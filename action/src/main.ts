import { appendFile, readFile } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { createClient, parseModelRef, withRetry, type LlmClient } from "@ai-changeguard/llm";
import {
  candidateProblem,
  cheaperCandidates,
  cleanCandidates,
  findSavings,
  formatSavingsMarkdown,
  formatSlimMarkdown,
  slimPrompts,
  formatRepairMarkdown,
  formatReplayMarkdown,
  formatWatchMarkdown,
  loadGoldenSet,
  parseBaseline,
  postSlack,
  REPLAY_CHECK_NAME,
  replayCheckText,
  replayConclusion,
  replayTitle,
  runReplayJob,
  runWatch,
  slackMessage,
} from "@ai-changeguard/replay";
import { addOnlinePackageChecks, buildCheck, buildReport, checkSwitchIds, compareMcp, describeServer, listMcpTools, planPromptEdit, planSwitch, scanDirectory, type Report } from "@ai-changeguard/scanner";
import { createHash } from "node:crypto";
import { GitHubError, githubApi, openEvidencePr, type GitHubApi } from "./pr.js";

const SOON_DAYS = 90;

export type Env = Record<string, string | undefined>;

export function input(name: string, env: Env): string {
  return (env[`INPUT_${name.replace(/ /g, "_").toUpperCase()}`] ?? "").trim();
}

function positive(value: string, name: string): number {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1) throw new Error(`Input "${name}" must be a whole number of 1 or more.`);
  return n;
}

function oneOf(value: string, name: string, allowed: string[]): string {
  if (!allowed.includes(value)) throw new Error(`Input "${name}" must be one of: ${allowed.join(", ")}.`);
  return value;
}

/** Whether ai-stack.lock differences fail the step: any, only code changes (outdated), only catalog changes, or none. */
export function lockFails(report: Report, failOn: string): boolean {
  if (failOn === "none" || !report.lock) return false;
  if ("error" in report.lock) return true;
  return report.lock.changes.some((c) => failOn === "any" || c.kind === failOn);
}

export function scanFails(report: Report, failOn: string): boolean {
  if (failOn === "none") return false;
  // A malicious AI package release fails the step at every level except "none".
  if (report.packages?.some((p) => p.issue === "compromised")) return true;
  const retired = report.models.some((m) => m.status === "retired" || m.status === "rerouted") || report.apis.some((a) => a.daysLeft <= 0);
  if (failOn === "retired") return retired;
  const soon =
    report.models.some((m) => (m.status === "retiring" || m.status === "rerouting") && (m.daysLeft ?? Infinity) <= SOON_DAYS) ||
    report.apis.some((a) => a.daysLeft > 0 && a.daysLeft <= SOON_DAYS);
  return retired || soon;
}

export interface RunDeps {
  env: Env;
  today: string;
  log: (line: string) => void;
  client?: LlmClient;
  github?: GitHubApi;
}

/**
 * The plain model IDs an evidence PR swaps. Only a switch within one provider (and one gateway
 * prefix) is a pure ID change; anything else needs different SDK code, which we never write.
 */
export function switchIds(oldModel: string, newModel: string): { from: string; to: string; prefix?: string } {
  const a = parseModelRef(oldModel);
  const b = parseModelRef(newModel);
  // Gemini accepts "models/<id>" and "<id>" alike, so that prefix is not a different route.
  const bare = (m: string) => m.replace(/^models\//, "");
  const prefix = (m: string) => (bare(m).includes("/") ? bare(m).slice(0, bare(m).lastIndexOf("/")) : "");
  if (a.provider !== b.provider || prefix(a.model) !== prefix(b.model)) {
    throw new Error(`open-pr only switches models within one provider: ${oldModel} to ${newModel} needs different SDK code, so change it by hand.`);
  }
  const last = (m: string) => m.slice(m.lastIndexOf("/") + 1);
  const ids = checkSwitchIds(last(a.model), last(b.model));
  return prefix(a.model) ? { ...ids, prefix: prefix(a.model) } : ids;
}

/**
 * Publishes the replay report as a check run on the commit, which is where the AI Change Guard dashboard reads
 * its migration report. Never fails the job: without "checks: write" it says how to turn it on.
 */
async function publishReplayCheck({ env, log, github, replay, prUrl, failOn }: {
  env: RunDeps["env"];
  log: RunDeps["log"];
  github: GitHubApi | undefined;
  replay: Parameters<typeof replayCheckText>[0];
  prUrl: string | undefined;
  failOn: "broken" | "changed" | "none";
}): Promise<void> {
  const repo = env.GITHUB_REPOSITORY ?? "";
  const sha = env.GITHUB_SHA ?? "";
  const token = input("github-token", env) || env.GITHUB_TOKEN || "";
  if (!repo || !sha || (!github && !token)) return;
  try {
    await (github ?? githubApi(token, env.GITHUB_API_URL))("POST", `/repos/${repo}/check-runs`, {
      name: REPLAY_CHECK_NAME,
      head_sha: sha,
      status: "completed",
      conclusion: replayConclusion(replay, failOn),
      output: { title: replayTitle(replay), summary: formatReplayMarkdown(replay).slice(0, 60_000), text: replayCheckText(replay, prUrl) },
    });
    log("AI Change Guard: saved the model switch report as a check run; it shows on your AI Change Guard dashboard");
  } catch (error) {
    const hint =
      error instanceof GitHubError && error.status === 403
        ? 'add "checks: write" to the workflow\'s permissions to see this report on your AI Change Guard dashboard'
        : error instanceof Error
          ? error.message
          : String(error);
    log(`::notice::AI Change Guard: could not save the model switch report as a check run (${hint}).`);
  }
}

/** Runs the Action and returns the process exit code. */
export async function run({ env, today, log, client, github }: RunDeps): Promise<number> {
  const workspace = env.GITHUB_WORKSPACE ?? process.cwd();
  const summary = async (markdown: string) => {
    if (env.GITHUB_STEP_SUMMARY) await appendFile(env.GITHUB_STEP_SUMMARY, `${markdown}\n\n`);
    else log(markdown);
  };

  const failOn = oneOf(input("fail-on", env) || "retired", "fail-on", ["retired", "retiring", "none"]);
  const lockFailOn = oneOf(input("lock-fail-on", env) || "none", "lock-fail-on", ["none", "any", "outdated", "catalog", "unverified", "mcp"]);
  const openPr = oneOf(input("open-pr", env) || "false", "open-pr", ["true", "false"]) === "true";
  const scanRoot = resolve(workspace, input("path", env) || ".");
  const scan = await scanDirectory(scanRoot);
  let report = buildReport(scan, env.GITHUB_REPOSITORY ?? workspace, today);
  if (oneOf(input("online-package-check", env) || "false", "online-package-check", ["true", "false"]) === "true") {
    report = await addOnlinePackageChecks(report, { today });
  }
  if (oneOf(input("mcp-tools", env) || "false", "mcp-tools", ["true", "false"]) === "true") {
    // Starts the repository's own MCP servers on this runner to compare their tools with ai-stack.lock.
    const servers = scan.mcpServers ?? [];
    if (!report.lock || !("file" in report.lock) || report.lock.file.mcp === undefined) {
      log("::notice::AI Change Guard: mcp-tools needs an ai-stack.lock with an MCP section, written with: ai-change-guard lock --mcp");
    } else {
      // Only the variables the workflow names in mcp-env are filled into the config: a config changed in a
      // pull request must not be able to send other runner variables (tokens) to a URL of its choosing.
      const allowed = input("mcp-env", env).split(",").map((v) => v.trim()).filter(Boolean);
      const mcpEnv = Object.fromEntries(allowed.map((name) => [name, env[name]]));
      if (servers.length) log(`AI Change Guard: starting ${servers.length} MCP server(s) to list their tools: ${servers.map(describeServer).join("; ")}`);
      // Servers that left the config are reported too, so the list is compared even when it is empty.
      const listed = await listMcpTools(servers, { cwd: scanRoot, env: mcpEnv });
      for (const s of listed) if (s.error) log(`::warning::AI Change Guard: could not list the tools of ${s.name} (${s.file}): ${s.error}`);
      report = { ...report, lock: { ...report.lock, changes: [...report.lock.changes, ...compareMcp(report.lock.file.mcp, listed)] } };
    }
  }
  const check = buildCheck(report, null);
  await summary(`## AI Change Guard: ${check.title}\n\n${check.summary}`);
  let failed = scanFails(report, failOn) || lockFails(report, lockFailOn);
  log(failed ? `::error::AI Change Guard: ${check.title}` : `AI Change Guard: ${check.title}`);

  const prompts = input("prompts", env);
  const oldModel = input("old-model", env);
  const newModel = input("new-model", env);

  // Watch: ask the model in use again and compare with the committed baseline.
  const baselinePath = input("baseline", env);
  if (baselinePath) {
    if (!prompts) throw new Error('A watch needs "prompts" (the golden set the baseline was made from).');
    if (oldModel || newModel || openPr) throw new Error('"baseline" runs a watch of one model; leave out old-model, new-model and open-pr.');
    const baseline = parseBaseline(await readFile(resolve(workspace, baselinePath), "utf8"));
    const items = await loadGoldenSet(resolve(workspace, prompts));
    const runsInput = input("runs", env);
    const watch = await runWatch({
      items,
      baseline,
      client: client ?? withRetry(createClient({ env })),
      concurrency: positive(input("concurrency", env) || "2", "concurrency"),
      maxCalls: positive(input("max-calls", env) || "300", "max-calls"),
      ...(runsInput ? { runs: positive(runsInput, "runs") } : {}),
    });
    await summary(formatWatchMarkdown(watch));
    const runUrl = env.GITHUB_SERVER_URL && env.GITHUB_REPOSITORY && env.GITHUB_RUN_ID ? `${env.GITHUB_SERVER_URL}/${env.GITHUB_REPOSITORY}/actions/runs/${env.GITHUB_RUN_ID}` : undefined;
    const webhook = input("slack-webhook", env);
    if (watch.alert && webhook) {
      await postSlack(webhook, slackMessage(watch, runUrl)).then(
        () => log("AI Change Guard watch: alert posted to Slack"),
        (error: unknown) => log(`::warning::AI Change Guard watch: could not post to Slack: ${error instanceof Error ? error.message : String(error)}`),
      );
    }
    const t = watch.report.totals;
    log(`${watch.alert ? "::error::" : ""}AI Change Guard watch: ${t.same} same, ${t.changed} changed, ${t.broken} broken${watch.reroutes.length ? `, rerouted to ${watch.reroutes.map((r) => r.served).join(", ")}` : ""}`);
    return failed || watch.alert ? 1 : 0;
  }

  if (oneOf(input("slim-prompts", env) || "false", "slim-prompts", ["true", "false"]) === "true") {
    // Verified Prompt Slimming: shorter system prompts, proved on the golden set with old-model.
    if (!prompts || !oldModel) throw new Error('slim-prompts needs "prompts" and "old-model" (the model you use now).');
    if (newModel) throw new Error('slim-prompts keeps your model: remove "new-model", or run the replay in a separate step.');
    if (input("find-savings", env) === "true") throw new Error("slim-prompts and find-savings each need their own step: run them as two steps.");
    const writer = input("writer-model", env) || oldModel;
    const token = input("github-token", env) || env.GITHUB_TOKEN || "";
    if (openPr && !github && !token) throw new Error('open-pr needs a token: keep the default "github-token" input or pass one.');
    const callsInput = input("calls-per-month", env);
    const result = await slimPrompts({
      items: await loadGoldenSet(resolve(workspace, prompts)),
      model: oldModel,
      writer,
      client: client ?? withRetry(createClient({ env })),
      runs: positive(input("runs", env) || "3", "runs"),
      concurrency: positive(input("concurrency", env) || "2", "concurrency"),
      maxCalls: positive(input("max-calls", env) || "300", "max-calls"),
      today,
      ...(callsInput ? { callsPerMonth: positive(callsInput, "calls-per-month") } : {}),
    });
    const card = formatSlimMarkdown(result);
    await summary(card);
    const passed = result.prompts.filter((p) => p.best);
    log(passed.length ? `AI Change Guard prompt slimming: ${passed.length} shorter system prompt(s) passed` : "AI Change Guard prompt slimming: no shorter prompt passed");
    // One pull request, for the prompt that saves the most (dollars when every one is priced, else the share); the card lists the others.
    const byMoney = passed.every((p) => p.best!.monthlySaving !== undefined);
    const worth = (p: (typeof passed)[number]) => (byMoney ? p.best!.monthlySaving! : (p.best!.savingShare ?? 0));
    const top = passed.sort((a, b) => worth(b) - worth(a))[0];
    if (openPr && top) {
      const notice = (text: string) => log(`::notice::AI Change Guard: ${text}`);
      const edit = top.skippedPrompts.length > 0 ? undefined : await planPromptEdit(scanRoot, top.system, top.best!.prompt);
      const flat = top.system.replace(/\s+/g, " ").trim();
      const label = flat.length <= 40 ? flat : `${flat.slice(0, 40).replace(/\s+\S*$/, "")}…`;
      const short = (t: string) => t.replace(/\s+/g, " ").trim().slice(0, 80);
      if (!edit) notice(`no pull request, because ${oldModel} did not answer ${top.skippedPrompts.length} prompt(s), so the comparison is incomplete.`);
      else if (edit.edits.length === 0) {
        notice(edit.skipped.length ? `no pull request: ${edit.skipped[0]!.file}:${edit.skipped[0]!.line}: ${edit.skipped[0]!.reason}.` : "no pull request, because the system prompt is not written as one string in the code (it may be built at runtime); paste the shorter one from the Summary by hand.");
      } else {
        const pr = await openEvidencePr({
          api: github ?? githubApi(token, env.GITHUB_API_URL),
          repo: env.GITHUB_REPOSITORY ?? "",
          ref: env.GITHUB_REF ?? "",
          sha: env.GITHUB_SHA ?? "",
          workspace,
          scanPath: relative(workspace, scanRoot).split(sep).join("/"),
          plan: {
            from: short(top.system),
            to: short(top.best!.prompt),
            edits: edit.edits.map((e) => ({ ...e, before: short(top.system), after: short(top.best!.prompt) })),
            skipped: edit.skipped.map((e) => ({ ...e, raw: short(top.system) })),
            files: edit.files,
          },
          replay: top.best!.report!,
          extra: card.replace(/^## .*\n\n/, ""),
          text: {
            subject: `Shorten the system prompt "${label}"`,
            branch: `ai-change-guard/slim-prompt-${createHash("sha1").update(top.system).digest("hex").slice(0, 10)}`,
            heading: "A shorter system prompt, tested on your real prompts",
            replayed: `on \`${oldModel}\` with the current and the shorter system prompt`,
            beforeMerge: [
              "- The shorter prompt was written by an AI model and passed your golden set: read it once to check it still says what you mean.",
              "- Prompts built at runtime or joined from several strings are not changed; they are listed above.",
            ],
          },
        });
        if (pr.action === "skipped") notice(`no pull request. ${pr.reason}`);
        else {
          log(`AI Change Guard ${pr.action} pull request #${pr.number}: ${pr.url}`);
          await summary(`**Shorter prompt pull request ${pr.action}:** [#${pr.number}](${pr.url})`);
        }
      }
    }
    return failed ? 1 : 0;
  }

  if (oneOf(input("find-savings", env) || "false", "find-savings", ["true", "false"]) === "true") {
    // Money-Finder: cheaper models on the same provider, judged on the golden set against old-model.
    if (!prompts || !oldModel) throw new Error('find-savings needs "prompts" and "old-model" (the model you use now).');
    if (newModel) throw new Error('find-savings picks the new model itself: remove "new-model", or run the replay in a separate step.');
    const candidatesInput = input("candidates", env);
    const candidates = candidatesInput ? cleanCandidates(oldModel, candidatesInput.split(",")) : cheaperCandidates(oldModel, today).map((c) => c.ref);
    if (candidatesInput && candidates.length === 0) throw new Error('"candidates" needs at least one model other than "old-model".');
    // A switch PR is planned for every testable candidate before any paid call: a bad ID or another provider fails here.
    const plans = new Map<string, Awaited<ReturnType<typeof planSwitch>>>();
    if (openPr) {
      for (const c of candidates) {
        if (candidateProblem(c, today)) continue;
        const ids = switchIds(oldModel, c);
        plans.set(c, await planSwitch(scanRoot, scan.findings, ids.from, ids.to, ids.prefix ? { prefix: ids.prefix } : {}));
      }
    }
    const callsInput = input("calls-per-month", env);
    const llm = client ?? withRetry(createClient({ env }));
    const result = await findSavings({
      candidates,
      items: await loadGoldenSet(resolve(workspace, prompts)),
      primary: oldModel,
      client: llm,
      runs: positive(input("runs", env) || "3", "runs"),
      concurrency: positive(input("concurrency", env) || "2", "concurrency"),
      maxCalls: positive(input("max-calls", env) || "300", "max-calls"),
      today,
      ...(callsInput ? { callsPerMonth: positive(callsInput, "calls-per-month") } : {}),
    });
    const card = formatSavingsMarkdown(result);
    await summary(card);
    log(result.best ? `AI Change Guard Money-Finder: ${result.best.model} passes and is ${Math.round((result.best.savingShare ?? 0) * 100)}% cheaper per call` : "AI Change Guard Money-Finder: no cheaper model passed");
    if (openPr && result.best) {
      const plan = plans.get(result.best.model)!;
      const notice = (text: string) => log(`::notice::AI Change Guard: ${text}`);
      if (result.skippedPrompts.length > 0) notice(`no pull request, because ${oldModel} did not answer ${result.skippedPrompts.length} prompt(s), so the comparison is incomplete.`);
      else if (plan.edits.length === 0) notice(`no pull request, because no code uses ${plan.from} as a plain model ID.`);
      else {
        const token = input("github-token", env) || env.GITHUB_TOKEN || "";
        if (!github && !token) throw new Error('open-pr needs a token: keep the default "github-token" input or pass one.');
        const pr = await openEvidencePr({
          api: github ?? githubApi(token, env.GITHUB_API_URL),
          repo: env.GITHUB_REPOSITORY ?? "",
          ref: env.GITHUB_REF ?? "",
          sha: env.GITHUB_SHA ?? "",
          workspace,
          scanPath: relative(workspace, scanRoot).split(sep).join("/"),
          plan,
          replay: result.best.report,
          extra: card.replace(/^## .*\n\n/, ""),
        });
        if (pr.action === "skipped") notice(`no pull request. ${pr.reason}`);
        else {
          log(`AI Change Guard ${pr.action} pull request #${pr.number}: ${pr.url}`);
          await summary(`**Savings pull request ${pr.action}:** [#${pr.number}](${pr.url})`);
        }
      }
    }
    return failed ? 1 : 0;
  }

  if (openPr && !(prompts && oldModel && newModel)) {
    throw new Error('open-pr needs a replay as proof: set "prompts", "old-model" and "new-model".');
  }
  if (prompts || oldModel || newModel) {
    if (!prompts || !oldModel || !newModel) throw new Error('A replay needs all three inputs: "prompts", "old-model" and "new-model".');
    // Plan the code change before the replay: bad IDs fail here, before any paid call.
    const ids = openPr ? switchIds(oldModel, newModel) : undefined;
    const plan = ids ? await planSwitch(scanRoot, scan.findings, ids.from, ids.to, ids.prefix ? { prefix: ids.prefix } : {}) : undefined;
    const replayFailOn = oneOf(input("replay-fail-on", env) || "broken", "replay-fail-on", ["broken", "changed", "none"]);
    const runs = positive(input("runs", env) || "3", "runs");
    const repair = oneOf(input("repair", env) || "false", "repair", ["true", "false"]) === "true";
    const concurrency = positive(input("concurrency", env) || "2", "concurrency");
    const maxCalls = positive(input("max-calls", env) || "300", "max-calls");
    const llm = client ?? withRetry(createClient({ env }));
    const items = await loadGoldenSet(resolve(workspace, prompts));
    const job = await runReplayJob({ items, oldModel, newModel, runs, concurrency, maxCalls, client: llm, repair });
    const replay = job.report;
    await summary(formatReplayMarkdown(replay));
    if (job.repair) {
      await summary(formatRepairMarkdown(job.repair));
      const fixed = job.repair.results.filter((r) => r.fixed).length;
      log(`AI Change Guard Prompt Repair: ${fixed} of ${job.repair.results.length} format breaks fixed (${job.repair.calls} extra calls); the fixes are in the Summary`);
    }
    if (job.repairSkipped) log(`::warning::AI Change Guard ${job.repairSkipped}`);
    const { broken, changed } = replay.totals;
    const replayFailed = (replayFailOn === "broken" && broken > 0) || (replayFailOn === "changed" && broken + changed > 0);
    log(`${replayFailed ? "::error::" : ""}AI Change Guard replay: ${replay.totals.same} same, ${changed} changed, ${broken} broken`);
    failed = failed || replayFailed;

    let prUrl: string | undefined;
    // The report is saved for the dashboard even when opening the pull request fails (permissions, a race).
    try {
      if (ids && plan) {
        const notice = (text: string) => log(`::notice::AI Change Guard: ${text}`);
        if (replayFailed || broken > 0 || replay.totals.error > 0) {
          notice("no pull request, because the replay did not pass (see the Summary).");
        } else if (plan.edits.length === 0) {
          notice(`no pull request, because no code uses ${ids.from} as a plain model ID${plan.skipped.length ? ` (${plan.skipped.length} use(s) need a manual change)` : ""}.`);
        } else {
          const token = input("github-token", env) || env.GITHUB_TOKEN || "";
          if (!github && !token) throw new Error('open-pr needs a token: keep the default "github-token" input or pass one.');
          const repo = env.GITHUB_REPOSITORY ?? "";
          const result = await openEvidencePr({
            api: github ?? githubApi(token, env.GITHUB_API_URL),
            repo,
            ref: env.GITHUB_REF ?? "",
            sha: env.GITHUB_SHA ?? "",
            workspace,
            scanPath: relative(workspace, scanRoot).split(sep).join("/"),
            plan,
            replay,
          });
          if (result.action === "skipped") notice(`no pull request. ${result.reason}`);
          else {
            log(`AI Change Guard ${result.action} pull request #${result.number}: ${result.url}`);
            await summary(`**Evidence pull request ${result.action}:** [#${result.number}](${result.url})`);
            prUrl = result.url;
          }
        }
      }
    } finally {
      await publishReplayCheck({ env, log, github, replay, prUrl, failOn: replayFailOn as "broken" | "changed" | "none" });
    }
  }
  return failed ? 1 : 0;
}

if (process.env.GITHUB_ACTIONS === "true") {
  run({ env: process.env, today: new Date().toISOString().slice(0, 10), log: (line) => console.log(line) }).then(
    (code) => {
      process.exitCode = code;
    },
    (error: unknown) => {
      console.log(`::error::${error instanceof Error ? error.message : String(error)}`);
      process.exitCode = 1;
    },
  );
}
