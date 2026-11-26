import { realpath } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, posix, relative, resolve, sep } from "node:path";
import { lookupModel, models as catalogModels, type Provider } from "@ai-changeguard/catalog";
import { createClient, parseModelRef, withRetry, type LlmClient } from "@ai-changeguard/llm";
import { formatRepairMarkdown, formatReplayMarkdown, loadGoldenSet, plannedCalls, ReplayCancelledError, runReplayJob } from "@ai-changeguard/replay";
import { buildReport, classifyModelId, daysBetween, formatText, planSwitch, scanDirectory } from "@ai-changeguard/scanner";
import { McpServer } from "@modelcontextprotocol/server";
import * as z from "zod/v4";

/**
 * The AI Change Guard MCP server. AI coding assistants (Claude Code, Cursor, ...) start it on the
 * developer's machine, so code and API keys never leave it. Every path is resolved inside the
 * project root (the folder the server was started in, or ACG_ROOT).
 */
export interface ServerOptions {
  /** Project root; paths outside it are refused. Default: ACG_ROOT or the working directory. */
  root?: string;
  /** Count days from this date (YYYY-MM-DD). Default: today in UTC. */
  today?: string;
  /** LLM client for test_model_switch. Default: real providers, keys from the environment. */
  client?: LlmClient;
}

export const VERSION = "0.1.0";

/** The project root: the option, else ACG_ROOT, else the working directory. */
export function resolveRoot(root?: string): string {
  return resolve(root ?? process.env.ACG_ROOT ?? process.cwd());
}

/**
 * Some MCP clients start servers in "/" or the home folder. Without an explicit root that would
 * make the whole disk "the project", so tools refuse to run until ACG_ROOT (or root) is set.
 */
function unsafeRoot(root: string, explicit: boolean): boolean {
  return !explicit && (root === resolve("/") || root === resolve(homedir()));
}

/** Last segment of a gateway-prefixed ID (openai/gpt-4o -> gpt-4o). */
const bareId = (id: string) => id.trim().slice(id.trim().lastIndexOf("/") + 1);

type ToolResult = { content: { type: "text"; text: string }[]; isError?: boolean };
const text = (...parts: string[]): ToolResult => ({ content: parts.map((t) => ({ type: "text" as const, text: t })) });
const fail = (message: string): ToolResult => ({ content: [{ type: "text", text: message }], isError: true });

export function createServer(options: ServerOptions = {}): McpServer {
  const root = resolveRoot(options.root);
  const rootProblem = unsafeRoot(root, Boolean(options.root ?? process.env.ACG_ROOT))
    ? fail(`The server was started in ${root}, not in a project. Set ACG_ROOT to your project folder in the MCP server settings (see the README), then try again.`)
    : null;
  const today = () => options.today ?? new Date().toISOString().slice(0, 10);

  const within = (base: string, full: string) => {
    const rel = relative(base, full);
    return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
  };
  /**
   * The absolute path for `path` if it lies inside the root, otherwise null. Symlinks are resolved
   * first, so a link inside the project that points outside it is refused too.
   */
  const inside = async (path: string | undefined): Promise<string | null> => {
    const full = resolve(root, path ?? ".");
    if (!within(root, full)) return null;
    const [realRoot, realFull] = await Promise.all([realpath(root).catch(() => root), realpath(full).catch(() => full)]);
    return within(realRoot, realFull) ? full : null;
  };
  const outside = (path: string) => fail(`"${path}" is outside the project folder (${root}). Use a path inside it.`);

  const server = new McpServer(
    { name: "ai-change-guard", version: VERSION },
    {
      instructions:
        "Finds AI models and AI APIs in this project that are retired or retiring, and tests a model switch on real prompts. " +
        "Start with scan_ai_models; use model_info or list_retirements for dates and replacements; plan_model_switch shows the exact lines to change; " +
        "test_model_switch replays a golden set on both models (it spends the user's API credits, so confirm with the user first).",
    },
  );

  server.registerTool(
    "scan_ai_models",
    {
      title: "Scan for AI models",
      description:
        "Finds every AI model ID and deprecated AI API in the project (code, .env, YAML, JSON) and reports which are retired or retiring, " +
        "when, the suggested replacement, and the file:line of each use. Runs locally; nothing is uploaded.",
      inputSchema: z.object({ path: z.string().optional().describe("Folder to scan, relative to the project root (default: the whole project)") }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ path }) => {
      if (rootProblem) return rootProblem;
      const folder = await inside(path);
      if (!folder) return outside(path ?? ".");
      try {
        // Report every location relative to the project root, whatever folder was scanned.
        const prefix = relative(root, folder).split(sep).join("/");
        const scan = await scanDirectory(folder);
        const fix = (file: string) => (prefix ? posix.join(prefix, file) : file);
        scan.findings = scan.findings.map((f) => ({ ...f, file: fix(f.file) }));
        scan.tokenTips = scan.tokenTips.map((t) => ({ ...t, file: fix(t.file) }));
        scan.security = scan.security.map((s) => ({ ...s, file: fix(s.file) }));
        const report = buildReport(scan, prefix || ".", today());
        const summary = report.models.map((m) => ({
          model: m.id,
          platform: m.platform,
          status: m.status,
          date: m.date,
          daysLeft: m.daysLeft,
          replacement: m.replacement ?? m.reroutesTo,
          dateVerified: m.verified,
          locations: m.locations,
        }));
        const security = (report.security ?? []).map((s) => ({ rule: s.rule, title: s.title, owasp: s.owasp, severity: s.severity, fix: s.fix, findings: s.findings }));
        return text(
          formatText(report),
          JSON.stringify({ models: summary, apis: report.apis, security, dynamicModelIds: report.dynamic.map((d) => d.location) }, null, 2),
        );
      } catch (error) {
        return fail(`Scan failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    },
  );

  server.registerTool(
    "model_info",
    {
      title: "Look up a model",
      description: "Retirement or reroute date, suggested replacement and official sources for one model ID, from the AI Change Guard catalog.",
      inputSchema: z.object({ model_id: z.string().min(1).describe("Model ID as written in code, e.g. gpt-4o-2024-08-06 or openai/gpt-4o") }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ model_id }) => {
      const id = classifyModelId(model_id)?.id ?? model_id.trim().toLowerCase();
      const entries = lookupModel(id);
      if (entries.length === 0) {
        return text(
          `${id}: no announced retirement in the AI Change Guard catalog. The catalog follows the providers' official pages; a model not listed may still change later.`,
        );
      }
      const lines = entries.map((e) => {
        const days = daysBetween(today(), e.retiresOn);
        const when = days <= 0 ? `since ${e.retiresOn}` : `on ${e.retiresOn} (in ${days} day${days === 1 ? "" : "s"})`;
        const what = e.reroutesTo ? `reroutes to ${e.reroutesTo} ${when}` : `${days <= 0 ? "retired" : "retires"} ${when}`;
        return [
          `${e.id} on ${e.platform}: ${what}.`,
          e.replacement ? `Suggested replacement: ${e.replacement}.` : "",
          e.verified ? "" : "This date is inferred, not yet confirmed on the provider's page.",
          e.note ?? "",
          `Sources: ${e.sources.map((s) => `${s.url} (checked ${s.checked})`).join(", ")}`,
        ]
          .filter(Boolean)
          .join(" ");
      });
      return text(lines.join("\n"));
    },
  );

  server.registerTool(
    "list_retirements",
    {
      title: "List upcoming retirements",
      description: "AI models from OpenAI, Anthropic, Google and Mistral whose retirement or reroute is within the given number of days, soonest first.",
      inputSchema: z.object({
        within_days: z.number().int().min(1).max(730).optional().describe("Look this many days ahead (default 90)"),
        provider: z.enum(["openai", "anthropic", "google", "mistral"]).optional(),
        include_retired: z.boolean().optional().describe("Also list models already retired (default false)"),
      }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ within_days, provider, include_retired }) => {
      const limit = within_days ?? 90;
      const rows = catalogModels
        .filter((m) => !provider || m.provider === (provider as Provider))
        .map((m) => ({ m, days: daysBetween(today(), m.retiresOn) }))
        .filter(({ days }) => days <= limit && (include_retired || days > 0))
        .sort((a, b) => a.days - b.days || a.m.id.localeCompare(b.m.id));
      if (rows.length === 0) return text(`No ${provider ? `${provider} ` : ""}model retirements in the next ${limit} days.`);
      return text(
        rows
          .map(({ m, days }) => {
            const when = days <= 0 ? `retired ${m.retiresOn}` : `${m.reroutesTo ? `reroutes to ${m.reroutesTo}` : "retires"} ${m.retiresOn} (${days} days)`;
            return `${m.id} [${m.platform}] ${when}${m.replacement ? ` → ${m.replacement}` : ""}${m.verified ? "" : " (date unconfirmed)"}`;
          })
          .join("\n"),
      );
    },
  );

  server.registerTool(
    "plan_model_switch",
    {
      title: "Plan a model switch",
      description:
        "Shows exactly which lines would change to switch from one model ID to another, and which uses need a manual change " +
        "(fine-tuned, Bedrock, Azure, other routes). Changes nothing: apply the edits yourself after reviewing them, ideally after test_model_switch passes.",
      inputSchema: z.object({
        from: z.string().min(1).describe("Model ID used today, e.g. gpt-4o"),
        to: z.string().min(1).describe("Model ID to switch to, e.g. gpt-6-luna"),
        path: z.string().optional().describe("Folder to look in, relative to the project root (default: whole project)"),
      }),
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ from, to, path }) => {
      if (rootProblem) return rootProblem;
      const folder = await inside(path);
      if (!folder) return outside(path ?? ".");
      try {
        const plan = await planSwitch(folder, (await scanDirectory(folder)).findings, bareId(from), bareId(to));
        if (plan.edits.length === 0 && plan.skipped.length === 0) return text(`No uses of ${plan.from} found.`);
        const where = (file: string, line: number) => `${relative(root, resolve(folder, file))}:${line}`;
        const lines = [
          `${plan.edits.length} line(s) in ${plan.files.size} file(s) would change from ${plan.from} to ${plan.to}:`,
          ...plan.edits.map((e) => `- ${where(e.file, e.line)}\n  - ${e.before}\n  + ${e.after}`),
          ...(plan.skipped.length ? ["", "Change by hand:", ...plan.skipped.map((s) => `- ${where(s.file, s.line)} (${s.raw}): ${s.reason}`)] : []),
        ];
        return text(lines.join("\n"));
      } catch (error) {
        return fail(error instanceof Error ? error.message : String(error));
      }
    },
  );

  server.registerTool(
    "test_model_switch",
    {
      title: "Test a model switch on real prompts",
      description:
        "Replays a golden set (JSONL of real prompts) on the old and the new model and reports, per prompt, same / changed / broken with the reason " +
        "(broken JSON, changed tool calls, missing text, new refusals, drift). Optionally tries prompt fixes when a JSON format breaks. " +
        "Calls the providers with the user's own API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, ...) and costs money: confirm with the user first.",
      inputSchema: z.object({
        prompts: z.string().min(1).describe("Golden set file, relative to the project root (e.g. tests/golden.jsonl)"),
        old_model: z.string().min(3).describe("provider:model used today, e.g. openai:gpt-4o"),
        new_model: z.string().min(3).describe("provider:model to switch to, e.g. openai:gpt-6-luna"),
        runs: z.number().int().min(1).max(5).optional().describe("Runs per prompt on each model (default 2)"),
        max_calls: z.number().int().min(2).max(500).optional().describe("Refuse to start above this many calls (default 60)"),
        repair: z.boolean().optional().describe("Try prompt fixes on the new model when a JSON format breaks (default false)"),
      }),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: false, openWorldHint: true },
    },
    async ({ prompts, old_model, new_model, runs = 2, max_calls = 60, repair = false }, ctx) => {
      if (rootProblem) return rootProblem;
      const file = await inside(prompts);
      if (!file) return outside(prompts);
      try {
        parseModelRef(old_model);
        parseModelRef(new_model);
        const items = await loadGoldenSet(file);
        const planned = plannedCalls(items.length, runs);
        if (planned > max_calls) {
          return fail(
            `This replay needs ${planned} calls (${items.length} prompts × ${runs} runs × 2 models), above max_calls ${max_calls}. Lower runs or raise max_calls.`,
          );
        }
        // Progress keeps long replays alive in clients that reset their timeout on progress, and
        // cancellation stops further paid calls as soon as the client gives up.
        const progressToken = ctx.mcpReq._meta?.progressToken;
        const job = await runReplayJob({
          items,
          oldModel: old_model,
          newModel: new_model,
          runs,
          maxCalls: max_calls,
          concurrency: 2,
          client: options.client ?? withRetry(createClient()),
          repair,
          signal: ctx.mcpReq.signal,
          ...(progressToken !== undefined
            ? {
                onProgress: (done: number, total: number) =>
                  void ctx.mcpReq
                    .notify({ method: "notifications/progress", params: { progressToken, progress: done, total, message: `${done} of ${total} calls` } })
                    .catch(() => {}),
              }
            : {}),
        });
        const parts = [formatReplayMarkdown(job.report)];
        if (job.repair) parts.push(formatRepairMarkdown(job.repair));
        if (job.repairSkipped) parts.push(job.repairSkipped);
        return text(parts.join("\n\n"));
      } catch (error) {
        if (error instanceof ReplayCancelledError) return fail(error.message);
        return fail(error instanceof Error ? error.message : String(error));
      }
    },
  );

  return server;
}
