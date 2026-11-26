import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join, posix } from "node:path";
import { formatReplayMarkdown, type ReplayReport } from "@ai-changeguard/replay";
import type { SwitchPlan } from "@ai-changeguard/scanner";

export class GitHubError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export type GitHubApi = <T>(method: "GET" | "POST" | "PATCH", path: string, body?: unknown) => Promise<T>;

export function githubApi(token: string, baseUrl = "https://api.github.com", fetchImpl: typeof fetch = fetch): GitHubApi {
  return async <T>(method: string, path: string, body?: unknown): Promise<T> => {
    const res = await fetchImpl(`${baseUrl.replace(/\/$/, "")}${path}`, {
      method,
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-change-guard-action",
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = ((await res.json().catch(() => ({}))) as { message?: string }).message ?? "";
      throw new GitHubError(res.status, `GitHub ${res.status} on ${method} ${path.split("?")[0]}${detail ? `: ${detail}` : ""}`);
    }
    return (res.status === 204 ? undefined : await res.json()) as T;
  };
}

export interface EvidencePrInput {
  api: GitHubApi;
  /** owner/name */
  repo: string;
  /** GITHUB_REF and GITHUB_SHA of the run. */
  ref: string;
  sha: string;
  workspace: string;
  /** Folder that was scanned, relative to the workspace ("" for the root). */
  scanPath: string;
  plan: SwitchPlan;
  replay: ReplayReport;
  /** Markdown shown under the heading, e.g. the Money-Finder savings card. */
  extra?: string;
  /** Wording for a pull request that is not a model switch (a shorter prompt). Default: a model switch. */
  text?: PrText;
}

export interface PrText {
  /** Commit subject; also how the branch is recognised as ours. */
  subject: string;
  branch: string;
  heading: string;
  /** What was replayed, after "replayed N of your real prompts M times each ". */
  replayed: string;
  beforeMerge: string[];
}

interface PullRequest {
  number: number;
  state: string;
  merged_at?: string | null;
  html_url: string;
}

export type EvidencePrResult =
  | { action: "created" | "updated"; number: number; url: string }
  | { action: "skipped"; reason: string };

const MAX_BODY = 60_000;
const MAX_LISTED = 50;

export function branchName(from: string, to: string): string {
  const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9.\-]+/g, "-").replace(/^-+|-+$/g, "");
  return `ai-change-guard/${slug(from)}-to-${slug(to)}`;
}

function code(text: string): string {
  return "`" + text.replace(/\s+/g, " ").slice(0, 200).replace(/`/g, "'") + "`";
}

function listed<T>(items: T[], render: (item: T) => string, more: string): string[] {
  return [...items.slice(0, MAX_LISTED).map(render), ...(items.length > MAX_LISTED ? [`- …and ${items.length - MAX_LISTED} more ${more}`] : [])];
}

export function evidenceBody(plan: SwitchPlan, replay: ReplayReport, scanPath: string, note?: string, extra?: string, text?: PrText): string {
  const where = (file: string, line: number) => `\`${posix.join(scanPath || ".", file)}:${line}\``;
  const t = replay.totals;
  const prompts = replay.items.length;
  const lines = [
    ...(note ? [`> ${note}`, ""] : []),
    text ? `## ${text.heading}` : `## Switch \`${plan.from}\` to \`${plan.to}\``,
    "",
    ...(extra ? [extra, ""] : []),
    `Before opening this pull request, AI Change Guard replayed **${prompts} of your real prompt${prompts === 1 ? "" : "s"}** ${replay.runs} time${replay.runs === 1 ? "" : "s"} each ${text ? text.replayed : `on \`${replay.oldModel}\` and \`${replay.newModel}\``}.`,
    "",
    "| Same | Changed | Broken | Errors |",
    "| ---: | ---: | ---: | ---: |",
    `| ${t.same} | ${t.changed} | ${t.broken} | ${t.error} |`,
    "",
    t.changed > 0
      ? `**${t.changed} prompt${t.changed === 1 ? "" : "s"} answered differently but passed your checks.** Read them in the full report below before merging.`
      : text
        ? "Every prompt passed your checks with the same kind of answer as before."
        : "Every prompt passed your checks with the same kind of answer on both models.",
    "",
    `### Code changes (${plan.edits.length})`,
    "",
    ...listed(plan.edits, (e) => `- ${where(e.file, e.line)}: ${code(e.before)} → ${code(e.after)}`, "(see the Files tab)"),
  ];
  if (plan.skipped.length > 0) {
    lines.push("", `### Not changed automatically (${plan.skipped.length})`, "");
    lines.push(...listed(plan.skipped, (s) => `- ${where(s.file, s.line)}: ${code(s.raw)}. ${s.reason}`, "(run `ai-change-guard switch` locally to see all)"));
  }
  lines.push(
    "",
    "### Before you merge",
    "",
    ...(text?.beforeMerge ?? [
      `- Settings are not checked yet: make sure every option you pass (temperature, max_tokens, tools, …) is supported by \`${plan.to}\`.`,
      "- Model IDs built at runtime (from variables or environment) are not changed; search for them by hand.",
    ]),
    "- A pull request opened with the default workflow token does not start other workflows. Close and reopen it, or give the Action a `github-token` from a GitHub App or a personal token, to run your CI on it.",
    "",
  );
  const top = lines.join("\n");
  const open = "\n<details><summary>Full replay report</summary>\n\n";
  const close = "\n\n</details>";
  const cut = "\n\n…cut to fit GitHub's limit; the full report is in the workflow run's Summary.";
  let report = formatReplayMarkdown(replay);
  const room = MAX_BODY - top.length - open.length - close.length;
  if (report.length > room) report = report.slice(0, Math.max(0, room - cut.length)) + cut;
  return top + open + report + close;
}

const PERMISSION_HELP =
  "Give the job `permissions: { contents: write, pull-requests: write }`, and in Settings → Actions → General turn on " +
  '"Allow GitHub Actions to create and approve pull requests".';

function permissionHint(error: unknown): unknown {
  if (error instanceof GitHubError && (error.status === 403 || error.status === 404)) return new Error(`${error.message.replace(/\.+$/, "")}. ${PERMISSION_HELP}`);
  return error;
}

/** The SHA git gives a file's bytes, to compare the runner's copy with the commit. */
export function gitBlobSha(content: Buffer): string {
  return createHash("sha1").update(`blob ${content.length}\0`).update(content).digest("hex");
}

/**
 * Keeps only files that git tracks at `sha` and that are byte-for-byte unchanged on the runner.
 * An untracked or gitignored file (a generated .env, say) or one an earlier step rewrote must
 * never be committed, so those changes move to the skipped list instead.
 */
async function trackedOnly(input: EvidencePrInput): Promise<{ plan: SwitchPlan; modes: Map<string, string> }> {
  const { api, repo, plan } = input;
  const tree = await api<{ truncated: boolean; tree: { path: string; mode: string; type: string; sha: string }[] }>(
    "GET",
    `/repos/${repo}/git/trees/${input.sha}?recursive=1`,
  );
  if (tree.truncated) throw new Error("The repository has too many files for GitHub to list in one tree, so AI Change Guard cannot check which files it may commit.");
  const tracked = new Map(tree.tree.filter((e) => e.type === "blob").map((e) => [e.path, e]));
  const keep: SwitchPlan = { ...plan, edits: [], skipped: [...plan.skipped], files: new Map() };
  const modes = new Map<string, string>();
  for (const [file, content] of plan.files) {
    const path = posix.join(input.scanPath || ".", file);
    const entry = tracked.get(path);
    const reason = !entry
      ? "Not committed in the repository (untracked or gitignored), so it is left out of the PR."
      : gitBlobSha(await readFile(join(input.workspace, path))) !== entry.sha
        ? "Changed by an earlier step of this job, so it is left out of the PR."
        : null;
    const edits = plan.edits.filter((e) => e.file === file);
    if (reason) {
      keep.skipped.push(...edits.map((e) => ({ file, line: e.line, raw: plan.from, reason })));
      continue;
    }
    keep.edits.push(...edits);
    keep.files.set(file, content);
    modes.set(path, entry!.mode);
  }
  keep.skipped.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line);
  return { plan: keep, modes };
}

/** Commits the planned switch to a branch and opens (or refreshes) one pull request for it. */
export async function openEvidencePr(input: EvidencePrInput): Promise<EvidencePrResult> {
  const { api, repo, replay } = input;
  const { default_branch: base } = await api<{ default_branch: string }>("GET", `/repos/${repo}`);
  if (input.ref !== `refs/heads/${base}`) {
    return { action: "skipped", reason: `Evidence PRs are opened from runs on ${base} (push, schedule or workflow_dispatch), not from ${input.ref}.` };
  }
  const head = await api<{ object: { sha: string } }>("GET", `/repos/${repo}/git/ref/heads/${encodeURIComponent(base)}`);
  if (head.object.sha !== input.sha) {
    return { action: "skipped", reason: `${base} moved on while this run was going; the next run opens the PR.` };
  }

  const { plan, modes } = await trackedOnly(input);
  if (plan.files.size === 0) return { action: "skipped", reason: "None of the files to change are committed and unchanged in the repository." };

  const branch = input.text?.branch ?? branchName(plan.from, plan.to);
  const owner = repo.split("/")[0];
  const t = replay.totals;
  const subject = input.text?.subject ?? `Switch ${plan.from} to ${plan.to}`;
  const message = `${subject}\n\nReplayed ${replay.items.length} prompt${replay.items.length === 1 ? "" : "s"} x ${replay.runs} run${replay.runs === 1 ? "" : "s"} with AI Change Guard: ${t.same} same, ${t.changed} changed, ${t.broken} broken.`;
  const body = evidenceBody(plan, replay, input.scanPath, undefined, input.extra, input.text);

  const listPulls = () =>
    api<PullRequest[]>("GET", `/repos/${repo}/pulls?state=all&per_page=20&head=${encodeURIComponent(`${owner}:${branch}`)}`);

  /** True when the branch holds nothing but one AI Change Guard commit, so it is ours to rebuild. */
  const ownsBranch = async (): Promise<{ owned: boolean; behind: number }> => {
    const cmp = await api<{ ahead_by: number; behind_by: number; commits: { commit: { message: string } }[] }>(
      "GET",
      `/repos/${repo}/compare/${encodeURIComponent(base)}...${encodeURIComponent(branch)}`,
    );
    const ours = cmp.ahead_by === 0 || (cmp.ahead_by === 1 && cmp.commits[0]?.commit.message.startsWith(`${subject}\n`) === true);
    return { owned: ours, behind: cmp.behind_by };
  };

  const commit = async (): Promise<string> => {
    const tree = [...plan.files].map(([file, content]) => {
      const path = posix.join(input.scanPath || ".", file);
      return { path, mode: modes.get(path)!, type: "blob", content };
    });
    const parent = await api<{ tree: { sha: string } }>("GET", `/repos/${repo}/git/commits/${input.sha}`);
    const newTree = await api<{ sha: string }>("POST", `/repos/${repo}/git/trees`, { base_tree: parent.tree.sha, tree });
    return (await api<{ sha: string }>("POST", `/repos/${repo}/git/commits`, { message, tree: newTree.sha, parents: [input.sha] })).sha;
  };

  /** The branch already holds these exact files (a rewrite written by a model can differ from run to run). */
  const branchHasFiles = async (): Promise<boolean> => {
    const tree = await api<{ tree: { path: string; sha: string }[] }>("GET", `/repos/${repo}/git/trees/${encodeURIComponent(branch)}?recursive=1`);
    const shas = new Map(tree.tree.map((e) => [e.path, e.sha]));
    return [...plan.files].every(([file, content]) => shas.get(posix.join(input.scanPath || ".", file)) === gitBlobSha(Buffer.from(content)));
  };

  const refresh = async (pr: PullRequest): Promise<EvidencePrResult> => {
    const { owned, behind } = await ownsBranch();
    // The description always shows this run's proof, so the code on the branch must be this run's code too.
    if (owned && (behind > 0 || (input.text && !(await branchHasFiles())))) {
      await api("PATCH", `/repos/${repo}/git/refs/heads/${branch}`, { sha: await commit(), force: true });
    }
    const note = owned ? undefined : `People have added commits to this branch, so AI Change Guard no longer changes its code. The lists below describe ${base} today.`;
    await api("PATCH", `/repos/${repo}/pulls/${pr.number}`, { body: owned ? body : evidenceBody(plan, replay, input.scanPath, note, input.extra, input.text) });
    return { action: "updated", number: pr.number, url: pr.html_url };
  };

  try {
    const existing = await listPulls();
    const open = existing.find((p) => p.state === "open");
    if (open) return await refresh(open);
    const closed = existing.find((p) => p.state === "closed" && !p.merged_at);
    if (closed) {
      return { action: "skipped", reason: `Pull request #${closed.number} for this switch was closed without merging, so no new one is opened. Reopen it to continue.` };
    }

    const sha = await commit();
    try {
      await api("POST", `/repos/${repo}/git/refs`, { ref: `refs/heads/${branch}`, sha });
    } catch (error) {
      if (!(error instanceof GitHubError) || error.status !== 422) throw error;
      // The branch exists: left over from a merged PR or an unfinished run, or made by a person.
      if (!(await ownsBranch()).owned) {
        return { action: "skipped", reason: `Branch ${branch} already exists with other commits; delete it to get a new pull request.` };
      }
      await api("PATCH", `/repos/${repo}/git/refs/heads/${branch}`, { sha, force: true });
    }

    try {
      const pr = await api<{ number: number; html_url: string }>("POST", `/repos/${repo}/pulls`, {
        title: `${subject} (tested on ${replay.items.length} real prompt${replay.items.length === 1 ? "" : "s"})`,
        head: branch,
        base,
        body,
      });
      return { action: "created", number: pr.number, url: pr.html_url };
    } catch (error) {
      // Another run opened it a moment ago.
      const raced = error instanceof GitHubError && error.status === 422 ? (await listPulls()).find((p) => p.state === "open") : undefined;
      if (!raced) throw error;
      await api("PATCH", `/repos/${repo}/pulls/${raced.number}`, { body });
      return { action: "updated", number: raced.number, url: raced.html_url };
    }
  } catch (error) {
    throw permissionHint(error);
  }
}
