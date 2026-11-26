import { createHash } from "node:crypto";
import type { LlmClient, LlmResponse } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import { buildReplayReport, isReroute, modelName, type ReplayReport } from "./report.js";
import { runReplay, type RunRecord } from "./run.js";

/**
 * Nightly watch: is the model you already use still answering your prompts the way it did?
 * A baseline stores N answers per prompt from one run. Each watch run asks the same model again
 * and judges the fresh answers against the stored ones with the replay's verdict rules, at a
 * stricter threshold (3 of 5) so one odd answer never raises an alarm. It also notices when the
 * provider serves a different model than the one requested, or a different snapshot than before.
 */

export const BASELINE_VERSION = 1;
/** Answers longer than this are cut, on both sides alike, so a long JSON answer is judged fairly. */
const MAX_TEXT = 32_000;
export const WATCH_SHARE = 0.6;

export interface BaselineRecord {
  itemId: string;
  run: number;
  response?: LlmResponse;
  error?: string;
}

export interface Baseline {
  version: number;
  model: string;
  createdAt: string;
  runs: number;
  /** One entry per prompt, with a hash of its request, so an edited prompt is not compared to old answers. */
  items: { id: string; hash: string }[];
  records: BaselineRecord[];
}

export interface WatchReport {
  model: string;
  baselineCreatedAt: string;
  report: ReplayReport;
  /** The provider served another model than the one requested. */
  reroutes: { requested: string; served: string }[];
  /** Model versions the API reported, at baseline time and now (a changed snapshot behind an alias). */
  servedBefore: string[];
  servedNow: string[];
  /** The API now reports a model version it did not report at baseline time. */
  snapshotChanged: boolean;
  /** Prompts with no usable baseline: new, or edited since the baseline was made. */
  notInBaseline: string[];
  /** Something changed that deserves a person's attention. */
  alert: boolean;
}

/** JSON with sorted keys, so re-saving a golden set with keys in another order changes nothing. */
function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stable((value as Record<string, unknown>)[k])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Fingerprint of what is sent to the model (not the id or the checks). */
export function itemHash(item: GoldenItem): string {
  const request = { system: item.system, messages: item.messages, tools: item.tools, maxTokens: item.maxTokens, temperature: item.temperature, jsonMode: item.jsonMode };
  return createHash("sha256").update(stable(request)).digest("hex").slice(0, 16);
}

function trimmed(response: LlmResponse): LlmResponse {
  return response.text.length > MAX_TEXT ? { ...response, text: response.text.slice(0, MAX_TEXT) } : response;
}

export interface BaselineOptions {
  items: GoldenItem[];
  model: string;
  client: LlmClient;
  runs?: number;
  concurrency?: number;
  maxCalls?: number;
  now?: Date;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

/** Runs every prompt `runs` times on `model` and keeps the answers as the reference. */
export async function createBaseline(options: BaselineOptions): Promise<Baseline> {
  const runs = options.runs ?? 5;
  const records = await runReplay({
    items: options.items,
    oldModel: options.model,
    newModel: options.model,
    runs,
    client: options.client,
    sides: ["new"],
    ...(options.concurrency ? { concurrency: options.concurrency } : {}),
    ...(options.maxCalls ? { maxCalls: options.maxCalls } : {}),
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.onProgress ? { onProgress: options.onProgress } : {}),
  });
  // Every prompt needs answers, or it could never be watched.
  const empty = options.items.filter((i) => !records.some((r) => r.itemId === i.id && r.response)).map((i) => i.id);
  if (empty.length > 0) {
    const error = records.find((r) => empty.includes(r.itemId))?.error ?? "";
    throw new Error(`No answers for ${empty.join(", ")} (${error}). If this is a rate limit, try again with a lower --concurrency.`);
  }
  return {
    version: BASELINE_VERSION,
    model: options.model,
    createdAt: (options.now ?? new Date()).toISOString(),
    runs,
    items: options.items.map((i) => ({ id: i.id, hash: itemHash(i) })),
    records: records.map((r) => ({ itemId: r.itemId, run: r.run, ...(r.response ? { response: trimmed(r.response) } : { error: r.error ?? "failed" }) })),
  };
}

export function parseBaseline(text: string): Baseline {
  let data: Baseline;
  try {
    data = JSON.parse(text) as Baseline;
  } catch {
    throw new Error("The baseline file is not valid JSON.");
  }
  if (data?.version !== BASELINE_VERSION || typeof data.model !== "string" || !Array.isArray(data.records) || !Array.isArray(data.items)) {
    throw new Error("This is not an AI Change Guard baseline file (or it was made by a newer version).");
  }
  return data;
}

export interface WatchOptions {
  items: GoldenItem[];
  baseline: Baseline;
  client: LlmClient;
  runs?: number;
  concurrency?: number;
  maxCalls?: number;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

export async function runWatch(options: WatchOptions): Promise<WatchReport> {
  const { baseline } = options;
  const known = new Map(baseline.items.map((i) => [i.id, i.hash]));
  const watched = options.items.filter((i) => known.get(i.id) === itemHash(i));
  const notInBaseline = options.items.filter((i) => known.get(i.id) !== itemHash(i)).map((i) => i.id);
  if (watched.length === 0) throw new Error("None of these prompts match the baseline. Make a new baseline for this golden set.");

  const runs = options.runs ?? baseline.runs;
  const live = await runReplay({
    items: watched,
    oldModel: baseline.model,
    newModel: baseline.model,
    runs,
    client: options.client,
    sides: ["new"],
    ...(options.concurrency ? { concurrency: options.concurrency } : {}),
    ...(options.maxCalls ? { maxCalls: options.maxCalls } : {}),
    ...(options.signal ? { signal: options.signal } : {}),
    ...(options.onProgress ? { onProgress: options.onProgress } : {}),
  });
  // Cut fresh answers exactly like the stored ones, so both sides are judged alike.
  for (const r of live) if (r.response) r.response = trimmed(r.response);
  const ids = new Set(watched.map((i) => i.id));
  const stored: RunRecord[] = baseline.records
    .filter((r) => ids.has(r.itemId))
    .map((r) => ({ itemId: r.itemId, side: "old" as const, run: r.run, ...(r.response ? { response: r.response } : { error: r.error ?? "failed" }) }));

  const report = buildReplayReport(watched, [...stored, ...live], { oldModel: baseline.model, newModel: baseline.model, runs, judge: { share: WATCH_SHARE } });
  const served = (records: { response?: LlmResponse }[]) => [...new Set(records.flatMap((r) => (r.response?.servedModel ? [r.response.servedModel] : [])))].sort();
  const servedBefore = served(stored);
  const servedNow = served(live);
  const requested = modelName(baseline.model);
  // Only what is new since the baseline counts: an alias the provider always served under another
  // name (claude-...-latest, gemini-flash-latest) is not a change.
  const isNew = (s: string) => !servedBefore.includes(s);
  const reroutes = servedNow.filter((s) => isNew(s) && isReroute(requested, s)).map((s) => ({ requested, served: s }));
  const snapshotChanged = servedBefore.length > 0 && servedNow.some(isNew);
  // "error" means a prompt could not be compared at all; an unwatched prompt must not stay silent.
  const alert = report.totals.broken + report.totals.changed + report.totals.error > 0 || reroutes.length > 0 || snapshotChanged;
  return { model: baseline.model, baselineCreatedAt: baseline.createdAt, report, reroutes, servedBefore, servedNow, snapshotChanged, notInBaseline, alert };
}

function headline(w: WatchReport): string {
  const t = w.report.totals;
  const n = w.report.items.length;
  if (!w.alert) return `No change: ${n} prompt${n === 1 ? "" : "s"} on ${w.model} still answer like the baseline of ${w.baselineCreatedAt.slice(0, 10)}.`;
  const parts: string[] = [];
  if (t.broken) parts.push(`${t.broken} broken`);
  if (t.changed) parts.push(`${t.changed} changed`);
  if (t.error) parts.push(`${t.error} could not be compared`);
  if (w.reroutes.length) parts.push(`rerouted to ${w.reroutes.map((r) => r.served).join(", ")}`);
  if (w.snapshotChanged) parts.push(`now served by ${w.servedNow.join(", ")} (was ${w.servedBefore.join(", ")})`);
  return `${w.model} changed since the baseline of ${w.baselineCreatedAt.slice(0, 10)}: ${parts.join(", ")}.`;
}

export function formatWatchText(w: WatchReport): string {
  const lines = [`AI Change Guard watch · ${w.model}`, headline(w)];
  for (const item of w.report.items) {
    if (item.verdict === "same") continue;
    lines.push(`  ${item.verdict.toUpperCase()}  ${item.id}`);
    for (const r of item.reasons) lines.push(`    ${r.code}: ${r.detail}`);
    if (item.error) lines.push(`    ${item.error}`);
  }
  if (w.notInBaseline.length) lines.push(`Not watched (new or edited since the baseline; make a new baseline to include them): ${w.notInBaseline.join(", ")}`);
  return lines.join("\n");
}

export function formatWatchMarkdown(w: WatchReport): string {
  const t = w.report.totals;
  const lines = [
    `## ${w.alert ? "AI model watch: change detected" : "AI model watch: no change"}`,
    "",
    headline(w),
    "",
    "| Same | Changed | Broken | Could not compare |",
    "| ---: | ---: | ---: | ---: |",
    `| ${t.same} | ${t.changed} | ${t.broken} | ${t.error} |`,
  ];
  const changed = w.report.items.filter((i) => i.verdict !== "same");
  if (changed.length) {
    lines.push("", "| Prompt | Verdict | Why |", "| --- | --- | --- |");
    for (const i of changed) {
      const why = (i.reasons.map((r) => r.detail).join("; ") || i.error || "").replace(/\s+/g, " ").replace(/\|/g, "\\|");
      lines.push(`| ${i.id.replace(/\|/g, "\\|")} | ${i.verdict} | ${why} |`);
    }
  }
  if (w.servedNow.length) lines.push("", `Served by: ${w.servedNow.join(", ")}${w.servedBefore.length ? ` (at baseline: ${w.servedBefore.join(", ")})` : ""}.`);
  if (w.notInBaseline.length) lines.push("", `Not watched (new or edited since the baseline): ${w.notInBaseline.join(", ")}.`);
  return lines.join("\n");
}

/** Slack incoming-webhook message: counts and prompt ids only, never prompt or answer text. */
export function slackMessage(w: WatchReport, link?: string): { text: string } {
  const ids = w.report.items.filter((i) => i.verdict === "broken" || i.verdict === "changed").map((i) => `${i.id} (${i.verdict})`);
  const text = [`:rotating_light: ${headline(w)}`, ids.length ? `Prompts: ${ids.slice(0, 20).join(", ")}${ids.length > 20 ? ", …" : ""}` : "", link ? `Details: ${link}` : ""]
    .filter(Boolean)
    .join("\n");
  return { text };
}

/** Posts to a Slack incoming webhook. Only https URLs; the URL is a secret and is never logged. */
export async function postSlack(webhookUrl: string, message: { text: string }, fetchImpl: typeof fetch = fetch): Promise<void> {
  if (!/^https:\/\//.test(webhookUrl)) throw new Error("The Slack webhook must be an https URL.");
  const res = await fetchImpl(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(message),
    signal: AbortSignal.timeout(10_000),
  });
  if (!res.ok) throw new Error(`Slack answered ${res.status}.`);
}
