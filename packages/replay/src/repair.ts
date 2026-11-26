import { parseModelRef, type LlmClient } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import { runReplay, SpendCapError, type RunRecord } from "./run.js";
import { judgeItem, type ItemResult } from "./verdict.js";

/**
 * Prompt Repair for format breaks. When the new model stops returning the JSON a prompt
 * expects, try a short list of fixed, explainable changes on the new model only, re-test each
 * with the same verdict rules, and report the first one that passes. Nothing is rewritten by an
 * AI: every repair is a change a person can read and apply.
 */
export type RepairKind = "format-instruction" | "json-mode";

export interface RepairPatch {
  kind: RepairKind;
  /** Text appended to the system prompt. */
  systemAddition: string;
  /** Also turn on the provider's JSON output mode. */
  jsonMode?: true;
}

export interface RepairAttempt {
  kind: RepairKind;
  fixed: boolean;
  detail: string;
}

export interface RepairResult {
  id: string;
  fixed: boolean;
  patch?: RepairPatch;
  attempts: RepairAttempt[];
}

export interface RepairReport {
  newModel: string;
  runs: number;
  calls: number;
  results: RepairResult[];
}

export interface RepairOptions {
  items: GoldenItem[];
  /** The replay's records: the old model's answers stay the reference. */
  records: RunRecord[];
  results: ItemResult[];
  newModel: string;
  client: LlmClient;
  runs: number;
  concurrency?: number;
  maxCalls?: number;
  signal?: AbortSignal;
}

const MAX_SCHEMA_CHARS = 2000;
// Anthropic's API has no JSON output switch; the other providers accept one.
const NO_JSON_MODE = new Set(["anthropic"]);

/** A prompt Repair can work on: broken, and only because the answer format broke. */
export function repairable(result: ItemResult): boolean {
  const broken = result.reasons.filter((r) => r.severity === "broken");
  return result.verdict === "broken" && broken.length > 0 && broken.every((r) => r.code === "format-broken");
}

export function formatInstruction(item: GoldenItem): string {
  const base = "Answer with one JSON value only: no explanation before or after it, no Markdown, no code fences.";
  if (!item.expect?.schema) return base;
  const schema = JSON.stringify(item.expect.schema);
  return schema.length <= MAX_SCHEMA_CHARS ? `${base} It must match this JSON Schema: ${schema}` : base;
}

function cell(text: string): string {
  return text.replace(/\s+/g, " ").replace(/\|/g, "\\|");
}

function code(text: string): string {
  return "`" + text.replace(/\s+/g, " ").replace(/`/g, "'") + "`";
}

function candidates(item: GoldenItem, newModel: string): RepairPatch[] {
  const systemAddition = formatInstruction(item);
  const list: RepairPatch[] = [{ kind: "format-instruction", systemAddition }];
  if (!item.jsonMode && !item.tools?.length && !NO_JSON_MODE.has(parseModelRef(newModel).provider)) {
    list.push({ kind: "json-mode", systemAddition, jsonMode: true });
  }
  return list;
}

export function applyRepair(item: GoldenItem, patch: RepairPatch): GoldenItem {
  const repaired: GoldenItem = { ...item, system: item.system ? `${item.system}\n\n${patch.systemAddition}` : patch.systemAddition };
  if (patch.jsonMode) repaired.jsonMode = true;
  return repaired;
}

/** The most calls a repair of these prompts can make. */
export function plannedRepairCalls(items: GoldenItem[], results: ItemResult[], newModel: string, runs: number): number {
  const byId = new Map(items.map((i) => [i.id, i]));
  return results.filter(repairable).reduce((sum, r) => sum + candidates(byId.get(r.id)!, newModel).length * runs, 0);
}

export async function runRepair(options: RepairOptions): Promise<RepairReport> {
  const { items, newModel, runs } = options;
  const byId = new Map(items.map((i) => [i.id, i]));
  const targets = options.results.filter(repairable).map((r) => ({ item: byId.get(r.id)!, tries: candidates(byId.get(r.id)!, newModel) }));
  const maxCalls = options.maxCalls ?? 500;
  const planned = targets.reduce((sum, t) => sum + t.tries.length * runs, 0);
  if (planned > maxCalls) {
    throw new SpendCapError(`Prompt Repair could need ${planned} AI calls, above the limit of ${maxCalls}. Lower the runs or raise the limit.`);
  }

  const results = new Map<string, RepairResult>(targets.map((t) => [t.item.id, { id: t.item.id, fixed: false, attempts: [] }]));
  const oldRecords = options.records.filter((r) => r.side === "old");
  let calls = 0;
  const rounds = Math.max(0, ...targets.map((t) => t.tries.length));

  for (let round = 0; round < rounds; round++) {
    const batch = targets.flatMap((t) => {
      const patch = t.tries[round];
      return patch && !results.get(t.item.id)!.fixed ? [{ original: t.item, patch, repaired: applyRepair(t.item, patch) }] : [];
    });
    if (batch.length === 0) break;
    const records = await runReplay({
      items: batch.map((b) => b.repaired),
      // Only the new model is called (sides); the old model's answers come from the replay.
      oldModel: newModel,
      newModel,
      client: options.client,
      runs,
      ...(options.concurrency ? { concurrency: options.concurrency } : {}),
      maxCalls: maxCalls - calls,
      sides: ["new"],
      ...(options.signal ? { signal: options.signal } : {}),
    });
    calls += records.length;
    const reference = [...oldRecords, ...records];
    for (const b of batch) {
      const judged = judgeItem(b.repaired, reference);
      const fixed = judged.verdict !== "broken";
      const format = judged.reasons.find((r) => r.code === "format-broken");
      const result = results.get(b.original.id)!;
      result.attempts.push({
        kind: b.patch.kind,
        fixed,
        detail: fixed
          ? `passes now (verdict: ${judged.verdict}, ${judged.new.runs - judged.new.errors} of ${judged.new.runs} calls answered)`
          : (format?.detail ?? judged.reasons.find((r) => r.severity === "broken")?.detail ?? "still broken"),
      });
      if (fixed) {
        result.fixed = true;
        result.patch = b.patch;
      }
    }
  }
  return { newModel, runs, calls, results: [...results.values()] };
}

const KIND_LABEL: Record<RepairKind, string> = {
  "format-instruction": "JSON-only instruction added to the system prompt",
  "json-mode": "JSON-only instruction plus the provider's JSON mode",
};

export function formatRepairMarkdown(report: RepairReport): string {
  if (report.results.length === 0) return "";
  const fixed = report.results.filter((r) => r.fixed);
  const lines = [
    "### Prompt Repair",
    "",
    `${fixed.length} of ${report.results.length} prompt${report.results.length === 1 ? "" : "s"} with a broken answer format fixed on \`${report.newModel}\` (${report.calls} extra calls, ${report.runs} run${report.runs === 1 ? "" : "s"} per try).`,
    "",
    "| Prompt | Result | Tried |",
    "| --- | --- | --- |",
    ...report.results.map(
      (r) =>
        `| ${cell(r.id)} | ${r.fixed ? `Fixed: ${KIND_LABEL[r.patch!.kind]}` : "Not fixed"} | ${r.attempts.map((a) => cell(`${a.kind}: ${a.detail}`)).join("<br>")} |`,
    ),
  ];
  for (const r of fixed) {
    lines.push("", `**Change for ${code(r.id)}:** add this to the end of the system prompt${r.patch!.jsonMode ? ", and turn on JSON output (`response_format: {type: \"json_object\"}` or `responseMimeType: \"application/json\"`)" : ""}:`, "", "```text", r.patch!.systemAddition, "```");
  }
  return lines.join("\n");
}

export function formatRepairText(report: RepairReport): string {
  if (report.results.length === 0) return "";
  const fixed = report.results.filter((r) => r.fixed).length;
  const lines = [`Prompt Repair: ${fixed} of ${report.results.length} format breaks fixed on ${report.newModel} (${report.calls} extra calls)`];
  for (const r of report.results) {
    lines.push(`  ${r.fixed ? "FIXED " : "NOT FIXED"} ${r.id}${r.fixed ? `: ${KIND_LABEL[r.patch!.kind]}` : ""}`);
    for (const a of r.attempts) lines.push(`    tried ${a.kind}: ${a.detail}`);
    if (r.fixed) {
      lines.push(`    add to the system prompt: ${r.patch!.systemAddition}`);
      if (r.patch!.jsonMode) lines.push("    and turn on the provider's JSON output mode");
    }
  }
  return lines.join("\n");
}

/** One golden-set line in the format parseGoldenSet reads, so a repaired set can be re-run. */
export function goldenLine(item: GoldenItem): string {
  const line: Record<string, unknown> = { id: item.id };
  if (item.system) line.system = item.system;
  line.messages = item.messages;
  if (item.tools) line.tools = item.tools;
  if (item.expect) line.expect = item.expect;
  if (item.maxTokens !== undefined) line.max_tokens = item.maxTokens;
  if (item.temperature !== undefined) line.temperature = item.temperature;
  if (item.jsonMode) line.json_mode = true;
  return JSON.stringify(line);
}
