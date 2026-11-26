import type { LlmResponse, ToolCall } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import type { RunRecord } from "./run.js";
import { extractJson, validateSchema } from "./schema.js";
import { similarity } from "./similarity.js";

export type Verdict = "same" | "changed" | "broken" | "error";

export type ReasonCode =
  | "calls-failing"
  | "format-broken"
  | "tool-change"
  | "tool-args-missing"
  | "missing-text"
  | "new-refusal"
  | "answer-drift";

export interface Reason {
  code: ReasonCode;
  severity: "broken" | "changed";
  detail: string;
}

export interface SideStats {
  runs: number;
  errors: number;
  avgInputTokens: number;
  avgOutputTokens: number;
  avgLatencyMs: number;
  avgChars: number;
  servedModels: string[];
}

export interface ItemResult {
  id: string;
  verdict: Verdict;
  reasons: Reason[];
  old: SideStats;
  new: SideStats;
  sampleOld?: string;
  sampleNew?: string;
  error?: string;
}

const average = (values: number[]) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);
const atLeast = (n: number, fraction: number) => Math.max(1, Math.ceil(n * fraction));
const count = (responses: LlmResponse[], test: (r: LlmResponse) => boolean) => responses.filter(test).length;

function sideStats(records: RunRecord[]): SideStats {
  const ok = records.flatMap((r) => (r.response ? [r.response] : []));
  return {
    runs: records.length,
    errors: records.length - ok.length,
    avgInputTokens: average(ok.map((r) => r.inputTokens)),
    avgOutputTokens: average(ok.map((r) => r.outputTokens)),
    avgLatencyMs: average(ok.map((r) => r.latencyMs)),
    avgChars: average(ok.map((r) => r.text.length)),
    servedModels: [...new Set(ok.flatMap((r) => (r.servedModel ? [r.servedModel] : [])))],
  };
}

function mostCommon(values: string[]): { value: string; share: number } {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best = { value: "", share: 0 };
  for (const [value, n] of counts) if (n / values.length > best.share) best = { value, share: n / values.length };
  return best;
}

function missingRequiredArg(item: GoldenItem, call: ToolCall): boolean {
  const required = item.tools?.find((t) => t.name === call.name)?.parameters.required;
  if (!Array.isArray(required)) return false;
  return required.some((key) => typeof key === "string" && !(key in call.arguments));
}

function firstError(records: RunRecord[]): string {
  return records.find((r) => r.error)?.error ?? "no runs";
}

export interface JudgeOptions {
  /**
   * Share of the new answers that must show a problem before it counts (default 0.4: 2 of 5).
   * The nightly watch uses 0.6 (3 of 5) so a single odd answer never raises an alarm.
   */
  share?: number;
}

export function judgeItem(item: GoldenItem, records: RunRecord[], options: JudgeOptions = {}): ItemResult {
  const share = options.share ?? 0.4;
  const oldRecords = records.filter((r) => r.itemId === item.id && r.side === "old");
  const newRecords = records.filter((r) => r.itemId === item.id && r.side === "new");
  const oldOk = oldRecords.flatMap((r) => (r.response ? [r.response] : []));
  const newOk = newRecords.flatMap((r) => (r.response ? [r.response] : []));

  const result: ItemResult = { id: item.id, verdict: "same", reasons: [], old: sideStats(oldRecords), new: sideStats(newRecords) };
  if (oldOk[0]) result.sampleOld = oldOk[0].text;
  if (newOk[0]) result.sampleNew = newOk[0].text;
  const add = (code: ReasonCode, severity: Reason["severity"], detail: string) => result.reasons.push({ code, severity, detail });

  if (oldOk.length === 0) {
    result.verdict = "error";
    const oldError = firstError(oldRecords);
    const newStatus =
      newOk.length > 0
        ? `The new model answered ${newOk.length} of ${newRecords.length} times.`
        : firstError(newRecords) === oldError
          ? "The new model failed the same way."
          : `The new model failed too: ${firstError(newRecords)}`;
    result.error = `The old model never answered: ${oldError} — ${newStatus}`;
    return result;
  }
  if (newOk.length === 0) {
    add("calls-failing", "broken", `All ${newRecords.length} calls to the new model failed: ${firstError(newRecords)}`);
    result.verdict = "broken";
    return result;
  }
  const newErrors = newRecords.length - newOk.length;
  if (newErrors >= atLeast(newRecords.length, share)) {
    add("calls-failing", "broken", `${newErrors} of ${newRecords.length} calls to the new model failed: ${firstError(newRecords)}`);
  }

  const threshold = atLeast(newOk.length, share);
  const expect = item.expect;

  if (expect?.json || expect?.schema) {
    const problem = (r: LlmResponse): string | null => {
      const parsed = extractJson(r.text);
      if (!parsed.ok) return "not valid JSON";
      return expect.schema ? (validateSchema(parsed.value, expect.schema)[0] ?? null) : null;
    };
    const newBad = count(newOk, (r) => problem(r) !== null);
    const oldBad = count(oldOk, (r) => problem(r) !== null);
    if (newBad >= threshold && newBad > oldBad) {
      const example = newOk.map(problem).find((p) => p !== null);
      add(
        "format-broken",
        "broken",
        `${newBad} of ${newOk.length} new answers fail the expected ${expect.schema ? "JSON schema" : "JSON format"} (old: ${oldBad} of ${oldOk.length}). First problem: ${example}`,
      );
    }
  }

  if (expect?.tool) {
    const wrong = (r: LlmResponse) => !r.toolCalls.some((c) => c.name === expect.tool);
    const newWrong = count(newOk, wrong);
    const oldWrong = count(oldOk, wrong);
    if (newWrong >= threshold && newWrong > oldWrong) {
      add("tool-change", "broken", `${newWrong} of ${newOk.length} new answers don't call ${expect.tool} (old: ${oldWrong} of ${oldOk.length})`);
    }
  } else if (item.tools?.length) {
    const firstTool = (r: LlmResponse) => r.toolCalls[0]?.name ?? "no tool";
    const oldMode = mostCommon(oldOk.map(firstTool));
    const newMode = mostCommon(newOk.map(firstTool));
    if (oldMode.share >= 0.6 && newMode.share >= 0.6 && oldMode.value !== newMode.value) {
      add("tool-change", "broken", `The old model called ${oldMode.value}; the new one calls ${newMode.value}`);
    }
  }

  if (item.tools?.length) {
    const missing = (r: LlmResponse) => r.toolCalls.some((c) => missingRequiredArg(item, c));
    const newMissing = count(newOk, missing);
    const oldMissing = count(oldOk, missing);
    if (newMissing >= threshold && newMissing > oldMissing) {
      add("tool-args-missing", "broken", `${newMissing} of ${newOk.length} new tool calls miss a required argument (old: ${oldMissing} of ${oldOk.length})`);
    }
  }

  if (expect?.contains?.length) {
    const wanted = expect.contains;
    const lacks = (r: LlmResponse) => wanted.some((s) => !r.text.toLowerCase().includes(s.toLowerCase()));
    const newLacks = count(newOk, lacks);
    const oldLacks = count(oldOk, lacks);
    if (newLacks >= threshold && newLacks > oldLacks) {
      add("missing-text", "broken", `${newLacks} of ${newOk.length} new answers miss required text: ${wanted.join(", ")} (old: ${oldLacks} of ${oldOk.length})`);
    }
  }

  const newRefusals = count(newOk, (r) => r.refused);
  const oldRefusals = count(oldOk, (r) => r.refused);
  const refusalChange = newRefusals >= atLeast(newOk.length, 0.6) && oldRefusals <= Math.floor(oldOk.length * 0.2);
  if (refusalChange) {
    add("new-refusal", "changed", `The new model refused ${newRefusals} of ${newOk.length} times (old: ${oldRefusals} of ${oldOk.length})`);
  }

  const oldTexts = oldOk.map((r) => r.text).filter((t) => t.trim());
  const newTexts = newOk.map((r) => r.text).filter((t) => t.trim());
  // Drift needs at least two old answers: they show how much the old model varies by itself.
  // With one, any rewording would look like drift.
  if (!refusalChange && oldTexts.length > 1 && newTexts.length > 0) {
    const pairs: number[] = [];
    for (let i = 0; i < oldTexts.length; i++) {
      for (let j = i + 1; j < oldTexts.length; j++) pairs.push(similarity(oldTexts[i]!, oldTexts[j]!));
    }
    const oldSelf = average(pairs);
    const cross = newTexts.map((t) => Math.max(...oldTexts.map((o) => similarity(t, o))));
    const drifted = cross.filter((c) => c < oldSelf * 0.7).length;
    if (drifted >= atLeast(newTexts.length, 0.6)) {
      add(
        "answer-drift",
        "changed",
        `${drifted} of ${newTexts.length} new answers read differently from the old ones (word overlap ${average(cross).toFixed(2)} vs ${oldSelf.toFixed(2)} between old answers; approximate)`,
      );
    }
  }

  result.verdict = result.reasons.some((r) => r.severity === "broken") ? "broken" : result.reasons.length > 0 ? "changed" : "same";
  return result;
}
