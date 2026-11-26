import { billedPriceOn, callCost, defaultPlatform, formatUsd, localToday, type Platform, type Provider } from "@ai-changeguard/catalog";
import type { GoldenItem } from "./golden.js";
import type { RunRecord, Side } from "./run.js";
import { judgeItem, type ItemResult, type JudgeOptions, type Verdict } from "./verdict.js";

export interface Reroute {
  side: Side;
  requested: string;
  served: string;
}

interface Usage {
  input: number;
  output: number;
}

export interface ReplayReport {
  oldModel: string;
  newModel: string;
  runs: number;
  calls: number;
  failedCalls: number;
  totals: Record<Verdict, number>;
  avgTokens: { old: Usage; new: Usage };
  avgLatencyMs: { old: number; new: number };
  /** US dollars per call at list price, from the average tokens; missing for a model the catalog has no price for. */
  costPerCall?: { old?: number; new?: number };
  comparedPrompts: number;
  rateLimited: boolean;
  reroutes: Reroute[];
  items: ItemResult[];
}

const VERDICT_ORDER: Record<Verdict, number> = { broken: 0, changed: 1, error: 2, same: 3 };

const average = (values: number[]) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);

/** The model part of a provider:model reference. */
export function modelName(ref: string): string {
  return ref.slice(ref.indexOf(":") + 1);
}

function lastSegment(name: string): string {
  return (name.split("/").at(-1) ?? name).toLowerCase();
}

export function isReroute(requested: string, served: string): boolean {
  const a = lastSegment(requested);
  const b = lastSegment(served);
  return !b.startsWith(a) && !a.startsWith(b);
}

/**
 * The catalog platform a provider:model reference is priced on. Undefined for gateways (groq, openrouter,
 * ollama), which bill their own prices: the catalog only knows the providers' own APIs.
 */
export function platformForRef(ref: string): Platform | undefined {
  const provider = ref.slice(0, ref.indexOf(":"));
  return Object.hasOwn(defaultPlatform, provider) ? defaultPlatform[provider as Provider] : undefined;
}

/** List-price cost of one call for a provider:model reference, or undefined when the catalog has no price. */
export function costOfCall(ref: string, usage: Usage, today: string): number | undefined {
  const platform = platformForRef(ref);
  const price = platform ? billedPriceOn(modelName(ref), platform, today) : undefined;
  return price ? callCost(price, usage.input, usage.output) : undefined;
}

export function buildReplayReport(
  items: GoldenItem[],
  records: RunRecord[],
  options: { oldModel: string; newModel: string; runs: number; judge?: JudgeOptions; today?: string },
): ReplayReport {
  const results = items.map((item) => judgeItem(item, records, options.judge)).sort((a, b) => VERDICT_ORDER[a.verdict] - VERDICT_ORDER[b.verdict]);
  const totals: Record<Verdict, number> = { same: 0, changed: 0, broken: 0, error: 0 };
  for (const r of results) totals[r.verdict]++;

  const ok = (side: Side) => records.flatMap((r) => (r.side === side && r.response ? [r.response] : []));
  // Compare only prompts both models answered, so a failed call doesn't skew the averages.
  const paired = results.filter((r) => r.old.runs > r.old.errors && r.new.runs > r.new.errors);
  const usage = (side: Side): Usage => ({
    input: average(paired.map((r) => r[side].avgInputTokens)),
    output: average(paired.map((r) => r[side].avgOutputTokens)),
  });

  const reroutes: Reroute[] = [];
  for (const side of ["old", "new"] as const) {
    const requested = modelName(side === "old" ? options.oldModel : options.newModel);
    const served = new Set(ok(side).flatMap((r) => (r.servedModel ? [r.servedModel] : [])));
    for (const s of served) if (isReroute(requested, s)) reroutes.push({ side, requested, served: s });
  }

  const today = options.today ?? localToday();
  const avgTokens = { old: usage("old"), new: usage("new") };
  const oldCost = paired.length ? costOfCall(options.oldModel, avgTokens.old, today) : undefined;
  const newCost = paired.length ? costOfCall(options.newModel, avgTokens.new, today) : undefined;

  return {
    oldModel: options.oldModel,
    newModel: options.newModel,
    runs: options.runs,
    calls: records.length,
    failedCalls: records.filter((r) => r.error).length,
    totals,
    avgTokens,
    avgLatencyMs: { old: average(paired.map((r) => r.old.avgLatencyMs)), new: average(paired.map((r) => r.new.avgLatencyMs)) },
    ...(oldCost !== undefined || newCost !== undefined ? { costPerCall: { ...(oldCost !== undefined ? { old: oldCost } : {}), ...(newCost !== undefined ? { new: newCost } : {}) } } : {}),
    comparedPrompts: paired.length,
    rateLimited: records.some((r) => r.error?.startsWith("HTTP 429")),
    reroutes,
    items: results,
  };
}

function change(before: number, after: number): string {
  if (before === 0) return "";
  const pct = Math.round(((after - before) / before) * 100);
  return ` (${pct > 0 ? "+" : ""}${pct}%)`;
}

/** "Cost per call at list price: $0.0012 → $0.0015 (+25%)", or "" when neither side has a price. */
export function costLine(report: ReplayReport): string {
  const c = report.costPerCall;
  if (!c) return "";
  const side = (v: number | undefined) => (v === undefined ? "no list price known" : formatUsd(v));
  const pct = c.old !== undefined && c.new !== undefined ? change(c.old, c.new) : "";
  return `Cost per call at list price: ${side(c.old)} → ${side(c.new)}${pct}`;
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

const LABEL: Record<Verdict, string> = { broken: "BROKEN ", changed: "CHANGED", error: "ERROR  ", same: "SAME   " };

export function formatReplayText(report: ReplayReport): string {
  const t = report.totals;
  const out: string[] = [];
  out.push(`Replay: ${report.oldModel} → ${report.newModel} · ${report.items.length} prompt${report.items.length === 1 ? "" : "s"} × ${report.runs} run${report.runs === 1 ? "" : "s"} · ${report.calls} calls (${report.failedCalls} failed)`);
  out.push(`Verdict: ${t.same} same · ${t.changed} changed · ${t.broken} broken · ${t.error} could not compare`);
  const o = report.avgTokens.old;
  const n = report.avgTokens.new;
  out.push(
    `Per call, over the ${report.comparedPrompts} prompts both models answered: input tokens ${Math.round(o.input)} → ${Math.round(n.input)}${change(o.input, n.input)} · ` +
      `output tokens ${Math.round(o.output)} → ${Math.round(n.output)}${change(o.output, n.output)} · ` +
      `latency ${seconds(report.avgLatencyMs.old)} → ${seconds(report.avgLatencyMs.new)}`,
  );
  const cost = costLine(report);
  if (cost) out.push(cost);
  if (report.rateLimited) {
    out.push("TIP: some calls hit the provider's rate limit. Re-run with --concurrency 1, or use a key with a higher quota.");
  }
  for (const r of report.reroutes) {
    out.push(`NOTE: ${r.side} model "${r.requested}" was answered by "${r.served}", a different model than requested.`);
  }
  out.push("");
  for (const item of report.items) {
    if (item.verdict === "same" || item.verdict === "error") continue;
    out.push(`${LABEL[item.verdict]}  ${item.id}`);
    for (const reason of item.reasons) out.push(`  ${reason.code}: ${reason.detail}`);
  }
  const errorsByMessage = new Map<string, string[]>();
  for (const item of report.items) {
    if (item.verdict !== "error") continue;
    const message = item.error ?? "Unknown error";
    errorsByMessage.set(message, [...(errorsByMessage.get(message) ?? []), item.id]);
  }
  for (const [message, ids] of errorsByMessage) {
    const shown = ids.length > 3 ? `${ids.slice(0, 3).join(", ")} and ${ids.length - 3} more` : ids.join(", ");
    out.push(`${LABEL.error}  ${ids.length === 1 ? ids[0] : `${ids.length} prompts (${shown})`}`);
    out.push(`  ${message}`);
  }
  if (t.same > 0) out.push(`${t.same} ${t.same === 1 ? "prompt behaves" : "prompts behave"} the same.`);
  return out.join("\n");
}

export function formatReplayMarkdown(report: ReplayReport): string {
  const t = report.totals;
  const o = report.avgTokens.old;
  const n = report.avgTokens.new;
  const out: string[] = [
    `## Model switch check: \`${report.oldModel}\` → \`${report.newModel}\``,
    "",
    `${report.items.length} real prompt${report.items.length === 1 ? "" : "s"}, ${report.runs} run${report.runs === 1 ? "" : "s"} each on both models.`,
    "",
    "| Same | Changed | Broken | Could not compare |",
    "| --- | --- | --- | --- |",
    `| ${t.same} | ${t.changed} | ${t.broken} | ${t.error} |`,
    "",
    `Per call: input tokens ${Math.round(o.input)} → ${Math.round(n.input)}${change(o.input, n.input)}, output tokens ${Math.round(o.output)} → ${Math.round(n.output)}${change(o.output, n.output)}, latency ${seconds(report.avgLatencyMs.old)} → ${seconds(report.avgLatencyMs.new)}.`,
  ];
  const cost = costLine(report);
  if (cost) out.push("", `${cost}.`);
  for (const r of report.reroutes) out.push("", `> **Note:** the ${r.side} model \`${r.requested}\` was answered by \`${r.served}\`.`);
  const flagged = report.items.filter((i) => i.verdict !== "same");
  if (flagged.length > 0) {
    out.push("", "### What changed");
    for (const item of flagged) {
      out.push("", `**${item.verdict.toUpperCase()}: ${item.id}**`);
      if (item.error) out.push(`- ${item.error}`);
      for (const reason of item.reasons) out.push(`- ${reason.detail}`);
    }
  }
  return out.join("\n");
}
