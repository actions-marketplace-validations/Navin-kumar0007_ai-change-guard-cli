// The replay report as the Action publishes it in a GitHub check run, so the AI Change Guard dashboard can show
// the migration report straight from the customer's repository. The data stays in their GitHub: the dashboard
// reads it when someone opens the page and stores nothing.
import type { ReplayReport, Reroute } from "./report.js";
import type { Verdict } from "./verdict.js";

/** The check run's name: the dashboard looks for exactly this. */
export const REPLAY_CHECK_NAME = "AI Change Guard model switch";
const MARK = "ai-change-guard:replay:v1";
/** GitHub caps a check run's text at 65,535 characters; stay under it with room for the wrapper. */
const TEXT_LIMIT = 60_000;

export interface PublishedItem {
  id: string;
  verdict: Verdict;
  reasons: string[];
  sampleOld?: string;
  sampleNew?: string;
  error?: string;
}

export interface PublishedReplay {
  v: 1;
  oldModel: string;
  newModel: string;
  runs: number;
  calls: number;
  failedCalls: number;
  totals: Record<Verdict, number>;
  avgTokens: ReplayReport["avgTokens"];
  avgLatencyMs: ReplayReport["avgLatencyMs"];
  costPerCall?: ReplayReport["costPerCall"];
  comparedPrompts: number;
  rateLimited: boolean;
  reroutes: Reroute[];
  /** Changed and broken prompts first. Long answers are cut, and when even that doesn't fit, the last items are left out. */
  items: PublishedItem[];
  omittedItems: number;
  prUrl?: string;
}

const cut = (s: string | undefined, n: number) => (s === undefined ? undefined : s.length > n ? `${s.slice(0, n)}…` : s);

function item(i: ReplayReport["items"][number], sample: number, sameSamples: boolean): PublishedItem {
  const withSamples = i.verdict !== "same" || sameSamples;
  return {
    id: i.id.slice(0, 200),
    verdict: i.verdict,
    reasons: i.reasons.map((r) => r.detail.slice(0, 500)),
    ...(withSamples && i.sampleOld !== undefined ? { sampleOld: cut(i.sampleOld, sample) } : {}),
    ...(withSamples && i.sampleNew !== undefined ? { sampleNew: cut(i.sampleNew, sample) } : {}),
    ...(i.error ? { error: i.error.slice(0, 500) } : {}),
  };
}

/** JSON that can sit inside an HTML comment: "<", ">" and "&" never appear literally, so "-->" can't end it early. */
const safeJson = (value: unknown) => JSON.stringify(value).replace(/</g, "\\u003c").replace(/>/g, "\\u003e").replace(/&/g, "\\u0026");

/** The check run's text: the report as data, in a comment GitHub doesn't show. */
export function replayCheckText(report: ReplayReport, prUrl?: string, limit = TEXT_LIMIT): string {
  const base: Omit<PublishedReplay, "items" | "omittedItems"> = {
    v: 1,
    oldModel: report.oldModel.slice(0, 200),
    newModel: report.newModel.slice(0, 200),
    runs: report.runs,
    calls: report.calls,
    failedCalls: report.failedCalls,
    totals: report.totals,
    avgTokens: report.avgTokens,
    avgLatencyMs: report.avgLatencyMs,
    ...(report.costPerCall ? { costPerCall: report.costPerCall } : {}),
    comparedPrompts: report.comparedPrompts,
    rateLimited: report.rateLimited,
    reroutes: report.reroutes.slice(0, 10).map((r) => ({ side: r.side, requested: r.requested.slice(0, 200), served: r.served.slice(0, 200) })),
    ...(prUrl ? { prUrl: prUrl.slice(0, 500) } : {}),
  };
  const wrap = (items: PublishedItem[], omitted: number) => `<!-- ${MARK} ${safeJson({ ...base, items, omittedItems: omitted })} -->`;
  // Shrink step by step: shorter answers, then no answers for unchanged prompts, then fewer prompts.
  for (const [sample, sameSamples] of [[2000, true], [2000, false], [800, false], [300, false]] as const) {
    const text = wrap(report.items.map((i) => item(i, sample, sameSamples)), 0);
    if (text.length <= limit) return text;
  }
  const items = report.items.map((i) => item(i, 300, false));
  let keep = items.length;
  while (keep > 0 && wrap(items.slice(0, keep), items.length - keep).length > limit) keep = Math.floor(keep * 0.8);
  // The header alone is bounded (cut names, at most 10 reroutes), so with no items it always fits.
  return wrap(items.slice(0, keep), items.length - keep);
}

const VERDICTS = new Set<Verdict>(["same", "changed", "broken", "error"]);
const num = (v: unknown, fallback = 0) => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : fallback);
const str = (v: unknown, max: number) => (typeof v === "string" ? v.slice(0, max) : undefined);
const record = (v: unknown): Record<string, unknown> => (v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const usage = (v: unknown) => ({ input: num(record(v).input), output: num(record(v).output) });

/**
 * The report from a check run's text, or null when there is none or it isn't one we can read. Anyone who can
 * write checks in the repository can write this text, so every field is checked and rebuilt, never passed through.
 */
export function readReplayCheckText(text: string | null | undefined): PublishedReplay | null {
  if (!text) return null;
  const at = text.indexOf(`<!-- ${MARK} `);
  if (at < 0) return null;
  const end = text.indexOf(" -->", at);
  if (end < 0) return null;
  let data: unknown;
  try {
    data = JSON.parse(text.slice(at + MARK.length + 6, end));
  } catch {
    return null;
  }
  const d = record(data);
  const totals = record(d.totals);
  const oldModel = str(d.oldModel, 200);
  const newModel = str(d.newModel, 200);
  if (d.v !== 1 || !oldModel || !newModel || !Array.isArray(d.items)) return null;
  if (!["same", "changed", "broken", "error"].every((k) => typeof totals[k] === "number" && Number.isFinite(totals[k]))) return null;

  const items: PublishedItem[] = [];
  for (const raw of d.items.slice(0, 5000)) {
    const i = record(raw);
    const id = str(i.id, 200);
    if (id === undefined || !VERDICTS.has(i.verdict as Verdict)) return null;
    const sampleOld = str(i.sampleOld, 4000);
    const sampleNew = str(i.sampleNew, 4000);
    const error = str(i.error, 500);
    items.push({
      id,
      verdict: i.verdict as Verdict,
      reasons: (Array.isArray(i.reasons) ? i.reasons : []).flatMap((r) => (typeof r === "string" ? [r.slice(0, 500)] : [])).slice(0, 20),
      ...(sampleOld !== undefined ? { sampleOld } : {}),
      ...(sampleNew !== undefined ? { sampleNew } : {}),
      ...(error !== undefined ? { error } : {}),
    });
  }
  const cost = record(d.costPerCall);
  const prUrl = str(d.prUrl, 500);
  return {
    v: 1,
    oldModel,
    newModel,
    runs: num(d.runs, 1),
    calls: num(d.calls),
    failedCalls: num(d.failedCalls),
    totals: { same: num(totals.same), changed: num(totals.changed), broken: num(totals.broken), error: num(totals.error) },
    avgTokens: { old: usage(record(d.avgTokens).old), new: usage(record(d.avgTokens).new) },
    avgLatencyMs: { old: num(record(d.avgLatencyMs).old), new: num(record(d.avgLatencyMs).new) },
    ...(d.costPerCall !== undefined
      ? { costPerCall: { ...(typeof cost.old === "number" ? { old: num(cost.old) } : {}), ...(typeof cost.new === "number" ? { new: num(cost.new) } : {}) } }
      : {}),
    comparedPrompts: num(d.comparedPrompts),
    rateLimited: d.rateLimited === true,
    reroutes: (Array.isArray(d.reroutes) ? d.reroutes : []).slice(0, 10).flatMap((r) => {
      const x = record(r);
      const requested = str(x.requested, 200);
      const served = str(x.served, 200);
      return (x.side === "old" || x.side === "new") && requested && served ? [{ side: x.side, requested, served }] : [];
    }),
    items,
    omittedItems: num(d.omittedItems),
    ...(prUrl ? { prUrl } : {}),
  };
}

/**
 * The check run's conclusion, matching the job: it fails only on what replay-fail-on fails the job for (broken
 * prompts by default), and asks for a look when anything changed, broke or couldn't be compared.
 */
export function replayConclusion(report: Pick<ReplayReport, "totals">, failOn: "broken" | "changed" | "none" = "broken"): "success" | "neutral" | "failure" {
  const t = report.totals;
  if ((failOn === "broken" && t.broken > 0) || (failOn === "changed" && t.broken + t.changed > 0)) return "failure";
  return t.broken + t.changed + t.error > 0 ? "neutral" : "success";
}

/** "45 same · 3 changed · 2 broken", the check run's title. */
export function replayTitle(report: Pick<ReplayReport, "totals">): string {
  const t = report.totals;
  const parts = [`${t.same} same`, `${t.changed} changed`, `${t.broken} broken`];
  if (t.error) parts.push(`${t.error} could not compare`);
  return parts.join(" · ");
}
