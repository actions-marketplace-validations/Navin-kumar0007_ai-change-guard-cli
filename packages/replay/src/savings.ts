// Money-Finder (plan 8.9): cheaper models that answer your real prompts the same way. Picks cheaper candidates
// on the same provider from the catalog, replays the golden set on them against your model, and keeps a
// candidate only with nothing broken and at most 2% of answers changed. Savings use the tokens measured in
// the replay (so a tokenizer that uses more tokens counts) and, when given, your monthly call volume.
import { billedPriceOn, callCost, formatUsd, localToday, lookupModel, prices, tokenRatio, type Platform, type TokenPrice } from "@ai-changeguard/catalog";
import { parseModelRef, type LlmClient } from "@ai-changeguard/llm";
import { testFallbacks } from "./fallback.js";
import type { GoldenItem } from "./golden.js";
import { modelName, platformForRef, type ReplayReport } from "./report.js";

/** At most this share of prompts may change for a cheaper model to count as a safe switch. */
export const MAX_CHANGED_SHARE = 0.02;
/** A candidate must be at least this much cheaper per call to be worth a switch. */
const MIN_SAVING = 0.05;
/** Prices are judged at the worse of today and this many days ahead, so an introductory price ending soon can't win. */
const PRICE_HORIZON_DAYS = 60;
/** Models that retire or reroute within this many days are not suggested. */
const RETIREMENT_DAYS = 90;
/** Token mix used to rank candidates before the replay has measured anything. */
const DEFAULT_TOKENS = { input: 1000, output: 250 };

export interface Candidate {
  ref: string;
  /** Estimated cost per call, at the token mix given (tokenizer difference included). */
  costPerCall: number;
}

function addDays(date: string, days: number): string {
  return new Date(Date.parse(`${date}T00:00:00Z`) + days * 86_400_000).toISOString().slice(0, 10);
}

/** The higher of the price billed today and the price in PRICE_HORIZON_DAYS, or undefined if either is unknown. */
function steadyPrice(id: string, platform: Platform, today: string): TokenPrice | undefined {
  const now = billedPriceOn(id, platform, today);
  const later = billedPriceOn(id, platform, addDays(today, PRICE_HORIZON_DAYS));
  if (!now || !later) return undefined;
  return { input: Math.max(now.input, later.input), output: Math.max(now.output, later.output) };
}

/**
 * Cheaper models on the same provider (so the same API key works), closest in price first: those are the
 * likeliest to answer the same way. Skips previews, dated snapshot IDs (the alias is suggested instead),
 * models retiring soon, and models whose price in two months is unknown.
 */
/**
 * Why a model is not offered as a cheaper switch, or undefined when it may be: a preview, a dated snapshot
 * ID, a model that retires or reroutes within 90 days, or one whose price two months out is unknown.
 */
export function candidateProblem(ref: string, today: string): string | undefined {
  const platform = platformForRef(ref);
  const id = modelName(ref);
  if (!platform) return "not priced in the catalog (a gateway or unknown provider)";
  if (/preview/.test(id)) return "a preview model";
  if (/-\d{8}$|-\d{4}-\d{2}-\d{2}$/.test(id)) return "a dated snapshot ID; test its alias instead";
  const retiring = lookupModel(id).find((m) => m.platform === platform && m.retiresOn <= addDays(today, RETIREMENT_DAYS));
  if (retiring) return `${retiring.reroutesTo ? "reroutes" : "retires"} on ${retiring.retiresOn}`;
  if (!steadyPrice(id, platform, today)) return "its price two months from now is not known";
  return undefined;
}

/** Cost of a call at the steady price (the higher of today's and the price in two months). */
export function steadyCost(ref: string, tokens: { input: number; output: number }, today: string): number | undefined {
  const platform = platformForRef(ref);
  const price = platform ? steadyPrice(modelName(ref), platform, today) : undefined;
  return price ? callCost(price, tokens.input, tokens.output) : undefined;
}

export function cheaperCandidates(primary: string, today: string, options: { tokens?: { input: number; output: number }; count?: number } = {}): Candidate[] {
  const { provider } = parseModelRef(primary);
  const platform = platformForRef(primary);
  if (!platform) return [];
  const id = modelName(primary);
  const tokens = options.tokens ?? DEFAULT_TOKENS;
  const current = steadyPrice(id, platform, today) ?? billedPriceOn(id, platform, today);
  if (!current) return [];
  const currentCost = callCost(current, tokens.input, tokens.output);

  const out: Candidate[] = [];
  for (const p of prices) {
    if (p.platform !== platform || p.id === id) continue;
    // Closed to new API users: most accounts would get a 404. Name it with --candidates if yours has access.
    if (p.availability === "existing-users") continue;
    if (candidateProblem(`${provider}:${p.id}`, today)) continue;
    const price = steadyPrice(p.id, platform, today)!;
    const ratio = tokenRatio({ id, platform }, { id: p.id, platform });
    const cost = callCost(price, tokens.input * ratio, tokens.output * ratio);
    if (cost <= currentCost * (1 - MIN_SAVING)) out.push({ ref: `${provider}:${p.id}`, costPerCall: cost });
  }
  return out.sort((a, b) => b.costPerCall - a.costPerCall).slice(0, options.count ?? 2);
}

export interface SavingsCandidate {
  model: string;
  qualifies: boolean;
  reason?: string;
  report: ReplayReport;
  /** Share of compared prompts whose answers changed. */
  changedShare: number;
  /** US dollars per call at list price, from the replay's measured tokens. */
  costPerCall?: { current: number; candidate: number };
  /** Share of the per-call cost saved (0.7 = 70% cheaper). */
  savingShare?: number;
  /** Dollars a month saved at the call volume given. */
  monthlySaving?: number;
}

export interface SavingsResult {
  primary: string;
  runs: number;
  calls: number;
  callsPerMonth?: number;
  candidates: SavingsCandidate[];
  /** Candidates not tested, and why (no calls spent on them). */
  excluded: { model: string; reason: string }[];
  /** Prompts the current model did not answer, so they could not be compared. */
  skippedPrompts: string[];
  /** The qualifying candidate that saves the most per call. */
  best?: SavingsCandidate;
}

export interface SavingsOptions {
  items: GoldenItem[];
  primary: string;
  client: LlmClient;
  /** provider:model candidates; picked from the catalog when left out. */
  candidates?: string[];
  callsPerMonth?: number;
  runs?: number;
  concurrency?: number;
  maxCalls?: number;
  today?: string;
  signal?: AbortSignal;
}

export async function findSavings(options: SavingsOptions): Promise<SavingsResult> {
  const today = options.today ?? localToday();
  const runs = options.runs ?? 3;
  const asked = options.candidates ?? cheaperCandidates(options.primary, today).map((c) => c.ref);
  // Given candidates get the same checks as picked ones, before any call is spent.
  const excluded = asked.flatMap((model) => {
    const reason = candidateProblem(model, today);
    return reason ? [{ model, reason }] : [];
  });
  const candidates = asked.filter((m) => !excluded.some((e) => e.model === m));
  const base = { primary: options.primary, runs, excluded, ...(options.callsPerMonth !== undefined ? { callsPerMonth: options.callsPerMonth } : {}) };
  if (candidates.length === 0) return { ...base, calls: 0, candidates: [], skippedPrompts: [] };

  const tested = await testFallbacks({
    items: options.items,
    primary: options.primary,
    candidates,
    client: options.client,
    runs,
    ...(options.concurrency ? { concurrency: options.concurrency } : {}),
    ...(options.maxCalls ? { maxCalls: options.maxCalls } : {}),
    ...(options.signal ? { signal: options.signal } : {}),
    today,
  });

  const results = tested.candidates.map((c): SavingsCandidate => {
    const compared = c.report.items.length || 1;
    const changedShare = c.report.totals.changed / compared;
    // Priced at the steady price on the tokens each model really used, so a promotion ending soon can't win.
    const current = steadyCost(options.primary, c.report.avgTokens.old, today) ?? c.report.costPerCall?.old;
    const candidate = steadyCost(c.model, c.report.avgTokens.new, today);
    const priced = current !== undefined && candidate !== undefined && c.report.comparedPrompts > 0 ? { current, candidate } : undefined;
    const savingShare = priced && priced.current > 0 ? 1 - priced.candidate / priced.current : undefined;
    // Say which prompt broke and how, so a skip can be checked by hand.
    const firstBroken = c.report.items.find((i) => i.verdict === "broken");
    const brokenDetail = firstBroken ? ` (first: ${firstBroken.id}, ${(firstBroken.reasons[0]?.detail ?? firstBroken.error ?? "").slice(0, 120)})` : "";
    const reason =
      (c.reason ? `${c.reason}${c.report.totals.broken > 0 ? brokenDetail : ""}` : undefined) ??
      (changedShare > MAX_CHANGED_SHARE
        ? `${c.report.totals.changed} of ${compared} prompts answered differently (more than ${Math.round(MAX_CHANGED_SHARE * 100)}%)`
        : !priced || savingShare === undefined
          ? "no list price known for one of the two models"
          : savingShare < 0
            ? `costs more on your prompts (+${Math.round(-savingShare * 100)}% per call)`
            : savingShare < MIN_SAVING
              ? `only ${Math.round(savingShare * 100)}% cheaper on your prompts`
              : undefined);
    return {
      model: c.model,
      qualifies: !reason,
      ...(reason ? { reason } : {}),
      report: c.report,
      changedShare,
      ...(priced ? { costPerCall: priced } : {}),
      ...(savingShare !== undefined ? { savingShare } : {}),
      ...(priced && options.callsPerMonth !== undefined ? { monthlySaving: (priced.current - priced.candidate) * options.callsPerMonth } : {}),
    };
  });
  const best = results.filter((r) => r.qualifies).sort((a, b) => (b.savingShare ?? 0) - (a.savingShare ?? 0))[0];
  return { ...base, calls: tested.calls, candidates: results, skippedPrompts: tested.skippedPrompts, ...(best ? { best } : {}) };
}

function usd(amount: number): string {
  return amount >= 100 ? `$${Math.round(amount).toLocaleString("en-US")}` : formatUsd(amount);
}

function verdictLine(c: SavingsCandidate): string {
  const t = c.report.totals;
  return `${t.same} same · ${t.changed} changed · ${t.broken} broken on ${c.report.items.length} prompts`;
}

function savingText(c: SavingsCandidate): string {
  if (!c.costPerCall || c.savingShare === undefined) return "";
  const pct = Math.round(c.savingShare * 100);
  const month = c.monthlySaving !== undefined ? `, about ${usd(c.monthlySaving)} a month` : "";
  return `${pct}% cheaper per call (${usd(c.costPerCall.current)} → ${usd(c.costPerCall.candidate)})${month}`;
}

function skippedNote(result: SavingsResult): string | undefined {
  const n = result.skippedPrompts.length;
  return n ? `${result.primary} did not answer ${n} prompt${n === 1 ? "" : "s"} (${result.skippedPrompts.slice(0, 5).join(", ")}${n > 5 ? ", ..." : ""}), so ${n === 1 ? "it was" : "they were"} not compared.` : undefined;
}

/** Text report. Pass volumeHint: false when a volume was asked for (e.g. a log) but none was found. */
export function formatSavingsText(result: SavingsResult, options: { volumeHint?: boolean } = {}): string {
  const out: string[] = [`Money-Finder: cheaper models for ${result.primary}`];
  for (const e of result.excluded) out.push(`  NOT TESTED  ${e.model}: ${e.reason}`);
  if (result.candidates.length === 0) {
    out.push(result.excluded.length ? "No candidate left to test." : "No cheaper model on the same provider to test (or no list price known for yours).");
    return out.join("\n");
  }
  if (result.best) {
    out.push(`SAVE  ${result.primary} → ${result.best.model}: ${savingText(result.best)}`);
    out.push(`      ${verdictLine(result.best)} (${result.runs} run${result.runs === 1 ? "" : "s"} each)`);
    out.push(`      Switch: ai-change-guard switch . --from ${modelName(result.primary)} --to ${modelName(result.best.model)} --write`);
  } else {
    out.push("No cheaper model answered your prompts the same way. Keep the model you have.");
  }
  for (const c of result.candidates) {
    if (c === result.best) continue;
    out.push(`  ${c.qualifies ? "ALSO" : "SKIP"}  ${c.model}: ${c.qualifies ? savingText(c) : c.reason}`);
  }
  const skipped = skippedNote(result);
  if (skipped) out.push(`WARNING: ${skipped}`);
  if (result.callsPerMonth === undefined && result.best && options.volumeHint !== false) out.push("Add --calls-per-month or --log ai-calls.jsonl to see the monthly saving.");
  out.push(`${result.calls} calls used. Costs use list prices, at the higher of today's and the price two months out (no batch, cache or volume discounts).`);
  return out.join("\n");
}

/** The savings card: Markdown for a pull request or the Action's Summary. */
export function formatSavingsMarkdown(result: SavingsResult): string {
  const out: string[] = [`## Money-Finder: \`${result.primary}\``, ""];
  if (result.candidates.length === 0) {
    return [...out, "No cheaper model on the same provider to test.", ...result.excluded.map((e) => `- Not tested: \`${e.model}\`: ${e.reason}`)].join("\n");
  }
  if (result.best) {
    out.push(`**Switch to \`${result.best.model}\`: ${savingText(result.best)}.**`, "", `On your real prompts: ${verdictLine(result.best)}, ${result.runs} runs each.`);
  } else {
    out.push("**No cheaper model answered your prompts the same way.** Keep the model you have.");
  }
  out.push("", "| Candidate | Result | Cost per call |", "| --- | --- | --- |");
  for (const c of result.candidates) {
    const cost = c.costPerCall ? `${usd(c.costPerCall.current)} → ${usd(c.costPerCall.candidate)}` : "unknown";
    out.push(`| \`${c.model}\` | ${c.qualifies ? `Passes: ${verdictLine(c)}` : `Skipped: ${c.reason}`} | ${cost} |`);
  }
  for (const e of result.excluded) out.push(`| \`${e.model}\` | Not tested: ${e.reason} | |`);
  const skipped = skippedNote(result);
  if (skipped) out.push("", `> **Warning:** ${skipped}`);
  out.push("", "List prices at the higher of today's and the price two months out; no batch, cache or volume discounts.");
  return out.join("\n");
}
