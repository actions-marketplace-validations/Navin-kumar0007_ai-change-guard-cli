// Monthly cost impact of announced price changes, from the app's real call volume: the recorder's
// ai-calls.jsonl (or numbers the user gives). Used by `ai-change-guard cost`.
import { billedPriceOn, callCost, defaultPlatform, lookupModel, lookupPrice, type Platform, type TokenPrice } from "@ai-changeguard/catalog";
import { buildCostRows, type CostRow } from "./costs.js";
import { classifyModelId } from "./detect.js";

export interface ModelUsage {
  model: string;
  platform?: Platform;
  /** Calls seen in the log (or given). */
  calls: number;
  /** Calls whose answer reported token counts (streamed answers are recorded without one). */
  callsWithTokens: number;
  inputTokens: number;
  outputTokens: number;
  first?: string;
  last?: string;
}

const HOST_PLATFORM: [RegExp, Platform][] = [
  [/(^|\.)api\.openai\.com$/, "openai-api"],
  [/\.openai\.azure\.com$/, "azure-openai"],
  [/(^|\.)api\.anthropic\.com$/, "anthropic-api"],
  [/(^|\.)generativelanguage\.googleapis\.com$/, "gemini-api"],
  [/aiplatform\.googleapis\.com$/, "vertex-ai"],
  [/(^|\.)api\.mistral\.ai$/, "mistral-api"],
];

function platformOf(url: unknown): Platform | undefined {
  if (typeof url !== "string") return undefined;
  try {
    const host = new URL(url).hostname.toLowerCase();
    return HOST_PLATFORM.find(([re]) => re.test(host))?.[1];
  } catch {
    return undefined;
  }
}

type Json = Record<string, unknown>;
const obj = (v: unknown): Json | undefined => (v && typeof v === "object" && !Array.isArray(v) ? (v as Json) : undefined);
const num = (v: unknown): number | undefined => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : undefined);

/**
 * Token counts from an OpenAI (Chat or Responses), Anthropic or Gemini answer. Input counts every input
 * token, cached ones included: Anthropic reports cache reads and writes apart from input_tokens.
 */
export function tokensOf(response: unknown): { input: number; output: number } | undefined {
  const r = obj(response);
  const usage = obj(r?.usage);
  if (usage) {
    const base = num(usage.prompt_tokens) ?? num(usage.input_tokens);
    const output = num(usage.completion_tokens) ?? num(usage.output_tokens);
    const cached = (num(usage.cache_read_input_tokens) ?? 0) + (num(usage.cache_creation_input_tokens) ?? 0);
    if (base !== undefined && output !== undefined) return { input: base + cached, output };
  }
  const meta = obj(r?.usageMetadata);
  if (meta) {
    const input = num(meta.promptTokenCount);
    const output = (num(meta.candidatesTokenCount) ?? 0) + (num(meta.thoughtsTokenCount) ?? 0);
    if (input !== undefined) return { input, output };
  }
  return undefined;
}

function modelOf(entry: Json): string | undefined {
  const raw = typeof entry.model === "string" ? entry.model : typeof obj(entry.request)?.model === "string" ? (obj(entry.request)!.model as string) : undefined;
  return raw?.replace(/^models\//, "");
}

/** Calls that are not billed as a generation: token counting, embeddings, moderation. */
const NOT_GENERATION = /\/count_tokens\b|:countTokens\b|\/embeddings\b|:(?:batchE|e)mbedContents?\b|\/moderations\b/i;

/** The platform a model ID usually runs on, when the log's host does not say (e.g. a custom gateway). */
export function guessPlatform(model: string): Platform | undefined {
  const hit = classifyModelId(model);
  return hit ? (hit.platformHint ?? defaultPlatform[hit.provider]) : undefined;
}

/** Reads the recorder's JSONL: one call per line. Lines that are not recorder calls are skipped. */
export function usageFromLog(text: string): ModelUsage[] {
  const byModel = new Map<string, ModelUsage>();
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    let entry: Json | undefined;
    try {
      entry = obj(JSON.parse(line));
    } catch {
      continue;
    }
    if (!entry) continue;
    if (typeof entry.url === "string" && NOT_GENERATION.test(entry.url)) continue;
    const model = modelOf(entry);
    if (!model) continue;
    const platform = platformOf(entry.url);
    const key = `${model}|${platform ?? ""}`;
    let u = byModel.get(key);
    if (!u) {
      u = { model, calls: 0, callsWithTokens: 0, inputTokens: 0, outputTokens: 0 };
      if (platform) u.platform = platform;
      byModel.set(key, u);
    }
    u.calls++;
    const tokens = tokensOf(entry.response);
    if (tokens) {
      u.callsWithTokens++;
      u.inputTokens += tokens.input;
      u.outputTokens += tokens.output;
    }
    const at = typeof entry.recorded_at === "string" && !Number.isNaN(Date.parse(entry.recorded_at)) ? entry.recorded_at : undefined;
    if (at) {
      if (!u.first || at < u.first) u.first = at;
      if (!u.last || at > u.last) u.last = at;
    }
  }
  return [...byModel.values()].sort((a, b) => b.calls - a.calls);
}

export interface ImpactRow {
  model: string;
  platform: Platform;
  callsPerMonth: number;
  avgInputTokens: number;
  avgOutputTokens: number;
  /** At today's price; missing when the catalog has no price. */
  monthlyNow?: number;
  /** inEffect: the change already applies, so monthlyNow already includes it and delta is what it added. */
  changes: { row: CostRow; inEffect: boolean; monthlyAfter?: number; delta?: number }[];
}

export interface ImpactSummary {
  /** Days the log covers (at least 1), used to scale to a month. */
  days?: number;
  sample: number;
  rows: ImpactRow[];
  /** Models in the log with no platform or no catalog price. */
  unpriced: string[];
}

const DAY_MS = 86_400_000;

/** Days between the first and last call of all usage, rounded up to at least one day. */
export function logDays(usage: ModelUsage[]): number | undefined {
  const firsts = usage.flatMap((u) => (u.first ? [Date.parse(u.first)] : []));
  const lasts = usage.flatMap((u) => (u.last ? [Date.parse(u.last)] : []));
  if (!firsts.length || !lasts.length) return undefined;
  return Math.max(1, (Math.max(...lasts) - Math.min(...firsts)) / DAY_MS);
}

/**
 * Monthly cost now and after each announced change. `perMonth` marks usage that is already per month
 * (numbers given by hand); otherwise the log is scaled from its own time span to 30 days, and divided
 * by the recorder's sample share.
 */
export function estimateImpact(
  usage: ModelUsage[],
  options: { today: string; sample?: number; perMonth?: boolean; platformFor?: (model: string) => Platform | undefined },
): ImpactSummary {
  const sample = options.sample ?? 1;
  const days = options.perMonth ? undefined : logDays(usage);
  const scale = options.perMonth ? 1 : (30 / (days ?? 30)) / sample;
  const rows: ImpactRow[] = [];
  const unpriced: string[] = [];

  for (const u of usage) {
    const platform = u.platform ?? options.platformFor?.(u.model) ?? guessPlatform(u.model);
    if (!platform || !lookupPrice(u.model, platform)) {
      unpriced.push(u.model);
      continue;
    }
    const avgIn = u.callsWithTokens ? u.inputTokens / u.callsWithTokens : 0;
    const avgOut = u.callsWithTokens ? u.outputTokens / u.callsWithTokens : 0;
    const callsPerMonth = u.calls * scale;
    const monthly = (price: TokenPrice | undefined) => (price && u.callsWithTokens ? callCost(price, avgIn, avgOut) * callsPerMonth : undefined);
    const now = monthly(billedPriceOn(u.model, platform, options.today));
    const reroute = lookupModel(u.model).find((e) => e.platform === platform && e.reroutesTo);
    const costs = buildCostRows(
      [{ id: u.model, platform, locations: [], ...(reroute?.reroutesTo ? { reroute: { on: reroute.retiresOn, to: reroute.reroutesTo } } : {}) }],
      options.today,
    );
    const row: ImpactRow = {
      model: u.model,
      platform,
      callsPerMonth,
      avgInputTokens: avgIn,
      avgOutputTokens: avgOut,
      changes: costs.map((c) => {
        const inEffect = c.daysLeft <= 0;
        const after = c.after ? monthly(c.after) : undefined;
        const before = monthly(c.before);
        return after !== undefined && before !== undefined ? { row: c, inEffect, monthlyAfter: after, delta: after - before } : { row: c, inEffect };
      }),
    };
    if (now !== undefined) row.monthlyNow = now;
    rows.push(row);
  }
  return { ...(days !== undefined ? { days } : {}), sample, rows, unpriced };
}
