import { parseModelRef, type LlmClient, type ProviderId } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import { buildReplayReport, type ReplayReport } from "./report.js";
import { plannedCalls, runReplay, SpendCapError } from "./run.js";

/**
 * Tested backup models. When a provider has an outage, a fallback only helps if it answers your
 * prompts well enough. This replays the golden set on the primary model once and on every
 * candidate, keeps the candidates with nothing broken (fewest changed answers first), and writes
 * a ready LiteLLM or OpenRouter fallback config with only those.
 */
export interface CandidateResult {
  model: string;
  report: ReplayReport;
  qualifies: boolean;
  /** Why a candidate was left out. */
  reason?: string;
}

export interface FallbackResult {
  primary: string;
  runs: number;
  calls: number;
  candidates: CandidateResult[];
  /** Qualifying candidates, best first. */
  fallbacks: string[];
  /** Prompts the primary never answered, so they could not be used to judge backups. */
  skippedPrompts: string[];
}

export interface FallbackOptions {
  items: GoldenItem[];
  primary: string;
  candidates: string[];
  client: LlmClient;
  runs?: number;
  concurrency?: number;
  maxCalls?: number;
  signal?: AbortSignal;
  /** Day the replay's cost per call is priced on (default: today). */
  today?: string;
}

export function plannedFallbackCalls(items: number, runs: number, candidates: number): number {
  return plannedCalls(items, runs, 1) * (1 + candidates);
}

/** Trimmed, de-duplicated candidates other than the primary (what the test actually runs). */
export function cleanCandidates(primary: string, candidates: string[]): string[] {
  return [...new Set(candidates.map((c) => c.trim()).filter(Boolean))].filter((c) => c !== primary);
}

/** A share this small means one broken run out of any number disqualifies a fallback. */
const ANY_RUN = 0.01;

export async function testFallbacks(options: FallbackOptions): Promise<FallbackResult> {
  const { primary, client } = options;
  const runs = options.runs ?? 3;
  const maxCalls = options.maxCalls ?? 500;
  parseModelRef(primary);
  const candidates = cleanCandidates(primary, options.candidates);
  candidates.forEach((c) => parseModelRef(c));
  if (candidates.length === 0) throw new Error("Give at least one candidate model that differs from the primary.");
  const planned = plannedFallbackCalls(options.items.length, runs, candidates.length);
  if (planned > maxCalls) {
    throw new SpendCapError(
      `Testing ${candidates.length} candidates needs ${planned} calls, above the limit of ${maxCalls}. Use fewer candidates or runs, or raise the limit.`,
    );
  }
  const common = { client, runs, ...(options.concurrency ? { concurrency: options.concurrency } : {}), ...(options.signal ? { signal: options.signal } : {}) };

  // The primary runs once; every candidate is compared against the same answers.
  const primaryRecords = await runReplay({ ...common, items: options.items, oldModel: primary, newModel: primary, sides: ["old"], maxCalls });
  // Without the primary's answers there is nothing to compare a backup with: stop before spending more.
  const answered = new Set(primaryRecords.filter((r) => r.response).map((r) => r.itemId));
  if (answered.size === 0) {
    throw new Error(`${primary} did not answer any prompt (${primaryRecords[0]?.error ?? "no calls"}). Fix that first: the other models are judged against its answers.`);
  }
  const items = options.items.filter((i) => answered.has(i.id));
  const skippedPrompts = options.items.filter((i) => !answered.has(i.id)).map((i) => i.id);
  const perCandidate = plannedCalls(items.length, runs, 1);

  // Candidates usually sit on different providers with their own rate limits, so they run side by side.
  const results = await Promise.all(
    candidates.map(async (model): Promise<CandidateResult & { calls: number }> => {
      const records = await runReplay({ ...common, items, oldModel: primary, newModel: model, sides: ["new"], maxCalls: perCandidate });
      const report = buildReplayReport(items, [...primaryRecords, ...records], { oldModel: primary, newModel: model, runs, judge: { share: ANY_RUN }, ...(options.today ? { today: options.today } : {}) });
      const t = report.totals;
      const rerouted = report.reroutes.filter((r) => r.side === "new");
      const reason =
        records.length > 0 && records.every((r) => r.error)
          ? `every call failed: ${(records[0]!.error ?? "").slice(0, 120)}`
          : rerouted.length > 0
            ? `served as ${rerouted.map((r) => r.served).join(", ")}, not the model requested`
            : t.broken > 0
              ? `${t.broken} prompt${t.broken === 1 ? "" : "s"} broken in at least one run`
              : t.error > 0
                ? `${t.error} prompt${t.error === 1 ? "" : "s"} could not be compared`
                : undefined;
      return { model, report, qualifies: !reason, ...(reason ? { reason } : {}), calls: records.length };
    }),
  );
  const fallbacks = results
    .filter((r) => r.qualifies)
    .sort((a, b) => a.report.totals.changed - b.report.totals.changed || a.report.avgLatencyMs.new - b.report.avgLatencyMs.new)
    .map((r) => r.model);
  return {
    primary,
    runs,
    calls: primaryRecords.length + results.reduce((n, r) => n + r.calls, 0),
    candidates: results.map(({ calls: _calls, ...r }) => r),
    fallbacks,
    skippedPrompts,
  };
}

// ---------- LiteLLM ----------

const LITELLM: Record<ProviderId, { prefix: string; key?: string; base?: string }> = {
  openai: { prefix: "openai", key: "OPENAI_API_KEY" },
  anthropic: { prefix: "anthropic", key: "ANTHROPIC_API_KEY" },
  google: { prefix: "gemini", key: "GEMINI_API_KEY" },
  mistral: { prefix: "mistral", key: "MISTRAL_API_KEY" },
  groq: { prefix: "groq", key: "GROQ_API_KEY" },
  openrouter: { prefix: "openrouter", key: "OPENROUTER_API_KEY" },
  ollama: { prefix: "ollama", base: "http://localhost:11434" },
};

const yamlString = (s: string) => JSON.stringify(s);

/** A LiteLLM proxy config.yaml: the primary and its tested fallbacks, tried in order. Keys come from the environment. */
export function litellmConfig(primary: string, fallbacks: string[], note?: string): string {
  const names = new Map<string, string>();
  const used = new Set<string>();
  for (const ref of [primary, ...fallbacks]) {
    const base = parseModelRef(ref).model.split("/").pop()!.replace(/[^\w.-]+/g, "-");
    let name = base;
    for (let n = 2; used.has(name); n++) name = `${base}-${n}`;
    used.add(name);
    names.set(ref, name);
  }
  const lines = [`# AI Change Guard: fallbacks that passed your golden set${note ? ` (${note})` : ""}`, "model_list:"];
  for (const ref of [primary, ...fallbacks]) {
    const { provider, model } = parseModelRef(ref);
    const p = LITELLM[provider];
    lines.push(`  - model_name: ${yamlString(names.get(ref)!)}`, "    litellm_params:", `      model: ${yamlString(`${p.prefix}/${model}`)}`);
    if (p.key) lines.push(`      api_key: ${yamlString(`os.environ/${p.key}`)}`);
    if (p.base) lines.push(`      api_base: ${yamlString(p.base)}`);
  }
  lines.push("", "router_settings:", `  fallbacks: [{${yamlString(names.get(primary)!)}: [${fallbacks.map((f) => yamlString(names.get(f)!)).join(", ")}]}]`, "");
  return lines.join("\n");
}

// ---------- OpenRouter ----------

const OPENROUTER_PREFIX: Partial<Record<ProviderId, string>> = { openai: "openai", anthropic: "anthropic", google: "google", mistral: "mistralai" };

/**
 * The OpenRouter "models" fallback list. OpenRouter names some models differently from the
 * providers (for example dots in version numbers), so ids not given as openrouter:... are listed
 * for the user to check against openrouter.ai/models.
 */
export function openrouterConfig(primary: string, fallbacks: string[]): { body: { models: string[] }; unchecked: string[]; skipped: string[] } {
  // The first id is the model OpenRouter tries first; dropping the primary would quietly promote a backup.
  if (parseModelRef(primary).provider !== "openrouter" && !OPENROUTER_PREFIX[parseModelRef(primary).provider]) {
    throw new Error(`OpenRouter does not serve ${primary}, so it cannot head an OpenRouter fallback list. Use --format litellm instead.`);
  }
  const models: string[] = [];
  const unchecked: string[] = [];
  const skipped: string[] = [];
  for (const ref of [primary, ...fallbacks]) {
    const { provider, model } = parseModelRef(ref);
    if (provider === "openrouter") models.push(model);
    else if (OPENROUTER_PREFIX[provider]) {
      const id = `${OPENROUTER_PREFIX[provider]}/${model}`;
      models.push(id);
      unchecked.push(id);
    } else skipped.push(ref);
  }
  if (models.length < 2) throw new Error("None of the tested backups is available on OpenRouter. Use --format litellm instead.");
  return { body: { models }, unchecked, skipped };
}
