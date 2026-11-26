// Verified Prompt Slimming (plan 8.14, the paid half of Token Saver): a shorter system prompt that answers your
// real prompts the same way. A model you choose rewrites each system prompt shorter; the golden set is replayed
// on your model with the old and the new prompt, and the shorter one is proposed only when nothing broke and at
// most 2% of answers changed: savings with proof, like Money-Finder.
import { formatUsd, localToday } from "@ai-changeguard/catalog";
import { parseModelRef, type LlmClient } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import { buildReplayReport, type ReplayReport } from "./report.js";
import { plannedCalls, runReplay, SpendCapError } from "./run.js";
import { MAX_CHANGED_SHARE, steadyCost } from "./savings.js";
import { extractJson } from "./schema.js";

/** Prompts that must use one system prompt before a shorter one can be proved on them. */
export const MIN_SLIM_ITEMS = 3;
/** A rewrite must be at least this much shorter (in characters) to be worth testing. */
const MIN_SHRINK = 0.1;
/** And must save at least this share of the cost (or input tokens) per call, measured. */
const MIN_SAVING = 0.03;
/** One broken run out of any number disqualifies a rewrite. */
const ANY_RUN = 0.01;

export type SlimStyle = "concise" | "minimal";
export const SLIM_STYLES: SlimStyle[] = ["concise", "minimal"];
const TARGET: Record<SlimStyle, string> = {
  concise: "about 30% shorter",
  minimal: "as short as it can be, about half its length or less",
};

const PLACEHOLDER_RE = /\$\{[^}]*\}|\{\{[^{}]*\}\}|\{[A-Za-z_][\w.]*\}/g;

/** Template placeholders ({name}, {{name}}, ${name}), sorted, so two prompts can be compared. */
export function placeholders(text: string): string[] {
  return [...text.matchAll(PLACEHOLDER_RE)].map((m) => m[0]).sort();
}

export function slimRequest(system: string, style: SlimStyle): string {
  return [
    "Below is the system prompt of one feature in an app. It is sent with every call, so every word costs money.",
    "",
    "<system_prompt>",
    system,
    "</system_prompt>",
    "",
    `Rewrite it ${TARGET[style]}, without changing what the model does or how it answers.`,
    "Keep: every rule, limit and refusal; the output format, JSON keys and field names; the structure of any examples;",
    "every template placeholder exactly as written ({name}, {{name}}, ${name}); names, numbers and links; the language it is written in.",
    "Remove: repetition, filler, politeness, explanations meant for people, and words that do not change the answer.",
    'Reply with only JSON: {"prompt": "..."}',
  ].join("\n");
}

/** The rewritten prompt in a writer answer ({"prompt": "..."}), or undefined. */
export function parseSlimAnswer(text: string): string | undefined {
  const pick = (value: unknown) => {
    const p = value && typeof value === "object" ? (value as { prompt?: unknown }).prompt : undefined;
    return typeof p === "string" && p.trim() ? p.trim() : undefined;
  };
  const parsed = extractJson(text);
  if (parsed.ok) return pick(parsed.value);
  const m = /\{[\s\S]*\}/.exec(text);
  const retry = m ? extractJson(m[0]) : parsed;
  return retry.ok ? pick(retry.value) : undefined;
}

/** Why a rewrite can't be tested, or undefined when it can: too little shorter, or placeholders changed. */
export function rewriteProblem(original: string, rewrite: string): string | undefined {
  if (!rewrite.trim()) return "the rewrite is empty";
  if (rewrite.length > original.length * (1 - MIN_SHRINK)) return `only ${Math.max(0, Math.round((1 - rewrite.length / original.length) * 100))}% shorter`;
  const a = placeholders(original).join("\n");
  const b = placeholders(rewrite).join("\n");
  if (a !== b) return "the rewrite changed the template placeholders";
  return undefined;
}

export interface SlimVariant {
  style: SlimStyle;
  /** The rewritten system prompt (empty when the writer gave none). */
  prompt: string;
  qualifies: boolean;
  reason?: string;
  report?: ReplayReport;
  changedShare?: number;
  /** Average input tokens per call, measured in the replay. */
  inputTokens?: { old: number; new: number };
  /** US dollars per call at list price (the higher of today's and the price two months out), measured tokens. */
  costPerCall?: { current: number; candidate: number };
  /** Share of the cost per call saved, or of the input tokens when the model has no list price. */
  savingShare?: number;
  /** Dollars a month saved, from the calls per month given and this prompt's share of the golden set. */
  monthlySaving?: number;
}

export interface SlimPrompt {
  /** The system prompt as the golden set has it. */
  system: string;
  /** Prompts in the golden set that use it. */
  items: number;
  variants: SlimVariant[];
  /** Why it was not tested (too few prompts, over the call limit, the model never answered). */
  reason?: string;
  /** Prompts your model did not answer with the original, so they were not compared. */
  skippedPrompts: string[];
  /** The qualifying rewrite that saves the most. */
  best?: SlimVariant;
}

export interface SlimResult {
  model: string;
  writer: string;
  runs: number;
  calls: number;
  callsPerMonth?: number;
  prompts: SlimPrompt[];
  /** Golden-set prompts with no system prompt, which have nothing to slim. */
  withoutSystem: number;
}

export interface SlimOptions {
  items: GoldenItem[];
  /** The model the app uses (provider:model): both sides of every replay run on it. */
  model: string;
  /** The model that writes the shorter prompts (default: `model`). */
  writer?: string;
  client: LlmClient;
  styles?: SlimStyle[];
  /** Most system prompts to slim, the most-used first (default 3). */
  maxPrompts?: number;
  callsPerMonth?: number;
  runs?: number;
  concurrency?: number;
  maxCalls?: number;
  today?: string;
  signal?: AbortSignal;
}

/** Calls a slimming run spends at most: one writer call and one replay per style, plus your model once per prompt. */
export function plannedSlimCalls(itemsPerPrompt: number[], runs: number, styles: number): number {
  return itemsPerPrompt.reduce((n, items) => n + styles + plannedCalls(items, runs, 1) * (1 + styles), 0);
}

function groupBySystem(items: GoldenItem[]): { system: string; items: GoldenItem[] }[] {
  const groups = new Map<string, GoldenItem[]>();
  for (const item of items) {
    if (!item.system?.trim()) continue;
    const list = groups.get(item.system);
    if (list) list.push(item);
    else groups.set(item.system, [item]);
  }
  return [...groups].map(([system, list]) => ({ system, items: list })).sort((a, b) => b.items.length - a.items.length);
}

/** The prompt counts of the system prompts a run will slim: the most-used first, each with enough prompts. */
export function slimTargets(items: GoldenItem[], maxPrompts = 3): number[] {
  return groupBySystem(items)
    .filter((g) => g.items.length >= MIN_SLIM_ITEMS)
    .slice(0, maxPrompts)
    .map((g) => g.items.length);
}

export async function slimPrompts(options: SlimOptions): Promise<SlimResult> {
  const { model, client } = options;
  const writer = options.writer ?? model;
  parseModelRef(model);
  parseModelRef(writer);
  const today = options.today ?? localToday();
  const runs = options.runs ?? 3;
  const styles = [...new Set(options.styles ?? SLIM_STYLES)];
  const maxCalls = options.maxCalls ?? 500;
  const groups = groupBySystem(options.items);
  const withSystem = groups.reduce((n, g) => n + g.items.length, 0);
  const common = { client, runs, ...(options.concurrency ? { concurrency: options.concurrency } : {}), ...(options.signal ? { signal: options.signal } : {}) };

  // Which prompts are tested is decided before any call, so the limit is checked on the real plan.
  const prompts: SlimPrompt[] = [];
  const testable: { group: (typeof groups)[number]; out: SlimPrompt }[] = [];
  for (const group of groups) {
    const out: SlimPrompt = { system: group.system, items: group.items.length, variants: [], skippedPrompts: [] };
    prompts.push(out);
    if (group.items.length < MIN_SLIM_ITEMS) out.reason = `only ${group.items.length} prompt${group.items.length === 1 ? "" : "s"} use${group.items.length === 1 ? "s" : ""} it; at least ${MIN_SLIM_ITEMS} are needed to prove a shorter one`;
    else if (testable.length >= (options.maxPrompts ?? 3)) out.reason = "not tested this time (more system prompts than --max-prompts)";
    else testable.push({ group, out });
  }
  const planned = plannedSlimCalls(testable.map((t) => t.group.items.length), runs, styles.length);
  if (planned > maxCalls) {
    throw new SpendCapError(`Slimming ${testable.length} system prompt${testable.length === 1 ? "" : "s"} needs up to ${planned} calls, above the limit of ${maxCalls}. Use fewer runs or prompts, or raise the limit.`);
  }

  let calls = 0;
  for (const { group, out } of testable) {
    const oldRecords = await runReplay({ ...common, items: group.items, oldModel: model, newModel: model, sides: ["old"], maxCalls });
    calls += oldRecords.length;
    const answered = new Set(oldRecords.filter((r) => r.response).map((r) => r.itemId));
    if (answered.size === 0) {
      out.reason = `${model} did not answer any of its prompts (${(oldRecords[0]?.error ?? "no calls").slice(0, 120)})`;
      continue;
    }
    const items = group.items.filter((i) => answered.has(i.id));
    out.skippedPrompts = group.items.filter((i) => !answered.has(i.id)).map((i) => i.id);

    const tried = new Set<string>();
    for (const style of styles) {
      let prompt = "";
      try {
        calls++;
        const res = await client({
          model: writer,
          system: "You shorten the system prompts of AI apps without changing what they make the model do.",
          messages: [{ role: "user", content: slimRequest(group.system, style) }],
          maxTokens: Math.min(8000, Math.max(1000, Math.ceil(group.system.length / 2))),
          jsonMode: true,
        });
        prompt = parseSlimAnswer(res.text) ?? "";
      } catch (error) {
        out.variants.push({ style, prompt: "", qualifies: false, reason: `the writer failed: ${(error instanceof Error ? error.message : String(error)).slice(0, 120)}` });
        continue;
      }
      const problem = !prompt ? "the writer did not return a prompt" : tried.has(prompt) ? "same rewrite as the one before" : rewriteProblem(group.system, prompt);
      if (problem) {
        out.variants.push({ style, prompt, qualifies: false, reason: problem });
        continue;
      }
      tried.add(prompt);

      const newItems = items.map((i) => ({ ...i, system: prompt }));
      const records = await runReplay({ ...common, items, newItems, oldModel: model, newModel: model, sides: ["new"], maxCalls: plannedCalls(items.length, runs, 1) });
      calls += records.length;
      const report = buildReplayReport(items, [...oldRecords, ...records], { oldModel: model, newModel: model, runs, judge: { share: ANY_RUN }, today });
      const t = report.totals;
      const compared = report.items.length || 1;
      const changedShare = t.changed / compared;
      const inputTokens = { old: report.avgTokens.old.input, new: report.avgTokens.new.input };
      const current = steadyCost(model, report.avgTokens.old, today);
      const candidate = steadyCost(model, report.avgTokens.new, today);
      const priced = current !== undefined && candidate !== undefined && report.comparedPrompts > 0 ? { current, candidate } : undefined;
      const savingShare = priced ? (priced.current > 0 ? 1 - priced.candidate / priced.current : undefined) : inputTokens.old > 0 && report.comparedPrompts > 0 ? 1 - inputTokens.new / inputTokens.old : undefined;
      const firstBroken = report.items.find((i) => i.verdict === "broken");
      const reason =
        records.length > 0 && records.every((r) => r.error)
          ? `every call failed: ${(records[0]!.error ?? "").slice(0, 120)}`
          : t.broken > 0
            ? `${t.broken} prompt${t.broken === 1 ? "" : "s"} broken in at least one run (first: ${firstBroken!.id}, ${(firstBroken!.reasons[0]?.detail ?? firstBroken!.error ?? "").slice(0, 120)})`
            : t.error > 0
              ? `${t.error} prompt${t.error === 1 ? "" : "s"} could not be compared`
              : changedShare > MAX_CHANGED_SHARE
                ? `${t.changed} of ${compared} prompts answered differently (more than ${Math.round(MAX_CHANGED_SHARE * 100)}%)`
                : savingShare === undefined
                  ? "no token counts came back, so the saving could not be measured"
                  : savingShare < MIN_SAVING
                    ? savingShare < 0
                      ? `costs more per call on your prompts (+${Math.round(-savingShare * 100)}%): longer answers outweigh the shorter prompt`
                      : `only ${Math.round(savingShare * 100)}% cheaper per call on your prompts`
                    : undefined;
      const share = withSystem > 0 ? group.items.length / withSystem : 0;
      out.variants.push({
        style,
        prompt,
        qualifies: !reason,
        ...(reason ? { reason } : {}),
        report,
        changedShare,
        inputTokens,
        ...(priced ? { costPerCall: priced } : {}),
        ...(savingShare !== undefined ? { savingShare } : {}),
        ...(priced && options.callsPerMonth !== undefined ? { monthlySaving: (priced.current - priced.candidate) * options.callsPerMonth * share } : {}),
      });
    }
    const best = out.variants.filter((v) => v.qualifies).sort((a, b) => (b.savingShare ?? 0) - (a.savingShare ?? 0))[0];
    if (best) out.best = best;
  }
  return { model, writer, runs, calls, prompts, withoutSystem: options.items.length - withSystem, ...(options.callsPerMonth !== undefined ? { callsPerMonth: options.callsPerMonth } : {}) };
}

// ---------- Reports ----------

function usd(amount: number): string {
  return amount >= 100 ? `$${Math.round(amount).toLocaleString("en-US")}` : formatUsd(amount);
}

function preview(text: string, max = 70): string {
  const one = text.replace(/\s+/g, " ").trim();
  return one.length > max ? `${one.slice(0, max - 1)}…` : one;
}

function savingText(v: SlimVariant): string {
  const pct = Math.round((v.savingShare ?? 0) * 100);
  const tokens = v.inputTokens ? `${Math.round(v.inputTokens.old)} → ${Math.round(v.inputTokens.new)} input tokens per call` : "";
  const cost = v.costPerCall ? `, ${usd(v.costPerCall.current)} → ${usd(v.costPerCall.candidate)} per call` : "";
  const month = v.monthlySaving !== undefined ? `, about ${usd(v.monthlySaving)} a month` : "";
  return `${pct}% ${v.costPerCall ? "cheaper per call" : "fewer input tokens"} (${tokens}${cost}${month})`;
}

function verdictLine(v: SlimVariant): string {
  const t = v.report!.totals;
  return `${t.same} same · ${t.changed} changed · ${t.broken} broken on ${v.report!.items.length} prompts`;
}

export function formatSlimText(result: SlimResult): string {
  const out = [`Prompt slimming on ${result.model} (rewrites by ${result.writer}, ${result.runs} run${result.runs === 1 ? "" : "s"} per prompt)`];
  if (result.prompts.length === 0) out.push("No prompt in the golden set has a system prompt, so there is nothing to slim.");
  for (const p of result.prompts) {
    out.push("", `System prompt "${preview(p.system)}" (${p.system.length} characters, ${p.items} prompt${p.items === 1 ? "" : "s"})`);
    if (p.reason) {
      out.push(`  NOT TESTED: ${p.reason}`);
      continue;
    }
    if (p.best) {
      out.push(`  SAVE  ${p.best.style} rewrite, ${p.best.prompt.length} characters: ${savingText(p.best)}`, `        ${verdictLine(p.best)}`, "        New prompt:");
      out.push(...p.best.prompt.split("\n").map((l) => `          ${l}`));
    } else out.push("  No shorter prompt answered your prompts the same way. Keep the prompt you have.");
    for (const v of p.variants) if (v !== p.best) out.push(`  ${v.qualifies ? "ALSO" : "SKIP"}  ${v.style}: ${v.qualifies ? savingText(v) : v.reason}`);
    if (p.skippedPrompts.length) out.push(`  WARNING: ${result.model} did not answer ${p.skippedPrompts.length} prompt(s) (${p.skippedPrompts.slice(0, 5).join(", ")}), so they were not compared.`);
  }
  if (result.withoutSystem > 0) out.push("", `${result.withoutSystem} prompt(s) have no system prompt and were left out.`);
  if (result.callsPerMonth === undefined && result.prompts.some((p) => p.best?.costPerCall)) out.push("Add --calls-per-month or --log ai-calls.jsonl to see the monthly saving.");
  out.push(`${result.calls} calls used. Costs use list prices, at the higher of today's and the price two months out (no batch, cache or volume discounts).`);
  return out.join("\n");
}

/** The slimming card: Markdown for a pull request or the Action's Summary. */
export function formatSlimMarkdown(result: SlimResult): string {
  const out = [`## Prompt slimming: \`${result.model}\``, ""];
  if (result.prompts.length === 0) return [...out, "No prompt in the golden set has a system prompt, so there is nothing to slim."].join("\n");
  for (const p of result.prompts) {
    out.push(`### "${preview(p.system, 60).replace(/[`|]/g, "'")}" (${p.system.length} characters, ${p.items} prompt${p.items === 1 ? "" : "s"})`, "");
    if (p.reason) {
      out.push(`Not tested: ${p.reason}.`, "");
      continue;
    }
    if (p.best) {
      out.push(`**Use the ${p.best.style} rewrite (${p.best.prompt.length} characters): ${savingText(p.best)}.**`, "", `On your real prompts, with your model: ${verdictLine(p.best)}, ${result.runs} runs each.`, "");
      out.push("<details><summary>The shorter prompt</summary>", "", "```text", p.best.prompt.replace(/```/g, "'''"), "```", "", "</details>", "");
    } else out.push("**No shorter prompt answered your prompts the same way.** Keep the prompt you have.", "");
    out.push("| Rewrite | Result |", "| --- | --- |");
    for (const v of p.variants) out.push(`| ${v.style} (${v.prompt.length} chars) | ${v.qualifies ? `Passes: ${verdictLine(v)}, ${savingText(v)}` : `Skipped: ${(v.reason ?? "").replace(/\|/g, "/")}`} |`);
    if (p.skippedPrompts.length) out.push("", `> **Warning:** ${result.model} did not answer ${p.skippedPrompts.length} prompt(s), so they were not compared.`);
    out.push("");
  }
  out.push(`Rewrites by \`${result.writer}\`. List prices at the higher of today's and the price two months out; no batch, cache or volume discounts.`);
  return out.join("\n");
}
