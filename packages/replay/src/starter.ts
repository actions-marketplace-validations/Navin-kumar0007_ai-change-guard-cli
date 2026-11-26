import type { LlmClient } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import { extractJson } from "./schema.js";

/**
 * A starter golden set for apps with no logs yet: for each system prompt found in the code,
 * one call to a model of the user's choice writes realistic user messages for that feature.
 * The system prompt stays exactly as written in the code; only the user messages are made up,
 * so the set is clearly marked as generated and real logs (import, recorder) replace it later.
 */
export interface StarterSource {
  /** Base for the item ids, e.g. "bot.py-4". */
  id: string;
  system: string;
  /** The code expects JSON answers, so every item gets a JSON check. */
  json: boolean;
}

export interface StarterOptions {
  sources: StarterSource[];
  model: string;
  client: LlmClient;
  /** User messages per system prompt (default 8). */
  count?: number;
}

export interface StarterResult {
  items: GoldenItem[];
  calls: number;
  failures: { id: string; error: string }[];
}

const MAX_MESSAGE_CHARS = 2000;

export function starterRequest(system: string, count: number): string {
  return [
    "Below is the system prompt of one feature in an app.",
    "",
    "<system_prompt>",
    system,
    "</system_prompt>",
    "",
    `Write ${count} messages that real users would send to this feature, to test it before a model change.`,
    "Mix everyday cases, harder edge cases, and one or two messages that are unclear or off-topic.",
    "Write each message the way a user would type it. Fill any template placeholders with realistic values.",
    `Reply with only JSON: {"messages": ["...", "..."]}`,
  ].join("\n");
}

function parseValue(value: unknown, count: number): string[] {
  const list = Array.isArray(value) ? value : value && typeof value === "object" ? (value as { messages?: unknown }).messages : undefined;
  if (!Array.isArray(list)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of list) {
    if (typeof v !== "string") continue;
    const t = v.trim();
    if (!t || t.length > MAX_MESSAGE_CHARS || seen.has(t)) continue;
    seen.add(t);
    out.push(t);
    if (out.length === count) break;
  }
  return out;
}

/** The messages in a generator answer: {"messages": [...]} or a bare list, strings only. */
export function parseStarterAnswer(text: string, count: number): string[] {
  const parsed = extractJson(text);
  if (parsed.ok) return parseValue(parsed.value, count);
  // Some models add a sentence around the JSON; take the outermost object or list.
  const m = /[[{][\s\S]*[\]}]/.exec(text);
  const retry = m ? extractJson(m[0]) : parsed;
  return retry.ok ? parseValue(retry.value, count) : [];
}

export async function generateStarterSet(options: StarterOptions): Promise<StarterResult> {
  const count = options.count ?? 8;
  const items: GoldenItem[] = [];
  const failures: StarterResult["failures"] = [];
  let calls = 0;
  for (const source of options.sources) {
    calls++;
    try {
      const res = await options.client({
        model: options.model,
        system: "You write realistic test inputs for AI-powered apps.",
        messages: [{ role: "user", content: starterRequest(source.system, count) }],
        maxTokens: 2000,
        jsonMode: true,
      });
      const messages = parseStarterAnswer(res.text, count);
      if (messages.length === 0) {
        failures.push({ id: source.id, error: "the model did not return a list of messages" });
        continue;
      }
      messages.forEach((content, i) => {
        const item: GoldenItem = { id: `${source.id}-${i + 1}`, system: source.system, messages: [{ role: "user", content }] };
        if (source.json) item.expect = { json: true };
        items.push(item);
      });
    } catch (error) {
      failures.push({ id: source.id, error: error instanceof Error ? error.message : String(error) });
    }
  }
  return { items, calls, failures };
}
