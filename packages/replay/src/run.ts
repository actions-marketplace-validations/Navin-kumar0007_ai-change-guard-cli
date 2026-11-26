import type { LlmClient, LlmRequest, LlmResponse } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";

export type Side = "old" | "new";

export interface RunRecord {
  itemId: string;
  side: Side;
  run: number;
  response?: LlmResponse;
  error?: string;
}

export interface ReplayOptions {
  items: GoldenItem[];
  oldModel: string;
  newModel: string;
  client: LlmClient;
  runs?: number;
  concurrency?: number;
  maxCalls?: number;
  defaultMaxTokens?: number;
  /**
   * Prompts sent on the new side in place of `items`, matched by id: the same prompts with a different system
   * prompt, for example (Verified Prompt Slimming). Default: `items` on both sides.
   */
  newItems?: GoldenItem[];
  /** Which models to call; a repair re-test calls only the new one. Default: both. */
  sides?: Side[];
  /** Stops starting new calls once aborted; the replay then rejects with a ReplayCancelledError. */
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

export class SpendCapError extends Error {}
export class ReplayCancelledError extends Error {}

export function plannedCalls(items: number, runs: number, sides = 2): number {
  return items * runs * sides;
}

function requestFor(item: GoldenItem, model: string, defaultMaxTokens: number): LlmRequest {
  const request: LlmRequest = { model, messages: item.messages, maxTokens: item.maxTokens ?? defaultMaxTokens };
  if (item.system) request.system = item.system;
  if (item.tools) request.tools = item.tools;
  if (item.temperature !== undefined) request.temperature = item.temperature;
  if (item.jsonMode) request.jsonMode = true;
  return request;
}

export async function runReplay(options: ReplayOptions): Promise<RunRecord[]> {
  const runs = options.runs ?? 5;
  const concurrency = Math.max(1, options.concurrency ?? 5);
  const maxCalls = options.maxCalls ?? 500;
  const sides = options.sides ?? (["old", "new"] as const);
  const total = plannedCalls(options.items.length, runs, sides.length);
  if (total > maxCalls) {
    throw new SpendCapError(
      `This replay needs ${total} AI calls, above the limit of ${maxCalls}. Lower --runs, use fewer prompts, or raise --max-calls.`,
    );
  }

  const newById = new Map((options.newItems ?? []).map((i) => [i.id, i]));
  const tasks: (() => Promise<RunRecord>)[] = [];
  for (const item of options.items) {
    for (let run = 0; run < runs; run++) {
      for (const side of sides) {
        const sent = side === "new" ? (newById.get(item.id) ?? item) : item;
        const request = requestFor(sent, side === "old" ? options.oldModel : options.newModel, options.defaultMaxTokens ?? 1024);
        tasks.push(async () => {
          try {
            return { itemId: item.id, side, run, response: await options.client(request) };
          } catch (error) {
            return { itemId: item.id, side, run, error: error instanceof Error ? error.message : String(error) };
          }
        });
      }
    }
  }

  const results: RunRecord[] = new Array(tasks.length);
  let next = 0;
  let done = 0;
  const worker = async () => {
    while (next < tasks.length && !options.signal?.aborted) {
      const index = next++;
      results[index] = await tasks[index]!();
      done++;
      options.onProgress?.(done, total);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, tasks.length) }, worker));
  if (options.signal?.aborted) throw new ReplayCancelledError(`Replay cancelled after ${done} of ${total} calls.`);
  return results;
}
