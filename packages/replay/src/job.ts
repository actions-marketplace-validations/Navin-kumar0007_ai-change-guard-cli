import type { LlmClient } from "@ai-changeguard/llm";
import type { GoldenItem } from "./golden.js";
import { plannedRepairCalls, repairable, runRepair, type RepairReport } from "./repair.js";
import { buildReplayReport, type ReplayReport } from "./report.js";
import { runReplay, type RunRecord } from "./run.js";

/**
 * One replay, optionally followed by Prompt Repair, under a single call budget. Shared by the
 * CLI, the GitHub Action and the MCP server so the spend rules live in one place: the replay
 * refuses to start above maxCalls, and repair only uses what the replay left.
 */
export interface ReplayJob {
  items: GoldenItem[];
  oldModel: string;
  newModel: string;
  runs: number;
  maxCalls: number;
  concurrency: number;
  client: LlmClient;
  repair?: boolean;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number) => void;
}

export interface ReplayJobResult {
  report: ReplayReport;
  records: RunRecord[];
  repair?: RepairReport;
  /** Why repair did not run although broken formats were found. */
  repairSkipped?: string;
}

export async function runReplayJob(job: ReplayJob): Promise<ReplayJobResult> {
  const { items, oldModel, newModel, runs, maxCalls, concurrency, client } = job;
  const records = await runReplay({
    items,
    oldModel,
    newModel,
    runs,
    concurrency,
    maxCalls,
    client,
    ...(job.signal ? { signal: job.signal } : {}),
    ...(job.onProgress ? { onProgress: job.onProgress } : {}),
  });
  const report = buildReplayReport(items, records, { oldModel, newModel, runs });
  if (!job.repair || !report.items.some(repairable)) return { report, records };

  const left = maxCalls - records.length;
  const needed = plannedRepairCalls(items, report.items, newModel, runs);
  if (needed > left) {
    return { report, records, repairSkipped: `Prompt Repair skipped: it could need ${needed} calls, and only ${left} are left under the limit of ${maxCalls}.` };
  }
  try {
    const repair = await runRepair({ items, records, results: report.items, newModel, client, runs, concurrency, maxCalls: left, ...(job.signal ? { signal: job.signal } : {}) });
    return { report, records, repair };
  } catch (error) {
    // A repair problem never hides the replay result.
    return { report, records, repairSkipped: `Prompt Repair failed: ${error instanceof Error ? error.message : String(error)}` };
  }
}
