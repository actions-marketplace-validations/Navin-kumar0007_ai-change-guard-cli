import type { LlmClient, LlmRequest } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import { parseGoldenSet } from "../src/golden.js";
import { applyRepair, formatRepairMarkdown, goldenLine, repairable, runRepair } from "../src/repair.js";
import { buildReplayReport } from "../src/report.js";
import { runReplay, SpendCapError } from "../src/run.js";

const SET = [
  '{"id":"order","system":"You extract orders.","input":"2 apples","expect":{"schema":{"type":"object","required":["qty"]}}}',
  '{"id":"chat","input":"Say hi","expect":{"contains":["hi"]}}',
].join("\n");

/** Old model always answers right. The new one wraps JSON in prose unless `fixedBy` says otherwise. */
function client(fixedBy: (req: LlmRequest) => boolean): { client: LlmClient; calls: LlmRequest[] } {
  const calls: LlmRequest[] = [];
  const fake: LlmClient = async (req) => {
    calls.push(req);
    const base = { toolCalls: [], inputTokens: 10, outputTokens: 5, latencyMs: 1, refused: false };
    if (req.messages[0]!.content === "Say hi") return { ...base, text: "hi there" };
    if (req.model === "openai:old" || fixedBy(req)) return { ...base, text: '{"qty": 2}' };
    return { ...base, text: 'Sure! Here is the order: {"qty": 2}' };
  };
  return { client: fake, calls };
}

async function replayThenRepair(fixedBy: (req: LlmRequest) => boolean, newModel = "openai:new", maxCalls = 500) {
  const items = parseGoldenSet(SET);
  const first = client(() => false);
  const records = await runReplay({ items, oldModel: "openai:old", newModel, client: first.client, runs: 2, concurrency: 1 });
  const report = buildReplayReport(items, records, { oldModel: "openai:old", newModel, runs: 2 });
  const second = client(fixedBy);
  const repair = await runRepair({ items, records, results: report.items, newModel, client: second.client, runs: 2, concurrency: 1, maxCalls });
  return { items, report, repair, calls: second.calls };
}

describe("Prompt Repair", () => {
  it("only takes prompts broken purely by their answer format", async () => {
    const { report } = await replayThenRepair(() => false);
    expect(report.items.filter(repairable).map((r) => r.id)).toEqual(["order"]);
  });

  it("stops at the first fix that passes and calls only the new model", async () => {
    const { repair, calls } = await replayThenRepair((req) => req.system?.includes("Answer with one JSON value only") === true);
    expect(repair.results).toEqual([
      {
        id: "order",
        fixed: true,
        patch: { kind: "format-instruction", systemAddition: expect.stringContaining('It must match this JSON Schema: {"type":"object","required":["qty"]}') },
        attempts: [{ kind: "format-instruction", fixed: true, detail: expect.stringMatching(/^passes now/) }],
      },
    ]);
    expect(repair.calls).toBe(2);
    expect(calls.map((c) => c.model)).toEqual(["openai:new", "openai:new"]);
    expect(calls[0]!.system).toMatch(/^You extract orders\.\n\nAnswer with one JSON value only/);
  });

  it("then tries the provider's JSON mode", async () => {
    const { repair } = await replayThenRepair((req) => req.jsonMode === true);
    expect(repair.results[0]).toMatchObject({ fixed: true, patch: { kind: "json-mode", jsonMode: true } });
    expect(repair.results[0]!.attempts.map((a) => [a.kind, a.fixed])).toEqual([
      ["format-instruction", false],
      ["json-mode", true],
    ]);
    expect(repair.calls).toBe(4);
  });

  it("skips JSON mode on Anthropic, which has none, and reports a prompt it could not fix", async () => {
    const { repair } = await replayThenRepair(() => false, "anthropic:new");
    expect(repair.results[0]).toMatchObject({ fixed: false, attempts: [{ kind: "format-instruction", fixed: false }] });
    expect(formatRepairMarkdown(repair)).toContain("| order | Not fixed |");
  });

  it("refuses to start above the spend cap", async () => {
    await expect(replayThenRepair(() => false, "openai:new", 3)).rejects.toThrow(SpendCapError);
  });

  it("writes a repaired golden set that parses back to the repaired prompt", async () => {
    const { items, repair } = await replayThenRepair((req) => req.jsonMode === true);
    const repaired = applyRepair(items[0]!, repair.results[0]!.patch!);
    const [back] = parseGoldenSet(goldenLine(repaired));
    expect(back).toEqual(repaired);
    expect(back!.jsonMode).toBe(true);
    expect(formatRepairMarkdown(repair)).toMatch(/\*\*Change for `order`:\*\* add this to the end of the system prompt, and turn on JSON output/);
  });
});
