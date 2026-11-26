import type { LlmClient, LlmRequest, LlmResponse } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import {
  buildReplayReport,
  extractJson,
  formatReplayMarkdown,
  formatReplayText,
  GoldenSetError,
  isReroute,
  judgeItem,
  parseGoldenSet,
  runReplay,
  SpendCapError,
  validateSchema,
  type GoldenItem,
  type RunRecord,
} from "../src/index.js";

function reply(text: string, extra: Partial<LlmResponse> = {}): LlmResponse {
  return { text, toolCalls: [], inputTokens: 10, outputTokens: 5, latencyMs: 100, refused: false, ...extra };
}

function records(itemId: string, oldReplies: (LlmResponse | Error)[], newReplies: (LlmResponse | Error)[]): RunRecord[] {
  const build = (side: "old" | "new", list: (LlmResponse | Error)[]) =>
    list.map((r, run) => (r instanceof Error ? { itemId, side, run, error: r.message } : { itemId, side, run, response: r }));
  return [...build("old", oldReplies), ...build("new", newReplies)];
}

const times = <T>(n: number, make: (i: number) => T) => Array.from({ length: n }, (_, i) => make(i));

describe("parseGoldenSet", () => {
  it("reads input lines, messages, tools and expectations", () => {
    const items = parseGoldenSet(
      [
        '{"id":"a","input":"Hi","system":"Be nice","expect":{"contains":["hello"]}}',
        "// comments and blank lines are skipped",
        "",
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"r"},{"role":"user","content":"q2"}],"max_tokens":50,"json_mode":true}',
        '{"id":"t","input":"x","tools":[{"name":"f","parameters":{"type":"object"}}],"expect":{"tool":"f","schema":{"type":"object"}}}',
      ].join("\n"),
    );
    expect(items.map((i) => i.id)).toEqual(["a", "line-4", "t"]);
    expect(items[0]?.messages).toEqual([{ role: "user", content: "Hi" }]);
    expect(items[1]?.maxTokens).toBe(50);
    expect(items[1]?.jsonMode).toBe(true);
    expect(items[2]?.expect?.tool).toBe("f");
  });

  it.each([
    ["not json", "Line 1: not valid JSON"],
    ['{"id":"x"}', 'needs "input"'],
    ['{"input":"a","id":1}\n{"input":"b","id":1}', 'duplicate id "1"'],
    ['{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}', "last message must be from the user"],
    ['{"input":"a","tools":[{"name":"f"}]}', "parameters"],
    ["", "empty"],
  ])("explains what is wrong: %s", (text, message) => {
    expect(() => parseGoldenSet(text)).toThrow(GoldenSetError);
    expect(() => parseGoldenSet(text)).toThrow(message);
  });
});

describe("schema checks", () => {
  const schema = {
    type: "object",
    required: ["status", "items"],
    additionalProperties: false,
    properties: { status: { enum: ["ok", "fail"] }, items: { type: "array", minItems: 1, items: { type: "integer" } } },
  };

  it("accepts valid values and lists problems in invalid ones", () => {
    expect(validateSchema({ status: "ok", items: [1, 2] }, schema)).toEqual([]);
    expect(validateSchema({ status: "maybe", items: [1.5], extra: true }, schema)).toEqual([
      "$.status: not one of the allowed values",
      "$.items[0]: expected integer, got number",
      '$: unexpected "extra"',
    ]);
  });

  it("extracts JSON from plain text or a fenced block", () => {
    expect(extractJson('{"a":1}')).toEqual({ ok: true, value: { a: 1 } });
    expect(extractJson('```json\n{"a":1}\n```')).toEqual({ ok: true, value: { a: 1 } });
    expect(extractJson("Here you go: {a:1}")).toEqual({ ok: false });
  });
});

describe("judgeItem", () => {
  const item = (extra: Partial<GoldenItem> = {}): GoldenItem => ({ id: "p1", messages: [{ role: "user", content: "q" }], ...extra });

  it("calls an unchanged answer the same", () => {
    const text = "Your order ships within two business days from our Pune warehouse.";
    const result = judgeItem(item(), records("p1", times(5, () => reply(text)), times(5, () => reply(text))));
    expect(result.verdict).toBe("same");
    expect(result.reasons).toEqual([]);
  });

  it("flags broken JSON only when the new model is worse than the old", () => {
    const expect_ = { schema: { type: "object", required: ["total"] } };
    const good = () => reply('{"total": 5}');
    const bad = () => reply("The total is 5.");
    const broken = judgeItem(item({ expect: expect_ }), records("p1", times(5, good), [good(), bad(), bad(), good(), good()]));
    expect(broken.verdict).toBe("broken");
    expect(broken.reasons[0]?.code).toBe("format-broken");
    expect(broken.reasons[0]?.detail).toContain("2 of 5 new answers fail the expected JSON schema (old: 0 of 5)");

    const alreadyFlaky = judgeItem(item({ expect: expect_ }), records("p1", [bad(), bad(), good(), good(), good()], [bad(), bad(), good(), good(), good()]));
    expect(alreadyFlaky.reasons.some((r) => r.code === "format-broken")).toBe(false);
  });

  it("flags a changed tool call and missing required arguments", () => {
    const tools = [
      { name: "refund", parameters: { type: "object", required: ["order_id"] } },
      { name: "escalate", parameters: { type: "object" } },
    ];
    const oldR = () => reply("", { toolCalls: [{ name: "refund", arguments: { order_id: "7" } }] });
    const newR = () => reply("", { toolCalls: [{ name: "escalate", arguments: {} }] });
    const changed = judgeItem(item({ tools }), records("p1", times(5, oldR), times(5, newR)));
    expect(changed.reasons.map((r) => r.code)).toEqual(["tool-change"]);

    const noArg = () => reply("", { toolCalls: [{ name: "refund", arguments: {} }] });
    const missing = judgeItem(item({ tools, expect: { tool: "refund" } }), records("p1", times(5, oldR), times(5, noArg)));
    expect(missing.reasons.map((r) => r.code)).toEqual(["tool-args-missing"]);
  });

  it("flags missing required text", () => {
    const result = judgeItem(
      item({ expect: { contains: ["refund"] } }),
      records("p1", times(5, () => reply("We will refund you today.")), times(5, () => reply("We will look into it."))),
    );
    expect(result.verdict).toBe("broken");
    expect(result.reasons.some((r) => r.code === "missing-text")).toBe(true);
  });

  it("flags new refusals as a change, not as drift", () => {
    const result = judgeItem(
      item(),
      records("p1", times(5, () => reply("Here is how to reset your password: open Settings.")), times(5, () => reply("I'm sorry, I can't help with that.", { refused: true }))),
    );
    expect(result.verdict).toBe("changed");
    expect(result.reasons.map((r) => r.code)).toEqual(["new-refusal"]);
  });

  it("flags answer drift relative to how much the old model varies", () => {
    const oldTexts = [
      "Restart the router, then wait two minutes for the lights to settle.",
      "Restart the router and wait about two minutes for the lights.",
      "Please restart the router, then wait two minutes.",
      "Restart your router and wait two minutes for the lights to settle.",
      "Restart the router, wait two minutes, and check the lights.",
    ];
    const drifting = times(5, () => reply("Our billing team will contact you about the invoice next week."));
    const similarNew = times(5, (i) => reply(oldTexts[i]!));
    expect(judgeItem(item(), records("p1", oldTexts.map((t) => reply(t)), drifting)).reasons[0]?.code).toBe("answer-drift");
    expect(judgeItem(item(), records("p1", oldTexts.map((t) => reply(t)), similarNew)).verdict).toBe("same");
  });

  it("handles failing calls on either side", () => {
    const allNewFail = judgeItem(item(), records("p1", times(5, () => reply("ok")), times(5, () => new Error("HTTP 404: model not found"))));
    expect(allNewFail.verdict).toBe("broken");
    expect(allNewFail.reasons[0]?.detail).toContain("All 5 calls to the new model failed: HTTP 404");

    const oldFails = judgeItem(item(), records("p1", times(5, () => new Error("HTTP 401")), times(5, () => reply("ok"))));
    expect(oldFails.verdict).toBe("error");
    expect(oldFails.error).toBe("The old model never answered: HTTP 401 — The new model answered 5 of 5 times.");

    const bothFail = judgeItem(item(), records("p1", times(2, () => new Error("HTTP 404: gone")), times(2, () => new Error("HTTP 404: gone"))));
    expect(bothFail.error).toBe("The old model never answered: HTTP 404: gone — The new model failed the same way.");
  });

  it("compares tokens only over prompts both models answered, and suggests slowing down on rate limits", () => {
    const goldenItems: GoldenItem[] = [
      { id: "small", messages: [{ role: "user", content: "a" }] },
      { id: "big", messages: [{ role: "user", content: "b" }] },
    ];
    const recs = [
      ...records("small", [reply("hi", { inputTokens: 10 })], [reply("hi", { inputTokens: 10 })]),
      ...records("big", [new Error("HTTP 429: quota")], [reply("hello", { inputTokens: 500 })]),
    ];
    const report = buildReplayReport(goldenItems, recs, { oldModel: "google:a", newModel: "google:b", runs: 1 });
    expect(report.comparedPrompts).toBe(1);
    expect(report.avgTokens.new.input).toBe(10);
    expect(report.rateLimited).toBe(true);
    expect(formatReplayText(report)).toContain("TIP: some calls hit the provider's rate limit");
  });

  it("prices a call at list price on both models, when the catalog knows them", () => {
    const goldenItems: GoldenItem[] = [{ id: "a", messages: [{ role: "user", content: "a" }] }];
    const recs = records("a", [reply("hi", { inputTokens: 1000, outputTokens: 200 })], [reply("hi", { inputTokens: 1300, outputTokens: 260 })]);
    const report = buildReplayReport(goldenItems, recs, {
      oldModel: "anthropic:claude-sonnet-4-5",
      newModel: "anthropic:claude-sonnet-5-5",
      runs: 1,
      today: "2026-11-23",
    });
    // Old: 1,000 × $3 + 200 × $15 per 1M = $0.006. New: 1,300 × $2 + 260 × $10 per 1M = $0.0052.
    expect(report.costPerCall?.old).toBeCloseTo(0.006);
    expect(report.costPerCall?.new).toBeCloseTo(0.0052);
    expect(formatReplayText(report)).toContain("Cost per call at list price: $0.006 → $0.0052 (-13%)");
    expect(formatReplayMarkdown(report)).toContain("Cost per call at list price: $0.006 → $0.0052 (-13%).");

    const unknown = buildReplayReport(goldenItems, recs, { oldModel: "ollama:llama3", newModel: "google:made-up", runs: 1 });
    expect(unknown.costPerCall).toBeUndefined();
    expect(formatReplayText(unknown)).not.toContain("Cost per call");
  });

  it("groups identical errors into one line in the text report", () => {
    const goldenItems: GoldenItem[] = ["a", "b", "c", "d", "e"].map((id) => ({ id, messages: [{ role: "user", content: id }] }));
    const recs = goldenItems.flatMap((g) => records(g.id, [new Error("HTTP 404: model retired")], [new Error("HTTP 404: model retired")]));
    const text = formatReplayText(buildReplayReport(goldenItems, recs, { oldModel: "google:x", newModel: "google:y", runs: 1 }));
    expect(text).toContain("ERROR    5 prompts (a, b, c and 2 more)");
    expect(text.match(/HTTP 404: model retired/g)).toHaveLength(1);
  });
});

describe("runReplay", () => {
  const items: GoldenItem[] = [
    { id: "a", messages: [{ role: "user", content: "1" }], system: "S", maxTokens: 30 },
    { id: "b", messages: [{ role: "user", content: "2" }], jsonMode: true },
  ];

  it("runs every prompt on both models, within the concurrency limit", async () => {
    const seen: LlmRequest[] = [];
    let inFlight = 0;
    let peak = 0;
    const client: LlmClient = async (req) => {
      seen.push(req);
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--;
      if (req.model === "openai:new" && req.messages[0]?.content === "2") throw new Error("boom");
      return reply(`answer from ${req.model}`);
    };
    const progress: number[] = [];
    const result = await runReplay({ items, oldModel: "openai:old", newModel: "openai:new", client, runs: 3, concurrency: 2, onProgress: (d) => progress.push(d) });

    expect(result).toHaveLength(12);
    expect(peak).toBeLessThanOrEqual(2);
    expect(progress.at(-1)).toBe(12);
    expect(result.filter((r) => r.error)).toHaveLength(3);
    expect(seen.find((r) => r.messages[0]?.content === "1")).toMatchObject({ system: "S", maxTokens: 30 });
    expect(seen.find((r) => r.messages[0]?.content === "2")).toMatchObject({ jsonMode: true, maxTokens: 1024 });
  });

  it("refuses to start above the call limit", async () => {
    const client: LlmClient = async () => reply("x");
    await expect(runReplay({ items, oldModel: "openai:a", newModel: "openai:b", client, runs: 5, maxCalls: 10 })).rejects.toThrow(SpendCapError);
  });
});

describe("report", () => {
  it("detects silent reroutes but not dated versions of the same model", () => {
    expect(isReroute("gpt-4o", "gpt-4o-2024-08-06")).toBe(false);
    expect(isReroute("gemini-2.5-flash", "gemini-3.5-flash-lite")).toBe(true);
    expect(isReroute("gpt-4o", "openai/gpt-4o")).toBe(false);
  });

  it("summarises verdicts, tokens and reroutes in text and markdown", () => {
    const goldenItems: GoldenItem[] = [
      { id: "same-one", messages: [{ role: "user", content: "a" }] },
      { id: "broken-one", messages: [{ role: "user", content: "b" }], expect: { json: true } },
    ];
    const recs = [
      ...records("same-one", times(2, () => reply("Same words here", { servedModel: "gemini-2.5-flash" })), times(2, () => reply("Same words here", { outputTokens: 10, servedModel: "gemini-3.5-flash-lite" }))),
      ...records("broken-one", times(2, () => reply('{"x":1}')), times(2, () => reply("no json"))),
    ];
    const report = buildReplayReport(goldenItems, recs, { oldModel: "google:gemini-2.5-flash", newModel: "google:gemini-2.5-flash", runs: 2 });

    expect(report.totals).toEqual({ same: 1, changed: 0, broken: 1, error: 0 });
    expect(report.items[0]?.id).toBe("broken-one");
    expect(report.reroutes).toEqual([{ side: "new", requested: "gemini-2.5-flash", served: "gemini-3.5-flash-lite" }]);

    const text = formatReplayText(report);
    expect(text).toContain("Verdict: 1 same · 0 changed · 1 broken · 0 could not compare");
    expect(text).toContain("output tokens 5 → 8 (+50%)");
    expect(text).toContain('was answered by "gemini-3.5-flash-lite"');
    expect(text).toContain("BROKEN   broken-one");

    const md = formatReplayMarkdown(report);
    expect(md).toContain("| 1 | 0 | 1 | 0 |");
    expect(md).toContain("**BROKEN: broken-one**");
  });
});

describe("answer drift needs a baseline", () => {
  it("does not judge drift from a single old answer, but does from two or more", async () => {
    const { judgeItem: judge } = await import("../src/verdict.js");
    const item = { id: "a", messages: [{ role: "user" as const, content: "hi" }] };
    const answer = (side: "old" | "new", run: number, text: string) => ({
      itemId: "a",
      side,
      run,
      response: { text, toolCalls: [], inputTokens: 1, outputTokens: 1, latencyMs: 1, refused: false },
    });
    const one = judge(item, [answer("old", 0, "Your refund is on its way today."), answer("new", 0, "We have started processing the return for you.")]);
    expect(one.verdict).toBe("same");
    const two = judge(item, [
      answer("old", 0, "Your refund is on its way today."),
      answer("old", 1, "Your refund is on its way today."),
      answer("new", 0, "We have started processing the return for you."),
      answer("new", 1, "We have started processing the return for you."),
    ]);
    expect(two.reasons.map((r) => r.code)).toEqual(["answer-drift"]);
  });
});

describe("cancelling a replay", () => {
  it("stops starting new calls once the signal aborts", async () => {
    const { runReplay: run, ReplayCancelledError: Cancelled } = await import("../src/run.js");
    const controller = new AbortController();
    let calls = 0;
    const client: LlmClient = async () => {
      calls++;
      if (calls === 2) controller.abort();
      return { text: "ok", toolCalls: [], inputTokens: 1, outputTokens: 1, latencyMs: 1, refused: false };
    };
    const items = [1, 2, 3, 4].map((i) => ({ id: String(i), messages: [{ role: "user" as const, content: "hi" }] }));
    await expect(run({ items, oldModel: "openai:a", newModel: "openai:b", client, runs: 1, concurrency: 1, signal: controller.signal })).rejects.toThrow(Cancelled);
    expect(calls).toBe(2);
  });
});
