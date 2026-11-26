import { describe, expect, it } from "vitest";
import { readReplayCheckText, replayCheckText, replayConclusion, replayTitle, type ReplayReport } from "../src/index.js";

const side = { runs: 3, errors: 0, avgInputTokens: 100, avgOutputTokens: 50, avgLatencyMs: 800, avgChars: 200, servedModels: [] };
function report(items: ReplayReport["items"]): ReplayReport {
  const totals = { same: 0, changed: 0, broken: 0, error: 0 };
  for (const i of items) totals[i.verdict]++;
  return {
    oldModel: "openai:gpt-4o",
    newModel: "openai:gpt-6-luna",
    runs: 3,
    calls: items.length * 6,
    failedCalls: 0,
    totals,
    avgTokens: { old: { input: 100, output: 50 }, new: { input: 90, output: 40 } },
    avgLatencyMs: { old: 800, new: 600 },
    costPerCall: { old: 0.00075, new: 0.00003 },
    comparedPrompts: items.length,
    rateLimited: false,
    reroutes: [],
    items,
  };
}
const item = (id: string, verdict: ReplayReport["items"][number]["verdict"], answer = "an answer"): ReplayReport["items"][number] => ({
  id,
  verdict,
  reasons: verdict === "same" ? [] : [{ code: "format-broken", severity: verdict === "broken" ? "broken" : "changed", detail: `${id} detail` }],
  old: side,
  new: side,
  sampleOld: `old ${answer}`,
  sampleNew: `new ${answer}`,
});

describe("the replay report published in a check run", () => {
  it("reads back what it wrote, hidden in a comment GitHub doesn't show", () => {
    const r = report([item("a", "broken"), item("b", "changed"), item("c", "same")]);
    const text = replayCheckText(r, "https://github.com/acme/bot/pull/7");
    expect(text.startsWith("<!-- ai-change-guard:replay:v1 ")).toBe(true);
    const back = readReplayCheckText(`Some heading\n\n${text}\n`);
    expect(back).toMatchObject({ v: 1, oldModel: "openai:gpt-4o", newModel: "openai:gpt-6-luna", totals: { same: 1, changed: 1, broken: 1, error: 0 }, omittedItems: 0 });
    expect(back?.prUrl).toBe("https://github.com/acme/bot/pull/7");
    expect(back?.costPerCall).toEqual({ old: 0.00075, new: 0.00003 });
    expect(back?.items[0]).toEqual({ id: "a", verdict: "broken", reasons: ["a detail"], sampleOld: "old an answer", sampleNew: "new an answer" });
  });

  it("can't be cut short by an answer that contains a comment's end", () => {
    const text = replayCheckText(report([item("a", "changed", "x --> <b>bold</b> & more")]));
    expect(text.indexOf("-->")).toBe(text.length - 3);
    expect(readReplayCheckText(text)?.items[0].sampleNew).toBe("new x --> <b>bold</b> & more");
  });

  it("stays under GitHub's limit: shorter answers first, then fewer prompts, saying how many were left out", () => {
    const many = report(Array.from({ length: 400 }, (_, i) => item(`p${i}`, i < 10 ? "changed" : "same", "y".repeat(3000))));
    const text = replayCheckText(many, undefined, 60_000);
    expect(text.length).toBeLessThanOrEqual(60_000);
    const back = readReplayCheckText(text)!;
    expect(back.items.length + back.omittedItems).toBe(400);
    expect(back.totals.same).toBe(390); // the totals always count every prompt
    expect(back.items[0].sampleNew!.length).toBeLessThan(3000);
    expect(back.items.find((i) => i.verdict === "same")?.sampleNew).toBeUndefined();

    const huge = report(Array.from({ length: 3000 }, (_, i) => item(`prompt-${i}`, "changed", "z".repeat(500))));
    const cut = readReplayCheckText(replayCheckText(huge, undefined, 60_000))!;
    expect(cut.omittedItems).toBeGreaterThan(0);
    expect(cut.items.length + cut.omittedItems).toBe(3000);
  });

  it("returns null for text that isn't a report it can read", () => {
    expect(readReplayCheckText(null)).toBeNull();
    expect(readReplayCheckText("just a summary")).toBeNull();
    expect(readReplayCheckText("<!-- ai-change-guard:replay:v1 {not json} -->")).toBeNull();
    expect(readReplayCheckText('<!-- ai-change-guard:replay:v1 {"v":2,"items":[]} -->')).toBeNull();
    expect(readReplayCheckText('<!-- ai-change-guard:replay:v1 {"v":1,"oldModel":"a","newModel":"b","totals":{"same":1},"items":[]} -->')).toBeNull();
  });

  it("rebuilds a partial or tampered report instead of passing it on", () => {
    const wrap = (o: unknown) => `<!-- ai-change-guard:replay:v1 ${JSON.stringify(o)} -->`;
    const partial = readReplayCheckText(
      wrap({
        v: 1,
        oldModel: "a",
        newModel: "b",
        totals: { same: 1, changed: 1, broken: 0, error: 0 },
        items: [{ id: "x", verdict: "changed", reasons: ["ok", { evil: true }, 5], sampleOld: { not: "text" } }],
        reroutes: "nope",
        costPerCall: { old: "free", new: 0.001 },
        prUrl: 7,
      }),
    )!;
    expect(partial).toMatchObject({ runs: 1, calls: 0, omittedItems: 0, reroutes: [], rateLimited: false, costPerCall: { new: 0.001 } });
    expect(partial.avgTokens).toEqual({ old: { input: 0, output: 0 }, new: { input: 0, output: 0 } });
    expect(partial.avgLatencyMs).toEqual({ old: 0, new: 0 });
    expect(partial.items).toEqual([{ id: "x", verdict: "changed", reasons: ["ok"] }]);
    expect(partial.prUrl).toBeUndefined();
    // An item without a real verdict makes the whole report unreadable rather than half shown.
    expect(readReplayCheckText(wrap({ v: 1, oldModel: "a", newModel: "b", totals: { same: 0, changed: 0, broken: 0, error: 0 }, items: [{ id: "x", verdict: "maybe" }] }))).toBeNull();
  });

  it("always fits, even when the models' names are absurdly long", () => {
    const long = { ...report([item("a", "changed")]), oldModel: `openai:${"m".repeat(70_000)}`, reroutes: Array.from({ length: 500 }, () => ({ side: "new" as const, requested: "x".repeat(300), served: "y".repeat(300) })) };
    const text = replayCheckText(long, undefined, 60_000);
    expect(text.length).toBeLessThanOrEqual(60_000);
    expect(readReplayCheckText(text)?.reroutes).toHaveLength(10);
  });

  it("fails the check only when the job fails, and asks for a look otherwise", () => {
    expect(replayConclusion({ totals: { same: 5, changed: 0, broken: 1, error: 0 } }, "none")).toBe("neutral");
    expect(replayConclusion({ totals: { same: 5, changed: 1, broken: 0, error: 0 } }, "changed")).toBe("failure");
    expect(replayConclusion({ totals: { same: 5, changed: 0, broken: 1, error: 0 } })).toBe("failure");
    expect(replayConclusion({ totals: { same: 5, changed: 1, broken: 0, error: 0 } })).toBe("neutral");
    expect(replayConclusion({ totals: { same: 5, changed: 0, broken: 0, error: 1 } })).toBe("neutral");
    expect(replayConclusion({ totals: { same: 5, changed: 0, broken: 0, error: 0 } })).toBe("success");
    expect(replayTitle({ totals: { same: 45, changed: 3, broken: 2, error: 1 } })).toBe("45 same · 3 changed · 2 broken · 1 could not compare");
  });
});
