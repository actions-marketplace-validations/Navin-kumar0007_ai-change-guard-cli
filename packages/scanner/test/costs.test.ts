import { billedPriceOn, callCost, lookupPrice, prices, priceChanges, priceOn, tokenRatio } from "@ai-changeguard/catalog";
import { describe, expect, it } from "vitest";
import { buildCheck } from "../src/check.js";
import { buildCostRows, describeCost, formatUsd, isPriceRise, priceChangeText } from "../src/costs.js";
import type { Finding } from "../src/detect.js";
import { estimateImpact, guessPlatform, tokensOf, usageFromLog } from "../src/impact.js";
import { buildReport, formatText, refreshReport } from "../src/report.js";

const TODAY = "2026-11-23";

describe("price catalog", () => {
  it("has a source and a positive price for every entry, and no duplicates", () => {
    const keys = new Set<string>();
    for (const p of prices) {
      expect(p.sources.length).toBeGreaterThan(0);
      expect(p.input).toBeGreaterThan(0);
      expect(p.output).toBeGreaterThan(0);
      const key = `${p.id}|${p.platform}`;
      expect(keys.has(key), key).toBe(false);
      keys.add(key);
    }
    for (const c of priceChanges) {
      expect(c.on).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(lookupPrice(c.id, c.platform), c.id).toBeDefined();
    }
  });

  it("applies announced changes from their first day", () => {
    expect(priceOn("gemini-3.8-flash", "gemini-api", "2026-12-31")).toEqual({ input: 0.75, output: 3.75 });
    expect(priceOn("gemini-3.8-flash", "gemini-api", "2027-01-01")).toEqual({ input: 1.5, output: 7.5 });
    // OpenAI has not published GPT-5.6 Sol's price after its promotion: unknown, not a guess.
    expect(priceOn("gpt-5.6-sol", "openai-api", "2026-11-30")).toEqual({ input: 4, output: 20 });
    expect(priceOn("gpt-5.6-sol", "openai-api", "2026-12-01")).toBeUndefined();
    expect(priceOn("no-such-model", "openai-api", TODAY)).toBeUndefined();
    // Prices are per platform: the Gemini API price says nothing about Vertex AI.
    expect(priceOn("gemini-3.8-flash", "vertex-ai", TODAY)).toBeUndefined();
  });

  it("knows Claude's newer tokenizer uses about 30% more tokens", () => {
    expect(tokenRatio({ id: "claude-sonnet-4-5", platform: "anthropic-api" }, { id: "claude-sonnet-5-5", platform: "anthropic-api" })).toBe(1.3);
    expect(tokenRatio({ id: "claude-sonnet-5-5", platform: "anthropic-api" }, { id: "claude-sonnet-4-6", platform: "anthropic-api" })).toBeCloseTo(1 / 1.3);
    expect(tokenRatio({ id: "claude-opus-4-7", platform: "anthropic-api" }, { id: "claude-sonnet-5-5", platform: "anthropic-api" })).toBe(1);
    expect(tokenRatio({ id: "gpt-4o", platform: "openai-api" }, { id: "gpt-6-luna", platform: "openai-api" })).toBe(1);
  });

  it("bills a rerouted model at the target's price from the reroute day", () => {
    expect(billedPriceOn("gemini-2.5-pro", "gemini-api", "2026-11-30")).toEqual({ input: 1.25, output: 10 });
    expect(billedPriceOn("gemini-2.5-pro", "gemini-api", "2026-12-15")).toEqual({ input: 0.75, output: 3.75 });
    expect(billedPriceOn("gemini-2.5-pro", "gemini-api", "2027-01-02")).toEqual({ input: 1.5, output: 7.5 });
  });

  it("prices a call", () => {
    expect(callCost({ input: 2, output: 10 }, 1_000_000, 500_000)).toBe(7);
  });
});

describe("cost rows", () => {
  it("reports an introductory price ending, with the new price", () => {
    const [row] = buildCostRows([{ id: "gemini-3.8-flash", platform: "gemini-api", locations: ["a.py:1"] }], TODAY);
    expect(row).toMatchObject({ kind: "intro-ends", on: "2027-01-01", daysLeft: 39, before: { input: 0.75, output: 3.75 }, after: { input: 1.5, output: 7.5 } });
    expect(describeCost(row!)).toBe(
      "gemini-3.8-flash: price rises on 2027-01-01 (in 39 days): $0.75 → $1.50 input, $3.75 → $7.50 output per 1M tokens (+100%)",
    );
  });

  it("says plainly when the new price is not published", () => {
    const [row] = buildCostRows([{ id: "gpt-5.6-sol", platform: "openai-api", locations: [] }], TODAY);
    expect(row?.after).toBeUndefined();
    expect(describeCost(row!)).toContain("the new price is not published yet");
  });

  it("follows a reroute: billed at the target's rates, then the target's own price change", () => {
    const rows = buildCostRows([{ id: "gemini-2.5-pro", platform: "gemini-api", reroute: { on: "2026-12-01", to: "gemini-3.8-flash" }, locations: [] }], TODAY);
    expect(rows.map((r) => [r.kind, r.on, r.billedAs])).toEqual([
      ["reroute-billing", "2026-12-01", "gemini-3.8-flash"],
      ["intro-ends", "2027-01-01", "gemini-3.8-flash"],
    ]);
    expect(rows[0]).toMatchObject({ before: { input: 1.25, output: 10 }, after: { input: 0.75, output: 3.75 } });
    expect(describeCost(rows[1]!)).toContain("gemini-2.5-pro (served as gemini-3.8-flash): price rises");
  });

  it("keeps a change for 30 days after it happens, then drops it", () => {
    const model = [{ id: "gemini-3.8-flash", platform: "gemini-api" as const, locations: [] }];
    expect(buildCostRows(model, "2027-01-31")[0]?.daysLeft).toBe(-30);
    expect(describeCost(buildCostRows(model, "2027-01-31")[0]!)).toContain("since 2027-01-01");
    expect(buildCostRows(model, "2027-02-01")).toEqual([]);
  });

  it("does not call a scheduled change with no published price an introductory price", () => {
    const row = { id: "m", platform: "openai-api" as const, kind: "price-change" as const, on: "2027-03-01", daysLeft: 98, before: { input: 1, output: 2 }, verified: true, locations: [] };
    expect(describeCost(row)).toBe("m: price changes on 2027-03-01 (in 98 days); the new price is not published yet (now $1.00 input, $2.00 output per 1M tokens)");
  });

  it("formats money and percentages", () => {
    expect(formatUsd(1.5)).toBe("$1.50");
    expect(formatUsd(0.1)).toBe("$0.10");
    expect(formatUsd(0.075)).toBe("$0.075");
    expect(formatUsd(0.01)).toBe("$0.01");
    expect(formatUsd(0.000423)).toBe("$0.000423");
    expect(formatUsd(0)).toBe("$0.00");
    expect(priceChangeText({ input: 1, output: 4 }, { input: 2, output: 8 })).toBe("+100%");
    expect(priceChangeText({ input: 1.25, output: 10 }, { input: 0.75, output: 3.75 })).toBe("input -40%, output -62%");
  });
});

function finding(file: string, line: number, id: string, provider: "openai" | "anthropic" | "google"): Finding {
  return { file, line, kind: "model", snippet: "", model: { provider, id, raw: id } };
}

describe("scan report", () => {
  const report = buildReport(
    {
      root: "/tmp/x",
      filesScanned: 2,
      aiFiles: 2,
      callSites: 3,
      findings: [finding("a.py", 3, "gemini-2.5-pro", "google"), finding("b.ts", 7, "claude-sonnet-4-5", "anthropic"), finding("c.py", 1, "gemini-3.8-flash", "google")],
      tokenTips: [],
    },
    "demo",
    TODAY,
  );

  it("adds today's price, the replacement's price and the tokenizer difference", () => {
    const sonnet = report.models.find((m) => m.id === "claude-sonnet-4-5")!;
    expect(sonnet.price).toEqual({ input: 3, output: 15 });
    expect(sonnet.replacementPrice).toEqual({ input: 2, output: 10 });
    expect(sonnet.replacementTokenRatio).toBe(1.3);
    expect(formatText(report)).toContain("uses about 30% more tokens for the same text");
  });

  it("lists price changes, including the rerouted model's", () => {
    expect(report.costs?.map((c) => `${c.id}:${c.kind}:${c.on}`)).toEqual([
      "gemini-2.5-pro:reroute-billing:2026-12-01",
      "gemini-2.5-pro:intro-ends:2027-01-01",
      "gemini-3.8-flash:intro-ends:2027-01-01",
    ]);
    const text = formatText(report);
    expect(text).toContain("Price changes");
    expect(text).toContain("c.py:1");
  });

  it("shows the price billed on the scan day, not the catalog's first price", () => {
    const later = buildReport(
      { root: "/tmp/x", filesScanned: 1, aiFiles: 1, callSites: 3, findings: [finding("a.py", 1, "gemini-3.8-flash", "google"), finding("b.py", 1, "gemini-2.5-pro", "google"), finding("c.py", 1, "gpt-4", "openai")], tokenTips: [] },
      "demo",
      "2027-01-10",
    );
    expect(later.models.find((m) => m.id === "gemini-3.8-flash")?.price).toEqual({ input: 1.5, output: 7.5 });
    expect(later.models.find((m) => m.id === "gemini-2.5-pro")?.price).toEqual({ input: 1.5, output: 7.5 });
    // gpt-4's replacement gpt-5.6-sol has no published price after its promotion ended: none shown.
    const gpt4 = later.models.find((m) => m.id === "gpt-4")!;
    expect(gpt4.replacement).toBe("gpt-5.6-sol");
    expect(gpt4.replacementPrice).toBeUndefined();
  });

  it("puts price changes in the pull request check without changing its conclusion", () => {
    const check = buildCheck(report, ["c.py"]);
    expect(check.summary).toContain("**Price changes** (advisory):");
    expect(check.summary).toContain("gemini-3.8-flash: price rises on 2027-01-01");
    const without = buildCheck({ ...report, costs: [] }, ["c.py"]);
    expect(check.conclusion).toBe(without.conclusion);
  });
});

describe("usage from the recorder log", () => {
  it("reads token counts from OpenAI, Anthropic and Gemini answers", () => {
    expect(tokensOf({ usage: { prompt_tokens: 10, completion_tokens: 4 } })).toEqual({ input: 10, output: 4 });
    expect(tokensOf({ usage: { input_tokens: 12, output_tokens: 6 } })).toEqual({ input: 12, output: 6 });
    // Gemini bills thinking tokens as output.
    expect(tokensOf({ usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 5, thoughtsTokenCount: 7 } })).toEqual({ input: 20, output: 12 });
    // Anthropic reports cache reads and writes apart from input_tokens; all of them are input.
    expect(tokensOf({ usage: { input_tokens: 200, cache_read_input_tokens: 10_000, cache_creation_input_tokens: 50, output_tokens: 30 } })).toEqual({ input: 10_250, output: 30 });
    expect(tokensOf(null)).toBeUndefined();
    expect(tokensOf({ usage: { total_tokens: 5 } })).toBeUndefined();
  });

  it("groups calls by model and platform and skips lines it cannot read", () => {
    const lines = [
      JSON.stringify({ recorded_at: "2026-11-01T00:00:00Z", url: "https://api.openai.com/v1/chat/completions", model: "gpt-4o", response: { usage: { prompt_tokens: 100, completion_tokens: 20 } } }),
      JSON.stringify({ recorded_at: "2026-11-03T00:00:00Z", url: "https://api.openai.com/v1/chat/completions", request: { model: "gpt-4o", stream: true }, response: null }),
      JSON.stringify({ recorded_at: "2026-11-02T00:00:00Z", url: "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent", model: "models/gemini-3.8-flash", response: { usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 10 } } }),
      JSON.stringify({ recorded_at: "2026-11-02T00:00:00Z", url: "https://api.anthropic.com/v1/messages/count_tokens", request: { model: "claude-sonnet-5-5" }, response: { input_tokens: 9 } }),
      JSON.stringify({ recorded_at: "2026-11-02T00:00:00Z", url: "https://api.openai.com/v1/embeddings", model: "text-embedding-3-small", response: { usage: { prompt_tokens: 5, total_tokens: 5 } } }),
      "not json",
      JSON.stringify({ hello: "world" }),
    ].join("\n");
    const usage = usageFromLog(lines);
    expect(usage).toEqual([
      { model: "gpt-4o", platform: "openai-api", calls: 2, callsWithTokens: 1, inputTokens: 100, outputTokens: 20, first: "2026-11-01T00:00:00Z", last: "2026-11-03T00:00:00Z" },
      { model: "gemini-3.8-flash", platform: "gemini-api", calls: 1, callsWithTokens: 1, inputTokens: 50, outputTokens: 10, first: "2026-11-02T00:00:00Z", last: "2026-11-02T00:00:00Z" },
    ]);
  });
});

describe("monthly impact", () => {
  const usage = [
    { model: "gemini-3.8-flash", platform: "gemini-api" as const, calls: 300, callsWithTokens: 300, inputTokens: 300 * 1000, outputTokens: 300 * 200, first: "2026-11-01T00:00:00Z", last: "2026-11-11T00:00:00Z" },
    { model: "my-own-model", calls: 5, callsWithTokens: 5, inputTokens: 5, outputTokens: 5 },
  ];

  it("scales the log's span to 30 days and prices the change", () => {
    const s = estimateImpact(usage, { today: TODAY });
    expect(s.days).toBe(10);
    const row = s.rows[0]!;
    expect(row.callsPerMonth).toBe(900);
    // 1,000 input + 200 output tokens: $0.0015 a call now, $0.003 after the change.
    expect(row.monthlyNow).toBeCloseTo(1.35);
    expect(row.changes[0]?.monthlyAfter).toBeCloseTo(2.7);
    expect(row.changes[0]?.delta).toBeCloseTo(1.35);
    expect(s.unpriced).toEqual(["my-own-model"]);
  });

  it("guesses the platform from the model name when the log's host does not say", () => {
    expect(guessPlatform("gpt-4o")).toBe("openai-api");
    expect(guessPlatform("claude-sonnet-5-5")).toBe("anthropic-api");
    expect(guessPlatform("gemini-3.8-flash")).toBe("gemini-api");
    expect(guessPlatform("my-own-model")).toBeUndefined();
    const s = estimateImpact([{ model: "gpt-4o", calls: 10, callsWithTokens: 10, inputTokens: 10_000, outputTokens: 1000 }], { today: TODAY, perMonth: true });
    expect(s.rows[0]?.platform).toBe("openai-api");
  });

  it("prices a rerouted model at what it is billed now, and marks a change that already applies", () => {
    const s = estimateImpact([{ model: "gemini-2.5-pro", platform: "gemini-api", calls: 50_000, callsWithTokens: 1, inputTokens: 1500, outputTokens: 400 }], { today: "2026-12-15", perMonth: true });
    expect(s.rows[0]?.monthlyNow).toBeCloseTo(131.25);
    expect(s.rows[0]?.changes.map((c) => [c.row.kind, c.inEffect])).toEqual([
      ["reroute-billing", true],
      ["intro-ends", false],
    ]);
  });

  it("divides by the recorder's sample share", () => {
    expect(estimateImpact(usage, { today: TODAY, sample: 0.2 }).rows[0]?.callsPerMonth).toBe(4500);
  });

  it("uses given numbers as already per month", () => {
    const s = estimateImpact([{ model: "gemini-2.5-pro", platform: "gemini-api", calls: 50_000, callsWithTokens: 1, inputTokens: 1500, outputTokens: 400 }], { today: TODAY, perMonth: true });
    expect(s.days).toBeUndefined();
    expect(s.rows[0]?.monthlyNow).toBeCloseTo(293.75);
    // Rerouted on 1 Dec and billed at gemini-3.8-flash's rates.
    expect(s.rows[0]?.changes.map((c) => Math.round(c.monthlyAfter ?? -1))).toEqual([131, 263]);
  });
});

describe("refreshing a stored report", () => {
  it("re-reads retirement, reroute and price data from the catalog and recounts the days", () => {
    // A report stored before the catalog knew gemini-2.5-pro reroutes: no date, no reroute, no prices.
    const stale = buildReport(
      { root: "/tmp/x", filesScanned: 1, aiFiles: 1, callSites: 1, findings: [finding("a.py", 3, "gemini-2.5-pro", "google")], tokenTips: [] },
      "demo",
      TODAY,
    );
    const stored = { ...stale, models: stale.models.map(({ id, provider, platform, fineTuned, gateways, otherPlatforms, locations }) => ({ id, provider, platform, status: "no-known-retirement" as const, fineTuned, gateways, otherPlatforms, locations })), costs: [] };
    const fresh = refreshReport(stored, "2026-12-15");
    expect(fresh.models[0]).toMatchObject({ status: "rerouted", date: "2026-12-01", daysLeft: -14, reroutesTo: "gemini-3.8-flash", price: { input: 0.75, output: 3.75 }, locations: ["a.py:3"] });
    expect(fresh.costs?.map((c) => c.kind)).toEqual(["reroute-billing", "intro-ends"]);
    expect(fresh.today).toBe("2026-12-15");
  });
});

describe("isPriceRise", () => {
  it("is true only when the bill goes up or the new price is unknown", () => {
    const before = { input: 1, output: 2 };
    expect(isPriceRise({ before, after: { input: 2, output: 4 } })).toBe(true);
    expect(isPriceRise({ before, after: { input: 1, output: 3 } })).toBe(true);
    expect(isPriceRise({ before })).toBe(true);
    expect(isPriceRise({ before, after: { input: 0.5, output: 2 } })).toBe(false);
    expect(isPriceRise({ before, after: { input: 1, output: 2 } })).toBe(false);
    expect(isPriceRise({ before, after: { input: 2, output: 1 } })).toBe(false);
  });
});

describe("refreshing a row the catalog does not know", () => {
  it("keeps the stored date and recounts the days", () => {
    const base = buildReport({ root: "/tmp/x", filesScanned: 1, aiFiles: 1, callSites: 1, findings: [], tokenTips: [] }, "demo", TODAY);
    const row = { id: "made-up-model", provider: "openai" as const, platform: "openai-api" as const, status: "rerouting" as const, date: "2026-12-01", daysLeft: 8, reroutesTo: "x", fineTuned: false, gateways: [], otherPlatforms: [], locations: ["a.py:1"] };
    const fresh = refreshReport({ ...base, models: [row] }, "2026-12-05");
    expect(fresh.models[0]).toMatchObject({ status: "rerouted", daysLeft: -4, date: "2026-12-01", reroutesTo: "x" });
  });
});
