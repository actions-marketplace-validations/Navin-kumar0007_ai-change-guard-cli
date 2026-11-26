import type { LlmClient, LlmRequest } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import { parseGoldenSet } from "../src/golden.js";
import { cheaperCandidates, findSavings, formatSavingsMarkdown, formatSavingsText } from "../src/savings.js";

const TODAY = "2026-11-23";
const SET = parseGoldenSet(
  ['{"id":"order","input":"Extract the order as JSON: 2 apples","expect":{"json":true}}', '{"id":"greet","input":"Greet Asha","expect":{"contains":["Asha"]}}'].join("\n"),
);
const ok = (text: string) => ({ text, toolCalls: [], inputTokens: 1000, outputTokens: 200, latencyMs: 1, refused: false });
const isOrder = (req: LlmRequest) => req.messages[0]!.content.startsWith("Extract");

/** gpt-5.4 answers like gpt-5.5; gpt-5.2 rewords the greeting; gpt-6-luna breaks the JSON. */
function client(calls: string[]): LlmClient {
  return async (req) => {
    calls.push(req.model);
    if (isOrder(req)) return ok(req.model.endsWith("gpt-6-luna") ? "Sure! 2 apples." : '{"item": "apple", "qty": 2}');
    return ok(req.model.endsWith("gpt-5.2") ? "Greetings and salutations to you, dear Asha, on this fine day." : "Hello Asha!");
  };
}

describe("cheaperCandidates", () => {
  it("picks the nearest cheaper models on the same provider", () => {
    expect(cheaperCandidates("openai:gpt-5.5", TODAY).map((c) => c.ref)).toEqual(["openai:gpt-5.4", "openai:gpt-5.2"]);
    expect(cheaperCandidates("openai:gpt-6-luna", TODAY)).toEqual([]);
    expect(cheaperCandidates("ollama:llama3", TODAY)).toEqual([]);
  });

  it("prices candidates at their price two months out, so an ending introductory price can't win", () => {
    // Gemini 3.8 Flash costs $0.75/$3.75 until 31 Dec and $1.50/$7.50 after: ranked at the higher price.
    const picks = cheaperCandidates("google:gemini-3.1-pro-preview", TODAY, { tokens: { input: 1000, output: 250 } });
    const flash = picks.find((c) => c.ref === "google:gemini-3.8-flash");
    expect(flash?.costPerCall).toBeCloseTo((1000 * 1.5 + 250 * 7.5) / 1e6);
  });

  it("counts the tokenizer: Sonnet 5.5 is cheaper than Sonnet 4.6 even with ~30% more tokens", () => {
    const picks = cheaperCandidates("anthropic:claude-sonnet-4-6", TODAY, { tokens: { input: 1000, output: 250 } });
    const s55 = picks.find((c) => c.ref === "anthropic:claude-sonnet-5-5");
    expect(s55?.costPerCall).toBeCloseTo((1300 * 2 + 325 * 10) / 1e6);
  });

  it("does not suggest models that retire soon or have no price in two months", () => {
    const refs = cheaperCandidates("openai:gpt-6.2-astra", TODAY, { count: 50 }).map((c) => c.ref);
    expect(refs).not.toContain("openai:gpt-4o"); // retires 14 Jan 2027
    expect(refs).not.toContain("openai:gpt-5.6-sol"); // price after its promotion is not published
  });
});

describe("findSavings", () => {
  it("passes a cheaper model only with nothing broken and at most 2% changed, with the measured saving", async () => {
    const calls: string[] = [];
    const r = await findSavings({
      items: SET,
      primary: "openai:gpt-5.5",
      candidates: ["openai:gpt-5.4", "openai:gpt-5.2", "openai:gpt-6-luna"],
      client: client(calls),
      runs: 2,
      concurrency: 1,
      callsPerMonth: 100_000,
      today: TODAY,
    });
    expect(r.candidates.map((c) => [c.model, c.qualifies])).toEqual([
      ["openai:gpt-5.4", true],
      ["openai:gpt-5.2", false],
      ["openai:gpt-6-luna", false],
    ]);
    expect(r.candidates[1]!.reason).toBe("1 of 2 prompts answered differently (more than 2%)");
    expect(r.candidates[2]!.reason).toContain("broken");
    // 1,000 input + 200 output tokens: gpt-5.5 $0.011 a call, gpt-5.4 $0.0055.
    expect(r.best?.model).toBe("openai:gpt-5.4");
    expect(r.best?.costPerCall?.current).toBeCloseTo(0.011);
    expect(r.best?.costPerCall?.candidate).toBeCloseTo(0.0055);
    expect(r.best?.savingShare).toBeCloseTo(0.5);
    expect(r.best?.monthlySaving).toBeCloseTo(550);
    expect(formatSavingsText(r)).toContain("SAVE  openai:gpt-5.5 → openai:gpt-5.4: 50% cheaper per call ($0.011 → $0.0055), about $550 a month");
    expect(formatSavingsText(r)).toContain("Switch: ai-change-guard switch . --from gpt-5.5 --to gpt-5.4 --write");
    expect(formatSavingsMarkdown(r)).toContain("**Switch to `openai:gpt-5.4`: 50% cheaper per call");
  });

  it("says to keep the model when nothing cheaper passes, and spends nothing without candidates", async () => {
    const none = await findSavings({ items: SET, primary: "openai:gpt-5.5", candidates: ["openai:gpt-6-luna"], client: client([]), runs: 1, today: TODAY });
    expect(none.best).toBeUndefined();
    expect(formatSavingsText(none)).toContain("No cheaper model answered your prompts the same way");
    const calls: string[] = [];
    const empty = await findSavings({ items: SET, primary: "openai:gpt-6-luna", client: client(calls), today: TODAY });
    expect(empty.candidates).toEqual([]);
    expect(calls).toEqual([]);
  });
});

describe("review fixes", () => {
  const same: LlmClient = async (req) => ok(isOrder(req) ? '{"item": "apple", "qty": 2}' : "Hello Asha!");

  it("prices the saving at the steady price, so an ending introductory price does not count", async () => {
    // 1,000 input + 200 output tokens. gemini-3.1-pro-preview $2/$12: $0.0044. gemini-3.8-flash at its
    // January price $1.50/$7.50: $0.003 (not the $0.0015 of its introductory price).
    const r = await findSavings({ items: SET, primary: "google:gemini-3.1-pro-preview", candidates: ["google:gemini-3.8-flash"], client: same, runs: 1, today: TODAY });
    expect(r.best?.costPerCall?.current).toBeCloseTo(0.0044);
    expect(r.best?.costPerCall?.candidate).toBeCloseTo(0.003);
  });

  it("does not test a given candidate that retires soon, a preview or a gateway, and spends nothing on it", async () => {
    const calls: string[] = [];
    const r = await findSavings({ items: SET, primary: "openai:gpt-5.5", candidates: ["openai:gpt-5-nano", "google:gemini-3-flash-preview", "groq:llama"], client: client(calls), today: TODAY });
    expect(r.excluded).toEqual([
      { model: "openai:gpt-5-nano", reason: "retires on 2026-12-11" },
      { model: "google:gemini-3-flash-preview", reason: "a preview model" },
      { model: "groq:llama", reason: "not priced in the catalog (a gateway or unknown provider)" },
    ]);
    expect(calls).toEqual([]);
    expect(formatSavingsText(r)).toContain("NOT TESTED  openai:gpt-5-nano: retires on 2026-12-11");
  });

  it("warns when the current model did not answer some prompts", async () => {
    const flaky: LlmClient = async (req) => {
      if (req.model === "openai:gpt-5.5" && !isOrder(req)) throw new Error("HTTP 500");
      return ok(isOrder(req) ? '{"item": "apple", "qty": 2}' : "Hello Asha!");
    };
    const r = await findSavings({ items: SET, primary: "openai:gpt-5.5", candidates: ["openai:gpt-5.4"], client: flaky, runs: 1, today: TODAY });
    expect(r.skippedPrompts).toEqual(["greet"]);
    expect(formatSavingsText(r)).toContain("WARNING: openai:gpt-5.5 did not answer 1 prompt (greet), so it was not compared.");
    expect(formatSavingsMarkdown(r)).toContain("> **Warning:**");
  });

  it("says a dearer candidate costs more, not a negative saving", async () => {
    const r = await findSavings({ items: SET, primary: "openai:gpt-5.4", candidates: ["openai:gpt-5.5"], client: same, runs: 1, today: TODAY });
    expect(r.candidates[0]?.reason).toBe("costs more on your prompts (+100% per call)");
  });
});

describe("live test fixes (2026-11-24)", () => {
  it("does not pick a model closed to new API users", () => {
    // Live test: gemini-2.5-flash-lite answered HTTP 404 "no longer available to new users".
    expect(cheaperCandidates("google:gemini-3.5-flash-lite", "2026-11-24").map((c) => c.ref)).toEqual(["google:gemini-3.1-flash-lite"]);
  });

  it("names the first broken prompt and why", async () => {
    const r = await findSavings({ items: SET, primary: "openai:gpt-5.5", candidates: ["openai:gpt-6-luna"], client: client([]), runs: 1, today: TODAY });
    expect(r.candidates[0]?.reason).toMatch(/^1 prompt broken in at least one run \(first: order, .+\)$/);
  });
});
