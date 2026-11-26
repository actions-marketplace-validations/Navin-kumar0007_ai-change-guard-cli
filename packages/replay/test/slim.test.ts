import type { LlmClient, LlmRequest } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import type { GoldenItem } from "../src/golden.js";
import { SpendCapError } from "../src/run.js";
import { formatSlimMarkdown, formatSlimText, parseSlimAnswer, placeholders, rewriteProblem, slimPrompts } from "../src/slim.js";

const TODAY = "2026-11-23";
const LONG =
  "You are the order assistant for Acme Store. Please always read the customer's message very carefully. " +
  "Extract the order from the message and reply with JSON only, with the keys item and qty. " +
  "Do not add any explanation, greeting or extra text of any kind. Never invent items that are not in the message.";
const CONCISE = "Extract the order as JSON with keys item and qty. JSON only, no extra text. Never invent items.";
const MINIMAL = "Reply JSON {item, qty}.";

const items = (system: string, n: number, prefix: string): GoldenItem[] =>
  Array.from({ length: n }, (_, i) => ({ id: `${prefix}-${i + 1}`, system, messages: [{ role: "user", content: `I want ${i + 2} apples` }], expect: { json: true } }));

const isWriter = (req: LlmRequest) => req.system?.startsWith("You shorten");
const answer = (text: string, system = "") => ({ text, toolCalls: [], inputTokens: 40 + Math.round(system.length / 4), outputTokens: 12, latencyMs: 1, refused: false });

/** The writer offers CONCISE then MINIMAL; the app model keeps the JSON with CONCISE and breaks it with MINIMAL. */
function fake(calls: { writer: number; app: number }, rewrites: string[] = [CONCISE, MINIMAL]): LlmClient {
  let next = 0;
  return async (req) => {
    if (isWriter(req)) {
      calls.writer++;
      return answer(JSON.stringify({ prompt: rewrites[next++ % rewrites.length] }));
    }
    calls.app++;
    const qty = /(\d+)/.exec(req.messages[0]!.content)![1];
    return answer(req.system === MINIMAL ? `Sure! ${qty} apples.` : `{"item": "apple", "qty": ${qty}}`, req.system);
  };
}

describe("rewrite checks", () => {
  it("compares placeholders and needs a real cut", () => {
    expect(placeholders("Hi {name}, see ${url} and {{literal}}")).toEqual(["${url}", "{name}", "{{literal}}"]);
    expect(rewriteProblem(LONG, CONCISE)).toBeUndefined();
    expect(rewriteProblem(LONG, LONG.slice(0, LONG.length - 5))).toMatch(/only \d+% shorter/);
    expect(rewriteProblem("Greet {name} warmly and kindly, always in English please.", "Greet warmly.")).toMatch(/placeholders/);
    expect(rewriteProblem(LONG, "  ")).toMatch(/empty/);
  });

  it("reads the writer's JSON, with or without text around it", () => {
    expect(parseSlimAnswer('{"prompt": " Short. "}')).toBe("Short.");
    expect(parseSlimAnswer('Here you go:\n{"prompt": "Short."}\nDone')).toBe("Short.");
    expect(parseSlimAnswer('{"text": "x"}')).toBeUndefined();
  });
});

describe("slimPrompts", () => {
  it("proposes the shorter prompt that keeps every answer, with the measured saving", async () => {
    const calls = { writer: 0, app: 0 };
    const r = await slimPrompts({ items: items(LONG, 3, "order"), model: "openai:gpt-5.5", client: fake(calls), runs: 2, concurrency: 1, callsPerMonth: 100_000, today: TODAY });
    const p = r.prompts[0]!;
    expect(p.best?.style).toBe("concise");
    expect(p.best?.prompt).toBe(CONCISE);
    expect(p.best?.report?.totals).toMatchObject({ broken: 0, changed: 0 });
    expect(p.best!.inputTokens!.new).toBeLessThan(p.best!.inputTokens!.old);
    expect(p.best!.savingShare!).toBeGreaterThan(0.03);
    expect(p.best!.monthlySaving!).toBeGreaterThan(0);
    const minimal = p.variants.find((v) => v.style === "minimal")!;
    expect(minimal.qualifies).toBe(false);
    expect(minimal.reason).toMatch(/broken in at least one run \(first: order-\d/);
    // Old prompt once (3 × 2), each rewrite once (3 × 2 each), plus 2 writer calls.
    expect(calls).toEqual({ writer: 2, app: 18 });
    expect(r.calls).toBe(20);
    expect(formatSlimText(r)).toContain(`SAVE  concise rewrite, ${CONCISE.length} characters`);
    const md = formatSlimMarkdown(r);
    expect(md).toContain("**Use the concise rewrite");
    expect(md).toContain(CONCISE);
    expect(md).toMatch(/\| minimal \(\d+ chars\) \| Skipped: /);
  });

  it("skips system prompts with too few prompts, and spends no replay on a rewrite that drops a placeholder", async () => {
    const calls = { writer: 0, app: 0 };
    const templated = "Answer the customer called {name} about their order, politely and briefly, in plain English always.";
    const set = [...items(templated, 3, "t"), ...items(LONG, 2, "few"), { id: "bare", messages: [{ role: "user" as const, content: "hi" }] }];
    const r = await slimPrompts({ items: set, model: "openai:gpt-5.5", client: fake(calls, ["Answer briefly."]), runs: 1, concurrency: 1, styles: ["concise"], today: TODAY });
    expect(r.withoutSystem).toBe(1);
    expect(r.prompts.find((p) => p.system === LONG)?.reason).toMatch(/only 2 prompts use it/);
    const t = r.prompts.find((p) => p.system === templated)!;
    expect(t.best).toBeUndefined();
    expect(t.variants[0]).toMatchObject({ qualifies: false, reason: "the rewrite changed the template placeholders" });
    expect(calls).toEqual({ writer: 1, app: 3 }); // the old prompt's answers only
    expect(formatSlimText(r)).toContain("No shorter prompt answered your prompts the same way");
  });

  it("refuses before any call when the plan is over the call limit", async () => {
    const calls = { writer: 0, app: 0 };
    await expect(slimPrompts({ items: items(LONG, 10, "o"), model: "openai:gpt-5.5", client: fake(calls), runs: 5, maxCalls: 100, today: TODAY })).rejects.toThrow(SpendCapError);
    expect(calls).toEqual({ writer: 0, app: 0 });
  });
});
