import type { LlmClient, LlmRequest } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import { generateStarterSet, parseStarterAnswer } from "../src/starter.js";

const base = { toolCalls: [], inputTokens: 1, outputTokens: 1, latencyMs: 1, refused: false };

describe("starter set", () => {
  it("keeps each system prompt as written and adds generated user messages, one call per prompt", async () => {
    const calls: LlmRequest[] = [];
    const client: LlmClient = async (req) => {
      calls.push(req);
      return { ...base, text: req.messages[0]!.content.includes("Classify") ? '{"messages": ["Charged twice!!", "where is my order"]}' : '["hola", "hi"]' };
    };
    const r = await generateStarterSet({
      sources: [
        { id: "bot.py-4", system: "Classify the ticket as JSON.", json: true },
        { id: "app.ts-2", system: "Translate to Hindi.", json: false },
      ],
      model: "openai:gpt-4o-mini",
      client,
      count: 2,
    });
    expect(r.items).toEqual([
      { id: "bot.py-4-1", system: "Classify the ticket as JSON.", messages: [{ role: "user", content: "Charged twice!!" }], expect: { json: true } },
      { id: "bot.py-4-2", system: "Classify the ticket as JSON.", messages: [{ role: "user", content: "where is my order" }], expect: { json: true } },
      { id: "app.ts-2-1", system: "Translate to Hindi.", messages: [{ role: "user", content: "hola" }] },
      { id: "app.ts-2-2", system: "Translate to Hindi.", messages: [{ role: "user", content: "hi" }] },
    ]);
    expect(r.calls).toBe(2);
    expect(calls.map((c) => [c.model, c.jsonMode])).toEqual([
      ["openai:gpt-4o-mini", true],
      ["openai:gpt-4o-mini", true],
    ]);
    expect(calls[0]!.messages[0]!.content).toContain("<system_prompt>\nClassify the ticket as JSON.\n</system_prompt>");
  });

  it("reports a prompt it could not generate for, and carries on", async () => {
    let n = 0;
    const client: LlmClient = async () => {
      n++;
      if (n === 1) throw new Error("HTTP 429");
      if (n === 2) return { ...base, text: "Sorry, I can't help with that." };
      return { ...base, text: '{"messages": ["ok"]}' };
    };
    const r = await generateStarterSet({
      sources: ["a", "b", "c"].map((id) => ({ id, system: "S", json: false })),
      model: "openai:x",
      client,
    });
    expect(r.failures).toEqual([
      { id: "a", error: "HTTP 429" },
      { id: "b", error: "the model did not return a list of messages" },
    ]);
    expect(r.items.map((i) => i.id)).toEqual(["c-1"]);
  });

  it("reads JSON wrapped in prose or fences, drops duplicates and non-strings, and stops at the count", () => {
    expect(parseStarterAnswer('Here you go:\n```json\n{"messages": ["a", "a", 3, " b ", ""]}\n```', 5)).toEqual(["a", "b"]);
    expect(parseStarterAnswer('Sure! {"messages": ["x", "y", "z"]} Hope that helps.', 2)).toEqual(["x", "y"]);
    expect(parseStarterAnswer("no json here", 3)).toEqual([]);
  });
});
