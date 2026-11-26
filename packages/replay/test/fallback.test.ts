import type { LlmClient, LlmRequest } from "@ai-changeguard/llm";
import { describe, expect, it } from "vitest";
import { litellmConfig, openrouterConfig, testFallbacks } from "../src/fallback.js";
import { parseGoldenSet } from "../src/golden.js";
import { SpendCapError } from "../src/run.js";

const SET = parseGoldenSet(
  ['{"id":"order","input":"Extract the order as JSON: 2 apples","expect":{"json":true}}', '{"id":"greet","input":"Greet Asha","expect":{"contains":["Asha"]}}'].join("\n"),
);
const ok = (text: string) => ({ text, toolCalls: [], inputTokens: 1, outputTokens: 1, latencyMs: 1, refused: false });
const isOrder = (req: LlmRequest) => req.messages[0]!.content.startsWith("Extract");

/** primary and "good" answer well, "drifty" rewords the greeting, "bad" breaks the JSON. */
function client(calls: string[]): LlmClient {
  return async (req) => {
    calls.push(req.model);
    if (isOrder(req)) return ok(req.model === "openai:bad" ? "Sure! 2 apples." : '{"item": "apple", "qty": 2}');
    return ok(req.model === "google:drifty" ? "Greetings and salutations to you, dear Asha, on this fine day." : "Hello Asha!");
  };
}

describe("testFallbacks", () => {
  it("keeps candidates with nothing broken, best first, and calls the primary only once", async () => {
    const calls: string[] = [];
    const r = await testFallbacks({ items: SET, primary: "openai:gpt-4o", candidates: ["openai:bad", "google:drifty", "anthropic:good"], client: client(calls), runs: 2, concurrency: 1 });
    expect(r.candidates.map((c) => [c.model, c.qualifies, c.reason])).toEqual([
      ["openai:bad", false, "1 prompt broken in at least one run"],
      ["google:drifty", true, undefined],
      ["anthropic:good", true, undefined],
    ]);
    expect(r.fallbacks).toEqual(["anthropic:good", "google:drifty"]);
    expect(calls.filter((m) => m === "openai:gpt-4o")).toHaveLength(4); // 2 prompts × 2 runs, once
    expect(r.calls).toBe(16);
  });

  it("refuses to start above the call limit, and needs a real candidate", async () => {
    await expect(testFallbacks({ items: SET, primary: "openai:a", candidates: ["openai:b", "openai:c"], client: client([]), runs: 3, maxCalls: 10 })).rejects.toThrow(SpendCapError);
    await expect(testFallbacks({ items: SET, primary: "openai:a", candidates: ["openai:a"], client: client([]) })).rejects.toThrow(/at least one candidate/);
  });
});

describe("configs", () => {
  it("writes a LiteLLM config with key names from the environment and fallbacks in order", () => {
    const yaml = litellmConfig("openai:gpt-4o", ["anthropic:claude-sonnet-5-5", "google:gemini-3.1-flash-lite"], "2 prompts × 3 runs");
    expect(yaml).toBe(
      [
        "# AI Change Guard: fallbacks that passed your golden set (2 prompts × 3 runs)",
        "model_list:",
        '  - model_name: "gpt-4o"',
        "    litellm_params:",
        '      model: "openai/gpt-4o"',
        '      api_key: "os.environ/OPENAI_API_KEY"',
        '  - model_name: "claude-sonnet-5-5"',
        "    litellm_params:",
        '      model: "anthropic/claude-sonnet-5-5"',
        '      api_key: "os.environ/ANTHROPIC_API_KEY"',
        '  - model_name: "gemini-3.1-flash-lite"',
        "    litellm_params:",
        '      model: "gemini/gemini-3.1-flash-lite"',
        '      api_key: "os.environ/GEMINI_API_KEY"',
        "",
        "router_settings:",
        '  fallbacks: [{"gpt-4o": ["claude-sonnet-5-5", "gemini-3.1-flash-lite"]}]',
        "",
      ].join("\n"),
    );
  });

  it("gives unique names when two providers serve the same model id", () => {
    const yaml = litellmConfig("openai:gpt-4o", ["openrouter:openai/gpt-4o"]);
    expect(yaml).toContain('fallbacks: [{"gpt-4o": ["gpt-4o-2"]}]');
  });

  it("lists OpenRouter ids in priority order and flags the ones to check", () => {
    expect(openrouterConfig("openrouter:openai/gpt-4o", ["anthropic:claude-sonnet-5-5", "mistral:mistral-large-2411", "ollama:llama3"])).toEqual({
      body: { models: ["openai/gpt-4o", "anthropic/claude-sonnet-5-5", "mistralai/mistral-large-2411"] },
      unchecked: ["anthropic/claude-sonnet-5-5", "mistralai/mistral-large-2411"],
      skipped: ["ollama:llama3"],
    });
  });

  it("says why a model that never answers was left out", async () => {
    const gone: LlmClient = async (req) => {
      if (req.model === "google:retired") throw new Error("HTTP 404: model not found");
      return ok(isOrder(req) ? '{"a": 1}' : "Hello Asha!");
    };
    const r = await testFallbacks({ items: SET, primary: "openai:gpt-4o", candidates: ["google:retired"], client: gone, runs: 1, concurrency: 1 });
    expect(r.candidates[0]).toMatchObject({ qualifies: false, reason: "every call failed: HTTP 404: model not found" });
    expect(r.fallbacks).toEqual([]);
  });

});

describe("review fixes", () => {
  const reply = (req: LlmRequest, served?: string) => ({ ...ok(isOrder(req) ? '{"item": "apple", "qty": 2}' : "Hello Asha!"), ...(served ? { servedModel: served } : {}) });

  it("stops before testing any backup when the primary does not answer at all", async () => {
    const calls: string[] = [];
    const down: LlmClient = async (req) => {
      calls.push(req.model);
      if (req.model === "openai:gpt-4o") throw new Error("HTTP 503: overloaded");
      return reply(req);
    };
    await expect(testFallbacks({ items: SET, primary: "openai:gpt-4o", candidates: ["anthropic:good"], client: down, runs: 1, concurrency: 1 })).rejects.toThrow(
      /openai:gpt-4o did not answer any prompt \(HTTP 503: overloaded\)/,
    );
    expect(calls.every((m) => m === "openai:gpt-4o")).toBe(true);
  });

  it("leaves out prompts the primary could not answer instead of blaming every backup", async () => {
    const partial: LlmClient = async (req) => {
      if (req.model === "openai:gpt-4o" && !isOrder(req)) throw new Error("HTTP 400: content filter");
      return reply(req);
    };
    const r = await testFallbacks({ items: SET, primary: "openai:gpt-4o", candidates: ["anthropic:good"], client: partial, runs: 2, concurrency: 1 });
    expect(r.skippedPrompts).toEqual(["greet"]);
    expect(r.fallbacks).toEqual(["anthropic:good"]);
  });

  it("disqualifies a backup that breaks a prompt in even one run", async () => {
    let n = 0;
    const flaky: LlmClient = async (req) => (req.model === "google:flaky" && isOrder(req) && n++ === 0 ? ok("Sure! 2 apples.") : reply(req));
    const r = await testFallbacks({ items: SET, primary: "openai:gpt-4o", candidates: ["google:flaky"], client: flaky, runs: 3, concurrency: 1 });
    expect(r.candidates[0]).toMatchObject({ qualifies: false, reason: "1 prompt broken in at least one run" });
  });

  it("disqualifies a backup the provider served as another model", async () => {
    const swapped: LlmClient = async (req) => reply(req, req.model === "openrouter:acme/cheap" ? "other/model-x" : undefined);
    const r = await testFallbacks({ items: SET, primary: "openai:gpt-4o", candidates: ["openrouter:acme/cheap"], client: swapped, runs: 1, concurrency: 1 });
    expect(r.candidates[0]).toMatchObject({ qualifies: false, reason: "served as other/model-x, not the model requested" });
  });

  it("will not build an OpenRouter list without the primary or without a backup", () => {
    expect(() => openrouterConfig("groq:llama-3.3-70b", ["anthropic:x"])).toThrow(/OpenRouter does not serve groq:llama-3.3-70b/);
    expect(() => openrouterConfig("openai:gpt-4o", ["ollama:llama3"])).toThrow(/None of the tested backups is available on OpenRouter/);
  });
});

