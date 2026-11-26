import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LlmClient } from "@ai-changeguard/llm";
import { afterEach, describe, expect, it } from "vitest";
import { run } from "../src/main.js";

const dirs: string[] = [];
afterEach(async () => {
  await Promise.all(dirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
});

async function workspace(files: Record<string, string>): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "acg-action-"));
  dirs.push(dir);
  for (const [path, content] of Object.entries(files)) {
    await mkdir(join(dir, path, ".."), { recursive: true });
    await writeFile(join(dir, path), content);
  }
  return dir;
}

async function runAction(files: Record<string, string>, inputs: Record<string, string>, client?: LlmClient) {
  const dir = await workspace(files);
  const env: Record<string, string> = { GITHUB_WORKSPACE: dir, GITHUB_STEP_SUMMARY: join(dir, "summary.md") };
  for (const [k, v] of Object.entries(inputs)) env[`INPUT_${k.toUpperCase()}`] = v;
  const lines: string[] = [];
  const code = await run({ env, today: "2026-11-21", log: (l) => lines.push(l), client });
  return { code, lines, summary: await readFile(env.GITHUB_STEP_SUMMARY!, "utf8") };
}

const RETIRED = { "bot.py": 'from anthropic import Anthropic\nclient.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100)\n' };

describe("GitHub Action", () => {
  it("fails on a retired model and writes a job summary without a PR column", async () => {
    const { code, lines, summary } = await runAction(RETIRED, {});
    expect(code).toBe(1);
    expect(lines[0]).toMatch(/^::error::AI Change Guard: AI models and APIs: 1 retired/);
    expect(summary).toContain("`claude-3-5-sonnet-20241022`");
    expect(summary).toContain("| Model | Status | Where |\n");
    expect(summary).not.toContain("In this PR");
  });

  it("passes with fail-on: none", async () => {
    expect((await runAction(RETIRED, { "fail-on": "none" })).code).toBe(0);
  });

  it("rejects an unknown fail-on value", async () => {
    await expect(runAction(RETIRED, { "fail-on": "sometimes" })).rejects.toThrow(/fail-on/);
  });

  it("needs all three replay inputs together", async () => {
    await expect(runAction({}, { "old-model": "openai:a" })).rejects.toThrow(/all three inputs/);
  });

  it("runs a replay and fails when the new model breaks JSON output", async () => {
    const client: LlmClient = async (req) => ({
      text: req.model === "openai:old" ? '{"ok": true}' : "Sure! Here you go: ok",
      toolCalls: [],
      inputTokens: 10,
      outputTokens: 5,
      latencyMs: 1,
      refused: false,
    });
    const files = { "golden.jsonl": '{"id":"a","input":"Reply in JSON","expect":{"json":true}}\n' };
    const { code, lines, summary } = await runAction(
      files,
      { prompts: "golden.jsonl", "old-model": "openai:old", "new-model": "openai:new", runs: "2", concurrency: "1" },
      client,
    );
    expect(code).toBe(1);
    expect(lines.at(-1)).toBe("::error::AI Change Guard replay: 0 same, 0 changed, 1 broken");
    expect(summary).toContain("AI Change Guard: 0 AI models found");
  });
});

describe("bundled catalog data", () => {
  it("matches packages/catalog/data (run npm run build:action after catalog changes)", async () => {
    for (const file of ["models.json", "apis.json"]) {
      const bundled = await readFile(new URL(`../data/${file}`, import.meta.url), "utf8");
      const source = await readFile(new URL(`../../packages/catalog/data/${file}`, import.meta.url), "utf8");
      expect(bundled, file).toBe(source);
    }
  });
});

describe("Prompt Repair in the Action", () => {
  it("shows the fix in the Summary and still fails the step until it is applied", async () => {
    const client: LlmClient = async (req) => ({
      text: req.model === "openai:old" || req.system?.includes("Answer with one JSON value only") ? '{"ok": true}' : "Sure! Here you go: ok",
      toolCalls: [],
      inputTokens: 10,
      outputTokens: 5,
      latencyMs: 1,
      refused: false,
    });
    const files = { "golden.jsonl": '{"id":"a","input":"Reply in JSON","expect":{"json":true}}\n' };
    const inputs = { prompts: "golden.jsonl", "old-model": "openai:old", "new-model": "openai:new", runs: "2", concurrency: "1", repair: "true" };
    const { code, lines, summary } = await runAction(files, inputs, client);
    expect(code).toBe(1);
    expect(lines).toContain("AI Change Guard Prompt Repair: 1 of 1 format breaks fixed (2 extra calls); the fixes are in the Summary");
    expect(summary).toContain("### Prompt Repair");
    expect(summary).toContain("**Change for `a`:** add this to the end of the system prompt:");
  });

  it("skips repair when max-calls has no room left, and still reports the replay", async () => {
    const client: LlmClient = async (req) => ({
      text: req.model === "openai:old" ? '{"ok": true}' : "Sure! Here you go: ok",
      toolCalls: [],
      inputTokens: 10,
      outputTokens: 5,
      latencyMs: 1,
      refused: false,
    });
    const files = { "golden.jsonl": '{"id":"a","input":"Reply in JSON","expect":{"json":true}}\n' };
    const inputs = { prompts: "golden.jsonl", "old-model": "openai:old", "new-model": "openai:new", runs: "2", concurrency: "1", "max-calls": "5", repair: "true" };
    const { code, lines } = await runAction(files, inputs, client);
    expect(code).toBe(1);
    expect(lines).toContain("::warning::AI Change Guard Prompt Repair skipped: it could need 4 calls, and only 1 are left under the limit of 5.");
    expect(lines.at(-1)).toBe("::error::AI Change Guard replay: 0 same, 0 changed, 1 broken");
  });
});

describe("watch mode", () => {
  it("passes quietly when the model answers like the baseline, and fails with a summary when it breaks", async () => {
    const { createBaseline, parseGoldenSet } = await import("@ai-changeguard/replay");
    const golden = '{"id":"a","input":"Reply in JSON","expect":{"json":true}}\n';
    const answer = (text: string) => ({ text, toolCalls: [], inputTokens: 1, outputTokens: 1, latencyMs: 1, refused: false, servedModel: "gpt-4o-2024-08-06" });
    const steady: LlmClient = async () => answer('{"ok": true}');
    const baseline = await createBaseline({ items: parseGoldenSet(golden), model: "openai:gpt-4o", client: steady, runs: 5, concurrency: 1 });
    const files = { "tests/golden.jsonl": golden, "tests/golden.baseline.json": JSON.stringify(baseline) };
    // runs: "" is what GitHub passes for the input's default; the watch then uses the baseline's 5 runs.
    const inputs = { "fail-on": "none", prompts: "tests/golden.jsonl", baseline: "tests/golden.baseline.json", concurrency: "1", runs: "" };

    const quiet = await runAction(files, inputs, steady);
    expect(quiet.code).toBe(0);
    expect(quiet.summary).toContain("AI model watch: no change");
    expect(quiet.lines.at(-1)).toBe("AI Change Guard watch: 1 same, 0 changed, 0 broken");
    expect(quiet.summary).toContain("| 1 | 0 | 0 | 0 |");

    const loud = await runAction(files, inputs, async () => answer("Sure, here you go"));
    expect(loud.code).toBe(1);
    expect(loud.summary).toContain("AI model watch: change detected");
    expect(loud.lines.at(-1)).toBe("::error::AI Change Guard watch: 0 same, 0 changed, 1 broken");
  });

  it("rejects mixing a watch with a model switch", async () => {
    await expect(runAction({}, { prompts: "g.jsonl", baseline: "b.json", "old-model": "openai:a" })).rejects.toThrow(/watch of one model/);
  });
});

describe("prompt slimming in the Action", () => {
  const SYSTEM = "You are the order assistant for Acme Store. Please always read the message very carefully. Reply with JSON only, keys item and qty, and never add extra text.";
  const SET = [1, 2, 3].map((n) => JSON.stringify({ id: `o${n}`, system: SYSTEM, input: `I want ${n} apples`, expect: { json: true } })).join("\n");
  const SHORT = "Reply JSON only: keys item, qty. No extra text.";
  const client: LlmClient = async (req) => ({
    text: req.system?.startsWith("You shorten") ? JSON.stringify({ prompt: SHORT }) : `{"item": "apple", "qty": ${/\d+/.exec(req.messages[0]!.content)![0]}}`,
    toolCalls: [],
    inputTokens: 20 + Math.round((req.system ?? "").length / 4),
    outputTokens: 10,
    latencyMs: 1,
    refused: false,
  });

  it("puts the slimming card in the Summary", async () => {
    const { code, summary, lines } = await runAction({ "golden.jsonl": SET }, { prompts: "golden.jsonl", "old-model": "openai:gpt-5.5", "slim-prompts": "true", "fail-on": "none", runs: "1" }, client);
    expect(code).toBe(0);
    expect(summary).toContain("## Prompt slimming: `openai:gpt-5.5`");
    expect(summary).toContain(SHORT);
    expect(lines).toContain("AI Change Guard prompt slimming: 1 shorter system prompt(s) passed");
  });

  it("needs prompts and old-model, and keeps the model", async () => {
    await expect(runAction({}, { "slim-prompts": "true", "old-model": "openai:gpt-5.5" })).rejects.toThrow(/slim-prompts needs/);
    await expect(runAction({ "golden.jsonl": SET }, { "slim-prompts": "true", prompts: "golden.jsonl", "old-model": "openai:a", "new-model": "openai:b" })).rejects.toThrow(/keeps your model/);
    await expect(runAction({ "golden.jsonl": SET }, { "slim-prompts": "true", "find-savings": "true", prompts: "golden.jsonl", "old-model": "openai:a" })).rejects.toThrow(/own step/);
  });
});
