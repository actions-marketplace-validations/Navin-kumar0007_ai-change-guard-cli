import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LlmClient } from "@ai-changeguard/llm";
import { Client, InMemoryTransport } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it } from "vitest";
import { createServer } from "../src/server.js";

const dirs: string[] = [];
const clients: Client[] = [];
afterEach(async () => {
  await Promise.all(clients.splice(0).map((c) => c.close()));
  await Promise.all(dirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
});

async function project(files: Record<string, string>): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "acg-mcp-"));
  dirs.push(dir);
  for (const [path, content] of Object.entries(files)) {
    await mkdir(join(dir, path, ".."), { recursive: true });
    await writeFile(join(dir, path), content);
  }
  return dir;
}

/** A real MCP client talking to the server over the SDK's in-memory transport pair. */
async function connect(root: string, llm?: LlmClient): Promise<Client> {
  const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
  await createServer({ root, today: "2026-11-23", ...(llm ? { client: llm } : {}) }).connect(serverSide);
  const client = new Client({ name: "test", version: "1.0.0" });
  await client.connect(clientSide);
  clients.push(client);
  return client;
}

async function call(client: Client, name: string, args: Record<string, unknown> = {}) {
  const result = await client.callTool({ name, arguments: args });
  const texts = (result.content as { type: string; text?: string }[]).map((c) => c.text ?? "");
  return { isError: result.isError === true, text: texts.join("\n") };
}

const APP = {
  "bot.py": 'from anthropic import Anthropic\nclient.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=100)\n',
  "chat.ts": 'import OpenAI from "openai";\nawait client.chat.completions.create({ model: "gpt-4o-2024-08-06", messages });\n',
};

describe("MCP server", () => {
  it("lists its five tools with read-only hints where nothing is changed", async () => {
    const client = await connect(await project({}));
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual(["list_retirements", "model_info", "plan_model_switch", "scan_ai_models", "test_model_switch"]);
    const hint = (name: string) => tools.find((t) => t.name === name)?.annotations?.readOnlyHint;
    expect(hint("scan_ai_models")).toBe(true);
    expect(hint("plan_model_switch")).toBe(true);
    expect(hint("test_model_switch")).toBe(false);
  });

  it("scans the project and reports retired and retiring models with file and line", async () => {
    const client = await connect(await project(APP));
    const r = await call(client, "scan_ai_models");
    expect(r.isError).toBe(false);
    expect(r.text).toContain("claude-3-5-sonnet-20241022");
    const json = JSON.parse(r.text.slice(r.text.indexOf('{\n  "models"')));
    expect(json.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ model: "claude-3-5-sonnet-20241022", status: "retired", replacement: "claude-sonnet-5-5", locations: ["bot.py:2"] }),
        expect.objectContaining({ model: "gpt-4o-2024-08-06", status: "retiring", date: "2027-01-14", daysLeft: 52, locations: ["chat.ts:2"] }),
      ]),
    );
  });

  it("looks up a model, including a gateway-prefixed id, and says when it knows nothing", async () => {
    const client = await connect(await project({}));
    const known = await call(client, "model_info", { model_id: "openai/gpt-4o-2024-08-06" });
    expect(known.text).toMatch(/^gpt-4o-2024-08-06 on openai-api: retires on 2027-01-14 \(in 52 days\)\. Suggested replacement: gpt-6-luna\./);
    expect(known.text).toContain("Sources: https://");
    const unknown = await call(client, "model_info", { model_id: "my-own-model-v1" });
    expect(unknown.text).toMatch(/no announced retirement/);
  });

  it("lists upcoming retirements soonest first, by provider", async () => {
    const client = await connect(await project({}));
    const r = await call(client, "list_retirements", { within_days: 60, provider: "openai" });
    const lines = r.text.split("\n");
    expect(lines.length).toBeGreaterThan(0);
    const days = lines.map((l) => Number(/\((\d+) days\)/.exec(l)?.[1]));
    expect(days).toEqual([...days].sort((a, b) => a - b));
    expect(days.every((d) => d > 0 && d <= 60)).toBe(true);
  });

  it("plans a switch without changing any file", async () => {
    const root = await project(APP);
    const client = await connect(root);
    const r = await call(client, "plan_model_switch", { from: "gpt-4o-2024-08-06", to: "gpt-6-luna" });
    expect(r.text).toContain("1 line(s) in 1 file(s) would change from gpt-4o-2024-08-06 to gpt-6-luna:");
    expect(r.text).toContain('+ await client.chat.completions.create({ model: "gpt-6-luna", messages });');
    expect(await readFile(join(root, "chat.ts"), "utf8")).toBe(APP["chat.ts"]);
  });

  it("refuses paths outside the project", async () => {
    const client = await connect(await project(APP));
    const cases: [string, Record<string, unknown>][] = [
      ["scan_ai_models", { path: "../../" }],
      ["plan_model_switch", { from: "a1", to: "b1", path: "/etc" }],
      ["test_model_switch", { prompts: "../secrets.jsonl", old_model: "openai:a", new_model: "openai:b" }],
    ];
    for (const [name, args] of cases) {
      const r = await call(client, name, args);
      expect(r.isError, name).toBe(true);
      expect(r.text).toMatch(/outside the project folder/);
    }
  });

  it("replays a golden set on both models, and refuses to start above max_calls", async () => {
    const root = await project({ "golden.jsonl": '{"id":"a","input":"Reply in JSON","expect":{"json":true}}\n' });
    const llm: LlmClient = async (req) => ({
      text: req.model === "openai:old" ? '{"ok": true}' : "Sure! ok",
      toolCalls: [],
      inputTokens: 1,
      outputTokens: 1,
      latencyMs: 1,
      refused: false,
    });
    const client = await connect(root, llm);
    const r = await call(client, "test_model_switch", { prompts: "golden.jsonl", old_model: "openai:old", new_model: "openai:new", runs: 2 });
    expect(r.isError).toBe(false);
    expect(r.text).toContain("| 0 | 0 | 1 | 0 |");
    const capped = await call(client, "test_model_switch", { prompts: "golden.jsonl", old_model: "openai:old", new_model: "openai:new", runs: 2, max_calls: 3 });
    expect(capped).toEqual({ isError: true, text: expect.stringMatching(/needs 4 calls .* above max_calls 3/) });
  });

  it("reports scan locations from the project root when a subfolder is scanned", async () => {
    const client = await connect(await project({ "src/app/chat.ts": APP["chat.ts"] }));
    const r = await call(client, "scan_ai_models", { path: "src" });
    expect(r.text).toContain('"src/app/chat.ts:2"');
  });

  it("accepts gateway-prefixed ids when planning a switch", async () => {
    const client = await connect(await project(APP));
    const r = await call(client, "plan_model_switch", { from: "openai/gpt-4o-2024-08-06", to: "openai/gpt-6-luna" });
    expect(r.isError).toBe(false);
    expect(r.text).toContain("would change from gpt-4o-2024-08-06 to gpt-6-luna");
  });

  it("refuses a symlink that leads outside the project, but allows names that start with two dots", async () => {
    const outsideDir = await project({ "secret.jsonl": '{"id":"s","input":"private"}\n' });
    const root = await project({ "..fixtures/golden.jsonl": '{"id":"a","input":"hi"}\n' });
    const { symlink } = await import("node:fs/promises");
    await symlink(join(outsideDir, "secret.jsonl"), join(root, "linked.jsonl"));
    const client = await connect(root, async () => ({ text: "ok", toolCalls: [], inputTokens: 1, outputTokens: 1, latencyMs: 1, refused: false }));
    const linked = await call(client, "test_model_switch", { prompts: "linked.jsonl", old_model: "openai:a", new_model: "openai:b", runs: 1 });
    expect(linked).toEqual({ isError: true, text: expect.stringMatching(/outside the project folder/) });
    const dotted = await call(client, "test_model_switch", { prompts: "..fixtures/golden.jsonl", old_model: "openai:a", new_model: "openai:b", runs: 1 });
    expect(dotted.isError).toBe(false);
  });

  it("will not treat the home folder as a project unless ACG_ROOT says so", async () => {
    const { homedir } = await import("node:os");
    const before = process.env.ACG_ROOT;
    const cwd = process.cwd();
    delete process.env.ACG_ROOT;
    process.chdir(homedir());
    try {
      const [clientSide, serverSide] = InMemoryTransport.createLinkedPair();
      await createServer({ today: "2026-11-23" }).connect(serverSide);
      const client = new Client({ name: "t", version: "1" });
      await client.connect(clientSide);
      clients.push(client);
      const r = await call(client, "scan_ai_models");
      expect(r).toEqual({ isError: true, text: expect.stringMatching(/Set ACG_ROOT to your project folder/) });
      // Catalog lookups need no project, so they still work.
      expect((await call(client, "model_info", { model_id: "gpt-4o-2024-08-06" })).isError).toBe(false);
    } finally {
      process.chdir(cwd);
      if (before !== undefined) process.env.ACG_ROOT = before;
    }
  });

  it("returns security findings in the JSON, with paths from the project root", async () => {
    const mcp = JSON.stringify({ mcpServers: { fs: { command: "npx", args: ["-y", "some-mcp"] } } });
    const client = await connect(await project({ "apps/web/.cursor/mcp.json": mcp }));
    const r = await call(client, "scan_ai_models", { path: "apps/web" });
    const json = JSON.parse(r.text.slice(r.text.indexOf('{\n  "models"')));
    expect(json.security).toEqual([
      expect.objectContaining({ rule: "unpinned-mcp-server", findings: [{ location: "apps/web/.cursor/mcp.json:1", detail: "fs: some-mcp" }] }),
    ]);
  });
});

