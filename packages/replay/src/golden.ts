import { readFile } from "node:fs/promises";
import type { ChatMessage, ToolDef } from "@ai-changeguard/llm";
import type { JsonSchema } from "./schema.js";

export interface GoldenExpect {
  json?: boolean;
  schema?: JsonSchema;
  tool?: string;
  contains?: string[];
}

export interface GoldenItem {
  id: string;
  system?: string;
  messages: ChatMessage[];
  tools?: ToolDef[];
  expect?: GoldenExpect;
  maxTokens?: number;
  temperature?: number;
  jsonMode?: boolean;
}

export class GoldenSetError extends Error {}

type Raw = Record<string, unknown>;

function isObject(value: unknown): value is Raw {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseMessages(raw: Raw, where: string): ChatMessage[] {
  if (typeof raw.input === "string" && raw.input.trim()) return [{ role: "user", content: raw.input }];
  const list = raw.messages;
  if (Array.isArray(list) && list.length > 0) {
    const messages = list.map((m, i) => {
      if (!isObject(m) || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string") {
        throw new GoldenSetError(`${where}: messages[${i}] needs a role of "user" or "assistant" and text content`);
      }
      return { role: m.role, content: m.content } as ChatMessage;
    });
    if (messages.at(-1)?.role !== "user") throw new GoldenSetError(`${where}: the last message must be from the user`);
    return messages;
  }
  throw new GoldenSetError(`${where}: needs "input" (text) or "messages" (a list of {role, content})`);
}

function parseTools(raw: unknown, where: string): ToolDef[] | undefined {
  if (raw === undefined) return undefined;
  if (!Array.isArray(raw)) throw new GoldenSetError(`${where}: "tools" must be a list`);
  return raw.map((t, i) => {
    if (!isObject(t) || typeof t.name !== "string" || !isObject(t.parameters)) {
      throw new GoldenSetError(`${where}: tools[${i}] needs a "name" and a "parameters" JSON schema`);
    }
    const tool: ToolDef = { name: t.name, parameters: t.parameters };
    if (typeof t.description === "string") tool.description = t.description;
    return tool;
  });
}

function parseExpect(raw: unknown, where: string): GoldenExpect | undefined {
  if (raw === undefined) return undefined;
  if (!isObject(raw)) throw new GoldenSetError(`${where}: "expect" must be an object`);
  const expect: GoldenExpect = {};
  if (raw.json !== undefined) expect.json = Boolean(raw.json);
  if (raw.schema !== undefined) {
    if (!isObject(raw.schema)) throw new GoldenSetError(`${where}: "expect.schema" must be a JSON schema object`);
    expect.schema = raw.schema as JsonSchema;
  }
  if (raw.tool !== undefined) {
    if (typeof raw.tool !== "string") throw new GoldenSetError(`${where}: "expect.tool" must be a tool name`);
    expect.tool = raw.tool;
  }
  if (raw.contains !== undefined) {
    if (!Array.isArray(raw.contains) || !raw.contains.every((s) => typeof s === "string")) {
      throw new GoldenSetError(`${where}: "expect.contains" must be a list of text`);
    }
    expect.contains = raw.contains as string[];
  }
  return expect;
}

export function parseGoldenSet(text: string): GoldenItem[] {
  const items: GoldenItem[] = [];
  const ids = new Set<string>();

  text.split(/\r?\n/).forEach((rawLine, index) => {
    const line = rawLine.trim();
    if (!line || line.startsWith("//")) return;
    const where = `Line ${index + 1}`;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch {
      throw new GoldenSetError(`${where}: not valid JSON`);
    }
    if (!isObject(raw)) throw new GoldenSetError(`${where}: each line must be a JSON object`);

    const id = raw.id === undefined ? `line-${index + 1}` : String(raw.id);
    if (ids.has(id)) throw new GoldenSetError(`${where}: duplicate id "${id}"`);
    ids.add(id);

    if (raw.system !== undefined && typeof raw.system !== "string") throw new GoldenSetError(`${where}: "system" must be text`);
    const item: GoldenItem = { id, messages: parseMessages(raw, where) };
    if (typeof raw.system === "string") item.system = raw.system;
    const tools = parseTools(raw.tools, where);
    if (tools) item.tools = tools;
    const expect = parseExpect(raw.expect, where);
    if (expect) item.expect = expect;
    if (typeof raw.max_tokens === "number") item.maxTokens = raw.max_tokens;
    if (typeof raw.temperature === "number") item.temperature = raw.temperature;
    if (raw.json_mode === true) item.jsonMode = true;
    items.push(item);
  });

  if (items.length === 0) throw new GoldenSetError("The golden set is empty.");
  return items;
}

export async function loadGoldenSet(path: string): Promise<GoldenItem[]> {
  return parseGoldenSet(await readFile(path, "utf8"));
}
