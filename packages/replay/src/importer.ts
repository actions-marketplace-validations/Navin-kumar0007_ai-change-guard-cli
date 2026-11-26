import type { ChatMessage, ToolDef } from "@ai-changeguard/llm";
import type { GoldenExpect, GoldenItem } from "./golden.js";
import type { JsonSchema } from "./schema.js";

/**
 * Builds a golden set from logs people already have, so nobody has to write test prompts.
 *
 * Tools export logs in different wrappers (Langfuse traces and observations, Helicone request
 * records, OpenTelemetry spans, plain JSONL of API requests), but the prompt inside is almost
 * always one of a few request shapes. So the importer walks every record, finds a request body
 * wherever it sits, and reads it by shape: OpenAI Chat Completions, OpenAI Responses,
 * Anthropic Messages, Gemini generateContent, or OpenTelemetry GenAI attributes.
 *
 * When the recorded answer is present it also derives checks: an answer that was JSON becomes
 * expect.json, and an answer that called a tool becomes expect.tool.
 */

export interface ImportOptions {
  /** Keep at most this many prompts, spread evenly over the log (default 50). */
  limit?: number;
  /** Keep only calls to this model (matched on the last path segment, case-insensitive). */
  model?: string;
}

export interface ImportResult {
  items: GoldenItem[];
  /** Records looked at (objects that could hold a call). */
  records: number;
  /** Distinct prompts found before the limit was applied. */
  found: number;
  /** Why records were left out, with counts. */
  skipped: Record<string, number>;
  /** Calls per model among the prompts found. */
  models: Record<string, number>;
  /** Prompts kept by a --model filter although the log did not name their model (raw Gemini bodies, say). */
  unknownModel: number;
}

type Json = unknown;
type Obj = Record<string, Json>;

interface Extracted {
  system?: string;
  messages: ChatMessage[];
  model?: string;
  tools?: ToolDef[];
  maxTokens?: number;
  temperature?: number;
  jsonMode?: boolean;
  schema?: JsonSchema;
  /** The call forced a tool (tool_choice), which replay does not reproduce. */
  forcedTool?: boolean;
}

class Skip extends Error {}

const isObj = (v: Json): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: Json): string | undefined => (typeof v === "string" ? v : undefined);
const num = (v: Json): number | undefined =>
  typeof v === "number" && Number.isFinite(v) ? v : typeof v === "string" && v.trim() && Number.isFinite(Number(v)) ? Number(v) : undefined;

/** Values that are JSON written as a string (Helicone bodies, Langfuse I/O, span attributes). */
function parsed(v: Json): Json {
  if (typeof v !== "string") return v;
  const t = v.trim();
  if (!(t.startsWith("{") || t.startsWith("["))) return v;
  try {
    return JSON.parse(t);
  } catch {
    return v;
  }
}

/** Gemini writes schema types in capitals ("OBJECT"); JSON Schema and the replay checks use lower case. */
function normalizeSchema(schema: Json): Json {
  if (Array.isArray(schema)) return schema.map(normalizeSchema);
  if (!isObj(schema)) return schema;
  const out: Obj = {};
  for (const [k, v] of Object.entries(schema)) {
    if (k === "type") out[k] = typeof v === "string" ? v.toLowerCase() : Array.isArray(v) ? v.map((t) => (typeof t === "string" ? t.toLowerCase() : t)) : v;
    else if (k === "properties" && isObj(v)) out[k] = Object.fromEntries(Object.entries(v).map(([pk, pv]) => [pk, normalizeSchema(pv)]));
    else if (k === "items" || k === "anyOf" || k === "oneOf" || k === "allOf") out[k] = normalizeSchema(v);
    else out[k] = v;
  }
  return out;
}

// ---------- reading the file ----------

function parseFile(text: string): { values: Json[]; badLines: number } {
  const trimmed = text.trim();
  if (!trimmed) return { values: [], badLines: 0 };
  try {
    return { values: [JSON.parse(trimmed)], badLines: 0 };
  } catch {
    // JSONL
  }
  const values: Json[] = [];
  let badLines = 0;
  for (const line of trimmed.split(/\r?\n/)) {
    const l = line.trim();
    if (!l) continue;
    try {
      values.push(JSON.parse(l));
    } catch {
      badLines++;
    }
  }
  return { values, badLines };
}

/** An OpenAI Evals output item: one recorded call, whose "results" are grader scores, not more records. */
function isEvalItem(value: Obj): value is Obj & { sample: Obj & { input: Json[] } } {
  return isObj(value.sample) && Array.isArray(value.sample.input);
}

const CONTAINERS = ["data", "observations", "spans", "requests", "items", "records", "results", "rows", "generations"];

/** Flattens wrappers (arrays, OTLP resourceSpans, {data:[...]}, a trace with observations) into records. */
function records(value: Json, out: Obj[] = []): Obj[] {
  if (Array.isArray(value)) {
    for (const v of value) records(v, out);
    return out;
  }
  if (!isObj(value)) return out;
  if (Array.isArray(value.resourceSpans)) {
    for (const rs of value.resourceSpans) {
      for (const ss of isObj(rs) && Array.isArray(rs.scopeSpans) ? rs.scopeSpans : []) records(isObj(ss) ? ss.spans : [], out);
    }
    return out;
  }
  const container = CONTAINERS.find((k) => Array.isArray(value[k]) && (value[k] as Json[]).some(isObj));
  if (container && !isEvalItem(value) && !Array.isArray(value.messages) && !Array.isArray(value.contents)) {
    records(value[container], out);
    return out;
  }
  out.push(value);
  return out;
}

// ---------- OpenTelemetry GenAI ----------

function otelValue(v: Json): Json {
  if (!isObj(v)) return v;
  if ("stringValue" in v) return v.stringValue;
  if ("intValue" in v) return num(v.intValue);
  if ("doubleValue" in v) return v.doubleValue;
  if ("boolValue" in v) return v.boolValue;
  if (isObj(v.arrayValue) && Array.isArray(v.arrayValue.values)) return v.arrayValue.values.map(otelValue);
  return v;
}

/** Span attributes as a flat map, from OTLP JSON ([{key, value}]) or an already flat object. */
function spanAttributes(record: Obj): Obj | null {
  const raw = record.attributes ?? record.span_attributes ?? record.spanAttributes;
  if (Array.isArray(raw)) {
    const map: Obj = {};
    for (const a of raw) if (isObj(a) && typeof a.key === "string") map[a.key] = otelValue(a.value);
    return map;
  }
  return isObj(raw) ? raw : null;
}

function partsText(parts: Json, onOther: (type: string) => void): string {
  if (!Array.isArray(parts)) return "";
  const texts: string[] = [];
  for (const p of parts) {
    if (!isObj(p)) continue;
    if (p.type === "text" && typeof p.content === "string") texts.push(p.content);
    else if (typeof p.type === "string") onOther(p.type);
  }
  return texts.join("\n");
}

function fromOtel(attrs: Obj): Extracted | null {
  const input = parsed(attrs["gen_ai.input.messages"]);
  const messages: ChatMessage[] = [];
  let system = "";
  const refuse = (type: string) => {
    if (type === "tool_call" || type === "tool_call_response") throw new Skip("tool-call conversation");
    if (type !== "reasoning") throw new Skip("non-text content (images, files or audio)");
  };
  if (Array.isArray(input)) {
    for (const m of input) {
      if (!isObj(m)) continue;
      const text = partsText(m.parts, refuse);
      if (m.role === "system") system = [system, text].filter(Boolean).join("\n\n");
      else if (m.role === "user" || m.role === "assistant") messages.push({ role: m.role, content: text });
      else if (m.role === "tool") throw new Skip("tool-call conversation");
    }
    const extra = partsText(parsed(attrs["gen_ai.system_instructions"]), () => {});
    if (extra) system = [extra, system].filter(Boolean).join("\n\n");
  } else {
    // Older instrumentations: gen_ai.prompt.0.role / gen_ai.prompt.0.content ...
    for (let i = 0; attrs[`gen_ai.prompt.${i}.role`] !== undefined || attrs[`gen_ai.prompt.${i}.content`] !== undefined; i++) {
      const role = str(attrs[`gen_ai.prompt.${i}.role`]);
      const content = str(attrs[`gen_ai.prompt.${i}.content`]) ?? "";
      if (role === "system") system = [system, content].filter(Boolean).join("\n\n");
      else if (role === "user" || role === "assistant") messages.push({ role, content });
      else if (role === "tool") throw new Skip("tool-call conversation");
    }
  }
  if (messages.length === 0) return null;
  const out: Extracted = { messages };
  if (system) out.system = system;
  const model = str(attrs["gen_ai.request.model"]) ?? str(attrs["gen_ai.response.model"]);
  if (model) out.model = model;
  const maxTokens = num(attrs["gen_ai.request.max_tokens"]);
  if (maxTokens !== undefined) out.maxTokens = maxTokens;
  const temperature = num(attrs["gen_ai.request.temperature"]);
  if (temperature !== undefined) out.temperature = temperature;
  if (attrs["gen_ai.output.type"] === "json") out.jsonMode = true;
  return out;
}

interface Answer {
  text?: string;
  tool?: string;
}

function otelAnswer(attrs: Obj): Answer | null {
  const output = parsed(attrs["gen_ai.output.messages"]);
  if (Array.isArray(output)) {
    for (const m of output) {
      if (!isObj(m) || !Array.isArray(m.parts)) continue;
      const call = m.parts.find((p) => isObj(p) && p.type === "tool_call") as Obj | undefined;
      if (call && typeof call.name === "string") return { tool: call.name };
      const text = partsText(m.parts, () => {});
      if (text) return { text };
    }
  }
  const legacy = str(attrs["gen_ai.completion.0.content"]);
  return legacy ? { text: legacy } : null;
}

// ---------- request bodies ----------

/** Text of an OpenAI/Anthropic content value: a string or a list of text blocks. */
function contentText(content: Json): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw new Skip("unreadable message content");
  const texts: string[] = [];
  for (const block of content) {
    if (typeof block === "string") texts.push(block);
    else if (isObj(block)) {
      const type = block.type;
      if ((type === "text" || type === "input_text" || type === "output_text") && typeof block.text === "string") texts.push(block.text);
      else if (type === "tool_use" || type === "tool_result" || type === "function_call" || type === "function_call_output") throw new Skip("tool-call conversation");
      else if (type !== "thinking" && type !== "redacted_thinking") throw new Skip("non-text content (images, files or audio)");
    }
  }
  return texts.join("\n");
}

function chatMessages(list: Json[]): { system: string; messages: ChatMessage[] } {
  let system = "";
  const messages: ChatMessage[] = [];
  for (const m of list) {
    if (!isObj(m)) continue;
    const role = m.role ?? (m.type === "message" ? "user" : undefined);
    if (role === "system" || role === "developer") system = [system, contentText(m.content)].filter(Boolean).join("\n\n");
    else if (role === "user" || role === "assistant") {
      if (role === "assistant" && ((Array.isArray(m.tool_calls) && m.tool_calls.length > 0) || isObj(m.function_call))) throw new Skip("tool-call conversation");
      messages.push({ role, content: contentText(m.content) });
    } else if (role === "tool" || role === "function" || m.type === "function_call" || m.type === "function_call_output") {
      throw new Skip("tool-call conversation");
    }
  }
  return { system, messages };
}

function geminiContents(contents: Json[]): ChatMessage[] {
  return contents.flatMap((c): ChatMessage[] => {
    if (!isObj(c) || !Array.isArray(c.parts)) return [];
    const texts: string[] = [];
    for (const p of c.parts) {
      if (!isObj(p)) continue;
      if (typeof p.text === "string") texts.push(p.text);
      else if ("functionCall" in p || "functionResponse" in p) throw new Skip("tool-call conversation");
      else throw new Skip("non-text content (images, files or audio)");
    }
    return [{ role: c.role === "model" ? "assistant" : "user", content: texts.join("\n") }];
  });
}

function toolDefs(body: Obj): ToolDef[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const defs: ToolDef[] = [];
  for (const t of body.tools) {
    if (!isObj(t)) continue;
    // Gemini: tools: [{functionDeclarations: [...]}]
    if (Array.isArray(t.functionDeclarations)) {
      for (const f of t.functionDeclarations) {
        if (isObj(f) && typeof f.name === "string" && isObj(f.parameters)) {
          defs.push({ name: f.name, parameters: f.parameters, ...(typeof f.description === "string" ? { description: f.description } : {}) });
        }
      }
      continue;
    }
    const fn = isObj(t.function) ? t.function : t; // OpenAI chat nests under function; Responses and Anthropic do not
    const name = str(fn.name);
    const parameters = isObj(fn.parameters) ? fn.parameters : isObj(fn.input_schema) ? fn.input_schema : undefined;
    if (!name || !parameters) continue;
    defs.push({ name, parameters, ...(typeof fn.description === "string" ? { description: fn.description } : {}) });
  }
  return defs.length ? defs : undefined;
}

function fromBody(body: Obj): Extracted | null {
  let system = "";
  let messages: ChatMessage[];
  if (Array.isArray(body.messages)) {
    const chat = chatMessages(body.messages);
    messages = chat.messages;
    system = chat.system;
    // Anthropic: top-level system as a string or text blocks.
    if (body.system !== undefined && body.system !== null) system = [contentText(body.system), system].filter(Boolean).join("\n\n");
  } else if (Array.isArray(body.contents)) {
    messages = geminiContents(body.contents);
    const si = body.systemInstruction ?? body.system_instruction;
    if (isObj(si) && Array.isArray(si.parts)) system = si.parts.flatMap((p) => (isObj(p) && typeof p.text === "string" ? [p.text] : [])).join("\n");
  } else if (typeof body.input === "string" && (body.model !== undefined || body.instructions !== undefined)) {
    messages = [{ role: "user", content: body.input }]; // OpenAI Responses
  } else if (Array.isArray(body.input) && (body.model !== undefined || body.instructions !== undefined)) {
    const chat = chatMessages(body.input);
    messages = chat.messages;
    system = chat.system;
  } else {
    return null;
  }
  if (typeof body.instructions === "string") system = [body.instructions, system].filter(Boolean).join("\n\n");
  if (messages.length === 0) return null;

  const out: Extracted = { messages };
  if (system) out.system = system;
  if (typeof body.model === "string") out.model = body.model;
  const gen = isObj(body.generationConfig) ? body.generationConfig : isObj(body.generation_config) ? body.generation_config : {};
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens) ?? num(body.max_output_tokens) ?? num(gen.maxOutputTokens);
  if (maxTokens !== undefined) out.maxTokens = maxTokens;
  const temperature = num(body.temperature) ?? num(gen.temperature);
  if (temperature !== undefined) out.temperature = temperature;
  const defs = toolDefs(body);
  if (defs) out.tools = defs;
  const choice = body.tool_choice;
  if (choice === "required" || choice === "any" || (isObj(choice) && choice.type !== "auto" && choice.type !== "none")) out.forcedTool = true;
  if (isObj(body.toolConfig) && isObj(body.toolConfig.functionCallingConfig) && body.toolConfig.functionCallingConfig.mode === "ANY") out.forcedTool = true;

  const rf = isObj(body.response_format) ? body.response_format : isObj(body.text) && isObj(body.text.format) ? body.text.format : null;
  if (rf?.type === "json_object") out.jsonMode = true;
  if (rf?.type === "json_schema") {
    out.jsonMode = true;
    const schema = isObj(rf.json_schema) ? rf.json_schema.schema : rf.schema;
    if (isObj(schema)) out.schema = normalizeSchema(schema) as JsonSchema;
  }
  if (gen.responseMimeType === "application/json") {
    out.jsonMode = true;
    const schema = gen.responseSchema ?? gen.responseJsonSchema;
    if (isObj(schema)) out.schema = normalizeSchema(schema) as JsonSchema;
  }
  return out;
}

// ---------- recorded answers ----------

function answerFrom(value: Json): Answer | null {
  const v = parsed(value);
  if (typeof v === "string") return v.trim() ? { text: v } : null;
  if (Array.isArray(v)) {
    for (const item of v) {
      const a = answerFrom(item);
      if (a) return a;
    }
    return null;
  }
  if (!isObj(v)) return null;
  // OpenAI chat
  if (Array.isArray(v.choices) && isObj(v.choices[0])) return answerFrom(v.choices[0].message ?? v.choices[0].text);
  if (Array.isArray(v.tool_calls) && isObj(v.tool_calls[0])) {
    const fn = isObj(v.tool_calls[0].function) ? v.tool_calls[0].function : v.tool_calls[0];
    if (typeof fn.name === "string") return { tool: fn.name };
  }
  // Gemini
  if (Array.isArray(v.candidates) && isObj(v.candidates[0]) && isObj(v.candidates[0].content)) {
    const parts = v.candidates[0].content.parts;
    if (Array.isArray(parts)) {
      const call = parts.find((p) => isObj(p) && isObj(p.functionCall)) as Obj | undefined;
      if (call && isObj(call.functionCall) && typeof call.functionCall.name === "string") return { tool: call.functionCall.name };
      const text = parts.flatMap((p) => (isObj(p) && typeof p.text === "string" ? [p.text] : [])).join("");
      return text ? { text } : null;
    }
  }
  // Anthropic content blocks, Responses output items
  const blocks = Array.isArray(v.content) ? v.content : Array.isArray(v.output) ? v.output : null;
  if (blocks) {
    for (const b of blocks) {
      if (isObj(b) && (b.type === "tool_use" || b.type === "function_call") && typeof b.name === "string") return { tool: b.name };
    }
    const text = blocks
      .flatMap((b) =>
        isObj(b) && typeof b.text === "string"
          ? [b.text]
          : isObj(b) && Array.isArray(b.content)
            ? b.content.flatMap((c) => (isObj(c) && typeof c.text === "string" ? [c.text] : []))
            : [],
      )
      .join("");
    if (text) return { text };
  }
  if (typeof v.content === "string") return { text: v.content };
  if (typeof v.output_text === "string") return { text: v.output_text };
  if (typeof v.text === "string") return { text: v.text };
  return null;
}

function looksLikeJson(text: string): boolean {
  let t = text.trim();
  const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(t);
  if (fenced?.[1] !== undefined) t = fenced[1];
  if (!t.startsWith("{") && !t.startsWith("[")) return false;
  try {
    const v: unknown = JSON.parse(t);
    return typeof v === "object" && v !== null;
  } catch {
    return false;
  }
}

// ---------- per record ----------

const BODY_KEYS = ["request_body", "requestBody", "request", "body", "input", "prompt"];
const ANSWER_KEYS = ["response_body", "responseBody", "response", "output", "completion", "result"];
const ID_KEYS = ["id", "request_id", "requestId", "spanId", "span_id"];

function extractRecord(record: Obj): { call: Extracted; answer: Answer | null; id?: string } | null {
  const attrs = spanAttributes(record);
  if (attrs && (attrs["gen_ai.input.messages"] !== undefined || attrs["gen_ai.prompt.0.content"] !== undefined || attrs["gen_ai.prompt.0.role"] !== undefined)) {
    const call = fromOtel(attrs);
    const id = str(record.spanId) ?? str(record.span_id);
    return call ? { call, answer: otelAnswer(attrs), ...(id ? { id } : {}) } : null;
  }

  // OpenAI Evals output items (saved with evals-export): the exact messages sent, the model and its answer. The
  // run's sampling_params (response format, tools), which the items don't repeat, come in as `sampling_params`.
  if (isEvalItem(record)) {
    const s = record.sample;
    if (isObj(s.error) && (s.error.code || s.error.message)) throw new Skip("the eval's call failed");
    const sp = isObj(record.sampling_params) ? record.sampling_params : {};
    const max = num(s.max_completion_tokens) ?? num(sp.max_completion_tokens) ?? num(sp.max_output_tokens);
    const evalCall = fromBody({
      messages: s.input,
      model: s.model,
      ...(max && max > 0 ? { max_completion_tokens: max } : {}),
      temperature: s.temperature ?? sp.temperature,
      ...(sp.response_format !== undefined ? { response_format: sp.response_format } : {}),
      ...(sp.text !== undefined ? { text: sp.text } : {}),
      ...(sp.tools !== undefined ? { tools: sp.tools } : {}),
    });
    if (!evalCall) return null;
    const itemId = num(record.datasource_item_id);
    const id = str(record.id) ?? (itemId !== undefined ? `item-${itemId}` : undefined);
    return { call: evalCall, answer: answerFrom(s.output), ...(id ? { id } : {}) };
  }

  let call: Extracted | null = null;
  for (const key of BODY_KEYS) {
    const v = parsed(record[key]);
    if (isObj(v)) call = fromBody(v);
    // Langfuse often stores a generation's input as the message list itself.
    else if (Array.isArray(v) && v.some((m) => isObj(m) && typeof m.role === "string")) call = fromBody({ messages: v });
    if (call) break;
  }
  call ??= fromBody(record);
  if (!call) return null;
  if (!call.model && typeof record.model === "string") call.model = record.model;

  let answer: Answer | null = null;
  for (const key of ANSWER_KEYS) {
    if (record[key] === undefined) continue;
    answer = answerFrom(record[key]);
    if (answer) break;
  }
  const id = ID_KEYS.map((k) => record[k]).find((v) => typeof v === "string" || typeof v === "number");
  return { call, answer, ...(id !== undefined ? { id: String(id) } : {}) };
}

function lastSegment(model: string): string {
  return (model.split("/").at(-1) ?? model).toLowerCase();
}

/** gpt-4o matches gpt-4o and its dated snapshots (gpt-4o-2024-08-06), never gpt-4o-mini. */
export function sameModel(logged: string, wanted: string): boolean {
  const a = lastSegment(logged);
  const b = lastSegment(wanted);
  if (a === b) return true;
  if (!a.startsWith(b)) return false;
  return /^(?:-\d{4}-?\d{2}-?\d{2}|-\d{3,4}|@\d{8}|-latest|-v\d+(?::\d+)?)$/.test(a.slice(b.length));
}

/** Up to `limit` items spread evenly over the list, so one busy hour does not fill the set. */
function spread<T>(list: T[], limit: number): T[] {
  if (list.length <= limit) return list;
  const step = list.length / limit;
  return Array.from({ length: limit }, (_, i) => list[Math.floor(i * step)]!);
}

const UNSAFE_ID = /[^A-Za-z0-9._-]+/g;

export function importPrompts(text: string, options: ImportOptions = {}): ImportResult {
  const { values, badLines } = parseFile(text);
  return importValues(values, options, badLines);
}

/** Like importPrompts, for records already parsed (several files at once, or records given extra fields). */
export function importValues(values: Json[], options: ImportOptions = {}, badLines = 0): ImportResult {
  const limit = options.limit ?? 50;
  const wantModel = options.model;
  const all = values.flatMap((v) => records(v));
  const skipped: Record<string, number> = {};
  const skip = (reason: string) => (skipped[reason] = (skipped[reason] ?? 0) + 1);
  if (badLines > 0) skipped["lines that are not JSON"] = badLines;

  const seen = new Set<string>();
  const ids = new Set<string>();
  const models: Record<string, number> = {};
  const found: GoldenItem[] = [];
  let unknownModel = 0;

  for (const record of all) {
    let got: ReturnType<typeof extractRecord>;
    try {
      got = extractRecord(record);
    } catch (error) {
      if (!(error instanceof Skip)) throw error;
      skip(error.message);
      continue;
    }
    if (!got) {
      skip("no prompt found in the record");
      continue;
    }
    const { call, answer } = got;
    // A logged conversation can end with the model's own reply; the prompt is everything before it.
    while (call.messages.at(-1)?.role === "assistant") call.messages.pop();
    if (call.messages.length === 0 || !call.messages.at(-1)!.content.trim()) {
      skip("empty prompt");
      continue;
    }
    if (wantModel && call.model && !sameModel(call.model, wantModel)) {
      skip("a different model");
      continue;
    }
    if (wantModel && !call.model) unknownModel++;
    const key = JSON.stringify([call.system ?? "", call.messages, call.tools?.map((t) => t.name) ?? [], call.jsonMode ?? false, call.schema ?? null]);
    if (seen.has(key)) {
      skip("duplicate of an earlier prompt");
      continue;
    }
    seen.add(key);
    if (call.model) models[call.model] = (models[call.model] ?? 0) + 1;

    const fallback = `log-${found.length + 1}`;
    const base = (got.id ?? fallback).replace(UNSAFE_ID, "-").slice(0, 60) || fallback;
    let id = base;
    for (let n = 2; ids.has(id); n++) id = `${base}-${n}`;
    ids.add(id);

    const item: GoldenItem = { id, messages: call.messages };
    if (call.system) item.system = call.system;
    if (call.tools) item.tools = call.tools;
    if (call.maxTokens !== undefined) item.maxTokens = call.maxTokens;
    if (call.temperature !== undefined) item.temperature = call.temperature;
    if (call.jsonMode) item.jsonMode = true;

    const expect: GoldenExpect = {};
    if (call.schema) expect.schema = call.schema;
    else if (call.jsonMode || (answer?.text !== undefined && looksLikeJson(answer.text))) expect.json = true;
    // A forced tool_choice is not replayed, so the model may answer in text; no tool check then.
    if (answer?.tool && !call.forcedTool && call.tools?.some((t) => t.name === answer.tool)) expect.tool = answer.tool;
    if (Object.keys(expect).length > 0) item.expect = expect;
    found.push(item);
  }

  return { items: spread(found, limit), records: all.length, found: found.length, skipped, models, unknownModel };
}
