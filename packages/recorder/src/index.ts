import { mkdirSync, statSync } from "node:fs";
import { appendFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

/**
 * Records your app's real AI calls to a JSONL file, so `ai-change-guard import` can turn them
 * into a golden set. Three lines:
 *
 *   import { record } from "@ai-changeguard/recorder";
 *   record();                       // writes ./ai-calls.jsonl
 *
 * It wraps globalThis.fetch, which the OpenAI, Anthropic and Google Gen AI SDKs use, and saves
 * the JSON request and response bodies of calls to known AI APIs. It never saves headers (where
 * API keys live) or the URL query (Gemini can carry ?key=), it never changes what your app
 * sends or receives, and a recorder error never reaches your code.
 */

export interface RecordOptions {
  /** Where to write. Default: ai-calls.jsonl in the working directory. */
  file?: string;
  /** Share of calls to keep, 0 to 1. Default 1. */
  sample?: number;
  /** Stop writing once the file reaches this size. Default 20 MB. */
  maxBytes?: number;
  /** More API hosts to record, e.g. your own gateway. */
  hosts?: string[];
  /** Change or remove data before it is written (return null to drop the call). */
  redact?: (entry: RecordedCall) => RecordedCall | null;
}

export interface RecordedCall {
  recorded_at: string;
  /** Origin and path only; the query string is never kept. */
  url: string;
  model?: string;
  status?: number;
  request: unknown;
  response: unknown;
}

export interface Recorder {
  /** Puts the original fetch back. */
  stop(): void;
  /** Waits for answers still being read and written. */
  flush(): Promise<void>;
}

const AI_HOSTS = [
  "api.openai.com",
  "api.anthropic.com",
  "generativelanguage.googleapis.com",
  "openrouter.ai",
  "api.groq.com",
  "api.mistral.ai",
  "api.together.xyz",
  "api.deepseek.com",
];
const AI_HOST_SUFFIXES = [".openai.azure.com", "-aiplatform.googleapis.com"];
const MARK = Symbol.for("ai-change-guard.recorder");

const noop: Recorder = { stop() {}, flush: async () => {} };

function bodyText(body: unknown): string | null {
  if (typeof body === "string") return body;
  if (body instanceof Uint8Array) return new TextDecoder().decode(body);
  if (body instanceof ArrayBuffer) return new TextDecoder().decode(new Uint8Array(body));
  return null;
}

function modelFrom(url: URL, request: unknown): string | undefined {
  if (request && typeof request === "object" && typeof (request as { model?: unknown }).model === "string") return (request as { model: string }).model;
  // Gemini and Vertex put the model in the path: .../models/gemini-2.0-flash:generateContent
  return /\/models\/([^/:]+)/.exec(url.pathname)?.[1];
}

export function record(options: RecordOptions = {}): Recorder {
  if (typeof process === "undefined" || process.env?.ACG_RECORD === "0") return noop;
  const original = globalThis.fetch;
  if (typeof original !== "function" || (original as unknown as Record<symbol, unknown>)[MARK]) return noop;

  const file = resolve(options.file ?? "ai-calls.jsonl");
  const sample = Math.min(1, Math.max(0, options.sample ?? 1));
  const maxBytes = options.maxBytes ?? 20 * 1024 * 1024;
  const extraHosts = new Set((options.hosts ?? []).map((h) => h.toLowerCase()));
  let size = 0;
  try {
    size = statSync(file).size;
  } catch {
    // new file
  }
  let full = size >= maxBytes;
  let stopped = false;
  let writing: Promise<void> = Promise.resolve();
  const pending = new Set<Promise<void>>();
  try {
    mkdirSync(dirname(file), { recursive: true });
  } catch {
    return noop;
  }

  const isAiHost = (host: string) => AI_HOSTS.includes(host) || extraHosts.has(host) || AI_HOST_SUFFIXES.some((s) => host.endsWith(s));

  const write = (entry: RecordedCall) => {
    if (full) return;
    const kept = options.redact ? options.redact(entry) : entry;
    if (!kept) return;
    const line = `${JSON.stringify(kept)}\n`;
    const bytes = Buffer.byteLength(line);
    if (size + bytes > maxBytes) {
      full = true;
      console.warn(`[ai-change-guard] ${file} reached its size limit; recording stopped.`);
      return;
    }
    size += bytes;
    // Queued, so lines stay whole and in order without blocking the event loop.
    writing = writing.then(() => appendFile(file, line)).catch(() => {});
  };

  const wrapped = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    // Read what we need before the call: a Request's body can only be read once.
    let plan: { url: URL; text: string | null; fromRequest: Promise<string | null> | null } | null = null;
    try {
      const href = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      const url = new URL(href);
      const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
      if (!stopped && !full && method === "POST" && isAiHost(url.hostname.toLowerCase()) && Math.random() < sample) {
        const text = bodyText(init?.body);
        // .catch at once: an aborted request must not become an unhandled rejection in your app.
        const fromRequest = text === null && input instanceof Request && !init?.body ? input.clone().text().catch(() => null) : null;
        plan = { url, text, fromRequest };
      }
    } catch {
      plan = null;
    }

    const response = await original(input, init);

    // Calls the provider rejected (4xx/5xx) are not recorded: they would only become broken tests.
    if (plan && response.ok) {
      const { url, text, fromRequest } = plan;
      // Copy the answer now, before your code reads it. A streamed answer is never copied: a copy
      // nobody reads would hold the whole stream in memory. Only the prompt is recorded then.
      let copy: Response | null = null;
      try {
        const streamed =
          /event-stream|ndjson/i.test(response.headers.get("content-type") ?? "") ||
          /:stream|alt=sse/i.test(url.pathname + url.search) ||
          (text !== null && /"stream"\s*:\s*true/.test(text));
        if (!streamed) copy = response.clone();
      } catch {
        copy = null;
      }
      const job = (async () => {
        try {
          const body = fromRequest ? await fromRequest : text;
          if (body === null) return;
          const request: unknown = JSON.parse(body);
          let answer: unknown = null;
          if (copy) {
            try {
              answer = JSON.parse(await copy.text());
            } catch {
              answer = null;
            }
          }
          const entry: RecordedCall = { recorded_at: new Date().toISOString(), url: `${url.origin}${url.pathname}`, status: response.status, request, response: answer };
          const model = modelFrom(url, request);
          if (model) entry.model = model;
          write(entry);
        } catch {
          // Never let recording break the app.
        }
      })();
      pending.add(job);
      void job.finally(() => pending.delete(job));
    }
    return response;
  };
  (wrapped as unknown as Record<symbol, unknown>)[MARK] = true;
  globalThis.fetch = wrapped as typeof fetch;

  return {
    stop() {
      // Clients that kept a reference to the wrapper go straight to the original from now on.
      stopped = true;
      if (globalThis.fetch === (wrapped as typeof fetch)) globalThis.fetch = original;
    },
    async flush() {
      await Promise.all([...pending]);
      await writing;
    },
  };
}
