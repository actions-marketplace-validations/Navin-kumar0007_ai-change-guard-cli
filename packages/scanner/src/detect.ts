import type { ApiEntry, Platform, Provider } from "@ai-changeguard/catalog";

export type FileKind = "code" | "config";

export interface ModelHit {
  provider: Provider;
  id: string;
  raw: string;
  platformHint?: Platform;
  gateway?: string;
  fineTuned?: boolean;
  weak?: boolean;
}

export interface Finding {
  file: string;
  line: number;
  kind: "model" | "dynamic-model" | "deprecated-api";
  snippet: string;
  model?: ModelHit;
  apiId?: string;
}

export interface FileScan {
  findings: Finding[];
  isAiFile: boolean;
  callSites: number;
}

export interface ApiRule {
  id: string;
  regexes: RegExp[];
}

const NOT_MODELS = new Set(["claude-code", "claude-desktop", "gemini-cli", "gemini-code-assist"]);

const GATEWAY_PLATFORMS: Record<string, Platform | undefined> = {
  openai: undefined,
  anthropic: undefined,
  google: undefined,
  gemini: undefined,
  mistral: undefined,
  mistralai: undefined,
  openrouter: undefined,
  litellm: undefined,
  together_ai: undefined,
  groq: undefined,
  vertex_ai: "vertex-ai",
  vertex: "vertex-ai",
  bedrock: "aws-bedrock",
  azure: "azure-openai",
};

const OPENAI_RE =
  /^(?:gpt-(?:\d|image-|realtime|audio)[a-z0-9.\-]*|chatgpt-(?:4o|image)-latest|dall-e-[23]|(?:text|code)-davinci-\d{3}|(?:babbage|davinci)-002|text-moderation-(?:latest|stable|\d{3})|codex-mini-latest|computer-use-preview(?:-\d{4}-\d{2}-\d{2})?|text-embedding-(?:3-small|3-large|ada-002))$/;
/** Short OpenAI names that could be other things in code (a file name, a voice): counted only near AI code. */
const OPENAI_WEAK_RE = /^(?:whisper-1|tts-1(?:-hd)?(?:-\d{4})?|sora-2(?:-pro)?(?:-\d{4}-\d{2}-\d{2})?)$/;
const OPENAI_O_SERIES_RE = /^o\d(?:-(?:mini|pro|preview|deep-research))*(?:-\d{4}-\d{2}-\d{2})?$/;
const CLAUDE_RE = /^claude-[a-z0-9.\-]+$/;
const BEDROCK_CLAUDE_RE = /^(?:(?:us|eu|apac|global)\.)?anthropic\.(claude-[a-z0-9.\-]+?)(?:-v\d+(?::\d+)?)?$/;
const GEMINI_RE = /^(?:models\/)?(gemini-[a-z0-9.\-]+)$/;
/** Google's other model families on the Gemini API and Vertex AI: embeddings, Imagen, Veo. */
const GOOGLE_RE = /^(?:models\/)?(text-embedding-0\d{2}|embedding-gecko-001|imagen-\d[a-z0-9.\-]*|veo-\d[a-z0-9.\-]*)$/;
const GOOGLE_WEAK_RE = /^(?:models\/)?(embedding-001)$/;
const MISTRAL_RE = /^(?:mistral|codestral|pixtral|ministral|magistral|devstral|open-mistral|open-mixtral)-[a-z0-9.\-]+$/;

export function classifyModelId(literal: string): ModelHit | null {
  let s = literal.trim().toLowerCase();
  if (s.length < 2 || s.length > 120) return null;

  let fineTuned = false;
  if (s.startsWith("ft:")) {
    s = s.slice(3).split(":")[0] ?? "";
    fineTuned = true;
  }

  let gateway: string | undefined;
  let platformHint: Platform | undefined;
  if (s.includes("/") && !s.startsWith("models/")) {
    const parts = s.split("/");
    const prefix = parts[0] ?? "";
    if (!(prefix in GATEWAY_PLATFORMS) || parts.length > 3) return null;
    gateway = prefix;
    platformHint = GATEWAY_PLATFORMS[prefix];
    s = parts[parts.length - 1] ?? "";
  }
  if (!s || NOT_MODELS.has(s)) return null;

  const base: Omit<ModelHit, "provider" | "id"> = { raw: literal };
  if (fineTuned) base.fineTuned = true;
  if (gateway) base.gateway = gateway;
  if (platformHint) base.platformHint = platformHint;

  const bedrock = BEDROCK_CLAUDE_RE.exec(s);
  if (bedrock?.[1]) return { ...base, provider: "anthropic", id: bedrock[1], platformHint: "aws-bedrock" };
  if (CLAUDE_RE.test(s) && /\d/.test(s)) return { ...base, provider: "anthropic", id: s };

  const gemini = GEMINI_RE.exec(s);
  if (gemini?.[1] && /\d/.test(gemini[1])) return { ...base, provider: "google", id: gemini[1] };

  const google = GOOGLE_RE.exec(s);
  if (google?.[1]) return { ...base, provider: "google", id: google[1] };
  const googleWeak = GOOGLE_WEAK_RE.exec(s);
  if (googleWeak?.[1]) return { ...base, provider: "google", id: googleWeak[1], weak: true };

  if (OPENAI_RE.test(s)) return { ...base, provider: "openai", id: s };
  if (OPENAI_O_SERIES_RE.test(s) || OPENAI_WEAK_RE.test(s)) return { ...base, provider: "openai", id: s, weak: true };

  if (MISTRAL_RE.test(s) && /\d|latest/.test(s)) return { ...base, provider: "mistral", id: s };
  return null;
}

const STRING_RE = /(["'`])((?:\\.|(?!\1)[^\\\n])*)\1/g;
const CONFIG_VALUE_RE = /^\s*(?:export\s+)?[\w.\-]+\s*[=:]\s*["']?([^"'#\s,]+)/;
const YAML_ITEM_RE = /^\s*-\s+["']?([^"'#\s,]+)/;

/** Where the values that `candidates` reads sit on the line: string literals, and a config value. */
export function valueSpans(line: string, kind: FileKind): { start: number; end: number }[] {
  const spans: { start: number; end: number }[] = [];
  for (const m of line.matchAll(STRING_RE)) {
    if (m[2]) spans.push({ start: m.index + 1, end: m.index + 1 + m[2].length });
  }
  if (kind === "config") {
    const m = new RegExp(CONFIG_VALUE_RE.source, "d").exec(line) ?? new RegExp(YAML_ITEM_RE.source, "d").exec(line);
    const at = m?.indices?.[1];
    if (at) spans.push({ start: at[0], end: at[1] });
  }
  return spans;
}

export function candidates(line: string, kind: FileKind): string[] {
  const out: string[] = [];
  for (const m of line.matchAll(STRING_RE)) {
    if (m[2]) out.push(m[2]);
  }
  if (kind === "config") {
    const value = CONFIG_VALUE_RE.exec(line)?.[1] ?? YAML_ITEM_RE.exec(line)?.[1];
    if (value) out.push(value);
  }
  return out;
}

const AI_MARKERS = [
  "openai",
  "anthropic",
  "google.genai",
  "google.generativeai",
  "from google import genai",
  "from google import generativeai",
  "@google/genai",
  "@google/generative-ai",
  "vertexai",
  "langchain",
  "litellm",
  "mistralai",
  "@ai-sdk/",
  "bedrock-runtime",
  "ollama",
];

const SDK_CALL_PATTERNS = [
  /\bchat\.completions\.create\b/,
  /\bresponses\.create\b/,
  /\bmessages\.(?:create|stream)\b/,
  /\bgenerate_?[cC]ontent(?:_?[sS]tream)?\b/,
  /\bembeddings\.create\b/,
  /\bChat(?:OpenAI|Anthropic|GoogleGenerativeAI|VertexAI|Mistral(?:AI)?)\s*\(/,
  /\blitellm\.a?completion\s*\(/,
  /\b(?:generateText|streamText|generateObject|streamObject)\s*\(/,
  /\bInvokeModel(?:Command|WithResponseStreamCommand)?\b|\binvoke_model\b/,
];

const DYNAMIC_MODEL_RE =
  /\bmodel(?:_?[nN]ame|_?[iI]d)?\s*[:=]\s*(?![\s"'`=])([A-Za-z_$][\w$]*(?:\.[\w$]+)*(?:\[[^\]]*\])?)/;

const COMMENT_LINE_RE = /^\s*(?:#|\/\/|\/\*|\*|--)/;

const TYPE_WORDS = new Set([
  "string",
  "str",
  "number",
  "int",
  "float",
  "any",
  "unknown",
  "boolean",
  "bool",
  "Optional",
  "Literal",
  "None",
  "null",
  "undefined",
  "true",
  "false",
  "String",
  "Model",
]);

export function isAiContent(content: string): boolean {
  const lower = content.toLowerCase();
  return AI_MARKERS.some((m) => lower.includes(m));
}

export function compileApiRules(apis: readonly ApiEntry[]): ApiRule[] {
  return apis.map((a) => ({ id: a.id, regexes: a.patterns.map((p) => new RegExp(p)) }));
}

export function scanText(file: string, content: string, kind: FileKind, apiRules: ApiRule[]): FileScan {
  const isAiFile = isAiContent(content);
  const findings: Finding[] = [];
  let callSites = 0;
  const lines = content.split(/\r?\n/);

  lines.forEach((line, index) => {
    const lineNo = index + 1;
    const snippet = line.trim().slice(0, 160);
    const seen = new Set<string>();

    for (const candidate of candidates(line, kind)) {
      const hit = classifyModelId(candidate);
      if (!hit || seen.has(hit.id)) continue;
      if (hit.weak && !isAiFile) continue;
      seen.add(hit.id);
      findings.push({ file, line: lineNo, kind: "model", snippet, model: hit });
    }

    if (kind === "code" && isAiFile && seen.size === 0 && !COMMENT_LINE_RE.test(line)) {
      const dynamic = DYNAMIC_MODEL_RE.exec(line);
      const value = dynamic?.[1];
      if (value && !TYPE_WORDS.has(value)) {
        findings.push({ file, line: lineNo, kind: "dynamic-model", snippet });
      }
    }

    if (kind === "code" && SDK_CALL_PATTERNS.some((re) => re.test(line))) callSites++;

    for (const rule of apiRules) {
      if (rule.regexes.some((re) => re.test(line))) {
        findings.push({ file, line: lineNo, kind: "deprecated-api", snippet, apiId: rule.id });
      }
    }
  });

  return { findings, isAiFile, callSites };
}
