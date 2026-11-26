import { apis } from "@ai-changeguard/catalog";
import { describe, expect, it } from "vitest";
import { classifyModelId, compileApiRules, scanText, type FileKind } from "../src/detect.js";

const rules = compileApiRules(apis);

function scan(content: string, kind: FileKind = "code", file = "app.py") {
  return scanText(file, content, kind, rules);
}

function modelIds(content: string, kind: FileKind = "code", file = "app.py") {
  return scan(content, kind, file)
    .findings.filter((f) => f.kind === "model")
    .map((f) => f.model?.id);
}

describe("classifyModelId", () => {
  it.each([
    ["gpt-4o", "openai", "gpt-4o"],
    ["gpt-4o-2024-08-06", "openai", "gpt-4o-2024-08-06"],
    ["claude-3-5-sonnet-20241022", "anthropic", "claude-3-5-sonnet-20241022"],
    ["us.anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic", "claude-3-5-sonnet-20241022"],
    ["models/gemini-2.5-flash", "google", "gemini-2.5-flash"],
    ["openai/gpt-4.1-mini", "openai", "gpt-4.1-mini"],
    ["ft:gpt-4o-mini-2024-07-18:acme::abc123", "openai", "gpt-4o-mini-2024-07-18"],
    ["mistral-large-latest", "mistral", "mistral-large-latest"],
    ["gpt-image-1.5", "openai", "gpt-image-1.5"],
    ["gpt-realtime-mini", "openai", "gpt-realtime-mini"],
    ["dall-e-3", "openai", "dall-e-3"],
    ["chatgpt-image-latest", "openai", "chatgpt-image-latest"],
    ["text-moderation-latest", "openai", "text-moderation-latest"],
    ["models/text-embedding-004", "google", "text-embedding-004"],
    ["imagen-4.0-generate-001", "google", "imagen-4.0-generate-001"],
    ["veo-3.0-generate-001", "google", "veo-3.0-generate-001"],
  ])("recognises %s", (literal, provider, id) => {
    const hit = classifyModelId(literal);
    expect(hit?.provider).toBe(provider);
    expect(hit?.id).toBe(id);
  });

  it("marks Bedrock and Vertex platforms", () => {
    expect(classifyModelId("anthropic.claude-3-opus-20240229-v1:0")?.platformHint).toBe("aws-bedrock");
    expect(classifyModelId("vertex_ai/gemini-2.5-pro")?.platformHint).toBe("vertex-ai");
  });

  it("counts short names that could be something else only in AI code", () => {
    for (const id of ["whisper-1", "tts-1", "tts-1-hd", "sora-2", "embedding-001"]) expect(classifyModelId(id)?.weak, id).toBe(true);
    for (const notModel of ["gpt-images", "tts-1x", "dall-e-4", "veo-latest"]) expect(classifyModelId(notModel), notModel).toBeNull();
  });

  it("flags fine-tuned models", () => {
    expect(classifyModelId("ft:gpt-4o-mini-2024-07-18:acme::abc123")?.fineTuned).toBe(true);
  });

  it.each(["claude-code", "gemini-cli", "gpt", "hello-world", "azure/my-deployment", "some/deep/path/gpt-4o", "gemini-pro"])(
    "ignores %s",
    (literal) => {
      expect(classifyModelId(literal)).toBeNull();
    },
  );
});

describe("scanText", () => {
  it("finds a model in Python code with its line number", () => {
    const code = ["from openai import OpenAI", "client = OpenAI()", 'r = client.chat.completions.create(model="gpt-4o", messages=m)'].join("\n");
    const result = scan(code);
    const model = result.findings.find((f) => f.kind === "model");
    expect(model?.model?.id).toBe("gpt-4o");
    expect(model?.line).toBe(3);
    expect(result.callSites).toBe(1);
    expect(result.isAiFile).toBe(true);
  });

  it("finds models in .env and YAML config", () => {
    expect(modelIds("OPENAI_MODEL=gpt-4.1-mini\n", "config", ".env")).toEqual(["gpt-4.1-mini"]);
    expect(modelIds("models:\n  - gemini-2.5-pro\n", "config", "config.yaml")).toEqual(["gemini-2.5-pro"]);
  });

  it("only trusts short o-series names in files that use an AI SDK", () => {
    expect(modelIds('const label = "o1";', "code", "a.ts")).toEqual([]);
    expect(modelIds('import OpenAI from "openai";\nconst m = "o3-mini";', "code", "a.ts")).toEqual(["o3-mini"]);
  });

  it("reports models chosen at runtime", () => {
    const code = "import openai\nresp = openai.chat.completions.create(model=settings.model, messages=m)";
    const dynamic = scan(code).findings.filter((f) => f.kind === "dynamic-model");
    expect(dynamic).toHaveLength(1);
    expect(dynamic[0]?.line).toBe(2);
  });

  it("does not report comments as runtime models", () => {
    const code = "import anthropic\n# Unknown model: every mode is selectable\n  // model: newer models reason";
    expect(scan(code).findings.filter((f) => f.kind === "dynamic-model")).toEqual([]);
  });

  it("does not report type annotations as runtime models", () => {
    const code = 'import OpenAI from "openai";\ninterface Opts {\n  model: string;\n  modelName?: string;\n}';
    expect(scan(code, "code", "a.ts").findings.filter((f) => f.kind === "dynamic-model")).toEqual([]);
  });

  it("detects the shut-down Assistants API and reusable prompt IDs", () => {
    const code = [
      "from openai import OpenAI",
      "a = client.beta.assistants.create(model='gpt-4o')",
      "r = client.responses.create(prompt={'id': 'pmpt_abc123def456'})",
    ].join("\n");
    const apiIds = scan(code).findings.filter((f) => f.kind === "deprecated-api").map((f) => f.apiId);
    expect(apiIds).toEqual(["openai-assistants-api", "openai-reusable-prompts-api"]);
  });

  it("does not double-count the same model twice on one line", () => {
    expect(modelIds('pick("gpt-4o") or pick("gpt-4o")')).toEqual(["gpt-4o"]);
  });
});
