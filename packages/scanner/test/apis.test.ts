import { apis } from "@ai-changeguard/catalog";
import { describe, expect, it } from "vitest";
import { compileApiRules, scanText, type FileKind } from "../src/detect.js";
import { fileKind } from "../src/walk.js";

const rules = compileApiRules(apis);
const hits = (file: string, content: string, kind: FileKind = "code") =>
  scanText(file, content, kind, rules).findings.filter((f) => f.kind === "deprecated-api").map((f) => `${f.apiId}@${f.line}`);

describe("AI API and SDK shutdowns", () => {
  it("finds the legacy Google SDKs in code and dependency files, but not their replacements", () => {
    expect(hits("app.py", "import google.generativeai as genai\nfrom google import genai as new\n")).toEqual(["google-generativeai-python-sdk@1"]);
    expect(hits("requirements.txt", "requests==2.32\ngoogle-generativeai==0.8.5\ngoogle-genai==1.2\n", "config")).toEqual(["google-generativeai-python-sdk@2"]);
    expect(hits("package.json", '{ "dependencies": { "@google/generative-ai": "^0.21.0", "@google/genai": "^1.0.0" } }', "config")).toEqual([
      "google-generative-ai-js-sdk@1",
    ]);
    expect(hits("v.py", "from vertexai.generative_models import GenerativeModel\nimport vertexai\n")).toEqual(["vertex-ai-sdk-generative-module@1"]);
  });

  it("finds retired and retiring OpenAI endpoints and beta headers", () => {
    const code = [
      'client.audio.translations.create(model="whisper-1", file=f)',
      "client.fine_tuning.jobs.create(training_file=fid, model=base)",
      'headers = {"OpenAI-Beta": "realtime=v1"}',
      'fetch("https://api.openai.com/v1/videos", opts)',
      'requests.post("https://api.openai.com/v1/edits", json=body)',
    ].join("\n");
    expect(hits("calls.py", code)).toEqual([
      "openai-audio-translations@1",
      "openai-fine-tuning-job-creation@2",
      "openai-realtime-beta-header@3",
      "openai-videos-api@4",
      "openai-legacy-endpoints@5",
    ]);
  });

  it("leaves look-alikes alone", () => {
    const code = [
      "youtube.videos.list(part='snippet')",
      "client.fine_tuning.jobs.list()",
      "client.audio.transcriptions.create(model='whisper-1', file=f)",
      'url = "https://example.com/v1/videos/123"',
    ].join("\n");
    expect(hits("misc.py", code)).toEqual([]);
  });

  it("reads requirements files as config", () => {
    expect(fileKind("requirements.txt")).toBe("config");
    expect(fileKind("requirements-dev.txt")).toBe("config");
    expect(fileKind("notes.txt")).toBeNull();
  });

  it("gives every catalog entry valid patterns, a date and a source", () => {
    for (const a of apis) {
      expect(a.retiresOn, a.id).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(a.sources.length, a.id).toBeGreaterThan(0);
      for (const p of a.patterns) expect(() => new RegExp(p), `${a.id}: ${p}`).not.toThrow();
    }
  });
});
