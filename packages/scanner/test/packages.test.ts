import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { aiPackageMatch, findCompromised } from "@ai-changeguard/catalog";
import { describe, expect, it } from "vitest";
import { buildCheck } from "../src/check.js";
import { lineOf, normalizeName, ownPackageName, parseDependencies, privateRegistries } from "../src/deps.js";
import { aiPackagesOf, checkPackagesOffline, checkPackagesOnline, type AiPackage } from "../src/packages.js";
import { addOnlinePackageChecks, buildReport, formatText } from "../src/report.js";
import { scanDirectory } from "../src/scan.js";

const names = (file: string, content: string) => parseDependencies(file, content).map((d) => `${d.name}@${d.version ?? "*"}`);

describe("dependency files", () => {
  it("reads package-lock.json (v3 and v1)", () => {
    const v3 = JSON.stringify({ lockfileVersion: 3, packages: { "": {}, "node_modules/openai": { version: "6.1.0" }, "node_modules/a/node_modules/@langchain/core": { version: "1.0.2" }, "node_modules/local": { link: true } } });
    expect(names("package-lock.json", v3)).toEqual(["openai@6.1.0", "@langchain/core@1.0.2"]);
    const v1 = JSON.stringify({ lockfileVersion: 1, dependencies: { ai: { version: "5.0.0", dependencies: { "@ai-sdk/provider": { version: "2.0.0" } } } } });
    expect(names("package-lock.json", v1)).toEqual(["ai@5.0.0", "@ai-sdk/provider@2.0.0"]);
  });

  it("reads package.json, keeping exact pins and skipping workspace, file and git specs", () => {
    const pkg = JSON.stringify({ name: "@acme/app", dependencies: { openai: "^6.0.0", "@mastra/core": "1.42.1", "@acme/agent": "workspace:*", llm: "github:x/y" }, devDependencies: { "@ai-sdk/openai": "=2.0.1" } });
    expect(names("apps/web/package.json", pkg)).toEqual(["openai@*", "@mastra/core@1.42.1", "@ai-sdk/openai@2.0.1"]);
    expect(ownPackageName("apps/web/package.json", pkg)).toBe("@acme/app");
  });

  it("reads pnpm-lock.yaml (v9, v6 and v5 keys)", () => {
    const lock = ["lockfileVersion: '9.0'", "packages:", "  '@mastra/core@1.42.1':", "    resolution: {}", "  openai@6.1.0:", "  /ai@5.0.0:", "  /@ai-sdk/openai/2.0.1:"].join("\n");
    expect(names("pnpm-lock.yaml", lock)).toEqual(["@mastra/core@1.42.1", "openai@6.1.0", "ai@5.0.0", "@ai-sdk/openai@2.0.1"]);
  });

  it("reads yarn.lock (classic and berry)", () => {
    const classic = ['"@mistralai/mistralai@^2.2.0":', '  version "2.2.3"', "", "openai@^6, openai@^6.1:", '  version "6.1.0"'].join("\n");
    expect(names("yarn.lock", classic)).toEqual(["@mistralai/mistralai@2.2.3", "openai@6.1.0"]);
    const berry = ['"ai@npm:^5.0.0":', "  version: 5.0.3", "  resolution: x"].join("\n");
    expect(names("yarn.lock", berry)).toEqual(["ai@5.0.3"]);
  });

  it("reads poetry.lock, uv.lock and Pipfile.lock, with PyPI names normalized", () => {
    const toml = ["[[package]]", 'name = "LiteLLM"', 'version = "1.82.8"', "", "[[package]]", 'name = "guardrails_ai"', 'version = "0.10.1"'].join("\n");
    expect(names("poetry.lock", toml)).toEqual(["litellm@1.82.8", "guardrails-ai@0.10.1"]);
    expect(names("uv.lock", toml)).toEqual(["litellm@1.82.8", "guardrails-ai@0.10.1"]);
    const pipfile = JSON.stringify({ default: { mistralai: { version: "==2.4.6" } }, develop: { anthropic: {} } });
    expect(names("Pipfile.lock", pipfile)).toEqual(["mistralai@2.4.6", "anthropic@*"]);
  });

  it("reads requirements files, and notes a private index", () => {
    const req = ["# comment", "openai[realtime]==1.99.0", "litellm == 1.82.8  # proxy", "langchain>=0.3", "-r other.txt", "torch==2.*"].join("\n");
    expect(names("requirements.txt", req)).toEqual(["openai@1.99.0", "litellm@1.82.8", "langchain@*", "torch@*"]);
    expect(parseDependencies("requirements.txt", req).some((d) => d.privateIndex)).toBe(false);
    expect(parseDependencies("requirements-dev.txt", "--extra-index-url https://pkgs.acme.dev\nacme-llm==1.0").every((d) => d.privateIndex)).toBe(true);
  });

  it("ignores files it cannot read, and finds lines", () => {
    expect(parseDependencies("package-lock.json", "{not json")).toEqual([]);
    expect(parseDependencies("README.md", "openai==1")).toEqual([]);
    expect(lineOf("a\nb\nlitellm==1\n", "litellm")).toBe(3);
    expect(normalizeName("PyPI", "Pydantic_AI.slim")).toBe("pydantic-ai-slim");
    expect(privateRegistries("@acme:registry=https://npm.acme.dev\n")).toEqual({ all: false, scopes: ["@acme"] });
    expect(privateRegistries("registry=https://npm.acme.dev\nsave-exact=true\n")).toEqual({ all: true, scopes: [] });
  });
});

describe("package catalog", () => {
  it("knows the 2026 malicious AI releases", () => {
    expect(findCompromised("PyPI", "litellm", "1.82.8")?.advisory).toBe("PYSEC-2026-2");
    expect(findCompromised("PyPI", "litellm", "1.107.0")?.advisory).toBe("MAL-2026-2144");
    expect(findCompromised("PyPI", "litellm", "1.82.6")).toBeUndefined();
    expect(findCompromised("npm", "@mistralai/mistralai", "2.2.3")).toBeDefined();
    expect(findCompromised("npm", "@mastra/core", "1.42.1")).toBeDefined();
  });

  it("tells a listed package from a family match", () => {
    expect(aiPackageMatch("PyPI", "langchain")).toBe("name");
    expect(aiPackageMatch("PyPI", "langchain-openai")).toBe("prefix");
    expect(aiPackageMatch("npm", "@langchain/core")).toBe("prefix");
    expect(aiPackageMatch("npm", "react")).toBeUndefined();
  });
});

const pkg = (over: Partial<AiPackage> & Pick<AiPackage, "name">): AiPackage => ({ ecosystem: "PyPI", versions: [], known: false, locations: ["requirements.txt:1"], ...over });

describe("offline check", () => {
  it("keeps AI packages only and flags exact malicious versions", () => {
    const deps = [
      { ecosystem: "PyPI" as const, name: "litellm", version: "1.82.8", file: "requirements.txt", line: 2 },
      { ecosystem: "PyPI" as const, name: "requests", version: "2.32.0", file: "requirements.txt", line: 3 },
      { ecosystem: "npm" as const, name: "my-gpt-helper", file: "package.json", line: 5 },
    ];
    const packages = aiPackagesOf(deps);
    expect(packages.map((p) => [p.name, p.known])).toEqual([
      ["my-gpt-helper", false],
      ["litellm", true],
    ]);
    expect(checkPackagesOffline(packages)).toMatchObject([{ name: "litellm", version: "1.82.8", issue: "compromised", advisory: "PYSEC-2026-2", source: "catalog", locations: ["requirements.txt:2"] }]);
  });
});

function fakeFetch(routes: Record<string, { status: number; body?: unknown }>, calls: string[] = []): typeof fetch {
  return (async (url: string | URL, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${String(url)}`;
    calls.push(key);
    const hit = routes[key];
    if (!hit) throw new Error(`unexpected ${key}`);
    return new Response(hit.body === undefined ? null : JSON.stringify(hit.body), { status: hit.status });
  }) as typeof fetch;
}

describe("online check", () => {
  it("reports a malicious OSV advisory the catalog does not have yet", async () => {
    const calls: string[] = [];
    const fetchFn = fakeFetch(
      {
        "POST https://api.osv.dev/v1/querybatch": { status: 200, body: { results: [{ vulns: [{ id: "GHSA-aaaa" }, { id: "MAL-2027-1" }] }] } },
        "GET https://api.osv.dev/v1/vulns/MAL-2027-1": { status: 200, body: { id: "MAL-2027-1", summary: "Malicious code in openai (PyPI)", published: "2027-01-05T00:00:00Z" } },
      },
      calls,
    );
    const { rows, failed } = await checkPackagesOnline([pkg({ name: "openai", versions: ["9.9.9"], known: true })], { fetch: fetchFn, today: "2027-01-10" });
    expect(failed).toBe(0);
    // Ordinary vulnerabilities (GHSA-...) are left to Dependabot and the like; only malicious code is reported.
    expect(rows).toEqual([expect.objectContaining({ name: "openai", version: "9.9.9", issue: "compromised", advisory: "MAL-2027-1", date: "2027-01-05", source: "osv" })]);
    // A known package is not looked up on the registry.
    expect(calls.some((c) => c.includes("pypi.org"))).toBe(false);
  });

  it("does not report a catalog release twice", async () => {
    const fetchFn = fakeFetch({ "POST https://api.osv.dev/v1/querybatch": { status: 200, body: { results: [{ vulns: [{ id: "MAL-2026-2144" }] }] } } });
    const { rows } = await checkPackagesOnline([pkg({ name: "litellm", versions: ["1.82.8"], known: true })], { fetch: fetchFn, today: "2026-11-23" });
    expect(rows).toEqual([]);
  });

  it("flags an AI-looking name that is not on the registry, or brand new", async () => {
    const fetchFn = fakeFetch({
      "GET https://pypi.org/pypi/langchain-gemini-turbo/json": { status: 404 },
      "GET https://registry.npmjs.org/mcp-server-fresh": { status: 200, body: { time: { created: "2026-11-20T10:00:00Z" } } },
      "GET https://registry.npmjs.org/old-llm-tool": { status: 200, body: { time: { created: "2023-01-01T00:00:00Z" } } },
    });
    const { rows } = await checkPackagesOnline(
      [pkg({ name: "langchain-gemini-turbo" }), pkg({ ecosystem: "npm", name: "mcp-server-fresh" }), pkg({ ecosystem: "npm", name: "old-llm-tool" })],
      { fetch: fetchFn, today: "2026-11-23" },
    );
    expect(rows.map((r) => [r.name, r.issue])).toEqual([
      ["langchain-gemini-turbo", "not-found"],
      ["mcp-server-fresh", "new-package"],
    ]);
    expect(rows[0]!.summary).toContain("anyone could publish");
  });

  it("does not call a private package missing", async () => {
    const fetchFn = fakeFetch({
      "GET https://pypi.org/pypi/acme-llm/json": { status: 404 },
      "GET https://registry.npmjs.org/@acme%2Fllm-kit": { status: 404 },
      "GET https://registry.npmjs.org/@langchain%2Fmade-up": { status: 404 },
    });
    const { rows } = await checkPackagesOnline(
      [pkg({ name: "acme-llm", privateIndex: true }), pkg({ ecosystem: "npm", name: "@acme/llm-kit" }), pkg({ ecosystem: "npm", name: "@langchain/made-up" })],
      { fetch: fetchFn, today: "2026-11-23" },
    );
    // Only the invented name in a scope that belongs to an AI company is reported, without "anyone could publish".
    expect(rows.map((r) => r.name)).toEqual(["@langchain/made-up"]);
    expect(rows[0]!.summary).not.toContain("anyone could publish");
  });

  it("counts lookups that fail instead of reporting them", async () => {
    const fetchFn = (async () => {
      throw new Error("offline");
    }) as typeof fetch;
    const { rows, failed } = await checkPackagesOnline([pkg({ name: "gpt-thing", versions: ["1.0"] })], { fetch: fetchFn, today: "2026-11-23" });
    expect(rows).toEqual([]);
    expect(failed).toBe(2);
  });
});

describe("scan, report and check", () => {
  it("finds a malicious release in a lockfile, skips the repo's own workspaces, and fails a PR that touches it", async () => {
    const dir = await mkdtemp(join(tmpdir(), "acg-pkg-"));
    await writeFile(join(dir, "requirements.txt"), "litellm==1.82.8\n");
    await writeFile(join(dir, "package.json"), JSON.stringify({ name: "@acme/agent-kit", dependencies: { "@acme/agent-kit": "1.0.0" } }));
    await writeFile(join(dir, "uv.lock"), '[[package]]\nname = "mistralai"\nversion = "2.4.6"\n');
    const report = buildReport(await scanDirectory(dir), "demo", "2026-11-23");
    expect(report.aiPackages?.map((p) => p.name)).toEqual(["litellm", "mistralai"]);
    expect(report.packages?.map((p) => `${p.name}:${p.issue}`)).toEqual(["litellm:compromised", "mistralai:compromised"]);
    expect(report.packages?.[1]?.locations).toEqual(["uv.lock:2"]);
    expect(formatText(report)).toContain("HIGH    litellm 1.82.8 (PyPI): release published with malicious code [PYSEC-2026-2]");

    const inPr = buildCheck(report, ["uv.lock"]);
    expect(inPr.conclusion).toBe("failure");
    expect(inPr.title).toContain("Malicious AI package release installed");
    expect(inPr.summary).toContain("This pull request installs an AI package release");
    expect(buildCheck(report, ["README.md"]).conclusion).toBe("neutral");
  });

  it("adds online findings to a report and says so", async () => {
    const base = buildReport({ root: "/x", filesScanned: 1, aiFiles: 0, callSites: 0, findings: [], tokenTips: [], security: [], dependencies: [{ ecosystem: "PyPI", name: "gpt-fake-sdk", file: "requirements.txt", line: 1 }] }, "demo", "2026-11-23");
    const online = await addOnlinePackageChecks(base, { today: "2026-11-23", fetch: fakeFetch({ "GET https://pypi.org/pypi/gpt-fake-sdk/json": { status: 404 } }) });
    expect(online.packagesOnline).toEqual({ checked: true, failed: 0 });
    expect(online.packages?.map((p) => p.issue)).toEqual(["not-found"]);
    expect(formatText(online)).toContain("REVIEW  gpt-fake-sdk (PyPI): gpt-fake-sdk is not on PyPI");
    expect(buildCheck(online, ["requirements.txt"]).conclusion).toBe("success");
  });
});

describe("refreshing a stored report", () => {
  it("re-checks stored AI packages against today's catalog and keeps online findings", async () => {
    const { refreshReport } = await import("../src/report.js");
    const base = buildReport({ root: "/x", filesScanned: 1, aiFiles: 0, callSites: 0, findings: [], tokenTips: [], security: [], dependencies: [{ ecosystem: "PyPI", name: "litellm", version: "1.107.0", file: "uv.lock", line: 9 }] }, "demo", "2026-11-23");
    // Stored before the catalog listed litellm 1.107.0, with one online finding.
    const online = { ecosystem: "npm" as const, name: "gpt-x", issue: "not-found" as const, summary: "Not on npm.", source: "registry" as const, locations: ["package.json:1"] };
    const stored = { ...base, packages: [online] };
    expect(refreshReport(stored, "2026-11-24").packages?.map((p) => `${p.name}:${p.issue}:${p.source}`)).toEqual(["litellm:compromised:catalog", "gpt-x:not-found:registry"]);
  });
});

describe("review fixes", () => {
  it("skips workspace folders in package-lock.json and reads aliases as the real package", () => {
    const lock = JSON.stringify({ lockfileVersion: 3, packages: { "": {}, "packages/my-agent-tools": { version: "1.0.0" }, "node_modules/my-agent-tools": { link: true }, "node_modules/my-ai": { name: "litellm-js", version: "1.2.3" } } });
    expect(names("package-lock.json", lock)).toEqual(["litellm-js@1.2.3"]);
    expect(names("package.json", JSON.stringify({ dependencies: { "my-ai": "npm:@mastra/core@1.42.1" } }))).toEqual(["@mastra/core@1.42.1"]);
    expect(names("yarn.lock", '"my-ai@npm:@mistralai/mistralai@^2":\n  version "2.2.3"\n')).toEqual(["@mistralai/mistralai@2.2.3"]);
  });

  it("reads CRLF lockfiles and pnpm v5 entries with peer suffixes", () => {
    expect(names("yarn.lock", 'openai@^6:\r\n  version "6.1.0"\r\n')).toEqual(["openai@6.1.0"]);
    expect(names("pnpm-lock.yaml", "packages:\n  /langchain/0.1.0_openai@4.0.0:\n")).toEqual(["langchain@0.1.0"]);
  });

  it("skips the project and local packages in uv.lock and poetry.lock, and notes other indexes", () => {
    const uv = [
      "[[package]]", 'name = "acme-llm-service"', 'version = "0.1.0"', 'source = { editable = "." }',
      "[[package]]", 'name = "agent-helpers"', 'version = "0.2.0"', 'source = { virtual = "libs/agent" }',
      "[[package]]", 'name = "acme-gpt-utils"', 'version = "1.0.0"', 'source = { registry = "https://pkgs.acme.dev/simple" }',
      "[[package]]", 'name = "openai"', 'version = "1.99.0"', 'source = { registry = "https://pypi.org/simple" }',
    ].join("\n");
    const deps = parseDependencies("uv.lock", uv);
    expect(deps.map((d) => [d.name, Boolean(d.privateIndex)])).toEqual([
      ["acme-gpt-utils", true],
      ["openai", false],
    ]);
    const poetry = ["[[package]]", 'name = "my-llm-lib"', 'version = "0.1.0"', "", "[package.source]", 'type = "directory"', 'url = "libs/llm"'].join("\n");
    expect(names("poetry.lock", poetry)).toEqual([]);
  });

  it("treats a scope-only .npmrc registry as private for that scope only", async () => {
    const dir = await mkdtemp(join(tmpdir(), "acg-npmrc-"));
    await writeFile(join(dir, ".npmrc"), "@acme:registry=https://npm.acme.dev\n");
    await writeFile(join(dir, "package.json"), JSON.stringify({ dependencies: { "@acme/llm-kit": "1.0.0", "openai-agent-sdk": "1.0.0" } }));
    const deps = (await scanDirectory(dir)).dependencies ?? [];
    expect(deps.map((d) => [d.name, Boolean(d.privateIndex)])).toEqual([
      ["@acme/llm-kit", true],
      ["openai-agent-sdk", false],
    ]);
  });

  it("counts an OSV or registry error as a failed lookup, not a clean result", async () => {
    const fetchFn = fakeFetch({ "POST https://api.osv.dev/v1/querybatch": { status: 503 }, "GET https://pypi.org/pypi/gpt-thing/json": { status: 500 } });
    const { rows, failed } = await checkPackagesOnline([pkg({ name: "gpt-thing", versions: ["1.0"] })], { fetch: fetchFn, today: "2026-11-23" });
    expect(rows).toEqual([]);
    expect(failed).toBe(2);
  });

  it("reports a version OSV adds to an advisory the catalog already knows", async () => {
    const fetchFn = fakeFetch({
      "POST https://api.osv.dev/v1/querybatch": { status: 200, body: { results: [{ vulns: [{ id: "MAL-2026-2144" }] }] } },
      "GET https://api.osv.dev/v1/vulns/MAL-2026-2144": { status: 200, body: { id: "MAL-2026-2144", summary: "Malicious code in litellm (PyPI)" } },
    });
    const { rows } = await checkPackagesOnline([pkg({ name: "litellm", versions: ["1.82.9"], known: true })], { fetch: fetchFn, today: "2026-11-23" });
    expect(rows.map((r) => `${r.version}:${r.advisory}`)).toEqual(["1.82.9:MAL-2026-2144"]);
  });
});

describe("website review fixes", () => {
  it("never sends private or unknown-scope names to a public registry", async () => {
    const calls: string[] = [];
    const fetchFn = fakeFetch({}, calls);
    await checkPackagesOnline([pkg({ name: "acme-llm", privateIndex: true }), pkg({ ecosystem: "npm", name: "@acme/llm-kit" })], { fetch: fetchFn, today: "2026-11-23" });
    expect(calls).toEqual([]);
  });

  it("shows a release once, even when OSV and the catalog both flag it", async () => {
    const { sortPackageRows, packageRowKey } = await import("../src/packages.js");
    const base = { ecosystem: "PyPI" as const, name: "litellm", version: "1.82.8", issue: "compromised" as const, summary: "x", locations: [] };
    const rows = sortPackageRows([{ ...base, advisory: "MAL-1", source: "osv" }, { ...base, advisory: "PYSEC-2026-2", source: "catalog" }, { ...base, advisory: "MAL-2", source: "osv" }]);
    expect(rows.map((r) => r.advisory)).toEqual(["PYSEC-2026-2"]);
    expect(packageRowKey(base)).toBe("PyPI|litellm|1.82.8|compromised");
  });

  it("replaces earlier online findings on a new online check, and lets a new package age out", async () => {
    const { refreshReport } = await import("../src/report.js");
    const base = buildReport({ root: "/x", filesScanned: 1, aiFiles: 0, callSites: 0, findings: [], tokenTips: [], security: [], dependencies: [{ ecosystem: "npm", name: "mcp-server-fresh", file: "package.json", line: 1 }] }, "demo", "2026-11-23");
    const first = await addOnlinePackageChecks(base, { today: "2026-11-23", fetch: fakeFetch({ "GET https://registry.npmjs.org/mcp-server-fresh": { status: 200, body: { time: { created: "2026-11-20T00:00:00Z" } } } }) });
    expect(first.packages?.map((p) => p.issue)).toEqual(["new-package"]);
    const again = await addOnlinePackageChecks(first, { today: "2026-11-24", fetch: fakeFetch({ "GET https://registry.npmjs.org/mcp-server-fresh": { status: 200, body: { time: { created: "2026-11-20T00:00:00Z" } } } }) });
    expect(again.packages).toHaveLength(1);
    expect(refreshReport(first, "2026-12-19").packages).toHaveLength(1);
    expect(refreshReport(first, "2026-12-20").packages).toEqual([]);
  });
});

describe("isScannablePath", () => {
  it("matches the walker: code, config and lockfiles, never dependency or build folders", async () => {
    const { isScannablePath } = await import("../src/walk.js");
    for (const p of ["app.py", "src/bot.ts", "backend/requirements.txt", "package-lock.json", "uv.lock", ".npmrc", "ai-stack.lock", ".mcp.json", ".env.example", "config/settings.yaml"]) expect(isScannablePath(p), p).toBe(true);
    for (const p of ["backend/venv/lib/python3.11/site-packages/x.py", "node_modules/openai/index.js", "dist/app.js", "logo.png", "Cargo.lock", "app.min.js", "README.md", "backend/git_alpha.db"]) expect(isScannablePath(p), p).toBe(false);
  });

  it("agrees with walk() on a real folder", async () => {
    const { isScannablePath, walk } = await import("../src/walk.js");
    const dir = await mkdtemp(join(tmpdir(), "acg-walk-"));
    const { mkdir } = await import("node:fs/promises");
    for (const f of ["a.py", "venv/lib/b.py", "web/package.json", "web/node_modules/x/index.js", "notes.md", "uv.lock"]) {
      await mkdir(join(dir, f, ".."), { recursive: true });
      await writeFile(join(dir, f), "x");
    }
    const walked: string[] = [];
    for await (const f of walk(dir)) walked.push(f.rel);
    expect(walked.sort()).toEqual(["a.py", "uv.lock", "web/package.json"]);
    expect(["a.py", "venv/lib/b.py", "web/package.json", "web/node_modules/x/index.js", "notes.md", "uv.lock"].filter(isScannablePath).sort()).toEqual(walked.sort());
  });
});
