// AI package guard (AI Security Check part B): AI SDKs, agent frameworks and MCP packages a project installs,
// checked for releases published with malicious code, and (online) for names that don't exist on the registry
// or were first published days ago: signs of an invented or fake package.
import { aiPackageMatch, findCompromised, type CompromisedRelease } from "@ai-changeguard/catalog";
import type { Dependency, Ecosystem } from "./deps.js";

/** Names that look AI-related, so an unknown one is worth a registry check (invented names often look like this). */
const AI_LOOKING = /(openai|anthropic|claude|gemini|gpt|llm|langchain|langgraph|llama|mistral|ollama|genai|mcp|agent|copilot|huggingface)/i;

export function looksLikeAiPackage(name: string): boolean {
  return AI_LOOKING.test(name);
}

/** The one filter for which dependencies the guard keeps: catalog AI packages and AI-looking names. */
export function isAiRelated(ecosystem: Ecosystem, name: string): boolean {
  return aiPackageMatch(ecosystem, name) !== undefined || looksLikeAiPackage(name);
}

/** One AI package as the project uses it. */
export interface AiPackage {
  ecosystem: Ecosystem;
  name: string;
  /** Exact versions found in lockfiles or pinned manifests. */
  versions: string[];
  /** True when the catalog lists this exact name. A family match (langchain-*) or an AI-looking name is false: those get a registry check online. */
  known: boolean;
  /** Installed from a private index or registry (see Dependency.privateIndex). */
  privateIndex?: true;
  locations: string[];
}

export type PackageIssue = "compromised" | "not-found" | "new-package";

export interface PackageRow {
  ecosystem: Ecosystem;
  name: string;
  version?: string;
  issue: PackageIssue;
  /** Advisory ID for a compromised release (e.g. MAL-2026-2144). */
  advisory?: string;
  summary: string;
  url?: string;
  /** Advisory date, or the package's first publish date for a new package. */
  date?: string;
  /** Where the finding came from: the bundled catalog, OSV, or the package registry. */
  source: "catalog" | "osv" | "registry";
  locations: string[];
}

/** Groups the AI-related dependencies by package. */
export function aiPackagesOf(deps: Dependency[]): AiPackage[] {
  const byName = new Map<string, AiPackage>();
  for (const d of deps) {
    if (!isAiRelated(d.ecosystem, d.name)) continue;
    const known = aiPackageMatch(d.ecosystem, d.name) === "name";
    const key = `${d.ecosystem}|${d.name}`;
    const pkg = byName.get(key) ?? { ecosystem: d.ecosystem, name: d.name, versions: [], known, locations: [] };
    if (d.version && !pkg.versions.includes(d.version)) pkg.versions.push(d.version);
    if (d.privateIndex) pkg.privateIndex = true;
    const location = `${d.file}:${d.line}`;
    if (!pkg.locations.includes(location)) pkg.locations.push(location);
    byName.set(key, pkg);
  }
  return [...byName.values()].sort((a, b) => a.ecosystem.localeCompare(b.ecosystem) || a.name.localeCompare(b.name));
}

function compromisedRow(pkg: AiPackage, version: string, release: CompromisedRelease): PackageRow {
  return {
    ecosystem: pkg.ecosystem,
    name: pkg.name,
    version,
    issue: "compromised",
    advisory: release.advisory,
    summary: release.summary,
    url: release.sources[0]?.url ?? `https://osv.dev/vulnerability/${release.advisory}`,
    date: release.published,
    source: "catalog",
    locations: pkg.locations,
  };
}

/** Offline check: exact versions that match a known malicious release in the catalog. */
export function checkPackagesOffline(packages: AiPackage[]): PackageRow[] {
  const rows: PackageRow[] = [];
  for (const pkg of packages) {
    for (const version of pkg.versions) {
      const release = findCompromised(pkg.ecosystem, pkg.name, version);
      if (release) rows.push(compromisedRow(pkg, version, release));
    }
  }
  return rows;
}

export interface OnlineOptions {
  fetch?: typeof fetch;
  /** YYYY-MM-DD, to judge how new a package is. */
  today: string;
  /** A package first published fewer days ago than this counts as new. Default 30. */
  newDays?: number;
  /** At most this many registry lookups (unknown AI-looking names only). Default 30. */
  maxLookups?: number;
  timeoutMs?: number;
}

const DAY_MS = 86_400_000;

/** JSON from a lookup. A 404 is an answer ("not there"); any other error throws, so it counts as a failed lookup. */
async function getJson(fetchFn: typeof fetch, url: string, timeoutMs: number, init?: RequestInit): Promise<{ status: number; body?: unknown }> {
  const res = await fetchFn(url, { ...init, signal: AbortSignal.timeout(timeoutMs), headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}) } });
  if (res.status === 404) return { status: 404 };
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${new URL(url).host}`);
  return { status: res.status, body: await res.json() };
}

/** When a package was first published, or "missing" when the registry does not have it. */
async function firstPublished(fetchFn: typeof fetch, pkg: AiPackage, timeoutMs: number): Promise<string | "missing" | undefined> {
  if (pkg.ecosystem === "npm") {
    const r = await getJson(fetchFn, `https://registry.npmjs.org/${pkg.name.replace("/", "%2F")}`, timeoutMs);
    if (r.status === 404) return "missing";
    const created = (r.body as { time?: { created?: string } } | undefined)?.time?.created;
    return typeof created === "string" ? created : undefined;
  }
  const r = await getJson(fetchFn, `https://pypi.org/pypi/${encodeURIComponent(pkg.name)}/json`, timeoutMs);
  if (r.status === 404) return "missing";
  const releases = (r.body as { releases?: Record<string, { upload_time_iso_8601?: string }[]> } | undefined)?.releases ?? {};
  const times = Object.values(releases).flatMap((files) => files.map((f) => f.upload_time_iso_8601).filter((t): t is string => typeof t === "string"));
  return times.sort()[0];
}

interface OsvVuln {
  id: string;
  summary?: string;
  published?: string;
  aliases?: string[];
}

/**
 * Online check. Asks OSV (osv.dev) about every exact version, for malicious-code advisories newer than the
 * bundled catalog, and asks npm or PyPI whether unknown AI-looking names exist and how old they are.
 * Sends package names and versions only. Lookups that fail are skipped, never reported as findings.
 */
export async function checkPackagesOnline(packages: AiPackage[], options: OnlineOptions): Promise<{ rows: PackageRow[]; failed: number }> {
  const fetchFn = options.fetch ?? fetch;
  const timeoutMs = options.timeoutMs ?? 10_000;
  const rows: PackageRow[] = [];
  let failed = 0;

  const queries = packages.flatMap((pkg) => pkg.versions.map((version) => ({ pkg, version })));
  if (queries.length > 0) {
    try {
      const batch = await getJson(fetchFn, "https://api.osv.dev/v1/querybatch", timeoutMs, {
        method: "POST",
        body: JSON.stringify({ queries: queries.map((q) => ({ package: { name: q.pkg.name, ecosystem: q.pkg.ecosystem }, version: q.version })) }),
      });
      const results = (batch.body as { results?: { vulns?: { id: string }[] }[] } | undefined)?.results;
      if (!results) throw new Error("OSV answered without results");
      for (let i = 0; i < queries.length; i++) {
        const { pkg, version } = queries[i]!;
        // A version the catalog lists is already reported offline; any other version OSV flags is new, even
        // under an advisory the catalog knows (advisories gain versions).
        if (findCompromised(pkg.ecosystem, pkg.name, version)) continue;
        const malicious = (results[i]?.vulns ?? []).filter((v) => v.id.startsWith("MAL-"));
        for (const v of malicious) {
          const detail = await getJson(fetchFn, `https://api.osv.dev/v1/vulns/${encodeURIComponent(v.id)}`, timeoutMs).catch(() => ({ status: 0 }));
          const vuln = (detail as { body?: OsvVuln }).body;
          rows.push({
            ecosystem: pkg.ecosystem,
            name: pkg.name,
            version,
            issue: "compromised",
            advisory: v.id,
            summary: vuln?.summary ?? `Malicious code in ${pkg.name} ${version}.`,
            url: `https://osv.dev/vulnerability/${v.id}`,
            ...(vuln?.published ? { date: vuln.published.slice(0, 10) } : {}),
            source: "osv",
            locations: pkg.locations,
          });
        }
      }
    } catch {
      failed++;
    }
  }

  const newDays = options.newDays ?? NEW_PACKAGE_DAYS;
  const unknown = packages.filter((p) => !p.known).slice(0, options.maxLookups ?? 30);
  const todayMs = Date.parse(`${options.today}T00:00:00Z`);
  for (let i = 0; i < unknown.length; i += 5) {
    await Promise.all(
      unknown.slice(i, i + 5).map(async (pkg) => {
        // A package from a private index, or in an npm scope we don't know as an AI scope (companies publish
        // their own scoped packages privately), is never looked up: its name stays private, and "missing
        // from the public registry" would prove nothing.
        const scope = pkg.ecosystem === "npm" && pkg.name.startsWith("@") ? pkg.name.split("/")[0] + "/" : undefined;
        if (pkg.privateIndex || (scope !== undefined && aiPackageMatch("npm", pkg.name) !== "prefix")) return;
        let first: string | "missing" | undefined;
        try {
          first = await firstPublished(fetchFn, pkg, timeoutMs);
        } catch {
          failed++;
          return;
        }
        if (first === "missing") {
          rows.push({
            ecosystem: pkg.ecosystem,
            name: pkg.name,
            issue: "not-found",
            // Only a scope's owner can publish under it; an unscoped name is open to anyone.
            summary: `${pkg.name} is not on ${pkg.ecosystem === "npm" ? "npm" : "PyPI"}. If an AI assistant suggested it, the name may be invented${scope ? "" : ", and anyone could publish a package under it"}.`,
            source: "registry",
            locations: pkg.locations,
          });
        } else if (first && todayMs - Date.parse(first) < newDays * DAY_MS) {
          const date = first.slice(0, 10);
          rows.push({
            ecosystem: pkg.ecosystem,
            name: pkg.name,
            issue: "new-package",
            summary: `${pkg.name} was first published on ${date}. A brand-new package with an AI-sounding name can be a fake; check its source and owner before installing.`,
            date,
            source: "registry",
            locations: pkg.locations,
          });
        }
      }),
    );
  }
  return { rows, failed };
}

const ISSUE_ORDER: Record<PackageIssue, number> = { compromised: 0, "not-found": 1, "new-package": 2 };

/** One key per finding: a release (name and version) and what is wrong with it. */
export function packageRowKey(row: Pick<PackageRow, "ecosystem" | "name" | "version" | "issue">): string {
  return `${row.ecosystem}|${row.name}|${row.version ?? ""}|${row.issue}`;
}

/** Sorted, with one row per finding: a release flagged by several advisories, or by OSV and the catalog, shows once (catalog first). */
export function sortPackageRows(rows: PackageRow[]): PackageRow[] {
  const byKey = new Map<string, PackageRow>();
  for (const row of [...rows].sort((a, b) => Number(b.source === "catalog") - Number(a.source === "catalog"))) {
    if (!byKey.has(packageRowKey(row))) byKey.set(packageRowKey(row), row);
  }
  return [...byKey.values()].sort((a, b) => ISSUE_ORDER[a.issue] - ISSUE_ORDER[b.issue] || a.name.localeCompare(b.name));
}

/** A "new package" finding stops being news after this many days. */
export const NEW_PACKAGE_DAYS = 30;

/** One line for a person. */
export function describePackage(row: PackageRow): string {
  const what = `${row.name}${row.version ? ` ${row.version}` : ""} (${row.ecosystem})`;
  if (row.issue === "compromised") return `${what}: release published with malicious code [${row.advisory}]. ${row.summary}`;
  return `${what}: ${row.summary}`;
}

export const PACKAGE_FIX: Record<PackageIssue, string> = {
  compromised:
    "Remove this version and install a clean one. Treat any machine or CI runner that installed it as compromised: rotate every key and token it could read, from a different machine.",
  "not-found": "Check the name against the provider's official docs; remove it or replace it with the real package.",
  "new-package": "Confirm the package's source repository and owner before installing; prefer the provider's official SDK.",
};
