// ai-stack.lock: everything an app's AI depends on, in one committed file, like package-lock.json for AI.
// It lists the models (with what the catalog said about them when the lock was written), deprecated AI APIs,
// AI packages and the golden-set baselines that verified a model. File names, not line numbers, and no
// timestamps of its own, so the file changes only when the AI stack changes.
// Comparing a lock with today's scan reports two kinds of change:
//   "outdated": the code changed and the lock was not updated (a new model, a new package version, ...);
//   "catalog":  the code is the same but the facts changed (a retirement announced or moved, a reroute,
//               a price change): what the nightly check alerts on.
import { lookupPrice, lookupPriceChanges, type Platform } from "@ai-changeguard/catalog";
import { classifyModelId } from "./detect.js";
import type { McpServerTools, McpToolPrint } from "./mcp-tools.js";
import type { Report } from "./report.js";

export const LOCK_FILE = "ai-stack.lock";
export const LOCKFILE_VERSION = 1;

export interface LockedModel {
  id: string;
  platform: string;
  files: string[];
  retiresOn?: string;
  reroutesTo?: string;
  replacement?: string;
  /**
   * The catalog's list price per 1M tokens and every announced price change (first day each applies), as the
   * catalog had them when the lock was written: fixed facts, so the lock does not change as days pass.
   */
  price?: { input: number; output: number };
  priceChanges?: string[];
}

export interface LockedApi {
  id: string;
  name: string;
  retiresOn: string;
  files: string[];
}

export interface LockedPackage {
  ecosystem: string;
  name: string;
  versions: string[];
  files: string[];
}

/** A golden-set baseline found in the repository: the model it verified and when. */
export interface LockedBaseline {
  file: string;
  model: string;
  createdAt: string;
  prompts: number;
}

/** An MCP server's tools, as listed when the lock was written with --mcp. */
export interface LockedMcpServer {
  name: string;
  file: string;
  tools: McpToolPrint[];
}

export interface AiStackLock {
  lockfileVersion: number;
  models: LockedModel[];
  apis: LockedApi[];
  packages: LockedPackage[];
  baselines: LockedBaseline[];
  /** Present once written with `lock --mcp`. */
  mcp?: LockedMcpServer[];
}

/**
 * outdated: the code changed and the lock was not rewritten (run `lock`).
 * catalog: the catalog changed since the lock was written (a retirement, reroute, replacement or price).
 * unverified: a baseline verifies a model the code no longer uses (make a new baseline).
 * mcp: an MCP server's tools changed since the lock was written (checked with --mcp only).
 */
export type LockChangeKind = "outdated" | "catalog" | "unverified" | "mcp";

export interface LockChange {
  kind: LockChangeKind;
  /** What it is about, e.g. "model gpt-4o (openai-api)". */
  subject: string;
  detail: string;
}

/** A baseline file as the scanner sees it; parsed loosely so the scanner does not depend on its full format. */
export function parseBaselineSummary(file: string, content: string): LockedBaseline | undefined {
  try {
    const data = JSON.parse(content) as { version?: unknown; model?: unknown; createdAt?: unknown; items?: unknown };
    if (typeof data.version !== "number" || typeof data.model !== "string" || typeof data.createdAt !== "string" || !Array.isArray(data.items)) return undefined;
    return { file, model: data.model, createdAt: data.createdAt, prompts: data.items.length };
  } catch {
    return undefined;
  }
}

const fileOf = (location: string) => location.replace(/:\d+$/, "");
/** Code-point order: the same on every machine and locale, so the committed file never reorders by itself. */
const codePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
const filesOf = (locations: string[]) => [...new Set(locations.map(fileOf))].sort(codePoint);
const byText = <T>(key: (t: T) => string) => (a: T, b: T) => codePoint(key(a), key(b));

/** The model ID a baseline's provider:model reference stands for, normalized like scanned IDs. */
export function baselineModelId(ref: string): string {
  const model = ref.slice(ref.indexOf(":") + 1);
  return classifyModelId(model)?.id ?? model.toLowerCase();
}

/** The lock for a scan report. `mcp` (servers whose tools were listed) adds the mcp section. */
export function buildLock(report: Report, baselines: LockedBaseline[] = [], mcp?: LockedMcpServer[]): AiStackLock {
  const models = report.models
    .map((m): LockedModel => {
      const listed = lookupPrice(m.id, m.platform as Platform);
      const changes = [...new Set(lookupPriceChanges(m.id, m.platform as Platform).map((c) => c.on))].sort(codePoint);
      return {
        id: m.id,
        platform: m.platform,
        files: filesOf(m.locations),
        ...(m.date ? { retiresOn: m.date } : {}),
        ...(m.reroutesTo ? { reroutesTo: m.reroutesTo } : {}),
        ...(m.replacement ? { replacement: m.replacement } : {}),
        ...(listed ? { price: { input: listed.input, output: listed.output } } : {}),
        ...(changes.length ? { priceChanges: changes } : {}),
      };
    })
    .sort(byText((m) => `${m.id}|${m.platform}`));
  return {
    lockfileVersion: LOCKFILE_VERSION,
    models,
    apis: report.apis.map((a) => ({ id: a.id, name: a.name, retiresOn: a.date, files: filesOf(a.locations) })).sort(byText((a) => a.id)),
    packages: (report.aiPackages ?? [])
      .map((p) => ({ ecosystem: p.ecosystem, name: p.name, versions: [...p.versions].sort(codePoint), files: filesOf(p.locations) }))
      .sort(byText((p) => `${p.ecosystem}|${p.name}`)),
    baselines: [...baselines].sort(byText((b) => b.file)),
    ...(mcp ? { mcp: [...mcp].sort(byText((s) => `${s.file}|${s.name}`)) } : {}),
  };
}

/**
 * The mcp section to write: servers listed now, and for a server that could not be listed this time, what the
 * old lock had (so a server that is down does not drop out of the lock).
 */
export function mcpSection(listed: McpServerTools[], old: LockedMcpServer[] | undefined): LockedMcpServer[] {
  return listed.flatMap((s): LockedMcpServer[] => {
    if (s.tools) return [{ name: s.name, file: s.file, tools: s.tools }];
    const kept = old?.find((o) => o.name === s.name && o.file === s.file);
    return kept ? [kept] : [];
  });
}

const short = (text: string) => (text.length > 140 ? `${text.slice(0, 137)}...` : text);

/** What changed in the MCP servers' tools since the lock: tools added, removed, or with a new description or inputs. */
export function compareMcp(locked: LockedMcpServer[] | undefined, listed: McpServerTools[]): LockChange[] {
  const changes: LockChange[] = [];
  const add = (subject: string, detail: string) => changes.push({ kind: "mcp", subject, detail });
  const before = new Map((locked ?? []).map((s) => [`${s.file}|${s.name}`, s]));
  for (const s of listed) {
    const subject = `MCP server ${s.name} (${s.file})`;
    const old = before.get(`${s.file}|${s.name}`);
    before.delete(`${s.file}|${s.name}`);
    if (!s.tools) {
      // A locked server that stops answering can't be verified; one never locked is only a warning for the caller.
      if (old) add(subject, `could not list its tools, so they could not be compared: ${s.error ?? "unknown error"}`);
      continue;
    }
    if (!old) {
      add(subject, `new, with ${s.tools.length} tool${s.tools.length === 1 ? "" : "s"}: ${s.tools.map((t) => t.name).join(", ")}`);
      continue;
    }
    const oldTools = new Map(old.tools.map((t) => [t.name, t]));
    for (const t of s.tools) {
      const was = oldTools.get(t.name);
      oldTools.delete(t.name);
      if (!was) add(subject, `new tool ${t.name}: ${short(t.description)}`);
      else {
        // Compared on the full description's fingerprint: a change past the shortened text still counts.
        if (was.descriptionHash !== t.descriptionHash) add(subject, `tool ${t.name} has a new description (read it before trusting it): "${short(t.description)}"`);
        if (was.inputs !== t.inputs) add(subject, `tool ${t.name} takes different inputs`);
      }
    }
    for (const name of oldTools.keys()) add(subject, `tool ${name} removed`);
  }
  for (const s of before.values()) add(`MCP server ${s.name} (${s.file})`, "no longer in the MCP config");
  return changes;
}

export function serializeLock(lock: AiStackLock): string {
  return `${JSON.stringify(lock, null, 2)}\n`;
}

export class LockError extends Error {}

const REWRITE = "Fix it, or delete it and write a new one with: ai-change-guard lock";
const isStr = (v: unknown): v is string => typeof v === "string";
const isStrArray = (v: unknown): v is string[] => Array.isArray(v) && v.every(isStr);
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** Reads a committed lock, checking every entry's shape, so a hand-edited or badly merged file gets a clear message. */
export function parseLock(text: string): AiStackLock {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new LockError(`${LOCK_FILE} is not valid JSON. ${REWRITE}`);
  }
  if (!isObj(data)) throw new LockError(`${LOCK_FILE} is not a lock file. ${REWRITE}`);
  if (data.lockfileVersion !== LOCKFILE_VERSION) {
    throw new LockError(`${LOCK_FILE} has lockfileVersion ${String(data.lockfileVersion)}; this version reads ${LOCKFILE_VERSION}. ${REWRITE}`);
  }
  const list = (key: string, valid: (e: Record<string, unknown>) => boolean): never[] => {
    const value = data[key as keyof typeof data] ?? [];
    if (!Array.isArray(value) || !value.every((e) => isObj(e) && valid(e))) throw new LockError(`${LOCK_FILE}: "${key}" is damaged (maybe a merge). ${REWRITE}`);
    return value as never[];
  };
  const price = (v: unknown) => v === undefined || (isObj(v) && typeof v.input === "number" && typeof v.output === "number");
  const tool = (t: unknown) => isObj(t) && isStr(t.name) && isStr(t.description) && isStr(t.descriptionHash) && isStr(t.inputs);
  const mcp = data.mcp === undefined ? undefined : list("mcp", (e) => isStr(e.name) && isStr(e.file) && Array.isArray(e.tools) && e.tools.every(tool));
  return {
    lockfileVersion: LOCKFILE_VERSION,
    models: list("models", (e) => isStr(e.id) && isStr(e.platform) && isStrArray(e.files) && price(e.price) && (e.priceChanges === undefined || isStrArray(e.priceChanges))),
    apis: list("apis", (e) => isStr(e.id) && isStr(e.name) && isStr(e.retiresOn) && isStrArray(e.files)),
    packages: list("packages", (e) => isStr(e.ecosystem) && isStr(e.name) && isStrArray(e.versions) && isStrArray(e.files)),
    baselines: list("baselines", (e) => isStr(e.file) && isStr(e.model) && isStr(e.createdAt) && typeof e.prompts === "number"),
    ...(mcp ? { mcp } : {}),
  };
}

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const modelKey = (m: { id: string; platform: string }) => `${m.id}|${m.platform}`;
const money = (p?: { input: number; output: number }) => (p ? `$${p.input} / $${p.output}` : "no list price");

/** What differs between a committed lock and the lock today's scan would write. */
export function compareLock(locked: AiStackLock, current: AiStackLock): LockChange[] {
  const changes: LockChange[] = [];
  const outdated = (subject: string, detail: string) => changes.push({ kind: "outdated", subject, detail });
  const catalog = (subject: string, detail: string) => changes.push({ kind: "catalog", subject, detail });
  const unverified = (subject: string, detail: string) => changes.push({ kind: "unverified", subject, detail });

  const before = new Map(locked.models.map((m) => [modelKey(m), m]));
  const now = new Map(current.models.map((m) => [modelKey(m), m]));
  for (const [key, m] of now) {
    const subject = `model ${m.id} (${m.platform})`;
    const old = before.get(key);
    if (!old) {
      outdated(subject, `new in the code (${m.files.join(", ")})`);
      continue;
    }
    if (!same(old.retiresOn, m.retiresOn) || !same(old.reroutesTo, m.reroutesTo)) {
      const was = old.retiresOn ? `${old.reroutesTo ? `reroute to ${old.reroutesTo}` : "retirement"} on ${old.retiresOn}` : "no retirement announced";
      const is = m.retiresOn ? `${m.reroutesTo ? `reroutes to ${m.reroutesTo}` : "retires"} on ${m.retiresOn}` : "no retirement announced";
      catalog(subject, `${was} when locked; now ${is}`);
    }
    if (!same(old.replacement, m.replacement)) catalog(subject, `replacement ${old.replacement ?? "none"} when locked; now ${m.replacement ?? "none"}`);
    if (!same(old.price, m.price)) catalog(subject, `list price ${money(old.price)} when locked; now ${money(m.price)}`);
    if (!same(old.priceChanges, m.priceChanges)) {
      const added = (m.priceChanges ?? []).filter((d) => !(old.priceChanges ?? []).includes(d));
      if (added.length) catalog(subject, `price change announced for ${added.join(", ")}`);
    }
    if (!same(old.files, m.files)) outdated(subject, `now used in ${m.files.join(", ")}`);
  }
  for (const [key, m] of before) if (!now.has(key)) outdated(`model ${m.id} (${m.platform})`, "no longer in the code");

  const apisBefore = new Map(locked.apis.map((a) => [a.id, a]));
  const apisNow = new Map(current.apis.map((a) => [a.id, a]));
  for (const [id, a] of apisNow) {
    const old = apisBefore.get(id);
    if (!old) outdated(`AI API ${a.name}`, `deprecated API newly used (${a.files.join(", ")}), shuts down ${a.retiresOn}`);
    else {
      if (old.retiresOn !== a.retiresOn || old.name !== a.name) catalog(`AI API ${a.name}`, `shutdown ${old.retiresOn} when locked; now ${a.retiresOn}`);
      if (!same(old.files, a.files)) outdated(`AI API ${a.name}`, `now used in ${a.files.join(", ")}`);
    }
  }
  for (const [id, a] of apisBefore) if (!apisNow.has(id)) outdated(`AI API ${a.name}`, "no longer in the code");

  const pkgKey = (p: LockedPackage) => `${p.ecosystem}|${p.name}`;
  const pkgsBefore = new Map(locked.packages.map((p) => [pkgKey(p), p]));
  const pkgsNow = new Map(current.packages.map((p) => [pkgKey(p), p]));
  for (const [key, p] of pkgsNow) {
    const old = pkgsBefore.get(key);
    const subject = `package ${p.name} (${p.ecosystem})`;
    if (!old) outdated(subject, `new${p.versions.length ? ` at ${p.versions.join(", ")}` : ""}`);
    else {
      if (!same(old.versions, p.versions)) outdated(subject, `${old.versions.join(", ") || "unpinned"} when locked; now ${p.versions.join(", ") || "unpinned"}`);
      if (!same(old.files, p.files)) outdated(subject, `now declared in ${p.files.join(", ")}`);
    }
  }
  for (const [key, p] of pkgsBefore) if (!pkgsNow.has(key)) outdated(`package ${p.name} (${p.ecosystem})`, "no longer installed");

  // A baseline verifies one model: if the code no longer uses it, the switch was not re-verified.
  const baselinesNow = new Map(current.baselines.map((b) => [b.file, b]));
  for (const b of locked.baselines) {
    const nowB = baselinesNow.get(b.file);
    if (!nowB) outdated(`baseline ${b.file}`, "removed");
    else if (nowB.model !== b.model || nowB.createdAt !== b.createdAt || nowB.prompts !== b.prompts) {
      outdated(`baseline ${b.file}`, `now verifies ${nowB.model} on ${nowB.prompts} prompts (made ${nowB.createdAt})`);
    }
  }
  for (const b of current.baselines) if (!locked.baselines.some((x) => x.file === b.file)) outdated(`baseline ${b.file}`, `new: verifies ${b.model}`);
  // A baseline verifies one model; one the code no longer uses means the switch was not re-verified.
  // Rewriting the lock does not fix this: only a new baseline does, so it is its own kind.
  const usedIds = new Set(current.models.map((m) => m.id));
  for (const b of current.baselines) {
    if (!usedIds.has(baselineModelId(b.model))) unverified(`baseline ${b.file}`, `verifies ${b.model}, which the code no longer uses`);
  }
  return changes;
}

/** The three groups of lock changes, in order, each with its heading and what to do: one wording for text and Markdown. */
export function groupLockChanges(changes: LockChange[]): { kind: LockChangeKind; heading: string; remedy: string; changes: LockChange[] }[] {
  const groups: { kind: LockChangeKind; heading: string; remedy: string }[] = [
    { kind: "catalog", heading: "changed in the catalog since the lock was written", remedy: "check these, then run ai-change-guard lock and commit" },
    { kind: "outdated", heading: "out of date with the code", remedy: "run ai-change-guard lock and commit it, so the pull request shows the AI stack change" },
    { kind: "unverified", heading: "model switched without a new baseline", remedy: "make a new baseline for the model in use with ai-change-guard baseline" },
    { kind: "mcp", heading: "MCP tools changed since the lock was written", remedy: "read the changes; if you trust them, run ai-change-guard lock --mcp and commit" },
  ];
  return groups.map((g) => ({ ...g, changes: changes.filter((c) => c.kind === g.kind) })).filter((g) => g.changes.length > 0);
}

export function formatLockChanges(changes: LockChange[]): string[] {
  if (changes.length === 0) return [`${LOCK_FILE}: up to date with the code and the catalog.`];
  return groupLockChanges(changes).flatMap((g) => [`${LOCK_FILE}: ${g.heading} (${g.remedy}):`, ...g.changes.map((c) => `  ${c.subject}: ${c.detail}`)]);
}
