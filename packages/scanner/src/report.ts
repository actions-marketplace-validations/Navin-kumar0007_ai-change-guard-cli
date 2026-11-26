import {
  apis,
  defaultPlatform,
  billedPriceOn,
  daysBetween,
  localToday,
  lookupModel,
  tokenRatio,
  type ModelEntry,
  type Platform,
  type Provider,
  type TokenPrice,
} from "@ai-changeguard/catalog";
import { buildCostRows, describeCost, formatPrice, type CostRow } from "./costs.js";
import { buildLock, compareLock, formatLockChanges, LOCK_FILE, LockError, parseLock, type AiStackLock, type LockChange, type LockedBaseline } from "./lock.js";
import { aiPackagesOf, checkPackagesOffline, checkPackagesOnline, describePackage, NEW_PACKAGE_DAYS, PACKAGE_FIX, sortPackageRows, type AiPackage, type OnlineOptions, type PackageRow } from "./packages.js";
import type { ScanResult } from "./scan.js";
import { SECURITY_RULES, type SecurityRule } from "./security.js";
import { TOKEN_RULES, type TokenRule } from "./tokens.js";

export interface SecurityRow {
  rule: SecurityRule;
  title: string;
  owasp: string;
  severity: "high" | "medium" | "review";
  why: string;
  fix: string;
  findings: { location: string; detail: string }[];
}

export interface TokenTipRow {
  rule: TokenRule;
  title: string;
  fix: string;
  saving: string;
  locations: string[];
}

export type RowStatus = "retired" | "rerouted" | "retiring" | "rerouting" | "no-known-retirement";

export interface ModelRow {
  id: string;
  provider: Provider;
  platform: Platform;
  status: RowStatus;
  date?: string;
  daysLeft?: number;
  reroutesTo?: string;
  replacement?: string;
  verified?: boolean;
  note?: string;
  fineTuned: boolean;
  gateways: string[];
  otherPlatforms: { platform: Platform; date: string }[];
  locations: string[];
  /** List price today, when the catalog has it. */
  price?: TokenPrice;
  /** The suggested replacement's list price, to compare before switching. */
  replacementPrice?: TokenPrice;
  /** About how many tokens the replacement uses for 1 token of this model's (tokenizer difference); 1 is left out. */
  replacementTokenRatio?: number;
}

export interface ApiRow {
  id: string;
  name: string;
  date: string;
  daysLeft: number;
  replacement?: string;
  verified: boolean;
  locations: string[];
}

export interface Report {
  target: string;
  today: string;
  filesScanned: number;
  aiFiles: number;
  callSites: number;
  models: ModelRow[];
  dynamic: { location: string; snippet: string }[];
  apis: ApiRow[];
  tokenTips: TokenTipRow[];
  /** AI Security Check (advisory). Missing in reports stored before it existed. */
  security?: SecurityRow[];
  /** Cost Change Guard: announced price changes for the models found. Missing in older stored reports. */
  costs?: CostRow[];
  /** AI package guard: the AI packages the project installs, and problems found with them. */
  aiPackages?: AiPackage[];
  packages?: PackageRow[];
  /** Set once the online package check ran (OSV and registries); "failed" counts lookups that did not answer. */
  packagesOnline?: { checked: true; failed: number };
  /** Golden-set baselines in the repository (they go into ai-stack.lock). */
  baselines?: LockedBaseline[];
  /** The repository's ai-stack.lock and how today's scan differs from it; or why it could not be read. */
  lock?: { file: AiStackLock; changes: LockChange[] } | { error: string };
}

/** Compares the committed lock text with what this report's lock would be. */
function lockStatus(lockText: string | undefined, report: Report): Report["lock"] {
  if (lockText === undefined) return undefined;
  try {
    const file = parseLock(lockText);
    return { file, changes: compareLock(file, buildLock(report, report.baselines ?? [])) };
  } catch (error) {
    return { error: error instanceof LockError ? error.message : String(error) };
  }
}

const SEVERITY_ORDER = { high: 0, medium: 1, review: 2 } as const;

const STATUS_ORDER: Record<RowStatus, number> = {
  retired: 0,
  rerouted: 1,
  retiring: 2,
  rerouting: 3,
  "no-known-retirement": 4,
};

export { daysBetween, localToday };

function statusFor(entry: ModelEntry | undefined, today: string): Pick<ModelRow, "status" | "daysLeft" | "date"> {
  if (!entry) return { status: "no-known-retirement" };
  const daysLeft = daysBetween(today, entry.retiresOn);
  const past = daysLeft <= 0;
  const status: RowStatus = entry.reroutesTo ? (past ? "rerouted" : "rerouting") : past ? "retired" : "retiring";
  return { status, daysLeft, date: entry.retiresOn };
}

/** Announced price changes for these report rows, from the current catalog (reroutes included). */
export function costRowsFor(models: ModelRow[], today: string): CostRow[] {
  return buildCostRows(
    models.map((m) => ({
      id: m.id,
      platform: m.platform,
      locations: m.locations,
      ...(m.reroutesTo && m.date ? { reroute: { on: m.date, to: m.reroutesTo } } : {}),
    })),
    today,
  );
}

/**
 * A stored report brought up to date: retirement, reroute and price data re-read from the current catalog
 * and every day count measured from `today`. Model IDs and locations stay as scanned.
 */
export function refreshReport(report: Report, today: string): Report {
  const fresh = refreshCatalogData(report, today);
  // The committed lock is compared again with today's catalog: a retirement announced since shows up.
  if (!report.lock || !("file" in report.lock)) return fresh;
  try {
    return { ...fresh, lock: { file: report.lock.file, changes: compareLock(report.lock.file, buildLock(fresh, fresh.baselines ?? [])) } };
  } catch (error) {
    // A stored lock that no longer compares is reported, never thrown at the caller.
    return { ...fresh, lock: { error: `${LOCK_FILE} could not be compared: ${error instanceof Error ? error.message : String(error)}` } };
  }
}

function refreshCatalogData(report: Report, today: string): Report {
  const models = report.models.map((row): ModelRow => {
    const entries = lookupModel(row.id);
    const primary = entries.find((e) => e.platform === row.platform);
    if (!primary) {
      // Not in today's catalog: keep what the scan stored, with the days counted again.
      const kept: ModelRow = { ...row };
      if (row.date) Object.assign(kept, statusFor({ retiresOn: row.date, ...(row.reroutesTo ? { reroutesTo: row.reroutesTo } : {}) } as ModelEntry, today));
      return kept;
    }
    const fresh: ModelRow = {
      id: row.id,
      provider: row.provider,
      platform: row.platform,
      ...statusFor(primary, today),
      fineTuned: row.fineTuned,
      gateways: row.gateways,
      otherPlatforms: entries.filter((e) => e !== primary).map((e) => ({ platform: e.platform, date: e.retiresOn })),
      locations: row.locations,
    };
    if (primary) {
      fresh.verified = primary.verified;
      if (primary.reroutesTo) fresh.reroutesTo = primary.reroutesTo;
      if (primary.replacement) fresh.replacement = primary.replacement;
      if (primary.note) fresh.note = primary.note;
    }
    addPrices(fresh, today);
    return fresh;
  });
  return {
    ...report,
    today,
    models,
    apis: report.apis.map((a) => ({ ...a, daysLeft: daysBetween(today, a.date) })),
    costs: costRowsFor(models, today),
    // Malicious releases added to the catalog since the scan show up too. Online findings are kept until the
    // next online check replaces them, except a "new package" that is no longer new.
    ...(report.aiPackages
      ? {
          packages: sortPackageRows([
            ...checkPackagesOffline(report.aiPackages),
            ...(report.packages ?? []).filter(
              (p) => p.source !== "catalog" && !(p.issue === "new-package" && p.date && daysBetween(p.date, today) >= NEW_PACKAGE_DAYS),
            ),
          ]),
        }
      : {}),
  };
}

/** What a call is billed at today (announced changes and reroutes included), and the replacement's price. */
function addPrices(row: ModelRow, today: string): void {
  const price = billedPriceOn(row.id, row.platform, today);
  if (price) row.price = price;
  const replacementPrice = row.replacement ? billedPriceOn(row.replacement, row.platform, today) : undefined;
  if (row.replacement && replacementPrice) {
    row.replacementPrice = replacementPrice;
    const ratio = tokenRatio({ id: row.id, platform: row.platform }, { id: row.replacement, platform: row.platform });
    if (ratio !== 1) row.replacementTokenRatio = ratio;
  }
}

export function buildReport(result: ScanResult, target: string, today: string): Report {
  const rows = new Map<string, ModelRow>();
  const dynamic: Report["dynamic"] = [];
  const apiLocations = new Map<string, string[]>();

  for (const f of result.findings) {
    const location = `${f.file}:${f.line}`;
    if (f.kind === "dynamic-model") {
      dynamic.push({ location, snippet: f.snippet });
    } else if (f.kind === "deprecated-api" && f.apiId) {
      const list = apiLocations.get(f.apiId) ?? [];
      list.push(location);
      apiLocations.set(f.apiId, list);
    } else if (f.kind === "model" && f.model) {
      const hit = f.model;
      const platform = hit.platformHint ?? defaultPlatform[hit.provider];
      const key = `${hit.id}|${platform}`;
      let row = rows.get(key);
      if (!row) {
        const entries = lookupModel(hit.id);
        const primary = entries.find((e) => e.platform === platform);
        row = {
          id: hit.id,
          provider: hit.provider,
          platform,
          ...statusFor(primary, today),
          fineTuned: false,
          gateways: [],
          otherPlatforms: entries
            .filter((e) => e !== primary)
            .map((e) => ({ platform: e.platform, date: e.retiresOn })),
          locations: [],
        };
        if (primary) {
          row.verified = primary.verified;
          if (primary.reroutesTo) row.reroutesTo = primary.reroutesTo;
          if (primary.replacement) row.replacement = primary.replacement;
          if (primary.note) row.note = primary.note;
        }
        addPrices(row, today);
        rows.set(key, row);
      }
      if (hit.fineTuned) row.fineTuned = true;
      if (hit.gateway && !row.gateways.includes(hit.gateway)) row.gateways.push(hit.gateway);
      row.locations.push(location);
    }
  }

  const models = [...rows.values()].sort(
    (a, b) =>
      STATUS_ORDER[a.status] - STATUS_ORDER[b.status] ||
      (a.daysLeft ?? Infinity) - (b.daysLeft ?? Infinity) ||
      a.id.localeCompare(b.id),
  );

  const apiRows: ApiRow[] = [];
  for (const [id, locations] of apiLocations) {
    const entry = apis.find((a) => a.id === id);
    if (!entry) continue;
    const row: ApiRow = {
      id,
      name: entry.name,
      date: entry.retiresOn,
      daysLeft: daysBetween(today, entry.retiresOn),
      verified: entry.verified,
      locations,
    };
    if (entry.replacement) row.replacement = entry.replacement;
    apiRows.push(row);
  }
  apiRows.sort((a, b) => a.daysLeft - b.daysLeft);

  const tipRows = new Map<TokenRule, TokenTipRow>();
  for (const tip of result.tokenTips) {
    let row = tipRows.get(tip.rule);
    if (!row) {
      row = { rule: tip.rule, ...TOKEN_RULES[tip.rule], locations: [] };
      tipRows.set(tip.rule, row);
    }
    row.locations.push(`${tip.file}:${tip.line}`);
  }
  const tokenTips = [...tipRows.values()].sort((a, b) => b.locations.length - a.locations.length);

  const securityRows = new Map<SecurityRule, SecurityRow>();
  for (const f of result.security ?? []) {
    let row = securityRows.get(f.rule);
    if (!row) {
      row = { rule: f.rule, ...SECURITY_RULES[f.rule], findings: [] };
      securityRows.set(f.rule, row);
    }
    row.findings.push({ location: `${f.file}:${f.line}`, detail: f.detail });
  }
  const security = [...securityRows.values()].sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] || b.findings.length - a.findings.length);

  const costs = costRowsFor(models, today);
  const aiPackages = aiPackagesOf(result.dependencies ?? []);

  const report: Report = {
    target,
    today,
    filesScanned: result.filesScanned,
    aiFiles: result.aiFiles,
    callSites: result.callSites,
    models,
    dynamic,
    apis: apiRows,
    tokenTips,
    security,
    costs,
    aiPackages,
    packages: sortPackageRows(checkPackagesOffline(aiPackages)),
    baselines: result.baselines ?? [],
  };
  const lock = lockStatus(result.lockText, report);
  return lock ? { ...report, lock } : report;
}

/** Adds the online package check (OSV advisories, registry existence and age), replacing any earlier online findings. */
export async function addOnlinePackageChecks(report: Report, options: OnlineOptions): Promise<Report> {
  const { rows, failed } = await checkPackagesOnline(report.aiPackages ?? [], options);
  const offline = (report.packages ?? []).filter((p) => p.source === "catalog");
  return { ...report, packages: sortPackageRows([...offline, ...rows]), packagesOnline: { checked: true, failed } };
}

/** "uses about 30% more tokens", for a tokenizer ratio. */
export function tokenRatioText(ratio: number): string {
  const pct = Math.round(Math.abs(ratio - 1) * 100);
  return `uses about ${pct}% ${ratio > 1 ? "more" : "fewer"} tokens for the same text`;
}

function statusLabel(row: ModelRow): string {
  const star = row.verified === false ? "*" : "";
  switch (row.status) {
    case "retired":
      return `RETIRED since ${row.date}${star}`;
    case "rerouted":
      return `REROUTED to ${row.reroutesTo} since ${row.date}${star}`;
    case "retiring":
      return `RETIRES in ${row.daysLeft} days (${row.date})${star}`;
    case "rerouting":
      return `REROUTES to ${row.reroutesTo} in ${row.daysLeft} days (${row.date})${star}`;
    default:
      return "no retirement known";
  }
}

function whereLabel(locations: string[]): string {
  const [first, ...rest] = locations;
  return rest.length > 0 ? `${first} (+${rest.length} more)` : (first ?? "");
}

function table(headers: string[], rows: string[][]): string[] {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? "").length)));
  const fmt = (cells: string[]) =>
    "  " + cells.map((c, i) => (i === cells.length - 1 ? c : c.padEnd(widths[i] ?? 0))).join("  ");
  return [fmt(headers), ...rows.map(fmt)];
}

export function formatText(report: Report): string {
  const out: string[] = [];
  out.push(`AI Change Guard scan: ${report.target}`);
  out.push(
    `${report.filesScanned} files scanned · ${report.aiFiles} files use an AI SDK · ` +
      `${report.callSites} AI call sites · ${report.models.length} model IDs found · today ${report.today}`,
  );
  out.push("");

  if (report.models.length === 0 && report.dynamic.length === 0 && report.apis.length === 0) {
    out.push("No AI model IDs found in code or config.");
    out.push("If this app uses AI, the model may be set in a database, a dashboard or a gateway config. Check there.");
    // Security findings (an exposed key, an unpinned MCP server) matter even without model IDs.
    if (!report.security?.length && !report.aiPackages?.length) return out.join("\n");
    out.push("");
  }

  if (report.models.length > 0) {
    out.push("Models");
    out.push(
      ...table(
        ["STATUS", "MODEL", "PLATFORM", "WHERE"],
        report.models.map((m) => [statusLabel(m), m.fineTuned ? `${m.id} (fine-tuned)` : m.id, m.platform, whereLabel(m.locations)]),
      ),
    );
    for (const m of report.models) {
      const extras: string[] = [];
      if (m.replacement) {
        const price = m.replacementPrice ? ` (${formatPrice(m.replacementPrice)}${m.price ? `; now ${formatPrice(m.price)}` : ""})` : "";
        const tokens = m.replacementTokenRatio ? `; it ${tokenRatioText(m.replacementTokenRatio)}` : "";
        extras.push(`replacement: ${m.replacement}${price}${tokens}`);
      }
      if (m.note) extras.push(m.note);
      for (const other of m.otherPlatforms) extras.push(`on ${other.platform}: retires ${other.date}`);
      if (m.gateways.length > 0) extras.push(`called through: ${m.gateways.join(", ")}`);
      if (extras.length > 0) out.push(`  ${m.id}: ${extras.join(" · ")}`);
    }
    out.push("");
  }

  if (report.apis.length > 0) {
    out.push("Deprecated AI APIs");
    for (const a of report.apis) {
      const when = a.daysLeft <= 0 ? `SHUT DOWN since ${a.date}` : `SHUTS DOWN in ${a.daysLeft} days (${a.date})`;
      out.push(`  ${when}${a.verified ? "" : "*"}  ${a.name}  ${whereLabel(a.locations)}`);
      if (a.replacement) out.push(`    replacement: ${a.replacement}`);
    }
    out.push("");
  }

  if (report.costs?.length) {
    out.push("Price changes");
    for (const c of report.costs) {
      out.push(`  ${describeCost(c)}${c.verified ? "" : "*"}  ${whereLabel(c.locations)}`);
      if (c.note) out.push(`    ${c.note}`);
    }
    out.push("");
  }

  if (report.tokenTips.length > 0) {
    out.push("Token Saver: ways to cut your AI bill");
    for (const t of report.tokenTips) {
      out.push(`  ${t.title}  ${whereLabel(t.locations)}`);
      out.push(`    fix: ${t.fix}`);
      out.push(`    why: ${t.saving}`);
    }
    out.push("");
  }

  if (report.security?.length) {
    out.push("AI security check (advisory; OWASP Top 10 for LLM and agentic apps)");
    for (const s of report.security) {
      out.push(`  ${s.severity.toUpperCase()}  ${s.title} [${s.owasp}]`);
      for (const f of s.findings.slice(0, 10)) out.push(`    ${f.location}  ${f.detail}`);
      if (s.findings.length > 10) out.push(`    ... and ${s.findings.length - 10} more`);
      out.push(`    fix: ${s.fix}`);
    }
    out.push("");
  }

  if (report.aiPackages?.length) {
    const how = report.packagesOnline ? "checked against the catalog and OSV, unknown names against the registry" : "checked against the catalog; add --online for OSV and registry checks";
    out.push(`AI packages: ${report.aiPackages.length} found, ${how}`);
    if (!report.packages?.length) out.push("  No known malicious release among them.");
    for (const p of report.packages ?? []) {
      out.push(`  ${p.issue === "compromised" ? "HIGH  " : "REVIEW"}  ${describePackage(p)}  ${whereLabel(p.locations)}`);
      out.push(`    fix: ${PACKAGE_FIX[p.issue]}`);
      if (p.url) out.push(`    advisory: ${p.url}`);
    }
    if (report.packagesOnline?.failed) out.push(`  ${report.packagesOnline.failed} online lookup${report.packagesOnline.failed === 1 ? "" : "s"} did not answer; run again to check them.`);
    out.push("");
  }

  if (report.lock) {
    if ("error" in report.lock) out.push(`ai-stack.lock: ${report.lock.error}`);
    else out.push(...formatLockChanges(report.lock.changes));
    out.push("");
  }

  if (report.dynamic.length > 0) {
    out.push("Model set at runtime (check these by hand)");
    for (const d of report.dynamic.slice(0, 20)) out.push(`  ${d.location}  ${d.snippet}`);
    if (report.dynamic.length > 20) out.push(`  ... and ${report.dynamic.length - 20} more`);
    out.push("");
  }

  if (report.models.some((m) => m.verified === false && m.status !== "no-known-retirement")) {
    out.push("* Date not yet verified against the provider's official page.");
  }
  return out.join("\n");
}
