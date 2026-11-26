import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { apis } from "@ai-changeguard/catalog";
import { compileApiRules, scanText, type Finding } from "./detect.js";
import { isDependencyFile, lineOf, ownPackageName, parseDependencies, privateRegistries, type Dependency } from "./deps.js";
import { isAiRelated } from "./packages.js";
import { LOCK_FILE, parseBaselineSummary, type LockedBaseline } from "./lock.js";
import { findSecurityIssues, isMcpConfig, parseMcpServers, type SecurityFinding } from "./security.js";
import type { McpServerConfig } from "./mcp-tools.js";
import { findTokenTips, type TokenTip } from "./tokens.js";
import { walk } from "./walk.js";

const MAX_FILE_BYTES = 1_000_000;
/** Lockfiles of big projects run to many megabytes. */
const MAX_LOCKFILE_BYTES = 30_000_000;

export interface ScanResult {
  root: string;
  filesScanned: number;
  aiFiles: number;
  callSites: number;
  findings: Finding[];
  tokenTips: TokenTip[];
  security: SecurityFinding[];
  /** AI-related dependencies from manifests and lockfiles (missing in results made before the package guard). */
  dependencies?: Dependency[];
  /** Golden-set baselines (*.baseline.json) found in the folder. */
  baselines?: LockedBaseline[];
  /** The folder's ai-stack.lock, as text, when it has one. */
  lockText?: string;
  /** MCP servers declared in the folder's MCP configs (env and header values included: never store these). */
  mcpServers?: McpServerConfig[];
}

export async function scanDirectory(root: string): Promise<ScanResult> {
  const info = await stat(root).catch(() => null);
  if (!info?.isDirectory()) throw new Error(`Not a folder: ${root}`);

  const rules = compileApiRules(apis);
  const result: ScanResult = { root, filesScanned: 0, aiFiles: 0, callSites: 0, findings: [], tokenTips: [], security: [], dependencies: [], baselines: [], mcpServers: [] };

  const ownPackages = new Set<string>();
  for await (const file of walk(root)) {
    const size = (await stat(file.path)).size;
    if (size > (file.lock ? MAX_LOCKFILE_BYTES : MAX_FILE_BYTES)) continue;
    const content = await readFile(file.path, "utf8");
    if (content.includes("\u0000")) continue;

    if (file.rel.endsWith(".baseline.json")) {
      const baseline = parseBaselineSummary(file.rel, content);
      if (baseline) {
        // A baseline holds recorded answers, not code: its model names are not models the app uses.
        result.baselines!.push(baseline);
        continue;
      }
    }
    const own = ownPackageName(file.rel, content);
    if (own) ownPackages.add(own);
    if (isDependencyFile(file.rel)) {
      // Only AI-related packages are kept; their lines are looked up here, once.
      for (const { needle, ...dep } of parseDependencies(file.rel, content)) {
        if (isAiRelated(dep.ecosystem, dep.name)) result.dependencies!.push({ ...dep, line: lineOf(content, needle) });
      }
    }
    if (file.lock) continue;

    const scanned = scanText(file.rel, content, file.kind, rules);
    result.filesScanned++;
    if (scanned.isAiFile && file.kind === "code") result.aiFiles++;
    result.callSites += scanned.callSites;
    result.findings.push(...scanned.findings);
    if (scanned.isAiFile) result.tokenTips.push(...findTokenTips(file.rel, content, file.kind));
    result.security.push(...findSecurityIssues(file.rel, content, file.kind));
    if (isMcpConfig(file.rel, content, file.kind)) result.mcpServers!.push(...parseMcpServers(file.rel, content));
  }
  // The repo's own workspace packages are not installed from a registry.
  result.dependencies = result.dependencies!.filter((d) => !(d.ecosystem === "npm" && ownPackages.has(d.name)));
  const npmrc = await readFile(join(root, ".npmrc"), "utf8").catch(() => "");
  const lockText = await readFile(join(root, LOCK_FILE), "utf8").catch(() => undefined);
  if (lockText !== undefined) result.lockText = lockText;
  const registries = privateRegistries(npmrc);
  for (const d of result.dependencies) {
    if (d.ecosystem === "npm" && (registries.all || registries.scopes.some((s) => d.name.startsWith(`${s}/`)))) d.privateIndex = true;
  }
  return result;
}
