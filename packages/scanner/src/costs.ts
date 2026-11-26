// Cost Change Guard: announced price changes (introductory prices ending, scheduled increases) and reroutes
// billed at another model's rates, for the models a repository uses. Prices come from the catalog.
import { daysBetween, formatUsd, lookupPrice, lookupPriceChanges, priceOn, type Platform, type PriceChange, type TokenPrice } from "@ai-changeguard/catalog";

export { daysBetween, formatUsd };

export type CostKind = "intro-ends" | "price-change" | "reroute-billing";

export interface CostRow {
  /** The model ID in the code. */
  id: string;
  platform: Platform;
  kind: CostKind;
  /** First day the new price applies. */
  on: string;
  daysLeft: number;
  before: TokenPrice;
  /** Missing when the provider has not published the new price. */
  after?: TokenPrice;
  /** Set when the code's model is rerouted and billed as another model. */
  billedAs?: string;
  verified: boolean;
  note?: string;
  locations: string[];
}

/** Changes that took effect up to this many days ago are still reported, so a missed one is not lost. */
export const RECENT_DAYS = 30;

export function dayBefore(date: string): string {
  return new Date(Date.parse(`${date}T00:00:00Z`) - 86_400_000).toISOString().slice(0, 10);
}

export interface CostModel {
  id: string;
  platform: Platform;
  /** Reroute date and target, when the provider reroutes this model. */
  reroute?: { on: string; to: string };
  locations: string[];
}

/** A row for one announced change of `priced` (the code's model, or the model it is billed as). */
function changeRow(m: CostModel, change: PriceChange, priced: string, today: string): CostRow | undefined {
  const before = priceOn(priced, m.platform, dayBefore(change.on));
  if (!before) return undefined;
  const row: CostRow = {
    id: m.id,
    platform: m.platform,
    kind: change.kind,
    on: change.on,
    daysLeft: daysBetween(today, change.on),
    before,
    verified: change.verified,
    locations: m.locations,
  };
  if (priced !== m.id) row.billedAs = priced;
  if (change.input !== undefined && change.output !== undefined) row.after = { input: change.input, output: change.output };
  if (change.note) row.note = change.note;
  return row;
}

export function buildCostRows(models: CostModel[], today: string): CostRow[] {
  const rows: CostRow[] = [];
  const keep = (on: string) => daysBetween(today, on) >= -RECENT_DAYS;
  const push = (row: CostRow | undefined) => row && rows.push(row);

  for (const m of models) {
    // Once rerouted, the model's own later price changes no longer apply: its calls are billed as the target.
    const ownUntil = m.reroute?.on;
    for (const change of lookupPriceChanges(m.id, m.platform)) {
      if (keep(change.on) && (!ownUntil || change.on < ownUntil)) push(changeRow(m, change, m.id, today));
    }

    const target = m.reroute ? lookupPrice(m.reroute.to, m.platform) : undefined;
    if (!m.reroute || !target) continue;
    const { on, to } = m.reroute;
    const before = priceOn(m.id, m.platform, dayBefore(on));
    const after = priceOn(to, m.platform, on);
    if (keep(on) && before && after && (before.input !== after.input || before.output !== after.output)) {
      rows.push({
        id: m.id,
        platform: m.platform,
        kind: "reroute-billing",
        on,
        daysLeft: daysBetween(today, on),
        before,
        after,
        billedAs: to,
        verified: target.verified,
        note: `Requests are answered by ${to} and billed at its rates.`,
        locations: m.locations,
      });
    }
    // After the reroute, the target's own price changes reach this code too.
    for (const change of lookupPriceChanges(to, m.platform)) {
      if (change.on > on && keep(change.on)) push(changeRow(m, change, to, today));
    }
  }
  return rows.sort((a, b) => a.on.localeCompare(b.on) || a.id.localeCompare(b.id));
}

export function formatPrice(price: TokenPrice): string {
  return `${formatUsd(price.input)} input, ${formatUsd(price.output)} output per 1M tokens`;
}

/** "$1.25 → $0.75 input, $10.00 → $3.75 output per 1M tokens". */
export function formatPriceMove(before: TokenPrice, after: TokenPrice): string {
  return `${formatUsd(before.input)} → ${formatUsd(after.input)} input, ${formatUsd(before.output)} → ${formatUsd(after.output)} output per 1M tokens`;
}

function percent(from: number, to: number): number {
  return from === 0 ? 0 : Math.round(((to - from) / from) * 100);
}

/** "+100%", or "input +100%, output +50%" when the two differ. */
export function priceChangeText(before: TokenPrice, after: TokenPrice): string {
  const i = percent(before.input, after.input);
  const o = percent(before.output, after.output);
  const sign = (n: number) => (n > 0 ? `+${n}%` : `${n}%`);
  return i === o ? sign(i) : `input ${sign(i)}, output ${sign(o)}`;
}

/** True when the change raises the bill (or its new price is unknown): either price goes up, neither goes down. */
export function isPriceRise(row: Pick<CostRow, "before" | "after">): boolean {
  if (!row.after) return true;
  const up = row.after.input > row.before.input || row.after.output > row.before.output;
  const down = row.after.input < row.before.input || row.after.output < row.before.output;
  return up && !down;
}

/** One line for a person: what changes, when, by how much. */
export function describeCost(row: CostRow): string {
  const subject = row.billedAs && row.kind !== "reroute-billing" ? `${row.id} (served as ${row.billedAs})` : row.id;
  const when = row.daysLeft > 0 ? `on ${row.on} (in ${row.daysLeft} day${row.daysLeft === 1 ? "" : "s"})` : `since ${row.on}`;
  if (row.kind === "reroute-billing" && row.after) {
    return `${subject} is billed as ${row.billedAs} ${when}: ${formatPriceMove(row.before, row.after)} (${priceChangeText(row.before, row.after)})`;
  }
  if (!row.after) {
    const what = row.kind === "intro-ends" ? `introductory price ${row.daysLeft > 0 ? "ends" : "ended"}` : `price ${row.daysLeft > 0 ? "changes" : "changed"}`;
    return `${subject}: ${what} ${when}; the new price is not published yet (${row.daysLeft > 0 ? "now" : "was"} ${formatPrice(row.before)})`;
  }
  return `${subject}: price ${isPriceRise(row) ? "rises" : "changes"} ${when}: ${formatPriceMove(row.before, row.after)} (${priceChangeText(row.before, row.after)})`;
}
