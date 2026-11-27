/** The command palette's search: which items a query keeps, and in what
 * order. Each word of the query must start a word of the item's label or
 * keywords (`ins fr` finds `Inspect · Frames`); a label that starts with the
 * query comes first, then the items in the order they were given, so groups
 * stay together. */

export interface PaletteItem {
  id: string;
  /** What the row says. */
  label: string;
  /** The group it is listed under (`Screens`, `Runs`, …). */
  group: string;
  /** Other words that find it (a run's label, a project's folder). */
  keywords?: string;
}

function words(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((w) => w.length > 0);
}

function matches(item: PaletteItem, query: string[]): boolean {
  const pool = words(`${item.label} ${item.keywords ?? ""}`);
  return query.every((q) => pool.some((w) => w.startsWith(q)));
}

export function filterItems<T extends PaletteItem>(
  items: readonly T[],
  query: string,
): T[] {
  const q = words(query);
  if (q.length === 0) return [...items];
  const kept = items.filter((item) => matches(item, q));
  const lead = query.trim().toLowerCase();
  const starts = (item: T) => item.label.toLowerCase().startsWith(lead);
  return [...kept.filter(starts), ...kept.filter((item) => !starts(item))];
}
