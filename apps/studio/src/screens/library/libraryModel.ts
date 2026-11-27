/** The Library's pure rules over a `scan_library` reply: each project as a
 * row (its latest run's verdict and RMS), the rows grouped `product →
 * device → project`, the filter the search box and chips make, and the
 * text a search matches. Nothing here reads the back end or draws. */
import type {
  DeviceIdentity,
  ProblemKind,
  ProjectEntry,
  RunInfo,
  Verdict,
} from "../../api/generated/types";
import { runRms, runVerdict, runsNewestFirst } from "../../project/projectModel";

/** The group of a device with no product, and of a project with no device. */
export const UNSPECIFIED_TYPE = "(unspecified type)";
export const NO_DEVICE = "(no device)";

/** One project of the scan as the table lists it. */
export interface ProjectRow {
  entry: ProjectEntry;
  /** The run the row reports: the newest starred run, else the newest. */
  latest: RunInfo | null;
  verdict: Verdict | null;
  /** Pooled reprojection RMS of `latest`, pixels. */
  rms: number | null;
  /** Views that entered `latest` and views its dataset formed. */
  views: { usable: number; total: number } | null;
  /** When the newest run was written, seconds since the epoch. */
  updated: number | null;
}

/** The newest run (by id) that is starred, else the newest at all. */
export function latestRun(runs: readonly RunInfo[]): RunInfo | null {
  const newest = runsNewestFirst(runs);
  return newest.find((r) => r.notes.starred) ?? newest[0] ?? null;
}

function runTime(run: RunInfo): number | null {
  return run.state.kind === "recorded" ? run.state.timestamp : null;
}

function runViews(run: RunInfo): ProjectRow["views"] {
  const state = run.state;
  if (state.kind === "recorded" && state.source.kind === "calibrated") {
    return { usable: state.source.usable_views, total: state.source.total_views };
  }
  if (state.kind === "unrecorded" && state.usable_views != null) {
    return { usable: state.usable_views, total: state.total_views ?? state.usable_views };
  }
  return null;
}

export function projectRow(entry: ProjectEntry): ProjectRow {
  const latest = latestRun(entry.runs);
  const times = entry.runs.map(runTime).filter((t): t is number => t !== null);
  return {
    entry,
    latest,
    verdict: latest ? runVerdict(latest) : null,
    rms: latest ? runRms(latest) : null,
    views: latest ? runViews(latest) : null,
    updated: times.length > 0 ? Math.max(...times) : null,
  };
}

/** The most severe of `verdicts` (`fail` over `warn` over `pass`); `null`
 * when none has one. */
export function worstVerdict(verdicts: readonly (Verdict | null)[]): Verdict | null {
  let worst: Verdict | null = null;
  const rank: Record<Verdict, number> = { pass: 0, warn: 1, fail: 2 };
  for (const v of verdicts) {
    if (v !== null && (worst === null || rank[v] > rank[worst])) worst = v;
  }
  return worst;
}

/** What the tree and table are filtered by. */
export interface LibraryFilter {
  /** Free text, matched against name, product, id, label, workflow and the
   * notes of the runs; empty matches everything. */
  query: string;
  /** Verdicts to keep; empty keeps all. `"none"` is a project with no
   * verdict (no run, or none judged). */
  verdicts: ReadonlySet<Verdict | "none">;
  /** A workflow to keep; `null` keeps all. */
  topology: ProblemKind | null;
}

export const NO_FILTER: LibraryFilter = {
  query: "",
  verdicts: new Set(),
  topology: null,
};

export function isFiltering(filter: LibraryFilter): boolean {
  return (
    filter.query.trim() !== "" || filter.verdicts.size > 0 || filter.topology !== null
  );
}

/** The fields of a row a search looks in, each lower-cased, for
 * `HighlightedText` to mark the same words in. */
export function searchFields(row: ProjectRow): string[] {
  const { entry } = row;
  const notes = entry.runs.flatMap((r) => [r.notes.label, r.notes.notes]);
  return [
    entry.name,
    entry.dir,
    entry.device?.product,
    entry.device?.id,
    entry.device?.label,
    entry.topology,
    ...notes,
  ]
    .filter((s): s is string => typeof s === "string" && s !== "")
    .map((s) => s.toLowerCase());
}

/** Whether `row` passes `filter`: every word of the query is in some field. */
export function matches(row: ProjectRow, filter: LibraryFilter): boolean {
  if (filter.topology !== null && row.entry.topology !== filter.topology) return false;
  if (filter.verdicts.size > 0 && !filter.verdicts.has(row.verdict ?? "none")) {
    return false;
  }
  const words = filter.query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return true;
  const fields = searchFields(row);
  return words.every((w) => fields.some((f) => f.includes(w)));
}

/** A device of the Library: its identity and its projects. */
export interface DeviceGroup {
  /** Tree id. */
  id: string;
  device: DeviceIdentity;
  rows: ProjectRow[];
}

/** A product (device type) of the Library. */
export interface ProductGroup {
  id: string;
  /** The product's name, or `UNSPECIFIED_TYPE`. */
  label: string;
  devices: DeviceGroup[];
}

export interface LibraryGroups {
  products: ProductGroup[];
  /** Projects that name no device, last in the tree. */
  ungrouped: ProjectRow[];
}

const byText = (a: string, b: string) =>
  a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });

/** The rows grouped by device product, then device id; projects sorted by
 * name; products and devices by name, with an unspecified type last. */
export function groupLibrary(rows: readonly ProjectRow[]): LibraryGroups {
  const products = new Map<string, ProductGroup>();
  const ungrouped: ProjectRow[] = [];
  for (const row of rows) {
    const device = row.entry.device;
    if (!device) {
      ungrouped.push(row);
      continue;
    }
    const product = device.product?.trim() || UNSPECIFIED_TYPE;
    let group = products.get(product);
    if (!group) {
      group = { id: `product:${product}`, label: product, devices: [] };
      products.set(product, group);
    }
    const id = `device:${product}\u0000${device.id}`;
    let dev = group.devices.find((d) => d.id === id);
    if (!dev) {
      dev = { id, device, rows: [] };
      group.devices.push(dev);
    }
    dev.rows.push(row);
  }
  const rank = (p: ProductGroup) => (p.label === UNSPECIFIED_TYPE ? 1 : 0);
  const sorted = [...products.values()].sort(
    (a, b) => rank(a) - rank(b) || byText(a.label, b.label),
  );
  for (const product of sorted) {
    product.devices.sort((a, b) => byText(a.device.id, b.device.id));
    for (const dev of product.devices) {
      dev.rows.sort((a, b) => byText(a.entry.name, b.entry.name));
    }
  }
  ungrouped.sort((a, b) => byText(a.entry.name, b.entry.name));
  return { products: sorted, ungrouped };
}

export const NO_DEVICE_ID = "ungrouped";
export const projectNodeId = (dir: string) => `project:${dir}`;

/** The rows a tree node stands for: a project, a device, a product, the
 * ungrouped group, or the whole library (`null`). */
export function rowsOf(groups: LibraryGroups, nodeId: string | null): ProjectRow[] {
  const all = [
    ...groups.products.flatMap((p) => p.devices.flatMap((d) => d.rows)),
    ...groups.ungrouped,
  ];
  if (nodeId === null) return all;
  if (nodeId === NO_DEVICE_ID) return groups.ungrouped;
  if (nodeId.startsWith("project:")) {
    return all.filter((r) => projectNodeId(r.entry.dir) === nodeId);
  }
  for (const product of groups.products) {
    if (product.id === nodeId) return product.devices.flatMap((d) => d.rows);
    const device = product.devices.find((d) => d.id === nodeId);
    if (device) return device.rows;
  }
  return [];
}

/** The device whose projects `nodeId` is, when it is a device or a project of one. */
export function deviceOf(
  groups: LibraryGroups,
  nodeId: string | null,
): DeviceGroup | null {
  if (nodeId === null) return null;
  for (const product of groups.products) {
    for (const dev of product.devices) {
      if (dev.id === nodeId) return dev;
      if (dev.rows.some((r) => projectNodeId(r.entry.dir) === nodeId)) return dev;
    }
  }
  return null;
}

/** Where a project opens: Inspect when it has a run to read, else
 * Calibrate. */
export function openScreen(row: ProjectRow): "inspect" | "calibrate" {
  return row.entry.runs.some((r) => r.state.kind !== "unreadable")
    ? "inspect"
    : "calibrate";
}

/** The location of a project relative to the library roots it lies under;
 * the path itself when under none. */
export function relativeToRoots(dir: string, roots: readonly string[]): string {
  const norm = (p: string) => p.replace(/\\/g, "/").replace(/\/+$/, "");
  const d = norm(dir);
  const best = roots
    .map(norm)
    .filter((r) => d === r || d.startsWith(`${r}/`))
    .sort((a, b) => b.length - a.length)[0];
  if (best === undefined) return d;
  const rest = d.slice(best.length).replace(/^\//, "");
  return rest === "" ? "." : rest;
}

/** The node whose table lists `nodeId`: a project's device (so its
 * siblings show beside it), or the ungrouped group; any other node is its
 * own. */
export function containerOf(groups: LibraryGroups, nodeId: string | null): string | null {
  if (nodeId === null || !nodeId.startsWith("project:")) return nodeId;
  const device = deviceOf(groups, nodeId);
  if (device) return device.id;
  return groups.ungrouped.some((r) => projectNodeId(r.entry.dir) === nodeId)
    ? NO_DEVICE_ID
    : null;
}

/** The ranges of `text` the words of `query` match, for `HighlightedText`. */
export function querySpans(
  text: string,
  query: string,
): { start: number; end: number; kind: "protected-term" }[] {
  const lower = text.toLowerCase();
  const spans: { start: number; end: number; kind: "protected-term" }[] = [];
  for (const word of query.toLowerCase().split(/\s+/).filter(Boolean)) {
    for (
      let at = lower.indexOf(word);
      at >= 0;
      at = lower.indexOf(word, at + word.length)
    ) {
      spans.push({ start: at, end: at + word.length, kind: "protected-term" });
    }
  }
  return spans;
}

/** The runs, projects and failures a header line counts. */
export function countLine(rows: readonly ProjectRow[]): string {
  const runs = rows.reduce((n, r) => n + r.entry.runs.length, 0);
  const p = rows.length;
  return `${runs} ${runs === 1 ? "run" : "runs"} · ${p} ${p === 1 ? "project" : "projects"}`;
}

/** `2026-11-02` of a time in seconds since the epoch, in the local zone. */
export function dateOf(seconds: number): string {
  const d = new Date(seconds * 1000);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}
