/** The per-feature residuals of a camera as rows, scatter series and
 * statistics rows, and the scope (which views) the residual map and the
 * records share. A residual is `observed − projected`, in pixels. Pure. */
import type {
  CameraStats,
  CellStats,
  ReprojectionStats,
  ResidualRecords,
  ResidualStats,
  ViewStats,
} from "../../api/generated/types";

/** Which views the map and the records are over. */
export type ViewScope =
  { kind: "all" } | { kind: "view" } | { kind: "subset"; views: number[] };

/** The view indices a scope names, as `residual_map`'s `views` option takes
 * them: `null` for every view. `selected` is the selection's view; with none
 * the scope "view" is every view (there is nothing to narrow to). */
export function scopeViews(scope: ViewScope, selected: number | null): number[] | null {
  if (scope.kind === "all") return null;
  if (scope.kind === "view") return selected === null ? null : [selected];
  return scope.views.length === 0 ? null : [...scope.views].sort((a, b) => a - b);
}

/** One record of the camera. */
export interface RecordRow {
  /** Its index in the export's records (what the map's outliers name). */
  index: number;
  view: number;
  feature: number;
  /** The observed pixel. */
  x: number;
  y: number;
  /** `observed − projected`. */
  dx: number;
  dy: number;
  magnitude: number;
  outlier: boolean;
}

/** The records that have a projection as rows (of `views` when given), and
 * how many diverged. */
export function recordRows(
  records: ResidualRecords,
  outliers: ReadonlySet<number>,
  views: readonly number[] | null,
): { rows: RecordRow[]; diverged: number } {
  const rows: RecordRow[] = [];
  let diverged = 0;
  const keep = views === null ? null : new Set(views);
  for (let i = 0; i < records.index.length; i++) {
    const view = records.view[i]!;
    if (keep && !keep.has(view)) continue;
    const r = records.residual_px[i];
    if (!r) {
      diverged += 1;
      continue;
    }
    const [x, y] = records.observed_px[i]!;
    rows.push({
      index: records.index[i]!,
      view,
      feature: records.feature[i]!,
      x,
      y,
      dx: r[0],
      dy: r[1],
      magnitude: Math.hypot(r[0], r[1]),
      outlier: outliers.has(records.index[i]!),
    });
  }
  return { rows, diverged };
}

/** The most series the scatter tells apart by name. */
export const MAX_VIEW_SERIES = 6;

/** The rows grouped into the scatter's series: one per view while there are
 * at most {@link MAX_VIEW_SERIES} views, else one for all. */
export function scatterGroups(
  rows: readonly RecordRow[],
): { name: string; rows: RecordRow[] }[] {
  const views = [...new Set(rows.map((r) => r.view))].sort((a, b) => a - b);
  if (views.length > MAX_VIEW_SERIES || views.length === 0) {
    return rows.length === 0 ? [] : [{ name: "all features", rows: [...rows] }];
  }
  return views.map((v) => ({
    name: `view ${v}`,
    rows: rows.filter((r) => r.view === v),
  }));
}

/** An equal, symmetric axis over every component: the largest magnitude
 * rounded up to 1, 2 or 5 × 10ⁿ (0.1 for no data). */
export function equalAxis(rows: readonly RecordRow[]): [number, number] {
  let m = 0;
  for (const r of rows) m = Math.max(m, Math.abs(r.dx), Math.abs(r.dy));
  if (!(m > 0)) return [-0.1, 0.1];
  const exponent = Math.floor(Math.log10(m));
  const base = 10 ** exponent;
  const d =
    [1, 2, 5, 10].map((k) => k * base).find((v) => v >= m * (1 - 1e-12)) ?? 10 * base;
  return [-d, d];
}

export type StatsScope = "camera" | "view" | "cell";

/** One row of the statistics table. */
export interface StatsRow {
  key: string;
  label: string;
  camera: number | null;
  view: number | null;
  count: number;
  mean: number;
  rms: number;
  max: number;
  /** `null` when the back end gives none (no error lengths). */
  median: number | null;
  p95: number | null;
  unprojected: number;
}

function row(
  key: string,
  label: string,
  camera: number | null,
  view: number | null,
  stats: ReprojectionStats,
  unprojected: number,
): StatsRow {
  return {
    key,
    label,
    camera,
    view,
    count: stats.count,
    mean: stats.mean,
    rms: stats.rms,
    max: stats.max,
    median: stats.median ?? null,
    p95: stats.p95 ?? null,
    unprojected,
  };
}

/** The rows of `scope` from the run's statistics. */
export function statsRows(stats: ResidualStats, scope: StatsScope): StatsRow[] {
  if (scope === "camera") {
    return stats.cameras.map((c: CameraStats) =>
      row(`c${c.camera}`, `camera ${c.camera}`, c.camera, null, c.stats, c.unprojected),
    );
  }
  if (scope === "view") {
    return stats.views.map((v: ViewStats) =>
      row(`v${v.view}`, `view ${v.view}`, null, v.view, v.stats, v.unprojected),
    );
  }
  return stats.cells.map((c: CellStats) =>
    row(
      `c${c.camera}v${c.view}`,
      `camera ${c.camera} · view ${c.view}`,
      c.camera,
      c.view,
      c.stats,
      c.unprojected,
    ),
  );
}
