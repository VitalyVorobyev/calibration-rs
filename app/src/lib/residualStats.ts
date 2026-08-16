/** Pure residual-aggregation helpers for the Diagnose stats panels
 *. Both operate on the `per_feature_residuals.target` array every
 * `*Export` carries (ADR 0012); neither touches React or Tauri, so they
 * unit-test in a plain Node environment.
 *
 * - {@link computePoseResidualStats} rolls the per-corner reprojection
 *   errors up to one row per pose (across all cameras) — the multi-pose
 *   stats table.
 * - {@link computeCameraPoseMatrix} keeps the `(camera, pose)` grain — the
 *   cross-camera residual matrix. */
import type { TargetFeatureResidual } from "../types";

/** Per-pose reprojection summary (aggregated over every camera that saw
 * the pose). */
export interface PoseResidualStat {
  pose: number;
  /** Corners with a finite reprojection error. */
  count: number;
  /** Corners whose projection diverged (`error_px` null). */
  diverged: number;
  /** Mean error in pixels (0 when `count === 0`). */
  mean: number;
  /** Median error in pixels (0 when `count === 0`). */
  median: number;
  /** Max error in pixels (0 when `count === 0`). */
  max: number;
}

/** Finite `error_px` values only. */
function validErrors(residuals: TargetFeatureResidual[]): number[] {
  return residuals
    .map((r) => r.error_px)
    .filter((e): e is number => typeof e === "number" && Number.isFinite(e));
}

function median(sortedAsc: number[]): number {
  const n = sortedAsc.length;
  if (n === 0) return 0;
  const mid = n >> 1;
  return n % 2 === 1 ? sortedAsc[mid] : (sortedAsc[mid - 1] + sortedAsc[mid]) / 2;
}

function summarize(errors: number[]): { mean: number; median: number; max: number } {
  if (errors.length === 0) return { mean: 0, median: 0, max: 0 };
  const sorted = [...errors].sort((a, b) => a - b);
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
  return { mean, median: median(sorted), max: sorted[sorted.length - 1] };
}

/** One row per pose, sorted by ascending pose index. Poses are taken
 * from the residual records themselves, so a pose with only diverged
 * corners still appears (with `count === 0`). */
export function computePoseResidualStats(
  residuals: TargetFeatureResidual[],
): PoseResidualStat[] {
  const byPose = new Map<number, TargetFeatureResidual[]>();
  for (const r of residuals) {
    const arr = byPose.get(r.pose);
    if (arr) arr.push(r);
    else byPose.set(r.pose, [r]);
  }
  const poses = [...byPose.keys()].sort((a, b) => a - b);
  return poses.map((pose) => {
    const recs = byPose.get(pose)!;
    const errors = validErrors(recs);
    const { mean, median, max } = summarize(errors);
    return {
      pose,
      count: errors.length,
      diverged: recs.length - errors.length,
      mean,
      median,
      max,
    };
  });
}

/** One cell of the cross-camera residual matrix. */
export interface CameraPoseCell {
  camera: number;
  pose: number;
  /** Corners with a finite error backing this cell's mean. */
  count: number;
  /** Mean error in pixels; `null` when the `(camera, pose)` slot has no
   * finite residual (either unobserved or fully diverged), so the UI can
   * render an empty cell rather than a misleading `0`. */
  mean: number | null;
}

export interface CameraPoseMatrix {
  /** Distinct camera indices, ascending. */
  cameras: number[];
  /** Distinct pose indices, ascending. */
  poses: number[];
  /** `camera → pose → cell` for O(1) lookup while rendering the grid. */
  cells: Map<number, Map<number, CameraPoseCell>>;
}

/** Build the `(camera, pose)` mean-residual matrix. Rows are cameras,
 * columns are poses; every observed slot gets a cell (a slot with only
 * diverged corners gets `mean: null`, `count: 0`). */
export function computeCameraPoseMatrix(
  residuals: TargetFeatureResidual[],
): CameraPoseMatrix {
  // Accumulate sum + count per (camera, pose) in one pass.
  const acc = new Map<number, Map<number, { sum: number; count: number }>>();
  const cameraSet = new Set<number>();
  const poseSet = new Set<number>();
  for (const r of residuals) {
    cameraSet.add(r.camera);
    poseSet.add(r.pose);
    let row = acc.get(r.camera);
    if (!row) {
      row = new Map();
      acc.set(r.camera, row);
    }
    let slot = row.get(r.pose);
    if (!slot) {
      slot = { sum: 0, count: 0 };
      row.set(r.pose, slot);
    }
    if (typeof r.error_px === "number" && Number.isFinite(r.error_px)) {
      slot.sum += r.error_px;
      slot.count += 1;
    }
  }

  const cells = new Map<number, Map<number, CameraPoseCell>>();
  for (const [camera, row] of acc) {
    const outRow = new Map<number, CameraPoseCell>();
    for (const [pose, { sum, count }] of row) {
      outRow.set(pose, {
        camera,
        pose,
        count,
        mean: count > 0 ? sum / count : null,
      });
    }
    cells.set(camera, outRow);
  }

  return {
    cameras: [...cameraSet].sort((a, b) => a - b),
    poses: [...poseSet].sort((a, b) => a - b),
    cells,
  };
}
