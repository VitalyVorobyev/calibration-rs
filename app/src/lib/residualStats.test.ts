import { describe, expect, it } from "vitest";
import type { TargetFeatureResidual } from "../types";
import { computeCameraPoseMatrix, computePoseResidualStats } from "./residualStats";

/** Compact residual factory — only the fields the aggregators read. */
function res(
  pose: number,
  camera: number,
  error_px: number | null,
): TargetFeatureResidual {
  return {
    pose,
    camera,
    feature: 0,
    target_xyz_m: [0, 0, 0],
    observed_px: [0, 0],
    projected_px: error_px == null ? null : [0, 0],
    error_px,
  };
}

describe("computePoseResidualStats", () => {
  it("returns an empty array for no residuals", () => {
    expect(computePoseResidualStats([])).toEqual([]);
  });

  it("aggregates per pose across cameras and sorts by pose", () => {
    const stats = computePoseResidualStats([
      res(1, 0, 4),
      res(1, 1, 2),
      res(0, 0, 1),
      res(0, 0, 3),
    ]);
    expect(stats.map((s) => s.pose)).toEqual([0, 1]);
    // pose 0: [1, 3] → mean 2, median 2, max 3
    expect(stats[0]).toMatchObject({
      pose: 0,
      count: 2,
      diverged: 0,
      mean: 2,
      median: 2,
      max: 3,
    });
    // pose 1: [2, 4] across cam0+cam1 → mean 3, median 3, max 4
    expect(stats[1]).toMatchObject({
      pose: 1,
      count: 2,
      diverged: 0,
      mean: 3,
      median: 3,
      max: 4,
    });
  });

  it("computes an odd-length median as the middle value", () => {
    const stats = computePoseResidualStats([res(0, 0, 5), res(0, 0, 1), res(0, 0, 3)]);
    expect(stats[0]).toMatchObject({ count: 3, median: 3, mean: 3, max: 5 });
  });

  it("counts diverged corners and excludes them from mean/median/max", () => {
    const stats = computePoseResidualStats([res(0, 0, 2), res(0, 0, null), res(0, 0, 4)]);
    expect(stats[0]).toMatchObject({ count: 2, diverged: 1, mean: 3, median: 3, max: 4 });
  });

  it("keeps a fully-diverged pose with zeroed stats", () => {
    const stats = computePoseResidualStats([res(7, 0, null), res(7, 0, null)]);
    expect(stats[0]).toMatchObject({
      pose: 7,
      count: 0,
      diverged: 2,
      mean: 0,
      median: 0,
      max: 0,
    });
  });
});

describe("computeCameraPoseMatrix", () => {
  it("returns empty axes for no residuals", () => {
    const m = computeCameraPoseMatrix([]);
    expect(m.cameras).toEqual([]);
    expect(m.poses).toEqual([]);
    expect(m.cells.size).toBe(0);
  });

  it("builds sorted axes and per-slot means", () => {
    const m = computeCameraPoseMatrix([
      res(0, 1, 2),
      res(0, 1, 4),
      res(1, 0, 3),
      res(0, 0, 1),
    ]);
    expect(m.cameras).toEqual([0, 1]);
    expect(m.poses).toEqual([0, 1]);
    expect(m.cells.get(1)?.get(0)?.mean).toBe(3);
    // cam1 pose0 mean over [2, 4] = 3
    expect(m.cells.get(1)?.get(0)).toMatchObject({
      camera: 1,
      pose: 0,
      count: 2,
      mean: 3,
    });
    expect(m.cells.get(0)?.get(0)?.mean).toBe(1);
    // cam0 observed pose1 (err 3), so that slot exists...
    expect(m.cells.get(0)?.get(1)?.mean).toBe(3);
    // ...but cam1 never saw pose1 → no cell.
    expect(m.cells.get(1)?.get(1)).toBeUndefined();
  });

  it("marks a fully-diverged slot with a null mean but keeps the cell", () => {
    const m = computeCameraPoseMatrix([res(2, 3, null), res(2, 3, null)]);
    const cell = m.cells.get(3)?.get(2);
    expect(cell).toMatchObject({ camera: 3, pose: 2, count: 0, mean: null });
  });
});
