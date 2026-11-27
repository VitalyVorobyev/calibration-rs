import { describe, expect, it } from "vitest";
import type { ResidualRecords, ResidualStats } from "../../api/generated/types";
import {
  equalAxis,
  recordRows,
  scatterGroups,
  scopeViews,
  statsRows,
} from "./residualModel";

/** Three records of view 0 (one diverged) and two of view 1. */
const RECORDS: ResidualRecords = {
  index: [0, 1, 2, 3, 4],
  view: [0, 0, 0, 1, 1],
  feature: [0, 1, 2, 0, 1],
  target_xyz_m: [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ],
  observed_px: [
    [10, 10],
    [20, 10],
    [30, 10],
    [10, 20],
    [20, 20],
  ],
  projected_px: [[9.9, 10.2], [20.3, 10], null, [10, 19.5], [20, 20]],
  residual_px: [[0.1, -0.2], [-0.3, 0], null, [0, 0.5], [0, 0]],
};

describe("the scope of views", () => {
  it("is every view, the selected one, or a chosen few", () => {
    expect(scopeViews({ kind: "all" }, 3)).toBeNull();
    expect(scopeViews({ kind: "view" }, 3)).toEqual([3]);
    expect(scopeViews({ kind: "subset", views: [4, 1] }, 3)).toEqual([1, 4]);
  });
  it("is every view when there is nothing to narrow to", () => {
    expect(scopeViews({ kind: "view" }, null)).toBeNull();
    expect(scopeViews({ kind: "subset", views: [] }, 2)).toBeNull();
  });
});

describe("recordRows", () => {
  it("is observed − projected as the library gives it, with the worst measured", () => {
    const { rows, diverged } = recordRows(RECORDS, new Set([1]), null);
    expect(diverged).toBe(1);
    expect(rows).toHaveLength(4);
    expect(rows[0]).toMatchObject({
      index: 0,
      view: 0,
      feature: 0,
      x: 10,
      y: 10,
      dx: 0.1,
      dy: -0.2,
    });
    expect(rows[1]).toMatchObject({ outlier: true });
    expect(rows[3]!.magnitude).toBe(0);
  });
  it("keeps the records of the chosen views", () => {
    const { rows, diverged } = recordRows(RECORDS, new Set(), [1]);
    expect(rows.map((r) => r.index)).toEqual([3, 4]);
    expect(diverged).toBe(0);
  });
});

describe("the scatter", () => {
  const { rows } = recordRows(RECORDS, new Set(), null);
  it("names a series per view while there are few, so colour is never the only key", () => {
    expect(scatterGroups(rows).map((g) => g.name)).toEqual(["view 0", "view 1"]);
  });
  it("is one series past six views", () => {
    const many = Array.from({ length: 7 }, (_, v) => ({ ...rows[0]!, view: v }));
    expect(scatterGroups(many).map((g) => g.name)).toEqual(["all features"]);
    expect(scatterGroups([])).toEqual([]);
  });
  it("has equal, symmetric axes over both components", () => {
    expect(equalAxis(rows)).toEqual([-0.5, 0.5]);
    expect(equalAxis([])).toEqual([-0.1, 0.1]);
  });
});

describe("statistics rows", () => {
  const stats: ResidualStats = {
    cameras: [
      {
        camera: 0,
        stats: { count: 9, mean: 0.1, rms: 0.2, max: 0.5 },
        unprojected: 1,
        views: 2,
      },
    ],
    views: [
      { view: 1, stats: { count: 4, mean: 0.1, rms: 0.3, max: 0.5 }, unprojected: 0 },
    ],
    cells: [
      {
        camera: 0,
        view: 1,
        stats: { count: 4, mean: 0.1, rms: 0.3, max: 0.5 },
        unprojected: 0,
      },
    ],
  };
  it("lists per camera, per view and per camera and view", () => {
    expect(statsRows(stats, "camera")[0]).toMatchObject({
      label: "camera 0",
      camera: 0,
      view: null,
      rms: 0.2,
      unprojected: 1,
    });
    expect(statsRows(stats, "view")[0]).toMatchObject({
      label: "view 1",
      camera: null,
      view: 1,
    });
    expect(statsRows(stats, "cell")[0]).toMatchObject({
      label: "camera 0 · view 1",
      camera: 0,
      view: 1,
    });
  });
});
