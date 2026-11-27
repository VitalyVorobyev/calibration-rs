import { describe, expect, it } from "vitest";
import { decodePlane } from "@vitavision/stage2d";
import { quantile } from "../../../lib/colorRange";
import {
  GRID,
  RESIDUAL_CHANNELS,
  distortionPlane,
  plane,
  residualPlane,
  vam1,
} from "../../../test/planeFixtures";
import {
  autoDomain,
  channelIndex,
  clipsHigh,
  maskedBelow,
  meanOf,
  nodeAt,
  nodeCentre,
  peakOf,
  planeRect,
  quiver,
  quiverEvery,
  readoutAt,
  readoutNumber,
  shareAtLeast,
  singleChannel,
  unitLabel,
} from "./mapModel";

describe("decoding", () => {
  it("reads the planes the back end writes: one decode, values as sent", () => {
    const p = decodePlane(vam1(distortionPlane()));
    expect([p.width, p.height, p.channels]).toEqual([4, 3, 3]);
    expect(p.values[1]).toBe(1);
  });
});

describe("domains", () => {
  const p = plane(2, 2, 2, (c, col, row) =>
    c === 0 ? col + 2 * row : (col + 2 * row) * -1,
  );
  it("runs a magnitude from zero to a round top", () => {
    expect(autoDomain(p, 0, false)).toEqual([0, 3]);
  });
  it("fits the data and clips only a real outlier", () => {
    const flat = plane(10, 10, 1, (_c, col, row) => 0.01 + (col + 10 * row) * 0.0002);
    const [, top] = autoDomain(flat, 0, false)!;
    expect(top).toBeGreaterThanOrEqual(0.0298);
    expect(top).toBeLessThan(0.032);
    expect(clipsHigh(flat, 0, [0, top])).toBe(false);
    const spiky = plane(40, 25, 1, (_c, col, row) => (col + 40 * row === 999 ? 5 : 0.01));
    const [, spikyTop] = autoDomain(spiky, 0, false)!;
    expect(spikyTop).toBeLessThan(0.1);
    expect(clipsHigh(spiky, 0, [0, spikyTop])).toBe(true);
  });
  it("centres a signed quantity on zero", () => {
    expect(autoDomain(p, 1, true)).toEqual([-3, 3]);
  });
  it("has none for a channel without a finite value", () => {
    const empty = plane(2, 2, 1, () => Number.NaN);
    expect(autoDomain(empty, 0, false)).toBeNull();
    expect(autoDomain(p, 7, false)).toBeNull();
  });
  it("says when the scale clips", () => {
    expect(clipsHigh(p, 0, [0, 2])).toBe(true);
    expect(clipsHigh(p, 0, [0, 3])).toBe(false);
  });
  it("takes nearest-rank quantiles", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2);
    expect(Number.isNaN(quantile([], 0.5))).toBe(true);
  });
});

describe("the grid", () => {
  it("lays the blocks edge to edge from the image's corner", () => {
    expect(planeRect(GRID)).toEqual({ x: -0.5, y: -0.5, width: 40, height: 30 });
  });
  it("puts a node at the centre of its block, a partial block on its pixels", () => {
    expect(nodeCentre(GRID, 0, 0)).toEqual([4.5, 4.5]);
    expect(nodeCentre(GRID, 3, 2)).toEqual([34.5, 24.5]);
    const partial = {
      columns: 2,
      rows: 1,
      stride: 10,
      image_size: [15, 10] as [number, number],
    };
    expect(nodeCentre(partial, 1, 0)[0]).toBe(12);
  });
  it("finds the node under a position, none outside the image", () => {
    expect(nodeAt(GRID, 0, 0)).toEqual({ column: 0, row: 0 });
    expect(nodeAt(GRID, 39, 29)).toEqual({ column: 3, row: 2 });
    expect(nodeAt(GRID, 14, 5)).toEqual({ column: 1, row: 0 });
    expect(nodeAt(GRID, -1, 5)).toBeNull();
    expect(nodeAt(GRID, 40, 5)).toBeNull();
  });
});

describe("readoutAt", () => {
  it("reads every channel of the node under the pointer", () => {
    const r = readoutAt(distortionPlane(), GRID, 25, 15)!;
    expect([r.column, r.row]).toEqual([2, 1]);
    expect(r.values).toEqual([2, 1, Math.hypot(2, 1)].map(Math.fround));
  });
  it("is empty outside the image", () => {
    expect(readoutAt(distortionPlane(), GRID, 100, 5)).toBeNull();
  });
  it("writes a missing value as no data, never as zero", () => {
    expect(readoutNumber(Number.NaN)).toBe("no data");
    expect(readoutNumber(undefined)).toBe("no data");
    expect(readoutNumber(0)).toBe("0");
    expect(readoutNumber(-0.012345)).toBe("−0.0123");
    expect(readoutNumber(1e-5)).toBe("1.00e-5");
    expect(unitLabel("chi")).toBe("χ");
    expect(unitLabel("px")).toBe("px");
  });
});

describe("the residual plane", () => {
  const p = residualPlane();
  const idx = (name: string) => channelIndex(RESIDUAL_CHANNELS, name);

  it("leaves unsupported nodes empty", () => {
    expect(finite(singleChannel(p, idx("mean_magnitude")))).toBe(9);
  });

  it("keeps a value only where the test passes", () => {
    const bright = maskedBelow(p, idx("mean_magnitude"), idx("significance"), 3.03);
    expect(bright.channels).toBe(1);
    expect(finite(bright)).toBe(3);
  });

  it("peaks where it is largest, and says where", () => {
    const peak = peakOf(p, GRID, idx("mean_magnitude"))!;
    expect(peak.value).toBeCloseTo(Math.hypot(0.3, 0.05));
    expect(peak.x).toBe(24.5);
  });

  it("counts the share of the testable area that passes the test", () => {
    const share = shareAtLeast(p, idx("mean_magnitude"), idx("significance"), 3.03)!;
    expect(share.tested).toBe(9);
    expect(share.share).toBeCloseTo(3 / 9);
    expect(meanOf(p, idx("rms"))).toBeCloseTo(0.3);
  });

  it("draws arrows at every n-th node with a value, filtered by the test", () => {
    const all = quiver(p, GRID, idx("mean_dx"), idx("mean_dy"), 1);
    expect(all.ids).toHaveLength(9);
    expect(all.vectors.slice(0, 2)).toEqual([Math.fround(0.1), -Math.fround(0.05)]);
    const kept = quiver(p, GRID, 0, 1, 1, {
      channel: idx("significance"),
      minimum: 3.03,
    });
    expect(kept.ids).toHaveLength(3);
    const faint = quiver(p, GRID, 0, 1, 1, {
      channel: idx("significance"),
      minimum: 3.03,
      below: true,
    });
    expect(faint.ids).toHaveLength(6);
    expect(quiver(p, GRID, 0, 1, 2).ids).toHaveLength(4);
    expect(quiver(p, GRID, -1, 1, 1).ids).toHaveLength(0);
    expect(quiverEvery(64)).toBe(4);
    expect(quiverEvery(4)).toBe(1);
  });
});

function finite(p: { values: Float32Array }): number {
  return Array.from(p.values).filter(Number.isFinite).length;
}
