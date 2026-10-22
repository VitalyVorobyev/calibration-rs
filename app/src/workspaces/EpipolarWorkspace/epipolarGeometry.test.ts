import { describe, expect, it } from "vitest";
import type { TargetFeatureResidual } from "../../types";
import {
  distanceToPolyline,
  featureDots,
  ghostDistanceTone,
  snapToFeature,
} from "./epipolarGeometry";

const feature = (id: number, x: number, y: number) =>
  ({ pose: 0, camera: 0, feature: id, observed_px: [x, y] }) as TargetFeatureResidual;

describe("snapToFeature", () => {
  const features = [feature(1, 10, 10), feature(2, 13, 10)];

  it("snaps to the nearest feature within the radius", () => {
    expect(snapToFeature([12.6, 10], features)).toEqual({ px: [13, 10], feature: 2 });
  });

  it("keeps a free pick outside the radius", () => {
    expect(snapToFeature([50, 50], features)).toEqual({ px: [50, 50], feature: null });
  });
});

describe("distanceToPolyline", () => {
  it("measures to the nearest segment, clamped to its ends", () => {
    const line: [number, number][] = [
      [0, 0],
      [10, 0],
      [10, 10],
    ];
    expect(distanceToPolyline([5, 3], line)).toBeCloseTo(3);
    expect(distanceToPolyline([13, 5], line)).toBeCloseTo(3);
    expect(distanceToPolyline([-3, -4], line)).toBeCloseTo(5);
  });

  it("handles a degenerate segment", () => {
    expect(
      distanceToPolyline(
        [3, 4],
        [
          [0, 0],
          [0, 0],
        ],
      ),
    ).toBeCloseTo(5);
  });
});

describe("markers", () => {
  it("draws every feature as a muted dot of the given size", () => {
    expect(featureDots([feature(1, 1, 2)], 6)).toEqual([
      { px: [1, 2], color: "var(--fg-muted)", dot: true, size: 6 },
    ]);
  });

  it("tones the epipolar residual", () => {
    expect(ghostDistanceTone(0.5)).toBe("signal");
    expect(ghostDistanceTone(2)).toBe("fg");
    expect(ghostDistanceTone(5)).toBe("defect");
  });
});
