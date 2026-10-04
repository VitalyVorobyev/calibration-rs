import { describe, expect, it } from "vitest";
import type { LaserFeatureResidual, TargetFeatureResidual } from "../types";
import { colorForError, colorForLaserError } from "../lib/errorColors";
import {
  ARROW_GAIN,
  ARROW_LENGTH_CAP_PX,
  buildLaserOverlay,
  buildResidualArrows,
} from "./frameOverlayPaths";

const FRAME = { pose: 1, camera: 0 };

function tgt(
  over: Partial<TargetFeatureResidual> & Pick<TargetFeatureResidual, "observed_px">,
): TargetFeatureResidual {
  return {
    pose: 1,
    camera: 0,
    feature: 0,
    target_xyz_m: [0, 0, 0],
    projected_px: null,
    error_px: null,
    ...over,
  };
}

function las(
  over: Partial<LaserFeatureResidual> & Pick<LaserFeatureResidual, "observed_px">,
): LaserFeatureResidual {
  return { pose: 1, camera: 0, feature: 0, ...over };
}

/** First two numbers of the path: where the shaft starts and, after `L`, ends. */
function shaft(d: string): { from: number[]; to: number[] } {
  const m = /^M([\d.e-]+) ([\d.e-]+)L([\d.e-]+) ([\d.e-]+)/.exec(d)!;
  return { from: [Number(m[1]), Number(m[2])], to: [Number(m[3]), Number(m[4])] };
}

describe("buildResidualArrows", () => {
  it("buckets arrows by error colour, one path per colour", () => {
    const res = [
      tgt({ observed_px: [10, 10], projected_px: [10.5, 10], error_px: 0.5 }),
      tgt({ observed_px: [20, 20], projected_px: [20.4, 20], error_px: 0.4 }),
      tgt({ observed_px: [30, 30], projected_px: [36, 30], error_px: 6 }),
    ];
    const { arrows } = buildResidualArrows(res, FRAME, 1);
    expect(arrows.map((a) => a.color)).toEqual([colorForError(0.5), colorForError(6)]);
    expect(arrows[0]!.d.match(/M/g)).toHaveLength(4); // two arrows x (shaft + 2nd barb)
    expect(arrows[1]!.d.match(/M/g)).toHaveLength(2);
  });

  it("falls back to the geometric magnitude when error_px is absent", () => {
    const res = [tgt({ observed_px: [0, 0], projected_px: [12, 0] })];
    expect(buildResidualArrows(res, FRAME, 1).arrows[0]!.color).toBe(colorForError(12));
  });

  it("magnifies short residuals by the gain and caps long ones in length", () => {
    const short = buildResidualArrows(
      [tgt({ observed_px: [5, 5], projected_px: [5.1, 5] })],
      FRAME,
      1,
    ).arrows[0]!;
    expect(shaft(short.d).to[0]! - 5).toBeCloseTo(0.1 * ARROW_GAIN, 3);
    const long = buildResidualArrows(
      [tgt({ observed_px: [5, 5], projected_px: [5, 105] })],
      FRAME,
      1,
    ).arrows[0]!;
    expect(shaft(long.d).to[1]! - 5).toBeCloseTo(ARROW_LENGTH_CAP_PX, 3);
  });

  it("skips zero-length residuals and those without a projection", () => {
    const res = [
      tgt({ observed_px: [1, 1], projected_px: [1, 1] }),
      tgt({ observed_px: [2, 2], projected_px: null }),
    ];
    expect(buildResidualArrows(res, FRAME, 1)).toEqual({ arrows: [], dots: "" });
  });

  it("keeps only the requested pose and camera", () => {
    const res = [
      tgt({ observed_px: [1, 1], projected_px: [2, 1], pose: 2 }),
      tgt({ observed_px: [1, 1], projected_px: [2, 1], camera: 3 }),
    ];
    expect(buildResidualArrows(res, FRAME, 1).arrows).toEqual([]);
  });

  it("draws in the ROI-local frame as given, with no offset applied", () => {
    const { arrows } = buildResidualArrows(
      [tgt({ observed_px: [7.5, 9.25], projected_px: [8.5, 9.25] })],
      FRAME,
      1,
    );
    expect(shaft(arrows[0]!.d).from).toEqual([7.5, 9.25]);
  });

  it("scales heads and dots with the screen-pixel unit, not the data", () => {
    const res = [tgt({ observed_px: [0, 0], projected_px: [1, 0] })];
    const one = buildResidualArrows(res, FRAME, 1);
    const half = buildResidualArrows(res, FRAME, 0.5);
    expect(shaft(one.arrows[0]!.d).to).toEqual(shaft(half.arrows[0]!.d).to);
    expect(one.arrows[0]!.d).not.toBe(half.arrows[0]!.d);
    expect(one.dots).toContain("a1.6 1.6");
    expect(half.dots).toContain("a0.8 0.8");
  });
});

describe("buildLaserOverlay", () => {
  it("returns the projected line from the first record that carries it", () => {
    const out = buildLaserOverlay(
      [
        las({ observed_px: [1, 1] }),
        las({
          observed_px: [2, 2],
          projected_line_px: [
            [0, 10],
            [100, 12],
          ],
        }),
      ],
      FRAME,
      1,
    );
    expect(out.line).toBe("M0 10L100 12");
  });

  it("has no line when no record carries one", () => {
    expect(buildLaserOverlay([las({ observed_px: [1, 1] })], FRAME, 1).line).toBeNull();
  });

  it("buckets dots by point-to-plane error in mm; null residuals go to one translucent path", () => {
    const out = buildLaserOverlay(
      [
        las({ observed_px: [1, 1], residual_m: 0.0001 }),
        las({ observed_px: [2, 2], residual_m: -0.0001 }),
        las({ observed_px: [3, 3], residual_m: 0.003 }),
        las({ observed_px: [4, 4], residual_m: null }),
        las({ observed_px: [5, 5] }),
        las({ observed_px: [6, 6], residual_m: 0, pose: 9 }),
      ],
      FRAME,
      1,
    );
    expect(out.dots.map((d) => d.color)).toEqual([
      colorForLaserError(0.1),
      colorForLaserError(3),
    ]);
    expect(out.dots[0]!.d.match(/M/g)).toHaveLength(2);
    expect(out.unresolved.match(/M/g)).toHaveLength(2);
  });
});
