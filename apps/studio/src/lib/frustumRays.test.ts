import { describe, expect, it } from "vitest";
import {
  BORDER_SAMPLES_PER_EDGE,
  pinholeBorderRays,
  raysFromUndistortedPixels,
} from "./frustumRays";

const K = { fx: 1000, fy: 500, cx: 320, cy: 240 };

describe("raysFromUndistortedPixels", () => {
  it("inverts K onto the z = 1 plane", () => {
    const rays = raysFromUndistortedPixels(K, [320, 240, 1320, 740]);
    expect(Array.from(rays)).toEqual([0, 0, 1, 1, 1, 1]);
  });
});

describe("pinholeBorderRays", () => {
  it("walks the border clockwise from the top-left corner", () => {
    const rays = pinholeBorderRays(K, 640, 480, 1);
    expect(Array.from(rays)).toEqual([
      -0.32, -0.48, 1, 0.32, -0.48, 1, 0.32, 0.48, 1, -0.32, 0.48, 1,
    ]);
  });

  it("samples every edge by default", () => {
    expect(pinholeBorderRays(K, 640, 480).length).toBe(4 * BORDER_SAMPLES_PER_EDGE * 3);
  });
});
