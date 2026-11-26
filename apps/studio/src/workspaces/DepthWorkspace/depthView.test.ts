import type { ValuePlane } from "@vitavision/stage2d";
import { describe, expect, it } from "vitest";
import { DEPTH, heatmap, overlay } from "./depthView";

/** 2 × 1: one matched pixel, one without a disparity. */
const plane: ValuePlane = {
  width: 2,
  height: 1,
  stride: 1,
  channels: 3,
  // disparity, depth, left luminance
  values: new Float32Array([4, NaN, 0.5, NaN, 100, 200]),
};

describe("depth views", () => {
  it("colours a channel and paints a pixel without a value black", () => {
    const rgba = heatmap(plane, DEPTH);
    expect(Array.from(rgba.slice(4))).toEqual([0, 0, 0, 255]);
    expect(rgba[3]).toBe(255);
    expect(rgba[0]! + rgba[1]! + rgba[2]!).toBeGreaterThan(0);
  });

  it("blends the disparity over the image and shows the image where unmatched", () => {
    const rgba = overlay(plane);
    expect(Array.from(rgba.slice(4))).toEqual([200, 200, 200, 255]);
    expect(rgba[7]).toBe(255);
    // Matched: neither the plain image nor the plain colour.
    expect(Array.from(rgba.slice(0, 3))).not.toEqual([100, 100, 100]);
  });
});
