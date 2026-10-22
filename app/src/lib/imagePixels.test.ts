import { describe, expect, it } from "vitest";
import { getPixelLum, rectHistogram, rgbaToLuminance } from "./imagePixels";

describe("rgbaToLuminance", () => {
  it("is Rec. 601 luma, one byte per pixel", () => {
    const rgba = [255, 255, 255, 255, 255, 0, 0, 255, 0, 0, 0, 0];
    expect([...rgbaToLuminance(rgba)]).toEqual([255, 76, 0]);
  });
});

describe("pixel queries", () => {
  // 3 × 2 image, luminance = index × 40.
  const data = {
    naturalWidth: 3,
    naturalHeight: 2,
    luminance: Uint8Array.from([0, 40, 80, 120, 160, 200]),
  };

  it("reads a pixel, null outside the image", () => {
    expect(getPixelLum(data, 2, 1)).toBe(200);
    expect(getPixelLum(data, 1.7, 0.2)).toBe(40);
    expect(getPixelLum(data, 3, 0)).toBeNull();
    expect(getPixelLum(data, -1, 0)).toBeNull();
  });

  it("bins a rectangle, clipped to the image", () => {
    const bins = rectHistogram(data, { x: 1, y: 0, w: 10, h: 10 }, 4);
    // 40, 80 → bin 0 and 1; 160, 200 → bins 2 and 3.
    expect(bins).toEqual([1, 1, 1, 1]);
    expect(rectHistogram(data, { x: 5, y: 5, w: 1, h: 1 }, 4)).toEqual([0, 0, 0, 0]);
  });
});
