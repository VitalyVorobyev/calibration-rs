import { describe, expect, it } from "vitest";
import { niceCeil, quantile, rangeTop } from "./colorRange";

describe("niceCeil", () => {
  it("rounds up to two significant digits", () => {
    expect(niceCeil(0.0329)).toBe(0.033);
    expect(niceCeil(0.03)).toBe(0.03);
    expect(niceCeil(0.11)).toBe(0.11);
    expect(niceCeil(2.01)).toBe(2.1);
    expect(niceCeil(54.3)).toBe(55);
    expect(niceCeil(0)).toBe(1);
    expect(niceCeil(Number.NaN)).toBe(1);
  });
});

describe("quantile", () => {
  it("is the nearest-rank value, whatever the order", () => {
    expect(quantile([5, 1, 3, 2, 4], 0.5)).toBe(3);
    expect(quantile([5, 1, 3, 2, 4], 1)).toBe(5);
    expect(quantile([], 0.5)).toBeNaN();
  });
});

describe("rangeTop", () => {
  it("reaches the data's maximum when no value is an outlier", () => {
    expect(rangeTop([0.01, 0.02, 0.0329])).toBe(0.033);
  });
  it("stops at the percentile for a real outlier", () => {
    const values = Array.from({ length: 1000 }, (_, i) => (i === 999 ? 5 : 0.01));
    expect(rangeTop(values)).toBe(0.01);
  });
  it("has none without a finite value", () => {
    expect(rangeTop([])).toBeNull();
    expect(rangeTop([Number.NaN])).toBeNull();
  });
});
