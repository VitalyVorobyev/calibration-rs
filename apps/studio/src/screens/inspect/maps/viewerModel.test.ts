import { describe, expect, it } from "vitest";
import {
  chosenDomain,
  effectiveColormap,
  heatOpacity,
  isoStep,
  legendClipped,
  legendMarkers,
} from "./viewerModel";

describe("the colour map", () => {
  it("is diverging for a signed quantity, whatever was chosen", () => {
    expect(effectiveColormap("signed", "magma", false)).toBe("blueOrange");
    expect(effectiveColormap("signed", "auto", true)).toBe("blueOrange");
  });
  it("is viridis on the canvas and inferno over an image, unless chosen", () => {
    expect(effectiveColormap("magnitude", "auto", false)).toBe("viridis");
    expect(effectiveColormap("magnitude", "auto", true)).toBe("inferno");
    expect(effectiveColormap("magnitude", "cividis", true)).toBe("cividis");
  });
  it("lets an image through", () => {
    expect(heatOpacity(true)).toBe(0.7);
    expect(heatOpacity(false)).toBe(1);
  });
});

describe("the range", () => {
  it("is the automatic one, or a fixed top", () => {
    expect(chosenDomain("magnitude", [0, 2], null)).toEqual([0, 2]);
    expect(chosenDomain("magnitude", [0, 2], 5)).toEqual([0, 5]);
    expect(chosenDomain("signed", [-2, 2], 5)).toEqual([-5, 5]);
  });
  it("is flat for a channel with no value", () => {
    expect(chosenDomain("magnitude", null, null)).toEqual([0, 1]);
    expect(chosenDomain("signed", null, null)).toEqual([-1, 1]);
  });
  it("ignores a top that is not positive", () => {
    expect(chosenDomain("magnitude", [0, 2], 0)).toEqual([0, 2]);
  });
});

describe("iso-lines", () => {
  it("are spaced 1, 2 or 5 × 10ⁿ giving a handful across the span", () => {
    expect(isoStep([0, 2])).toBe(0.5);
    expect(isoStep([0, 0.05])).toBe(0.01);
    expect(isoStep([-1, 1])).toBe(0.5);
    expect(isoStep([0, 50])).toBe(10);
    expect(isoStep([0, 0])).toBe(1);
  });
});

describe("the legend", () => {
  it("says which end clips", () => {
    expect(legendClipped(true, false)).toBe("high");
    expect(legendClipped(false, true)).toBe("low");
    expect(legendClipped(true, true)).toBe("both");
    expect(legendClipped(false, false)).toBeUndefined();
  });
  it("marks a named limit and the value under the pointer, with the unit", () => {
    expect(legendMarkers({ value: 3.03, label: "p = 0.01" }, 0.25, "px")).toEqual([
      { value: 3.03, label: "p = 0.01" },
      { value: 0.25, label: "0.25 px" },
    ]);
    expect(legendMarkers(undefined, Number.NaN, "px")).toEqual([]);
    expect(legendMarkers(undefined, undefined, "px")).toEqual([]);
  });
});
