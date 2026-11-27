import { describe, expect, it } from "vitest";
import {
  DISTORTION_CHANNELS,
  GRID,
  RESIDUAL_CHANNELS,
  distortionPlane,
  plane,
  residualPlane,
  residualReply,
  sigmaPlane,
} from "../../../test/planeFixtures";
import {
  distortionStatement,
  gridText,
  placeWords,
  residualProvenance,
  residualStatement,
  sigmaStatement,
} from "./mapText";

describe("map statements", () => {
  it("names the grid", () => {
    expect(gridText(GRID)).toBe("4 × 3 nodes of 10 px");
  });

  it("puts a position in words", () => {
    expect(placeWords(GRID, 34, 25)).toBe("in the bottom-right corner");
    expect(placeWords(GRID, 20, 15)).toBe("near the centre");
    expect(placeWords(GRID, 20, 2)).toBe("at the top edge");
  });

  it("says how far the lens moves a pixel, and where", () => {
    const text = distortionStatement(
      { plane: distortionPlane(), grid: GRID, channels: DISTORTION_CHANNELS },
      "lens",
    );
    expect(text).toBe(
      "The lens displaces a pixel by up to 3.61 px in the bottom-right corner (35, 25).",
    );
  });

  it("says where the systematic error peaks and how much of the area is significant", () => {
    const text = residualStatement({
      plane: residualPlane(),
      grid: GRID,
      channels: RESIDUAL_CHANNELS,
    });
    expect(text).toContain("The systematic error peaks at 0.304 px");
    expect(text).toContain(
      "33 % of the area that could be tested differs from zero (χ ≥ 3.03, p < 0.01)",
    );
  });

  it("makes no claim of significance for a plane without a test", () => {
    const untested = plane(4, 3, 8, (c, col) => (c === 5 ? Number.NaN : 0.1 * (col + 1)));
    const text = residualStatement({
      plane: untested,
      grid: GRID,
      channels: RESIDUAL_CHANNELS,
    });
    expect(text).toMatch(/^The systematic error peaks at/);
    expect(text).not.toContain("differs from zero");
  });

  it("says nothing of an empty map", () => {
    const empty = plane(4, 3, 8, () => Number.NaN);
    expect(
      residualStatement({ plane: empty, grid: GRID, channels: RESIDUAL_CHANNELS }),
    ).toBeNull();
  });

  it("gives the projection σ at the centre and at its worst", () => {
    const text = sigmaStatement({
      plane: sigmaPlane(),
      grid: GRID,
      channels: [{ name: "sigma", unit: "px" }],
    });
    expect(text).toContain("up to 0.35 px in the bottom-right corner");
    expect(text).toContain("(1σ) at the centre");
  });

  it("writes the provenance of a residual map", () => {
    const line = residualProvenance(
      residualReply({ bandwidth_selection: { kind: "given" }, views: 1 }),
      "run-007",
    );
    expect(line).toBe(
      "run-007 · camera 0 · residual map · 4 × 3 nodes of 10 px · bandwidth 12 px (given) · 1 view · 40 features",
    );
  });
});
