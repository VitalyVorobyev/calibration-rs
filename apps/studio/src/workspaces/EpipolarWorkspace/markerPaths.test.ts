import { describe, expect, it } from "vitest";
import { buildMarkerPaths } from "./markerPaths";

describe("buildMarkerPaths", () => {
  it("batches markers of one appearance into a single path", () => {
    const out = buildMarkerPaths(
      [
        { px: [1, 1], color: "a", dot: true, size: 6 },
        { px: [2, 2], color: "a", dot: true, size: 6 },
        { px: [3, 3], color: "a", dot: true, size: 4 },
        { px: [4, 4], color: "b" },
        { px: [5, 5], color: "b" },
      ],
      1,
    );
    expect(out.map((p) => [p.kind, p.color])).toEqual([
      ["dot", "a"],
      ["dot", "a"],
      ["cross", "b"],
    ]);
    expect(out[0]!.d.match(/M/g)).toHaveLength(2);
    expect(out[2]!.d.match(/M/g)).toHaveLength(6); // 3 sub-paths x 2 markers
  });

  it("keeps the on-screen size constant by scaling the radius with the unit", () => {
    const at = (unit: number) =>
      buildMarkerPaths([{ px: [10, 10], color: "a" }], unit)[0]!.d;
    expect(at(1)).toContain("M4 10h12M10 4v12");
    expect(at(0.5)).toContain("M7 10h6M10 7v6");
  });

  it("returns nothing for no markers", () => {
    expect(buildMarkerPaths([], 1)).toEqual([]);
  });
});
