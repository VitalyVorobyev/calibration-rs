import { describe, expect, it } from "vitest";
import { asPoseList, formatPoseList, isPoseListPath, parsePoseList } from "./poseList";

describe("parsePoseList", () => {
  it("reads numbers and ranges, sorted and without repeats", () => {
    expect(parsePoseList("5-7, 0 3 3")).toEqual({ value: [0, 3, 5, 6, 7] });
    expect(parsePoseList("")).toEqual({ value: [] });
    expect(parsePoseList("  ")).toEqual({ value: [] });
  });

  it("says what it cannot read", () => {
    expect(parsePoseList("1, x")).toEqual({
      problem: "“x” is not a view number or a range like 2-4.",
    });
    expect(parsePoseList("-1")).toHaveProperty("problem");
    expect(parsePoseList("1.5")).toHaveProperty("problem");
    expect(parsePoseList("7-3")).toEqual({ problem: "The range 7-3 runs backwards." });
  });
});

describe("formatPoseList", () => {
  it("collapses runs of three or more into ranges", () => {
    expect(formatPoseList([0, 1, 2, 4, 6, 7])).toBe("0-2, 4, 6, 7");
    expect(formatPoseList([])).toBe("");
  });

  it("round-trips through the parser", () => {
    const list = [0, 2, 3, 4, 5, 9];
    expect(parsePoseList(formatPoseList(list))).toEqual({ value: list });
  });
});

describe("asPoseList and the path", () => {
  it("keeps only non-negative integers", () => {
    expect(asPoseList([0, 2, -1, 1.5, "3"])).toEqual([0, 2]);
    expect(asPoseList(undefined)).toEqual([]);
  });

  it("matches every stage's fix_poses and nothing else", () => {
    expect(isPoseListPath("optimize.fix_poses")).toBe(true);
    expect(isPoseListPath("fix_poses")).toBe(true);
    expect(isPoseListPath("optimize.fix_poses_extra")).toBe(false);
    expect(isPoseListPath("camera.fix")).toBe(false);
  });
});
