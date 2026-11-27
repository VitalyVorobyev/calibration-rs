import { describe, expect, it } from "vitest";
import { stepIndex } from "./useViewKeys";

describe("stepIndex", () => {
  it("steps and stays inside", () => {
    expect(stepIndex(3, 1, 10)).toBe(4);
    expect(stepIndex(0, -1, 10)).toBe(0);
    expect(stepIndex(9, 1, 10)).toBe(9);
  });

  it("starts at an end when nothing is selected, and has nothing to step in an empty list", () => {
    expect(stepIndex(null, 1, 10)).toBe(0);
    expect(stepIndex(null, -1, 10)).toBe(9);
    expect(stepIndex(null, 1, 0)).toBeNull();
  });
});
