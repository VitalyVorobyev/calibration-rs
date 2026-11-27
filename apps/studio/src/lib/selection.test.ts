import { describe, expect, it } from "vitest";
import {
  escapeSelection,
  NO_SELECTION,
  sameSelection,
  type Selection,
  stepId,
  stepWrap,
  withCamera,
  withFeature,
  withView,
} from "./selection";

describe("selection changes", () => {
  it("drops the feature when the view or the camera changes, keeps the view across cameras", () => {
    const s = { camera: 0, view: 3, feature: 12 };
    expect(withView(s, 4)).toEqual({ camera: 0, view: 4, feature: null });
    expect(withCamera(s, 1)).toEqual({ camera: 1, view: 3, feature: null });
    expect(withFeature(s, 5)).toEqual({ camera: 0, view: 3, feature: 5 });
  });

  it("returns the same object when nothing changes", () => {
    const s = { camera: 0, view: 3, feature: 12 };
    expect(withView(s, 3)).toBe(s);
    expect(withCamera(s, 0)).toBe(s);
    expect(withFeature(s, 12)).toBe(s);
    expect(sameSelection(s, { ...s })).toBe(true);
    expect(sameSelection(s, withFeature(s, null))).toBe(false);
  });

  it("clears with Esc one level at a time, innermost first", () => {
    let s: Selection = { camera: 1, view: 2, feature: 3 };
    s = escapeSelection(s);
    expect(s).toEqual({ camera: 1, view: 2, feature: null });
    s = escapeSelection(s);
    expect(s).toEqual({ camera: 1, view: null, feature: null });
    s = escapeSelection(s);
    expect(s).toEqual(NO_SELECTION);
    expect(escapeSelection(s)).toBe(s);
  });
});

describe("stepWrap", () => {
  it("steps through the values and wraps at the ends", () => {
    expect(stepWrap([0, 2, 5], 2, 1)).toBe(5);
    expect(stepWrap([0, 2, 5], 5, 1)).toBe(0);
    expect(stepWrap([0, 2, 5], 0, -1)).toBe(5);
    expect(stepWrap([0, 2, 5], 0, 4)).toBe(2);
  });
  it("starts at an end when the current value is not in the list", () => {
    expect(stepWrap([0, 2, 5], null, 1)).toBe(0);
    expect(stepWrap([0, 2, 5], 9, -1)).toBe(5);
    expect(stepWrap([], 1, 1)).toBeNull();
  });
});

describe("stepId", () => {
  const ids = ["run-003", "run-002", "run-001"];
  it("stops at the ends instead of wrapping", () => {
    expect(stepId(ids, "run-002", 1)).toBe("run-001");
    expect(stepId(ids, "run-001", 1)).toBe("run-001");
    expect(stepId(ids, "run-003", -1)).toBe("run-003");
  });
  it("starts at the newest when there is no current run", () => {
    expect(stepId(ids, null, 1)).toBe("run-003");
    expect(stepId([], null, 1)).toBeNull();
  });
});
