import { describe, expect, it } from "vitest";
import { advice, errorFieldPath, errorMessages, stageOf, tidy } from "./configCheck";

const CONFIG = {
  solver: { max_iters: 0, backend: "tiny_solver" },
  camera: { fix: { distortion: ["p9"] }, model: { distortion: "brown_conrady5" } },
  optimize: { reprojection_loss: { kind: "huber", scale: -1 } },
};

describe("errorFieldPath", () => {
  it("names the field the library's message starts with", () => {
    expect(
      errorFieldPath("invalid input: solver.max_iters must be positive", CONFIG),
    ).toBe("solver.max_iters");
    expect(
      errorFieldPath(
        "invalid input: optimize.reprojection_loss (huber): scale must be finite",
        CONFIG,
      ),
    ).toBe("optimize.reprojection_loss");
  });

  it("cuts back to the longest path the config has", () => {
    expect(
      errorFieldPath(
        "invalid input: camera.fix: invalid input: unknown distortion",
        CONFIG,
      ),
    ).toBe("camera.fix");
    expect(errorFieldPath("invalid input: solver.nothing.here is odd", CONFIG)).toBe(
      "solver",
    );
  });

  it("is the form's own when the message names no field", () => {
    expect(errorFieldPath("invalid type: string", CONFIG)).toBe("");
    expect(errorFieldPath("serialization error: unknown field `bogus`", CONFIG)).toBe("");
  });
});

describe("errorMessages", () => {
  it("keys the tidied message by its field, and is empty without an error", () => {
    expect(
      errorMessages(
        { message: "invalid input: solver.max_iters must be positive" },
        CONFIG,
      ),
    ).toEqual({ "solver.max_iters": "solver.max_iters must be positive" });
    expect(errorMessages(null, CONFIG)).toEqual({});
    expect(tidy("serialization error: bad")).toBe("bad");
  });
});

describe("a failed calibration", () => {
  it("says where it stopped, from the last report", () => {
    expect(stageOf({ kind: "detection", done: 3, total: 9 })).toBe(
      "while detecting the images (3 of 9)",
    );
    // The last count reaching the total: what failed came after detection.
    expect(stageOf({ kind: "detection", done: 9, total: 9 })).toBe(
      "after detecting the images",
    );
    expect(stageOf({ kind: "iteration", iter: 12 })).toBe("while solving (iteration 12)");
    expect(stageOf({ kind: "step", name: "rig init" })).toBe('in the step "rig init"');
    expect(stageOf(null)).toBeNull();
  });

  it("says what to try by the kind of failure", () => {
    expect(advice("calibration", "need 3, got 2 views")).toMatch(/too few usable views/);
    expect(advice("calibration", "diverged")).toMatch(/robust loss/);
    expect(advice("config", "x")).toMatch(/setting is not valid/);
    expect(advice("manifest", "x")).toMatch(/Dataset/);
  });
});
