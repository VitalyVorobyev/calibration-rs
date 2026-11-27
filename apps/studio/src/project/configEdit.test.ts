import { describe, expect, it } from "vitest";
import {
  changedPaths,
  isEmptyPatch,
  patchChanges,
  patchOver,
  pathLabel,
} from "./configEdit";

const BASE = {
  camera: { model: { projection: "pinhole", distortion: "brown_conrady5" }, fix: {} },
  camera_overrides: [] as unknown[],
  solver: { max_iters: 50, backend: "tiny_solver" },
  loss: { kind: "none" },
};

describe("changedPaths", () => {
  it("is empty for equal configurations", () => {
    expect(changedPaths(BASE, structuredClone(BASE))).toEqual([]);
  });

  it("names each changed leaf", () => {
    const edited = { ...BASE, solver: { ...BASE.solver, max_iters: 20 } };
    expect(changedPaths(edited, BASE)).toEqual(["solver.max_iters"]);
  });

  it("counts a list as one value, however it differs", () => {
    const edited = {
      ...BASE,
      camera_overrides: [{ camera: 1, model: { projection: "pinhole" } }],
    };
    expect(changedPaths(edited, BASE)).toEqual(["camera_overrides"]);
    const longer = {
      ...BASE,
      camera_overrides: [{ camera: 1, model: { projection: "equidistant" } }],
    };
    expect(changedPaths(longer, edited)).toEqual(["camera_overrides"]);
  });
});

describe("patchOver", () => {
  it("is empty when nothing changed", () => {
    expect(patchOver(BASE, BASE)).toEqual({});
    expect(isEmptyPatch(patchOver(BASE, BASE))).toBe(true);
  });

  it("holds only what differs from the base, nested like the config", () => {
    const edited = {
      ...BASE,
      solver: { ...BASE.solver, max_iters: 20 },
      loss: { kind: "huber", scale: 1 },
    };
    expect(patchOver(edited, BASE)).toEqual({
      solver: { max_iters: 20 },
      loss: { kind: "huber", scale: 1 },
    });
  });

  it("writes a value the project changed back to the default explicitly", () => {
    // The project config says 200; the editor is back at the default 50:
    // the patch must say 50, because it merges over the project's.
    const project = { ...BASE, solver: { ...BASE.solver, max_iters: 200 } };
    expect(patchOver(BASE, project)).toEqual({ solver: { max_iters: 50 } });
  });

  it("sets a removed optional field to null", () => {
    const project = { ...BASE, extra: { a: 1 } };
    expect(patchOver(BASE, project)).toEqual({ extra: null });
  });

  it("replaces a changed list whole", () => {
    const edited = { ...BASE, camera_overrides: [{ camera: 2 }] };
    expect(patchOver(edited, BASE)).toEqual({ camera_overrides: [{ camera: 2 }] });
  });
});

describe("patchChanges", () => {
  it("compares two patches", () => {
    expect(
      patchChanges({ solver: { max_iters: 20 } }, { solver: { max_iters: 20 } }),
    ).toEqual([]);
    expect(patchChanges({ solver: { max_iters: 20 } }, undefined)).toEqual([
      "solver.max_iters",
    ]);
  });
});

describe("pathLabel", () => {
  it("writes the last segments as words", () => {
    expect(pathLabel("rig.refine_intrinsics")).toBe("rig · refine intrinsics");
    expect(pathLabel("handeye.rig.refine_intrinsics")).toBe("rig · refine intrinsics");
    expect(pathLabel("")).toBe("configuration");
  });
});
