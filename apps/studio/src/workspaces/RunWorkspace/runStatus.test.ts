import { describe, expect, it } from "vitest";
import { StudioError } from "../../api";
import type { RunInfo } from "../../api/generated/types";
import {
  newRunId,
  runSummaryLine,
  statusFromError,
  withProgress,
  type RunStatus,
} from "./runStatus";
import {
  configSummary,
  manifestSummary,
  robustLossLabel,
  runBlockReason,
  targetKindOf,
} from "./setupSummary";

describe("statusFromError", () => {
  it("asks for an open choice with its field and suggestions", () => {
    const e = new StudioError({
      kind: "needs_input",
      message: "How are views paired?",
      field: "pose_pairing",
      suggestions: ["by_index"],
    });
    expect(statusFromError(e)).toEqual({
      kind: "ask_user",
      field: "pose_pairing",
      prompt: "How are views paired?",
      suggestions: ["by_index"],
    });
  });

  it("maps a cancellation and every other failure to its banner", () => {
    expect(statusFromError(new StudioError({ kind: "cancelled", message: "c" }))).toEqual(
      { kind: "cancelled" },
    );
    expect(
      statusFromError({ kind: "manifest", message: "unresolved", field: "target" }),
    ).toEqual({ kind: "error", category: "manifest", message: "unresolved (target)" });
    expect(statusFromError(new Error("ipc down"))).toEqual({
      kind: "error",
      category: "internal",
      message: "ipc down",
    });
  });

  it("issues distinct run ids", () => {
    expect(newRunId()).not.toBe(newRunId());
  });
});

describe("runSummaryLine", () => {
  it("says what a calibrated run kept and where it went", () => {
    const run = {
      id: "run-003",
      dir: "/p/runs/run-003",
      notes: {},
      state: {
        kind: "recorded",
        version: 1,
        timestamp: 1,
        tool: "t",
        library_version: "1",
        summary: { kind: "planar_intrinsics", verdict: "warn", cameras: [] },
        source: {
          kind: "calibrated",
          usable_views: 9,
          total_views: 10,
          duration_ms: 42,
          left_out: [],
          redetected: false,
          digest: {} as never,
          inputs: {} as never,
        },
      },
    } satisfies RunInfo;
    expect(runSummaryLine(run)).toBe(
      "42 ms · 9/10 usable views · saved as runs/run-003 · verdict warn",
    );
  });
});

describe("setup summaries", () => {
  const manifest = {
    topology: "rig_extrinsics",
    target: { kind: "charuco" },
    cameras: [{}, {}],
  };

  it("summarizes the manifest", () => {
    expect(targetKindOf(manifest)).toBe("charuco");
    expect(manifestSummary(manifest)).toBe("rig_extrinsics · charuco · 2 cameras");
    expect(manifestSummary({ cameras: [{}] })).toBe("? · ? · 1 camera");
  });

  it("summarizes the config's robust loss by its kind", () => {
    expect(robustLossLabel({ kind: "none" })).toBe("none");
    expect(robustLossLabel({ kind: "huber", scale: 1 })).toBe("huber");
    expect(robustLossLabel(undefined)).toBe("none");
    expect(
      configSummary({
        solver: { max_iters: 50 },
        optimize: { reprojection_loss: { kind: "cauchy", scale: 1 } },
      }),
    ).toBe("max_iters=50 · loss=cauchy");
  });

  it("says why Run is blocked, most fundamental reason first", () => {
    const base = { supported: true, unresolved: 0, hasProject: true, configLoaded: true };
    expect(runBlockReason(base)).toBeUndefined();
    expect(runBlockReason({ ...base, hasProject: false, unresolved: 2 })).toBe(
      "Open or create a project first",
    );
    expect(runBlockReason({ ...base, configLoaded: false })).toBe(
      "The calibration config has not loaded",
    );
    expect(runBlockReason({ ...base, unresolved: 2 })).toBe(
      "Resolve 2 unresolved manifest fields first",
    );
    expect(
      runBlockReason({ ...base, supported: false, unsupportedReason: "unknown" }),
    ).toBe("unknown");
  });
});

describe("withProgress", () => {
  const running: Extract<RunStatus, { kind: "running" }> = {
    kind: "running",
    runId: "r",
    stage: null,
    report: null,
    step: null,
    startedAt: 0,
    cancelRequested: false,
  };

  it("keeps the step for the iterations that follow it", () => {
    const stepped = withProgress(running, { kind: "step", name: "optimize" });
    const iterated = withProgress(stepped, {
      kind: "iteration",
      iter: 2,
      max_iters: 50,
      cost: 1.5,
    });
    expect(iterated).toMatchObject({ stage: "solve", step: "optimize" });
    expect(iterated.report).toEqual({
      kind: "iteration",
      iter: 2,
      max_iters: 50,
      cost: 1.5,
    });
  });

  it("takes the stage from the report's kind", () => {
    expect(withProgress(running, { kind: "detection", done: 1, total: 2 }).stage).toBe(
      "detect",
    );
    expect(
      withProgress(running, { kind: "laser_extraction", done: 1, total: 2 }).stage,
    ).toBe("detect");
    expect(withProgress(running, { kind: "step", name: "init" }).stage).toBe("solve");
  });
});
