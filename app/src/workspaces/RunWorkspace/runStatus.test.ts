import { describe, expect, it } from "vitest";
import { newRunId, statusFromResponse } from "./runStatus";
import {
  configSummary,
  manifestSummary,
  robustLossLabel,
  runBlockReason,
  targetKindOf,
} from "./setupSummary";

describe("statusFromResponse", () => {
  it("maps a success, with the project run it was saved as", () => {
    const ok = {
      kind: "ok" as const,
      export: {},
      duration_ms: 42,
      usable_views: 9,
      total_views: 10,
      cache_used: true,
    };
    expect(statusFromResponse(ok)).toEqual({
      kind: "ok",
      durationMs: 42,
      usable: 9,
      total: 10,
      cacheUsed: true,
      savedAs: null,
    });
    expect(statusFromResponse(ok, "run-003")).toMatchObject({ savedAs: "run-003" });
  });

  it("maps every terminal failure to its banner", () => {
    expect(statusFromResponse({ kind: "cancelled" })).toEqual({ kind: "cancelled" });
    expect(statusFromResponse({ kind: "validation_failed", message: "m" })).toEqual({
      kind: "validation",
      message: "m",
    });
    expect(statusFromResponse({ kind: "failed", category: "io", message: "m" })).toEqual({
      kind: "error",
      category: "io",
      message: "m",
    });
    expect(
      statusFromResponse({
        kind: "ask_user",
        field: "f",
        prompt: "p",
        suggestions: ["a"],
      }),
    ).toEqual({ kind: "ask_user", field: "f", prompt: "p", suggestions: ["a"] });
  });

  it("issues distinct run ids", () => {
    expect(newRunId()).not.toBe(newRunId());
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

  it("summarizes the config's robust loss in each wire shape", () => {
    expect(robustLossLabel("None")).toBe("None");
    expect(robustLossLabel({ type: "huber", scale: 1 })).toBe("huber");
    expect(robustLossLabel({ Cauchy: { scale: 1 } })).toBe("Cauchy");
    expect(robustLossLabel(undefined)).toBe("None");
    expect(configSummary({ max_iters: 50, robust_loss: "None" })).toBe(
      "max_iters=50 · loss=None",
    );
  });

  it("says why Run is blocked, most fundamental reason first", () => {
    const base = { supported: true, unresolved: 0, manifestDir: "/d" };
    expect(runBlockReason(base)).toBeUndefined();
    expect(runBlockReason({ ...base, manifestDir: null })).toBe(
      "Set a dataset folder first",
    );
    expect(runBlockReason({ ...base, unresolved: 2, manifestDir: null })).toBe(
      "Resolve 2 unresolved manifest fields first",
    );
    expect(
      runBlockReason({
        ...base,
        supported: false,
        unsupportedReason: "unknown",
        unresolved: 1,
      }),
    ).toBe("unknown");
  });
});
