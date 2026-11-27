import { describe, expect, it } from "vitest";
import { StudioError } from "../api";
import type { RunInfo } from "../api/generated/types";
import {
  asProjectError,
  folderName,
  inlineManifest,
  runLabel,
  runMeta,
  staleParts,
  withManifest,
} from "./projectModel";

const DIGEST = {
  inputs: "sha256:i",
  dataset: "sha256:d",
  exclusions: "sha256:e",
  config: "sha256:c",
  quality: "sha256:q",
  detections: "sha256:x",
};

/** A calibrated run with `DIGEST`. */
function calibrated(): RunInfo {
  return {
    id: "run-002",
    dir: "/p/runs/run-002",
    notes: {},
    state: {
      kind: "recorded",
      version: 1,
      timestamp: 1,
      tool: "test",
      library_version: "1.0.0",
      summary: { kind: "planar_intrinsics", verdict: "pass", cameras: [] },
      source: {
        kind: "calibrated",
        digest: DIGEST,
        usable_views: 7,
        total_views: 8,
        left_out: [],
        redetected: false,
        duration_ms: 1,
        inputs: {} as never,
      },
    },
  };
}

describe("folderName", () => {
  it("takes the last segment of either separator, trailing ones ignored", () => {
    expect(folderName("/data/kuka_1")).toBe("kuka_1");
    expect(folderName("/data/kuka_1/")).toBe("kuka_1");
    expect(folderName("C:\\data\\rig")).toBe("rig");
    expect(folderName("plain")).toBe("plain");
  });
});

describe("asProjectError", () => {
  it("keeps the back end's error envelope with its field", () => {
    expect(
      asProjectError(
        new StudioError({
          kind: "exclusion",
          message: "stale",
          field: "exclude.images[0]",
        }),
      ),
    ).toEqual({ kind: "exclusion", message: "stale", field: "exclude.images[0]" });
    expect(asProjectError({ kind: "not_a_project", message: "no project" })).toEqual({
      kind: "not_a_project",
      message: "no project",
    });
  });

  it("reports anything else as an io error with its text", () => {
    expect(asProjectError(new Error("ipc down"))).toEqual({
      kind: "io",
      message: "ipc down",
    });
  });
});

describe("runs", () => {
  it("labels a run by its notes and summarises its record", () => {
    const run = calibrated();
    expect(runLabel(run)).toBe("run-002");
    expect(runMeta(run)).toBe("7/8 views · pass");
    expect(runLabel({ ...run, notes: { label: "tight", starred: true } })).toBe(
      "★ run-002 · tight",
    );
    expect(
      runMeta({
        ...run,
        state: { kind: "unrecorded", usable_views: 2, total_views: 2 },
      }),
    ).toBe("2/2 views");
    expect(runMeta({ ...run, state: { kind: "unreadable", message: "v9" } })).toBe(
      "unreadable",
    );
  });

  it("is stale in the parts whose digest differs from the project's", () => {
    const run = calibrated();
    const current = {
      dataset: "sha256:d",
      exclusions: "sha256:e",
      config: "sha256:c",
      quality: "sha256:q",
    };
    expect(staleParts(run, current)).toEqual([]);
    expect(staleParts(run, { ...current, config: "sha256:other" })).toEqual(["config"]);
    expect(staleParts(run, null)).toBeNull();
    expect(
      staleParts(
        { ...run, state: { kind: "unrecorded", usable_views: null, total_views: null } },
        current,
      ),
    ).toBeNull();
  });
});

describe("the dataset in a project file", () => {
  it("is edited only when written in the file", () => {
    const manifest = { version: 1, topology: "planar_intrinsics", cameras: [] } as never;
    const inline = { version: 2, dataset: { root: "data", manifest } };
    expect(inlineManifest(inline)).toBe(manifest);
    expect(
      inlineManifest({ version: 2, dataset: { manifest: "dataset.toml" } }),
    ).toBeNull();
    const next = { ...(manifest as object), description: "x" } as never;
    expect(withManifest(inline, next)).toEqual({
      version: 2,
      dataset: { root: "data", manifest: next },
    });
  });
});
