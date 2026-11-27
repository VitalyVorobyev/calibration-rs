import { describe, expect, it } from "vitest";
import type { ProjectView, RunInfo } from "../api/generated/types";
import { calibratedRun, projectView } from "../test/projectFixtures";
import {
  differencesFromEditor,
  leafText,
  lowestRmsRun,
  recordedPatch,
  runRow,
  staleSince,
  type RunRow,
} from "./runRows";

/** A calibrated run of `dir` with a recorded patch and per-camera RMS. */
function run(
  dir: string,
  id: string,
  patch: Record<string, unknown>,
  rms: (number | null)[],
  verdict: "pass" | "warn" | "fail" = "pass",
): RunInfo {
  const base = calibratedRun(dir, id);
  if (base.state.kind !== "recorded" || base.state.source.kind !== "calibrated") {
    throw new Error("fixture is a calibrated run");
  }
  return {
    ...base,
    state: {
      ...base.state,
      summary: {
        kind: "rig_extrinsics",
        verdict,
        cameras: rms.map((r, i) => ({
          id: `cam${i}`,
          rms_px: r,
          features: 100,
          views: 9,
        })),
      },
      source: {
        ...base.state.source,
        inputs: { config_patch: patch } as never,
      },
    },
  };
}

const project = (): ProjectView => ({
  ...projectView("/work/p"),
  spec: {
    version: 2,
    dataset: { root: ".", manifest: "dataset.toml" },
    config: { solver: { max_iters: 100 } },
  },
});

describe("runRow", () => {
  it("reads the record: RMS pooled and per camera, views, features, changes", () => {
    const p = project();
    const r = run(
      p.dir,
      "run-001",
      { solver: { max_iters: 100 }, optimize: { reprojection_loss: { kind: "huber" } } },
      [0.3, 0.4],
    );
    const row = runRow(r, p);
    expect(row.rms).toBeCloseTo(Math.sqrt((0.09 + 0.16) / 2), 12);
    expect(row.cameras.map((c) => c.id)).toEqual(["cam0", "cam1"]);
    expect(row.features).toBe(200);
    expect(row.views).toEqual({ used: 9, total: 10 });
    // The project's own change is not the run's: only the loss differs.
    expect(row.changes).toEqual(["optimize.reprojection_loss.kind"]);
    expect(row.verdict).toBe("pass");
  });

  it("has no changes to name for a run with no record of its settings", () => {
    const p = project();
    const legacy: RunInfo = {
      id: "run-000",
      dir: `${p.dir}/runs/run-000`,
      notes: {},
      state: { kind: "unrecorded", summary: null, usable_views: 6, total_views: 8 },
    };
    const row = runRow(legacy, p);
    expect(row.changes).toBeNull();
    expect(row.views).toEqual({ used: 6, total: 8 });
    expect(row.state).toBe("Legacy");
    expect(recordedPatch(legacy)).toBeNull();
  });

  it("carries an unreadable run's message", () => {
    const p = project();
    const bad: RunInfo = {
      id: "run-002",
      dir: `${p.dir}/runs/run-002`,
      notes: {},
      state: { kind: "unreadable", message: "export.json: missing" },
    };
    expect(runRow(bad, p)).toMatchObject({
      state: "Unreadable",
      problem: "export.json: missing",
      rms: null,
    });
  });
});

describe("staleSince", () => {
  const OTHER = {
    dataset: "sha256:d",
    exclusions: "sha256:e",
    config: "sha256:new",
    quality: "sha256:q",
  };

  it("does not call a run's settings of its own a change of the project", () => {
    const p = project();
    // The run took a loss the project does not have: the library's config
    // digest differs, and the recorded changes say why.
    const r = run(
      p.dir,
      "run-001",
      { optimize: { reprojection_loss: { kind: "huber" } } },
      [0.3],
    );
    expect(staleSince(r, OTHER, p.spec.config)).toEqual([]);
  });

  it("still says when the project moved on: its configuration, with none of the run's own", () => {
    const p = project();
    const same = run(p.dir, "run-001", { solver: { max_iters: 100 } }, [0.3]);
    expect(staleSince(same, OTHER, p.spec.config)).toEqual(["config"]);
    expect(staleSince(same, { ...OTHER, dataset: "sha256:x" }, p.spec.config)).toEqual([
      "dataset",
      "config",
    ]);
  });

  it("keeps the other parts a run's changes do not explain", () => {
    const p = project();
    const r = run(p.dir, "run-001", { solver: { max_iters: 7 } }, [0.3]);
    expect(staleSince(r, { ...OTHER, exclusions: "sha256:x" }, p.spec.config)).toEqual([
      "exclusions",
    ]);
  });
});

describe("lowestRmsRun", () => {
  it("marks no run when two read the same to the thousandth", () => {
    const row = (id: string, rms: number): RunRow =>
      ({ id, rms, verdict: "pass" }) as RunRow;
    expect(lowestRmsRun([row("run-001", 0.1523), row("run-002", 0.1521)])).toBeNull();
    expect(lowestRmsRun([row("run-001", 0.1523), row("run-002", 0.2)])).toBe("run-001");
  });

  const row = (id: string, rms: number | null, verdict: RunRow["verdict"]): RunRow =>
    ({ id, rms, verdict }) as RunRow;

  it("is the lowest error among the runs that did not fail", () => {
    expect(
      lowestRmsRun([
        row("run-001", 0.4, "pass"),
        row("run-002", 0.1, "fail"),
        row("run-003", 0.3, "warn"),
      ]),
    ).toBe("run-003");
  });

  it("says nothing with fewer than two runs to be lowest among", () => {
    expect(lowestRmsRun([row("run-001", 0.4, "pass")])).toBeNull();
    expect(
      lowestRmsRun([row("run-001", 0.4, "pass"), row("run-002", null, null)]),
    ).toBeNull();
  });
});

describe("differencesFromEditor", () => {
  it("lists the leaves the run and the editor set differently", () => {
    const diffs = differencesFromEditor(
      { solver: { max_iters: 20 }, a: { b: "x" } },
      { solver: { max_iters: 50 } },
    );
    expect(diffs).toEqual([
      { path: "a.b", run: '"x"', editor: "default" },
      { path: "solver.max_iters", run: "20", editor: "50" },
    ]);
    expect(leafText('"x"')).toBe("x");
    expect(leafText("20")).toBe("20");
    expect(leafText(undefined)).toBe("default");
  });
});
