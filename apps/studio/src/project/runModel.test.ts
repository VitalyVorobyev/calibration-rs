import { describe, expect, it } from "vitest";
import type { RunInfo } from "../api/generated/types";
import { calibratedRun } from "../test/projectFixtures";
import {
  defaultRun,
  runRms,
  runsNewestFirst,
  runStateName,
  runVerdict,
} from "./projectModel";

function withCameras(run: RunInfo, cameras: { features: number; rms: number | null }[]) {
  if (run.state.kind !== "recorded") throw new Error("fixture");
  return {
    ...run,
    state: {
      ...run.state,
      summary: {
        ...run.state.summary,
        cameras: cameras.map((c, i) => ({
          id: `cam${i}`,
          features: c.features,
          views: 3,
          rms_px: c.rms,
        })),
      },
    },
  } as RunInfo;
}

const unreadable = (id: string): RunInfo => ({
  id,
  dir: `/p/runs/${id}`,
  notes: {},
  state: { kind: "unreadable", message: "v9" },
});

describe("the order of runs", () => {
  it("is newest first, numerically", () => {
    const runs = ["run-002", "run-010", "run-001", "run-009"].map((id) =>
      calibratedRun("/p", id),
    );
    expect(runsNewestFirst(runs).map((r) => r.id)).toEqual([
      "run-010",
      "run-009",
      "run-002",
      "run-001",
    ]);
  });

  it("opens on the newest run that reads", () => {
    const runs = [calibratedRun("/p", "run-001"), unreadable("run-002")];
    expect(defaultRun(runs)?.id).toBe("run-001");
    expect(defaultRun([unreadable("run-001")])).toBeNull();
    expect(defaultRun([])).toBeNull();
  });
});

describe("what a run says of itself", () => {
  it("names the state of its files", () => {
    const run = calibratedRun("/p", "run-001");
    expect(runStateName(run)).toBe("Recorded");
    expect(runStateName(unreadable("run-002"))).toBe("Unreadable");
    expect(
      runStateName({
        ...run,
        state: { kind: "unrecorded", usable_views: null, total_views: null },
      }),
    ).toBe("Legacy");
    if (run.state.kind !== "recorded") throw new Error("fixture");
    expect(
      runStateName({
        ...run,
        state: { ...run.state, source: { kind: "imported", file_name: "e.json" } },
      }),
    ).toBe("Imported");
  });

  it("reads its verdict", () => {
    expect(runVerdict(calibratedRun("/p", "run-001"))).toBe("pass");
    expect(runVerdict(unreadable("run-002"))).toBeNull();
  });

  it("pools the cameras' RMS by their feature counts", () => {
    const run = withCameras(calibratedRun("/p", "run-001"), [
      { features: 100, rms: 0.2 },
      { features: 300, rms: 0.4 },
    ]);
    // sqrt((100·0.04 + 300·0.16) / 400)
    expect(runRms(run)).toBeCloseTo(Math.sqrt((4 + 48) / 400), 12);
  });

  it("skips a camera without an RMS, and has none when no camera has", () => {
    const run = withCameras(calibratedRun("/p", "run-001"), [
      { features: 100, rms: 0.2 },
      { features: 50, rms: null },
    ]);
    expect(runRms(run)).toBeCloseTo(0.2, 12);
    expect(runRms(withCameras(run, [{ features: 5, rms: null }]))).toBeNull();
    expect(runRms(unreadable("run-003"))).toBeNull();
  });
});
