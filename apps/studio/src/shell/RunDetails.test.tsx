// @vitest-environment jsdom
/** The run's details: the seed its record says it started from. */
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { RunInfo, RunSeed } from "../api/generated/types";
import { runRow } from "../project/runRows";
import { calibratedRun, projectView } from "../test/projectFixtures";
import { PREVIEW_RUN } from "../screens/device/__fixtures__/deviceFixtures";
import { RunDetails } from "./RunDetails";

afterEach(cleanup);

const DIR = "/work/cell-a";

function withSeed(seed: RunSeed | null): RunInfo {
  const run = calibratedRun(DIR, "run-002");
  if (run.state.kind !== "recorded" || run.state.source.kind !== "calibrated") {
    throw new Error("fixture");
  }
  return {
    ...run,
    state: {
      ...run.state,
      source: {
        ...run.state.source,
        inputs: { ...run.state.source.inputs, ...(seed ? { seed } : {}) },
      },
    },
  };
}

const show = (run: RunInfo) =>
  render(<RunDetails run={run} row={runRow(run, projectView(DIR, [run]))} />);

describe("the seed of a run", () => {
  it("says where its values came from and how many it took", () => {
    show(
      withSeed({
        applied: ["cam0.lens", "cam1.lens", "handeye"],
        ignored: ["cam0.laser_plane"],
        source: { run: "/work/unit-7/runs/run-003" },
        value: PREVIEW_RUN.seed,
      }),
    );
    expect(screen.getByText("Seed")).toBeTruthy();
    expect(screen.getByText("run /work/unit-7/runs/run-003")).toBeTruthy();
    expect(screen.getByText("took 3 values, 1 not used")).toBeTruthy();
  });

  it("says none for a run that started from nothing", () => {
    show(withSeed(null));
    expect(screen.getByText("Seed")).toBeTruthy();
    expect(screen.getByText("none")).toBeTruthy();
  });
});
