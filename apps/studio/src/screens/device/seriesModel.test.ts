import { describe, expect, it } from "vitest";
import {
  SERIES_SN1,
  SERIES_UNAVAILABLE,
  SN1,
  SPREAD_FX,
} from "./__fixtures__/deviceFixtures";
import { runKey } from "./deviceModel";
import {
  camerasIn,
  chartSeries,
  chartSpecs,
  gapNote,
  groupsPresent,
  markersFor,
  nearestRun,
  runMetaMap,
  spreadLine,
  xFormatter,
  xOfRun,
} from "./seriesModel";

const sets = [SERIES_SN1];
const meta = runMetaMap([SN1]);
const spec = (
  tab: Parameters<typeof chartSpecs>[1],
  key: string,
  x: "order" | "time" = "order",
) => chartSpecs(sets, tab, "cam0", x).find((s) => s.key === key)!;

describe("the groups and cameras", () => {
  it("lists the groups that have a parameter, in the tabs' order", () => {
    expect(groupsPresent(sets)).toEqual(["intrinsics", "distortion", "handeye"]);
    expect(camerasIn(sets, "intrinsics")).toEqual(["cam0"]);
    expect(camerasIn(sets, "handeye")).toEqual([]);
  });
});

describe("a parameter's line", () => {
  it("has a point per run, and a gap (never a zero) where a run lacks the value", () => {
    const line = spec("intrinsics", "cam0.fx").lines[0]!;
    expect(line.entries.map((e) => e.y)).toEqual([200, 200.6, Number.NaN, Number.NaN]);
    expect(line.entries.map((e) => e.x)).toEqual([1, 2, 3, 4]);
    expect(line.missing).toBe(2);
    const series = chartSeries(line, 0, "fx", "value");
    expect(series.points.map((p) => p.y).filter(Number.isNaN)).toHaveLength(2);
  });

  it("draws σ as a band of ±1σ, with none where a value has no σ", () => {
    const line = spec("distortion", "cam0.k1").lines[0]!;
    expect(line.entries[0]?.lower).toBeCloseTo(-0.13);
    expect(line.entries[0]?.upper).toBeCloseTo(-0.11);
    expect(Number.isNaN(line.entries[1]!.lower)).toBe(true);
    expect(line.withoutSigma).toBe(1);
    expect(gapNote(line, "order")).toBe(
      "1 run has no value (a gap); 1 without a σ (no band)",
    );
  });

  it("says in the legend when σ is of a perturbation", () => {
    const value = chartSeries(spec("intrinsics", "cam0.fx").lines[0]!, 0, "fx", "value");
    expect(value.band?.label).toBe("±1σ");
    const pose = spec("handeye", "gripper_se3_rig.tx");
    expect(pose.sigmaOf).toBe("perturbation");
    expect(chartSeries(pose.lines[0]!, 0, "tx", pose.sigmaOf).band?.label).toBe(
      "±1σ of the perturbation",
    );
  });

  it("puts runs on a time axis by their recorded time and leaves out those with none", () => {
    const line = spec("intrinsics", "cam0.fx", "time").lines[0]!;
    // Run 0 and run 3 have no time.
    expect(line.entries.map((e) => e.x)).toEqual([1_700_100_000, 1_700_200_000]);
    expect(line.undated).toBe(2);
    expect(gapNote(line, "time")).toContain("2 with no recorded time (not drawn)");
    expect(xFormatter("order")(3)).toBe("3");
    // No tick between two runs.
    expect(xFormatter("order")(2.5)).toBe("");
    expect(xFormatter("time")(1_700_100_000)).toMatch(/^2023-11-\d\d$/);
  });
});

describe("the marks on the axis", () => {
  it("mark the representative run, starred runs and values on a bound", () => {
    const k1 = markersFor(spec("distortion", "cam0.k1").lines, meta);
    expect(k1).toEqual([
      { position: 2, tone: "signal", label: "run-002 (representative)" },
      { position: 2, tone: "warn", label: "run-002: ended on a bound" },
    ]);
  });

  it("name the unit when there are several", () => {
    const two = [SERIES_SN1, { ...SERIES_SN1, device: { id: "SN-0057" } }];
    const [fx] = chartSpecs(two, "intrinsics", "cam0", "order");
    expect(fx?.lines.map((l) => l.unit)).toEqual(["SN-0042", "SN-0057"]);
    expect(markersFor(fx!.lines, meta)[0]?.label).toBe(
      "SN-0042: run-002 (representative)",
    );
  });
});

describe("the linked selection", () => {
  const lines = spec("intrinsics", "cam0.fx").lines;

  it("finds the nearest run with a value, never a gap", () => {
    expect(nearestRun(lines, 1.2)).toBe(runKey("/data/bench", "run-001"));
    expect(nearestRun(lines, 1.8)).toBe(runKey("/data/bench", "run-002"));
    // Past the last value, still the last run that has one.
    expect(nearestRun(lines, 4)).toBe(runKey("/data/bench", "run-002"));
    expect(nearestRun([], 1)).toBeNull();
  });

  it("puts the hovered run's rule at its x in every chart, and nowhere else", () => {
    const key = runKey("/data/field", "run-001");
    expect(xOfRun(lines, key)).toBe(3);
    expect(xOfRun(lines, "nope")).toBeNull();
    expect(xOfRun(lines, null)).toBeNull();
  });
});

describe("the spread", () => {
  it("is the library's, beside a camera parameter", () => {
    const stats = spec("intrinsics", "cam0.fx").lines[0]!.stats;
    expect(stats).toEqual({ kind: "spread", spread: SPREAD_FX, runs: 3 });
    expect(spreadLine(SPREAD_FX)).toBe(
      "median 200 · robust σ 0.4 · min 199.8 · max 200.6",
    );
  });

  it("says why a pose has none, and why none is available", () => {
    const pose = spec("handeye", "gripper_se3_rig.tx").lines[0]!.stats;
    expect(pose.kind === "none" && pose.reason).toMatch(/own parametrization/);
    const [fx] = chartSpecs(SERIES_UNAVAILABLE.devices, "intrinsics", "cam0", "order");
    const stats = fx!.lines[0]!.stats;
    expect(stats.kind === "none" && stats.reason).toMatch(/different cameras/);
  });
});
