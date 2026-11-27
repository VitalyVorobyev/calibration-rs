import { describe, expect, it } from "vitest";
import type { ProjectEntry, RunInfo, Verdict } from "../../api/generated/types";
import { calibratedRun } from "../../test/projectFixtures";
import {
  containerOf,
  groupLibrary,
  isFiltering,
  latestRun,
  matches,
  NO_FILTER,
  openScreen,
  projectNodeId,
  projectRow,
  querySpans,
  relativeToRoots,
  rowsOf,
  UNSPECIFIED_TYPE,
  worstVerdict,
  type LibraryFilter,
} from "./libraryModel";

function run(
  id: string,
  verdict: Verdict | null,
  rms: number | null,
  extra: Partial<RunInfo["notes"]> = {},
  at = 100,
): RunInfo {
  const base = calibratedRun("/lib/p", id);
  if (base.state.kind !== "recorded") throw new Error("fixture");
  return {
    ...base,
    notes: extra,
    state: {
      ...base.state,
      timestamp: at,
      summary: {
        ...base.state.summary,
        verdict,
        cameras:
          rms === null ? [] : [{ id: "cam0", features: 100, views: 9, rms_px: rms }],
      },
    },
  };
}

function entry(
  name: string,
  device: ProjectEntry["device"] | undefined,
  runs: RunInfo[] = [],
  topology: ProjectEntry["topology"] = "planar_intrinsics",
): ProjectEntry {
  return { name, dir: `/lib/${name}`, device: device ?? null, runs, topology };
}

describe("latestRun and projectRow", () => {
  it("prefers the newest starred run, else the newest", () => {
    const runs = [
      run("run-001", "pass", 0.2, { starred: true }),
      run("run-002", "warn", 0.4),
      run("run-010", "fail", 0.9),
    ];
    expect(latestRun(runs)?.id).toBe("run-001");
    expect(latestRun(runs.slice(1))?.id).toBe("run-010");
    expect(latestRun([])).toBeNull();
  });

  it("reads verdict, RMS, views and time from the latest run", () => {
    const row = projectRow(entry("a", null, [run("run-001", "warn", 0.31, {}, 500)]));
    expect(row.verdict).toBe("warn");
    expect(row.rms).toBeCloseTo(0.31);
    expect(row.views).toEqual({ usable: 9, total: 10 });
    expect(row.updated).toBe(500);
    const empty = projectRow(entry("b", null));
    expect(empty.verdict).toBeNull();
    expect(empty.rms).toBeNull();
    expect(empty.updated).toBeNull();
  });
});

describe("worstVerdict", () => {
  it("ranks fail over warn over pass and ignores no verdict", () => {
    expect(worstVerdict(["pass", "warn", null])).toBe("warn");
    expect(worstVerdict(["pass", "fail", "warn"])).toBe("fail");
    expect(worstVerdict([null, null])).toBeNull();
    expect(worstVerdict([])).toBeNull();
  });
});

describe("groupLibrary", () => {
  const rows = [
    projectRow(entry("zeta", { id: "SN2", product: "Stereo-X1" })),
    projectRow(entry("alpha", { id: "SN10", product: "Stereo-X1" })),
    projectRow(entry("beta", { id: "SN2", product: "Stereo-X1" })),
    projectRow(entry("gamma", { id: "Q1" })),
    projectRow(entry("loose", null)),
    projectRow(entry("mono", { id: "M1", product: "Mono-Z" })),
  ];
  const groups = groupLibrary(rows);

  it("groups product then device, naturally sorted, unspecified type last", () => {
    expect(groups.products.map((p) => p.label)).toEqual([
      "Mono-Z",
      "Stereo-X1",
      UNSPECIFIED_TYPE,
    ]);
    const stereo = groups.products[1]!;
    expect(stereo.devices.map((d) => d.device.id)).toEqual(["SN2", "SN10"]);
    expect(stereo.devices[0]!.rows.map((r) => r.entry.name)).toEqual(["beta", "zeta"]);
  });

  it("keeps projects with no device apart", () => {
    expect(groups.ungrouped.map((r) => r.entry.name)).toEqual(["loose"]);
  });

  it("resolves a node to its rows", () => {
    expect(rowsOf(groups, null)).toHaveLength(6);
    expect(rowsOf(groups, "ungrouped")).toHaveLength(1);
    expect(rowsOf(groups, groups.products[1]!.id)).toHaveLength(3);
    expect(rowsOf(groups, groups.products[1]!.devices[0]!.id)).toHaveLength(2);
    expect(rowsOf(groups, projectNodeId("/lib/mono"))).toHaveLength(1);
    expect(rowsOf(groups, "nope")).toEqual([]);
  });
});

describe("matches", () => {
  const row = projectRow(
    entry(
      "bench-2026",
      { id: "SN0042", product: "Line scanner", label: "Cell 3" },
      [run("run-001", "pass", 0.2, { label: "tight loss" })],
      "rig_handeye",
    ),
  );
  const filter = (f: Partial<LibraryFilter>): LibraryFilter => ({ ...NO_FILTER, ...f });

  it("matches every word in any field, ignoring case", () => {
    expect(matches(row, filter({ query: "line 0042" }))).toBe(true);
    expect(matches(row, filter({ query: "TIGHT" }))).toBe(true);
    expect(matches(row, filter({ query: "cell bench" }))).toBe(true);
    expect(matches(row, filter({ query: "line nowhere" }))).toBe(false);
  });

  it("filters by verdict, including none, and by workflow", () => {
    expect(matches(row, filter({ verdicts: new Set(["pass"]) }))).toBe(true);
    expect(matches(row, filter({ verdicts: new Set(["fail", "warn"]) }))).toBe(false);
    const none = projectRow(entry("x", null));
    expect(matches(none, filter({ verdicts: new Set(["none"]) }))).toBe(true);
    expect(matches(row, filter({ topology: "rig_handeye" }))).toBe(true);
    expect(matches(row, filter({ topology: "planar_intrinsics" }))).toBe(false);
  });

  it("says when a filter is on", () => {
    expect(isFiltering(NO_FILTER)).toBe(false);
    expect(isFiltering(filter({ query: " a " }))).toBe(true);
    expect(isFiltering(filter({ query: "  " }))).toBe(false);
  });
});

describe("containerOf and querySpans", () => {
  const groups = groupLibrary([
    projectRow(entry("a", { id: "SN1", product: "P" })),
    projectRow(entry("loose", null)),
  ]);

  it("lists a project among its device's projects, or the ungrouped ones", () => {
    expect(containerOf(groups, projectNodeId("/lib/a"))).toBe(
      groups.products[0]!.devices[0]!.id,
    );
    expect(containerOf(groups, projectNodeId("/lib/loose"))).toBe("ungrouped");
    expect(containerOf(groups, "ungrouped")).toBe("ungrouped");
    expect(containerOf(groups, null)).toBeNull();
    expect(containerOf(groups, projectNodeId("/lib/missing"))).toBeNull();
  });

  it("marks every occurrence of every word", () => {
    expect(querySpans("Bench bench 2", "bench 2")).toEqual([
      { start: 0, end: 5, kind: "protected-term" },
      { start: 6, end: 11, kind: "protected-term" },
      { start: 12, end: 13, kind: "protected-term" },
    ]);
    expect(querySpans("abc", "")).toEqual([]);
  });
});

describe("openScreen and relativeToRoots", () => {
  it("opens on Inspect only when a run reads", () => {
    expect(openScreen(projectRow(entry("a", null)))).toBe("calibrate");
    expect(openScreen(projectRow(entry("a", null, [run("run-001", "pass", 0.1)])))).toBe(
      "inspect",
    );
  });

  it("names a project relative to the deepest root it is under", () => {
    expect(relativeToRoots("/data/a/b", ["/data", "/data/a"])).toBe("b");
    expect(relativeToRoots("/data", ["/data"])).toBe(".");
    expect(relativeToRoots("/else/x", ["/data"])).toBe("/else/x");
  });
});
