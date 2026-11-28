import { describe, expect, it } from "vitest";
import { groupLibrary, projectRow } from "../library/libraryModel";
import { SN1, SN2 } from "./__fixtures__/deviceFixtures";
import {
  cameraIds,
  deviceParamsOf,
  fromParams,
  historyRows,
  rmsOf,
  runKey,
  seedLine,
  selectorOf,
  targetOfDevice,
  toParams,
  typeOf,
  unitRows,
  whenText,
} from "./deviceModel";

const view = (query: string) => fromParams(new URLSearchParams(query));

describe("the address", () => {
  it("names a device, a type, the tab and the switches", () => {
    expect(view("product=Stereo-X1&device=SN-0042").target).toEqual({
      scope: "device",
      product: "Stereo-X1",
      id: "SN-0042",
    });
    expect(view("product=Stereo-X1&scope=type&tab=seed&starred=1&x=time")).toEqual({
      target: { scope: "type", product: "Stereo-X1", id: null },
      tab: "seed",
      starredOnly: true,
      xMode: "time",
    });
  });

  it("falls back to the open project's device, History and run order", () => {
    expect(view("")).toEqual({
      target: null,
      tab: "history",
      starredOnly: false,
      xMode: "order",
    });
    // A type needs its product; an unknown tab is History.
    expect(view("scope=type").target).toBeNull();
    expect(view("tab=nonsense").tab).toBe("history");
  });

  it("round-trips, leaving the project and run alone and dropping defaults", () => {
    const params = new URLSearchParams("project=%2Fp&run=run-001&tab=trends");
    const next = toParams(
      params,
      view("product=Stereo-X1&scope=type&x=time&starred=1&tab=trends"),
    );
    expect(next.get("project")).toBe("/p");
    expect(next.get("scope")).toBe("type");
    expect(next.get("x")).toBe("time");
    expect(next.get("starred")).toBe("1");
    const cleared = toParams(next, view(""));
    expect(cleared.toString()).toBe("project=%2Fp&run=run-001");
  });
});

describe("what is asked for", () => {
  it("a device by its id and type, a type by its product", () => {
    const device = targetOfDevice({ id: "SN-0042", product: " Stereo-X1 " })!;
    expect(selectorOf(device)).toEqual({
      kind: "device",
      id: "SN-0042",
      product: "Stereo-X1",
    });
    expect(selectorOf({ scope: "device", product: null, id: "X" })).toEqual({
      kind: "device",
      id: "X",
    });
    expect(selectorOf(typeOf(device)!)).toEqual({ kind: "type", product: "Stereo-X1" });
    expect(typeOf({ scope: "device", product: null, id: "X" })).toBeNull();
    expect(targetOfDevice(null)).toBeNull();
  });
});

describe("the history", () => {
  it("lists the runs in the library's order with their place in the history", () => {
    const rows = historyRows(SN1, false);
    expect(rows.map((r) => [r.order, r.run.id])).toEqual([
      [0, "run-001"],
      [1, "run-002"],
      [2, "run-001"],
      [3, "run-002"],
    ]);
    // Ids repeat across projects: the key is the project's too.
    expect(new Set(rows.map((r) => r.key)).size).toBe(4);
    expect(rows[1]?.key).toBe(runKey("/data/bench", "run-002"));
  });

  it("keeps the starred runs without renumbering them", () => {
    expect(historyRows(SN1, true).map((r) => [r.order, r.run.id])).toEqual([
      [1, "run-002"],
    ]);
  });

  it("reads the cameras and their RMS", () => {
    expect(cameraIds(historyRows(SN1, false))).toEqual(["cam0", "cam1"]);
    expect(rmsOf(SN1.runs[1]!, "cam1")).toBe(0.247);
    expect(rmsOf(SN1.runs[3]!, "cam0")).toBeNull();
  });

  it("says what a seeded run started from, applied and ignored", () => {
    const line = seedLine(SN1.runs[2]!.seed)!;
    expect(line.text).toBe("the calibrated run run-002 · 2 applied · 1 ignored");
    expect(line.title).toContain("applied: cam0.lens, cam1.lens");
    expect(line.title).toContain("ignored: handeye");
    expect(seedLine(null)).toBeNull();
  });

  it("writes a time, or nothing", () => {
    expect(whenText(null)).toBe("–");
    expect(whenText(1_700_000_000)).toMatch(/^2023-11-1[45] \d\d:\d\d$/);
  });
});

describe("the units of a type", () => {
  it("is a row per unit with its representative run", () => {
    const rows = unitRows([SN1, SN2]);
    expect(rows.map((r) => [r.device.id, r.runs, r.projects, r.starred])).toEqual([
      ["SN-0042", 4, 2, 1],
      ["SN-0057", 1, 1, 0],
    ]);
    expect(rows.map((r) => r.representative?.id)).toEqual(["run-002", "run-001"]);
  });
});

describe("a Library node's device", () => {
  const entry = (name: string, device: { id: string; product?: string } | null) =>
    projectRow({ name, dir: `/data/${name}`, device, runs: [], topology: null });
  const groups = groupLibrary([
    entry("a", { id: "SN1", product: "X1" }),
    entry("b", { id: "SN2" }),
  ]);

  it("opens a device, or a type", () => {
    const device = groups.products.find((p) => p.label === "X1")!.devices[0]!;
    expect(deviceParamsOf(groups, device.id)?.toString()).toBe("product=X1&device=SN1");
    expect(deviceParamsOf(groups, "product:X1")?.toString()).toBe(
      "product=X1&scope=type",
    );
  });

  it("has none for other nodes or a type nobody names", () => {
    expect(deviceParamsOf(groups, null)).toBeNull();
    expect(deviceParamsOf(groups, "ungrouped")).toBeNull();
    expect(deviceParamsOf(groups, "project:/data/a")).toBeNull();
    expect(deviceParamsOf(groups, "product:(unspecified type)")).toBeNull();
  });
});
