import { describe, expect, it } from "vitest";
import type { ProjectEntry } from "../../api/generated/types";
import { calibratedRun } from "../../test/projectFixtures";
import { projectRow } from "./libraryModel";
import { rowDetails, visibleColumns } from "./tableColumns";

const keys = (width: number | null) => [...visibleColumns(width)];

describe("visibleColumns", () => {
  it("keeps every column where the width is not known or is wide", () => {
    expect(visibleColumns(null).size).toBe(8);
    expect(visibleColumns(900).size).toBe(8);
  });

  it("gives up the run and views, then the date, then the workflow", () => {
    expect(keys(899)).toEqual([
      "project",
      "workflow",
      "verdict",
      "rms",
      "updated",
      "actions",
    ]);
    expect(keys(699)).toEqual(["project", "workflow", "verdict", "rms", "actions"]);
    expect(keys(559)).toEqual(["project", "verdict", "rms", "actions"]);
  });

  it("always keeps the name, the verdict, the RMS and the actions", () => {
    for (const width of [0, 300, 600, 800, 1200]) {
      const kept = visibleColumns(width);
      for (const key of ["project", "verdict", "rms", "actions"]) {
        expect(kept.has(key)).toBe(true);
      }
    }
  });
});

describe("rowDetails", () => {
  const entry = (runs: ProjectEntry["runs"]): ProjectEntry => ({
    name: "a",
    dir: "/lib/a",
    device: null,
    runs,
    topology: "planar_intrinsics",
  });

  it("says what the dropped columns say", () => {
    const text = rowDetails(
      projectRow(
        entry([calibratedRun("/lib/a", "run-001"), calibratedRun("/lib/a", "run-002")]),
      ),
    );
    expect(text).toContain("Planar intrinsics");
    expect(text).toContain("latest run run-002 of 2");
    expect(text).toContain("9/10 views");
  });

  it("says there is no run", () => {
    expect(rowDetails(projectRow(entry([])))).toBe("Planar intrinsics · no run");
  });
});
