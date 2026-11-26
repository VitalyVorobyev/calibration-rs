import { describe, expect, it } from "vitest";
import { StudioError } from "../api";
import { asProjectError, projectManifestPath, runExportPath } from "./projectApi";

describe("asProjectError", () => {
  it("keeps the back end's error envelope", () => {
    expect(asProjectError({ kind: "run_exists", message: "taken" })).toEqual({
      kind: "run_exists",
      message: "taken",
    });
    expect(
      asProjectError(new StudioError({ kind: "not_a_project", message: "no project" })),
    ).toEqual({ kind: "not_a_project", message: "no project" });
  });

  it("reports anything else as an io error with its text", () => {
    expect(asProjectError(new Error("ipc down"))).toEqual({
      kind: "io",
      message: "ipc down",
    });
    expect(asProjectError("plain")).toEqual({ kind: "io", message: "plain" });
  });
});

describe("project paths", () => {
  it("names the manifest and a run's export with / separators", () => {
    expect(projectManifestPath("C:\\p")).toBe("C:/p/dataset.json");
    expect(runExportPath("/p/", "run-001")).toBe("/p/runs/run-001/export.json");
  });
});
