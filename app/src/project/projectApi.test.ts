import { describe, expect, it } from "vitest";
import { asProjectError, projectManifestPath, runExportPath } from "./projectApi";

describe("asProjectError", () => {
  it("keeps a backend ProjectError", () => {
    expect(asProjectError({ kind: "run_exists", message: "taken" })).toEqual({
      kind: "run_exists",
      message: "taken",
    });
  });

  it("reports anything else as an io error with its text", () => {
    expect(asProjectError(new Error("ipc down"))).toEqual({
      kind: "io",
      message: "ipc down",
    });
    expect(asProjectError("plain")).toEqual({ kind: "io", message: "plain" });
    expect(asProjectError({ kind: "bogus", message: "x" }).kind).toBe("io");
  });
});

describe("project paths", () => {
  it("names the manifest and a run's export with / separators", () => {
    expect(projectManifestPath("C:\\p")).toBe("C:/p/dataset.json");
    expect(runExportPath("/p/", "run-001")).toBe("/p/runs/run-001/export.json");
  });
});
