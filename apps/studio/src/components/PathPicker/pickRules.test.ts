import { describe, expect, it } from "vitest";
import type { DirEntry } from "../../api/generated/types";
import { breadcrumb, selectable, showsFiles } from "./pickRules";

const dir = (name: string, is_project = false): DirEntry => ({
  name,
  path: `/r/${name}`,
  kind: "dir",
  is_project,
});
const file = (name: string): DirEntry => ({
  name,
  path: `/r/${name}`,
  kind: "file",
  is_project: false,
  size: 1,
});

describe("pick rules", () => {
  it("offers what each mode asks for", () => {
    expect(selectable(dir("a"), "folder")).toBe(true);
    expect(selectable(dir("a"), "project")).toBe(false);
    expect(selectable(dir("p", true), "project")).toBe(true);
    expect(selectable(file("dataset.TOML"), "manifest")).toBe(true);
    expect(selectable(file("export.json"), "manifest")).toBe(true);
    expect(selectable(file("a.png"), "manifest")).toBe(false);
    expect(selectable(file("dataset.toml"), "export")).toBe(false);
    expect(selectable(file("export.json"), "export")).toBe(true);
    expect(showsFiles("folder")).toBe(false);
    expect(showsFiles("export")).toBe(true);
  });

  it("builds the breadcrumb from the root a path lies under", () => {
    expect(breadcrumb(null, ["/data"])).toEqual([{ label: "Folders", path: null }]);
    expect(breadcrumb("/data/rigs/a", ["/data", "/data/rigs/other"])).toEqual([
      { label: "Folders", path: null },
      { label: "/data", path: "/data" },
      { label: "rigs", path: "/data/rigs" },
      { label: "a", path: "/data/rigs/a" },
    ]);
    // A sibling with the root's name as a prefix is not under it.
    expect(breadcrumb("/database/x", ["/data"]).map((c) => c.path)).toEqual([
      null,
      "/database/x",
    ]);
    expect(breadcrumb("C:\\d\\e", ["C:\\d"]).map((c) => c.label)).toEqual([
      "Folders",
      "C:\\d",
      "e",
    ]);
  });
});
