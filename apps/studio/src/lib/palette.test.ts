import { describe, expect, it } from "vitest";
import { filterItems, type PaletteItem } from "./palette";

const items: PaletteItem[] = [
  { id: "s-inspect", label: "Inspect", group: "Screens" },
  { id: "s-calibrate", label: "Calibrate", group: "Screens" },
  { id: "r-7", label: "run-007 · tight loss", group: "Runs", keywords: "starred" },
  { id: "r-8", label: "run-008", group: "Runs" },
  { id: "c-1", label: "Camera 1", group: "Cameras" },
  { id: "p-1", label: "rig-demo", group: "Projects", keywords: "/data/rig-demo" },
];

describe("filterItems", () => {
  it("keeps everything, in order, for an empty query", () => {
    expect(filterItems(items, "  ").map((i) => i.id)).toEqual(items.map((i) => i.id));
  });

  it("matches the start of any word of the label or keywords", () => {
    expect(filterItems(items, "loss").map((i) => i.id)).toEqual(["r-7"]);
    expect(filterItems(items, "star").map((i) => i.id)).toEqual(["r-7"]);
    expect(filterItems(items, "demo").map((i) => i.id)).toEqual(["p-1"]);
  });

  it("needs every word of the query", () => {
    expect(filterItems(items, "run tight").map((i) => i.id)).toEqual(["r-7"]);
    expect(filterItems(items, "run nothing")).toEqual([]);
  });

  it("is case-insensitive and does not match the middle of a word", () => {
    expect(filterItems(items, "CAM").map((i) => i.id)).toEqual(["c-1"]);
    expect(filterItems(items, "nspect")).toEqual([]);
  });

  it("puts labels that start with the query first", () => {
    const list: PaletteItem[] = [
      { id: "a", label: "Open run-001", group: "x" },
      { id: "b", label: "run-001", group: "x" },
    ];
    expect(filterItems(list, "run").map((i) => i.id)).toEqual(["b", "a"]);
  });
});
