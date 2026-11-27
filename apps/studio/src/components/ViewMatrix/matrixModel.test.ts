import { describe, expect, it } from "vitest";
import type { ViewRow } from "../../api/generated/types";
import {
  cameraSummaries,
  filterViews,
  findView,
  hasGap,
  imagePath,
  missingCameras,
  NO_ROW_FILTER,
} from "./matrixModel";

const views: ViewRow[] = [
  { token: "0001", images: ["cam0/0001.png", "cam1/0001.png"] },
  { token: "0002", images: ["cam0/0002.png", null] },
  { token: "0010", images: [null, "cam1/0010.png"] },
];

describe("matrixModel", () => {
  it("joins a cell's image onto the data folder, as the image resource wants", () => {
    expect(imagePath("/data/set1", views[0]!, 1)).toBe("/data/set1/cam1/0001.png");
    expect(imagePath("/data/set1", views[1]!, 1)).toBeNull();
    expect(imagePath("/data", { token: "x", images: ["/abs/a.png"] }, 0)).toBe(
      "/abs/a.png",
    );
  });

  it("finds the gaps of a view", () => {
    expect(hasGap(views[0]!)).toBe(false);
    expect(missingCameras(views[1]!)).toEqual([1]);
    expect(missingCameras(views[2]!)).toEqual([0]);
  });

  it("filters rows by problem and by token text", () => {
    expect(filterViews(views, NO_ROW_FILTER)).toHaveLength(3);
    expect(
      filterViews(views, { problemsOnly: true, token: "" }).map((v) => v.token),
    ).toEqual(["0002", "0010"]);
    expect(
      filterViews(views, { problemsOnly: false, token: " 001 " }).map((v) => v.token),
    ).toEqual(["0001", "0010"]);
  });

  it("finds a view by token and counts each camera's images", () => {
    expect(findView(views, "0010")?.index).toBe(2);
    expect(findView(views, "nope")).toBeNull();
    expect(cameraSummaries({ cameras: ["a", "b"], views })).toEqual([
      { id: "a", images: 2 },
      { id: "b", images: 2 },
    ]);
  });
});
