import { describe, expect, it } from "vitest";
import type { TargetSpec } from "../../../api/generated/types";
import { glyphGrid } from "./glyphGrid";

const target = (t: Record<string, unknown>) => t as unknown as TargetSpec;

describe("glyphGrid", () => {
  it("counts a chessboard's squares one more than its interior corners", () => {
    expect(glyphGrid(target({ kind: "chessboard", rows: 5, cols: 7 }))).toEqual({
      cols: 8,
      rows: 6,
      shape: "squares",
    });
  });

  it("draws a ChArUco board as squares and a ring grid as marks", () => {
    expect(glyphGrid(target({ kind: "charuco", rows: 4, cols: 6 }))).toMatchObject({
      cols: 6,
      shape: "squares",
    });
    expect(glyphGrid(target({ kind: "ringgrid", rows: 9, long_row_cols: 15 }))).toEqual({
      cols: 15,
      rows: 9,
      shape: "marks",
    });
  });

  it("has nothing to draw for the sniffer's placeholder or an unsized target", () => {
    expect(glyphGrid(target({ kind: "chessboard", rows: 0, cols: 0 }))).toBeNull();
    expect(glyphGrid(target({ kind: "puzzleboard", cell_size_m: 0.01 }))).toBeNull();
  });

  it("keeps a huge grid a sketch", () => {
    expect(glyphGrid(target({ kind: "chessboard", rows: 100, cols: 3 }))).toMatchObject({
      rows: 24,
      cols: 4,
    });
  });
});
