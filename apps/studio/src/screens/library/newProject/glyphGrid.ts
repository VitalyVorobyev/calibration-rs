/** The size of the sketch of a target: how many cells across and down and
 * whether they are squares (a chessboard: `cols` and `rows` count interior
 * corners, so the board has one more square each way) or marks. `null` for
 * a target with no size (the sniffer's placeholder). */
import type { TargetSpec } from "../../../api/generated/types";

export interface GlyphGrid {
  cols: number;
  rows: number;
  shape: "squares" | "marks";
}

/** At most this many cells either way, so a large ring grid stays a sketch. */
const MAX_CELLS = 24;

export function glyphGrid(target: TargetSpec): GlyphGrid | null {
  const t = target as unknown as Record<string, unknown>;
  const num = (...keys: string[]): number | null => {
    for (const k of keys) {
      const v = t[k];
      if (typeof v === "number" && Number.isFinite(v) && v > 0) return Math.floor(v);
    }
    return null;
  };
  const rows = num("rows");
  const cols = num("cols", "long_row_cols");
  if (rows === null || cols === null) return null;
  const clamp = (n: number) => Math.min(n, MAX_CELLS);
  if (target.kind === "chessboard") {
    return { cols: clamp(cols + 1), rows: clamp(rows + 1), shape: "squares" };
  }
  // A ChArUco board counts squares; a ring grid counts marks.
  return {
    cols: clamp(cols),
    rows: clamp(rows),
    shape: target.kind === "charuco" ? "squares" : "marks",
  };
}
