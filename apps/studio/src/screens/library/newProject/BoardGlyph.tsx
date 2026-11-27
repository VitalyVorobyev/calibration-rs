import type { TargetSpec } from "../../../api/generated/types";
import { glyphGrid } from "./glyphGrid";

/** The lattice a target spec describes, drawn small: a chessboard as
 * squares, any other family as the grid of its marks. It is a sketch of the
 * numbers (rows, columns), not of the print. Colours are the app's tokens. */
export function BoardGlyph({ target }: { target: TargetSpec }) {
  const grid = glyphGrid(target);
  if (grid === null) {
    return (
      <div className="grid h-28 w-full place-items-center text-xs text-fg-subtle">
        no size to draw yet
      </div>
    );
  }
  const unit = 100 / Math.max(grid.cols, grid.rows);
  const w = grid.cols * unit;
  const h = grid.rows * unit;
  const cells: React.ReactNode[] = [];
  for (let r = 0; r < grid.rows; r++) {
    for (let c = 0; c < grid.cols; c++) {
      const x = c * unit;
      const y = r * unit;
      if (grid.shape === "squares") {
        cells.push(
          <rect
            key={`${r}-${c}`}
            x={x}
            y={y}
            width={unit}
            height={unit}
            className={(r + c) % 2 === 0 ? "fill-fg" : "fill-surface"}
          />,
        );
      } else {
        cells.push(
          <circle
            key={`${r}-${c}`}
            cx={x + unit / 2}
            cy={y + unit / 2}
            r={unit * 0.28}
            className="fill-fg"
          />,
        );
      }
    }
  }
  return (
    <svg
      role="img"
      aria-label={`${grid.cols} by ${grid.rows} ${grid.shape === "squares" ? "squares" : "marks"}`}
      viewBox={`-1 -1 ${w + 2} ${h + 2}`}
      className="w-full max-w-44 rounded-sm bg-surface ring-1 ring-line"
    >
      {cells}
      <rect
        x={0}
        y={0}
        width={w}
        height={h}
        className="fill-none stroke-line-strong"
        strokeWidth={0.6}
      />
    </svg>
  );
}
