import { cn } from "@vitavision/ui";
import { Ban, ImageOff, TriangleAlert } from "lucide-react";
import type { CellStatus, CellTone } from "./detectModel";

const TEXT: Record<CellTone, string> = {
  normal: "text-fg",
  warn: "text-warn",
  defect: "text-defect",
  pending: "text-fg-subtle",
  excluded: "text-fg-subtle",
  missing: "text-defect",
};

const BAR: Record<CellTone, string> = {
  normal: "bg-normal",
  warn: "bg-warn",
  defect: "bg-defect",
  pending: "bg-line",
  excluded: "bg-line",
  missing: "bg-defect",
};

/** One cell of the overview matrix: the number of features found in mono, a
 * bar scaled to the board's expected count, and a mark where something is
 * wrong. An excluded cell is struck through and stays readable; a cell with
 * no image says so; a cell not detected yet shows `·`. The whole cell's
 * meaning is also a title and the matrix's accessible name, never colour
 * alone. */
export function CountCell({ status }: { status: CellStatus }) {
  const { tone } = status;
  return (
    <span
      title={status.sentence}
      className={cn(
        "relative flex size-full flex-col items-center justify-center gap-0.5 px-1",
        tone === "defect" || tone === "missing" ? "bg-defect/8" : "",
        tone === "warn" ? "bg-warn/8" : "",
        tone === "excluded" ? "bg-raised" : "",
      )}
    >
      {tone === "missing" ? (
        <ImageOff aria-hidden className="size-4 text-defect" />
      ) : (
        <span
          className={cn(
            "font-mono text-xs tabular-nums",
            TEXT[tone],
            tone === "excluded" && "line-through",
          )}
        >
          {status.text}
        </span>
      )}
      {status.fraction !== null && tone !== "excluded" && (
        <span aria-hidden className="h-1 w-3/5 overflow-hidden rounded-full bg-line">
          <span
            className={cn("block h-full", BAR[tone])}
            style={{ width: `${Math.round(status.fraction * 100)}%` }}
          />
        </span>
      )}
      {(tone === "warn" || tone === "defect") && status.count !== null && (
        <TriangleAlert
          aria-hidden
          className={cn(
            "absolute top-0.5 right-0.5 size-3",
            tone === "warn" ? "text-warn" : "text-defect",
          )}
        />
      )}
      {tone === "excluded" && (
        <Ban aria-hidden className="absolute top-0.5 right-0.5 size-3 text-fg-subtle" />
      )}
    </span>
  );
}
