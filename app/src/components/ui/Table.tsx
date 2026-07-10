import type { TableHTMLAttributes, ThHTMLAttributes, TdHTMLAttributes } from "react";
import { cx } from "./cx";

export type TableSize = "sm" | "xs";

export interface TableProps extends TableHTMLAttributes<HTMLTableElement> {
  /** `sm` = 11px (PoseStatsTable's scale) · `xs` = 10px (the denser
   * cross-camera matrix). Kept distinct rather than forced to one size —
   * the matrix trades type size for grid density on purpose. */
  size?: TableSize;
}

const SIZE_CLASSES: Record<TableSize, string> = {
  sm: "text-[11px]",
  xs: "text-[10px]",
};

/** Shared table shell: border-collapse + font-mono + tabular-nums, the
 * part that was identical between PoseStatsTable and CameraResidualMatrix.
 * Row/cell coloring (hover, active, severity) stays with each caller —
 * this isn't a generic sortable-DataTable engine, just the repeated chrome. */
export function Table({ size = "sm", className, ...rest }: TableProps) {
  return (
    <table
      className={cx(
        "border-collapse font-mono tabular-nums",
        SIZE_CLASSES[size],
        className,
      )}
      {...rest}
    />
  );
}

export interface ThProps extends ThHTMLAttributes<HTMLTableCellElement> {
  align?: "left" | "right" | "center";
  /** Sticks the cell to the left edge on horizontal scroll (the matrix's
   * `cam\pose` corner + per-row camera index column). */
  sticky?: boolean;
}

const ALIGN_CLASSES = { left: "text-left", right: "text-right", center: "text-center" };

export function Th({
  align = "left",
  sticky,
  scope = "col",
  className,
  ...rest
}: ThProps) {
  return (
    <th
      scope={scope}
      className={cx(
        "font-medium",
        ALIGN_CLASSES[align],
        sticky && "sticky left-0 z-10 bg-surface",
        className,
      )}
      {...rest}
    />
  );
}

export interface TdProps extends TdHTMLAttributes<HTMLTableCellElement> {
  align?: "left" | "right" | "center";
}

export function Td({ align = "left", className, ...rest }: TdProps) {
  return <td className={cx(ALIGN_CLASSES[align], className)} {...rest} />;
}
