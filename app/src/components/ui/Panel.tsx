import type { HTMLAttributes } from "react";
import { cx } from "./cx";

export type PanelTag = "div" | "aside" | "section";

export interface PanelProps extends HTMLAttributes<HTMLElement> {
  /** Rendered element — `aside` for side panels, `section` for grouped
   * content, `div` (default) otherwise. */
  as?: PanelTag;
}

/** The bordered card surface used for side panels and grouped content
 * (Diagnose's stats rail, the 3D viewer's info rail). Layout (flex
 * direction, gap, width, overflow) stays with the caller via
 * `className` — this only owns the border/background/radius/padding
 * that every such panel shared verbatim before this pass. */
export function Panel({ as = "div", className, ...rest }: PanelProps) {
  const Tag = as;
  return (
    <Tag
      className={cx("rounded-md border border-border bg-surface p-3", className)}
      {...rest}
    />
  );
}
