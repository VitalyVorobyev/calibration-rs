import type { ReactNode } from "react";

/** The one toolbar of the viewers: a wrapping row of labelled groups of
 * controls (segmented controls, chips, selects), in the toolbar size. */
export function ViewerToolbar({
  label,
  children,
}: {
  /** Names the toolbar for assistive technology. */
  label: string;
  children: ReactNode;
}) {
  return (
    <div
      role="toolbar"
      aria-label={label}
      className="flex flex-wrap items-center gap-x-4 gap-y-1.5"
    >
      {children}
    </div>
  );
}

/** One group of a `ViewerToolbar`: a small caption and its controls. */
export function ToolbarGroup({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div role="group" aria-label={label} className="flex items-center gap-1.5">
      <span className="text-[11px] uppercase tracking-wider text-fg-muted">{label}</span>
      {children}
    </div>
  );
}
