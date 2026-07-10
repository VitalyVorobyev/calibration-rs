import type { ReactNode } from "react";

export interface EmptyStateProps {
  /** Workspace title, repeated in the header even while empty so the
   * workspace never looks unlabeled. */
  title: string;
  body: ReactNode;
}

/** Whole-workspace "no data loaded yet" placeholder: title + a centered
 * dashed-border message box filling the remaining space. Viewer3D,
 * Epipolar and Depth each hand-rolled this identically; DiagnoseWorkspace's
 * inline canvas-area placeholder is a different, smaller idiom (a note
 * inside an already-visible canvas area, not a whole-workspace state) and
 * is intentionally left as-is. */
export function EmptyState({ title, body }: EmptyStateProps) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <p className="max-w-[28rem] p-6 text-center text-[13px] text-muted-foreground">
          {body}
        </p>
      </div>
    </div>
  );
}
