/** What every per-camera tab of Inspect shares: the run shown, the camera
 * (the app-wide selection's, else the first the run has), and the words for
 * the two states before there is anything to show. */
import { Empty, Select, Skeleton } from "@vitavision/ui";
import type { ReactNode } from "react";
import { useProject } from "../../project/projectStore";
import { useSelection } from "../../session/selection";
import { useStore } from "../../store";
import { useTarget, type Target } from "./target";

/** The camera chooser, in the toolbar's size. */
export function CameraSelect({ target }: { target: Target }) {
  return (
    <Select
      aria-label="Camera"
      value={String(target.camera)}
      onValueChange={(v) => v && useSelection.getState().selectCamera(Number(v))}
      options={target.cameras.map((c) => ({ value: String(c), label: `Camera ${c}` }))}
      className="w-36"
    />
  );
}

/** A per-camera tab: `title`, the camera chooser (when the run has more
 * than one camera) and `children(target)`; the plain words when no run is
 * shown. */
export function CameraFrame({
  title,
  children,
}: {
  title: string;
  children: (target: Target) => ReactNode;
}) {
  const { target } = useTarget();
  // A run that failed to load is said by the shell, above the tabs.
  const failed = useStore((s) => s.loadError !== null);
  // A project with runs is opening one: the shell reads it in.
  const runLoading = useProject((s) => (s.project?.runs.length ?? 0) > 0);
  if (target === null) {
    if (failed) return null;
    return runLoading ? (
      <div className="flex flex-col gap-2" aria-busy="true">
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-64 w-full" />
      </div>
    ) : (
      <Empty>
        No run is shown. Choose one in the run list, or calibrate the project first.
      </Empty>
    );
  }
  return (
    <section aria-label={title} className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-sm font-medium text-fg">{title}</h2>
        {target.cameras.length > 1 && <CameraSelect target={target} />}
        <span className="font-mono text-xs text-fg-muted">
          {target.run.id}
          {target.cameras.length === 1 ? ` · camera ${target.camera}` : ""}
        </span>
      </div>
      {children(target)}
    </section>
  );
}
