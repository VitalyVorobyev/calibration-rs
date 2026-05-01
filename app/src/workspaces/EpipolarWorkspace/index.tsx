import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportShape";

/** Placeholder for B2's epipolar-line workspace. Two FrameCanvas panes
 * + click-to-overlay epipolar lines computed server-side via a Tauri
 * command (`compute_epipolar_overlay`). */
export function EpipolarWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);

  if (!data || !kind) {
    return (
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <p className="max-w-[28rem] p-6 text-center text-[13px] text-muted-foreground">
          Load a rig export to inspect epipolar geometry between cameras.
          B2 will let you click a feature in pane A and see the matching
          epipolar line in pane B.
        </p>
      </div>
    );
  }

  const isRig =
    kind === "rig_extrinsics" ||
    kind === "rig_handeye" ||
    kind === "rig_laserline_device";

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">
          Epipolar geometry
        </h2>
        <span className="font-mono text-[11px] text-muted-foreground">
          {exportKindLabel(kind)}
        </span>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <div className="max-w-[36rem] p-6 text-center">
          <h3 className="mb-2 text-sm font-semibold">
            Two-pane epipolar view lands in B2
          </h3>
          <p className="text-[12px] text-muted-foreground">
            Pick pose + cam A + cam B; click a feature in pane A; a Tauri
            command computes the epipolar line through the rig's relative
            extrinsics and distortion-aware projection, then renders it as
            an SVG polyline over pane B.
          </p>
          {!isRig && (
            <p className="mt-4 rounded-md border-l-2 border-brand bg-brand/[0.06] p-2.5 text-left text-[11px] text-foreground">
              The current export is single-camera ({exportKindLabel(kind)}).
              Epipolar geometry only applies to rig exports
              (rig_extrinsics, rig_handeye, rig_laserline_device).
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
