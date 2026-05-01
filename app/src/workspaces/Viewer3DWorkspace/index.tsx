import { useMemo, useState } from "react";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportShape";
import type { FrameKey } from "../../types";
import { Scene } from "./Scene";

/** Build a per-camera (width, height) map from the manifest's ROI
 * entries. For tiled rigs (puzzle 130×130: one 4320×540 PNG per pose
 * with six 720×540 ROIs) this gives every frustum its real sensor
 * dimensions; for single-image-per-camera shapes the ROI is absent and
 * the camera falls back to the workspace default. */
function cameraDimensionsFromFrames(
  frames: FrameKey[],
): Map<number, { width: number; height: number }> {
  const dims = new Map<number, { width: number; height: number }>();
  for (const f of frames) {
    if (dims.has(f.camera)) continue;
    if (f.roi) dims.set(f.camera, { width: f.roi.w, height: f.roi.h });
  }
  return dims;
}

/** B1.2 — 3D rig scene. Renders the rig origin, per-camera frustums,
 * and the active pose's target board. Click a frustum to select that
 * camera across all workspaces; click a board to select that pose. */
export function Viewer3DWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);
  const frames = useStore((s) => s.frames);
  const cameraA = useStore((s) => s.cameraA);
  const selectedPose = useStore((s) => s.selectedPose);
  const [showAllPoses, setShowAllPoses] = useState(false);

  const cameraDimensions = useMemo(
    () => cameraDimensionsFromFrames(frames),
    [frames],
  );

  if (!data || !kind) {
    return <Empty body="Load a rig export to see cameras and target poses in 3D." />;
  }

  const isRig =
    Array.isArray(data.cameras) &&
    Array.isArray(data.cam_se3_rig) &&
    Array.isArray(data.rig_se3_target) &&
    data.cameras.length > 0;

  if (!isRig) {
    return (
      <Empty
        body={`The 3D viewer needs a rig export (cameras + cam_se3_rig + rig_se3_target). The current export is ${exportKindLabel(
          kind,
        )} — single-camera shapes will land in a follow-up.`}
      />
    );
  }

  const numCameras = data.cameras?.length ?? 0;
  const numPoses = data.rig_se3_target?.length ?? 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">3D viewer</h2>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowAllPoses((v) => !v)}
            className={`h-7 px-2 font-mono text-[11px] ${
              showAllPoses ? "border-brand text-brand" : ""
            }`}
            title="Toggle ghost rendering of every target pose"
          >
            {showAllPoses ? "All poses ✓" : "All poses"}
          </button>
          <span className="font-mono text-[11px] text-muted-foreground">
            {exportKindLabel(kind)}
          </span>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1 overflow-hidden rounded-md border border-border">
        <Scene
          data={data}
          showAllPoses={showAllPoses}
          cameraDimensions={cameraDimensions}
          fallbackImage={{ width: 1024, height: 768 }}
        />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between border-t border-border bg-bg-soft/80 px-3 py-1.5 font-mono text-[10px] text-muted-foreground backdrop-blur-sm">
          <span>
            cameras {numCameras} · poses {numPoses}
          </span>
          <span>
            cam {cameraA} · pose {selectedPose}
          </span>
          <span>drag · scroll to zoom · click a frustum to select</span>
        </div>
      </div>
    </div>
  );
}

function Empty({ body }: { body: string }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-tight">3D viewer</h2>
      </header>
      <div className="flex min-h-0 flex-1 items-center justify-center rounded-md border border-dashed border-border bg-bg-soft">
        <p className="max-w-[28rem] p-6 text-center text-[13px] text-muted-foreground">
          {body}
        </p>
      </div>
    </div>
  );
}
