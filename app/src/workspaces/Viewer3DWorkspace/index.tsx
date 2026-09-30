import {
  DensityProvider,
  Empty,
  Panel,
  Select,
  ToggleChip,
  Tooltip,
} from "@vitavision/ui";
import { useMemo, useState } from "react";
import { PoseStepper } from "../../components/PoseStepper";
import {
  iso3DistanceM,
  iso3EulerXYZDeg,
  iso3RotationAngleDeg,
  relativeCameraPose,
  targetInCameraPose,
} from "../../lib/se3";
import { adaptSceneExport } from "../../lib/sceneExport";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportKind";
import type { Iso3Wire, PinholeCameraWire } from "../../store/types";
import type { FrameKey } from "../../types";
import { Scene } from "./Scene";

/** The quiet right-aligned caption in an info-rail panel's header. */
const SUBTITLE = "font-mono text-[10px] text-fg-muted";

/** 3D rig scene + side panel that breaks down the
 * selected camera's intrinsics, the camera→target extrinsic for the
 * active pose, and the selected camera's pose relative to a chosen
 * reference camera (default cam 0). Click a frustum to pick a camera;
 * click a board to pick a pose. */
export function Viewer3DWorkspace() {
  const data = useStore((s) => s.data);
  const kind = useStore((s) => s.kind);
  const frames = useStore((s) => s.frames);
  const cameraA = useStore((s) => s.cameraA);
  const selectedPose = useStore((s) => s.selectedPose);
  const setSelectedPose = useStore((s) => s.setSelectedPose);
  const [showAllPoses, setShowAllPoses] = useState(false);
  const [showLaserPlanes, setShowLaserPlanes] = useState(true);
  const [referenceCamera, setReferenceCamera] = useState<number>(0);

  const cameraDimensions = useMemo(() => cameraDimensionsFromFrames(frames), [frames]);

  // Rig exports render directly; a single-camera laserline export is
  // lifted into a one-camera rig at the origin so its laser plane + poses
  // show up too.
  const sceneData = useMemo(() => (data ? adaptSceneExport(data) : null), [data]);

  if (!data || !kind || !sceneData) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <h2 className="text-sm font-semibold tracking-tight text-fg">3D viewer</h2>
        <Empty className="flex-1 justify-center rounded-panel border border-dashed border-line bg-surface px-6 [&>p]:max-w-md">
          Load a rig export to see cameras and target poses in 3D.
        </Empty>
      </div>
    );
  }

  const camerasArr = sceneData.cameras;
  const camSe3Rig = sceneData.cam_se3_rig;
  const rigSe3Target = sceneData.rig_se3_target;
  const isRig =
    Array.isArray(camerasArr) &&
    Array.isArray(camSe3Rig) &&
    Array.isArray(rigSe3Target) &&
    camerasArr.length > 0;

  if (!isRig) {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-3">
        <h2 className="text-sm font-semibold tracking-tight text-fg">3D viewer</h2>
        <Empty className="flex-1 justify-center rounded-panel border border-dashed border-line bg-surface px-6 [&>p]:max-w-md">
          {`The 3D viewer needs a rig export (cameras + cam_se3_rig + rig_se3_target) or a single-camera laserline device. The current export is ${exportKindLabel(
            kind,
          )} — its remaining single-camera shapes will land in a follow-up.`}
        </Empty>
      </div>
    );
  }

  const numCameras = camerasArr.length;
  const numPoses = rigSe3Target.length;
  const hasLaserPlanes = (sceneData.laser_planes_rig?.length ?? 0) > 0;
  const cameraIndices = Array.from({ length: numCameras }, (_, i) => i);
  const poseIndices = Array.from({ length: numPoses }, (_, i) => i);

  // Bound the picked indices to what the export actually has so an
  // older selection from a previous export doesn't render an out-of-
  // range readout.
  const safeCameraA = cameraA < numCameras ? cameraA : 0;
  const safeRefCamera = referenceCamera < numCameras ? referenceCamera : 0;
  const safePose = selectedPose < numPoses ? selectedPose : 0;

  const selectedCameraData = camerasArr[safeCameraA];
  const selectedCamSe3Rig = camSe3Rig[safeCameraA];
  const referenceCamSe3Rig = camSe3Rig[safeRefCamera];
  const targetPose = rigSe3Target[safePose];

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold tracking-tight">3D viewer</h2>
        <div className="flex flex-wrap items-center gap-3">
          {numPoses > 0 && (
            <PoseStepper
              poseValues={poseIndices}
              selectedPose={safePose}
              onSelectPose={(next) => setSelectedPose(next, "A")}
            />
          )}
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[11px] tracking-wider text-fg-muted uppercase">
              ref cam
            </span>
            <Tooltip content="Reference camera for the relative-pose readout">
              <span>
                <Select
                  aria-label="ref cam"
                  className="h-7 w-16 font-mono text-xs"
                  value={String(safeRefCamera)}
                  options={cameraIndices.map((i) => ({
                    value: String(i),
                    label: String(i),
                  }))}
                  onValueChange={(v) => setReferenceCamera(Number(v))}
                />
              </span>
            </Tooltip>
          </div>
          <ToggleChip
            checked={showAllPoses}
            onCheckedChange={setShowAllPoses}
            title="Toggle ghost rendering of every target pose"
          >
            All poses
          </ToggleChip>
          {hasLaserPlanes && (
            <ToggleChip
              checked={showLaserPlanes}
              onCheckedChange={setShowLaserPlanes}
              title="Toggle the calibrated laser planes (rig frame)"
            >
              Laser planes
            </ToggleChip>
          )}
          <span className="font-mono text-[11px] text-fg-muted">
            {exportKindLabel(kind)}
          </span>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_18rem] gap-3 overflow-hidden">
        <div className="relative overflow-hidden rounded-control border border-line">
          <Scene
            data={sceneData}
            showAllPoses={showAllPoses}
            showLaserPlanes={hasLaserPlanes && showLaserPlanes}
            cameraDimensions={cameraDimensions}
            fallbackImage={{ width: 1024, height: 768 }}
          />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between border-t border-line bg-raised/80 px-3 py-1.5 font-mono text-[10px] text-fg-muted backdrop-blur-sm">
            <span>
              cameras {numCameras} · poses {numPoses}
            </span>
            <span>
              cam {safeCameraA} · pose {safePose} · ref {safeRefCamera}
            </span>
            <span>drag · scroll to zoom · click a frustum to select</span>
          </div>
        </div>

        <DensityProvider value="compact">
          <aside className="flex min-h-0 flex-col gap-2.5 overflow-y-auto text-[12px]">
            <CameraIntrinsicsPanel
              cameraIndex={safeCameraA}
              camera={selectedCameraData}
            />
            <TargetExtrinsicsPanel
              cameraIndex={safeCameraA}
              poseIndex={safePose}
              camSe3Rig={selectedCamSe3Rig}
              rigSe3Target={targetPose}
            />
            <RelativePosePanel
              referenceCamera={safeRefCamera}
              selectedCamera={safeCameraA}
              referenceCamSe3Rig={referenceCamSe3Rig}
              selectedCamSe3Rig={selectedCamSe3Rig}
            />
          </aside>
        </DensityProvider>
      </div>
    </div>
  );
}

interface CameraIntrinsicsPanelProps {
  cameraIndex: number;
  camera: PinholeCameraWire | undefined;
}

function CameraIntrinsicsPanel({ cameraIndex, camera }: CameraIntrinsicsPanelProps) {
  return (
    <Panel
      title="Selected camera"
      actions={<span className={SUBTITLE}>{`cam ${cameraIndex}`}</span>}
    >
      {camera ? (
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
          <div className="col-span-2 mt-1 text-[10px] uppercase tracking-wider text-fg-muted">
            intrinsics
          </div>
          <span className="text-fg-muted">fx</span>
          <span>{camera.k.fx.toFixed(2)} px</span>
          <span className="text-fg-muted">fy</span>
          <span>{camera.k.fy.toFixed(2)} px</span>
          <span className="text-fg-muted">cx</span>
          <span>{camera.k.cx.toFixed(2)} px</span>
          <span className="text-fg-muted">cy</span>
          <span>{camera.k.cy.toFixed(2)} px</span>
          {camera.k.skew !== 0 && (
            <>
              <span className="text-fg-muted">skew</span>
              <span>{camera.k.skew.toFixed(4)}</span>
            </>
          )}
          {camera.dist && (
            <>
              <div className="col-span-2 mt-2 text-[10px] uppercase tracking-wider text-fg-muted">
                distortion (Brown-Conrady)
              </div>
              <span className="text-fg-muted">k1</span>
              <span>{camera.dist.k1.toFixed(5)}</span>
              <span className="text-fg-muted">k2</span>
              <span>{camera.dist.k2.toFixed(5)}</span>
              <span className="text-fg-muted">p1</span>
              <span>{camera.dist.p1.toFixed(5)}</span>
              <span className="text-fg-muted">p2</span>
              <span>{camera.dist.p2.toFixed(5)}</span>
              <span className="text-fg-muted">k3</span>
              <span>{camera.dist.k3.toFixed(5)}</span>
            </>
          )}
        </div>
      ) : (
        <p className="text-[11px] text-fg-muted">no camera at this index</p>
      )}
    </Panel>
  );
}

interface TargetExtrinsicsPanelProps {
  cameraIndex: number;
  poseIndex: number;
  camSe3Rig: Iso3Wire | undefined;
  rigSe3Target: Iso3Wire | undefined;
}

function TargetExtrinsicsPanel({
  cameraIndex,
  poseIndex,
  camSe3Rig,
  rigSe3Target,
}: TargetExtrinsicsPanelProps) {
  if (!camSe3Rig || !rigSe3Target) {
    return (
      <Panel title="Target extrinsic" actions={<span className={SUBTITLE}>—</span>}>
        <p className="text-[11px] text-fg-muted">no pose data</p>
      </Panel>
    );
  }
  const t = targetInCameraPose(camSe3Rig, rigSe3Target);
  const dist = iso3DistanceM(t);
  const euler = iso3EulerXYZDeg(t);
  const angle = iso3RotationAngleDeg(t);
  return (
    <Panel
      title="Target extrinsic"
      actions={
        <span className={SUBTITLE}>{`cam ${cameraIndex} → pose ${poseIndex}`}</span>
      }
    >
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
        <span className="text-fg-muted">distance</span>
        <span>
          {(dist * 1000).toFixed(1)} mm
          <span className="ml-1 text-fg-muted">({dist.toFixed(3)} m)</span>
        </span>
        <span className="text-fg-muted">euler X</span>
        <span>{formatDeg(euler.x)}</span>
        <span className="text-fg-muted">euler Y</span>
        <span>{formatDeg(euler.y)}</span>
        <span className="text-fg-muted">euler Z</span>
        <span>{formatDeg(euler.z)}</span>
        <span className="text-fg-muted">|rot|</span>
        <span>{formatDeg(angle)}</span>
      </div>
    </Panel>
  );
}

interface RelativePosePanelProps {
  referenceCamera: number;
  selectedCamera: number;
  referenceCamSe3Rig: Iso3Wire | undefined;
  selectedCamSe3Rig: Iso3Wire | undefined;
}

function RelativePosePanel({
  referenceCamera,
  selectedCamera,
  referenceCamSe3Rig,
  selectedCamSe3Rig,
}: RelativePosePanelProps) {
  if (!referenceCamSe3Rig || !selectedCamSe3Rig) {
    return (
      <Panel title="Relative pose" actions={<span className={SUBTITLE}>—</span>}>
        <p className="text-[11px] text-fg-muted">no pose data</p>
      </Panel>
    );
  }
  if (referenceCamera === selectedCamera) {
    return (
      <Panel
        title="Relative pose"
        actions={
          <span
            className={SUBTITLE}
          >{`cam ${referenceCamera} ↔ cam ${selectedCamera}`}</span>
        }
      >
        <p className="text-[11px] text-fg-muted">
          selected camera is the reference — pick a different camera (click a frustum) to
          see a relative pose.
        </p>
      </Panel>
    );
  }
  const rel = relativeCameraPose(referenceCamSe3Rig, selectedCamSe3Rig);
  const dist = iso3DistanceM(rel);
  const euler = iso3EulerXYZDeg(rel);
  const angle = iso3RotationAngleDeg(rel);
  return (
    <Panel
      title="Relative pose"
      actions={
        <span
          className={SUBTITLE}
        >{`cam ${referenceCamera} ⇒ cam ${selectedCamera}`}</span>
      }
    >
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
        <span className="text-fg-muted">baseline</span>
        <span>
          {(dist * 1000).toFixed(1)} mm
          <span className="ml-1 text-fg-muted">({dist.toFixed(4)} m)</span>
        </span>
        <span className="text-fg-muted">tx</span>
        <span>{(rel.translation[0] * 1000).toFixed(1)} mm</span>
        <span className="text-fg-muted">ty</span>
        <span>{(rel.translation[1] * 1000).toFixed(1)} mm</span>
        <span className="text-fg-muted">tz</span>
        <span>{(rel.translation[2] * 1000).toFixed(1)} mm</span>
        <span className="text-fg-muted">euler X</span>
        <span>{formatDeg(euler.x)}</span>
        <span className="text-fg-muted">euler Y</span>
        <span>{formatDeg(euler.y)}</span>
        <span className="text-fg-muted">euler Z</span>
        <span>{formatDeg(euler.z)}</span>
        <span className="text-fg-muted">|rot|</span>
        <span>{formatDeg(angle)}</span>
      </div>
    </Panel>
  );
}

function formatDeg(value: number): string {
  // Tabular alignment helper: always sign + at least one decimal.
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}°`;
}

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
