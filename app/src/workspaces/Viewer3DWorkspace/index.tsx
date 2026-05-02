import { useMemo, useState } from "react";
import { PoseStepper } from "../../components/PoseStepper";
import {
  iso3DistanceM,
  iso3EulerXYZDeg,
  iso3RotationAngleDeg,
  relativeCameraPose,
  targetInCameraPose,
} from "../../lib/se3";
import { useStore } from "../../store";
import { exportKindLabel } from "../../store/exportShape";
import type { Iso3Wire, PinholeCameraWire } from "../../store/types";
import type { FrameKey } from "../../types";
import { Scene } from "./Scene";

/** B1.2 + B2.4 — 3D rig scene + side panel that breaks down the
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
  const [referenceCamera, setReferenceCamera] = useState<number>(0);

  const cameraDimensions = useMemo(
    () => cameraDimensionsFromFrames(frames),
    [frames],
  );

  if (!data || !kind) {
    return (
      <Empty body="Load a rig export to see cameras and target poses in 3D." />
    );
  }

  const camerasArr = data.cameras;
  const camSe3Rig = data.cam_se3_rig;
  const rigSe3Target = data.rig_se3_target;
  const isRig =
    Array.isArray(camerasArr) &&
    Array.isArray(camSe3Rig) &&
    Array.isArray(rigSe3Target) &&
    camerasArr.length > 0;

  if (!isRig) {
    return (
      <Empty
        body={`The 3D viewer needs a rig export (cameras + cam_se3_rig + rig_se3_target). The current export is ${exportKindLabel(
          kind,
        )} — single-camera shapes will land in a follow-up.`}
      />
    );
  }

  const numCameras = camerasArr.length;
  const numPoses = rigSe3Target.length;
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
          <RefCameraSelect
            cameraIndices={cameraIndices}
            value={safeRefCamera}
            onChange={setReferenceCamera}
          />
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

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_18rem] gap-3 overflow-hidden">
        <div className="relative overflow-hidden rounded-md border border-border">
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
              cam {safeCameraA} · pose {safePose} · ref {safeRefCamera}
            </span>
            <span>drag · scroll to zoom · click a frustum to select</span>
          </div>
        </div>

        <aside className="flex min-h-0 flex-col gap-3 overflow-y-auto rounded-md border border-border bg-surface p-3 text-[12px]">
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
      </div>
    </div>
  );
}

interface RefCameraSelectProps {
  cameraIndices: number[];
  value: number;
  onChange: (v: number) => void;
}

function RefCameraSelect({
  cameraIndices,
  value,
  onChange,
}: RefCameraSelectProps) {
  return (
    <label className="flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
      <span className="uppercase tracking-wider">ref cam</span>
      <select
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-7 rounded-md border border-border bg-surface px-1 text-foreground"
        title="Reference camera for the relative-pose readout"
      >
        {cameraIndices.map((i) => (
          <option key={i} value={i}>
            {i}
          </option>
        ))}
      </select>
    </label>
  );
}

interface CameraIntrinsicsPanelProps {
  cameraIndex: number;
  camera: PinholeCameraWire | undefined;
}

function CameraIntrinsicsPanel({
  cameraIndex,
  camera,
}: CameraIntrinsicsPanelProps) {
  return (
    <section>
      <PanelHeading title="Selected camera" subtitle={`cam ${cameraIndex}`} />
      {camera ? (
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
          <div className="col-span-2 mt-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            intrinsics
          </div>
          <span className="text-muted-foreground">fx</span>
          <span>{camera.k.fx.toFixed(2)} px</span>
          <span className="text-muted-foreground">fy</span>
          <span>{camera.k.fy.toFixed(2)} px</span>
          <span className="text-muted-foreground">cx</span>
          <span>{camera.k.cx.toFixed(2)} px</span>
          <span className="text-muted-foreground">cy</span>
          <span>{camera.k.cy.toFixed(2)} px</span>
          {camera.k.skew !== 0 && (
            <>
              <span className="text-muted-foreground">skew</span>
              <span>{camera.k.skew.toFixed(4)}</span>
            </>
          )}
          {camera.dist && (
            <>
              <div className="col-span-2 mt-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                distortion (Brown-Conrady)
              </div>
              <span className="text-muted-foreground">k1</span>
              <span>{camera.dist.k1.toFixed(5)}</span>
              <span className="text-muted-foreground">k2</span>
              <span>{camera.dist.k2.toFixed(5)}</span>
              <span className="text-muted-foreground">p1</span>
              <span>{camera.dist.p1.toFixed(5)}</span>
              <span className="text-muted-foreground">p2</span>
              <span>{camera.dist.p2.toFixed(5)}</span>
              <span className="text-muted-foreground">k3</span>
              <span>{camera.dist.k3.toFixed(5)}</span>
            </>
          )}
        </div>
      ) : (
        <p className="text-[11px] text-muted-foreground">no camera at this index</p>
      )}
    </section>
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
      <section>
        <PanelHeading title="Target extrinsic" subtitle="—" />
        <p className="text-[11px] text-muted-foreground">no pose data</p>
      </section>
    );
  }
  const t = targetInCameraPose(camSe3Rig, rigSe3Target);
  const dist = iso3DistanceM(t);
  const euler = iso3EulerXYZDeg(t);
  const angle = iso3RotationAngleDeg(t);
  return (
    <section>
      <PanelHeading
        title="Target extrinsic"
        subtitle={`cam ${cameraIndex} → pose ${poseIndex}`}
      />
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
        <span className="text-muted-foreground">distance</span>
        <span>
          {(dist * 1000).toFixed(1)} mm
          <span className="ml-1 text-muted-foreground">({dist.toFixed(3)} m)</span>
        </span>
        <span className="text-muted-foreground">euler X</span>
        <span>{formatDeg(euler.x)}</span>
        <span className="text-muted-foreground">euler Y</span>
        <span>{formatDeg(euler.y)}</span>
        <span className="text-muted-foreground">euler Z</span>
        <span>{formatDeg(euler.z)}</span>
        <span className="text-muted-foreground">|rot|</span>
        <span>{formatDeg(angle)}</span>
      </div>
    </section>
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
      <section>
        <PanelHeading title="Relative pose" subtitle="—" />
        <p className="text-[11px] text-muted-foreground">no pose data</p>
      </section>
    );
  }
  if (referenceCamera === selectedCamera) {
    return (
      <section>
        <PanelHeading
          title="Relative pose"
          subtitle={`cam ${referenceCamera} ↔ cam ${selectedCamera}`}
        />
        <p className="text-[11px] text-muted-foreground">
          selected camera is the reference — pick a different camera (click a
          frustum) to see a relative pose.
        </p>
      </section>
    );
  }
  const rel = relativeCameraPose(referenceCamSe3Rig, selectedCamSe3Rig);
  const dist = iso3DistanceM(rel);
  const euler = iso3EulerXYZDeg(rel);
  const angle = iso3RotationAngleDeg(rel);
  return (
    <section>
      <PanelHeading
        title="Relative pose"
        subtitle={`cam ${referenceCamera} ⇒ cam ${selectedCamera}`}
      />
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
        <span className="text-muted-foreground">baseline</span>
        <span>
          {(dist * 1000).toFixed(1)} mm
          <span className="ml-1 text-muted-foreground">({dist.toFixed(4)} m)</span>
        </span>
        <span className="text-muted-foreground">tx</span>
        <span>{(rel.translation[0] * 1000).toFixed(1)} mm</span>
        <span className="text-muted-foreground">ty</span>
        <span>{(rel.translation[1] * 1000).toFixed(1)} mm</span>
        <span className="text-muted-foreground">tz</span>
        <span>{(rel.translation[2] * 1000).toFixed(1)} mm</span>
        <span className="text-muted-foreground">euler X</span>
        <span>{formatDeg(euler.x)}</span>
        <span className="text-muted-foreground">euler Y</span>
        <span>{formatDeg(euler.y)}</span>
        <span className="text-muted-foreground">euler Z</span>
        <span>{formatDeg(euler.z)}</span>
        <span className="text-muted-foreground">|rot|</span>
        <span>{formatDeg(angle)}</span>
      </div>
    </section>
  );
}

function PanelHeading({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <header className="mb-1.5 flex items-baseline justify-between border-b border-border pb-1">
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-foreground">
        {title}
      </h3>
      <span className="font-mono text-[10px] text-muted-foreground">
        {subtitle}
      </span>
    </header>
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
