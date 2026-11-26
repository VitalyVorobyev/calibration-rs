import { Panel } from "@vitavision/ui";
import {
  iso3DistanceM,
  iso3EulerXYZDeg,
  iso3RotationAngleDeg,
  relativeCameraPose,
  targetInCameraPose,
} from "../../lib/se3";
import type { CameraWire, Iso3Wire } from "../../store/types";

/** The quiet right-aligned caption in an info-rail panel's header. */
const SUBTITLE = "font-mono text-[10px] text-fg-muted";
const GRID =
  "grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums";
const GROUP = "col-span-2 mt-2 text-[10px] uppercase tracking-wider text-fg-muted";

interface CameraIntrinsicsPanelProps {
  cameraIndex: number;
  camera: CameraWire | undefined;
}

/** The selected camera's intrinsics, sensor tilt and distortion. */
export function CameraIntrinsicsPanel({
  cameraIndex,
  camera,
}: CameraIntrinsicsPanelProps) {
  return (
    <Panel
      title="Selected camera"
      actions={<span className={SUBTITLE}>{`cam ${cameraIndex}`}</span>}
    >
      {camera ? (
        <div className={GRID}>
          <div className="col-span-2 mt-1 text-[10px] uppercase tracking-wider text-fg-muted">
            intrinsics
          </div>
          <span className="text-fg-muted">fx</span>
          <span>{camera.intrinsics.fx.toFixed(2)} px</span>
          <span className="text-fg-muted">fy</span>
          <span>{camera.intrinsics.fy.toFixed(2)} px</span>
          <span className="text-fg-muted">cx</span>
          <span>{camera.intrinsics.cx.toFixed(2)} px</span>
          <span className="text-fg-muted">cy</span>
          <span>{camera.intrinsics.cy.toFixed(2)} px</span>
          {camera.sensor.type === "scheimpflug" && (
            <>
              <div className={GROUP}>sensor tilt (Scheimpflug)</div>
              <span className="text-fg-muted">tilt x</span>
              <span>{camera.sensor.tilt_x.toFixed(5)} rad</span>
              <span className="text-fg-muted">tilt y</span>
              <span>{camera.sensor.tilt_y.toFixed(5)} rad</span>
            </>
          )}
          {camera.distortion.type === "brown_conrady5" && (
            <>
              <div className={GROUP}>distortion (Brown-Conrady)</div>
              <span className="text-fg-muted">k1</span>
              <span>{camera.distortion.k1.toFixed(5)}</span>
              <span className="text-fg-muted">k2</span>
              <span>{camera.distortion.k2.toFixed(5)}</span>
              <span className="text-fg-muted">p1</span>
              <span>{camera.distortion.p1.toFixed(5)}</span>
              <span className="text-fg-muted">p2</span>
              <span>{camera.distortion.p2.toFixed(5)}</span>
              <span className="text-fg-muted">k3</span>
              <span>{camera.distortion.k3.toFixed(5)}</span>
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

/** The target's pose in the selected camera at the active pose. */
export function TargetExtrinsicsPanel({
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
  return (
    <Panel
      title="Target extrinsic"
      actions={
        <span className={SUBTITLE}>{`cam ${cameraIndex} → pose ${poseIndex}`}</span>
      }
    >
      <div className={GRID}>
        <span className="text-fg-muted">distance</span>
        <span>
          {(dist * 1000).toFixed(1)} mm
          <span className="ml-1 text-fg-muted">({dist.toFixed(3)} m)</span>
        </span>
        <RotationRows pose={t} />
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

/** The selected camera's pose relative to the reference camera. */
export function RelativePosePanel({
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
  return (
    <Panel
      title="Relative pose"
      actions={
        <span
          className={SUBTITLE}
        >{`cam ${referenceCamera} ⇒ cam ${selectedCamera}`}</span>
      }
    >
      <div className={GRID}>
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
        <RotationRows pose={rel} />
      </div>
    </Panel>
  );
}

/** Euler XYZ angles and the total rotation angle of a pose, as grid rows. */
function RotationRows({ pose }: { pose: Iso3Wire }) {
  const euler = iso3EulerXYZDeg(pose);
  return (
    <>
      <span className="text-fg-muted">euler X</span>
      <span>{formatDeg(euler.x)}</span>
      <span className="text-fg-muted">euler Y</span>
      <span>{formatDeg(euler.y)}</span>
      <span className="text-fg-muted">euler Z</span>
      <span>{formatDeg(euler.z)}</span>
      <span className="text-fg-muted">|rot|</span>
      <span>{formatDeg(iso3RotationAngleDeg(pose))}</span>
    </>
  );
}

/** Tabular alignment helper: always signed, two decimals. */
function formatDeg(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}°`;
}
