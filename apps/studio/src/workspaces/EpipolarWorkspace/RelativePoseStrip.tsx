import {
  iso3DistanceM,
  iso3EulerXYZDeg,
  iso3RotationAngleDeg,
  type relativeCameraPose,
} from "../../lib/se3";
import { ghostDistanceTone, type GhostTone } from "./epipolarGeometry";

interface RelativePoseStripProps {
  relativePose: ReturnType<typeof relativeCameraPose> | null;
  cameraA: number;
  cameraB: number;
  ghostDistancePx: number | null;
  pickedFeature: number | null;
  samplesClipped: number | null;
}

const TONE_CLASS: Record<GhostTone, string> = {
  signal: "text-signal",
  fg: "text-fg",
  defect: "text-defect",
};

/** Camera B's pose relative to camera A, and the picked feature's
 * epipolar residual. */
export function RelativePoseStrip({
  relativePose,
  cameraA,
  cameraB,
  ghostDistancePx,
  pickedFeature,
  samplesClipped,
}: RelativePoseStripProps) {
  if (!relativePose) {
    return (
      <div className="font-mono text-[11px] text-fg-muted">
        cam {cameraA} ↔ cam {cameraB}: pick two distinct cameras for relative pose
      </div>
    );
  }
  const dist = iso3DistanceM(relativePose);
  const euler = iso3EulerXYZDeg(relativePose);
  const angle = iso3RotationAngleDeg(relativePose);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-fg-muted">
      <span>
        cam {cameraA} ⇒ cam {cameraB}:
      </span>
      <span>
        baseline{" "}
        <span className="text-fg tabular-nums">{(dist * 1000).toFixed(1)} mm</span>
      </span>
      <span>
        rot <span className="text-fg tabular-nums">{angle.toFixed(2)}°</span>
        <span className="ml-1">
          ({euler.x.toFixed(1)}, {euler.y.toFixed(1)}, {euler.z.toFixed(1)})°
        </span>
      </span>
      {pickedFeature != null && (
        <span>
          feature <span className="text-fg tabular-nums">#{pickedFeature}</span>
        </span>
      )}
      {ghostDistancePx != null && (
        <span>
          ghost Δ
          <span
            className={`ml-1 tabular-nums ${TONE_CLASS[ghostDistanceTone(ghostDistancePx)]}`}
          >
            {ghostDistancePx.toFixed(2)} px
          </span>
        </span>
      )}
      {samplesClipped != null && samplesClipped > 0 && (
        <span title="number of depth samples whose projection diverged">
          clipped <span className="text-fg tabular-nums">{samplesClipped}</span>
        </span>
      )}
    </div>
  );
}
