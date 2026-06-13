import { useMemo } from "react";
import { Quaternion, Vector3 } from "three";
import { iso3FromWire } from "../../lib/se3";
import type { Iso3Wire, LaserPlaneWire } from "../../store/types";
import type { TargetFeatureResidual } from "../../types";
import { computeBoardBbox, type Bbox2 } from "./TargetBoard";

interface LaserTargetCutsProps {
  /** Active pose `rig_se3_target` (T_R_T). */
  rigSe3Target: Iso3Wire;
  /** Active-pose target residuals, used to clip cuts to the board bbox. */
  residuals: TargetFeatureResidual[];
  /** Laser planes in rig frame. */
  planesRig: LaserPlaneWire[];
}

const CUT_COLORS = ["#38bdf8", "#f97316", "#a3e635", "#f43f5e", "#c084fc", "#facc15"];
const EPS = 1e-9;

export function LaserTargetCuts({
  rigSe3Target,
  residuals,
  planesRig,
}: LaserTargetCutsProps) {
  const matrix = useMemo(() => iso3FromWire(rigSe3Target), [rigSe3Target]);
  const bbox = useMemo(() => computeBoardBbox(residuals), [residuals]);
  const segments = useMemo(
    () =>
      planesRig
        .map((plane, idx) => ({
          idx,
          segment: intersectLaserPlaneWithTarget(plane, rigSe3Target, bbox),
        }))
        .filter((v): v is { idx: number; segment: Segment2 } => v.segment != null),
    [planesRig, rigSe3Target, bbox],
  );

  if (segments.length === 0) return null;

  return (
    <group matrix={matrix} matrixAutoUpdate={false}>
      {segments.map(({ idx, segment }) => (
        <line key={`laser-cut-${idx}`}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              args={[
                new Float32Array([
                  segment.a[0],
                  segment.a[1],
                  0.0004,
                  segment.b[0],
                  segment.b[1],
                  0.0004,
                ]),
                3,
              ]}
            />
          </bufferGeometry>
          <lineBasicMaterial
            color={CUT_COLORS[idx % CUT_COLORS.length]}
            transparent
            opacity={0.95}
          />
        </line>
      ))}
    </group>
  );
}

interface Segment2 {
  a: [number, number];
  b: [number, number];
}

export function intersectLaserPlaneWithTarget(
  planeRig: LaserPlaneWire,
  rigSe3Target: Iso3Wire,
  bbox: Bbox2,
): Segment2 | null {
  const [qx, qy, qz, qw] = rigSe3Target.rotation;
  const qInv = new Quaternion(-qx, -qy, -qz, qw);
  const nRig = new Vector3(...planeRig.normal).normalize();
  const nTarget = nRig.clone().applyQuaternion(qInv);
  const tRig = new Vector3(...rigSe3Target.translation);
  const dTarget = planeRig.distance + nRig.dot(tRig);

  const a = nTarget.x;
  const b = nTarget.y;
  if (Math.hypot(a, b) < EPS || !Number.isFinite(dTarget)) return null;
  return clipImplicitLineToBbox(a, b, dTarget, bbox);
}

function clipImplicitLineToBbox(a: number, b: number, c: number, bbox: Bbox2): Segment2 | null {
  const pts: [number, number][] = [];
  const push = (x: number, y: number) => {
    if (
      x < bbox.x0 - EPS ||
      x > bbox.x1 + EPS ||
      y < bbox.y0 - EPS ||
      y > bbox.y1 + EPS ||
      !Number.isFinite(x) ||
      !Number.isFinite(y)
    ) {
      return;
    }
    if (!pts.some(([px, py]) => Math.hypot(px - x, py - y) < 1e-7)) {
      pts.push([x, y]);
    }
  };

  if (Math.abs(b) > EPS) {
    push(bbox.x0, (-c - a * bbox.x0) / b);
    push(bbox.x1, (-c - a * bbox.x1) / b);
  }
  if (Math.abs(a) > EPS) {
    push((-c - b * bbox.y0) / a, bbox.y0);
    push((-c - b * bbox.y1) / a, bbox.y1);
  }

  if (pts.length < 2) return null;
  return { a: pts[0], b: pts[1] };
}
