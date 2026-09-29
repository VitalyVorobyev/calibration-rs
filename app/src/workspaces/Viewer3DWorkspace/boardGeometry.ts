import { Quaternion, Vector3 } from "three";
import type { Iso3Wire, LaserPlaneWire } from "../../store/types";
import type { TargetFeatureResidual } from "../../types";

const EPS = 1e-9;

export interface Bbox2 {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function computeBoardBbox(residuals: TargetFeatureResidual[]): Bbox2 {
  if (residuals.length === 0) {
    // Fallback when residuals don't carry geometry — a 100 mm × 100 mm
    // square centred on the target origin.
    return { x0: -0.05, y0: -0.05, x1: 0.05, y1: 0.05 };
  }
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const r of residuals) {
    const [x, y] = r.target_xyz_m;
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  // Pad by 5 % to keep the marker dots from sitting on the outline.
  const pad = Math.max(x1 - x0, y1 - y0) * 0.05;
  return { x0: x0 - pad, y0: y0 - pad, x1: x1 + pad, y1: y1 + pad };
}

export interface Segment2 {
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

function clipImplicitLineToBbox(
  a: number,
  b: number,
  c: number,
  bbox: Bbox2,
): Segment2 | null {
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
  return { a: pts[0]!, b: pts[1]! };
}
