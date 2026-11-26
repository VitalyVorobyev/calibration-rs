/** Pure picking and marker logic of the Epipolar workspace. */
import type { TargetFeatureResidual } from "../../types";
import type { OverlayPoint } from "./EpipolarOverlay";

type Px = [number, number];

/** Pixel radius around a click within which it snaps to the nearest
 * feature. Outside it the click is a free pixel pick (no pane-B ghost). */
export const FEATURE_SNAP_PX = 8;

/** The click `px`, snapped to the nearest feature within `radius`. */
export function snapToFeature(
  px: Px,
  residuals: readonly TargetFeatureResidual[],
  radius: number = FEATURE_SNAP_PX,
): { px: Px; feature: number | null } {
  let best = radius;
  let hit: { px: Px; feature: number | null } = { px, feature: null };
  for (const r of residuals) {
    const d = Math.hypot(r.observed_px[0] - px[0], r.observed_px[1] - px[1]);
    if (d < best) {
      best = d;
      hit = { px: [r.observed_px[0], r.observed_px[1]], feature: r.feature };
    }
  }
  return hit;
}

/** Shortest distance from `point` to a polyline of at least one segment. */
export function distanceToPolyline(point: Px, polyline: readonly Px[]): number {
  let best = Infinity;
  for (let i = 0; i < polyline.length - 1; i++) {
    const d = distanceToSegment(point, polyline[i]!, polyline[i + 1]!);
    if (d < best) best = d;
  }
  return best;
}

function distanceToSegment(p: Px, a: Px, b: Px): number {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const lenSq = abx * abx + aby * aby;
  if (lenSq === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / lenSq));
  return Math.hypot(p[0] - (a[0] + t * abx), p[1] - (a[1] + t * aby));
}

/** Every feature as a quiet dot of `size` px. */
export function featureDots(
  residuals: readonly TargetFeatureResidual[],
  size: number,
): OverlayPoint[] {
  return residuals.map((r) => ({
    px: r.observed_px,
    color: "var(--fg-muted)",
    dot: true,
    size,
  }));
}

/** The design token an epipolar residual is drawn in: on the line under
 * 1 px, plain under 5 px, a defect beyond. */
export type GhostTone = "signal" | "fg" | "defect";

export function ghostDistanceTone(distancePx: number): GhostTone {
  if (distancePx < 1) return "signal";
  if (distancePx < 5) return "fg";
  return "defect";
}
