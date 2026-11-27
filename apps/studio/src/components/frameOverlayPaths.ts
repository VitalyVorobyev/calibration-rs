/** Pure SVG path building for the residual and laser overlays drawn over a
 * frame. Everything is in the frame's image coordinates (ROI-local, pixel
 * centre at the integer — the convention the detectors and camera models
 * use); `unit` is the image-pixel length of one screen pixel
 * (`useScreenPx()(1)`), so strokes, heads and dots stay a constant on-screen
 * size at any zoom. Output is batched by appearance (one path per colour
 * bucket), never one element per residual (lab-ui ADR-0004). */
import type {
  FrameKey,
  LaserFeatureResidual,
  LaserLineTrace,
  TargetFeatureResidual,
} from "../types";
import { colorForError, colorForLaserError } from "../lib/errorColors";

/** Residual arrows are magnified by this factor... */
export const ARROW_GAIN = 30;
/** ...but never drawn longer than this many image pixels. */
export const ARROW_LENGTH_CAP_PX = 40;
/** On-screen px: stroke width of arrows and the laser line. */
export const STROKE_PX = 1.5;
/** On-screen px: length of each arrowhead barb. */
export const ARROW_HEAD_PX = 4;
/** On-screen px: radius of the observed dot under an arrow. */
export const OBSERVED_DOT_PX = 1.6;
/** On-screen px: radius of the ring around the selected feature. */
export const SELECTION_RING_PX = 7;
/** On-screen px: radius of a laser observation dot. */
export const LASER_DOT_PX = 1.4;
/** Fill of the observed dots. */
export const OBSERVED_DOT_FILL = "rgba(255,255,255,0.85)";
/** Stroke of the projected laser line. */
export const LASER_LINE_STROKE = "rgba(64, 156, 255, 0.6)";
/** Fill of laser dots that carry no point-to-plane residual. */
export const LASER_UNRESOLVED_FILL = "rgba(255,255,255,0.4)";

const ZERO_LENGTH = 1e-6;
const HEAD_ANGLE = Math.PI / 6;

/** One path of the overlay and the colour it is painted in. */
export interface ColoredPath {
  color: string;
  d: string;
}

export interface ResidualArrowPaths {
  /** Shaft + arrowhead of every arrow, one stroked path per error colour. */
  arrows: ColoredPath[];
  /** All observed dots as one filled path (empty string when none). */
  dots: string;
}

export interface LaserOverlayPaths {
  /** The projected laser line, or `null` when the frame has none. */
  line: string | null;
  /** Observed dots with a residual, one filled path per error colour. */
  dots: ColoredPath[];
  /** Observed dots without a residual, one filled path. */
  unresolved: string;
}

const n = (v: number): string => String(Math.round(v * 1e4) / 1e4);

function circle(cx: number, cy: number, r: number): string {
  return `M${n(cx - r)} ${n(cy)}a${n(r)} ${n(r)} 0 1 0 ${n(2 * r)} 0a${n(r)} ${n(r)} 0 1 0 ${n(-2 * r)} 0`;
}

function ofFrame<T extends { view: number; camera: number }>(
  all: readonly T[],
  frame: Pick<FrameKey, "pose" | "camera">,
): T[] {
  return all.filter((r) => r.view === frame.pose && r.camera === frame.camera);
}

/** Group `d` fragments by colour, keeping first-seen order. */
function bucket(items: Iterable<[string, string]>): ColoredPath[] {
  const byColor = new Map<string, string>();
  for (const [color, d] of items) byColor.set(color, (byColor.get(color) ?? "") + d);
  return [...byColor].map(([color, d]) => ({ color, d }));
}

/** One arrow per residual of the frame, anchored at the observed corner
 * and pointing along the residual `observed − projected` (the sign of the
 * library's residuals and maps: where the data put the corner relative to
 * the model), magnified by {@link ARROW_GAIN} up to
 * {@link ARROW_LENGTH_CAP_PX}. */
export function buildResidualArrows(
  all: readonly TargetFeatureResidual[],
  frame: Pick<FrameKey, "pose" | "camera">,
  unit: number,
): ResidualArrowPaths {
  const head = ARROW_HEAD_PX * unit;
  const dotR = OBSERVED_DOT_PX * unit;
  const arrows: [string, string][] = [];
  let dots = "";
  for (const r of ofFrame(all, frame)) {
    if (!r.projected_px) continue;
    const [ox, oy] = r.observed_px;
    // observed − projected.
    const dx0 = ox - r.projected_px[0];
    const dy0 = oy - r.projected_px[1];
    const mag = Math.hypot(dx0, dy0);
    if (mag < ZERO_LENGTH) continue;
    const gain = Math.min(ARROW_GAIN, ARROW_LENGTH_CAP_PX / mag);
    const x2 = ox + dx0 * gain;
    const y2 = oy + dy0 * gain;
    const angle = Math.atan2(y2 - oy, x2 - ox);
    const x3 = x2 - head * Math.cos(angle - HEAD_ANGLE);
    const y3 = y2 - head * Math.sin(angle - HEAD_ANGLE);
    const x4 = x2 - head * Math.cos(angle + HEAD_ANGLE);
    const y4 = y2 - head * Math.sin(angle + HEAD_ANGLE);
    arrows.push([
      colorForError(r.error_px ?? mag),
      `M${n(ox)} ${n(oy)}L${n(x2)} ${n(y2)}L${n(x3)} ${n(y3)}M${n(x2)} ${n(y2)}L${n(x4)} ${n(y4)}`,
    ]);
    dots += circle(ox, oy, dotR);
  }
  return { arrows: bucket(arrows), dots };
}

/** A ring around the observed corner of feature `feature` of the frame, for
 * the feature the selection names; `null` when the frame has no such one. */
export function buildFeatureRing(
  all: readonly TargetFeatureResidual[],
  frame: Pick<FrameKey, "pose" | "camera">,
  feature: number | null,
  unit: number,
): string | null {
  if (feature === null) return null;
  const r = ofFrame(all, frame).find((x) => x.feature === feature);
  if (!r) return null;
  return circle(r.observed_px[0], r.observed_px[1], SELECTION_RING_PX * unit);
}

export function buildLaserOverlay(
  all: readonly LaserFeatureResidual[],
  lines: readonly LaserLineTrace[],
  frame: Pick<FrameKey, "pose" | "camera">,
  unit: number,
): LaserOverlayPaths {
  const records = ofFrame(all, frame);
  const trace = lines.find((l) => l.view === frame.pose && l.camera === frame.camera);
  const line =
    trace && trace.points_px.length >= 2
      ? trace.points_px
          .map(([x, y], i) => `${i === 0 ? "M" : "L"}${n(x)} ${n(y)}`)
          .join("")
      : null;
  const r = LASER_DOT_PX * unit;
  const dots: [string, string][] = [];
  let unresolved = "";
  for (const rec of records) {
    const [x, y] = rec.observed_px;
    if (rec.residual_m != null) {
      dots.push([colorForLaserError(Math.abs(rec.residual_m) * 1e3), circle(x, y, r)]);
    } else {
      unresolved += circle(x, y, r);
    }
  }
  return { line, dots: bucket(dots), unresolved };
}
