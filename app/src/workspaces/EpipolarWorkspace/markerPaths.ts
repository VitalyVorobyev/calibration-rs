/** Pure SVG path building for the epipolar overlay's markers. Positions are
 * in the frame's image coordinates; `unit` is the image-pixel length of one
 * screen pixel, so a marker keeps its on-screen size at any zoom. Markers
 * are batched by appearance (colour, size, kind): one path per group, not
 * one element per marker. */

export interface MarkerSpec {
  px: [number, number];
  color: string;
  /** Radius in screen px; defaults to 6. */
  size?: number | undefined;
  /** Filled dot instead of a crosshair. */
  dot?: boolean | undefined;
}

export interface MarkerPath {
  /** Stable identity of the appearance group (kind, colour, size). */
  key: string;
  kind: "dot" | "cross";
  color: string;
  d: string;
}

/** Stroke width of crosshairs, in screen px. */
export const CROSS_STROKE_PX = 1.4;
/** Opacity of filled dots. */
export const DOT_OPACITY = 0.75;

const n = (v: number): string => String(Math.round(v * 1e4) / 1e4);

function circle(cx: number, cy: number, r: number): string {
  return `M${n(cx - r)} ${n(cy)}a${n(r)} ${n(r)} 0 1 0 ${n(2 * r)} 0a${n(r)} ${n(r)} 0 1 0 ${n(-2 * r)} 0`;
}

export function buildMarkerPaths(
  markers: readonly MarkerSpec[],
  unit: number,
): MarkerPath[] {
  const groups = new Map<string, MarkerPath>();
  for (const m of markers) {
    const size = (m.size ?? 6) * unit;
    const [x, y] = m.px;
    const kind = m.dot ? "dot" : "cross";
    const key = `${kind}|${m.color}|${m.size ?? 6}`;
    let g = groups.get(key);
    if (!g) {
      g = { key, kind, color: m.color, d: "" };
      groups.set(key, g);
    }
    g.d += m.dot
      ? circle(x, y, size * 0.35)
      : `M${n(x - size)} ${n(y)}h${n(2 * size)}M${n(x)} ${n(y - size)}v${n(2 * size)}${circle(x, y, size * 0.6)}`;
  }
  return [...groups.values()];
}
