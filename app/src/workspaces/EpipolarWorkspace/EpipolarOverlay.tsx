import type { ViewportTransform } from "../../types";

export interface OverlayPoint {
  /** Pixel coordinates in the canvas's image-pixel frame (ROI-local). */
  px: [number, number];
  /** Visual color (typically token-derived). */
  color: string;
  /** Optional radius in CSS pixels; defaults to 6. */
  size?: number;
  /** When true, render as a filled dot rather than a crosshair. */
  dot?: boolean;
}

interface EpipolarOverlayProps {
  /** Viewport transform of the underlying FrameCanvas. The polyline is
   * drawn in image-pixel space so it follows zoom/pan; crosshairs are
   * drawn in canvas-pixel space so they stay a fixed on-screen size
   * regardless of zoom. */
  transform: ViewportTransform;
  /** Polyline in image-pixel coordinates. Renders inside the scaled
   * group so the line scales with the image. */
  polyline?: [number, number][];
  /** Polyline stroke color. */
  polylineColor?: string;
  /** Crosshair markers (selected feature, hover ghost, tie-line dots). */
  markers?: OverlayPoint[];
  /** Optional label drawn at the top-left of the overlay. */
  caption?: string;
}

/** SVG layer drawn over a FrameCanvas. The SVG fills the canvas's
 * container; an inner group is transformed by the viewport so the
 * polyline lines up with the underlying image, while crosshairs and
 * dots are drawn in the outer canvas-pixel space so they keep a stable
 * on-screen size as the user zooms. */
export function EpipolarOverlay({
  transform,
  polyline,
  polylineColor = "currentColor",
  markers,
  caption,
}: EpipolarOverlayProps) {
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      xmlns="http://www.w3.org/2000/svg"
    >
      {polyline && polyline.length > 1 && (
        <g
          transform={`translate(${transform.tx} ${transform.ty}) scale(${transform.scale})`}
        >
          <polyline
            points={polyline.map((p) => `${p[0]},${p[1]}`).join(" ")}
            fill="none"
            stroke={polylineColor}
            strokeWidth={1 / transform.scale}
            strokeLinejoin="round"
          />
        </g>
      )}
      {markers?.map((m, i) => {
        const cx = m.px[0] * transform.scale + transform.tx;
        const cy = m.px[1] * transform.scale + transform.ty;
        const size = m.size ?? 6;
        return m.dot ? (
          <circle
            key={i}
            cx={cx}
            cy={cy}
            r={size * 0.35}
            fill={m.color}
            opacity={0.75}
          />
        ) : (
          <g key={i} stroke={m.color} strokeWidth={1.4}>
            <line x1={cx - size} y1={cy} x2={cx + size} y2={cy} />
            <line x1={cx} y1={cy - size} x2={cx} y2={cy + size} />
            <circle cx={cx} cy={cy} r={size * 0.6} fill="none" />
          </g>
        );
      })}
      {caption && (
        <text
          x={8}
          y={16}
          fontFamily="var(--font-mono, monospace)"
          fontSize={11}
          fill="currentColor"
          opacity={0.7}
        >
          {caption}
        </text>
      )}
    </svg>
  );
}
