import { imageViewBox, useScreenPx, useStage } from "@vitavision/stage2d";
import {
  buildMarkerPaths,
  CROSS_STROKE_PX,
  DOT_OPACITY,
  type MarkerSpec,
} from "./markerPaths";

export type OverlayPoint = MarkerSpec;

interface EpipolarOverlayProps {
  /** Polyline in image-pixel coordinates; follows zoom/pan. */
  polyline?: [number, number][] | undefined;
  /** Polyline stroke color. */
  polylineColor?: string | undefined;
  /** Crosshair markers (selected feature, hover ghost, tie-line dots),
   * kept a fixed on-screen size regardless of zoom. */
  markers?: OverlayPoint[] | undefined;
  /** Optional label drawn at the top-left of the viewport. */
  caption?: string | undefined;
  /** Optional pixel-anchored text annotation drawn near `px` (image
   * pixels, fixed on-screen size). Used to flag the picked feature's
   * residual distance to the epipolar polyline. */
  annotation?: { px: [number, number]; text: string; color: string } | undefined;
}

/** Stage layer for the epipolar workspace. Everything is drawn in the
 * frame's image coordinates (inside the stage's transform); sizes that must
 * not follow the zoom (markers, text, strokes) come from the screen-pixel
 * unit, and the caption is placed at a fixed viewport position by mapping
 * that position back into image coordinates. */
export function EpipolarOverlay({
  polyline,
  polylineColor = "currentColor",
  markers,
  caption,
  annotation,
}: EpipolarOverlayProps) {
  const { image, view } = useStage();
  const px = useScreenPx();
  const unit = px(1);
  const paths = buildMarkerPaths(markers ?? [], unit);
  // Viewport (css) -> image-coordinate: the stage is translated by `tx`
  // and scaled by `scale`, and the viewBox is shifted by the pixel centre.
  const atViewport = (cx: number, cy: number) => ({
    x: (cx - view.tx) / view.scale - 0.5,
    y: (cy - view.ty) / view.scale - 0.5,
  });
  const captionAt = atViewport(8, 16);
  return (
    <svg
      viewBox={imageViewBox(image)}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {polyline && polyline.length > 1 && (
        <polyline
          points={polyline.map((p) => `${p[0]},${p[1]}`).join(" ")}
          fill="none"
          stroke={polylineColor}
          strokeWidth={unit}
          strokeLinejoin="round"
        />
      )}
      {paths.map((p) =>
        p.kind === "dot" ? (
          <path key={p.key} d={p.d} fill={p.color} opacity={DOT_OPACITY} />
        ) : (
          <path
            key={p.key}
            d={p.d}
            fill="none"
            stroke={p.color}
            strokeWidth={CROSS_STROKE_PX * unit}
          />
        ),
      )}
      {caption && (
        <text
          x={captionAt.x}
          y={captionAt.y}
          fontFamily="var(--font-mono, monospace)"
          fontSize={px(11)}
          fill="currentColor"
          opacity={0.7}
        >
          {caption}
        </text>
      )}
      {annotation && (
        // Offset upward-right of the anchor so the label clears the
        // crosshair drawn at the same pixel.
        <text
          x={annotation.px[0] + px(10)}
          y={annotation.px[1] - px(8)}
          fontFamily="var(--font-mono, monospace)"
          fontSize={px(11)}
          fill={annotation.color}
          stroke="var(--raised)"
          strokeWidth={px(3)}
          paintOrder="stroke"
        >
          {annotation.text}
        </text>
      )}
    </svg>
  );
}
