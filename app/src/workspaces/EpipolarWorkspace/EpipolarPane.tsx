import { useMemo, type RefObject } from "react";
import { FrameCanvas, type FrameCanvasHandle } from "../../components/FrameCanvas";
import type { FrameKey, ViewportTransform } from "../../types";
import { EpipolarOverlay, type OverlayPoint } from "./EpipolarOverlay";

export interface EpipolarPaneProps {
  frame: FrameKey | null;
  transform: ViewportTransform | null;
  onTransformChange: (t: ViewportTransform) => void;
  onPick?: ((pixel: { x: number; y: number }) => void) | undefined;
  polyline?: [number, number][] | undefined;
  polylineColor?: string | undefined;
  markers?: OverlayPoint[] | undefined;
  caption?: string | undefined;
  image: HTMLImageElement | null;
  handleRef: RefObject<FrameCanvasHandle | null>;
  annotation?: { px: [number, number]; text: string; color: string } | undefined;
}

/** One pane of the epipolar pair: a camera's undistorted frame with the
 * overlay, or a placeholder when the (pose, camera) slot has no frame. */
export function EpipolarPane(props: EpipolarPaneProps) {
  const { frame } = props;
  if (!frame) {
    return (
      <div className="flex h-full items-center justify-center rounded-control border border-dashed border-line text-[12px] text-fg-muted">
        no frame for this (pose, camera)
      </div>
    );
  }
  return <PaneInner key={frame.abs_path} {...props} frame={frame} />;
}

function PaneInner({
  frame,
  transform,
  onTransformChange,
  onPick,
  polyline,
  polylineColor,
  markers,
  caption,
  image,
  handleRef,
  annotation,
}: EpipolarPaneProps & { frame: FrameKey }) {
  // The undistorted image is already ROI-local: no second crop.
  const displayFrame = useMemo<FrameKey>(() => ({ ...frame, roi: undefined }), [frame]);
  return (
    <div className="relative flex h-full flex-col">
      <div className="relative flex-1 overflow-hidden">
        <FrameCanvas
          ref={handleRef}
          frame={displayFrame}
          residuals={[]}
          image={image}
          transform={transform}
          onTransformChange={onTransformChange}
          onPick={onPick}
        >
          <EpipolarOverlay
            polyline={polyline}
            polylineColor={polylineColor}
            markers={markers}
            caption={caption}
            annotation={annotation}
          />
        </FrameCanvas>
      </div>
    </div>
  );
}
