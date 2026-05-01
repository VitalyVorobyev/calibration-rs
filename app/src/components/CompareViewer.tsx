import { useCallback, useEffect, useRef, useState } from "react";
import { useImageData, getPixelLum } from "../hooks/useImageData";
import type {
  CursorReadout,
  FrameKey,
  TargetFeatureResidual,
  ViewportTransform,
} from "../types";
import { IDENTITY_TRANSFORM } from "../types";
import { FrameCanvas, type FrameCanvasHandle } from "./FrameCanvas";

interface CompareViewerProps {
  leftFrame: FrameKey;
  rightFrame: FrameKey;
  residuals: TargetFeatureResidual[];
  /** Active pane receives stepper input + zoom/key shortcuts. */
  activePane: "left" | "right";
  onActivePane: (next: "left" | "right") => void;
  /** When true the two panes share a single viewport transform — wheel
   * zoom + drag pan in one applies to both, so the same image-pixel
   * sits at the same canvas pixel on both sides (the workhorse of
   * cross-camera comparison). */
  linked: boolean;
  onCursorChange?: (cursor: CursorReadout | null) => void;
  onError?: (msg: string) => void;
}

/** Imperative handle the toolbar uses to drive zoom/fit on whichever
 * pane is currently active (or both panes when linked). */
export interface CompareViewerHandle {
  fitActive(): void;
  reset1to1Active(): void;
  zoomActiveBy(factor: number): void;
}

export function CompareViewer({
  leftFrame,
  rightFrame,
  residuals,
  activePane,
  onActivePane,
  linked,
  onCursorChange,
  onError,
  innerRef,
}: CompareViewerProps & {
  innerRef?: React.MutableRefObject<CompareViewerHandle | null>;
}) {
  const leftRef = useRef<FrameCanvasHandle | null>(null);
  const rightRef = useRef<FrameCanvasHandle | null>(null);

  const [linkedTransform, setLinkedTransform] = useState<ViewportTransform>(
    IDENTITY_TRANSFORM,
  );
  const leftImageData = useImageData(leftFrame, onError);
  const rightImageData = useImageData(rightFrame, onError);

  // Re-fit the linked transform whenever either frame's image becomes
  // ready and we don't have a meaningful transform yet (initial mount
  // or post-toggle). After that, the user drives transforms via
  // wheel / drag / toolbar.
  const seededRef = useRef(false);
  useEffect(() => {
    if (!linked) return;
    if (seededRef.current) return;
    if (!leftImageData) return;
    seededRef.current = true;
    leftRef.current?.fit();
  }, [linked, leftImageData]);

  // When `linked` flips on, force a re-fit so both panes start aligned.
  // When it flips off, clear the seeded flag so the next link toggle
  // re-fits.
  useEffect(() => {
    if (linked) {
      leftRef.current?.fit();
    } else {
      seededRef.current = false;
    }
  }, [linked]);

  const handleCursor = useCallback(
    (pane: "left" | "right", c: { x: number; y: number } | null) => {
      const data = pane === "left" ? leftImageData : rightImageData;
      const frame = pane === "left" ? leftFrame : rightFrame;
      if (!c || !data) {
        onCursorChange?.(null);
        return;
      }
      const roi = frame.roi;
      const srcX = (roi?.x ?? 0) + c.x;
      const srcY = (roi?.y ?? 0) + c.y;
      onCursorChange?.({
        x: c.x,
        y: c.y,
        intensity: getPixelLum(data, srcX, srcY),
      });
    },
    [leftImageData, rightImageData, leftFrame, rightFrame, onCursorChange],
  );

  // Wire the imperative handle so ResidualViewer's toolbar can drive
  // whichever pane is active. In linked mode the canvases share state,
  // so calling either pane's methods produces the same effect.
  useEffect(() => {
    if (!innerRef) return;
    innerRef.current = {
      fitActive: () => {
        if (linked) {
          leftRef.current?.fit();
        } else {
          (activePane === "left" ? leftRef : rightRef).current?.fit();
        }
      },
      reset1to1Active: () => {
        if (linked) {
          leftRef.current?.reset1to1();
        } else {
          (activePane === "left" ? leftRef : rightRef).current?.reset1to1();
        }
      },
      zoomActiveBy: (factor) => {
        if (linked) {
          leftRef.current?.zoomBy(factor);
        } else {
          (activePane === "left" ? leftRef : rightRef).current?.zoomBy(factor);
        }
      },
    };
    return () => {
      if (innerRef) innerRef.current = null;
    };
  }, [innerRef, linked, activePane]);

  // Tab swaps the active pane in unlinked mode; in linked mode it's
  // still useful for routing stepper input to the other side.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (
        tgt &&
        (tgt.tagName === "INPUT" ||
          tgt.tagName === "SELECT" ||
          tgt.tagName === "TEXTAREA" ||
          tgt.isContentEditable)
      ) {
        return;
      }
      if (e.key === "Tab") {
        e.preventDefault();
        onActivePane(activePane === "left" ? "right" : "left");
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [activePane, onActivePane]);

  return (
    <div className="grid h-full w-full grid-cols-2 gap-2">
      <div
        className="flex h-full flex-col"
        onMouseDown={() => onActivePane("left")}
      >
        <FrameCanvas
          ref={leftRef}
          frame={leftFrame}
          residuals={residuals}
          image={leftImageData?.image ?? null}
          transform={linked ? linkedTransform : undefined}
          onTransformChange={linked ? setLinkedTransform : undefined}
          onCursor={(c) => handleCursor("left", c)}
          onError={onError}
          active={activePane === "left"}
        />
        <PaneLabel frame={leftFrame} active={activePane === "left"} />
      </div>
      <div
        className="flex h-full flex-col"
        onMouseDown={() => onActivePane("right")}
      >
        <FrameCanvas
          ref={rightRef}
          frame={rightFrame}
          residuals={residuals}
          image={rightImageData?.image ?? null}
          transform={linked ? linkedTransform : undefined}
          onTransformChange={linked ? setLinkedTransform : undefined}
          onCursor={(c) => handleCursor("right", c)}
          onError={onError}
          active={activePane === "right"}
        />
        <PaneLabel frame={rightFrame} active={activePane === "right"} />
      </div>
    </div>
  );
}

function PaneLabel({ frame, active }: { frame: FrameKey; active: boolean }) {
  return (
    <div
      className={`mt-1 font-mono text-[11px] ${
        active ? "text-brand" : "text-muted-foreground"
      }`}
    >
      pose {frame.pose} · cam {frame.camera}
    </div>
  );
}
