import { Button } from "./ui";

export interface ZoomControlsProps {
  onFit: () => void;
  onOneToOne: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  /** Show keyboard-shortcut hints in the button titles — only where the
   * workspace actually wires those global shortcuts (Diagnose does via
   * `useKeyboardNav` + its own window keydown handler; Epipolar doesn't). */
  hints?: boolean;
}

/** Fit / 1:1 / zoom in / zoom out cluster for a `FrameCanvas`-backed
 * viewer. DiagnoseWorkspace's `ZoomControls` and EpipolarWorkspace's
 * `ZoomBar` were the same four buttons twice over — this is the one
 * definition both now share. */
export function ZoomControls({
  onFit,
  onOneToOne,
  onZoomIn,
  onZoomOut,
  hints,
}: ZoomControlsProps) {
  return (
    <div className="flex items-center gap-1">
      <Button
        size="icon"
        onClick={onZoomOut}
        title={hints ? "Zoom out (−)" : "Zoom out"}
        aria-label="Zoom out"
      >
        −
      </Button>
      <Button
        size="icon"
        onClick={onZoomIn}
        title={hints ? "Zoom in (+)" : "Zoom in"}
        aria-label="Zoom in"
      >
        +
      </Button>
      <Button onClick={onFit} title={hints ? "Fit (f)" : "Fit"}>
        Fit
      </Button>
      <Button onClick={onOneToOne} title={hints ? "1:1 (1)" : "1:1"}>
        1:1
      </Button>
    </div>
  );
}
