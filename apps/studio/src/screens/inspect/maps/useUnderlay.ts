import { useMemo } from "react";
import { api } from "../../../api";
import { useSelection } from "../../../session/selection";
import { useStore } from "../../../store";
import type { FrameKey } from "../../../types";

/** The frame a map of `camera` can sit over: the selected view's, else the
 * camera's first; `reason` says why there is none. A tiled camera's frame is
 * its region of interest of a larger image, which the full-sensor map does
 * not line up with, so it has none. */
export function underlayFrame(
  frames: readonly FrameKey[],
  camera: number,
  view: number | null,
): { frame: FrameKey | null; reason: string } {
  const own = frames.filter((f) => f.camera === camera && f.abs_path !== null);
  const frame = own.find((f) => f.pose === view) ?? own[0] ?? null;
  if (frame === null) {
    return { frame: null, reason: "The run lists no image for this camera." };
  }
  if (frame.roi) {
    return {
      frame: null,
      reason: "A tiled camera's frame does not line up with its map.",
    };
  }
  return { frame, reason: "" };
}

/** The image under `camera`'s maps, for the `MapViewer`'s `underlay`. */
export function useUnderlay(camera: number): { url: string | null; reason: string } {
  const frames = useStore((s) => s.frames);
  const view = useSelection((s) => s.selection.view);
  return useMemo(() => {
    const { frame, reason } = underlayFrame(frames, camera, view);
    return {
      url: frame?.abs_path ? api.urls.image({ path: frame.abs_path }) : null,
      reason,
    };
  }, [frames, camera, view]);
}
