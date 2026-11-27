/** What a per-camera tab of Inspect looks at: the run shown and the camera
 * (the app-wide selection's when the run has it, else its first). */
import type { RunRef } from "../../api/generated/types";
import { useSelection } from "../../session/selection";
import { useStore } from "../../store";

export interface Target {
  run: RunRef;
  camera: number;
  cameras: number[];
}

/** The camera of the selection when the run has it, else its first. */
export function pickCamera(
  cameras: readonly number[],
  selected: number | null,
): number | null {
  if (selected !== null && cameras.includes(selected)) return selected;
  return cameras[0] ?? null;
}

export function useTarget(): { target: Target | null } {
  const run = useStore((s) => s.run);
  const cameras = useStore((s) => s.cameraValues);
  const selected = useSelection((s) => s.selection.camera);
  const camera = pickCamera(cameras, selected);
  return {
    target: run !== null && camera !== null ? { run, camera, cameras } : null,
  };
}
