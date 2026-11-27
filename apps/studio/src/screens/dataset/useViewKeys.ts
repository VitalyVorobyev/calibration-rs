/** `[` `]` step the selected view and `,` `.` the camera on the Dataset
 * screen. With a run open the shell's keys already do it (through the
 * selection bridge), so this stands down then and never steps twice. */
import { useEffect } from "react";
import { useSelection } from "../../session/selection";
import { isTyping } from "../../shell/keymap";

/** The index `delta` places from `current` among `count`, kept inside; the
 * first one when there is no current. */
export function stepIndex(
  current: number | null,
  delta: number,
  count: number,
): number | null {
  if (count <= 0) return null;
  if (current === null) return delta > 0 ? 0 : count - 1;
  return Math.min(count - 1, Math.max(0, current + delta));
}

export function useViewKeys(enabled: boolean, views: number, cameras: number): void {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || isTyping(e.target)) return;
      if (document.querySelector('[role="dialog"], [role="alertdialog"]') !== null)
        return;
      const { selection, setSelection } = useSelection.getState();
      const view = e.key === "[" ? -1 : e.key === "]" ? 1 : 0;
      const camera = e.key === "," ? -1 : e.key === "." ? 1 : 0;
      if (view === 0 && camera === 0) return;
      e.preventDefault();
      setSelection({
        camera:
          camera === 0
            ? (selection.camera ?? (cameras > 0 ? 0 : null))
            : stepIndex(selection.camera, camera, cameras),
        view: view === 0 ? selection.view : stepIndex(selection.view, view, views),
        feature: null,
      });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled, views, cameras]);
}
