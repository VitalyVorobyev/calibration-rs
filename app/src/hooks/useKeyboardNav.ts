import { useEffect } from "react";

interface KeyboardNavOptions {
  /** Called on ArrowLeft / ArrowRight (signed step). No-op when an
   * editable control has focus. */
  onPoseStep?: (delta: number) => void;
  /** Called on ArrowUp / ArrowDown (signed step). No-op when an
   * editable control has focus. */
  onCameraStep?: (delta: number) => void;
  /** When false, listeners are not registered. */
  enabled?: boolean;
}

/** Window-scoped arrow-key navigation. Ignores the keystroke when an
 * `<input>` / `<select>` / `<textarea>` (or anything `contenteditable`)
 * has focus, so typing into a search field still works. */
export function useKeyboardNav(opts: KeyboardNavOptions): void {
  const { onPoseStep, onCameraStep, enabled = true } = opts;
  useEffect(() => {
    if (!enabled) return;
    const handler = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement | null;
      if (tgt) {
        const tag = tgt.tagName;
        if (
          tag === "INPUT" ||
          tag === "SELECT" ||
          tag === "TEXTAREA" ||
          tgt.isContentEditable
        ) {
          return;
        }
      }
      switch (e.key) {
        case "ArrowLeft":
          onPoseStep?.(-1);
          e.preventDefault();
          break;
        case "ArrowRight":
          onPoseStep?.(+1);
          e.preventDefault();
          break;
        case "ArrowUp":
          onCameraStep?.(-1);
          e.preventDefault();
          break;
        case "ArrowDown":
          onCameraStep?.(+1);
          e.preventDefault();
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onPoseStep, onCameraStep, enabled]);
}
