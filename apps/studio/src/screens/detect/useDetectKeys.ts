/** The Detect screen's own keys: `X` leaves the picked features out (or, with
 * none picked, this camera's image), `⇧X` the whole view, `⌘Z` undoes the last
 * exclusion. Stepping views and cameras is `useViewKeys`. Nothing fires while
 * typing or while a dialog is open. */
import { useEffect, useRef } from "react";
import { isTyping } from "../../shell/keymap";
import { undoExclusion } from "./useExclusions";

export interface DetectKeyActions {
  /** `X`: the picked features, else this camera's image. */
  exclude: () => void;
  /** `⇧X`: the whole view. */
  excludeView: () => void;
}

export function useDetectKeys(actions: DetectKeyActions, enabled: boolean): void {
  const latest = useRef(actions);
  useEffect(() => {
    latest.current = actions;
  });
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey || isTyping(e.target)) return;
      if (document.querySelector('[role="dialog"], [role="alertdialog"]') !== null)
        return;
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.code === "KeyZ") {
        e.preventDefault();
        void undoExclusion();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.code !== "KeyX") return;
      e.preventDefault();
      if (e.shiftKey) latest.current.excludeView();
      else latest.current.exclude();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}
