import { useEffect, useEffectEvent } from "react";

/** The zoom commands of a frame view. */
export interface ZoomCommands {
  fit: () => void;
  oneToOne: () => void;
  zoomBy: (factor: number) => void;
}

/** One zoom step of `+` / `-`. */
export const ZOOM_STEP = 1.25;

/** What a key does to a frame view, or `null` for a key it ignores:
 * `f` fit, `1` one-to-one, `+`/`=` zoom in, `-`/`_` zoom out, `l` toggle
 * linking. */
export function zoomKeyAction(
  key: string,
): "fit" | "oneToOne" | "zoomIn" | "zoomOut" | "link" | null {
  switch (key) {
    case "f":
    case "F":
      return "fit";
    case "1":
      return "oneToOne";
    case "+":
    case "=":
      return "zoomIn";
    case "-":
    case "_":
      return "zoomOut";
    case "l":
    case "L":
      return "link";
    default:
      return null;
  }
}

/** True when a keystroke belongs to a text field rather than the view. */
export function isEditableTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "SELECT" ||
    el.tagName === "TEXTAREA" ||
    el.isContentEditable
  );
}

/** Window-level single-key zoom shortcuts, so they work without the view
 * having focus; ignored over editable elements. `onLink` (when given) is
 * bound to `l`. */
export function useZoomShortcuts(
  zoom: ZoomCommands,
  options: { enabled: boolean; onLink?: (() => void) | undefined },
): void {
  const { enabled, onLink } = options;
  const handle = useEffectEvent((e: KeyboardEvent) => {
    if (isEditableTarget(e.target)) return;
    const action = zoomKeyAction(e.key);
    if (action === null) return;
    if (action === "link") {
      if (!onLink) return;
      onLink();
    } else if (action === "fit") zoom.fit();
    else if (action === "oneToOne") zoom.oneToOne();
    else zoom.zoomBy(action === "zoomIn" ? ZOOM_STEP : 1 / ZOOM_STEP);
    e.preventDefault();
  });

  useEffect(() => {
    if (!enabled) return;
    const listener = (e: KeyboardEvent) => handle(e);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [enabled]);
}
