/** The app's global keys, as pure rules: a keystroke and whether the user
 * is typing become an action or nothing. The hook that listens
 * (`useShellKeys`) carries the action out.
 *
 * `⌘`/`Ctrl` chords work even in a text field (a screen switch, the
 * palette, save); bare keys (`[`, `,`, `?`, `Esc`) never do, so typing is
 * never taken. Brackets, comma and period are read by their physical key
 * (`code`), so `⇧[` is the run step on any keyboard layout. */
import { screenForShortcut, type ScreenId } from "./screens";

export type KeyAction =
  | { kind: "screen"; id: ScreenId }
  | { kind: "view"; delta: -1 | 1 }
  | { kind: "camera"; delta: -1 | 1 }
  | { kind: "run"; delta: -1 | 1 }
  | { kind: "escape" }
  | { kind: "help" }
  | { kind: "palette" }
  | { kind: "save" };

export interface KeyLike {
  key: string;
  code: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}

/** The action `e` means, or `null`. `typing`: focus is in a text control. */
export function keyAction(e: KeyLike, typing: boolean): KeyAction | null {
  const chord = e.metaKey || e.ctrlKey;
  if (chord) {
    if (e.altKey) return null;
    const id = screenForShortcut(e);
    if (id !== null) return { kind: "screen", id };
    if (!e.shiftKey && e.key.toLowerCase() === "k") return { kind: "palette" };
    if (!e.shiftKey && e.key.toLowerCase() === "s") return { kind: "save" };
    return null;
  }
  if (typing || e.altKey) return null;
  switch (e.code) {
    case "BracketLeft":
      return e.shiftKey ? { kind: "run", delta: -1 } : { kind: "view", delta: -1 };
    case "BracketRight":
      return e.shiftKey ? { kind: "run", delta: 1 } : { kind: "view", delta: 1 };
    case "Comma":
      return e.shiftKey ? null : { kind: "camera", delta: -1 };
    case "Period":
      return e.shiftKey ? null : { kind: "camera", delta: 1 };
    default:
      break;
  }
  if (e.key === "Escape") return { kind: "escape" };
  if (e.key === "?") return { kind: "help" };
  return null;
}

/** Whether focus is in something the user types into. */
export function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== "string") return false;
  const tag = el.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    el.isContentEditable ||
    el.getAttribute("role") === "textbox" ||
    el.getAttribute("role") === "combobox"
  );
}
