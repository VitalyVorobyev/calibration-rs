/** The app's workspaces, in rail order, and their `⌘1`…`⌘5` shortcuts. */

export interface Workspace {
  /** The route. */
  to: string;
  label: string;
  /** Needs the desktop app (dialogs, the file system, the calibration
   * runner): not offered by the browser viewer. */
  desktopOnly?: boolean;
}

export const WORKSPACES: readonly Workspace[] = [
  { to: "/diagnose", label: "Diagnose" },
  { to: "/viewer3d", label: "3D viewer" },
  { to: "/epipolar", label: "Epipolar" },
  { to: "/depth", label: "Depth (dense stereo)" },
  { to: "/run", label: "Run calibration", desktopOnly: true },
];

/** The workspaces the back end offers, in rail order. */
export function availableWorkspaces(desktop: boolean): readonly Workspace[] {
  return WORKSPACES.filter((w) => desktop || !w.desktopOnly);
}

/** The workspace route a keystroke selects: `⌘` (or `Ctrl`) with the digit
 * of its position in `workspaces`, no other modifier. `null` for any other
 * key. */
export function workspaceForShortcut(
  e: {
    key: string;
    metaKey: boolean;
    ctrlKey: boolean;
    altKey: boolean;
    shiftKey: boolean;
  },
  workspaces: readonly Workspace[] = WORKSPACES,
): string | null {
  if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return null;
  const index = Number(e.key) - 1;
  if (!Number.isInteger(index)) return null;
  return workspaces[index]?.to ?? null;
}
