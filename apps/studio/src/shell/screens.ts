/** The app's seven screens in workflow order, their routes, their
 * `⌘1`…`⌘7` shortcuts, and the address of a place in the app.
 *
 * An address is `#/<screen>[/<tab>]?project=<dir>&run=<id>`: the screen,
 * for Inspect its tab, and the project and run it shows, so a link
 * reopens the same place. The router has no params of its own for the
 * project and run: they are the query, owned by `useUrlSync`. */

export type ScreenId =
  "library" | "dataset" | "detect" | "calibrate" | "inspect" | "compare" | "device";

export interface Screen {
  id: ScreenId;
  label: string;
  /** The route (Inspect's is the base of its tabs). */
  path: string;
  /** Whether a project must be open to use it. */
  needsProject: boolean;
  /** Whether the screen exists yet: an unbuilt one is disabled in the rail
   * and says so when its address is opened. */
  built: boolean;
}

/** In rail order, which is the workflow's. `⌘n` is the n-th. */
export const SCREENS: readonly Screen[] = [
  { id: "library", label: "Library", path: "/library", needsProject: false, built: true },
  { id: "dataset", label: "Dataset", path: "/dataset", needsProject: true, built: true },
  { id: "detect", label: "Detect", path: "/detect", needsProject: true, built: true },
  {
    id: "calibrate",
    label: "Calibrate",
    path: "/calibrate",
    needsProject: true,
    built: true,
  },
  { id: "inspect", label: "Inspect", path: "/inspect", needsProject: true, built: true },
  { id: "compare", label: "Compare", path: "/compare", needsProject: true, built: true },
  { id: "device", label: "Device", path: "/device", needsProject: false, built: true },
];

export function screenById(id: ScreenId): Screen {
  return SCREENS.find((s) => s.id === id) ?? SCREENS[0]!;
}

/** Inspect's tabs, in the order they are offered: the viewers, then the
 * per-camera panels of a run. */
export const INSPECT_TABS = [
  "scene",
  "frames",
  "parameters",
  "distortion",
  "residuals",
  "sigma",
  "quality",
  "epipolar",
  "depth",
] as const;
export type InspectTab = (typeof INSPECT_TABS)[number];

/** The tab Inspect opens on: the frame viewer (what the app opened on
 * before there were screens). */
export const DEFAULT_TAB: InspectTab = "frames";

export function isInspectTab(value: string | undefined): value is InspectTab {
  return (INSPECT_TABS as readonly string[]).includes(value ?? "");
}

/** Where the app is: the screen, Inspect's tab, the project and run. */
export interface Place {
  screen: ScreenId;
  tab: InspectTab | null;
  /** The project folder, absolute. */
  project: string | null;
  run: string | null;
}

/** The place an address names; an address of no screen is the Library. */
export function parsePlace(pathname: string, search: string): Place {
  const [, first = "", second] = pathname.split("/");
  const screen = SCREENS.find((s) => s.path === `/${first}`);
  const params = new URLSearchParams(search);
  return {
    screen: screen?.id ?? "library",
    tab: screen?.id === "inspect" ? (isInspectTab(second) ? second : DEFAULT_TAB) : null,
    project: params.get("project") || null,
    run: params.get("run") || null,
  };
}

/** The address of `place`: path and query (`/inspect/frames?project=…`). */
export function placePath(place: Place): string {
  const screen = screenById(place.screen);
  const path =
    screen.id === "inspect" ? `${screen.path}/${place.tab ?? DEFAULT_TAB}` : screen.path;
  const params = new URLSearchParams();
  if (place.project) params.set("project", place.project);
  if (place.project && place.run) params.set("run", place.run);
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

/** The address of screen `id`, keeping the project and run of `from`. */
export function screenPlace(id: ScreenId, from: Place): Place {
  return {
    screen: id,
    tab: id === "inspect" ? (from.tab ?? DEFAULT_TAB) : null,
    project: from.project,
    run: from.run,
  };
}

/** Whether `screen` can be entered: built, and with a project when it
 * needs one. */
export function isEnabled(screen: Screen, hasProject: boolean): boolean {
  return screen.built && (hasProject || !screen.needsProject);
}

/** The screen `⌘`/`Ctrl` + a digit selects (no other modifier); `null`
 * for any other key. */
export function screenForShortcut(e: {
  key: string;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
}): ScreenId | null {
  if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey) return null;
  if (!/^[1-9]$/.test(e.key)) return null;
  return SCREENS[Number(e.key) - 1]?.id ?? null;
}
