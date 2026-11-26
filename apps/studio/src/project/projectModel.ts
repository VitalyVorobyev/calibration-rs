/** Pure project-model rules: run names and the recent-projects list. */

/** How many recent projects the shell remembers. */
export const RECENT_LIMIT = 8;

/** The next free `run-NNN` name: one past the highest numbered run, so a
 * deleted run's number is not reused while later runs exist. */
export function nextRunName(existing: readonly string[]): string {
  let highest = 0;
  for (const name of existing) {
    const match = /^run-(\d+)$/.exec(name);
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  return `run-${String(highest + 1).padStart(3, "0")}`;
}

/** `recent` with `dir` moved to the front, without duplicates, capped at
 * `RECENT_LIMIT`. */
export function rememberRecent(recent: readonly string[], dir: string): string[] {
  return [dir, ...recent.filter((d) => d !== dir)].slice(0, RECENT_LIMIT);
}

/** `recent` without `dir` (a project that no longer opens). */
export function forgetRecent(recent: readonly string[], dir: string): string[] {
  return recent.filter((d) => d !== dir);
}

/** The last path segment, for labels. */
export function folderName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return idx >= 0 ? trimmed.slice(idx + 1) : trimmed;
}

const RECENT_KEY = "calib-recent-projects";

/** The remembered recent projects. A per-viewer convenience: an
 * unavailable or corrupt store reads as empty. */
export function loadRecent(storage: Storage | null = safeStorage()): string[] {
  try {
    const parsed: unknown = JSON.parse(storage?.getItem(RECENT_KEY) ?? "[]");
    return Array.isArray(parsed)
      ? parsed.filter((d): d is string => typeof d === "string").slice(0, RECENT_LIMIT)
      : [];
  } catch {
    return [];
  }
}

/** Remember `recent`; a store that refuses the write is ignored. */
export function saveRecent(
  recent: readonly string[],
  storage: Storage | null = safeStorage(),
): void {
  try {
    storage?.setItem(RECENT_KEY, JSON.stringify(recent));
  } catch {
    // Private mode or blocked storage: the list lives for this session only.
  }
}

function safeStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}
