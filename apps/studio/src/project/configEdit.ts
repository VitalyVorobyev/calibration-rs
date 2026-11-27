/** Pure rules of editing a run's configuration: what differs between two
 * configurations, the one-off patch that turns the project's configuration
 * into the edited one, and how many changes a patch makes over another.
 *
 * A configuration is the workflow config object as a form edits it. A patch
 * is the library's: tables merge key by key, any other value (a number, a
 * list, `null`) replaces what it lands on. A list is therefore one value:
 * a difference inside a list is a difference of the whole list. */
import { diffPaths, getAtPath, setAtPath } from "@vitavision/forms";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** `path` cut at the first array it passes through (the array is the
 * value a patch replaces), checked against `root`. */
function wholeArrayPath(root: unknown, path: string): string {
  if (path === "") return path;
  const segments = path.split(".");
  let current: unknown = root;
  for (let i = 0; i < segments.length; i++) {
    if (Array.isArray(current)) return segments.slice(0, i).join(".");
    if (!isRecord(current)) return path;
    current = current[segments[i]!];
  }
  return path;
}

/** The paths at which `config` differs from `base`, each list a single
 * path, sorted and without repeats. */
export function changedPaths(config: unknown, base: unknown): string[] {
  const paths = diffPaths(config, base).map((p) => {
    // The shorter of the two readings: a path inside a list in either.
    const a = wholeArrayPath(config, p);
    const b = wholeArrayPath(base, p);
    return a.length <= b.length ? a : b;
  });
  return [...new Set(paths)].sort();
}

/** The patch that makes `config` from `base` when merged over a patch that
 * made `base`: each changed path set to its value in `config`, a removed
 * optional field set to `null`. Empty (`{}`) when they are the same. */
export function patchOver(config: unknown, base: unknown): Record<string, unknown> {
  let patch: unknown = {};
  for (const path of changedPaths(config, base)) {
    if (path === "") return isRecord(config) ? { ...config } : {};
    const value = getAtPath(config, path);
    patch = setAtPath(patch, path, value === undefined ? null : value);
  }
  return isRecord(patch) ? patch : {};
}

/** A patch as its leaves: each table descended into, anything else (a
 * number, a list, `null`) one value, keyed by its dot path. */
export function patchLeaves(patch: unknown, prefix = ""): Map<string, string> {
  const leaves = new Map<string, string>();
  if (isRecord(patch) && Object.keys(patch).length > 0) {
    for (const [key, value] of Object.entries(patch)) {
      for (const [path, text] of patchLeaves(value, prefix ? `${prefix}.${key}` : key)) {
        leaves.set(path, text);
      }
    }
  } else if (prefix !== "") {
    leaves.set(prefix, JSON.stringify(patch));
  }
  return leaves;
}

/** The leaves at which patch `a` is not patch `b`: set differently, or
 * by one only. Sorted. */
export function patchChanges(a: unknown, b: unknown): string[] {
  const left = patchLeaves(a);
  const right = patchLeaves(b);
  const paths = new Set([...left.keys(), ...right.keys()]);
  return [...paths].filter((p) => left.get(p) !== right.get(p)).sort();
}

/** A path as words, the last `depth` segments: `rig.refine_intrinsics` →
 * `rig · refine intrinsics`. */
export function pathLabel(path: string, depth = 2): string {
  if (path === "") return "configuration";
  return path
    .split(".")
    .slice(-depth)
    .map((s) => s.replace(/_/g, " "))
    .join(" · ");
}

/** Whether `value` holds no key: an empty patch. */
export function isEmptyPatch(patch: unknown): boolean {
  return !isRecord(patch) || Object.keys(patch).length === 0;
}

/** A configuration value as short text for a "was → is" line. */
export function shortValue(value: unknown, limit = 28): string {
  let text: string;
  if (value === undefined) text = "default";
  else if (value === null) text = "none";
  else if (typeof value === "string") text = value;
  else if (typeof value === "number" || typeof value === "boolean") text = String(value);
  else if (Array.isArray(value) && value.every((v) => typeof v === "string"))
    text = value.length === 0 ? "none" : value.join(" ");
  else if (Array.isArray(value) && value.some((v) => typeof v === "object" && v !== null))
    text =
      value.length === 0
        ? "none"
        : `${value.length} item${value.length === 1 ? "" : "s"}`;
  else text = JSON.stringify(value) ?? "";
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
