/** What the path picker offers for each kind of pick, and its breadcrumb:
 * pure rules over the back end's `list_dir` entries. */
import type { DirEntry } from "../../api/generated/types";

/** What is being picked. */
export type PickMode = "folder" | "project" | "manifest" | "export" | "file";

/** The noun of each mode, for the dialog's text. */
export const MODE_NOUN: Record<PickMode, string> = {
  folder: "a folder",
  project: "a project",
  manifest: "a dataset manifest (.toml or .json)",
  export: "a calibration export (.json)",
  file: "a file",
};

/** Whether the listing shows files for `mode` (folders always show). */
export function showsFiles(mode: PickMode): boolean {
  return mode === "manifest" || mode === "export" || mode === "file";
}

function extension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** Whether `entry` itself is a valid answer for `mode`. */
export function selectable(entry: DirEntry, mode: PickMode): boolean {
  switch (mode) {
    case "folder":
      return entry.kind === "dir";
    case "project":
      return entry.kind === "dir" && entry.is_project;
    case "manifest":
      return entry.kind === "file" && ["toml", "json"].includes(extension(entry.name));
    case "export":
      return entry.kind === "file" && extension(entry.name) === "json";
    case "file":
      return entry.kind === "file";
  }
}

/** One step of the breadcrumb: its label and the folder it opens (`null`:
 * the roots). */
export interface Crumb {
  label: string;
  path: string | null;
}

/** The breadcrumb of `path`: the roots, the root it lies under (by its full
 * path), then each folder below it. `roots` are canonical paths. */
export function breadcrumb(path: string | null, roots: readonly string[]): Crumb[] {
  const crumbs: Crumb[] = [{ label: "Folders", path: null }];
  if (path === null) return crumbs;
  const sep = path.includes("\\") && !path.includes("/") ? "\\" : "/";
  const root = roots
    .filter((r) => path === r || path.startsWith(r.endsWith(sep) ? r : r + sep))
    .sort((a, b) => b.length - a.length)[0];
  if (root === undefined) return [...crumbs, { label: path, path }];
  crumbs.push({ label: root, path: root });
  let current = root;
  for (const part of path.slice(root.length).split(sep).filter(Boolean)) {
    current = current.endsWith(sep) ? current + part : current + sep + part;
    crumbs.push({ label: part, path: current });
  }
  return crumbs;
}
