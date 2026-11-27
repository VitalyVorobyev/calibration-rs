/** What a project leaves out of its dataset, as pure operations on the
 * project's `[exclude]`: toggle a view, one camera's image of a view, or
 * single features of one image, and ask whether something is excluded.
 * Every operation returns a new `Exclusions`; none is written here (the
 * screen saves the result through `save_project`).
 *
 * A feature is named by its target point as the detector labelled it
 * (`world_xyz`, metres); the library matches a point to a feature within a
 * micrometre in each coordinate, and so does `samePoint`. */
import type {
  Exclusions,
  FeatureExclusion,
  ImageExclusion,
  ViewExclusion,
} from "../../api/generated/types";

export type Point3 = readonly [number, number, number];

/** The library's tolerance when it matches a point to a feature, metres. */
const POINT_TOLERANCE_M = 1e-6;

export function samePoint(a: Point3, b: Point3): boolean {
  return a.every((v, i) => Math.abs(v - b[i]!) <= POINT_TOLERANCE_M);
}

/** The view's identity as the screen knows it: its token and the images
 * it holds (relative paths, `null` where a camera has none). */
export interface ViewIdentity {
  token: string;
  images: readonly (string | null)[];
}

/** The excluding entries of `view`, however they name it (by token or by
 * one of its images). */
function viewEntries(exclude: Exclusions, view: ViewIdentity): ViewExclusion[] {
  return (exclude.views ?? []).filter(
    (e) =>
      ("token" in e.view && e.view.token === view.token) ||
      ("image" in e.view && view.images.includes(e.view.image)),
  );
}

export function isViewExcluded(exclude: Exclusions, view: ViewIdentity): boolean {
  return viewEntries(exclude, view).length > 0;
}

export function isImageExcluded(
  exclude: Exclusions,
  camera: string,
  image: string,
): boolean {
  return (exclude.images ?? []).some((e) => e.camera === camera && e.image === image);
}

/** Whether camera `camera`'s `image` of `view` is left out, by its own
 * entry or because the whole view is. */
export function isImageOut(
  exclude: Exclusions,
  view: ViewIdentity,
  camera: string,
  image: string,
): boolean {
  return isViewExcluded(exclude, view) || isImageExcluded(exclude, camera, image);
}

export function isFeatureExcluded(
  exclude: Exclusions,
  camera: string,
  image: string,
  point: Point3,
): boolean {
  return (exclude.features ?? []).some(
    (e) =>
      e.camera === camera &&
      e.image === image &&
      e.points.some((p) => samePoint(p, point)),
  );
}

/** The view left out, or put back when it was. */
export function toggleView(exclude: Exclusions, view: ViewIdentity): Exclusions {
  if (isViewExcluded(exclude, view)) {
    const gone = new Set(viewEntries(exclude, view));
    return { ...exclude, views: (exclude.views ?? []).filter((e) => !gone.has(e)) };
  }
  return {
    ...exclude,
    views: [...(exclude.views ?? []), { view: { token: view.token } }],
  };
}

/** One camera's image left out, or put back when it was. */
export function toggleImage(
  exclude: Exclusions,
  camera: string,
  image: string,
): Exclusions {
  if (isImageExcluded(exclude, camera, image)) {
    return {
      ...exclude,
      images: (exclude.images ?? []).filter(
        (e) => !(e.camera === camera && e.image === image),
      ),
    };
  }
  const entry: ImageExclusion = { camera, image };
  return { ...exclude, images: [...(exclude.images ?? []), entry] };
}

/** `points` of one image left out, all of them unless every one already
 * was, in which case they are put back (one gesture, `X`, twice). An entry
 * left with no point is dropped. */
export function toggleFeatures(
  exclude: Exclusions,
  camera: string,
  image: string,
  points: readonly Point3[],
): Exclusions {
  if (points.length === 0) return exclude;
  const all = points.every((p) => isFeatureExcluded(exclude, camera, image, p));
  const mine = (e: FeatureExclusion) => e.camera === camera && e.image === image;
  const existing = (exclude.features ?? []).find(mine);
  const kept = (existing?.points ?? []).filter(
    (p) => !points.some((q) => samePoint(p, q)),
  );
  const next: Point3[] = all
    ? kept
    : [...kept, ...points].map((p) => [p[0], p[1], p[2]] as [number, number, number]);
  const others = (exclude.features ?? []).filter((e) => !mine(e));
  const entry: FeatureExclusion = {
    ...(existing ?? {}),
    camera,
    image,
    points: next as [number, number, number][],
  };
  return { ...exclude, features: next.length === 0 ? others : [...others, entry] };
}

/** The entries' notes: set the note of one entry (empty clears it). */
export function withNote<K extends "views" | "images" | "features">(
  exclude: Exclusions,
  kind: K,
  index: number,
  note: string,
): Exclusions {
  const list = [...(exclude[kind] ?? [])] as { note?: string | null }[];
  const entry = list[index];
  if (!entry) return exclude;
  const { note: _old, ...rest } = entry;
  list[index] = note.trim() === "" ? rest : { ...rest, note: note.trim() };
  return { ...exclude, [kind]: list };
}

/** The entries of one scope, removed. */
export function without(
  exclude: Exclusions,
  kind: "views" | "images" | "features",
  index: number,
): Exclusions {
  return { ...exclude, [kind]: (exclude[kind] ?? []).filter((_, i) => i !== index) };
}

export interface ExclusionCounts {
  views: number;
  images: number;
  features: number;
}

/** `3 views · 2 images · 14 features` as numbers. */
export function countExclusions(exclude: Exclusions): ExclusionCounts {
  return {
    views: (exclude.views ?? []).length,
    images: (exclude.images ?? []).length,
    features: (exclude.features ?? []).reduce((n, e) => n + e.points.length, 0),
  };
}

export function totalExcluded(counts: ExclusionCounts): number {
  return counts.views + counts.images + counts.features;
}

/** The counts as the tray says them; zero counts are left out. */
export function describeCounts(counts: ExclusionCounts): string {
  const parts = [
    [counts.views, "view", "views"],
    [counts.images, "image", "images"],
    [counts.features, "feature", "features"],
  ] as const;
  const said = parts
    .filter(([n]) => n > 0)
    .map(([n, one, many]) => `${n} ${n === 1 ? one : many}`);
  return said.length === 0 ? "Nothing excluded" : said.join(" · ");
}

/** A phrase for an operation, for a toast: "Excluded 3 features". */
export function describeChange(before: Exclusions, after: Exclusions): string {
  const a = countExclusions(before);
  const b = countExclusions(after);
  const change = (one: string, many: string, d: number) =>
    d === 0
      ? null
      : `${d > 0 ? "Excluded" : "Restored"} ${Math.abs(d)} ${Math.abs(d) === 1 ? one : many}`;
  return (
    change("view", "views", b.views - a.views) ??
    change("image", "images", b.images - a.images) ??
    change("feature", "features", b.features - a.features) ??
    "No change"
  );
}

/** The entries of `exclude.features` for one image whose points match no
 * feature of that image's detection (`points` are its features' target
 * points): they leave nothing out, which the panel says. */
export function unmatchedPoints(
  exclude: Exclusions,
  camera: string,
  image: string,
  detected: readonly Point3[],
): Point3[] {
  return (exclude.features ?? [])
    .filter((e) => e.camera === camera && e.image === image)
    .flatMap((e) => e.points)
    .filter((p) => !detected.some((d) => samePoint(p, d)));
}
