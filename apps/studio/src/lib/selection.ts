/** The app-wide selection and how it changes. Every panel reads and writes
 * the same `{ camera, view, feature }` (plus the current run, which the
 * project owns); a hover is a transient selection of the same shape.
 *
 * The rules are here, pure: choosing a view drops the feature (it belongs
 * to a view), choosing a camera keeps the view (the same pose seen by
 * another camera) and drops the feature, and `Esc` clears the selection one
 * level at a time, innermost first. */

export interface Selection {
  camera: number | null;
  view: number | null;
  feature: number | null;
}

export const NO_SELECTION: Selection = { camera: null, view: null, feature: null };

export function sameSelection(a: Selection, b: Selection): boolean {
  return a.camera === b.camera && a.view === b.view && a.feature === b.feature;
}

export function withCamera(selection: Selection, camera: number | null): Selection {
  return camera === selection.camera
    ? selection
    : { ...selection, camera, feature: null };
}

export function withView(selection: Selection, view: number | null): Selection {
  return view === selection.view ? selection : { ...selection, view, feature: null };
}

export function withFeature(selection: Selection, feature: number | null): Selection {
  return feature === selection.feature ? selection : { ...selection, feature };
}

/** One `Esc`: the feature goes first, then the view, then the camera. The
 * same selection when nothing is selected (the next `Esc` closes the
 * transient panels). */
export function escapeSelection(selection: Selection): Selection {
  if (selection.feature !== null) return { ...selection, feature: null };
  if (selection.view !== null) return { ...selection, view: null };
  if (selection.camera !== null) return { ...selection, camera: null };
  return selection;
}

/** The value `delta` places from `current` in `values`, wrapping round the
 * ends; the first (or last, going back) value when `current` is not in the
 * list; `null` for an empty list. */
export function stepWrap(
  values: readonly number[],
  current: number | null,
  delta: number,
): number | null {
  if (values.length === 0) return null;
  const at = current === null ? -1 : values.indexOf(current);
  if (at < 0) return delta >= 0 ? values[0]! : values[values.length - 1]!;
  const n = values.length;
  return values[(((at + delta) % n) + n) % n]!;
}

/** The id `delta` places from `current` in `ids` (listed newest first),
 * stopping at the ends rather than wrapping: the oldest run is not next to
 * the newest. */
export function stepId(
  ids: readonly string[],
  current: string | null,
  delta: number,
): string | null {
  if (ids.length === 0) return null;
  const at = current === null ? -1 : ids.indexOf(current);
  if (at < 0) return ids[0]!;
  return ids[Math.min(Math.max(at + delta, 0), ids.length - 1)]!;
}
