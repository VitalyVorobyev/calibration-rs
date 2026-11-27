/** The camera × view matrix of a dataset as `SequenceMatrix` takes it: the
 * rows and columns from a `project_views` reply, the absolute path of a
 * cell's image (what the `image` resource wants), and what is wrong with a
 * view. Pure; shared by the New project preview and the Dataset screen. */
import type { ProjectViews, ViewRow } from "../../api/generated/types";
import { joinPath } from "../../lib/paths";

/** A cell's address. */
export interface CellAddress {
  /** The view's token. */
  token: string;
  /** The camera's index in the manifest. */
  camera: number;
}

/** The absolute path of `view`'s image for camera `camera`, or `null`
 * where the camera has none for it. `dataRoot` is the folder the manifest's
 * relative paths resolve against. */
export function imagePath(
  dataRoot: string,
  view: ViewRow,
  camera: number,
): string | null {
  const rel = view.images[camera];
  return rel == null ? null : joinPath(dataRoot, rel);
}

/** Whether `view` lacks an image of some camera. */
export function hasGap(view: ViewRow): boolean {
  return view.images.some((image) => image == null);
}

/** The cameras whose image of `view` is missing, as indices. */
export function missingCameras(view: ViewRow): number[] {
  return view.images.flatMap((image, i) => (image == null ? [i] : []));
}

/** What narrows the rows of the matrix. */
export interface RowFilter {
  /** Only views with a missing image. */
  problemsOnly: boolean;
  /** Only views whose token contains this text (case-insensitive). */
  token: string;
}

export const NO_ROW_FILTER: RowFilter = { problemsOnly: false, token: "" };

/** The views that pass `filter`, in their order. */
export function filterViews(views: readonly ViewRow[], filter: RowFilter): ViewRow[] {
  const needle = filter.token.trim().toLowerCase();
  return views.filter(
    (v) =>
      (!filter.problemsOnly || hasGap(v)) &&
      (needle === "" || v.token.toLowerCase().includes(needle)),
  );
}

/** The view with `token`, and its position among `views`. */
export function findView(
  views: readonly ViewRow[],
  token: string,
): { view: ViewRow; index: number } | null {
  const index = views.findIndex((v) => v.token === token);
  const view = views[index];
  return view === undefined ? null : { view, index };
}

/** The reply's cameras as columns need them: id and how many views have an
 * image of the camera. */
export function cameraSummaries(
  reply: Pick<ProjectViews, "cameras" | "views">,
): { id: string; images: number }[] {
  return reply.cameras.map((id, i) => ({
    id,
    images: reply.views.reduce((n, v) => n + (v.images[i] == null ? 0 : 1), 0),
  }));
}
