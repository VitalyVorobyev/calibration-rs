/** The image the screen is on: the selection's camera and view looked up in
 * the overview, with the paths the stage and the commands need. */
import type {
  DetectionOverview,
  ImageDetection,
  ViewDetections,
} from "../../api/generated/types";
import { joinPath } from "../../lib/paths";
import type { Selection } from "../../lib/selection";

export interface Cell {
  /** Index into `overview.views`. */
  viewIndex: number;
  /** Index into `overview.cameras`. */
  camera: number;
  cameraId: string;
  token: string;
  view: ViewDetections;
  /** The camera's image of the view; `null` where it has none. */
  detection: ImageDetection | null;
  /** The image relative to the data folder, as commands and exclusions name it. */
  image: string | null;
  /** The image's absolute path, for the `image` resource. */
  path: string | null;
}

/** The cell `selection` names, or `null` without a camera and a view. */
export function currentCell(
  overview: DetectionOverview | undefined,
  selection: Selection,
  dataRoot: string,
): Cell | null {
  if (!overview || selection.camera === null || selection.view === null) return null;
  const view = overview.views[selection.view];
  const cameraId = overview.cameras[selection.camera];
  if (!view || cameraId === undefined) return null;
  const detection = view.images[selection.camera] ?? null;
  const image = detection?.image ?? null;
  return {
    viewIndex: selection.view,
    camera: selection.camera,
    cameraId,
    token: view.token,
    view,
    detection,
    image,
    path: image === null ? null : joinPath(dataRoot, image),
  };
}
