import { useCallback, useSyncExternalStore } from "react";
import { api } from "../../api";
import { imageLoader, ThumbQueue, type ThumbState } from "../../lib/thumbQueue";

/** The app's one thumbnail queue: six loads at a time, never blocking a
 * grid (a cell is a placeholder until its image is ready). */
export const thumbs = new ThumbQueue(imageLoader, 6);

const IDLE: ThumbState = { kind: "idle" };

/** The state of the thumbnail of the image at `path` (a longest side of
 * `max` px: 160, 320, 640 or 1280); asking for it queues the load, and the
 * cell leaving the page withdraws the ask. */
export function useThumb(path: string | null, max: 160 | 320 | 640 | 1280): ThumbState {
  const url = path === null ? null : api.urls.image({ path, max });
  const subscribe = useCallback(
    (listener: () => void) => (url === null ? () => {} : thumbs.subscribe(url, listener)),
    [url],
  );
  return useSyncExternalStore(subscribe, () => (url === null ? IDLE : thumbs.get(url)));
}
