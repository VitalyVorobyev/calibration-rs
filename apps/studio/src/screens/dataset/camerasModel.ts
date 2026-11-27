/** Editing the list of cameras of a manifest: the next free id, a new
 * camera, a duplicate (its id the next one, its image pattern following the
 * id), moving one. Pure. */
import type { CameraSource, ImagePattern } from "../../api/generated/types";

/** The id `cam<n>` with the smallest `n` not taken (counting from the
 * number of cameras, so a list `cam0, cam1` offers `cam2`). */
export function nextCameraId(cameras: readonly CameraSource[]): string {
  const taken = new Set(cameras.map((c) => c.id));
  for (let n = cameras.length; ; n++) {
    if (!taken.has(`cam${n}`)) return `cam${n}`;
  }
}

/** A camera whose images are `<id>/*.png`. */
export function newCamera(id: string): CameraSource {
  return { id, images: { kind: "glob", pattern: `${id}/*.png` } };
}

/** The next id after `id`: its trailing number counted up past any taken id
 * (`left` has none and becomes `left2`). */
export function nextIdAfter(id: string, taken: ReadonlySet<string>): string {
  const m = /^(.*?)(\d*)$/.exec(id);
  const stem = m?.[1] ?? id;
  const digits = m?.[2] ?? "";
  let n = digits === "" ? 2 : Number(digits) + 1;
  const width = digits.length;
  for (;;) {
    const candidate = `${stem}${String(n).padStart(width, "0")}`;
    if (!taken.has(candidate)) return candidate;
    n++;
  }
}

/** `pattern` with the old id replaced by the new where it appears (a
 * `cam0/*.png` of camera `cam0` becomes `cam1/*.png`). */
function followId(pattern: ImagePattern, from: string, to: string): ImagePattern {
  if (from === "") return pattern;
  // The id as a whole name (`cam0` in `cam0/*.png` or `cam0_*.png`), never
  // inside another word (`a` in `frames`).
  const name = new RegExp(
    `(^|[^A-Za-z0-9])${from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![A-Za-z0-9])`,
    "g",
  );
  const swap = (s: string) => s.replace(name, `$1${to}`);
  return pattern.kind === "glob"
    ? { kind: "glob", pattern: swap(pattern.pattern) }
    : { kind: "list", paths: pattern.paths.map(swap) };
}

/** A copy of `camera` for the list `cameras`: the next free id, the image
 * patterns following it. */
export function duplicateCamera(
  camera: CameraSource,
  cameras: readonly CameraSource[],
): CameraSource {
  const id = nextIdAfter(camera.id, new Set(cameras.map((c) => c.id)));
  const copy: CameraSource = {
    ...camera,
    id,
    images: followId(camera.images, camera.id, id),
  };
  if (camera.laser_images)
    copy.laser_images = followId(camera.laser_images, camera.id, id);
  return copy;
}

/** `list` with the item at `index` moved by `by` places (no-op at an end). */
export function move<T>(list: readonly T[], index: number, by: -1 | 1): T[] {
  const to = index + by;
  if (to < 0 || to >= list.length) return [...list];
  const next = [...list];
  const [item] = next.splice(index, 1);
  next.splice(to, 0, item as T);
  return next;
}
