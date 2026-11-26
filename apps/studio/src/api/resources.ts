/** Reading a resource's bytes by its URL (`api.urls.*`), on either
 * transport: the browser's `fetch` follows `/api/r/…` and the desktop's
 * `studio:` scheme alike. A failure is the back end's `StudioError`
 * envelope; `blob_expired` means the store dropped the blob and its command
 * must run again. */
import { decodePlane, type ValuePlane } from "@vitavision/stage2d";
import { asStudioError, StudioError } from "./transport";

/** The bytes at `url`; rejects with a `StudioError`. */
export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (e) {
    throw asStudioError(e);
  }
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // Not the envelope: the status says what went wrong.
    }
    if (typeof body === "object" && body !== null && "kind" in body) {
      throw asStudioError(body);
    }
    throw new StudioError({
      kind: "internal",
      message: `${response.status} ${response.statusText}`.trim(),
    });
  }
  return response.arrayBuffer();
}

/** The value plane (`VAM1`) at `url`. */
export async function fetchPlane(url: string): Promise<ValuePlane> {
  const bytes = await fetchBytes(url);
  try {
    return decodePlane(bytes);
  } catch (e) {
    throw asStudioError(e);
  }
}

/** The little-endian `f32` values at `url` (a point cloud's positions). */
export async function fetchFloats(url: string): Promise<Float32Array> {
  const bytes = await fetchBytes(url);
  if (bytes.byteLength % 4 !== 0) {
    throw new StudioError({ kind: "internal", message: `${url}: not a list of f32` });
  }
  return new Float32Array(bytes);
}
