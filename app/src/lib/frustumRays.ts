/** Border rays for the 3D viewer's camera frustums.
 *
 * `@vitavision/three`'s `CameraFrustum` does no camera math: it takes the
 * image border's viewing rays as camera-frame points on the `z = 1` plane
 * and draws them. This module produces those rays from the two sources the
 * frontend has:
 *
 * - **Exact**: the `undistort_points` Tauri command back-projects raw
 *   pixels through the camera's full model (distortion, Scheimpflug) and
 *   returns them in the undistorted pixel frame, i.e. `K · ray`. Applying
 *   `K⁻¹` here gives the true border rays; a distorted border bows.
 * - **Pinhole fallback**: `K⁻¹` of the raw border pixels, distortion
 *   ignored. Used outside a Tauri webview, and for exports the backend
 *   cannot back-project (a lifted single-camera `laserline_device`).
 *
 * Pure and side-effect free so it unit-tests without React/Tauri. */
import { imageBorderPixels } from "@vitavision/three";
import type { FxFyCxCySkew } from "../types/generated/diagnose-wire";

/** Samples per image edge. A pinhole border is straight, but a distorted
 * one is not, so the exact source needs more than the four corners. */
export const BORDER_SAMPLES_PER_EDGE = 8;

/** `K⁻¹` of flat `[u0, v0, u1, v1, …]` undistorted pixels: flat
 * `[x0, y0, 1, x1, …]` points on the camera's `z = 1` plane. Mirrors
 * `undistorted_pixel_to_ray` in `src-tauri/src/epipolar.rs`, skew included. */
export function raysFromUndistortedPixels(
  k: FxFyCxCySkew,
  pixels: ArrayLike<number>,
): Float64Array {
  const n = Math.floor(pixels.length / 2);
  const out = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const y = (pixels[2 * i + 1]! - k.cy) / k.fy;
    const x = (pixels[2 * i]! - k.cx - k.skew * y) / k.fx;
    out[3 * i] = x;
    out[3 * i + 1] = y;
    out[3 * i + 2] = 1;
  }
  return out;
}

/** Distortion-free border rays of `k` for a `width × height` image
 * (clockwise from the top-left corner, see `imageBorderPixels`). */
export function pinholeBorderRays(
  k: FxFyCxCySkew,
  width: number,
  height: number,
  perEdge = BORDER_SAMPLES_PER_EDGE,
): Float64Array {
  return raysFromUndistortedPixels(k, imageBorderPixels(width, height, perEdge));
}
