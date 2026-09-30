import { invoke } from "@tauri-apps/api/core";
import { imageBorderPixels } from "@vitavision/three";
import { useEffect, useMemo, useState } from "react";
import {
  BORDER_SAMPLES_PER_EDGE,
  pinholeBorderRays,
  raysFromUndistortedPixels,
} from "../../lib/frustumRays";
import { isTauriContext } from "../../lib/tauri";
import type { FxFyCxCySkew } from "../../types/generated/diagnose-wire";

/** The image border's viewing rays for camera `cameraIndex` (flat
 * `[x, y, 1, …]` on the camera's `z = 1` plane), for `CameraFrustum`.
 *
 * Starts from the distortion-free pinhole rays, then — inside Tauri —
 * asks the backend to back-project the border through the full camera
 * model (`undistort_points`) and switches to those. Stays on the pinhole
 * rays when the backend can't (no Tauri, or an export without a `cameras`
 * array such as a lifted `laserline_device`). See `lib/frustumRays.ts`. */
export function useBorderRays(
  cameraIndex: number,
  k: FxFyCxCySkew,
  width: number,
  height: number,
): Float64Array {
  const pinhole = useMemo(() => pinholeBorderRays(k, width, height), [k, width, height]);
  // Keyed by the pinhole rays it refines, so a stale answer for an
  // earlier camera/size is never shown.
  const [exact, setExact] = useState<{ for: Float64Array; rays: Float64Array } | null>(
    null,
  );

  useEffect(() => {
    if (!isTauriContext()) return;
    let cancelled = false;
    const px = imageBorderPixels(width, height, BORDER_SAMPLES_PER_EDGE);
    const pointsPx: [number, number][] = [];
    for (let i = 0; i < px.length; i += 2) pointsPx.push([px[i]!, px[i + 1]!]);
    invoke<[number, number][] | null>("undistort_points", {
      camera: cameraIndex,
      pointsPx,
    })
      .then((out) => {
        if (cancelled || !Array.isArray(out) || out.length !== pointsPx.length) return;
        const flat = out.flat();
        if (!flat.every(Number.isFinite)) return;
        setExact({ for: pinhole, rays: raysFromUndistortedPixels(k, flat) });
      })
      .catch(() => {
        // Keep the pinhole rays: the frustum is a guide, not a measurement.
      });
    return () => {
      cancelled = true;
    };
  }, [cameraIndex, k, width, height, pinhole]);

  return exact?.for === pinhole ? exact.rays : pinhole;
}
