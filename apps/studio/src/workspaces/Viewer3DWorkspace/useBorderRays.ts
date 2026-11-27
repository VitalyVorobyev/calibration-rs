import { api } from "../../api";
import { imageBorderPixels } from "@vitavision/three";
import { useEffect, useMemo, useState } from "react";
import {
  BORDER_SAMPLES_PER_EDGE,
  pinholeBorderRays,
  raysFromUndistortedPixels,
} from "../../lib/frustumRays";
import type { FxFyCxCy } from "../../api/generated/types";

/** The image border's viewing rays for camera `cameraIndex` (flat
 * `[x, y, 1, …]` on the camera's `z = 1` plane), for `CameraFrustum`.
 *
 * Starts from the distortion-free pinhole rays, then — with a back end —
 * asks it to back-project the border through the full camera model
 * (`undistort_points`) and switches to those. Stays on the pinhole rays
 * when the back end can't (none, or an export without a `cameras` array
 * such as a lifted `laserline_device`). See `lib/frustumRays.ts`. */
export function useBorderRays(
  cameraIndex: number,
  k: FxFyCxCy,
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
    let cancelled = false;
    const px = imageBorderPixels(width, height, BORDER_SAMPLES_PER_EDGE);
    const pointsPx: [number, number][] = [];
    for (let i = 0; i < px.length; i += 2) pointsPx.push([px[i]!, px[i + 1]!]);
    api
      .undistortPoints({ camera: cameraIndex, points_px: pointsPx })
      .then((out) => {
        if (cancelled || !Array.isArray(out) || out.length !== pointsPx.length) return;
        const flat: number[] = [];
        for (const pt of out) {
          if (!pt || !Number.isFinite(pt[0]) || !Number.isFinite(pt[1])) return;
          flat.push(pt[0], pt[1]);
        }
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
