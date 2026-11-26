import { api } from "../../api";
import { useEffect, useState } from "react";
import type { TargetFeatureResidual } from "../../types";

type Px = [number, number];
type NullablePx = Px | null;

/** `residuals` with their observed pixels mapped into `camera`'s
 * undistorted frame by the backend (the Epipolar panes show undistorted
 * images); points without a finite unit-depth ray are skipped.
 * Empty until the mapping arrives; `onError` hears a failure. */
export function useUndistortedResiduals(
  camera: number,
  residuals: TargetFeatureResidual[],
  onError: (msg: string) => void,
): TargetFeatureResidual[] {
  const [mapped, setMapped] = useState<{
    source: TargetFeatureResidual[];
    camera: number;
    value: TargetFeatureResidual[];
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const points: Promise<NullablePx[]> =
      residuals.length > 0
        ? api.undistortPoints({
            camera,
            points_px: residuals.map((r) => r.observed_px),
          })
        : Promise.resolve([]);
    points
      .then((px) => {
        if (cancelled) return;
        const next: TargetFeatureResidual[] = [];
        for (let i = 0; i < residuals.length; i++) {
          const out = px[i];
          if (out && Number.isFinite(out[0]) && Number.isFinite(out[1])) {
            next.push({ ...residuals[i]!, observed_px: out });
          }
        }
        setMapped({
          source: residuals,
          camera,
          value: next,
        });
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setMapped({ source: residuals, camera, value: [] });
        onError(`undistort points failed: ${String(e)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [camera, residuals, onError]);

  return mapped && mapped.source === residuals && mapped.camera === camera
    ? mapped.value
    : NONE;
}

const NONE: TargetFeatureResidual[] = [];
