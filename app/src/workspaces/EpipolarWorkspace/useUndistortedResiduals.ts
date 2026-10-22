import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import type { TargetFeatureResidual } from "../../types";

type Px = [number, number];

/** `residuals` with their observed pixels mapped into `camera`'s
 * undistorted frame by the backend (the Epipolar panes show undistorted
 * images). Empty until the mapping arrives; `onError` hears a failure. */
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
    const points: Promise<Px[]> =
      residuals.length > 0
        ? invoke<Px[]>("undistort_points", {
            camera,
            pointsPx: residuals.map((r) => r.observed_px),
          })
        : Promise.resolve([]);
    points
      .then((px) => {
        if (cancelled) return;
        setMapped({
          source: residuals,
          camera,
          value: residuals.map((r, i) => ({ ...r, observed_px: px[i] ?? r.observed_px })),
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
