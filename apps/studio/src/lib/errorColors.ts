/** The colours of the frame overlay: a residual's size on the same
 * sequential colour map and range rule as the residual map, so the arrows
 * and the map read alike. Colours come from the charts package's colour
 * maps; nothing here names a colour. */
import { colormapValue, type ColormapName } from "@vitavision/charts";
import { rangeTop, UNDERLAY_COLORMAP } from "./colorRange";
import type { LaserFeatureResidual, TargetFeatureResidual } from "../types";

/** A colour scale for the frame overlay: `[0, top]` on `map`, `clipped`
 * when some value lies above `top`. */
export interface ErrorRamp {
  map: ColormapName;
  domain: [number, number];
  unit: string;
  clipped: boolean;
}

/** Colours are taken at this many steps, so a frame's arrows batch into a
 * few paths instead of one per distinct error. */
const LEVELS = 24;

/** The ramp over `values` (their absolute values, in `unit`): zero to a
 * round top that fits them. */
export function errorRamp(values: readonly number[], unit: string): ErrorRamp {
  const magnitudes = values.map(Math.abs);
  const top = rangeTop(magnitudes) ?? 1;
  return {
    map: UNDERLAY_COLORMAP,
    domain: [0, top],
    unit,
    clipped: magnitudes.some((v) => v > top),
  };
}

/** The colour of `value` on `ramp`, in `LEVELS` steps (saturating at the ends). */
export function colorForValue(ramp: ErrorRamp, value: number): string {
  const [low, high] = ramp.domain;
  const t = Math.min(1, Math.max(0, (Math.abs(value) - low) / (high - low)));
  return colormapValue(
    ramp.map,
    low + (Math.round(t * LEVELS) / LEVELS) * (high - low),
    ramp.domain,
  );
}

/** A target residual's size, px: the export's `error_px`, else the distance
 * between the observed and projected corners; `null` when it has neither. */
export function targetError(r: TargetFeatureResidual): number | null {
  if (typeof r.error_px === "number") return r.error_px;
  if (!r.projected_px) return null;
  return Math.hypot(
    r.observed_px[0] - r.projected_px[0],
    r.observed_px[1] - r.projected_px[1],
  );
}

/** The ramp of every target residual of a run, px. */
export function targetRamp(all: readonly TargetFeatureResidual[]): ErrorRamp {
  return errorRamp(
    all.flatMap((r) => targetError(r) ?? []),
    "px",
  );
}

/** A laser residual's point-to-plane distance, mm; `null` without one. */
export function laserError(r: LaserFeatureResidual): number | null {
  return typeof r.residual_m === "number" ? Math.abs(r.residual_m) * 1e3 : null;
}

/** The ramp of every laser residual of a run, mm. */
export function laserRamp(all: readonly LaserFeatureResidual[]): ErrorRamp {
  return errorRamp(
    all.flatMap((r) => laserError(r) ?? []),
    "mm",
  );
}
