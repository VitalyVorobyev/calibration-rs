// Reading a calibration export's parameter uncertainty for display: the
// covariance's rows in display units, its correlation matrix and a sentence
// for an unavailable reason.

import type {
  ParameterCovariance,
  Uncertainty,
  UncertaintyUnavailable,
} from "../types/generated/diagnose-wire";
import type { AnyExport } from "../store/types";

/** The export's uncertainty: top-level on every export except the laserline
 * device, which keeps it with its estimate. `null` when the export has none
 * (written without uncertainty). */
export function exportUncertainty(data: AnyExport): Uncertainty | null {
  return data.uncertainty ?? data.estimate?.uncertainty ?? null;
}

/** A display unit and the factor from the label's own unit to it. */
interface DisplayUnit {
  unit: string;
  scale: number;
}

const DEGREES: DisplayUnit = { unit: "°", scale: 180 / Math.PI };
const MILLIMETRES: DisplayUnit = { unit: "mm", scale: 1000 };
const PIXELS: DisplayUnit = { unit: "px", scale: 1 };
const PLAIN: DisplayUnit = { unit: "", scale: 1 };

/** How a label is shown: rotations (`*.rx`, `tilt_x`) in degrees,
 * translations (`*.tx`, `plane.d`) in millimetres, intrinsics in pixels,
 * coefficients and normal components as they are. */
export function displayUnit(label: string): DisplayUnit {
  const name = label.slice(label.lastIndexOf(".") + 1);
  if (/^r[xyz]$/.test(name) || /^tilt_[xy]$/.test(name)) return DEGREES;
  if (/^t[xyz]$/.test(name) || label.endsWith("plane.d")) return MILLIMETRES;
  if (/^(fx|fy|cx|cy)$/.test(name)) return PIXELS;
  return PLAIN;
}

export interface ParameterRow {
  label: string;
  /** The value in display units; `null` for a pose or plane coordinate,
   * whose covariance is over a perturbation rather than a value. */
  value: number | null;
  /** Standard deviation in display units. */
  sigma: number;
  unit: string;
}

/** One row per shared parameter, `values` aligned with `cov.labels`. */
export function parameterRows(
  cov: ParameterCovariance,
  values: readonly (number | null)[],
): ParameterRow[] {
  return cov.labels.map((label, i) => {
    const { unit, scale } = displayUnit(label);
    const value = values[i];
    return {
      label,
      value: value == null ? null : value * scale,
      sigma: Math.sqrt(Math.max(0, cov.covariance[i]?.[i] ?? 0)) * scale,
      unit,
    };
  });
}

/** The correlation matrix of the shared parameters; `NaN` where a variance
 * is zero. */
export function correlationMatrix(cov: ParameterCovariance): number[][] {
  const sd = cov.labels.map((_, i) =>
    Math.sqrt(Math.max(0, cov.covariance[i]?.[i] ?? 0)),
  );
  return cov.covariance.map((row, i) =>
    row.map((v, j) => {
      const denom = sd[i]! * sd[j]!;
      return denom > 0 ? Math.max(-1, Math.min(1, v / denom)) : Number.NaN;
    }),
  );
}

/** A sentence for why a result has no covariance. */
export function unavailableText(reason: UncertaintyUnavailable): string {
  switch (reason.kind) {
    case "not_converged":
      return "The solve stopped on its iteration budget, short of a minimum.";
    case "rank_deficient":
      return `The data does not determine: ${reason.unobservable.join(", ")}.`;
    case "at_bound":
      return `These parameters ended on a bound: ${reason.labels.join(", ")}.`;
    case "no_redundancy":
      return "There are no more observations than estimated parameters.";
    case "not_computed":
      return "The export was written without uncertainty.";
  }
}
