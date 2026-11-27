/** The range a colour scale covers, shared by the maps and the frame
 * overlay so the two read alike: from zero to a round top that fits the
 * data, a few extreme values clipped (and said to be) instead of washing
 * out the rest. */
import type { ColormapName } from "@vitavision/charts";

/** The sequential colour map of a magnitude drawn over an image (low values
 * go dark, the image shows through) and on the bare canvas. */
export const UNDERLAY_COLORMAP: ColormapName = "inferno";
export const CANVAS_COLORMAP: ColormapName = "viridis";

/** The share of values that sets the top of an automatic range. */
const TOP_PERCENTILE = 0.995;

/** A maximum this close above the percentile is the data's own, not an
 * outlier: the range reaches it and nothing clips. */
const OUTLIER_FACTOR = 1.25;

/** `value` rounded up to two significant digits (0.0329 → 0.033, 54.3 →
 * 55): a round number that still fits the data; 1 for a non-positive or
 * non-finite one. */
export function niceCeil(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const exponent = Math.floor(Math.log10(value));
  const unit = 10 ** (exponent - 1);
  const rounded = Math.ceil(value / unit - 1e-9) * unit;
  // Trim the float noise of the product (0.033000000000000002).
  return Number(rounded.toPrecision(2));
}

/** The nearest-rank `p` quantile of `values` (any order); `NaN` when empty. */
export function quantile(values: readonly number[], p: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const at = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[at]!;
}

/** The top of the range for non-negative `magnitudes`: their largest when it
 * is within a quarter of the 99.5th percentile, else the percentile, rounded
 * up to two significant digits; `null` without a finite value. */
export function rangeTop(magnitudes: readonly number[]): number | null {
  const finite = magnitudes.filter(Number.isFinite);
  if (finite.length === 0) return null;
  const top = quantile(finite, TOP_PERCENTILE);
  const largest = finite.reduce((m, v) => Math.max(m, v), 0);
  return niceCeil(largest <= top * OUTLIER_FACTOR ? largest : top);
}
