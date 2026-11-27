/** How the app writes a number: a value with its uncertainty, and a
 * quantity with its unit. Every table, tooltip and readout goes through
 * here, so a value is written one way everywhere.
 *
 * A parameter is `1412.37 ± 0.21`: the decimals come from σ (its first two
 * significant digits), and the value is rounded to the same decimal. A
 * value without an uncertainty is `1412.37 ·`, the reason on hover (fixed,
 * or the typed reason it is unavailable). Nothing here parses or rounds a
 * value the back end sent: it only writes it. */

/** What is known of a value's uncertainty. */
export type Uncertainty =
  /** Its standard deviation, in the value's unit. */
  | { kind: "sigma"; sigma: number }
  /** Held fixed by the configuration: it has none. */
  | { kind: "fixed" }
  /** Estimated, but the back end could not give one; `reason` says why. */
  | { kind: "unavailable"; reason: string };

/** A value written for display. */
export interface FormattedValue {
  /** The value's digits alone (`1412.37`). */
  value: string;
  /** σ's digits (`0.21`), or `null` when there is none. */
  sigma: string | null;
  /** The whole text: `1412.37 ± 0.21 px`, or `1412.37 · px`. */
  text: string;
  /** What to say on hover when there is no σ (fixed, unavailable); else
   * `null`. */
  title: string | null;
}

/** Decimals that keep two significant digits of `sigma`; 0 for σ ≥ 10. */
export function sigmaDecimals(sigma: number): number {
  if (!Number.isFinite(sigma) || sigma <= 0) return 0;
  const decimals = 1 - Math.floor(Math.log10(sigma));
  return Math.min(Math.max(decimals, 0), 12);
}

/** `value` at `decimals`, never a negative zero (`-0.00` reads as a sign
 * error), and a non-finite value as its name. */
export function fixed(value: number, decimals: number): string {
  if (Number.isNaN(value)) return "NaN";
  if (!Number.isFinite(value)) return value > 0 ? "∞" : "−∞";
  const text = value.toFixed(Math.min(Math.max(decimals, 0), 20));
  const clean = /^-0(\.0*)?$/.test(text) ? text.slice(1) : text;
  return clean.replace("-", "−");
}

/** `value` to `digits` significant digits, without trailing zeros: the
 * way a value with no σ is written. */
export function significant(value: number, digits = 6): string {
  if (!Number.isFinite(value)) return fixed(value, 0);
  if (value === 0) return "0";
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  if (magnitude >= 9 || magnitude <= -6) {
    return value
      .toExponential(digits - 1)
      .replace(/\.?0+e/, "e")
      .replace("-", "−");
  }
  const decimals = Math.max(digits - 1 - magnitude, 0);
  const text = fixed(value, decimals);
  return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
}

/** `value ± σ unit`, with the decimals of σ; `value · unit` and a reason
 * when there is no σ. */
export function formatWithSigma(
  value: number,
  uncertainty: Uncertainty | null | undefined,
  unit?: string,
): FormattedValue {
  const suffix = unit ? ` ${unit}` : "";
  if (
    uncertainty?.kind === "sigma" &&
    Number.isFinite(uncertainty.sigma) &&
    uncertainty.sigma > 0 &&
    Number.isFinite(value)
  ) {
    const decimals = sigmaDecimals(uncertainty.sigma);
    const v = fixed(value, decimals);
    const s = fixed(uncertainty.sigma, decimals);
    return { value: v, sigma: s, text: `${v} ± ${s}${suffix}`, title: null };
  }
  const v = significant(value);
  const title =
    uncertainty?.kind === "fixed"
      ? "fixed: held at this value, not estimated"
      : uncertainty?.kind === "unavailable"
        ? `unavailable: ${uncertainty.reason}`
        : uncertainty?.kind === "sigma"
          ? "unavailable: σ is not a positive number"
          : "no uncertainty known";
  return { value: v, sigma: null, text: `${v} ·${suffix}`, title };
}

/** Pixels, at `decimals` (3: a reprojection RMS). */
export function formatPixels(px: number, decimals = 3): string {
  return `${fixed(px, decimals)} px`;
}

/** A length in metres, written in millimetres below one metre. */
export function formatLength(metres: number): string {
  if (!Number.isFinite(metres)) return fixed(metres, 0);
  return Math.abs(metres) < 1
    ? `${significant(metres * 1000, 4)} mm`
    : `${significant(metres, 4)} m`;
}

/** An angle in radians, written in degrees. */
export function formatAngle(radians: number): string {
  return `${significant((radians * 180) / Math.PI, 4)}°`;
}

/** A solver cost, `3.2e-1`: two digits, always an exponent. */
export function formatCost(cost: number): string {
  if (!Number.isFinite(cost)) return fixed(cost, 0);
  return cost.toExponential(1).replace("e+", "e").replace("-", "−");
}

/** A whole number with thin grouping (`12 345`). */
export function formatCount(n: number): string {
  if (!Number.isFinite(n)) return fixed(n, 0);
  return Math.round(n)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

/** A duration in milliseconds: `42 ms`, `3.2 s`, `2 min 05 s`. */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "–";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${fixed(seconds, seconds < 10 ? 1 : 0)} s`;
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  return `${minutes} min ${String(whole % 60).padStart(2, "0")} s`;
}
