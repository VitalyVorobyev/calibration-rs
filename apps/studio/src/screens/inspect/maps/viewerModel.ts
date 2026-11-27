/** The rules of the shared map viewer that are not drawing: which colour
 * map a quantity gets, the domain of a fixed or automatic range, the
 * iso-line step, and what the legend marks. */
import type { ColormapName } from "@vitavision/charts";
import { readoutNumber } from "./mapModel";

/** A magnitude runs `[0, D]` on a sequential map; a signed quantity
 * `[−D, D]` on a diverging one. */
export type MapKind = "magnitude" | "signed";

export type ColormapChoice = "auto" | Exclude<ColormapName, "blueOrange">;

export const SEQUENTIAL_CHOICES: readonly ColormapChoice[] = [
  "auto",
  "viridis",
  "cividis",
  "magma",
  "inferno",
];

/** The colour map drawn: the diverging one for a signed quantity (never a
 * choice); for a magnitude the person's choice, else viridis on the canvas
 * and inferno over an image (low values go dark, the image shows through). */
export function effectiveColormap(
  kind: MapKind,
  choice: ColormapChoice,
  underlay: boolean,
): ColormapName {
  if (kind === "signed") return "blueOrange";
  if (choice !== "auto") return choice;
  return underlay ? "inferno" : "viridis";
}

/** The opacity of the heatmap: over an image it lets the image through. */
export function heatOpacity(underlay: boolean): number {
  return underlay ? 0.7 : 1;
}

/** The domain of the colours: `fixedMax` (a person's top value) when set,
 * else the automatic one; a flat `[0, 1]` when the channel has no value. */
export function chosenDomain(
  kind: MapKind,
  auto: [number, number] | null,
  fixedMax: number | null,
): [number, number] {
  if (fixedMax !== null && fixedMax > 0) {
    return kind === "signed" ? [-fixedMax, fixedMax] : [0, fixedMax];
  }
  return auto ?? (kind === "signed" ? [-1, 1] : [0, 1]);
}

/** The spacing of iso-lines over `domain`: 1, 2 or 5 × 10ⁿ giving about
 * three to six lines across the span. */
export function isoStep(domain: readonly [number, number]): number {
  const span = domain[1] - domain[0];
  if (!(span > 0)) return 1;
  const raw = span / 6;
  const exponent = Math.floor(Math.log10(raw));
  const base = 10 ** exponent;
  for (const m of [1, 2, 5, 10]) if (m * base >= raw * (1 - 1e-12)) return m * base;
  return 10 * base;
}

/** What the legend says of a saturated scale. */
export function legendClipped(
  high: boolean,
  low: boolean,
): "high" | "low" | "both" | undefined {
  if (high && low) return "both";
  if (high) return "high";
  if (low) return "low";
  return undefined;
}

/** The ticks the legend carries: a named threshold (a limit that is a
 * place on the scale, not a colour change) and the value under the pointer. */
export function legendMarkers(
  threshold: { value: number; label: string } | undefined,
  cursor: number | undefined,
  unit: string,
): { value: number; label?: string }[] {
  const markers: { value: number; label?: string }[] = [];
  if (threshold) markers.push({ value: threshold.value, label: threshold.label });
  if (cursor !== undefined && Number.isFinite(cursor)) {
    markers.push({
      value: cursor,
      label: `${readoutNumber(cursor)}${unit ? ` ${unit}` : ""}`,
    });
  }
  return markers;
}
