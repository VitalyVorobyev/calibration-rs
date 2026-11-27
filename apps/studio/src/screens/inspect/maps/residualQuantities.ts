import type { MapKind } from "./viewerModel";

export interface ResidualQuantity {
  channel: string;
  label: string;
  kind: MapKind;
}

/** What the residual map can show, in the order of the menu. The default is
 * the systematic error. Signed components take the diverging map; the rest
 * are magnitudes. */
export const RESIDUAL_QUANTITIES: readonly ResidualQuantity[] = [
  { channel: "mean_magnitude", label: "Systematic |μ|", kind: "magnitude" },
  { channel: "rms", label: "RMS", kind: "magnitude" },
  { channel: "mean_dx", label: "μx", kind: "signed" },
  { channel: "mean_dy", label: "μy", kind: "signed" },
  { channel: "std_error", label: "Standard error", kind: "magnitude" },
  { channel: "significance", label: "Significance χ", kind: "magnitude" },
  { channel: "effective_samples", label: "Support: samples", kind: "magnitude" },
  { channel: "effective_views", label: "Support: views", kind: "magnitude" },
];
