import { Histogram } from "@vitavision/charts";

/** Luminance range the pre-binned counts span: 8-bit values, upper edge exclusive. */
const LUMINANCE_DOMAIN: [number, number] = [0, 256];

/** The ROI luminance distribution. `counts` is the equal-width pre-binned
 * histogram from `rectHistogram`; `intensity` is the cursor's luminance in
 * data units (not a bin index) — the chart highlights the bin containing it.
 * Empty `counts` draws the chart's empty frame. */
export function RoiHistogram({
  counts,
  intensity,
}: {
  counts: number[];
  intensity: number | null;
}) {
  return (
    <div className="w-60">
      <Histogram
        counts={counts}
        domain={LUMINANCE_DOMAIN}
        {...(intensity != null ? { cursor: intensity } : {})}
        variant="fluid"
        height={96}
        xLabel="luminance"
        label="ROI luminance histogram"
      />
    </div>
  );
}
