/** The words a map says about itself, from its own values: where it is
 * largest and how much of it is distinguishable from zero. Only what the
 * plane holds, with the library's own significance cut; no thresholds of
 * our own. Pure. */
import type { MapGrid, ResidualMap } from "../../../api/generated/types";
import type { MapSource } from "./mapSource";
import {
  channelIndex,
  nodeAt,
  peakOf,
  readoutNumber,
  shareAtLeast,
  SIGNIFICANCE_P01,
} from "./mapModel";

/** `64 × 48 nodes of 20 px` */
export function gridText(grid: MapGrid): string {
  return `${grid.columns} × ${grid.rows} nodes of ${grid.stride} px`;
}

function at(x: number, y: number): string {
  return `(${Math.round(x)}, ${Math.round(y)})`;
}

/** Where on the image a position is, in words: its corner or edge, or the
 * middle. */
export function placeWords(grid: MapGrid, x: number, y: number): string {
  const [w, h] = grid.image_size;
  const col = x < w / 3 ? "left" : x > (2 * w) / 3 ? "right" : "";
  const row = y < h / 3 ? "top" : y > (2 * h) / 3 ? "bottom" : "";
  if (!col && !row) return "near the centre";
  if (col && row) return `in the ${row}-${col} corner`;
  return `at the ${col || row} edge`;
}

/** The largest value of `channel` and where. */
function largest(source: MapSource, channel: string): string | null {
  const { plane, grid, channels } = source;
  const index = channelIndex(channels, channel);
  const peak = peakOf(plane, grid, index);
  if (!peak) return null;
  const unit = channels[index]?.unit ?? "";
  return `${readoutNumber(peak.value)} ${unit} ${placeWords(grid, peak.x, peak.y)} ${at(peak.x, peak.y)}`;
}

export function distortionStatement(source: MapSource, layer: string): string | null {
  const text = largest(source, "magnitude");
  return text ? `The ${layer} displaces a pixel by up to ${text}.` : null;
}

export function sigmaStatement(source: MapSource): string | null {
  const { plane, grid, channels } = source;
  const index = channelIndex(channels, "sigma");
  const peak = peakOf(plane, grid, index);
  if (!peak) return null;
  const centre = nodeAt(grid, grid.image_size[0] / 2, grid.image_size[1] / 2);
  const size = plane.width * plane.height;
  const centreValue = centre
    ? plane.values[index * size + centre.row * plane.width + centre.column]!
    : Number.NaN;
  const max = `${readoutNumber(peak.value)} px ${placeWords(grid, peak.x, peak.y)} ${at(peak.x, peak.y)}`;
  return Number.isFinite(centreValue)
    ? `Where a ray lands is known to ${readoutNumber(centreValue)} px (1σ) at the centre and up to ${max}.`
    : `Where a ray lands is known to up to ${max} (1σ).`;
}

/** What a residual map says: where the systematic error peaks and, when the
 * map has a significance test, how much of the mapped area it covers. */
export function residualStatement(source: MapSource): string | null {
  const { plane, channels } = source;
  const peak = largest(source, "mean_magnitude");
  if (!peak) return null;
  const magnitude = channelIndex(channels, "mean_magnitude");
  const significance = channelIndex(channels, "significance");
  const share =
    significance >= 0
      ? shareAtLeast(plane, magnitude, significance, SIGNIFICANCE_P01)
      : null;
  const head = `The systematic error peaks at ${peak}.`;
  if (!share) return head;
  const percent = Math.round(share.share * 100);
  return `${head} ${percent} % of the area that could be tested differs from zero (χ ≥ ${SIGNIFICANCE_P01}, p < 0.01).`;
}

/** The provenance line of a residual map: its kernel and its sample. */
export function residualProvenance(map: ResidualMap, run: string): string {
  const how =
    map.bandwidth_selection.kind === "cross_validated" ? "cross-validated" : "given";
  const views = `${map.views} view${map.views === 1 ? "" : "s"}`;
  return [
    run,
    `camera ${map.camera}`,
    "residual map",
    gridText(map.grid),
    `bandwidth ${Math.round(map.bandwidth_px)} px (${how})`,
    views,
    `${map.used} features`,
  ].join(" · ");
}
