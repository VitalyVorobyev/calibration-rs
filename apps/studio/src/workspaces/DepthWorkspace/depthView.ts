/** What the Depth workspace shows of a disparity job's output: the back end
 * answers the fields as blobs (a `VAM1` plane of disparity, depth and the
 * rectified left image; the point cloud as raw floats; the rectified pair as
 * a PNG), and the colour mapping happens here, with the design system's
 * colour map. */
import { colormapRgb } from "@vitavision/charts";
import { rasterizePlane, type ValuePlane } from "@vitavision/stage2d";
import { api, fetchPlane } from "../../api";
import { fetchFloats } from "../../api/resources";
import type { DisparityResult } from "../../api/generated/types";

/** The plane's channels, in the order the back end writes them. */
export const DISPARITY = 0;
export const DEPTH = 1;
export const LEFT = 2;

/** How much of the overlay is the heatmap (the rest is the image). */
const OVERLAY_WEIGHT = 0.65;

const viridis = (t: number) => colormapRgb("viridis", t);

/** `channel` of `plane` through the colour map; a pixel without a value
 * is opaque black. */
export function heatmap(plane: ValuePlane, channel: number): Uint8ClampedArray {
  const rgba = rasterizePlane(plane, viridis, undefined, channel);
  for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
  return rgba;
}

/** The disparity heatmap blended over the rectified left image; where there
 * is no disparity, the image alone. */
export function overlay(plane: ValuePlane): Uint8ClampedArray {
  const colours = rasterizePlane(plane, viridis, undefined, DISPARITY);
  const size = plane.width * plane.height;
  const out = new Uint8ClampedArray(size * 4);
  for (let k = 0; k < size; k++) {
    const g = plane.values[LEFT * size + k] ?? 0;
    const matched = colours[4 * k + 3] !== 0;
    for (let c = 0; c < 3; c++) {
      const colour = colours[4 * k + c] ?? 0;
      out[4 * k + c] = matched ? OVERLAY_WEIGHT * colour + (1 - OVERLAY_WEIGHT) * g : g;
    }
    out[4 * k + 3] = 255;
  }
  return out;
}

/** RGBA pixels as a PNG data URL, for an `<img>`. */
function toUrl(rgba: Uint8ClampedArray, width: number, height: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("no 2D canvas");
  const pixels = new ImageData(width, height);
  pixels.data.set(rgba);
  context.putImageData(pixels, 0, 0);
  return canvas.toDataURL("image/png");
}

/** A disparity job's output, ready to show. */
export interface DisparityView {
  result: DisparityResult;
  /** Image URLs per view. */
  images: { rectified: string; disparity: string; overlay: string; depth: string };
  cloud: { positions: Float32Array; colors: Float32Array };
}

/** Fetch a disparity output's blobs and colour them. Rejects with
 * `blob_expired` when the store has dropped one (run the job again). */
export async function loadDisparityView(result: DisparityResult): Promise<DisparityView> {
  const url = (id: string) => api.urls.blob({ id });
  const [plane, positions, colors] = await Promise.all([
    fetchPlane(url(result.planes.id)),
    fetchFloats(url(result.point_positions.id)),
    fetchFloats(url(result.point_colors.id)),
  ]);
  const { width, height } = plane;
  return {
    result,
    images: {
      rectified: url(result.rectified_pair.id),
      disparity: toUrl(heatmap(plane, DISPARITY), width, height),
      overlay: toUrl(overlay(plane), width, height),
      depth: toUrl(heatmap(plane, DEPTH), width, height),
    },
    cloud: { positions, colors },
  };
}
