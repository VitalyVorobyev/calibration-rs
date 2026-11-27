/** What the map viewer computes from a decoded plane, pure: the colour
 * domain, where a node sits in the image, what the pointer is over, the
 * arrows, the significance mask and the statement a map makes about itself.
 *
 * A plane is `columns × rows` nodes, each the centre of a `stride × stride`
 * block of the camera's image (`MapGrid`), row-major from the top-left;
 * pixel centres are at integer coordinates, so block `(c, r)` spans
 * `[c·stride − ½, (c+1)·stride − ½)` and a trailing partial block is
 * centred on its in-image pixels. `NaN` is "no value", never a colour. */
import type { Rect, ValuePlane } from "@vitavision/stage2d";
import type { MapGrid, PlaneChannel } from "../../../api/generated/types";

/** The χ above which a residual field differs from zero at p < 0.01 (the
 * library's value, `residual_map`'s documentation). */
export const SIGNIFICANCE_P01 = 3.03;

/** The share of finite values that sets the top of an automatic domain:
 * a few extreme nodes do not wash out the rest. */
const DOMAIN_PERCENTILE = 0.995;

/** The smallest of 1, 2, 5 × 10ⁿ that is at least `value`; 1 for a
 * non-positive or non-finite one. */
export function niceCeil(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  for (const step of [1, 2, 5, 10]) {
    if (step * base >= value * (1 - 1e-12)) return step * base;
  }
  return 10 * base;
}

/** The index of the channel named `name` in `channels`; −1 when absent. */
export function channelIndex(channels: readonly PlaneChannel[], name: string): number {
  return channels.findIndex((c) => c.name === name);
}

/** The finite values of a channel. */
export function finiteValues(plane: ValuePlane, channel: number): number[] {
  const size = plane.width * plane.height;
  const out: number[] = [];
  if (channel < 0 || channel >= plane.channels) return out;
  const base = channel * size;
  for (let i = 0; i < size; i++) {
    const v = plane.values[base + i]!;
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

/** The nearest-rank `p` quantile of `values` (any order); `NaN` when empty. */
export function quantile(values: readonly number[], p: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const at = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[at]!;
}

/** The domain a channel is coloured over by default: `[0, D]` for a
 * magnitude and `[−D, D]` for a signed channel, `D` the 99.5th percentile
 * of the finite magnitudes rounded up to 1, 2 or 5 × 10ⁿ. `null` for a
 * channel with no finite value. */
export function autoDomain(
  plane: ValuePlane,
  channel: number,
  signed: boolean,
): [number, number] | null {
  const values = finiteValues(plane, channel);
  if (values.length === 0) return null;
  const top = quantile(signed ? values.map(Math.abs) : values, DOMAIN_PERCENTILE);
  const d = niceCeil(top);
  return signed ? [-d, d] : [0, d];
}

/** Whether any finite value lies above the top of `domain`: the legend then
 * says the scale clips. */
export function clipsHigh(
  plane: ValuePlane,
  channel: number,
  domain: readonly [number, number],
): boolean {
  return finiteValues(plane, channel).some((v) => v > domain[1]);
}

/** Whether any finite value lies below the bottom of `domain`. */
export function clipsLow(
  plane: ValuePlane,
  channel: number,
  domain: readonly [number, number],
): boolean {
  return finiteValues(plane, channel).some((v) => v < domain[0]);
}

/** The image-coordinate rectangle a plane of `grid` is stretched over: the
 * blocks laid edge to edge from the image's top-left corner. A partial
 * last block makes it run past the image, which the stage clips. */
export function planeRect(grid: MapGrid): Rect {
  return {
    x: -0.5,
    y: -0.5,
    width: grid.columns * grid.stride,
    height: grid.rows * grid.stride,
  };
}

/** The image position of the centre of node `(column, row)`. */
export function nodeCentre(grid: MapGrid, column: number, row: number): [number, number] {
  const [width, height] = grid.image_size;
  const s = grid.stride;
  return [
    (column * s + Math.min((column + 1) * s, width) - 1) / 2,
    (row * s + Math.min((row + 1) * s, height) - 1) / 2,
  ];
}

/** The node whose block holds image position `(x, y)`; `null` outside the
 * image. */
export function nodeAt(
  grid: MapGrid,
  x: number,
  y: number,
): { column: number; row: number } | null {
  const [width, height] = grid.image_size;
  if (x < -0.5 || y < -0.5 || x >= width - 0.5 || y >= height - 0.5) return null;
  return {
    column: Math.min(grid.columns - 1, Math.max(0, Math.floor((x + 0.5) / grid.stride))),
    row: Math.min(grid.rows - 1, Math.max(0, Math.floor((y + 0.5) / grid.stride))),
  };
}

/** What the pointer is over: the node, its centre and every channel's value
 * there (`NaN` where there is none). */
export interface Readout {
  column: number;
  row: number;
  /** The pointer's image position. */
  x: number;
  y: number;
  values: number[];
}

export function readoutAt(
  plane: ValuePlane,
  grid: MapGrid,
  x: number,
  y: number,
): Readout | null {
  const node = nodeAt(grid, x, y);
  if (!node) return null;
  const size = plane.width * plane.height;
  const at = node.row * plane.width + node.column;
  const values: number[] = [];
  for (let c = 0; c < plane.channels; c++)
    values.push(plane.values[c * size + at] ?? NaN);
  return { ...node, x, y, values };
}

/** Arrows for a quiver: every `every`-th node in both directions that has a
 * finite `(dx, dy)` (and, with `keep`, a `keep.channel` of at least
 * `keep.minimum`, or, with `below`, of less or none), as `VectorField` takes them. `ids` is the node's flat
 * index. */
export function quiver(
  plane: ValuePlane,
  grid: MapGrid,
  dxChannel: number,
  dyChannel: number,
  every: number,
  keep?: { channel: number; minimum: number; below?: boolean },
): { anchors: number[]; vectors: number[]; ids: number[] } {
  const anchors: number[] = [];
  const vectors: number[] = [];
  const ids: number[] = [];
  if (dxChannel < 0 || dyChannel < 0) return { anchors, vectors, ids };
  const size = plane.width * plane.height;
  for (let r = 0; r < plane.height; r += every) {
    for (let c = 0; c < plane.width; c += every) {
      const at = r * plane.width + c;
      const dx = plane.values[dxChannel * size + at]!;
      const dy = plane.values[dyChannel * size + at]!;
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
      if (keep) {
        const passes = plane.values[keep.channel * size + at]! >= keep.minimum;
        if (passes === (keep.below ?? false)) continue;
      }
      const [x, y] = nodeCentre(grid, c, r);
      anchors.push(x, y);
      vectors.push(dx, dy);
      ids.push(at);
    }
  }
  return { anchors, vectors, ids };
}

/** The spacing of arrows, in nodes, that keeps a quiver readable: about 16
 * across a 64-column plane. */
export function quiverEvery(columns: number): number {
  return Math.max(1, Math.round(columns / 16));
}

/** `plane` with `channel` kept only where `testChannel` is at least
 * `minimum`: the bright layer of a significance mask. A node whose test
 * value is `NaN` is not kept. Returns a one-channel plane. */
export function maskedBelow(
  plane: ValuePlane,
  channel: number,
  testChannel: number,
  minimum: number,
): ValuePlane {
  const size = plane.width * plane.height;
  const values = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const test = plane.values[testChannel * size + i]!;
    values[i] = test >= minimum ? plane.values[channel * size + i]! : Number.NaN;
  }
  return { ...plane, channels: 1, values };
}

/** `plane`'s one channel as a plane of its own (what a layer draws). */
export function singleChannel(plane: ValuePlane, channel: number): ValuePlane {
  const size = plane.width * plane.height;
  return {
    ...plane,
    channels: 1,
    values: plane.values.subarray(channel * size, (channel + 1) * size),
  };
}

/** Where a channel is largest: the node's image position and the value;
 * `null` for no finite value. */
export function peakOf(
  plane: ValuePlane,
  grid: MapGrid,
  channel: number,
): { value: number; x: number; y: number } | null {
  if (channel < 0 || channel >= plane.channels) return null;
  const size = plane.width * plane.height;
  let best = -Infinity;
  let at = -1;
  for (let i = 0; i < size; i++) {
    const v = plane.values[channel * size + i]!;
    if (Number.isFinite(v) && v > best) {
      best = v;
      at = i;
    }
  }
  if (at < 0) return null;
  const [x, y] = nodeCentre(grid, at % plane.width, Math.floor(at / plane.width));
  return { value: best, x, y };
}

/** Of the nodes with a value in `channel` and a test value in `testChannel`
 * (a node the test could not judge is not counted), the share whose test
 * value is at least `minimum`; `null` when no node could be tested. */
export function shareAtLeast(
  plane: ValuePlane,
  channel: number,
  testChannel: number,
  minimum: number,
): { tested: number; share: number } | null {
  const size = plane.width * plane.height;
  let tested = 0;
  let over = 0;
  for (let i = 0; i < size; i++) {
    if (!Number.isFinite(plane.values[channel * size + i]!)) continue;
    const test = plane.values[testChannel * size + i]!;
    if (!Number.isFinite(test)) continue;
    tested += 1;
    if (test >= minimum) over += 1;
  }
  return tested === 0 ? null : { tested, share: over / tested };
}

/** The mean of a channel's finite values; `NaN` when there is none. */
export function meanOf(plane: ValuePlane, channel: number): number {
  const values = finiteValues(plane, channel);
  return values.length === 0
    ? Number.NaN
    : values.reduce((a, b) => a + b, 0) / values.length;
}

/** A channel's unit as written beside a value: the significance's `chi`
 * is the letter. */
export function unitLabel(unit: string): string {
  return unit === "chi" ? "χ" : unit;
}

/** A number for a readout: three significant digits, `no data` for `NaN`. */
export function readoutNumber(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "no data";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(2);
  return String(Number(value.toPrecision(3))).replace("-", "−");
}
