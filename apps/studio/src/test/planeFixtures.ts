/** Value planes and map replies for tests: a `VAM1` buffer as the back end
 * writes it (`planes.rs`), and the typed replies of the map commands over a
 * small grid. */
import type { ValuePlane } from "@vitavision/stage2d";
import type {
  BlobRef,
  DistortionMap,
  MapGrid,
  MapPlane,
  PlaneChannel,
  ProjectionSigmaMap,
  ResidualMap,
} from "../api/generated/types";

/** The bytes of a plane: magic, `u32` LE width, height, stride, channels,
 * 4 reserved bytes, then the `f32` LE values, plane-major. */
export function vam1(plane: ValuePlane): ArrayBuffer {
  const buffer = new ArrayBuffer(24 + plane.values.length * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x56, 0x41, 0x4d, 0x31]);
  const view = new DataView(buffer);
  view.setUint32(4, plane.width, true);
  view.setUint32(8, plane.height, true);
  view.setUint32(12, plane.stride, true);
  view.setUint32(16, plane.channels, true);
  plane.values.forEach((v, i) => view.setFloat32(24 + i * 4, v, true));
  return buffer;
}

/** A plane of `width × height` nodes whose channel `c` is `f(c, column, row)`. */
export function plane(
  width: number,
  height: number,
  channels: number,
  f: (channel: number, column: number, row: number) => number,
  stride = 10,
): ValuePlane {
  const values = new Float32Array(width * height * channels);
  for (let c = 0; c < channels; c++)
    for (let r = 0; r < height; r++)
      for (let col = 0; col < width; col++)
        values[(c * height + r) * width + col] = f(c, col, r);
  return { width, height, stride, channels, values };
}

/** A 4 × 3 grid of 10 px blocks over a 40 × 30 image. */
export const GRID: MapGrid = { columns: 4, rows: 3, stride: 10, image_size: [40, 30] };

const blob = (id: string): BlobRef => ({
  id,
  len: 100,
  media_type: "application/x-vam1",
});
const ch = (name: string, unit = "px"): PlaneChannel => ({ name, unit, domain: [0, 1] });

function mapPlane(id: string, channels: PlaneChannel[]): MapPlane {
  return { blob: blob(id), channels };
}

export const DISTORTION_CHANNELS = [ch("dx"), ch("dy"), ch("magnitude")];

/** Distortion replies: planes `lens`, `tilt` and `total`. Each plane is
 * `(dx, dy, |d|) = (column, row, hypot)` so a reading identifies its node. */
export function distortionReply(hasSensorLayer = false): DistortionMap {
  return {
    camera: 0,
    grid: GRID,
    has_sensor_layer: hasSensorLayer,
    valid_nodes: 12,
    distortion: mapPlane("lens", DISTORTION_CHANNELS),
    sensor: mapPlane("tilt", DISTORTION_CHANNELS),
    total: mapPlane("total", DISTORTION_CHANNELS),
  };
}

export function distortionPlane(): ValuePlane {
  return plane(4, 3, 3, (c, col, row) =>
    c === 0 ? col : c === 1 ? row : Math.hypot(col, row),
  );
}

export const RESIDUAL_CHANNELS = [
  ch("mean_dx"),
  ch("mean_dy"),
  ch("mean_magnitude"),
  ch("rms"),
  ch("std_error"),
  ch("significance", "chi"),
  ch("effective_samples", "samples"),
  ch("effective_views", "views"),
];

export function residualReply(over: Partial<ResidualMap> = {}): ResidualMap {
  return {
    bandwidth_px: 12.4,
    bandwidth_selection: { kind: "given" },
    camera: 0,
    chain_level: false,
    grid: GRID,
    options: {},
    outliers: [],
    plane: mapPlane("residual", RESIDUAL_CHANNELS),
    unprojected: 0,
    used: 40,
    views: 6,
    ...over,
  };
}

/** A residual plane: the mean grows along `column`; the right-hand column has
 * no value (thin support), significance passes the cut on the bottom row. */
export function residualPlane(): ValuePlane {
  return plane(4, 3, 8, (c, col, row) => {
    if (col === 3) return Number.NaN;
    const m = 0.1 * (col + 1);
    switch (c) {
      case 0:
        return m;
      case 1:
        return -0.05;
      case 2:
        return Math.hypot(m, 0.05);
      case 3:
        return 0.3;
      case 4:
        return 0.02;
      case 5:
        return row === 2 ? 5 : 1;
      case 6:
        return 20;
      default:
        return 6;
    }
  });
}

export function sigmaReply(): ProjectionSigmaMap {
  return {
    status: "available",
    camera: 0,
    grid: GRID,
    plane: mapPlane("sigma", [ch("sigma")]),
  };
}

export function sigmaPlane(): ValuePlane {
  return plane(4, 3, 1, (_c, col, row) => 0.1 + 0.05 * (col + row));
}
