// TypeScript projections of the subset of vision-calibration's export
// types that the v0 viewer reads. The Rust backend serializes the full
// PlanarIntrinsicsExport as untyped JSON; we narrow it here to keep
// component code typed without forcing a brittle 1:1 binding to every
// Rust field.

import type { StageView } from "@vitavision/stage2d";

export interface PixelRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** What a frame depicts (ADR 0016 §5). Absent on the wire means
 * "target" — exports written before the discriminator omit it. */
export type FrameKind = "target" | "laser";

export interface FrameRef {
  pose: number;
  camera: number;
  path: string;
  roi?: PixelRect;
  kind?: FrameKind;
}

export interface ImageManifest {
  root: string;
  frames: FrameRef[];
}

export interface TargetFeatureResidual {
  pose: number;
  camera: number;
  feature: number;
  target_xyz_m: [number, number, number];
  observed_px: [number, number];
  projected_px?: [number, number] | null;
  error_px?: number | null;
}

/** Per-laser-pixel residual record (laser problem types only). */
export interface LaserFeatureResidual {
  pose: number;
  camera: number;
  feature: number;
  observed_px: [number, number];
  /** Point-to-plane distance in meters; null when the back-projected
   * ray does not intersect the target plane. */
  residual_m?: number | null;
  /** Distance to the projected laser line in pixels, as the laser solve
   * measures it (on the undistorted image plane). */
  residual_px?: number | null;
}

/** The calibrated laser line in one view's image: a polyline (distortion
 * bends it) spanning the view's laser pixels. */
export interface LaserLineTrace {
  pose: number;
  camera: number;
  points_px: [number, number][];
}

/** Aggregate residual histogram (fixed [<=1, <=2, <=5, <=10, >10] px buckets). */
export interface FeatureResidualHistogram {
  bucket_edges_px: [number, number, number, number];
  counts: [number, number, number, number, number];
  count: number;
  mean: number;
  max: number;
}

export interface PerFeatureResiduals {
  target: TargetFeatureResidual[];
  /** Per-laser-pixel records; absent/empty on non-laser problem types. */
  laser?: LaserFeatureResidual[];
  /** One projected laser line per (view, camera) with laser pixels. */
  laser_lines?: LaserLineTrace[];
  target_hist_per_camera?: FeatureResidualHistogram[] | null;
  laser_hist_per_camera?: FeatureResidualHistogram[] | null;
}

/** The subset of PlanarIntrinsicsExport the viewer reads. */
export interface PlanarExport {
  per_feature_residuals: PerFeatureResiduals;
  image_manifest?: ImageManifest;
  /** Optional on the wire — older export files may omit it. */
  mean_reproj_error?: number;
}

/** Tauri command response from `load_export`. */
export interface LoadExportResult {
  /** The raw export JSON. */
  export: PlanarExport;
  /** Absolute filesystem directory containing the export.json. */
  export_dir: string;
}

/** A `(pose, camera)` selector entry materialised from the manifest. */
export interface FrameKey {
  pose: number;
  camera: number;
  label: string;
  /** Absolute path to the PNG on disk. */
  abs_path: string;
  roi?: PixelRect | undefined;
}

/** Viewport transform of a frame viewer: the stage's `StageView`.
 * `scale` is CSS pixels per image pixel; `(tx, ty)` is the offset of the
 * image's top-left corner in CSS pixels. */
export type ViewportTransform = StageView;

/** Cursor readout emitted by FrameCanvas on mouse move. Coordinates
 * are in image-pixel space (the ROI-local frame the residuals live
 * in, pixel centres at integers); `intensity` is luminance ∈ [0, 255] when the underlying pixel
 * is decodable, or null at the canvas's edge / outside the image. */
export interface CursorReadout {
  x: number;
  y: number;
  intensity: number | null;
}
