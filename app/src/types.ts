// TypeScript projections of the subset of vision-calibration's export
// types that the v0 viewer reads. The Rust backend serializes the full
// PlanarIntrinsicsExport as untyped JSON; we narrow it here to keep
// component code typed without forcing a brittle 1:1 binding to every
// Rust field.

export interface PixelRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FrameRef {
  pose: number;
  camera: number;
  path: string;
  roi?: PixelRect;
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

export interface PerFeatureResiduals {
  target: TargetFeatureResidual[];
  // laser, target_hist_per_camera, laser_hist_per_camera exist on the
  // wire but the v0 viewer does not consume them.
}

/** The subset of PlanarIntrinsicsExport the viewer reads. */
export interface PlanarExport {
  per_feature_residuals: PerFeatureResiduals;
  image_manifest?: ImageManifest;
  mean_reproj_error: number;
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
  roi?: PixelRect;
}

/** Viewport transform applied to the canvas before drawing the image
 * + residuals. `scale` is the zoom factor; `(tx, ty)` is the
 * translation in canvas pixels. Identity is `{ scale: 1, tx: 0, ty: 0 }`.
 * Wheel zoom anchors at the cursor; reset on frame change. */
export interface ViewportTransform {
  scale: number;
  tx: number;
  ty: number;
}

export const IDENTITY_TRANSFORM: ViewportTransform = { scale: 1, tx: 0, ty: 0 };

/** Cursor readout emitted by FrameCanvas on mouse move. Coordinates
 * are in image-pixel space (the ROI-local frame the residuals live
 * in); `intensity` is luminance ∈ [0, 255] when the underlying pixel
 * is decodable, or null at the canvas's edge / outside the image. */
export interface CursorReadout {
  x: number;
  y: number;
  intensity: number | null;
}
