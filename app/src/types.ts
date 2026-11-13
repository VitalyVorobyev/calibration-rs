// TypeScript projections of the subset of vision-calibration's export
// types that the v0 viewer reads. The Rust backend serializes the full
// PlanarIntrinsicsExport as untyped JSON; we narrow it here to keep
// component code typed without forcing a brittle 1:1 binding to every
// Rust field.

import type { StageView } from "@vitavision/stage2d";
import type {
  FeatureResidualHistogram,
  FrameKind,
  LaserFeatureResidual,
  LaserLineTrace,
  PixelRect,
  TargetFeatureResidual,
} from "./types/generated/diagnose-wire";

// The wire types the viewer reads with the same shape as their generated
// definition are that definition.
export type {
  FeatureResidualHistogram,
  FrameKind,
  LaserFeatureResidual,
  LaserLineTrace,
  PixelRect,
  TargetFeatureResidual,
} from "./types/generated/diagnose-wire";

// The rest narrow their generated definition: the viewer reads an export
// whose `diagnostics` group carries `per_feature_residuals` (with `target`)
// and whose optional fields are absent rather than `null`.

export interface FrameRef {
  view: number;
  camera: number;
  path: string;
  roi?: PixelRect;
  /** Absent on the wire means "target". */
  kind?: FrameKind;
}

export interface ImageManifest {
  root: string;
  frames: FrameRef[];
}

export interface PerFeatureResiduals {
  target: TargetFeatureResidual[];
  /** Per-laser-pixel records; absent/empty on non-laser problem types. */
  laser?: LaserFeatureResidual[];
  /** One projected laser line per (view, camera) with laser pixels. */
  laser_lines?: LaserLineTrace[];
  target_hist_per_cam?: FeatureResidualHistogram[] | null;
  laser_hist_per_cam?: FeatureResidualHistogram[] | null;
}

/** An export's `diagnostics` group: every feature's residual and, when
 * the images are known, where they are. */
export interface ExportDiagnostics {
  per_feature_residuals: PerFeatureResiduals;
  image_manifest?: ImageManifest;
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
