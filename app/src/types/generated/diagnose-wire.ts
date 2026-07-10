/**
 * DO NOT EDIT — generated from the Rust wire types by
 * `bun run generate:types` (B-QUAL2). The source of truth is the
 * `#[derive(schemars::JsonSchema)]` types in the calibration workspace and
 * the diagnose Tauri commands; edit those and regenerate.
 */

/**
 * What a frame depicts, distinguishing target images from laser-on images.
 *
 * Laser problem types capture two images per `(pose, camera)` slot: one of
 * the calibration target and one with the laser line on. Both can appear in
 * the same [`ImageManifest`], discriminated by [`FrameRef::kind`]
 * (ADR 0021 §5).
 *
 * Serialized in `snake_case`; absent in JSON means [`FrameKind::Target`],
 * which keeps pre-existing exports byte-stable and forward-readable.
 */
export type FrameKind = "target" | "laser";
/**
 * Identifies which calibration `*Export` a JSON payload holds.
 *
 * One variant per pipeline problem type, serialized as the snake_case module
 * name in the `kind` field of every `*Export`. Consumers (the desktop app's
 * `detectExportKind`, downstream tooling) read this single tag rather than
 * sniffing field presence.
 */
export type ExportKind =
  | "planar_intrinsics"
  | "scheimpflug_intrinsics"
  | "single_cam_handeye"
  | "laserline_device"
  | "rig_extrinsics"
  | "rig_handeye"
  | "rig_laserline_device"
  | "rig_handeye_laserline";
/**
 * Serializable distortion model parameters.
 */
export type DistortionParams =
  | {
      type: "none";
    }
  | {
      /**
       * Iterations for undistortion.
       */
      iters: number;
      /**
       * Radial coefficient k1.
       */
      k1: number;
      /**
       * Radial coefficient k2.
       */
      k2: number;
      /**
       * Radial coefficient k3.
       */
      k3: number;
      /**
       * Tangential coefficient p1.
       */
      p1: number;
      /**
       * Tangential coefficient p2.
       */
      p2: number;
      type: "brown_conrady5";
    }
  | {
      /**
       * Iterations for undistortion (0 → 10).
       */
      iters: number;
      /**
       * Numerator radial coefficient k1.
       */
      k1: number;
      /**
       * Numerator radial coefficient k2.
       */
      k2: number;
      /**
       * Numerator radial coefficient k3.
       */
      k3: number;
      /**
       * Denominator radial coefficient k4.
       */
      k4: number;
      /**
       * Denominator radial coefficient k5.
       */
      k5: number;
      /**
       * Denominator radial coefficient k6.
       */
      k6: number;
      /**
       * Tangential coefficient p1.
       */
      p1: number;
      /**
       * Tangential coefficient p2.
       */
      p2: number;
      type: "rational";
    }
  | {
      /**
       * Iterations for undistortion (0 → 10).
       */
      iters: number;
      /**
       * Radial coefficient k1.
       */
      k1: number;
      /**
       * Radial coefficient k2.
       */
      k2: number;
      /**
       * Radial coefficient k3.
       */
      k3: number;
      /**
       * Tangential coefficient p1.
       */
      p1: number;
      /**
       * Tangential coefficient p2.
       */
      p2: number;
      /**
       * Thin-prism coefficient s1 (x correction, r²).
       */
      s1: number;
      /**
       * Thin-prism coefficient s2 (x correction, r⁴).
       */
      s2: number;
      /**
       * Thin-prism coefficient s3 (y correction, r²).
       */
      s3: number;
      /**
       * Thin-prism coefficient s4 (y correction, r⁴).
       */
      s4: number;
      type: "thin_prism";
    }
  | {
      /**
       * Division distortion coefficient.
       */
      lambda: number;
      type: "division";
    };
/**
 * Serializable intrinsics parameters.
 */
export type IntrinsicsParams = {
  /**
   * Principal point X coordinate in pixels.
   */
  cx: number;
  /**
   * Principal point Y coordinate in pixels.
   */
  cy: number;
  /**
   * Focal length in pixels along X.
   */
  fx: number;
  /**
   * Focal length in pixels along Y.
   */
  fy: number;
  /**
   * Skew term (typically 0).
   */
  skew: number;
  type: "fx_fy_cx_cy_skew";
};
/**
 * Serializable projection model parameters.
 */
export type ProjectionParams = {
  type: "pinhole";
};
/**
 * Serializable sensor model parameters.
 */
export type SensorParams =
  | {
      type: "identity";
    }
  | {
      /**
       * Row-major homography matrix mapping normalized to sensor coordinates.
       *
       * @minItems 3
       * @maxItems 3
       */
      h: [[number, number, number], [number, number, number], [number, number, number]];
      type: "homography";
    }
  | {
      /**
       * Tilt around X axis in radians (alias: tau_x).
       */
      tilt_x: number;
      /**
       * Tilt around Y axis in radians (alias: tau_y).
       */
      tilt_y: number;
      type: "scheimpflug";
    };
/**
 * Classic pinhole projection model.
 */
export type Pinhole = null;
/**
 * Identity sensor model.
 */
export type IdentitySensor = null;
/**
 * Hand-eye calibration mode.
 *
 * Specifies the transform chain used for hand-eye calibration.
 */
export type HandEyeMode = "EyeInHand" | "EyeToHand";

/**
 * Generated wire types for the diagnose app (B-QUAL2). Do not edit by hand — run `bun run generate:types`. The top-level wrapper only anchors the `definitions`; consumers import the individual interfaces.
 */
export interface DiagnoseWireTypes {
  disparity_result?: DisparityResult;
  epipolar_overlay?: EpipolarOverlay;
  laserline_device_export?: LaserlineDeviceExport;
  planar_intrinsics_export?: PlanarIntrinsicsExport;
  rig_extrinsics_export?: RigExtrinsicsExport;
  rig_handeye_export?: RigHandeyeExport;
  rig_handeye_laserline_export?: RigHandeyeLaserlineExport;
  rig_laserline_device_export?: RigLaserlineDeviceExport;
  scheimpflug_intrinsics_export?: ScheimpflugIntrinsicsExport;
  single_cam_handeye_export?: SingleCamHandeyeExport;
}
/**
 * Result returned to the Depth workspace. All images are `data:image/png`
 * base64 URLs the webview can drop straight into `<img>`.
 */
export interface DisparityResult {
  /**
   * Stereo baseline in calibration units (m).
   */
  baselineM: number;
  /**
   * Fraction of pixels with a valid disparity.
   */
  density: number;
  /**
   * Depth colormap (jet over metric depth `Z = f·B / d`; invalid black).
   */
  depthPng: string;
  /**
   * See [`DisparityResult::disp_min`].
   */
  dispMax: number;
  /**
   * Min / max recovered disparity (px).
   */
  dispMin: number;
  /**
   * Disparity colormap (jet; invalid pixels black).
   */
  disparityPng: string;
  /**
   * Matched (downscaled) image height.
   */
  height: number;
  /**
   * Disparity colormap blended over the rectified left image.
   */
  overlayPng: string;
  /**
   * Inlier pixel count of that planar fit.
   */
  planeInliers: number;
  /**
   * Robust planar-fit RMS over the dominant surface (px) — low ⇒ coherent.
   */
  planeRms: number;
  /**
   * Reprojected 3D point cloud (reference-camera frame, metres).
   */
  pointCloud: PointCloud;
  /**
   * Rectified left | right with shared epipolar rows (rectification sanity).
   */
  rectifiedPairPng: string;
  /**
   * Whether semi-global aggregation was used.
   */
  semiGlobal: boolean;
  /**
   * Matched (downscaled) image width.
   */
  width: number;
}
/**
 * A reprojected 3D point cloud for the frontend's WebGL renderer.
 */
export interface PointCloud {
  /**
   * Flat `[r, g, b, …]` per-point grayscale colours in `[0, 1]`.
   */
  colors: number[];
  /**
   * Number of points (`positions.len() / 3`).
   */
  count: number;
  /**
   * Flat `[x, y, z, …]` positions in the reference-camera frame (metres).
   */
  positions: number[];
}
/**
 * Result of a single epipolar overlay computation.
 */
export interface EpipolarOverlay {
  /**
   * Pane-B pixel coordinate of cam A's optical center. `None` when
   * the cameras are arranged so the epipole is at infinity (or
   * behind cam B).
   *
   * @minItems 2
   * @maxItems 2
   */
  epipole_b?: [number, number] | null;
  /**
   * Pane-B distorted pixel coordinates of each depth sample that
   * projected successfully. `[]` when the ray never crosses pane B's
   * image plane in front of cam B.
   */
  line_b: [number, number][];
  /**
   * Number of depth samples whose projection diverged (point behind
   * cam B, distortion fixed-point failure, …). For diagnostic UI.
   */
  samples_clipped: number;
}
/**
 * Export type for laserline device calibration.
 */
export interface LaserlineDeviceExport {
  /**
   * Pipeline output including optimized parameters and summary statistics.
   */
  estimate: LaserlineEstimate;
  /**
   * Optional image manifest (ADR 0014, viewer-side contract). Per
   * accepted view (pose = kept-view index, camera = 0): one frame
   * for the *target* image plus one of kind `laser` for the laser
   * image (ADR 0021 §5). `None` means "no images shipped"; the
   * calibration pipeline never reads this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::LaserlineDevice`].
   */
  kind: ExportKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (single element for single-camera workflows).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-feature reprojection + laser residuals (ADR 0012). Single-camera:
   * `target_hist_per_camera` and `laser_hist_per_camera` each carry
   * `Some(vec![one_entry])`.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Laserline statistics payload.
   */
  stats: LaserlineStats;
}
/**
 * Result of laserline bundle adjustment.
 */
export interface LaserlineEstimate {
  /**
   * Refined bundle parameters.
   */
  params: LaserlineParams;
  /**
   * Backend solve report.
   */
  report: SolveReport;
}
/**
 * Initial values for laserline bundle adjustment.
 */
export interface LaserlineParams {
  /**
   * Brown-Conrady distortion parameters.
   */
  distortion: BrownConrady5;
  /**
   * Camera intrinsics.
   */
  intrinsics: FxFyCxCySkew;
  /**
   * Laser plane parameters in camera frame.
   */
  plane: LaserPlane;
  /**
   * Per-view target poses (`camera_se3_target`).
   */
  poses: Iso3Schema[];
  /**
   * Scheimpflug sensor parameters (use defaults for identity sensor).
   */
  sensor: ScheimpflugParams;
}
/**
 * Brown-Conrady 5-parameter radial-tangential distortion model.
 */
export interface BrownConrady5 {
  /**
   * Iterations for undistortion.
   */
  iters: number;
  /**
   * Radial coefficient k1.
   */
  k1: number;
  /**
   * Radial coefficient k2.
   */
  k2: number;
  /**
   * Radial coefficient k3.
   */
  k3: number;
  /**
   * Tangential coefficient p1.
   */
  p1: number;
  /**
   * Tangential coefficient p2.
   */
  p2: number;
}
/**
 * Standard pinhole intrinsics with optional skew.
 */
export interface FxFyCxCySkew {
  /**
   * Principal point X coordinate in pixels.
   */
  cx: number;
  /**
   * Principal point Y coordinate in pixels.
   */
  cy: number;
  /**
   * Focal length in pixels along X.
   */
  fx: number;
  /**
   * Focal length in pixels along Y.
   */
  fy: number;
  /**
   * Skew term (typically 0).
   */
  skew: number;
}
/**
 * Laser plane in camera frame: unit normal + signed distance.
 *
 * The plane equation is: n̂ · p + d = 0
 * where p is a point in camera coordinates.
 */
export interface LaserPlane {
  /**
   * Signed distance from camera origin
   */
  distance: number;
  /**
   * Unit normal vector in camera frame. Serializes as `[nx, ny, nz]`.
   *
   * @minItems 3
   * @maxItems 3
   */
  normal: [number, number, number];
}
/**
 * JSON Schema proxy for [`Iso3`] (`nalgebra::Isometry3<f64>`).
 *
 * `nalgebra` does not implement [`schemars::JsonSchema`], so `*Export` and
 * parameter types that embed [`Iso3`] annotate the field with
 * `#[cfg_attr(feature = "schemars", schemars(with = "Iso3Schema"))]`
 * (or `Vec<Iso3Schema>` / `Option<Iso3Schema>`). This proxy mirrors the exact
 * serde wire format of `Isometry3`:
 * `{ "rotation": [qx, qy, qz, qw], "translation": [tx, ty, tz] }`.
 *
 * It exists only to describe that shape to `schemars`; it is never
 * constructed at runtime.
 */
export interface Iso3Schema {
  /**
   * Unit quaternion `[qx, qy, qz, qw]` (i, j, k, w order).
   *
   * @minItems 4
   * @maxItems 4
   */
  rotation: [number, number, number, number];
  /**
   * Translation `[tx, ty, tz]` in meters.
   *
   * @minItems 3
   * @maxItems 3
   */
  translation: [number, number, number];
}
/**
 * Scheimpflug tilt parameters (OpenCV-compatible).
 */
export interface ScheimpflugParams {
  /**
   * Tilt around X axis in radians (alias: tau_x).
   */
  tilt_x: number;
  /**
   * Tilt around Y axis in radians (alias: tau_y).
   */
  tilt_y: number;
}
/**
 * Summary of backend solve outcome.
 */
export interface SolveReport {
  /**
   * Final objective value reported by backend.
   */
  final_cost: number;
  /**
   * Number of outer solver iterations executed.
   */
  num_iters?: number;
}
/**
 * Image manifest carried alongside a calibration export.
 *
 * `root` is interpreted relative to the directory of the `export.json` that
 * embeds the manifest. Each [`FrameRef::path`] is interpreted relative to
 * `root`. Both layers of indirection make exports portable across machines
 * as long as the image directory is co-located with the export.
 */
export interface ImageManifest {
  /**
   * One entry per `(pose, camera)` slot the viewer can render.
   * Pose-major; aligns with the indexing of `PerFeatureResiduals.target`.
   */
  frames: FrameRef[];
  /**
   * Image root directory, relative to the export file's directory.
   */
  root: string;
}
/**
 * Reference to a single image (or sub-image) for one `(pose, camera)`.
 */
export interface FrameRef {
  /**
   * Camera index. `0` for single-camera problem types.
   */
  camera: number;
  /**
   * What the frame depicts. Absent in JSON means [`FrameKind::Target`],
   * so exports written before ADR 0021 §5 deserialize unchanged.
   */
  kind?: FrameKind;
  /**
   * Image path relative to the enclosing [`ImageManifest::root`].
   */
  path: string;
  /**
   * Pose / view index in the input dataset.
   */
  pose: number;
  /**
   * Sub-image rectangle in pixel coordinates. `None` means the entire
   * image is this `(pose, camera)`. Used for tiled multi-camera frames.
   *
   * **Coordinate convention:** see the module-level `# Coordinate
   * convention` block. ROI is a render-time crop hint only —
   * per-feature residuals are already in the ROI-local pixel frame
   * (i.e. the camera's own image frame, since intrinsics were
   * calibrated against the cropped tile), so a viewer must not
   * subtract `(x, y)` from `observed_px` / `projected_px` when
   * drawing.
   */
  roi?: PixelRect | null;
}
/**
 * Inclusive-exclusive pixel rectangle: `[x, x+w) × [y, y+h)`.
 */
export interface PixelRect {
  /**
   * Height in pixels.
   */
  h: number;
  /**
   * Width in pixels.
   */
  w: number;
  /**
   * Left edge in pixels.
   */
  x: number;
  /**
   * Top edge in pixels.
   */
  y: number;
}
/**
 * Container bundled into every `*Export` carrying the per-feature drill-down
 * data the diagnose UI consumes.
 *
 * Empty `Vec`s mean "this problem type does not produce observations of that
 * flavor" (e.g., `PlanarIntrinsicsExport.per_feature_residuals.laser` is
 * always empty). The histogram fields are `Option`: `None` means the problem
 * type chose not to produce a per-camera aggregate; `Some(vec)` length must
 * match `num_cameras`.
 */
export interface PerFeatureResiduals {
  /**
   * Per-laser-pixel residual records, pose-major.
   */
  laser?: LaserFeatureResidual[];
  /**
   * Per-camera laser pixel-distance histogram (length = `num_cameras`).
   */
  laser_hist_per_camera?: FeatureResidualHistogram[] | null;
  /**
   * Per-target-corner reprojection records, pose-major.
   */
  target?: TargetFeatureResidual[];
  /**
   * Per-camera target reprojection histogram (length = `num_cameras`).
   */
  target_hist_per_camera?: FeatureResidualHistogram[] | null;
}
/**
 * Reprojection record for a single laser pixel in a single view.
 */
export interface LaserFeatureResidual {
  /**
   * Camera index. `0` for single-camera problem types.
   */
  camera: number;
  /**
   * Pixel index within the view's laser pixel list.
   */
  feature: number;
  /**
   * Observed laser pixel coordinate.
   *
   * @minItems 2
   * @maxItems 2
   */
  observed_px: [number, number];
  /**
   * Pose / view index in the input dataset.
   */
  pose: number;
  /**
   * Two endpoints `[[x0, y0], [x1, y1]]` of the projected laser line in
   * image space. Useful for 2D overlays. `None` if the line cannot be
   * synthesized in this view.
   *
   * @minItems 2
   * @maxItems 2
   */
  projected_line_px?: [[number, number], [number, number]] | null;
  /**
   * Point-to-plane distance in meters between the back-projected ray and
   * the calibrated laser plane. `None` if the ray does not intersect.
   */
  residual_m?: number | null;
  /**
   * Pixel-domain residual: distance from `observed_px` to the projected
   * laser line in undistorted pixel space. `None` if unavailable.
   */
  residual_px?: number | null;
}
/**
 * Aggregate residual histogram for a (set of) reprojection error samples.
 *
 * Buckets are fixed at `[<=1, <=2, <=5, <=10, >10]` pixels — see
 * [`REPROJECTION_HISTOGRAM_EDGES_PX`].
 */
export interface FeatureResidualHistogram {
  /**
   * Bucket edges in pixels: `[1.0, 2.0, 5.0, 10.0]`.
   *
   * @minItems 4
   * @maxItems 4
   */
  bucket_edges_px: [number, number, number, number];
  /**
   * Total number of error samples included in the histogram.
   */
  count: number;
  /**
   * Counts in each bucket: `[<=1, <=2, <=5, <=10, >10]`.
   * `counts.iter().sum() == count`.
   *
   * @minItems 5
   * @maxItems 5
   */
  counts: [number, number, number, number, number];
  /**
   * Maximum error (pixels). `0.0` when `count == 0`.
   */
  max: number;
  /**
   * Mean error (pixels). `0.0` when `count == 0`.
   */
  mean: number;
}
/**
 * Reprojection record for a single target feature in a single view.
 */
export interface TargetFeatureResidual {
  /**
   * Camera index. `0` for single-camera problem types.
   */
  camera: number;
  /**
   * Euclidean pixel distance `|projected - observed|`.
   * `None` iff `projected_px` is `None`.
   */
  error_px?: number | null;
  /**
   * Feature index within the view's `points_3d`.
   */
  feature: number;
  /**
   * Observed pixel coordinate.
   *
   * @minItems 2
   * @maxItems 2
   */
  observed_px: [number, number];
  /**
   * Pose / view index in the input dataset.
   */
  pose: number;
  /**
   * Projected pixel using the calibrated camera + recovered pose.
   * `None` when projection diverges (point behind camera, distortion failure).
   *
   * @minItems 2
   * @maxItems 2
   */
  projected_px?: [number, number] | null;
  /**
   * 3D point in target / world frame (meters).
   *
   * @minItems 3
   * @maxItems 3
   */
  target_xyz_m: [number, number, number];
}
/**
 * Summary statistics for a laserline calibration result.
 */
export interface LaserlineStats {
  /**
   * Mean laser residual (units depend on residual type).
   */
  mean_laser_error: number;
  /**
   * Mean reprojection error for calibration points (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-view mean laser residual (same units as `mean_laser_error`).
   */
  per_view_laser_errors: number[];
  /**
   * Per-view mean reprojection error (pixels).
   */
  per_view_reproj_errors: number[];
}
/**
 * Export format for planar intrinsics calibration.
 */
export interface PlanarIntrinsicsExport {
  /**
   * Optional image manifest (ADR 0014, viewer-side contract). When
   * populated, downstream viewers (the diagnose UI) can locate the source
   * image for each `(pose, camera)` slot. `None` means "no images
   * shipped"; the calibration pipeline never reads this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::PlanarIntrinsics`].
   */
  kind: ExportKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Calibrated parameters.
   */
  params: PlanarIntrinsicsParams;
  /**
   * Per-camera reprojection errors (single element for single-camera workflows).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-feature reprojection residuals (ADR 0012). For planar intrinsics
   * only `target` is populated; `laser` is empty. `target_hist_per_camera`
   * is `Some(vec![one_entry])` since this problem type is single-camera.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Solver report.
   */
  report: SolveReport;
}
/**
 * Optimization result for planar intrinsics.
 */
export interface PlanarIntrinsicsParams {
  /**
   * Refined camera model (model-agnostic serializable parameters).
   */
  camera: CameraParams;
  /**
   * Refined target-to-camera poses.
   */
  camera_se3_target: Iso3Schema[];
}
/**
 * Serializable camera parameters for building a runtime model.
 */
export interface CameraParams {
  /**
   * Distortion model parameters.
   */
  distortion: DistortionParams;
  /**
   * Intrinsics model parameters.
   */
  intrinsics: IntrinsicsParams;
  /**
   * Projection model parameters.
   */
  projection: ProjectionParams;
  /**
   * Sensor model parameters.
   */
  sensor: SensorParams;
}
/**
 * Export format for rig extrinsics calibration.
 *
 * Common to pinhole and Scheimpflug rigs. `sensors` is `None` for pinhole
 * rigs and `Some(_)` for Scheimpflug rigs, matching the configured
 * [`SensorMode`].
 */
export interface RigExtrinsicsExport {
  /**
   * Per-camera extrinsics: `cam_se3_rig` (T_C_R).
   * Transform from rig frame to camera frame.
   */
  cam_se3_rig: Iso3Schema[];
  /**
   * Per-camera calibrated intrinsics + distortion (pinhole core).
   */
  cameras: Camera[];
  /**
   * Optional image manifest (ADR 0014, viewer-side contract). When
   * populated, downstream viewers (the diagnose UI) can locate the source
   * image for each `(pose, camera)` slot. Tiled multi-camera frames
   * (e.g. 6× 720×540 horizontal strips on the puzzle 130×130 rig) point
   * multiple `FrameRef`s at the same `path` with disjoint ROIs. `None`
   * means "no images shipped"; the calibration pipeline never reads
   * this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::RigExtrinsics`].
   */
  kind: ExportKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (pixels).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-feature reprojection residuals (ADR 0012). For rig extrinsics
   * `target` is populated and `laser` is empty. `target_hist_per_camera`
   * is `Some(vec)` with one entry per camera.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Per-view rig poses: `rig_se3_target` (T_R_T).
   */
  rig_se3_target: Iso3Schema[];
  /**
   * Per-camera Scheimpflug sensor parameters. `None` for pinhole rigs;
   * `Some(_)` for Scheimpflug rigs (one entry per camera).
   */
  sensors?: ScheimpflugParams[] | null;
}
/**
 * A composable camera model: projection -> distortion -> sensor -> intrinsics.
 */
export interface Camera {
  _phantom: null;
  /**
   * Distortion model (e.g. Brown-Conrady).
   */
  dist: BrownConrady5;
  /**
   * Intrinsics model (K).
   */
  k: FxFyCxCySkew;
  /**
   * Projection model (e.g. pinhole).
   */
  proj: Pinhole;
  /**
   * Sensor model (e.g. identity or tilted homography).
   */
  sensor: IdentitySensor;
}
/**
 * Export format for rig hand-eye calibration.
 *
 * Common to pinhole and Scheimpflug rigs. `sensors` is `None` for pinhole
 * rigs and `Some(_)` for Scheimpflug rigs, matching the configured
 * [`SensorMode`].
 */
export interface RigHandeyeExport {
  /**
   * Eye-in-hand: base-to-target transform `base_se3_target` (T_B_T).
   *
   * `None` for EyeToHand mode.
   */
  base_se3_target?: Iso3Schema | null;
  /**
   * Per-camera extrinsics: `cam_se3_rig` (T_C_R).
   * Transform from rig frame to camera frame.
   */
  cam_se3_rig: Iso3Schema[];
  /**
   * Per-camera calibrated intrinsics + distortion (pinhole core).
   */
  cameras: Camera[];
  /**
   * Eye-in-hand: gripper-to-rig transform `gripper_se3_rig` (T_G_R).
   *
   * `None` for EyeToHand mode.
   */
  gripper_se3_rig?: Iso3Schema | null;
  /**
   * Eye-to-hand: gripper-to-target transform `gripper_se3_target` (T_G_T).
   *
   * `None` for EyeInHand mode.
   */
  gripper_se3_target?: Iso3Schema | null;
  /**
   * Hand-eye mode used to interpret mode-dependent transforms.
   */
  handeye_mode: HandEyeMode;
  /**
   * Optional image manifest (ADR 0014, viewer-side contract). When
   * populated, downstream viewers (the diagnose UI) can locate the source
   * image for each `(pose, camera)` slot. Tiled multi-camera frames
   * (e.g. 6× 720×540 horizontal strips on the puzzle 130×130 rig) point
   * multiple `FrameRef`s at the same `path` with disjoint ROIs. `None`
   * means "no images shipped"; the calibration pipeline never reads
   * this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::RigHandeye`].
   */
  kind: ExportKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (pixels).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-feature reprojection residuals (ADR 0012). Per-view
   * `rig_se3_target` is derived from the handeye chain
   * (see [`handeye_observer_se3_target`](vision_calibration_optim::handeye_observer_se3_target)),
   * then composed with `cam_se3_rig` for projection.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Eye-to-hand: rig-to-base transform `rig_se3_base` (T_R_B).
   *
   * `None` for EyeInHand mode.
   */
  rig_se3_base?: Iso3Schema | null;
  /**
   * Per-view rig poses: `rig_se3_target` (T_R_T), derived from the
   * hand-eye chain (`handeye_observer_se3_target`) so downstream
   * viewers (3D scene, epipolar overlay) can read board poses
   * without re-implementing the chain. One entry per input view.
   * `#[serde(default)]` keeps older exports forward-compatible at
   * load time; they decode with an empty Vec.
   */
  rig_se3_target?: Iso3Schema[];
  /**
   * Per-view robot pose corrections (if refinement enabled).
   * Each element is [rx, ry, rz, tx, ty, tz] in se(3).
   */
  robot_deltas?: [number, number, number, number, number, number][] | null;
  /**
   * Per-camera Scheimpflug sensor parameters. `None` for pinhole rigs;
   * `Some(_)` for Scheimpflug rigs (one entry per camera).
   */
  sensors?: ScheimpflugParams[] | null;
}
/**
 * Export format for joint rig hand-eye laserline calibration.
 */
export interface RigHandeyeLaserlineExport {
  /**
   * Eye-in-hand target reference pose `base_se3_target`.
   */
  base_se3_target?: Iso3Schema | null;
  /**
   * Per-camera extrinsics `cam_se3_rig` (T_C_R).
   */
  cam_se3_rig: Iso3Schema[];
  /**
   * Per-camera pinhole camera parameters.
   */
  cameras: Camera[];
  /**
   * Eye-in-hand hand-eye transform `gripper_se3_rig`.
   */
  gripper_se3_rig?: Iso3Schema | null;
  /**
   * Eye-to-hand target reference pose `gripper_se3_target`.
   */
  gripper_se3_target?: Iso3Schema | null;
  /**
   * Hand-eye mode used to interpret mode-dependent transforms.
   */
  handeye_mode: HandEyeMode;
  /**
   * Optional image manifest populated by app/dataset runners.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::RigHandeyeLaserline`].
   */
  kind: ExportKind;
  /**
   * Per-camera laser planes in camera frame.
   */
  laser_planes_cam: LaserPlane[];
  /**
   * Per-camera laser planes in rig frame.
   */
  laser_planes_rig: LaserPlane[];
  /**
   * Mean target reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera target reprojection errors (pixels).
   */
  per_cam_reproj_errors: number[];
  /**
   * Joint per-camera stats.
   */
  per_camera_stats: RigHandeyeLaserlinePerCamStats[];
  /**
   * Per-feature target and laser residuals.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Eye-to-hand hand-eye transform `rig_se3_base`.
   */
  rig_se3_base?: Iso3Schema | null;
  /**
   * Per-view target poses `rig_se3_target` (T_R_T).
   */
  rig_se3_target: Iso3Schema[];
  /**
   * Optional optimized robot pose deltas.
   */
  robot_deltas?: [number, number, number, number, number, number][] | null;
  /**
   * Per-camera Scheimpflug sensor parameters.
   */
  sensors: ScheimpflugParams[];
}
/**
 * Per-camera statistics for [`RigHandeyeLaserlineEstimate`].
 */
export interface RigHandeyeLaserlinePerCamStats {
  /**
   * Number of laser pixel residuals evaluated for this camera.
   */
  laser_count: number;
  /**
   * Laser point-to-plane histogram with buckets
   * `[<=0.1mm, <=1mm, <=10mm, <=100mm, >100mm]`.
   *
   * @minItems 5
   * @maxItems 5
   */
  laser_histogram_m: [number, number, number, number, number];
  /**
   * Maximum absolute laser point-to-plane distance (meters).
   */
  max_laser_err_m: number;
  /**
   * Maximum absolute laser line-distance in undistorted pixel space (pixels).
   */
  max_laser_err_px: number;
  /**
   * Maximum target-corner reprojection error (pixels).
   */
  max_reproj_error_px: number;
  /**
   * Mean laser point-to-plane distance (meters).
   */
  mean_laser_err_m: number;
  /**
   * Mean laser line-distance in undistorted pixel space (pixels).
   */
  mean_laser_err_px: number;
  /**
   * Mean target-corner reprojection error (pixels).
   */
  mean_reproj_error_px: number;
  /**
   * Number of target corner residuals evaluated for this camera.
   */
  reproj_count: number;
  /**
   * Reprojection error histogram with buckets `[<=1, <=2, <=5, <=10, >10]` pixels.
   *
   * @minItems 5
   * @maxItems 5
   */
  reproj_histogram_px: [number, number, number, number, number];
}
/**
 * Export format for rig laserline calibration.
 */
export interface RigLaserlineDeviceExport {
  /**
   * Frozen upstream per-camera extrinsics `T_C_R`.
   */
  cam_se3_rig?: Iso3Schema[];
  /**
   * Frozen upstream cameras (pinhole part), echoed so the export is
   * self-contained for downstream viewers (3D rig scene, epipolar) —
   * same field names as `RigHandeyeExport`. Empty on pre-B-laser
   * exports (`serde(default)`).
   */
  cameras?: Camera[];
  /**
   * Optional image manifest (ADR 0014, viewer-side contract). When
   * populated, downstream viewers (the diagnose / 3D / epipolar UIs)
   * can locate the source image for each `(pose, camera)` slot.
   * Tiled multi-camera frames (e.g. 6× 720×540 horizontal strips on
   * the puzzle 130×130 rig) point multiple `FrameRef`s at the same
   * `path` with disjoint ROIs. `None` means "no images shipped"; the
   * calibration pipeline never reads this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::RigLaserlineDevice`].
   */
  kind: ExportKind;
  /**
   * Per-camera laser planes in their own camera frames.
   */
  laser_planes_cam: LaserPlane[];
  /**
   * Per-camera laser planes in rig frame.
   */
  laser_planes_rig: LaserPlane[];
  /**
   * Mean target reprojection error (pixels) against the frozen
   * upstream. Diagnostic echo — this problem type does not optimize
   * reprojection.
   */
  mean_reproj_error?: number;
  /**
   * Per-camera stats (reprojection + laser residuals).
   */
  per_camera_stats: LaserlineStats[];
  /**
   * Per-feature reprojection + laser residuals (ADR 0012). Multi-camera
   * rig: `target` covers per-corner reprojection (when present) and
   * `laser` covers per-pixel laser distances. Both per-camera histograms
   * are populated.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Frozen upstream per-view rig poses `T_R_T`.
   */
  rig_se3_target?: Iso3Schema[];
  /**
   * Frozen upstream Scheimpflug sensor parameters (zero tilt for
   * pinhole rigs), aligned with `cameras`.
   */
  sensors?: ScheimpflugParams[] | null;
}
/**
 * Export format for Scheimpflug intrinsics calibration.
 */
export interface ScheimpflugIntrinsicsExport {
  /**
   * Optional pointer to the source images behind this export. When
   * populated, downstream viewers (the diagnose UI) can locate the
   * source image for each view. `None` means "no images shipped";
   * the calibration pipeline never reads this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::ScheimpflugIntrinsics`].
   */
  kind: ExportKind;
  /**
   * Mean per-point reprojection error in pixels.
   */
  mean_reproj_error: number;
  /**
   * Estimated parameters.
   */
  params: ScheimpflugIntrinsicsParams;
  /**
   * Per-camera reprojection errors (single element for single-camera workflows).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-feature reprojection residuals (ADR 0012). Single-camera, target
   * only — `laser` is empty; `target_hist_per_camera` is
   * `Some(vec![one_entry])`.
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Backend solve report.
   */
  report: SolveReport;
}
/**
 * Output parameter pack for Scheimpflug intrinsics calibration.
 */
export interface ScheimpflugIntrinsicsParams {
  /**
   * Estimated camera model including intrinsics, distortion, and sensor parameters.
   */
  camera: CameraParams;
  /**
   * Estimated pose `camera_se3_target` for each view.
   */
  camera_se3_target: Iso3Schema[];
}
/**
 * Export format for single-camera hand-eye calibration.
 */
export interface SingleCamHandeyeExport {
  /**
   * Eye-in-hand: base_se3_target (T_B_T).
   *
   * `None` for EyeToHand mode.
   */
  base_se3_target?: Iso3Schema | null;
  /**
   * Calibrated camera (intrinsics + distortion).
   */
  camera: Camera;
  /**
   * Eye-to-hand: camera_se3_base (T_C_B).
   *
   * `None` for EyeInHand mode.
   */
  camera_se3_base?: Iso3Schema | null;
  /**
   * Eye-in-hand: gripper_se3_camera (T_G_C).
   *
   * `None` for EyeToHand mode.
   */
  gripper_se3_camera?: Iso3Schema | null;
  /**
   * Eye-to-hand: gripper_se3_target (T_G_T).
   *
   * `None` for EyeInHand mode.
   */
  gripper_se3_target?: Iso3Schema | null;
  /**
   * Hand-eye mode used to interpret the transforms.
   */
  handeye_mode: HandEyeMode;
  /**
   * Optional pointer to the source images behind this export. When
   * populated, downstream viewers (the diagnose UI) can locate the
   * source image for each view. `None` means "no images shipped";
   * the calibration pipeline never reads this field.
   */
  image_manifest?: ImageManifest | null;
  /**
   * Export-type discriminator (R7) — always [`ExportKind::SingleCamHandeye`].
   */
  kind: ExportKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (pixels). Single element for single-camera.
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-feature reprojection residuals (ADR 0012). Single-camera, target
   * only. Per-view `cam_se3_target` is derived from the handeye chain
   * (see [`handeye_observer_se3_target`](vision_calibration_optim::handeye_observer_se3_target)).
   */
  per_feature_residuals?: PerFeatureResiduals;
  /**
   * Per-view robot pose deltas (se(3) tangent: [rx, ry, rz, tx, ty, tz]).
   * Only present if robot refinement was enabled.
   */
  robot_deltas?: [number, number, number, number, number, number][] | null;
}
