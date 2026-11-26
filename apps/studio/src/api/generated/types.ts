/**
 * DO NOT EDIT — the back end's wire types, generated from the back end's command table by
 * `bun run generate:api`. The source of truth is studio-core's
 * `api::table` and the `#[derive(schemars::JsonSchema)]` types; edit those
 * and regenerate.
 */

/**
 * An absolute path on the back end's machine, inside one of its root folders. The back end refuses any other.
 */
export type UserPath = string;
export type CameraParameterValuesReply = (number | null)[];
/**
 * Serializable distortion model parameters.
 */
export type DistortionParams =
  | {
      type: "none";
    }
  | {
      /**
       * Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
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
       * Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
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
       * Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
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
    }
  | {
      /**
       * Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
       */
      iters: number;
      /**
       * Coefficient of `r²`.
       */
      k1: number;
      /**
       * Coefficient of `r⁴`.
       */
      k2: number;
      /**
       * Coefficient of `r⁶`.
       */
      k3: number;
      /**
       * Coefficient of `r⁸`.
       */
      k4: number;
      type: "kannala_brandt4";
    };
/**
 * Serializable projection model parameters.
 */
export type ProjectionParams =
  | {
      type: "pinhole";
    }
  | {
      type: "equidistant";
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
 * How image paths for a camera are listed.
 */
export type ImagePattern =
  | {
      kind: "glob";
      /**
       * The glob pattern.
       */
      pattern: string;
    }
  | {
      kind: "list";
      /**
       * Paths relative to the manifest directory unless absolute.
       */
      paths: string[];
    };
/**
 * The pixel-noise level behind a corner covariance: estimated from the
 * image, or stated by the user. The manifest's copy of the detector's
 * image-noise option (see [`DetectorSpec`]).
 *
 * In JSON / TOML: `"estimate"` or `{ stated = 2.0 }`.
 */
export type ImageNoiseSpec =
  | "estimate"
  | {
      stated: number;
    };
/**
 * The corner detector a chessboard-like detector runs. The manifest's copy
 * of the detector's corner strategy (see [`DetectorSpec`]).
 */
export type CornerStrategySpec = "chess" | "radon";
/**
 * The calibration problem a dataset is for, and the kind of an export.
 *
 * One value per problem type, serialized as its snake_case name
 * (`"planar_intrinsics"`, …). A manifest names it in its `topology` field,
 * which selects the problem a run dispatches to; every export carries it in
 * its `kind` field, so a consumer narrows on one value instead of probing
 * which fields are present. The tag is required on deserialize.
 *
 * A problem kind describes the data (how many cameras, robot poses, laser
 * images), not the camera model: a tilted (Scheimpflug) single camera is a
 * `planar_intrinsics` dataset whose calibration config names the sensor.
 */
export type ProblemKind =
  | "planar_intrinsics"
  | "single_cam_handeye"
  | "laserline_device"
  | "rig_extrinsics"
  | "rig_handeye"
  | "rig_handeye_laserline"
  | "rig_laserline_device";
/**
 * Laser scan direction.
 */
export type LaserScanAxis = "cols" | "rows";
/**
 * Rotation parameterisation in the file.
 */
export type RotationFormat =
  | "quat_xyzw"
  | "quat_wxyz"
  | "euler_xyz_deg"
  | "euler_xyz_rad"
  | "euler_zyx_deg"
  | "euler_zyx_rad"
  | "axis_angle_rad"
  | "matrix4x4_row_major";
/**
 * Which transform a pose row encodes.
 */
export type TransformConvention =
  "t_base_tcp" | "t_tcp_base" | "t_world_tcp" | "t_tcp_world";
/**
 * Translation units in the file.
 */
export type TranslationUnits = "m" | "mm";
/**
 * How per-camera image lists align across views and (optionally)
 * align with robot poses.
 */
export type PosePairing =
  | {
      kind: "by_index";
    }
  | {
      /**
       * Name of the capture group whose value identifies the view.
       */
      group: string;
      kind: "shared_filename_token";
      /**
       * Regex with at least one named capture group.
       */
      regex: string;
    };
/**
 * Robot-pose file format.
 */
export type RobotPoseFormat = "csv" | "json" | "jsonl" | "rowmajor4x4" | "counted4x4";
/**
 * Calibration-target metadata. Tagged on `kind`.
 */
export type TargetSpec =
  | {
      /**
       * Number of interior corners across the board (the image's
       * columns axis when the board is seen as declared).
       */
      cols: number;
      kind: "chessboard";
      /**
       * Measure the blur at each corner: the covariance `Σ` (px², image
       * axes, including the pixel aperture) of the Gaussian kernel that
       * best blurs the printed pattern around the corner into the image,
       * with the covariance of that estimate, carried into each
       * observation. A corner the estimate declines has none. It costs a
       * few milliseconds per corner, single-threaded (a 476-corner board
       * adds about 1–2 s per image), hence the default `false`. Corner
       * positions are the same with or without it.
       */
      measure_blur?: boolean;
      /**
       * Number of interior corners down the board (the image's rows
       * axis when the board is seen as declared).
       */
      rows: number;
      /**
       * Edge length of one square in metres.
       */
      square_size_m: number;
    }
  | {
      /**
       * Number of squares along the cols axis.
       */
      cols: number;
      /**
       * ArUco dictionary identifier (e.g. `"DICT_4X4_50"`). The
       * detector validates this against the supported set.
       */
      dictionary: string;
      kind: "charuco";
      /**
       * Edge length of an embedded marker in metres.
       */
      marker_size_m: number;
      /**
       * Number of squares along the rows axis (full grid, not
       * interior corners).
       */
      rows: number;
      /**
       * Edge length of one square in metres.
       */
      square_size_m: number;
    }
  | {
      /**
       * Edge length of one cell in metres.
       */
      cell_size_m: number;
      kind: "puzzleboard";
      /**
       * Board dimensions as `puzzle_<rows>x<cols>` (e.g.
       * `"puzzle_12x12"`; the `puzzle_` prefix is optional).
       */
      layout: string;
      /**
       * Measure the blur at each corner: the covariance `Σ` (px², image
       * axes, including the pixel aperture) of the Gaussian kernel that
       * best blurs the printed pattern around the corner into the image,
       * with the covariance of that estimate, carried into each
       * observation. A corner the estimate declines has none. It costs a
       * few milliseconds per corner, single-threaded (a 476-corner board
       * adds about 1–2 s per image), hence the default `false`. Corner
       * positions are the same with or without it.
       */
      measure_blur?: boolean;
    }
  | {
      /**
       * How each marker's centre is estimated. Omitted keeps the
       * ellipse-based centre; `"photometric"` fits the blurred marker
       * pattern instead, for blurred or low-contrast images.
       */
      center_method?: RinggridCenterSpec;
      /**
       * How the markers are told apart. Omitted keeps `"coded"`.
       */
      coding?: RinggridCodingSpec;
      kind: "ringgrid";
      /**
       * The marker lattice. Omitted keeps `"hex"`.
       */
      lattice?: RinggridLatticeSpec;
      /**
       * Number of columns: on a hexagonal lattice, of the longest
       * (even-indexed) rows, the shorter rows being derived by the
       * layout; on a rectangular lattice, of every row.
       */
      long_row_cols: number;
      /**
       * Inner ring radius in metres.
       */
      marker_inner_radius_m: number;
      /**
       * Outer ring radius in metres.
       */
      marker_outer_radius_m: number;
      /**
       * Width of each ring band in metres. Coded markers need it, greater
       * than zero; plain markers ignore it, so it may be left out for
       * them.
       */
      marker_ring_width_m?: number;
      /**
       * Floor on the detector's per-marker confidence, in `[0, 1]`.
       * Omitted keeps the detector's floor of 0.03. The confidence is a
       * product of factors in `[0, 1]`, one of them the edge sharpness
       * `1 / (1 + (w / 3 px)²)`, so a floor `c` keeps only markers whose
       * outer edge width `w` is at most `3 · √(1/c − 1)` px (0.17: 6.6 px).
       */
      min_confidence?: number | null;
      /**
       * Center-to-center spacing between adjacent markers in metres. On a
       * rectangular lattice it is both the row and the column step.
       */
      pitch_m: number;
      /**
       * Number of marker rows.
       */
      rows: number;
    };
/**
 * The ring-grid centre estimator: the manifest's own copy of the detector's
 * centre method, which has the same JSON spelling.
 */
export type RinggridCenterSpec = "ellipse" | "photometric";
/**
 * How ring-grid markers are told apart: the manifest's own copy of the
 * detector's coding, which has the same JSON spelling.
 *
 * In JSON: `"coded"`, or `{"plain": {"origin_dots": true}}`; in TOML,
 * `coding = { plain = { origin_dots = true } }`.
 */
export type RinggridCodingSpec =
  | "coded"
  | {
      plain: {
        /**
         * The board carries the triad of filled origin dots that the
         * target generator places near its centre. Defaults to `true`.
         *
         * A view is labelled in the board frame whenever all three dots are
         * in it among detected markers, however small a part of the board
         * it shows; a view without them, or with only one or two, yields no
         * features. `false`, a plain board without dots, is not supported:
         * its views have no board coordinates, and a manifest that sets it
         * is rejected before detection.
         */
        origin_dots?: boolean;
      };
    };
/**
 * The ring-grid marker lattice: the manifest's own copy of the detector's
 * lattice, which has the same JSON spelling.
 */
export type RinggridLatticeSpec = "hex" | "rect";
export type EntryKind = "dir" | "file";
export type DisparityEvent = null;
/**
 * The kinds of [`StudioError`].
 */
export type ErrorKind =
  | "forbidden_path"
  | "unknown_command"
  | "invalid_args"
  | "busy"
  | "too_large"
  | "job_not_found"
  | "blob_expired"
  | "unauthorized"
  | "roots_fixed"
  | "internal"
  | "not_a_project"
  | "project_exists"
  | "unsupported_version"
  | "invalid_run_name"
  | "run_exists"
  | "io"
  | "parse"
  | "manifest"
  | "config"
  | "invalid"
  | "cancelled";
/**
 * What happened.
 */
export type EventBody = JobEvent & {
  kind: "job";
};
/**
 * Where a job is in its life.
 */
export type JobState =
  | {
      kind: "queued";
      position: number;
    }
  | {
      kind: "running";
    }
  | {
      kind: "done";
      output: unknown;
    }
  | {
      error: StudioError;
      kind: "failed";
    }
  | {
      kind: "cancelled";
    };
/**
 * What a frame depicts, distinguishing target images from laser-on images.
 *
 * Laser problem types capture two images per `(view, camera)` slot: one of
 * the calibration target and one with the laser line on. Both can appear in
 * the same [`ImageManifest`], discriminated by [`FrameRef::kind`].
 *
 * Serialized in `snake_case`; absent in JSON means [`FrameKind::Target`],
 * which keeps exports without laser frames byte-stable and forward-readable.
 */
export type FrameKind = "target" | "laser";
export type FetchSampleEvent = null;
export type FetchSampleOutput = string;
export type ForgetRecentReply = string[];
/**
 * The solution of a rig hand-eye calibration: the hand-eye of its mode and
 * the pose of the static target in the frame that mode fixes it in.
 *
 * Each mode holds exactly the two poses it estimates, so an export with the
 * wrong pose for its mode cannot be written or read. In JSON it is
 * internally tagged by `mode`:
 *
 * ```json
 * {"mode": "eye_in_hand", "gripper_se3_rig": {…}, "base_se3_target": {…}}
 * ```
 */
export type HandeyeSolution =
  | {
      /**
       * The target's pose in the robot's base frame (`base_se3_target`).
       */
      base_se3_target: Iso3Schema;
      /**
       * The rig's pose on the gripper (`gripper_se3_rig`, rig frame →
       * gripper frame).
       */
      gripper_se3_rig: Iso3Schema;
      mode: "eye_in_hand";
    }
  | {
      /**
       * The target's pose on the gripper (`gripper_se3_target`).
       */
      gripper_se3_target: Iso3Schema;
      mode: "eye_to_hand";
      /**
       * The base's pose in the rig (`rig_se3_base`, base frame → rig
       * frame).
       */
      rig_se3_base: Iso3Schema;
    };
/**
 * The uncertainty of a result's estimated parameters.
 */
export type Uncertainty =
  | (ParameterCovariance & {
      status: "available";
    })
  | {
      /**
       * Why the covariance is not available.
       */
      reason: UncertaintyUnavailable;
      status: "unavailable";
    };
/**
 * The estimated noise level of one kind of residual.
 */
export type ResidualNoise = {
  /**
   * The camera, for a result that estimates each camera's noise apart
   * (the rig laserline, which solves each camera's plane on its own);
   * `None` for one level shared by every camera.
   */
  camera?: number | null;
  /**
   * Number of scalar residuals of this kind.
   */
  count: number;
  /**
   * The kind of residual.
   */
  kind: ResidualKind;
} & ResidualNoise1;
/**
 * A kind of residual, which sets its unit.
 */
export type ResidualKind =
  | "reprojection"
  | "laser_plane"
  | "laser_line"
  | {
      custom: string;
    };
export type ResidualNoise1 =
  | {
      sigma: number;
    }
  | {
      sigma0: number;
    };
/**
 * Why a result has no covariance.
 */
export type UncertaintyUnavailable =
  | {
      kind: "not_converged";
    }
  | {
      kind: "rank_deficient";
      /**
       * Labels of the parameters involved.
       */
      unobservable: string[];
    }
  | {
      kind: "at_bound";
      /**
       * Labels of the parameters on a bound.
       */
      labels: string[];
    }
  | {
      kind: "no_redundancy";
    }
  | {
      kind: "not_computed";
    }
  | {
      kind: "calibration_without_covariance";
    };
/**
 * What a listing shows.
 */
export type Show = "all" | "dirs";
export type ListJobsReply = JobInfo[];
/**
 * What a long-running call reports to its observer.
 *
 * In JSON the variant is the `kind` field:
 * `{"kind": "iteration", "iter": 3, "max_iters": 100, "cost": 12.5}`.
 */
export type Progress =
  | {
      /**
       * Target images done so far, `1..=total`.
       */
      done: number;
      kind: "detection";
      /**
       * Target images to detect, over every camera.
       */
      total: number;
    }
  | {
      /**
       * Laser images done so far, `1..=total`.
       */
      done: number;
      kind: "laser_extraction";
      /**
       * Laser images to extract, over every camera.
       */
      total: number;
    }
  | {
      kind: "step";
      /**
       * The step's name.
       */
      name: string;
    }
  | {
      /**
       * The objective after the iteration, `½ Σᵢ ρᵢ(‖rᵢ‖²)` (the
       * convention of a solve report's `final_cost`).
       */
      cost: number;
      /**
       * The iteration, `1..=max_iters`.
       */
      iter: number;
      kind: "iteration";
      /**
       * The solve's iteration budget.
       */
      max_iters: number;
    };
export type RecentProjectsReply = string[];
export type ResolveConfigReply = unknown;
/**
 * Parts of a [`RobotPoseSigma`]: its rotation, its translation or both.
 */
export type RobotSigmaParts = "rotation" | "translation" | "both";
/**
 * Pipeline stage a run is currently executing. Within a stage the
 * library's own reports ([`Progress`]: each image detected, each
 * calibration step, each solver iteration) ride along in
 * [`RunProgress::report`].
 */
export type RunStage = "detect" | "solve" | "export";
/**
 * Tagged response shape so the React layer can match on `kind` rather
 * than parsing free-form error strings.
 */
export type RunResponse =
  | (RunSuccess & {
      kind: "ok";
    })
  | {
      kind: "cancelled";
    }
  | {
      /**
       * Dotted field path the runtime needs help with.
       */
      field: string;
      kind: "ask_user";
      /**
       * Human-friendly prompt copy for the modal.
       */
      prompt: string;
      /**
       * Optional preset suggestions (e.g. `["t_base_tcp", "t_tcp_base"]`).
       */
      suggestions: string[];
    }
  | {
      kind: "validation_failed";
      message: string;
    }
  | {
      /**
       * Stable kind tag for the frontend.
       */
      category: string;
      kind: "failed";
      /**
       * Human-readable detail.
       */
      message: string;
    };
export type SaveProjectManifestReply = null;
export type SetActiveExportReply = null;
/**
 * The solution of a single-camera hand-eye calibration: the hand-eye of its
 * mode and the pose of the static target in the frame that mode fixes it in.
 *
 * Each mode holds exactly the two poses it estimates, so an export with the
 * wrong pose for its mode cannot be written or read. It is the rig
 * solution ([`HandeyeSolution`](crate::common::HandeyeSolution)) of a rig
 * that is the camera, with the camera named where the rig is. In JSON it is
 * internally tagged by `mode`:
 *
 * ```json
 * {"mode": "eye_in_hand", "gripper_se3_cam": {…}, "base_se3_target": {…}}
 * ```
 */
export type SingleCamHandeyeSolution =
  | {
      /**
       * The target's pose in the robot's base frame (`base_se3_target`).
       */
      base_se3_target: Iso3Schema;
      /**
       * The camera's pose on the gripper (`gripper_se3_cam`, camera
       * frame → gripper frame).
       */
      gripper_se3_cam: Iso3Schema;
      mode: "eye_in_hand";
    }
  | {
      /**
       * The base's pose in the camera (`cam_se3_base`, base frame →
       * camera frame).
       */
      cam_se3_base: Iso3Schema;
      /**
       * The target's pose on the gripper (`gripper_se3_target`).
       */
      gripper_se3_target: Iso3Schema;
      mode: "eye_to_hand";
    };
export type UndistortPointsReply = ([number, number] | null)[];

export interface StudioApiTypes {
  AddRootArgs?: AddRootArgs;
  BlobQuery?: BlobQuery;
  BlobRef?: BlobRef;
  CameraParameterValuesArgs?: CameraParameterValuesArgs;
  CameraParameterValuesReply?: CameraParameterValuesReply;
  CameraParams?: CameraParams;
  CameraSource?: CameraSource;
  CancelJobArgs?: CancelJobArgs;
  ChessCornersDetectorSpec?: ChessCornersDetectorSpec;
  CombinationTerm?: CombinationTerm;
  ConfigFormArgs?: ConfigFormArgs;
  ConfigPatch?: ConfigPatch;
  CornerCovarianceSpec?: CornerCovarianceSpec;
  CornerStrategySpec?: CornerStrategySpec;
  CreateProjectArgs?: CreateProjectArgs;
  DatasetFormArgs?: DatasetFormArgs;
  DatasetSpec?: DatasetSpec;
  DetectorSpec?: DetectorSpec;
  DirEntry?: DirEntry;
  DirListing?: DirListing;
  DisparityArgs?: DisparityArgs;
  DisparityEvent?: DisparityEvent;
  DisparityMetrics?: DisparityMetrics;
  DisparityParams?: DisparityParams;
  DisparityResult?: DisparityResult;
  DistortionParams?: DistortionParams;
  EntryKind?: EntryKind;
  EpipolarOverlay?: EpipolarOverlay;
  EpipolarOverlayArgs?: EpipolarOverlayArgs;
  EpipolarOverlayUndistortedArgs?: EpipolarOverlayUndistortedArgs;
  ErrorKind?: ErrorKind;
  EventBody?: EventBody;
  ExportDiagnostics?: ExportDiagnostics;
  FeatureResidualHistogram?: FeatureResidualHistogram;
  FetchSampleArgs?: FetchSampleArgs;
  FetchSampleEvent?: FetchSampleEvent;
  FetchSampleOutput?: FetchSampleOutput;
  ForgetRecentArgs?: ForgetRecentArgs;
  ForgetRecentReply?: ForgetRecentReply;
  FormSchema?: FormSchema;
  FrameKind?: FrameKind;
  FrameRef?: FrameRef;
  FxFyCxCy?: FxFyCxCy;
  HandeyeSolution?: HandeyeSolution;
  ImageManifest?: ImageManifest;
  ImageNoiseSpec?: ImageNoiseSpec;
  ImagePattern?: ImagePattern;
  ImageQuery?: ImageQuery;
  Iso3Schema?: Iso3Schema;
  JobEvent?: JobEvent;
  JobInfo?: JobInfo;
  JobState?: JobState;
  JobStatusArgs?: JobStatusArgs;
  LaserExtractionSpec?: LaserExtractionSpec;
  LaserFeatureResidual?: LaserFeatureResidual;
  LaserLineTrace?: LaserLineTrace;
  LaserPlane?: LaserPlane;
  LaserScanAxis?: LaserScanAxis;
  LaserlineDeviceExport?: LaserlineDeviceExport;
  LaserlineStats?: LaserlineStats;
  ListDirArgs?: ListDirArgs;
  ListJobsArgs?: ListJobsArgs;
  ListJobsReply?: ListJobsReply;
  LoadExportArgs?: LoadExportArgs;
  LoadExportResult?: LoadExportResult;
  OpenProjectArgs?: OpenProjectArgs;
  ParameterCovariance?: ParameterCovariance;
  PerFeatureResiduals?: PerFeatureResiduals;
  PixelRect?: PixelRect;
  PlanarIntrinsicsExport?: PlanarIntrinsicsExport;
  PoseColumnMap?: PoseColumnMap;
  PoseConvention?: PoseConvention;
  PosePairing?: PosePairing;
  ProblemKind?: ProblemKind;
  Progress?: Progress;
  ProjectInfo?: ProjectInfo;
  ProjectionParams?: ProjectionParams;
  ProjectionUncertainty?: ProjectionUncertainty;
  ProjectionUncertaintyArgs?: ProjectionUncertaintyArgs;
  ReadManifestArgs?: ReadManifestArgs;
  RecentProjectsArgs?: RecentProjectsArgs;
  RecentProjectsReply?: RecentProjectsReply;
  RemoveRootArgs?: RemoveRootArgs;
  ResidualKind?: ResidualKind;
  ResidualNoise?: ResidualNoise;
  ResolveConfigArgs?: ResolveConfigArgs;
  ResolveConfigReply?: ResolveConfigReply;
  RigExtrinsicsExport?: RigExtrinsicsExport;
  RigHandeyeExport?: RigHandeyeExport;
  RigHandeyeLaserlineExport?: RigHandeyeLaserlineExport;
  RigHandeyeLaserlinePerCamStats?: RigHandeyeLaserlinePerCamStats;
  RigLaserlineDeviceExport?: RigLaserlineDeviceExport;
  RinggridCenterSpec?: RinggridCenterSpec;
  RinggridCodingSpec?: RinggridCodingSpec;
  RinggridLatticeSpec?: RinggridLatticeSpec;
  RobotPoseFormat?: RobotPoseFormat;
  RobotPoseSigma?: RobotPoseSigma;
  RobotPoseSource?: RobotPoseSource;
  RobotRefinement?: RobotRefinement;
  RobotSigmaParts?: RobotSigmaParts;
  RootFolder?: RootFolder;
  RotationFormat?: RotationFormat;
  RunCalibrationArgs?: RunCalibrationArgs;
  RunDiagnostics?: RunDiagnostics;
  RunProgress?: RunProgress;
  RunRecord?: RunRecord;
  RunResponse?: RunResponse;
  RunStage?: RunStage;
  RunSuccess?: RunSuccess;
  RunSummary?: RunSummary;
  SaveProjectManifestArgs?: SaveProjectManifestArgs;
  SaveProjectManifestReply?: SaveProjectManifestReply;
  SaveProjectRunArgs?: SaveProjectRunArgs;
  SecondOrderError?: SecondOrderError;
  SensorParams?: SensorParams;
  ServerEvent?: ServerEvent;
  ServerInfo?: ServerInfo;
  ServerInfoArgs?: ServerInfoArgs;
  SetActiveExportArgs?: SetActiveExportArgs;
  SetActiveExportReply?: SetActiveExportReply;
  Show?: Show;
  SingleCamHandeyeExport?: SingleCamHandeyeExport;
  SingleCamHandeyeSolution?: SingleCamHandeyeSolution;
  SniffFolderArgs?: SniffFolderArgs;
  StudioError?: StudioError;
  TargetFeatureResidual?: TargetFeatureResidual;
  TargetSpec?: TargetSpec;
  TransformConvention?: TransformConvention;
  TranslationUnits?: TranslationUnits;
  Uncertainty?: Uncertainty;
  UncertaintyUnavailable?: UncertaintyUnavailable;
  UndistortPointsArgs?: UndistortPointsArgs;
  UndistortPointsReply?: UndistortPointsReply;
  UndistortedImageQuery?: UndistortedImageQuery;
  UserPath?: UserPath;
  ViewCovariance?: ViewCovariance;
}
/**
 * Add the folder `path` to the root folders (the desktop: the folder the
 * user picked in the native dialog, the one act that grants a folder).
 * `roots_fixed` on a server.
 */
export interface AddRootArgs {
  path: UserPath;
}
/**
 * The blob `id` a command made (a value plane, raw floats, a PNG).
 * `blob_expired` once the store has evicted it: run the command again.
 */
export interface BlobQuery {
  id: string;
}
/**
 * A blob a command made: fetch it from the `blob` resource with its id.
 */
export interface BlobRef {
  id: string;
  /**
   * Its length in bytes.
   */
  len: number;
  /**
   * `application/x-vam1` (a value plane), `application/x-f32` (raw
   * little-endian floats), `image/png`, …
   */
  media_type: string;
}
/**
 * The value of each labelled parameter of the active export that is a
 * camera parameter, aligned with `labels`; `null` for poses and planes.
 */
export interface CameraParameterValuesArgs {
  labels: string[];
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
   * Pinhole intrinsics (`fx, fy, cx, cy` in pixels).
   */
  intrinsics: FxFyCxCy;
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
 * Standard pinhole intrinsics: focal lengths and principal point in pixels.
 *
 * The pixel grid is assumed orthogonal, so `K` carries no skew term.
 */
export interface FxFyCxCy {
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
}
/**
 * Per-camera image source.
 */
export interface CameraSource {
  /**
   * Stable camera identifier (e.g. `"cam0"`). Used as a key in
   * pose pairings and downstream exports.
   */
  id: string;
  /**
   * Glob pattern (e.g. `"cam0/*.png"`) or explicit list of image
   * paths, relative to the manifest's directory unless absolute.
   */
  images: ImagePattern;
  /**
   * Laser-frame image source for the laser topologies.
   * One laser image per view, captured by the same sensor with the
   * line projector on; `roi_xywh` applies to laser frames too.
   * Laser images pair with target views through `pose_pairing`.
   */
  laser_images?: ImagePattern | null;
  /**
   * Optional rectangular ROI in source-image pixels — `[x, y, w, h]`.
   * When set, only the cropped region is fed to the detector and
   * stored in the export's image manifest.
   *
   * @minItems 4
   * @maxItems 4
   */
  roi_xywh?: [number, number, number, number] | null;
}
/**
 * Cancel the job `id`: a queued job at once, a running one at its next
 * image or solver iteration. A finished job is answered as it is.
 */
export interface CancelJobArgs {
  id: string;
}
/**
 * ChESS corner extractor overrides. The manifest's copy of the detector's
 * ChESS options (see [`DetectorSpec`]).
 */
export interface ChessCornersDetectorSpec {
  /**
   * Attach the detector's per-corner position covariance to every
   * observation. Absent (the default) = off.
   *
   * The covariance is the image noise propagated through the corner
   * localizer, plus the localizer's systematic-error model. When present,
   * a calibration weights each corner by it (whitened residuals) and its
   * report gives the unit-weight scale factor σ̂₀ instead of an RMS in
   * pixels.
   *
   * Off by default, and not recommended for calibration. On real
   * chessboard and ChArUco images, weighting by it did not improve the
   * accuracy on held-out views and made it slightly worse on some
   * cameras; its stated uncertainty was no closer to the real scatter.
   *
   * Supported by the `chessboard` and `charuco` targets only: the
   * PuzzleBoard detector runs on an upscaled image, for which no
   * covariance is reported, and ring grids have none. A manifest that
   * sets it for another target is an error.
   */
  covariance?: CornerCovarianceSpec | null;
  /**
   * Minimum corner strength for a detected corner to enter the grid
   * builder. `None` keeps the detector default (`33.0`). The floor is on
   * the strategy's response scale; under Radon the default never binds.
   *
   * The default drops weakly-firing corners, which is the right trade
   * whenever corners are plentiful. Set `0.0` to disable it on small or
   * soft image tiles, where the floor can remove a third of the board and
   * the resulting sparse view conditions worse than the weak corners did.
   */
  min_corner_strength?: number | null;
  /**
   * Corner detector. `None` keeps ChESS.
   */
  strategy?: CornerStrategySpec | null;
  /**
   * Acceptance threshold on the corner response. `None` keeps the
   * strategy's default.
   *
   * Under ChESS it is an absolute floor on the raw response: a corner is
   * kept when its response exceeds it. Under Radon it is a fraction in
   * `(0, 1]` of the frame's maximum response.
   *
   * A sibling `threshold_mode` key is rejected by `deny_unknown_fields`:
   * the corner detector has no threshold mode.
   */
  threshold_value?: number | null;
}
/**
 * Per-corner covariance options; see [`ChessCornersDetectorSpec::covariance`].
 * The manifest's copy of the detector's covariance options (see
 * [`DetectorSpec`]).
 */
export interface CornerCovarianceSpec {
  /**
   * The image noise that the propagated covariance is expressed in.
   * Defaults to `estimate`: the noise is estimated from each image, which
   * is unreliable on JPEG or demosaiced images; give `stated` for those.
   */
  image_noise?: ImageNoiseSpec & string;
}
/**
 * One term of a combination of parameters: a parameter and its weight, in
 * units of the parameter's σ.
 */
export interface CombinationTerm {
  /**
   * The parameter's label.
   */
  label: string;
  /**
   * The weight of `parameter / σ(parameter)` in the combination, in
   * `[-1, 1]` with the largest term at +1. Terms of equal weight and
   * opposite sign are a difference the combination measures.
   */
  weight: number;
}
/**
 * The config form of the workflow `topology`: the schema of its config
 * (`PlanarIntrinsicsConfig`, `RigHandeyeConfig`, …) and the library's
 * default configuration.
 */
export interface ConfigFormArgs {
  topology: ProblemKind;
}
/**
 * Changes to a workflow's default configuration: what the patch names
 * replaces the default, what it leaves out keeps it.
 *
 * A patch is keyed like the problem's own config, a table of its fields
 * (`{"solver": {"max_iters": 20}}`, or in TOML `solver.max_iters = 20`),
 * not the tagged `{"topology": …, "config": …}` form of
 * [`CalibrationConfig`]. It applies by one rule: two tables merge key by
 * key, and any other value (a number, a string, a list, `null`) replaces
 * what it lands on. [`CalibrationConfig::patched`] applies a patch to the
 * defaults of a topology and checks the result; a field the workflow does
 * not have is an error there, so a misspelt name never passes silently.
 *
 * TOML has no `null`, so a patch read from TOML cannot set an optional
 * field back to none; JSON can.
 *
 * The patch serializes as its table, with the keys of every table sorted.
 *
 * ```
 * use vision_calibration_dataset::ProblemKind;
 * use vision_calibration_pipeline::dataset_runner::{CalibrationConfig, ConfigPatch};
 *
 * let patch = ConfigPatch::from_toml_str("solver.max_iters = 20")?;
 * let config = CalibrationConfig::patched(ProblemKind::PlanarIntrinsics, &patch)?;
 * let CalibrationConfig::PlanarIntrinsics(planar) = &config else {
 *     unreachable!("the patch applies to the topology's own config");
 * };
 * assert_eq!(planar.solver.max_iters, 20);
 * // The smallest patch that makes the configuration from the defaults.
 * assert_eq!(ConfigPatch::from_config(&config), patch);
 * # Ok::<(), vision_calibration_pipeline::Error>(())
 * ```
 */
export interface ConfigPatch {
  [k: string]: unknown;
}
/**
 * Make the folder `dir` a project for the dataset in `dataset_dir`,
 * described by `manifest`. Fails with `project_exists` when `dir` already
 * holds one, and with `forbidden_path` when the manifest names a path
 * outside the roots.
 */
export interface CreateProjectArgs {
  dataset_dir: UserPath;
  dir: UserPath;
  /**
   * The dataset manifest (`DatasetSpec` JSON), stored as `dataset.json`.
   */
  manifest: unknown;
}
/**
 * The dataset manifest form: the schema of `DatasetSpec` and the manifest
 * a new setup starts from (one chessboard camera, planar intrinsics).
 */
export interface DatasetFormArgs {}
/**
 * Top-level dataset manifest. Authoritative description of where
 * the user's calibration data lives and how to interpret it.
 */
export interface DatasetSpec {
  /**
   * Field paths the manifest generator could not determine.
   * Validated to be empty before the runner accepts the manifest.
   * Populated as e.g. `["pose_convention.transform"]`.
   */
  _unresolved?: string[];
  /**
   * Per-camera image sources. Length must match the topology's
   * expected camera count (a single-camera topology requires exactly
   * one entry, a rig topology at least two, `rig_handeye_laserline` at
   * least one).
   */
  cameras: CameraSource[];
  /**
   * Free-form human-readable description.
   */
  description?: string | null;
  /**
   * Detector-stage overrides shared by target families. Defaults
   * preserve the detector's built-in behavior.
   */
  detector?: DetectorSpec | null;
  /**
   * Laser-line extraction parameters for the laser topologies.
   * `None` means defaults — extraction tuning has safe
   * defaults and is not a fail-fast ambiguity. Rejected on
   * non-laser topologies.
   */
  laser?: LaserExtractionSpec | null;
  /**
   * Frame conventions for robot poses. **No defaults** — every
   * component is required when `robot_poses` is set.
   */
  pose_convention?: PoseConvention | null;
  /**
   * How per-camera images are paired across views (and optionally
   * with robot poses). `None` is permitted only where pairing is
   * irrelevant: a single camera without robot poses.
   */
  pose_pairing?: PosePairing | null;
  /**
   * Robot-pose source for hand-eye topologies. `None` for
   * pure-intrinsics calibrations.
   */
  robot_poses?: RobotPoseSource | null;
  /**
   * Calibration-target specification. Tagged enum on `kind`.
   */
  target: TargetSpec;
  /**
   * Calibration topology — selects which problem type the runner
   * will dispatch to.
   */
  topology: ProblemKind;
  /**
   * Path to a frozen upstream `RigHandeyeExport` JSON, relative to
   * the manifest directory unless absolute. Required for
   * `ProblemKind::RigLaserlineDevice`, rejected elsewhere. Joint hand-eye
   * laserline calibration computes its own hand-eye warm start and does not
   * use this field.
   */
  upstream_calibration?: string | null;
  /**
   * Manifest format version; must be [`DATASET_SPEC_VERSION`] (`1`).
   * Required: a manifest without it, or with another value, is rejected.
   */
  version: number;
}
/**
 * Target-detector overrides. These options affect feature detection
 * only; they are hashed into the detection-cache key by the runner.
 *
 * The manifest declares its own detector types (this one,
 * [`ChessCornersDetectorSpec`], [`CornerStrategySpec`],
 * [`CornerCovarianceSpec`], [`ImageNoiseSpec`]) instead of reusing the
 * detector crate's: the manifest crate and the detector crate do not depend
 * on each other, so a manifest is read and validated without the detectors.
 * The runner maps each to its detector counterpart, which has the same JSON
 * spelling.
 */
export interface DetectorSpec {
  /**
   * ChESS corner extractor options shared by chessboard-like
   * detectors (plain chessboard and ChArUco).
   */
  chess_corners?: ChessCornersDetectorSpec | null;
  /**
   * Minimum detected features for a view (or a camera's slot in a rig
   * view) to be kept, in every topology. `None` keeps the runner default
   * (4, the homography minimum). Raise it for detectors whose sparse
   * detections are unreliable — e.g. ring-grid decodes with < 8 markers
   * are noisy enough to poison a solve. A plain chessboard needs no
   * floor to stay consistent: where views share one target frame the
   * runner already keeps only views that span the whole board.
   */
  min_features_per_view?: number | null;
}
/**
 * Laser-line extraction parameters. Consumed by the
 * runner's injected `LaserPixelExtractor`; hashed (canonical JSON)
 * into the detection-cache key so edits re-extract.
 */
export interface LaserExtractionSpec {
  /**
   * Minimum extracted points for a usable laser view. Views below
   * the bar are dropped (single camera) or become gap slots (rig).
   * Must be at least 2 (a line needs two points); 0/1 would let an
   * empty extraction pass as usable and leave the plane
   * unconstrained.
   */
  min_points?: number;
  /**
   * Negative-edge response threshold.
   */
  neg_thresh?: number;
  /**
   * Positive-edge response threshold.
   */
  pos_thresh?: number;
  /**
   * Scan direction. `cols` walks image columns (one point per
   * column; for laser lines running mostly left-to-right), `rows`
   * walks image rows.
   */
  scan_axis?: LaserScanAxis & string;
  /**
   * Gaussian derivative sigma in pixels for the edge detector.
   */
  sigma?: number;
}
/**
 * Frame conventions for the robot pose stream.
 *
 * All three components are required (no defaults). Each is a closed
 * enum so the runtime supports a finite, validated set of conventions.
 */
export interface PoseConvention {
  /**
   * How rotation is parameterised in the file.
   */
  rotation_format: RotationFormat;
  /**
   * Which transform the pose row encodes.
   */
  transform: TransformConvention;
  /**
   * Translation units in the file.
   */
  translation_units: TranslationUnits;
}
/**
 * Where and how robot poses are stored on disk.
 */
export interface RobotPoseSource {
  /**
   * Column / field mapping from the user's file to the canonical
   * fields the converter consumes. Required for the tabular formats
   * (`csv` / `json` / `jsonl`) unless `matrix_field` is set; must be
   * absent for `rowmajor4x4` and `counted4x4`, whose matrix layout is
   * fixed.
   */
  columns?: PoseColumnMap | null;
  /**
   * File-format selector.
   */
  format: RobotPoseFormat;
  /**
   * For `json` / `jsonl` files whose rows store the whole pose as a
   * single nested 4×4 (or flat-16) array field, names that field
   * (e.g. `"tcp2base"`). Mutually exclusive with `columns`; requires
   * `pose_convention.rotation_format = matrix4x4_row_major`.
   * Translation comes from the matrix's fourth column, scaled by
   * `translation_units`. No `pose_id` ⇒ `by_index` pairing only.
   */
  matrix_field?: string | null;
  /**
   * Path to the pose file, relative to the manifest directory
   * unless absolute.
   */
  path: string;
}
/**
 * Mapping from user-defined column / field names to the canonical
 * pose fields. All seven of `tx, ty, tz, qx, qy, qz, qw` are required
 * when `rotation_format` is a quaternion; the runner rejects partial
 * mappings as a fail-fast event.
 */
export interface PoseColumnMap {
  /**
   * Optional view-id column (string or integer). When absent, the
   * pose row index is used as the implicit pose id.
   */
  pose_id?: string | null;
  /**
   * Rotation columns. Length and meaning depend on the
   * `pose_convention.rotation_format`.
   */
  rotation: string[];
  /**
   * Translation x column.
   */
  tx: string;
  /**
   * Translation y column.
   */
  ty: string;
  /**
   * Translation z column.
   */
  tz: string;
}
/**
 * One entry of a listing.
 */
export interface DirEntry {
  /**
   * A folder holding a calibration project.
   */
  is_project: boolean;
  kind: EntryKind;
  name: string;
  /**
   * Its canonical path, for the next `list_dir` or a command.
   */
  path: string;
  /**
   * A file's size in bytes.
   */
  size?: number | null;
}
/**
 * A folder's entries.
 */
export interface DirListing {
  /**
   * Folders first, then files, each by name.
   */
  entries: DirEntry[];
  /**
   * Its parent while that is inside a root; `null` at a root (go back to
   * the roots) and for the roots.
   */
  parent?: string | null;
  /**
   * The folder listed (canonical); `null` for the roots.
   */
  path?: string | null;
  /**
   * Whether entries were left out past the limit.
   */
  truncated: boolean;
}
/**
 * A disparity map of the stereo pair `path_a` (camera `cam_a`) and
 * `path_b` (camera `cam_b`) at `pose`: rectified from the active export and
 * dense-matched. The output names its fields as blobs: a value plane of
 * disparity, depth and the rectified left image, the point cloud, and the
 * rectified pair. The same request again answers from the blob store while
 * it still holds them.
 */
export interface DisparityArgs {
  cam_a: number;
  cam_b: number;
  params: DisparityParams;
  path_a: UserPath;
  path_b: UserPath;
  pose: number;
}
/**
 * Frontend-supplied matching controls.
 */
export interface DisparityParams {
  /**
   * Matching-window side length (odd). 11 is a good default for real imagery.
   */
  block_size: number;
  /**
   * Integer image downscale before matching (keeps the search tractable).
   */
  downscale: number;
  /**
   * Use semi-global aggregation (fills low-texture regions) vs block matching.
   */
  semi_global: boolean;
}
/**
 * The quality of a dense match.
 */
export interface DisparityMetrics {
  /**
   * Stereo baseline in calibration units (m).
   */
  baseline_m: number;
  /**
   * Fraction of pixels with a valid disparity.
   */
  density: number;
  /**
   * See [`DisparityMetrics::disp_min`].
   */
  disp_max: number;
  /**
   * Min / max recovered disparity (px).
   */
  disp_min: number;
  /**
   * Inlier pixel count of that planar fit.
   */
  plane_inliers: number;
  /**
   * Robust planar-fit RMS over the dominant surface (px) — low ⇒ coherent;
   * `None` when fewer than 3 finite disparities are available or the fit is degenerate.
   */
  plane_rms?: number | null;
  /**
   * Whether semi-global aggregation was used.
   */
  semi_global: boolean;
}
/**
 * A dense match, its fields as blobs (`blob` resource).
 */
export interface DisparityResult {
  /**
   * Matched (downscaled) image height.
   */
  height: number;
  metrics: DisparityMetrics;
  /**
   * A `VAM1` value plane of the matched size with three channels:
   * disparity (px) and metric depth (m), both `NaN` where there is no
   * match, and the rectified left image's luminance (0–255).
   */
  planes: BlobRef;
  /**
   * Its colours: `3 · point_count` little-endian `f32` in `[0, 1]`.
   */
  point_colors: BlobRef;
  point_count: number;
  /**
   * The point cloud's positions: `3 · point_count` little-endian `f32`
   * (x, y, z in the reference camera's frame, metres).
   */
  point_positions: BlobRef;
  /**
   * Rectified left | right side by side with shared horizontal epipolar
   * lines (a PNG).
   */
  rectified_pair: BlobRef;
  /**
   * Matched (downscaled) image width.
   */
  width: number;
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
 * The epipolar polyline and epipole in camera `cam_b` (distorted pixels)
 * of the pixel `point_px` of camera `cam_a`, through each camera's full
 * model.
 */
export interface EpipolarOverlayArgs {
  cam_a: number;
  cam_b: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  point_px: [number, number];
}
/**
 * A straight epipolar segment in camera `cam_b`'s undistorted frame,
 * clipped to its `image_width_b` × `image_height_b` image, of the pixel
 * `point_px` of camera `cam_a`'s undistorted frame.
 */
export interface EpipolarOverlayUndistortedArgs {
  cam_a: number;
  cam_b: number;
  image_height_b: number;
  image_width_b: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  point_px: [number, number];
}
/**
 * A job's state, and the progress report that came with it.
 */
export interface JobEvent {
  /**
   * The job command that started it.
   */
  command: string;
  /**
   * The job's progress report (its command's event type), when this
   * event carries one.
   */
  event?: unknown;
  /**
   * The job's id.
   */
  id: string;
  state: JobState;
}
/**
 * Why a command, job or resource failed.
 */
export interface StudioError {
  /**
   * What the UI branches on.
   */
  kind: ErrorKind;
  /**
   * What it shows.
   */
  message: string;
  /**
   * The path the failure is about, when there is one.
   */
  path?: string | null;
}
/**
 * Evidence about how an export's result was reached: what a viewer or a
 * debugging session reads, not what a consumer needs to use the
 * calibration.
 *
 * The result fields of an export (camera parameters, poses, uncertainty,
 * mean reprojection error) are the stable contract; this group may gain
 * detail between releases.
 */
export interface ExportDiagnostics {
  /**
   * Where each view's source images are, for a viewer that overlays the
   * residuals on them. The calibration never reads it; `None` means no
   * images were recorded (the dataset runner and the desktop app fill it).
   */
  image_manifest?: ImageManifest | null;
  /**
   * The reprojection (and, for a laser workflow, laser) residual of every
   * feature, with per-camera histograms.
   */
  per_feature_residuals?: PerFeatureResiduals;
}
/**
 * Image manifest carried alongside a calibration export.
 *
 * `root` is interpreted relative to the directory of the `export.json` that
 * embeds the manifest. Each [`FrameRef::path`] is interpreted relative to
 * `root`. Both layers of indirection make exports portable across machines
 * as long as the image directory is co-located with the export.
 *
 * The dataset runner fills the manifest before the export has a file: its
 * `root` is `.`, the dataset folder its manifest's paths resolve against,
 * and each frame's path is relative to that folder (absolute for an image
 * outside it). A writer that puts the export in another folder restates
 * the root with [`rebased`](Self::rebased).
 */
export interface ImageManifest {
  /**
   * One entry per `(view, camera)` slot the viewer can render.
   * View-major; aligns with the indexing of `PerFeatureResiduals.target`.
   */
  frames: FrameRef[];
  /**
   * Image root directory, relative to the export file's directory.
   */
  root: string;
}
/**
 * Reference to a single image (or sub-image) for one `(view, camera)`.
 */
export interface FrameRef {
  /**
   * Camera index. `0` for single-camera problem types.
   */
  camera: number;
  /**
   * What the frame depicts. Absent in JSON means [`FrameKind::Target`],
   * so target-only exports deserialize unchanged.
   */
  kind?: FrameKind;
  /**
   * Image path relative to the enclosing [`ImageManifest::root`].
   */
  path: string;
  /**
   * Sub-image rectangle in pixel coordinates. `None` means the entire
   * image is this `(view, camera)`. Used for tiled multi-camera frames.
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
  /**
   * View index in the input dataset.
   */
  view: number;
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
   * Per-laser-pixel residual records, view-major.
   */
  laser?: LaserFeatureResidual[];
  /**
   * Per-camera laser pixel-distance histogram (length = `num_cameras`).
   */
  laser_hist_per_cam?: FeatureResidualHistogram[] | null;
  /**
   * The projected laser line of each (view, camera) with laser pixels,
   * view-major.
   */
  laser_lines?: LaserLineTrace[];
  /**
   * Per-target-corner reprojection records, view-major.
   */
  target?: TargetFeatureResidual[];
  /**
   * Per-camera target reprojection histogram (length = `num_cameras`).
   */
  target_hist_per_cam?: FeatureResidualHistogram[] | null;
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
   * Point-to-plane distance in meters between the back-projected ray and
   * the calibrated laser plane. `None` if the ray does not intersect.
   */
  residual_m?: number | null;
  /**
   * Distance (non-negative) from `observed_px` to the projected laser
   * line, in pixels: the magnitude of the line-distance residual the laser
   * solve minimizes, measured on the undistorted image plane and scaled by
   * `sqrt(fx·fy)`. For parallel laser and target planes the line is their
   * common vanishing line. `None` if the planes share no line with a
   * finite image (see the module documentation).
   */
  residual_px?: number | null;
  /**
   * View index in the input dataset.
   */
  view: number;
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
 * The calibrated laser line in one view's image.
 */
export interface LaserLineTrace {
  /**
   * Camera index. `0` for single-camera problem types.
   */
  camera: number;
  /**
   * Points of the line in pixels, in order along it, spanning the view's
   * laser pixels.
   */
  points_px: [number, number][];
  /**
   * View index in the input dataset.
   */
  view: number;
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
   * The error whitened by the feature's stated covariance,
   * `√(eᵀ Σ⁻¹ e)`: its size in units of the feature's standard deviation
   * along it. `None` when the input states no covariance or the
   * projection failed.
   */
  error_whitened?: number | null;
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
  /**
   * View index in the input dataset.
   */
  view: number;
}
/**
 * Make the sample dataset `id` (one of the library's sample catalogue)
 * available locally and answer the absolute path of its manifest. The
 * files are downloaded into the back end's samples folder (always one of
 * its roots), checked against their SHA-256 and reused later, so the app
 * carries no dataset. A back end built without samples answers
 * `invalid` (`server_info.samples` is false).
 */
export interface FetchSampleArgs {
  id: string;
}
/**
 * Drop `dir` from the recent projects (one that no longer opens).
 */
export interface ForgetRecentArgs {
  dir: UserPath;
}
/**
 * A form: the JSON Schema it renders and the value it starts from.
 */
export interface FormSchema {
  /**
   * The value a new form starts from: the type's default.
   */
  default: unknown;
  /**
   * A JSON Schema (draft 2020-12, as the library emits it).
   */
  schema: unknown;
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
 * The image file at `path`, as a browser can decode it: JPEG, PNG, WebP,
 * GIF and BMP are the file itself; TIFF, PNM and the other formats the
 * back end reads become an 8-bit PNG (a 16-bit image reduced as the
 * detectors reduce it). With `max`, a JPEG thumbnail whose longest side is
 * at most `max`, snapped up to 160, 320, 640 or 1280.
 */
export interface ImageQuery {
  max?: number | null;
  path: UserPath;
}
/**
 * A job as the registry knows it.
 */
export interface JobInfo {
  /**
   * The job command that started it.
   */
  command: string;
  /**
   * `<counter>-<8 random hex digits>`: unique within the process and not
   * reused by the next one.
   */
  id: string;
  /**
   * Its latest progress report, if it sent one.
   */
  last_event?: unknown;
  state: JobState;
}
/**
 * The job `id`: its state, its last progress report, its output once
 * done.
 */
export interface JobStatusArgs {
  id: string;
}
/**
 * Laser plane in camera frame: unit normal + signed distance.
 *
 * The plane equation is: n̂ · p + d = 0
 * where p is a point in camera coordinates.
 *
 * A plane is valid by construction: its normal has unit length (within
 * `1e-9`) and every value is finite, and a plane read from JSON is checked
 * the same way.
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
 * Export type for laserline device calibration.
 */
export interface LaserlineDeviceExport {
  /**
   * Per-view target poses (`cam_se3_target`).
   */
  cam_se3_target: Iso3Schema[];
  /**
   * The calibrated camera, in its model.
   */
  camera: CameraParams;
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * Export-type discriminator — always [`ProblemKind::LaserlineDevice`].
   */
  kind: ProblemKind;
  /**
   * The laser plane in the camera frame.
   */
  laser_plane: LaserPlane;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (single element for single-camera workflows).
   */
  per_cam_reproj_errors: number[];
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Laserline statistics payload.
   */
  stats: LaserlineStats;
  /**
   * Covariance of the calibrated parameters (camera and laser plane), with
   * the noise level the residuals imply.
   */
  uncertainty?: Uncertainty;
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
 * The covariance of a result's estimated parameters.
 */
export interface ParameterCovariance {
  /**
   * Covariance of the shared parameters: one row per label, symmetric.
   */
  covariance: number[][];
  /**
   * Residuals minus estimated parameters.
   */
  dof: number;
  /**
   * Labels of the parameters held fixed. The covariance is conditional on
   * them.
   */
  fixed?: string[];
  /**
   * Labels of the shared parameters, in the order of `covariance`.
   */
  labels: string[];
  /**
   * The noise level of each kind of residual, estimated from the
   * residuals at the solution.
   */
  noise: ResidualNoise[];
  /**
   * Number of scalar residuals, priors included.
   */
  num_residuals: number;
  /**
   * How far the error of the shared parameters is predicted to depart
   * from `covariance` once its second-order term is counted. `None` when
   * the result has no shared parameters or more than 64 (the prediction
   * costs `s (s + 1)` residual evaluations for `s` of them), its data is
   * noise-free, or the covariance was derived from others rather than
   * from one solve.
   */
  second_order?: SecondOrderError | null;
  /**
   * How many times the variance of the shared parameters across views
   * exceeds what the residual noise predicts, measured by leaving out one
   * view at a time. The residual noise model takes every corner's error
   * as independent; errors shared by a view's corners (a lens model or
   * target that does not fit exactly, a detector bias) are not in it, and
   * on real images make the estimate vary between view sets several
   * times more than predicted. `covariance` and `views` are already
   * scaled by it. `None` when the views scatter no more than the noise
   * predicts (within the scatter's own sampling noise, at 97.5 %), and
   * below six views.
   */
  view_scatter?: number | null;
  /**
   * Per-view parameters, one block per view that has free ones.
   */
  views?: ViewCovariance[];
}
/**
 * The error of a result's shared parameters at second order, against
 * their covariance.
 *
 * The covariance is the first-order error: linear in the noise. Where the
 * views pin a combination of parameters down only through large errors of
 * others that cancel in it (centred, mildly tilted views leave a camera's
 * rotation and its principal point to trade off this way), the error's
 * second-order term, quadratic in the noise, can exceed the first along
 * that combination while every parameter's own σ still holds. A consumer
 * that takes the whole covariance as a Gaussian (a prior, a test of two
 * calibrations against each other) then trusts the combination too much.
 *
 * The prediction comes from the residuals' curvature along simulated
 * first-order errors, at the result's own noise level; it grows with the
 * noise's variance.
 */
export interface SecondOrderError {
  /**
   * The worst combination, `Σ wᵢ pᵢ / σᵢ` over the shared parameters
   * `pᵢ`: its largest terms, by decreasing magnitude of the weight `wᵢ`
   * (the largest +1, the others down to a quarter of it, at most six).
   */
  combination: CombinationTerm[];
  /**
   * The predicted scatter of the worst combination of the shared
   * parameters, over the σ the covariance states for it: 1 when the
   * first order describes the error, 3 when the error there is three
   * times its stated σ.
   */
  max_scatter_ratio: number;
  /**
   * The predicted mean normalized estimation error squared of the shared
   * parameters (`eᵀ Σ⁻¹ e`), per parameter: 1 when the first order
   * describes the error.
   */
  nees_ratio: number;
}
/**
 * The covariance of one view's own parameters.
 */
export interface ViewCovariance {
  /**
   * Their covariance: one row per label, symmetric.
   */
  covariance: number[][];
  /**
   * Labels of the view's parameters, e.g. `cam_se3_target.rx`.
   */
  labels: string[];
  /**
   * View index, as in the result's per-view lists.
   */
  view: number;
}
/**
 * The folder `path`, or the root folders without one. Entries whose name
 * starts with a dot are left out, and so is a symbolic link that leads out
 * of the roots; past 5 000 entries the listing is `truncated`.
 */
export interface ListDirArgs {
  path?: UserPath | null;
  show?: Show;
}
/**
 * Every job the back end still knows of, live or recently finished, in
 * start order.
 */
export interface ListJobsArgs {}
/**
 * Read and parse the calibration export at `path` and check its schema
 * version. It does not become active: the frontend validates it first and
 * then calls `set_active_export`. Its images are read through the `image`
 * resource, so they too must lie inside a root.
 */
export interface LoadExportArgs {
  path: UserPath;
}
/**
 * A loaded export.
 */
export interface LoadExportResult {
  /**
   * The parsed export JSON; the frontend narrows it by its `kind`.
   */
  export: unknown;
  /**
   * The folder holding the export (absolute, `/`-separated): its image
   * manifest's `root` resolves against it.
   */
  export_dir: string;
}
/**
 * Open the project in `dir`; `not_a_project` for any other folder,
 * `forbidden_path` when its dataset lies outside the roots. It becomes
 * the most recent project.
 */
export interface OpenProjectArgs {
  dir: UserPath;
}
/**
 * Export format for planar intrinsics calibration.
 */
export interface PlanarIntrinsicsExport {
  /**
   * Per-view target poses (`cam_se3_target`).
   */
  cam_se3_target: Iso3Schema[];
  /**
   * The calibrated camera, in its model; `camera.sensor` states whether a
   * tilted sensor was calibrated.
   */
  camera: CameraParams;
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * Export-type discriminator — always [`ProblemKind::PlanarIntrinsics`].
   */
  kind: ProblemKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (single element for single-camera workflows).
   */
  per_cam_reproj_errors: number[];
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Covariance of the calibrated parameters: `fx ± σ` (and `tilt_x ± σ`
   * for a tilted sensor), correlations, and the noise level the residuals
   * imply.
   */
  uncertainty?: Uncertainty;
}
/**
 * An opened project, as the UI shows it.
 */
export interface ProjectInfo {
  /**
   * The folder the manifest's relative paths resolve against (absolute).
   */
  dataset_dir: string;
  /**
   * The project folder (absolute).
   */
  dir: string;
  /**
   * The dataset manifest (`dataset.json`), as stored.
   */
  manifest: unknown;
  /**
   * The saved runs, sorted by name.
   */
  runs: RunSummary[];
}
/**
 * One saved run: its folder name and diagnostics.
 */
export interface RunSummary {
  /**
   * `diagnostics.json`, or `None` when it is missing or does not parse
   * (the run is still listed, so its export can be opened).
   */
  diagnostics?: RunDiagnostics | null;
  /**
   * The run's folder name under `runs/`.
   */
  name: string;
}
/**
 * `diagnostics.json`: how a run went.
 */
export interface RunDiagnostics {
  /**
   * Whether detections were served from the detection cache.
   */
  cache_used: boolean;
  /**
   * Detection and calibration time, in milliseconds.
   */
  duration_ms: number;
  /**
   * The manifest topology the run calibrated.
   */
  topology: string;
  /**
   * Views attempted.
   */
  total_views: number;
  /**
   * Views with enough detected features to use.
   */
  usable_views: number;
}
/**
 * The projection uncertainty of one camera over its image.
 *
 * Grid nodes are evenly spaced from pixel `(0, 0)` to
 * `(width − 1, height − 1)`, corners included, and listed row by row from
 * the top-left. Each value is the major-axis standard deviation, in pixels,
 * of where the ray through the node projects.
 */
export interface ProjectionUncertainty {
  /**
   * σ at the image centre, pixels.
   */
  center: number;
  /**
   * Nodes per grid row.
   */
  columns: number;
  /**
   * The largest σ of the four corners, pixels; `None` when no ray reaches
   * any of them.
   */
  corner?: number | null;
  /**
   * `[width, height]` of the image, pixels.
   *
   * @minItems 2
   * @maxItems 2
   */
  image_size: [number, number];
  /**
   * The largest σ of the grid and the centre, pixels.
   */
  max: number;
  /**
   * Grid rows.
   */
  rows: number;
  /**
   * Major-axis σ at each node, pixels. `None` where no ray reaches the
   * node, as where a strong distortion folds the image over.
   */
  sigma: (number | null)[];
}
/**
 * The projection uncertainty of `camera` of the active export over an
 * image of `image_width` × `image_height` pixels.
 */
export interface ProjectionUncertaintyArgs {
  camera: number;
  image_height: number;
  image_width: number;
}
/**
 * Read the dataset manifest at `path`: TOML for a `.toml` extension, JSON
 * otherwise.
 */
export interface ReadManifestArgs {
  path: UserPath;
}
/**
 * The projects opened most recently on this back end, newest first.
 */
export interface RecentProjectsArgs {}
/**
 * Remove `path` from the root folders (the desktop). The samples' folder
 * is always a root.
 */
export interface RemoveRootArgs {
  path: UserPath;
}
/**
 * The configuration of the workflow `topology` that `patch` makes from its
 * defaults: tables merge key by key, anything else replaces (the library's
 * rule, as `vcal run --config` applies it), then checked. A field the
 * workflow does not have, or an invalid value, is a `config` error.
 */
export interface ResolveConfigArgs {
  patch?: ConfigPatch;
  topology: ProblemKind;
}
/**
 * Export format for rig extrinsics calibration.
 */
export interface RigExtrinsicsExport {
  /**
   * Per-camera extrinsics: `cam_se3_rig` (T_C_R).
   * Transform from rig frame to camera frame.
   */
  cam_se3_rig: Iso3Schema[];
  /**
   * Per-camera calibrated parameters, each in its camera's model.
   */
  cameras: CameraParams[];
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * Export-type discriminator — always [`ProblemKind::RigExtrinsics`].
   */
  kind: ProblemKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (pixels).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-view rig poses: `rig_se3_target` (T_R_T).
   */
  rig_se3_target: Iso3Schema[];
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Covariance of the calibrated parameters: `fx ± σ`, correlations, and
   * the noise level the residuals imply.
   */
  uncertainty?: Uncertainty;
}
/**
 * Export format for rig hand-eye calibration.
 */
export interface RigHandeyeExport {
  /**
   * Per-camera extrinsics: `cam_se3_rig` (T_C_R).
   * Transform from rig frame to camera frame.
   */
  cam_se3_rig: Iso3Schema[];
  /**
   * Per-camera calibrated parameters, each in its camera's model.
   */
  cameras: CameraParams[];
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * The hand-eye and the target reference pose of the calibration's mode.
   */
  handeye: HandeyeSolution;
  /**
   * Export-type discriminator — always [`ProblemKind::RigHandeye`].
   */
  kind: ProblemKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (pixels).
   */
  per_cam_reproj_errors: number[];
  /**
   * Per-view rig poses: `rig_se3_target` (T_R_T), derived from the
   * hand-eye chain (`handeye_observer_se3_target`) so downstream
   * viewers (3D scene, epipolar overlay) can read board poses
   * without re-implementing the chain. One entry per input view.
   * `#[serde(default)]` lets exports without this field load;
   * they decode with an empty Vec.
   */
  rig_se3_target?: Iso3Schema[];
  /**
   * The per-view robot-pose corrections and the pose error they were
   * weighed with: the configured `robot_poses` prior, or its estimate with
   * `estimate_sigma`. Only present if robot refinement was enabled.
   */
  robot_refinement?: RobotRefinement | null;
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Covariance of the calibrated parameters: `fx ± σ`, correlations, and
   * the noise level the residuals imply.
   */
  uncertainty?: Uncertainty;
}
/**
 * The robot-pose correction a hand-eye solve estimated with
 * `refine_robot_poses`.
 *
 * Absent from a result when the robot poses were taken as exact.
 */
export interface RobotRefinement {
  /**
   * Per-view robot-pose corrections, one se(3) tangent
   * `[rx, ry, rz, tx, ty, tz]` per view, in the gripper frame: the view's
   * corrected pose is `base_se3_gripper · exp(δ)`.
   */
  deltas: [number, number, number, number, number, number][];
  /**
   * The robot's pose error the corrections were weighed with: the
   * configured one, or the estimate when it was estimated.
   */
  sigma: RobotPoseSigma;
  /**
   * The parts of an estimated `sigma` that are upper bounds rather than
   * estimates: the corrections could not separate the robot's error
   * there from corner noise correlated within views, so that part of
   * `sigma` counts some of the corner noise as the robot's. Reported only
   * when the estimate leaves view-correlated corner noise out
   * (`discount_view_scatter`); `None` when every estimated part was
   * resolved, without that option, and when `sigma` was configured.
   */
  unresolved?: RobotSigmaParts | null;
}
/**
 * A robot's pose error: the standard deviations of the tool's orientation
 * error about each gripper axis and of its position error along each.
 */
export interface RobotPoseSigma {
  /**
   * Orientation error, radians.
   */
  rot_sigma: number;
  /**
   * Position error, metres.
   */
  trans_sigma: number;
}
/**
 * Export format for joint rig hand-eye laserline calibration.
 */
export interface RigHandeyeLaserlineExport {
  /**
   * Per-camera extrinsics `cam_se3_rig` (T_C_R).
   */
  cam_se3_rig: Iso3Schema[];
  /**
   * Per-camera parameters, each in its camera's model.
   */
  cameras: CameraParams[];
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * The hand-eye and the target reference pose of the calibration's mode.
   */
  handeye: HandeyeSolution;
  /**
   * Export-type discriminator — always [`ProblemKind::RigHandeyeLaserline`].
   */
  kind: ProblemKind;
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
  per_cam_stats: RigHandeyeLaserlinePerCamStats[];
  /**
   * Per-view target poses `rig_se3_target` (T_R_T).
   */
  rig_se3_target: Iso3Schema[];
  /**
   * The per-view robot-pose corrections and the pose error they were
   * weighed with: the configured `robot_poses` prior, or its estimate with
   * `estimate_sigma`. Only present if robot refinement was enabled.
   */
  robot_refinement?: RobotRefinement | null;
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Covariance of the jointly refined parameters, labelled as the export
   * fields (`cam1.fx`, `cam1.cam_se3_rig.tx`, `gripper_se3_rig.rz`), with
   * each camera's laser plane in its camera frame (`cam1.plane.nx`).
   */
  uncertainty?: Uncertainty;
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
  max_laser_error_m: number;
  /**
   * Maximum absolute laser line-distance in undistorted pixel space (pixels).
   */
  max_laser_error_px: number;
  /**
   * Maximum target-corner reprojection error (pixels).
   */
  max_reproj_error: number;
  /**
   * Mean laser point-to-plane distance (meters).
   */
  mean_laser_error_m: number;
  /**
   * Mean laser line-distance in undistorted pixel space (pixels).
   */
  mean_laser_error_px: number;
  /**
   * Mean target-corner reprojection error (pixels).
   */
  mean_reproj_error: number;
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
  reproj_histogram: [number, number, number, number, number];
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
   * Frozen upstream cameras, echoed so the export is self-contained for
   * downstream viewers (3D rig scene, epipolar) — same field name as
   * `RigHandeyeExport`. Empty when absent in the JSON (`serde(default)`).
   */
  cameras?: CameraParams[];
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * Export-type discriminator — always [`ProblemKind::RigLaserlineDevice`].
   */
  kind: ProblemKind;
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
   * Per-camera mean target reprojection errors (pixels) against the frozen
   * upstream, in camera order, over every target view of the camera
   * (`per_cam_stats[i].mean_reproj_error` averages only the views that also
   * have laser pixels); `0.0` for a camera with no projected feature, and
   * empty in exports serialized before this field existed.
   */
  per_cam_reproj_errors?: number[];
  /**
   * Per-camera stats (reprojection + laser residuals).
   */
  per_cam_stats: LaserlineStats[];
  /**
   * Frozen upstream per-view rig poses `T_R_T`.
   */
  rig_se3_target?: Iso3Schema[];
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Covariance of the laser planes in their camera frames
   * (`cam1.plane.nx … cam1.plane.d`). When the upstream calibration
   * carries its covariance, the planes' carries its share, and the
   * cameras' parameters and extrinsics are listed with it, correlated
   * with the planes: a laser point's σ then has the whole calibration in
   * it. The views' target poses enter the planes' share but are not
   * listed. Without one, the covariance is conditional on the upstream
   * calibration (its parameters are among `fixed`) and the planes of
   * different cameras are uncorrelated. Each camera's plane is solved on
   * its own, so each camera has its own noise levels.
   */
  uncertainty?: Uncertainty;
}
/**
 * One root folder.
 */
export interface RootFolder {
  /**
   * Its canonical path.
   */
  path: string;
  /**
   * The Quick Start samples' folder: always a root, never removed.
   */
  samples: boolean;
}
/**
 * Run a calibration end to end: detection (cached), the solve and the
 * export, reporting each stage and the library's reports within it. The
 * folder `manifest_dir` is what the manifest's relative paths resolve
 * against; every path the manifest names must lie inside a root. A
 * successful run's export becomes the active export.
 */
export interface RunCalibrationArgs {
  /**
   * The topology's `*Config` JSON.
   */
  config: unknown;
  /**
   * The dataset manifest (`DatasetSpec` JSON).
   */
  manifest: unknown;
  manifest_dir: UserPath;
}
/**
 * A progress report of a calibration run: the `run_calibration` job's
 * event. One is sent as each `RunStage` begins (no `report`) and then for
 * the library's reports within it; the outcome is the job's output (a
 * `RunResponse`), or its `cancelled` state.
 */
export interface RunProgress {
  /**
   * The library's report within the stage, absent when the stage
   * begins.
   */
  report?: Progress | null;
  /**
   * The stage under way.
   */
  stage: RunStage;
}
/**
 * Everything a saved run records (`runs/<name>/*.json`).
 */
export interface RunRecord {
  config: unknown;
  /**
   * The folder the run resolved the manifest's paths against.
   */
  dataset_dir: UserPath;
  diagnostics: RunDiagnostics;
  /**
   * The export as the runner returned it: its image manifest's `root` is
   * relative to `dataset_dir`.
   */
  export: unknown;
  manifest: unknown;
}
/**
 * Successful calibration run.
 */
export interface RunSuccess {
  /**
   * Whether at least one detection was served from the cache.
   * (Sanity check for "second run is faster"; precise per-image
   * hit/miss counts are not surfaced yet.)
   */
  cache_used: boolean;
  /**
   * Total wall-clock time for detection + calibration.
   */
  duration_ms: number;
  /**
   * Final export JSON, ready to drop into the diagnose viewer.
   */
  export: unknown;
  /**
   * Total number of views attempted (images for single-camera
   * topologies, paired rig views for rig topologies).
   */
  total_views: number;
  /**
   * Number of usable views after detection. For single-camera
   * topologies: views with >= 4 features. For rig topologies: views
   * where at least one camera reached >= 4 features.
   */
  usable_views: number;
}
/**
 * Replace the project's dataset manifest (`dataset.json`); every path it
 * names must lie inside a root.
 */
export interface SaveProjectManifestArgs {
  dir: UserPath;
  manifest: unknown;
}
/**
 * Save a run as `runs/<name>/`; `run_exists` rather than overwrite one.
 */
export interface SaveProjectRunArgs {
  dir: UserPath;
  name: string;
  record: RunRecord;
}
/**
 * One numbered event.
 */
export interface ServerEvent {
  body: EventBody;
  /**
   * Process-wide, increasing; a subscriber resumes after the last one it
   * saw.
   */
  seq: number;
}
/**
 * This back end.
 */
export interface ServerInfo {
  /**
   * The root folders: the only ones it reads and writes, where the folder
   * browser starts.
   */
  roots: RootFolder[];
  /**
   * Whether `add_root` and `remove_root` change them (the desktop); a
   * server's are fixed when it starts.
   */
  roots_editable: boolean;
  /**
   * Whether the Quick Start samples can be downloaded.
   */
  samples: boolean;
  /**
   * The shell: `desktop` or `server`.
   */
  shell: string;
  /**
   * The app's version.
   */
  version: string;
}
/**
 * What this back end is and what it serves. The frontend branches on
 * these values, never on which transport carries them.
 */
export interface ServerInfoArgs {}
/**
 * Make `export` the active export, the one every viewer command reads.
 */
export interface SetActiveExportArgs {
  export: unknown;
}
/**
 * Export format for single-camera hand-eye calibration.
 */
export interface SingleCamHandeyeExport {
  /**
   * Calibrated camera, in the configured model.
   */
  camera: CameraParams;
  /**
   * Residuals and the image manifest: evidence for viewers and
   * debugging, not part of the result.
   */
  diagnostics?: ExportDiagnostics;
  /**
   * The hand-eye and the target pose of the calibration's mode.
   */
  handeye: SingleCamHandeyeSolution;
  /**
   * Export-type discriminator — always [`ProblemKind::SingleCamHandeye`].
   */
  kind: ProblemKind;
  /**
   * Mean reprojection error (pixels).
   */
  mean_reproj_error: number;
  /**
   * Per-camera reprojection errors (pixels). Single element for single-camera.
   */
  per_cam_reproj_errors: number[];
  /**
   * The per-view robot-pose corrections and the pose error they were
   * weighed with: the configured `handeye_optimize.robot_poses` prior, or its estimate with
   * `estimate_sigma`. Only present if robot refinement was enabled.
   */
  robot_refinement?: RobotRefinement | null;
  /**
   * Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
   * Reading an export of another version is an error.
   */
  schema_version: number;
  /**
   * Covariance of the calibrated parameters: `fx ± σ`, correlations, and
   * the noise level the residuals imply.
   */
  uncertainty?: Uncertainty;
}
/**
 * Infer a dataset manifest from the folder `folder`: images grouped into
 * cameras, a robot-pose file, a guessed topology. What cannot be
 * determined is left at a placeholder and listed in the manifest's
 * `_unresolved` (the same inference as `vcal init`).
 */
export interface SniffFolderArgs {
  folder: UserPath;
}
/**
 * Map raw (distorted) pixels of `camera` into its undistorted pixel frame;
 * `null` where back-projection has no finite ray.
 */
export interface UndistortPointsArgs {
  camera: number;
  points_px: [number, number][];
}
/**
 * The image at `path` rendered into `camera`'s undistorted pixel frame of
 * the active export (a PNG), cropped to `roi` when given.
 */
export interface UndistortedImageQuery {
  camera: number;
  path: UserPath;
  roi?: PixelRect | null;
}
