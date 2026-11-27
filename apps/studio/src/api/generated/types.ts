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
/**
 * How [`residual_map`] chooses the kernel's bandwidth.
 */
export type Bandwidth =
  | "auto"
  | {
      pixels: number;
    };
/**
 * How [`residual_map`] chose its bandwidth.
 */
export type BandwidthSelection =
  | {
      kind: "given";
    }
  | {
      /**
       * The number of folds.
       */
      folds: number;
      kind: "cross_validated";
      /**
       * The scored ladder entries, pixels, ascending. Empty when no
       * entry predicted any held-out record; the bandwidth is then
       * `0.1·diagonal`.
       */
      ladder_px: number[];
      /**
       * The standard error of each score, from its per-fold scores, px².
       */
      score_std_errors_px2: number[];
      /**
       * The mean squared held-out error at each scored entry, px².
       */
      scores_px2: number[];
    }
  | {
      kind: "few_views";
    };
/**
 * The configuration of one calibration, tagged by the manifest topology it
 * applies to.
 *
 * In JSON the variant is the `topology` field and the problem's config is
 * the `config` field:
 * `{"topology": "planar_intrinsics", "config": {...}}`.
 */
export type CalibrationConfig =
  | {
      config: PlanarIntrinsicsConfig;
      topology: "planar_intrinsics";
    }
  | {
      config: SingleCamHandeyeConfig;
      topology: "single_cam_handeye";
    }
  | {
      config: LaserlineDeviceConfig;
      topology: "laserline_device";
    }
  | {
      config: RigExtrinsicsConfig;
      topology: "rig_extrinsics";
    }
  | {
      config: RigHandeyeConfig;
      topology: "rig_handeye";
    }
  | {
      config: RigLaserlineDeviceConfig;
      topology: "rig_laserline_device";
    }
  | {
      config: RigHandeyeLaserlineConfig;
      topology: "rig_handeye_laserline";
    };
/**
 * Which distortion model a camera carries.
 *
 * `dim() == 0` means the camera has no distortion parameter block.
 */
export type DistortionKind =
  | "none"
  | "brown_conrady5"
  | "rational8"
  | "thin_prism9"
  | "division1"
  | "kannala_brandt4";
/**
 * Which projection model a camera uses.
 */
export type ProjectionKind = "pinhole" | "equidistant";
/**
 * Which sensor model a camera uses.
 */
export type SensorKind = "identity" | "scheimpflug";
/**
 * Robust loss applied to a residual block.
 *
 * Each residual block has its own loss; per-point robustification is achieved
 * by using one residual block per observation.
 */
export type RobustLoss =
  | {
      kind: "none";
    }
  | {
      kind: "huber";
      /**
       * Transition residual norm, in residual units (pixels for
       * reprojection).
       */
      scale: number;
    }
  | {
      kind: "cauchy";
      /**
       * Residual norm where down-weighting sets in, in residual units.
       */
      scale: number;
    }
  | {
      kind: "arctan";
      /**
       * Transition in squared-residual units: down-weighting sets in near
       * `‖r‖ ≈ √scale`.
       */
      scale: number;
    };
/**
 * Which engine linearizes the problem.
 *
 * Both run the same Levenberg–Marquardt loop over the same residual
 * kernels and reach the same minimizer; they differ in how they build the
 * Jacobian and how they fold in a robust loss.
 */
export type SolverBackend = "tiny_solver" | "factrs";
/**
 * Hand-eye calibration mode.
 *
 * Specifies the transform chain used for hand-eye calibration.
 */
export type HandeyeMode = "eye_in_hand" | "eye_to_hand";
/**
 * The closed-form solver of a hand-eye initialization.
 *
 * All four need robot rotations about at least two non-parallel axes; they
 * differ in how they weigh the views and in what they estimate jointly.
 */
export type HandeyeMethod = "tsai_lenz" | "park_martin" | "daniilidis" | "shah";
/**
 * Robot-pose refinement options for hand-eye bundle adjustment.
 *
 * Shared by `SingleCamHandeyeConfig`, `HandeyeOptimizeConfig`, and the joint
 * rig+hand-eye+laser BA config (`JointOptimizeConfig`).
 *
 * [`Refine`](Self::Refine) gives every view's measured robot pose an se(3)
 * correction in the gripper frame, `base_se3_gripper · exp(δ)`, under a
 * zero-mean prior (see [`RobotPosePrior`]). [`Exact`](Self::Exact) takes the
 * robot poses as exact. Serialized as `{"kind": "exact"}` or
 * `{"kind": "refine", "rot_sigma": …, "trans_sigma": …, "estimate_sigma": …}`.
 * The default is `Refine` with the default prior: robot pose errors are
 * rarely below the corner noise, and taking them as exact pushes them into
 * the camera parameters.
 */
export type RobotPoseConfig =
  | {
      kind: "exact";
    }
  | {
      /**
       * With `estimate_sigma`, leave corner noise correlated within views out
       * of the estimated robot error. Each correction carries the pose-like
       * part of its view's corner noise; when the residuals are correlated
       * within views (a significant view scatter), that part is larger than
       * the independent-noise model says, and the plain estimate counts the
       * excess as the robot's. With this option the excess is taken out at
       * the view scatter's scale, measured on the shared parameters.
       *
       * It helps where the correlation shifts a view's corners in patches,
       * as a detector bias over a tile of the board does: there the plain
       * estimate can be several times the robot's error and the hand-eye
       * leans on the corners more than it should. Where the corrections do
       * not resolve the remaining robot error (by two standard errors over
       * the views), the estimate stays at the largest robot error they
       * cannot tell from none; where the noise would explain more than the
       * corrections hold, nothing is taken out and the estimate is the plain
       * one. `robot_refinement.unresolved` names the parts of the estimate
       * that are such upper bounds.
       *
       * Off by default: the scale comes from the shared parameters, and how
       * far it holds for the poses has been measured only on synthetic
       * single-camera scenes. Requires `estimate_sigma`.
       */
      discount_view_scatter?: boolean;
      /**
       * Estimate the robot's error from the corrections, starting from
       * `rot_sigma` and `trans_sigma`: each prior's variance factor is
       * estimated from the corrections and their redundancy, and the solve
       * repeats until it settles. The export reports the estimate
       * (`robot_refinement.sigma`). On by default: the stated accuracy of an
       * arm is rarely its error in one calibration, and a prior that is too
       * loose or too tight misstates the hand-eye and target standard
       * deviations. Off, `rot_sigma` and `trans_sigma` are taken as the
       * robot's true accuracy.
       */
      estimate_sigma: boolean;
      kind: "refine";
      /**
       * Standard deviation of the tool's orientation error about each gripper
       * axis (radians, > 0). Default 0.5°, a loose bound for an industrial
       * arm.
       */
      rot_sigma: number;
      /**
       * Standard deviation of the tool's position error along each gripper
       * axis (metres, > 0). Default 1 mm, a loose bound for an industrial
       * arm.
       */
      trans_sigma: number;
    };
/**
 * Type of laser plane residual to use.
 */
export type LaserlineResidualType = "point_to_plane" | "line_dist_normalized";
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
 * How an exclusion names a view.
 *
 * A view has no index that survives an image added to the dataset; its
 * pairing token and its images do. Serialized as `{ token = "0012" }` or
 * `{ image = "cam0/0012.png" }`.
 */
export type ViewRef =
  | {
      token: string;
    }
  | {
      image: string;
    };
/**
 * Why the runner left a target detection out of the calibration input.
 *
 * Serialized in `snake_case` (`"too_few_features"`, `"excluded"`, ...).
 */
export type RejectReason =
  | "too_few_features"
  | "partial_chessboard"
  | "turned_chessboard"
  | "ambiguous_chessboard_turn"
  | "excluded";
/**
 * Which stage of the camera model a parameter belongs to.
 */
export type ParameterGroup = "intrinsics" | "projection" | "distortion" | "sensor";
/**
 * A parameter's standard deviation, or why it has none.
 */
export type ParameterSigma =
  | {
      kind: "estimated";
      sigma: number;
    }
  | {
      kind: "unavailable";
      reason: SigmaUnavailable;
    };
/**
 * Why a parameter (or a camera's projection) has no σ.
 */
export type SigmaUnavailable =
  | {
      kind: "no_covariance";
      reason: UncertaintyUnavailable;
    }
  | {
      kind: "fixed";
    }
  | {
      kind: "not_listed";
    }
  | {
      kind: "unsupported_model";
      message: string;
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
 * What a check examines; [`assess`] describes each check.
 */
export type CheckKind =
  | "observability"
  | "convergence"
  | "variance_factor"
  | "projection_sigma"
  | "feature_coverage"
  | "view_outliers"
  | "parameter_correlation"
  | "covariance_linearity"
  | "parameter_conditioning";
/**
 * The outcome of one check.
 */
export type CheckStatus = "pass" | "warn" | "fail" | "not_applicable";
/**
 * A project's configuration as it resolves.
 */
export type ConfigState =
  | {
      config: unknown;
      kind: "ok";
    }
  | {
      error: StudioError;
      kind: "invalid";
    };
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
  | "ambiguous_project"
  | "project_exists"
  | "unsupported_version"
  | "changed_on_disk"
  | "io"
  | "parse"
  | "manifest"
  | "config"
  | "exclusion"
  | "run_not_found"
  | "import_mismatch"
  | "needs_input"
  | "detection"
  | "calibration"
  | "invalid"
  | "cancelled";
/**
 * A manifest file's path, or the manifest.
 */
export type DatasetManifest = string | DatasetSpec;
/**
 * A project's dataset manifest as it resolves.
 */
export type DatasetState =
  | {
      kind: "ok";
      spec: DatasetSpec;
    }
  | {
      error: StudioError;
      kind: "unreadable";
    };
/**
 * What a run's files say.
 */
export type RunState =
  | (RunRecord & {
      kind: "recorded";
    })
  | (UnrecordedRun & {
      kind: "unrecorded";
    })
  | {
      kind: "unreadable";
      /**
       * Why.
       */
      message: string;
    };
/**
 * Where a run's export came from.
 */
export type RunSource =
  | (CalibratedRun & {
      kind: "calibrated";
    })
  | {
      /**
       * The imported file's name (not its path).
       */
      file_name: string;
      kind: "imported";
    };
/**
 * The judgment of a result, or of one check.
 *
 * Ordered by severity: `Pass < Warn < Fail`.
 */
export type Verdict = "pass" | "warn" | "fail";
export type DeleteRunReply = RunInfo[];
/**
 * The outcome of one image's detection.
 */
export type DetectionOutcome =
  | {
      status: "detected";
    }
  | {
      /**
       * The detector's reason, for a reader.
       */
      reason: string;
      status: "not_found";
    }
  | {
      /**
       * Features whose pixel repeats an earlier feature's.
       */
      repeated_pixels: number;
      /**
       * Features whose target point repeats an earlier feature's.
       */
      repeated_points: number;
      status: "ambiguous";
    }
  | {
      status: "filtered";
    };
/**
 * Why a point candidate was left out.
 */
export type PointRejection =
  | {
      kind: "unlabelled";
      /**
       * The corner detector's response.
       */
      strength: number;
    }
  | {
      kind: "outside_board";
    }
  | {
      /**
       * The detector's confidence in the centre, in `[0, 1]`.
       */
      confidence: number;
      kind: "unanchored";
    }
  | {
      /**
       * The detector's confidence in the centre, in `[0, 1]`.
       */
      confidence: number;
      kind: "uncorrected_center";
    };
/**
 * Overlay data of one target kind.
 */
export type TargetDetail =
  | (ChessboardDetail & {
      kind: "chessboard";
    })
  | (CharucoDetail & {
      kind: "charuco";
    })
  | (PuzzleboardDetail & {
      kind: "puzzleboard";
    })
  | (RinggridDetail & {
      kind: "ringgrid";
    });
/**
 * Which estimator produced a ring marker's centre.
 */
export type RingCenter = "outer_ellipse" | "projective" | "photometric";
/**
 * What the cache holds of one camera's image.
 */
export type ImageDetectionState =
  | {
      kind: "not_detected";
    }
  | {
      features: number;
      kind: "detected";
      outcome?: DetectionOutcome | null;
    }
  | {
      error: StudioError;
      kind: "failed";
    };
export type EntryKind = "dir" | "file";
export type DisparityEvent = null;
/**
 * What happened.
 */
export type EventBody =
  | (JobEvent & {
      kind: "job";
    })
  | {
      /**
       * The project folder.
       */
      dir: string;
      kind: "project_changed";
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
 * What a listing shows.
 */
export type Show = "all" | "dirs";
export type ListJobsReply = JobInfo[];
export type ListRunsReply = RunInfo[];
/**
 * The unit of a measured value.
 */
export type Unit = "pixel" | "fraction" | "ratio" | "sigma" | "correlation";
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
/**
 * A camera's projection σ on the maps' grid, or why it has none.
 */
export type ProjectionSigmaMap =
  | {
      camera: number;
      grid: MapGrid;
      plane: MapPlane;
      status: "available";
    }
  | {
      camera: number;
      reason: SigmaUnavailable;
      status: "unavailable";
    };
export type RecentProjectsReply = string[];
export type ResolveConfigReply = unknown;
/**
 * Parts of a [`RobotPoseSigma`]: its rotation, its translation or both.
 */
export type RobotSigmaParts = "rotation" | "translation" | "both";
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
  Bandwidth?: Bandwidth;
  BandwidthSelection?: BandwidthSelection;
  BlobQuery?: BlobQuery;
  BlobRef?: BlobRef;
  CalibrateArgs?: CalibrateArgs;
  CalibratedRun?: CalibratedRun;
  CalibrationConfig?: CalibrationConfig;
  CameraConfig?: CameraConfig;
  CameraFix?: CameraFix;
  CameraModelSpec?: CameraModelSpec;
  CameraOverride?: CameraOverride;
  CameraParameter?: CameraParameter;
  CameraParameterSet?: CameraParameterSet;
  CameraParameterValuesArgs?: CameraParameterValuesArgs;
  CameraParameters?: CameraParameters;
  CameraParams?: CameraParams;
  CameraSource?: CameraSource;
  CameraStats?: CameraStats;
  CameraSummary?: CameraSummary;
  CancelJobArgs?: CancelJobArgs;
  CellStats?: CellStats;
  CharucoDetail?: CharucoDetail;
  CheckKind?: CheckKind;
  CheckStatus?: CheckStatus;
  ChessCornersDetectorSpec?: ChessCornersDetectorSpec;
  ChessboardDetail?: ChessboardDetail;
  CombinationTerm?: CombinationTerm;
  ConfigFormArgs?: ConfigFormArgs;
  ConfigPatch?: ConfigPatch;
  ConfigPatchArgs?: ConfigPatchArgs;
  ConfigState?: ConfigState;
  CornerCovarianceSpec?: CornerCovarianceSpec;
  CornerDetail?: CornerDetail;
  CornerQuality?: CornerQuality;
  CornerStrategySpec?: CornerStrategySpec;
  CreateProjectArgs?: CreateProjectArgs;
  DatasetFormArgs?: DatasetFormArgs;
  DatasetManifest?: DatasetManifest;
  DatasetSource?: DatasetSource;
  DatasetSpec?: DatasetSpec;
  DatasetState?: DatasetState;
  DeleteRunArgs?: DeleteRunArgs;
  DeleteRunReply?: DeleteRunReply;
  DetectAllArgs?: DetectAllArgs;
  DetectPreviewArgs?: DetectPreviewArgs;
  DetectProgress?: DetectProgress;
  DetectionDetail?: DetectionDetail;
  DetectionDetailArgs?: DetectionDetailArgs;
  DetectionOutcome?: DetectionOutcome;
  DetectionOverview?: DetectionOverview;
  DetectionOverviewArgs?: DetectionOverviewArgs;
  DetectorSpec?: DetectorSpec;
  DeviceIdentity?: DeviceIdentity;
  DeviceOptimizeConfig?: DeviceOptimizeConfig;
  DirEntry?: DirEntry;
  DirListing?: DirListing;
  DisparityArgs?: DisparityArgs;
  DisparityEvent?: DisparityEvent;
  DisparityMetrics?: DisparityMetrics;
  DisparityParams?: DisparityParams;
  DisparityResult?: DisparityResult;
  DistortionKind?: DistortionKind;
  DistortionMap?: DistortionMap;
  DistortionMapArgs?: DistortionMapArgs;
  DistortionParams?: DistortionParams;
  EdgeBitDetail?: EdgeBitDetail;
  EllipseDetail?: EllipseDetail;
  EntryKind?: EntryKind;
  EpipolarOverlay?: EpipolarOverlay;
  EpipolarOverlayArgs?: EpipolarOverlayArgs;
  EpipolarOverlayUndistortedArgs?: EpipolarOverlayUndistortedArgs;
  ErrorKind?: ErrorKind;
  EventBody?: EventBody;
  Exclusions?: Exclusions;
  ExportDiagnostics?: ExportDiagnostics;
  Feature?: Feature;
  FeatureBlur?: FeatureBlur;
  FeatureExclusion?: FeatureExclusion;
  FeatureResidualHistogram?: FeatureResidualHistogram;
  FetchSampleArgs?: FetchSampleArgs;
  FetchSampleEvent?: FetchSampleEvent;
  ForgetRecentArgs?: ForgetRecentArgs;
  ForgetRecentReply?: ForgetRecentReply;
  FormSchema?: FormSchema;
  FrameKind?: FrameKind;
  FrameRef?: FrameRef;
  FxFyCxCy?: FxFyCxCy;
  HandeyeInitConfig?: HandeyeInitConfig;
  HandeyeMethod?: HandeyeMethod;
  HandeyeMode?: HandeyeMode;
  HandeyeOptimizeConfig?: HandeyeOptimizeConfig;
  HandeyeSolution?: HandeyeSolution;
  ImageDetection?: ImageDetection;
  ImageDetectionState?: ImageDetectionState;
  ImageExclusion?: ImageExclusion;
  ImageManifest?: ImageManifest;
  ImageNoiseSpec?: ImageNoiseSpec;
  ImagePattern?: ImagePattern;
  ImageQuery?: ImageQuery;
  ImportExportArgs?: ImportExportArgs;
  InputDigest?: InputDigest;
  IntrinsicsInitConfig?: IntrinsicsInitConfig;
  IntrinsicsOptimizeConfig?: IntrinsicsOptimizeConfig;
  Iso3Schema?: Iso3Schema;
  JobEvent?: JobEvent;
  JobInfo?: JobInfo;
  JobState?: JobState;
  JobStatusArgs?: JobStatusArgs;
  JointOptimizeConfig?: JointOptimizeConfig;
  LaserExtractionSpec?: LaserExtractionSpec;
  LaserFeatureResidual?: LaserFeatureResidual;
  LaserLineTrace?: LaserLineTrace;
  LaserPlane?: LaserPlane;
  LaserScanAxis?: LaserScanAxis;
  LaserlineDeviceConfig?: LaserlineDeviceConfig;
  LaserlineDeviceExport?: LaserlineDeviceExport;
  LaserlineResidualType?: LaserlineResidualType;
  LaserlineStats?: LaserlineStats;
  LeftOutImage?: LeftOutImage;
  LibraryScan?: LibraryScan;
  ListDirArgs?: ListDirArgs;
  ListJobsArgs?: ListJobsArgs;
  ListJobsReply?: ListJobsReply;
  ListRunsArgs?: ListRunsArgs;
  ListRunsReply?: ListRunsReply;
  MapGrid?: MapGrid;
  MapPlane?: MapPlane;
  MarkerDetail?: MarkerDetail;
  Measurement?: Measurement;
  OpenProjectArgs?: OpenProjectArgs;
  ParameterCovariance?: ParameterCovariance;
  ParameterGroup?: ParameterGroup;
  ParameterSigma?: ParameterSigma;
  PerFeatureResiduals?: PerFeatureResiduals;
  PixelRect?: PixelRect;
  PlanarIntrinsicsConfig?: PlanarIntrinsicsConfig;
  PlanarIntrinsicsExport?: PlanarIntrinsicsExport;
  PlanarOptimizeConfig?: PlanarOptimizeConfig;
  PlaneChannel?: PlaneChannel;
  PlaneOptimizeConfig?: PlaneOptimizeConfig;
  PointRejection?: PointRejection;
  PoseColumnMap?: PoseColumnMap;
  PoseConvention?: PoseConvention;
  PosePairing?: PosePairing;
  ProblemKind?: ProblemKind;
  Progress?: Progress;
  ProjectDigest?: ProjectDigest;
  ProjectEntry?: ProjectEntry;
  ProjectSpec?: ProjectSpec;
  ProjectView?: ProjectView;
  ProjectViews?: ProjectViews;
  ProjectViewsArgs?: ProjectViewsArgs;
  ProjectionKind?: ProjectionKind;
  ProjectionParams?: ProjectionParams;
  ProjectionSigmaMap?: ProjectionSigmaMap;
  ProjectionSigmaMapArgs?: ProjectionSigmaMapArgs;
  ProjectionUncertainty?: ProjectionUncertainty;
  ProjectionUncertaintyArgs?: ProjectionUncertaintyArgs;
  PuzzleboardDecode?: PuzzleboardDecode;
  PuzzleboardDetail?: PuzzleboardDetail;
  QualityCheck?: QualityCheck;
  QualityReport?: QualityReport;
  QualityThresholds?: QualityThresholds;
  ReadManifestArgs?: ReadManifestArgs;
  RecentProjectsArgs?: RecentProjectsArgs;
  RecentProjectsReply?: RecentProjectsReply;
  RejectReason?: RejectReason;
  RejectedMarker?: RejectedMarker;
  RejectedPoint?: RejectedPoint;
  RemoveRootArgs?: RemoveRootArgs;
  ReportQuery?: ReportQuery;
  ReprojectionStats?: ReprojectionStats;
  ResidualKind?: ResidualKind;
  ResidualMap?: ResidualMap;
  ResidualMapArgs?: ResidualMapArgs;
  ResidualMapOptions?: ResidualMapOptions;
  ResidualNoise?: ResidualNoise;
  ResidualRecords?: ResidualRecords;
  ResidualRecordsArgs?: ResidualRecordsArgs;
  ResidualStats?: ResidualStats;
  ResidualStatsArgs?: ResidualStatsArgs;
  ResolveConfigArgs?: ResolveConfigArgs;
  ResolveConfigReply?: ResolveConfigReply;
  RigConfig?: RigConfig;
  RigExtrinsicsConfig?: RigExtrinsicsConfig;
  RigExtrinsicsExport?: RigExtrinsicsExport;
  RigHandeyeConfig?: RigHandeyeConfig;
  RigHandeyeExport?: RigHandeyeExport;
  RigHandeyeLaserlineConfig?: RigHandeyeLaserlineConfig;
  RigHandeyeLaserlineExport?: RigHandeyeLaserlineExport;
  RigHandeyeLaserlinePerCamStats?: RigHandeyeLaserlinePerCamStats;
  RigIntrinsicsManualInit?: RigIntrinsicsManualInit;
  RigLaserlineDeviceConfig?: RigLaserlineDeviceConfig;
  RigLaserlineDeviceExport?: RigLaserlineDeviceExport;
  RingCenter?: RingCenter;
  RingDetail?: RingDetail;
  RinggridCenterSpec?: RinggridCenterSpec;
  RinggridCodingSpec?: RinggridCodingSpec;
  RinggridDetail?: RinggridDetail;
  RinggridLatticeSpec?: RinggridLatticeSpec;
  RobotPoseConfig?: RobotPoseConfig;
  RobotPoseFormat?: RobotPoseFormat;
  RobotPoseSigma?: RobotPoseSigma;
  RobotPoseSource?: RobotPoseSource;
  RobotRefinement?: RobotRefinement;
  RobotSigmaParts?: RobotSigmaParts;
  RobustLoss?: RobustLoss;
  RootFolder?: RootFolder;
  RotationFormat?: RotationFormat;
  RunExport?: RunExport;
  RunExportArgs?: RunExportArgs;
  RunInfo?: RunInfo;
  RunInfoArgs?: RunInfoArgs;
  RunInputs?: RunInputs;
  RunNotes?: RunNotes;
  RunQualityArgs?: RunQualityArgs;
  RunRecord?: RunRecord;
  RunRef?: RunRef;
  RunSource?: RunSource;
  RunState?: RunState;
  RunSummary?: RunSummary;
  SampleInfo?: SampleInfo;
  SampleProject?: SampleProject;
  SaveProjectArgs?: SaveProjectArgs;
  ScanFailure?: ScanFailure;
  ScanLibraryArgs?: ScanLibraryArgs;
  ScheimpflugParams?: ScheimpflugParams;
  SecondOrderError?: SecondOrderError;
  SensorKind?: SensorKind;
  SensorParams?: SensorParams;
  ServerEvent?: ServerEvent;
  ServerInfo?: ServerInfo;
  ServerInfoArgs?: ServerInfoArgs;
  SetRunNotesArgs?: SetRunNotesArgs;
  Show?: Show;
  SigmaUnavailable?: SigmaUnavailable;
  SingleCamHandeyeConfig?: SingleCamHandeyeConfig;
  SingleCamHandeyeExport?: SingleCamHandeyeExport;
  SingleCamHandeyeOptimizeConfig?: SingleCamHandeyeOptimizeConfig;
  SingleCamHandeyeSolution?: SingleCamHandeyeSolution;
  SniffFolderArgs?: SniffFolderArgs;
  SniffedFolder?: SniffedFolder;
  SolverBackend?: SolverBackend;
  SolverConfig?: SolverConfig;
  StudioError?: StudioError;
  TargetDetail?: TargetDetail;
  TargetFeatureResidual?: TargetFeatureResidual;
  TargetSpec?: TargetSpec;
  TransformConvention?: TransformConvention;
  TranslationUnits?: TranslationUnits;
  Uncertainty?: Uncertainty;
  UncertaintyUnavailable?: UncertaintyUnavailable;
  UndistortPointsArgs?: UndistortPointsArgs;
  UndistortPointsReply?: UndistortPointsReply;
  UndistortedImageQuery?: UndistortedImageQuery;
  Unit?: Unit;
  UnrecordedRun?: UnrecordedRun;
  UserPath?: UserPath;
  Verdict?: Verdict;
  ViewCovariance?: ViewCovariance;
  ViewDetections?: ViewDetections;
  ViewExclusion?: ViewExclusion;
  ViewRef?: ViewRef;
  ViewRow?: ViewRow;
  ViewStats?: ViewStats;
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
 * Calibrate the project as it is on disk and write the run as its next
 * `runs/run-NNN/` (export, quality report, record, the HTML report, and
 * `notes` when given). `config_patch` changes the configuration for this
 * run only, over the project's, and is recorded with it; `redetect`
 * detects every image again instead of reading the detection cache. Every
 * file the project names must lie inside a root. Its events are the
 * library's progress reports; a cancelled run writes nothing and ends the
 * job `cancelled`.
 */
export interface CalibrateArgs {
  config_patch?: ConfigPatch;
  /**
   * The project folder.
   */
  dir: UserPath;
  notes?: RunNotes;
  redetect?: boolean;
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
 * What a person notes on a run, `notes.json`: the only file of a run that
 * changes after it is written.
 */
export interface RunNotes {
  /**
   * A short name for the run.
   */
  label?: string | null;
  /**
   * Free text.
   */
  notes?: string | null;
  /**
   * Whether the run is marked as one to keep.
   */
  starred?: boolean;
}
/**
 * A run the project calibrated: its inputs, their digest and what the
 * detection kept.
 */
export interface CalibratedRun {
  /**
   * The digest of the inputs.
   */
  digest: InputDigest;
  /**
   * How long the run took, milliseconds; for display only.
   */
  duration_ms: number;
  /**
   * What the run was given.
   */
  inputs: RunInputs;
  /**
   * The images left out, rejected by the screening or excluded.
   */
  left_out: LeftOutImage[];
  /**
   * Whether every image was detected again rather than read from the
   * detection cache.
   */
  redetected: boolean;
  /**
   * Views the dataset's images formed before screening.
   */
  total_views: number;
  /**
   * Feature exclusions whose points matched no feature of the detection
   * they were applied to, with those points only. An image left out
   * before its features were screened is listed in `left_out` instead.
   */
  unmatched_exclusions?: FeatureExclusion[];
  /**
   * Views that entered the calibration.
   */
  usable_views: number;
}
/**
 * The digest of a calibrated run's inputs: each `sha256:<hex>` of the
 * canonical JSON (keys sorted) of what it covers.
 *
 * Two runs with equal `inputs` digests, by the same library version, have
 * byte-identical `export.json` and `quality.json`. The data root's
 * location is not part of any digest: a dataset moved whole keeps them.
 */
export interface InputDigest {
  /**
   * The resolved configuration.
   */
  config: string;
  /**
   * The manifest, and the content of the files it names other than
   * images (the robot pose file, the upstream calibration).
   */
  dataset: string;
  /**
   * The sorted set of detection cache keys the run read or wrote: each
   * image's content, its detector and detector configuration and the
   * detector's version.
   */
  detections: string;
  /**
   * The exclusions, notes left out, each list sorted and without
   * repeats.
   */
  exclusions: string;
  /**
   * Over the other five, in their order here.
   */
  inputs: string;
  /**
   * The quality thresholds.
   */
  quality: string;
}
/**
 * What a calibrated run was given, resolved.
 */
export interface RunInputs {
  /**
   * The configuration they made.
   */
  config: CalibrationConfig;
  /**
   * The configuration changes: the project's, then the run's own.
   */
  config_patch: ConfigPatch;
  /**
   * The data root as the project stated it: the inline manifest's
   * `root`, or the manifest file's path. For the reader only.
   */
  data_root: string;
  /**
   * The manifest as the run used it.
   */
  dataset: DatasetSpec;
  /**
   * The project's exclusions.
   */
  exclusions: Exclusions;
  /**
   * The thresholds the quality report was judged by.
   */
  quality: QualityThresholds;
}
/**
 * Configuration for planar intrinsics calibration.
 *
 * Grouped by stage: the camera model with its fix lists in
 * [`CameraConfig`], the linear initialization in [`IntrinsicsInitConfig`],
 * the non-linear refinement in [`PlanarOptimizeConfig`], and the solver
 * engine in [`SolverConfig`].
 */
export interface PlanarIntrinsicsConfig {
  /**
   * The camera model and its fixed parameters.
   *
   * Any distortion kind with an identity sensor; a Scheimpflug sensor
   * needs a distortion model. A tilted sensor's solve without a tilt seed
   * runs a multi-start tilt sweep (see `step_optimize`).
   */
  camera?: CameraConfig;
  /**
   * Linear-initialization stage settings (`step_init`). The linear stage
   * fits only the distortion terms the camera's solves estimate.
   */
  intrinsics_init: IntrinsicsInitConfig;
  /**
   * Non-linear refinement stage settings (`step_optimize`).
   */
  optimize: PlanarOptimizeConfig;
  /**
   * Solver engine settings. `solver.max_iters` defaults to 120, sized for
   * a tilted sensor's tilt valley; the solver stops at convergence, so
   * the cap only bounds runs that do not converge.
   */
  solver: SolverConfig;
}
/**
 * The camera model a workflow calibrates, and which of its parameters stay
 * fixed.
 *
 * The model names one kind per slot — projection, distortion and sensor —
 * so a tilted (Scheimpflug) sensor is a value of `model.sensor`, not a
 * separate workflow. Behaviour that depends on the sensor (default fix
 * lists, the linear stage's tangential terms, the tilt multi-start) follows
 * from the model.
 */
export interface CameraConfig {
  /**
   * Parameters held fixed during the non-linear solve, named per slot
   * (`{"intrinsics": ["cx", "cy"], "distortion": ["k3"], "sensor":
   * ["tilt_y"]}`). A slot left out keeps the model's default: `k3` for
   * Brown-Conrady, and everything but the low-order radial terms (`k1`,
   * `k2`; the division model's `lambda`) under a Scheimpflug sensor,
   * because tangential and higher-order terms trade off against the tilt.
   * An unknown name is a configuration error.
   */
  fix?: CameraFix;
  /**
   * The camera model, one kind per slot. Default: pinhole projection,
   * Brown-Conrady (`k1, k2, k3, p1, p2`) distortion and an identity
   * (untilted) sensor — the model every workflow accepts.
   */
  model: CameraModelSpec;
}
/**
 * Camera parameters held fixed during a solve, named per slot.
 *
 * Each list names parameters of the model in that slot (`fx`, `cy`, `k3`,
 * `tilt_y`, …). `None` means the model's default
 * ([`CameraModelSpec::default_fix`]); an empty list fixes nothing in that
 * slot. Names are checked against the model by [`CameraFix::resolve`], before
 * any work.
 */
export interface CameraFix {
  /**
   * Fixed distortion coefficients; default: `k3` for Brown-Conrady, and
   * everything but the low-order radial terms under a Scheimpflug sensor.
   */
  distortion?: string[] | null;
  /**
   * Fixed intrinsics (`fx`, `fy`, `cx`, `cy`); default: none.
   */
  intrinsics?: string[] | null;
  /**
   * Fixed projection parameters; default: none.
   */
  projection?: string[] | null;
  /**
   * Fixed sensor parameters (`tilt_x`, `tilt_y`); default: none.
   */
  sensor?: string[] | null;
}
/**
 * A camera model: the kind in each slot. The values live in
 * [`CameraParams`](super::CameraParams); [`CameraParams::spec`](super::CameraParams::spec)
 * recovers the spec from them.
 */
export interface CameraModelSpec {
  /**
   * Distortion slot.
   */
  distortion: DistortionKind;
  /**
   * Projection slot.
   */
  projection: ProjectionKind;
  /**
   * Sensor slot.
   */
  sensor: SensorKind;
}
/**
 * Per-camera linear-initialization stage (Zhang's method + iterative
 * Brown-Conrady distortion fit).
 *
 * Shared by every intrinsics-bearing problem's `intrinsics_init` config
 * group.
 */
export interface IntrinsicsInitConfig {
  /**
   * Hold `k3` at zero in the linear stage even where the camera's
   * non-linear solves estimate it (default `true`: a sixth-order term
   * fitted linearly overfits typical data, and the solve that frees it
   * starts it from zero).
   *
   * Which terms the linear stage fits otherwise follows the camera: a
   * term the camera holds fixed, or its
   * model lacks, stays at zero, and so do the tangential terms of a
   * camera whose tilt is estimated.
   */
  fix_k3: boolean;
  /**
   * Number of iterations for iterative intrinsics estimation.
   */
  iterations: number;
}
/**
 * The non-linear refinement stage of a planar intrinsics calibration
 * (`step_optimize`).
 */
export interface PlanarOptimizeConfig {
  /**
   * Indices of poses to fix during optimization (e.g., \[0\] to fix first
   * pose). Default: none — the board fixes the target frame, and a fixed
   * pose makes the reported uncertainty conditional on it.
   */
  fix_poses: number[];
  /**
   * Robust loss on the target-corner reprojection residuals; its scale is
   * in pixels. Default `none`: plain least squares, the
   * maximum-likelihood estimate under Gaussian corner noise. A tilted
   * sensor's tilt sweep runs plain least squares and applies this loss
   * in its final solve.
   */
  reprojection_loss: RobustLoss;
}
/**
 * The solver engine: iteration budget, verbosity and backend, shared by
 * every problem's non-linear stages.
 *
 * It holds engine settings only. What a stage minimizes — its robust losses
 * and weights — belongs to the stage's own group (`optimize`,
 * `intrinsics_optimize`, `rig`, `handeye_optimize`, `joint_optimize`), named
 * by the residual family it applies to (`reprojection_loss`,
 * `laser_loss`).
 *
 * `max_iters` has no single workspace-wide default — problems override it
 * per solve-stage cost. Documented per-problem defaults: 50 standard; 120
 * for planar intrinsics (sized for a tilted sensor's tilt valley; the
 * solver stops at convergence, so the cap only bounds runs that do not
 * converge); 30 for a warm-started joint
 * rig+hand-eye+laser polish; 200 for the rig-laserline frozen-geometry
 * stage (cheap 1-DOF-per-view problem). This struct's own [`Default`] is
 * 50 — the plain value used by problems with no reason to deviate.
 */
export interface SolverConfig {
  /**
   * Engine that linearizes the problem for the Levenberg–Marquardt loop.
   * Both reach the same minimizer; see [`SolverBackend`].
   */
  backend?: SolverBackend & string;
  /**
   * Maximum iterations for the non-linear optimizer.
   */
  max_iters: number;
  /**
   * Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
   */
  verbosity: number;
}
/**
 * Configuration for single-camera hand-eye calibration.
 *
 * Grouped by stage, in the order the steps run: the camera model, the
 * intrinsics linear init and refinement, the hand-eye linear init and
 * bundle adjustment, and the solver engine. Every group's own `Default` is
 * the plain workspace default (no per-problem override is needed here,
 * unlike `PlanarIntrinsicsConfig`'s iteration cap).
 *
 * Two stages solve the target corners, each with its own
 * `reprojection_loss`: `intrinsics_optimize` and `handeye_optimize`. A
 * robust calibration sets both; the final stage's loss is the estimator.
 */
export interface SingleCamHandeyeConfig {
  /**
   * The camera model and the parameters its solves (the intrinsics stage
   * and the hand-eye bundle adjustment) hold fixed. Any distortion kind
   * with an identity sensor; a Scheimpflug sensor needs a distortion
   * model. A tilted sensor's intrinsics stage runs a multi-start tilt
   * sweep; seed its tilt through the intrinsics manual init.
   */
  camera?: CameraConfig;
  /**
   * Hand-eye linear-initialization stage settings (`step_handeye_init`).
   */
  handeye_init: HandeyeInitConfig;
  /**
   * Hand-eye bundle-adjustment stage settings (`step_handeye_optimize`).
   */
  handeye_optimize: SingleCamHandeyeOptimizeConfig;
  /**
   * Linear-initialization stage settings (`step_intrinsics_init`). The
   * linear stage fits only the distortion terms the camera's solves
   * estimate.
   */
  intrinsics_init: IntrinsicsInitConfig;
  /**
   * Intrinsics refinement stage settings (`step_intrinsics_optimize`).
   */
  intrinsics_optimize: IntrinsicsOptimizeConfig;
  /**
   * Solver engine settings, for both non-linear stages.
   */
  solver: SolverConfig;
}
/**
 * Hand-eye linear-initialization stage options.
 *
 * Shared by every hand-eye problem's `handeye_init` config group.
 */
export interface HandeyeInitConfig {
  /**
   * Hand-eye mode: `eye_in_hand` or `eye_to_hand`.
   */
  handeye_mode: HandeyeMode;
  /**
   * The closed-form solver of the initialization: `tsai_lenz` (default),
   * `park_martin`, `daniilidis` or `shah`. They differ in the accuracy of
   * the starting point, not in where the bundle adjustment ends: on
   * synthetic captures with robot rotations down to ±2° it converged from
   * each. `daniilidis` is the most accurate rotation when the robot
   * rotates by only a few degrees; `shah` estimates the fixed target
   * (eye-in-hand) or camera (eye-to-hand) pose over all views, where the
   * others close it through the first view.
   */
  method?: HandeyeMethod & string;
  /**
   * Minimum rotation (degrees) between two views, in the robot and in the
   * camera, for the pair to enter the linear solve; the pairs that clear
   * it must rotate the robot about two non-parallel axes. Positive.
   * Default 5°: smaller rotations amplify noise into the translation.
   */
  min_motion_angle_deg: number;
}
/**
 * The hand-eye bundle adjustment of a single-camera hand-eye calibration
 * (`step_handeye_optimize`): the camera, the hand-eye transform, the
 * target pose and, by default, the robot poses refined together, the
 * camera's `fix` lists held.
 */
export interface SingleCamHandeyeOptimizeConfig {
  /**
   * Robust loss on the target-corner reprojection residuals; its scale is
   * in pixels. Default `none`.
   */
  reprojection_loss: RobustLoss;
  /**
   * Robot-pose refinement.
   */
  robot_poses: RobotPoseConfig;
}
/**
 * The per-camera intrinsics refinement stage (`intrinsics_optimize`): each
 * camera solved on its own views, before the joint stages.
 *
 * Shared by the single-camera hand-eye and the two rig problems. The
 * stage's iteration budget and backend come from `solver`.
 */
export interface IntrinsicsOptimizeConfig {
  /**
   * Robust loss on the target-corner reprojection residuals of this
   * stage; its scale is in pixels. Default `none`: plain least squares,
   * the maximum-likelihood estimate under Gaussian corner noise. The
   * later stages have their own: a redescending loss (`cauchy`,
   * `arctan`) needs a good start, so a convex `huber` here and the
   * redescending one in the final stage is a sound pairing.
   */
  reprojection_loss: RobustLoss;
}
/**
 * Configuration for laserline device calibration.
 *
 * Settings are grouped by stage; the camera model and its fixed parameters
 * live in [`CameraConfig`].
 */
export interface LaserlineDeviceConfig {
  /**
   * The camera model and the parameters the bundle adjustment holds
   * fixed. Any distortion kind, and an identity or Scheimpflug sensor;
   * the line-distance laser residual needs a projection that keeps lines
   * straight. A Scheimpflug sensor's tilt is held by default at the
   * manual-init `sensor` seed (zero when unseeded): a device's mount tilt
   * is measured separately. An explicit `fix.sensor` list estimates it
   * instead (`[]` refines both angles; unseeded, the tilt then comes from
   * the linear tilt sweep and a staged multi-start).
   */
  camera?: CameraConfig;
  /**
   * Linear-initialization stage settings (`step_init`). The linear stage
   * fits only the distortion terms the bundle adjustment estimates
   * ([`Self::effective_fix`]). A Scheimpflug camera whose tilt is
   * estimated and not seeded is solved alone by the staged tilt
   * multi-start before the plane fit, in plain least squares.
   */
  intrinsics_init: IntrinsicsInitConfig;
  /**
   * Bundle-adjustment stage settings (`step_optimize`).
   */
  optimize: DeviceOptimizeConfig;
  /**
   * Solver engine settings.
   */
  solver: SolverConfig;
}
/**
 * Bundle-adjustment options for laserline device calibration.
 */
export interface DeviceOptimizeConfig {
  /**
   * Fix laser plane parameters during optimization.
   */
  fix_plane: boolean;
  /**
   * Indices of target poses to hold fixed. None by default: each view's
   * pose is measured by its own corners, so there is no gauge to fix. A
   * held pose keeps its initial value, and the linear init is not robust
   * to outlying corners: holding one keeps the solve off the minimum when
   * some corners are outliers.
   */
  fix_poses: number[];
  /**
   * Robust loss on the laser residuals. Its scale is in the residual's
   * unit: pixels for the line distance (the default residual), meters for
   * point-to-plane.
   */
  laser_loss: RobustLoss;
  /**
   * Laser residual type.
   */
  laser_residual_type: LaserlineResidualType;
  /**
   * Weight of the laser residuals.
   */
  laser_weight: number;
  /**
   * Robust loss on the target-corner reprojection residuals; its scale is
   * in pixels.
   */
  reprojection_loss: RobustLoss;
  /**
   * Weight of the reprojection residuals.
   */
  reprojection_weight: number;
}
/**
 * Configuration for multi-camera rig extrinsics calibration.
 *
 * Grouped by stage. Every camera has the model of `camera` unless
 * `camera_overrides` names it, so one rig can mix lenses and sensors.
 *
 * Two stages solve the target corners, each with its own
 * `reprojection_loss`: `intrinsics_optimize` (each camera alone) and `rig`
 * (the rig bundle adjustment). A robust calibration sets both; the final
 * stage's loss is the estimator.
 *
 * The reference-camera gauge fix (`rig.reference_camera_idx`) alone removes
 * the full 6-DOF rig gauge; an extra per-view pose constraint would be
 * redundant and mildly pessimizing. View-0's rig-from-target pose is always free in the rig BA.
 */
export interface RigExtrinsicsConfig {
  /**
   * The camera model of every rig camera and the parameters its solves
   * hold fixed. A tilted (Scheimpflug) sensor's per-camera solve runs a
   * multi-start tilt sweep.
   */
  camera?: CameraConfig;
  /**
   * Cameras whose model differs from `camera`, at most one entry per
   * camera index.
   */
  camera_overrides?: CameraOverride[];
  /**
   * Per-camera linear-initialization stage settings
   * (`step_intrinsics_init_all`). The linear stage fits only the
   * distortion terms each camera's solves estimate.
   */
  intrinsics_init: IntrinsicsInitConfig;
  /**
   * Per-camera refinement stage settings (`step_intrinsics_optimize_all`).
   */
  intrinsics_optimize: IntrinsicsOptimizeConfig;
  /**
   * Rig stage settings (`step_rig_init`, `step_rig_optimize`): the
   * reference camera, and the scope and loss of the rig bundle
   * adjustment.
   */
  rig: RigConfig;
  /**
   * Solver engine settings, for every non-linear stage.
   */
  solver: SolverConfig;
}
/**
 * One camera of a rig whose model differs from the rig's `camera`, so a
 * rig can mix lenses (a tilted laser camera beside a frontal one, a fisheye
 * beside a narrow lens).
 */
export interface CameraOverride {
  /**
   * Index of the camera in the rig (`0..num_cameras`).
   */
  camera: number;
  /**
   * Parameters held fixed during the non-linear solves, named per slot;
   * a slot left out keeps the model's default (see [`CameraConfig::fix`]).
   */
  fix?: CameraFix;
  /**
   * The camera's model, one kind per slot.
   */
  model: CameraModelSpec;
}
/**
 * The rig stage: the reference-camera gauge, and the scope and loss of the
 * rig bundle adjustment.
 *
 * The `rig` group of `RigExtrinsicsConfig` and `RigHandeyeConfig`.
 */
export interface RigConfig {
  /**
   * Reference camera index for the rig frame (identity extrinsics).
   */
  reference_camera_idx: number;
  /**
   * Refine per-camera intrinsics and distortion jointly with the rig in
   * its bundle adjustment, holding the camera's `fix` lists. Default:
   * `true`. Each camera's solve fits them from its own views only; the
   * joint solve is the estimate that uses every view, and its covariance
   * carries the intrinsics and their correlation with the extrinsics.
   * Held fixed, the extrinsics' standard deviations are conditional on
   * the intrinsics and understate the uncertainty (several-fold on real
   * stereo rigs).
   */
  refine_intrinsics: boolean;
  /**
   * Refine the sensor tilt of each Scheimpflug camera in the rig bundle
   * adjustment, holding its `fix.sensor` list. Default: `true`, for the
   * reason given on `refine_intrinsics`: a camera's own views pin its
   * tilt poorly (it trades with the principal point), the views of the
   * whole rig pin it several times better, and a tilt held at the
   * per-camera estimate leaves its error in the focal length and the
   * principal point while the covariance, conditional on the held tilt,
   * understates their standard deviations several-fold. `false` holds
   * the tilt at the per-camera estimate; the covariance then lists it as
   * fixed. Cameras with an untilted sensor have no tilt.
   */
  refine_sensor?: boolean;
  /**
   * Robust loss on the target-corner reprojection residuals of the rig
   * bundle adjustment; its scale is in pixels. Default `none`.
   */
  reprojection_loss: RobustLoss;
}
/**
 * Configuration for multi-camera rig hand-eye calibration.
 *
 * Grouped by stage. Every camera has the model of `camera` unless
 * `camera_overrides` names it, so one rig can mix lenses and sensors.
 *
 * Three stages solve the target corners, each with its own
 * `reprojection_loss`: `intrinsics_optimize` (each camera alone), `rig`
 * (the rig bundle adjustment) and `handeye_optimize`. A robust calibration
 * sets all three; the final stage's loss is the estimator.
 *
 * The reference-camera gauge fix (`rig.reference_camera_idx`) alone removes
 * the full 6-DOF rig gauge; an extra per-view pose constraint would be
 * redundant and mildly pessimizing (as for `RigExtrinsicsConfig`).
 * View-0's rig-from-target pose is always free in the rig BA.
 */
export interface RigHandeyeConfig {
  /**
   * The camera model of every rig camera and the parameters its solves
   * hold fixed. A tilted (Scheimpflug) sensor's per-camera solve runs a
   * multi-start tilt sweep; seed its tilt through `manual_init`.
   */
  camera?: CameraConfig;
  /**
   * Cameras whose model differs from `camera`, at most one entry per
   * camera index.
   */
  camera_overrides?: CameraOverride[];
  /**
   * Hand-eye linear-initialization stage settings (`step_handeye_init`).
   */
  handeye_init: HandeyeInitConfig;
  /**
   * Final hand-eye bundle-adjustment stage settings
   * (`step_handeye_optimize`).
   */
  handeye_optimize: HandeyeOptimizeConfig;
  /**
   * Per-camera linear-initialization stage settings
   * (`step_intrinsics_init_all`). The linear stage fits only the
   * distortion terms each camera's solves estimate.
   */
  intrinsics_init: IntrinsicsInitConfig;
  /**
   * Per-camera refinement stage settings (`step_intrinsics_optimize_all`).
   */
  intrinsics_optimize: IntrinsicsOptimizeConfig;
  /**
   * Optional manual seeds for the per-camera intrinsics stage. They seed
   * `step_intrinsics_init_all` and `run_calibration` alike;
   * `step_intrinsics_init_all_with_seed` takes an explicit seed that
   * overrides this one.
   */
  manual_init?: RigIntrinsicsManualInit | null;
  /**
   * Rig stage settings (`step_rig_init`, `step_rig_optimize`): the
   * reference camera, and the scope and loss of the rig bundle
   * adjustment.
   */
  rig: RigConfig;
  /**
   * Solver engine settings, for every non-linear stage.
   */
  solver: SolverConfig;
}
/**
 * The final hand-eye bundle adjustment of a rig hand-eye calibration
 * (`step_handeye_optimize`): its loss, robot-pose refinement and scope.
 */
export interface HandeyeOptimizeConfig {
  /**
   * Refine cam_se3_rig in hand-eye BA (default: false).
   * When true, rig extrinsics are further refined during hand-eye optimization.
   */
  refine_cam_se3_rig: boolean;
  /**
   * Refine each camera's intrinsics and distortion in the hand-eye BA,
   * holding the camera's `fix` lists. Default: `true`, for the reason
   * given on `rig.refine_intrinsics`: the final solve's covariance then
   * includes the intrinsics.
   */
  refine_intrinsics: boolean;
  /**
   * Refine the sensor tilt of each Scheimpflug camera in the hand-eye
   * BA, holding its `fix.sensor` list. Default: `true`, for the reason
   * given on `rig.refine_sensor`: the final solve's covariance then
   * includes the tilts. Cameras with an untilted sensor have no tilt.
   */
  refine_sensor?: boolean;
  /**
   * Robust loss on the target-corner reprojection residuals; its scale is
   * in pixels. Default `none`.
   */
  reprojection_loss: RobustLoss;
  /**
   * Robot-pose refinement settings for the final hand-eye BA.
   */
  robot_poses: RobotPoseConfig;
}
/**
 * Manual seeds for the **per-camera intrinsics stage** of a rig
 * calibration.
 *
 * All fields are `Option<T>`. When `per_cam_intrinsics` is `Some`, Zhang's
 * auto-fit is skipped for *every* camera; per-camera target poses are then
 * recovered from per-camera homographies using the seeded intrinsics. When
 * `None`, each camera is auto-initialized.
 *
 * When intrinsics are seeded but distortion is not, distortion starts at
 * zero; when intrinsics are auto-initialized, the fitted distortion is used
 * unless distortion is seeded. Each distortion seed is a value in its
 * camera's own distortion model: its kind must equal that camera's
 * configured distortion kind (a typed error otherwise) and it is used as
 * given.
 *
 * `per_cam_sensors` seeds the tilt of each camera with a Scheimpflug
 * sensor (a seeded tilt skips the tilt sweep of the linear stage); a camera
 * with an untilted sensor takes only a zero tilt.
 *
 * Per-camera vectors must have one entry per camera.
 */
export interface RigIntrinsicsManualInit {
  /**
   * Per-camera distortion seeds, each of the kind its camera's model names.
   */
  per_cam_distortion?: DistortionParams[] | null;
  /**
   * Per-camera intrinsics seeds. `None` runs Zhang's method per camera.
   */
  per_cam_intrinsics?: FxFyCxCy[] | null;
  /**
   * Per-camera sensor tilt seeds (Scheimpflug cameras; zero for the
   * others).
   */
  per_cam_sensors?: ScheimpflugParams[] | null;
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
 * Configuration for rig laserline calibration.
 *
 * Grouped by stage. `solver.max_iters` defaults to 200: this stage
 * refines only per-camera laser-plane parameters against an already-frozen
 * rig geometry (three parameters per camera), so iterations are nearly
 * free.
 */
export interface RigLaserlineDeviceConfig {
  /**
   * Laser-plane refinement stage settings (`step_optimize`).
   */
  optimize: PlaneOptimizeConfig;
  /**
   * Solver engine settings.
   */
  solver: SolverConfig;
}
/**
 * The laser-plane refinement of a rig laserline calibration
 * (`step_optimize`): each camera's plane fitted to its laser pixels, the
 * cameras and the target poses held at the upstream calibration. The
 * target corners only place the target; with everything they constrain
 * held, they have no loss or weight here.
 */
export interface PlaneOptimizeConfig {
  /**
   * Robust loss on the laser residuals, its scale in the residual's unit:
   * pixels for the line distance, meters for point-to-plane. Default
   * Huber at 1, a pixel for the line distance; for point-to-plane a
   * scale of 1 m is plain least squares in effect, so state one in
   * meters to robustify it.
   */
  laser_loss: RobustLoss;
  /**
   * The laser residual: the line distance in pixels (the default) or the
   * point-to-plane distance in meters.
   */
  laser_residual_type: LaserlineResidualType;
}
/**
 * Configuration for joint rig hand-eye laserline calibration.
 *
 * The calibration runs two warm-start problems and then its own joint
 * solve. Each warm start is configured by its problem's whole config —
 * `handeye` a rig hand-eye calibration, `laserline_init` a rig laserline
 * calibration of the planes against that frozen geometry — and every one
 * of their fields applies to that stage. The joint bundle adjustment is
 * configured by `joint_optimize` and `solver`, the same shape as a
 * single-camera laserline device's `optimize` and `solver`.
 */
export interface RigHandeyeLaserlineConfig {
  /**
   * The warm-start rig hand-eye calibration: cameras, extrinsics,
   * hand-eye transform and target reference pose. Its `rig`
   * reference camera also fixes the rig gauge of the joint solve.
   */
  handeye: RigHandeyeConfig;
  /**
   * The joint bundle-adjustment stage.
   */
  joint_optimize: JointOptimizeConfig;
  /**
   * The warm-start laser-plane fit against the frozen hand-eye geometry.
   */
  laserline_init: RigLaserlineDeviceConfig;
  /**
   * Solver engine settings of the joint stage. `max_iters` defaults to
   * 30: the stage starts from both warm starts and needs few iterations.
   */
  solver: SolverConfig;
}
/**
 * Final joint BA options.
 *
 * The joint stage always pins the upstream rig's reference camera
 * (`handeye.rig.reference_camera_idx`) — see `joint_fix_extrinsics`. Each
 * camera's fixed parameters are named lists in that camera's model
 * (`fix`).
 */
export interface JointOptimizeConfig {
  /**
   * Camera parameters held fixed in the joint stage, one entry per camera
   * in camera order, each named per slot in that camera's model
   * (`{"distortion": ["k3", "p1", "p2"], "sensor": ["tilt_x"]}`); a slot
   * left out of an entry keeps the model's default. Empty (the default)
   * gives every camera the stage default: intrinsics, the low-order
   * radial terms (`k1`, `k2`; the division model's `lambda`) and the
   * sensor tilt refined, the higher-order distortion terms held at their
   * warm-start values — they trade off against the extrinsics and the
   * laser planes in the joint solve. The tilt is refined like the
   * intrinsics: held, the covariance would be conditional on it and
   * understate the intrinsics' standard deviations. A name the camera's
   * model does not have is a configuration error.
   */
  fix?: CameraFix[];
  /**
   * Fix the hand-eye transform during the joint stage.
   */
  fix_handeye: boolean;
  /**
   * Fix the target reference pose during the joint stage.
   */
  fix_target_ref: boolean;
  /**
   * Robust loss on the laser residuals, its scale in the residual's unit
   * (meters for point-to-plane, pixels for the line distance).
   */
  laser_loss: RobustLoss;
  /**
   * Which laser residual drives the joint solve.
   */
  laser_residual_type: LaserlineResidualType;
  /**
   * Weight of the laser residuals.
   */
  laser_weight: number;
  /**
   * Robust loss on the target-corner reprojection residuals; its scale is
   * in pixels.
   */
  reprojection_loss: RobustLoss;
  /**
   * Weight of the reprojection residuals.
   */
  reprojection_weight: number;
  /**
   * Robot-pose refinement settings for the joint stage.
   */
  robot_poses: RobotPoseConfig;
}
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
 * What a project leaves out of its dataset.
 *
 * Image paths are relative to the dataset's folder unless absolute, and
 * `/`-separated; they match the images the manifest lists after both are
 * made absolute and their `.` and `..` folded.
 *
 * In TOML:
 *
 * ```toml
 * [exclude]
 * views  = [{ view = { token = "0012" }, note = "robot pose glitch" }]
 * images = [{ camera = "cam1", image = "cam1/0003.png" }]
 * [[exclude.features]]
 * camera = "cam0"
 * image  = "cam0/0005.png"
 * points = [[0.03, 0.06, 0.0]]
 * ```
 */
export interface Exclusions {
  /**
   * Single features of one camera's detection in one image.
   */
  features?: FeatureExclusion[];
  /**
   * One camera's image of a view; the view keeps its other cameras.
   */
  images?: ImageExclusion[];
  /**
   * Whole views: every camera's image of the view, its robot pose and
   * its laser frames.
   */
  views?: ViewExclusion[];
}
/**
 * Features of one camera's detection in one image, left out.
 *
 * A feature is named by its target point as the detector labelled it,
 * before a rig turns a plain chessboard's labels to agree with its other
 * cameras: the detection a review shows. A point matches a feature within
 * a micrometre in each coordinate. A point that matches no feature of the
 * detection leaves nothing out and is not an error, since what the
 * detector finds is known only once it has run.
 */
export interface FeatureExclusion {
  /**
   * The camera's id in the manifest.
   */
  camera: string;
  /**
   * The image, one of the camera's.
   */
  image: string;
  /**
   * Why, for the reader; the calibration does not read it.
   */
  note?: string | null;
  /**
   * The target points of the features to leave out, metres.
   */
  points: [number, number, number][];
}
/**
 * One camera's image of a view, left out.
 *
 * The camera is named with the image because a tiled image holds several
 * cameras' views.
 */
export interface ImageExclusion {
  /**
   * The camera's id in the manifest.
   */
  camera: string;
  /**
   * The image, one of the camera's.
   */
  image: string;
  /**
   * Why, for the reader; the calibration does not read it.
   */
  note?: string | null;
}
/**
 * A view left out, with every camera's image of it.
 */
export interface ViewExclusion {
  /**
   * Why, for the reader; the calibration does not read it.
   */
  note?: string | null;
  /**
   * Which view.
   */
  view: ViewRef;
}
/**
 * Thresholds of the quality checks.
 *
 * Every default is a limit a well-planned calibration of a perspective
 * camera clears, and that a calibration with a known planning flaw (views
 * bunched in one part of the image, near-identical board poses) does not.
 * The expected feature noise has no default: it depends on the detector,
 * the target and the optics.
 */
export interface QualityThresholds {
  /**
   * Share of the image's coverage cells, in `[0, 1]` and at most
   * `coverage_warn_below`, below which the coverage check fails. Default
   * `0.3`: the views reached little more than one quadrant of the image,
   * so the rest of it, corners included, is extrapolated.
   */
  coverage_fail_below?: number;
  /**
   * Share of the image's coverage cells, in `[0, 1]`, below which the
   * coverage check warns. Default `0.5`: features in fewer than half of
   * the cells leave the distortion to be extrapolated over the rest.
   */
  coverage_warn_below?: number;
  /**
   * Expected noise of one feature coordinate, σ₀, in pixels per axis
   * (> 0). `None` (the default) leaves the variance-factor check not
   * applicable: there is no universal value, since it depends on the
   * detector, the target and the image quality. State it to test whether
   * the residuals are consistent with it.
   */
  expected_feature_noise_px?: number | null;
  /**
   * Largest acceptable absolute correlation of two parameters of one
   * camera, in `(0, 1]`; a pair above it is named and the check warns.
   * Default `0.99`: at that level the data determines the pair's sum or
   * difference, not the two parameters, so each one's value is mostly the
   * trade-off's choice. The two focal lengths `fx` and `fy` are left out:
   * the board's distance scales both, so they correlate almost fully in
   * every planar calibration, while their ratio is well determined.
   */
  max_parameter_correlation?: number;
  /**
   * Largest acceptable projection σ anywhere in the image, in pixels
   * (> 0); above it the check warns. Default `1.0`: where the views
   * covered the image, σ stays a fraction of a pixel; it exceeds a pixel
   * where the distortion is extrapolated, typically in corners the board
   * never reached. The map is conservative for uses that re-estimate the
   * camera's pose, so the check warns rather than fails.
   */
  max_projection_sigma_px?: number;
  /**
   * Largest acceptable predicted scatter of the worst combination of the
   * shared parameters over its stated σ, once the error's second-order
   * term is counted (≥ 1); above it the covariance-linearity check warns,
   * naming the combination. Default `1.2`: on synthetic stereo rigs and
   * single cameras whose error the covariance describes the prediction
   * stayed at or below 1.06, and every stereo capture predicted above 1.3
   * scattered 1.4–3 times its σ along that combination.
   */
  max_second_order_scatter?: number;
  /**
   * Largest acceptable σ inflation `1/√(1 − R²)` of one camera parameter,
   * R its multiple correlation with the camera's other parameters (≥ 1);
   * above it the conditioning check warns, naming the parameter and the
   * partners it trades off against. The other focal length is left out
   * of a focal length's regressors, and the other radial coefficients out
   * of a radial coefficient's. Default `7.0`: a pair at the correlation
   * limit 0.99 inflates both by 7.09; well-determined sensor tilts
   * measured 1.2–5, tilts the data cannot separate from the principal
   * point or the tangential terms 14–2900.
   */
  max_sigma_inflation?: number;
  /**
   * Largest acceptable mean normalized residual of one view, in units of
   * the noise level σ̂ (> 0); a view above it is named and the check warns.
   * Default `3.0`: the mean of `|r| / σ̂` over Gaussian inliers is about
   * 1.25, so a view at 3 has residuals well over twice the typical ones,
   * as from a misdetection, motion blur or a wrong robot pose.
   */
  max_view_residual?: number;
  /**
   * Significance level α of the two-sided χ² variance-factor test, in
   * `(0, 1)`. Default `0.01`: a result consistent with σ₀ fails the test
   * once in a hundred calibrations.
   */
  variance_test_alpha?: number;
}
/**
 * An image a run left out, with the reason.
 */
export interface LeftOutImage {
  /**
   * The camera's id in the manifest.
   */
  camera: string;
  /**
   * The image, relative to the data root (absolute when outside it),
   * `/`-separated.
   */
  image: string;
  /**
   * Why it was left out.
   */
  reason: RejectReason;
}
/**
 * One parameter of a camera.
 */
export interface CameraParameter {
  /**
   * The solve ended with it on a box bound (the reason the run has no
   * covariance). The bounds themselves are the workflow's own and not
   * recorded in the export.
   */
  at_bound: boolean;
  group: ParameterGroup;
  /**
   * Its label in the export's uncertainty (`fx`, `cam1.k1`).
   */
  label: string;
  /**
   * Its name in the camera model (`fx`, `k1`, `tilt_x`).
   */
  name: string;
  sigma: ParameterSigma;
  /**
   * `px` (intrinsics), `rad` (sensor tilt) or empty (a coefficient).
   */
  unit: string;
  value: number;
}
/**
 * One camera's model and parameters.
 */
export interface CameraParameterSet {
  camera: number;
  distortion: DistortionKind;
  /**
   * Its parameters in the solver's order; empty for a model the solver
   * does not estimate (`unsupported` says why).
   */
  parameters: CameraParameter[];
  projection: ProjectionKind;
  /**
   * `null` for a general sensor homography.
   */
  sensor?: SensorKind | null;
  unsupported?: string | null;
}
/**
 * Every parameter of every camera of the run, whatever its model: its
 * label in the export's uncertainty, name, group, value and unit, its σ
 * or why it has none (no covariance and the library's reason, held fixed,
 * not listed), and whether the solve ended with it on a bound.
 */
export interface CameraParameterValuesArgs {
  run: RunRef;
}
/**
 * A run of a project: the project folder and the run's id (its folder
 * under `runs/`).
 */
export interface RunRef {
  /**
   * The project folder.
   */
  dir: UserPath;
  /**
   * The run's id, as `run-007`.
   */
  id: string;
}
/**
 * Every camera's parameters, in camera order.
 */
export interface CameraParameters {
  cameras: CameraParameterSet[];
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
 * One camera's statistics.
 */
export interface CameraStats {
  camera: number;
  /**
   * Mean, RMS and largest error length `|observed − projected|`, pixels,
   * over the records with a projection (all zero when there is none).
   */
  stats: ReprojectionStats;
  /**
   * Records without a projection (it diverged), not in `stats`.
   */
  unprojected: number;
  /**
   * The views it has records in.
   */
  views: number;
}
/**
 * Summary statistics for reprojection errors.
 *
 * The percentiles (`median`, `p95`) interpolate linearly between order
 * statistics of the ascending-sorted errors, the default of numpy and
 * pandas: the `q`-quantile of `n` errors lies at fractional rank
 * `q * (n - 1)`, so the median of an even count is the mean of the middle
 * two. They are present when the statistics were computed from the errors
 * ([`from_errors`](Self::from_errors)), and absent when built from
 * aggregates ([`from_summary`](Self::from_summary)).
 */
export interface ReprojectionStats {
  /**
   * Number of points evaluated.
   */
  count: number;
  /**
   * Maximum reprojection error in pixels.
   */
  max: number;
  /**
   * Mean reprojection error in pixels.
   */
  mean: number;
  /**
   * Median reprojection error in pixels (the 50th percentile, linearly
   * interpolated); `None` when built from aggregates. `0.0` for no
   * errors, as the other statistics.
   */
  median?: number | null;
  /**
   * 95th percentile of the reprojection error in pixels, linearly
   * interpolated; `None` when built from aggregates.
   */
  p95?: number | null;
  /**
   * Root mean square error in pixels.
   */
  rms: number;
}
/**
 * One camera of a [`RunSummary`].
 */
export interface CameraSummary {
  /**
   * Its features in the calibration.
   */
  features: number;
  /**
   * The camera's id in the manifest.
   */
  id: string;
  /**
   * The root-mean-square reprojection error of its features, pixels:
   * √(mean of the squared error lengths); `None` without residuals.
   */
  rms_px?: number | null;
  /**
   * The views it saw.
   */
  views: number;
}
/**
 * Cancel the job `id`: a queued job at once, a running one at its next
 * image or solver iteration. A finished job is answered as it is.
 */
export interface CancelJobArgs {
  id: string;
}
/**
 * One camera's statistics in one view.
 */
export interface CellStats {
  camera: number;
  /**
   * Mean, RMS and largest error length `|observed − projected|`, pixels,
   * over the records with a projection (all zero when there is none).
   */
  stats: ReprojectionStats;
  /**
   * Records without a projection (it diverged), not in `stats`.
   */
  unprojected: number;
  view: number;
}
/**
 * A ChArUco board's overlay data.
 */
export interface CharucoDetail {
  /**
   * One per feature, in feature order.
   */
  corners: CornerDetail[];
  /**
   * The markers the board alignment kept.
   */
  markers: MarkerDetail[];
  /**
   * How well the corners are localized.
   */
  quality: CornerQuality;
  /**
   * Marker cells, of the board the detection was built from, whose bits
   * match another marker of the board better than the one the chosen
   * alignment places there: a misread or wrongly placed marker.
   */
  rejected_markers?: RejectedMarker[];
}
/**
 * One labelled board corner, aligned with a [`Feature`].
 */
export interface CornerDetail {
  /**
   * Directions of the two grid lines through the corner, radians in
   * `[0, 2π)`, as the corner stage measured them. Present on a chessboard
   * corner whose corner stage fitted at least one axis; `None` on ChArUco
   * and PuzzleBoard corners, whose detectors do not report them.
   *
   * @minItems 2
   * @maxItems 2
   */
  axes_rad?: [number, number] | null;
  /**
   * The detector's lattice label `[u, v]`: `u` along the board's +x, `v`
   * along its +y. A plain chessboard labels each view from its own
   * top-left corner; a ChArUco board and a PuzzleBoard label it on the
   * board (the PuzzleBoard on its master pattern).
   *
   * @minItems 2
   * @maxItems 2
   */
  grid: [number, number];
  /**
   * The ChArUco corner id, or the PuzzleBoard master corner id; `None`
   * on a chessboard.
   */
  id?: number | null;
  /**
   * The detector's corner score: higher is better, not comparable across
   * detectors.
   */
  score: number;
}
/**
 * One decoded ArUco marker of a ChArUco board.
 */
export interface MarkerDetail {
  /**
   * The board square it sits in, `[column, row]`.
   *
   * @minItems 2
   * @maxItems 2
   */
  cell: [number, number];
  /**
   * Bits read in the image that differ from the marker's code.
   */
  hamming: number;
  /**
   * The marker's dictionary id.
   */
  id: number;
  /**
   * The square's corners in image pixels, in the board's order: the
   * square's top-left on the board (the marker's own top-left) first,
   * then clockwise on the board.
   *
   * @minItems 4
   * @maxItems 4
   */
  quad_px: [[number, number], [number, number], [number, number], [number, number]];
  /**
   * The rotation, in quarter turns, that aligns the bits read in the
   * image with the dictionary's code.
   */
  rotation: number;
  /**
   * The decoding's confidence, in `[0, 1]`.
   */
  score: number;
}
/**
 * How well a detection's corners are localized, as the detector measured
 * it on the corners it returned.
 */
export interface CornerQuality {
  /**
   * Number of corners with enough labelled neighbours (six of the eight)
   * for the leave-one-out check: each such corner's position is
   * predicted from its neighbours.
   */
  local_residual_count: number;
  /**
   * RMS length of the leave-one-out residuals (observed minus predicted
   * position) over those corners, px; `None` when no corner qualifies.
   * The detectors report this aggregate only, not each corner's residual.
   */
  local_residual_rms_px?: number | null;
  /**
   * Median over the corners of their 1σ position uncertainty along the
   * least certain direction, px; `None` when no corner carries a
   * covariance.
   */
  median_sigma_px?: number | null;
  /**
   * The image noise level, gray levels (1σ), the corner covariances were
   * computed with; `None` when no covariance was computed.
   */
  noise_sigma?: number | null;
}
/**
 * A ChArUco marker cell whose bits match another marker of the board
 * better than the one the board places there.
 */
export interface RejectedMarker {
  /**
   * The board marker the cell's bits match best, over every marker of the
   * board and every rotation.
   */
  decoded_id: number;
  /**
   * The marker the board places in this cell under the chosen alignment.
   */
  expected_id: number;
  /**
   * The cell's corners in image pixels, in the board's order (as
   * [`MarkerDetail::quad_px`]).
   *
   * @minItems 4
   * @maxItems 4
   */
  quad_px: [[number, number], [number, number], [number, number], [number, number]];
  /**
   * The matcher's log-likelihood of the cell's bits under the decoded
   * marker: higher is better, comparable within one image.
   */
  score: number;
}
/**
 * A chessboard's overlay data.
 */
export interface ChessboardDetail {
  /**
   * The board's cell size in pixels (median labelled edge length);
   * `None` when no board was detected.
   */
  cell_size_px?: number | null;
  /**
   * One per feature, in feature order.
   */
  corners: CornerDetail[];
  /**
   * How well the corners are localized.
   */
  quality: CornerQuality;
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
 * The configuration changes that make `config` (the workflow `topology`'s
 * config object, as a form edits it) from the library's defaults: the
 * fewest that do, as a project file stores them. The inverse of
 * `resolve_config`; an invalid config is a `config` error.
 */
export interface ConfigPatchArgs {
  config: unknown;
  topology: ProblemKind;
}
/**
 * Why a command, job or resource failed.
 */
export interface StudioError {
  /**
   * The field it is about, as a dotted path into the project file or
   * the manifest (`device.id`, `exclude.images[0]`, `pose_pairing`),
   * when there is one.
   */
  field?: string | null;
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
  /**
   * The values to offer for `field` (`needs_input`).
   */
  suggestions?: string[];
}
/**
 * Make the folder `dir` (created when missing, inside a root) a project of
 * `spec`, written as `calibration-project.toml`. Every path the spec names
 * must lie inside a root; `project_exists` when the folder already holds a
 * project. It becomes the most recent project.
 */
export interface CreateProjectArgs {
  dir: UserPath;
  spec: ProjectSpec;
}
/**
 * A calibration project's definition: its dataset, the changes to its
 * workflow's default configuration, the quality thresholds its runs are
 * judged by and what it leaves out of the dataset.
 *
 * It is the content of `calibration-project.toml` (or of
 * `calibration-project.json`). An unknown field is an error, so a misspelt
 * section such as `[exlude]` never passes silently.
 *
 * ```toml
 * version = 2
 * name = "Bench camera"
 *
 * [device]
 * product = "line scanner"
 * id = "SN-0042"
 *
 * [dataset]
 * manifest = "captures/dataset.toml"
 *
 * [config]
 * solver.max_iters = 80
 *
 * [exclude]
 * images = [{ camera = "cam0", image = "cam0/0003.png" }]
 * ```
 */
export interface ProjectSpec {
  /**
   * Changes to the default configuration of the dataset's workflow.
   */
  config?: ConfigPatch;
  /**
   * The dataset: its manifest, written in the project file or kept in a
   * file of its own.
   */
  dataset: DatasetSource;
  /**
   * What the project is about, for the reader.
   */
  description?: string | null;
  /**
   * The device the project calibrates: the key its runs are grouped by
   * across projects.
   */
  device?: DeviceIdentity | null;
  /**
   * What the runs leave out of the dataset.
   */
  exclude?: Exclusions;
  /**
   * The project's name, for display; without one, the project's folder
   * name is shown.
   */
  name?: string | null;
  /**
   * The thresholds the runs' quality is judged by; `None` for the
   * defaults.
   */
  quality?: QualityThresholds | null;
  /**
   * The project file's format version, [`PROJECT_VERSION`].
   */
  version: number;
}
/**
 * Where a project's dataset manifest is: `manifest` the path of a
 * manifest file of its own, or the manifest itself, written in the project
 * file, with `root` the folder its relative paths resolve against.
 */
export interface DatasetSource {
  /**
   * The manifest file's path (relative to the project folder unless
   * absolute), or the manifest.
   */
  manifest: DatasetManifest;
  /**
   * For a manifest written in the project file: the folder its relative
   * paths resolve against, relative to the project folder unless
   * absolute. Default `.`. Not allowed beside a manifest file.
   */
  root?: string | null;
}
/**
 * The device a project calibrates.
 *
 * Projects of one device, in any folder, form its calibration history:
 * the library groups them by `product` and `id` together.
 */
export interface DeviceIdentity {
  /**
   * The device's own identifier: a serial number, a MAC address. Not
   * empty, without leading or trailing white space, compared as written.
   */
  id: string;
  /**
   * A name for display.
   */
  label?: string | null;
  /**
   * The device type or product line it belongs to; `None` when the id
   * alone names the device.
   */
  product?: string | null;
}
/**
 * The dataset manifest form: the schema of `DatasetSpec` and the manifest
 * a new setup starts from (one chessboard camera, planar intrinsics).
 */
export interface DatasetFormArgs {}
/**
 * Delete run `id` with every file in it. Answers the runs left.
 */
export interface DeleteRunArgs {
  dir: UserPath;
  id: string;
}
/**
 * One run of a project, as a listing shows it.
 */
export interface RunInfo {
  /**
   * The run's folder, absolute.
   */
  dir: string;
  /**
   * The run's id: its folder's name under `runs/`.
   */
  id: string;
  /**
   * Its notes.
   */
  notes: RunNotes;
  /**
   * What its files say.
   */
  state: RunState;
}
/**
 * The record of one run, `run.json`: what went in, a digest of it, and a
 * summary of what came out. Written once with the run and never changed.
 *
 * `timestamp` and `CalibratedRun::duration_ms` are the only clock
 * readings in a run's files; neither enters a digest, the export or the
 * quality report.
 */
export interface RunRecord {
  /**
   * The version of this library that wrote the run.
   */
  library_version: string;
  /**
   * Where the run's export came from.
   */
  source: RunSource;
  /**
   * What came out, for listing without reading the export.
   */
  summary: RunSummary;
  /**
   * When the run was written, seconds since the Unix epoch; for display
   * and ordering only.
   */
  timestamp: number;
  /**
   * The program that made the run, as it named itself (`vcal 1.1.0`).
   */
  tool: string;
  /**
   * The record's format version, [`RUN_RECORD_VERSION`](super::RUN_RECORD_VERSION).
   */
  version: number;
}
/**
 * What a run produced, in brief: enough to list runs without reading
 * their exports, and recomputable from the export.
 */
export interface RunSummary {
  /**
   * Each camera, in camera order.
   */
  cameras: CameraSummary[];
  /**
   * The workflow that produced the export.
   */
  kind: ProblemKind;
  /**
   * The quality report's verdict; `None` when the quality could not be
   * assessed.
   */
  verdict?: Verdict | null;
}
/**
 * A run of the desktop app's earlier format.
 */
export interface UnrecordedRun {
  /**
   * The summary of its export, when the export reads.
   */
  summary?: RunSummary | null;
  /**
   * Views attempted, from its `diagnostics.json`.
   */
  total_views?: number | null;
  /**
   * Views that entered the calibration, from its `diagnostics.json`.
   */
  usable_views?: number | null;
}
/**
 * Detect every image of the project into its detection cache (images it
 * already holds are read from it, unless `redetect`), several at a time;
 * its events count the images done, its output is the
 * `detection_overview` of the result. An image that cannot be read or
 * detected is reported in the overview, not as the job's failure.
 * Cancelling stops between images; what was detected stays cached.
 */
export interface DetectAllArgs {
  /**
   * The project folder.
   */
  dir: UserPath;
  redetect?: boolean;
}
/**
 * `detection_detail` of one image with `detector` in place of the
 * project's detector settings (the manifest's `detector` section; `{}` is
 * the detectors' defaults), to preview an edit: detected now, and nothing
 * is written (neither the project nor its detection cache).
 */
export interface DetectPreviewArgs {
  camera: number;
  detector: DetectorSpec;
  /**
   * The project folder.
   */
  dir: UserPath;
  image: string;
}
/**
 * How far `detect_all` is.
 */
export interface DetectProgress {
  /**
   * The image last done: its camera and path (relative to the data
   * folder).
   */
  camera: number;
  /**
   * Images done.
   */
  done: number;
  image: string;
  /**
   * Images in all.
   */
  total: number;
}
/**
 * What a detector found in one image: the correspondences
 * [`DetectorConfig::detect`](crate::DetectorConfig::detect) returns, and
 * what an overlay needs to judge them.
 */
export interface DetectionDetail {
  /**
   * The features, equal bit for bit to what
   * [`DetectorConfig::detect`](crate::DetectorConfig::detect) returns for
   * the same image and config.
   */
  features: Feature[];
  /**
   * The image-level outcome.
   */
  outcome: DetectionOutcome;
  /**
   * Point candidates the detector or this crate saw and did not keep,
   * each with its reason. Empty when there were none, and on an
   * [`DetectionOutcome::Ambiguous`] image (its dropped points are not
   * listed).
   */
  rejected?: RejectedPoint[];
  /**
   * Target-specific overlay data. Its per-feature vector (`corners` or
   * `rings`) holds one entry per feature, in the order of
   * [`features`](Self::features).
   */
  target: TargetDetail;
}
/**
 * Single 2D-3D feature correspondence in canonical units.
 *
 * `image_xy` is in source-image pixels (pre-ROI-crop if a ROI is
 * declared in the manifest, the converter adjusts as needed).
 * `world_xyz` is in metres, expressed in the calibration target's
 * own frame (target origin at `(0, 0, 0)`, target plane at `z = 0`).
 */
export interface Feature {
  /**
   * The blur the detector measured at this feature, with its covariance
   * ([`FeatureBlur`]). Present when the detector was asked to measure one
   * (`measure_blur` on the chessboard and PuzzleBoard configs) and the
   * estimate at this corner was not declined; `None` otherwise.
   *
   * A measurement only: no calibration reads it.
   */
  blur?: FeatureBlur | null;
  /**
   * Covariance of `image_xy` as `[xx, xy, yy]` in px², in the image
   * frame of `image_xy`. Present only when the detector was asked for it
   * (`ChessCornersConfig::covariance`) and measured one; `None` otherwise.
   *
   * @minItems 3
   * @maxItems 3
   */
  cov_px?: [number, number, number] | null;
  /**
   * Pixel coordinates `[x, y]` (column, row).
   *
   * @minItems 2
   * @maxItems 2
   */
  image_xy: [number, number];
  /**
   * World coordinates `[x, y, z]` in metres in the target frame.
   *
   * @minItems 3
   * @maxItems 3
   */
  world_xyz: [number, number, number];
}
/**
 * The blur a detector measured at one feature: the Gaussian point-spread
 * function (PSF) that best explains the blurred print around the feature,
 * and the covariance of that estimate.
 *
 * **`psf_px`** is the PSF's covariance matrix `Σ = [[xx, xy], [xy, yy]]`
 * in px², stored as `[xx, xy, yy]` (the layout of
 * [`Feature::cov_px`]), in the image frame of `image_xy`. Its principal
 * axes are the directions of least and most blur, and `√(tr Σ / 2) =
 * √((xx + yy) / 2)` is the PSF's width σ in pixels.
 *
 * `Σ` is the blur **as the image shows it, including the pixel's area
 * integration**: for a Gaussian optical PSF of covariance `Σ_o` seen by
 * square pixels, `Σ = Σ_o + I/12`, so a perfectly focused camera has
 * `Σ = I/12`. A model of the optics that predicts `Σ_o` adds `I/12` to its
 * prediction before comparing it with `psf_px`. The constant is not
 * subtracted here: it is what the detector measured, its covariance is
 * unchanged by a constant shift, and on a sharp feature `Σ − I/12` is
 * within the estimate's error of zero and need not be positive definite.
 *
 * **`cov`** is the 3×3 covariance of `(xx, xy, yy)`, px⁴, row-major. How
 * far it can be trusted is the detector's statement. For chessboard and
 * PuzzleBoard corners it allows for residuals correlated between
 * neighbouring pixels, and a combination of `Σ` the print around the
 * corner constrains weakly (with one edge direction, the width along that
 * edge) shows as a large variance.
 *
 * Under a non-Gaussian PSF (a defocus disc, a halo) `Σ` is the covariance
 * of the Gaussian that blurs the print most like the lens does, not the
 * PSF's second moment; it then depends on the cell size and the pattern,
 * so compare values measured on the same board.
 */
export interface FeatureBlur {
  /**
   * The covariance of `psf_px`, px⁴, row-major.
   *
   * @minItems 3
   * @maxItems 3
   */
  cov: [[number, number, number], [number, number, number], [number, number, number]];
  /**
   * The PSF covariance `[xx, xy, yy]`, px², including the pixel's area
   * integration.
   *
   * @minItems 3
   * @maxItems 3
   */
  psf_px: [number, number, number];
}
/**
 * A point candidate the detector or this crate saw and did not keep.
 */
export interface RejectedPoint {
  /**
   * Its lattice label, when it had one.
   *
   * @minItems 2
   * @maxItems 2
   */
  grid?: [number, number] | null;
  /**
   * Its id (a ring's codebook id, a PuzzleBoard master corner id), when
   * it had one.
   */
  id?: number | null;
  /**
   * Its position, image pixels.
   *
   * @minItems 2
   * @maxItems 2
   */
  image_xy: [number, number];
  /**
   * Why it was left out.
   */
  reason: PointRejection;
  /**
   * The target point it would have had, metres, when it had a label.
   *
   * @minItems 3
   * @maxItems 3
   */
  world_xyz?: [number, number, number] | null;
}
/**
 * A PuzzleBoard's overlay data.
 */
export interface PuzzleboardDetail {
  /**
   * One per feature, in feature order.
   */
  corners: CornerDetail[];
  /**
   * The decode summary; `None` when no board was decoded.
   */
  decode?: PuzzleboardDecode | null;
  /**
   * The code bits read on the edges between two kept corners.
   */
  edge_bits?: EdgeBitDetail[];
  /**
   * How well the corners are localized.
   */
  quality: CornerQuality;
}
/**
 * How a PuzzleBoard decoded.
 */
export interface PuzzleboardDecode {
  /**
   * Fraction of the read bits that disagree with the master pattern.
   */
  bit_error_rate: number;
  /**
   * Fraction of the read confidence that lost its majority vote; high
   * means noisy dots or a mislabelled grid.
   */
  dot_dissent_rate: number;
  /**
   * Edges whose bit matched the master pattern after alignment.
   */
  edges_matched: number;
  /**
   * Edges whose bit was read.
   */
  edges_observed: number;
  /**
   * The same, over the distinct master bits the view reads, after the
   * repeated reads of each were majority-voted.
   */
  logical_bit_error_rate: number;
  /**
   * Where the view's local corner `(0, 0)` lies on the master pattern,
   * `[row, column]`.
   *
   * @minItems 2
   * @maxItems 2
   */
  master_origin: [number, number];
}
/**
 * One code bit read on a PuzzleBoard edge.
 */
export interface EdgeBitDetail {
  /**
   * The bit read: the dot on the edge is black (`0`) or white (`1`).
   */
  bit: number;
  /**
   * The read's confidence, in `[0, 1]`.
   */
  confidence: number;
  /**
   * The two corners the edge joins, image pixels. An edge is listed only
   * when both are features.
   *
   * @minItems 2
   * @maxItems 2
   */
  endpoints_px: [[number, number], [number, number]];
  /**
   * Whether the bit equals the master pattern's bit at this edge under
   * the decoded alignment.
   */
  matches_master: boolean;
}
/**
 * A ring grid's overlay data.
 */
export interface RinggridDetail {
  /**
   * Whether the labels are absolute board positions: always for a coded
   * board that was labelled, for a plain one when its origin dots placed
   * the view on the board.
   */
  anchored: boolean;
  /**
   * Whether every marker of the board was detected; `None` when nothing
   * was labelled.
   */
  board_complete?: boolean | null;
  /**
   * The origin dots of a plain board, predicted through the board-to-image
   * homography (no lens model), image pixels. The detector does not
   * report the dots it found, so these are where the board places them.
   * Empty for a coded board and for a view not placed on the board.
   */
  origin_dots_px?: [number, number][];
  /**
   * One per feature, in feature order.
   */
  rings: RingDetail[];
}
/**
 * One ring marker, aligned with a [`Feature`].
 */
export interface RingDetail {
  /**
   * Which estimator produced the feature's position.
   */
  center: RingCenter;
  /**
   * The covariance of the centre as the detector stated it, `[xx, xy,
   * yy]` px²: photometric centres only. The calibration does not read it
   * ([`Feature::cov_px`] stays `None` for ring markers).
   *
   * @minItems 3
   * @maxItems 3
   */
  center_cov_px?: [number, number, number] | null;
  /**
   * The detector's confidence in the centre, in `[0, 1]`.
   */
  confidence: number;
  /**
   * The lattice cell `[u, v]` (axial coordinates on a hex lattice).
   *
   * @minItems 2
   * @maxItems 2
   */
  grid?: [number, number] | null;
  /**
   * The decoded codebook id (coded boards); `None` on a plain board.
   */
  id?: number | null;
  /**
   * The fitted inner ellipse.
   */
  inner?: EllipseDetail | null;
  /**
   * The fitted outer ellipse.
   */
  outer?: EllipseDetail | null;
}
/**
 * An ellipse in image pixels.
 */
export interface EllipseDetail {
  /**
   * The direction of the major axis, radians.
   */
  angle_rad: number;
  /**
   * The centre, px.
   *
   * @minItems 2
   * @maxItems 2
   */
  center_px: [number, number];
  /**
   * The semi-axes `[a, b]`, `a ≥ b`, px.
   *
   * @minItems 2
   * @maxItems 2
   */
  semi_axes_px: [number, number];
}
/**
 * Camera `camera`'s detection of `image` (relative to the data folder, as
 * `project_views` lists it) with what an overlay needs: the features (in
 * the camera's region of interest when it has one), the target's per-kind
 * detail aligned with them, the candidates the detector left out and the
 * image's outcome. Read from the project's detection cache, or detected
 * and stored there (features and detail) when it holds none: the
 * detection a calibration of the project uses. A feature is excluded by
 * its `world_xyz`. `invalid` for an image that is not one of the
 * camera's.
 */
export interface DetectionDetailArgs {
  camera: number;
  /**
   * The project folder.
   */
  dir: UserPath;
  image: string;
}
/**
 * The project's images as its detection cache knows them.
 */
export interface DetectionOverview {
  /**
   * The cameras' ids, in manifest order.
   */
  cameras: string[];
  /**
   * The features a complete detection of the board has (corners of a
   * chessboard or ChArUco board, markers of a ring grid); `null` for a
   * PuzzleBoard, which is made to be seen in part.
   */
  expected?: number | null;
  /**
   * The run whose `left_out` reasons are given: the one asked for, else
   * the project's latest calibrated run; `null` when there is none.
   */
  run?: string | null;
  /**
   * The views, in pairing order.
   */
  views: ViewDetections[];
}
/**
 * One view: each camera's image.
 */
export interface ViewDetections {
  /**
   * One entry per camera; `null` where the camera has no image of the
   * view.
   */
  images: (ImageDetection | null)[];
  /**
   * The view's pairing token (as `project_views` lists it).
   */
  token: string;
}
/**
 * One camera's image of a view.
 */
export interface ImageDetection {
  /**
   * The image, relative to the data folder (absolute outside it),
   * `/`-separated: as `project_views` lists it, and as an exclusion
   * names it.
   */
  image: string;
  /**
   * Why the overview's run left the image out; `null` when it used it
   * (or there is no run).
   */
  left_out?: RejectReason | null;
  state: ImageDetectionState;
}
/**
 * What the project's detection cache holds of every image, view by view
 * and camera by camera (the views of `project_views`): the features found
 * and the detector's outcome, or that the image has no cached detection
 * with the project's detector settings. No image is decoded or detected;
 * each is read to find its cache entry. `left_out` gives why run `run`
 * (default: the latest calibrated run) left an image out.
 */
export interface DetectionOverviewArgs {
  /**
   * The project folder.
   */
  dir: UserPath;
  run?: string | null;
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
 * `path_b` (camera `cam_b`) at `pose`: rectified from the run's export and
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
  run: RunRef;
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
 * A camera's distortion map (see `distortion_map`).
 */
export interface DistortionMap {
  camera: number;
  distortion: MapPlane;
  grid: MapGrid;
  /**
   * Whether the sensor model can add a layer (a tilted or homography
   * sensor); without one `sensor` is zero.
   */
  has_sensor_layer: boolean;
  sensor: MapPlane;
  total: MapPlane;
  /**
   * Nodes with a value (the others are `NaN` in every layer).
   */
  valid_nodes: number;
}
/**
 * A value plane of a map: a `VAM1` blob (the `blob` resource) whose
 * channels, in order, are `channels`. `NaN` marks a node without a value.
 */
export interface MapPlane {
  blob: BlobRef;
  channels: PlaneChannel[];
}
/**
 * One channel of a [`MapPlane`].
 */
export interface PlaneChannel {
  /**
   * `[min, max]` of its finite values; `null` when no node has one.
   *
   * @minItems 2
   * @maxItems 2
   */
  domain?: [number, number] | null;
  /**
   * What the channel holds (`dx`, `dy`, `magnitude`, `rms`, …).
   */
  name: string;
  /**
   * Its unit: `px`, `chi` (a significance), `samples`, `views`.
   */
  unit: string;
}
/**
 * The nodes every plane of a map is sampled at: the centres of
 * `stride × stride` pixel blocks over the camera's image (its region of
 * interest for a tiled camera), row-major from the top-left. Pixel
 * centres are at integer coordinates; block `(c, r)`'s node is at
 * `x = (c·stride + min((c + 1)·stride, width) − 1) / 2` (likewise `y`), so
 * a trailing partial block is centred on its in-image pixels. The
 * `VAM1` header of every plane carries `columns`, `rows` and `stride`.
 */
export interface MapGrid {
  columns: number;
  /**
   * `[width, height]` of the camera's image, pixels.
   *
   * @minItems 2
   * @maxItems 2
   */
  image_size: [number, number];
  rows: number;
  /**
   * The block size, pixels.
   */
  stride: number;
}
/**
 * The lens and sensor displacement of camera `camera` of the run over its
 * image: three layers, each a plane of `dx`, `dy` and `magnitude` in
 * pixels, `actual − reference` at the same ray (where the ray lands minus
 * where it would land without the stage), indexed by the camera's own
 * pixel. `distortion` is the lens alone (the camera without its sensor
 * tilt minus the camera without distortion), `sensor` what the sensor
 * model adds (zero for an untilted sensor), `total` their sum. `NaN` where
 * no ray reaches a node. `columns` (default 64, at most 1024) sets the
 * grid every map of the camera shares.
 */
export interface DistortionMapArgs {
  camera: number;
  columns?: number | null;
  run: RunRef;
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
 * model of the run.
 */
export interface EpipolarOverlayArgs {
  cam_a: number;
  cam_b: number;
  /**
   * @minItems 2
   * @maxItems 2
   */
  point_px: [number, number];
  run: RunRef;
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
  run: RunRef;
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
 * Make the sample dataset `id` (one of `server_info.samples`) available
 * as a project and answer its folder: the files are downloaded into the
 * back end's samples folder (always one of its roots), checked against
 * their SHA-256 and reused later, and the folder becomes a project
 * referencing the sample's manifest (once; an existing project is kept
 * with its runs). A back end built without samples answers `invalid`.
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
 * Add an export calibrated elsewhere to the project as its next run: an
 * export file, a `vcal run -o` folder (its `calibration.json`) or a run
 * folder (its `export.json`). `import_mismatch` for another workflow or
 * camera count than the project's; the export, and every image its image
 * manifest names, must lie inside a root.
 */
export interface ImportExportArgs {
  dir: UserPath;
  export: UserPath;
  notes?: RunNotes;
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
 * What a scan found: the projects, and what it could not read.
 */
export interface LibraryScan {
  /**
   * The folders and projects that could not be read, sorted by path.
   */
  failures: ScanFailure[];
  /**
   * The projects, sorted by folder.
   */
  projects: ProjectEntry[];
}
/**
 * A folder or project a scan could not read. A scan never fails on one.
 */
export interface ScanFailure {
  /**
   * Why it could not be read.
   */
  message: string;
  /**
   * The folder.
   */
  path: string;
}
/**
 * One project a scan found.
 */
export interface ProjectEntry {
  /**
   * The device it calibrates.
   */
  device?: DeviceIdentity | null;
  /**
   * The project folder, absolute.
   */
  dir: string;
  /**
   * The project's name, or its folder's.
   */
  name: string;
  /**
   * Its runs, in listing order.
   */
  runs: RunInfo[];
  /**
   * Its dataset's workflow; `None` when a referenced manifest cannot be
   * read.
   */
  topology?: ProblemKind | null;
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
 * The project's runs (as `open_project` lists them).
 */
export interface ListRunsArgs {
  dir: UserPath;
}
/**
 * A check's measured value and the limit that decided its status.
 */
export interface Measurement {
  /**
   * The limit it was compared to: the one it crossed, or, for a pass, the
   * nearest one.
   */
  threshold: number;
  /**
   * The unit of both.
   */
  unit: Unit;
  /**
   * The measured value; for a check over several items (views, parameter
   * pairs), the worst one.
   */
  value: number;
}
/**
 * Open the project in `dir`: its definition, the resolved dataset and
 * configuration, the digests of its inputs and its runs. A version 1
 * project of the app's earlier format opens as it is (it is migrated when
 * saved). `not_a_project` for a folder without a project file,
 * `forbidden_path` when the project names a data folder or manifest
 * outside the roots. It becomes the most recent project. Opening again is
 * how a client reloads a project: the back end keeps none open.
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
 * The digests of a project's inputs as they stand, for what is known
 * before any image is read: each part of an [`InputDigest`] but the
 * detections, computed as a run computes it. A calibrated run whose
 * digest differs from its project's in one of these parts was made from
 * other inputs than the project now states (a run's own configuration
 * changes count as such a difference).
 */
export interface ProjectDigest {
  /**
   * As [`InputDigest::config`], of the project's configuration.
   */
  config: string;
  /**
   * As [`InputDigest::dataset`].
   */
  dataset: string;
  /**
   * As [`InputDigest::exclusions`].
   */
  exclusions: string;
  /**
   * As [`InputDigest::quality`].
   */
  quality: string;
}
/**
 * An open project: its definition as on disk, what it resolves to, and its
 * runs.
 */
export interface ProjectView {
  /**
   * The workflow's configuration: its defaults with the project's
   * changes, or why they do not make a valid one.
   */
  config: ConfigState;
  /**
   * The folder the manifest's relative paths resolve against.
   */
  data_root: string;
  /**
   * The dataset's manifest, or why it cannot be read: a broken
   * referenced manifest still opens the project.
   */
  dataset: DatasetState;
  /**
   * Whether the manifest is the project file's (and edited with it); a
   * referenced manifest file is never written.
   */
  dataset_editable: boolean;
  /**
   * The project folder.
   */
  dir: string;
  /**
   * The project file's format version on disk: 1 for a project of the
   * app's earlier format until it is saved (then 2).
   */
  file_version: number;
  /**
   * The digests of the project's inputs as they stand (no image read),
   * to compare with each run's: absent when the dataset or the
   * configuration does not resolve.
   */
  input_digest?: ProjectDigest | null;
  /**
   * Its name, or its folder's.
   */
  name: string;
  /**
   * The project file's content revision as read: `save_project` sends it
   * back, and is refused with `changed_on_disk` when the file changed
   * since.
   */
  revision: string;
  /**
   * The runs: numbered runs by number, then others by name.
   */
  runs: RunInfo[];
  /**
   * The project file's definition.
   */
  spec: ProjectSpec;
}
/**
 * The dataset's views and the project's problems.
 */
export interface ProjectViews {
  /**
   * The cameras' ids, in manifest order.
   */
  cameras: string[];
  /**
   * What a run would refuse, each with the field it is about when there
   * is one; empty when the project would run.
   */
  messages: StudioError[];
  /**
   * The views, in pairing order; empty when the images cannot be
   * listed or paired (a message says why).
   */
  views: ViewRow[];
}
/**
 * One view of the dataset.
 */
export interface ViewRow {
  /**
   * Each camera's image, relative to the data folder (absolute outside
   * it), `/`-separated; `null` where a camera has none.
   */
  images: (string | null)[];
  /**
   * Its pairing token (the view's index under index pairing).
   */
  token: string;
}
/**
 * The views the project's images pair into, without reading an image, and
 * what is wrong with the project as a run would find it before detection:
 * of the saved project, or of the draft `spec` when given (nothing is
 * written). Each problem is a message with the field it is about.
 */
export interface ProjectViewsArgs {
  dir: UserPath;
  spec?: ProjectSpec | null;
}
/**
 * The projection uncertainty of camera `camera` on the maps' grid: one
 * plane, `sigma`, the major-axis σ in pixels of where the camera-frame ray
 * through each node lands, from the covariance of the camera's
 * parameters; `NaN` where no ray reaches a node. `unavailable`, with the
 * reason, for a run without a covariance or whose camera parameters were
 * all held fixed. `columns` as `distortion_map`'s.
 */
export interface ProjectionSigmaMapArgs {
  camera: number;
  columns?: number | null;
  run: RunRef;
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
 * The projection uncertainty of `camera` of the run over an image of
 * `image_width` × `image_height` pixels.
 */
export interface ProjectionUncertaintyArgs {
  camera: number;
  image_height: number;
  image_width: number;
  run: RunRef;
}
/**
 * One check of a [`QualityReport`].
 */
export interface QualityCheck {
  /**
   * The camera, for a check made per camera; `None` for one about the
   * whole result.
   */
  camera?: number | null;
  /**
   * What the check examines.
   */
  kind: CheckKind;
  /**
   * The measured value; absent for a check not applicable, and for the
   * observability and convergence checks, which read a status.
   */
  measurement?: Measurement | null;
  /**
   * One sentence on the outcome, or on why the check was not applicable.
   */
  message: string;
  /**
   * Its outcome.
   */
  status: CheckStatus;
  /**
   * What the check names: unobservable or at-bound parameter labels,
   * correlated parameter pairs (`fx/cx`), or outlier views (`view 3`).
   */
  subjects?: string[];
}
/**
 * The verdict on a calibration result and the checks behind it.
 */
export interface QualityReport {
  /**
   * Every check, the result-wide ones first, then each camera's.
   */
  checks: QualityCheck[];
  /**
   * The worst verdict of the assessed checks; `warn` when no check could
   * be assessed, since then nothing vouches for the result.
   */
  verdict: Verdict;
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
 * The run's page, `report.html` (written with every calibration this app
 * runs), served sandboxed: its scripts run without access to the app.
 * `io` when the run has none.
 */
export interface ReportQuery {
  run: RunRef;
}
/**
 * A camera's residual map (see `residual_map`). The residual is
 * `observed − projected`, the sign of a distortion map's layer.
 */
export interface ResidualMap {
  /**
   * The Gaussian kernel's σ the map used, pixels.
   */
  bandwidth_px: number;
  bandwidth_selection: BandwidthSelection;
  camera: number;
  /**
   * Whether the run's board poses come through a rig or robot chain
   * (rig and hand-eye workflows): the residuals then include the chain's
   * pose error, with a smooth pattern per view, besides the camera
   * model's misfit.
   */
  chain_level: boolean;
  grid: MapGrid;
  /**
   * The options as used (`views` sorted): `min_effective_samples` is
   * the support boundary, the `effective_samples` level where the map
   * ends; `min_effective_views` where `std_error` and `significance`
   * end.
   */
  options: ResidualMapOptions;
  /**
   * The records left out as outliers, as indices into the export's
   * `per_feature_residuals.target` (and `residual_records`' `index`).
   */
  outliers: number[];
  plane: MapPlane;
  /**
   * The selected records without a projection, left out.
   */
  unprojected: number;
  /**
   * The records mapped.
   */
  used: number;
  /**
   * The number of views mapped.
   */
  views: number;
}
/**
 * Options of [`residual_map`].
 */
export interface ResidualMapOptions {
  /**
   * The kernel's bandwidth. Default [`Bandwidth::Auto`]: a field's scale
   * differs from capture to capture, and cross-validation over views
   * finds it without fitting each view's pose error as a field.
   */
  bandwidth?: Bandwidth & string;
  /**
   * The fewest effective samples `1/Σ l_k²` a node needs for a value;
   * finite, ≥ 1. Default 8: below it the local fit is mostly one or two
   * records' noise.
   */
  min_effective_samples?: number;
  /**
   * The fewest effective views a node needs for a standard error and a
   * significance; finite, ≥ 2. Default 5: the view-clustered standard
   * error is estimated from that many independent captures, and with
   * fewer it understates the scatter.
   */
  min_effective_views?: number;
  /**
   * Records with `|e|` above this many robust σ (`median |e| / 1.1774`,
   * the Rayleigh median) are left out as outliers; finite, > 0. Default
   * 8, a tail of `exp(−32)` under Gaussian noise: a genuine systematic
   * field at the corners can reach several σ, and a global threshold must
   * not clip the signal the map is meant to show.
   */
  outlier_factor?: number;
  /**
   * The views to map, by their index in the input; `None` (the default)
   * maps all. With one view (or fewer effective views than
   * [`min_effective_views`](Self::min_effective_views)) the mean includes
   * that view's pose error and has no significance.
   */
  views?: number[] | null;
}
/**
 * The systematic part of camera `camera`'s reprojection residuals over its
 * image, the library's local-linear residual map: one plane of eight
 * channels, `mean_dx`, `mean_dy`, `mean_magnitude` (the smoothed mean of
 * `observed − projected`, px), `rms` (px), `std_error` (px, view-clustered),
 * `significance` (χ: above 3.03 is p < 0.01), `effective_samples` and
 * `effective_views`. Records sit at their projection. `NaN` where the
 * support is too thin (`effective_samples` below
 * `options.min_effective_samples`) and, for `std_error` and
 * `significance`, with too few effective views. `options.views` maps a
 * subset of the views (their indices in the export); one view, or too few
 * effective views, gives a mean that includes the views' pose error and no
 * significance. `columns` as `distortion_map`'s. `invalid` for a run
 * without per-feature residuals of the camera.
 */
export interface ResidualMapArgs {
  camera: number;
  columns?: number | null;
  options?: ResidualMapOptions;
  run: RunRef;
}
/**
 * A camera's per-feature residual records, one column per field; entry
 * `i` of every column is one record.
 */
export interface ResidualRecords {
  /**
   * Its feature's index in the view's detection.
   */
  feature: number[];
  /**
   * The record's index in the export's `per_feature_residuals.target`
   * (what `residual_map`'s `outliers` name).
   */
  index: number[];
  /**
   * The detected pixel.
   */
  observed_px: [number, number][];
  /**
   * The model's projection of the target point; `null` where it
   * diverged.
   */
  projected_px: ([number, number] | null)[];
  /**
   * `observed − projected`, pixels; `null` where the projection
   * diverged.
   */
  residual_px: ([number, number] | null)[];
  /**
   * The target point, metres, as the calibration used it.
   */
  target_xyz_m: [number, number, number][];
  /**
   * Its view.
   */
  view: number[];
}
/**
 * Camera `camera`'s per-feature residual records (of one `view`, when
 * given), one column per field, in the export's order: each record's
 * index in the export, view, feature, target point, observed and
 * projected pixel and `observed − projected`. The target point is the one
 * the calibration used: an exclusion names a feature by its detection's
 * `world_xyz` (`detection_detail`), which a rig of plain chessboards may
 * have relabelled.
 */
export interface ResidualRecordsArgs {
  camera: number;
  run: RunRef;
  view?: number | null;
}
/**
 * The reprojection statistics of a run's per-feature records: per camera,
 * per view (cameras pooled) and per (camera, view), each ascending.
 */
export interface ResidualStats {
  cameras: CameraStats[];
  cells: CellStats[];
  views: ViewStats[];
}
/**
 * One view's statistics, over every camera that saw it.
 */
export interface ViewStats {
  /**
   * Mean, RMS and largest error length `|observed − projected|`, pixels,
   * over the records with a projection (all zero when there is none).
   */
  stats: ReprojectionStats;
  /**
   * Records without a projection (it diverged), not in `stats`.
   */
  unprojected: number;
  view: number;
}
/**
 * The run's reprojection statistics from its per-feature residuals (the
 * library's mean, RMS, largest error and count of `|observed −
 * projected|`, px), per camera, per view over every camera, and per
 * camera and view, with the records whose projection diverged.
 */
export interface ResidualStatsArgs {
  run: RunRef;
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
 * A run's export.
 */
export interface RunExport {
  /**
   * The export JSON; the frontend narrows it by its `kind`.
   */
  export: unknown;
  /**
   * The folder the image manifest's frame paths resolve against
   * (absolute, `/`-separated): the project's data folder for a run it
   * calibrated, else the manifest's root as the run folder states it.
   * `null` without an image manifest.
   */
  image_root?: string | null;
}
/**
 * A run's export, with the folder its image manifest's frames resolve
 * against.
 */
export interface RunExportArgs {
  run: RunRef;
}
/**
 * A run's listing entry: its state (with the record of a run this library
 * wrote) and its notes.
 */
export interface RunInfoArgs {
  run: RunRef;
}
/**
 * A run's quality report; `not found` (`io`) for an imported export whose
 * quality could not be assessed.
 */
export interface RunQualityArgs {
  run: RunRef;
}
/**
 * One Quick Start sample.
 */
export interface SampleInfo {
  /**
   * What it holds: the workflow, the views, the target.
   */
  description: string;
  /**
   * The id `fetch_sample` takes.
   */
  id: string;
  /**
   * Its licence (SPDX; `NOASSERTION` when the publisher states none).
   */
  licence: string;
  /**
   * Its workflow.
   */
  topology: ProblemKind;
}
/**
 * A Quick Start sample, ready to open.
 */
export interface SampleProject {
  /**
   * The sample's project folder.
   */
  dir: string;
}
/**
 * Replace the project's definition with `spec` and write it, in its
 * format (a version 1 project becomes version 2). `revision` is the one
 * `open_project` answered: when the file changed on disk since, nothing is
 * written and the answer is `changed_on_disk` (open it again and apply the
 * edit). Every path the spec names must lie inside a root.
 */
export interface SaveProjectArgs {
  dir: UserPath;
  revision: string;
  spec: ProjectSpec;
}
/**
 * Every project under the root folders, with its runs; a folder or
 * project that cannot be read is listed as a failure, never an error.
 */
export interface ScanLibraryArgs {}
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
   * The Quick Start samples `fetch_sample` makes into projects: the
   * library's sample catalogue, empty when this back end was built
   * without them.
   */
  samples: SampleInfo[];
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
 * Replace the notes (label, notes, star) of run `id`; empty notes remove
 * them. Answers the run as listed after.
 */
export interface SetRunNotesArgs {
  dir: UserPath;
  id: string;
  notes: RunNotes;
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
 * Infer a draft dataset manifest from the folder `folder` (the same
 * inference as `vcal project init`): images grouped into cameras, a
 * robot-pose file, a guessed topology. What cannot be determined is left
 * at a placeholder and listed in `unresolved`; a run refuses the manifest
 * until each is set and cleared from its `_unresolved`.
 */
export interface SniffFolderArgs {
  folder: UserPath;
}
/**
 * A sniffed folder.
 */
export interface SniffedFolder {
  /**
   * The draft manifest, its relative paths against the folder.
   */
  spec: DatasetSpec;
  /**
   * The fields the sniffer could not determine, as dotted paths into the
   * manifest (`target`, `topology`, `pose_pairing`, …): the draft's
   * `_unresolved`.
   */
  unresolved: string[];
}
/**
 * Map raw (distorted) pixels of `camera` of the run into its undistorted
 * pixel frame; `null` where back-projection has no finite ray.
 */
export interface UndistortPointsArgs {
  camera: number;
  points_px: [number, number][];
  run: RunRef;
}
/**
 * The image at `path` rendered into `camera`'s undistorted pixel frame of
 * the run's export (a PNG), cropped to `roi` when given.
 */
export interface UndistortedImageQuery {
  camera: number;
  path: UserPath;
  roi?: PixelRect | null;
  run: RunRef;
}
