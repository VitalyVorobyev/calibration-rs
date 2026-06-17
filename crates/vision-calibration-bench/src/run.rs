//! Run a calibration problem end-to-end and return a [`crate::record::BenchRecord`] with
//! captured metrics.
//!
//! Tier-A would replay a frozen fixture through the math/serde path; that lands
//! in a later phase. The Tier-B path here globs a camera's images, detects the
//! board in each, builds the problem `Input`, runs the facade pipeline, and
//! captures both bench-recomputed and self-reported reprojection metrics so any
//! divergence between them is visible in the record.

#[cfg(feature = "tier-b")]
pub use tier_b::{
    diagnose_intrinsics, run_planar_intrinsics, run_rig_extrinsics, run_rig_handeye,
    run_single_cam_handeye,
};

#[cfg(feature = "tier-b")]
pub mod tier_b {
    //! Tier-B entry points that require detection and image-loading capabilities.

    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{Context, Result};
    use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
    use vision_calibration::analysis::{
        RigStageReprojection, planar_intrinsics_report, rig_extrinsics_report,
        rig_handeye_report_with_rig_stage, single_cam_handeye_report,
    };
    use vision_calibration::core::{
        CorrespondenceView, Iso3, NoMeta, RigDataset, RigView, RigViewObs,
    };
    use vision_calibration::optim::{HandEyeMode, RobotPoseMeta};
    use vision_calibration::planar_intrinsics::{
        PlanarIntrinsicsExport, PlanarIntrinsicsProblem, step_init, step_optimize,
    };
    use vision_calibration::rig_extrinsics::{
        RigExtrinsicsExport, RigExtrinsicsProblem, step_intrinsics_init_all,
        step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
    };
    // Rig hand-eye step fns share names with the rig-extrinsics ones, so alias.
    use vision_calibration::rig_handeye::{
        RigHandeyeConfig, RigHandeyeExport, RigHandeyeIntrinsicsManualInit, RigHandeyeProblem,
        step_handeye_init as rh_handeye_init, step_handeye_optimize as rh_handeye_optimize,
        step_intrinsics_init_all_with_seed as rh_intrinsics_init_all_with_seed,
        step_intrinsics_optimize_all as rh_intrinsics_optimize_all, step_rig_init as rh_rig_init,
        step_rig_optimize as rh_rig_optimize,
    };
    use vision_calibration::rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
        run_calibration as run_rig_laserline_device_calibration,
    };
    use vision_calibration::session::CalibrationSession;
    use vision_calibration::single_cam_handeye::{
        HandeyeMeta, SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeInput,
        SingleCamHandeyeProblem, SingleCamHandeyeView, step_handeye_init, step_handeye_optimize,
        step_intrinsics_init, step_intrinsics_optimize,
    };
    use vision_calibration_core::{
        BrownConrady5, Camera, CameraFixMask, DistortionFixMask, FeatureResidualHistogram,
        FxFyCxCySkew, IntrinsicsFixMask, Pinhole, PinholeCamera, PlanarDataset, Pt2,
        ReprojectionStats, ScheimpflugParams, TargetFeatureResidual, View,
        compute_planar_target_residuals,
    };
    use vision_calibration_optim::{
        BackendSolveOptions, LaserlineResidualType, RigHandeyeLaserlineDataset,
        RigHandeyeLaserlineParams, RigHandeyeLaserlinePerCamStats, RigHandeyeLaserlineSolveOptions,
        RigLaserlineView, RobustLoss, ScheimpflugFixMask, ScheimpflugIntrinsicsParams,
        ScheimpflugIntrinsicsSolveOptions, SolveReport, optimize_rig_handeye_laserline,
        optimize_scheimpflug_intrinsics,
    };
    #[cfg(feature = "laser")]
    use vision_metrology::{
        ColAccess, Edge1DConfig, ImageView, LaserExtractConfig, LaserExtractor, ScanAxis,
    };

    use crate::detect::{
        DetectorKind, charuco_params_for, chess_config_for_override, detect_chessboard_view,
        glob_sorted_images, load_image, puzzleboard_params_for,
    };
    #[cfg(feature = "laser")]
    use crate::record::LaserCamStat;
    use crate::record::{
        BENCH_SCHEMA_VERSION, BenchRecord, CalibrationArtifacts, CameraArtifact, Convergence,
        Detection, DetectionStat, DistortionArtifact, Fit, Ident, IntrinsicsArtifact, LaserMetrics,
        ResidualSidecar, RobotCorrectionSummary, ScheimpflugArtifact, StageTiming, Timing,
        TransformArtifact, compact_reproj_report,
    };
    use crate::registry::{BenchEntry, BoardGeometry, CameraLayout, DetectorOverride, ProblemKind};

    /// Lightweight per-dataset stage profile for diagnosing detector/extractor
    /// cost before running a full calibration solve.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct DatasetStageProfile {
        /// Dataset id.
        pub dataset_id: String,
        /// Problem kind.
        pub problem: ProblemKind,
        /// Number of cameras in the registry entry.
        pub camera_count: usize,
        /// Number of robot/target poses found when a pose source exists.
        pub pose_count: Option<usize>,
        /// Maximum images profiled per camera.
        pub max_images_per_camera: Option<usize>,
        /// Target detector profile.
        pub target_detection: TargetDetectionProfile,
        /// Laser extraction profile, when the dataset declares laser data and
        /// the binary was built with the `laser` feature.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub laser_extraction: Option<LaserExtractionProfile>,
    }

    /// Target detector timing/count profile.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct TargetDetectionProfile {
        /// Board layout used to select the detector.
        pub detector: String,
        /// Expected full-board feature count.
        pub expected_features_per_board: usize,
        /// Total wall-clock time in milliseconds.
        pub total_ms: u64,
        /// Per-camera target detection profiles.
        pub per_camera: Vec<ProfileCameraStat>,
    }

    /// Laser extractor timing/count profile.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct LaserExtractionProfile {
        /// Whether the laser feature was enabled in this binary.
        pub enabled: bool,
        /// Total wall-clock time in milliseconds.
        pub total_ms: u64,
        /// Per-camera laser extraction profiles.
        pub per_camera: Vec<ProfileCameraStat>,
    }

    /// Per-camera profile summary.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct ProfileCameraStat {
        /// Camera id.
        pub camera_id: String,
        /// Number of images available to this stage.
        pub images_total: usize,
        /// Number of images actually profiled.
        pub images_profiled: usize,
        /// Number of images with a non-empty detection/extraction.
        pub images_used: usize,
        /// Total features or laser pixels detected.
        pub points_total: usize,
        /// Maximum features or laser pixels detected in one image.
        pub points_max: usize,
        /// Total image load/decode/tile time in milliseconds.
        pub load_ms: u64,
        /// Total detector/extractor time in milliseconds.
        pub stage_ms: u64,
        /// Mean detector/extractor time per profiled image.
        pub mean_stage_ms: f64,
        /// Max detector/extractor time for one image.
        pub max_stage_ms: u64,
    }

    /// Private rtv3d-focused per-camera intrinsics diagnostic report.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsDiagnoseReport {
        /// Dataset id.
        pub dataset_id: String,
        /// Raw all-corner mean gate in pixels.
        pub gate_px: f64,
        /// Default ChESS threshold used by the registry/detector path, when known.
        pub chess_threshold_abs: Option<f32>,
        /// Per-camera intrinsic solve diagnostics.
        pub cameras: Vec<IntrinsicsCameraReport>,
        /// True only when every camera's gate case passes.
        pub pass: bool,
    }

    /// Per-camera diagnostic result.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsCameraReport {
        /// Camera id from the bench registry.
        pub camera_id: String,
        /// ROI-local image size `[width, height]`.
        pub image_size: [u32; 2],
        /// Number of matched images for this camera.
        pub images_total: usize,
        /// Number of images with accepted target detections.
        pub images_used: usize,
        /// Total detected target features.
        pub features_detected: usize,
        /// Whether all observed points lie inside `[0,width] x [0,height]`.
        pub roi_local_coordinates: bool,
        /// Best centered/fixed-principal-point solve used for the quality gate.
        pub gate_case: IntrinsicsCaseReport,
        /// Additional non-gating model-variant diagnostics, only populated when
        /// the centered gate case fails.
        pub diagnostic_cases: Vec<IntrinsicsCaseReport>,
        /// Per-pose residual statistics for the gate case.
        pub per_pose: Vec<IntrinsicsPoseStats>,
        /// Largest residuals for drill-down.
        pub top_outliers: Vec<IntrinsicsOutlier>,
    }

    /// One staged solve result.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsCaseReport {
        /// Case label.
        pub case: String,
        /// Whether this case is allowed to satisfy the official gate.
        pub accepted_for_gate: bool,
        /// Whether the raw all-corner mean is below the gate.
        pub passed_gate: bool,
        /// Seed that initialized the solve.
        pub seed: IntrinsicsSeed,
        /// Refined camera parameters.
        pub params: IntrinsicsParamReport,
        /// Raw residual distribution from per-feature recomputation.
        pub stats: IntrinsicsResidualStats,
        /// Mean reported by `optimize_scheimpflug_intrinsics`.
        pub solver_mean_reproj_error: f64,
        /// Absolute difference between solver-reported and direct raw means.
        pub mean_crosscheck_abs_diff: f64,
        /// Backend solve summary from the final polish stage.
        pub solve_report: SolveReport,
        /// Optional note for non-gating diagnostics.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub note: Option<String>,
    }

    /// Intrinsic seed used for a staged solve.
    #[derive(Debug, Clone, Copy, serde::Serialize)]
    pub struct IntrinsicsSeed {
        /// Initial `fx` and `fy` value.
        pub focal_px: f64,
        /// Initial/fixed principal point x.
        pub cx: f64,
        /// Initial/fixed principal point y.
        pub cy: f64,
        /// Initial Scheimpflug tilt around x.
        pub tau_x: f64,
        /// Initial Scheimpflug tilt around y.
        pub tau_y: f64,
    }

    /// Refined camera parameters used in the diagnostic output.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsParamReport {
        pub fx: f64,
        pub fy: f64,
        pub cx: f64,
        pub cy: f64,
        pub skew: f64,
        pub k1: f64,
        pub k2: f64,
        pub k3: f64,
        pub p1: f64,
        pub p2: f64,
        pub tau_x: f64,
        pub tau_y: f64,
    }

    /// Residual distribution for raw Euclidean reprojection errors.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsResidualStats {
        pub mean: f64,
        pub rms: f64,
        pub median: f64,
        pub p90: f64,
        pub p95: f64,
        pub p99: f64,
        pub max: f64,
        pub count: usize,
        pub count_le_0_4: usize,
        pub count_le_1: usize,
        pub count_gt_2: usize,
        pub count_gt_5: usize,
        /// Diagnostic only; never used to satisfy the quality gate.
        pub trimmed_mean_95: f64,
    }

    /// Residual statistics for one target pose/view.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsPoseStats {
        pub pose: usize,
        pub stats: IntrinsicsResidualStats,
    }

    /// One large-residual feature for drill-down.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsOutlier {
        pub pose: usize,
        pub feature: usize,
        pub target_xyz_m: [f64; 3],
        pub observed_px: [f64; 2],
        pub projected_px: Option<[f64; 2]>,
        pub error_px: f64,
    }

    #[derive(Debug, Clone)]
    struct LaserObservationSet {
        views: Vec<RigLaserlineView>,
        robot_poses: Vec<Iso3>,
        view_indices: Vec<usize>,
        metrics: LaserMetrics,
    }

    #[derive(Debug, Clone)]
    struct DetectedIntrinsicsCamera {
        camera_id: String,
        image_size: [u32; 2],
        images_total: usize,
        images_used: usize,
        features_detected: usize,
        roi_local_coordinates: bool,
        dataset: PlanarDataset,
    }

    #[derive(Debug, Clone)]
    struct StagedIntrinsicsSolve {
        report: IntrinsicsCaseReport,
        params: ScheimpflugIntrinsicsParams,
        residuals: Vec<TargetFeatureResidual>,
    }

    #[derive(Debug, Clone)]
    struct IntrinsicsVariantConfig {
        case: &'static str,
        fix_intrinsics: IntrinsicsFixMask,
        fix_distortion: DistortionFixMask,
        note: Option<String>,
    }

    #[derive(Debug, Clone)]
    struct JointV5Result {
        mean_reproj_error_px: f64,
        per_cam_stats: Vec<RigHandeyeLaserlinePerCamStats>,
        laser_metrics: LaserMetrics,
    }

    /// Elapsed whole-milliseconds since `start` — DRYs the
    /// `start.elapsed().as_millis() as u64` idiom repeated across the runners and
    /// per-stage timers.
    fn ms_since(start: Instant) -> u64 {
        start.elapsed().as_millis() as u64
    }

    /// Run a single-camera planar-intrinsics calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Expects exactly one [`crate::registry::CameraLayout`] and a
    /// [`BoardGeometry`] on the entry. Globs the camera folder under
    /// `entry.data_root`, detects the chessboard in every image (skipping images
    /// with no board), runs `step_init` → `step_optimize`, and assembles the
    /// record. The returned record's `ident` carries placeholder provenance
    /// (`git_sha` / `timestamp_rfc3339` / `features` are filled by the caller).
    pub fn run_planar_intrinsics(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::PlanarIntrinsics,
            "run_planar_intrinsics called for {:?}",
            entry.problem
        );
        let board = entry
            .board
            .as_ref()
            .context("planar_intrinsics entry needs a `board` geometry")?;
        anyhow::ensure!(
            entry.cameras.len() == 1,
            "planar_intrinsics expects exactly one camera, got {}",
            entry.cameras.len()
        );
        let cam = &entry.cameras[0];

        let folder = entry.data_root.join(&cam.folder);
        let paths = glob_sorted_images(&folder, &cam.filename_glob)?;
        anyhow::ensure!(
            !paths.is_empty(),
            "no images matched {}/{} under {}",
            cam.folder,
            cam.filename_glob,
            entry.data_root.display()
        );

        // ── Detection ──────────────────────────────────────────────────────
        progress(
            entry,
            format!("detecting {} images for {}", paths.len(), cam.id),
        );
        let detect_start = Instant::now();
        let mut views = Vec::new();
        let mut images_used = 0usize;
        let mut features_detected = 0usize;
        let mut max_corners_per_image = 0usize;
        let chess_config = chess_config_for_override(entry.detector.as_ref());
        for (image_idx, path) in paths.iter().enumerate() {
            progress_images(entry, "detect", &cam.id, image_idx, paths.len());
            let img = load_image(path)?;
            match detect_chessboard_view(
                &img,
                board.rows,
                board.cols,
                board.cell_size_m,
                board.strict_grid,
                &chess_config,
            ) {
                Ok(Some(view)) => {
                    images_used += 1;
                    features_detected += view.len();
                    max_corners_per_image = max_corners_per_image.max(view.len());
                    views.push(View::without_meta(view));
                }
                Ok(None) => {}
                Err(e) => {
                    // A malformed image is a hard error; a board-not-found is the
                    // Ok(None) arm above.
                    return Err(e.context(format!("detection failed for {}", path.display())));
                }
            }
        }
        let detection_ms = detect_start.elapsed().as_millis() as u64;
        progress(
            entry,
            format!(
                "detection finished: {images_used}/{} images, {features_detected} features, {detection_ms} ms",
                paths.len()
            ),
        );

        anyhow::ensure!(
            views.len() >= 3,
            "need >= 3 detected views for planar intrinsics, got {}",
            views.len()
        );

        // Coverage denominator: the full-board corner count. For loose
        // checkerboard datasets we take the max corners seen in any single
        // image because old manifests can disagree with the auto-discovered
        // grid. Strict-grid datasets use the declared board size because every
        // accepted detection has already been validated against it.
        let features_per_board = max_corners_per_image.max(board_feature_count(board));
        let features_expected = images_used * features_per_board;
        let coverage_pct = if features_expected > 0 {
            100.0 * features_detected as f64 / features_expected as f64
        } else {
            0.0
        };

        let detection = Detection {
            per_camera: vec![DetectionStat {
                camera_id: cam.id.clone(),
                images_total: paths.len(),
                images_used,
                features_detected,
                features_expected,
                coverage_pct,
                detect_ms: detection_ms,
            }],
            total_detected: features_detected,
            total_expected: features_expected,
        };

        // ── Calibration ────────────────────────────────────────────────────
        // Keep a copy of the detected views for the multi-level reproj report
        // (the dataset is moved into the session below).
        let views_for_report = views.clone();
        let dataset = PlanarDataset::new(views).context("failed to build PlanarDataset")?;
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(dataset).context("set_input failed")?;

        progress(entry, "initializing planar intrinsics");
        let init_start = Instant::now();
        let init_ok = step_init(&mut session, None).is_ok();
        let init_ms = init_start.elapsed().as_millis() as u64;
        anyhow::ensure!(init_ok, "step_init failed for {}", entry.id);

        progress(entry, "optimizing planar intrinsics");
        let opt_start = Instant::now();
        let _ = step_optimize(&mut session, None).context("step_optimize failed")?;
        let optimize_ms = opt_start.elapsed().as_millis() as u64;

        let export = session.export().context("session.export failed")?;

        // ── Fit metrics ────────────────────────────────────────────────────
        // `overall` is recomputed from every per-feature error_px (the spec's
        // bench-computed aggregate); `reported_*` mirror what the export itself
        // claimed so a divergence surfaces.
        let errors: Vec<f64> = export
            .per_feature_residuals
            .target
            .iter()
            .filter_map(|r| r.error_px)
            .collect();
        let overall = ReprojectionStats::from_errors(&errors);

        // Single-camera problem: per-camera == overall, and the per-camera
        // histogram comes straight from the export.
        let per_camera = vec![overall];
        let per_camera_hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .clone()
            .unwrap_or_else(|| vec![FeatureResidualHistogram::default()]);

        let fit = Fit {
            overall,
            per_camera,
            per_camera_hist,
            reported_mean_reproj_px: export.mean_reproj_error,
            reported_per_cam_px: export.per_cam_reproj_errors.clone(),
        };

        let convergence = Convergence {
            init_ok,
            // The planar solver does not expose an explicit convergence flag; a
            // successful `step_optimize` that produced an export is treated as
            // converged. `report` carries the underlying cost/iteration detail.
            converged: true,
            report: export.report.clone(),
        };

        let total_ms = init_ms
            .saturating_add(optimize_ms)
            .saturating_add(detection_ms);
        let timing = Timing {
            init_ms,
            optimize_ms,
            total_ms,
            detection_ms,
            stages: None,
        };

        // Hierarchical reprojection report. Planar intrinsics has a single
        // (Intrinsic) level whose mean equals the headline; the report adds the
        // per-camera / per-view breakdown.
        let (reproj_report, residual_sidecar) = split_reproj_report(
            entry,
            planar_intrinsics_report(&export, &views_for_report).ok(),
        );

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "planar_intrinsics"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser: None,
            robot_corrections: None,
            artifacts: Some(artifacts_from_planar(&cam.id, &export)),
            delta_to_prior: None,
            timing,
            reproj_report,
            residual_sidecar,
        })
    }

    /// Run a multi-camera rig-extrinsics calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Mirrors `crates/vision-calibration/examples/stereo_session.rs` (and its
    /// ChArUco sibling) to the step: it globs each camera folder, detects the
    /// board per image with the entry's detector (chessboard or ChArUco —
    /// selected from [`BoardGeometry::layout`]), pairs views *by filename suffix*
    /// (the shared portion after a per-camera prefix), and runs
    /// `step_intrinsics_init_all` → `step_intrinsics_optimize_all` →
    /// `step_rig_init` → `step_rig_optimize`. The bench-computed `Fit::overall`
    /// is recomputed from the export's per-camera per-feature residuals; the
    /// `reported_*` fields mirror the export so any divergence is visible.
    ///
    /// Pairing matches the examples' semantics: a view is kept when *either*
    /// camera detected the board (the missing camera contributes an empty
    /// `CorrespondenceView`), and at least 3 usable views per camera are
    /// required. Image filenames are paired on the longest common suffix after
    /// stripping each camera's leading non-shared prefix, which reproduces the
    /// examples' `Im_L_N`/`Im_R_N` and `Cam1_…`/`Cam2_…` pairing without
    /// hard-coding those prefixes.
    pub fn run_rig_extrinsics(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::RigExtrinsics,
            "run_rig_extrinsics called for {:?}",
            entry.problem
        );
        let board = entry
            .board
            .as_ref()
            .context("rig_extrinsics entry needs a `board` geometry")?;
        anyhow::ensure!(
            entry.cameras.len() == 2,
            "rig_extrinsics currently expects exactly two cameras, got {}",
            entry.cameras.len()
        );
        let detector = detector_for(board, entry.detector.as_ref())?;

        // ── Detection (per camera, then pair by filename suffix) ────────────
        progress(
            entry,
            format!("detecting target in {} cameras", entry.cameras.len()),
        );
        let detect_start = Instant::now();
        let mut per_cam_views: Vec<Vec<(String, Option<CorrespondenceView>)>> = Vec::new();
        let mut detect_stats: Vec<DetectionStat> = Vec::new();
        let mut max_corners_per_image = 0usize;
        for cam in &entry.cameras {
            let folder = entry.data_root.join(&cam.folder);
            let paths = glob_sorted_images(&folder, &cam.filename_glob)?;
            anyhow::ensure!(
                !paths.is_empty(),
                "no images matched {}/{} under {}",
                cam.folder,
                cam.filename_glob,
                entry.data_root.display()
            );
            let mut views: Vec<(String, Option<CorrespondenceView>)> = Vec::new();
            let mut images_used = 0usize;
            let mut features_detected = 0usize;
            progress(
                entry,
                format!("detecting {} images for {}", paths.len(), cam.id),
            );
            for (image_idx, path) in paths.iter().enumerate() {
                progress_images(entry, "detect", &cam.id, image_idx, paths.len());
                let key = pairing_key(path);
                let img = load_image(path)?;
                match detector.detect(&img) {
                    Ok(Some(view)) => {
                        images_used += 1;
                        features_detected += view.len();
                        max_corners_per_image = max_corners_per_image.max(view.len());
                        views.push((key, Some(view)));
                    }
                    Ok(None) => views.push((key, None)),
                    Err(e) => {
                        return Err(e.context(format!("detection failed for {}", path.display())));
                    }
                }
            }
            detect_stats.push(DetectionStat {
                camera_id: cam.id.clone(),
                images_total: paths.len(),
                images_used,
                features_detected,
                // `features_expected` is filled once the full-board corner count
                // is known (after all cameras are processed).
                features_expected: 0,
                coverage_pct: 0.0,
                detect_ms: 0,
            });
            per_cam_views.push(views);
        }
        let detection_ms = detect_start.elapsed().as_millis() as u64;
        progress(
            entry,
            format!("target detection finished in {detection_ms} ms"),
        );

        // Pair views: collect the suffix keys present in *every* camera, in the
        // first camera's order. (`stereo_session` pairs the index intersection;
        // we generalise to the suffix intersection so charuco's descriptive
        // filenames pair too.)
        let mut common: BTreeSet<String> =
            per_cam_views[0].iter().map(|(k, _)| k.clone()).collect();
        for cam_views in &per_cam_views[1..] {
            let here: BTreeSet<String> = cam_views.iter().map(|(k, _)| k.clone()).collect();
            common = common.intersection(&here).cloned().collect();
        }
        anyhow::ensure!(
            !common.is_empty(),
            "no shared image keys across the {} cameras",
            entry.cameras.len()
        );

        let mut rig_views: Vec<RigView<NoMeta>> = Vec::new();
        let mut usable_per_cam = vec![0usize; entry.cameras.len()];
        for key in &common {
            // `RigViewObs::cameras` is `Vec<Option<CorrespondenceView>>`: a
            // camera that did not detect the board in this view contributes
            // `None` (matching the rig dataset's per-camera-optional shape).
            let mut cams: Vec<Option<CorrespondenceView>> = Vec::with_capacity(entry.cameras.len());
            let mut any = false;
            for (ci, cam_views) in per_cam_views.iter().enumerate() {
                let view = cam_views
                    .iter()
                    .find(|(k, _)| k == key)
                    .and_then(|(_, v)| v.clone());
                if view.is_some() {
                    usable_per_cam[ci] += 1;
                    any = true;
                }
                cams.push(view);
            }
            if any {
                rig_views.push(RigView {
                    meta: NoMeta,
                    obs: RigViewObs { cameras: cams },
                });
            }
        }
        for (ci, &u) in usable_per_cam.iter().enumerate() {
            anyhow::ensure!(
                u >= 3,
                "need >= 3 usable views for camera {} ({}), got {}",
                ci,
                entry.cameras[ci].id,
                u
            );
        }

        let features_per_board = max_corners_per_image.max(board_feature_count(board));

        // ── Calibration ────────────────────────────────────────────────────
        let input = RigDataset::new(rig_views, entry.cameras.len())
            .map_err(|e| anyhow::anyhow!("failed to build RigDataset: {e}"))?;
        // Keep a copy for the multi-level reproj report (the dataset is moved
        // into the session below).
        let dataset_for_report = input.clone();
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(input).context("set_input failed")?;

        progress(entry, "initializing per-camera intrinsics");
        let init_start = Instant::now();
        let init_ok = step_intrinsics_init_all(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init_all failed for {}", entry.id);
        let init_ms = init_start.elapsed().as_millis() as u64;

        progress(entry, "optimizing intrinsics and rig extrinsics");
        let s = Instant::now();
        let _ = step_intrinsics_optimize_all(&mut session, None)
            .context("step_intrinsics_optimize_all failed")?;
        let intrinsics_optimize_ms = ms_since(s);
        let s = Instant::now();
        let _ = step_rig_init(&mut session).context("step_rig_init failed")?;
        let rig_init_ms = ms_since(s);
        let s = Instant::now();
        let _ = step_rig_optimize(&mut session, None).context("step_rig_optimize failed")?;
        let rig_optimize_ms = ms_since(s);
        let optimize_ms = intrinsics_optimize_ms
            .saturating_add(rig_init_ms)
            .saturating_add(rig_optimize_ms);
        let stages = StageTiming {
            intrinsics_init_ms: Some(init_ms),
            intrinsics_optimize_ms: Some(intrinsics_optimize_ms),
            rig_init_ms: Some(rig_init_ms),
            rig_optimize_ms: Some(rig_optimize_ms),
            ..StageTiming::default()
        };

        let export = session.export().context("session.export failed")?;

        // ── Fit metrics ────────────────────────────────────────────────────
        // `PerFeatureResiduals` keeps one flat `target` list with a `camera`
        // index per `TargetFeatureResidual`. The bench splits it on
        // `camera` for per-camera stats and pools all finite `error_px` for
        // `overall` (the spec's bench-computed aggregate). `reported_*` mirror
        // the export's own `mean_reproj_error` / `per_cam_reproj_errors` so any
        // divergence between bench-computed and self-reported numbers surfaces.
        let n_cam = entry.cameras.len();
        let mut per_cam_errs: Vec<Vec<f64>> = vec![Vec::new(); n_cam];
        let mut all_errors: Vec<f64> = Vec::new();
        for r in &export.per_feature_residuals.target {
            if let Some(e) = r.error_px {
                all_errors.push(e);
                let ci = r.camera;
                if ci < n_cam {
                    per_cam_errs[ci].push(e);
                }
            }
        }
        let overall = ReprojectionStats::from_errors(&all_errors);
        let per_camera: Vec<ReprojectionStats> = per_cam_errs
            .iter()
            .map(|errs| ReprojectionStats::from_errors(errs))
            .collect();
        let per_camera_hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .clone()
            .unwrap_or_else(|| vec![FeatureResidualHistogram::default(); n_cam]);

        let fit = Fit {
            overall,
            per_camera,
            per_camera_hist,
            reported_mean_reproj_px: export.mean_reproj_error,
            reported_per_cam_px: export.per_cam_reproj_errors.clone(),
        };

        // Finalize detection coverage now that the full-board count is known.
        let mut total_detected = 0usize;
        let mut total_expected = 0usize;
        for stat in &mut detect_stats {
            let expected = stat.images_used * features_per_board;
            stat.features_expected = expected;
            stat.coverage_pct = if expected > 0 {
                100.0 * stat.features_detected as f64 / expected as f64
            } else {
                0.0
            };
            stat.detect_ms = detection_ms;
            total_detected += stat.features_detected;
            total_expected += expected;
        }
        let detection = Detection {
            per_camera: detect_stats,
            total_detected,
            total_expected,
        };

        let convergence = Convergence {
            init_ok,
            converged: true,
            report: synth_report(export.mean_reproj_error),
        };
        let total_ms = init_ms
            .saturating_add(optimize_ms)
            .saturating_add(detection_ms);
        let timing = Timing {
            init_ms,
            optimize_ms,
            total_ms,
            detection_ms,
            stages: Some(stages),
        };

        // Hierarchical reprojection report: Intrinsic floor (free per-(cam,view)
        // PnP pose) + RigExtrinsic level (shared board pose through the rig).
        let (reproj_report, residual_sidecar) = split_reproj_report(
            entry,
            rig_extrinsics_report(&export, &dataset_for_report).ok(),
        );

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "rig_extrinsics"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser: None,
            robot_corrections: None,
            artifacts: Some(artifacts_from_rig_extrinsics(&entry.cameras, &export)),
            delta_to_prior: None,
            timing,
            reproj_report,
            residual_sidecar,
        })
    }

    /// Run a single-camera hand-eye calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Supports two pose-pairing modes, dispatched on
    /// [`crate::registry::PoseSource::format`]:
    ///
    /// - **`"rowmajor4x4"` (legacy)**: mirrors
    ///   `crates/vision-calibration/examples/handeye_session.rs`. Loads the
    ///   robot poses (row-major 4×4 per line) and pairs image `{i+1:02}.png`
    ///   with pose row `i`. Board is detected with a chessboard detector.
    ///
    /// - **`"snap_list_json"`**: loads `poses.json` as a JSON list of snap
    ///   objects (`{target_image, tcp2base, …}`), iterates in file order, and
    ///   loads the named `target_image` from the camera folder. Board detector
    ///   is chosen from `board.layout` (ChArUco or chessboard).
    ///
    /// In both modes, views with no board are skipped, runs
    /// `step_intrinsics_init` → `step_intrinsics_optimize` →
    /// `step_handeye_init` → `step_handeye_optimize`, then exports.
    pub fn run_single_cam_handeye(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::SingleCamHandeye,
            "run_single_cam_handeye called for {:?}",
            entry.problem
        );
        let board = entry
            .board
            .as_ref()
            .context("single_cam_handeye entry needs a `board` geometry")?;
        anyhow::ensure!(
            entry.cameras.len() == 1,
            "single_cam_handeye expects exactly one camera, got {}",
            entry.cameras.len()
        );
        let cam = &entry.cameras[0];
        let pose_src = entry
            .robot_poses
            .as_ref()
            .context("single_cam_handeye entry needs `robot_poses`")?;

        let trans_scale = match pose_src.units.as_deref() {
            Some("mm") => 1.0e-3,
            Some("m") | None => 1.0,
            Some(other) => anyhow::bail!("unsupported pose units '{other}' (use 'mm' or 'm')"),
        };

        // ── Detection (pairing mode depends on pose format) ─────────────────
        let detect_start = Instant::now();
        let mut views: Vec<SingleCamHandeyeView> = Vec::new();
        let mut features_detected = 0usize;
        let mut max_corners_per_image = 0usize;
        let mut images_used = 0usize;
        let total_poses: usize;

        if pose_src.format == "snap_list_json" {
            // ── snap_list_json path: named image files ───────────────────────
            let mut snap_pairs =
                load_snap_list_json(&entry.data_root.join(&pose_src.path), trans_scale)?;
            for (_target_image, robot_pose) in &mut snap_pairs {
                normalize_robot_pose_convention(robot_pose, &pose_src.convention)?;
            }
            total_poses = snap_pairs.len();
            let detector = detector_for(board, entry.detector.as_ref())?;
            let cam_root = if cam.folder.is_empty() {
                entry.data_root.clone()
            } else {
                entry.data_root.join(&cam.folder)
            };
            progress(
                entry,
                format!("detecting {} snap images for {}", snap_pairs.len(), cam.id),
            );
            for (image_idx, (target_image, robot_pose)) in snap_pairs.iter().enumerate() {
                progress_images(entry, "detect", &cam.id, image_idx, snap_pairs.len());
                let img_path = cam_root.join(target_image);
                anyhow::ensure!(
                    img_path.exists(),
                    "missing image {} (from snap_list_json)",
                    img_path.display()
                );
                let img = load_image(&img_path)?;
                let img = apply_tile(&img, cam.tile);
                if let Some(view) = detector.detect(&img)? {
                    images_used += 1;
                    features_detected += view.len();
                    max_corners_per_image = max_corners_per_image.max(view.len());
                    views.push(SingleCamHandeyeView {
                        obs: view,
                        meta: HandeyeMeta {
                            base_se3_gripper: *robot_pose,
                        },
                    });
                }
            }
        } else {
            // ── Legacy rowmajor4x4 path (kuka_1) ────────────────────────────
            let robot_poses =
                load_robot_poses_for(pose_src, &entry.data_root.join(&pose_src.path))?;
            total_poses = robot_poses.len();
            let square_size_m = board.cell_size_m;
            let folder = entry.data_root.join(&cam.folder);
            let chess_config = chess_config_for_override(entry.detector.as_ref());
            progress(
                entry,
                format!("detecting {} pose images for {}", robot_poses.len(), cam.id),
            );
            for (idx, robot_pose) in robot_poses.iter().enumerate() {
                progress_images(entry, "detect", &cam.id, idx, robot_poses.len());
                let image_index = idx + 1;
                let img_path = folder.join(format!("{image_index:02}.png"));
                anyhow::ensure!(
                    img_path.exists(),
                    "missing image {} for pose row {}",
                    img_path.display(),
                    image_index
                );
                let img = load_image(&img_path)?;
                match detect_chessboard_view(
                    &img,
                    board.rows,
                    board.cols,
                    square_size_m,
                    board.strict_grid,
                    &chess_config,
                ) {
                    Ok(Some(view)) => {
                        images_used += 1;
                        features_detected += view.len();
                        max_corners_per_image = max_corners_per_image.max(view.len());
                        views.push(SingleCamHandeyeView {
                            obs: view,
                            meta: HandeyeMeta {
                                base_se3_gripper: *robot_pose,
                            },
                        });
                    }
                    Ok(None) => {}
                    Err(e) => {
                        return Err(
                            e.context(format!("detection failed for {}", img_path.display()))
                        );
                    }
                }
            }
        }
        let detection_ms = detect_start.elapsed().as_millis() as u64;
        progress(
            entry,
            format!("target detection finished in {detection_ms} ms"),
        );
        anyhow::ensure!(
            views.len() >= 3,
            "need >= 3 detected views for single-cam hand-eye, got {}",
            views.len()
        );

        let features_per_board = max_corners_per_image.max(board_feature_count(board));
        let features_expected = images_used * features_per_board;
        let coverage_pct = if features_expected > 0 {
            100.0 * features_detected as f64 / features_expected as f64
        } else {
            0.0
        };
        let detection = Detection {
            per_camera: vec![DetectionStat {
                camera_id: cam.id.clone(),
                images_total: total_poses,
                images_used,
                features_detected,
                features_expected,
                coverage_pct,
                detect_ms: detection_ms,
            }],
            total_detected: features_detected,
            total_expected: features_expected,
        };

        // ── Calibration ────────────────────────────────────────────────────
        // Keep a copy of the detected views for the multi-level reproj report
        // (the input is moved into the session below).
        let views_for_report = views.clone();
        let input =
            SingleCamHandeyeInput::new(views).map_err(|e| anyhow::anyhow!("set_input: {e}"))?;
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        let mut config = SingleCamHandeyeConfig::default();
        if let Some(overrides) = &entry.single_cam_handeye {
            overrides.apply_to(&mut config);
        }
        let robot_rot_sigma = config.robot_rot_sigma;
        let robot_trans_sigma = config.robot_trans_sigma;
        session
            .set_config(config)
            .context("set single_cam_handeye config failed")?;
        session.set_input(input).context("set_input failed")?;

        progress(entry, "initializing single-camera intrinsics");
        let init_start = Instant::now();
        let init_ok = step_intrinsics_init(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init failed for {}", entry.id);
        let init_ms = init_start.elapsed().as_millis() as u64;

        progress(entry, "optimizing intrinsics and hand-eye");
        let opt_start = Instant::now();
        let _ = step_intrinsics_optimize(&mut session, None)
            .context("step_intrinsics_optimize failed")?;
        let _ = step_handeye_init(&mut session, None).context("step_handeye_init failed")?;
        let _ =
            step_handeye_optimize(&mut session, None).context("step_handeye_optimize failed")?;
        let optimize_ms = opt_start.elapsed().as_millis() as u64;

        let export = session.export().context("session.export failed")?;
        // Reference `HandEyeMode` so the import is load-bearing (the export
        // carries the mode; the bench does not branch on it).
        debug_assert!(matches!(
            export.handeye_mode,
            HandEyeMode::EyeInHand | HandEyeMode::EyeToHand
        ));

        // Single camera: `overall == per_camera[0]`, pooled from the export's
        // per-feature `error_px` (the spec's bench-computed aggregate); the
        // per-camera histogram comes straight from the export. `reported_*`
        // mirror the export's self-reported numbers so divergence surfaces.
        let errors: Vec<f64> = export
            .per_feature_residuals
            .target
            .iter()
            .filter_map(|r| r.error_px)
            .collect();
        let overall = ReprojectionStats::from_errors(&errors);
        let per_camera_hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .clone()
            .unwrap_or_else(|| vec![FeatureResidualHistogram::default()]);

        let fit = Fit {
            overall,
            per_camera: vec![overall],
            per_camera_hist,
            reported_mean_reproj_px: export.mean_reproj_error,
            reported_per_cam_px: export.per_cam_reproj_errors.clone(),
        };
        let convergence = Convergence {
            init_ok,
            converged: true,
            report: synth_report(export.mean_reproj_error),
        };
        let total_ms = init_ms
            .saturating_add(optimize_ms)
            .saturating_add(detection_ms);
        let timing = Timing {
            init_ms,
            optimize_ms,
            total_ms,
            detection_ms,
            stages: None,
        };

        // Hierarchical reprojection report: Intrinsic floor (free per-view PnP
        // pose) + HandEye level (board pose through robot + hand-eye chain). The
        // gap between the two localizes the kuka-style "why is it 1.2 px" error.
        let (reproj_report, residual_sidecar) = split_reproj_report(
            entry,
            single_cam_handeye_report(&export, &views_for_report).ok(),
        );
        let robot_corrections = export.robot_deltas.as_deref().and_then(|deltas| {
            RobotCorrectionSummary::from_deltas_with_priors(
                deltas,
                Some(robot_rot_sigma),
                Some(robot_trans_sigma),
            )
        });

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "single_cam_handeye"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser: None,
            robot_corrections,
            artifacts: Some(artifacts_from_single_cam(&cam.id, &export)),
            delta_to_prior: None,
            timing,
            reproj_report,
            residual_sidecar,
        })
    }

    /// Run a multi-camera rig hand-eye calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Generalizes the rig path to N cameras mounted on a robot end-effector
    /// (DS8: two JAI BB-500 + a Creative Senz3D, pinhole). Unlike
    /// [`run_rig_extrinsics`], cameras are paired **by sorted image index**, not
    /// filename suffix — DS8's cameras use incompatible naming
    /// (`imageN.png` vs `colorFrame_0_NNNNN.png`) but are frame-synchronized, so
    /// view `i` is `(cam0[i], cam1[i], …, pose i)`; a camera that did not detect
    /// the board in that frame contributes `None`. Robot poses are loaded per the
    /// entry's [`crate::registry::PoseSource`] (`format` + `units`). Runs the full
    /// six-step rig hand-eye pipeline (intrinsics → rig → hand-eye) with default
    /// options (so robot-pose refinement is on), then exports.
    pub fn run_rig_handeye(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::RigHandeye,
            "run_rig_handeye called for {:?}",
            entry.problem
        );
        let board = entry
            .board
            .as_ref()
            .context("rig_handeye entry needs a `board` geometry")?;
        anyhow::ensure!(
            entry.cameras.len() >= 2,
            "rig_handeye expects >= 2 cameras, got {}",
            entry.cameras.len()
        );
        let pose_src = entry
            .robot_poses
            .as_ref()
            .context("rig_handeye entry needs `robot_poses`")?;
        let detector = detector_for(board, entry.detector.as_ref())?;
        let robot_poses = load_robot_poses_for(pose_src, &entry.data_root.join(&pose_src.path))?;

        // ── Detection (per camera, image-order) ─────────────────────────────
        progress(
            entry,
            format!("detecting target in {} cameras", entry.cameras.len()),
        );
        let detect_start = Instant::now();
        let mut per_cam_dets: Vec<Vec<Option<CorrespondenceView>>> = Vec::new();
        let mut per_cam_paths: Vec<Vec<PathBuf>> = Vec::new();
        let mut detect_stats: Vec<DetectionStat> = Vec::new();
        let mut max_corners_per_image = 0usize;
        for cam in &entry.cameras {
            let folder = entry.data_root.join(&cam.folder);
            let paths = glob_sorted_images(&folder, &cam.filename_glob)?;
            anyhow::ensure!(
                !paths.is_empty(),
                "no images matched {}/{} under {}",
                cam.folder,
                cam.filename_glob,
                entry.data_root.display()
            );
            let mut dets: Vec<Option<CorrespondenceView>> = Vec::with_capacity(paths.len());
            let mut images_used = 0usize;
            let mut features_detected = 0usize;
            progress(
                entry,
                format!("detecting {} images for {}", paths.len(), cam.id),
            );
            for (image_idx, path) in paths.iter().enumerate() {
                progress_images(entry, "detect", &cam.id, image_idx, paths.len());
                let img = load_image(path)?;
                // Tiled rigs (e.g. 6 cameras side-by-side in one frame) crop each
                // camera's ROI from the shared image; `tile` is `None` for the
                // one-file-per-camera rigs (DS8), leaving the image untouched.
                let img = apply_tile(&img, cam.tile);
                match detector.detect(&img) {
                    Ok(Some(view)) => {
                        images_used += 1;
                        features_detected += view.len();
                        max_corners_per_image = max_corners_per_image.max(view.len());
                        dets.push(Some(view));
                    }
                    Ok(None) => dets.push(None),
                    Err(e) => {
                        return Err(e.context(format!("detection failed for {}", path.display())));
                    }
                }
            }
            detect_stats.push(DetectionStat {
                camera_id: cam.id.clone(),
                images_total: paths.len(),
                images_used,
                features_detected,
                features_expected: 0, // filled once full-board count is known
                coverage_pct: 0.0,
                detect_ms: 0,
            });
            per_cam_dets.push(dets);
            per_cam_paths.push(paths);
        }
        let detection_ms = detect_start.elapsed().as_millis() as u64;
        progress(
            entry,
            format!("target detection finished in {detection_ms} ms"),
        );

        // Pair by index: view i = (cam0[i], …, camN[i], pose i), bounded by the
        // shortest camera and the pose count.
        let n_cam = entry.cameras.len();
        let n_views = per_cam_dets
            .iter()
            .map(|d| d.len())
            .min()
            .unwrap_or(0)
            .min(robot_poses.len());
        anyhow::ensure!(
            n_views >= 3,
            "need >= 3 paired views (min over cameras/poses), got {n_views}"
        );

        let mut rig_views: Vec<RigView<RobotPoseMeta>> = Vec::with_capacity(n_views);
        let mut usable_per_cam = vec![0usize; n_cam];
        for v in 0..n_views {
            let mut cams: Vec<Option<CorrespondenceView>> = Vec::with_capacity(n_cam);
            let mut any = false;
            for (ci, dets) in per_cam_dets.iter().enumerate() {
                let view = dets[v].clone();
                if view.is_some() {
                    usable_per_cam[ci] += 1;
                    any = true;
                }
                cams.push(view);
            }
            if any {
                rig_views.push(RigView {
                    meta: RobotPoseMeta {
                        base_se3_gripper: robot_poses[v],
                    },
                    obs: RigViewObs { cameras: cams },
                });
            }
        }
        for (ci, &u) in usable_per_cam.iter().enumerate() {
            anyhow::ensure!(
                u >= 3,
                "need >= 3 usable views for camera {} ({}), got {} \
                 (low-res/color cameras may detect the board poorly)",
                ci,
                entry.cameras[ci].id,
                u
            );
        }

        let features_per_board = max_corners_per_image.max(board_feature_count(board));
        let laser_observations = extract_laser_observations(
            entry,
            &per_cam_dets,
            &per_cam_paths,
            &robot_poses,
            n_views,
        )?;
        let mut laser = laser_observations.as_ref().map(|obs| obs.metrics.clone());

        // ── Calibration ─────────────────────────────────────────────────────
        let mut config = RigHandeyeConfig::default();
        if let Some(overrides) = &entry.rig_handeye {
            overrides.apply_to(&mut config);
        }
        if let Some(seed) = &entry.seed {
            let manual: RigHandeyeIntrinsicsManualInit = serde_json::from_value(seed.0.clone())
                .with_context(|| {
                    format!("failed to parse rig hand-eye manual seed for {}", entry.id)
                })?;
            config.intrinsics.manual_init = Some(manual);
        }
        if config.intrinsics.manual_init.is_none() && entry.id == "rtv3d" {
            config.intrinsics.manual_init = Some(rtv3d_manual_intrinsics_seed(entry));
            config.intrinsics.fix_tangential = true;
        }
        let robot_rot_sigma = config.handeye_ba.robot_rot_sigma;
        let robot_trans_sigma = config.handeye_ba.robot_trans_sigma;
        let manual_intrinsics = config.intrinsics.manual_init.clone().unwrap_or_default();
        let input = RigDataset::new(rig_views, n_cam)
            .map_err(|e| anyhow::anyhow!("failed to build RigDataset: {e}"))?;
        let dataset_for_report = input.clone();
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session
            .set_config(config)
            .context("set rig_handeye config failed")?;
        session.set_input(input).context("set_input failed")?;

        progress(entry, "initializing per-camera intrinsics");
        let init_start = Instant::now();
        let init_ok =
            rh_intrinsics_init_all_with_seed(&mut session, manual_intrinsics, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init_all failed for {}", entry.id);
        let init_ms = init_start.elapsed().as_millis() as u64;

        progress(entry, "optimizing intrinsics, rig, and hand-eye");
        let s = Instant::now();
        let _ = rh_intrinsics_optimize_all(&mut session, None)
            .context("step_intrinsics_optimize_all failed")?;
        let intrinsics_optimize_ms = ms_since(s);
        let s = Instant::now();
        let _ = rh_rig_init(&mut session).context("step_rig_init failed")?;
        let rig_init_ms = ms_since(s);
        let s = Instant::now();
        let rig_stage_output =
            rh_rig_optimize(&mut session, None).context("step_rig_optimize failed")?;
        let rig_optimize_ms = ms_since(s);
        let s = Instant::now();
        let _ = rh_handeye_init(&mut session, None).context("step_handeye_init failed")?;
        let handeye_init_ms = ms_since(s);
        let s = Instant::now();
        let _ = rh_handeye_optimize(&mut session, None).context("step_handeye_optimize failed")?;
        let handeye_optimize_ms = ms_since(s);
        let optimize_ms = intrinsics_optimize_ms
            .saturating_add(rig_init_ms)
            .saturating_add(rig_optimize_ms)
            .saturating_add(handeye_init_ms)
            .saturating_add(handeye_optimize_ms);
        let stages = StageTiming {
            intrinsics_init_ms: Some(init_ms),
            intrinsics_optimize_ms: Some(intrinsics_optimize_ms),
            rig_init_ms: Some(rig_init_ms),
            rig_optimize_ms: Some(rig_optimize_ms),
            handeye_init_ms: Some(handeye_init_ms),
            handeye_optimize_ms: Some(handeye_optimize_ms),
        };

        let export = session.export().context("session.export failed")?;
        let joint_result = run_rig_handeye_laserline_v5(
            entry,
            &export,
            laser_observations,
            robot_rot_sigma,
            robot_trans_sigma,
        )?;
        if let Some(joint) = &joint_result {
            laser = Some(joint.laser_metrics.clone());
        }

        // ── Fit metrics (per-camera split of the export's per-feature errors) ─
        let mut per_cam_errs: Vec<Vec<f64>> = vec![Vec::new(); n_cam];
        let mut all_errors: Vec<f64> = Vec::new();
        for r in &export.per_feature_residuals.target {
            if let Some(e) = r.error_px {
                all_errors.push(e);
                if r.camera < n_cam {
                    per_cam_errs[r.camera].push(e);
                }
            }
        }
        let overall = ReprojectionStats::from_errors(&all_errors);
        let per_camera: Vec<ReprojectionStats> = per_cam_errs
            .iter()
            .map(|errs| ReprojectionStats::from_errors(errs))
            .collect();
        let per_camera_hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .clone()
            .unwrap_or_else(|| vec![FeatureResidualHistogram::default(); n_cam]);
        let mut fit = Fit {
            overall,
            per_camera,
            per_camera_hist,
            reported_mean_reproj_px: export.mean_reproj_error,
            reported_per_cam_px: export.per_cam_reproj_errors.clone(),
        };
        if let Some(joint) = &joint_result {
            fit.overall = reproj_stats_from_joint(&joint.per_cam_stats);
            fit.per_camera = joint
                .per_cam_stats
                .iter()
                .map(reproj_stats_from_joint_cam)
                .collect();
            fit.reported_mean_reproj_px = joint.mean_reproj_error_px;
            fit.reported_per_cam_px = joint
                .per_cam_stats
                .iter()
                .map(|s| s.mean_reproj_error_px)
                .collect();
        }

        // Finalize detection coverage now that the full-board count is known.
        let mut total_detected = 0usize;
        let mut total_expected = 0usize;
        for stat in &mut detect_stats {
            let expected = stat.images_used * features_per_board;
            stat.features_expected = expected;
            stat.coverage_pct = if expected > 0 {
                100.0 * stat.features_detected as f64 / expected as f64
            } else {
                0.0
            };
            stat.detect_ms = detection_ms;
            total_detected += stat.features_detected;
            total_expected += expected;
        }
        let detection = Detection {
            per_camera: detect_stats,
            total_detected,
            total_expected,
        };

        let convergence = Convergence {
            init_ok,
            converged: true,
            report: synth_report(
                joint_result
                    .as_ref()
                    .map(|j| j.mean_reproj_error_px)
                    .unwrap_or(export.mean_reproj_error),
            ),
        };
        let laser_ms = laser.as_ref().map(|m| m.extract_ms).unwrap_or(0);
        let total_ms = init_ms
            .saturating_add(optimize_ms)
            .saturating_add(detection_ms)
            .saturating_add(laser_ms);
        let timing = Timing {
            init_ms,
            optimize_ms,
            total_ms,
            detection_ms,
            stages: Some(stages),
        };

        // Hierarchical report: Intrinsic floor (free per-(cam,view) PnP) +
        // HandEye level (board pose through the robot + hand-eye chain).
        let rig_stage = RigStageReprojection {
            cam_se3_rig: rig_stage_output.cam_se3_rig,
            rig_se3_target: rig_stage_output.rig_se3_target,
        };
        let (reproj_report, residual_sidecar) = split_reproj_report(
            entry,
            rig_handeye_report_with_rig_stage(&export, &dataset_for_report, &rig_stage).ok(),
        );
        let robot_corrections = export.robot_deltas.as_deref().and_then(|deltas| {
            RobotCorrectionSummary::from_deltas_with_priors(
                deltas,
                Some(robot_rot_sigma),
                Some(robot_trans_sigma),
            )
        });

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "rig_handeye"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser,
            robot_corrections,
            artifacts: Some(artifacts_from_rig_handeye(&entry.cameras, &export)),
            delta_to_prior: None,
            timing,
            reproj_report,
            residual_sidecar,
        })
    }

    /// Diagnose per-camera Scheimpflug intrinsics only: detect the calibration
    /// target, run staged multistart single-camera solves, and report raw
    /// all-corner reprojection distributions.
    pub fn diagnose_intrinsics(entry: &BenchEntry) -> Result<IntrinsicsDiagnoseReport> {
        const GATE_PX: f64 = 0.4;

        let board = entry
            .board
            .as_ref()
            .context("diagnose intrinsics needs a `board` geometry")?;
        anyhow::ensure!(
            !entry.cameras.is_empty(),
            "diagnose intrinsics needs at least one camera"
        );
        let detector_override = intrinsics_detector_override(entry);
        let detector = detector_for(board, Some(&detector_override))?;

        progress(
            entry,
            format!(
                "diagnosing Scheimpflug intrinsics for {} cameras",
                entry.cameras.len()
            ),
        );

        let mut camera_reports = Vec::with_capacity(entry.cameras.len());
        for cam in &entry.cameras {
            progress(entry, format!("detecting intrinsic dataset for {}", cam.id));
            let detected = detect_intrinsics_camera(entry, board, cam, &detector)?;
            anyhow::ensure!(
                detected.dataset.num_views() >= 3,
                "need >= 3 detected views for {}, got {}",
                cam.id,
                detected.dataset.num_views()
            );
            let center = (
                f64::from(detected.image_size[0]) * 0.5,
                f64::from(detected.image_size[1]) * 0.5,
            );
            let seeds = default_intrinsics_seeds(center.0, center.1);
            progress(
                entry,
                format!(
                    "{}: solving {} centered Scheimpflug seeds",
                    cam.id,
                    seeds.len()
                ),
            );
            let gate_solve = solve_centered_multistart(&detected.dataset, &seeds, GATE_PX)
                .with_context(|| format!("centered multistart failed for {}", cam.id))?;
            let mut diagnostic_cases = Vec::new();
            if !gate_solve.report.passed_gate {
                progress(
                    entry,
                    format!(
                        "{}: gate mean {:.4} px; running diagnostic model variants",
                        cam.id, gate_solve.report.stats.mean
                    ),
                );
                diagnostic_cases = solve_intrinsics_diagnostic_variants(
                    &detected.dataset,
                    &gate_solve,
                    center,
                    GATE_PX,
                )?
                .into_iter()
                .map(|solve| solve.report)
                .collect();
            }
            let top_outliers = top_intrinsics_outliers(&gate_solve.residuals, 25);
            let per_pose = intrinsics_pose_stats(&gate_solve.residuals);
            camera_reports.push(IntrinsicsCameraReport {
                camera_id: detected.camera_id,
                image_size: detected.image_size,
                images_total: detected.images_total,
                images_used: detected.images_used,
                features_detected: detected.features_detected,
                roi_local_coordinates: detected.roi_local_coordinates,
                gate_case: gate_solve.report,
                diagnostic_cases,
                per_pose,
                top_outliers,
            });
        }

        let pass = camera_reports
            .iter()
            .all(|camera| camera.gate_case.accepted_for_gate && camera.gate_case.passed_gate);
        Ok(IntrinsicsDiagnoseReport {
            dataset_id: entry.id.clone(),
            gate_px: GATE_PX,
            chess_threshold_abs: chess_threshold_abs(Some(&detector_override)),
            cameras: camera_reports,
            pass,
        })
    }

    fn intrinsics_detector_override(entry: &BenchEntry) -> DetectorOverride {
        let mut detector = entry.detector.clone().unwrap_or_default();
        let needs_default = detector
            .chess_corners
            .as_ref()
            .map(|chess| chess.threshold_mode.is_none() && chess.threshold_value.is_none())
            .unwrap_or(true);
        if needs_default {
            detector.chess_corners = Some(crate::registry::ChessCornersDetectorOverride {
                threshold_mode: Some(crate::registry::BenchChessThresholdMode::Absolute),
                threshold_value: Some(30.0),
            });
        }
        detector
    }

    fn detect_intrinsics_camera(
        entry: &BenchEntry,
        board: &BoardGeometry,
        cam: &CameraLayout,
        detector: &DetectorKind,
    ) -> Result<DetectedIntrinsicsCamera> {
        let folder = entry.data_root.join(&cam.folder);
        let paths = glob_sorted_images(&folder, &cam.filename_glob)?;
        anyhow::ensure!(
            !paths.is_empty(),
            "no images matched {}/{} under {}",
            cam.folder,
            cam.filename_glob,
            entry.data_root.display()
        );

        let mut views = Vec::new();
        let mut image_size = None;
        let mut images_used = 0usize;
        let mut features_detected = 0usize;
        for (image_idx, path) in paths.iter().enumerate() {
            progress_images(entry, "detect-intrinsics", &cam.id, image_idx, paths.len());
            let img = load_image(path)?;
            let img = apply_tile(&img, cam.tile);
            let size = [img.width(), img.height()];
            if let Some(prev) = image_size {
                anyhow::ensure!(
                    prev == size,
                    "{} image size changed from {:?} to {:?}",
                    cam.id,
                    prev,
                    size
                );
            }
            image_size = Some(size);
            match detector.detect(&img) {
                Ok(Some(view)) => {
                    images_used += 1;
                    features_detected += view.len();
                    views.push(View::without_meta(view));
                }
                Ok(None) => {}
                Err(e) => return Err(e.context(format!("detection failed for {}", path.display()))),
            }
        }

        let image_size = image_size
            .or_else(|| cam.tile.map(|[_x, _y, w, h]| [w, h]))
            .or(cam.expected_size)
            .context("could not determine image size")?;
        let roi_local_coordinates = views_are_roi_local(&views, image_size);
        let dataset = PlanarDataset::new(views).context("failed to build PlanarDataset")?;
        let features_per_board = board_feature_count(board);
        progress(
            entry,
            format!(
                "{} detections: {images_used}/{} images, {features_detected} features (full board {})",
                cam.id,
                paths.len(),
                features_per_board
            ),
        );
        Ok(DetectedIntrinsicsCamera {
            camera_id: cam.id.clone(),
            image_size,
            images_total: paths.len(),
            images_used,
            features_detected,
            roi_local_coordinates,
            dataset,
        })
    }

    fn views_are_roi_local<M>(views: &[View<M>], image_size: [u32; 2]) -> bool {
        let w = f64::from(image_size[0]);
        let h = f64::from(image_size[1]);
        views.iter().all(|view| {
            view.obs.points_2d.iter().all(|p| {
                p.x.is_finite()
                    && p.y.is_finite()
                    && p.x >= 0.0
                    && p.x <= w
                    && p.y >= 0.0
                    && p.y <= h
            })
        })
    }

    fn default_intrinsics_seeds(cx: f64, cy: f64) -> Vec<IntrinsicsSeed> {
        let mut seeds = Vec::new();
        for focal_px in [1600.0, 1800.0, 2000.0, 2200.0] {
            for tau_x in [-0.12, -0.08, -0.04, 0.0, 0.04] {
                for tau_y in [-0.02, 0.0, 0.02] {
                    seeds.push(IntrinsicsSeed {
                        focal_px,
                        cx,
                        cy,
                        tau_x,
                        tau_y,
                    });
                }
            }
        }
        seeds
    }

    fn solve_centered_multistart(
        dataset: &PlanarDataset,
        seeds: &[IntrinsicsSeed],
        gate_px: f64,
    ) -> Result<StagedIntrinsicsSolve> {
        let mut best: Option<StagedIntrinsicsSolve> = None;
        let mut failures = Vec::new();
        for &seed in seeds {
            match solve_centered_seed(dataset, seed, gate_px) {
                Ok(solve) => {
                    let replace = best
                        .as_ref()
                        .map(|b| solve.report.stats.mean < b.report.stats.mean)
                        .unwrap_or(true);
                    if replace {
                        best = Some(solve);
                    }
                }
                Err(err) => failures.push(format!(
                    "f={:.0} tau=({:.3},{:.3}): {err}",
                    seed.focal_px, seed.tau_x, seed.tau_y
                )),
            }
        }
        best.with_context(|| {
            let sample = failures.into_iter().take(5).collect::<Vec<_>>().join("; ");
            format!("all centered intrinsic seeds failed: {sample}")
        })
    }

    fn solve_centered_seed(
        dataset: &PlanarDataset,
        seed: IntrinsicsSeed,
        gate_px: f64,
    ) -> Result<StagedIntrinsicsSolve> {
        let initial = initial_intrinsics_params(dataset, seed)?;
        let stage_a = optimize_scheimpflug_intrinsics(
            dataset,
            &initial,
            ScheimpflugIntrinsicsSolveOptions {
                robust_loss: RobustLoss::None,
                fix_intrinsics: IntrinsicsFixMask::all_fixed(),
                fix_distortion: DistortionFixMask::all_fixed(),
                fix_scheimpflug: ScheimpflugFixMask {
                    tilt_x: true,
                    tilt_y: true,
                },
                fix_poses: Vec::new(),
                bounds: None,
            },
            BackendSolveOptions {
                max_iters: 30,
                verbosity: 0,
                ..Default::default()
            },
        )
        .context("stage A pose-only Scheimpflug polish failed")?;

        let centered_opts = ScheimpflugIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 1.0 },
            fix_intrinsics: IntrinsicsFixMask {
                fx: false,
                fy: false,
                cx: true,
                cy: true,
            },
            fix_distortion: DistortionFixMask {
                k1: false,
                k2: false,
                k3: true,
                p1: true,
                p2: true,
            },
            fix_scheimpflug: ScheimpflugFixMask {
                tilt_x: false,
                tilt_y: false,
            },
            fix_poses: Vec::new(),
            bounds: None,
        };
        let stage_b = optimize_scheimpflug_intrinsics(
            dataset,
            &stage_a.params,
            centered_opts.clone(),
            BackendSolveOptions {
                max_iters: 100,
                verbosity: 0,
                ..Default::default()
            },
        )
        .context("stage B robust centered Scheimpflug solve failed")?;

        let stage_c = optimize_scheimpflug_intrinsics(
            dataset,
            &stage_b.params,
            ScheimpflugIntrinsicsSolveOptions {
                robust_loss: RobustLoss::None,
                ..centered_opts
            },
            BackendSolveOptions {
                max_iters: 50,
                verbosity: 0,
                ..Default::default()
            },
        )
        .context("stage C L2 centered Scheimpflug polish failed")?;

        staged_case_from_estimate(
            dataset,
            seed,
            "centered_p1p2_k3_fixed",
            true,
            gate_px,
            stage_c,
            None,
        )
    }

    fn initial_intrinsics_params(
        dataset: &PlanarDataset,
        seed: IntrinsicsSeed,
    ) -> Result<ScheimpflugIntrinsicsParams> {
        let intrinsics = FxFyCxCySkew {
            fx: seed.focal_px,
            fy: seed.focal_px,
            cx: seed.cx,
            cy: seed.cy,
            skew: 0.0,
        };
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let sensor = ScheimpflugParams {
            tilt_x: seed.tau_x,
            tilt_y: seed.tau_y,
        };
        let poses = initial_poses_for_dataset(dataset, &intrinsics)?;
        ScheimpflugIntrinsicsParams::new(intrinsics, distortion, sensor, poses)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    fn initial_poses_for_dataset(
        dataset: &PlanarDataset,
        k: &FxFyCxCySkew<f64>,
    ) -> Result<Vec<Iso3>> {
        let mut poses = Vec::with_capacity(dataset.views.len());
        for (view_idx, view) in dataset.views.iter().enumerate() {
            let obs_3d = &view.obs.points_3d;
            let obs_2d = &view.obs.points_2d;
            anyhow::ensure!(
                obs_3d.len() >= 4 && obs_3d.len() == obs_2d.len(),
                "view {view_idx} has invalid correspondence count"
            );
            let max_z = obs_3d.iter().fold(0.0_f64, |m, p| m.max(p.z.abs()));
            let pose = if max_z < 1e-6 {
                let world2d: Vec<Pt2> = obs_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
                let h =
                    vision_calibration::linear::homography::HomographySolver::dlt(&world2d, obs_2d)
                        .with_context(|| format!("homography seed failed for view {view_idx}"))?;
                vision_calibration::linear::planar_pose::estimate_planar_pose_from_h(
                    &k.k_matrix(),
                    &h,
                )
                .with_context(|| format!("planar pose seed failed for view {view_idx}"))?
            } else {
                vision_calibration::linear::pnp::PnpSolver::epnp(obs_3d, obs_2d, k)
                    .with_context(|| format!("EPnP seed failed for view {view_idx}"))?
            };
            poses.push(pose);
        }
        Ok(poses)
    }

    fn solve_intrinsics_diagnostic_variants(
        dataset: &PlanarDataset,
        gate_solve: &StagedIntrinsicsSolve,
        center: (f64, f64),
        gate_px: f64,
    ) -> Result<Vec<StagedIntrinsicsSolve>> {
        let seed = gate_solve.report.seed;
        let mut out = Vec::new();
        let cxy = solve_intrinsics_variant(
            dataset,
            seed,
            &gate_solve.params,
            gate_px,
            IntrinsicsVariantConfig {
                case: "diagnostic_cxcy_free",
                fix_intrinsics: IntrinsicsFixMask {
                    fx: false,
                    fy: false,
                    cx: false,
                    cy: false,
                },
                fix_distortion: DistortionFixMask {
                    k1: false,
                    k2: false,
                    k3: true,
                    p1: true,
                    p2: true,
                },
                note: Some(
                    "diagnostic only: cx/cy released; gate still uses centered solution"
                        .to_string(),
                ),
            },
        )?;
        let cxy_dx = (cxy.report.params.cx - center.0).abs();
        let cxy_dy = (cxy.report.params.cy - center.1).abs();
        let cxy = if cxy_dx > 40.0 || cxy_dy > 40.0 {
            let mut cxy = cxy;
            cxy.report.note = Some(format!(
                "diagnostic only: principal point moved {:.2}/{:.2} px from center",
                cxy_dx, cxy_dy
            ));
            cxy
        } else {
            cxy
        };
        out.push(cxy);

        out.push(solve_intrinsics_variant(
            dataset,
            seed,
            &gate_solve.params,
            gate_px,
            IntrinsicsVariantConfig {
                case: "diagnostic_k3_free",
                fix_intrinsics: IntrinsicsFixMask {
                    fx: false,
                    fy: false,
                    cx: true,
                    cy: true,
                },
                fix_distortion: DistortionFixMask {
                    k1: false,
                    k2: false,
                    k3: false,
                    p1: true,
                    p2: true,
                },
                note: Some("diagnostic only: k3 released".to_string()),
            },
        )?);

        out.push(solve_intrinsics_variant(
            dataset,
            seed,
            &gate_solve.params,
            gate_px,
            IntrinsicsVariantConfig {
                case: "diagnostic_p1p2_free",
                fix_intrinsics: IntrinsicsFixMask {
                    fx: false,
                    fy: false,
                    cx: true,
                    cy: true,
                },
                fix_distortion: DistortionFixMask {
                    k1: false,
                    k2: false,
                    k3: true,
                    p1: false,
                    p2: false,
                },
                note: Some(
                    "diagnostic only: p1/p2 released to test missing model terms".to_string(),
                ),
            },
        )?);

        Ok(out)
    }

    fn solve_intrinsics_variant(
        dataset: &PlanarDataset,
        seed: IntrinsicsSeed,
        initial: &ScheimpflugIntrinsicsParams,
        gate_px: f64,
        config: IntrinsicsVariantConfig,
    ) -> Result<StagedIntrinsicsSolve> {
        let opts = ScheimpflugIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 1.0 },
            fix_intrinsics: config.fix_intrinsics,
            fix_distortion: config.fix_distortion,
            fix_scheimpflug: ScheimpflugFixMask {
                tilt_x: false,
                tilt_y: false,
            },
            fix_poses: Vec::new(),
            bounds: None,
        };
        let robust = optimize_scheimpflug_intrinsics(
            dataset,
            initial,
            opts.clone(),
            BackendSolveOptions {
                max_iters: 100,
                verbosity: 0,
                ..Default::default()
            },
        )
        .with_context(|| format!("{} robust solve failed", config.case))?;
        let polished = optimize_scheimpflug_intrinsics(
            dataset,
            &robust.params,
            ScheimpflugIntrinsicsSolveOptions {
                robust_loss: RobustLoss::None,
                ..opts
            },
            BackendSolveOptions {
                max_iters: 50,
                verbosity: 0,
                ..Default::default()
            },
        )
        .with_context(|| format!("{} L2 polish failed", config.case))?;
        staged_case_from_estimate(
            dataset,
            seed,
            config.case,
            false,
            gate_px,
            polished,
            config.note,
        )
    }

    fn staged_case_from_estimate(
        dataset: &PlanarDataset,
        seed: IntrinsicsSeed,
        case: &str,
        accepted_for_gate: bool,
        gate_px: f64,
        estimate: vision_calibration_optim::ScheimpflugIntrinsicsEstimate,
        note: Option<String>,
    ) -> Result<StagedIntrinsicsSolve> {
        let camera = Camera::new(
            Pinhole,
            estimate.params.distortion,
            estimate.params.sensor.compile(),
            estimate.params.intrinsics,
        );
        let residuals =
            compute_planar_target_residuals(&camera, dataset, &estimate.params.camera_se3_target)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        let stats = intrinsics_stats_from_residuals(&residuals);
        let diff = (stats.mean - estimate.mean_reproj_error).abs();
        anyhow::ensure!(
            diff <= 1e-9,
            "{case}: direct raw mean {:.12} != solver mean {:.12} (diff {diff:.3e})",
            stats.mean,
            estimate.mean_reproj_error
        );
        let params = IntrinsicsParamReport {
            fx: estimate.params.intrinsics.fx,
            fy: estimate.params.intrinsics.fy,
            cx: estimate.params.intrinsics.cx,
            cy: estimate.params.intrinsics.cy,
            skew: estimate.params.intrinsics.skew,
            k1: estimate.params.distortion.k1,
            k2: estimate.params.distortion.k2,
            k3: estimate.params.distortion.k3,
            p1: estimate.params.distortion.p1,
            p2: estimate.params.distortion.p2,
            tau_x: estimate.params.sensor.tilt_x,
            tau_y: estimate.params.sensor.tilt_y,
        };
        let report = IntrinsicsCaseReport {
            case: case.to_string(),
            accepted_for_gate,
            passed_gate: stats.mean < gate_px,
            seed,
            params,
            stats,
            solver_mean_reproj_error: estimate.mean_reproj_error,
            mean_crosscheck_abs_diff: diff,
            solve_report: estimate.report,
            note,
        };
        Ok(StagedIntrinsicsSolve {
            report,
            params: estimate.params,
            residuals,
        })
    }

    fn intrinsics_stats_from_residuals(
        residuals: &[TargetFeatureResidual],
    ) -> IntrinsicsResidualStats {
        let mut errors: Vec<f64> = residuals
            .iter()
            .filter_map(|r| r.error_px)
            .filter(|e| e.is_finite())
            .collect();
        intrinsics_stats_from_errors(&mut errors)
    }

    fn intrinsics_stats_from_errors(errors: &mut [f64]) -> IntrinsicsResidualStats {
        if errors.is_empty() {
            return IntrinsicsResidualStats {
                mean: 0.0,
                rms: 0.0,
                median: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                max: 0.0,
                count: 0,
                count_le_0_4: 0,
                count_le_1: 0,
                count_gt_2: 0,
                count_gt_5: 0,
                trimmed_mean_95: 0.0,
            };
        }
        let sum: f64 = errors.iter().sum();
        let sum_sq: f64 = errors.iter().map(|e| e * e).sum();
        let count_le_0_4 = errors.iter().filter(|e| **e <= 0.4).count();
        let count_le_1 = errors.iter().filter(|e| **e <= 1.0).count();
        let count_gt_2 = errors.iter().filter(|e| **e > 2.0).count();
        let count_gt_5 = errors.iter().filter(|e| **e > 5.0).count();
        errors.sort_by(|a, b| a.total_cmp(b));
        let count = errors.len();
        let trimmed_count = ((count as f64) * 0.95).ceil().clamp(1.0, count as f64) as usize;
        let trimmed_mean_95 = errors[..trimmed_count].iter().sum::<f64>() / trimmed_count as f64;
        IntrinsicsResidualStats {
            mean: sum / count as f64,
            rms: (sum_sq / count as f64).sqrt(),
            median: percentile_sorted(errors, 0.5),
            p90: percentile_sorted(errors, 0.90),
            p95: percentile_sorted(errors, 0.95),
            p99: percentile_sorted(errors, 0.99),
            max: errors[count - 1],
            count,
            count_le_0_4,
            count_le_1,
            count_gt_2,
            count_gt_5,
            trimmed_mean_95,
        }
    }

    fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
        debug_assert!(!sorted.is_empty());
        if sorted.len() == 1 {
            return sorted[0];
        }
        let rank = q * (sorted.len() as f64 - 1.0);
        let lo = rank.floor() as usize;
        let hi = rank.ceil() as usize;
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }

    fn top_intrinsics_outliers(
        residuals: &[TargetFeatureResidual],
        limit: usize,
    ) -> Vec<IntrinsicsOutlier> {
        let mut finite: Vec<_> = residuals
            .iter()
            .filter_map(|r| r.error_px.filter(|e| e.is_finite()).map(|e| (e, r)))
            .collect();
        finite.sort_by(|(a, _), (b, _)| b.total_cmp(a));
        finite
            .into_iter()
            .take(limit)
            .map(|(error_px, r)| IntrinsicsOutlier {
                pose: r.pose,
                feature: r.feature,
                target_xyz_m: r.target_xyz_m,
                observed_px: r.observed_px,
                projected_px: r.projected_px,
                error_px,
            })
            .collect()
    }

    fn intrinsics_pose_stats(residuals: &[TargetFeatureResidual]) -> Vec<IntrinsicsPoseStats> {
        let pose_count = residuals
            .iter()
            .map(|residual| residual.pose)
            .max()
            .map(|idx| idx + 1)
            .unwrap_or(0);
        let mut per_pose = vec![Vec::new(); pose_count];
        for residual in residuals {
            if let Some(error) = residual.error_px.filter(|e| e.is_finite())
                && let Some(slot) = per_pose.get_mut(residual.pose)
            {
                slot.push(error);
            }
        }
        per_pose
            .into_iter()
            .enumerate()
            .map(|(pose, mut errors)| IntrinsicsPoseStats {
                pose,
                stats: intrinsics_stats_from_errors(&mut errors),
            })
            .collect()
    }

    fn chess_threshold_abs(detector: Option<&DetectorOverride>) -> Option<f32> {
        let chess = detector.and_then(|d| d.chess_corners.as_ref())?;
        match chess.threshold_mode {
            Some(crate::registry::BenchChessThresholdMode::Absolute) => chess.threshold_value,
            None => chess.threshold_value,
            Some(crate::registry::BenchChessThresholdMode::Relative) => None,
        }
    }

    /// Profile target detection and laser extraction without running a solver.
    ///
    /// This intentionally keeps the diagnostic scoped to I/O + detector stages:
    /// if a puzzle dataset is already slow here, the full calibration path will
    /// not reach initialization quickly enough to be a useful first diagnostic.
    pub fn profile_dataset_stages(
        entry: &BenchEntry,
        max_images_per_camera: Option<usize>,
    ) -> Result<DatasetStageProfile> {
        let board = entry
            .board
            .as_ref()
            .context("stage profiling needs a `board` geometry")?;
        let detector = detector_for(board, entry.detector.as_ref())?;
        let detector_name = board
            .layout
            .as_deref()
            .unwrap_or("checkerboard")
            .to_string();
        let profile_limit = max_images_per_camera.unwrap_or(usize::MAX);
        let target_start = Instant::now();
        let mut target_per_camera = Vec::with_capacity(entry.cameras.len());

        for cam in &entry.cameras {
            let folder = entry.data_root.join(&cam.folder);
            let paths = glob_sorted_images(&folder, &cam.filename_glob)?;
            let selected: Vec<_> = paths.iter().take(profile_limit).collect();
            let mut images_used = 0usize;
            let mut points_total = 0usize;
            let mut points_max = 0usize;
            let mut load_ms = 0u64;
            let mut stage_ms = 0u64;
            let mut max_stage_ms = 0u64;

            for path in &selected {
                let load_start = Instant::now();
                let img = load_image(path)?;
                let img = apply_tile(&img, cam.tile);
                load_ms = load_ms.saturating_add(load_start.elapsed().as_millis() as u64);

                let stage_start = Instant::now();
                let detection = detector.detect(&img)?;
                let elapsed = stage_start.elapsed().as_millis() as u64;
                stage_ms = stage_ms.saturating_add(elapsed);
                max_stage_ms = max_stage_ms.max(elapsed);
                if let Some(view) = detection {
                    images_used += 1;
                    points_total += view.len();
                    points_max = points_max.max(view.len());
                }
            }
            target_per_camera.push(ProfileCameraStat {
                camera_id: cam.id.clone(),
                images_total: paths.len(),
                images_profiled: selected.len(),
                images_used,
                points_total,
                points_max,
                load_ms,
                stage_ms,
                mean_stage_ms: mean_ms(stage_ms, selected.len()),
                max_stage_ms,
            });
        }

        let laser_extraction = profile_laser_extraction(entry, profile_limit)?;
        let pose_count = entry
            .robot_poses
            .as_ref()
            .and_then(|pose_src| count_robot_poses(entry, pose_src).ok());

        Ok(DatasetStageProfile {
            dataset_id: entry.id.clone(),
            problem: entry.problem,
            camera_count: entry.cameras.len(),
            pose_count,
            max_images_per_camera,
            target_detection: TargetDetectionProfile {
                detector: detector_name,
                expected_features_per_board: board_feature_count(board),
                total_ms: target_start.elapsed().as_millis() as u64,
                per_camera: target_per_camera,
            },
            laser_extraction,
        })
    }

    fn mean_ms(total_ms: u64, count: usize) -> f64 {
        if count == 0 {
            0.0
        } else {
            total_ms as f64 / count as f64
        }
    }

    fn progress(entry: &BenchEntry, message: impl std::fmt::Display) {
        if std::env::var_os("CALIB_BENCH_QUIET").is_none() {
            eprintln!("[calib-bench:{}] {message}", entry.id);
        }
    }

    fn progress_images(
        entry: &BenchEntry,
        stage: &str,
        camera_id: &str,
        image_idx: usize,
        total: usize,
    ) {
        let n = image_idx + 1;
        if n == 1 || n == total || n.is_multiple_of(10) {
            progress(entry, format!("{stage} {camera_id}: {n}/{total} images"));
        }
    }

    fn count_robot_poses(
        entry: &BenchEntry,
        pose_src: &crate::registry::PoseSource,
    ) -> Result<usize> {
        let path = entry.data_root.join(&pose_src.path);
        match pose_src.format.as_str() {
            "snap_list_json" => Ok(load_snap_list_json(&path, 1.0)?.len()),
            "counted4x4" | "rowmajor4x4" => Ok(load_robot_poses_for(pose_src, &path)?.len()),
            other => anyhow::bail!("unsupported robot-pose format '{other}'"),
        }
    }

    #[cfg(not(feature = "laser"))]
    fn profile_laser_extraction(
        entry: &BenchEntry,
        _profile_limit: usize,
    ) -> Result<Option<LaserExtractionProfile>> {
        if entry.laser.is_some() {
            return Ok(Some(LaserExtractionProfile {
                enabled: false,
                total_ms: 0,
                per_camera: Vec::new(),
            }));
        }
        Ok(None)
    }

    #[cfg(feature = "laser")]
    fn profile_laser_extraction(
        entry: &BenchEntry,
        profile_limit: usize,
    ) -> Result<Option<LaserExtractionProfile>> {
        if entry.laser.is_none() {
            return Ok(None);
        }

        let paths = laser_image_paths(entry)?;
        let selected: Vec<_> = paths.iter().take(profile_limit).collect();
        let total_start = Instant::now();
        let mut per_camera = Vec::with_capacity(entry.cameras.len());
        for cam in &entry.cameras {
            let mut images_used = 0usize;
            let mut points_total = 0usize;
            let mut points_max = 0usize;
            let mut load_ms = 0u64;
            let mut stage_ms = 0u64;
            let mut max_stage_ms = 0u64;

            for path in &selected {
                let load_start = Instant::now();
                let img = load_image(path)?;
                let img = apply_tile(&img, cam.tile);
                load_ms = load_ms.saturating_add(load_start.elapsed().as_millis() as u64);

                let stage_start = Instant::now();
                let point_count = extract_laser_points(&img);
                let elapsed = stage_start.elapsed().as_millis() as u64;
                stage_ms = stage_ms.saturating_add(elapsed);
                max_stage_ms = max_stage_ms.max(elapsed);
                if point_count > 0 {
                    images_used += 1;
                    points_total += point_count;
                    points_max = points_max.max(point_count);
                }
            }

            per_camera.push(ProfileCameraStat {
                camera_id: cam.id.clone(),
                images_total: paths.len(),
                images_profiled: selected.len(),
                images_used,
                points_total,
                points_max,
                load_ms,
                stage_ms,
                mean_stage_ms: mean_ms(stage_ms, selected.len()),
                max_stage_ms,
            });
        }

        Ok(Some(LaserExtractionProfile {
            enabled: true,
            total_ms: total_start.elapsed().as_millis() as u64,
            per_camera,
        }))
    }

    fn split_reproj_report(
        entry: &BenchEntry,
        report: Option<vision_calibration::analysis::ReprojReport>,
    ) -> (
        Option<crate::record::CompactReprojReport>,
        Option<ResidualSidecar>,
    ) {
        match report {
            Some(report) => {
                let (compact, sidecar) = compact_reproj_report(&entry.id, report);
                (Some(compact), Some(sidecar))
            }
            None => (None, None),
        }
    }

    /// Crop a loaded image to a tile ROI `[x, y, w, h]`, if specified.
    ///
    /// Used when a single image file contains multiple cameras side-by-side;
    /// each camera's [`crate::registry::CameraLayout::tile`] selects its slice.
    /// Returns the original image unmodified when `tile` is `None`.
    fn apply_tile(img: &image::DynamicImage, tile: Option<[u32; 4]>) -> image::DynamicImage {
        match tile {
            Some([x, y, w, h]) => img.crop_imm(x, y, w, h),
            None => img.clone(),
        }
    }

    fn rtv3d_manual_intrinsics_seed(entry: &BenchEntry) -> RigHandeyeIntrinsicsManualInit {
        let (cx, cy) = entry
            .cameras
            .first()
            .and_then(|cam| cam.tile)
            .map(|[_x, _y, w, h]| (w as f64 * 0.5, h as f64 * 0.5))
            .unwrap_or((360.0, 270.0));
        let n = entry.cameras.len();
        let mut seed = RigHandeyeIntrinsicsManualInit::default();
        seed.per_cam_intrinsics = Some(vec![
            FxFyCxCySkew {
                fx: 2000.0,
                fy: 2000.0,
                cx,
                cy,
                skew: 0.0,
            };
            n
        ]);
        seed.per_cam_distortion = Some(vec![
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            };
            n
        ]);
        seed.per_cam_sensors = Some(vec![
            ScheimpflugParams {
                tilt_x: 0.0,
                tilt_y: 0.0,
            };
            n
        ]);
        seed
    }

    #[cfg(not(feature = "laser"))]
    fn extract_laser_observations(
        entry: &BenchEntry,
        _per_cam_dets: &[Vec<Option<CorrespondenceView>>],
        _per_cam_paths: &[Vec<PathBuf>],
        _robot_poses: &[Iso3],
        _n_views: usize,
    ) -> Result<Option<LaserObservationSet>> {
        if entry.laser.is_some() {
            anyhow::bail!(
                "dataset '{}' declares laser data; rebuild with --features 'tier-b laser'",
                entry.id
            );
        }
        Ok(None)
    }

    #[cfg(feature = "laser")]
    fn extract_laser_observations(
        entry: &BenchEntry,
        per_cam_dets: &[Vec<Option<CorrespondenceView>>],
        per_cam_paths: &[Vec<PathBuf>],
        robot_poses: &[Iso3],
        n_views: usize,
    ) -> Result<Option<LaserObservationSet>> {
        if entry.laser.is_none() {
            return Ok(None);
        }
        let keyed_lasers = laser_image_paths_by_target(entry)?;
        let indexed_lasers = if keyed_lasers.is_empty() {
            laser_image_paths(entry)?
        } else {
            Vec::new()
        };
        if keyed_lasers.is_empty() && indexed_lasers.is_empty() {
            return Ok(None);
        }

        progress(entry, "extracting laser observations for V5 joint BA");
        let n_cam = entry.cameras.len();
        let mut per_cam_images_used = vec![0usize; n_cam];
        let mut per_cam_points = vec![0usize; n_cam];
        let mut per_cam_ms = vec![0u64; n_cam];
        let mut total_points = 0usize;
        let mut total_images_used = 0usize;
        let mut candidate_views = 0usize;
        let total_start = Instant::now();
        let mut views = Vec::new();
        let mut laser_robot_poses = Vec::new();
        let mut laser_view_indices = Vec::new();

        for view_idx in 0..n_views {
            let laser_path = if keyed_lasers.is_empty() {
                indexed_lasers.get(view_idx)
            } else {
                let target_name = per_cam_paths[0][view_idx]
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                keyed_lasers.get(target_name)
            };
            let Some(laser_path) = laser_path else {
                continue;
            };
            candidate_views += 1;
            progress_images(entry, "laser", "all", candidate_views - 1, n_views);
            let img = load_image(laser_path)?;
            let mut laser_pixels = Vec::with_capacity(n_cam);
            let mut any_laser = false;
            for (cam_idx, cam) in entry.cameras.iter().enumerate() {
                let cam_start = Instant::now();
                let tile = apply_tile(&img, cam.tile);
                let pixels = extract_laser_pixels(&tile);
                per_cam_ms[cam_idx] =
                    per_cam_ms[cam_idx].saturating_add(cam_start.elapsed().as_millis() as u64);
                if pixels.is_empty() {
                    laser_pixels.push(None);
                } else {
                    any_laser = true;
                    per_cam_images_used[cam_idx] += 1;
                    per_cam_points[cam_idx] += pixels.len();
                    total_points += pixels.len();
                    laser_pixels.push(Some(pixels));
                }
            }
            if any_laser {
                total_images_used += 1;
                views.push(RigLaserlineView {
                    cameras: per_cam_dets
                        .iter()
                        .map(|dets| dets[view_idx].clone())
                        .collect(),
                    laser_pixels,
                });
                laser_robot_poses.push(robot_poses[view_idx]);
                laser_view_indices.push(view_idx);
            }
        }

        if views.is_empty() {
            return Ok(None);
        }

        let per_camera = entry
            .cameras
            .iter()
            .enumerate()
            .map(|(cam_idx, cam)| LaserCamStat {
                camera_id: cam.id.clone(),
                images_total: candidate_views,
                images_used: per_cam_images_used[cam_idx],
                points_extracted: per_cam_points[cam_idx],
                extract_ms: per_cam_ms[cam_idx],
                plane_residual_m: None,
                line_residual_px: None,
                inlier_ratio: None,
            })
            .collect();

        Ok(Some(LaserObservationSet {
            views,
            robot_poses: laser_robot_poses,
            view_indices: laser_view_indices,
            metrics: LaserMetrics {
                per_camera,
                total_points,
                total_images_used,
                extract_ms: total_start.elapsed().as_millis() as u64,
            },
        }))
    }

    #[cfg(feature = "laser")]
    fn extract_laser_points(img: &image::DynamicImage) -> usize {
        extract_laser_pixels(img).len()
    }

    #[cfg(feature = "laser")]
    fn extract_laser_pixels(img: &image::DynamicImage) -> Vec<Pt2> {
        let luma = img.to_luma8();
        let width = luma.width() as usize;
        let height = luma.height() as usize;
        let Ok(view) = ImageView::<u8>::from_slice(width, height, width, luma.as_raw()) else {
            return Vec::new();
        };
        let cfg = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Gather,
            },
            edge_cfg: Edge1DConfig {
                sigma: 1.2,
                pos_thresh: 4.0,
                neg_thresh: 4.0,
                ..Edge1DConfig::default()
            },
            ..LaserExtractConfig::default()
        };
        let mut extractor = LaserExtractor::new(cfg.edge_cfg.sigma);
        extractor
            .extract_line_u8(&view, 0..width, &cfg, None)
            .points
            .into_iter()
            .map(|p| Pt2::new(p.x as f64, p.y as f64))
            .collect()
    }

    #[cfg(feature = "laser")]
    #[derive(serde::Deserialize)]
    struct LaserPoseEntry {
        target_image: Option<String>,
        laser_image: Option<String>,
        #[serde(rename = "type")]
        snap_type: Option<String>,
    }

    #[cfg(feature = "laser")]
    fn laser_image_paths_by_target(
        entry: &BenchEntry,
    ) -> Result<std::collections::BTreeMap<String, PathBuf>> {
        let Some(pose_src) = &entry.robot_poses else {
            return Ok(std::collections::BTreeMap::new());
        };
        if pose_src.format != "snap_list_json" {
            return Ok(std::collections::BTreeMap::new());
        }
        let path = entry.data_root.join(&pose_src.path);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read laser pose list {}", path.display()))?;
        let entries: Vec<LaserPoseEntry> = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse laser pose list {}", path.display()))?;
        Ok(entries
            .into_iter()
            .filter(|entry| entry.snap_type.as_deref() == Some("double_snap"))
            .filter_map(|entry| Some((entry.target_image?, entry.laser_image?)))
            .map(|(target, laser)| {
                let target_name = Path::new(&target)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&target)
                    .to_string();
                (target_name, entry.data_root.join(laser))
            })
            .collect())
    }

    #[cfg(feature = "laser")]
    fn laser_image_paths(entry: &BenchEntry) -> Result<Vec<PathBuf>> {
        if let Some(pose_src) = &entry.robot_poses
            && pose_src.format == "snap_list_json"
        {
            let path = entry.data_root.join(&pose_src.path);
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read laser pose list {}", path.display()))?;
            let entries: Vec<LaserPoseEntry> = serde_json::from_str(&text)
                .with_context(|| format!("failed to parse laser pose list {}", path.display()))?;
            let paths: Vec<PathBuf> = entries
                .into_iter()
                .filter(|entry| entry.snap_type.as_deref() == Some("double_snap"))
                .filter_map(|entry| entry.laser_image)
                .map(|image| entry.data_root.join(image))
                .collect();
            if !paths.is_empty() {
                return Ok(paths);
            }
        }
        glob_sorted_images(&entry.data_root, "laser_*.png")
    }

    fn run_rig_handeye_laserline_v5(
        entry: &BenchEntry,
        rig_export: &RigHandeyeExport,
        observations: Option<LaserObservationSet>,
        robot_rot_sigma: f64,
        robot_trans_sigma: f64,
    ) -> Result<Option<JointV5Result>> {
        let Some(observations) = observations else {
            return Ok(None);
        };
        if observations.views.is_empty() {
            return Ok(None);
        }

        progress(
            entry,
            format!(
                "V5 laserline + joint BA over {} laser views",
                observations.views.len()
            ),
        );
        let n_cam = entry.cameras.len();
        let laserline_dataset =
            vision_calibration_optim::RigLaserlineDataset::new(observations.views.clone(), n_cam)
                .map_err(|e| anyhow::anyhow!("build RigLaserlineDataset: {e}"))?;
        let rig_se3_target: Vec<Iso3> = observations
            .robot_poses
            .iter()
            .map(|pose| rig_target_for_handeye_mode(rig_export, *pose))
            .collect::<Result<Vec<_>>>()?;
        let upstream = rig_export
            .to_upstream_calibration(rig_se3_target)
            .context("build rig laserline upstream from rig hand-eye export")?;
        let laserline_input = RigLaserlineDeviceInput {
            dataset: laserline_dataset.clone(),
            upstream,
            initial_planes_cam: None,
        };

        let mut laser_session =
            CalibrationSession::<RigLaserlineDeviceProblem>::with_description("bench_v5_laser");
        laser_session
            .set_input(laserline_input)
            .context("set laserline input failed")?;
        let mut laser_cfg = RigLaserlineDeviceConfig::default();
        laser_cfg.max_iters = Some(200);
        laser_cfg.verbosity = Some(0);
        laser_cfg.laser_residual_type = LaserlineResidualType::PointToPlane;
        laser_session
            .set_config(laser_cfg)
            .context("set laserline config failed")?;
        run_rig_laserline_device_calibration(&mut laser_session)
            .context("RigLaserlineDevice V5 stage failed")?;
        let laser_export = laser_session
            .export()
            .context("export V5 laserline stage failed")?;

        let mut laser_metrics = observations.metrics;
        apply_laserline_stage_stats(&mut laser_metrics, &laser_export.per_camera_stats);

        let initial_robot_deltas = rig_export.robot_deltas.as_ref().map(|deltas| {
            observations
                .view_indices
                .iter()
                .filter_map(|&idx| deltas.get(idx).copied())
                .collect::<Vec<_>>()
        });
        let initial_robot_deltas =
            initial_robot_deltas.filter(|deltas| deltas.len() == observations.views.len());

        let joint_dataset = RigHandeyeLaserlineDataset::from_rig_dataset(
            laserline_dataset,
            observations.robot_poses,
            rig_export.handeye_mode,
        )
        .map_err(|e| anyhow::anyhow!("build RigHandeyeLaserlineDataset: {e}"))?;

        let handeye = match rig_export.handeye_mode {
            HandEyeMode::EyeInHand => rig_export
                .gripper_se3_rig
                .context("EyeInHand rig export missing gripper_se3_rig")?,
            HandEyeMode::EyeToHand => rig_export
                .rig_se3_base
                .context("EyeToHand rig export missing rig_se3_base")?,
        };
        let target_ref = match rig_export.handeye_mode {
            HandEyeMode::EyeInHand => rig_export
                .base_se3_target
                .context("EyeInHand rig export missing base_se3_target")?,
            HandEyeMode::EyeToHand => rig_export
                .gripper_se3_target
                .context("EyeToHand rig export missing gripper_se3_target")?,
        };
        let sensors = rig_export
            .sensors
            .clone()
            .unwrap_or_else(|| vec![ScheimpflugParams::default(); n_cam]);
        let joint_initial = RigHandeyeLaserlineParams {
            cameras: rig_export.cameras.clone(),
            sensors,
            cam_to_rig: rig_export.cam_se3_rig.iter().map(|t| t.inverse()).collect(),
            handeye,
            target_ref,
            planes_cam: laser_export.laser_planes_cam.clone(),
        };

        let cam_fix = CameraFixMask {
            intrinsics: IntrinsicsFixMask::all_free(),
            distortion: DistortionFixMask {
                k1: false,
                k2: false,
                k3: true,
                p1: true,
                p2: true,
            },
        };
        let joint_opts = RigHandeyeLaserlineSolveOptions {
            laser_residual_type: LaserlineResidualType::PointToPlane,
            laser_weight: 1.0e4,
            calib_weight: 1.0,
            refine_robot_poses: true,
            robot_rot_sigma,
            robot_trans_sigma,
            initial_robot_deltas,
            fix_intrinsics: vec![cam_fix; n_cam],
            fix_extrinsics: (0..n_cam).map(|i| i == 0).collect(),
            fix_scheimpflug: vec![
                ScheimpflugFixMask {
                    tilt_x: true,
                    tilt_y: true,
                };
                n_cam
            ],
            ..Default::default()
        };
        let backend_opts = BackendSolveOptions {
            max_iters: 30,
            verbosity: 0,
            ..Default::default()
        };
        let joint_est =
            optimize_rig_handeye_laserline(joint_dataset, joint_initial, joint_opts, backend_opts)
                .context("final V5 optimize_rig_handeye_laserline failed")?;
        apply_joint_laser_stats(&mut laser_metrics, &joint_est.per_cam_stats);

        Ok(Some(JointV5Result {
            mean_reproj_error_px: joint_est.mean_reproj_error_px,
            per_cam_stats: joint_est.per_cam_stats,
            laser_metrics,
        }))
    }

    fn rig_target_for_handeye_mode(
        export: &RigHandeyeExport,
        base_se3_gripper: Iso3,
    ) -> Result<Iso3> {
        match export.handeye_mode {
            HandEyeMode::EyeInHand => {
                let gripper_se3_rig = export
                    .gripper_se3_rig
                    .context("EyeInHand rig export missing gripper_se3_rig")?;
                let base_se3_target = export
                    .base_se3_target
                    .context("EyeInHand rig export missing base_se3_target")?;
                Ok(gripper_se3_rig.inverse() * base_se3_gripper.inverse() * base_se3_target)
            }
            HandEyeMode::EyeToHand => {
                let rig_se3_base = export
                    .rig_se3_base
                    .context("EyeToHand rig export missing rig_se3_base")?;
                let gripper_se3_target = export
                    .gripper_se3_target
                    .context("EyeToHand rig export missing gripper_se3_target")?;
                Ok(rig_se3_base * base_se3_gripper * gripper_se3_target)
            }
        }
    }

    fn apply_laserline_stage_stats(
        metrics: &mut LaserMetrics,
        stats: &[vision_calibration_optim::LaserlineStats],
    ) {
        for (metric, stat) in metrics.per_camera.iter_mut().zip(stats) {
            metric.plane_residual_m =
                Some(ReprojectionStats::from_errors(&stat.per_view_laser_errors));
        }
    }

    fn apply_joint_laser_stats(
        metrics: &mut LaserMetrics,
        stats: &[RigHandeyeLaserlinePerCamStats],
    ) {
        for (metric, stat) in metrics.per_camera.iter_mut().zip(stats) {
            metric.plane_residual_m = Some(ReprojectionStats::from_summary(
                stat.mean_laser_err_m,
                stat.mean_laser_err_m,
                stat.max_laser_err_m,
                stat.laser_count,
            ));
            metric.line_residual_px = Some(ReprojectionStats::from_summary(
                stat.mean_laser_err_px,
                stat.mean_laser_err_px,
                stat.max_laser_err_px,
                stat.laser_count,
            ));
            metric.inlier_ratio = Some(1.0);
        }
    }

    fn reproj_stats_from_joint(stats: &[RigHandeyeLaserlinePerCamStats]) -> ReprojectionStats {
        let total_count: usize = stats.iter().map(|s| s.reproj_count).sum();
        if total_count == 0 {
            return ReprojectionStats::from_errors(&[]);
        }
        let mean = stats
            .iter()
            .map(|s| s.mean_reproj_error_px * s.reproj_count as f64)
            .sum::<f64>()
            / total_count as f64;
        let rms = (stats
            .iter()
            .map(|s| s.mean_reproj_error_px.powi(2) * s.reproj_count as f64)
            .sum::<f64>()
            / total_count as f64)
            .sqrt();
        let max = stats
            .iter()
            .map(|s| s.max_reproj_error_px)
            .fold(0.0_f64, f64::max);
        ReprojectionStats::from_summary(mean, rms, max, total_count)
    }

    fn reproj_stats_from_joint_cam(stat: &RigHandeyeLaserlinePerCamStats) -> ReprojectionStats {
        ReprojectionStats::from_summary(
            stat.mean_reproj_error_px,
            stat.mean_reproj_error_px,
            stat.max_reproj_error_px,
            stat.reproj_count,
        )
    }

    /// Synthesize a [`SolveReport`] for the record's [`Convergence`].
    ///
    /// The rig and hand-eye exports do not surface the solver's final cost /
    /// iteration count (unlike the planar path), so the bench records the
    /// export's mean reprojection error as a stand-in `final_cost` and `0`
    /// iterations. The record's authoritative reprojection numbers live in
    /// `Fit`; this keeps `Convergence.report` populated without inventing
    /// solver internals.
    fn synth_report(mean_reproj_px: f64) -> SolveReport {
        SolveReport {
            final_cost: mean_reproj_px,
            num_iters: 0,
        }
    }

    /// Pairing key for a rig image: the filename after a leading per-camera
    /// prefix has been stripped.
    ///
    /// The prefix is the longest leading run of ASCII alphanumerics (the camera
    /// tag — `Im_L`, `Cam1`, …) followed by one separator (`_`/`-`). For
    /// `Im_L_2.png` the run is `Im`, then `_`, leaving `L_2.png`; one more pass
    /// strips `L` + `_` to `2.png`. For `Cam1_<suffix>.png` the run `Cam1` + `_`
    /// is stripped in a single pass to `<suffix>.png`. Applying the same rule to
    /// every camera makes the shared tail the pairing key, reproducing the
    /// stereo examples' index/suffix pairing without hard-coding prefixes.
    fn pairing_key(path: &Path) -> String {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let mut s = name;
        // Peel up to two `<alphanumeric-run><sep>` segments. Two passes cover
        // both `Im_L_` (two segments) and `Cam1_` (one segment); the loop stops
        // early once the head is no longer a tag+separator.
        for _ in 0..2 {
            let after_run = s.trim_start_matches(|c: char| c.is_ascii_alphanumeric());
            if after_run.len() == s.len() {
                break; // no leading alphanumeric run — nothing to strip
            }
            match after_run.strip_prefix(['_', '-']) {
                Some(rest) => s = rest,
                None => break, // run not followed by a separator — keep as-is
            }
        }
        s.to_string()
    }

    /// Resolve the per-image detector for a rig from its board geometry.
    ///
    /// `layout` selects the detector: `"charuco"` builds a ChArUco detector from
    /// the board's rows/cols/cell-size/dictionary/marker_size_rel; anything else
    /// (incl. the `"checkerboard"` default) uses the chessboard detector with the
    /// board's metric cell size.
    fn detector_for(
        board: &BoardGeometry,
        detector_override: Option<&DetectorOverride>,
    ) -> Result<DetectorKind> {
        let layout = board.layout.as_deref().unwrap_or("checkerboard");
        let chess_config = chess_config_for_override(detector_override);
        if layout.eq_ignore_ascii_case("charuco") {
            let dict = board
                .dictionary
                .as_deref()
                .context("charuco board needs a `dictionary`")?;
            // Marker scale: from BoardGeometry if present, else the OpenCV
            // ChArUco default (0.75) used by the stereo_charuco example.
            let marker_scale = board.marker_size_rel.unwrap_or(0.75);
            let params = charuco_params_for(
                board.rows as u32,
                board.cols as u32,
                board.cell_size_m,
                marker_scale,
                dict,
            )?;
            Ok(DetectorKind::Charuco {
                params: Box::new(params),
                chess_config,
            })
        } else if layout.eq_ignore_ascii_case("puzzleboard") {
            let params =
                puzzleboard_params_for(board.rows as u32, board.cols as u32, board.cell_size_m)?;
            Ok(DetectorKind::Puzzleboard(Box::new(params)))
        } else {
            Ok(DetectorKind::Chessboard {
                rows: board.rows,
                cols: board.cols,
                require_known_grid: board.strict_grid,
                square_size_m: board.cell_size_m,
                chess_config,
            })
        }
    }

    /// Expected full-board feature count in the same semantics as the selected
    /// detector. PuzzleBoard registry dimensions are square counts, so the
    /// actual inner-corner grid is `(rows - 1) * (cols - 1)`.
    fn board_feature_count(board: &BoardGeometry) -> usize {
        match board.layout.as_deref() {
            Some(layout) if layout.eq_ignore_ascii_case("puzzleboard") => {
                board.rows.saturating_sub(1) * board.cols.saturating_sub(1)
            }
            _ => board.rows * board.cols,
        }
    }

    /// Build an [`Iso3`] from 16 row-major 4×4 values, scaling the translation.
    ///
    /// Rotation comes from the upper-left 3×3, translation from the 4th column
    /// scaled by `trans_scale` (`1.0` for metres, `1e-3` for millimetre files).
    /// The result is the `base_se3_gripper` (EyeInHand robot-pose prior).
    fn build_pose_from_4x4(values: &[f64], trans_scale: f64) -> Result<Iso3> {
        anyhow::ensure!(
            values.len() == 16,
            "expected 16 values for a 4x4 pose, got {}",
            values.len()
        );
        let r = Matrix3::new(
            values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
            values[10],
        );
        let t = Vector3::new(
            values[3] * trans_scale,
            values[7] * trans_scale,
            values[11] * trans_scale,
        );
        let rot = Rotation3::from_matrix_unchecked(r);
        Ok(Iso3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_rotation_matrix(&rot),
        ))
    }

    /// Parse one row-major 4×4 robot-pose line into an [`Iso3`] (translation in
    /// the file's native metres).
    ///
    /// Port of `support/handeye_io.rs::parse_pose_line`.
    fn parse_pose_line(line: &str, idx: usize) -> Result<Iso3> {
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("invalid float in robot pose line {}", idx + 1))?;
        build_pose_from_4x4(&values, 1.0).with_context(|| format!("robot pose line {}", idx + 1))
    }

    /// Load robot poses from a whitespace-delimited 4×4-per-line file
    /// (`rowmajor4x4`, translation in metres). Used by the kuka single-cam path.
    fn load_robot_poses(path: &PathBuf) -> Result<Vec<Iso3>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read robot poses from {}", path.display()))?;
        let mut poses = Vec::new();
        for (idx, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            poses.push(parse_pose_line(line, idx)?);
        }
        anyhow::ensure!(
            !poses.is_empty(),
            "robot pose file {} is empty",
            path.display()
        );
        Ok(poses)
    }

    /// Load robot poses from a counted-block file: a leading integer count, then
    /// that many row-major 4×4 matrices as whitespace-separated values (DS8's
    /// `robot_cali.txt`). `trans_scale` converts translations to metres.
    fn load_robot_poses_counted(path: &PathBuf, trans_scale: f64) -> Result<Vec<Iso3>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read robot poses from {}", path.display()))?;
        let mut tokens = text.split_whitespace();
        let count: usize = tokens
            .next()
            .context("counted pose file is empty")?
            .parse()
            .context("first token of a counted pose file must be the pose count")?;
        let vals: Vec<f64> = tokens
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .context("invalid float in counted pose file")?;
        anyhow::ensure!(
            vals.len() >= count * 16,
            "counted pose file declares {} poses ({} values) but has {}",
            count,
            count * 16,
            vals.len()
        );
        let mut poses = Vec::with_capacity(count);
        for k in 0..count {
            poses.push(build_pose_from_4x4(
                &vals[k * 16..k * 16 + 16],
                trans_scale,
            )?);
        }
        Ok(poses)
    }

    /// Deserialise one entry from `poses.json` (snap_list_json format).
    #[derive(serde::Deserialize)]
    struct SnapEntry {
        target_image: String,
        /// Row-major 4×4 homogeneous matrix: tcp→base (= `base_se3_gripper`).
        tcp2base: [[f64; 4]; 4],
    }

    /// Load robot poses from a JSON list of snap objects (`poses.json`).
    ///
    /// Returns `(target_image_filename, base_se3_gripper)` pairs in file order.
    /// `trans_scale` converts the translation column to metres (`1e-3` for mm).
    /// Rotation is taken from the upper-left 3×3 and re-normalised via
    /// `UnitQuaternion::from_rotation_matrix` to guard against floating-point
    /// near-orthogonality.
    fn load_snap_list_json(path: &PathBuf, trans_scale: f64) -> Result<Vec<(String, Iso3)>> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read snap list from {}", path.display()))?;
        let entries: Vec<SnapEntry> = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse snap list JSON {}", path.display()))?;
        anyhow::ensure!(
            !entries.is_empty(),
            "snap list JSON {} is empty",
            path.display()
        );
        let mut out = Vec::with_capacity(entries.len());
        for e in &entries {
            let m = &e.tcp2base;
            // Upper-left 3×3 is the rotation.
            let r = Matrix3::new(
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
            );
            // 4th column (column-major index 3) is the translation.
            let t = Vector3::new(
                m[0][3] * trans_scale,
                m[1][3] * trans_scale,
                m[2][3] * trans_scale,
            );
            let rot = Rotation3::from_matrix_unchecked(r);
            let iso = Iso3::from_parts(
                Translation3::from(t),
                UnitQuaternion::from_rotation_matrix(&rot),
            );
            out.push((e.target_image.clone(), iso));
        }
        Ok(out)
    }

    /// Load robot poses for an entry, dispatching on
    /// [`crate::registry::PoseSource::format`] and scaling by `units`.
    fn load_robot_poses_for(
        pose_src: &crate::registry::PoseSource,
        path: &PathBuf,
    ) -> Result<Vec<Iso3>> {
        let trans_scale = match pose_src.units.as_deref() {
            Some("mm") => 1.0e-3,
            Some("m") | None => 1.0,
            Some(other) => anyhow::bail!("unsupported pose units '{other}' (use 'mm' or 'm')"),
        };
        let mut poses = match pose_src.format.as_str() {
            "counted4x4" => load_robot_poses_counted(path, trans_scale),
            "rowmajor4x4" => {
                let mut poses = load_robot_poses(path)?;
                if trans_scale != 1.0 {
                    for p in &mut poses {
                        p.translation.vector *= trans_scale;
                    }
                }
                Ok(poses)
            }
            // Snap-list JSON pairs each pose with a `target_image`; for the rig
            // path (index-pairing) we drop the names and return poses in file
            // order. The registry's per-camera glob must natural-sort to the same
            // order as this list (e.g. `target_*.png` ↔ `target_0..N`), exactly
            // as the counted/rowmajor formats already assume pose `i` ↔ frame `i`.
            "snap_list_json" => Ok(load_snap_list_json(path, trans_scale)?
                .into_iter()
                .map(|(_image, pose)| pose)
                .collect()),
            other => anyhow::bail!(
                "unsupported robot-pose format '{other}' \
                 (use 'rowmajor4x4', 'counted4x4', or 'snap_list_json')"
            ),
        }?;

        apply_robot_pose_convention(&mut poses, &pose_src.convention)?;
        Ok(poses)
    }

    fn apply_robot_pose_convention(poses: &mut [Iso3], convention: &str) -> Result<()> {
        for pose in poses {
            normalize_robot_pose_convention(pose, convention)?;
        }
        Ok(())
    }

    fn normalize_robot_pose_convention(pose: &mut Iso3, convention: &str) -> Result<()> {
        match convention {
            "base_se3_gripper" => {}
            "gripper_se3_base" => {
                *pose = pose.inverse();
            }
            other => anyhow::bail!(
                "unsupported robot-pose convention '{other}' \
                 (use 'base_se3_gripper' or 'gripper_se3_base')"
            ),
        }
        Ok(())
    }

    fn artifacts_from_planar(
        camera_id: &str,
        export: &PlanarIntrinsicsExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: vec![camera_artifact(camera_id, &export.params.camera, None)],
            transforms: Vec::new(),
        };
        for (view_idx, pose) in export.params.camera_se3_target.iter().enumerate() {
            artifacts.transforms.push(transform_artifact(
                format!("{camera_id}_se3_target_view{view_idx}"),
                camera_id,
                format!("target/view_{view_idx}"),
                pose,
            ));
        }
        artifacts
    }

    fn artifacts_from_single_cam(
        camera_id: &str,
        export: &SingleCamHandeyeExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: vec![camera_artifact(camera_id, &export.camera, None)],
            transforms: Vec::new(),
        };
        if let Some(t) = &export.gripper_se3_camera {
            artifacts.transforms.push(transform_artifact(
                "gripper_se3_camera",
                "gripper",
                camera_id,
                t,
            ));
        }
        if let Some(t) = &export.camera_se3_base {
            artifacts
                .transforms
                .push(transform_artifact("camera_se3_base", camera_id, "base", t));
        }
        if let Some(t) = &export.base_se3_target {
            artifacts
                .transforms
                .push(transform_artifact("base_se3_target", "base", "target", t));
        }
        if let Some(t) = &export.gripper_se3_target {
            artifacts.transforms.push(transform_artifact(
                "gripper_se3_target",
                "gripper",
                "target",
                t,
            ));
        }
        artifacts
    }

    fn artifacts_from_rig_extrinsics(
        cameras: &[CameraLayout],
        export: &RigExtrinsicsExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: camera_artifacts(cameras, &export.cameras, export.sensors.as_deref()),
            transforms: Vec::new(),
        };
        for (cam_idx, pose) in export.cam_se3_rig.iter().enumerate() {
            let camera_id = camera_id(cameras, cam_idx);
            artifacts.transforms.push(transform_artifact(
                format!("{camera_id}_se3_rig"),
                camera_id,
                "rig",
                pose,
            ));
        }
        for (view_idx, pose) in export.rig_se3_target.iter().enumerate() {
            artifacts.transforms.push(transform_artifact(
                format!("rig_se3_target_view{view_idx}"),
                "rig",
                format!("target/view_{view_idx}"),
                pose,
            ));
        }
        artifacts
    }

    fn artifacts_from_rig_handeye(
        cameras: &[CameraLayout],
        export: &RigHandeyeExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: camera_artifacts(cameras, &export.cameras, export.sensors.as_deref()),
            transforms: Vec::new(),
        };
        for (cam_idx, pose) in export.cam_se3_rig.iter().enumerate() {
            let camera_id = camera_id(cameras, cam_idx);
            artifacts.transforms.push(transform_artifact(
                format!("{camera_id}_se3_rig"),
                camera_id,
                "rig",
                pose,
            ));
        }
        if let Some(t) = &export.gripper_se3_rig {
            artifacts
                .transforms
                .push(transform_artifact("gripper_se3_rig", "gripper", "rig", t));
        }
        if let Some(t) = &export.rig_se3_base {
            artifacts
                .transforms
                .push(transform_artifact("rig_se3_base", "rig", "base", t));
        }
        if let Some(t) = &export.base_se3_target {
            artifacts
                .transforms
                .push(transform_artifact("base_se3_target", "base", "target", t));
        }
        if let Some(t) = &export.gripper_se3_target {
            artifacts.transforms.push(transform_artifact(
                "gripper_se3_target",
                "gripper",
                "target",
                t,
            ));
        }
        for (view_idx, pose) in export.rig_se3_target.iter().enumerate() {
            artifacts.transforms.push(transform_artifact(
                format!("rig_se3_target_view{view_idx}"),
                "rig",
                format!("target/view_{view_idx}"),
                pose,
            ));
        }
        artifacts
    }

    fn camera_artifacts(
        layouts: &[CameraLayout],
        cameras: &[PinholeCamera],
        sensors: Option<&[ScheimpflugParams]>,
    ) -> Vec<CameraArtifact> {
        cameras
            .iter()
            .enumerate()
            .map(|(idx, camera)| {
                camera_artifact(
                    camera_id(layouts, idx),
                    camera,
                    sensors.and_then(|s| s.get(idx)),
                )
            })
            .collect()
    }

    fn camera_id(layouts: &[CameraLayout], index: usize) -> &str {
        layouts
            .get(index)
            .map(|layout| layout.id.as_str())
            .unwrap_or("camera")
    }

    fn camera_artifact(
        camera_id: &str,
        camera: &PinholeCamera,
        sensor: Option<&ScheimpflugParams>,
    ) -> CameraArtifact {
        let k = camera.k;
        let d = camera.dist;
        CameraArtifact {
            camera_id: camera_id.to_string(),
            camera_matrix_px: [[k.fx, k.skew, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]],
            intrinsics_px: IntrinsicsArtifact {
                fx: k.fx,
                fy: k.fy,
                cx: k.cx,
                cy: k.cy,
                skew: k.skew,
            },
            distortion_model: "brown_conrady5".to_string(),
            distortion: DistortionArtifact {
                k1: d.k1,
                k2: d.k2,
                k3: d.k3,
                p1: d.p1,
                p2: d.p2,
            },
            scheimpflug: sensor.map(|s| ScheimpflugArtifact {
                tilt_x_rad: s.tilt_x,
                tilt_y_rad: s.tilt_y,
            }),
        }
    }

    fn transform_artifact(
        name: impl Into<String>,
        to_frame: impl Into<String>,
        from_frame: impl Into<String>,
        transform: &Iso3,
    ) -> TransformArtifact {
        let q = transform.rotation.quaternion();
        let rv = transform.rotation.scaled_axis();
        TransformArtifact {
            name: name.into(),
            to_frame: to_frame.into(),
            from_frame: from_frame.into(),
            translation_mm: [
                transform.translation.vector.x * 1000.0,
                transform.translation.vector.y * 1000.0,
                transform.translation.vector.z * 1000.0,
            ],
            rotation_quat_xyzw: [q.i, q.j, q.k, q.w],
            rotation_rotvec_deg: [rv.x.to_degrees(), rv.y.to_degrees(), rv.z.to_degrees()],
        }
    }

    /// Build an [`Ident`] with the run-intrinsic fields filled and the
    /// externally-injected provenance fields left as placeholders for the caller
    /// (`calib_bench.rs`) to overwrite.
    fn placeholder_ident(entry: &BenchEntry, problem: &str) -> Ident {
        Ident {
            dataset_id: entry.id.clone(),
            problem: problem.to_string(),
            tier: "b".to_string(),
            git_sha: String::new(),
            timestamp_rfc3339: String::new(),
            config_hash: 0,
            bench_schema_version: BENCH_SCHEMA_VERSION,
            features: Vec::new(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use vision_calibration_core::{CameraProject, Pt3};

        #[test]
        fn detector_for_puzzleboard_layout_dispatches() {
            let board = BoardGeometry {
                rows: 130,
                cols: 130,
                cell_size_m: 0.001014,
                dictionary: None,
                layout: Some("puzzleboard".to_string()),
                marker_size_rel: None,
                strict_grid: false,
            };

            let detector = detector_for(&board, None).expect("detector");
            match detector {
                DetectorKind::Puzzleboard(params) => {
                    assert_eq!(params.board.rows, 130);
                    assert_eq!(params.board.cols, 130);
                }
                _ => panic!("expected puzzleboard detector"),
            }
        }

        #[test]
        fn staged_multistart_recovers_synthetic_scheimpflug_intrinsics() {
            let dataset = synthetic_scheimpflug_dataset();
            let seeds = [
                IntrinsicsSeed {
                    focal_px: 1600.0,
                    cx: 360.0,
                    cy: 270.0,
                    tau_x: 0.04,
                    tau_y: 0.02,
                },
                IntrinsicsSeed {
                    focal_px: 2200.0,
                    cx: 360.0,
                    cy: 270.0,
                    tau_x: -0.12,
                    tau_y: 0.0,
                },
            ];

            let solve = solve_centered_multistart(&dataset, &seeds, 0.4).expect("synthetic solve");
            assert!(
                solve.report.stats.mean < 0.05,
                "synthetic Scheimpflug solve should reach near-zero raw mean, got {}",
                solve.report.stats.mean
            );
            assert!(solve.report.mean_crosscheck_abs_diff <= 1e-9);
            assert_eq!(solve.report.params.cx, 360.0);
            assert_eq!(solve.report.params.cy, 270.0);
            assert_eq!(solve.report.params.k3, 0.0);
            assert_eq!(solve.report.params.p1, 0.0);
            assert_eq!(solve.report.params.p2, 0.0);
        }

        #[test]
        fn intrinsics_diagnose_json_shape_is_stable() {
            let mut errors = vec![0.2, 0.4, 0.8, 2.5, 5.5];
            let stats = intrinsics_stats_from_errors(&mut errors);
            let report = IntrinsicsDiagnoseReport {
                dataset_id: "synthetic".to_string(),
                gate_px: 0.4,
                chess_threshold_abs: Some(30.0),
                pass: false,
                cameras: vec![IntrinsicsCameraReport {
                    camera_id: "cam0".to_string(),
                    image_size: [720, 540],
                    images_total: 2,
                    images_used: 2,
                    features_detected: 5,
                    roi_local_coordinates: true,
                    gate_case: IntrinsicsCaseReport {
                        case: "centered_p1p2_k3_fixed".to_string(),
                        accepted_for_gate: true,
                        passed_gate: false,
                        seed: IntrinsicsSeed {
                            focal_px: 2000.0,
                            cx: 360.0,
                            cy: 270.0,
                            tau_x: -0.08,
                            tau_y: 0.0,
                        },
                        params: IntrinsicsParamReport {
                            fx: 2000.0,
                            fy: 1990.0,
                            cx: 360.0,
                            cy: 270.0,
                            skew: 0.0,
                            k1: -0.1,
                            k2: 0.2,
                            k3: 0.0,
                            p1: 0.0,
                            p2: 0.0,
                            tau_x: -0.08,
                            tau_y: 0.0,
                        },
                        stats: stats.clone(),
                        solver_mean_reproj_error: stats.mean,
                        mean_crosscheck_abs_diff: 0.0,
                        solve_report: SolveReport {
                            final_cost: 1.0,
                            num_iters: 3,
                        },
                        note: None,
                    },
                    diagnostic_cases: Vec::new(),
                    per_pose: vec![IntrinsicsPoseStats {
                        pose: 0,
                        stats: stats.clone(),
                    }],
                    top_outliers: Vec::new(),
                }],
            };

            let value = serde_json::to_value(&report).expect("json");
            assert_eq!(value["dataset_id"], "synthetic");
            assert_eq!(value["gate_px"], 0.4);
            assert_eq!(value["cameras"][0]["gate_case"]["stats"]["count"], 5);
            assert_eq!(value["cameras"][0]["gate_case"]["params"]["tau_x"], -0.08);
            assert_eq!(value["cameras"][0]["gate_case"]["seed"]["focal_px"], 2000.0);
            assert_eq!(value["cameras"][0]["per_pose"][0]["pose"], 0);
            assert_eq!(value["cameras"][0]["per_pose"][0]["stats"]["count"], 5);
        }

        fn synthetic_scheimpflug_dataset() -> PlanarDataset {
            let points = synthetic_board(7, 9, 0.0052);
            let k = FxFyCxCySkew {
                fx: 1950.0,
                fy: 1900.0,
                cx: 360.0,
                cy: 270.0,
                skew: 0.0,
            };
            let dist = BrownConrady5 {
                k1: -0.14,
                k2: 0.06,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            };
            let sensor = ScheimpflugParams {
                tilt_x: -0.08,
                tilt_y: 0.01,
            };
            let camera = Camera::new(Pinhole, dist, sensor.compile(), k);
            let poses = [
                synthetic_pose(-0.05, 0.03, 0.01, -0.022, -0.016, 0.19),
                synthetic_pose(0.08, -0.04, 0.02, -0.020, -0.014, 0.21),
                synthetic_pose(-0.03, -0.08, -0.02, -0.024, -0.017, 0.18),
                synthetic_pose(0.04, 0.06, 0.04, -0.019, -0.018, 0.22),
                synthetic_pose(-0.07, 0.01, -0.03, -0.023, -0.015, 0.20),
            ];
            let views = poses
                .iter()
                .map(|pose| {
                    let image: Vec<_> = points
                        .iter()
                        .map(|p| {
                            let p_cam = pose * p;
                            camera
                                .project_camera_point(&p_cam.coords)
                                .expect("synthetic point projects")
                        })
                        .collect();
                    View::without_meta(
                        CorrespondenceView::new(points.clone(), image).expect("view"),
                    )
                })
                .collect();
            PlanarDataset::new(views).expect("dataset")
        }

        fn synthetic_board(rows: usize, cols: usize, pitch: f64) -> Vec<Pt3> {
            let mut points = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    points.push(Pt3::new(c as f64 * pitch, r as f64 * pitch, 0.0));
                }
            }
            points
        }

        fn synthetic_pose(rx: f64, ry: f64, rz: f64, tx: f64, ty: f64, tz: f64) -> Iso3 {
            let rot = UnitQuaternion::from_scaled_axis(Vector3::new(rx, ry, rz));
            Iso3::from_parts(Translation3::new(tx, ty, tz), rot)
        }
    }
}
