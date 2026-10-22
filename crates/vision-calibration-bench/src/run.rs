//! Run a calibration problem end-to-end and return a [`crate::record::BenchRecord`] with
//! captured metrics.
//!
//! Tier-A would replay a frozen fixture through the math/serde path; that lands
//! in a later phase. The Tier-B path here builds the problem `Input` from the
//! entry's manifest with the pipeline's dataset runner (`crate::detect`),
//! runs the problem's steps one by one (to time each stage), and captures both
//! bench-recomputed and self-reported reprojection metrics so any divergence
//! between them is visible in the record.
//!
//! What stays here, on top of the runner: the seeded official route (the
//! device-spec seed of a Scheimpflug camera, a registry `seed`), registry
//! config overrides, stage timing, the record, and the rig hand-eye entries'
//! joint laserline stage — laser frames paired through the snap list,
//! extracted with `vision-metrology`, and refined jointly with the rig
//! hand-eye result.

#[cfg(feature = "tier-b")]
pub use tier_b::{
    DetectedScheimpflugCamera, ScheimpflugSeededSolve, detect_scheimpflug_seeded_input,
    diagnose_intrinsics, progress_label, run_planar_intrinsics, run_rig_extrinsics,
    run_rig_handeye, run_scheimpflug_intrinsics, run_single_cam_handeye, solve_scheimpflug_seeded,
};

#[cfg(feature = "tier-b")]
pub mod tier_b {
    //! Tier-B entry points that require detection and image-loading capabilities.

    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{Context, Result};
    use vision_calibration::analysis::{
        QualityReport, RigStageReprojection, assess, planar_intrinsics_report,
        rig_extrinsics_report, rig_handeye_report_with_rig_stage, single_cam_handeye_report,
    };
    use vision_calibration::common::CalibratedCameras;
    use vision_calibration::core::{Iso3, RigDataset};
    use vision_calibration::optim::{HandeyeMode, RobotPoseMeta};
    use vision_calibration::planar_intrinsics::{
        PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem, step_init,
        step_init_with_seed, step_optimize,
    };
    use vision_calibration::rig_extrinsics::{
        RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsProblem, step_intrinsics_init_all,
        step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
    };
    // Rig hand-eye step fns share names with the rig-extrinsics ones, so alias.
    use vision_calibration::rig_handeye::{
        IntrinsicsManualInit as RigIntrinsicsManualInit, RigHandeyeConfig, RigHandeyeExport,
        RigHandeyeProblem, step_handeye_init as rh_handeye_init,
        step_handeye_optimize as rh_handeye_optimize,
        step_intrinsics_init_all_with_seed as rh_intrinsics_init_all_with_seed,
        step_intrinsics_optimize_all as rh_intrinsics_optimize_all, step_rig_init as rh_rig_init,
        step_rig_optimize as rh_rig_optimize,
    };
    use vision_calibration::rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
        run_calibration as run_rig_laserline_device_calibration,
    };
    // Named at module scope so `DetectedScheimpflugCamera` can carry the seed;
    // the seeded route's step functions are the planar ones imported above.
    use vision_calibration::planar_intrinsics::IntrinsicsManualInit;
    use vision_calibration::session::CalibrationSession;
    use vision_calibration::single_cam_handeye::{
        SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeProblem, step_handeye_init,
        step_handeye_optimize, step_intrinsics_init, step_intrinsics_optimize,
    };
    use vision_calibration_core::{
        BrownConrady5, Camera, CameraFix, CameraModelSpec, CameraParams, DistortionKind,
        DistortionParams, FeatureResidualHistogram, FxFyCxCy, Pinhole, PinholeCamera,
        PlanarDataset, ProjectionParams, Pt2, ReprojectionStats, ScheimpflugParams, SensorKind,
        SensorParams, TargetFeatureResidual, View, compute_planar_target_residuals,
        make_pinhole_camera,
    };
    use vision_calibration_optim::{
        BackendSolveOptions, LaserlineResidualType, PlanarIntrinsicsEstimate,
        PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigHandeyeLaserlineDataset,
        RigHandeyeLaserlineParams, RigHandeyeLaserlinePerCamStats, RigHandeyeLaserlineSolveOptions,
        RigLaserlineView, RobustLoss, SolveReport, SolverBackend, optimize_planar_intrinsics,
        optimize_rig_handeye_laserline,
    };
    #[cfg(feature = "laser")]
    use vision_metrology::{
        ColAccess, Edge1DConfig, ImageView, LaserExtractConfig, LaserExtractor, ScanAxis,
    };

    use vision_calibration_dataset::{
        CameraSource, ChessCornersDetectorSpec, DatasetSpec, DetectorSpec, TargetSpec,
    };
    use vision_calibration_detect::DetectionCache;
    use vision_calibration_pipeline::dataset_runner::{RunError, build_planar_input};

    use crate::detect::{
        RigViewPaths, TimingCache, board_feature_count, camera_spec, detect_planar,
        detect_rig_extrinsics, detect_rig_handeye, detect_single_cam_handeye, detection_cache, ms,
    };
    use crate::manifest::{dataset_spec, resolve_against};
    #[cfg(feature = "laser")]
    use crate::record::LaserCamStat;
    use crate::record::{
        BENCH_SCHEMA_VERSION, BenchRecord, CalibrationArtifacts, CameraArtifact, Convergence,
        Detection, DistortionArtifact, Fit, Ident, IntrinsicsArtifact, LaserMetrics,
        ResidualSidecar, RobotCorrectionSummary, ScheimpflugArtifact, StageTiming, Timing,
        TransformArtifact, compact_reproj_report,
    };
    use crate::registry::{BenchEntry, ProblemKind};

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
        /// Target kind of the manifest (`chessboard`, `charuco`, …).
        pub detector: String,
        /// Expected full-board feature count of a registry board (zero for a
        /// manifest target).
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
        /// Total image load/decode/tile time in milliseconds, when measured
        /// apart from the stage. Target detection decodes inside the dataset
        /// runner, so its `stage_ms` includes decoding and this is absent.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub load_ms: Option<u64>,
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
        /// Mean reported by `optimize_planar_intrinsics`.
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
        /// Horizontal focal length in pixels.
        pub fx: f64,
        /// Vertical focal length in pixels.
        pub fy: f64,
        /// Principal-point x-coordinate in pixels.
        pub cx: f64,
        /// Principal-point y-coordinate in pixels.
        pub cy: f64,
        /// First radial distortion coefficient.
        pub k1: f64,
        /// Second radial distortion coefficient.
        pub k2: f64,
        /// Third radial distortion coefficient.
        pub k3: f64,
        /// First tangential distortion coefficient.
        pub p1: f64,
        /// Second tangential distortion coefficient.
        pub p2: f64,
        /// Scheimpflug tilt angle about the x-axis in radians.
        pub tau_x: f64,
        /// Scheimpflug tilt angle about the y-axis in radians.
        pub tau_y: f64,
    }

    /// Residual distribution for raw Euclidean reprojection errors.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsResidualStats {
        /// Mean reprojection error in pixels.
        pub mean: f64,
        /// Root-mean-square reprojection error in pixels.
        pub rms: f64,
        /// Median reprojection error in pixels.
        pub median: f64,
        /// 90th-percentile reprojection error in pixels.
        pub p90: f64,
        /// 95th-percentile reprojection error in pixels.
        pub p95: f64,
        /// 99th-percentile reprojection error in pixels.
        pub p99: f64,
        /// Maximum reprojection error in pixels.
        pub max: f64,
        /// Total number of reprojected feature observations.
        pub count: usize,
        /// Number of observations with error ≤ 0.4 px.
        pub count_le_0_4: usize,
        /// Number of observations with error ≤ 1 px.
        pub count_le_1: usize,
        /// Number of observations with error > 2 px.
        pub count_gt_2: usize,
        /// Number of observations with error > 5 px.
        pub count_gt_5: usize,
        /// Diagnostic only; never used to satisfy the quality gate.
        pub trimmed_mean_95: f64,
    }

    /// Residual statistics for one target pose/view.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsPoseStats {
        /// Zero-based index of the target pose (view).
        pub pose: usize,
        /// Residual statistics for this pose.
        pub stats: IntrinsicsResidualStats,
    }

    /// One large-residual feature for drill-down.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct IntrinsicsOutlier {
        /// Zero-based index of the target pose containing this outlier.
        pub pose: usize,
        /// Zero-based index of the feature within the pose.
        pub feature: usize,
        /// 3-D target point in metres.
        pub target_xyz_m: [f64; 3],
        /// Observed image coordinate in pixels.
        pub observed_px: [f64; 2],
        /// Projected image coordinate in pixels, or `None` if projection failed.
        pub projected_px: Option<[f64; 2]>,
        /// Euclidean reprojection error in pixels.
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
        params: PlanarIntrinsicsParams,
        residuals: Vec<TargetFeatureResidual>,
    }

    #[derive(Debug, Clone)]
    struct IntrinsicsVariantConfig {
        case: &'static str,
        fix: CameraFix,
        note: Option<String>,
    }

    #[derive(Debug, Clone)]
    struct JointV5Result {
        mean_reproj_error_px: f64,
        per_cam_stats: Vec<RigHandeyeLaserlinePerCamStats>,
        laser_metrics: LaserMetrics,
    }

    /// The library's quality report on `export`, under the entry's acceptance
    /// thresholds (the defaults when it sets none).
    fn quality_report<E: CalibratedCameras>(
        entry: &BenchEntry,
        export: &E,
        image_sizes: &[[u32; 2]],
    ) -> Result<QualityReport> {
        let thresholds = entry
            .accept
            .and_then(|gate| gate.quality)
            .unwrap_or_default();
        assess(export, image_sizes, &thresholds)
            .with_context(|| format!("entry `{}`: quality assessment failed", entry.id))
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
    /// Detects the entry's manifest through the dataset runner (views with no
    /// board are dropped), runs `step_init` → `step_optimize`, and assembles
    /// the record. The returned record's `ident` carries placeholder
    /// provenance (`git_sha` / `timestamp_rfc3339` / `features` are filled by
    /// the caller).
    pub fn run_planar_intrinsics(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::PlanarIntrinsics,
            "run_planar_intrinsics called for {:?}",
            entry.problem
        );
        let spec = dataset_spec(entry)?;
        let camera_id = single_camera_id(&spec)?;

        // ── Detection ──────────────────────────────────────────────────────
        progress(entry, format!("detecting views for {camera_id}"));
        let detected = detect_planar(
            &spec,
            &entry.data_root,
            entry.board.as_ref(),
            &detection_cache(),
        )
        .with_context(|| format!("entry `{}`: detection failed", entry.id))?;
        let detection_ms = detected.detection_ms;
        progress(
            entry,
            format!(
                "detection finished: {}/{} images, {} features, {detection_ms} ms",
                detected.detection.per_camera[0].images_used,
                detected.detection.per_camera[0].images_total,
                detected.detection.total_detected,
            ),
        );
        anyhow::ensure!(
            detected.input.num_views() >= 3,
            "need >= 3 detected views for planar intrinsics, got {}",
            detected.input.num_views()
        );

        // ── Calibration ────────────────────────────────────────────────────
        // Keep a copy of the detected views for the multi-level reproj report
        // (the dataset is moved into the session below).
        let views_for_report = detected.input.views.clone();
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        let mut config = PlanarIntrinsicsConfig::default();
        config.solver.backend = entry.solver_backend();
        session.set_config(config).context("set_config failed")?;
        session
            .set_input(detected.input)
            .context("set_input failed")?;

        progress(entry, "initializing planar intrinsics");
        let init_start = Instant::now();
        let init_ok = step_init(&mut session, None).is_ok();
        let init_ms = ms_since(init_start);
        anyhow::ensure!(init_ok, "step_init failed for {}", entry.id);

        progress(entry, "optimizing planar intrinsics");
        let opt_start = Instant::now();
        let _ = step_optimize(&mut session, None).context("step_optimize failed")?;
        let optimize_ms = ms_since(opt_start);

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
            detection: Some(detected.detection),
            laser: None,
            robot_corrections: None,
            artifacts: Some(artifacts_from_planar(camera_id, &export)),
            delta_to_prior: None,
            timing,
            quality: Some(quality_report(entry, &export, &detected.image_sizes)?),
            reproj_report,
            residual_sidecar,
        })
    }

    /// The id of a single-camera manifest's camera.
    fn single_camera_id(spec: &DatasetSpec) -> Result<&str> {
        anyhow::ensure!(
            spec.cameras.len() == 1,
            "a single-camera problem needs exactly one camera, got {}",
            spec.cameras.len()
        );
        Ok(&spec.cameras[0].id)
    }

    /// Detected views + device-spec-derived seed for a single-camera
    /// Scheimpflug intrinsics entry (ADR 0011) — the seed-independent
    /// half of [`run_scheimpflug_intrinsics`].
    ///
    /// Manifest-driven: the entry's manifest (its `spec`, or the one its
    /// registry fields describe) must be a single-camera `PlanarIntrinsics`
    /// dataset; the entry's `device_spec` (the ADR 0011 `spec.json`
    /// sidecar) provides the seed via `device_seed::scheimpflug_seed`.
    ///
    /// Detection is the expensive step. [`run_scheimpflug_intrinsics`] calls
    /// this once and immediately solves; the convergence-basin study
    /// (`vision_calibration_bench::basin`) calls this once per camera and
    /// replays [`solve_scheimpflug_seeded`] over many perturbed copies of
    /// `seed` without re-detecting.
    #[derive(Debug, Clone)]
    pub struct DetectedScheimpflugCamera {
        /// Camera id (the manifest's single camera).
        pub camera_id: String,
        /// Detected planar dataset. Cheap to clone across seed variants.
        pub dataset: PlanarDataset,
        /// Device-spec-derived seed. `intrinsics` and `sensor` are always
        /// `Some` — see `device_seed::scheimpflug_seed`.
        pub seed: IntrinsicsManualInit,
        /// Detection stats for the `BenchRecord`.
        pub detection: Detection,
        /// `[width, height]` of the camera's images (its region of
        /// interest, when the manifest crops one).
        pub image_size: [u32; 2],
        /// Wall-clock detection time in milliseconds.
        pub detection_ms: u64,
    }

    /// Detect `entry`'s single camera and derive its ADR 0011 seed. See
    /// [`DetectedScheimpflugCamera`].
    pub fn detect_scheimpflug_seeded_input(
        entry: &BenchEntry,
    ) -> Result<DetectedScheimpflugCamera> {
        use vision_calibration::device_seed::{DeviceSpec, scheimpflug_seed};

        anyhow::ensure!(
            entry.is_scheimpflug_planar(),
            "detect_scheimpflug_seeded_input called for {:?} without a Scheimpflug sensor",
            entry.problem
        );
        let spec = dataset_spec(entry)?;
        let camera_id = single_camera_id(&spec)?.to_string();

        let device_spec_rel = entry.device_spec.as_ref().with_context(|| {
            format!(
                "entry `{}`: Scheimpflug planar acceptance needs `device_spec` \
                 (the ADR 0011 spec.json sidecar)",
                entry.id
            )
        })?;
        let device_spec_path = resolve_against(&entry.data_root, device_spec_rel);
        let device = DeviceSpec::from_path(&device_spec_path)
            .with_context(|| format!("load device spec {}", device_spec_path.display()))?;
        let seed = scheimpflug_seed(&device, &camera_id)
            .with_context(|| format!("derive seed for camera `{camera_id}` from device spec"))?;

        // ── Detection (manifest-driven, cached) ────────────────────────────
        progress(entry, format!("detecting views for {camera_id} (manifest)"));
        let detected = detect_planar(
            &spec,
            &entry.data_root,
            entry.board.as_ref(),
            &detection_cache(),
        )
        .with_context(|| format!("entry `{}`: manifest detection failed", entry.id))?;

        Ok(DetectedScheimpflugCamera {
            camera_id,
            dataset: detected.input,
            seed,
            detection: detected.detection,
            image_size: detected.image_sizes[0],
            detection_ms: detected.detection_ms,
        })
    }

    /// Result of [`solve_scheimpflug_seeded`].
    #[derive(Debug, Clone)]
    pub struct ScheimpflugSeededSolve {
        /// Session export.
        pub export: PlanarIntrinsicsExport,
        /// Seeded-init wall-clock time in milliseconds.
        pub init_ms: u64,
        /// Optimize wall-clock time in milliseconds.
        pub optimize_ms: u64,
    }

    /// Run the seeded Scheimpflug intrinsics init + optimize route (ADR
    /// 0022) on an already-detected `dataset` with the given `seed`. The
    /// solve configuration mirrors the hard-gated private examples: both
    /// tilts free, `k3` + tangential fixed (default), Huber(1.0), 120
    /// iterations.
    ///
    /// `label` only affects progress logging (`entry.id` from
    /// [`run_scheimpflug_intrinsics`], or a per-cell label from the
    /// basin study).
    pub fn solve_scheimpflug_seeded(
        dataset: PlanarDataset,
        seed: IntrinsicsManualInit,
        label: &str,
        backend: SolverBackend,
    ) -> Result<ScheimpflugSeededSolve> {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(dataset).context("set_input failed")?;
        let mut config = PlanarIntrinsicsConfig::default();
        config.camera.model = tilted_bc5_model();
        config.solver.max_iters = 120;
        config.solver.backend = backend;
        config.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
        session.set_config(config).context("set_config failed")?;

        progress_label(label, "seeded scheimpflug init");
        let init_start = Instant::now();
        step_init_with_seed(&mut session, seed, None).context("seeded init failed")?;
        let init_ms = ms_since(init_start);

        progress_label(label, "optimizing scheimpflug intrinsics");
        let opt_start = Instant::now();
        step_optimize(&mut session, None).context("optimize failed")?;
        let optimize_ms = ms_since(opt_start);

        let export = session.export().context("session.export failed")?;
        Ok(ScheimpflugSeededSolve {
            export,
            init_ms,
            optimize_ms,
        })
    }

    /// Run the seeded single-camera Scheimpflug intrinsics **acceptance
    /// route** (ADR 0011) for `entry` and build a [`BenchRecord`].
    ///
    /// Detects via [`detect_scheimpflug_seeded_input`] and solves via
    /// [`solve_scheimpflug_seeded`] — see both for the manifest/seed and
    /// solve-configuration details.
    ///
    /// The from-scratch multistart stays in [`diagnose_intrinsics`]
    /// (informational); this seeded path is what
    /// `calib-bench accept` gates on.
    pub fn run_scheimpflug_intrinsics(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.is_scheimpflug_planar(),
            "run_scheimpflug_intrinsics called for {:?} without a Scheimpflug sensor",
            entry.problem
        );
        let detected = detect_scheimpflug_seeded_input(entry)?;
        let image_sizes = [detected.image_size];
        let solve = solve_scheimpflug_seeded(
            detected.dataset,
            detected.seed,
            &entry.id,
            entry.solver_backend(),
        )
        .with_context(|| format!("entry `{}`: seeded solve failed", entry.id))?;
        let export = solve.export;

        // ── Fit metrics (bench-recomputed + export-reported) ───────────────
        let errors: Vec<f64> = export
            .per_feature_residuals
            .target
            .iter()
            .filter_map(|r| r.error_px)
            .collect();
        let overall = ReprojectionStats::from_errors(&errors);
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
            init_ok: true,
            converged: true,
            report: export.report.clone(),
        };
        let total_ms = solve
            .init_ms
            .saturating_add(solve.optimize_ms)
            .saturating_add(detected.detection_ms);
        let timing = Timing {
            init_ms: solve.init_ms,
            optimize_ms: solve.optimize_ms,
            total_ms,
            detection_ms: detected.detection_ms,
            stages: None,
        };

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "planar_intrinsics"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detected.detection),
            laser: None,
            robot_corrections: None,
            // Single-camera Scheimpflug intrinsics emits Fit-only metrics and no
            // camera artifact. The `camera_artifact("brown_conrady5", …)` path is
            // reached only by the rig runs, which stay Brown-Conrady even after the
            // Distortion-model selection (rig BA is BC5-typed).
            artifacts: None,
            delta_to_prior: None,
            timing,
            quality: Some(quality_report(entry, &export, &image_sizes)?),
            reproj_report: None,
            residual_sidecar: None,
        })
    }

    /// Every rig camera must see the target in at least three views.
    fn ensure_three_views_per_camera<M>(spec: &DatasetSpec, input: &RigDataset<M>) -> Result<()> {
        for (ci, camera) in spec.cameras.iter().enumerate() {
            let usable = input
                .views
                .iter()
                .filter(|v| v.obs.cameras[ci].is_some())
                .count();
            anyhow::ensure!(
                usable >= 3,
                "need >= 3 usable views for camera {ci} ({}), got {usable}",
                camera.id
            );
        }
        Ok(())
    }

    /// Run a multi-camera rig-extrinsics calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Detects the entry's manifest through the dataset runner and runs
    /// `step_intrinsics_init_all` → `step_intrinsics_optimize_all` →
    /// `step_rig_init` → `step_rig_optimize`. The bench-computed
    /// `Fit::overall` is recomputed from the export's per-camera per-feature
    /// residuals; the `reported_*` fields mirror the export so any divergence
    /// is visible.
    ///
    /// A view is kept when any camera detected the board (a camera that did
    /// not contributes no observation), and every camera needs at least 3
    /// usable views. A registry entry's images pair by filename token: the
    /// name after the per-camera tag (`Im_L_2.png` ↔ `Im_R_2.png`,
    /// `Cam1_<pose>.png` ↔ `Cam2_<pose>.png`).
    pub fn run_rig_extrinsics(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::RigExtrinsics,
            "run_rig_extrinsics called for {:?}",
            entry.problem
        );
        let spec = dataset_spec(entry)?;

        // ── Detection ──────────────────────────────────────────────────────
        progress(
            entry,
            format!("detecting target in {} cameras", spec.cameras.len()),
        );
        let detected = detect_rig_extrinsics(
            &spec,
            &entry.data_root,
            entry.board.as_ref(),
            &detection_cache(),
        )
        .with_context(|| format!("entry `{}`: detection failed", entry.id))?;
        let detection_ms = detected.detection_ms;
        progress(
            entry,
            format!("target detection finished in {detection_ms} ms"),
        );
        ensure_three_views_per_camera(&spec, &detected.input)?;

        // ── Calibration ────────────────────────────────────────────────────
        // Keep a copy for the multi-level reproj report (the dataset is moved
        // into the session below).
        let dataset_for_report = detected.input.clone();
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        let mut config = RigExtrinsicsConfig::default();
        config.solver.backend = entry.solver_backend();
        session.set_config(config).context("set_config failed")?;
        session
            .set_input(detected.input)
            .context("set_input failed")?;

        progress(entry, "initializing per-camera intrinsics");
        let init_start = Instant::now();
        let init_ok = step_intrinsics_init_all(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init_all failed for {}", entry.id);
        let init_ms = ms_since(init_start);

        progress(entry, "optimizing intrinsics and rig extrinsics");
        let s = Instant::now();
        let _ = step_intrinsics_optimize_all(&mut session, None)
            .context("step_intrinsics_optimize_all failed")?;
        let intrinsics_optimize_ms = ms_since(s);
        let s = Instant::now();
        let _ = step_rig_init(&mut session, None).context("step_rig_init failed")?;
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
        let n_cam = spec.cameras.len();
        let (overall, per_camera) =
            split_target_residuals(&export.per_feature_residuals.target, n_cam);
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
            detection: Some(detected.detection),
            laser: None,
            robot_corrections: None,
            artifacts: Some(artifacts_from_rig_extrinsics(&spec.cameras, &export)),
            delta_to_prior: None,
            timing,
            quality: Some(quality_report(entry, &export, &detected.image_sizes)?),
            reproj_report,
            residual_sidecar,
        })
    }

    /// Pool the finite per-feature errors of `residuals` (overall) and split
    /// them by camera.
    fn split_target_residuals(
        residuals: &[TargetFeatureResidual],
        n_cam: usize,
    ) -> (ReprojectionStats, Vec<ReprojectionStats>) {
        let mut per_cam_errs: Vec<Vec<f64>> = vec![Vec::new(); n_cam];
        let mut all_errors: Vec<f64> = Vec::new();
        for r in residuals {
            if let Some(e) = r.error_px {
                all_errors.push(e);
                if r.camera < n_cam {
                    per_cam_errs[r.camera].push(e);
                }
            }
        }
        (
            ReprojectionStats::from_errors(&all_errors),
            per_cam_errs
                .iter()
                .map(|errs| ReprojectionStats::from_errors(errs))
                .collect(),
        )
    }

    /// Run a single-camera hand-eye calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Detects the entry's manifest through the dataset runner (image `i`, in
    /// natural filename order, pairs with pose row `i`; views with no board
    /// are dropped), runs `step_intrinsics_init` → `step_intrinsics_optimize`
    /// → `step_handeye_init` → `step_handeye_optimize`, then exports.
    pub fn run_single_cam_handeye(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::SingleCamHandeye,
            "run_single_cam_handeye called for {:?}",
            entry.problem
        );
        let spec = dataset_spec(entry)?;
        let camera_id = single_camera_id(&spec)?;

        // ── Detection ──────────────────────────────────────────────────────
        progress(entry, format!("detecting pose images for {camera_id}"));
        let detected = detect_single_cam_handeye(
            &spec,
            &entry.data_root,
            entry.board.as_ref(),
            &detection_cache(),
        )
        .with_context(|| format!("entry `{}`: detection failed", entry.id))?;
        let detection_ms = detected.detection_ms;
        progress(
            entry,
            format!("target detection finished in {detection_ms} ms"),
        );
        anyhow::ensure!(
            detected.input.views.len() >= 3,
            "need >= 3 detected views for single-cam hand-eye, got {}",
            detected.input.views.len()
        );

        // ── Calibration ────────────────────────────────────────────────────
        // Keep a copy of the detected views for the multi-level reproj report
        // (the input is moved into the session below).
        let views_for_report = detected.input.views.clone();
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        let mut config = SingleCamHandeyeConfig::default();
        if let Some(overrides) = &entry.single_cam_handeye {
            overrides.apply_to(&mut config);
        }
        config.solver.backend = entry.solver_backend();
        let robot_rot_sigma = config.robot_poses.rot_sigma;
        let robot_trans_sigma = config.robot_poses.trans_sigma;
        session
            .set_config(config)
            .context("set single_cam_handeye config failed")?;
        session
            .set_input(detected.input)
            .context("set_input failed")?;

        progress(entry, "initializing single-camera intrinsics");
        let init_start = Instant::now();
        let init_ok = step_intrinsics_init(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init failed for {}", entry.id);
        let init_ms = ms_since(init_start);

        progress(entry, "optimizing intrinsics and hand-eye");
        let opt_start = Instant::now();
        let _ = step_intrinsics_optimize(&mut session, None)
            .context("step_intrinsics_optimize failed")?;
        let _ = step_handeye_init(&mut session, None).context("step_handeye_init failed")?;
        let _ =
            step_handeye_optimize(&mut session, None).context("step_handeye_optimize failed")?;
        let optimize_ms = ms_since(opt_start);

        let export = session.export().context("session.export failed")?;

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
        // pose) + Handeye level (board pose through robot + hand-eye chain). The
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
            detection: Some(detected.detection),
            laser: None,
            robot_corrections,
            artifacts: Some(artifacts_from_single_cam(camera_id, &export)),
            delta_to_prior: None,
            timing,
            quality: Some(quality_report(entry, &export, &detected.image_sizes)?),
            reproj_report,
            residual_sidecar,
        })
    }

    /// Run a multi-camera rig hand-eye calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Generalizes the rig path to N cameras on a robot (DS8: two JAI BB-500
    /// and a Creative Senz3D, pinhole). The registry fields pair image `i` of
    /// every camera (natural filename order) with pose row `i`; a camera
    /// that did not detect the board in a view contributes no observation.
    /// Runs the full six-step rig hand-eye pipeline (intrinsics → rig →
    /// hand-eye) with default options (so robot-pose refinement is on), then
    /// exports. An entry with `laser` data adds the joint laserline stage.
    pub fn run_rig_handeye(entry: &BenchEntry) -> Result<BenchRecord> {
        anyhow::ensure!(
            entry.problem == ProblemKind::RigHandeye,
            "run_rig_handeye called for {:?}",
            entry.problem
        );
        let spec = dataset_spec(entry)?;

        // ── Detection ──────────────────────────────────────────────────────
        progress(
            entry,
            format!("detecting target in {} cameras", spec.cameras.len()),
        );
        let (detected, view_paths) = detect_rig_handeye(
            &spec,
            &entry.data_root,
            entry.board.as_ref(),
            &detection_cache(),
        )
        .with_context(|| format!("entry `{}`: detection failed", entry.id))?;
        let detection_ms = detected.detection_ms;
        progress(
            entry,
            format!("target detection finished in {detection_ms} ms"),
        );
        ensure_three_views_per_camera(&spec, &detected.input)?;

        let n_cam = spec.cameras.len();
        let laser_observations =
            extract_laser_observations(entry, &spec, &detected.input, &view_paths)?;
        let mut laser = laser_observations.as_ref().map(|obs| obs.metrics.clone());

        // ── Calibration ─────────────────────────────────────────────────────
        let mut config = RigHandeyeConfig::default();
        if let Some(overrides) = &entry.rig_handeye {
            overrides.apply_to(&mut config);
        }
        config.solver.backend = entry.solver_backend();
        if let Some(seed) = &entry.seed {
            let manual: RigIntrinsicsManualInit = serde_json::from_value(seed.0.clone())
                .with_context(|| {
                    format!("failed to parse rig hand-eye manual seed for {}", entry.id)
                })?;
            config.manual_init = Some(manual);
        }
        if config.manual_init.is_none() && entry.id == "rtv3d" {
            config.manual_init = Some(rtv3d_manual_intrinsics_seed(&spec));
            config.intrinsics.fix_tangential = Some(true);
        }
        let robot_rot_sigma = config.handeye_optimize.robot_poses.rot_sigma;
        let robot_trans_sigma = config.handeye_optimize.robot_poses.trans_sigma;
        let manual_intrinsics = config.manual_init.clone().unwrap_or_default();
        let dataset_for_report = detected.input.clone();
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session
            .set_config(config)
            .context("set rig_handeye config failed")?;
        session
            .set_input(detected.input)
            .context("set_input failed")?;

        progress(entry, "initializing per-camera intrinsics");
        let init_start = Instant::now();
        let init_ok =
            rh_intrinsics_init_all_with_seed(&mut session, manual_intrinsics, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init_all failed for {}", entry.id);
        let init_ms = ms_since(init_start);

        progress(entry, "optimizing intrinsics, rig, and hand-eye");
        let s = Instant::now();
        let _ = rh_intrinsics_optimize_all(&mut session, None)
            .context("step_intrinsics_optimize_all failed")?;
        let intrinsics_optimize_ms = ms_since(s);
        let s = Instant::now();
        let _ = rh_rig_init(&mut session, None).context("step_rig_init failed")?;
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
        let (overall, per_camera) =
            split_target_residuals(&export.per_feature_residuals.target, n_cam);
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
        // Handeye level (board pose through the robot + hand-eye chain).
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
            detection: Some(detected.detection),
            laser,
            robot_corrections,
            artifacts: Some(artifacts_from_rig_handeye(&spec.cameras, &export)),
            delta_to_prior: None,
            timing,
            quality: Some(quality_report(entry, &export, &detected.image_sizes)?),
            reproj_report,
            residual_sidecar,
        })
    }

    /// Diagnose per-camera Scheimpflug intrinsics only: detect the calibration
    /// target, run staged multistart single-camera solves, and report raw
    /// all-corner reprojection distributions.
    ///
    /// **Informational only:** the 0.4 px threshold below
    /// grades the *from-scratch* multistart floor, which is an experimental
    /// path. The acceptance gate is the seeded official
    /// route (`calib-bench accept`, per-entry `AcceptGate`); a `pass:
    /// false` in this report is a data point, not a failure of the dataset.
    pub fn diagnose_intrinsics(entry: &BenchEntry) -> Result<IntrinsicsDiagnoseReport> {
        const GATE_PX: f64 = 0.4;

        let mut spec = dataset_spec(entry)?;
        anyhow::ensure!(
            !spec.cameras.is_empty(),
            "diagnose intrinsics needs at least one camera"
        );
        let chess_corners = intrinsics_chess_corners(&mut spec);

        progress(
            entry,
            format!(
                "diagnosing Scheimpflug intrinsics for {} cameras",
                spec.cameras.len()
            ),
        );

        let cache = detection_cache();
        let mut camera_reports = Vec::with_capacity(spec.cameras.len());
        for camera in 0..spec.cameras.len() {
            let camera_id = spec.cameras[camera].id.clone();
            progress(
                entry,
                format!("detecting intrinsic dataset for {camera_id}"),
            );
            let detected = detect_intrinsics_camera(entry, &spec, camera, &cache)?;
            anyhow::ensure!(
                detected.dataset.num_views() >= 3,
                "need >= 3 detected views for {camera_id}, got {}",
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
                    "{camera_id}: solving {} centered Scheimpflug seeds",
                    seeds.len()
                ),
            );
            let gate_solve = solve_centered_multistart(&detected.dataset, &seeds, GATE_PX)
                .with_context(|| format!("centered multistart failed for {camera_id}"))?;
            let mut diagnostic_cases = Vec::new();
            if !gate_solve.report.passed_gate {
                progress(
                    entry,
                    format!(
                        "{camera_id}: gate mean {:.4} px; running diagnostic model variants",
                        gate_solve.report.stats.mean
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
            chess_threshold_abs: chess_corners.threshold_value,
            cameras: camera_reports,
            pass,
        })
    }

    /// The intrinsics diagnostic detects with a ChESS threshold of 30 unless
    /// the manifest sets one; returns the corner options it detects with.
    fn intrinsics_chess_corners(spec: &mut DatasetSpec) -> ChessCornersDetectorSpec {
        let detector = spec.detector.get_or_insert_with(DetectorSpec::default);
        let chess = detector.chess_corners.get_or_insert_with(Default::default);
        chess.threshold_value.get_or_insert(30.0);
        *chess
    }

    fn detect_intrinsics_camera(
        entry: &BenchEntry,
        spec: &DatasetSpec,
        camera: usize,
        cache: &dyn DetectionCache,
    ) -> Result<DetectedIntrinsicsCamera> {
        let camera_spec = camera_spec(spec, camera);
        let camera_id = camera_spec.cameras[0].id.clone();
        let built = build_planar_input(&camera_spec, &entry.data_root, cache, false)
            .with_context(|| format!("detection failed for {camera_id}"))?;
        let views = &built.input.views;
        let features_detected = views.iter().map(|v| v.obs.len()).sum::<usize>();

        let expected_size = entry
            .cameras
            .iter()
            .find(|c| c.id == camera_id)
            .and_then(|c| c.expected_size);
        let image_size = match camera_spec.cameras[0].roi_xywh {
            Some([_, _, w, h]) => [w, h],
            None => match built.view_paths.first() {
                Some(path) => {
                    let (w, h) = image::image_dimensions(path)
                        .with_context(|| format!("read the size of {}", path.display()))?;
                    [w, h]
                }
                None => expected_size.context("could not determine image size")?,
            },
        };
        let roi_local_coordinates = views_are_roi_local(views, image_size);
        progress(
            entry,
            format!(
                "{camera_id} detections: {}/{} images, {features_detected} features",
                built.usable_views, built.total_views
            ),
        );
        Ok(DetectedIntrinsicsCamera {
            camera_id,
            image_size,
            images_total: built.total_views,
            images_used: built.usable_views,
            features_detected,
            roi_local_coordinates,
            dataset: built.input,
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

    /// The camera model of the bench's tilted single-camera solves: pinhole,
    /// Brown-Conrady, Scheimpflug sensor.
    fn tilted_bc5_model() -> CameraModelSpec {
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug)
    }

    /// Planar-solve parameters of a tilted Brown-Conrady camera.
    fn tilted_bc5_params(
        intrinsics: FxFyCxCy<f64>,
        distortion: BrownConrady5<f64>,
        sensor: ScheimpflugParams,
        poses: Vec<Iso3>,
    ) -> Result<PlanarIntrinsicsParams> {
        PlanarIntrinsicsParams::new(
            CameraParams {
                projection: ProjectionParams::Pinhole,
                distortion: DistortionParams::BrownConrady5 { params: distortion },
                sensor: SensorParams::Scheimpflug { params: sensor },
                intrinsics,
            },
            poses,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// The tilt of a planar solve's camera (zero for an untilted one).
    fn planar_tilt(params: &PlanarIntrinsicsParams) -> ScheimpflugParams {
        match params.camera.sensor {
            SensorParams::Scheimpflug { params } => params,
            _ => ScheimpflugParams::default(),
        }
    }

    /// The fix lists of a tilted Brown-Conrady camera: the named intrinsics
    /// and distortion terms held, the tilt free.
    fn tilted_bc5_fix(intrinsics: &[&str], distortion: &[&str]) -> CameraFix {
        let names = |n: &[&str]| Some(n.iter().map(|s| s.to_string()).collect());
        CameraFix {
            intrinsics: names(intrinsics),
            projection: Some(Vec::new()),
            distortion: names(distortion),
            sensor: Some(Vec::new()),
        }
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
        let stage_a = optimize_planar_intrinsics(
            dataset,
            &initial,
            PlanarIntrinsicsSolveOptions {
                robust_loss: RobustLoss::None,
                fix: CameraFix::all(tilted_bc5_model()),
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

        let centered_opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 1.0 },
            fix: tilted_bc5_fix(&["cx", "cy"], &["k3", "p1", "p2"]),
            fix_poses: Vec::new(),
            bounds: None,
        };
        let stage_b = optimize_planar_intrinsics(
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

        let stage_c = optimize_planar_intrinsics(
            dataset,
            &stage_b.params,
            PlanarIntrinsicsSolveOptions {
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
    ) -> Result<PlanarIntrinsicsParams> {
        let intrinsics = FxFyCxCy {
            fx: seed.focal_px,
            fy: seed.focal_px,
            cx: seed.cx,
            cy: seed.cy,
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
        tilted_bc5_params(intrinsics, distortion, sensor, poses)
    }

    fn initial_poses_for_dataset(dataset: &PlanarDataset, k: &FxFyCxCy<f64>) -> Result<Vec<Iso3>> {
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
                let h = vision_calibration::geometry::homography::dlt_homography(&world2d, obs_2d)
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
                fix: tilted_bc5_fix(&[], &["k3", "p1", "p2"]),
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
                fix: tilted_bc5_fix(&["cx", "cy"], &["p1", "p2"]),
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
                fix: tilted_bc5_fix(&["cx", "cy"], &["k3"]),
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
        initial: &PlanarIntrinsicsParams,
        gate_px: f64,
        config: IntrinsicsVariantConfig,
    ) -> Result<StagedIntrinsicsSolve> {
        let opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 1.0 },
            fix: config.fix,
            fix_poses: Vec::new(),
            bounds: None,
        };
        let robust = optimize_planar_intrinsics(
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
        let polished = optimize_planar_intrinsics(
            dataset,
            &robust.params,
            PlanarIntrinsicsSolveOptions {
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
        estimate: PlanarIntrinsicsEstimate,
        note: Option<String>,
    ) -> Result<StagedIntrinsicsSolve> {
        // Bench staged solves are Brown-Conrady (the committed defaults);
        // `IntrinsicsParamReport` is BC5-shaped, so read the concrete coefficients.
        let distortion = estimate.params.distortion().unwrap_or_default();
        let tilt = planar_tilt(&estimate.params);
        let camera = Camera::new(
            Pinhole,
            distortion,
            tilt.compile(),
            estimate.params.camera.intrinsics,
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
            fx: estimate.params.camera.intrinsics.fx,
            fy: estimate.params.camera.intrinsics.fy,
            cx: estimate.params.camera.intrinsics.cx,
            cy: estimate.params.camera.intrinsics.cy,
            k1: distortion.k1,
            k2: distortion.k2,
            k3: distortion.k3,
            p1: distortion.p1,
            p2: distortion.p2,
            tau_x: tilt.tilt_x,
            tau_y: tilt.tilt_y,
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

    /// Profile target detection and laser extraction without running a solver.
    ///
    /// This intentionally keeps the diagnostic scoped to I/O + detector stages:
    /// if a puzzle dataset is already slow here, the full calibration path will
    /// not reach initialization quickly enough to be a useful first diagnostic.
    /// Each camera's images go through the dataset runner uncached, so every
    /// image is detected.
    pub fn profile_dataset_stages(
        entry: &BenchEntry,
        max_images_per_camera: Option<usize>,
    ) -> Result<DatasetStageProfile> {
        let spec = dataset_spec(entry)?;
        let target_start = Instant::now();
        let mut target_per_camera = Vec::with_capacity(spec.cameras.len());

        for camera in 0..spec.cameras.len() {
            let camera_spec = camera_spec(&spec, camera);
            let timing = TimingCache::new(max_images_per_camera);
            let (views, images_total) =
                match build_planar_input(&camera_spec, &entry.data_root, &timing, false) {
                    Ok(built) => (built.input.views, built.total_views),
                    // A camera that found no board is a profile, not a failure.
                    Err(RunError::InsufficientUsableViews { total, .. }) => (Vec::new(), total),
                    Err(e) => {
                        return Err(anyhow::Error::new(e).context(format!(
                            "profile detection for {}",
                            camera_spec.cameras[0].id
                        )));
                    }
                };
            let durations: Vec<u64> = timing.durations().into_iter().map(ms).collect();
            let stage_ms = durations.iter().sum::<u64>();
            target_per_camera.push(ProfileCameraStat {
                camera_id: camera_spec.cameras[0].id.clone(),
                images_total,
                images_profiled: durations.len(),
                images_used: views.len(),
                points_total: views.iter().map(|v| v.obs.len()).sum(),
                points_max: views.iter().map(|v| v.obs.len()).max().unwrap_or(0),
                load_ms: None,
                stage_ms,
                mean_stage_ms: mean_ms(stage_ms, durations.len()),
                max_stage_ms: durations.iter().copied().max().unwrap_or(0),
            });
        }

        let laser_extraction =
            profile_laser_extraction(entry, &spec, max_images_per_camera.unwrap_or(usize::MAX))?;

        Ok(DatasetStageProfile {
            dataset_id: entry.id.clone(),
            problem: entry.problem,
            camera_count: spec.cameras.len(),
            max_images_per_camera,
            target_detection: TargetDetectionProfile {
                detector: target_kind(&spec.target).to_string(),
                expected_features_per_board: entry.board.as_ref().map_or(0, board_feature_count),
                total_ms: ms_since(target_start),
                per_camera: target_per_camera,
            },
            laser_extraction,
        })
    }

    /// The manifest name of a target's kind.
    fn target_kind(target: &TargetSpec) -> &'static str {
        match target {
            TargetSpec::Chessboard { .. } => "chessboard",
            TargetSpec::Charuco { .. } => "charuco",
            TargetSpec::Puzzleboard { .. } => "puzzleboard",
            TargetSpec::Ringgrid { .. } => "ringgrid",
        }
    }

    fn mean_ms(total_ms: u64, count: usize) -> f64 {
        if count == 0 {
            0.0
        } else {
            total_ms as f64 / count as f64
        }
    }

    fn progress(entry: &BenchEntry, message: impl std::fmt::Display) {
        progress_label(&entry.id, message);
    }

    /// Same as `progress` but keyed by an arbitrary label rather than a
    /// [`BenchEntry`] — used by [`solve_scheimpflug_seeded`], whose caller
    /// may be a plain dataset id (`run_scheimpflug_intrinsics`) or a
    /// per-cell label (the convergence-basin study).
    pub fn progress_label(label: &str, message: impl std::fmt::Display) {
        if std::env::var_os("CALIB_BENCH_QUIET").is_none() {
            eprintln!("[calib-bench:{label}] {message}");
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

    #[cfg(not(feature = "laser"))]
    fn profile_laser_extraction(
        entry: &BenchEntry,
        _spec: &DatasetSpec,
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
        spec: &DatasetSpec,
        profile_limit: usize,
    ) -> Result<Option<LaserExtractionProfile>> {
        if entry.laser.is_none() {
            return Ok(None);
        }
        let paths: Vec<PathBuf> = laser_frames(entry)?.into_values().collect();
        let selected: Vec<_> = paths.iter().take(profile_limit).collect();
        let total_start = Instant::now();
        let mut per_camera = Vec::with_capacity(spec.cameras.len());
        for camera in &spec.cameras {
            let mut images_used = 0usize;
            let mut points_total = 0usize;
            let mut points_max = 0usize;
            let mut load_ms = 0u64;
            let mut stage_ms = 0u64;
            let mut max_stage_ms = 0u64;

            for path in &selected {
                let load_start = Instant::now();
                let img = load_image(path)?;
                let img = crop_to_roi(&img, camera.roi_xywh);
                load_ms = load_ms.saturating_add(ms_since(load_start));

                let stage_start = Instant::now();
                let point_count = extract_laser_pixels(&img).len();
                let elapsed = ms_since(stage_start);
                stage_ms = stage_ms.saturating_add(elapsed);
                max_stage_ms = max_stage_ms.max(elapsed);
                if point_count > 0 {
                    images_used += 1;
                    points_total += point_count;
                    points_max = points_max.max(point_count);
                }
            }

            per_camera.push(ProfileCameraStat {
                camera_id: camera.id.clone(),
                images_total: paths.len(),
                images_profiled: selected.len(),
                images_used,
                points_total,
                points_max,
                load_ms: Some(load_ms),
                stage_ms,
                mean_stage_ms: mean_ms(stage_ms, selected.len()),
                max_stage_ms,
            });
        }

        Ok(Some(LaserExtractionProfile {
            enabled: true,
            total_ms: ms_since(total_start),
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

    /// Crop an image to a camera's region of interest `[x, y, w, h]` (a tiled
    /// multi-camera frame holds every camera side by side), or keep it whole.
    #[cfg(feature = "laser")]
    fn crop_to_roi(img: &image::DynamicImage, roi: Option<[u32; 4]>) -> image::DynamicImage {
        match roi {
            Some([x, y, w, h]) => img.crop_imm(x, y, w, h),
            None => img.clone(),
        }
    }

    #[cfg(feature = "laser")]
    fn load_image(path: &Path) -> Result<image::DynamicImage> {
        image::open(path).with_context(|| format!("failed to load {}", path.display()))
    }

    fn rtv3d_manual_intrinsics_seed(spec: &DatasetSpec) -> RigIntrinsicsManualInit {
        let (cx, cy) = spec
            .cameras
            .first()
            .and_then(|camera| camera.roi_xywh)
            .map(|[_x, _y, w, h]| (w as f64 * 0.5, h as f64 * 0.5))
            .unwrap_or((360.0, 270.0));
        let n = spec.cameras.len();
        let mut seed = RigIntrinsicsManualInit::default();
        seed.per_cam_intrinsics = Some(vec![
            FxFyCxCy {
                fx: 2000.0,
                fy: 2000.0,
                cx,
                cy,
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
        _spec: &DatasetSpec,
        _input: &RigDataset<RobotPoseMeta>,
        _view_paths: &RigViewPaths,
    ) -> Result<Option<LaserObservationSet>> {
        if entry.laser.is_some() {
            anyhow::bail!(
                "dataset '{}' declares laser data; rebuild with --features 'tier-b laser'",
                entry.id
            );
        }
        Ok(None)
    }

    /// Laser observations of the detected rig views: each view whose target
    /// frame has a laser frame (paired through the snap list) contributes the
    /// laser pixels every camera extracts from its region of that frame.
    #[cfg(feature = "laser")]
    fn extract_laser_observations(
        entry: &BenchEntry,
        spec: &DatasetSpec,
        input: &RigDataset<RobotPoseMeta>,
        view_paths: &RigViewPaths,
    ) -> Result<Option<LaserObservationSet>> {
        if entry.laser.is_none() {
            return Ok(None);
        }
        let laser_frames = laser_frames(entry)?;

        progress(
            entry,
            "extracting laser observations for the joint bundle adjustment",
        );
        let n_cam = spec.cameras.len();
        let n_views = input.views.len();
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

        for (view_idx, (view, paths)) in input.views.iter().zip(view_paths).enumerate() {
            // Every camera of a tiled rig reads the same target frame.
            let target_name = paths
                .iter()
                .flatten()
                .next()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let Some(laser_path) = laser_frames.get(target_name) else {
                continue;
            };
            candidate_views += 1;
            progress_images(entry, "laser", "all", candidate_views - 1, n_views);
            let img = load_image(laser_path)?;
            let mut laser_pixels = Vec::with_capacity(n_cam);
            let mut any_laser = false;
            for (cam_idx, camera) in spec.cameras.iter().enumerate() {
                let cam_start = Instant::now();
                let tile = crop_to_roi(&img, camera.roi_xywh);
                let pixels = extract_laser_pixels(&tile);
                per_cam_ms[cam_idx] = per_cam_ms[cam_idx].saturating_add(ms_since(cam_start));
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
                    cameras: view.obs.cameras.clone(),
                    laser_pixels,
                });
                laser_robot_poses.push(view.meta.base_se3_gripper);
                laser_view_indices.push(view_idx);
            }
        }

        if views.is_empty() {
            return Ok(None);
        }

        let per_camera = spec
            .cameras
            .iter()
            .enumerate()
            .map(|(cam_idx, camera)| LaserCamStat {
                camera_id: camera.id.clone(),
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
                extract_ms: ms_since(total_start),
            },
        }))
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

    /// The laser frame of each target frame, by target file name: the
    /// `double_snap` entries of the entry's snap-list pose file, which name
    /// both frames of a robot pose.
    #[cfg(feature = "laser")]
    fn laser_frames(entry: &BenchEntry) -> Result<std::collections::BTreeMap<String, PathBuf>> {
        let pose_src = entry
            .robot_poses
            .as_ref()
            .filter(|src| src.format == "snap_list_json")
            .with_context(|| {
                format!(
                    "entry `{}` declares laser data, which pairs with target frames through \
                     a snap_list_json pose file",
                    entry.id
                )
            })?;
        let path = entry.data_root.join(&pose_src.path);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read laser pose list {}", path.display()))?;
        let entries: Vec<LaserPoseEntry> = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse laser pose list {}", path.display()))?;
        let frames: std::collections::BTreeMap<String, PathBuf> = entries
            .into_iter()
            .filter(|snap| snap.snap_type.as_deref() == Some("double_snap"))
            .filter_map(|snap| Some((snap.target_image?, snap.laser_image?)))
            .map(|(target, laser)| {
                let target_name = Path::new(&target)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&target)
                    .to_string();
                (target_name, entry.data_root.join(laser))
            })
            .collect();
        anyhow::ensure!(
            !frames.is_empty(),
            "entry `{}`: pose list {} has no double_snap entry naming a laser frame",
            entry.id,
            path.display()
        );
        Ok(frames)
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
                "laserline + joint BA over {} laser views",
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
        laser_cfg.solver.max_iters = 200;
        laser_cfg.solver.backend = entry.solver_backend();
        laser_cfg.solver.verbosity = 0;
        laser_cfg.laser_residual_type = LaserlineResidualType::PointToPlane;
        laser_session
            .set_config(laser_cfg)
            .context("set laserline config failed")?;
        run_rig_laserline_device_calibration(&mut laser_session)
            .context("RigLaserlineDevice stage failed")?;
        let laser_export = laser_session
            .export()
            .context("export laserline stage failed")?;

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
            HandeyeMode::EyeInHand => rig_export
                .gripper_se3_rig
                .context("EyeInHand rig export missing gripper_se3_rig")?,
            HandeyeMode::EyeToHand => rig_export
                .rig_se3_base
                .context("EyeToHand rig export missing rig_se3_base")?,
        };
        let target_ref = match rig_export.handeye_mode {
            HandeyeMode::EyeInHand => rig_export
                .base_se3_target
                .context("EyeInHand rig export missing base_se3_target")?,
            HandeyeMode::EyeToHand => rig_export
                .gripper_se3_target
                .context("EyeToHand rig export missing gripper_se3_target")?,
        };
        let joint_initial = RigHandeyeLaserlineParams {
            cameras: rig_export.cameras.clone(),
            cam_to_rig: rig_export.cam_se3_rig.iter().map(|t| t.inverse()).collect(),
            handeye,
            target_ref,
            planes_cam: laser_export.laser_planes_cam.clone(),
        };

        // Intrinsics and the low-order radial terms free; k3, the tangential
        // terms and any sensor tilt held.
        let fix = rig_export
            .cameras
            .iter()
            .map(|camera| {
                let model = camera
                    .spec()
                    .context("rig camera with a general homography sensor")?;
                let held = |names: &[&str], held: &[&str]| -> Option<Vec<String>> {
                    Some(
                        names
                            .iter()
                            .filter(|n| held.contains(n))
                            .map(|n| n.to_string())
                            .collect(),
                    )
                };
                Ok(CameraFix {
                    intrinsics: Some(Vec::new()),
                    projection: Some(Vec::new()),
                    distortion: held(model.distortion.param_names(), &["k3", "p1", "p2"]),
                    sensor: held(model.sensor.param_names(), &["tilt_x", "tilt_y"]),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let joint_opts = RigHandeyeLaserlineSolveOptions {
            laser_residual_type: LaserlineResidualType::PointToPlane,
            laser_weight: 1.0e4,
            calib_weight: 1.0,
            refine_robot_poses: true,
            robot_rot_sigma,
            robot_trans_sigma,
            initial_robot_deltas,
            fix,
            fix_extrinsics: (0..n_cam).map(|i| i == 0).collect(),
            ..Default::default()
        };
        let backend_opts = BackendSolveOptions {
            backend: entry.solver_backend(),
            max_iters: 30,
            verbosity: 0,
            ..Default::default()
        };
        let joint_est =
            optimize_rig_handeye_laserline(joint_dataset, joint_initial, joint_opts, backend_opts)
                .context("final optimize_rig_handeye_laserline failed")?;
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
            HandeyeMode::EyeInHand => {
                let gripper_se3_rig = export
                    .gripper_se3_rig
                    .context("EyeInHand rig export missing gripper_se3_rig")?;
                let base_se3_target = export
                    .base_se3_target
                    .context("EyeInHand rig export missing base_se3_target")?;
                Ok(gripper_se3_rig.inverse() * base_se3_gripper.inverse() * base_se3_target)
            }
            HandeyeMode::EyeToHand => {
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

    fn artifacts_from_planar(
        camera_id: &str,
        export: &PlanarIntrinsicsExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: vec![params_artifact(camera_id, &export.params.camera)],
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
            cameras: vec![params_artifact(camera_id, &export.camera)],
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
        cameras: &[CameraSource],
        export: &RigExtrinsicsExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: camera_artifacts(cameras, &export.cameras),
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
        cameras: &[CameraSource],
        export: &RigHandeyeExport,
    ) -> CalibrationArtifacts {
        let mut artifacts = CalibrationArtifacts {
            spatial_unit: "mm".to_string(),
            angle_unit: "deg".to_string(),
            cameras: camera_artifacts(cameras, &export.cameras),
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

    /// Artifacts of rig cameras. The artifact format is Brown-Conrady: a
    /// camera of another distortion model reports zero coefficients under
    /// its own model name.
    fn camera_artifacts(sources: &[CameraSource], cameras: &[CameraParams]) -> Vec<CameraArtifact> {
        cameras
            .iter()
            .enumerate()
            .map(|(idx, camera)| params_artifact(camera_id(sources, idx), camera))
            .collect()
    }

    /// The artifact of a camera in any model: its Brown-Conrady terms (zero
    /// for another distortion model, which `distortion_model` names) and its
    /// tilt.
    fn params_artifact(camera_id: &str, camera: &CameraParams) -> CameraArtifact {
        let dist = match &camera.distortion {
            DistortionParams::BrownConrady5 { params } => *params,
            _ => BrownConrady5::default(),
        };
        let sensor = match &camera.sensor {
            SensorParams::Scheimpflug { params } => Some(params),
            _ => None,
        };
        let mut artifact = camera_artifact(
            camera_id,
            &make_pinhole_camera(camera.intrinsics, dist),
            sensor,
        );
        if let Some(model) = camera.spec() {
            artifact.distortion_model = serde_json::to_value(model.distortion)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or(artifact.distortion_model);
        }
        artifact
    }

    fn camera_id(sources: &[CameraSource], index: usize) -> &str {
        sources
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
            camera_matrix_px: [[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]],
            intrinsics_px: IntrinsicsArtifact {
                fx: k.fx,
                fy: k.fy,
                cx: k.cx,
                cy: k.cy,
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
        use nalgebra::{Translation3, UnitQuaternion, Vector3};
        use vision_calibration_core::{CameraProject, CorrespondenceView, Pt3};

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
            let k = FxFyCxCy {
                fx: 1950.0,
                fy: 1900.0,
                cx: 360.0,
                cy: 270.0,
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
