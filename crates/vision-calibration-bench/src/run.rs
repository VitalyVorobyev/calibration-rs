//! Run a calibration problem end-to-end and return a [`BenchRecord`] with
//! captured metrics.
//!
//! Tier-A would replay a frozen fixture through the math/serde path; that lands
//! in a later phase. The Tier-B path here globs a camera's images, detects the
//! board in each, builds the problem `Input`, runs the facade pipeline, and
//! captures both bench-recomputed and self-reported reprojection metrics so any
//! divergence between them is visible in the record.

#[cfg(feature = "tier-b")]
pub use tier_b::{
    run_planar_intrinsics, run_rig_extrinsics, run_rig_handeye, run_single_cam_handeye,
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
        planar_intrinsics_report, rig_extrinsics_report, rig_handeye_report,
        single_cam_handeye_report,
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
        RigHandeyeConfig, RigHandeyeExport, RigHandeyeProblem,
        step_handeye_init as rh_handeye_init, step_handeye_optimize as rh_handeye_optimize,
        step_intrinsics_init_all as rh_intrinsics_init_all,
        step_intrinsics_optimize_all as rh_intrinsics_optimize_all, step_rig_init as rh_rig_init,
        step_rig_optimize as rh_rig_optimize,
    };
    use vision_calibration::session::CalibrationSession;
    use vision_calibration::single_cam_handeye::{
        HandeyeMeta, SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeInput,
        SingleCamHandeyeProblem, SingleCamHandeyeView, step_handeye_init, step_handeye_optimize,
        step_intrinsics_init, step_intrinsics_optimize,
    };
    use vision_calibration_core::{
        FeatureResidualHistogram, PinholeCamera, PlanarDataset, ReprojectionStats,
        ScheimpflugParams, View,
    };
    use vision_calibration_optim::SolveReport;
    #[cfg(feature = "laser")]
    use vision_metrology::{
        ColAccess, Edge1DConfig, ImageView, LaserExtractConfig, LaserExtractor, ScanAxis,
    };

    use crate::detect::{
        DetectorKind, charuco_params_for, detect_chessboard_view, glob_sorted_images, load_image,
        puzzleboard_params_for,
    };
    #[cfg(feature = "laser")]
    use crate::record::LaserCamStat;
    use crate::record::{
        BENCH_SCHEMA_VERSION, BenchRecord, CalibrationArtifacts, CameraArtifact, Convergence,
        Detection, DetectionStat, DistortionArtifact, Fit, Ident, IntrinsicsArtifact, LaserMetrics,
        ResidualSidecar, RobotCorrectionSummary, ScheimpflugArtifact, Timing, TransformArtifact,
        compact_reproj_report,
    };
    use crate::registry::{BenchEntry, BoardGeometry, CameraLayout, ProblemKind};

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
        for (image_idx, path) in paths.iter().enumerate() {
            progress_images(entry, "detect", &cam.id, image_idx, paths.len());
            let img = load_image(path)?;
            match detect_chessboard_view(
                &img,
                board.rows,
                board.cols,
                board.cell_size_m,
                board.strict_grid,
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
        let detector = detector_for(board)?;

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
        let opt_start = Instant::now();
        let _ = step_intrinsics_optimize_all(&mut session, None)
            .context("step_intrinsics_optimize_all failed")?;
        let _ = step_rig_init(&mut session).context("step_rig_init failed")?;
        let _ = step_rig_optimize(&mut session, None).context("step_rig_optimize failed")?;
        let optimize_ms = opt_start.elapsed().as_millis() as u64;

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
            let detector = detector_for(board)?;
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
        };

        // Hierarchical reprojection report: Intrinsic floor (free per-view PnP
        // pose) + HandEye level (board pose through robot + hand-eye chain). The
        // gap between the two localizes the kuka-style "why is it 1.2 px" error.
        let (reproj_report, residual_sidecar) = split_reproj_report(
            entry,
            single_cam_handeye_report(&export, &views_for_report).ok(),
        );
        let robot_corrections = export
            .robot_deltas
            .as_deref()
            .and_then(RobotCorrectionSummary::from_deltas);

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
        let detector = detector_for(board)?;
        let robot_poses = load_robot_poses_for(pose_src, &entry.data_root.join(&pose_src.path))?;

        // ── Detection (per camera, image-order) ─────────────────────────────
        progress(
            entry,
            format!("detecting target in {} cameras", entry.cameras.len()),
        );
        let detect_start = Instant::now();
        let mut per_cam_dets: Vec<Vec<Option<CorrespondenceView>>> = Vec::new();
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
        let laser = extract_laser_metrics(entry)?;

        // ── Calibration ─────────────────────────────────────────────────────
        let mut config = RigHandeyeConfig::default();
        if let Some(overrides) = &entry.rig_handeye {
            overrides.apply_to(&mut config);
        }
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
        let init_ok = rh_intrinsics_init_all(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init_all failed for {}", entry.id);
        let init_ms = init_start.elapsed().as_millis() as u64;

        progress(entry, "optimizing intrinsics, rig, and hand-eye");
        let opt_start = Instant::now();
        let _ = rh_intrinsics_optimize_all(&mut session, None)
            .context("step_intrinsics_optimize_all failed")?;
        let _ = rh_rig_init(&mut session).context("step_rig_init failed")?;
        let _ = rh_rig_optimize(&mut session, None).context("step_rig_optimize failed")?;
        let _ = rh_handeye_init(&mut session, None).context("step_handeye_init failed")?;
        let _ = rh_handeye_optimize(&mut session, None).context("step_handeye_optimize failed")?;
        let optimize_ms = opt_start.elapsed().as_millis() as u64;

        let export = session.export().context("session.export failed")?;

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
        };

        // Hierarchical report: Intrinsic floor (free per-(cam,view) PnP) +
        // HandEye level (board pose through the robot + hand-eye chain).
        let (reproj_report, residual_sidecar) =
            split_reproj_report(entry, rig_handeye_report(&export, &dataset_for_report).ok());
        let robot_corrections = export
            .robot_deltas
            .as_deref()
            .and_then(RobotCorrectionSummary::from_deltas);

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
        let detector = detector_for(board)?;
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

    #[cfg(not(feature = "laser"))]
    fn extract_laser_metrics(entry: &BenchEntry) -> Result<Option<LaserMetrics>> {
        if entry.laser.is_some() {
            anyhow::bail!(
                "dataset '{}' declares laser data; rebuild with --features 'tier-b laser'",
                entry.id
            );
        }
        Ok(None)
    }

    #[cfg(feature = "laser")]
    fn extract_laser_metrics(entry: &BenchEntry) -> Result<Option<LaserMetrics>> {
        if entry.laser.is_none() {
            return Ok(None);
        }
        let paths = laser_image_paths(entry)?;
        if paths.is_empty() {
            return Ok(None);
        }

        progress(
            entry,
            format!(
                "extracting laser lines from {} images across {} cameras",
                paths.len(),
                entry.cameras.len()
            ),
        );
        let mut per_camera = Vec::with_capacity(entry.cameras.len());
        let mut total_points = 0usize;
        let mut total_images_used = 0usize;
        let total_start = Instant::now();

        for cam in &entry.cameras {
            let cam_start = Instant::now();
            let mut images_used = 0usize;
            let mut points_extracted = 0usize;
            for (image_idx, path) in paths.iter().enumerate() {
                progress_images(entry, "laser", &cam.id, image_idx, paths.len());
                let img = load_image(path)?;
                let img = apply_tile(&img, cam.tile);
                let point_count = extract_laser_points(&img);
                if point_count > 0 {
                    images_used += 1;
                    points_extracted += point_count;
                }
            }
            let extract_ms = cam_start.elapsed().as_millis() as u64;
            progress(
                entry,
                format!(
                    "laser {}: {images_used}/{} images, {points_extracted} points, {extract_ms} ms",
                    cam.id,
                    paths.len()
                ),
            );
            total_points += points_extracted;
            total_images_used += images_used;
            per_camera.push(LaserCamStat {
                camera_id: cam.id.clone(),
                images_total: paths.len(),
                images_used,
                points_extracted,
                extract_ms,
                plane_residual_m: None,
                line_residual_px: None,
                inlier_ratio: None,
            });
        }

        Ok(Some(LaserMetrics {
            per_camera,
            total_points,
            total_images_used,
            extract_ms: total_start.elapsed().as_millis() as u64,
        }))
    }

    #[cfg(feature = "laser")]
    fn extract_laser_points(img: &image::DynamicImage) -> usize {
        let luma = img.to_luma8();
        let width = luma.width() as usize;
        let height = luma.height() as usize;
        let Ok(view) = ImageView::<u8>::from_slice(width, height, width, luma.as_raw()) else {
            return 0;
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
            .len()
    }

    #[cfg(feature = "laser")]
    #[derive(serde::Deserialize)]
    struct LaserPoseEntry {
        laser_image: Option<String>,
        #[serde(rename = "type")]
        snap_type: Option<String>,
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
    fn detector_for(board: &BoardGeometry) -> Result<DetectorKind> {
        let layout = board.layout.as_deref().unwrap_or("checkerboard");
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
            Ok(DetectorKind::Charuco(Box::new(params)))
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

            let detector = detector_for(&board).expect("detector");
            match detector {
                DetectorKind::Puzzleboard(params) => {
                    assert_eq!(params.board.rows, 130);
                    assert_eq!(params.board.cols, 130);
                }
                _ => panic!("expected puzzleboard detector"),
            }
        }
    }
}
