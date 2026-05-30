//! Run a calibration problem end-to-end and return a [`BenchRecord`] with
//! captured metrics.
//!
//! Tier-A would replay a frozen fixture through the math/serde path; that lands
//! in a later phase. The Tier-B path here globs a camera's images, detects the
//! board in each, builds the problem `Input`, runs the facade pipeline, and
//! captures both bench-recomputed and self-reported reprojection metrics so any
//! divergence between them is visible in the record.

#[cfg(feature = "tier-b")]
pub use tier_b::{run_planar_intrinsics, run_rig_extrinsics, run_single_cam_handeye};

#[cfg(feature = "tier-b")]
pub mod tier_b {
    //! Tier-B entry points that require detection and image-loading capabilities.

    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{Context, Result};
    use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
    use vision_calibration::analysis::{
        planar_intrinsics_report, rig_extrinsics_report, single_cam_handeye_report,
    };
    use vision_calibration::core::{
        CorrespondenceView, Iso3, NoMeta, RigDataset, RigView, RigViewObs,
    };
    use vision_calibration::optim::HandEyeMode;
    use vision_calibration::planar_intrinsics::{
        PlanarIntrinsicsProblem, step_init, step_optimize,
    };
    use vision_calibration::rig_extrinsics::{
        RigExtrinsicsProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
        step_rig_init, step_rig_optimize,
    };
    use vision_calibration::session::CalibrationSession;
    use vision_calibration::single_cam_handeye::{
        HandeyeMeta, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeView,
        step_handeye_init, step_handeye_optimize, step_intrinsics_init, step_intrinsics_optimize,
    };
    use vision_calibration_core::{
        FeatureResidualHistogram, PlanarDataset, ReprojectionStats, View,
    };
    use vision_calibration_optim::SolveReport;

    use crate::detect::{
        DetectorKind, charuco_params_for, detect_chessboard_view, glob_sorted_images, load_image,
    };
    use crate::record::{
        BENCH_SCHEMA_VERSION, BenchRecord, Convergence, Detection, DetectionStat, Fit, Ident,
        Timing,
    };
    use crate::registry::{BenchEntry, BoardGeometry, ProblemKind};

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
        let detect_start = Instant::now();
        let mut views = Vec::new();
        let mut images_used = 0usize;
        let mut features_detected = 0usize;
        let mut max_corners_per_image = 0usize;
        for path in &paths {
            let img = load_image(path)?;
            match detect_chessboard_view(&img, board.rows, board.cols, board.cell_size_m) {
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

        anyhow::ensure!(
            views.len() >= 3,
            "need >= 3 detected views for planar intrinsics, got {}",
            views.len()
        );

        // Coverage denominator: the full-board corner count. We take the max
        // corners seen in any single image rather than the registry's declared
        // `rows*cols`, because the `calib-targets` chessboard detector
        // auto-discovers the interior-corner grid and can disagree with a
        // hand-written manifest. (For `stereo_left` the detector finds a 10x11
        // = 110-corner grid while dataset_left.toml declares 7x11 = 77; using
        // the manifest count would yield a >100% coverage figure.) The declared
        // `board.rows`/`board.cols` are still threaded through to
        // `detect_chessboard_view` for API symmetry.
        let features_per_board = max_corners_per_image.max(board.rows * board.cols);
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

        let init_start = Instant::now();
        let init_ok = step_init(&mut session, None).is_ok();
        let init_ms = init_start.elapsed().as_millis() as u64;
        anyhow::ensure!(init_ok, "step_init failed for {}", entry.id);

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
        let reproj_report = planar_intrinsics_report(&export, &views_for_report).ok();

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "planar_intrinsics"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser: None,
            delta_to_prior: None,
            timing,
            reproj_report,
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
            for path in &paths {
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

        let features_per_board = max_corners_per_image.max(board.rows * board.cols);

        // ── Calibration ────────────────────────────────────────────────────
        let input = RigDataset::new(rig_views, entry.cameras.len())
            .map_err(|e| anyhow::anyhow!("failed to build RigDataset: {e}"))?;
        // Keep a copy for the multi-level reproj report (the dataset is moved
        // into the session below).
        let dataset_for_report = input.clone();
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(input).context("set_input failed")?;

        let init_start = Instant::now();
        let init_ok = step_intrinsics_init_all(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init_all failed for {}", entry.id);
        let init_ms = init_start.elapsed().as_millis() as u64;

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
        let reproj_report = rig_extrinsics_report(&export, &dataset_for_report).ok();

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "rig_extrinsics"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser: None,
            delta_to_prior: None,
            timing,
            reproj_report,
        })
    }

    /// Run a single-camera hand-eye calibration for `entry` and build a
    /// [`BenchRecord`].
    ///
    /// Mirrors `crates/vision-calibration/examples/handeye_session.rs`: loads
    /// the robot poses (row-major 4×4 per line, the `base_se3_gripper` /
    /// EyeInHand convention parsed exactly as the example's
    /// `support/handeye_io.rs`), then for image `i` (`{i+1:02}.png`) detects the
    /// chessboard and pairs it with pose `i`; views with no board are skipped,
    /// shifting later poses' image index but keeping the pose↔image alignment
    /// the example relies on (image `NN.png` pairs with pose row `NN`). Runs
    /// `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init`
    /// → `step_handeye_optimize`, then exports. The board square size is read
    /// from the registry `board.cell_size_m`.
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
        let square_size_m = board.cell_size_m;

        let robot_poses = load_robot_poses(&entry.data_root.join(&pose_src.path))?;

        // ── Detection (image i ↔ pose i, mirroring handeye_session.rs) ──────
        let folder = entry.data_root.join(&cam.folder);
        let detect_start = Instant::now();
        let mut views: Vec<SingleCamHandeyeView> = Vec::new();
        let mut features_detected = 0usize;
        let mut max_corners_per_image = 0usize;
        let mut images_used = 0usize;
        for (idx, robot_pose) in robot_poses.iter().enumerate() {
            let image_index = idx + 1;
            let img_path = folder.join(format!("{image_index:02}.png"));
            anyhow::ensure!(
                img_path.exists(),
                "missing image {} for pose row {}",
                img_path.display(),
                image_index
            );
            let img = load_image(&img_path)?;
            match detect_chessboard_view(&img, board.rows, board.cols, square_size_m) {
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
                    return Err(e.context(format!("detection failed for {}", img_path.display())));
                }
            }
        }
        let detection_ms = detect_start.elapsed().as_millis() as u64;
        anyhow::ensure!(
            views.len() >= 3,
            "need >= 3 detected views for single-cam hand-eye, got {}",
            views.len()
        );

        let features_per_board = max_corners_per_image.max(board.rows * board.cols);
        let features_expected = images_used * features_per_board;
        let coverage_pct = if features_expected > 0 {
            100.0 * features_detected as f64 / features_expected as f64
        } else {
            0.0
        };
        let detection = Detection {
            per_camera: vec![DetectionStat {
                camera_id: cam.id.clone(),
                images_total: robot_poses.len(),
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
        session.set_input(input).context("set_input failed")?;

        let init_start = Instant::now();
        let init_ok = step_intrinsics_init(&mut session, None).is_ok();
        anyhow::ensure!(init_ok, "step_intrinsics_init failed for {}", entry.id);
        let init_ms = init_start.elapsed().as_millis() as u64;

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
        let reproj_report = single_cam_handeye_report(&export, &views_for_report).ok();

        Ok(BenchRecord {
            ident: placeholder_ident(entry, "single_cam_handeye"),
            convergence,
            fit,
            generalization: None,
            stability: None,
            detection: Some(detection),
            laser: None,
            delta_to_prior: None,
            timing,
            reproj_report,
        })
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
    /// the board's rows/cols/cell-size/dictionary; anything else (incl. the
    /// `"checkerboard"` default) uses the chessboard detector with the board's
    /// metric cell size.
    fn detector_for(board: &BoardGeometry) -> Result<DetectorKind> {
        let layout = board.layout.as_deref().unwrap_or("checkerboard");
        if layout.eq_ignore_ascii_case("charuco") {
            let dict = board
                .dictionary
                .as_deref()
                .context("charuco board needs a `dictionary`")?;
            // Marker scale is not in `BoardGeometry`; the stereo_charuco example
            // uses 0.75, the OpenCV ChArUco default. Hard-code it to match.
            let params = charuco_params_for(
                board.rows as u32,
                board.cols as u32,
                board.cell_size_m,
                0.75,
                dict,
            )?;
            Ok(DetectorKind::Charuco(Box::new(params)))
        } else {
            Ok(DetectorKind::Chessboard {
                square_size_m: board.cell_size_m,
            })
        }
    }

    /// Parse one row-major 4×4 robot-pose line into an [`Iso3`].
    ///
    /// Verbatim port of `support/handeye_io.rs::parse_pose_line`: 16 values,
    /// row-major; rotation from the upper-left 3×3, translation from the 4th
    /// column. The resulting `Iso3` is the `base_se3_gripper` the example feeds
    /// as the EyeInHand robot-pose prior.
    fn parse_pose_line(line: &str, idx: usize) -> Result<Iso3> {
        let values: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("invalid float in robot pose line {}", idx + 1))?;
        anyhow::ensure!(
            values.len() == 16,
            "robot pose line {} expected 16 values, got {}",
            idx + 1,
            values.len()
        );
        let r = Matrix3::new(
            values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
            values[10],
        );
        let t = Vector3::new(values[3], values[7], values[11]);
        let rot = Rotation3::from_matrix_unchecked(r);
        Ok(Iso3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_rotation_matrix(&rot),
        ))
    }

    /// Load all robot poses from a whitespace-delimited 4×4-per-line file.
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
}
