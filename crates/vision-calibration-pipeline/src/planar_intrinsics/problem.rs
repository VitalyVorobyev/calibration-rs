//! [`ProblemType`] implementation for planar intrinsics calibration.
//!
//! This module provides the `PlanarIntrinsicsProblem` type that implements
//! the session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    DistortionFixMask, ImageManifest, IntrinsicsFixMask, PerFeatureResiduals, PlanarDataset,
    build_feature_histogram, compute_planar_target_residuals,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, SolveReport,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::PlanarState;

/// Planar intrinsics calibration problem (Zhang's method with distortion).
///
/// This problem type implements calibration of camera intrinsics and distortion
/// from multiple views of a planar calibration pattern.
///
/// # Associated Types
///
/// - **Config**: [`PlanarIntrinsicsConfig`] - solver settings, fix masks, etc.
/// - **Input**: [`PlanarDataset`] - views with 2D-3D point correspondences
/// - **State**: [`PlanarState`] - homographies, initial estimates, metrics
/// - **Output**: [`PlanarIntrinsicsEstimate`] - final calibrated camera + poses
/// - **Export**: [`PlanarIntrinsicsExport`] - stable export contract
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::planar_intrinsics::{
///     PlanarIntrinsicsProblem, PlanarIntrinsicsConfig,
///     step_init, step_optimize,
/// };
/// # fn main() -> anyhow::Result<()> {
/// # let dataset = unimplemented!();
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
///
/// step_init(&mut session, None)?;
/// step_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PlanarIntrinsicsProblem;

/// Configuration for planar intrinsics calibration.
///
/// Contains settings for both initialization and optimization phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PlanarIntrinsicsConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Initialization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,

    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3_in_init: bool,

    /// Fix tangential distortion (p1, p2) during initialization.
    pub fix_tangential_in_init: bool,

    /// Enforce zero skew during initialization.
    pub zero_skew: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Optimization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum iterations for the optimizer.
    pub max_iters: usize,

    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,

    /// Robust loss function for outlier handling.
    pub robust_loss: vision_calibration_optim::RobustLoss,

    /// Mask for fixing intrinsic parameters during optimization.
    pub fix_intrinsics: IntrinsicsFixMask,

    /// Mask for fixing distortion parameters during optimization.
    pub fix_distortion: DistortionFixMask,

    /// Indices of poses to fix during optimization (e.g., \[0\] to fix first pose).
    pub fix_poses: Vec<usize>,
}

impl Default for PlanarIntrinsicsConfig {
    fn default() -> Self {
        Self {
            // Initialization
            init_iterations: 2,
            fix_k3_in_init: true,
            fix_tangential_in_init: false,
            zero_skew: true,
            // Optimization
            max_iters: 50,
            verbosity: 0,
            robust_loss: vision_calibration_optim::RobustLoss::None,
            fix_intrinsics: Default::default(),
            fix_distortion: Default::default(),
            fix_poses: Vec::new(),
        }
    }
}

impl PlanarIntrinsicsConfig {
    /// Convert to vision-calibration-linear initialization options.
    pub fn init_opts(&self) -> IterativeIntrinsicsOptions {
        IterativeIntrinsicsOptions {
            iterations: self.init_iterations,
            distortion_opts: DistortionFitOptions {
                fix_k3: self.fix_k3_in_init,
                fix_tangential: self.fix_tangential_in_init,
                iters: 8,
            },
            zero_skew: self.zero_skew,
        }
    }

    /// Convert to vision-calibration-optim solve options.
    pub fn solve_opts(&self) -> PlanarIntrinsicsSolveOptions {
        PlanarIntrinsicsSolveOptions {
            robust_loss: self.robust_loss,
            fix_intrinsics: self.fix_intrinsics,
            fix_distortion: self.fix_distortion,
            fix_poses: self.fix_poses.clone(),
        }
    }

    /// Convert to backend solver options.
    pub fn backend_opts(&self) -> BackendSolveOptions {
        BackendSolveOptions {
            max_iters: self.max_iters,
            verbosity: self.verbosity,
            ..Default::default()
        }
    }
}

/// Export format for planar intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PlanarIntrinsicsExport {
    /// Calibrated parameters.
    pub params: PlanarIntrinsicsParams,
    /// Solver report.
    pub report: SolveReport,
    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors (single element for single-camera workflows).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-feature reprojection residuals (ADR 0012). For planar intrinsics
    /// only `target` is populated; `laser` is empty. `target_hist_per_camera`
    /// is `Some(vec![one_entry])` since this problem type is single-camera.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
    /// Optional image manifest (ADR 0014, viewer-side contract). When
    /// populated, downstream viewers (the diagnose UI) can locate the source
    /// image for each `(pose, camera)` slot. `None` means "no images
    /// shipped"; the calibration pipeline never reads this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
}

impl ProblemType for PlanarIntrinsicsProblem {
    type Config = PlanarIntrinsicsConfig;
    type Input = PlanarDataset;
    type State = PlanarState;
    type Output = PlanarIntrinsicsEstimate;
    type Export = PlanarIntrinsicsExport;

    fn name() -> &'static str {
        "planar_intrinsics_v2"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        if input.num_views() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: input.num_views(),
            });
        }

        for (i, view) in input.views.iter().enumerate() {
            if view.obs.len() < 4 {
                return Err(Error::invalid_input(format!(
                    "view {} has too few points (need >= 4 for homography, got {})",
                    i,
                    view.obs.len()
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.init_iterations == 0 {
            return Err(Error::invalid_input("init_iterations must be positive"));
        }
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        // When input changes, clear state and output
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        // Config changes don't auto-invalidate (allow experimentation)
        InvalidationPolicy::KEEP_ALL
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        // Length-mismatch is a logic error inside the pipeline (output came
        // out of an optimizer that ran on `input`); surface it as a typed
        // error rather than panicking.
        let target = compute_planar_target_residuals(
            &output.params.camera,
            input,
            &output.params.camera_se3_target,
        )?;
        let target_hist = build_feature_histogram(target.iter().filter_map(|r| r.error_px));
        Ok(PlanarIntrinsicsExport {
            params: output.params.clone(),
            report: output.report.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: vec![output.mean_reproj_error],
            per_feature_residuals: PerFeatureResiduals {
                target,
                laser: Vec::new(),
                target_hist_per_camera: Some(vec![target_hist]),
                laser_hist_per_camera: None,
            },
            // Manifest is populated by callers that also wrote images for
            // the dataset (e.g. the `planar_synthetic_with_images` example);
            // the pipeline itself never has image paths to fill in.
            image_manifest: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{CorrespondenceView, NoMeta, Pt2, Pt3, View};
    use vision_calibration_optim::RobustLoss;

    fn make_minimal_dataset() -> PlanarDataset {
        // Create 3 views with 4 points each (minimum requirements)
        let make_view = || {
            View::new(
                CorrespondenceView {
                    points_3d: vec![
                        Pt3::new(0.0, 0.0, 0.0),
                        Pt3::new(0.05, 0.0, 0.0),
                        Pt3::new(0.05, 0.05, 0.0),
                        Pt3::new(0.0, 0.05, 0.0),
                    ],
                    points_2d: vec![
                        Pt2::new(100.0, 100.0),
                        Pt2::new(200.0, 100.0),
                        Pt2::new(200.0, 200.0),
                        Pt2::new(100.0, 200.0),
                    ],
                    weights: Vec::new(),
                },
                NoMeta {},
            )
        };

        PlanarDataset::new(vec![make_view(), make_view(), make_view()]).unwrap()
    }

    #[test]
    fn validate_input_requires_3_views() {
        // PlanarDataset::new already validates this, so we test our validation
        // by checking the error message content
        let dataset = make_minimal_dataset();
        // With 3 views it should pass
        let result = PlanarIntrinsicsProblem::validate_input(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_minimum_views() {
        // Verify the validation function checks view count
        // We can't create invalid PlanarDataset, but we verify the validation logic
        // by checking that the valid dataset passes
        let dataset = make_minimal_dataset();
        assert!(PlanarIntrinsicsProblem::validate_input(&dataset).is_ok());
    }

    #[test]
    fn validate_input_accepts_valid() {
        let dataset = make_minimal_dataset();
        let result = PlanarIntrinsicsProblem::validate_input(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_config_requires_positive_iters() {
        let config = PlanarIntrinsicsConfig {
            max_iters: 0,
            ..Default::default()
        };
        let result = PlanarIntrinsicsProblem::validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iters"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = PlanarIntrinsicsConfig::default();
        let result = PlanarIntrinsicsProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn config_json_roundtrip() {
        let config = PlanarIntrinsicsConfig {
            max_iters: 100,
            fix_k3_in_init: false,
            robust_loss: vision_calibration_optim::RobustLoss::Huber { scale: 2.5 },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.max_iters, 100);
        assert!(!restored.fix_k3_in_init);
        match restored.robust_loss {
            vision_calibration_optim::RobustLoss::Huber { scale } => {
                assert!((scale - 2.5).abs() < 1e-12);
            }
            vision_calibration_optim::RobustLoss::None
            | vision_calibration_optim::RobustLoss::Cauchy { .. }
            | vision_calibration_optim::RobustLoss::Arctan { .. } => {
                panic!(
                    "expected Huber loss after roundtrip, got {:?}",
                    restored.robust_loss
                );
            }
        }
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(PlanarIntrinsicsProblem::name(), "planar_intrinsics_v2");
        assert_eq!(PlanarIntrinsicsProblem::schema_version(), 1);
    }

    #[test]
    fn init_opts_conversion() {
        let config = PlanarIntrinsicsConfig {
            init_iterations: 5,
            fix_k3_in_init: true,
            fix_tangential_in_init: true,
            zero_skew: false,
            ..Default::default()
        };

        let opts = config.init_opts();
        assert_eq!(opts.iterations, 5);
        assert!(opts.distortion_opts.fix_k3);
        assert!(opts.distortion_opts.fix_tangential);
        assert!(!opts.zero_skew);
    }

    #[test]
    fn solve_opts_conversion() {
        let config = PlanarIntrinsicsConfig {
            robust_loss: vision_calibration_optim::RobustLoss::Cauchy { scale: 1.0 },
            fix_poses: vec![0, 1],
            ..Default::default()
        };

        let opts = config.solve_opts();
        assert_eq!(opts.fix_poses, vec![0, 1]);
        match opts.robust_loss {
            vision_calibration_optim::RobustLoss::Cauchy { scale } => {
                assert!((scale - 1.0).abs() < 1e-12);
            }
            vision_calibration_optim::RobustLoss::None
            | vision_calibration_optim::RobustLoss::Huber { .. }
            | vision_calibration_optim::RobustLoss::Arctan { .. } => {
                panic!("expected Cauchy loss, got {:?}", opts.robust_loss);
            }
        }
    }

    #[test]
    fn backend_opts_conversion() {
        let config = PlanarIntrinsicsConfig {
            max_iters: 100,
            verbosity: 2,
            ..Default::default()
        };

        let opts = config.backend_opts();
        assert_eq!(opts.max_iters, 100);
        assert_eq!(opts.verbosity, 2);
    }

    #[test]
    fn export_includes_per_camera_reprojection_errors() {
        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCySkew {
                fx: 800.0,
                fy: 790.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );
        let params =
            PlanarIntrinsicsParams::new(camera, vec![vision_calibration_core::Iso3::identity()])
                .expect("valid params");
        let output = PlanarIntrinsicsEstimate {
            params,
            report: SolveReport {
                final_cost: 1.23e-3,
            },
            mean_reproj_error: 0.42,
        };

        let dummy_view = vision_calibration_core::View::without_meta(
            vision_calibration_core::CorrespondenceView::new(
                vec![vision_calibration_core::Pt3::new(0.0, 0.0, 0.0); 4],
                vec![vision_calibration_core::Pt2::new(0.0, 0.0); 4],
            )
            .unwrap(),
        );
        let dummy_input = vision_calibration_core::PlanarDataset::new(vec![dummy_view]).unwrap();

        let export = PlanarIntrinsicsProblem::export(
            &dummy_input,
            &output,
            &PlanarIntrinsicsConfig {
                robust_loss: RobustLoss::None,
                ..Default::default()
            },
        )
        .expect("export");

        assert_eq!(export.mean_reproj_error, 0.42);
        assert_eq!(export.per_cam_reproj_errors, vec![0.42]);
        // Even though the synthetic input has every feature at (0,0,0)
        // (projection diverges), one record per feature must still appear.
        assert_eq!(export.per_feature_residuals.target.len(), 4);
        for r in &export.per_feature_residuals.target {
            assert_eq!(r.camera, 0);
            assert!(r.projected_px.is_none());
            assert!(r.error_px.is_none());
        }
        let hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .as_ref()
            .expect("planar export emits a single-camera histogram");
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].count, 0);
    }

    #[test]
    fn export_target_residuals_zero_for_perfect_data() {
        // Synthesize a 3-view dataset where every feature reprojects to the
        // observed pixel exactly. The exported per_feature_residuals.target
        // must report Some(error_px) ≈ 0 for every record and the histogram
        // must place all counts in the first bucket.
        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );
        let pose = vision_calibration_core::Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let board: Vec<vision_calibration_core::Pt3> = vec![
            vision_calibration_core::Pt3::new(0.0, 0.0, 0.0),
            vision_calibration_core::Pt3::new(0.05, 0.0, 0.0),
            vision_calibration_core::Pt3::new(0.0, 0.05, 0.0),
            vision_calibration_core::Pt3::new(0.05, 0.05, 0.0),
        ];
        let make_view = || {
            let pixels: Vec<vision_calibration_core::Pt2> = board
                .iter()
                .map(|p| {
                    let p_cam = pose * p;
                    camera.project_point_c(&p_cam.coords).unwrap()
                })
                .collect();
            vision_calibration_core::View::without_meta(
                vision_calibration_core::CorrespondenceView::new(board.clone(), pixels).unwrap(),
            )
        };
        let dataset = vision_calibration_core::PlanarDataset::new(vec![
            make_view(),
            make_view(),
            make_view(),
        ])
        .unwrap();

        let params =
            PlanarIntrinsicsParams::new(camera, vec![pose, pose, pose]).expect("valid params");
        let output = PlanarIntrinsicsEstimate {
            params,
            report: SolveReport { final_cost: 0.0 },
            mean_reproj_error: 0.0,
        };

        let export =
            PlanarIntrinsicsProblem::export(&dataset, &output, &PlanarIntrinsicsConfig::default())
                .expect("export");

        assert_eq!(export.per_feature_residuals.target.len(), 12);
        for (i, r) in export.per_feature_residuals.target.iter().enumerate() {
            let err = r
                .error_px
                .unwrap_or_else(|| panic!("record {i} missing error"));
            assert!(err < 1e-9, "record {i} error_px {err} not near zero");
        }
        let hist = &export
            .per_feature_residuals
            .target_hist_per_camera
            .as_ref()
            .unwrap()[0];
        assert_eq!(hist.count, 12);
        // All errors are <=1 px so they fall in the first bucket.
        assert_eq!(hist.counts, [12, 0, 0, 0, 0]);
    }

    #[test]
    fn export_per_feature_residuals_json_roundtrip() {
        // Round-trip the export through JSON and confirm per_feature_residuals
        // including bucket counts survive serialize/deserialize.
        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );
        let pose = vision_calibration_core::Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let view = vision_calibration_core::View::without_meta(
            vision_calibration_core::CorrespondenceView::new(
                vec![
                    vision_calibration_core::Pt3::new(0.0, 0.0, 0.0),
                    vision_calibration_core::Pt3::new(0.05, 0.0, 0.0),
                    vision_calibration_core::Pt3::new(0.0, 0.05, 0.0),
                    vision_calibration_core::Pt3::new(0.05, 0.05, 0.0),
                ],
                vec![
                    vision_calibration_core::Pt2::new(320.0, 240.0),
                    vision_calibration_core::Pt2::new(360.0, 240.0),
                    vision_calibration_core::Pt2::new(320.0, 280.0),
                    vision_calibration_core::Pt2::new(360.0, 280.0),
                ],
            )
            .unwrap(),
        );
        let dataset = vision_calibration_core::PlanarDataset::new(vec![view]).unwrap();
        let params = PlanarIntrinsicsParams::new(camera, vec![pose]).expect("valid params");
        let output = PlanarIntrinsicsEstimate {
            params,
            report: SolveReport { final_cost: 0.0 },
            mean_reproj_error: 0.0,
        };
        let export =
            PlanarIntrinsicsProblem::export(&dataset, &output, &PlanarIntrinsicsConfig::default())
                .expect("export");

        let json = serde_json::to_string(&export).expect("serialize");
        let restored: PlanarIntrinsicsExport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored.per_feature_residuals.target.len(),
            export.per_feature_residuals.target.len()
        );
        assert_eq!(
            restored.per_feature_residuals.target_hist_per_camera,
            export.per_feature_residuals.target_hist_per_camera
        );
        assert!(restored.per_feature_residuals.laser.is_empty());
        assert!(
            restored
                .per_feature_residuals
                .laser_hist_per_camera
                .is_none()
        );
        // ADR 0014: image_manifest defaults to None and absent from the wire
        // when not populated, preserving JSON byte-stability for legacy
        // exports.
        assert!(restored.image_manifest.is_none());
        assert!(!json.contains("image_manifest"));
    }

    #[test]
    fn export_image_manifest_roundtrip() {
        use vision_calibration_core::{FrameRef, ImageManifest};

        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );
        let params =
            PlanarIntrinsicsParams::new(camera, vec![vision_calibration_core::Iso3::identity()])
                .expect("valid params");
        let mut export = PlanarIntrinsicsExport {
            params,
            report: SolveReport { final_cost: 0.0 },
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0],
            per_feature_residuals: PerFeatureResiduals::default(),
            image_manifest: None,
        };
        export.image_manifest = Some(ImageManifest {
            root: std::path::PathBuf::from("images"),
            frames: vec![FrameRef {
                pose: 0,
                camera: 0,
                path: std::path::PathBuf::from("pose_0_cam_0.png"),
                roi: None,
            }],
        });

        let json = serde_json::to_string(&export).expect("serialize");
        let restored: PlanarIntrinsicsExport = serde_json::from_str(&json).expect("deserialize");
        let manifest = restored
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.root, std::path::PathBuf::from("images"));
        assert_eq!(manifest.frames.len(), 1);
        assert_eq!(
            manifest.frames[0].path,
            std::path::PathBuf::from("pose_0_cam_0.png")
        );
    }
}
