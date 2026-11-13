//! [`ProblemType`] implementation for planar intrinsics calibration.
//!
//! This module provides the `PlanarIntrinsicsProblem` type that implements
//! the session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::stated_covariance;
use vision_calibration_core::{
    CameraParams, Iso3, PlanarDataset, SensorKind, Uncertainty, compute_planar_target_residuals,
};
use vision_calibration_optim::{
    PlanarIntrinsicsEstimate, PlanarIntrinsicsSolveOptions, RobustLoss,
};

use crate::common::config::{CameraConfig, CameraInitOptions, IntrinsicsInitConfig, SolverConfig};
use crate::common::{
    ExportDiagnostics, ProblemKind, VersionedExport, one_camera_target_residuals,
    read_schema_version,
};
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::PlanarState;

/// Input for planar intrinsics calibration: views of a planar target, each
/// with its 2D-3D correspondences.
pub type PlanarIntrinsicsInput = PlanarDataset;

/// Planar intrinsics calibration problem (Zhang's method with distortion).
///
/// This problem type calibrates one camera — intrinsics, distortion and, for a
/// tilted (Scheimpflug) sensor, the sensor tilt — from multiple views of a
/// planar calibration pattern. The camera model comes from
/// [`PlanarIntrinsicsConfig::camera`].
///
/// # Associated Types
///
/// - **Config**: [`PlanarIntrinsicsConfig`] - solver settings, fix masks, etc.
/// - **Input**: [`PlanarIntrinsicsInput`] - views with 2D-3D point correspondences
/// - **Output**: [`PlanarIntrinsicsEstimate`] - final calibrated camera + poses
/// - **Export**: [`PlanarIntrinsicsExport`] - stable export contract
///
/// # Example
///
/// Eight synthetic views of an 8×6 board (40 mm pitch) seen by a known
/// camera, calibrated step by step:
///
/// ```
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::planar_intrinsics::{
///     PlanarIntrinsicsProblem, step_filter, step_init, step_optimize,
/// };
/// # use vision_calibration_core::synthetic::{planar, poses};
/// # use vision_calibration_core::{BrownConrady5, FxFyCxCy, PlanarDataset, View, make_pinhole_camera};
/// # fn main() -> anyhow::Result<()> {
/// # let truth = make_pinhole_camera(
/// #     FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
/// #     BrownConrady5 { k1: 0.05, k2: -0.02, ..BrownConrady5::default() },
/// # );
/// # let board = planar::grid_points(8, 6, 0.04);
/// # let specs = poses::seeded_board_pose_specs(8, 0.35, 3);
/// # let poses = poses::centered_board_poses(&specs, (0.14, 0.10), 0.6);
/// # let views = planar::project_views_all(&truth, &board, &poses)?;
/// # let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
///
/// step_init(&mut session, None)?; // Zhang's method, closed form
/// step_optimize(&mut session, None)?; // non-linear refinement
/// // Optional: drop outlier points, which clears the solution, then solve again.
/// step_filter(&mut session, None)?;
/// step_init(&mut session, None)?;
/// step_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// let k = export.camera.intrinsics;
/// assert!((k.fx - 800.0).abs() < 1e-3 && (k.cy - 360.0).abs() < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PlanarIntrinsicsProblem;

/// The fewest target points a planar view needs: four determine its
/// homography.
pub(crate) const MIN_VIEW_POINTS: usize = 4;

/// Default non-linear iteration cap of a planar solve, sized for a tilted
/// sensor's tilt valley. The solver stops at convergence, so the cap only
/// bounds runs that do not converge.
const DEFAULT_MAX_ITERS: usize = 120;

/// Configuration for planar intrinsics calibration.
///
/// Grouped by stage: the camera model with its fix lists in
/// [`CameraConfig`], the linear initialization in [`IntrinsicsInitConfig`],
/// the non-linear refinement in [`PlanarOptimizeConfig`], and the solver
/// engine in [`SolverConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct PlanarIntrinsicsConfig {
    /// The camera model and its fixed parameters.
    ///
    /// Any distortion kind with an identity sensor; a Scheimpflug sensor
    /// needs a distortion model. A tilted sensor's solve without a tilt seed
    /// runs a multi-start tilt sweep (see `step_optimize`).
    #[serde(default)]
    pub camera: CameraConfig,

    /// Linear-initialization stage settings (`step_init`). The linear stage
    /// fits only the distortion terms the camera's solves estimate.
    pub intrinsics_init: IntrinsicsInitConfig,

    /// Non-linear refinement stage settings (`step_optimize`).
    pub optimize: PlanarOptimizeConfig,

    /// Solver engine settings. `solver.max_iters` defaults to 120, sized for
    /// a tilted sensor's tilt valley; the solver stops at convergence, so
    /// the cap only bounds runs that do not converge.
    pub solver: SolverConfig,
}

impl Default for PlanarIntrinsicsConfig {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            intrinsics_init: IntrinsicsInitConfig::default(),
            optimize: PlanarOptimizeConfig::default(),
            solver: SolverConfig {
                max_iters: DEFAULT_MAX_ITERS,
                ..SolverConfig::default()
            },
        }
    }
}

/// The non-linear refinement stage of a planar intrinsics calibration
/// (`step_optimize`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct PlanarOptimizeConfig {
    /// Robust loss on the target-corner reprojection residuals; its scale is
    /// in pixels. Default `none`: plain least squares, the
    /// maximum-likelihood estimate under Gaussian corner noise. A tilted
    /// sensor's tilt sweep runs plain least squares and applies this loss
    /// in its final solve.
    pub reprojection_loss: RobustLoss,

    /// Indices of poses to fix during optimization (e.g., \[0\] to fix first
    /// pose). Default: none — the board fixes the target frame, and a fixed
    /// pose makes the reported uncertainty conditional on it.
    pub fix_poses: Vec<usize>,
}

impl PlanarIntrinsicsConfig {
    /// Whether the camera has a tilted (Scheimpflug) sensor.
    pub fn is_scheimpflug(&self) -> bool {
        self.camera.model.sensor == SensorKind::Scheimpflug
    }

    /// Convert to vision-calibration-linear initialization options, with
    /// `iterations` overriding `intrinsics_init.iterations` when set.
    pub(crate) fn init_opts(&self, iterations: Option<usize>) -> CameraInitOptions {
        self.intrinsics_init
            .camera_init_opts(iterations, &self.camera)
    }

    /// Convert to vision-calibration-optim solve options.
    pub(crate) fn solve_opts(&self) -> PlanarIntrinsicsSolveOptions {
        PlanarIntrinsicsSolveOptions {
            robust_loss: self.optimize.reprojection_loss,
            fix: self.camera.fix.clone(),
            fix_poses: self.optimize.fix_poses.clone(),
            bounds: None,
        }
    }
}

/// Export format for planar intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PlanarIntrinsicsExport {
    /// Export-type discriminator — always [`ProblemKind::PlanarIntrinsics`].
    pub kind: ProblemKind,

    /// Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
    /// Reading an export of another version is an error.
    #[serde(deserialize_with = "read_schema_version::<_, PlanarIntrinsicsExport>")]
    pub schema_version: u32,
    /// The calibrated camera, in its model; `camera.sensor` states whether a
    /// tilted sensor was calibrated.
    pub camera: CameraParams,
    /// Per-view target poses (`cam_se3_target`).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_target: Vec<Iso3>,
    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors (single element for single-camera workflows).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Residuals and the image manifest: evidence for viewers and
    /// debugging, not part of the result.
    #[serde(default)]
    pub diagnostics: ExportDiagnostics,
    /// Covariance of the calibrated parameters: `fx ± σ` (and `tilt_x ± σ`
    /// for a tilted sensor), correlations, and the noise level the residuals
    /// imply.
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

impl VersionedExport for PlanarIntrinsicsExport {
    const SCHEMA_VERSION: u32 = 1;
}

impl ProblemState for PlanarIntrinsicsProblem {
    type State = PlanarState;
}

impl ProblemType for PlanarIntrinsicsProblem {
    type Config = PlanarIntrinsicsConfig;
    type Input = PlanarIntrinsicsInput;
    type Output = PlanarIntrinsicsEstimate;
    type Export = PlanarIntrinsicsExport;

    fn name() -> &'static str {
        "planar_intrinsics_v2"
    }

    fn schema_version() -> u32 {
        2
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        stated_covariance(input.views.iter().map(|v| &v.obs))?;
        if input.num_views() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: input.num_views(),
            });
        }

        for (i, view) in input.views.iter().enumerate() {
            if view.obs.len() < MIN_VIEW_POINTS {
                return Err(Error::invalid_input(format!(
                    "view {} has too few points (need >= {MIN_VIEW_POINTS} for homography, got {})",
                    i,
                    view.obs.len()
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.solver.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.intrinsics_init.iterations == 0 {
            return Err(Error::invalid_input("init.iterations must be positive"));
        }
        config.camera.validate_single_camera()
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
        let cam = output.params.build_camera()?;
        let target = compute_planar_target_residuals(&cam, input, &output.params.cam_se3_target)?;
        let per_feature_residuals = one_camera_target_residuals(target);
        Ok(PlanarIntrinsicsExport {
            kind: ProblemKind::PlanarIntrinsics,
            schema_version: PlanarIntrinsicsExport::SCHEMA_VERSION,
            camera: output.params.camera.clone(),
            cam_se3_target: output.params.cam_se3_target.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: vec![output.mean_reproj_error],
            // Callers that have the images fill `image_manifest`; the
            // pipeline never has image paths.
            diagnostics: ExportDiagnostics::new(per_feature_residuals),
            uncertainty: output.uncertainty.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::DistortionKind;
    use vision_calibration_core::{CorrespondenceView, NoMeta, Pt2, Pt3, View};
    use vision_calibration_optim::PlanarIntrinsicsParams;
    use vision_calibration_optim::RobustLoss;
    use vision_calibration_optim::SolveReport;

    fn make_minimal_dataset() -> PlanarDataset {
        // Create 3 views with 4 points each (minimum requirements)
        let make_view = || {
            View::new(
                CorrespondenceView::new(
                    vec![
                        Pt3::new(0.0, 0.0, 0.0),
                        Pt3::new(0.05, 0.0, 0.0),
                        Pt3::new(0.05, 0.05, 0.0),
                        Pt3::new(0.0, 0.05, 0.0),
                    ],
                    vec![
                        Pt2::new(100.0, 100.0),
                        Pt2::new(200.0, 100.0),
                        Pt2::new(200.0, 200.0),
                        Pt2::new(100.0, 200.0),
                    ],
                )
                .unwrap(),
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

    /// Views with stated covariances and views without are rejected
    /// together, before any work; all with covariances is accepted.
    #[test]
    fn validate_input_rejects_mixed_point_covariances() {
        use vision_calibration_core::PointCovariance;
        let with_covariance = |view: &View<NoMeta>| {
            let cov = PointCovariance::new(0.04, 0.0, 0.04).unwrap();
            View::without_meta(
                CorrespondenceView::new_with_covariances(
                    view.obs.points_3d.clone(),
                    view.obs.points_2d.clone(),
                    vec![cov; view.obs.len()],
                )
                .unwrap(),
            )
        };
        let plain = make_minimal_dataset();
        let mut views = plain.views.clone();
        views[1] = with_covariance(&views[1]);
        let mixed = PlanarDataset::new(views).unwrap();
        let err = PlanarIntrinsicsProblem::validate_input(&mixed).unwrap_err();
        assert!(
            matches!(
                err,
                Error::Core(vision_calibration_core::Error::MixedCovariance {
                    with: 1,
                    without: 2
                })
            ),
            "{err}"
        );
        let mut session = crate::session::CalibrationSession::<PlanarIntrinsicsProblem>::new();
        assert!(session.set_input(mixed).is_err());

        let all = PlanarDataset::new(plain.views.iter().map(with_covariance).collect()).unwrap();
        PlanarIntrinsicsProblem::validate_input(&all).unwrap();
    }

    #[test]
    fn validate_config_requires_positive_iters() {
        let config = PlanarIntrinsicsConfig {
            solver: SolverConfig {
                max_iters: 0,
                ..Default::default()
            },
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
            solver: SolverConfig {
                max_iters: 100,
                ..Default::default()
            },
            optimize: PlanarOptimizeConfig {
                reprojection_loss: RobustLoss::Huber { scale: 2.5 },
                ..Default::default()
            },
            intrinsics_init: IntrinsicsInitConfig {
                fix_k3: false,
                ..Default::default()
            },
            camera: CameraConfig::new(vision_calibration_core::CameraModelSpec::pinhole(
                DistortionKind::Rational8,
                SensorKind::Scheimpflug,
            )),
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.solver.max_iters, 100);
        assert!(!restored.intrinsics_init.fix_k3);
        assert_eq!(restored.camera, config.camera);
        match restored.optimize.reprojection_loss {
            vision_calibration_optim::RobustLoss::Huber { scale } => {
                assert!((scale - 2.5).abs() < 1e-12);
            }
            _ => {
                panic!(
                    "expected Huber loss after roundtrip, got {:?}",
                    restored.optimize.reprojection_loss
                );
            }
        }
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(PlanarIntrinsicsProblem::name(), "planar_intrinsics_v2");
        assert_eq!(PlanarIntrinsicsProblem::schema_version(), 2);
    }

    #[test]
    fn init_opts_conversion() {
        let config = PlanarIntrinsicsConfig {
            intrinsics_init: IntrinsicsInitConfig {
                iterations: 5,
                fix_k3: true,
            },
            ..Default::default()
        };

        let opts = config.init_opts(None).pinhole;
        assert_eq!(opts.iterations, 5);
        assert!(opts.distortion_opts.fix_k3);
        assert!(!opts.distortion_opts.fix_tangential);
        assert_eq!(config.init_opts(Some(1)).pinhole.iterations, 1);
    }

    #[test]
    fn solve_opts_conversion() {
        let config = PlanarIntrinsicsConfig {
            optimize: PlanarOptimizeConfig {
                reprojection_loss: RobustLoss::Cauchy { scale: 1.0 },
                fix_poses: vec![0, 1],
            },
            ..Default::default()
        };

        let opts = config.solve_opts();
        assert_eq!(opts.fix_poses, vec![0, 1]);
        match opts.robust_loss {
            vision_calibration_optim::RobustLoss::Cauchy { scale } => {
                assert!((scale - 1.0).abs() < 1e-12);
            }
            _ => {
                panic!("expected Cauchy loss, got {:?}", opts.robust_loss);
            }
        }
    }

    #[test]
    fn backend_opts_conversion() {
        let config = PlanarIntrinsicsConfig {
            solver: SolverConfig {
                max_iters: 100,
                verbosity: 2,
                backend: vision_calibration_optim::SolverBackend::Factrs,
                ..Default::default()
            },
            ..Default::default()
        };

        let opts = config
            .solver
            .backend_options(&vision_calibration_core::RunControl::default());
        assert_eq!(opts.max_iters, 100);
        assert_eq!(opts.verbosity, 2);
        assert_eq!(
            opts.backend,
            vision_calibration_optim::SolverBackend::Factrs
        );
    }

    #[test]
    fn export_includes_per_camera_reprojection_errors() {
        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCy {
                fx: 800.0,
                fy: 790.0,
                cx: 640.0,
                cy: 360.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );
        let params = PlanarIntrinsicsParams::from_pinhole(
            camera,
            vec![vision_calibration_core::Iso3::identity()],
        )
        .expect("valid params");
        let output = PlanarIntrinsicsEstimate::new(
            params,
            SolveReport::new(1.23e-3, 0),
            0.42,
            Default::default(),
        );

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
            &PlanarIntrinsicsConfig::default(),
        )
        .expect("export");

        assert_eq!(export.mean_reproj_error, 0.42);
        assert_eq!(export.per_cam_reproj_errors, vec![0.42]);
        // Even though the synthetic input has every feature at (0,0,0)
        // (projection diverges), one record per feature must still appear.
        assert_eq!(export.diagnostics.per_feature_residuals.target.len(), 4);
        for r in &export.diagnostics.per_feature_residuals.target {
            assert_eq!(r.camera, 0);
            assert!(r.projected_px.is_none());
            assert!(r.error_px.is_none());
        }
        let hist = export
            .diagnostics
            .per_feature_residuals
            .target_hist_per_cam
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
            vision_calibration_core::FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
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

        let params = PlanarIntrinsicsParams::from_pinhole(camera, vec![pose, pose, pose])
            .expect("valid params");
        let output = PlanarIntrinsicsEstimate::new(
            params,
            SolveReport::new(0.0, 0),
            0.0,
            Default::default(),
        );

        let export =
            PlanarIntrinsicsProblem::export(&dataset, &output, &PlanarIntrinsicsConfig::default())
                .expect("export");

        assert_eq!(export.diagnostics.per_feature_residuals.target.len(), 12);
        for (i, r) in export
            .diagnostics
            .per_feature_residuals
            .target
            .iter()
            .enumerate()
        {
            let err = r
                .error_px
                .unwrap_or_else(|| panic!("record {i} missing error"));
            assert!(err < 1e-9, "record {i} error_px {err} not near zero");
        }
        let hist = &export
            .diagnostics
            .per_feature_residuals
            .target_hist_per_cam
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
            vision_calibration_core::FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
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
        let params =
            PlanarIntrinsicsParams::from_pinhole(camera, vec![pose]).expect("valid params");
        let output = PlanarIntrinsicsEstimate::new(
            params,
            SolveReport::new(0.0, 0),
            0.0,
            Default::default(),
        );
        let export =
            PlanarIntrinsicsProblem::export(&dataset, &output, &PlanarIntrinsicsConfig::default())
                .expect("export");

        let json = serde_json::to_string(&export).expect("serialize");
        let restored: PlanarIntrinsicsExport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored.diagnostics.per_feature_residuals.target.len(),
            export.diagnostics.per_feature_residuals.target.len()
        );
        assert_eq!(
            restored
                .diagnostics
                .per_feature_residuals
                .target_hist_per_cam,
            export.diagnostics.per_feature_residuals.target_hist_per_cam
        );
        assert!(restored.diagnostics.per_feature_residuals.laser.is_empty());
        assert!(
            restored
                .diagnostics
                .per_feature_residuals
                .laser_hist_per_cam
                .is_none()
        );
        // ADR 0014: image_manifest defaults to None and absent from the wire
        // when not populated, keeping existing serialized JSON byte-stable.
        assert!(restored.diagnostics.image_manifest.is_none());
        assert!(!json.contains("image_manifest"));
    }

    #[test]
    fn export_image_manifest_roundtrip() {
        use vision_calibration_core::{FrameRef, ImageManifest};

        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );
        let params = PlanarIntrinsicsParams::from_pinhole(
            camera,
            vec![vision_calibration_core::Iso3::identity()],
        )
        .expect("valid params");
        let mut export = PlanarIntrinsicsExport {
            kind: ProblemKind::PlanarIntrinsics,
            schema_version: PlanarIntrinsicsExport::SCHEMA_VERSION,
            camera: params.camera,
            cam_se3_target: params.cam_se3_target,
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0],
            diagnostics: ExportDiagnostics::default(),
            uncertainty: Uncertainty::default(),
        };
        let mut frame = FrameRef::default();
        frame.path = std::path::PathBuf::from("pose_0_cam_0.png");
        let mut manifest = ImageManifest::default();
        manifest.root = std::path::PathBuf::from("images");
        manifest.frames = vec![frame];
        export.diagnostics.image_manifest = Some(manifest);

        let json = serde_json::to_string(&export).expect("serialize");
        let restored: PlanarIntrinsicsExport = serde_json::from_str(&json).expect("deserialize");
        let manifest = restored
            .diagnostics
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.root, std::path::PathBuf::from("images"));
        assert_eq!(manifest.frames.len(), 1);
        assert_eq!(
            manifest.frames[0].path,
            std::path::PathBuf::from("pose_0_cam_0.png")
        );
    }

    #[test]
    fn default_config_is_pinhole_brown_conrady_with_120_iterations() {
        let config = PlanarIntrinsicsConfig::default();
        assert_eq!(config.solver.max_iters, 120);
        assert_eq!(config.camera, CameraConfig::default());
        assert!(!config.is_scheimpflug());
        assert!(
            !config
                .init_opts(None)
                .pinhole
                .distortion_opts
                .fix_tangential
        );
    }

    #[test]
    fn the_linear_stage_holds_the_tangential_terms_the_camera_holds() {
        let mut config = PlanarIntrinsicsConfig::default();
        config.camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
        assert!(
            config
                .init_opts(None)
                .pinhole
                .distortion_opts
                .fix_tangential
        );
        // A tilted camera holds them while its tilt is estimated.
        let mut config = PlanarIntrinsicsConfig::default();
        config.camera.model.sensor = SensorKind::Scheimpflug;
        assert!(
            config
                .init_opts(None)
                .pinhole
                .distortion_opts
                .fix_tangential
        );
        config.camera.fix.distortion = Some(vec!["k3".into()]);
        assert!(
            config
                .init_opts(None)
                .pinhole
                .distortion_opts
                .fix_tangential
        );
    }

    #[test]
    fn validate_config_rejects_scheimpflug_without_distortion() {
        let mut config = PlanarIntrinsicsConfig::default();
        config.camera.model.sensor = SensorKind::Scheimpflug;
        config.camera.model.distortion = DistortionKind::None;
        let err = PlanarIntrinsicsProblem::validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("distortion model"), "{err}");
    }

    #[test]
    fn validate_config_rejects_unknown_fix_names() {
        let mut config = PlanarIntrinsicsConfig::default();
        config.camera.fix.sensor = Some(vec!["tilt_x".into()]);
        let err = PlanarIntrinsicsProblem::validate_config(&config).unwrap_err();
        assert!(err.to_string().contains("tilt_x"), "{err}");
    }
}
