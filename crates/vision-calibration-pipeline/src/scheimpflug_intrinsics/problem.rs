//! [`ProblemType`] implementation for Scheimpflug planar intrinsics calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, DistortionFixMask, IntrinsicsFixMask, Iso3, PerFeatureResiduals, PlanarDataset,
    build_feature_histogram, compute_planar_target_residuals,
};
use vision_calibration_optim::{RobustLoss, SolveReport};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::ScheimpflugIntrinsicsState;

/// Planar intrinsics calibration problem with Scheimpflug sensor tilt.
#[derive(Debug)]
pub struct ScheimpflugIntrinsicsProblem;

/// Input dataset for Scheimpflug intrinsics calibration.
pub type ScheimpflugIntrinsicsInput = PlanarDataset;

/// Optimization mask for Scheimpflug tilt parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ScheimpflugFixMask {
    /// Keep `tilt_x` fixed during optimization.
    pub tilt_x: bool,
    /// Keep `tilt_y` fixed during optimization.
    pub tilt_y: bool,
}

impl ScheimpflugFixMask {
    /// Convert this mask into fixed parameter indices `[tilt_x, tilt_y] -> [0, 1]`.
    pub fn to_indices(self) -> Vec<usize> {
        let mut indices = Vec::new();
        if self.tilt_x {
            indices.push(0);
        }
        if self.tilt_y {
            indices.push(1);
        }
        indices
    }
}

/// Configuration for planar Scheimpflug intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScheimpflugIntrinsicsConfig {
    /// Number of iterative linear intrinsics initialization rounds.
    pub init_iterations: usize,
    /// Keep `k3` fixed in the linear initialization stage.
    pub fix_k3_in_init: bool,
    /// Enforce zero skew in initialization.
    pub zero_skew: bool,
    /// Maximum LM iterations for non-linear optimization.
    pub max_iters: usize,
    /// Backend verbosity level.
    pub verbosity: usize,
    /// Robust loss applied per reprojection residual.
    pub robust_loss: RobustLoss,
    /// Intrinsics parameter fix mask for non-linear optimization.
    pub fix_intrinsics: IntrinsicsFixMask,
    /// Distortion parameter fix mask for non-linear optimization.
    pub fix_distortion: DistortionFixMask,
    /// Scheimpflug tilt parameter fix mask for non-linear optimization.
    pub fix_scheimpflug: ScheimpflugFixMask,
    /// Keep the first pose fixed to remove gauge ambiguity.
    pub fix_first_pose: bool,
}

impl Default for ScheimpflugIntrinsicsConfig {
    fn default() -> Self {
        Self {
            init_iterations: 2,
            fix_k3_in_init: true,
            zero_skew: true,
            max_iters: 120,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::default(),
            fix_distortion: DistortionFixMask::radial_only(),
            fix_scheimpflug: ScheimpflugFixMask::default(),
            fix_first_pose: true,
        }
    }
}

/// Output parameter pack for Scheimpflug intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsParams {
    /// Estimated camera model including intrinsics, distortion, and sensor parameters.
    pub camera: CameraParams,
    /// Estimated pose `camera_se3_target` for each view.
    pub camera_se3_target: Vec<Iso3>,
}

/// Calibration output including parameters, solver report, and summary metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsResult {
    /// Estimated parameters.
    pub params: ScheimpflugIntrinsicsParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean per-point reprojection error in pixels.
    pub mean_reproj_error: f64,
}

/// Export format for Scheimpflug intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScheimpflugIntrinsicsExport {
    /// Estimated parameters.
    pub params: ScheimpflugIntrinsicsParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean per-point reprojection error in pixels.
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors (single element for single-camera workflows).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-feature reprojection residuals (ADR 0012). Single-camera, target
    /// only — `laser` is empty; `target_hist_per_camera` is
    /// `Some(vec![one_entry])`.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
}

impl ProblemType for ScheimpflugIntrinsicsProblem {
    type Config = ScheimpflugIntrinsicsConfig;
    type Input = ScheimpflugIntrinsicsInput;
    type State = ScheimpflugIntrinsicsState;
    type Output = ScheimpflugIntrinsicsResult;
    type Export = ScheimpflugIntrinsicsExport;

    fn name() -> &'static str {
        "scheimpflug_intrinsics_v1"
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

        for (view_idx, view) in input.views.iter().enumerate() {
            if view.obs.len() < 4 {
                return Err(Error::invalid_input(format!(
                    "view {} has too few points (need >= 4, got {})",
                    view_idx,
                    view.obs.len()
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.init_iterations == 0 {
            return Err(Error::invalid_input("init_iterations must be positive"));
        }
        if config.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        let camera = output.params.camera.build();
        let target =
            compute_planar_target_residuals(&camera, input, &output.params.camera_se3_target)?;
        let target_hist = build_feature_histogram(target.iter().filter_map(|r| r.error_px));

        Ok(ScheimpflugIntrinsicsExport {
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        CorrespondenceView, DistortionParams, IntrinsicsParams, NoMeta, ProjectionParams,
        SensorParams, View,
    };

    fn make_minimal_dataset() -> PlanarDataset {
        let make_view = || {
            View::new(
                CorrespondenceView {
                    points_3d: vec![
                        vision_calibration_core::Pt3::new(0.0, 0.0, 0.0),
                        vision_calibration_core::Pt3::new(0.05, 0.0, 0.0),
                        vision_calibration_core::Pt3::new(0.05, 0.05, 0.0),
                        vision_calibration_core::Pt3::new(0.0, 0.05, 0.0),
                    ],
                    points_2d: vec![
                        vision_calibration_core::Pt2::new(100.0, 100.0),
                        vision_calibration_core::Pt2::new(200.0, 100.0),
                        vision_calibration_core::Pt2::new(200.0, 200.0),
                        vision_calibration_core::Pt2::new(100.0, 200.0),
                    ],
                    weights: Vec::new(),
                },
                NoMeta,
            )
        };

        PlanarDataset::new(vec![make_view(), make_view(), make_view()]).expect("valid dataset")
    }

    #[test]
    fn problem_name_and_schema_version() {
        assert_eq!(
            ScheimpflugIntrinsicsProblem::name(),
            "scheimpflug_intrinsics_v1"
        );
        assert_eq!(ScheimpflugIntrinsicsProblem::schema_version(), 1);
    }

    #[test]
    fn validate_input_rejects_too_few_views() {
        let view = make_minimal_dataset().views[0].clone();
        let dataset = PlanarDataset::new(vec![view]).expect("single-view dataset");
        let err = ScheimpflugIntrinsicsProblem::validate_input(&dataset)
            .expect_err("should reject less than 3 views");
        assert!(err.to_string().contains("need 3"));
    }

    #[test]
    fn validate_input_rejects_too_few_points() {
        let mut dataset = make_minimal_dataset();
        let reduced = CorrespondenceView::new(
            dataset.views[0]
                .obs
                .points_3d
                .iter()
                .take(3)
                .copied()
                .collect(),
            dataset.views[0]
                .obs
                .points_2d
                .iter()
                .take(3)
                .copied()
                .collect(),
        )
        .expect("valid reduced correspondence");
        dataset.views[0] = View::without_meta(reduced);

        let err = ScheimpflugIntrinsicsProblem::validate_input(&dataset)
            .expect_err("should reject less than 4 points");
        assert!(err.to_string().contains("too few points"));
    }

    #[test]
    fn validate_config_rejects_zero_iterations() {
        let config = ScheimpflugIntrinsicsConfig {
            init_iterations: 0,
            ..Default::default()
        };
        let err = ScheimpflugIntrinsicsProblem::validate_config(&config)
            .expect_err("invalid config expected");
        assert!(err.to_string().contains("init_iterations"));
    }

    #[test]
    fn config_json_roundtrip() {
        let config = ScheimpflugIntrinsicsConfig {
            init_iterations: 3,
            max_iters: 70,
            robust_loss: RobustLoss::Huber { scale: 1.2 },
            fix_scheimpflug: ScheimpflugFixMask {
                tilt_x: true,
                tilt_y: false,
            },
            ..Default::default()
        };

        let json = serde_json::to_string(&config).expect("serialize config");
        let restored: ScheimpflugIntrinsicsConfig =
            serde_json::from_str(&json).expect("deserialize config");

        assert_eq!(restored.init_iterations, 3);
        assert_eq!(restored.max_iters, 70);
        assert!(matches!(
            restored.robust_loss,
            RobustLoss::Huber { scale } if (scale - 1.2).abs() < 1e-12
        ));
        assert!(restored.fix_scheimpflug.tilt_x);
        assert!(!restored.fix_scheimpflug.tilt_y);
    }

    #[test]
    fn export_returns_output_clone() {
        let output = ScheimpflugIntrinsicsResult {
            params: ScheimpflugIntrinsicsParams {
                camera: CameraParams {
                    projection: ProjectionParams::Pinhole,
                    distortion: DistortionParams::BrownConrady5 {
                        params: vision_calibration_core::BrownConrady5::default(),
                    },
                    sensor: SensorParams::Scheimpflug {
                        params: vision_calibration_core::ScheimpflugParams::default(),
                    },
                    intrinsics: IntrinsicsParams::FxFyCxCySkew {
                        params: vision_calibration_core::FxFyCxCySkew {
                            fx: 800.0,
                            fy: 780.0,
                            cx: 640.0,
                            cy: 360.0,
                            skew: 0.0,
                        },
                    },
                },
                camera_se3_target: vec![Iso3::identity()],
            },
            report: SolveReport { final_cost: 1.0 },
            mean_reproj_error: 0.25,
        };

        let dummy_view = vision_calibration_core::View::without_meta(
            vision_calibration_core::CorrespondenceView::new(
                vec![vision_calibration_core::Pt3::new(0.0, 0.0, 0.0); 4],
                vec![vision_calibration_core::Pt2::new(0.0, 0.0); 4],
            )
            .unwrap(),
        );
        let dummy_input = vision_calibration_core::PlanarDataset::new(vec![dummy_view]).unwrap();
        let exported =
            ScheimpflugIntrinsicsProblem::export(&dummy_input, &output, &Default::default())
                .expect("export should succeed");

        assert_eq!(exported.mean_reproj_error, output.mean_reproj_error);
        assert_eq!(
            exported.per_cam_reproj_errors,
            vec![output.mean_reproj_error]
        );
        assert_eq!(exported.params.camera_se3_target.len(), 1);
    }
}
