//! [`ProblemType`] implementation for multi-camera rig extrinsics calibration.
//!
//! This module provides the `RigExtrinsicsProblem` type that implements
//! the session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    Camera, FeatureResidualHistogram, Iso3, NoMeta, PerFeatureResiduals, Pinhole, PinholeCamera,
    RigDataset, ScheimpflugParams, build_feature_histogram, compute_rig_target_residuals,
};
use vision_calibration_optim::{
    RigExtrinsicsEstimate as PinholeRigExtrinsicsEstimate,
    RigExtrinsicsScheimpflugEstimate as ScheimpflugRigExtrinsicsEstimate, RobustLoss,
};

pub use crate::rig_family::SensorMode;

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigExtrinsicsState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Type
// ─────────────────────────────────────────────────────────────────────────────

/// Input for rig extrinsics calibration.
///
/// Reuses `RigDataset<NoMeta>` from vision_calibration_core.
pub type RigExtrinsicsInput = RigDataset<NoMeta>;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-camera rig extrinsics calibration.
///
/// Shared between pinhole and Scheimpflug rigs; the [`SensorMode`] field
/// `sensor` selects the sensor flavour.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigExtrinsicsConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Per-camera intrinsics options
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of iterations for iterative intrinsics estimation.
    pub intrinsics_init_iterations: usize,

    /// Fix k3 during intrinsics calibration.
    pub fix_k3: bool,

    /// Fix tangential distortion (p1, p2).
    pub fix_tangential: bool,

    /// Enforce zero skew.
    pub zero_skew: bool,

    /// Sensor flavour (pinhole or Scheimpflug).
    #[serde(default)]
    pub sensor: SensorMode,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig options
    // ─────────────────────────────────────────────────────────────────────────
    /// Reference camera index for rig frame (identity extrinsics).
    pub reference_camera_idx: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Optimization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum iterations for optimization.
    pub max_iters: usize,

    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,

    /// Robust loss function for outlier handling.
    pub robust_loss: RobustLoss,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig BA options
    // ─────────────────────────────────────────────────────────────────────────
    /// Re-refine intrinsics in rig BA (default: false).
    pub refine_intrinsics_in_rig_ba: bool,

    /// Fix first rig pose for gauge freedom (default: true, fixes view 0).
    pub fix_first_rig_pose: bool,
}

impl Default for RigExtrinsicsConfig {
    fn default() -> Self {
        Self {
            // Intrinsics
            intrinsics_init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            sensor: SensorMode::default(),
            // Rig
            reference_camera_idx: 0,
            // Optimization
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            // Rig BA
            refine_intrinsics_in_rig_ba: false,
            fix_first_rig_pose: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the rig BA stage. Pinhole and Scheimpflug rigs return
/// structurally similar but type-distinct optim estimates; this enum
/// preserves both as `Self::Output` for [`RigExtrinsicsProblem`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RigExtrinsicsOutput {
    /// Pinhole rig BA estimate.
    Pinhole(PinholeRigExtrinsicsEstimate),
    /// Scheimpflug rig BA estimate (per-camera sensors included).
    Scheimpflug(ScheimpflugRigExtrinsicsEstimate),
}

impl RigExtrinsicsOutput {
    pub fn final_cost(&self) -> f64 {
        match self {
            Self::Pinhole(e) => e.report.final_cost,
            Self::Scheimpflug(e) => e.report.final_cost,
        }
    }

    pub fn mean_reproj_error(&self) -> f64 {
        match self {
            Self::Pinhole(e) => e.mean_reproj_error,
            Self::Scheimpflug(e) => e.mean_reproj_error,
        }
    }

    pub fn cameras(&self) -> &[PinholeCamera] {
        match self {
            Self::Pinhole(e) => &e.params.cameras,
            Self::Scheimpflug(e) => &e.params.cameras,
        }
    }

    pub fn cam_to_rig(&self) -> &[Iso3] {
        match self {
            Self::Pinhole(e) => &e.params.cam_to_rig,
            Self::Scheimpflug(e) => &e.params.cam_to_rig,
        }
    }

    pub fn rig_from_target(&self) -> &[Iso3] {
        match self {
            Self::Pinhole(e) => &e.params.rig_from_target,
            Self::Scheimpflug(e) => &e.params.rig_from_target,
        }
    }

    pub fn sensors(&self) -> Option<&[ScheimpflugParams]> {
        match self {
            Self::Pinhole(_) => None,
            Self::Scheimpflug(e) => Some(&e.params.sensors),
        }
    }

    pub fn per_cam_reproj_errors(&self) -> &[f64] {
        match self {
            Self::Pinhole(e) => &e.per_cam_reproj_errors,
            Self::Scheimpflug(e) => &e.per_cam_reproj_errors,
        }
    }
}

/// Export format for rig extrinsics calibration.
///
/// Common to pinhole and Scheimpflug rigs. `sensors` is `None` for pinhole
/// rigs and `Some(_)` for Scheimpflug rigs, matching the configured
/// [`SensorMode`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigExtrinsicsExport {
    /// Per-camera calibrated intrinsics + distortion (pinhole core).
    pub cameras: Vec<PinholeCamera>,

    /// Per-camera Scheimpflug sensor parameters. `None` for pinhole rigs;
    /// `Some(_)` for Scheimpflug rigs (one entry per camera).
    #[serde(default)]
    pub sensors: Option<Vec<ScheimpflugParams>>,

    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    /// Transform from rig frame to camera frame.
    pub cam_se3_rig: Vec<Iso3>,

    /// Per-view rig poses: `rig_se3_target` (T_R_T).
    pub rig_se3_target: Vec<Iso3>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,

    /// Per-camera reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,

    /// Per-feature reprojection residuals (ADR 0012). For rig extrinsics
    /// `target` is populated and `laser` is empty. `target_hist_per_camera`
    /// is `Some(vec)` with one entry per camera.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
}

// ─────────────────────────────────────────────────────────────────────────────
// ProblemType Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-camera rig extrinsics calibration problem.
///
/// Calibrates a multi-camera rig, including:
/// - Per-camera intrinsics and distortion
/// - Per-camera extrinsics (camera-to-rig transforms)
/// - Per-view rig poses (rig-to-target)
///
/// # Conventions
///
/// - `cam_se3_rig` = T_C_R (transform from rig to camera frame)
/// - Reference camera has identity extrinsics (defines rig frame)
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::rig_extrinsics::{
///     RigExtrinsicsProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
///     step_rig_init, step_rig_optimize,
/// };
/// # fn main() -> anyhow::Result<()> {
/// # let rig_dataset = unimplemented!();
///
/// let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
/// session.set_input(rig_dataset)?;
///
/// step_intrinsics_init_all(&mut session, None)?;
/// step_intrinsics_optimize_all(&mut session, None)?;
/// step_rig_init(&mut session)?;
/// step_rig_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RigExtrinsicsProblem;

impl ProblemType for RigExtrinsicsProblem {
    type Config = RigExtrinsicsConfig;
    type Input = RigExtrinsicsInput;
    type State = RigExtrinsicsState;
    type Output = RigExtrinsicsOutput;
    type Export = RigExtrinsicsExport;

    fn name() -> &'static str {
        "rig_extrinsics_v2"
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

        if input.num_cameras < 2 {
            return Err(Error::InsufficientData {
                need: 2,
                got: input.num_cameras,
            });
        }

        // Check each view has correct number of cameras
        for (i, view) in input.views.iter().enumerate() {
            if view.obs.cameras.len() != input.num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {} has {} cameras, expected {}",
                    i,
                    view.obs.cameras.len(),
                    input.num_cameras
                )));
            }

            // Check at least one camera has observations in this view
            let has_obs = view.obs.cameras.iter().any(|c| c.is_some());
            if !has_obs {
                return Err(Error::invalid_input(format!(
                    "view {} has no observations from any camera",
                    i
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.intrinsics_init_iterations == 0 {
            return Err(Error::invalid_input(
                "intrinsics_init_iterations must be positive",
            ));
        }
        Ok(())
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        if config.reference_camera_idx >= input.num_cameras {
            return Err(Error::invalid_input(format!(
                "reference_camera_idx {} is out of range (num_cameras = {})",
                config.reference_camera_idx, input.num_cameras
            )));
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
        let cam_se3_rig: Vec<Iso3> = output.cam_to_rig().iter().map(|t| t.inverse()).collect();
        let rig_se3_target = output.rig_from_target().to_vec();
        let num_cameras = input.num_cameras;

        // Build the projection-ready cameras. For Scheimpflug rigs we splice
        // the per-camera sensor into the pinhole core so `compute_rig_target_residuals`
        // sees the full tilted projection chain; pinhole rigs use the cameras
        // verbatim (the trait works via the same `RigCameraProject`).
        let target = match output {
            RigExtrinsicsOutput::Pinhole(estimate) => compute_rig_target_residuals(
                &estimate.params.cameras,
                input,
                &cam_se3_rig,
                &rig_se3_target,
            )?,
            RigExtrinsicsOutput::Scheimpflug(estimate) => {
                let scheimpflug_cameras: Vec<_> = estimate
                    .params
                    .cameras
                    .iter()
                    .zip(estimate.params.sensors.iter())
                    .map(|(cam, sensor)| Camera::new(Pinhole, cam.dist, sensor.compile(), cam.k))
                    .collect();
                compute_rig_target_residuals(
                    &scheimpflug_cameras,
                    input,
                    &cam_se3_rig,
                    &rig_se3_target,
                )?
            }
        };

        let target_hist_per_camera: Vec<FeatureResidualHistogram> = (0..num_cameras)
            .map(|cam_idx| {
                build_feature_histogram(
                    target
                        .iter()
                        .filter(|r| r.camera == cam_idx)
                        .filter_map(|r| r.error_px),
                )
            })
            .collect();

        Ok(RigExtrinsicsExport {
            cameras: output.cameras().to_vec(),
            sensors: output.sensors().map(|s| s.to_vec()),
            cam_se3_rig,
            rig_se3_target,
            mean_reproj_error: output.mean_reproj_error(),
            per_cam_reproj_errors: output.per_cam_reproj_errors().to_vec(),
            per_feature_residuals: PerFeatureResiduals {
                target,
                laser: Vec::new(),
                target_hist_per_camera: Some(target_hist_per_camera),
                laser_hist_per_camera: None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{CorrespondenceView, Pt2, Pt3, RigView, RigViewObs};

    fn make_minimal_obs() -> CorrespondenceView {
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
        .unwrap()
    }

    fn make_minimal_input() -> RigExtrinsicsInput {
        let views = (0..3)
            .map(|_| RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs()), Some(make_minimal_obs())],
                },
            })
            .collect();

        RigDataset::new(views, 2).unwrap()
    }

    #[test]
    fn validate_input_requires_3_views() {
        let input = make_minimal_input();
        let result = RigExtrinsicsProblem::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_requires_2_cameras() {
        // Create single-camera input (should fail)
        let views = (0..3)
            .map(|_| RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs())],
                },
            })
            .collect();

        let input = RigDataset::new(views, 1).unwrap();
        let result = RigExtrinsicsProblem::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("need 2"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = RigExtrinsicsConfig::default();
        let result = RigExtrinsicsProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_config_checks_reference_camera() {
        let input = make_minimal_input();
        let config = RigExtrinsicsConfig {
            reference_camera_idx: 5, // Out of range
            ..RigExtrinsicsConfig::default()
        };

        let result = RigExtrinsicsProblem::validate_input_config(&input, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reference_camera_idx")
        );
    }

    #[test]
    fn config_json_roundtrip() {
        let config = RigExtrinsicsConfig {
            max_iters: 100,
            reference_camera_idx: 1,
            refine_intrinsics_in_rig_ba: true,
            robust_loss: RobustLoss::Huber { scale: 2.5 },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigExtrinsicsConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.max_iters, 100);
        assert_eq!(restored.reference_camera_idx, 1);
        assert!(restored.refine_intrinsics_in_rig_ba);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigExtrinsicsProblem::name(), "rig_extrinsics_v2");
        assert_eq!(RigExtrinsicsProblem::schema_version(), 1);
    }

    #[test]
    fn export_target_residuals_zero_for_perfect_rig_data() {
        // Synthesize a 2-camera 3-view rig where every observed pixel was
        // generated by projecting through the calibrated camera + extrinsics
        // + target poses. The exported per_feature_residuals.target must
        // therefore have error_px ~ 0 for every record, and per-camera
        // histograms must place all counts in the <=1 px bucket.
        use vision_calibration_core::{BrownConrady5, FxFyCxCySkew, View, make_pinhole_camera};
        use vision_calibration_optim::{RigExtrinsicsParams, SolveReport};

        let make_camera = || {
            make_pinhole_camera(
                FxFyCxCySkew {
                    fx: 800.0,
                    fy: 800.0,
                    cx: 320.0,
                    cy: 240.0,
                    skew: 0.0,
                },
                BrownConrady5::default(),
            )
        };
        let cam0 = make_camera();
        let cam1 = make_camera();
        // T_R_C convention: cam_to_rig (line 30 of rig_extrinsics.rs).
        let cam_to_rig = vec![
            Iso3::identity(),
            Iso3::from_parts(
                nalgebra::Translation3::new(-0.2, 0.0, 0.0),
                nalgebra::UnitQuaternion::identity(),
            ),
        ];
        let cam_se3_rig: Vec<Iso3> = cam_to_rig.iter().map(|t| t.inverse()).collect();
        let rig_from_target = vec![
            Iso3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 1.0),
                nalgebra::UnitQuaternion::identity(),
            ),
            Iso3::from_parts(
                nalgebra::Translation3::new(0.05, 0.0, 1.1),
                nalgebra::UnitQuaternion::identity(),
            ),
            Iso3::from_parts(
                nalgebra::Translation3::new(0.0, 0.05, 1.05),
                nalgebra::UnitQuaternion::identity(),
            ),
        ];
        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];

        let mut views = Vec::new();
        for rig_se3_target in &rig_from_target {
            let mut cameras = Vec::new();
            for (cam_idx, cam) in [&cam0, &cam1].iter().enumerate() {
                let cam_se3_target = cam_se3_rig[cam_idx] * rig_se3_target;
                let pixels: Vec<Pt2> = board
                    .iter()
                    .map(|p| {
                        let p_cam = cam_se3_target * p;
                        cam.project_point_c(&p_cam.coords).unwrap()
                    })
                    .collect();
                cameras.push(Some(
                    CorrespondenceView::new(board.clone(), pixels).unwrap(),
                ));
            }
            views.push(RigView {
                meta: NoMeta,
                obs: RigViewObs { cameras },
            });
        }
        let _ = View::<NoMeta>::without_meta; // silence unused import warning if any
        let dataset = RigDataset::new(views, 2).unwrap();

        let output = RigExtrinsicsOutput::Pinhole(PinholeRigExtrinsicsEstimate {
            params: RigExtrinsicsParams {
                cameras: vec![cam0, cam1],
                cam_to_rig,
                rig_from_target,
            },
            report: SolveReport { final_cost: 0.0 },
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0, 0.0],
        });

        let export =
            RigExtrinsicsProblem::export(&dataset, &output, &RigExtrinsicsConfig::default())
                .expect("export");

        // 2 cameras × 3 views × 4 features = 24 records.
        assert_eq!(export.per_feature_residuals.target.len(), 24);
        for r in &export.per_feature_residuals.target {
            let err = r.error_px.unwrap_or_else(|| panic!("residual missing"));
            assert!(err < 1e-9, "error_px {err} not near zero");
        }
        let hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .as_ref()
            .expect("rig export emits per-camera histograms");
        assert_eq!(hist.len(), 2);
        for (i, h) in hist.iter().enumerate() {
            assert_eq!(h.count, 12, "cam {i} count mismatch");
            assert_eq!(h.counts, [12, 0, 0, 0, 0], "cam {i} buckets");
        }
        assert!(export.per_feature_residuals.laser.is_empty());
        assert!(export.per_feature_residuals.laser_hist_per_camera.is_none());
    }

    #[test]
    fn export_per_feature_residuals_json_roundtrip() {
        // Round-trip the rig export through JSON.
        use vision_calibration_core::{BrownConrady5, FxFyCxCySkew, make_pinhole_camera};
        use vision_calibration_optim::{RigExtrinsicsParams, SolveReport};

        let camera = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        // One pose, one view, two cameras with the same model — produces
        // 8 records.
        let cam_to_rig = vec![Iso3::identity(), Iso3::identity()];
        let rig_from_target = vec![Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        )];
        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let cam_se3_target = rig_from_target[0];
        let pixels: Vec<Pt2> = board
            .iter()
            .map(|p| {
                let p_cam = cam_se3_target * p;
                camera.project_point_c(&p_cam.coords).unwrap()
            })
            .collect();
        let view = RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![
                    Some(CorrespondenceView::new(board.clone(), pixels.clone()).unwrap()),
                    Some(CorrespondenceView::new(board.clone(), pixels).unwrap()),
                ],
            },
        };
        let dataset = RigDataset::new(vec![view], 2).unwrap();

        let output = RigExtrinsicsOutput::Pinhole(PinholeRigExtrinsicsEstimate {
            params: RigExtrinsicsParams {
                cameras: vec![camera.clone(), camera],
                cam_to_rig,
                rig_from_target,
            },
            report: SolveReport { final_cost: 0.0 },
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0, 0.0],
        });
        let export =
            RigExtrinsicsProblem::export(&dataset, &output, &RigExtrinsicsConfig::default())
                .expect("export");

        let json = serde_json::to_string(&export).unwrap();
        let restored: RigExtrinsicsExport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.per_feature_residuals.target.len(),
            export.per_feature_residuals.target.len()
        );
        assert_eq!(
            restored.per_feature_residuals.target_hist_per_camera,
            export.per_feature_residuals.target_hist_per_camera
        );
    }
}
