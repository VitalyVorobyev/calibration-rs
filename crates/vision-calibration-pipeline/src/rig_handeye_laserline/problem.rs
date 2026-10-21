//! [`ProblemType`] implementation for joint rig hand-eye laserline calibration.

use crate::Error;
use crate::common::ExportKind;
use crate::common::config::{RobotPoseConfig, SolverConfig};
use crate::rig_handeye::{RigHandeyeConfig, RigHandeyeProblem};
use crate::rig_laserline_device::{RigLaserlineDeviceConfig, RigLaserlineDeviceProblem};
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraFix, CameraModelSpec, CameraParams, FeatureResidualHistogram, ImageManifest, Iso3,
    NoMeta, PerFeatureResiduals, Real, RigDataset, RigView, RigViewObs, Uncertainty,
    build_feature_histogram, compute_rig_target_residuals,
};
use vision_calibration_optim::{
    HandEyeMode, LaserPlane, LaserlineResidualType, RigHandeyeLaserlineDataset,
    RigHandeyeLaserlineEstimate, RigHandeyeLaserlinePerCamStats, RigHandeyeLaserlineView,
    RigLaserlineDataset, RobustLoss, compute_rig_laserline_feature_residuals,
};

use crate::rig_family::rig_camera_models;
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::RigHandeyeLaserlineState;

/// Input for joint rig hand-eye laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RigHandeyeLaserlineInput {
    /// Per-view target observations, laser pixels, and robot poses.
    pub views: Vec<RigHandeyeLaserlineView>,
    /// Number of cameras in the rig.
    pub num_cameras: usize,
}

impl RigHandeyeLaserlineInput {
    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    pub(crate) fn laserline_dataset(&self) -> Result<RigLaserlineDataset, Error> {
        RigLaserlineDataset::new(
            self.views.iter().map(|v| v.obs.clone()).collect(),
            self.num_cameras,
        )
        .map_err(Error::from)
    }

    pub(crate) fn joint_dataset(
        &self,
        mode: HandEyeMode,
    ) -> Result<RigHandeyeLaserlineDataset, Error> {
        RigHandeyeLaserlineDataset::new(self.views.clone(), self.num_cameras, mode)
            .map_err(Error::from)
    }

    pub(crate) fn handeye_dataset(&self) -> RigDataset<vision_calibration_optim::RobotPoseMeta> {
        RigDataset {
            num_cameras: self.num_cameras,
            views: self
                .views
                .iter()
                .map(|v| RigView {
                    meta: v.meta.clone(),
                    obs: RigViewObs {
                        cameras: v.obs.cameras.clone(),
                    },
                })
                .collect(),
        }
    }

    fn target_dataset_no_meta(&self) -> RigDataset<NoMeta> {
        RigDataset {
            num_cameras: self.num_cameras,
            views: self
                .views
                .iter()
                .map(|v| RigView {
                    meta: NoMeta,
                    obs: RigViewObs {
                        cameras: v.obs.cameras.clone(),
                    },
                })
                .collect(),
        }
    }
}

/// Pipeline output for joint rig hand-eye laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeLaserlineOutput {
    /// Joint optimizer estimate.
    pub estimate: RigHandeyeLaserlineEstimate,
    /// Hand-eye mode used to build the optimizer dataset.
    pub handeye_mode: HandEyeMode,
}

/// Configuration for joint rig hand-eye laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigHandeyeLaserlineConfig {
    /// Warm-start rig hand-eye stage.
    pub handeye: RigHandeyeConfig,
    /// Frozen-geometry laser plane initialization stage.
    pub laserline_init: RigLaserlineDeviceConfig,
    /// Final joint bundle adjustment stage.
    pub joint_ba: RigHandeyeLaserlineBaConfig,
}

impl Default for RigHandeyeLaserlineConfig {
    fn default() -> Self {
        Self {
            handeye: RigHandeyeConfig::default(),
            laserline_init: RigLaserlineDeviceConfig {
                solver: SolverConfig {
                    max_iters: 200,
                    verbosity: 0,
                    ..SolverConfig::default()
                },
                laser_residual_type: LaserlineResidualType::PointToPlane,
            },
            joint_ba: RigHandeyeLaserlineBaConfig::default(),
        }
    }
}

/// Final joint BA options.
///
/// The joint stage always pins the upstream rig's reference camera
/// (`handeye.rig.reference_camera_idx`) — see `joint_fix_extrinsics`. Each
/// camera's fixed parameters are named lists in that camera's model
/// (`fix`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigHandeyeLaserlineBaConfig {
    /// Non-linear solve stage settings for the joint stage.
    ///
    /// `robust_loss` is **not consulted** by this problem: laser-carrying
    /// stages track calibration and laser residuals as independent families
    /// with their own robust losses (`calib_loss`, `laser_loss`).
    /// `max_iters`, `verbosity` and `backend` apply.
    pub solver: SolverConfig,
    /// Which laser residual drives the joint solve.
    pub laser_residual_type: LaserlineResidualType,
    /// Robust loss applied to target residuals.
    pub calib_loss: RobustLoss,
    /// Robust loss applied to laser residuals.
    pub laser_loss: RobustLoss,
    /// Weight for target corner residuals.
    pub calib_weight: f64,
    /// Weight for laser residuals.
    pub laser_weight: f64,
    /// Camera parameters held fixed in the joint stage, one entry per camera
    /// in camera order, each named per slot in that camera's model
    /// (`{"distortion": ["k3", "p1", "p2"], "sensor": ["tilt_x"]}`); a slot
    /// left out of an entry keeps the model's default. Empty (the default)
    /// gives every camera the stage default: intrinsics and the low-order
    /// radial terms (`k1`, `k2`; the division model's `lambda`) refined,
    /// the higher-order distortion terms and the sensor tilt held at their
    /// warm-start values — they trade off against the extrinsics and the
    /// laser planes in the joint solve. A name the camera's model does not
    /// have is a configuration error.
    #[serde(default)]
    pub fix: Vec<CameraFix>,
    /// Fix the hand-eye transform during the joint stage.
    pub fix_handeye: bool,
    /// Fix the target reference pose during the joint stage.
    pub fix_target_ref: bool,
    /// Robot-pose refinement settings for the joint stage.
    pub robot_poses: RobotPoseConfig,
}

impl Default for RigHandeyeLaserlineBaConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig {
                max_iters: 30,
                verbosity: 0,
                ..SolverConfig::default()
            },
            laser_residual_type: LaserlineResidualType::PointToPlane,
            calib_loss: RobustLoss::None,
            laser_loss: RobustLoss::None,
            calib_weight: 1.0,
            laser_weight: 1.0e4,
            fix: Vec::new(),
            fix_handeye: false,
            fix_target_ref: false,
            robot_poses: RobotPoseConfig::default(),
        }
    }
}

/// Export format for joint rig hand-eye laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RigHandeyeLaserlineExport {
    /// Export-type discriminator — always [`ExportKind::RigHandeyeLaserline`].
    pub kind: ExportKind,
    /// Per-camera laser planes in rig frame.
    pub laser_planes_rig: Vec<LaserPlane>,
    /// Per-camera laser planes in camera frame.
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Joint per-camera stats.
    pub per_camera_stats: Vec<RigHandeyeLaserlinePerCamStats>,
    /// Per-camera parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera extrinsics `cam_se3_rig` (T_C_R).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_rig: Vec<Iso3>,
    /// Per-view target poses `rig_se3_target` (T_R_T).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub rig_se3_target: Vec<Iso3>,
    /// Hand-eye mode used to interpret mode-dependent transforms.
    pub handeye_mode: HandEyeMode,
    /// Eye-in-hand hand-eye transform `gripper_se3_rig`.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub gripper_se3_rig: Option<Iso3>,
    /// Eye-to-hand hand-eye transform `rig_se3_base`.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub rig_se3_base: Option<Iso3>,
    /// Eye-in-hand target reference pose `base_se3_target`.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub base_se3_target: Option<Iso3>,
    /// Eye-to-hand target reference pose `gripper_se3_target`.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub gripper_se3_target: Option<Iso3>,
    /// Optional optimized robot pose deltas.
    pub robot_deltas: Option<Vec<[Real; 6]>>,
    /// Mean target reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera target reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-feature target and laser residuals.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
    /// Optional image manifest populated by app/dataset runners.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
    /// Covariance of the jointly refined parameters, labelled as the export
    /// fields (`cam1.fx`, `cam1.cam_se3_rig.tx`, `gripper_se3_rig.rz`), with
    /// each camera's laser plane in its camera frame (`cam1.plane.nx`).
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

/// Joint rig hand-eye laserline problem.
#[derive(Debug)]
pub struct RigHandeyeLaserlineProblem;

impl ProblemState for RigHandeyeLaserlineProblem {
    type State = RigHandeyeLaserlineState;
}

impl ProblemType for RigHandeyeLaserlineProblem {
    type Config = RigHandeyeLaserlineConfig;
    type Input = RigHandeyeLaserlineInput;
    type Output = RigHandeyeLaserlineOutput;
    type Export = RigHandeyeLaserlineExport;

    fn name() -> &'static str {
        "rig_handeye_laserline_v1"
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        if input.views.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        if input.num_cameras < 2 {
            return Err(Error::InsufficientData {
                need: 2,
                got: input.num_cameras,
            });
        }
        for (view_idx, view) in input.views.iter().enumerate() {
            if view.obs.cameras.len() != input.num_cameras
                || view.obs.laser_pixels.len() != input.num_cameras
            {
                return Err(Error::invalid_input(format!(
                    "view {view_idx} has {}/{} target/laser slots, expected {}",
                    view.obs.cameras.len(),
                    view.obs.laser_pixels.len(),
                    input.num_cameras
                )));
            }
        }
        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        RigHandeyeProblem::validate_config(&config.handeye)?;
        RigLaserlineDeviceProblem::validate_config(&config.laserline_init)?;
        if config.joint_ba.solver.max_iters == 0 {
            return Err(Error::invalid_input(
                "joint_ba.solver.max_iters must be positive",
            ));
        }
        config.joint_ba.robot_poses.validate()?;
        Ok(())
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        crate::rig_handeye::RigHandeyeProblem::validate_input_config(
            &input.handeye_dataset(),
            &config.handeye,
        )?;
        let models: Vec<CameraModelSpec> = (0..input.num_cameras)
            .map(|i| config.handeye.camera_for(i).model)
            .collect();
        joint_fix(&config.joint_ba, &models).map(|_| ())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        export_joint(input, output)
    }
}

pub(crate) fn export_joint(
    input: &RigHandeyeLaserlineInput,
    output: &RigHandeyeLaserlineOutput,
) -> Result<RigHandeyeLaserlineExport, Error> {
    let estimate = &output.estimate;
    let params = &estimate.params;
    let mode = output.handeye_mode;
    let cam_se3_rig: Vec<Iso3> = params.cam_to_rig.iter().map(|t| t.inverse()).collect();
    let rig_se3_target =
        materialize_rig_se3_target(input, params, mode, estimate.robot_deltas.as_deref());
    let target_dataset = input.target_dataset_no_meta();
    let target = compute_rig_target_residuals(
        &rig_camera_models(&params.cameras)?,
        &target_dataset,
        &cam_se3_rig,
        &rig_se3_target,
    )?;
    let target_hist_per_camera: Vec<FeatureResidualHistogram> = (0..input.num_cameras)
        .map(|cam_idx| {
            build_feature_histogram(
                target
                    .iter()
                    .filter(|r| r.camera == cam_idx)
                    .filter_map(|r| r.error_px),
            )
        })
        .collect();

    let laser_dataset = input.laserline_dataset()?;
    let laser = compute_rig_laserline_feature_residuals(
        &laser_dataset,
        &params.cameras,
        &cam_se3_rig,
        &rig_se3_target,
        &params.planes_cam,
    )?;
    let laser_hist_per_camera: Vec<FeatureResidualHistogram> = (0..input.num_cameras)
        .map(|cam_idx| {
            build_feature_histogram(
                laser
                    .features
                    .iter()
                    .filter(|r| r.camera == cam_idx)
                    .filter_map(|r| r.residual_px),
            )
        })
        .collect();

    let mut per_feature_residuals = PerFeatureResiduals::default();
    per_feature_residuals.target = target;
    per_feature_residuals.laser = laser.features;
    per_feature_residuals.laser_lines = laser.lines;
    per_feature_residuals.target_hist_per_camera = Some(target_hist_per_camera);
    per_feature_residuals.laser_hist_per_camera = Some(laser_hist_per_camera);

    let (gripper_se3_rig, rig_se3_base, base_se3_target, gripper_se3_target) = match mode {
        HandEyeMode::EyeInHand => (Some(params.handeye), None, Some(params.target_ref), None),
        HandEyeMode::EyeToHand => (None, Some(params.handeye), None, Some(params.target_ref)),
    };

    Ok(RigHandeyeLaserlineExport {
        kind: ExportKind::RigHandeyeLaserline,
        laser_planes_rig: estimate.planes_rig.clone(),
        laser_planes_cam: params.planes_cam.clone(),
        per_camera_stats: estimate.per_cam_stats.clone(),
        cameras: params.cameras.clone(),
        cam_se3_rig,
        rig_se3_target,
        handeye_mode: mode,
        gripper_se3_rig,
        rig_se3_base,
        base_se3_target,
        gripper_se3_target,
        robot_deltas: estimate.robot_deltas.clone(),
        mean_reproj_error: estimate.mean_reproj_error_px,
        per_cam_reproj_errors: output
            .estimate
            .per_cam_stats
            .iter()
            .map(|s| s.mean_reproj_error_px)
            .collect(),
        per_feature_residuals,
        image_manifest: None,
        uncertainty: estimate.uncertainty.clone(),
    })
}

pub(crate) fn materialize_rig_se3_target(
    input: &RigHandeyeLaserlineInput,
    params: &vision_calibration_optim::RigHandeyeLaserlineParams,
    mode: HandEyeMode,
    robot_deltas: Option<&[[Real; 6]]>,
) -> Vec<Iso3> {
    input
        .views
        .iter()
        .enumerate()
        .map(|(view_idx, view)| {
            let robot_pose = if let Some(deltas) = robot_deltas {
                corrected_robot_pose(view.meta.base_se3_gripper, deltas[view_idx])
            } else {
                view.meta.base_se3_gripper
            };
            match mode {
                HandEyeMode::EyeInHand => {
                    params.handeye.inverse() * robot_pose.inverse() * params.target_ref
                }
                HandEyeMode::EyeToHand => params.handeye * robot_pose * params.target_ref,
            }
        })
        .collect()
}

fn corrected_robot_pose(robot_pose: Iso3, delta: [Real; 6]) -> Iso3 {
    let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
    let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);
    let angle = rot_vec.norm();
    let delta_rot = if angle > 1e-12 {
        UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
    } else {
        UnitQuaternion::identity()
    };
    let delta_iso = Iso3::from_parts(Translation3::from(trans_vec), delta_rot);
    delta_iso * robot_pose
}

/// The joint stage's fixed parameters of each camera of `models`: its entry
/// of `config.fix`, or the stage default when the list is empty.
///
/// # Errors
///
/// [`Error::InvalidInput`] when `config.fix` has neither zero nor one entry
/// per camera, or an entry names a parameter its camera's model does not
/// have.
pub(crate) fn joint_fix(
    config: &RigHandeyeLaserlineBaConfig,
    models: &[CameraModelSpec],
) -> Result<Vec<CameraFix>, Error> {
    if config.fix.is_empty() {
        return Ok(models.iter().map(|m| joint_stage_default_fix(*m)).collect());
    }
    if config.fix.len() != models.len() {
        return Err(Error::invalid_input(format!(
            "joint_ba.fix has {} entries; expected none or one per camera ({})",
            config.fix.len(),
            models.len()
        )));
    }
    for (i, (fix, model)) in config.fix.iter().zip(models).enumerate() {
        fix.resolve(*model)
            .map_err(|e| Error::invalid_input(format!("joint_ba.fix[{i}]: {e}")))?;
    }
    Ok(config.fix.clone())
}

/// The joint stage's default fix of a camera of `model`: intrinsics and the
/// low-order radial terms refined, the higher-order distortion terms and
/// the sensor tilt held.
fn joint_stage_default_fix(model: CameraModelSpec) -> CameraFix {
    let names = |v: &[&str]| Some(v.iter().map(|n| n.to_string()).collect());
    CameraFix {
        intrinsics: Some(Vec::new()),
        projection: Some(Vec::new()),
        distortion: names(&model.distortion.higher_order_names()),
        sensor: names(model.sensor.param_names()),
    }
}

/// The joint stage pins the extrinsic of the configured reference camera
/// (`reference_camera_idx`), not camera 0.
pub(crate) fn joint_fix_extrinsics(reference_camera_idx: usize, n: usize) -> Vec<bool> {
    (0..n).map(|idx| idx == reference_camera_idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use vision_calibration_core::{DistortionKind, SensorKind};

    #[test]
    fn config_json_roundtrip_preserves_joint_defaults() {
        let cfg = RigHandeyeLaserlineConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: RigHandeyeLaserlineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.joint_ba.laser_residual_type,
            LaserlineResidualType::PointToPlane
        );
        assert_eq!(restored.joint_ba.laser_weight, 1.0e4);
        assert!(restored.joint_ba.robot_poses.refine);
        assert!(restored.joint_ba.fix.is_empty());
    }

    #[test]
    fn joint_fix_defaults_hold_higher_order_terms_and_tilt() {
        let untilted =
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
        let tilted =
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
        let division = CameraModelSpec::pinhole(DistortionKind::Division1, SensorKind::Identity);
        let config = RigHandeyeLaserlineBaConfig::default();
        let fix = joint_fix(&config, &[untilted, tilted, division]).unwrap();
        let resolved: Vec<_> = fix
            .iter()
            .zip([untilted, tilted, division])
            .map(|(f, m)| f.resolve(m).unwrap())
            .collect();
        assert!(resolved[0].intrinsics.is_empty());
        assert_eq!(resolved[0].distortion, vec![2, 3, 4]);
        assert!(resolved[0].sensor.is_empty());
        assert_eq!(resolved[1].distortion, vec![2, 3, 4]);
        assert_eq!(resolved[1].sensor, vec![0, 1]);
        assert!(resolved[2].distortion.is_empty());
    }

    #[test]
    fn joint_fix_entries_are_checked_per_camera() {
        let untilted =
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
        let mut config = RigHandeyeLaserlineBaConfig {
            fix: vec![CameraFix::default()],
            ..RigHandeyeLaserlineBaConfig::default()
        };
        let err = joint_fix(&config, &[untilted, untilted])
            .unwrap_err()
            .to_string();
        assert!(err.contains("one per camera"), "{err}");
        config.fix = vec![
            CameraFix::default(),
            CameraFix {
                sensor: Some(vec!["tilt_x".into()]),
                ..CameraFix::default()
            },
        ];
        let err = joint_fix(&config, &[untilted, untilted])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("joint_ba.fix[1]") && err.contains("tilt_x"),
            "{err}"
        );
    }
}
