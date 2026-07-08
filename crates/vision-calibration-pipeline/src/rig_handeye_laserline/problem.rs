//! [`ProblemType`] implementation for joint rig hand-eye laserline calibration.

use crate::Error;
use crate::common::config::{RobotPoseConfig, SolverConfig};
use crate::rig_handeye::{RigHandeyeConfig, RigHandeyeProblem};
use crate::rig_laserline_device::{RigLaserlineDeviceConfig, RigLaserlineDeviceProblem};
use nalgebra::{Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    Camera, CameraFixMask, DistortionFixMask, FeatureResidualHistogram, ImageManifest,
    IntrinsicsFixMask, Iso3, NoMeta, PerFeatureResiduals, Pinhole, PinholeCamera, Real, RigDataset,
    RigView, RigViewObs, ScheimpflugParams, build_feature_histogram, compute_rig_target_residuals,
    make_pinhole_camera,
};
use vision_calibration_optim::{
    HandEyeMode, LaserPlane, LaserlineResidualType, RigHandeyeLaserlineDataset,
    RigHandeyeLaserlineEstimate, RigHandeyeLaserlinePerCamStats, RigHandeyeLaserlineView,
    RigLaserlineDataset, RobustLoss, ScheimpflugFixMask, compute_rig_laserline_feature_residuals,
};

use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::RigHandeyeLaserlineState;

/// Input for joint rig hand-eye laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Final joint BA options (ADR 0024).
///
/// D2 removed `fix_first_camera_extrinsic: bool`: the joint stage now always
/// pins the upstream rig's reference camera (`handeye.rig.reference_camera_idx`)
/// — see `joint_fix_extrinsics`, fixing the old index-0 hard-coding bug.
/// `JointCameraFixMask` (structurally identical to core's [`CameraFixMask`])
/// is deleted in favor of the core type directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RigHandeyeLaserlineBaConfig {
    /// Non-linear solve stage settings for the joint stage.
    ///
    /// `robust_loss` is **not consulted** by this problem: laser-carrying
    /// stages track calibration and laser residuals as independent families
    /// with their own robust losses (`calib_loss`, `laser_loss` — ADR 0024
    /// D3). Only `max_iters`/`verbosity` apply here.
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
    /// Default camera parameter fix mask applied to every camera.
    pub default_camera_fix: CameraFixMask,
    /// Fix Scheimpflug tilt parameters during the joint stage.
    pub fix_scheimpflug: ScheimpflugFixMask,
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
            default_camera_fix: CameraFixMask {
                intrinsics: IntrinsicsFixMask::all_free(),
                distortion: DistortionFixMask {
                    k1: false,
                    k2: false,
                    k3: true,
                    p1: true,
                    p2: true,
                },
            },
            fix_scheimpflug: ScheimpflugFixMask {
                tilt_x: true,
                tilt_y: true,
            },
            fix_handeye: false,
            fix_target_ref: false,
            robot_poses: RobotPoseConfig::default(),
        }
    }
}

/// Export format for joint rig hand-eye laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigHandeyeLaserlineExport {
    /// Per-camera laser planes in rig frame.
    pub laser_planes_rig: Vec<LaserPlane>,
    /// Per-camera laser planes in camera frame.
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Joint per-camera stats.
    pub per_camera_stats: Vec<RigHandeyeLaserlinePerCamStats>,
    /// Per-camera pinhole camera parameters.
    pub cameras: Vec<PinholeCamera>,
    /// Per-camera Scheimpflug sensor parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera extrinsics `cam_se3_rig` (T_C_R).
    pub cam_se3_rig: Vec<Iso3>,
    /// Per-view target poses `rig_se3_target` (T_R_T).
    pub rig_se3_target: Vec<Iso3>,
    /// Hand-eye mode used to interpret mode-dependent transforms.
    pub handeye_mode: HandEyeMode,
    /// Eye-in-hand hand-eye transform `gripper_se3_rig`.
    pub gripper_se3_rig: Option<Iso3>,
    /// Eye-to-hand hand-eye transform `rig_se3_base`.
    pub rig_se3_base: Option<Iso3>,
    /// Eye-in-hand target reference pose `base_se3_target`.
    pub base_se3_target: Option<Iso3>,
    /// Eye-to-hand target reference pose `gripper_se3_target`.
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
        )
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
    let cameras_for_residuals: Vec<_> = params
        .cameras
        .iter()
        .zip(params.sensors.iter())
        .map(|(cam, sensor)| Camera::new(Pinhole, cam.dist, sensor.compile(), cam.k))
        .collect();
    let target = compute_rig_target_residuals(
        &cameras_for_residuals,
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
    let laser_intrinsics: Vec<_> = params.cameras.iter().map(|c| c.k).collect();
    let laser_distortion: Vec<_> = params.cameras.iter().map(|c| c.dist).collect();
    let laser = compute_rig_laserline_feature_residuals(
        &laser_dataset,
        &laser_intrinsics,
        &laser_distortion,
        &params.sensors,
        &cam_se3_rig,
        &rig_se3_target,
        &params.planes_cam,
    )?;
    let laser_hist_per_camera: Vec<FeatureResidualHistogram> = (0..input.num_cameras)
        .map(|cam_idx| {
            build_feature_histogram(
                laser
                    .iter()
                    .filter(|r| r.camera == cam_idx)
                    .filter_map(|r| r.residual_px),
            )
        })
        .collect();

    let mut per_feature_residuals = PerFeatureResiduals::default();
    per_feature_residuals.target = target;
    per_feature_residuals.laser = laser;
    per_feature_residuals.target_hist_per_camera = Some(target_hist_per_camera);
    per_feature_residuals.laser_hist_per_camera = Some(laser_hist_per_camera);

    let (gripper_se3_rig, rig_se3_base, base_se3_target, gripper_se3_target) = match mode {
        HandEyeMode::EyeInHand => (Some(params.handeye), None, Some(params.target_ref), None),
        HandEyeMode::EyeToHand => (None, Some(params.handeye), None, Some(params.target_ref)),
    };

    Ok(RigHandeyeLaserlineExport {
        laser_planes_rig: estimate.planes_rig.clone(),
        laser_planes_cam: params.planes_cam.clone(),
        per_camera_stats: estimate.per_cam_stats.clone(),
        cameras: params
            .cameras
            .iter()
            .map(|c| make_pinhole_camera(c.k, c.dist))
            .collect(),
        sensors: params.sensors.clone(),
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

pub(crate) fn joint_fix_intrinsics(
    config: &RigHandeyeLaserlineBaConfig,
    n: usize,
) -> Vec<CameraFixMask> {
    vec![config.default_camera_fix; n]
}

pub(crate) fn joint_fix_scheimpflug(
    config: &RigHandeyeLaserlineBaConfig,
    n: usize,
) -> Vec<ScheimpflugFixMask> {
    vec![config.fix_scheimpflug; n]
}

/// D2 (ADR 0024): the joint stage always pins the upstream rig's reference
/// camera extrinsic — the old `fix_first_camera_extrinsic: bool` knob
/// hard-coded index 0 regardless of `reference_camera_idx` (a bug); this
/// always honors the configured reference camera instead.
pub(crate) fn joint_fix_extrinsics(reference_camera_idx: usize, n: usize) -> Vec<bool> {
    (0..n).map(|idx| idx == reference_camera_idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(restored.joint_ba.default_camera_fix.distortion.p1);
        assert!(restored.joint_ba.default_camera_fix.distortion.p2);
    }
}
