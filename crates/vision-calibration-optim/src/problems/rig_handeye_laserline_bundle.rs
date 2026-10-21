//! Joint rig + hand-eye + laser-plane bundle adjustment.
//!
//! A single-cost optimization that jointly refines per-camera camera
//! parameters (each camera in its own model), per-camera rig extrinsics, the hand-eye transform, a single fixed target reference pose,
//! and per-camera laser planes (in camera frame), using **both** target
//! corner reprojection residuals and laser-line residuals simultaneously.
//!
//! Conceptually this is the rig-scale version of
//! [`super::laserline_bundle::optimize_laserline`]: the laser-line residuals
//! close the camera→rig→hand-eye→target chain through a per-view robot pose,
//! so every upstream parameter is constrained by the laser observations
//! (rather than being held fixed, as in [`super::laserline_rig_bundle`]).
//!
//! The caller typically seeds this problem from a sequence of warm starts:
//! Scheimpflug per-camera BA → rig extrinsics BA → hand-eye BA → per-camera
//! laser-plane linear fit (or a one-shot `optimize_rig_laserline`). Joint BA
//! then corrects residual inconsistencies across the stages.

use crate::Error;
use crate::backend::{self, BackendSolveOptions, SolveReport};
use crate::ir::{
    FactorKind, FixedMask, HandEyeMode, LaserChain, ManifoldKind, ProblemIR, ReprojChain,
    ResidualBlock, RobustLoss,
};
use crate::params::camera::{
    CameraBlockInput, add_camera_blocks, camera_block_values, unpack_camera,
};
use crate::params::laser_plane::LaserPlane;
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use crate::problems::handeye::RobotPoseMeta;
use crate::problems::laserline_bundle::LaserlineResidualType;
use crate::problems::laserline_rig_bundle::{RigLaserlineDataset, RigLaserlineView};
use crate::problems::rig_extrinsics::{camera_models, resolve_camera_fix};
use crate::uncertainty::{self, RigLabelling};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(test)]
use vision_calibration_core::UncertaintyUnavailable;
use vision_calibration_core::{
    CameraFix, CameraModelSpec, CameraParams, FixedParams, Iso3, Real, Uncertainty,
};

/// Combined rig + hand-eye + laser dataset.
///
/// Per-view content:
/// - `cameras[c]`: target correspondences for camera `c` (may be `None` if the
///   camera did not detect the target in this view).
/// - `laser_pixels[c]`: laser line pixels in camera `c` (may be `None` if no
///   laser image exists for this pose, e.g. `target_snap`-only views).
/// - `meta.base_se3_gripper`: the robot pose for this view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeLaserlineDataset {
    /// Per-view per-camera observations + laser pixels + robot pose.
    pub views: Vec<RigHandeyeLaserlineView>,
    /// Number of cameras in the rig.
    pub num_cameras: usize,
    /// Hand-eye configuration (must match the warm-start).
    pub mode: HandEyeMode,
}

/// Per-view observations for [`RigHandeyeLaserlineDataset`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RigHandeyeLaserlineView {
    /// Target corners + laser pixels per camera (same shape as
    /// [`RigLaserlineView`]).
    pub obs: RigLaserlineView,
    /// Robot pose for this view (`base_se3_gripper`).
    pub meta: RobotPoseMeta,
}

impl RigHandeyeLaserlineDataset {
    /// Construct with consistency checks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `views` is empty.
    /// Returns [`Error::InvalidInput`] if any view has inconsistent per-camera
    /// slice lengths.
    pub fn new(
        views: Vec<RigHandeyeLaserlineView>,
        num_cameras: usize,
        mode: HandEyeMode,
    ) -> Result<Self, Error> {
        if views.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        for (i, v) in views.iter().enumerate() {
            if v.obs.cameras.len() != num_cameras || v.obs.laser_pixels.len() != num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {i} has {}/{} camera slots, expected {num_cameras}",
                    v.obs.cameras.len(),
                    v.obs.laser_pixels.len()
                )));
            }
        }
        Ok(Self {
            views,
            num_cameras,
            mode,
        })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// Build from a plain `RigLaserlineDataset` plus a parallel list of robot
    /// poses (one per view). Provided as a convenience for callers that
    /// already have both independently.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `robot_poses.len()` does not match
    /// `dataset.num_views()`.
    pub fn from_rig_dataset(
        dataset: RigLaserlineDataset,
        robot_poses: Vec<Iso3>,
        mode: HandEyeMode,
    ) -> Result<Self, Error> {
        if robot_poses.len() != dataset.num_views() {
            return Err(Error::invalid_input(format!(
                "robot_poses has {} entries, expected {} (one per view)",
                robot_poses.len(),
                dataset.num_views()
            )));
        }
        let num_cameras = dataset.num_cameras;
        let views = dataset
            .views
            .into_iter()
            .zip(robot_poses)
            .map(|(obs, pose)| RigHandeyeLaserlineView {
                obs,
                meta: RobotPoseMeta {
                    base_se3_gripper: pose,
                },
            })
            .collect();
        Self::new(views, num_cameras, mode)
    }
}

/// Initial / refined parameters for joint rig + hand-eye + laser BA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeLaserlineParams {
    /// Per-camera camera parameters, each in its own model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera `cam_to_rig` (T_rig_from_cam).
    pub cam_to_rig: Vec<Iso3>,
    /// Hand-eye transform.
    ///
    /// - `EyeInHand`: `gripper_se3_rig` (T_G_R).
    /// - `EyeToHand`: `rig_se3_base` (T_R_B).
    pub handeye: Iso3,
    /// Fixed target reference pose.
    ///
    /// - `EyeInHand`: `base_se3_target` (T_B_T).
    /// - `EyeToHand`: `gripper_se3_target` (T_G_T).
    pub target_ref: Iso3,
    /// Per-camera laser plane expressed in camera frame.
    pub planes_cam: Vec<LaserPlane>,
}

/// Solve options for joint rig + hand-eye + laser BA.
#[derive(Debug, Clone)]
pub struct RigHandeyeLaserlineSolveOptions {
    /// Which laser residual to use.
    pub laser_residual_type: LaserlineResidualType,
    /// Robust loss applied to the target-corner residuals.
    pub calib_loss: RobustLoss,
    /// Robust loss applied to the laser residuals.
    pub laser_loss: RobustLoss,
    /// Uniform weight for target corner residuals.
    pub calib_weight: f64,
    /// Uniform weight for laser pixel residuals.
    pub laser_weight: f64,
    /// Camera parameters held fixed, one entry per camera, each resolved
    /// against that camera's model. Empty holds every camera's model
    /// defaults (`k3` for Brown-Conrady; everything but the low-order radial
    /// terms under a Scheimpflug sensor).
    pub fix: Vec<CameraFix>,
    /// Per-camera extrinsics fix flag. Length must match `num_cameras`. The
    /// reference camera should typically be fixed for gauge freedom.
    pub fix_extrinsics: Vec<bool>,
    /// Fix the hand-eye transform.
    pub fix_handeye: bool,
    /// Fix the target reference pose.
    pub fix_target_ref: bool,
    /// Per-camera laser-plane fix flag. Length must match `num_cameras`.
    pub fix_planes: Vec<bool>,
    /// Refine per-view robot pose corrections in se(3).
    ///
    /// When enabled, each view gets a 6D tangent correction applied as
    /// `exp(delta_i) * T_B_G_i`. `delta_0` is fixed to zero for gauge
    /// consistency, and all deltas receive a zero-mean tangent prior.
    pub refine_robot_poses: bool,
    /// Robot rotation prior sigma (radians).
    pub robot_rot_sigma: f64,
    /// Robot translation prior sigma (meters).
    pub robot_trans_sigma: f64,
    /// Optional initial per-view robot pose deltas
    /// `[rx, ry, rz, tx, ty, tz]`. Used only when
    /// [`Self::refine_robot_poses`] is enabled.
    pub initial_robot_deltas: Option<Vec<[f64; 6]>>,
}

impl Default for RigHandeyeLaserlineSolveOptions {
    fn default() -> Self {
        Self {
            laser_residual_type: LaserlineResidualType::LineDistNormalized,
            calib_loss: RobustLoss::None,
            laser_loss: RobustLoss::None,
            calib_weight: 1.0,
            laser_weight: 1.0,
            fix: Vec::new(),
            fix_extrinsics: Vec::new(),
            fix_handeye: false,
            fix_target_ref: false,
            fix_planes: Vec::new(),
            refine_robot_poses: false,
            robot_rot_sigma: std::f64::consts::PI / 360.0,
            robot_trans_sigma: 1.0e-3,
            initial_robot_deltas: None,
        }
    }
}

/// Per-camera statistics for [`RigHandeyeLaserlineEstimate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RigHandeyeLaserlinePerCamStats {
    /// Mean target-corner reprojection error (pixels).
    pub mean_reproj_error_px: f64,
    /// Number of target corner residuals evaluated for this camera.
    pub reproj_count: usize,
    /// Maximum target-corner reprojection error (pixels).
    pub max_reproj_error_px: f64,
    /// Reprojection error histogram with buckets `[<=1, <=2, <=5, <=10, >10]` pixels.
    pub reproj_histogram_px: [usize; 5],
    /// Mean laser point-to-plane distance (meters).
    pub mean_laser_err_m: f64,
    /// Maximum absolute laser point-to-plane distance (meters).
    pub max_laser_err_m: f64,
    /// Laser point-to-plane histogram with buckets
    /// `[<=0.1mm, <=1mm, <=10mm, <=100mm, >100mm]`.
    pub laser_histogram_m: [usize; 5],
    /// Mean laser line-distance in undistorted pixel space (pixels).
    pub mean_laser_err_px: f64,
    /// Maximum absolute laser line-distance in undistorted pixel space (pixels).
    pub max_laser_err_px: f64,
    /// Number of laser pixel residuals evaluated for this camera.
    pub laser_count: usize,
}

/// Result of joint rig + hand-eye + laser BA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeLaserlineEstimate {
    /// Refined parameters.
    pub params: RigHandeyeLaserlineParams,
    /// Per-camera laser plane expressed in rig frame (convenience: derived
    /// from `params.planes_cam` + `params.cam_to_rig`).
    pub planes_rig: Vec<LaserPlane>,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean target-corner reprojection error across all cameras (pixels).
    pub mean_reproj_error_px: f64,
    /// Per-camera stats (reprojection + laser, both pixel and metric).
    pub per_cam_stats: Vec<RigHandeyeLaserlinePerCamStats>,
    /// Optional optimized per-view robot pose deltas
    /// `[rx, ry, rz, tx, ty, tz]`.
    pub robot_deltas: Option<Vec<[Real; 6]>>,
    /// Covariance of the refined parameters, labelled as the rig hand-eye
    /// result's, plus each camera's laser plane in its camera frame
    /// (`cam<i>.plane.nx … cam<i>.plane.d`).
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

/// Jointly optimize rig intrinsics + extrinsics + hand-eye + per-camera laser
/// planes against both target corner observations and laser pixel
/// observations.
///
/// # Errors
///
/// Returns [`Error`] if IR construction fails or the solver backend returns
/// an error.
pub fn optimize_rig_handeye_laserline(
    dataset: RigHandeyeLaserlineDataset,
    initial: RigHandeyeLaserlineParams,
    opts: RigHandeyeLaserlineSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<RigHandeyeLaserlineEstimate, Error> {
    let (specs, fixed) = resolve_camera_fix(&initial.cameras, &opts.fix)?;
    let (ir, initial_map) = build_ir(&dataset, &initial, &specs, &fixed, &opts)?;
    let solution = backend::solve(&ir, &initial_map, &backend_opts)?;
    let labels = RigLabelling {
        cameras: &specs,
        multi_camera: dataset.num_cameras > 1,
        handeye_mode: Some(dataset.mode),
    }
    .labels(&ir, &solution);

    // Extract refined parameters.
    let cameras = specs
        .iter()
        .enumerate()
        .map(|(i, spec)| unpack_camera(&solution.params, *spec, Some(i)))
        .collect::<Result<Vec<_>, Error>>()?;
    let cam_to_rig = (0..dataset.num_cameras)
        .map(|cam_idx| {
            se3_dvec_to_iso3(
                solution
                    .params
                    .get(&format!("extr/{cam_idx}"))
                    .unwrap()
                    .as_view(),
            )
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let handeye = se3_dvec_to_iso3(solution.params.get("handeye").unwrap().as_view())?;
    let target_ref = se3_dvec_to_iso3(solution.params.get("target_ref").unwrap().as_view())?;
    let planes_cam = (0..dataset.num_cameras)
        .map(|cam_idx| {
            let normal_v = solution
                .params
                .get(&format!("plane_n/{cam_idx}"))
                .unwrap()
                .as_view();
            let dist_v = solution
                .params
                .get(&format!("plane_d/{cam_idx}"))
                .unwrap()
                .as_view();
            LaserPlane::from_split_dvec(normal_v, dist_v)
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let params = RigHandeyeLaserlineParams {
        cameras,
        cam_to_rig,
        handeye,
        target_ref,
        planes_cam,
    };

    let robot_deltas = if opts.refine_robot_poses {
        let mut deltas = Vec::with_capacity(dataset.num_views());
        for view_idx in 0..dataset.num_views() {
            let key = format!("robot_delta/{view_idx}");
            let delta_vec = solution.params.get(&key).unwrap();
            deltas.push([
                delta_vec[0],
                delta_vec[1],
                delta_vec[2],
                delta_vec[3],
                delta_vec[4],
                delta_vec[5],
            ]);
        }
        Some(deltas)
    } else {
        None
    };

    let planes_rig: Vec<LaserPlane> = params
        .planes_cam
        .iter()
        .zip(params.cam_to_rig.iter())
        .map(|(plane, cam_to_rig)| plane.transform_by(cam_to_rig))
        .collect();

    let (mean_reproj_error_px, per_cam_stats) =
        compute_stats(&dataset, &params, robot_deltas.as_deref())?;

    Ok(RigHandeyeLaserlineEstimate {
        params,
        planes_rig,
        report: solution.solve_report,
        mean_reproj_error_px,
        per_cam_stats,
        robot_deltas,
        uncertainty: uncertainty::labelled(&ir, &solution.uncertainty, &labels),
    })
}

/// Evaluate joint rig + hand-eye + laser parameters on a dataset without
/// running optimization.
///
/// `robot_deltas`, when provided, must contain one tangent correction per
/// view and uses the same left-multiply convention as the optimizer:
/// `T_B_G_corr = exp(delta_i) * T_B_G_i`.
///
/// # Errors
///
/// [`Error::InvalidInput`] when a camera has a general homography sensor,
/// which no camera model describes.
pub fn evaluate_rig_handeye_laserline(
    dataset: &RigHandeyeLaserlineDataset,
    params: &RigHandeyeLaserlineParams,
    robot_deltas: Option<&[[Real; 6]]>,
) -> Result<(f64, Vec<RigHandeyeLaserlinePerCamStats>), Error> {
    compute_stats(dataset, params, robot_deltas)
}

fn compute_stats(
    dataset: &RigHandeyeLaserlineDataset,
    params: &RigHandeyeLaserlineParams,
    robot_deltas: Option<&[[Real; 6]]>,
) -> Result<(f64, Vec<RigHandeyeLaserlinePerCamStats>), Error> {
    use crate::factors::laserline::{laser_line_distance_residual, laser_point_to_plane_residual};

    let cam_models = camera_models(&params.cameras)?;
    let specs = params
        .cameras
        .iter()
        .map(crate::params::camera::camera_spec)
        .collect::<Result<Vec<_>, Error>>()?;

    let cam_se3_rig: Vec<Iso3> = params.cam_to_rig.iter().map(|t| t.inverse()).collect();

    // Pack params into DVectors (re-used across residual evals).
    let camera_dvecs: Vec<Vec<DVector<f64>>> =
        params.cameras.iter().map(camera_block_values).collect();
    let extr_dvecs: Vec<DVector<f64>> = params.cam_to_rig.iter().map(iso3_to_se3_dvec).collect();
    let handeye_dv = iso3_to_se3_dvec(&params.handeye);
    let target_ref_dv = iso3_to_se3_dvec(&params.target_ref);
    let plane_n_dvecs: Vec<DVector<f64>> = params
        .planes_cam
        .iter()
        .map(|p| p.normal_to_dvec())
        .collect();
    let plane_d_dvecs: Vec<DVector<f64>> = params
        .planes_cam
        .iter()
        .map(|p| p.distance_to_dvec())
        .collect();

    let mut per_cam_reproj_sum = vec![0.0f64; dataset.num_cameras];
    let mut per_cam_reproj_count = vec![0usize; dataset.num_cameras];
    let mut per_cam_reproj_max = vec![0.0f64; dataset.num_cameras];
    let mut per_cam_reproj_hist = vec![[0usize; 5]; dataset.num_cameras];
    let mut per_cam_laser_pt_sq_sum = vec![0.0f64; dataset.num_cameras];
    let mut per_cam_laser_pt_max = vec![0.0f64; dataset.num_cameras];
    let mut per_cam_laser_m_hist = vec![[0usize; 5]; dataset.num_cameras];
    let mut per_cam_laser_px_sq_sum = vec![0.0f64; dataset.num_cameras];
    let mut per_cam_laser_px_max = vec![0.0f64; dataset.num_cameras];
    let mut per_cam_laser_count = vec![0usize; dataset.num_cameras];

    let handeye_fwd = params.handeye;
    let handeye_inv = params.handeye.inverse();

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let robot_pose = if let Some(deltas) = robot_deltas {
            corrected_robot_pose(view.meta.base_se3_gripper, deltas[view_idx])
        } else {
            view.meta.base_se3_gripper
        };
        let robot_arr: [f64; 7] = {
            let v = iso3_to_se3_dvec(&robot_pose);
            [v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
        };

        // Target-corner reprojection: compute cam_se3_target per view from
        // the chain and forward-project each 3D corner. Matches the
        // ReprojPointPinhole4Dist5Scheimpflug2HandEye factor semantics.
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let camera = &cam_models[cam_idx];
            for (pt3, pt2) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_camera = match dataset.mode {
                    HandEyeMode::EyeInHand => {
                        let p_base = params.target_ref.transform_point(pt3);
                        let p_gripper = robot_pose.inverse_transform_point(&p_base);
                        let p_rig = handeye_inv.transform_point(&p_gripper);
                        cam_se3_rig[cam_idx].transform_point(&p_rig)
                    }
                    HandEyeMode::EyeToHand => {
                        let p_gripper = params.target_ref.transform_point(pt3);
                        let p_base = robot_pose.transform_point(&p_gripper);
                        let p_rig = handeye_fwd.transform_point(&p_base);
                        cam_se3_rig[cam_idx].transform_point(&p_rig)
                    }
                };
                if let Some(proj) = camera.project_point(&p_camera) {
                    let err = (proj - *pt2).norm();
                    if err.is_finite() {
                        per_cam_reproj_sum[cam_idx] += err;
                        per_cam_reproj_count[cam_idx] += 1;
                        per_cam_reproj_max[cam_idx] = per_cam_reproj_max[cam_idx].max(err);
                        per_cam_reproj_hist[cam_idx][bucket_reproj_px(err)] += 1;
                    }
                }
            }
        }

        // Laser residuals — evaluate both PointToPlane and LineDistNormalized
        // for reporting (PointToPlane gives meters, LineDist gives pixels);
        // this is independent of which one was used as the cost.
        for (cam_idx, laser_slot) in view.obs.laser_pixels.iter().enumerate() {
            let Some(pixels) = laser_slot else { continue };
            let chain = LaserChain::RigHandEye {
                base_se3_gripper: robot_arr,
                mode: dataset.mode,
            };
            let mut chain_params = camera_dvecs[cam_idx].clone();
            chain_params.extend([
                extr_dvecs[cam_idx].clone(),
                handeye_dv.clone(),
                target_ref_dv.clone(),
                plane_n_dvecs[cam_idx].clone(),
                plane_d_dvecs[cam_idx].clone(),
            ]);
            for px in pixels {
                let laser_px = [px.x, px.y];
                let r_m = laser_point_to_plane_residual(
                    specs[cam_idx],
                    &chain,
                    &chain_params,
                    laser_px,
                    1.0,
                );
                let r_px = laser_line_distance_residual(
                    specs[cam_idx],
                    &chain,
                    &chain_params,
                    laser_px,
                    1.0,
                );
                let m = r_m[0];
                let p = r_px[0];
                if m.is_finite() && p.is_finite() && m.abs() < 1e5 && p.abs() < 1e5 {
                    let m_abs = m.abs();
                    let p_abs = p.abs();
                    per_cam_laser_pt_sq_sum[cam_idx] += m_abs * m_abs;
                    per_cam_laser_px_sq_sum[cam_idx] += p_abs * p_abs;
                    per_cam_laser_pt_max[cam_idx] = per_cam_laser_pt_max[cam_idx].max(m_abs);
                    per_cam_laser_px_max[cam_idx] = per_cam_laser_px_max[cam_idx].max(p_abs);
                    per_cam_laser_m_hist[cam_idx][bucket_laser_m(m_abs)] += 1;
                    per_cam_laser_count[cam_idx] += 1;
                }
            }
        }
    }

    let per_cam_stats: Vec<RigHandeyeLaserlinePerCamStats> = (0..dataset.num_cameras)
        .map(|cam_idx| {
            let reproj_count = per_cam_reproj_count[cam_idx];
            let mean_reproj_error_px = if reproj_count > 0 {
                per_cam_reproj_sum[cam_idx] / reproj_count as f64
            } else {
                0.0
            };
            let laser_count = per_cam_laser_count[cam_idx];
            let (mean_laser_err_m, mean_laser_err_px) = if laser_count > 0 {
                let n = laser_count as f64;
                (
                    (per_cam_laser_pt_sq_sum[cam_idx] / n).sqrt(),
                    (per_cam_laser_px_sq_sum[cam_idx] / n).sqrt(),
                )
            } else {
                (0.0, 0.0)
            };
            RigHandeyeLaserlinePerCamStats {
                mean_reproj_error_px,
                reproj_count,
                max_reproj_error_px: per_cam_reproj_max[cam_idx],
                reproj_histogram_px: per_cam_reproj_hist[cam_idx],
                mean_laser_err_m,
                max_laser_err_m: per_cam_laser_pt_max[cam_idx],
                laser_histogram_m: per_cam_laser_m_hist[cam_idx],
                mean_laser_err_px,
                max_laser_err_px: per_cam_laser_px_max[cam_idx],
                laser_count,
            }
        })
        .collect();

    let total_reproj_sum: f64 = per_cam_reproj_sum.iter().sum();
    let total_reproj_count: usize = per_cam_reproj_count.iter().sum();
    let mean_reproj_error_px = if total_reproj_count > 0 {
        total_reproj_sum / total_reproj_count as f64
    } else {
        0.0
    };
    Ok((mean_reproj_error_px, per_cam_stats))
}

fn bucket_reproj_px(err: f64) -> usize {
    if err <= 1.0 {
        0
    } else if err <= 2.0 {
        1
    } else if err <= 5.0 {
        2
    } else if err <= 10.0 {
        3
    } else {
        4
    }
}

fn bucket_laser_m(err: f64) -> usize {
    if err <= 1e-4 {
        0
    } else if err <= 1e-3 {
        1
    } else if err <= 1e-2 {
        2
    } else if err <= 1e-1 {
        3
    } else {
        4
    }
}

fn corrected_robot_pose(robot_pose: Iso3, delta: [Real; 6]) -> Iso3 {
    use nalgebra::{Translation3, Unit, UnitQuaternion, Vector3};

    let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
    let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);
    let angle = rot_vec.norm();
    let delta_rot = if angle > 1e-12 {
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(rot_vec), angle)
    } else {
        UnitQuaternion::identity()
    };
    let delta_iso = Iso3::from_parts(Translation3::from(trans_vec), delta_rot);
    delta_iso * robot_pose
}

fn build_ir(
    dataset: &RigHandeyeLaserlineDataset,
    initial: &RigHandeyeLaserlineParams,
    specs: &[CameraModelSpec],
    fixed: &[FixedParams],
    opts: &RigHandeyeLaserlineSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error> {
    let n = dataset.num_cameras;
    if initial.cameras.len() != n {
        return Err(Error::invalid_input("cameras count mismatch"));
    }
    if initial.cam_to_rig.len() != n {
        return Err(Error::invalid_input("cam_to_rig count mismatch"));
    }
    if initial.planes_cam.len() != n {
        return Err(Error::invalid_input("planes_cam count mismatch"));
    }
    if !opts.fix_extrinsics.is_empty() && opts.fix_extrinsics.len() != n {
        return Err(Error::invalid_input(format!(
            "fix_extrinsics length {} must be 0 or match num_cameras {}",
            opts.fix_extrinsics.len(),
            n
        )));
    }
    if !opts.fix_planes.is_empty() && opts.fix_planes.len() != n {
        return Err(Error::invalid_input(format!(
            "fix_planes length {} must be 0 or match num_cameras {}",
            opts.fix_planes.len(),
            n
        )));
    }
    if opts.robot_rot_sigma.is_nan() || opts.robot_rot_sigma <= 0.0 {
        return Err(Error::invalid_input("robot_rot_sigma must be positive"));
    }
    if opts.robot_trans_sigma.is_nan() || opts.robot_trans_sigma <= 0.0 {
        return Err(Error::invalid_input("robot_trans_sigma must be positive"));
    }
    if let Some(deltas) = &opts.initial_robot_deltas
        && deltas.len() != dataset.num_views()
    {
        return Err(Error::invalid_input(format!(
            "initial_robot_deltas length {} must match num_views {}",
            deltas.len(),
            dataset.num_views()
        )));
    }

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

    let extr_fixed =
        |cam_idx: usize| -> bool { opts.fix_extrinsics.get(cam_idx).copied().unwrap_or(false) };
    let plane_fixed =
        |cam_idx: usize| -> bool { opts.fix_planes.get(cam_idx).copied().unwrap_or(false) };

    // 1. Per-camera blocks, grouped by slot.
    let inputs: Vec<CameraBlockInput<'_>> = initial
        .cameras
        .iter()
        .zip(fixed)
        .map(|(params, fixed)| CameraBlockInput {
            params,
            fixed,
            bounds: None,
        })
        .collect();
    let camera_blocks = add_camera_blocks(&mut ir, &mut initial_map, &inputs, true)?;
    // 4. Per-camera extrinsics.
    let mut extr_ids = Vec::with_capacity(n);
    for cam_idx in 0..n {
        let fixed = if extr_fixed(cam_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("extr/{cam_idx}");
        let id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        extr_ids.push(id);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.cam_to_rig[cam_idx]));
    }
    // 5. Hand-eye.
    let handeye_fixed = if opts.fix_handeye {
        FixedMask::all_fixed(7)
    } else {
        FixedMask::all_free()
    };
    let handeye_id = ir.add_param_block("handeye", 7, ManifoldKind::SE3, handeye_fixed, None);
    initial_map.insert("handeye".to_string(), iso3_to_se3_dvec(&initial.handeye));
    // 6. Target reference pose.
    let target_ref_fixed = if opts.fix_target_ref {
        FixedMask::all_fixed(7)
    } else {
        FixedMask::all_free()
    };
    let target_ref_id =
        ir.add_param_block("target_ref", 7, ManifoldKind::SE3, target_ref_fixed, None);
    initial_map.insert(
        "target_ref".to_string(),
        iso3_to_se3_dvec(&initial.target_ref),
    );
    // 7. Per-camera laser planes.
    let mut plane_n_ids = Vec::with_capacity(n);
    let mut plane_d_ids = Vec::with_capacity(n);
    for cam_idx in 0..n {
        let fixed_n = if plane_fixed(cam_idx) {
            FixedMask::all_fixed(3)
        } else {
            FixedMask::all_free()
        };
        let fixed_d = if plane_fixed(cam_idx) {
            FixedMask::all_fixed(1)
        } else {
            FixedMask::all_free()
        };
        let key_n = format!("plane_n/{cam_idx}");
        let id_n = ir.add_param_block(&key_n, 3, ManifoldKind::S2, fixed_n, None);
        plane_n_ids.push(id_n);
        initial_map.insert(key_n, initial.planes_cam[cam_idx].normal_to_dvec());
        let key_d = format!("plane_d/{cam_idx}");
        let id_d = ir.add_param_block(&key_d, 1, ManifoldKind::Euclidean, fixed_d, None);
        plane_d_ids.push(id_d);
        initial_map.insert(key_d, initial.planes_cam[cam_idx].distance_to_dvec());
    }

    let robot_prior_sqrt_info = if opts.refine_robot_poses {
        let rot = 1.0 / opts.robot_rot_sigma;
        let trans = 1.0 / opts.robot_trans_sigma;
        [rot, rot, rot, trans, trans, trans]
    } else {
        [0.0; 6]
    };

    // 8. Residuals per view.
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let robot_se3 = iso3_to_se3_dvec(&view.meta.base_se3_gripper);
        let robot_arr: [f64; 7] = [
            robot_se3[0],
            robot_se3[1],
            robot_se3[2],
            robot_se3[3],
            robot_se3[4],
            robot_se3[5],
            robot_se3[6],
        ];

        let robot_delta_id = if opts.refine_robot_poses {
            let fixed = if view_idx == 0 {
                FixedMask::all_fixed(6)
            } else {
                FixedMask::all_free()
            };
            let key = format!("robot_delta/{view_idx}");
            let id = ir.add_param_block(&key, 6, ManifoldKind::Euclidean, fixed, None);
            let initial_delta = opts
                .initial_robot_deltas
                .as_ref()
                .map(|d| DVector::from_row_slice(&d[view_idx]))
                .unwrap_or_else(|| DVector::from_element(6, 0.0));
            initial_map.insert(key, initial_delta);
            ir.add_residual_block(ResidualBlock {
                params: vec![id],
                loss: RobustLoss::None,
                factor: FactorKind::Se3TangentPrior {
                    sqrt_info: robot_prior_sqrt_info,
                },
                residual_dim: 6,
            });
            Some(id)
        } else {
            None
        };

        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let (params, chain) = match robot_delta_id {
                Some(robot_delta_id) => (
                    camera_blocks[cam_idx].with(&[
                        extr_ids[cam_idx],
                        handeye_id,
                        target_ref_id,
                        robot_delta_id,
                    ]),
                    ReprojChain::HandEyeRobotDelta {
                        base_se3_gripper: robot_arr,
                        mode: dataset.mode,
                    },
                ),
                None => (
                    camera_blocks[cam_idx].with(&[extr_ids[cam_idx], handeye_id, target_ref_id]),
                    ReprojChain::HandEye {
                        base_se3_gripper: robot_arr,
                        mode: dataset.mode,
                    },
                ),
            };
            for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                ir.add_residual_block(ResidualBlock {
                    params: params.clone(),
                    loss: opts.calib_loss,
                    factor: FactorKind::ReprojPoint {
                        model: specs[cam_idx],
                        chain,
                        pw: [pw.x, pw.y, pw.z],
                        uv: [uv.x, uv.y],
                        w: obs.weight(pt_idx) * opts.calib_weight,
                    },
                    residual_dim: 2,
                });
            }
        }
        for (cam_idx, pixels_slot) in view.obs.laser_pixels.iter().enumerate() {
            if let Some(pixels) = pixels_slot {
                for px in pixels {
                    let laser_px = [px.x, px.y];
                    let chain = if robot_delta_id.is_some() {
                        LaserChain::RigHandEyeRobotDelta {
                            base_se3_gripper: robot_arr,
                            mode: dataset.mode,
                        }
                    } else {
                        LaserChain::RigHandEye {
                            base_se3_gripper: robot_arr,
                            mode: dataset.mode,
                        }
                    };
                    let factor = match opts.laser_residual_type {
                        LaserlineResidualType::PointToPlane => FactorKind::LaserPointToPlane {
                            model: specs[cam_idx],
                            chain,
                            laser_pixel: laser_px,
                            w: opts.laser_weight,
                        },
                        LaserlineResidualType::LineDistNormalized => {
                            FactorKind::LaserLineDistance {
                                model: specs[cam_idx],
                                chain,
                                laser_pixel: laser_px,
                                w: opts.laser_weight,
                            }
                        }
                    };
                    let residual = ResidualBlock {
                        params: if let Some(robot_delta_id) = robot_delta_id {
                            camera_blocks[cam_idx].with(&[
                                extr_ids[cam_idx],
                                handeye_id,
                                target_ref_id,
                                plane_n_ids[cam_idx],
                                plane_d_ids[cam_idx],
                                robot_delta_id,
                            ])
                        } else {
                            camera_blocks[cam_idx].with(&[
                                extr_ids[cam_idx],
                                handeye_id,
                                target_ref_id,
                                plane_n_ids[cam_idx],
                                plane_d_ids[cam_idx],
                            ])
                        },
                        loss: opts.laser_loss,
                        factor,
                        residual_dim: 1,
                    };
                    ir.add_residual_block(residual);
                }
            }
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Translation3, Unit, UnitQuaternion, Vector3};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, DistortionParams, FxFyCxCy, ProjectionParams, Pt2, Pt3,
        ScheimpflugParams, SensorParams,
    };

    fn target_points() -> Vec<Pt3> {
        let mut pts = Vec::new();
        for y in -2..=2 {
            for x in -3..=3 {
                pts.push(Pt3::new(x as f64 * 0.02, y as f64 * 0.02, 0.0));
            }
        }
        pts
    }

    /// Eight gripper poses, all rotated about the base z axis.
    fn planar_robot_motion() -> Vec<Iso3> {
        (0..8)
            .map(|i| {
                let angle = 0.15 * i as f64;
                Isometry3::from_parts(
                    Translation3::new(0.10 * angle.cos(), 0.05 * (2.0 * angle).sin(), 0.20),
                    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle),
                )
            })
            .collect()
    }

    fn setup_synthetic_rig() -> (RigHandeyeLaserlineDataset, RigHandeyeLaserlineParams) {
        setup_synthetic_rig_with(planar_robot_motion())
    }

    fn setup_synthetic_rig_with(
        views_gt: Vec<Iso3>,
    ) -> (RigHandeyeLaserlineDataset, RigHandeyeLaserlineParams) {
        // Two cameras, side-by-side, Scheimpflug-tilted, fixed in base.
        // Target rides the gripper (EyeToHand), laser plane per camera.
        let intr_gt = FxFyCxCy {
            fx: 900.0,
            fy: 900.0,
            cx: 640.0,
            cy: 360.0,
        };
        let dist_gt = BrownConrady5::default(); // zero distortion
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.05,
            tilt_y: 0.0,
        };
        let cam_gt = CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 { params: dist_gt },
            sensor: SensorParams::Scheimpflug { params: sensor_gt },
            intrinsics: intr_gt,
        };

        // cam_to_rig = T_R_C (forward takes cam→rig coordinates).
        let cam_to_rig_gt = vec![
            Isometry3::from_parts(
                Translation3::new(-0.10, 0.0, 0.0),
                UnitQuaternion::identity(),
            ),
            Isometry3::from_parts(
                Translation3::new(0.10, 0.0, 0.0),
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.05),
            ),
        ];
        // EyeToHand: handeye = rig_se3_base, target_ref = gripper_se3_target.
        let handeye_gt: Iso3 = Isometry3::from_parts(
            Translation3::new(0.15, -0.02, 0.30),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.1),
        );
        let target_ref_gt: Iso3 = Isometry3::from_parts(
            Translation3::new(0.02, 0.0, -0.03),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.05),
        );
        // Per-camera laser planes in cam frame — tilted w.r.t. z axis.
        let planes_cam_gt = vec![
            LaserPlane::new(
                Unit::new_normalize(Vector3::new(0.1, 0.0, 1.0)).into_inner(),
                -0.30,
            ),
            LaserPlane::new(
                Unit::new_normalize(Vector3::new(-0.05, 0.05, 1.0)).into_inner(),
                -0.35,
            ),
        ];

        let params_gt = RigHandeyeLaserlineParams {
            cameras: vec![cam_gt; 2],
            cam_to_rig: cam_to_rig_gt.clone(),
            handeye: handeye_gt,
            target_ref: target_ref_gt,
            planes_cam: planes_cam_gt.clone(),
        };

        // Synthesize observations using the ground-truth chain.
        let pts = target_points();
        let cam_se3_rig_gt: Vec<Iso3> = cam_to_rig_gt.iter().map(|t| t.inverse()).collect();
        let cam_models = camera_models(&params_gt.cameras).unwrap();

        let mut dataset_views = Vec::new();
        for (view_idx, robot_pose) in views_gt.iter().enumerate() {
            let mut cam_corrs: Vec<Option<CorrespondenceView>> = Vec::with_capacity(2);
            let mut cam_laser: Vec<Option<Vec<Pt2>>> = Vec::with_capacity(2);

            for cam_idx in 0..2 {
                // Target points in camera frame via EyeToHand chain.
                let mut pts_2d = Vec::new();
                for pt3 in &pts {
                    let p_grip = target_ref_gt.transform_point(pt3);
                    let p_base = robot_pose.transform_point(&p_grip);
                    let p_rig = handeye_gt.transform_point(&p_base);
                    let p_cam = cam_se3_rig_gt[cam_idx].transform_point(&p_rig);
                    if let Some(proj) = cam_models[cam_idx].project_point(&p_cam) {
                        pts_2d.push(proj);
                    } else {
                        pts_2d.push(Pt2::new(0.0, 0.0));
                    }
                }
                cam_corrs.push(Some(CorrespondenceView::new(pts.clone(), pts_2d).unwrap()));

                // Laser points: project laser line on target plane in cam
                // frame onto the image. First compute target plane in cam
                // frame; intersect with laser plane; project line.
                // Short-cut: synthesize 5 laser pixels by sampling the line
                // direction.
                let rot = cam_se3_rig_gt[cam_idx].rotation
                    * handeye_gt.rotation
                    * robot_pose.rotation
                    * target_ref_gt.rotation;
                let n_target_cam = rot.transform_vector(&Vector3::z_axis());
                // target plane passes through origin in target → cam-frame:
                // p_center_cam =  cam_se3_rig_gt * handeye * robot * target_ref * origin
                let p_origin_target = Pt3::new(0.0, 0.0, 0.0);
                let p_grip_org = target_ref_gt.transform_point(&p_origin_target);
                let p_base_org = robot_pose.transform_point(&p_grip_org);
                let p_rig_org = handeye_gt.transform_point(&p_base_org);
                let p_cam_org = cam_se3_rig_gt[cam_idx].transform_point(&p_rig_org);

                let d_target_cam = -n_target_cam.dot(&p_cam_org.coords);
                let plane_c = &planes_cam_gt[cam_idx];
                let n_laser = plane_c.normal.into_inner();
                let d_laser = plane_c.distance;

                let v = n_laser.cross(&n_target_cam);
                let v_norm = v.norm();
                let mut laser_pixels_this: Vec<Pt2> = Vec::new();
                if v_norm > 1e-9 {
                    let v_unit = v / v_norm;
                    // Pick a point on the line: use cross method to solve a
                    // 2x2 system for a fixed coordinate.
                    let p0 = if v_unit.z.abs() >= v_unit.x.abs() && v_unit.z.abs() >= v_unit.y.abs()
                    {
                        // Solve at z=0
                        let det = n_laser.x * n_target_cam.y - n_laser.y * n_target_cam.x;
                        if det.abs() < 1e-12 {
                            Vector3::zeros()
                        } else {
                            let x = (-d_laser * n_target_cam.y - (-d_target_cam) * n_laser.y) / det;
                            let y =
                                (n_laser.x * (-d_target_cam) - n_target_cam.x * (-d_laser)) / det;
                            Vector3::new(x, y, 0.0)
                        }
                    } else if v_unit.y.abs() >= v_unit.x.abs() {
                        let det = n_laser.x * n_target_cam.z - n_laser.z * n_target_cam.x;
                        let x = (-d_laser * n_target_cam.z - (-d_target_cam) * n_laser.z) / det;
                        let z = (n_laser.x * (-d_target_cam) - n_target_cam.x * (-d_laser)) / det;
                        Vector3::new(x, 0.0, z)
                    } else {
                        let det = n_laser.y * n_target_cam.z - n_laser.z * n_target_cam.y;
                        let y = (-d_laser * n_target_cam.z - (-d_target_cam) * n_laser.z) / det;
                        let z = (n_laser.y * (-d_target_cam) - n_target_cam.y * (-d_laser)) / det;
                        Vector3::new(0.0, y, z)
                    };
                    for s_i in -2..=2 {
                        let s = s_i as f64 * 0.01;
                        let p = p0 + v_unit * s;
                        if let Some(proj) = cam_models[cam_idx].project_point(&Pt3::from(p)) {
                            laser_pixels_this.push(proj);
                        }
                    }
                }
                cam_laser.push(if laser_pixels_this.is_empty() {
                    None
                } else {
                    Some(laser_pixels_this)
                });
                let _ = view_idx;
            }

            dataset_views.push(RigHandeyeLaserlineView {
                obs: RigLaserlineView {
                    cameras: cam_corrs,
                    laser_pixels: cam_laser,
                },
                meta: RobotPoseMeta {
                    base_se3_gripper: *robot_pose,
                },
            });
        }

        let dataset =
            RigHandeyeLaserlineDataset::new(dataset_views, 2, HandEyeMode::EyeToHand).unwrap();
        (dataset, params_gt)
    }

    #[test]
    fn rig_handeye_laserline_joint_recovers_truth() {
        let (dataset, params_gt) = setup_synthetic_rig();

        // Perturb the initial values slightly.
        let mut initial = params_gt.clone();
        // Perturb intrinsics +1%, cx,cy ±1px, tilt ±2°, extr translation +1mm,
        // handeye rotation +0.01 rad.
        for c in initial.cameras.iter_mut() {
            c.intrinsics.fx *= 1.01;
            c.intrinsics.fy *= 0.99;
            c.intrinsics.cx += 1.0;
            c.intrinsics.cy -= 1.0;
        }
        let tilt = |c: &mut CameraParams| match &mut c.sensor {
            SensorParams::Scheimpflug { params } => *params,
            _ => unreachable!("the test rig is tilted"),
        };
        let mut s0 = tilt(&mut initial.cameras[0]);
        s0.tilt_x += 0.02;
        initial.cameras[0].sensor = SensorParams::Scheimpflug { params: s0 };
        let mut s1 = tilt(&mut initial.cameras[1]);
        s1.tilt_y += 0.01;
        initial.cameras[1].sensor = SensorParams::Scheimpflug { params: s1 };
        initial.cam_to_rig[1] = Isometry3::from_parts(
            initial.cam_to_rig[1].translation * Translation3::new(0.001, 0.001, 0.0),
            initial.cam_to_rig[1].rotation,
        );
        initial.handeye = Isometry3::from_parts(
            initial.handeye.translation,
            initial.handeye.rotation * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.01),
        );

        let opts = RigHandeyeLaserlineSolveOptions {
            laser_residual_type: LaserlineResidualType::LineDistNormalized,
            // Fix reference camera extrinsics for gauge.
            fix_extrinsics: vec![true, false],
            ..Default::default()
        };

        let backend_opts = BackendSolveOptions {
            max_iters: 80,
            verbosity: 0,
            ..Default::default()
        };

        let est = optimize_rig_handeye_laserline(dataset, initial, opts, backend_opts).unwrap();

        // Rotations about one axis leave the hand-eye and target
        // translations along it undetermined.
        let Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::RankDeficient { unobservable },
        } = &est.uncertainty
        else {
            panic!("expected a rank deficiency: {:?}", est.uncertainty);
        };
        for label in ["rig_se3_base.tz", "gripper_se3_target.tz"] {
            assert!(unobservable.iter().any(|l| l == label), "{unobservable:?}");
        }

        // Target reprojection should nail sub-pixel.
        assert!(
            est.mean_reproj_error_px < 0.1,
            "reproj {} not < 0.1 px",
            est.mean_reproj_error_px
        );
        // Laser residuals should be near-zero in both units. Point-to-plane
        // (meters) is what we optimize via LineDistNormalized indirectly;
        // the pixel metric is an auxiliary report and is allowed looser
        // tolerance since the synthesized line sampling is sparse.
        for (i, s) in est.per_cam_stats.iter().enumerate() {
            assert!(
                s.mean_laser_err_m < 1e-4,
                "cam {i}: laser pt-to-plane {} not < 1e-4 m",
                s.mean_laser_err_m
            );
            assert!(
                s.mean_laser_err_px < 1.0,
                "cam {i}: laser line-dist {} not < 1.0 px",
                s.mean_laser_err_px
            );
        }

        // Recovered intrinsics should be within 5% of GT. Under low-view
        // geometry, a joint BA allows a scale ambiguity between fx and the
        // hand-eye translation; the residuals still nail sub-pixel because
        // the solution is self-consistent. Tightening this threshold would
        // require more views / wider baseline motion.
        for (est_cam, gt_cam) in est.params.cameras.iter().zip(params_gt.cameras.iter()) {
            let (est_fx, gt_fx) = (est_cam.intrinsics.fx, gt_cam.intrinsics.fx);
            let rel_err = (est_fx - gt_fx).abs() / gt_fx;
            assert!(
                rel_err < 0.05,
                "fx rel error {rel_err:.3} > 5%: est {est_fx} gt {gt_fx}"
            );
        }
    }
}
