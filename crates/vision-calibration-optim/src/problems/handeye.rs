//! Hand-eye calibration for robot-mounted cameras.
//!
//! Default optimization state (fixed target):
//! - per-camera camera parameters, each camera in its own model (any
//!   distortion kind, an identity or Scheimpflug sensor)
//! - per-camera extrinsics (camera-to-rig)
//! - hand-eye transform (mode-dependent; see [`HandeyeParams::handeye`])
//! - fixed target pose (mode-dependent; see [`HandeyeParams::target_poses`])
//! - optional per-view robot pose corrections delta_i in se(3), with zero-mean priors
//!
//! Transform chain (eye-in-hand):
//! `T_C_T_i = (T_B_E_i * X)^-1 * Y` where `X` is hand-eye and `Y` is target in base.
//! With robot refinement: `T_B_E_i_corr = exp(delta_i) * T_B_E_i` (left-multiply).
//!
//! Residuals:
//! - reprojection errors for observed target points using the robot pose chain
//! - optional priors on delta_i (anisotropic rotation/translation sigmas),
//!   weighed against the reprojection residuals' estimated pixel noise
//!
//! Per-view target mode (`relax_target_poses`) relaxes the target pose per view,
//! but that is discouraged for a physically fixed target because it weakens
//! hand-eye observability.

use crate::Error;
use crate::backend::{self, BackendSolveOptions, SolveReport};
use crate::factors::reprojection_model::se3_exp;
use crate::ir::ReprojSqrtInfo;
use crate::ir::{
    FactorKind, FixedMask, HandeyeMode, ManifoldKind, ProblemIR, ReprojChain, ResidualBlock,
    RobustLoss,
};
use crate::params::camera::{CameraBlockInput, add_camera_blocks, unpack_camera};
use crate::params::pose_se3::iso3_to_se3_dvec;
use crate::problems::rig_extrinsics::{
    ReprojectionSums, camera_models, resolve_camera_fix, solved_pose,
};
use crate::problems::robot_poses::{self, RobotPosePriors};
use crate::uncertainty::{self, RigLabelling};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vision_calibration_core::{
    CameraFix, CameraModelSpec, CameraParams, FixedParams, Iso3, Real, RigDataset, RigView,
    Uncertainty,
};

/// Per-view robot metadata required by hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RobotPoseMeta {
    /// Gripper pose expressed in the robot base frame (T_B_G).
    #[serde(alias = "robot_pose")]
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "vision_calibration_core::Iso3Schema")
    )]
    pub base_se3_gripper: Iso3,
}

/// Dataset wrapper for hand-eye optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandeyeDataset {
    /// Multi-camera observations with robot metadata.
    pub data: RigDataset<RobotPoseMeta>,
    /// Hand-eye mode controlling transform chain.
    pub mode: HandeyeMode,
}

impl HandeyeDataset {
    /// Create dataset from views.
    pub fn new(
        views: Vec<RigView<RobotPoseMeta>>,
        num_cameras: usize,
        mode: HandeyeMode,
    ) -> Result<Self, Error> {
        if views.is_empty() {
            return Err(Error::invalid_input("need at least one view"));
        }
        for (idx, view) in views.iter().enumerate() {
            if view.obs.cameras.len() != num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {} has {} cameras, expected {}",
                    idx,
                    view.obs.cameras.len(),
                    num_cameras
                )));
            }
        }
        Ok(Self {
            data: RigDataset { views, num_cameras },
            mode,
        })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.data.views.len()
    }
}

/// Initial values for hand-eye calibration.
///
/// - `cam_to_rig`: camera pose in the rig frame (`T_R_C`)
/// - `handeye`:
///   - `EyeInHand`: gripper pose in the rig frame (`T_G_R`)
///   - `EyeToHand`: rig pose in the base frame (`T_R_B`)
/// - `target_poses`: fixed target pose(s) depending on the selected mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandeyeParams {
    /// Per-camera camera parameters, each in its own model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera extrinsics (camera-to-rig transforms).
    pub cam_to_rig: Vec<Iso3>,
    /// Hand-eye transform (mode-dependent).
    ///
    /// - `EyeInHand`: `handeye = gripper_from_rig` (`T_G_R`)
    /// - `EyeToHand`: `handeye = rig_from_base` (`T_R_B`)
    pub handeye: Iso3,
    /// Calibration target poses.
    ///
    /// - Default (fixed target):
    ///   - `EyeInHand`: the first pose is used as the initial `base_from_target` (`T_B_T`)
    ///   - `EyeToHand`: the first pose is used as the initial `gripper_from_target` (`T_G_T`)
    /// - Per-view target mode (`relax_target_poses = true`): one pose per view is required.
    pub target_poses: Vec<Iso3>,
}

/// Solve options for hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandeyeSolveOptions {
    /// Robust loss applied to reprojection residuals.
    pub robust_loss: RobustLoss,

    /// Camera parameters held fixed, one entry per camera, each resolved
    /// against that camera's model. Empty holds every camera's model
    /// defaults (`k3` for Brown-Conrady; everything but the low-order radial
    /// terms under a Scheimpflug sensor).
    pub fix: Vec<CameraFix>,
    /// Per-camera extrinsics masking (length must match num_cameras).
    pub fix_extrinsics: Vec<bool>,
    /// Fix hand-eye transform (for testing with known hand-eye).
    pub fix_handeye: bool,
    /// View indices to fix (per-view target mode only).
    pub fix_target_poses: Vec<usize>,
    /// Per-view target mode: relax per-view target poses instead of a fixed target.
    pub relax_target_poses: bool,
    /// Refine robot poses with per-view se(3) corrections under zero-mean
    /// priors of [`Self::robot_rot_sigma`] and [`Self::robot_trans_sigma`].
    ///
    /// The priors are weighed against the reprojection residuals at their
    /// estimated pixel noise: the problem is solved once at 1 px, then again
    /// from that solution at the noise its residuals show.
    pub refine_robot_poses: bool,
    /// Standard deviation of the robot's rotation error about each axis
    /// (radians).
    pub robot_rot_sigma: Real,
    /// Standard deviation of the robot's translation error along each axis
    /// (metres).
    pub robot_trans_sigma: Real,
}

impl Default for HandeyeSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix: Vec::new(),
            fix_extrinsics: Vec::new(),
            fix_handeye: false,
            fix_target_poses: Vec::new(),
            relax_target_poses: false,
            refine_robot_poses: false,
            robot_rot_sigma: std::f64::consts::PI / 360.0, // 0.5 deg
            robot_trans_sigma: 1.0e-3,                     // 1 mm
        }
    }
}

/// Result of hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandeyeEstimate {
    /// Refined hand-eye parameters.
    pub params: HandeyeParams,
    /// Covariance of the estimated parameters at the solution.
    #[serde(default)]
    pub uncertainty: Uncertainty,
    /// Backend solve report.
    pub report: SolveReport,
    /// Optional per-view robot pose deltas (se(3) tangent, `[rx, ry, rz, tx, ty, tz]`).
    pub robot_deltas: Option<Vec<[Real; 6]>>,
    /// Mean reprojection error in pixels (computed post-optimization).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Apply a per-view se(3) robot pose correction `[rx, ry, rz, tx, ty, tz]` to
/// a base-to-gripper pose, exactly as the hand-eye optimizer does:
///
/// `T_B_G_corr = exp(delta) * T_B_G` (left-multiply, `exp` the SE(3)
/// exponential).
///
/// A delta of all zeros returns `base_se3_gripper` unchanged.
pub fn apply_robot_delta(base_se3_gripper: &Iso3, delta: &[Real; 6]) -> Iso3 {
    let (rotation, translation) = se3_exp(DVector::from_row_slice(delta).as_view());
    Iso3::from_parts(translation.into(), rotation) * base_se3_gripper
}

/// Derive per-view "observer-to-target" poses from a hand-eye chain.
///
/// The returned slice parallels `base_se3_gripper`: index `i` gives the
/// target pose in the observer frame (camera for single-cam, rig for
/// rig-based problems) for view `i`.
///
/// Mode-dependent semantics:
/// - `EyeInHand`: `handeye = T_G_R` (or `T_G_C` for single-camera) and
///   `mode_target_pose = T_B_T`. Result: `T_R_T_i = handeye^-1 * T_G_B_i * T_B_T`.
/// - `EyeToHand`: `handeye = T_R_B` (or `T_C_B` for single-camera) and
///   `mode_target_pose = T_G_T`. Result: `T_R_T_i = handeye * T_B_G_i * T_G_T`.
///
/// When `robot_deltas` is `Some`, each `T_B_G_i` is first corrected via
/// [`apply_robot_delta`] before being plugged into the chain. This matches
/// the pose corrections the hand-eye optimizer applies internally when
/// `refine_robot_poses = true`. Pass `None` to use the raw input poses
/// unchanged.
///
/// For single-camera workflows the observer frame *is* the camera, so the
/// returned poses are `T_C_T` directly. For multi-camera rigs the caller
/// must compose with `cam_se3_rig` to project per camera.
pub fn handeye_observer_se3_target(
    mode: HandeyeMode,
    handeye: &Iso3,
    mode_target_pose: &Iso3,
    base_se3_gripper: &[Iso3],
    robot_deltas: Option<&[[Real; 6]]>,
) -> Vec<Iso3> {
    base_se3_gripper
        .iter()
        .enumerate()
        .map(|(i, tbg)| {
            let tbg_corr = match robot_deltas {
                Some(deltas) if i < deltas.len() => apply_robot_delta(tbg, &deltas[i]),
                _ => *tbg,
            };
            match mode {
                HandeyeMode::EyeInHand => handeye.inverse() * tbg_corr.inverse() * mode_target_pose,
                HandeyeMode::EyeToHand => handeye * tbg_corr * mode_target_pose,
            }
        })
        .collect()
}

/// Optimize hand-eye calibration using specified backend.
///
/// Each camera keeps its own model, and `opts.fix` is resolved against it.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when a camera has a general homography sensor,
///   `opts.fix` names a parameter a camera's model does not have or has the
///   wrong length, or the parameter counts do not match the dataset.
/// - Any IR construction or solver backend failure.
pub fn optimize_handeye(
    dataset: HandeyeDataset,
    initial: HandeyeParams,
    opts: HandeyeSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<HandeyeEstimate, Error> {
    let (specs, fixed) = resolve_camera_fix(&initial.cameras, &opts.fix)?;
    let build =
        |pixel_sigma| build_handeye_ir(&dataset, &initial, &specs, &fixed, &opts, pixel_sigma);
    let (ir, solution) = if opts.refine_robot_poses {
        robot_poses::solve_weighted(build, &backend_opts)?
    } else {
        let (ir, initial_map) = build(1.0)?;
        let solution = backend::solve(&ir, &initial_map, &backend_opts)?;
        (ir, solution)
    };

    let cameras = specs
        .iter()
        .enumerate()
        .map(|(i, spec)| unpack_camera(&solution.params, *spec, Some(i)))
        .collect::<Result<Vec<_>, Error>>()?;
    let cam_to_rig = (0..dataset.data.num_cameras)
        .map(|i| solved_pose(&solution.params, &format!("extr/{i}")))
        .collect::<Result<Vec<_>, Error>>()?;
    let handeye = solved_pose(&solution.params, "handeye")?;

    let target_poses = if opts.relax_target_poses {
        (0..dataset.num_views())
            .map(|i| solved_pose(&solution.params, &format!("target/{i}")))
            .collect::<Result<Vec<_>, Error>>()?
    } else {
        vec![solved_pose(&solution.params, "target")?; dataset.num_views()]
    };

    let robot_deltas = opts
        .refine_robot_poses
        .then(|| robot_poses::solved_deltas(&solution.params, dataset.num_views()))
        .transpose()?;

    let params = HandeyeParams {
        cameras,
        cam_to_rig,
        handeye,
        target_poses,
    };
    let (mean_reproj_error, per_cam_reproj_errors) =
        compute_handeye_reproj_error(&dataset, &params, robot_deltas.as_deref())?;
    let labels = RigLabelling {
        cameras: &specs,
        multi_camera: dataset.data.num_cameras > 1,
        handeye_mode: Some(dataset.mode),
    }
    .labels(&ir, &solution);

    Ok(HandeyeEstimate {
        params,
        uncertainty: uncertainty::labelled(&ir, &solution.uncertainty, &solution.noise, &labels),
        report: solution.solve_report,
        robot_deltas,
        mean_reproj_error,
        per_cam_reproj_errors,
    })
}

/// Mean reprojection error of a hand-eye result over the rig and per camera.
///
/// Points that do not project to a finite pixel are skipped; a camera without
/// a valid point reports zero. In per-view target mode each view uses its own
/// target pose.
fn compute_handeye_reproj_error(
    dataset: &HandeyeDataset,
    params: &HandeyeParams,
    robot_deltas: Option<&[[Real; 6]]>,
) -> Result<(f64, Vec<f64>), Error> {
    let models = camera_models(&params.cameras)?;
    let cam_se3_rig: Vec<Iso3> = params.cam_to_rig.iter().map(|t| t.inverse()).collect();
    let handeye_inv = params.handeye.inverse();
    let target_pose = params
        .target_poses
        .first()
        .cloned()
        .unwrap_or(Iso3::identity());

    let mut sums = ReprojectionSums::new(dataset.data.num_cameras);
    for (view_idx, view) in dataset.data.views.iter().enumerate() {
        let robot_pose = match robot_deltas {
            Some(deltas) => apply_robot_delta(&view.meta.base_se3_gripper, &deltas[view_idx]),
            None => view.meta.base_se3_gripper,
        };
        let view_target = if params.target_poses.len() == dataset.num_views() {
            params.target_poses[view_idx]
        } else {
            target_pose
        };

        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            for (pt3, pt2) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_camera = match dataset.mode {
                    HandeyeMode::EyeInHand => {
                        // target -> base -> gripper -> rig -> camera
                        let p_base = view_target.transform_point(pt3);
                        let p_gripper = robot_pose.inverse_transform_point(&p_base);
                        let p_rig = handeye_inv.transform_point(&p_gripper);
                        cam_se3_rig[cam_idx].transform_point(&p_rig)
                    }
                    HandeyeMode::EyeToHand => {
                        // target -> gripper -> base -> rig -> camera
                        let p_gripper = view_target.transform_point(pt3);
                        let p_base = robot_pose.transform_point(&p_gripper);
                        let p_rig = params.handeye.transform_point(&p_base);
                        cam_se3_rig[cam_idx].transform_point(&p_rig)
                    }
                };
                let error = models[cam_idx]
                    .project_point(&p_camera)
                    .map(|proj| (proj - *pt2).norm());
                sums.add(cam_idx, error);
            }
        }
    }
    Ok(sums.means())
}

/// Build IR for hand-eye calibration.
///
/// State vector:
/// - the camera blocks, extrinsics, hand-eye transform
/// - target pose (fixed by default, per-view in per-view target mode)
/// - optional per-view robot pose deltas (se(3) tangent)
///
/// Residuals:
/// - per-point reprojection error using robot pose measurements
/// - optional zero-mean priors on robot pose deltas, weighed against pixel
///   noise `pixel_sigma`
fn build_handeye_ir(
    dataset: &HandeyeDataset,
    initial: &HandeyeParams,
    specs: &[CameraModelSpec],
    fixed: &[FixedParams],
    opts: &HandeyeSolveOptions,
    pixel_sigma: f64,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error> {
    if initial.cameras.len() != dataset.data.num_cameras {
        return Err(Error::invalid_input(format!(
            "camera count {} != num_cameras {}",
            initial.cameras.len(),
            dataset.data.num_cameras
        )));
    }
    if initial.cam_to_rig.len() != dataset.data.num_cameras {
        return Err(Error::invalid_input(format!(
            "cam_to_rig count {} != num_cameras {}",
            initial.cam_to_rig.len(),
            dataset.data.num_cameras
        )));
    }
    if initial.target_poses.is_empty() {
        return Err(Error::invalid_input(
            "target_poses must contain at least one pose",
        ));
    }
    if opts.relax_target_poses && initial.target_poses.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "target_poses count {} != num_views {}",
            initial.target_poses.len(),
            dataset.num_views()
        )));
    }
    if !opts.relax_target_poses && !opts.fix_target_poses.is_empty() {
        return Err(Error::invalid_input(
            "fix_target_poses is only supported when relax_target_poses is true",
        ));
    }
    let robot_priors = RobotPosePriors {
        rot_sigma: opts.robot_rot_sigma,
        trans_sigma: opts.robot_trans_sigma,
    };
    if opts.refine_robot_poses {
        robot_priors.validate()?;
    }

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

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

    // 2. Per-camera extrinsics
    let mut extr_ids = Vec::new();
    for cam_idx in 0..dataset.data.num_cameras {
        let fixed = if opts.fix_extrinsics.get(cam_idx).copied().unwrap_or(false) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("extr/{}", cam_idx);
        let id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        extr_ids.push(id);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.cam_to_rig[cam_idx]));
    }

    // 3. Hand-eye transform
    let handeye_fixed = if opts.fix_handeye {
        FixedMask::all_fixed(7)
    } else {
        FixedMask::all_free()
    };
    let handeye_id = ir.add_param_block("handeye", 7, ManifoldKind::SE3, handeye_fixed, None);
    initial_map.insert("handeye".to_string(), iso3_to_se3_dvec(&initial.handeye));

    // 4. Target pose (fixed by default) + residuals
    let target_id = if opts.relax_target_poses {
        None
    } else {
        let target_seed = initial.target_poses[0];
        let id = ir.add_param_block("target", 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial_map.insert("target".to_string(), iso3_to_se3_dvec(&target_seed));
        Some(id)
    };

    for (view_idx, view) in dataset.data.views.iter().enumerate() {
        let target_id = match target_id {
            Some(id) => id,
            None => {
                let fixed = if opts.fix_target_poses.contains(&view_idx) {
                    FixedMask::all_fixed(7)
                } else {
                    FixedMask::all_free()
                };
                let key = format!("target/{}", view_idx);
                let target_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
                initial_map.insert(key, iso3_to_se3_dvec(&initial.target_poses[view_idx]));
                target_id
            }
        };

        let robot_delta_id = opts
            .refine_robot_poses
            .then(|| robot_priors.add_view(&mut ir, &mut initial_map, view_idx, None, pixel_sigma));

        // Convert robot pose to SE3 array for factor
        let robot_se3 = iso3_to_se3_dvec(&view.meta.base_se3_gripper);
        let robot_se3_array: [f64; 7] = [
            robot_se3[0],
            robot_se3[1],
            robot_se3[2],
            robot_se3[3],
            robot_se3[4],
            robot_se3[5],
            robot_se3[6],
        ];

        // Add residuals for each camera observation
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let (chain_params, chain) = match robot_delta_id {
                Some(robot_delta_id) => (
                    camera_blocks[cam_idx].with(&[
                        extr_ids[cam_idx],
                        handeye_id,
                        target_id,
                        robot_delta_id,
                    ]),
                    ReprojChain::HandeyeRobotDelta {
                        base_se3_gripper: robot_se3_array,
                        mode: dataset.mode,
                    },
                ),
                None => (
                    camera_blocks[cam_idx].with(&[extr_ids[cam_idx], handeye_id, target_id]),
                    ReprojChain::Handeye {
                        base_se3_gripper: robot_se3_array,
                        mode: dataset.mode,
                    },
                ),
            };
            for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                ir.add_residual_block(ResidualBlock {
                    params: chain_params.clone(),
                    loss: opts.robust_loss,
                    factor: FactorKind::ReprojPoint {
                        model: specs[cam_idx],
                        chain,
                        pw: [pw.x, pw.y, pw.z],
                        uv: [uv.x, uv.y],
                        sqrt_info: ReprojSqrtInfo::for_point(obs, pt_idx),
                    },
                    residual_dim: 2,
                });
            }
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCy, PinholeCamera, PointCovariance, Pt2, Pt3,
        RigView, RigViewObs, make_pinhole_camera, pinhole_camera_params,
    };

    fn make_test_camera() -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 600.0,
                fy: 590.0,
                cx: 320.0,
                cy: 240.0,
            },
            BrownConrady5::default(),
        )
    }

    fn project_view(
        camera: &PinholeCamera,
        cam_se3_target: &Iso3,
        board_pts: &[Pt3],
    ) -> CorrespondenceView {
        let pixels: Vec<Pt2> = board_pts
            .iter()
            .map(|p| {
                let p_cam = cam_se3_target.transform_point(p);
                camera
                    .project_point_c(&p_cam.coords)
                    .expect("point should be in front of camera")
            })
            .collect();

        CorrespondenceView::new(board_pts.to_vec(), pixels).unwrap()
    }

    #[test]
    fn compute_reproj_error_matches_ground_truth_chain() {
        let camera = make_test_camera();
        let handeye = Iso3::identity(); // gripper and camera frames coincide
        let target_in_base =
            Iso3::from_parts(Translation3::new(0.0, 0.0, 1.0), UnitQuaternion::identity());

        // Simple square target
        let board_pts = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.1, 0.0, 0.0),
            Pt3::new(0.1, 0.1, 0.0),
            Pt3::new(0.0, 0.1, 0.0),
        ];

        let robot_poses = [
            Iso3::identity(),
            Iso3::from_parts(
                Translation3::new(0.05, 0.0, 0.0),
                UnitQuaternion::identity(),
            ),
        ];

        let views: Vec<RigView<RobotPoseMeta>> = robot_poses
            .iter()
            .map(|robot_pose| {
                // Camera pose: T_C_T = (T_B_G * T_G_C)^-1 * T_B_T
                let cam_se3_target = (robot_pose * handeye).inverse() * target_in_base;
                let obs = project_view(&camera, &cam_se3_target, &board_pts);

                RigView {
                    meta: RobotPoseMeta {
                        base_se3_gripper: *robot_pose,
                    },
                    obs: RigViewObs {
                        cameras: vec![Some(obs)],
                    },
                }
            })
            .collect();

        let dataset = HandeyeDataset::new(views, 1, HandeyeMode::EyeInHand).unwrap();
        let params = HandeyeParams {
            cameras: vec![pinhole_camera_params(&camera)],
            cam_to_rig: vec![Iso3::identity()],
            handeye,
            target_poses: vec![target_in_base],
        };

        let (mean, per_cam) = compute_handeye_reproj_error(&dataset, &params, None).unwrap();
        assert!(mean < 1e-9, "mean reproj err too large: {}", mean);
        assert!(
            per_cam[0] < 1e-9,
            "per-cam reproj err too large: {}",
            per_cam[0]
        );
    }

    /// Weighing the robot-pose priors by the pixel noise `σ` (their residual
    /// counted at standard deviation `σ`) is the maximum-likelihood objective
    /// scaled by `σ²`: the same estimate and the same covariance as weighing
    /// every pixel residual by `1/σ` with unit priors.
    #[test]
    fn priors_weighed_by_pixel_noise_match_pixels_weighed_by_its_inverse() {
        use crate::backend::covariance::ParamUncertainty;
        use vision_calibration_core::synthetic::noise::UniformPixelNoise;

        let camera = make_test_camera();
        let gripper_se3_camera = Iso3::from_parts(
            Translation3::new(0.03, -0.02, 0.08),
            UnitQuaternion::from_euler_angles(0.1, -0.05, 0.3),
        );
        let base_se3_target = Iso3::from_parts(
            Translation3::new(0.6, -0.1, 0.02),
            UnitQuaternion::from_euler_angles(0.0, 0.05, 1.2),
        );
        let board: Vec<Pt3> = (0..42)
            .map(|i| {
                Pt3::new(
                    (i % 7) as f64 * 0.02 - 0.06,
                    (i / 7) as f64 * 0.02 - 0.05,
                    0.0,
                )
            })
            .collect();
        let pixel_noise = UniformPixelNoise {
            seed: 11,
            max_abs_px: 0.2,
        };
        let robot_noise = UniformPixelNoise {
            seed: 12,
            max_abs_px: 1.0,
        };
        let sigma = 0.1;
        // `variance`: each point's stated isotropic covariance, px², if any.
        let dataset = |variance: Option<f64>| {
            let views = [
                (0.0, 0.0, 0.0, -0.01, -0.01, 0.45),
                (0.25, 0.0, 0.1, -0.02, 0.0, 0.50),
                (-0.2, 0.1, 0.0, 0.01, -0.02, 0.48),
                (0.0, 0.25, -0.1, -0.01, 0.01, 0.52),
                (0.1, -0.25, 0.0, 0.02, -0.01, 0.47),
                (-0.15, -0.15, 0.2, 0.0, 0.02, 0.55),
            ]
            .iter()
            .enumerate()
            .map(|(v, &(rx, ry, rz, tx, ty, tz))| {
                let camera_se3_target = Iso3::from_parts(
                    Translation3::new(tx, ty, tz),
                    UnitQuaternion::from_euler_angles(rx, ry, rz),
                );
                let base_se3_gripper =
                    base_se3_target * camera_se3_target.inverse() * gripper_se3_camera.inverse();
                let (u, w) = (robot_noise.sample(v, 0), robot_noise.sample(v, 1));
                let t = robot_noise.sample(v, 2);
                let eps = [
                    u.x * 1e-3,
                    u.y * 1e-3,
                    w.x * 1e-3,
                    w.y * 1e-3,
                    t.x * 1e-3,
                    t.y * 1e-3,
                ];
                let mut obs = project_view(&camera, &camera_se3_target, &board);
                for (i, p) in obs.points_2d.iter_mut().enumerate() {
                    *p += pixel_noise.sample(v, i);
                }
                let obs = match variance {
                    None => obs,
                    Some(v) => CorrespondenceView::new_with_covariances(
                        obs.points_3d.clone(),
                        obs.points_2d.clone(),
                        vec![PointCovariance::new(v, 0.0, v).unwrap(); board.len()],
                    )
                    .unwrap(),
                };
                RigView {
                    meta: RobotPoseMeta {
                        base_se3_gripper: apply_robot_delta(&base_se3_gripper, &eps),
                    },
                    obs: RigViewObs {
                        cameras: vec![Some(obs)],
                    },
                }
            })
            .collect();
            HandeyeDataset::new(views, 1, HandeyeMode::EyeInHand).unwrap()
        };
        let initial = HandeyeParams {
            cameras: vec![pinhole_camera_params(&camera)],
            cam_to_rig: vec![Iso3::identity()],
            handeye: gripper_se3_camera,
            target_poses: vec![base_se3_target],
        };
        let opts = HandeyeSolveOptions {
            fix_extrinsics: vec![true],
            refine_robot_poses: true,
            robot_rot_sigma: 1e-3,
            robot_trans_sigma: 1e-3,
            ..Default::default()
        };
        let backend_opts = BackendSolveOptions {
            max_iters: 200,
            min_abs_decrease: Some(0.0),
            min_rel_decrease: Some(1e-15),
            ..Default::default()
        };
        let (specs, fixed) = resolve_camera_fix(&initial.cameras, &opts.fix).unwrap();
        let solve = |data: &HandeyeDataset, pixel_sigma: f64| {
            let (ir, init) =
                build_handeye_ir(data, &initial, &specs, &fixed, &opts, pixel_sigma).unwrap();
            backend::solve(&ir, &init, &backend_opts).unwrap()
        };
        let priors_weighed = solve(&dataset(None), sigma);
        let pixels_weighed = solve(&dataset(Some(sigma * sigma)), 1.0);

        for (key, a) in &priors_weighed.params {
            let b = &pixels_weighed.params[key];
            let diff = (a - b).norm() / a.norm().max(1e-3);
            assert!(diff < 1e-7, "{key}: {a} vs {b}");
        }
        let (ParamUncertainty::Available(a), ParamUncertainty::Available(b)) =
            (&priors_weighed.uncertainty, &pixels_weighed.uncertainty)
        else {
            panic!("covariance unavailable");
        };
        let diff = (&a.matrix - &b.matrix).norm() / a.matrix.norm();
        assert!(diff < 1e-6, "covariance differs by {diff:.2e}");
    }
}
