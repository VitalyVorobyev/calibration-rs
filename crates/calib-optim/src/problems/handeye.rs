//! Hand-eye calibration for robot-mounted cameras.
//!
//! Default optimization state (fixed target):
//! - per-camera intrinsics and distortion
//! - per-camera extrinsics (camera-to-rig)
//! - hand-eye transform (gripper-from-rig for eye-in-hand, base-from-rig for eye-to-hand)
//! - fixed target pose in base frame (shared across views)
//! - optional per-view robot pose corrections delta_i in se(3), with zero-mean priors
//!
//! Transform chain (eye-in-hand):
//! `T_C_T_i = (T_B_E_i * X)^-1 * Y` where `X` is hand-eye and `Y` is target in base.
//! With robot refinement: `T_B_E_i_corr = exp(delta_i) * T_B_E_i` (left-multiply).
//!
//! Residuals:
//! - reprojection errors for observed target points using the robot pose chain
//! - optional priors on delta_i (anisotropic rotation/translation sigmas)
//! - when delta_i are enabled, delta_0 is fixed to zero to remove gauge freedom
//!
//! Legacy mode can relax per-view target poses, but that is discouraged for a
//! physically fixed target because it weakens hand-eye observability.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{
    FactorKind, FixedMask, HandEyeMode, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss,
};
use crate::params::distortion::{pack_distortion, unpack_distortion, DISTORTION_DIM};
use crate::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
use crate::params::pose_se3::iso3_to_se3_dvec;
use anyhow::{ensure, Result};
use calib_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Real, Vec2,
};
use nalgebra::DVector;
use std::collections::HashMap;

/// Camera type for hand-eye calibration.
pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

/// Observations from one camera in one robot pose view.
#[derive(Debug, Clone)]
pub struct CameraViewObservations {
    pub points_3d: Vec<Pt3>,
    pub points_2d: Vec<Vec2>,
    pub weights: Option<Vec<f64>>,
}

impl CameraViewObservations {
    /// Create observations from 3D and 2D points.
    pub fn new(points_3d: Vec<Pt3>, points_2d: Vec<Vec2>) -> Result<Self> {
        ensure!(
            points_3d.len() == points_2d.len(),
            "3D/2D point count mismatch: {} vs {}",
            points_3d.len(),
            points_2d.len()
        );
        Ok(Self {
            points_3d,
            points_2d,
            weights: None,
        })
    }

    /// Get weight for point at index.
    pub fn weight(&self, idx: usize) -> f64 {
        self.weights.as_ref().map_or(1.0, |w| w[idx])
    }
}

/// Multi-camera rig observations for one robot pose.
#[derive(Debug, Clone)]
pub struct RigViewObservations {
    /// Observations for each camera (None if camera didn't observe in this view).
    pub cameras: Vec<Option<CameraViewObservations>>,
    /// Known robot pose (base-to-gripper) for this view.
    pub robot_pose: Iso3,
}

/// Complete hand-eye calibration dataset.
#[derive(Debug, Clone)]
pub struct HandEyeDataset {
    pub views: Vec<RigViewObservations>,
    pub num_cameras: usize,
    pub mode: HandEyeMode,
}

impl HandEyeDataset {
    /// Create dataset from views.
    pub fn new(
        views: Vec<RigViewObservations>,
        num_cameras: usize,
        mode: HandEyeMode,
    ) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view");
        for (idx, view) in views.iter().enumerate() {
            ensure!(
                view.cameras.len() == num_cameras,
                "view {} has {} cameras, expected {}",
                idx,
                view.cameras.len(),
                num_cameras
            );
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
}

/// Initial values for hand-eye calibration.
#[derive(Debug, Clone)]
pub struct HandEyeInit {
    /// Per-camera intrinsics (usually same values for homogeneous rig).
    pub intrinsics: Vec<FxFyCxCySkew<Real>>,
    /// Per-camera distortion (usually same values for homogeneous rig).
    pub distortion: Vec<BrownConrady5<Real>>,
    /// Per-camera extrinsics (camera-to-rig transforms).
    pub cam_to_rig: Vec<Iso3>,
    /// Hand-eye transform (gripper-from-rig for eye-in-hand, base-from-rig for eye-to-hand).
    pub handeye: Iso3,
    /// Calibration target poses.
    ///
    /// - Default (fixed target): the first pose is used as the initial `T_B_T`.
    /// - Legacy (`relax_target_poses = true`): one pose per view is required.
    pub target_poses: Vec<Iso3>,
}

/// Solve options for hand-eye calibration.
#[derive(Debug, Clone)]
pub struct HandEyeSolveOptions {
    pub robust_loss: RobustLoss,

    // Default intrinsics flags applied to all cameras
    pub fix_fx: bool,
    pub fix_fy: bool,
    pub fix_cx: bool,
    pub fix_cy: bool,

    // Default distortion flags applied to all cameras
    pub fix_k1: bool,
    pub fix_k2: bool,
    pub fix_k3: bool,
    pub fix_p1: bool,
    pub fix_p2: bool,

    /// Optional per-camera override: fix ALL intrinsics for camera i.
    pub fix_intrinsics: Vec<bool>,
    /// Optional per-camera override: fix ALL distortion for camera i.
    pub fix_distortion: Vec<bool>,
    /// Per-camera extrinsics masking (length must match num_cameras).
    pub fix_extrinsics: Vec<bool>,
    /// Fix hand-eye transform (for testing with known hand-eye).
    pub fix_handeye: bool,
    /// View indices to fix (legacy per-view target mode only).
    pub fix_target_poses: Vec<usize>,
    /// Legacy mode: relax per-view target poses instead of a fixed target.
    pub relax_target_poses: bool,
    /// Refine robot poses with per-view se(3) corrections and strong priors.
    ///
    /// When enabled, delta_0 is fixed to zero for gauge consistency.
    pub refine_robot_poses: bool,
    /// Robot rotation prior sigma (radians).
    pub robot_rot_sigma: Real,
    /// Robot translation prior sigma (meters).
    pub robot_trans_sigma: Real,
}

impl Default for HandEyeSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix_fx: false,
            fix_fy: false,
            fix_cx: false,
            fix_cy: false,
            fix_k1: false,
            fix_k2: false,
            fix_k3: true, // k3 often overfits
            fix_p1: false,
            fix_p2: false,
            fix_intrinsics: Vec::new(),
            fix_distortion: Vec::new(),
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
#[derive(Debug, Clone)]
pub struct HandEyeResult {
    /// Per-camera calibrated parameters.
    pub cameras: Vec<PinholeCamera>,
    /// Per-camera extrinsics (camera-to-rig transforms).
    pub cam_to_rig: Vec<Iso3>,
    /// Hand-eye transform.
    pub handeye: Iso3,
    /// Calibration target poses (fixed-target mode returns the same pose per view).
    pub target_poses: Vec<Iso3>,
    pub final_cost: f64,
}

/// Build IR for hand-eye calibration.
///
/// State vector:
/// - intrinsics/distortion, extrinsics, hand-eye transform
/// - target pose (fixed by default, per-view in legacy mode)
/// - optional per-view robot pose deltas (se(3) tangent)
///
/// Residuals:
/// - per-point reprojection error using robot pose measurements
/// - optional zero-mean priors on robot pose deltas
pub fn build_handeye_ir(
    dataset: &HandEyeDataset,
    initial: &HandEyeInit,
    opts: &HandEyeSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        initial.intrinsics.len() == dataset.num_cameras,
        "intrinsics count {} != num_cameras {}",
        initial.intrinsics.len(),
        dataset.num_cameras
    );
    ensure!(
        initial.distortion.len() == dataset.num_cameras,
        "distortion count {} != num_cameras {}",
        initial.distortion.len(),
        dataset.num_cameras
    );
    ensure!(
        initial.cam_to_rig.len() == dataset.num_cameras,
        "cam_to_rig count {} != num_cameras {}",
        initial.cam_to_rig.len(),
        dataset.num_cameras
    );
    ensure!(
        !initial.target_poses.is_empty(),
        "target_poses must contain at least one pose"
    );
    if opts.relax_target_poses {
        ensure!(
            initial.target_poses.len() == dataset.num_views(),
            "target_poses count {} != num_views {}",
            initial.target_poses.len(),
            dataset.num_views()
        );
    }
    ensure!(
        opts.relax_target_poses || opts.fix_target_poses.is_empty(),
        "fix_target_poses is only supported when relax_target_poses is true"
    );
    if opts.refine_robot_poses {
        ensure!(
            opts.robot_rot_sigma > 0.0,
            "robot_rot_sigma must be positive"
        );
        ensure!(
            opts.robot_trans_sigma > 0.0,
            "robot_trans_sigma must be positive"
        );
    }

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

    // 1. Per-camera intrinsics blocks
    let mut cam_ids = Vec::new();
    for cam_idx in 0..dataset.num_cameras {
        let mut cam_fixed = Vec::new();
        if opts.fix_fx {
            cam_fixed.push(0);
        }
        if opts.fix_fy {
            cam_fixed.push(1);
        }
        if opts.fix_cx {
            cam_fixed.push(2);
        }
        if opts.fix_cy {
            cam_fixed.push(3);
        }

        // Check per-camera override
        let fixed_mask = if opts.fix_intrinsics.get(cam_idx).copied().unwrap_or(false) {
            FixedMask::all_fixed(INTRINSICS_DIM)
        } else {
            FixedMask::fix_indices(&cam_fixed)
        };

        let key = format!("cam/{}", cam_idx);
        let cam_id = ir.add_param_block(
            &key,
            INTRINSICS_DIM,
            ManifoldKind::Euclidean,
            fixed_mask,
            None,
        );
        cam_ids.push(cam_id);
        initial_map.insert(key, pack_intrinsics(&initial.intrinsics[cam_idx])?);
    }

    // 2. Per-camera distortion blocks
    let mut dist_ids = Vec::new();
    for cam_idx in 0..dataset.num_cameras {
        let mut dist_fixed = Vec::new();
        if opts.fix_k1 {
            dist_fixed.push(0);
        }
        if opts.fix_k2 {
            dist_fixed.push(1);
        }
        if opts.fix_k3 {
            dist_fixed.push(2);
        }
        if opts.fix_p1 {
            dist_fixed.push(3);
        }
        if opts.fix_p2 {
            dist_fixed.push(4);
        }

        let fixed_mask = if opts.fix_distortion.get(cam_idx).copied().unwrap_or(false) {
            FixedMask::all_fixed(DISTORTION_DIM)
        } else {
            FixedMask::fix_indices(&dist_fixed)
        };

        let key = format!("dist/{}", cam_idx);
        let dist_id = ir.add_param_block(
            &key,
            DISTORTION_DIM,
            ManifoldKind::Euclidean,
            fixed_mask,
            None,
        );
        dist_ids.push(dist_id);
        initial_map.insert(key, pack_distortion(&initial.distortion[cam_idx]));
    }

    // 3. Per-camera extrinsics
    let mut extr_ids = Vec::new();
    for cam_idx in 0..dataset.num_cameras {
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

    // 4. Hand-eye transform
    let handeye_fixed = if opts.fix_handeye {
        FixedMask::all_fixed(7)
    } else {
        FixedMask::all_free()
    };
    let handeye_id = ir.add_param_block("handeye", 7, ManifoldKind::SE3, handeye_fixed, None);
    initial_map.insert("handeye".to_string(), iso3_to_se3_dvec(&initial.handeye));

    // 5. Target pose (fixed by default) + residuals
    let target_id = if opts.relax_target_poses {
        None
    } else {
        let target_seed = initial.target_poses[0];
        let id = ir.add_param_block("target", 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial_map.insert("target".to_string(), iso3_to_se3_dvec(&target_seed));
        Some(id)
    };

    let robot_prior_sqrt_info = if opts.refine_robot_poses {
        let rot = 1.0 / opts.robot_rot_sigma;
        let trans = 1.0 / opts.robot_trans_sigma;
        [rot, rot, rot, trans, trans, trans]
    } else {
        [0.0; 6]
    };

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let target_id = if opts.relax_target_poses {
            let fixed = if opts.fix_target_poses.contains(&view_idx) {
                FixedMask::all_fixed(7)
            } else {
                FixedMask::all_free()
            };
            let key = format!("target/{}", view_idx);
            let target_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
            initial_map.insert(key, iso3_to_se3_dvec(&initial.target_poses[view_idx]));
            target_id
        } else {
            target_id.expect("target id should be set for fixed-target mode")
        };

        let robot_delta_id = if opts.refine_robot_poses {
            let fixed = if view_idx == 0 {
                FixedMask::all_fixed(6)
            } else {
                FixedMask::all_free()
            };
            let key = format!("robot_delta/{}", view_idx);
            let id = ir.add_param_block(&key, 6, ManifoldKind::Euclidean, fixed, None);
            initial_map.insert(key, DVector::from_element(6, 0.0));
            let prior = ResidualBlock {
                params: vec![id],
                loss: RobustLoss::None,
                factor: FactorKind::Se3TangentPrior {
                    sqrt_info: robot_prior_sqrt_info,
                },
                residual_dim: 6,
            };
            ir.add_residual_block(prior);
            Some(id)
        } else {
            None
        };

        // Convert robot pose to SE3 array for factor
        let robot_se3 = iso3_to_se3_dvec(&view.robot_pose);
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
        for (cam_idx, cam_obs) in view.cameras.iter().enumerate() {
            if let Some(obs) = cam_obs {
                for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                    let residual = if let Some(robot_delta_id) = robot_delta_id {
                        ResidualBlock {
                            params: vec![
                                cam_ids[cam_idx],
                                dist_ids[cam_idx],
                                extr_ids[cam_idx],
                                handeye_id,
                                target_id,
                                robot_delta_id,
                            ],
                            loss: opts.robust_loss,
                            factor: FactorKind::ReprojPointPinhole4Dist5HandEyeRobotDelta {
                                pw: [pw.x, pw.y, pw.z],
                                uv: [uv.x, uv.y],
                                w: obs.weight(pt_idx),
                                base_to_gripper_se3: robot_se3_array,
                                mode: dataset.mode,
                            },
                            residual_dim: 2,
                        }
                    } else {
                        ResidualBlock {
                            params: vec![
                                cam_ids[cam_idx],
                                dist_ids[cam_idx],
                                extr_ids[cam_idx],
                                handeye_id,
                                target_id,
                            ],
                            loss: opts.robust_loss,
                            factor: FactorKind::ReprojPointPinhole4Dist5HandEye {
                                pw: [pw.x, pw.y, pw.z],
                                uv: [uv.x, uv.y],
                                w: obs.weight(pt_idx),
                                base_to_gripper_se3: robot_se3_array,
                                mode: dataset.mode,
                            },
                            residual_dim: 2,
                        }
                    };
                    ir.add_residual_block(residual);
                }
            }
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

/// Optimize hand-eye calibration using specified backend.
pub fn optimize_handeye(
    dataset: HandEyeDataset,
    initial: HandEyeInit,
    opts: HandEyeSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<HandEyeResult> {
    let (ir, initial_map) = build_handeye_ir(&dataset, &initial, &opts)?;
    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, &backend_opts)?;

    // Extract per-camera calibrated parameters
    let cameras = (0..dataset.num_cameras)
        .map(|cam_idx| {
            let intrinsics = unpack_intrinsics(
                solution
                    .params
                    .get(&format!("cam/{}", cam_idx))
                    .unwrap()
                    .as_view(),
            )?;
            let distortion = unpack_distortion(
                solution
                    .params
                    .get(&format!("dist/{}", cam_idx))
                    .unwrap()
                    .as_view(),
            )?;
            Ok(Camera::new(Pinhole, distortion, IdentitySensor, intrinsics))
        })
        .collect::<Result<Vec<_>>>()?;

    // Extract extrinsics
    let cam_to_rig = (0..dataset.num_cameras)
        .map(|i| {
            let key = format!("extr/{}", i);
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>>>()?;

    // Extract hand-eye transform
    let handeye = crate::params::pose_se3::se3_dvec_to_iso3(
        solution.params.get("handeye").unwrap().as_view(),
    )?;

    // Extract target poses
    let target_poses = if opts.relax_target_poses {
        (0..dataset.num_views())
            .map(|i| {
                let key = format!("target/{}", i);
                crate::params::pose_se3::se3_dvec_to_iso3(
                    solution.params.get(&key).unwrap().as_view(),
                )
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let target_pose = crate::params::pose_se3::se3_dvec_to_iso3(
            solution.params.get("target").unwrap().as_view(),
        )?;
        vec![target_pose; dataset.num_views()]
    };

    Ok(HandEyeResult {
        cameras,
        cam_to_rig,
        handeye,
        target_poses,
        final_cost: solution.final_cost,
    })
}
