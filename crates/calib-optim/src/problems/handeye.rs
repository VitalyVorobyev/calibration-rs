//! Hand-eye calibration for robot-mounted cameras.
//!
//! Optimizes per-camera intrinsics, distortion, camera-to-rig extrinsics,
//! hand-eye transform (rig-to-robot or robot-to-rig depending on mode),
//! and calibration target poses.
//!
//! Supports both eye-in-hand (camera on gripper) and eye-to-hand (camera fixed
//! in workspace) configurations.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{
    FactorKind, FixedMask, HandEyeMode, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss,
};
use crate::params::distortion::BrownConrady5Params;
use crate::params::intrinsics::Intrinsics4;
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
    pub intrinsics: Vec<Intrinsics4>,
    /// Per-camera distortion (usually same values for homogeneous rig).
    pub distortion: Vec<BrownConrady5Params>,
    /// Per-camera extrinsics (camera-to-rig transforms).
    pub cam_to_rig: Vec<Iso3>,
    /// Hand-eye transform (rig-to-robot for eye-in-hand, robot-to-rig for eye-to-hand).
    pub handeye: Iso3,
    /// Calibration target poses (one per view).
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
    /// View indices to fix (e.g., first view for gauge freedom).
    pub fix_target_poses: Vec<usize>,
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
    /// Calibration target poses.
    pub target_poses: Vec<Iso3>,
    pub final_cost: f64,
}

/// Build IR for hand-eye calibration.
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
        initial.target_poses.len() == dataset.num_views(),
        "target_poses count {} != num_views {}",
        initial.target_poses.len(),
        dataset.num_views()
    );

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
            FixedMask::all_fixed(4)
        } else {
            FixedMask::fix_indices(&cam_fixed)
        };

        let key = format!("cam/{}", cam_idx);
        let cam_id = ir.add_param_block(&key, 4, ManifoldKind::Euclidean, fixed_mask, None);
        cam_ids.push(cam_id);
        initial_map.insert(key, initial.intrinsics[cam_idx].to_dvec());
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
            FixedMask::all_fixed(5)
        } else {
            FixedMask::fix_indices(&dist_fixed)
        };

        let key = format!("dist/{}", cam_idx);
        let dist_id = ir.add_param_block(&key, 5, ManifoldKind::Euclidean, fixed_mask, None);
        dist_ids.push(dist_id);
        initial_map.insert(key, initial.distortion[cam_idx].to_dvec());
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

    // 5. Per-view target poses + residuals
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let fixed = if opts.fix_target_poses.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("target/{}", view_idx);
        let target_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.target_poses[view_idx]));

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
                    let residual = ResidualBlock {
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
            let intrinsics = Intrinsics4::from_dvec(
                solution
                    .params
                    .get(&format!("cam/{}", cam_idx))
                    .unwrap()
                    .as_view(),
            )?;
            let distortion = BrownConrady5Params::from_dvec(
                solution
                    .params
                    .get(&format!("dist/{}", cam_idx))
                    .unwrap()
                    .as_view(),
            )?;
            Ok(Camera::new(
                Pinhole,
                distortion.to_core(),
                IdentitySensor,
                intrinsics.to_core(),
            ))
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
    let target_poses = (0..dataset.num_views())
        .map(|i| {
            let key = format!("target/{}", i);
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(HandEyeResult {
        cameras,
        cam_to_rig,
        handeye,
        target_poses,
        final_cost: solution.final_cost,
    })
}
