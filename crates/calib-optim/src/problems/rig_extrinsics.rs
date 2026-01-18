//! Multi-camera rig extrinsics calibration.
//!
//! Optimizes per-camera intrinsics and distortion, per-camera extrinsics (`T_R_C`, camera-to-rig),
//! and per-view rig poses (`T_R_T`, target-to-rig).

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::{pack_distortion, unpack_distortion, DISTORTION_DIM};
use crate::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
use crate::params::pose_se3::iso3_to_se3_dvec;
use anyhow::{ensure, Result};
use calib_core::{
    BrownConrady5, Camera, CameraFixMask, CorrespondenceView, FxFyCxCySkew, IdentitySensor, Iso3,
    Pinhole, Real, PinholeCamera,
};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-camera observations for one rig view.
#[derive(Debug, Clone)]
pub struct RigViewObservations {
    /// Observations for each camera (None if camera didn't observe in this view).
    pub cameras: Vec<Option<CorrespondenceView>>,
}

/// Complete rig extrinsics dataset.
#[derive(Debug, Clone)]
pub struct RigExtrinsicsDataset {
    pub views: Vec<RigViewObservations>,
    pub num_cameras: usize,
}

impl RigExtrinsicsDataset {
    /// Create dataset from views.
    pub fn new(views: Vec<RigViewObservations>, num_cameras: usize) -> Result<Self> {
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
        Ok(Self { views, num_cameras })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}

/// Initial values for rig extrinsics optimization.
#[derive(Debug, Clone)]
pub struct RigExtrinsicsInit {
    /// Per-camera intrinsics (usually same values for homogeneous rig).
    pub intrinsics: Vec<FxFyCxCySkew<Real>>,
    /// Per-camera distortion (usually same values for homogeneous rig).
    pub distortion: Vec<BrownConrady5<Real>>,
    pub cam_to_rig: Vec<Iso3>,
    pub rig_from_target: Vec<Iso3>,
}

/// Solve options for rig extrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Default mask for fixing camera parameters (applied to all cameras).
    pub default_fix: CameraFixMask,
    /// Optional per-camera overrides (None = use default_fix).
    pub camera_overrides: Vec<Option<CameraFixMask>>,
    /// Per-camera extrinsics masking (fix camera-to-rig transform).
    pub fix_extrinsics: Vec<bool>,
    /// View indices to fix (e.g., first view for gauge freedom).
    pub fix_rig_poses: Vec<usize>,
}

impl Default for RigExtrinsicsSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            default_fix: CameraFixMask::default(), // k3 fixed by default
            camera_overrides: Vec::new(),
            fix_extrinsics: Vec::new(),
            fix_rig_poses: Vec::new(),
        }
    }
}

/// Result of rig extrinsics optimization.
#[derive(Debug, Clone)]
pub struct RigExtrinsicsResult {
    /// Per-camera calibrated parameters.
    pub cameras: Vec<PinholeCamera>,
    pub cam_to_rig: Vec<Iso3>,
    pub rig_from_target: Vec<Iso3>,
    pub final_cost: f64,
}

/// Build IR for rig extrinsics optimization.
pub fn build_rig_extrinsics_ir(
    dataset: &RigExtrinsicsDataset,
    initial: &RigExtrinsicsInit,
    opts: &RigExtrinsicsSolveOptions,
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
        initial.rig_from_target.len() == dataset.num_views(),
        "rig_from_target count {} != num_views {}",
        initial.rig_from_target.len(),
        dataset.num_views()
    );

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

    // Helper to get camera fix mask (per-camera override or default)
    let get_camera_mask = |cam_idx: usize| -> &CameraFixMask {
        opts.camera_overrides
            .get(cam_idx)
            .and_then(|o| o.as_ref())
            .unwrap_or(&opts.default_fix)
    };

    // 1. Per-camera intrinsics blocks
    let mut cam_ids = Vec::new();
    for cam_idx in 0..dataset.num_cameras {
        let mask = get_camera_mask(cam_idx);
        let fixed_mask = if mask.intrinsics.all_are_fixed() {
            FixedMask::all_fixed(INTRINSICS_DIM)
        } else {
            FixedMask::fix_indices(&mask.intrinsics.to_indices())
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
        let mask = get_camera_mask(cam_idx);
        let fixed_mask = if mask.distortion.all_are_fixed() {
            FixedMask::all_fixed(DISTORTION_DIM)
        } else {
            FixedMask::fix_indices(&mask.distortion.to_indices())
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

    // 4. Per-view rig poses + residuals
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let fixed = if opts.fix_rig_poses.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("rig_pose/{}", view_idx);
        let rig_pose_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.rig_from_target[view_idx]));

        // Add residuals for each camera observation
        for (cam_idx, cam_obs) in view.cameras.iter().enumerate() {
            if let Some(obs) = cam_obs {
                for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                    let residual = ResidualBlock {
                        params: vec![
                            cam_ids[cam_idx],
                            dist_ids[cam_idx],
                            extr_ids[cam_idx],
                            rig_pose_id,
                        ],
                        loss: opts.robust_loss,
                        factor: FactorKind::ReprojPointPinhole4Dist5TwoSE3 {
                            pw: [pw.x, pw.y, pw.z],
                            uv: [uv.x, uv.y],
                            w: obs.weight(pt_idx),
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

/// Optimize rig extrinsics using specified backend.
pub fn optimize_rig_extrinsics(
    dataset: RigExtrinsicsDataset,
    initial: RigExtrinsicsInit,
    opts: RigExtrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<RigExtrinsicsResult> {
    let (ir, initial_map) = build_rig_extrinsics_ir(&dataset, &initial, &opts)?;
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

    // Extract rig poses
    let rig_from_target = (0..dataset.num_views())
        .map(|i| {
            let key = format!("rig_pose/{}", i);
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RigExtrinsicsResult {
        cameras,
        cam_to_rig,
        rig_from_target,
        final_cost: solution.solve_report.final_cost,
    })
}
