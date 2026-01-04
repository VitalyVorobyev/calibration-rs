//! Multi-camera rig extrinsics calibration.
//!
//! Optimizes shared intrinsics, shared distortion, per-camera extrinsics (camera-to-rig),
//! and per-view rig poses (rig-to-target).

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::BrownConrady5Params;
use crate::params::intrinsics::Intrinsics4;
use crate::params::pose_se3::iso3_to_se3_dvec;
use anyhow::{ensure, Result};
use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Real, Vec2};
use nalgebra::DVector;
use std::collections::HashMap;

/// Camera type for rig extrinsics optimization.
pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

/// Observations from one camera in one view.
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

/// Multi-camera observations for one rig view.
#[derive(Debug, Clone)]
pub struct RigViewObservations {
    /// Observations for each camera (None if camera didn't observe in this view).
    pub cameras: Vec<Option<CameraViewObservations>>,
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
    pub intrinsics: Intrinsics4,
    pub distortion: BrownConrady5Params,
    pub cam_to_rig: Vec<Iso3>,
    pub rig_to_target: Vec<Iso3>,
}

/// Solve options for rig extrinsics.
#[derive(Debug, Clone)]
pub struct RigExtrinsicsSolveOptions {
    pub robust_loss: RobustLoss,
    pub fix_fx: bool,
    pub fix_fy: bool,
    pub fix_cx: bool,
    pub fix_cy: bool,
    pub fix_k1: bool,
    pub fix_k2: bool,
    pub fix_k3: bool,
    pub fix_p1: bool,
    pub fix_p2: bool,
    /// Per-camera extrinsics masking (length must match num_cameras).
    pub fix_extrinsics: Vec<bool>,
    /// View indices to fix (e.g., first view for gauge freedom).
    pub fix_rig_poses: Vec<usize>,
}

impl Default for RigExtrinsicsSolveOptions {
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
            fix_extrinsics: Vec::new(),
            fix_rig_poses: Vec::new(),
        }
    }
}

/// Result of rig extrinsics optimization.
#[derive(Debug, Clone)]
pub struct RigExtrinsicsResult {
    pub camera: PinholeCamera,
    pub cam_to_rig: Vec<Iso3>,
    pub rig_to_target: Vec<Iso3>,
    pub final_cost: f64,
}

/// Build IR for rig extrinsics optimization.
pub fn build_rig_extrinsics_ir(
    dataset: &RigExtrinsicsDataset,
    initial: &RigExtrinsicsInit,
    opts: &RigExtrinsicsSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        initial.cam_to_rig.len() == dataset.num_cameras,
        "cam_to_rig count {} != num_cameras {}",
        initial.cam_to_rig.len(),
        dataset.num_cameras
    );
    ensure!(
        initial.rig_to_target.len() == dataset.num_views(),
        "rig_to_target count {} != num_views {}",
        initial.rig_to_target.len(),
        dataset.num_views()
    );

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

    // 1. Shared intrinsics
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

    let cam_id = ir.add_param_block(
        "cam",
        4,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&cam_fixed),
        None,
    );
    initial_map.insert("cam".to_string(), initial.intrinsics.to_dvec());

    // 2. Shared distortion
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

    let dist_id = ir.add_param_block(
        "dist",
        5,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&dist_fixed),
        None,
    );
    initial_map.insert("dist".to_string(), initial.distortion.to_dvec());

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
        initial_map.insert(key, iso3_to_se3_dvec(&initial.rig_to_target[view_idx]));

        // Add residuals for each camera observation
        for (cam_idx, cam_obs) in view.cameras.iter().enumerate() {
            if let Some(obs) = cam_obs {
                for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                    let residual = ResidualBlock {
                        params: vec![cam_id, dist_id, extr_ids[cam_idx], rig_pose_id],
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

    // Extract camera
    let intrinsics = Intrinsics4::from_dvec(solution.params.get("cam").unwrap().as_view())?;
    let distortion = BrownConrady5Params::from_dvec(solution.params.get("dist").unwrap().as_view())?;
    let camera = Camera::new(
        Pinhole,
        distortion.to_core(),
        IdentitySensor,
        intrinsics.to_core(),
    );

    // Extract extrinsics
    let cam_to_rig = (0..dataset.num_cameras)
        .map(|i| {
            let key = format!("extr/{}", i);
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>>>()?;

    // Extract rig poses
    let rig_to_target = (0..dataset.num_views())
        .map(|i| {
            let key = format!("rig_pose/{}", i);
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RigExtrinsicsResult {
        camera,
        cam_to_rig,
        rig_to_target,
        final_cost: solution.final_cost,
    })
}
