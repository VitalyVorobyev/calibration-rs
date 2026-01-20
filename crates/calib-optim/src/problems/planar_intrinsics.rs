//! Planar intrinsics optimization using the backend-agnostic IR.
//!
//! Each observation contributes a residual block with two residuals (u, v),
//! enabling robust loss to operate per point rather than per view.

use crate::backend::{solve_with_backend, BackendKind, BackendSolveOptions, SolveReport};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::{pack_distortion, unpack_distortion, DISTORTION_DIM};
use crate::params::intrinsics::{pack_intrinsics, unpack_intrinsics, INTRINSICS_DIM};
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use anyhow::{anyhow, ensure, Result};
use calib_core::{
    compute_mean_reproj_error, make_pinhole_camera, BrownConrady5, DistortionFixMask, FxFyCxCySkew,
    IntrinsicsFixMask, Iso3, PinholeCamera, Real, View, PlanarDataset
};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Optimization result for planar intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsParams {
    /// Refined camera with intrinsics and distortion.
    pub camera: PinholeCamera,
    /// Refined target-to-camera poses.
    pub camera_se3_target: Vec<Iso3>,
}

impl PlanarIntrinsicsParams {
    pub fn new(camera: PinholeCamera, camera_se3_target: Vec<Iso3>) -> Result<Self> {
        ensure!(!camera_se3_target.is_empty(), "need at least one pose");
        Ok(Self {
            camera,
            camera_se3_target,
        })
    }

    /// Create from individual components.
    pub fn new_from_components(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        poses: Vec<Iso3>,
    ) -> Result<Self> {
        Self::new(make_pinhole_camera(intrinsics, distortion), poses)
    }

    /// Create with zero distortion (pinhole model only).
    pub fn from_intrinsics(
        intrinsics: FxFyCxCySkew<Real>,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self> {
        Self::new(
            make_pinhole_camera(intrinsics, BrownConrady5::default()),
            camera_se3_target,
        )
    }

    /// Get intrinsics.
    pub fn intrinsics(&self) -> FxFyCxCySkew<Real> {
        self.camera.k
    }

    /// Get distortion.
    pub fn distortion(&self) -> BrownConrady5<Real> {
        self.camera.dist
    }

    /// Get poses.
    pub fn poses(&self) -> &[Iso3] {
        &self.camera_se3_target
    }
}

#[derive(Clone, Debug)]
pub struct PlanarIntrinsicsEstimate {
    pub params: PlanarIntrinsicsParams,
    pub report: SolveReport,
    pub mean_reproj_error: f64,
}

/// Solve options specific to planar intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Mask for fixing intrinsics parameters.
    pub fix_intrinsics: IntrinsicsFixMask,
    /// Mask for fixing distortion parameters (k3 fixed by default).
    pub fix_distortion: DistortionFixMask,
    /// Indices of poses to keep fixed.
    pub fix_poses: Vec<usize>,
}

impl Default for PlanarIntrinsicsSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::default(),
            fix_distortion: DistortionFixMask::default(), // k3 fixed by default
            fix_poses: Vec::new(),
        }
    }
}

/// Build the backend-agnostic IR and initial values for planar intrinsics.
///
/// This is the canonical problem builder reused by all backends.
fn build_planar_intrinsics_ir(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    opts: &PlanarIntrinsicsSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        dataset.num_views() == initial.camera_se3_target.len(),
        "pose count ({}) must match number of views ({})",
        initial.camera_se3_target.len(),
        dataset.num_views()
    );
    for &idx in &opts.fix_poses {
        ensure!(
            idx < dataset.num_views(),
            "fixed pose index {} out of range ({} views)",
            idx,
            dataset.num_views()
        );
    }

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

    let cam_id = ir.add_param_block(
        "cam",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_intrinsics.to_indices()),
        None,
    );
    initial_map.insert("cam".to_string(), pack_intrinsics(&initial.camera.k)?);

    // Add distortion parameter block
    let dist_id = ir.add_param_block(
        "dist",
        DISTORTION_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_distortion.to_indices()),
        None,
    );
    initial_map.insert("dist".to_string(), pack_distortion(&initial.camera.dist));

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_key = format!("pose/{}", view_idx);
        let fixed = if opts.fix_poses.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let pose_id = ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(
            pose_key.clone(),
            iso3_to_se3_dvec(&initial.camera_se3_target[view_idx]),
        );

        for (pt_idx, (pw, uv)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            let factor = FactorKind::ReprojPointPinhole4Dist5 {
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: view.obs.weight(pt_idx),
            };
            let residual = ResidualBlock {
                params: vec![cam_id, dist_id, pose_id],
                loss: opts.robust_loss,
                factor,
                residual_dim: 2,
            };
            ir.add_residual_block(residual);
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

/// Optimize planar intrinsics using the default tiny-solver backend.
pub fn optimize_planar_intrinsics(
    dataset: &PlanarDataset,
    initial: PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate> {
    optimize_planar_intrinsics_with_backend(
        dataset,
        initial,
        opts,
        BackendKind::TinySolver,
        backend_opts,
    )
}

/// Optimize planar intrinsics using the selected backend.
pub fn optimize_planar_intrinsics_with_backend(
    dataset: &PlanarDataset,
    initial: PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate> {
    let (ir, initial_map) = build_planar_intrinsics_ir(&dataset, &initial, &opts)?;
    let solution = solve_with_backend(backend, &ir, &initial_map, &backend_opts)?;

    let cam_vec = solution
        .params
        .get("cam")
        .ok_or_else(|| anyhow!("missing camera parameters in solution"))?;
    let intrinsics = unpack_intrinsics(cam_vec.as_view())?;

    let dist_vec = solution
        .params
        .get("dist")
        .ok_or_else(|| anyhow!("missing distortion parameters in solution"))?;
    let distortion = unpack_distortion(dist_vec.as_view())?;

    let mut poses = Vec::with_capacity(dataset.num_views());
    for i in 0..dataset.num_views() {
        let key = format!("pose/{}", i);
        let pose_vec = solution
            .params
            .get(&key)
            .ok_or_else(|| anyhow!("missing pose {} in solution", i))?;
        poses.push(se3_dvec_to_iso3(pose_vec.as_view())?);
    }

    let camera = make_pinhole_camera(intrinsics, distortion);
    let views_with_poses: Vec<View<Iso3>> = dataset
        .views
        .iter()
        .zip(poses.iter().cloned())
        .map(|(view, pose)| View::new(view.obs.clone(), pose))
        .collect();
    let mean_reproj_error = compute_mean_reproj_error(&camera, &views_with_poses)?;

    Ok(PlanarIntrinsicsEstimate {
        params: PlanarIntrinsicsParams::new(camera, poses)?,
        report: solution.solve_report,
        mean_reproj_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_validation_catches_missing_param() {
        let mut ir = ProblemIR::new();
        let cam_id = ir.add_param_block(
            "cam",
            INTRINSICS_DIM,
            ManifoldKind::Euclidean,
            FixedMask::all_free(),
            None,
        );
        let residual = ResidualBlock {
            params: vec![cam_id, crate::ir::ParamId(42)],
            loss: RobustLoss::None,
            factor: FactorKind::ReprojPointPinhole4 {
                pw: [0.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                w: 1.0,
            },
            residual_dim: 2,
        };
        ir.add_residual_block(residual);

        let err = ir.validate().unwrap_err().to_string();
        assert!(
            err.contains("references missing param"),
            "unexpected validation error: {}",
            err
        );
    }
}
