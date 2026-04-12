//! Planar intrinsics optimization using the backend-agnostic IR.
//!
//! Each observation contributes a residual block with two residuals (u, v),
//! enabling robust loss to operate per point rather than per view.

use crate::Error;
use crate::backend::{BackendKind, BackendSolveOptions, SolveReport, solve_with_backend};
use crate::ir::RobustLoss;
use crate::params::distortion::unpack_distortion;
use crate::params::intrinsics::unpack_intrinsics;
use crate::params::pose_se3::se3_dvec_to_iso3;
use crate::problems::planar_family_shared::{
    PlanarReprojectionFactorModel, PlanarReprojectionIrOptions, build_planar_reprojection_ir,
};
use anyhow::{Result as AnyhowResult, anyhow};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, DistortionFixMask, FxFyCxCySkew, IntrinsicsFixMask, Iso3, PinholeCamera,
    PlanarDataset, Real, TargetPose, View, compute_mean_reproj_error, make_pinhole_camera,
};

/// Optimization result for planar intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsParams {
    /// Refined camera with intrinsics and distortion.
    pub camera: PinholeCamera,
    /// Refined target-to-camera poses.
    pub camera_se3_target: Vec<Iso3>,
}

impl PlanarIntrinsicsParams {
    /// Construct parameter pack with non-empty pose validation.
    pub fn new(camera: PinholeCamera, camera_se3_target: Vec<Iso3>) -> Result<Self, Error> {
        if camera_se3_target.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
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
    ) -> Result<Self, Error> {
        Self::new(make_pinhole_camera(intrinsics, distortion), poses)
    }

    /// Create with zero distortion (pinhole model only).
    pub fn from_intrinsics(
        intrinsics: FxFyCxCySkew<Real>,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self, Error> {
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

/// Output of planar intrinsics optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanarIntrinsicsEstimate {
    /// Refined camera and poses.
    pub params: PlanarIntrinsicsParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean reprojection error in pixels.
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
) -> AnyhowResult<(
    crate::ir::ProblemIR,
    std::collections::HashMap<String, nalgebra::DVector<f64>>,
)> {
    build_planar_reprojection_ir(
        dataset,
        &initial.camera.k,
        &initial.camera.dist,
        &initial.camera_se3_target,
        &PlanarReprojectionIrOptions {
            robust_loss: opts.robust_loss,
            fix_intrinsics_indices: opts.fix_intrinsics.to_indices(),
            fix_distortion_indices: opts.fix_distortion.to_indices(),
            fix_pose_indices: opts.fix_poses.clone(),
            sensor: None,
            factor_model: PlanarReprojectionFactorModel::PinholeDistortion,
        },
    )
}

/// Optimize planar intrinsics using the default tiny-solver backend.
pub fn optimize_planar_intrinsics(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
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
    initial: &PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    let (ir, initial_map) = build_planar_intrinsics_ir(dataset, initial, &opts)?;
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
    let views_with_poses: Vec<View<TargetPose>> = dataset
        .views
        .iter()
        .zip(poses.iter().cloned())
        .map(|(view, camera_se3_target)| {
            View::new(view.obs.clone(), TargetPose { camera_se3_target })
        })
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
    use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock};
    use crate::params::intrinsics::INTRINSICS_DIM;

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
