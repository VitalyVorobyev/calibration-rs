//! Planar intrinsics optimization using the backend-agnostic IR.
//!
//! Each observation contributes a residual block with two residuals (u, v),
//! enabling robust loss to operate per point rather than per view.

use crate::Error;
use crate::backend::{BackendKind, BackendSolveOptions, SolveReport, solve_with_backend};
use crate::ir::{
    CameraModelDesc, DistortionKind, FactorKind, FixedMask, ManifoldKind, ProblemIR, ReprojChain,
    ResidualBlock, RobustLoss,
};
use crate::params::distortion::{pack_distortion_params, unpack_distortion_params};
use crate::params::intrinsics::{INTRINSICS_DIM, pack_intrinsics, unpack_intrinsics};
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraModel, CameraParams, DistortionFixMask, DistortionParams, FxFyCxCySkew,
    IntrinsicsFixMask, IntrinsicsParams, Iso3, PinholeCamera, PlanarDataset, ProjectionParams,
    Real, SensorParams, TargetPose, View, compute_mean_reproj_error, make_pinhole_camera,
    pinhole_camera_params,
};

/// Optimization result for planar intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsParams {
    /// Refined camera model (model-agnostic serializable parameters).
    pub camera: CameraParams,
    /// Refined target-to-camera poses.
    pub camera_se3_target: Vec<Iso3>,
}

impl PlanarIntrinsicsParams {
    /// Construct parameter pack from model-agnostic [`CameraParams`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `camera_se3_target` is empty.
    pub fn new(camera: CameraParams, camera_se3_target: Vec<Iso3>) -> Result<Self, Error> {
        if camera_se3_target.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        Ok(Self {
            camera,
            camera_se3_target,
        })
    }

    /// Construct from a concrete [`PinholeCamera`] (Brown-Conrady) and poses.
    ///
    /// Convenience back-compat wrapper used by rig calibration and tests that
    /// operate with the concrete `PinholeCamera` type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `camera_se3_target` is empty.
    pub fn from_pinhole(
        camera: PinholeCamera,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self, Error> {
        Self::new(pinhole_camera_params(&camera), camera_se3_target)
    }

    /// Reconstruct a [`PinholeCamera`] (Brown-Conrady) from the stored params.
    ///
    /// Only valid when the stored distortion is `BrownConrady5` or `None`.
    /// Rig calibration always uses Brown-Conrady, so this path is always safe
    /// for rig consumers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the stored distortion is not a
    /// Brown-Conrady or `None` variant, or if intrinsics are unavailable.
    pub fn pinhole_camera(&self) -> Result<PinholeCamera, Error> {
        let k = self.intrinsics();
        let dist = match &self.camera.distortion {
            DistortionParams::None => BrownConrady5::default(),
            DistortionParams::BrownConrady5 { params } => *params,
            other => {
                return Err(Error::invalid_input(format!(
                    "pinhole_camera() requires BrownConrady5 or None distortion, got {:?}",
                    std::mem::discriminant(other)
                )));
            }
        };
        Ok(make_pinhole_camera(k, dist))
    }

    /// Build the runtime [`CameraModel`] for residual computation.
    ///
    /// Panics only if a stored homography sensor is not invertible (cannot
    /// arise from the standard planar calibration path).
    pub fn build_camera(&self) -> CameraModel {
        self.camera.build()
    }

    /// Create from individual components (Brown-Conrady).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `poses` is empty.
    pub fn new_from_components(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        poses: Vec<Iso3>,
    ) -> Result<Self, Error> {
        Self::from_pinhole(make_pinhole_camera(intrinsics, distortion), poses)
    }

    /// Create with zero distortion (pinhole model only).
    pub fn from_intrinsics(
        intrinsics: FxFyCxCySkew<Real>,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self, Error> {
        Self::from_pinhole(
            make_pinhole_camera(intrinsics, BrownConrady5::default()),
            camera_se3_target,
        )
    }

    /// Get intrinsics.
    pub fn intrinsics(&self) -> FxFyCxCySkew<Real> {
        match self.camera.intrinsics {
            IntrinsicsParams::FxFyCxCySkew { params } => params,
        }
    }

    /// Get distortion as Brown-Conrady (for BC-only consumers).
    ///
    /// Returns `None` when the active model is not Brown-Conrady or None.
    pub fn distortion(&self) -> Option<BrownConrady5<Real>> {
        match &self.camera.distortion {
            DistortionParams::None => Some(BrownConrady5::default()),
            DistortionParams::BrownConrady5 { params } => Some(*params),
            _ => None,
        }
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
    ///
    /// This mask is applied only when the distortion model is BrownConrady5.
    /// For other models all distortion coefficients are free by default.
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

/// Map a [`DistortionParams`] variant to the corresponding [`DistortionKind`].
fn distortion_params_to_kind(d: &DistortionParams) -> DistortionKind {
    match d {
        DistortionParams::None => DistortionKind::None,
        DistortionParams::BrownConrady5 { .. } => DistortionKind::BrownConrady5,
        DistortionParams::Rational { .. } => DistortionKind::Rational8,
        DistortionParams::ThinPrism { .. } => DistortionKind::ThinPrism9,
        DistortionParams::Division { .. } => DistortionKind::Division1,
    }
}

/// Select the [`CameraModelDesc`] matching the distortion kind.
fn camera_model_desc_for_kind(kind: DistortionKind) -> CameraModelDesc {
    match kind {
        DistortionKind::None => CameraModelDesc::PINHOLE4,
        DistortionKind::BrownConrady5 => CameraModelDesc::PINHOLE4_DIST5,
        DistortionKind::Rational8 => CameraModelDesc::PINHOLE4_RATIONAL8,
        DistortionKind::ThinPrism9 => CameraModelDesc::PINHOLE4_THINPRISM9,
        DistortionKind::Division1 => CameraModelDesc::PINHOLE4_DIVISION1,
    }
}

/// Build the backend-agnostic IR and initial values for planar intrinsics.
///
/// This is the canonical problem builder reused by all backends.
fn build_planar_intrinsics_ir(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    opts: &PlanarIntrinsicsSolveOptions,
) -> Result<
    (
        crate::ir::ProblemIR,
        std::collections::HashMap<String, nalgebra::DVector<f64>>,
    ),
    Error,
> {
    let kind = distortion_params_to_kind(&initial.camera.distortion);
    let model = camera_model_desc_for_kind(kind);

    // For BC5 the existing `fix_distortion` indices apply; for other models
    // all distortion params are free (the mask is BC5-shaped and doesn't
    // translate to other orderings).
    let fix_dist_indices = if kind == DistortionKind::BrownConrady5 {
        opts.fix_distortion.to_indices()
    } else {
        vec![]
    };

    use std::collections::{HashMap, HashSet};

    let intrinsics = initial.intrinsics();
    let dist_vec = pack_distortion_params(&initial.camera.distortion);

    let poses = &initial.camera_se3_target;

    let fixed_pose_set: HashSet<usize> = opts.fix_poses.iter().copied().collect();

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, nalgebra::DVector<f64>> = HashMap::new();

    let cam_id = ir.add_param_block(
        "cam",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&opts.fix_intrinsics.to_indices()),
        None,
    );
    initial_map.insert("cam".to_string(), pack_intrinsics(&intrinsics)?);

    // Only add a distortion block when the model has one.
    let dist_id = if kind != DistortionKind::None {
        let id = ir.add_param_block(
            "dist",
            kind.dim(),
            ManifoldKind::Euclidean,
            FixedMask::fix_indices(&fix_dist_indices),
            None,
        );
        initial_map.insert("dist".to_string(), dist_vec);
        Some(id)
    } else {
        None
    };

    let mut pose_ids = Vec::with_capacity(poses.len());
    for (view_idx, pose) in poses.iter().enumerate() {
        let fixed = if fixed_pose_set.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let pose_key = format!("pose/{view_idx}");
        let pose_id = ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(pose_key, iso3_to_se3_dvec(pose));
        pose_ids.push(pose_id);
    }

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_id = pose_ids[view_idx];
        for (point_idx, (pw, uv)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            let mut params = vec![cam_id];
            if let Some(did) = dist_id {
                params.push(did);
            }
            params.push(pose_id);
            let factor = FactorKind::ReprojPoint {
                model,
                chain: ReprojChain::SinglePose,
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: view.obs.weight(point_idx),
            };
            ir.add_residual_block(ResidualBlock {
                params,
                loss: opts.robust_loss,
                factor,
                residual_dim: 2,
            });
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

/// Optimize planar intrinsics using the default tiny-solver backend.
///
/// # Errors
///
/// Returns [`Error`] if IR construction or solver backend fails.
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
///
/// # Errors
///
/// Returns [`Error`] if IR construction or solver backend fails.
pub fn optimize_planar_intrinsics_with_backend(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    let kind = distortion_params_to_kind(&initial.camera.distortion);

    let (ir, initial_map) = build_planar_intrinsics_ir(dataset, initial, &opts)?;
    let solution = solve_with_backend(backend, &ir, &initial_map, &backend_opts)?;

    let cam_vec = solution
        .params
        .get("cam")
        .ok_or_else(|| Error::numerical("missing camera parameters in solution"))?;
    let intrinsics = unpack_intrinsics(cam_vec.as_view())?;

    // Unpack distortion (may be empty for DistortionKind::None).
    let distortion = if kind != DistortionKind::None {
        let dist_vec = solution
            .params
            .get("dist")
            .ok_or_else(|| Error::numerical("missing distortion parameters in solution"))?;
        unpack_distortion_params(kind, dist_vec.as_view())?
    } else {
        DistortionParams::None
    };

    let mut poses = Vec::with_capacity(dataset.num_views());
    for i in 0..dataset.num_views() {
        let key = format!("pose/{}", i);
        let pose_vec = solution
            .params
            .get(&key)
            .ok_or_else(|| Error::numerical(format!("missing pose {} in solution", i)))?;
        poses.push(se3_dvec_to_iso3(pose_vec.as_view())?);
    }

    let camera_params = CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion,
        sensor: SensorParams::Identity,
        intrinsics: IntrinsicsParams::FxFyCxCySkew { params: intrinsics },
    };
    let camera_model = camera_params.build();

    let views_with_poses: Vec<View<TargetPose>> = dataset
        .views
        .iter()
        .zip(poses.iter().cloned())
        .map(|(view, camera_se3_target)| {
            View::new(view.obs.clone(), TargetPose { camera_se3_target })
        })
        .collect();
    let mean_reproj_error = compute_mean_reproj_error(&camera_model, &views_with_poses)?;

    Ok(PlanarIntrinsicsEstimate {
        params: PlanarIntrinsicsParams::new(camera_params, poses)?,
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
            factor: FactorKind::ReprojPoint {
                model: crate::ir::CameraModelDesc::PINHOLE4,
                chain: crate::ir::ReprojChain::SinglePose,
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
