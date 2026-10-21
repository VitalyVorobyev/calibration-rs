//! Planar intrinsics optimization using the backend-agnostic IR.
//!
//! One builder serves every camera model: the camera's spec
//! ([`CameraParams::spec`]) selects the parameter blocks, and a
//! [`CameraFix`] names the parameters held fixed. Each observation
//! contributes a residual block with two residuals (u, v), so a robust loss
//! acts per point rather than per view.

use std::collections::{HashMap, HashSet};

use crate::Error;
use crate::backend::{self, BackendSolveOptions, SolveReport};
use crate::ir::{
    FactorKind, FixedMask, ManifoldKind, ProblemIR, ReprojChain, ResidualBlock, RobustLoss,
    SensorKind,
};
pub use crate::params::camera::CameraBounds;
use crate::params::camera::{CameraBlockInput, add_camera_blocks, camera_spec, unpack_camera};
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use crate::uncertainty;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraModel, CameraParams, DistortionParams, FixedParams, FxFyCxCy,
    Iso3, PinholeCamera, PlanarDataset, Real, TargetPose, Uncertainty, View,
    compute_mean_reproj_error, make_pinhole_camera, pinhole_camera_params,
};

/// Optimization result for planar intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct PlanarIntrinsicsParams {
    /// Refined camera model (model-agnostic serializable parameters).
    pub camera: CameraParams,
    /// Refined target-to-camera poses.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
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

    /// Construct from a concrete [`PinholeCamera`] (Brown-Conrady, untilted)
    /// and poses — the camera the linear initialization returns.
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

    /// Build the runtime [`CameraModel`] for residual computation.
    ///
    /// # Errors
    ///
    /// Returns [`vision_calibration_core::Error::Singular`] if a stored
    /// homography sensor is not invertible. The planar solve writes an
    /// identity or Scheimpflug sensor, so this is reachable only via a
    /// hand-edited or corrupted export.
    pub fn build_camera(&self) -> Result<CameraModel, vision_calibration_core::Error> {
        self.camera.build()
    }

    /// Create from individual components (Brown-Conrady).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `poses` is empty.
    pub fn new_from_components(
        intrinsics: FxFyCxCy<Real>,
        distortion: BrownConrady5<Real>,
        poses: Vec<Iso3>,
    ) -> Result<Self, Error> {
        Self::from_pinhole(make_pinhole_camera(intrinsics, distortion), poses)
    }

    /// Create with zero distortion (pinhole model only).
    pub fn from_intrinsics(
        intrinsics: FxFyCxCy<Real>,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self, Error> {
        Self::from_pinhole(
            make_pinhole_camera(intrinsics, BrownConrady5::default()),
            camera_se3_target,
        )
    }

    /// Get intrinsics.
    pub fn intrinsics(&self) -> FxFyCxCy<Real> {
        self.camera.intrinsics
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
    /// Covariance of the estimated camera parameters and poses.
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

/// Solve options for planar intrinsics, for any camera model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanarIntrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Camera parameters held fixed, named per slot and resolved against the
    /// initial camera's model. The default holds the model's defaults (`k3`
    /// for Brown-Conrady; everything but the low-order radial terms under a
    /// Scheimpflug sensor).
    pub fix: CameraFix,
    /// Indices of poses to keep fixed.
    pub fix_poses: Vec<usize>,
    /// Optional box bounds on intrinsics and tilt. `None` ⇒ unbounded.
    #[serde(default)]
    pub bounds: Option<CameraBounds>,
}

/// IR options of one planar solve, with the fix lists already resolved.
#[derive(Debug, Clone)]
pub(crate) struct PlanarIrOptions<'a> {
    pub robust_loss: RobustLoss,
    pub fixed: &'a FixedParams,
    pub fix_poses: &'a [usize],
    pub bounds: Option<&'a CameraBounds>,
}

/// Build the backend-agnostic IR and initial values for a planar solve of
/// `camera` (any camera model) and one pose per view.
///
/// Blocks are the camera's (`cam`, then `proj`, `dist` and `sensor` for the
/// kinds that have parameters) and `pose/{i}`, in that order; the covariance
/// labels rely on these names.
pub(crate) fn build_planar_reprojection_ir(
    dataset: &PlanarDataset,
    camera: &CameraParams,
    poses: &[Iso3],
    opts: &PlanarIrOptions<'_>,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error> {
    let model = camera_spec(camera)?;
    if dataset.num_views() != poses.len() {
        return Err(Error::invalid_input(format!(
            "pose count ({}) must match number of views ({})",
            poses.len(),
            dataset.num_views()
        )));
    }
    if let Some(&bad) = opts.fix_poses.iter().find(|&&i| i >= dataset.num_views()) {
        return Err(Error::invalid_input(format!(
            "fixed pose index {} out of range ({} views)",
            bad,
            dataset.num_views()
        )));
    }

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();
    let fixed_pose_set: HashSet<usize> = opts.fix_poses.iter().copied().collect();

    let camera_blocks = add_camera_blocks(
        &mut ir,
        &mut initial_map,
        &[CameraBlockInput {
            params: camera,
            fixed: opts.fixed,
            bounds: opts.bounds,
        }],
        false,
    )?
    .remove(0);

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
            let factor = FactorKind::ReprojPoint {
                model,
                chain: ReprojChain::SinglePose,
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: view.obs.weight(point_idx),
            };
            ir.add_residual_block(ResidualBlock {
                params: camera_blocks.with(&[pose_id]),
                loss: opts.robust_loss,
                factor,
                residual_dim: 2,
            });
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

/// Build the IR of [`optimize_planar_intrinsics`] (for tests that inspect it).
#[cfg(test)]
pub(crate) fn build_planar_intrinsics_ir(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    opts: &PlanarIntrinsicsSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error> {
    let fixed = opts.fix.resolve(camera_spec(&initial.camera)?)?;
    build_planar_reprojection_ir(
        dataset,
        &initial.camera,
        &initial.camera_se3_target,
        &PlanarIrOptions {
            robust_loss: opts.robust_loss,
            fixed: &fixed,
            fix_poses: &opts.fix_poses,
            bounds: opts.bounds.as_ref(),
        },
    )
}

/// Optimize planar intrinsics, distortion, sensor and per-view poses with
/// the backend `backend_opts` selects.
///
/// The camera model is the one `initial.camera` holds: any distortion, and
/// an identity or Scheimpflug sensor. `opts.fix` is resolved against it.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when the camera has a general homography
///   sensor, a fix list names a parameter the model does not have, tilt
///   bounds are given for a non-Scheimpflug sensor, or poses and views do
///   not match.
/// - Any IR construction or solver backend failure.
pub fn optimize_planar_intrinsics(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    let spec = camera_spec(&initial.camera)?;
    let fixed = opts.fix.resolve(spec)?;
    let (ir, initial_map) = build_planar_reprojection_ir(
        dataset,
        &initial.camera,
        &initial.camera_se3_target,
        &PlanarIrOptions {
            robust_loss: opts.robust_loss,
            fixed: &fixed,
            fix_poses: &opts.fix_poses,
            bounds: opts.bounds.as_ref(),
        },
    )?;
    let solution = backend::solve(&ir, &initial_map, &backend_opts)?;

    let camera = unpack_camera(&solution.params, spec, None)?;
    let mut poses = Vec::with_capacity(dataset.num_views());
    for i in 0..dataset.num_views() {
        let name = format!("pose/{i}");
        let block = solution
            .params
            .get(&name)
            .ok_or_else(|| Error::numerical(format!("missing {name} solution block")))?;
        poses.push(se3_dvec_to_iso3(block.as_view())?);
    }

    // The sensor is identity or Scheimpflug (compiled from its tilt angles),
    // both invertible by construction; only a general homography can fail.
    let camera_model = camera
        .build()
        .expect("identity and Scheimpflug sensors are invertible");
    let mean_reproj_error = match spec.sensor {
        SensorKind::Identity => mean_reproj_error(&camera_model, dataset, &poses)?,
        SensorKind::Scheimpflug => finite_mean_reproj_error(&camera_model, dataset, &poses),
    };

    let labels: Vec<_> = ir
        .params
        .iter()
        .map(|block| uncertainty::planar_block_labels(block, spec))
        .collect();
    Ok(PlanarIntrinsicsEstimate {
        params: PlanarIntrinsicsParams::new(camera, poses)?,
        report: solution.solve_report,
        mean_reproj_error,
        uncertainty: uncertainty::labelled(&ir, &solution.uncertainty, &labels),
    })
}

/// Mean reprojection error over every projectable point; an error when no
/// point projects.
fn mean_reproj_error(
    camera: &CameraModel,
    dataset: &PlanarDataset,
    poses: &[Iso3],
) -> Result<f64, Error> {
    let views: Vec<View<TargetPose>> = dataset
        .views
        .iter()
        .zip(poses.iter().cloned())
        .map(|(view, camera_se3_target)| {
            View::new(view.obs.clone(), TargetPose { camera_se3_target })
        })
        .collect();
    Ok(compute_mean_reproj_error(camera, &views)?)
}

/// Mean reprojection error over the points that project to a finite pixel;
/// `INFINITY` when none does.
///
/// A tilted sensor is solved as a multi-start whose stages rank candidates
/// by this value, so a stage that sends a few points to a non-finite pixel
/// is still scored on the rest rather than dropped.
fn finite_mean_reproj_error(camera: &CameraModel, dataset: &PlanarDataset, poses: &[Iso3]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (view, pose) in dataset.views.iter().zip(poses.iter()) {
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2d).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DistortionKind, SensorKind};
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
                model: crate::ir::CameraModelSpec::pinhole(
                    DistortionKind::None,
                    SensorKind::Identity,
                ),
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
