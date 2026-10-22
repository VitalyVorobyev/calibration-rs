//! Multi-camera rig extrinsics calibration.
//!
//! Optimizes per-camera camera parameters, per-camera extrinsics (`T_R_C`,
//! camera-to-rig) and per-view rig poses (`T_R_T`, target-to-rig). Each
//! camera has its own model ([`CameraParams::spec`]): any distortion kind and
//! an identity or Scheimpflug sensor, so one rig can mix lenses.

use crate::Error;
use crate::backend::{self, BackendSolveOptions, SolveReport};
use crate::ir::ReprojSqrtInfo;
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ReprojChain, ResidualBlock};
use crate::ir::{ParamId, RobustLoss};
use crate::params::camera::{CameraBlockInput, add_camera_blocks, camera_spec, unpack_camera};
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use crate::uncertainty::{self, RigLabelling};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vision_calibration_core::{
    CameraFix, CameraModel, CameraModelSpec, CameraParams, FixedParams, Iso3, NoMeta, RigDataset,
    Uncertainty,
};

/// Dataset type for rig extrinsics optimization.
pub type RigExtrinsicsDataset = RigDataset<NoMeta>;

/// Result of rig extrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsParams {
    /// Per-camera calibrated parameters, each in its own model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera extrinsics (`camera -> rig`, `T_R_C`).
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view rig poses (`target -> rig`, `T_R_T`).
    pub rig_from_target: Vec<Iso3>,
}

/// Output of rig extrinsics optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RigExtrinsicsEstimate {
    /// Refined rig extrinsics parameters.
    pub params: RigExtrinsicsParams,
    /// Covariance of the estimated parameters at the solution.
    #[serde(default)]
    pub uncertainty: Uncertainty,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean reprojection error in pixels (computed post-optimization).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Solve options for rig extrinsics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigExtrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Camera parameters held fixed, one entry per camera, each resolved
    /// against that camera's model. Empty holds every camera's model
    /// defaults (`k3` for Brown-Conrady; everything but the low-order radial
    /// terms under a Scheimpflug sensor).
    pub fix: Vec<CameraFix>,
    /// Per-camera extrinsics masking (fix camera-to-rig transform).
    pub fix_extrinsics: Vec<bool>,
    /// View indices to fix (e.g., first view for gauge freedom).
    pub fix_rig_poses: Vec<usize>,
}

/// Each camera's model and resolved fixed parameters: `fix` holds one entry
/// per camera, or none for the model defaults.
pub(crate) fn resolve_camera_fix(
    cameras: &[CameraParams],
    fix: &[CameraFix],
) -> Result<(Vec<CameraModelSpec>, Vec<FixedParams>), Error> {
    if !fix.is_empty() && fix.len() != cameras.len() {
        return Err(Error::invalid_input(format!(
            "fix has {} entries; expected none or one per camera ({})",
            fix.len(),
            cameras.len()
        )));
    }
    let specs = cameras
        .iter()
        .map(camera_spec)
        .collect::<Result<Vec<_>, _>>()?;
    let default = CameraFix::default();
    let fixed = specs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            fix.get(i)
                .unwrap_or(&default)
                .resolve(*spec)
                .map_err(|e| Error::invalid_input(format!("camera {i}: {e}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((specs, fixed))
}

/// The runtime models of rig cameras.
pub(crate) fn camera_models(cameras: &[CameraParams]) -> Result<Vec<CameraModel>, Error> {
    cameras
        .iter()
        .map(|c| {
            c.build()
                .map_err(|e| Error::invalid_input(format!("camera model: {e}")))
        })
        .collect()
}

/// Per-camera reprojection sums, for the mean errors a rig solve reports.
#[derive(Debug, Clone)]
pub(crate) struct ReprojectionSums {
    sums: Vec<f64>,
    counts: Vec<usize>,
}

impl ReprojectionSums {
    pub(crate) fn new(num_cameras: usize) -> Self {
        Self {
            sums: vec![0.0; num_cameras],
            counts: vec![0; num_cameras],
        }
    }

    /// Add camera `cam`'s error for one point; a point that does not project
    /// to a finite pixel is skipped.
    pub(crate) fn add(&mut self, cam: usize, error: Option<f64>) {
        if let Some(e) = error.filter(|e| e.is_finite()) {
            self.sums[cam] += e;
            self.counts[cam] += 1;
        }
    }

    /// The mean over all cameras and each camera's mean (zero for a camera
    /// without a valid point).
    pub(crate) fn means(&self) -> (f64, Vec<f64>) {
        let per_cam = self
            .sums
            .iter()
            .zip(&self.counts)
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();
        let total: usize = self.counts.iter().sum();
        let mean = if total > 0 {
            self.sums.iter().sum::<f64>() / total as f64
        } else {
            0.0
        };
        (mean, per_cam)
    }
}

/// Build IR for rig extrinsics optimization.
fn build_rig_extrinsics_ir(
    dataset: &RigExtrinsicsDataset,
    initial: &RigExtrinsicsParams,
    fixed: &[FixedParams],
    opts: &RigExtrinsicsSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error> {
    if initial.cameras.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "camera count {} != num_cameras {}",
            initial.cameras.len(),
            dataset.num_cameras
        )));
    }
    if initial.cam_to_rig.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "cam_to_rig count {} != num_cameras {}",
            initial.cam_to_rig.len(),
            dataset.num_cameras
        )));
    }
    if initial.rig_from_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "rig_from_target count {} != num_views {}",
            initial.rig_from_target.len(),
            dataset.num_views()
        )));
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
    let models: Vec<CameraModelSpec> = initial
        .cameras
        .iter()
        .map(camera_spec)
        .collect::<Result<_, _>>()?;

    // 2. Per-camera extrinsics
    let mut extr_ids: Vec<ParamId> = Vec::new();
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

    // 3. Per-view rig poses + residuals
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let fixed = if opts.fix_rig_poses.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("rig_pose/{}", view_idx);
        let rig_pose_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.rig_from_target[view_idx]));

        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let params = camera_blocks[cam_idx].with(&[extr_ids[cam_idx], rig_pose_id]);
            for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                ir.add_residual_block(ResidualBlock {
                    params: params.clone(),
                    loss: opts.robust_loss,
                    factor: FactorKind::ReprojPoint {
                        model: models[cam_idx],
                        chain: ReprojChain::TwoSe3,
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

/// The pose a solution holds in block `name`.
pub(crate) fn solved_pose(
    solution: &HashMap<String, DVector<f64>>,
    name: &str,
) -> Result<Iso3, Error> {
    let block = solution
        .get(name)
        .ok_or_else(|| Error::numerical(format!("missing {name} solution block")))?;
    se3_dvec_to_iso3(block.as_view())
}

/// Optimize rig extrinsics using specified backend.
///
/// Each camera keeps its own model, and `opts.fix` is resolved against it.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when a camera has a general homography sensor,
///   `opts.fix` names a parameter a camera's model does not have or has the
///   wrong length, or the parameter counts do not match the dataset.
/// - Any IR construction or solver backend failure.
pub fn optimize_rig_extrinsics(
    dataset: RigExtrinsicsDataset,
    initial: RigExtrinsicsParams,
    opts: RigExtrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<RigExtrinsicsEstimate, Error> {
    let (specs, fixed) = resolve_camera_fix(&initial.cameras, &opts.fix)?;
    let (ir, initial_map) = build_rig_extrinsics_ir(&dataset, &initial, &fixed, &opts)?;
    let solution = backend::solve(&ir, &initial_map, &backend_opts)?;

    let cameras = specs
        .iter()
        .enumerate()
        .map(|(i, spec)| unpack_camera(&solution.params, *spec, Some(i)))
        .collect::<Result<Vec<_>, Error>>()?;
    let cam_to_rig = (0..dataset.num_cameras)
        .map(|i| solved_pose(&solution.params, &format!("extr/{i}")))
        .collect::<Result<Vec<_>, Error>>()?;
    let rig_from_target = (0..dataset.num_views())
        .map(|i| solved_pose(&solution.params, &format!("rig_pose/{i}")))
        .collect::<Result<Vec<_>, Error>>()?;

    let (mean_reproj_error, per_cam_reproj_errors) =
        rig_reprojection_errors(&dataset, &cameras, &cam_to_rig, &rig_from_target)?;

    let labels = RigLabelling {
        cameras: &specs,
        multi_camera: dataset.num_cameras > 1,
        handeye_mode: None,
    }
    .labels(&ir, &solution);
    let uncertainty = uncertainty::labelled(&ir, &solution.uncertainty, &solution.noise, &labels);

    Ok(RigExtrinsicsEstimate {
        params: RigExtrinsicsParams {
            cameras,
            cam_to_rig,
            rig_from_target,
        },
        uncertainty,
        report: solution.solve_report,
        mean_reproj_error,
        per_cam_reproj_errors,
    })
}

/// Mean reprojection error over the rig and per camera, through
/// `cam_se3_rig · rig_se3_target`; points that do not project to a finite
/// pixel are skipped, and a camera without a valid point reports zero.
fn rig_reprojection_errors(
    dataset: &RigExtrinsicsDataset,
    cameras: &[CameraParams],
    cam_to_rig: &[Iso3],
    rig_from_target: &[Iso3],
) -> Result<(f64, Vec<f64>), Error> {
    let models = camera_models(cameras)?;
    let cam_se3_rig: Vec<Iso3> = cam_to_rig.iter().map(|t| t.inverse()).collect();
    let mut sums = ReprojectionSums::new(dataset.num_cameras);
    for (view_idx, view) in dataset.views.iter().enumerate() {
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let cam_se3_target = cam_se3_rig[cam_idx] * rig_from_target[view_idx];
            for (p3d, p2d) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_cam = cam_se3_target.transform_point(p3d);
                let error = models[cam_idx]
                    .project_point(&p_cam)
                    .map(|projected| (projected - *p2d).norm());
                sums.add(cam_idx, error);
            }
        }
    }
    Ok(sums.means())
}
