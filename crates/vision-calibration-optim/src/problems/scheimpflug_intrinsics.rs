//! Scheimpflug planar intrinsics optimization using the backend-agnostic IR.

use crate::backend::{BackendKind, BackendSolveOptions, SolveReport, solve_with_backend};
use crate::ir::RobustLoss;
use crate::params::distortion::unpack_distortion;
use crate::params::intrinsics::unpack_intrinsics;
use crate::params::pose_se3::se3_dvec_to_iso3;
use crate::problems::planar_family_shared::{
    PlanarReprojectionFactorModel, PlanarReprojectionIrOptions, PlanarSensorIrOptions,
    build_planar_reprojection_ir,
};
use anyhow::{Result, anyhow, ensure};
use nalgebra::DVectorView;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, DistortionFixMask, FxFyCxCySkew, IntrinsicsFixMask, Iso3, Pinhole,
    PlanarDataset, Real, ScheimpflugParams,
};

/// Mask for Scheimpflug tilt parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ScheimpflugFixMask {
    /// Keep `tilt_x` fixed during optimization.
    pub tilt_x: bool,
    /// Keep `tilt_y` fixed during optimization.
    pub tilt_y: bool,
}

impl ScheimpflugFixMask {
    fn as_flags(self) -> [bool; 2] {
        [self.tilt_x, self.tilt_y]
    }
}

/// Initial/refined parameters for Scheimpflug intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsParams {
    /// Camera intrinsics.
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Brown-Conrady distortion parameters.
    pub distortion: BrownConrady5<Real>,
    /// Scheimpflug sensor tilt parameters.
    pub sensor: ScheimpflugParams,
    /// Target poses per view (`camera_se3_target`).
    pub camera_se3_target: Vec<Iso3>,
}

impl ScheimpflugIntrinsicsParams {
    /// Construct parameter pack with validation.
    pub fn new(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        sensor: ScheimpflugParams,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self> {
        ensure!(!camera_se3_target.is_empty(), "need at least one pose");
        Ok(Self {
            intrinsics,
            distortion,
            sensor,
            camera_se3_target,
        })
    }
}

/// Solve options for Scheimpflug intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Mask for fixing intrinsics parameters.
    pub fix_intrinsics: IntrinsicsFixMask,
    /// Mask for fixing distortion parameters.
    pub fix_distortion: DistortionFixMask,
    /// Mask for fixing Scheimpflug tilt parameters.
    pub fix_scheimpflug: ScheimpflugFixMask,
    /// Indices of poses to keep fixed.
    pub fix_poses: Vec<usize>,
}

impl Default for ScheimpflugIntrinsicsSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::default(),
            fix_distortion: DistortionFixMask::radial_only(),
            fix_scheimpflug: ScheimpflugFixMask::default(),
            fix_poses: vec![0],
        }
    }
}

/// Optimization result for Scheimpflug intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsEstimate {
    /// Refined parameters.
    pub params: ScheimpflugIntrinsicsParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
}

fn build_scheimpflug_intrinsics_ir(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    opts: &ScheimpflugIntrinsicsSolveOptions,
) -> Result<(
    crate::ir::ProblemIR,
    std::collections::HashMap<String, nalgebra::DVector<f64>>,
)> {
    build_planar_reprojection_ir(
        dataset,
        &initial.intrinsics,
        &initial.distortion,
        &initial.camera_se3_target,
        &PlanarReprojectionIrOptions {
            robust_loss: opts.robust_loss,
            fix_intrinsics_indices: opts.fix_intrinsics.to_indices(),
            fix_distortion_indices: opts.fix_distortion.to_indices(),
            fix_pose_indices: opts.fix_poses.clone(),
            sensor: Some(PlanarSensorIrOptions {
                params: initial.sensor,
                fix_indices: opts.fix_scheimpflug.as_flags(),
            }),
            factor_model: PlanarReprojectionFactorModel::PinholeDistortionScheimpflug,
        },
    )
}

/// Optimize Scheimpflug intrinsics using the default tiny-solver backend.
pub fn optimize_scheimpflug_intrinsics(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    opts: ScheimpflugIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<ScheimpflugIntrinsicsEstimate> {
    optimize_scheimpflug_intrinsics_with_backend(
        dataset,
        initial,
        opts,
        BackendKind::TinySolver,
        backend_opts,
    )
}

/// Optimize Scheimpflug intrinsics using the selected backend.
pub fn optimize_scheimpflug_intrinsics_with_backend(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    opts: ScheimpflugIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: BackendSolveOptions,
) -> Result<ScheimpflugIntrinsicsEstimate> {
    let (ir, initial_map) = build_scheimpflug_intrinsics_ir(dataset, initial, &opts)?;
    let solution = solve_with_backend(backend, &ir, &initial_map, &backend_opts)?;

    let intrinsics = unpack_intrinsics(
        solution
            .params
            .get("cam")
            .ok_or_else(|| anyhow!("missing intrinsics solution block"))?
            .as_view(),
    )?;
    let distortion = unpack_distortion(
        solution
            .params
            .get("dist")
            .ok_or_else(|| anyhow!("missing distortion solution block"))?
            .as_view(),
    )?;
    let sensor = unpack_scheimpflug(
        solution
            .params
            .get("sensor")
            .ok_or_else(|| anyhow!("missing sensor solution block"))?
            .as_view(),
    )?;

    let mut optimized_poses = Vec::with_capacity(dataset.num_views());
    for view_idx in 0..dataset.num_views() {
        let key = format!("pose/{view_idx}");
        let pose = solution
            .params
            .get(&key)
            .ok_or_else(|| anyhow!("missing {key} solution block"))?;
        optimized_poses.push(se3_dvec_to_iso3(pose.as_view())?);
    }

    let mean_reproj_error =
        compute_mean_reproj_error(dataset, intrinsics, distortion, sensor, &optimized_poses);

    Ok(ScheimpflugIntrinsicsEstimate {
        params: ScheimpflugIntrinsicsParams {
            intrinsics,
            distortion,
            sensor,
            camera_se3_target: optimized_poses,
        },
        report: solution.solve_report,
        mean_reproj_error,
    })
}

fn unpack_scheimpflug(values: DVectorView<'_, f64>) -> Result<ScheimpflugParams> {
    ensure!(values.len() == 2, "scheimpflug block must have 2 entries");
    Ok(ScheimpflugParams {
        tilt_x: values[0],
        tilt_y: values[1],
    })
}

fn compute_mean_reproj_error(
    dataset: &PlanarDataset,
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    poses: &[Iso3],
) -> f64 {
    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
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

    if count == 0 { 0.0 } else { sum / count as f64 }
}
