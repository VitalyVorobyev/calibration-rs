//! Multi-camera rig extrinsics calibration with Scheimpflug-tilted sensors.
//!
//! Mirrors [`super::rig_extrinsics`] but adds a per-camera Scheimpflug sensor block
//! (`tilt_x`, `tilt_y`) and routes residuals through the
//! `ReprojPointPinhole4Dist5Scheimpflug2TwoSE3` factor.

use crate::Error;
use crate::backend::{BackendKind, BackendSolveOptions, SolveReport, solve_with_backend};
use crate::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use crate::params::distortion::{DISTORTION_DIM, pack_distortion, unpack_distortion};
use crate::params::intrinsics::{INTRINSICS_DIM, pack_intrinsics, unpack_intrinsics};
use crate::params::pose_se3::iso3_to_se3_dvec;
use crate::problems::scheimpflug_intrinsics::ScheimpflugFixMask;
use anyhow::ensure;
type AnyhowResult<T> = anyhow::Result<T>;
use nalgebra::{DVector, DVectorView};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vision_calibration_core::{
    BrownConrady5, Camera, CameraFixMask, FxFyCxCySkew, Iso3, NoMeta, Pinhole, PinholeCamera, Real,
    RigDataset, ScheimpflugParams, make_pinhole_camera,
};

/// Dataset type for Scheimpflug rig extrinsics optimization.
pub type RigExtrinsicsScheimpflugDataset = RigDataset<NoMeta>;

/// Parameter bundle consumed and produced by
/// [`optimize_rig_extrinsics_scheimpflug`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsScheimpflugParams {
    /// Per-camera calibrated parameters (pinhole + Brown-Conrady distortion).
    pub cameras: Vec<PinholeCamera>,
    /// Per-camera Scheimpflug sensor tilt parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera extrinsics (`camera -> rig`, `T_R_C`).
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view rig poses (`target -> rig`, `T_R_T`).
    pub rig_from_target: Vec<Iso3>,
}

/// Output of Scheimpflug rig extrinsics optimization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RigExtrinsicsScheimpflugEstimate {
    /// Refined parameters.
    pub params: RigExtrinsicsScheimpflugParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Solve options for Scheimpflug rig extrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsScheimpflugSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Default camera fix mask (applied to all cameras).
    pub default_fix: CameraFixMask,
    /// Optional per-camera camera-mask overrides (None = use default_fix).
    pub camera_overrides: Vec<Option<CameraFixMask>>,
    /// Default Scheimpflug fix mask.
    pub default_scheimpflug_fix: ScheimpflugFixMask,
    /// Optional per-camera Scheimpflug mask overrides.
    pub scheimpflug_overrides: Vec<Option<ScheimpflugFixMask>>,
    /// Per-camera extrinsics masking (`true` = fix camera-to-rig transform).
    pub fix_extrinsics: Vec<bool>,
    /// View indices to fix (e.g., first view for gauge freedom).
    pub fix_rig_poses: Vec<usize>,
}

impl Default for RigExtrinsicsScheimpflugSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            default_fix: CameraFixMask::default(),
            camera_overrides: Vec::new(),
            default_scheimpflug_fix: ScheimpflugFixMask::default(),
            scheimpflug_overrides: Vec::new(),
            fix_extrinsics: Vec::new(),
            fix_rig_poses: Vec::new(),
        }
    }
}

fn pack_scheimpflug(sensor: &ScheimpflugParams) -> DVector<f64> {
    DVector::from_row_slice(&[sensor.tilt_x, sensor.tilt_y])
}

fn unpack_scheimpflug(values: DVectorView<'_, f64>) -> AnyhowResult<ScheimpflugParams> {
    ensure!(values.len() == 2, "scheimpflug block must have 2 entries");
    Ok(ScheimpflugParams {
        tilt_x: values[0],
        tilt_y: values[1],
    })
}

fn fix_scheimpflug_mask(mask: ScheimpflugFixMask) -> FixedMask {
    let mut indices = Vec::with_capacity(2);
    if mask.tilt_x {
        indices.push(0);
    }
    if mask.tilt_y {
        indices.push(1);
    }
    if indices.len() == 2 {
        FixedMask::all_fixed(2)
    } else if indices.is_empty() {
        FixedMask::all_free()
    } else {
        FixedMask::fix_indices(&indices)
    }
}

fn build_rig_extrinsics_scheimpflug_ir(
    dataset: &RigExtrinsicsScheimpflugDataset,
    initial: &RigExtrinsicsScheimpflugParams,
    opts: &RigExtrinsicsScheimpflugSolveOptions,
) -> AnyhowResult<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        initial.cameras.len() == dataset.num_cameras,
        "intrinsics count {} != num_cameras {}",
        initial.cameras.len(),
        dataset.num_cameras
    );
    ensure!(
        initial.sensors.len() == dataset.num_cameras,
        "sensors count {} != num_cameras {}",
        initial.sensors.len(),
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
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

    let get_camera_mask = |cam_idx: usize| -> &CameraFixMask {
        opts.camera_overrides
            .get(cam_idx)
            .and_then(|o| o.as_ref())
            .unwrap_or(&opts.default_fix)
    };
    let get_scheimpflug_mask = |cam_idx: usize| -> ScheimpflugFixMask {
        opts.scheimpflug_overrides
            .get(cam_idx)
            .and_then(|o| *o)
            .unwrap_or(opts.default_scheimpflug_fix)
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

        let key = format!("cam/{cam_idx}");
        let cam_id = ir.add_param_block(
            &key,
            INTRINSICS_DIM,
            ManifoldKind::Euclidean,
            fixed_mask,
            None,
        );
        cam_ids.push(cam_id);
        initial_map.insert(key, pack_intrinsics(&initial.cameras[cam_idx].k)?);
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

        let key = format!("dist/{cam_idx}");
        let dist_id = ir.add_param_block(
            &key,
            DISTORTION_DIM,
            ManifoldKind::Euclidean,
            fixed_mask,
            None,
        );
        dist_ids.push(dist_id);
        initial_map.insert(key, pack_distortion(&initial.cameras[cam_idx].dist));
    }

    // 3. Per-camera Scheimpflug sensor blocks
    let mut sensor_ids = Vec::new();
    for cam_idx in 0..dataset.num_cameras {
        let mask = get_scheimpflug_mask(cam_idx);
        let fixed_mask = fix_scheimpflug_mask(mask);
        let key = format!("sensor/{cam_idx}");
        let sensor_id = ir.add_param_block(&key, 2, ManifoldKind::Euclidean, fixed_mask, None);
        sensor_ids.push(sensor_id);
        initial_map.insert(key, pack_scheimpflug(&initial.sensors[cam_idx]));
    }

    // 4. Per-camera extrinsics
    let mut extr_ids = Vec::new();
    for cam_idx in 0..dataset.num_cameras {
        let fixed = if opts.fix_extrinsics.get(cam_idx).copied().unwrap_or(false) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("extr/{cam_idx}");
        let id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        extr_ids.push(id);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.cam_to_rig[cam_idx]));
    }

    // 5. Per-view rig poses + residuals
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let fixed = if opts.fix_rig_poses.contains(&view_idx) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("rig_pose/{view_idx}");
        let rig_pose_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        initial_map.insert(key, iso3_to_se3_dvec(&initial.rig_from_target[view_idx]));

        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            if let Some(obs) = cam_obs {
                for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                    let residual = ResidualBlock {
                        params: vec![
                            cam_ids[cam_idx],
                            dist_ids[cam_idx],
                            sensor_ids[cam_idx],
                            extr_ids[cam_idx],
                            rig_pose_id,
                        ],
                        loss: opts.robust_loss,
                        factor: FactorKind::ReprojPointPinhole4Dist5Scheimpflug2TwoSE3 {
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

/// Compute per-camera and overall reprojection error against a Scheimpflug-tilted camera chain.
fn compute_rig_scheimpflug_reprojection(
    dataset: &RigExtrinsicsScheimpflugDataset,
    cameras: &[PinholeCamera],
    sensors: &[ScheimpflugParams],
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> (Vec<f64>, f64) {
    let mut per_cam_sums = vec![0.0_f64; dataset.num_cameras];
    let mut per_cam_counts = vec![0_usize; dataset.num_cameras];

    let cam_models: Vec<Camera<Real, Pinhole, BrownConrady5<Real>, _, FxFyCxCySkew<Real>>> =
        cameras
            .iter()
            .zip(sensors.iter())
            .map(|(cam, sensor)| Camera::new(Pinhole, cam.dist, sensor.compile(), cam.k))
            .collect();

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let rig_from_target = rig_se3_target[view_idx];
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let cam_from_rig = cam_se3_rig[cam_idx];
            let cam_from_target = cam_from_rig * rig_from_target;
            for (p3d, p2d) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_cam = cam_from_target.transform_point(p3d);
                let Some(projected) = cam_models[cam_idx].project_point(&p_cam) else {
                    continue;
                };
                let err = (projected - *p2d).norm();
                if err.is_finite() {
                    per_cam_sums[cam_idx] += err;
                    per_cam_counts[cam_idx] += 1;
                }
            }
        }
    }

    let per_cam: Vec<f64> = per_cam_sums
        .iter()
        .zip(per_cam_counts.iter())
        .map(|(s, &c)| if c == 0 { 0.0 } else { s / c as f64 })
        .collect();
    let total_sum: f64 = per_cam_sums.iter().sum();
    let total_count: usize = per_cam_counts.iter().sum();
    let mean = if total_count == 0 {
        0.0
    } else {
        total_sum / total_count as f64
    };
    (per_cam, mean)
}

/// Optimize Scheimpflug rig extrinsics via the default tiny-solver backend.
///
/// # Errors
///
/// Returns [`Error`] if IR construction or the solver backend fails.
pub fn optimize_rig_extrinsics_scheimpflug(
    dataset: RigExtrinsicsScheimpflugDataset,
    initial: RigExtrinsicsScheimpflugParams,
    opts: RigExtrinsicsScheimpflugSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<RigExtrinsicsScheimpflugEstimate, Error> {
    let (ir, initial_map) = build_rig_extrinsics_scheimpflug_ir(&dataset, &initial, &opts)?;
    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, &backend_opts)?;

    let cameras = (0..dataset.num_cameras)
        .map(|cam_idx| {
            let intrinsics = unpack_intrinsics(
                solution
                    .params
                    .get(&format!("cam/{cam_idx}"))
                    .unwrap()
                    .as_view(),
            )?;
            let distortion = unpack_distortion(
                solution
                    .params
                    .get(&format!("dist/{cam_idx}"))
                    .unwrap()
                    .as_view(),
            )?;
            Ok(make_pinhole_camera(intrinsics, distortion))
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let sensors = (0..dataset.num_cameras)
        .map(|cam_idx| {
            let view = solution
                .params
                .get(&format!("sensor/{cam_idx}"))
                .unwrap()
                .as_view();
            unpack_scheimpflug(view).map_err(Error::from)
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let cam_to_rig = (0..dataset.num_cameras)
        .map(|i| {
            let key = format!("extr/{i}");
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let rig_from_target = (0..dataset.num_views())
        .map(|i| {
            let key = format!("rig_pose/{i}");
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let cam_se3_rig: Vec<Iso3> = cam_to_rig.iter().map(|t| t.inverse()).collect();
    let (per_cam_reproj_errors, mean_reproj_error) = compute_rig_scheimpflug_reprojection(
        &dataset,
        &cameras,
        &sensors,
        &cam_se3_rig,
        &rig_from_target,
    );

    Ok(RigExtrinsicsScheimpflugEstimate {
        params: RigExtrinsicsScheimpflugParams {
            cameras,
            sensors,
            cam_to_rig,
            rig_from_target,
        },
        report: solution.solve_report,
        mean_reproj_error,
        per_cam_reproj_errors,
    })
}
