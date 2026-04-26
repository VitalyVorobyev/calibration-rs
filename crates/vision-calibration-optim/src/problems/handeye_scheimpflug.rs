//! Hand-eye calibration for robot-mounted cameras with Scheimpflug-tilted sensors.
//!
//! Mirrors [`super::handeye`] but adds a per-camera Scheimpflug sensor block
//! (`tilt_x`, `tilt_y`) and routes residuals through
//! `ReprojPointPinhole4Dist5Scheimpflug2HandEye(RobotDelta)` factors.
//!
//! Only EyeInHand / EyeToHand parity is maintained at the factor level; the
//! default behaviour mirrors [`super::handeye`] with the added sensor degree
//! of freedom.

use crate::Error;
use crate::backend::{BackendKind, BackendSolveOptions, SolveReport, solve_with_backend};
use crate::ir::{
    FactorKind, FixedMask, HandEyeMode, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss,
};
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
    BrownConrady5, Camera, CameraFixMask, FxFyCxCySkew, Iso3, Pinhole, PinholeCamera, Real,
    RigDataset, RigView, ScheimpflugParams, make_pinhole_camera,
};

use super::handeye::RobotPoseMeta;

/// Dataset for Scheimpflug hand-eye optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeScheimpflugDataset {
    /// Multi-camera observations with robot metadata.
    pub data: RigDataset<RobotPoseMeta>,
    /// Hand-eye mode.
    pub mode: HandEyeMode,
}

impl HandEyeScheimpflugDataset {
    /// Create a dataset from views.
    pub fn new(
        views: Vec<RigView<RobotPoseMeta>>,
        num_cameras: usize,
        mode: HandEyeMode,
    ) -> AnyhowResult<Self> {
        ensure!(!views.is_empty(), "need at least one view");
        for (idx, view) in views.iter().enumerate() {
            ensure!(
                view.obs.cameras.len() == num_cameras,
                "view {idx} has {} cameras, expected {num_cameras}",
                view.obs.cameras.len()
            );
        }
        Ok(Self {
            data: RigDataset { views, num_cameras },
            mode,
        })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.data.views.len()
    }
}

/// Initial / refined parameters for Scheimpflug hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeScheimpflugParams {
    /// Per-camera intrinsics and distortion.
    pub cameras: Vec<PinholeCamera>,
    /// Per-camera Scheimpflug sensor tilt parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera extrinsics (camera-to-rig, `T_R_C`).
    pub cam_to_rig: Vec<Iso3>,
    /// Hand-eye transform.
    ///
    /// - `EyeInHand`: `gripper_from_rig` (`T_G_R`)
    /// - `EyeToHand`: `rig_from_base` (`T_R_B`)
    pub handeye: Iso3,
    /// Calibration target poses. Either one pose (fixed-target mode) or one
    /// per view (legacy `relax_target_poses`).
    pub target_poses: Vec<Iso3>,
}

/// Solve options for Scheimpflug hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeScheimpflugSolveOptions {
    /// Robust loss applied to reprojection residuals.
    pub robust_loss: RobustLoss,
    /// Default mask for fixing camera parameters.
    pub default_fix: CameraFixMask,
    /// Optional per-camera camera-mask overrides.
    pub camera_overrides: Vec<Option<CameraFixMask>>,
    /// Default Scheimpflug fix mask.
    pub default_scheimpflug_fix: ScheimpflugFixMask,
    /// Optional per-camera Scheimpflug mask overrides.
    pub scheimpflug_overrides: Vec<Option<ScheimpflugFixMask>>,
    /// Per-camera extrinsics masking.
    pub fix_extrinsics: Vec<bool>,
    /// Fix hand-eye transform.
    pub fix_handeye: bool,
    /// View indices to fix (legacy per-view target mode only).
    pub fix_target_poses: Vec<usize>,
    /// Legacy mode: relax per-view target poses.
    pub relax_target_poses: bool,
    /// Refine robot poses with per-view se(3) corrections.
    pub refine_robot_poses: bool,
    /// Robot rotation prior sigma (radians).
    pub robot_rot_sigma: Real,
    /// Robot translation prior sigma (meters).
    pub robot_trans_sigma: Real,
}

impl Default for HandEyeScheimpflugSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            default_fix: CameraFixMask::default(),
            camera_overrides: Vec::new(),
            default_scheimpflug_fix: ScheimpflugFixMask::default(),
            scheimpflug_overrides: Vec::new(),
            fix_extrinsics: Vec::new(),
            fix_handeye: false,
            fix_target_poses: Vec::new(),
            relax_target_poses: false,
            refine_robot_poses: false,
            robot_rot_sigma: std::f64::consts::PI / 360.0,
            robot_trans_sigma: 1.0e-3,
        }
    }
}

/// Result of Scheimpflug hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeScheimpflugEstimate {
    /// Refined parameters.
    pub params: HandEyeScheimpflugParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Optional per-view robot pose deltas.
    pub robot_deltas: Option<Vec<[Real; 6]>>,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors.
    pub per_cam_reproj_errors: Vec<f64>,
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

/// Optimize Scheimpflug hand-eye calibration.
///
/// # Errors
///
/// Returns [`Error`] if IR construction or the solver fails.
pub fn optimize_handeye_scheimpflug(
    dataset: HandEyeScheimpflugDataset,
    initial: HandEyeScheimpflugParams,
    opts: HandEyeScheimpflugSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<HandEyeScheimpflugEstimate, Error> {
    let (ir, initial_map) = build_handeye_scheimpflug_ir(&dataset, &initial, &opts)?;
    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, &backend_opts)?;

    let cameras = (0..dataset.data.num_cameras)
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

    let sensors = (0..dataset.data.num_cameras)
        .map(|cam_idx| {
            let view = solution
                .params
                .get(&format!("sensor/{cam_idx}"))
                .unwrap()
                .as_view();
            unpack_scheimpflug(view).map_err(Error::from)
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let cam_to_rig = (0..dataset.data.num_cameras)
        .map(|i| {
            let key = format!("extr/{i}");
            crate::params::pose_se3::se3_dvec_to_iso3(solution.params.get(&key).unwrap().as_view())
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let handeye = crate::params::pose_se3::se3_dvec_to_iso3(
        solution.params.get("handeye").unwrap().as_view(),
    )?;

    let target_poses = if opts.relax_target_poses {
        (0..dataset.num_views())
            .map(|i| {
                let key = format!("target/{i}");
                crate::params::pose_se3::se3_dvec_to_iso3(
                    solution.params.get(&key).unwrap().as_view(),
                )
            })
            .collect::<Result<Vec<_>, Error>>()?
    } else {
        let target_pose = crate::params::pose_se3::se3_dvec_to_iso3(
            solution.params.get("target").unwrap().as_view(),
        )?;
        vec![target_pose; dataset.num_views()]
    };

    let robot_deltas = if opts.refine_robot_poses {
        let mut deltas = Vec::with_capacity(dataset.num_views());
        for i in 0..dataset.num_views() {
            let key = format!("robot_delta/{i}");
            let delta_vec = solution.params.get(&key).unwrap();
            deltas.push([
                delta_vec[0],
                delta_vec[1],
                delta_vec[2],
                delta_vec[3],
                delta_vec[4],
                delta_vec[5],
            ]);
        }
        Some(deltas)
    } else {
        None
    };

    let params = HandEyeScheimpflugParams {
        cameras,
        sensors,
        cam_to_rig,
        handeye,
        target_poses,
    };
    let (mean_reproj_error, per_cam_reproj_errors) =
        compute_reproj_error(&dataset, &params, robot_deltas.as_ref());

    Ok(HandEyeScheimpflugEstimate {
        params,
        report: solution.solve_report,
        robot_deltas,
        mean_reproj_error,
        per_cam_reproj_errors,
    })
}

fn compute_reproj_error(
    dataset: &HandEyeScheimpflugDataset,
    params: &HandEyeScheimpflugParams,
    robot_deltas: Option<&Vec<[Real; 6]>>,
) -> (f64, Vec<f64>) {
    use nalgebra::{UnitQuaternion, Vector3};

    let num_cameras = dataset.data.num_cameras;
    let mut per_cam_sum = vec![0.0f64; num_cameras];
    let mut per_cam_count = vec![0usize; num_cameras];

    let cam_models: Vec<Camera<Real, Pinhole, BrownConrady5<Real>, _, FxFyCxCySkew<Real>>> = params
        .cameras
        .iter()
        .zip(params.sensors.iter())
        .map(|(cam, sensor)| Camera::new(Pinhole, cam.dist, sensor.compile(), cam.k))
        .collect();

    let cam_se3_rig: Vec<Iso3> = params.cam_to_rig.iter().map(|t| t.inverse()).collect();
    let handeye_inv = params.handeye.inverse();
    let target_pose = params
        .target_poses
        .first()
        .cloned()
        .unwrap_or(Iso3::identity());

    for (view_idx, view) in dataset.data.views.iter().enumerate() {
        let robot_pose = view.meta.base_se3_gripper;
        let robot_pose = if let Some(deltas) = robot_deltas {
            let delta = &deltas[view_idx];
            let rot_vec = Vector3::new(delta[0], delta[1], delta[2]);
            let trans_vec = Vector3::new(delta[3], delta[4], delta[5]);
            let angle = rot_vec.norm();
            let delta_rot = if angle > 1e-12 {
                UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(rot_vec), angle)
            } else {
                UnitQuaternion::identity()
            };
            let delta_iso = Iso3::from_parts(nalgebra::Translation3::from(trans_vec), delta_rot);
            delta_iso * robot_pose
        } else {
            robot_pose
        };

        // Use per-view target pose if in relax mode.
        let view_target = if params.target_poses.len() == dataset.num_views() {
            params.target_poses[view_idx]
        } else {
            target_pose
        };

        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let camera = &cam_models[cam_idx];

            for (pt3, pt2) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_camera = match dataset.mode {
                    HandEyeMode::EyeInHand => {
                        let p_base = view_target.transform_point(pt3);
                        let p_gripper = robot_pose.inverse_transform_point(&p_base);
                        let p_rig = handeye_inv.transform_point(&p_gripper);
                        cam_se3_rig[cam_idx].transform_point(&p_rig)
                    }
                    HandEyeMode::EyeToHand => {
                        let p_gripper = view_target.transform_point(pt3);
                        let p_base = robot_pose.transform_point(&p_gripper);
                        let p_rig = params.handeye.transform_point(&p_base);
                        cam_se3_rig[cam_idx].transform_point(&p_rig)
                    }
                };

                if let Some(proj) = camera.project_point(&p_camera) {
                    let err = (proj - *pt2).norm();
                    if err.is_finite() {
                        per_cam_sum[cam_idx] += err;
                        per_cam_count[cam_idx] += 1;
                    }
                }
            }
        }
    }

    let per_cam: Vec<f64> = per_cam_sum
        .iter()
        .zip(per_cam_count.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    let total_sum: f64 = per_cam_sum.iter().sum();
    let total_count: usize = per_cam_count.iter().sum();
    let mean = if total_count > 0 {
        total_sum / total_count as f64
    } else {
        0.0
    };
    (mean, per_cam)
}

fn build_handeye_scheimpflug_ir(
    dataset: &HandEyeScheimpflugDataset,
    initial: &HandEyeScheimpflugParams,
    opts: &HandEyeScheimpflugSolveOptions,
) -> AnyhowResult<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        initial.cameras.len() == dataset.data.num_cameras,
        "cameras count {} != num_cameras {}",
        initial.cameras.len(),
        dataset.data.num_cameras
    );
    ensure!(
        initial.sensors.len() == dataset.data.num_cameras,
        "sensors count {} != num_cameras {}",
        initial.sensors.len(),
        dataset.data.num_cameras
    );
    ensure!(
        initial.cam_to_rig.len() == dataset.data.num_cameras,
        "cam_to_rig count {} != num_cameras {}",
        initial.cam_to_rig.len(),
        dataset.data.num_cameras
    );
    ensure!(
        !initial.target_poses.is_empty(),
        "target_poses must contain at least one pose"
    );
    if opts.relax_target_poses {
        ensure!(
            initial.target_poses.len() == dataset.num_views(),
            "target_poses count {} != num_views {}",
            initial.target_poses.len(),
            dataset.num_views()
        );
    }
    ensure!(
        opts.relax_target_poses || opts.fix_target_poses.is_empty(),
        "fix_target_poses requires relax_target_poses"
    );
    if opts.refine_robot_poses {
        ensure!(
            opts.robot_rot_sigma > 0.0,
            "robot_rot_sigma must be positive"
        );
        ensure!(
            opts.robot_trans_sigma > 0.0,
            "robot_trans_sigma must be positive"
        );
    }

    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

    let get_camera_mask = |cam_idx: usize| -> CameraFixMask {
        opts.camera_overrides
            .get(cam_idx)
            .copied()
            .flatten()
            .unwrap_or(opts.default_fix)
    };
    let get_scheimpflug_mask = |cam_idx: usize| -> ScheimpflugFixMask {
        opts.scheimpflug_overrides
            .get(cam_idx)
            .and_then(|o| *o)
            .unwrap_or(opts.default_scheimpflug_fix)
    };

    // 1. Per-camera intrinsics
    let mut cam_ids = Vec::new();
    for cam_idx in 0..dataset.data.num_cameras {
        let cam_fix = get_camera_mask(cam_idx);
        let fixed_mask = FixedMask::fix_indices(&cam_fix.intrinsics.to_indices());
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

    // 2. Per-camera distortion
    let mut dist_ids = Vec::new();
    for cam_idx in 0..dataset.data.num_cameras {
        let cam_fix = get_camera_mask(cam_idx);
        let fixed_mask = FixedMask::fix_indices(&cam_fix.distortion.to_indices());
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

    // 3. Per-camera Scheimpflug sensor
    let mut sensor_ids = Vec::new();
    for cam_idx in 0..dataset.data.num_cameras {
        let mask = fix_scheimpflug_mask(get_scheimpflug_mask(cam_idx));
        let key = format!("sensor/{cam_idx}");
        let id = ir.add_param_block(&key, 2, ManifoldKind::Euclidean, mask, None);
        sensor_ids.push(id);
        initial_map.insert(key, pack_scheimpflug(&initial.sensors[cam_idx]));
    }

    // 4. Per-camera extrinsics
    let mut extr_ids = Vec::new();
    for cam_idx in 0..dataset.data.num_cameras {
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

    // 5. Hand-eye
    let handeye_fixed = if opts.fix_handeye {
        FixedMask::all_fixed(7)
    } else {
        FixedMask::all_free()
    };
    let handeye_id = ir.add_param_block("handeye", 7, ManifoldKind::SE3, handeye_fixed, None);
    initial_map.insert("handeye".to_string(), iso3_to_se3_dvec(&initial.handeye));

    // 6. Target pose(s)
    let target_id = if opts.relax_target_poses {
        None
    } else {
        let id = ir.add_param_block("target", 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial_map.insert(
            "target".to_string(),
            iso3_to_se3_dvec(&initial.target_poses[0]),
        );
        Some(id)
    };

    let robot_prior_sqrt_info = if opts.refine_robot_poses {
        let rot = 1.0 / opts.robot_rot_sigma;
        let trans = 1.0 / opts.robot_trans_sigma;
        [rot, rot, rot, trans, trans, trans]
    } else {
        [0.0; 6]
    };

    for (view_idx, view) in dataset.data.views.iter().enumerate() {
        let target_id = if opts.relax_target_poses {
            let fixed = if opts.fix_target_poses.contains(&view_idx) {
                FixedMask::all_fixed(7)
            } else {
                FixedMask::all_free()
            };
            let key = format!("target/{view_idx}");
            let id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
            initial_map.insert(key, iso3_to_se3_dvec(&initial.target_poses[view_idx]));
            id
        } else {
            target_id.expect("target id set for fixed-target mode")
        };

        let robot_delta_id = if opts.refine_robot_poses {
            let fixed = if view_idx == 0 {
                FixedMask::all_fixed(6)
            } else {
                FixedMask::all_free()
            };
            let key = format!("robot_delta/{view_idx}");
            let id = ir.add_param_block(&key, 6, ManifoldKind::Euclidean, fixed, None);
            initial_map.insert(key, DVector::from_element(6, 0.0));
            let prior = ResidualBlock {
                params: vec![id],
                loss: RobustLoss::None,
                factor: FactorKind::Se3TangentPrior {
                    sqrt_info: robot_prior_sqrt_info,
                },
                residual_dim: 6,
            };
            ir.add_residual_block(prior);
            Some(id)
        } else {
            None
        };

        let robot_se3 = iso3_to_se3_dvec(&view.meta.base_se3_gripper);
        let robot_se3_array: [f64; 7] = [
            robot_se3[0],
            robot_se3[1],
            robot_se3[2],
            robot_se3[3],
            robot_se3[4],
            robot_se3[5],
            robot_se3[6],
        ];

        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            if let Some(obs) = cam_obs {
                for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                    let residual = if let Some(robot_delta_id) = robot_delta_id {
                        ResidualBlock {
                            params: vec![
                                cam_ids[cam_idx],
                                dist_ids[cam_idx],
                                sensor_ids[cam_idx],
                                extr_ids[cam_idx],
                                handeye_id,
                                target_id,
                                robot_delta_id,
                            ],
                            loss: opts.robust_loss,
                            factor:
                                FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta {
                                    pw: [pw.x, pw.y, pw.z],
                                    uv: [uv.x, uv.y],
                                    w: obs.weight(pt_idx),
                                    base_to_gripper_se3: robot_se3_array,
                                    mode: dataset.mode,
                                },
                            residual_dim: 2,
                        }
                    } else {
                        ResidualBlock {
                            params: vec![
                                cam_ids[cam_idx],
                                dist_ids[cam_idx],
                                sensor_ids[cam_idx],
                                extr_ids[cam_idx],
                                handeye_id,
                                target_id,
                            ],
                            loss: opts.robust_loss,
                            factor: FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEye {
                                pw: [pw.x, pw.y, pw.z],
                                uv: [uv.x, uv.y],
                                w: obs.weight(pt_idx),
                                base_to_gripper_se3: robot_se3_array,
                                mode: dataset.mode,
                            },
                            residual_dim: 2,
                        }
                    };
                    ir.add_residual_block(residual);
                }
            }
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}
