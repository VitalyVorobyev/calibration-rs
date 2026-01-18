//! Multi-camera rig + robot hand-eye session problem.

use anyhow::Result;
use calib_core::{CorrespondenceView, Iso3, Real};
use serde::{Deserialize, Serialize};

use crate::session::ProblemType;

/// Multi-camera rig + robot hand-eye calibration problem.
///
/// Calibrates:
/// - per-camera intrinsics + distortion,
/// - camera-to-rig extrinsics,
/// - the rig↔robot hand-eye transform (mode-dependent),
/// - a single fixed target pose (mode-dependent),
/// - optional per-view robot pose corrections (priors).
pub struct RigHandEyeProblem;

/// Observations for rig + hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandEyeObservations {
    pub views: Vec<RigHandEyeViewData>,
    pub num_cameras: usize,
    pub mode: calib_optim::ir::HandEyeMode,
}

/// Single robot view observations from a rig (one or more cameras).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandEyeViewData {
    /// Per-camera observations (None if camera didn't observe target).
    pub cameras: Vec<Option<CorrespondenceView>>,
    /// Robot pose measurement for this view (base_from_gripper, `T_B_G`).
    pub base_from_gripper: Iso3,
}

/// Initial values from rig + hand-eye initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandEyeInitial {
    pub mode: calib_optim::ir::HandEyeMode,
    /// Reference camera index defining the rig frame.
    pub ref_cam_idx: usize,
    /// Per-camera calibrated parameters (K + distortion) from the rig seed stage.
    pub cameras: Vec<calib_core::CameraParams>,
    /// Camera-to-rig transforms.
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view target-to-rig poses from the rig seed stage.
    pub rig_from_target: Vec<Iso3>,
    /// Hand-eye transform (mode-dependent).
    ///
    /// - `EyeInHand`: `handeye = gripper_from_rig` (`T_G_R`)
    /// - `EyeToHand`: `handeye = rig_from_base` (`T_R_B`)
    pub handeye: Iso3,
    /// Target pose values (mode-dependent).
    ///
    /// In fixed-target mode, the optimizer uses only the first pose internally,
    /// but returns one pose per view for convenience.
    ///
    /// - `EyeInHand`: `base_from_target` (`T_B_T`)
    /// - `EyeToHand`: `gripper_from_target` (`T_G_T`)
    pub target_poses: Vec<Iso3>,
}

/// Optimized results from joint rig + hand-eye bundle adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandEyeOptimized {
    pub mode: calib_optim::ir::HandEyeMode,
    pub ref_cam_idx: usize,
    pub cameras: Vec<calib_core::CameraParams>,
    pub cam_to_rig: Vec<Iso3>,
    pub handeye: Iso3,
    pub target_poses: Vec<Iso3>,
    pub robot_deltas: Option<Vec<[Real; 6]>>,
    pub final_cost: f64,
}

/// Options for rig + hand-eye initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandEyeInitOptions {
    /// Rig seed stage init (per-camera intrinsics, rig extrinsics init).
    #[serde(default)]
    pub rig_init: super::RigExtrinsicsInitOptions,
    /// Optional rig seed stage pre-optimization (improves `rig_from_target` before hand-eye DLT).
    #[serde(default)]
    pub rig_preopt: Option<super::RigExtrinsicsOptimOptions>,
    /// Minimum motion angle (degrees) used when building DLT motion pairs.
    #[serde(default = "default_min_angle_deg")]
    pub min_motion_angle_deg: Real,
}

fn default_min_angle_deg() -> Real {
    1.0
}

fn default_rig_preopt() -> super::RigExtrinsicsOptimOptions {
    use calib_optim::ir::RobustLoss;

    let mut opts = super::RigExtrinsicsOptimOptions::default();
    opts.solve_opts.robust_loss = RobustLoss::Huber { scale: 2.0 };
    opts.solve_opts.default_fix = calib_core::CameraFixMask::all_fixed();
    opts.backend_opts.max_iters = 60;
    opts
}

impl Default for RigHandEyeInitOptions {
    fn default() -> Self {
        Self {
            rig_init: Default::default(),
            rig_preopt: Some(default_rig_preopt()),
            min_motion_angle_deg: default_min_angle_deg(),
        }
    }
}

/// Options for rig + hand-eye optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandEyeOptimOptions {
    pub solve_opts: calib_optim::problems::handeye::HandEyeSolveOptions,
    pub backend_opts: calib_optim::backend::BackendSolveOptions,
}

impl Default for RigHandEyeOptimOptions {
    fn default() -> Self {
        use calib_optim::ir::RobustLoss;
        use calib_optim::problems::handeye::HandEyeSolveOptions;

        let solve_opts = HandEyeSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 2.0 },
            default_fix: calib_core::CameraFixMask::all_fixed(),
            refine_robot_poses: true,
            ..Default::default()
        };

        Self {
            solve_opts,
            backend_opts: calib_optim::backend::BackendSolveOptions::default(),
        }
    }
}

impl ProblemType for RigHandEyeProblem {
    type Observations = RigHandEyeObservations;
    type InitialValues = RigHandEyeInitial;
    type OptimizedResults = RigHandEyeOptimized;
    type InitOptions = RigHandEyeInitOptions;
    type OptimOptions = RigHandEyeOptimOptions;

    fn problem_name() -> &'static str {
        "rig_handeye"
    }

    fn initialize(
        obs: &Self::Observations,
        opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        use anyhow::{ensure, Context};

        ensure!(obs.num_cameras > 0, "need at least one camera");
        ensure!(!obs.views.is_empty(), "need at least one view");

        // --- Stage 1: Rig seed (intrinsics + camera-to-rig + per-view rig poses) ---
        let rig_obs = super::RigExtrinsicsObservations {
            views: obs
                .views
                .iter()
                .map(|v| super::RigViewData {
                    cameras: v.cameras.clone(),
                })
                .collect(),
            num_cameras: obs.num_cameras,
        };

        let rig_init = super::RigExtrinsicsProblem::initialize(&rig_obs, &opts.rig_init)
            .context("rig seed initialization failed")?;

        let ref_cam_idx = rig_init.ref_cam_idx;

        let (cameras, cam_to_rig, rig_from_target) = if let Some(rig_preopt) = &opts.rig_preopt {
            let rig_optim = super::RigExtrinsicsProblem::optimize(&rig_obs, &rig_init, rig_preopt)
                .context("rig seed pre-optimization failed")?;
            (
                rig_optim.cameras,
                rig_optim.cam_to_rig,
                rig_optim.rig_from_target,
            )
        } else {
            (
                rig_init.cameras,
                rig_init.cam_to_rig,
                rig_init.rig_from_target,
            )
        };

        ensure!(
            ref_cam_idx < obs.num_cameras,
            "ref_cam_idx {} out of bounds for {} cameras",
            ref_cam_idx,
            obs.num_cameras
        );
        ensure!(
            rig_from_target.len() == obs.views.len(),
            "rig_from_target count {} != num_views {}",
            rig_from_target.len(),
            obs.views.len()
        );

        // --- Stage 2: Hand-eye DLT init from rig poses + robot poses ---
        let robot_poses: Vec<Iso3> = obs.views.iter().map(|v| v.base_from_gripper).collect();
        let (handeye, target_poses) = match obs.mode {
            calib_optim::ir::HandEyeMode::EyeInHand => {
                // Solve for `handeye = gripper_from_rig` from {base_from_gripper, target_from_rig}.
                let target_from_rig: Vec<Iso3> =
                    rig_from_target.iter().map(|rt| rt.inverse()).collect();

                let handeye = calib_linear::handeye::estimate_handeye_dlt(
                    &robot_poses,
                    &target_from_rig,
                    opts.min_motion_angle_deg,
                )
                .context("hand-eye DLT initialization failed")?;

                // Seed `base_from_target` (fixed) by averaging per-view estimates.
                let mut base_from_target_candidates: Vec<Iso3> = robot_poses
                    .iter()
                    .zip(rig_from_target.iter())
                    .map(|(base_from_gripper, rig_from_target)| {
                        base_from_gripper * handeye * rig_from_target
                    })
                    .collect();

                let base_from_target =
                    calib_linear::extrinsics::average_isometries(&base_from_target_candidates)
                        .context("target pose averaging failed")?;
                base_from_target_candidates[0] = base_from_target;

                (handeye, base_from_target_candidates)
            }
            calib_optim::ir::HandEyeMode::EyeToHand => {
                // Eye-to-hand has two unknown constants:
                //   rig_from_base (handeye) and gripper_from_target (target pose).
                // Use a Tsai–Lenz solve for `gripper_from_target`, then average to get `rig_from_base`.

                let mut gripper_from_target = calib_linear::handeye::estimate_handeye_dlt(
                    &robot_poses,
                    &rig_from_target,
                    opts.min_motion_angle_deg,
                )
                .context("eye-to-hand DLT (gripper_from_target) failed")?;

                let mut rig_from_base_candidates: Vec<Iso3> = robot_poses
                    .iter()
                    .zip(rig_from_target.iter())
                    .map(|(base_from_gripper, rig_from_target)| {
                        rig_from_target
                            * gripper_from_target.inverse()
                            * base_from_gripper.inverse()
                    })
                    .collect();
                let mut rig_from_base =
                    calib_linear::extrinsics::average_isometries(&rig_from_base_candidates)
                        .context("rig_from_base averaging failed")?;

                // One refinement pass: recompute `gripper_from_target` using `rig_from_base`.
                let gripper_from_target_candidates: Vec<Iso3> = robot_poses
                    .iter()
                    .zip(rig_from_target.iter())
                    .map(|(base_from_gripper, rig_from_target)| {
                        base_from_gripper.inverse() * rig_from_base.inverse() * rig_from_target
                    })
                    .collect();
                gripper_from_target =
                    calib_linear::extrinsics::average_isometries(&gripper_from_target_candidates)
                        .context("gripper_from_target averaging failed")?;

                // And update `rig_from_base` once more.
                rig_from_base_candidates = robot_poses
                    .iter()
                    .zip(rig_from_target.iter())
                    .map(|(base_from_gripper, rig_from_target)| {
                        rig_from_target
                            * gripper_from_target.inverse()
                            * base_from_gripper.inverse()
                    })
                    .collect();
                rig_from_base =
                    calib_linear::extrinsics::average_isometries(&rig_from_base_candidates)
                        .context("rig_from_base averaging failed")?;

                let target_poses = vec![gripper_from_target; obs.views.len()];
                (rig_from_base, target_poses)
            }
        };

        Ok(RigHandEyeInitial {
            mode: obs.mode,
            ref_cam_idx,
            cameras,
            cam_to_rig,
            rig_from_target,
            handeye,
            target_poses,
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        use anyhow::{ensure, Context};
        use calib_optim::problems::handeye::*;

        ensure!(obs.num_cameras > 0, "need at least one camera");
        ensure!(
            obs.num_cameras == init.cameras.len(),
            "init cameras count {} != num_cameras {}",
            init.cameras.len(),
            obs.num_cameras
        );
        ensure!(
            obs.num_cameras == init.cam_to_rig.len(),
            "init cam_to_rig count {} != num_cameras {}",
            init.cam_to_rig.len(),
            obs.num_cameras
        );
        ensure!(
            init.target_poses.len() == obs.views.len(),
            "init target_poses count {} != num_views {}",
            init.target_poses.len(),
            obs.views.len()
        );

        // Convert observations to calib-optim format.
        let views: Vec<RigViewObservations> = obs
            .views
            .iter()
            .map(|view| {
                let cameras = view.cameras.clone();
                RigViewObservations {
                    cameras,
                    robot_pose: view.base_from_gripper,
                }
            })
            .collect();

        let dataset = HandEyeDataset::new(views, obs.num_cameras, obs.mode)?;

        // Extract initial values.
        let intrinsics = init
            .cameras
            .iter()
            .map(|cam| match cam.intrinsics {
                calib_core::IntrinsicsParams::FxFyCxCySkew { params } => params,
            })
            .collect();

        let distortion = init
            .cameras
            .iter()
            .map(|cam| match cam.distortion {
                calib_core::DistortionParams::BrownConrady5 { params } => params,
                calib_core::DistortionParams::None => calib_core::BrownConrady5 {
                    k1: 0.0,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            })
            .collect();

        let initial = HandEyeInit {
            intrinsics,
            distortion,
            cam_to_rig: init.cam_to_rig.clone(),
            handeye: init.handeye,
            target_poses: init.target_poses.clone(),
        };

        let mut solve_opts = opts.solve_opts.clone();
        if solve_opts.fix_extrinsics.is_empty() {
            solve_opts.fix_extrinsics = vec![false; obs.num_cameras];
        }
        ensure!(
            init.ref_cam_idx < obs.num_cameras,
            "ref_cam_idx {} out of bounds for {} cameras",
            init.ref_cam_idx,
            obs.num_cameras
        );
        ensure!(
            solve_opts.fix_extrinsics.len() == obs.num_cameras,
            "fix_extrinsics length {} != num_cameras {}",
            solve_opts.fix_extrinsics.len(),
            obs.num_cameras
        );
        solve_opts.fix_extrinsics[init.ref_cam_idx] = true;

        let result = calib_optim::problems::handeye::optimize_handeye_with_diagnostics(
            dataset,
            initial,
            solve_opts,
            opts.backend_opts.clone(),
        )
        .context("hand-eye optimization failed")?;

        let cameras: Vec<calib_core::CameraParams> = result
            .result
            .cameras
            .into_iter()
            .map(|cam| calib_core::CameraParams {
                projection: calib_core::ProjectionParams::Pinhole,
                distortion: calib_core::DistortionParams::BrownConrady5 { params: cam.dist },
                sensor: calib_core::SensorParams::Identity,
                intrinsics: calib_core::IntrinsicsParams::FxFyCxCySkew { params: cam.k },
            })
            .collect();

        Ok(RigHandEyeOptimized {
            mode: obs.mode,
            ref_cam_idx: init.ref_cam_idx,
            cameras,
            cam_to_rig: result.result.cam_to_rig,
            handeye: result.result.handeye,
            target_poses: result.result.target_poses,
            robot_deltas: result.robot_deltas,
            final_cost: result.result.final_cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::make_pinhole_camera;
    use crate::session::CalibrationSession;
    use calib_core::{BrownConrady5, FxFyCxCySkew, Iso3, Pt3, Vec2};
    use nalgebra::{UnitQuaternion, Vector3};

    fn pose_error(a: &Iso3, b: &Iso3) -> (Real, Real) {
        let dt = (a.translation.vector - b.translation.vector).norm();

        let r_a = a.rotation.to_rotation_matrix();
        let r_b = b.rotation.to_rotation_matrix();
        let r_diff = r_a.transpose() * r_b;
        let trace = r_diff.matrix().trace();
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let angle = cos_theta.acos();

        (dt, angle)
    }

    fn mean_reproj_error(
        obs: &RigHandEyeObservations,
        cams: &[calib_core::CameraParams],
        cam_to_rig: &[Iso3],
        handeye: &Iso3,
        target_poses: &[Iso3],
    ) -> Result<Real> {
        use calib_core::{Camera, IdentitySensor, Pinhole};

        let mut total_error = 0.0;
        let mut total_n = 0usize;

        for (view_idx, view) in obs.views.iter().enumerate() {
            let base_from_gripper = view.base_from_gripper;
            for (cam_idx, cam_view_opt) in view.cameras.iter().enumerate() {
                let Some(cam_view) = cam_view_opt else {
                    continue;
                };

                let k = match cams[cam_idx].intrinsics {
                    calib_core::IntrinsicsParams::FxFyCxCySkew { params } => params,
                };
                let dist = match cams[cam_idx].distortion {
                    calib_core::DistortionParams::BrownConrady5 { params } => params,
                    calib_core::DistortionParams::None => BrownConrady5 {
                        k1: 0.0,
                        k2: 0.0,
                        k3: 0.0,
                        p1: 0.0,
                        p2: 0.0,
                        iters: 8,
                    },
                };

                let camera = Camera::new(Pinhole, dist, IdentitySensor, k);
                let target_pose = target_poses[view_idx];

                for (pw, uv) in cam_view.points_3d.iter().zip(cam_view.points_2d.iter()) {
                    let p_cam = match obs.mode {
                        calib_optim::ir::HandEyeMode::EyeInHand => {
                            let p_base = target_pose.transform_point(pw);
                            let p_gripper = base_from_gripper.inverse_transform_point(&p_base);
                            let p_rig = handeye.inverse_transform_point(&p_gripper);
                            cam_to_rig[cam_idx].inverse_transform_point(&p_rig)
                        }
                        calib_optim::ir::HandEyeMode::EyeToHand => {
                            let p_gripper = target_pose.transform_point(pw);
                            let p_base = base_from_gripper.transform_point(&p_gripper);
                            let p_rig = handeye.transform_point(&p_base);
                            cam_to_rig[cam_idx].inverse_transform_point(&p_rig)
                        }
                    };

                    let Some(proj) = camera.project_point(&p_cam) else {
                        continue;
                    };
                    total_error += (proj - *uv).norm();
                    total_n += 1;
                }
            }
        }

        anyhow::ensure!(total_n > 0, "no valid projections");
        Ok(total_error / total_n as Real)
    }

    #[test]
    fn rig_handeye_eye_in_hand_synthetic_converges() -> Result<()> {
        // Camera model
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };

        let cam0 = make_pinhole_camera(k_gt, dist_gt);
        let cam1 = make_pinhole_camera(k_gt, dist_gt);

        // Rig extrinsics (camera -> rig)
        let cam0_to_rig = Iso3::identity();
        let cam1_to_rig = Iso3::from_parts(
            Vector3::new(0.12, 0.01, 0.0).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.03, 0.0)),
        );

        // Hand-eye: gripper_from_rig (rig -> gripper)
        let gripper_from_rig = Iso3::from_parts(
            Vector3::new(0.02, -0.01, 0.05).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.02, 0.01, -0.03)),
        );

        // Fixed target in base: base_from_target
        let base_from_target = Iso3::from_parts(
            Vector3::new(0.3, -0.1, 0.8).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.05, 0.02)),
        );

        // Checkerboard points
        let nx = 6;
        let ny = 5;
        let spacing = 0.04_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        // Views
        let mut views = Vec::new();
        for view_idx in 0..6 {
            let base_from_gripper = Iso3::from_parts(
                Vector3::new(
                    0.02 * view_idx as f64 - 0.05,
                    0.015 * view_idx as f64 - 0.03,
                    0.25 + 0.04 * view_idx as f64,
                )
                .into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(
                    0.03 * view_idx as f64 - 0.04,
                    0.05 * view_idx as f64 - 0.02,
                    0.02 * view_idx as f64 - 0.03,
                )),
            );

            let base_from_rig = base_from_gripper * gripper_from_rig;
            let rig_from_target = base_from_rig.inverse() * base_from_target;

            let cam0_from_target = cam0_to_rig.inverse() * rig_from_target;
            let cam1_from_target = cam1_to_rig.inverse() * rig_from_target;

            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();
            for pw in &board_points {
                if let Some(p) = cam0.project_point(&cam0_from_target.transform_point(pw)) {
                    cam0_pixels.push(Vec2::new(p.x, p.y));
                }
                if let Some(p) = cam1.project_point(&cam1_from_target.transform_point(pw)) {
                    cam1_pixels.push(Vec2::new(p.x, p.y));
                }
            }

            views.push(RigHandEyeViewData {
                cameras: vec![
                    Some(CorrespondenceView {
                        points_3d: board_points.clone(),
                        points_2d: cam0_pixels,
                        weights: None,
                    }),
                    Some(CorrespondenceView {
                        points_3d: board_points.clone(),
                        points_2d: cam1_pixels,
                        weights: None,
                    }),
                ],
                base_from_gripper,
            });
        }

        let obs = RigHandEyeObservations {
            views,
            num_cameras: 2,
            mode: calib_optim::ir::HandEyeMode::EyeInHand,
        };

        let mut session = CalibrationSession::<RigHandEyeProblem>::new();
        session.set_observations(obs.clone());

        session.initialize(RigHandEyeInitOptions {
            rig_preopt: None,
            ..Default::default()
        })?;

        let mut optim_opts = RigHandEyeOptimOptions::default();
        optim_opts.solve_opts.refine_robot_poses = false;
        optim_opts.backend_opts.max_iters = 200;
        optim_opts.backend_opts.min_abs_decrease = Some(1e-12);
        optim_opts.backend_opts.min_rel_decrease = Some(1e-12);
        session.optimize(optim_opts)?;
        let report = session.export()?;

        let err = mean_reproj_error(
            &obs,
            &report.cameras,
            &report.cam_to_rig,
            &report.handeye,
            &report.target_poses,
        )?;
        assert!(err < 1e-3, "mean reprojection error too high: {err}");

        let (dt_cam1, ang_cam1) = pose_error(&report.cam_to_rig[1], &cam1_to_rig);
        assert!(dt_cam1 < 1e-3, "cam1 translation error too high: {dt_cam1}");
        assert!(ang_cam1 < 1e-3, "cam1 rotation error too high: {ang_cam1}");

        let (dt_he, ang_he) = pose_error(&report.handeye, &gripper_from_rig);
        assert!(dt_he < 1e-3, "handeye translation error too high: {dt_he}");
        assert!(ang_he < 1e-3, "handeye rotation error too high: {ang_he}");

        let (dt_t, ang_t) = pose_error(&report.target_poses[0], &base_from_target);
        assert!(dt_t < 1e-3, "target translation error too high: {dt_t}");
        assert!(ang_t < 1e-3, "target rotation error too high: {ang_t}");

        Ok(())
    }

    #[test]
    fn rig_handeye_eye_to_hand_synthetic_converges() -> Result<()> {
        // Camera model
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };

        let cam0 = make_pinhole_camera(k_gt, dist_gt);
        let cam1 = make_pinhole_camera(k_gt, dist_gt);

        // Rig extrinsics (camera -> rig)
        let cam0_to_rig = Iso3::identity();
        let cam1_to_rig = Iso3::from_parts(
            Vector3::new(0.18, -0.02, 0.0).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, -0.04, 0.0)),
        );

        // Hand-eye: rig_from_base (base -> rig)
        let rig_from_base = Iso3::from_parts(
            Vector3::new(0.05, 0.02, 0.01).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.01, 0.02, 0.04)),
        );

        // Fixed target on gripper: gripper_from_target
        let gripper_from_target = Iso3::from_parts(
            Vector3::new(0.0, 0.0, 0.4).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.1, 0.0, 0.0)),
        );

        // Checkerboard points
        let nx = 6;
        let ny = 5;
        let spacing = 0.04_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for view_idx in 0..6 {
            let base_from_gripper = Iso3::from_parts(
                Vector3::new(
                    -0.03 + 0.015 * view_idx as f64,
                    0.02 * view_idx as f64 - 0.04,
                    0.3 + 0.03 * view_idx as f64,
                )
                .into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(
                    0.02 * view_idx as f64 - 0.01,
                    0.045 * view_idx as f64 - 0.03,
                    0.015 * view_idx as f64,
                )),
            );

            let rig_from_target = rig_from_base * base_from_gripper * gripper_from_target;

            let cam0_from_target = cam0_to_rig.inverse() * rig_from_target;
            let cam1_from_target = cam1_to_rig.inverse() * rig_from_target;

            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();
            for pw in &board_points {
                if let Some(p) = cam0.project_point(&cam0_from_target.transform_point(pw)) {
                    cam0_pixels.push(Vec2::new(p.x, p.y));
                }
                if let Some(p) = cam1.project_point(&cam1_from_target.transform_point(pw)) {
                    cam1_pixels.push(Vec2::new(p.x, p.y));
                }
            }

            views.push(RigHandEyeViewData {
                cameras: vec![
                    Some(CorrespondenceView {
                        points_3d: board_points.clone(),
                        points_2d: cam0_pixels,
                        weights: None,
                    }),
                    Some(CorrespondenceView {
                        points_3d: board_points.clone(),
                        points_2d: cam1_pixels,
                        weights: None,
                    }),
                ],
                base_from_gripper,
            });
        }

        let obs = RigHandEyeObservations {
            views,
            num_cameras: 2,
            mode: calib_optim::ir::HandEyeMode::EyeToHand,
        };

        let mut session = CalibrationSession::<RigHandEyeProblem>::new();
        session.set_observations(obs.clone());
        session.initialize(RigHandEyeInitOptions::default())?;

        let mut optim_opts = RigHandEyeOptimOptions::default();
        optim_opts.solve_opts.refine_robot_poses = false;
        optim_opts.solve_opts.default_fix = calib_core::CameraFixMask::truly_all_free();
        optim_opts.backend_opts.max_iters = 200;
        optim_opts.backend_opts.min_abs_decrease = Some(1e-12);
        optim_opts.backend_opts.min_rel_decrease = Some(1e-12);
        session.optimize(optim_opts)?;
        let report = session.export()?;

        let err = mean_reproj_error(
            &obs,
            &report.cameras,
            &report.cam_to_rig,
            &report.handeye,
            &report.target_poses,
        )?;
        assert!(err < 1e-3, "mean reprojection error too high: {err}");

        let (dt_cam1, ang_cam1) = pose_error(&report.cam_to_rig[1], &cam1_to_rig);
        assert!(dt_cam1 < 1e-3, "cam1 translation error too high: {dt_cam1}");
        assert!(ang_cam1 < 1e-3, "cam1 rotation error too high: {ang_cam1}");

        let (dt_he, ang_he) = pose_error(&report.handeye, &rig_from_base);
        assert!(dt_he < 1e-3, "handeye translation error too high: {dt_he}");
        assert!(ang_he < 1e-3, "handeye rotation error too high: {ang_he}");

        let (dt_t, ang_t) = pose_error(&report.target_poses[0], &gripper_from_target);
        assert!(dt_t < 1e-3, "target translation error too high: {dt_t}");
        assert!(ang_t < 1e-3, "target rotation error too high: {ang_t}");

        Ok(())
    }

    #[test]
    fn rig_handeye_session_json_roundtrip() {
        let obs = RigHandEyeObservations {
            views: vec![RigHandEyeViewData {
                cameras: vec![Some(CorrespondenceView {
                    points_3d: vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.05, 0.0, 0.0)],
                    points_2d: vec![Vec2::new(100.0, 100.0), Vec2::new(200.0, 100.0)],
                    weights: None,
                })],
                base_from_gripper: Iso3::identity(),
            }],
            num_cameras: 1,
            mode: calib_optim::ir::HandEyeMode::EyeInHand,
        };

        let mut session = CalibrationSession::<RigHandEyeProblem>::new();
        session.set_observations(obs);

        let json = session.to_json().unwrap();
        assert!(json.contains("rig_handeye"));
        assert!(json.contains("Uninitialized"));

        let restored: CalibrationSession<RigHandEyeProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(
            restored.stage(),
            crate::session::SessionStage::Uninitialized
        );
        assert_eq!(restored.observations().unwrap().views.len(), 1);
        assert_eq!(restored.observations().unwrap().num_cameras, 1);
    }
}
