//! Multi-camera rig extrinsics session problem.

use anyhow::Result;
use calib_core::{BrownConrady5, FxFyCxCySkew, Iso3, Pt2, Real, Vec2};
use serde::{Deserialize, Serialize};

use crate::session::ProblemType;

/// Multi-camera rig calibration problem.
///
/// Calibrates separate intrinsics and distortion for each camera in a rig,
/// along with camera-to-rig extrinsics and per-view rig poses.
pub struct RigExtrinsicsProblem;

/// Observations for multi-camera rig calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsObservations {
    pub views: Vec<RigViewData>,
    pub num_cameras: usize,
}

/// Single view observations from a rig (one or more cameras).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigViewData {
    /// Per-camera observations (None if camera didn't observe target).
    pub cameras: Vec<Option<CameraViewData>>,
}

pub use crate::CameraViewData;

/// Initial values from rig extrinsics initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsInitial {
    /// Per-camera calibrated parameters (K + distortion).
    pub cameras: Vec<calib_core::CameraParams>,
    /// Reference camera index defining the rig frame.
    pub ref_cam_idx: usize,
    /// Camera-to-rig transforms.
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view target-to-rig poses.
    pub rig_from_target: Vec<Iso3>,
}

/// Optimized results from rig extrinsics bundle adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsOptimized {
    /// Per-camera calibrated parameters.
    pub cameras: Vec<calib_core::CameraParams>,
    /// Camera-to-rig transforms.
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view target-to-rig poses.
    pub rig_from_target: Vec<Iso3>,
    /// Final optimization cost.
    pub final_cost: f64,
}

/// Options for rig extrinsics initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsInitOptions {
    /// Reference camera index (defines rig frame). Default: 0.
    #[serde(default)]
    pub ref_cam_idx: usize,
    /// Per-camera intrinsics initialization options (iterative Zhang + distortion).
    #[serde(default)]
    pub intrinsics_init_opts: calib_linear::iterative_intrinsics::IterativeIntrinsicsOptions,
    /// Optional per-camera non-linear planar intrinsics refinement before rig initialization.
    ///
    /// When enabled, runs a single-camera planar intrinsics optimization for each camera using
    /// the views where that camera observed the target. This typically improves the camera poses
    /// used for rig extrinsics initialization and helps avoid local minima in the subsequent rig BA.
    #[serde(default)]
    pub intrinsics_refine: Option<crate::PlanarIntrinsicsConfig>,
}

impl Default for RigExtrinsicsInitOptions {
    fn default() -> Self {
        Self {
            ref_cam_idx: 0,
            intrinsics_init_opts: Default::default(),
            intrinsics_refine: Some(Default::default()),
        }
    }
}

/// Options for rig extrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsOptimOptions {
    pub solve_opts: calib_optim::problems::rig_extrinsics::RigExtrinsicsSolveOptions,
    pub backend_opts: calib_optim::backend::BackendSolveOptions,
}

impl Default for RigExtrinsicsOptimOptions {
    fn default() -> Self {
        Self {
            // Gauge is fixed by forcing `fix_extrinsics[ref_cam_idx] = true` during optimization.
            solve_opts: Default::default(),
            backend_opts: calib_optim::backend::BackendSolveOptions {
                // Rig BA often has a large number of residuals; use stricter termination
                // thresholds so noise-free synthetic problems converge close to zero.
                min_abs_decrease: Some(1e-12),
                min_rel_decrease: Some(1e-12),
                ..Default::default()
            },
        }
    }
}

impl ProblemType for RigExtrinsicsProblem {
    type Observations = RigExtrinsicsObservations;
    type InitialValues = RigExtrinsicsInitial;
    type OptimizedResults = RigExtrinsicsOptimized;
    type InitOptions = RigExtrinsicsInitOptions;
    type OptimOptions = RigExtrinsicsOptimOptions;

    fn problem_name() -> &'static str {
        "rig_extrinsics"
    }

    fn initialize(
        obs: &Self::Observations,
        opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        use anyhow::{bail, Context};
        use calib_core::{DistortionModel, IntrinsicsParams, Mat3, Vec3};
        use calib_linear::iterative_intrinsics::{IterativeCalibView, IterativeIntrinsicsSolver};
        use calib_linear::{homography, planar_pose};
        use calib_optim::planar_intrinsics::PlanarIntrinsicsInit;

        let num_cameras = obs.num_cameras;
        let num_views = obs.views.len();

        anyhow::ensure!(num_cameras > 0, "need at least one camera");
        anyhow::ensure!(num_views > 0, "need at least one view");

        // Step 1: Per-camera planar intrinsics calibration
        let mut cameras = Vec::with_capacity(num_cameras);
        let mut cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];
        let mut cam_views: Vec<Vec<(usize, &CameraViewData)>> = vec![Vec::new(); num_cameras];

        for (view_idx, view) in obs.views.iter().enumerate() {
            anyhow::ensure!(
                view.cameras.len() == num_cameras,
                "view {} has {} cameras, expected {}",
                view_idx,
                view.cameras.len(),
                num_cameras
            );
            for (cam_idx, cam_opt) in view.cameras.iter().enumerate() {
                if let Some(cam_data) = cam_opt {
                    cam_views[cam_idx].push((view_idx, cam_data));
                }
            }
        }

        for (cam_idx, cam_views) in cam_views.iter().enumerate() {
            if cam_views.is_empty() {
                bail!("camera {} has no observations", cam_idx);
            }

            anyhow::ensure!(
                cam_views.len() >= 3,
                "camera {} needs at least 3 views for intrinsics initialization (got {})",
                cam_idx,
                cam_views.len()
            );

            // 1a) Per-camera iterative intrinsics + distortion initialization.
            let mut calib_views = Vec::with_capacity(cam_views.len());
            for (_view_idx, cam_data) in cam_views {
                anyhow::ensure!(
                    cam_data.points_3d.len() == cam_data.points_2d.len(),
                    "camera {} has mismatched 3D/2D point count",
                    cam_idx
                );
                anyhow::ensure!(
                    cam_data.points_3d.len() >= 4,
                    "camera {} needs at least 4 points per view",
                    cam_idx
                );
                if let Some(weights) = cam_data.weights.as_ref() {
                    anyhow::ensure!(
                        weights.len() == cam_data.points_3d.len(),
                        "camera {} weights must match point count",
                        cam_idx
                    );
                    anyhow::ensure!(
                        weights.iter().all(|w| *w >= 0.0),
                        "camera {} has negative weights",
                        cam_idx
                    );
                }
                let board_2d: Vec<Pt2> = cam_data
                    .points_3d
                    .iter()
                    .map(|p| Pt2::new(p.x, p.y))
                    .collect();
                let pixel_2d: Vec<Pt2> = cam_data
                    .points_2d
                    .iter()
                    .map(|v| Pt2::new(v.x, v.y))
                    .collect();
                calib_views.push(IterativeCalibView::new(board_2d, pixel_2d));
            }

            let intr_res =
                IterativeIntrinsicsSolver::estimate(&calib_views, opts.intrinsics_init_opts)
                    .with_context(|| {
                        format!("iterative intrinsics failed for camera {}", cam_idx)
                    })?;

            let mut intrinsics = intr_res.intrinsics;
            intrinsics.skew = 0.0;
            let distortion = intr_res.distortion;

            let k = Mat3::new(
                intrinsics.fx,
                intrinsics.skew,
                intrinsics.cx,
                0.0,
                intrinsics.fy,
                intrinsics.cy,
                0.0,
                0.0,
                1.0,
            );

            let cam_cfg = calib_core::CameraParams {
                projection: calib_core::ProjectionParams::Pinhole,
                distortion: calib_core::DistortionParams::BrownConrady5 { params: distortion },
                sensor: calib_core::SensorParams::Identity,
                intrinsics: IntrinsicsParams::FxFyCxCySkew { params: intrinsics },
            };

            // Helper: undistort pixels into an ideal pinhole image for homography/pose recovery.
            fn undistort_pixels(
                pixels: &[Vec2],
                kmtx: &Mat3,
                distortion: &BrownConrady5<Real>,
            ) -> Result<Vec<Pt2>> {
                let k_inv = kmtx
                    .try_inverse()
                    .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

                let mut undistorted = Vec::with_capacity(pixels.len());
                for p in pixels {
                    let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
                    let n_dist = Vec2::new(v_h.x / v_h.z, v_h.y / v_h.z);
                    let n_undist = distortion.undistort(&n_dist);
                    let p_h = kmtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
                    undistorted.push(Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z));
                }

                Ok(undistorted)
            }

            // 1b) Per-view pose seeds from undistorted homographies.
            let mut view_indices = Vec::with_capacity(cam_views.len());
            let mut pose_seeds = Vec::with_capacity(cam_views.len());
            for (view_idx, cam_data) in cam_views {
                view_indices.push(*view_idx);

                let board_2d: Vec<Pt2> = cam_data
                    .points_3d
                    .iter()
                    .map(|p| Pt2::new(p.x, p.y))
                    .collect();

                let undistorted_pixels = undistort_pixels(&cam_data.points_2d, &k, &distortion)
                    .with_context(|| {
                        format!(
                            "pixel undistortion failed for view {} camera {}",
                            view_idx, cam_idx
                        )
                    })?;

                let h = homography::dlt_homography(&board_2d, &undistorted_pixels).with_context(
                    || format!("view {} homography failed for camera {}", view_idx, cam_idx),
                )?;

                let cam_from_target = planar_pose::estimate_planar_pose_from_h(&k, &h)
                    .with_context(|| {
                        format!(
                            "pose recovery failed for view {} camera {}",
                            view_idx, cam_idx
                        )
                    })?;
                pose_seeds.push(cam_from_target);
            }

            // Optional: per-camera planar intrinsics refinement.
            if let Some(refine_cfg) = opts.intrinsics_refine.as_ref() {
                let planar_input = crate::PlanarIntrinsicsInput {
                    views: cam_views.iter().map(|(_, v)| (*v).clone()).collect(),
                };
                let dataset = crate::build_planar_dataset(&planar_input).with_context(|| {
                    format!(
                        "failed to build planar dataset for intrinsics refinement (camera {})",
                        cam_idx
                    )
                })?;
                let init = PlanarIntrinsicsInit::new(intrinsics, distortion, pose_seeds)
                    .with_context(|| {
                        format!(
                            "invalid planar init for intrinsics refinement (camera {})",
                            cam_idx
                        )
                    })?;
                let result = crate::optimize_planar_intrinsics_with_init(dataset, init, refine_cfg)
                    .with_context(|| {
                        format!("planar intrinsics refinement failed for camera {}", cam_idx)
                    })?;

                cameras.push(crate::pinhole_camera_params(&result.camera));

                anyhow::ensure!(
                    result.poses.len() == view_indices.len(),
                    "intrinsics refinement returned {} poses, expected {} (camera {})",
                    result.poses.len(),
                    view_indices.len(),
                    cam_idx
                );
                for (pose, view_idx) in result.poses.iter().zip(view_indices.iter()) {
                    cam_target_poses[*view_idx][cam_idx] = Some(pose.inverse());
                }
            } else {
                cameras.push(cam_cfg);
                for (pose, view_idx) in pose_seeds.iter().zip(view_indices.iter()) {
                    cam_target_poses[*view_idx][cam_idx] = Some(pose.inverse());
                }
            }
        }

        // Step 2: Estimate rig extrinsics from camera-target poses
        let extrinsics = calib_linear::extrinsics::estimate_extrinsics_from_cam_target_poses(
            &cam_target_poses,
            opts.ref_cam_idx,
        )?;

        Ok(RigExtrinsicsInitial {
            cameras,
            ref_cam_idx: opts.ref_cam_idx,
            cam_to_rig: extrinsics.cam_to_rig,
            rig_from_target: extrinsics.rig_from_target,
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        use calib_optim::problems::rig_extrinsics::*;

        anyhow::ensure!(
            init.ref_cam_idx < obs.num_cameras,
            "ref_cam_idx {} out of bounds for {} cameras",
            init.ref_cam_idx,
            obs.num_cameras
        );

        // Convert observations to calib-optim format
        let rig_views: Vec<RigViewObservations> = obs
            .views
            .iter()
            .map(|view| {
                let cameras: Vec<Option<CameraViewObservations>> = view
                    .cameras
                    .iter()
                    .map(|cam_opt| {
                        cam_opt.as_ref().map(|cam_data| CameraViewObservations {
                            points_3d: cam_data.points_3d.clone(),
                            points_2d: cam_data.points_2d.clone(),
                            weights: cam_data.weights.clone(),
                        })
                    })
                    .collect();
                RigViewObservations { cameras }
            })
            .collect();

        let dataset = RigExtrinsicsDataset::new(rig_views, obs.num_cameras)?;

        // Extract initial values
        let intrinsics: Vec<FxFyCxCySkew<Real>> = init
            .cameras
            .iter()
            .map(|cam| match &cam.intrinsics {
                calib_core::IntrinsicsParams::FxFyCxCySkew { params } => *params,
            })
            .collect();

        let distortion: Vec<BrownConrady5<Real>> = init
            .cameras
            .iter()
            .map(|cam| match &cam.distortion {
                calib_core::DistortionParams::BrownConrady5 { params } => *params,
                calib_core::DistortionParams::None => BrownConrady5 {
                    k1: 0.0,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            })
            .collect();

        let initial = RigExtrinsicsInit {
            intrinsics,
            distortion,
            cam_to_rig: init.cam_to_rig.clone(),
            rig_from_target: init.rig_from_target.clone(),
        };

        let mut solve_opts = opts.solve_opts.clone();
        if solve_opts.fix_extrinsics.is_empty() {
            solve_opts.fix_extrinsics = vec![false; obs.num_cameras];
        }
        anyhow::ensure!(
            solve_opts.fix_extrinsics.len() == obs.num_cameras,
            "fix_extrinsics length {} != num_cameras {}",
            solve_opts.fix_extrinsics.len(),
            obs.num_cameras
        );
        solve_opts.fix_extrinsics[init.ref_cam_idx] = true;

        // Run optimization
        let result = calib_optim::problems::rig_extrinsics::optimize_rig_extrinsics(
            dataset,
            initial,
            solve_opts,
            opts.backend_opts.clone(),
        )?;

        // Convert Camera to CameraParams
        let cameras: Vec<calib_core::CameraParams> = result
            .cameras
            .into_iter()
            .map(|cam| calib_core::CameraParams {
                projection: calib_core::ProjectionParams::Pinhole,
                distortion: calib_core::DistortionParams::BrownConrady5 { params: cam.dist },
                sensor: calib_core::SensorParams::Identity,
                intrinsics: calib_core::IntrinsicsParams::FxFyCxCySkew { params: cam.k },
            })
            .collect();

        Ok(RigExtrinsicsOptimized {
            cameras,
            cam_to_rig: result.cam_to_rig,
            rig_from_target: result.rig_from_target,
            final_cost: result.final_cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_pinhole_camera, session::CalibrationSession};
    use calib_core::{BrownConrady5, FxFyCxCySkew, Iso3, Pt3, Vec2};
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn rig_extrinsics_problem_synthetic_convergence() {
        // Ground truth parameters
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

        // Create two cameras with same intrinsics
        let cam0 = make_pinhole_camera(k_gt, dist_gt);
        let cam1 = make_pinhole_camera(k_gt, dist_gt);

        // Ground truth rig extrinsics
        let cam0_to_rig = Iso3::identity();
        let cam1_to_rig = Iso3::from_parts(
            Vector3::new(0.1, 0.0, 0.0).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.05, 0.0)),
        );

        // Generate checkerboard
        let nx = 5;
        let ny = 4;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        // Generate views
        let mut views = Vec::new();
        let mut rig_from_target_gt = Vec::new();
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let rig_from_target = Iso3::from_parts(translation.into(), rotation);
            rig_from_target_gt.push(rig_from_target);

            // Compute target -> camera poses: T_C_T = (T_R_C)^-1 * (T_R_T)
            let cam0_from_target = cam0_to_rig.inverse() * rig_from_target;
            let cam1_from_target = cam1_to_rig.inverse() * rig_from_target;

            // Project through cameras
            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();

            for pw in &board_points {
                let pc0 = cam0_from_target.transform_point(pw);
                let pc1 = cam1_from_target.transform_point(pw);

                if let Some(proj) = cam0.project_point(&pc0) {
                    cam0_pixels.push(Vec2::new(proj.x, proj.y));
                }
                if let Some(proj) = cam1.project_point(&pc1) {
                    cam1_pixels.push(Vec2::new(proj.x, proj.y));
                }
            }

            views.push(RigViewData {
                cameras: vec![
                    Some(CameraViewData {
                        points_3d: board_points.clone(),
                        points_2d: cam0_pixels,
                        weights: None,
                    }),
                    Some(CameraViewData {
                        points_3d: board_points.clone(),
                        points_2d: cam1_pixels,
                        weights: None,
                    }),
                ],
            });
        }

        // Create session and run pipeline
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        let input = RigExtrinsicsObservations {
            views,
            num_cameras: 2,
        };
        session.set_observations(input.clone());

        // Initialize
        let init_result = session.initialize(RigExtrinsicsInitOptions {
            ref_cam_idx: 0,
            ..Default::default()
        });
        assert!(
            init_result.is_ok(),
            "Initialization failed: {:?}",
            init_result.err()
        );

        // Optimize
        let optim_result = session.optimize(RigExtrinsicsOptimOptions::default());
        assert!(
            optim_result.is_ok(),
            "Optimization failed: {:?}",
            optim_result.err()
        );

        // Export
        let export_result = session.export();
        assert!(export_result.is_ok(), "Export failed");

        let final_result = export_result.unwrap();
        let reproj =
            crate::rig_extrinsics::rig_reprojection_errors_from_report(&input, &final_result)
                .unwrap();
        let mean_px = reproj.mean_px.unwrap();
        let rmse_px = reproj.rmse_px.unwrap();
        assert!(
            mean_px < 1e-3,
            "mean reprojection error too high: {mean_px}"
        );
        assert!(
            rmse_px < 1e-3,
            "rmse reprojection error too high: {rmse_px}"
        );

        // Verify cam0_to_rig is identity (reference camera)
        let cam0_trans = final_result.cam_to_rig[0].translation.vector.norm();
        let cam0_rot = final_result.cam_to_rig[0].rotation.angle();
        assert!(
            cam0_trans < 1e-12,
            "cam0 translation not identity: {cam0_trans}"
        );
        assert!(cam0_rot < 1e-12, "cam0 rotation not identity: {cam0_rot}");

        // Verify cam1_to_rig translation + rotation
        let cam1_trans_err =
            (final_result.cam_to_rig[1].translation.vector - cam1_to_rig.translation.vector).norm();
        let cam1_rot_err =
            (final_result.cam_to_rig[1].rotation * cam1_to_rig.rotation.inverse()).angle();
        assert!(
            cam1_trans_err < 1e-3,
            "cam1 translation error too large: {cam1_trans_err}"
        );
        assert!(
            cam1_rot_err < 1e-3,
            "cam1 rotation error too large: {cam1_rot_err}"
        );

        // Verify per-view rig poses
        assert_eq!(final_result.rig_from_target.len(), rig_from_target_gt.len());
        for (view_idx, (rig_est, rig_gt)) in final_result
            .rig_from_target
            .iter()
            .zip(rig_from_target_gt.iter())
            .enumerate()
        {
            let trans_err = (rig_est.translation.vector - rig_gt.translation.vector).norm();
            let rot_err = (rig_est.rotation * rig_gt.rotation.inverse()).angle();
            assert!(
                trans_err < 1e-3,
                "view {} rig translation error too large: {}",
                view_idx,
                trans_err
            );
            assert!(
                rot_err < 1e-3,
                "view {} rig rotation error too large: {}",
                view_idx,
                rot_err
            );
        }
    }

    #[test]
    fn rig_extrinsics_handles_missing_observations() {
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

        let cam0_to_rig = Iso3::identity();
        let cam1_to_rig = Iso3::from_parts(
            Vector3::new(0.1, 0.0, 0.0).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.05, 0.0)),
        );

        let nx = 5;
        let ny = 4;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for view_idx in 0..4 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let rig_from_target = Iso3::from_parts(translation.into(), rotation);

            let cam0_from_target = cam0_to_rig.inverse() * rig_from_target;
            let cam1_from_target = cam1_to_rig.inverse() * rig_from_target;

            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();
            for pw in &board_points {
                let pc0 = cam0_from_target.transform_point(pw);
                let pc1 = cam1_from_target.transform_point(pw);
                if let Some(proj) = cam0.project_point(&pc0) {
                    cam0_pixels.push(Vec2::new(proj.x, proj.y));
                }
                if let Some(proj) = cam1.project_point(&pc1) {
                    cam1_pixels.push(Vec2::new(proj.x, proj.y));
                }
            }

            views.push(RigViewData {
                cameras: vec![
                    Some(CameraViewData {
                        points_3d: board_points.clone(),
                        points_2d: cam0_pixels,
                        weights: None,
                    }),
                    if view_idx == 0 {
                        None
                    } else {
                        Some(CameraViewData {
                            points_3d: board_points.clone(),
                            points_2d: cam1_pixels,
                            weights: None,
                        })
                    },
                ],
            });
        }

        let input = RigExtrinsicsObservations {
            views,
            num_cameras: 2,
        };

        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_observations(input.clone());
        session
            .initialize(RigExtrinsicsInitOptions {
                ref_cam_idx: 0,
                ..Default::default()
            })
            .unwrap();
        session
            .optimize(RigExtrinsicsOptimOptions::default())
            .unwrap();
        let report = session.export().unwrap();

        let reproj =
            crate::rig_extrinsics::rig_reprojection_errors_from_report(&input, &report).unwrap();
        let mean_px = reproj.mean_px.unwrap();
        assert!(
            mean_px < 1e-3,
            "mean reprojection error too high: {mean_px}"
        );
        assert_eq!(report.rig_from_target.len(), 4);
    }

    #[test]
    fn rig_extrinsics_cost_improves_after_optimization() {
        // Ground truth parameters
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

        let cam0_to_rig = Iso3::identity();
        let cam1_to_rig = Iso3::from_parts(
            Vector3::new(0.1, 0.0, 0.0).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.05, 0.0)),
        );

        // Checkerboard
        let nx = 5;
        let ny = 4;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let rig_from_target = Iso3::from_parts(translation.into(), rotation);

            let cam0_from_target = cam0_to_rig.inverse() * rig_from_target;
            let cam1_from_target = cam1_to_rig.inverse() * rig_from_target;

            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();
            for pw in &board_points {
                let pc0 = cam0_from_target.transform_point(pw);
                let pc1 = cam1_from_target.transform_point(pw);
                if let Some(proj) = cam0.project_point(&pc0) {
                    cam0_pixels.push(Vec2::new(proj.x, proj.y));
                }
                if let Some(proj) = cam1.project_point(&pc1) {
                    cam1_pixels.push(Vec2::new(proj.x, proj.y));
                }
            }

            views.push(RigViewData {
                cameras: vec![
                    Some(CameraViewData {
                        points_3d: board_points.clone(),
                        points_2d: cam0_pixels,
                        weights: None,
                    }),
                    Some(CameraViewData {
                        points_3d: board_points.clone(),
                        points_2d: cam1_pixels,
                        weights: None,
                    }),
                ],
            });
        }

        // Initial cost: run optimize with 0 iterations (evaluates seed)
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_observations(RigExtrinsicsObservations {
            views: views.clone(),
            num_cameras: 2,
        });
        session
            .initialize(RigExtrinsicsInitOptions {
                ref_cam_idx: 0,
                ..Default::default()
            })
            .unwrap();

        let mut opts_eval = RigExtrinsicsOptimOptions::default();
        opts_eval.backend_opts.max_iters = 0;
        let cost0 = session.optimize(opts_eval).unwrap().final_cost;

        // Optimized cost: new session to avoid stage conflicts
        let mut session_opt = CalibrationSession::<RigExtrinsicsProblem>::new();
        session_opt.set_observations(RigExtrinsicsObservations {
            views,
            num_cameras: 2,
        });
        session_opt
            .initialize(RigExtrinsicsInitOptions {
                ref_cam_idx: 0,
                ..Default::default()
            })
            .unwrap();
        let result = session_opt
            .optimize(RigExtrinsicsOptimOptions::default())
            .unwrap();
        let cost1 = result.final_cost;

        assert!(
            cost1 < cost0,
            "optimization did not reduce cost: initial {} vs optimized {}",
            cost0,
            cost1
        );
    }

    #[test]
    fn rig_extrinsics_session_json_roundtrip() {
        let views = vec![RigViewData {
            cameras: vec![
                Some(CameraViewData {
                    points_3d: vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.05, 0.0, 0.0)],
                    points_2d: vec![Vec2::new(100.0, 100.0), Vec2::new(200.0, 100.0)],
                    weights: None,
                }),
                Some(CameraViewData {
                    points_3d: vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.05, 0.0, 0.0)],
                    points_2d: vec![Vec2::new(150.0, 100.0), Vec2::new(250.0, 100.0)],
                    weights: None,
                }),
            ],
        }];

        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_observations(RigExtrinsicsObservations {
            views,
            num_cameras: 2,
        });

        // Serialize
        let json = session.to_json().unwrap();
        assert!(json.contains("rig_extrinsics"));

        // Deserialize
        let restored: CalibrationSession<RigExtrinsicsProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(
            restored.stage(),
            crate::session::SessionStage::Uninitialized
        );
        assert_eq!(restored.observations().unwrap().views.len(), 1);
        assert_eq!(restored.observations().unwrap().num_cameras, 2);
    }
}
