//! Stepwise pipeline helpers for single-camera hand-eye calibration.

use crate::{
    build_planar_dataset, k_matrix_from_intrinsics, make_pinhole_camera,
    planar_homographies_from_views, poses_from_homographies, PlanarIntrinsicsInput, PlanarViewData,
};
use anyhow::{ensure, Result};
use calib_core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, Iso3,
    ProjectionParams, Pt2, Real, SensorParams,
};
use calib_linear::handeye::estimate_handeye_dlt;
use calib_linear::homography::dlt_homography_ransac;
pub use calib_linear::iterative_intrinsics::IterativeIntrinsicsOptions;
pub use calib_optim::backend::BackendSolveOptions;
pub use calib_optim::handeye::HandEyeSolveOptions;
use calib_optim::handeye::{
    optimize_handeye, CameraViewObservations, HandEyeDataset, HandEyeInit, RigViewObservations,
};
pub use calib_optim::ir::HandEyeMode;
pub use calib_optim::ir::RobustLoss;
pub use calib_optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;
use calib_optim::planar_intrinsics::{optimize_planar_intrinsics, PlanarIntrinsicsInit};

/// Input view for single-camera hand-eye calibration.
#[derive(Debug, Clone)]
pub struct HandEyeView {
    /// Planar target observations in this view.
    pub view: PlanarViewData,
    /// Robot pose (base-to-gripper) for this view.
    pub robot_pose: Iso3,
}

/// Intrinsics stage output with per-view poses and reprojection error.
#[derive(Debug, Clone)]
pub struct IntrinsicsStage {
    /// Pinhole intrinsics (fx, fy, cx, cy, skew).
    pub intrinsics: calib_core::FxFyCxCySkew<Real>,
    /// Brown-Conrady distortion (k1-k3, p1-p2).
    pub distortion: calib_core::BrownConrady5<Real>,
    /// Per-view pose estimates: camera-from-target (T_C_T).
    pub poses: Vec<Iso3>,
    /// Mean reprojection error in pixels (all points).
    pub mean_reproj_error: Real,
    /// Optional optimizer final cost.
    pub final_cost: Option<Real>,
}

/// Pose RANSAC stage output with filtered views and inlier stats.
#[derive(Debug, Clone)]
pub struct PoseRansacStage {
    /// Filtered views containing only inlier points.
    pub views: Vec<HandEyeView>,
    /// Per-view pose estimates: camera-from-target (T_C_T).
    pub poses: Vec<Iso3>,
    /// Mean reprojection error in pixels (inliers only).
    pub mean_reproj_error: Real,
    /// Inlier counts per view.
    pub inliers_per_view: Vec<usize>,
    /// Number of dropped views.
    pub dropped_views: usize,
}

/// Hand-eye stage output with target poses and reprojection error.
#[derive(Debug, Clone)]
pub struct HandEyeStage {
    /// Hand-eye transform (eye-in-hand: gripper-from-camera, eye-to-hand: camera-from-base).
    pub handeye: Iso3,
    /// Target poses: eye-in-hand uses base-from-target, eye-to-hand uses gripper-from-target.
    pub target_poses: Vec<Iso3>,
    /// Mean reprojection error in pixels (inliers only).
    pub mean_reproj_error: Real,
    /// Optional optimizer final cost.
    pub final_cost: Option<Real>,
}

/// Full report for stepwise hand-eye calibration.
#[derive(Debug, Clone)]
pub struct HandEyeSingleReport {
    pub intrinsics_init: IntrinsicsStage,
    pub intrinsics_optimized: IntrinsicsStage,
    pub pose_ransac: PoseRansacStage,
    pub handeye_init: HandEyeStage,
    pub handeye_optimized: HandEyeStage,
}

/// RANSAC defaults for per-view planar pose estimation.
pub const RANSAC_THRESH_PX: Real = 1.0;
/// Minimum inliers for accepting a pose.
pub const RANSAC_MIN_INLIERS: usize = 8;
/// Maximum RANSAC iterations per view.
pub const RANSAC_MAX_ITERS: usize = 500;
/// RANSAC confidence level.
pub const RANSAC_CONFIDENCE: Real = 0.99;
/// Deterministic RANSAC seed.
pub const RANSAC_SEED: u64 = 1_234_567;

/// Options for per-view planar pose RANSAC.
#[derive(Debug, Clone)]
pub struct PoseRansacOptions {
    /// Reprojection threshold in pixels.
    pub thresh_px: Real,
    /// Minimum inliers to accept a pose.
    pub min_inliers: usize,
    /// Maximum RANSAC iterations.
    pub max_iters: usize,
    /// RANSAC confidence.
    pub confidence: Real,
    /// Deterministic RNG seed.
    pub seed: u64,
}

impl Default for PoseRansacOptions {
    fn default() -> Self {
        Self {
            thresh_px: RANSAC_THRESH_PX,
            min_inliers: RANSAC_MIN_INLIERS,
            max_iters: RANSAC_MAX_ITERS,
            confidence: RANSAC_CONFIDENCE,
            seed: RANSAC_SEED,
        }
    }
}

/// Initialize intrinsics and distortion from planar views.
pub fn init_intrinsics(
    views: &[HandEyeView],
    opts: &IterativeIntrinsicsOptions,
) -> Result<IntrinsicsStage> {
    let planar_views: Vec<PlanarViewData> = views.iter().map(|v| v.view.clone()).collect();
    let init = crate::helpers::initialize_planar_intrinsics(&planar_views, opts)?;

    let homographies = planar_homographies_from_views(&planar_views)?;
    let kmtx = k_matrix_from_intrinsics(&init.intrinsics);
    let poses = poses_from_homographies(&kmtx, &homographies)?;

    let mean_reproj_error =
        mean_reproj_error_planar(&planar_views, &init.intrinsics, &init.distortion, &poses)?;

    Ok(IntrinsicsStage {
        intrinsics: init.intrinsics,
        distortion: init.distortion,
        poses,
        mean_reproj_error,
        final_cost: None,
    })
}

/// Build an intrinsics stage from known parameters (recomputes poses + error).
pub fn intrinsics_stage_from_params(
    views: &[HandEyeView],
    intrinsics: FxFyCxCySkew<Real>,
    distortion: BrownConrady5<Real>,
) -> Result<IntrinsicsStage> {
    let planar_views: Vec<PlanarViewData> = views.iter().map(|v| v.view.clone()).collect();
    let homographies = planar_homographies_from_views(&planar_views)?;
    let kmtx = k_matrix_from_intrinsics(&intrinsics);
    let poses = poses_from_homographies(&kmtx, &homographies)?;
    let mean_reproj_error =
        mean_reproj_error_planar(&planar_views, &intrinsics, &distortion, &poses)?;

    Ok(IntrinsicsStage {
        intrinsics,
        distortion,
        poses,
        mean_reproj_error,
        final_cost: None,
    })
}

/// Extract pinhole intrinsics and Brown-Conrady distortion from camera params.
///
/// Requires `ProjectionParams::Pinhole` and `SensorParams::Identity`.
pub fn pinhole_from_camera_params(
    camera: &CameraParams,
) -> Result<(FxFyCxCySkew<Real>, BrownConrady5<Real>)> {
    ensure!(
        matches!(camera.projection, ProjectionParams::Pinhole),
        "camera projection is not pinhole"
    );
    ensure!(
        matches!(camera.sensor, SensorParams::Identity),
        "camera sensor is not identity"
    );

    let intrinsics = match camera.intrinsics {
        IntrinsicsParams::FxFyCxCySkew { params } => params,
    };
    let distortion = match camera.distortion {
        DistortionParams::BrownConrady5 { params } => params,
        DistortionParams::None => BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
    };

    Ok((intrinsics, distortion))
}

/// Optimize intrinsics and distortion from an initial stage.
pub fn optimize_intrinsics(
    views: &[HandEyeView],
    init: &IntrinsicsStage,
    solve_opts: &PlanarIntrinsicsSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<IntrinsicsStage> {
    let planar_views: Vec<PlanarViewData> = views.iter().map(|v| v.view.clone()).collect();
    let dataset = build_planar_dataset(&PlanarIntrinsicsInput {
        views: planar_views.clone(),
    })?;

    let init = PlanarIntrinsicsInit::new(init.intrinsics, init.distortion, init.poses.clone())?;
    let result =
        optimize_planar_intrinsics(dataset, init, solve_opts.clone(), backend_opts.clone())?;

    let mean_reproj_error = mean_reproj_error_planar(
        &planar_views,
        &result.camera.k,
        &result.camera.dist,
        &result.poses,
    )?;

    Ok(IntrinsicsStage {
        intrinsics: result.camera.k,
        distortion: result.camera.dist,
        poses: result.poses,
        mean_reproj_error,
        final_cost: Some(result.final_cost),
    })
}

/// Estimate per-view poses with RANSAC and filter outliers.
pub fn ransac_planar_poses(
    views: &[HandEyeView],
    intrinsics: &calib_core::FxFyCxCySkew<Real>,
    distortion: &calib_core::BrownConrady5<Real>,
    opts: &PoseRansacOptions,
) -> Result<PoseRansacStage> {
    use calib_core::RansacOptions;
    use calib_linear::planar_pose::estimate_planar_pose_from_h;

    let ransac_opts = RansacOptions {
        max_iters: opts.max_iters,
        thresh: opts.thresh_px,
        min_inliers: opts.min_inliers,
        confidence: opts.confidence,
        seed: opts.seed,
        refit_on_inliers: true,
    };

    let kmtx = k_matrix_from_intrinsics(intrinsics);
    let mut filtered_views = Vec::new();
    let mut poses = Vec::new();
    let mut inliers_per_view = Vec::new();
    let mut dropped = 0usize;

    for view in views {
        let board_2d: Vec<Pt2> = view
            .view
            .points_3d
            .iter()
            .map(|p| Pt2::new(p.x, p.y))
            .collect();
        let pixel_2d: Vec<Pt2> = view
            .view
            .points_2d
            .iter()
            .map(|p| Pt2::new(p.x, p.y))
            .collect();

        let (h, inliers) = match dlt_homography_ransac(&board_2d, &pixel_2d, &ransac_opts) {
            Ok(res) => res,
            Err(_) => {
                dropped += 1;
                continue;
            }
        };

        let pose = match estimate_planar_pose_from_h(&kmtx, &h) {
            Ok(pose) => pose,
            Err(_) => {
                dropped += 1;
                continue;
            }
        };

        let filtered_view = filter_view_inliers(&view.view, &inliers);
        let inlier_count = filtered_view.points_2d.len();
        if inlier_count < opts.min_inliers {
            dropped += 1;
            continue;
        }

        filtered_views.push(HandEyeView {
            view: filtered_view,
            robot_pose: view.robot_pose.clone(),
        });
        poses.push(pose);
        inliers_per_view.push(inlier_count);
    }

    ensure!(
        !filtered_views.is_empty(),
        "all views dropped by pose RANSAC"
    );

    let mean_reproj_error = mean_reproj_error_planar(
        &filtered_views
            .iter()
            .map(|v| v.view.clone())
            .collect::<Vec<_>>(),
        intrinsics,
        distortion,
        &poses,
    )?;

    Ok(PoseRansacStage {
        views: filtered_views,
        poses,
        mean_reproj_error,
        inliers_per_view,
        dropped_views: dropped,
    })
}

/// Initialize hand-eye using linear DLT from base-to-gripper and camera-from-target poses.
///
/// `cam_from_target` are per-view poses `T_C_T` from planar pose estimation.
pub fn init_handeye(
    views: &[HandEyeView],
    cam_from_target: &[Iso3],
    intrinsics: &calib_core::FxFyCxCySkew<Real>,
    distortion: &calib_core::BrownConrady5<Real>,
    mode: HandEyeMode,
) -> Result<HandEyeStage> {
    ensure!(
        views.len() == cam_from_target.len(),
        "view count ({}) must match pose count ({})",
        views.len(),
        cam_from_target.len()
    );

    let base_to_gripper: Vec<Iso3> = views.iter().map(|v| v.robot_pose.clone()).collect();
    let cam_in_target: Vec<Iso3> = cam_from_target.iter().map(|pose| pose.inverse()).collect();

    let handeye = estimate_handeye_dlt(&base_to_gripper, &cam_in_target, 1.0)?;
    let target_poses = compute_target_poses(&base_to_gripper, cam_from_target, &handeye, mode);

    let mut intrinsics_fixed = *intrinsics;
    intrinsics_fixed.skew = 0.0;
    let mean_reproj_error = mean_reproj_error_handeye(
        views,
        &intrinsics_fixed,
        distortion,
        &handeye,
        &target_poses,
        mode,
    )?;

    Ok(HandEyeStage {
        handeye,
        target_poses,
        mean_reproj_error,
        final_cost: None,
    })
}

/// Optimize hand-eye with fixed intrinsics and distortion.
pub fn optimize_handeye_stage(
    views: &[HandEyeView],
    init: &HandEyeStage,
    intrinsics: &calib_core::FxFyCxCySkew<Real>,
    distortion: &calib_core::BrownConrady5<Real>,
    mode: HandEyeMode,
    solve_opts: &HandEyeSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<HandEyeStage> {
    let mut rig_views = Vec::new();
    for view in views {
        let obs =
            CameraViewObservations::new(view.view.points_3d.clone(), view.view.points_2d.clone())?;
        rig_views.push(RigViewObservations {
            cameras: vec![Some(obs)],
            robot_pose: view.robot_pose.clone(),
        });
    }

    let dataset = HandEyeDataset::new(rig_views, 1, mode)?;
    let mut intrinsics_fixed = *intrinsics;
    intrinsics_fixed.skew = 0.0;

    let init = HandEyeInit {
        intrinsics: vec![intrinsics_fixed],
        distortion: vec![*distortion],
        cam_to_rig: vec![Iso3::identity()],
        handeye: init.handeye.clone(),
        target_poses: init.target_poses.clone(),
    };

    let result = optimize_handeye(dataset, init, solve_opts.clone(), backend_opts.clone())?;

    let mean_reproj_error = mean_reproj_error_handeye(
        views,
        &intrinsics_fixed,
        distortion,
        &result.handeye,
        &result.target_poses,
        mode,
    )?;

    Ok(HandEyeStage {
        handeye: result.handeye,
        target_poses: result.target_poses,
        mean_reproj_error,
        final_cost: Some(result.final_cost),
    })
}

/// Run the full stepwise hand-eye pipeline with fixed intrinsics/distortion.
pub fn run_handeye_single(
    views: &[HandEyeView],
    intrinsics_init_opts: &IterativeIntrinsicsOptions,
    intrinsics_solve_opts: &PlanarIntrinsicsSolveOptions,
    intrinsics_backend_opts: &BackendSolveOptions,
    ransac_opts: &PoseRansacOptions,
    mode: HandEyeMode,
    handeye_solve_opts: &HandEyeSolveOptions,
    handeye_backend_opts: &BackendSolveOptions,
) -> Result<HandEyeSingleReport> {
    let intrinsics_init = init_intrinsics(views, intrinsics_init_opts)?;
    let intrinsics_optimized = optimize_intrinsics(
        views,
        &intrinsics_init,
        intrinsics_solve_opts,
        intrinsics_backend_opts,
    )?;
    let pose_ransac = ransac_planar_poses(
        views,
        &intrinsics_optimized.intrinsics,
        &intrinsics_optimized.distortion,
        ransac_opts,
    )?;
    let handeye_init = init_handeye(
        &pose_ransac.views,
        &pose_ransac.poses,
        &intrinsics_optimized.intrinsics,
        &intrinsics_optimized.distortion,
        mode,
    )?;
    let mut handeye_opts = handeye_solve_opts.clone();
    apply_handeye_fixed_intrinsics(&mut handeye_opts);
    ensure_handeye_defaults(&mut handeye_opts, 1);

    let handeye_optimized = optimize_handeye_stage(
        &pose_ransac.views,
        &handeye_init,
        &intrinsics_optimized.intrinsics,
        &intrinsics_optimized.distortion,
        mode,
        &handeye_opts,
        handeye_backend_opts,
    )?;

    Ok(HandEyeSingleReport {
        intrinsics_init,
        intrinsics_optimized,
        pose_ransac,
        handeye_init,
        handeye_optimized,
    })
}

fn apply_handeye_fixed_intrinsics(opts: &mut HandEyeSolveOptions) {
    opts.fix_fx = true;
    opts.fix_fy = true;
    opts.fix_cx = true;
    opts.fix_cy = true;
    opts.fix_k1 = true;
    opts.fix_k2 = true;
    opts.fix_k3 = true;
    opts.fix_p1 = true;
    opts.fix_p2 = true;
}

fn ensure_handeye_defaults(opts: &mut HandEyeSolveOptions, num_cameras: usize) {
    if opts.fix_extrinsics.is_empty() {
        opts.fix_extrinsics = vec![true; num_cameras];
    }
    if opts.fix_target_poses.is_empty() {
        opts.fix_target_poses = vec![0];
    }
}

fn compute_target_poses(
    base_to_gripper: &[Iso3],
    cam_from_target: &[Iso3],
    handeye: &Iso3,
    mode: HandEyeMode,
) -> Vec<Iso3> {
    base_to_gripper
        .iter()
        .zip(cam_from_target.iter())
        .map(|(base_to_gripper, cam_from_target)| match mode {
            HandEyeMode::EyeInHand => base_to_gripper * handeye * cam_from_target,
            HandEyeMode::EyeToHand => {
                base_to_gripper.inverse() * handeye.inverse() * cam_from_target
            }
        })
        .collect()
}

fn filter_view_inliers(view: &PlanarViewData, inliers: &[usize]) -> PlanarViewData {
    let mut points_3d = Vec::with_capacity(inliers.len());
    let mut points_2d = Vec::with_capacity(inliers.len());

    for &idx in inliers {
        points_3d.push(view.points_3d[idx]);
        points_2d.push(view.points_2d[idx]);
    }

    let weights = view
        .weights
        .as_ref()
        .map(|w| inliers.iter().map(|&idx| w[idx]).collect::<Vec<_>>());

    PlanarViewData {
        points_3d,
        points_2d,
        weights,
    }
}

fn mean_reproj_error_planar(
    views: &[PlanarViewData],
    intrinsics: &calib_core::FxFyCxCySkew<Real>,
    distortion: &calib_core::BrownConrady5<Real>,
    poses: &[Iso3],
) -> Result<Real> {
    use calib_core::{Camera, IdentitySensor, Pinhole};

    let camera = Camera::new(Pinhole, *distortion, IdentitySensor, *intrinsics);

    let mut total_error = 0.0;
    let mut total_points = 0usize;

    for (view, pose) in views.iter().zip(poses.iter()) {
        for (p3d, p2d) in view.points_3d.iter().zip(view.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            if let Some(projected) = camera.project_point_c(&p_cam.coords) {
                let error = (projected - *p2d).norm();
                total_error += error;
                total_points += 1;
            }
        }
    }

    ensure!(
        total_points > 0,
        "no valid projections for error computation"
    );
    Ok(total_error / total_points as Real)
}

fn mean_reproj_error_handeye(
    views: &[HandEyeView],
    intrinsics: &calib_core::FxFyCxCySkew<Real>,
    distortion: &calib_core::BrownConrady5<Real>,
    handeye: &Iso3,
    target_poses: &[Iso3],
    mode: HandEyeMode,
) -> Result<Real> {
    ensure!(
        views.len() == target_poses.len(),
        "target pose count ({}) must match view count ({})",
        target_poses.len(),
        views.len()
    );

    let camera = make_pinhole_camera(*intrinsics, *distortion);
    let cam_to_rig = Iso3::identity();

    let mut total_error = 0.0;
    let mut total_points = 0usize;

    for (view, target_pose) in views.iter().zip(target_poses.iter()) {
        let base_to_gripper = &view.robot_pose;
        for (p3d, p2d) in view.view.points_3d.iter().zip(view.view.points_2d.iter()) {
            let p_cam = match mode {
                HandEyeMode::EyeInHand => {
                    let p_base = target_pose.transform_point(p3d);
                    let p_gripper = base_to_gripper.inverse_transform_point(&p_base);
                    let p_rig = handeye.inverse_transform_point(&p_gripper);
                    cam_to_rig.inverse_transform_point(&p_rig)
                }
                HandEyeMode::EyeToHand => {
                    let p_gripper = target_pose.transform_point(p3d);
                    let p_base = base_to_gripper.transform_point(&p_gripper);
                    let p_rig = handeye.transform_point(&p_base);
                    cam_to_rig.inverse_transform_point(&p_rig)
                }
            };

            if let Some(projected) = camera.project_point(&p_cam) {
                let error = (projected - *p2d).norm();
                total_error += error;
                total_points += 1;
            }
        }
    }

    ensure!(
        total_points > 0,
        "no valid projections for error computation"
    );
    Ok(total_error / total_points as Real)
}
