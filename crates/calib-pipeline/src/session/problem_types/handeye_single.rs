//! Single-camera hand-eye session problem.

use crate::handeye_single::{
    init_handeye, init_intrinsics, optimize_handeye_stage, optimize_intrinsics,
    ransac_planar_poses, BackendSolveOptions, HandEyeMode, HandEyeSolveOptions, HandEyeStage,
    HandEyeView, IntrinsicsStage, IterativeIntrinsicsOptions, PlanarIntrinsicsSolveOptions,
    PoseRansacOptions, PoseRansacStage, RobustLoss,
};
use anyhow::{ensure, Result};
use calib_core::Real;
use serde::{Deserialize, Serialize};

use crate::session::ProblemType;

/// Single-camera hand-eye calibration problem (session-friendly pipeline).
///
/// Runs a compact, deterministic sequence:
/// 1) Intrinsics initialization
/// 2) Intrinsics optimization
/// 3) Pose RANSAC (inlier filtering)
/// 4) Intrinsics re-optimization on inliers
/// 5) Hand-eye DLT initialization (from inlier poses)
/// 6) Hand-eye optimization (fixed robot poses)
/// 7) Optional hand-eye optimization with robot pose refinement
pub struct HandEyeSingleProblem;

/// Serializable hand-eye mode for session inputs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HandEyeModeConfig {
    EyeInHand,
    EyeToHand,
}

impl From<HandEyeModeConfig> for HandEyeMode {
    fn from(mode: HandEyeModeConfig) -> Self {
        match mode {
            HandEyeModeConfig::EyeInHand => HandEyeMode::EyeInHand,
            HandEyeModeConfig::EyeToHand => HandEyeMode::EyeToHand,
        }
    }
}

impl From<HandEyeMode> for HandEyeModeConfig {
    fn from(mode: HandEyeMode) -> Self {
        match mode {
            HandEyeMode::EyeInHand => HandEyeModeConfig::EyeInHand,
            HandEyeMode::EyeToHand => HandEyeModeConfig::EyeToHand,
        }
    }
}

/// Observations for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeSingleObservations {
    pub views: Vec<HandEyeView>,
    pub mode: HandEyeModeConfig,
}

/// Initial values for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeSingleInitial {
    pub intrinsics_init: IntrinsicsStage,
}

/// Optimized results from the hand-eye single-camera pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeSingleOptimized {
    pub intrinsics_optimized: IntrinsicsStage,
    pub pose_ransac: PoseRansacStage,
    pub intrinsics_optimized_inliers: IntrinsicsStage,
    pub handeye_init: HandEyeStage,
    pub handeye_optimized: HandEyeStage,
    pub handeye_optimized_refined: Option<HandEyeStage>,
}

/// Options for hand-eye initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HandEyeSingleInitOptions {
    pub intrinsics_init_opts: IterativeIntrinsicsOptions,
}

/// Options for the hand-eye single-camera pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandEyeSingleOptimOptions {
    pub intrinsics_solve_opts: PlanarIntrinsicsSolveOptions,
    pub intrinsics_backend_opts: BackendSolveOptions,
    pub ransac_opts: PoseRansacOptions,
    pub handeye_solve_opts: HandEyeSolveOptions,
    pub handeye_backend_opts: BackendSolveOptions,
    /// Run a second hand-eye solve with robot pose refinement enabled.
    pub refine_robot_poses: bool,
    /// Robot rotation prior sigma (radians).
    pub robot_rot_sigma: Real,
    /// Robot translation prior sigma (meters).
    pub robot_trans_sigma: Real,
}

impl Default for HandEyeSingleOptimOptions {
    fn default() -> Self {
        let intrinsics_solve_opts = PlanarIntrinsicsSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 2.0 },
            ..Default::default()
        };

        let mut handeye_solve_opts = HandEyeSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 2.0 },
            default_fix: calib_core::CameraFixMask::all_fixed(),
            fix_extrinsics: vec![true],
            ..Default::default()
        };
        handeye_solve_opts.refine_robot_poses = false;

        Self {
            intrinsics_solve_opts,
            intrinsics_backend_opts: BackendSolveOptions {
                max_iters: 60,
                ..Default::default()
            },
            ransac_opts: PoseRansacOptions::default(),
            handeye_solve_opts,
            handeye_backend_opts: BackendSolveOptions::default(),
            refine_robot_poses: false,
            robot_rot_sigma: 0.5_f64.to_radians(),
            robot_trans_sigma: 1.0e-3,
        }
    }
}

impl ProblemType for HandEyeSingleProblem {
    type Observations = HandEyeSingleObservations;
    type InitialValues = HandEyeSingleInitial;
    type OptimizedResults = HandEyeSingleOptimized;
    type InitOptions = HandEyeSingleInitOptions;
    type OptimOptions = HandEyeSingleOptimOptions;

    fn problem_name() -> &'static str {
        "handeye_single"
    }

    fn initialize(
        obs: &Self::Observations,
        opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        ensure!(!obs.views.is_empty(), "need at least one view");
        let intrinsics_init = init_intrinsics(&obs.views, &opts.intrinsics_init_opts)?;
        Ok(HandEyeSingleInitial { intrinsics_init })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        let intrinsics_optimized = optimize_intrinsics(
            &obs.views,
            &init.intrinsics_init,
            &opts.intrinsics_solve_opts,
            &opts.intrinsics_backend_opts,
        )?;

        let pose_ransac = ransac_planar_poses(
            &obs.views,
            &intrinsics_optimized.intrinsics,
            &intrinsics_optimized.distortion,
            &opts.ransac_opts,
        )?;

        let intrinsics_optimized_inliers = optimize_intrinsics(
            &pose_ransac.views,
            &intrinsics_optimized,
            &opts.intrinsics_solve_opts,
            &opts.intrinsics_backend_opts,
        )?;

        let mode = HandEyeMode::from(obs.mode);
        let handeye_init = init_handeye(
            &pose_ransac.views,
            &intrinsics_optimized_inliers.poses,
            &intrinsics_optimized_inliers.intrinsics,
            &intrinsics_optimized_inliers.distortion,
            mode,
        )?;

        let mut handeye_opts = opts.handeye_solve_opts.clone();
        ensure_handeye_defaults(&mut handeye_opts, 1);
        handeye_opts.refine_robot_poses = false;

        let handeye_optimized = optimize_handeye_stage(
            &pose_ransac.views,
            &handeye_init,
            &intrinsics_optimized_inliers.intrinsics,
            &intrinsics_optimized_inliers.distortion,
            mode,
            &handeye_opts,
            &opts.handeye_backend_opts,
        )?;

        let handeye_optimized_refined = if opts.refine_robot_poses {
            let mut refine_opts = handeye_opts.clone();
            refine_opts.refine_robot_poses = true;
            refine_opts.robot_rot_sigma = opts.robot_rot_sigma;
            refine_opts.robot_trans_sigma = opts.robot_trans_sigma;

            Some(optimize_handeye_stage(
                &pose_ransac.views,
                &handeye_init,
                &intrinsics_optimized_inliers.intrinsics,
                &intrinsics_optimized_inliers.distortion,
                mode,
                &refine_opts,
                &opts.handeye_backend_opts,
            )?)
        } else {
            None
        };

        Ok(HandEyeSingleOptimized {
            intrinsics_optimized,
            pose_ransac,
            intrinsics_optimized_inliers,
            handeye_init,
            handeye_optimized,
            handeye_optimized_refined,
        })
    }
}

fn ensure_handeye_defaults(opts: &mut HandEyeSolveOptions, num_cameras: usize) {
    if opts.fix_extrinsics.is_empty() {
        opts.fix_extrinsics = vec![true; num_cameras];
    }
    if opts.relax_target_poses && opts.fix_target_poses.is_empty() {
        opts.fix_target_poses = vec![0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{make_pinhole_camera, session::CalibrationSession};
    use calib_core::{synthetic::planar, BrownConrady5, FxFyCxCySkew, Iso3};
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn handeye_single_problem_full_pipeline() {
        let k_gt = FxFyCxCySkew {
            fx: 820.0,
            fy: 810.0,
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
        let cam_gt = make_pinhole_camera(k_gt, dist_gt);

        let handeye_gt = Iso3::from_parts(
            Vector3::new(0.05, -0.02, 0.12).into(),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.05, -0.03, 0.08)),
        );
        let target_pose = Iso3::identity();
        let cam_to_rig = Iso3::identity();

        let robot_poses = vec![
            Iso3::from_parts(
                Vector3::new(0.0, 0.0, -1.0).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.0)),
            ),
            Iso3::from_parts(
                Vector3::new(-0.05, 0.04, -1.05).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.02, -0.03, 0.06)),
            ),
            Iso3::from_parts(
                Vector3::new(0.06, -0.03, -0.95).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(-0.03, 0.02, -0.05)),
            ),
            Iso3::from_parts(
                Vector3::new(-0.02, 0.02, -1.02).into(),
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01, 0.02, 0.03)),
            ),
        ];

        let board_points = planar::grid_points(5, 4, 0.05);

        let mut views = Vec::new();
        for robot_pose in &robot_poses {
            let cam_from_target =
                cam_to_rig.inverse() * handeye_gt.inverse() * robot_pose.inverse() * target_pose;
            let view = planar::project_view_all(&cam_gt, &cam_from_target, &board_points).unwrap();
            views.push(HandEyeView {
                view,
                robot_pose: *robot_pose,
            });
        }

        let mut session = CalibrationSession::<HandEyeSingleProblem>::new();
        session.set_observations(HandEyeSingleObservations {
            views,
            mode: HandEyeModeConfig::EyeInHand,
        });

        session
            .initialize(HandEyeSingleInitOptions::default())
            .unwrap();
        session
            .optimize(HandEyeSingleOptimOptions::default())
            .unwrap();
        let report = session.export().unwrap();

        let t_err = (report.handeye_optimized.handeye.translation.vector
            - handeye_gt.translation.vector)
            .norm();
        let r_final = report
            .handeye_optimized
            .handeye
            .rotation
            .to_rotation_matrix();
        let r_gt = handeye_gt.rotation.to_rotation_matrix();
        let r_diff = r_final.transpose() * r_gt;
        let angle = ((r_diff.matrix().trace() - 1.0) * 0.5)
            .clamp(-1.0, 1.0)
            .acos();

        assert!(
            t_err < 1e-2,
            "handeye translation error too large: {}",
            t_err
        );
        assert!(angle < 1e-2, "handeye rotation error too large: {}", angle);
    }
}
