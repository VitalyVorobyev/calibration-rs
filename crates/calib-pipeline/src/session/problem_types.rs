//! Problem type implementations for the calibration session framework.

use super::ProblemType;
use crate::handeye_single::{
    init_handeye, init_intrinsics, optimize_handeye_stage, optimize_intrinsics,
    ransac_planar_poses, BackendSolveOptions, HandEyeMode, HandEyeSolveOptions, HandEyeStage,
    HandEyeView, IntrinsicsStage, IterativeIntrinsicsOptions, PlanarIntrinsicsSolveOptions,
    PoseRansacOptions, PoseRansacStage, RobustLoss,
};
use crate::{
    optimize_planar_intrinsics_with_init, pinhole_camera_params, planar_init_seed_from_views,
    PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarIntrinsicsReport, PlanarViewData,
};
use anyhow::{ensure, Result};
use calib_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt3, Real, Vec2,
};
use serde::{Deserialize, Serialize};

/// Planar intrinsics calibration problem (Zhang's method with distortion).
///
/// Estimates camera intrinsics (fx, fy, cx, cy, skew) and Brown-Conrady distortion
/// (k1, k2, k3, p1, p2) from observations of a planar calibration pattern.
pub struct PlanarIntrinsicsProblem;

/// Observations for planar intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsObservations {
    pub views: Vec<PlanarViewData>,
}

/// Initial values from linear initialization (iterative Zhang's method).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitial {
    /// Initial intrinsics and distortion estimates (for compatibility).
    pub report: PlanarIntrinsicsReport,
    /// Serialized seed for nonlinear refinement.
    #[serde(default)]
    pub init: Option<PlanarIntrinsicsInitState>,
}

/// Optimized results from non-linear refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimized {
    /// Final calibration report with refined parameters.
    pub report: PlanarIntrinsicsReport,
    /// Refined board-to-camera poses.
    pub poses: Vec<Iso3>,
    /// Mean reprojection error after optimization (pixels).
    pub mean_reproj_error: Real,
}

/// Serializable seed for planar intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitState {
    pub intrinsics: FxFyCxCySkew<Real>,
    pub distortion: BrownConrady5<Real>,
    pub poses: Vec<Iso3>,
}

impl PlanarIntrinsicsInitState {
    pub fn from_seed(seed: &calib_optim::planar_intrinsics::PlanarIntrinsicsInit) -> Self {
        Self {
            intrinsics: FxFyCxCySkew {
                fx: seed.intrinsics.fx,
                fy: seed.intrinsics.fy,
                cx: seed.intrinsics.cx,
                cy: seed.intrinsics.cy,
                skew: 0.0,
            },
            distortion: BrownConrady5 {
                k1: seed.distortion.k1,
                k2: seed.distortion.k2,
                k3: seed.distortion.k3,
                p1: seed.distortion.p1,
                p2: seed.distortion.p2,
                iters: 8,
            },
            poses: seed.poses.clone(),
        }
    }

    pub fn to_seed(&self) -> Result<calib_optim::planar_intrinsics::PlanarIntrinsicsInit> {
        calib_optim::planar_intrinsics::PlanarIntrinsicsInit::new(
            self.intrinsics,
            self.distortion,
            self.poses.clone(),
        )
    }
}

/// Options for linear initialization.
///
/// Currently uses default options from the existing pipeline.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitOptions {
    /// Placeholder for future init-specific options.
    #[serde(skip)]
    _private: (),
}

/// Options for non-linear optimization.
///
/// Wraps the existing PlanarIntrinsicsConfig.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimOptions {
    #[serde(flatten)]
    pub config: PlanarIntrinsicsConfig,
}

impl ProblemType for PlanarIntrinsicsProblem {
    type Observations = PlanarIntrinsicsObservations;
    type InitialValues = PlanarIntrinsicsInitial;
    type OptimizedResults = PlanarIntrinsicsOptimized;
    type InitOptions = PlanarIntrinsicsInitOptions;
    type OptimOptions = PlanarIntrinsicsOptimOptions;

    fn problem_name() -> &'static str {
        "planar_intrinsics"
    }

    fn initialize(
        obs: &Self::Observations,
        _opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        let (seed, cam) = planar_init_seed_from_views(&obs.views)?;
        let init_state = PlanarIntrinsicsInitState::from_seed(&seed);

        let report = PlanarIntrinsicsReport {
            camera: pinhole_camera_params(&cam),
            final_cost: 0.0,
        };

        Ok(PlanarIntrinsicsInitial {
            report,
            init: Some(init_state),
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        let input = PlanarIntrinsicsInput {
            views: obs.views.clone(),
        };

        let dataset = crate::build_planar_dataset(&input)?;
        let seed = if let Some(state) = init.init.as_ref() {
            state.to_seed()?
        } else {
            planar_init_seed_from_views(&input.views)?.0
        };

        let result = optimize_planar_intrinsics_with_init(dataset, seed, &opts.config)?;

        let mean_reproj_error = mean_reproj_error_planar(
            &obs.views,
            &result.camera.k,
            &result.camera.dist,
            &result.poses,
        )?;

        let report = PlanarIntrinsicsReport {
            camera: pinhole_camera_params(&result.camera),
            final_cost: result.final_cost,
        };

        Ok(PlanarIntrinsicsOptimized {
            report,
            poses: result.poses,
            mean_reproj_error,
        })
    }
}

fn mean_reproj_error_planar(
    views: &[PlanarViewData],
    intrinsics: &FxFyCxCySkew<Real>,
    distortion: &BrownConrady5<Real>,
    poses: &[Iso3],
) -> Result<Real> {
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

//─────────────────────────────────────────────────────────────────────────────
// HandEyeSingleProblem: Single-camera hand-eye calibration
//─────────────────────────────────────────────────────────────────────────────

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
#[derive(Debug, Clone)]
pub struct HandEyeSingleInitOptions {
    pub intrinsics_init_opts: IterativeIntrinsicsOptions,
}

impl Default for HandEyeSingleInitOptions {
    fn default() -> Self {
        Self {
            intrinsics_init_opts: IterativeIntrinsicsOptions::default(),
        }
    }
}

// Manual Serialize/Deserialize to satisfy session trait bounds.
impl Serialize for HandEyeSingleInitOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let state = serializer.serialize_struct("HandEyeSingleInitOptions", 0)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HandEyeSingleInitOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionsVisitor;
        impl<'de> serde::de::Visitor<'de> for OptionsVisitor {
            type Value = HandEyeSingleInitOptions;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct HandEyeSingleInitOptions")
            }

            fn visit_map<A>(self, mut _map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                Ok(HandEyeSingleInitOptions::default())
            }
        }

        deserializer.deserialize_struct("HandEyeSingleInitOptions", &[], OptionsVisitor)
    }
}

/// Options for the hand-eye single-camera pipeline.
#[derive(Debug, Clone)]
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
            fix_k3: true,
            ..Default::default()
        };

        let mut handeye_solve_opts = HandEyeSolveOptions {
            robust_loss: RobustLoss::Huber { scale: 2.0 },
            fix_fx: true,
            fix_fy: true,
            fix_cx: true,
            fix_cy: true,
            fix_k1: true,
            fix_k2: true,
            fix_k3: true,
            fix_p1: true,
            fix_p2: true,
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

// Manual Serialize/Deserialize to satisfy session trait bounds.
impl Serialize for HandEyeSingleOptimOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let state = serializer.serialize_struct("HandEyeSingleOptimOptions", 0)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HandEyeSingleOptimOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionsVisitor;
        impl<'de> serde::de::Visitor<'de> for OptionsVisitor {
            type Value = HandEyeSingleOptimOptions;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct HandEyeSingleOptimOptions")
            }

            fn visit_map<A>(self, mut _map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                Ok(HandEyeSingleOptimOptions::default())
            }
        }

        deserializer.deserialize_struct("HandEyeSingleOptimOptions", &[], OptionsVisitor)
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

//─────────────────────────────────────────────────────────────────────────────
// RigExtrinsicsProblem: Multi-camera rig calibration
//─────────────────────────────────────────────────────────────────────────────

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

/// Observations from one camera in one view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraViewData {
    pub points_3d: Vec<Pt3>,
    pub points_2d: Vec<Vec2>,
    pub weights: Option<Vec<f64>>,
}

/// Initial values from rig extrinsics initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsInitial {
    /// Per-camera calibration reports.
    pub per_camera_calibrations: Vec<PlanarIntrinsicsReport>,
    /// Camera-to-rig transforms.
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view rig-to-target poses.
    pub rig_to_target: Vec<Iso3>,
}

/// Optimized results from rig extrinsics bundle adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsOptimized {
    /// Per-camera calibrated parameters.
    pub cameras: Vec<calib_core::CameraParams>,
    /// Camera-to-rig transforms.
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view rig-to-target poses.
    pub rig_to_target: Vec<Iso3>,
    /// Final optimization cost.
    pub final_cost: f64,
}

/// Options for rig extrinsics initialization.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsInitOptions {
    /// Reference camera index (defines rig frame). Default: 0.
    #[serde(default)]
    pub ref_cam_idx: usize,
}

/// Options for rig extrinsics optimization.
#[derive(Debug, Clone)]
pub struct RigExtrinsicsOptimOptions {
    pub solve_opts: calib_optim::problems::rig_extrinsics::RigExtrinsicsSolveOptions,
    pub backend_opts: calib_optim::backend::BackendSolveOptions,
}

// Manual Serialize/Deserialize to handle non-serializable inner types
impl Serialize for RigExtrinsicsOptimOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let state = serializer.serialize_struct("RigExtrinsicsOptimOptions", 0)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RigExtrinsicsOptimOptions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OptionsVisitor;
        impl<'de> serde::de::Visitor<'de> for OptionsVisitor {
            type Value = RigExtrinsicsOptimOptions;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct RigExtrinsicsOptimOptions")
            }

            fn visit_map<A>(self, mut _map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                Ok(RigExtrinsicsOptimOptions::default())
            }
        }

        deserializer.deserialize_struct("RigExtrinsicsOptimOptions", &[], OptionsVisitor)
    }
}

impl Default for RigExtrinsicsOptimOptions {
    fn default() -> Self {
        Self {
            solve_opts: calib_optim::problems::rig_extrinsics::RigExtrinsicsSolveOptions {
                fix_rig_poses: vec![0], // Fix first view for gauge freedom
                ..Default::default()
            },
            backend_opts: Default::default(),
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
        use calib_linear::{homography, planar_pose, zhang_intrinsics};

        let num_cameras = obs.num_cameras;
        let num_views = obs.views.len();

        anyhow::ensure!(num_cameras > 0, "need at least one camera");
        anyhow::ensure!(num_views > 0, "need at least one view");

        // Step 1: Per-camera planar intrinsics calibration
        let mut per_camera_calibrations = Vec::with_capacity(num_cameras);
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

            // Compute homographies for this camera
            let mut homographies = Vec::with_capacity(cam_views.len());
            for (view_idx, cam_data) in cam_views {
                let board_2d: Vec<calib_core::Pt2> = cam_data
                    .points_3d
                    .iter()
                    .map(|p| calib_core::Pt2::new(p.x, p.y))
                    .collect();

                let pixel_2d: Vec<calib_core::Pt2> = cam_data
                    .points_2d
                    .iter()
                    .map(|v| calib_core::Pt2::new(v.x, v.y))
                    .collect();

                let h = homography::dlt_homography(&board_2d, &pixel_2d).with_context(|| {
                    format!("view {} homography failed for camera {}", view_idx, cam_idx)
                })?;
                homographies.push((*view_idx, h));
            }

            anyhow::ensure!(
                homographies.len() >= 3,
                "camera {} needs at least 3 views for Zhang initialization (got {})",
                cam_idx,
                homographies.len()
            );

            let homography_mats: Vec<calib_core::Mat3> =
                homographies.iter().map(|(_, h)| *h).collect();
            let intrinsics =
                zhang_intrinsics::estimate_intrinsics_from_homographies(&homography_mats)
                    .with_context(|| {
                        format!("Zhang initialization failed for camera {}", cam_idx)
                    })?;

            let k = calib_core::Mat3::new(
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

            // Zero distortion seed; refined in joint BA
            let dist = calib_core::DistortionParams::BrownConrady5 {
                params: calib_core::BrownConrady5 {
                    k1: 0.0,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            };

            let cam_cfg = calib_core::CameraParams {
                projection: calib_core::ProjectionParams::Pinhole,
                distortion: dist,
                sensor: calib_core::SensorParams::Identity,
                intrinsics: calib_core::IntrinsicsParams::FxFyCxCySkew { params: intrinsics },
            };

            per_camera_calibrations.push(PlanarIntrinsicsReport {
                camera: cam_cfg,
                final_cost: 0.0,
            });

            // Compute per-view camera-to-target poses from homographies
            for (view_idx, h) in homographies.iter() {
                let pose = planar_pose::estimate_planar_pose_from_h(&k, h).with_context(|| {
                    format!(
                        "failed to recover pose for view {} camera {}",
                        view_idx, cam_idx
                    )
                })?;
                cam_target_poses[*view_idx][cam_idx] = Some(pose);
            }
        }

        // Step 2: Estimate rig extrinsics from camera-target poses
        let extrinsics = calib_linear::extrinsics::estimate_extrinsics_from_cam_target_poses(
            &cam_target_poses,
            opts.ref_cam_idx,
        )?;

        Ok(RigExtrinsicsInitial {
            per_camera_calibrations,
            cam_to_rig: extrinsics.cam_to_rig,
            rig_to_target: extrinsics.rig_to_target,
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        use calib_optim::problems::rig_extrinsics::*;

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
            .per_camera_calibrations
            .iter()
            .map(|report| match &report.camera.intrinsics {
                calib_core::IntrinsicsParams::FxFyCxCySkew { params } => *params,
            })
            .collect();

        let distortion: Vec<BrownConrady5<Real>> = init
            .per_camera_calibrations
            .iter()
            .map(|report| {
                if let calib_core::DistortionParams::BrownConrady5 { params } =
                    &report.camera.distortion
                {
                    *params
                } else {
                    unreachable!("planar intrinsics always returns BrownConrady5")
                }
            })
            .collect();

        let initial = RigExtrinsicsInit {
            intrinsics,
            distortion,
            cam_to_rig: init.cam_to_rig.clone(),
            rig_to_target: init.rig_to_target.clone(),
        };

        // Run optimization
        let result = calib_optim::problems::rig_extrinsics::optimize_rig_extrinsics(
            dataset,
            initial,
            opts.solve_opts.clone(),
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
            rig_to_target: result.rig_to_target,
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
    fn planar_intrinsics_problem_full_pipeline() {
        // Generate synthetic calibration data
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
        let cam_gt = make_pinhole_camera(k_gt, dist_gt);

        // Generate checkerboard points
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
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let pose = Iso3::from_parts(translation.into(), rotation);

            let mut points_2d = Vec::new();
            for pw in &board_points {
                let pc = pose.transform_point(pw);
                let proj = cam_gt.project_point(&pc).unwrap();
                points_2d.push(Vec2::new(proj.x, proj.y));
            }

            views.push(PlanarViewData {
                points_3d: board_points.clone(),
                points_2d,
                weights: None,
            });
        }

        // Create session and run pipeline
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
            "Synthetic planar intrinsics test".to_string(),
        );

        let obs = PlanarIntrinsicsObservations { views };
        session.set_observations(obs);

        // Initialize
        let init_result = session.initialize(PlanarIntrinsicsInitOptions::default());
        assert!(init_result.is_ok(), "Initialization failed");
        assert_eq!(
            session.stage(),
            super::super::SessionStage::Initialized,
            "Should be in Initialized stage"
        );

        // Optimize
        let optim_result = session.optimize(PlanarIntrinsicsOptimOptions::default());
        assert!(optim_result.is_ok(), "Optimization failed");
        assert_eq!(
            session.stage(),
            super::super::SessionStage::Optimized,
            "Should be in Optimized stage"
        );

        // Export
        let export_result = session.export();
        assert!(export_result.is_ok(), "Export failed");
        assert_eq!(
            session.stage(),
            super::super::SessionStage::Exported,
            "Should be in Exported stage"
        );

        let final_report = export_result.unwrap().report;
        assert!(
            final_report.final_cost < 1e-6,
            "Final cost too high: {}",
            final_report.final_cost
        );

        // Verify intrinsics are close to ground truth
        let cam_cfg = &final_report.camera;
        let k_est = match &cam_cfg.intrinsics {
            calib_core::IntrinsicsParams::FxFyCxCySkew { params } => FxFyCxCySkew {
                fx: params.fx,
                fy: params.fy,
                cx: params.cx,
                cy: params.cy,
                skew: 0.0,
            },
        };

        assert!((k_est.fx - k_gt.fx).abs() < 20.0, "fx estimate off");
        assert!((k_est.fy - k_gt.fy).abs() < 20.0, "fy estimate off");
        assert!((k_est.cx - k_gt.cx).abs() < 20.0, "cx estimate off");
        assert!((k_est.cy - k_gt.cy).abs() < 20.0, "cy estimate off");
    }

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
        for robot_pose in &robot_poses {
            let mut points_3d = Vec::new();
            let mut points_2d = Vec::new();
            for pw in &board_points {
                let p_base = target_pose.transform_point(pw);
                let p_gripper = robot_pose.inverse_transform_point(&p_base);
                let p_rig = handeye_gt.inverse_transform_point(&p_gripper);
                let p_cam = cam_to_rig.inverse_transform_point(&p_rig);
                if let Some(pixel) = cam_gt.project_point(&p_cam) {
                    points_3d.push(*pw);
                    points_2d.push(Vec2::new(pixel.x, pixel.y));
                }
            }

            views.push(HandEyeView {
                view: PlanarViewData {
                    points_3d,
                    points_2d,
                    weights: None,
                },
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

    #[test]
    fn planar_intrinsics_session_json_checkpoint() {
        // Create a simple session with observations
        let views = vec![PlanarViewData {
            points_3d: vec![
                Pt3::new(0.0, 0.0, 0.0),
                Pt3::new(0.05, 0.0, 0.0),
                Pt3::new(0.05, 0.05, 0.0),
                Pt3::new(0.0, 0.05, 0.0),
            ],
            points_2d: vec![
                Vec2::new(100.0, 100.0),
                Vec2::new(200.0, 100.0),
                Vec2::new(200.0, 200.0),
                Vec2::new(100.0, 200.0),
            ],
            weights: None,
        }];

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_observations(PlanarIntrinsicsObservations { views });

        // Serialize
        let json = session.to_json().unwrap();
        assert!(json.contains("planar_intrinsics"));
        assert!(json.contains("Uninitialized"));

        // Deserialize
        let restored: CalibrationSession<PlanarIntrinsicsProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(restored.stage(), super::super::SessionStage::Uninitialized);
        assert_eq!(restored.observations().unwrap().views.len(), 1);
    }

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
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let rig_to_target = Iso3::from_parts(translation.into(), rotation);

            // Compute camera poses
            let cam0_to_target = cam0_to_rig * rig_to_target;
            let cam1_to_target = cam1_to_rig * rig_to_target;

            // Project through cameras
            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();

            for pw in &board_points {
                let pc0 = cam0_to_target.transform_point(pw);
                let pc1 = cam1_to_target.transform_point(pw);

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
        session.set_observations(RigExtrinsicsObservations {
            views,
            num_cameras: 2,
        });

        // Initialize
        let init_result = session.initialize(RigExtrinsicsInitOptions { ref_cam_idx: 0 });
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
        assert!(
            final_result.final_cost < 1e-4,
            "Final cost too high: {}",
            final_result.final_cost
        );

        // Verify cam0_to_rig is identity (reference camera)
        let cam0_rig_trans = final_result.cam_to_rig[0].translation.vector.norm();
        assert!(
            cam0_rig_trans < 0.01,
            "cam0 translation not identity: {}",
            cam0_rig_trans
        );

        // Verify cam1_to_rig is reasonable (loose tolerance for initialization + optimization)
        // Linear initialization can have ~20% error without distortion
        let cam1_rig_trans_error =
            (final_result.cam_to_rig[1].translation.vector - cam1_to_rig.translation.vector).norm();
        assert!(
            cam1_rig_trans_error < 0.25,
            "cam1 translation error too large: {}",
            cam1_rig_trans_error
        );
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
            let rig_to_target = Iso3::from_parts(translation.into(), rotation);

            let cam0_to_target = cam0_to_rig * rig_to_target;
            let cam1_to_target = cam1_to_rig * rig_to_target;

            let mut cam0_pixels = Vec::new();
            let mut cam1_pixels = Vec::new();
            for pw in &board_points {
                let pc0 = cam0_to_target.transform_point(pw);
                let pc1 = cam1_to_target.transform_point(pw);
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
            .initialize(RigExtrinsicsInitOptions { ref_cam_idx: 0 })
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
            .initialize(RigExtrinsicsInitOptions { ref_cam_idx: 0 })
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
        assert_eq!(restored.stage(), super::super::SessionStage::Uninitialized);
        assert_eq!(restored.observations().unwrap().views.len(), 1);
        assert_eq!(restored.observations().unwrap().num_cameras, 2);
    }
}
