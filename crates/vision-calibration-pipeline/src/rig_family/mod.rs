//! Shared helpers for multi-camera rig calibration workflows.
//!
//! Internal to the pipeline crate. Holds the rig stages the rig problem
//! types open with ([`stages`]: each camera's intrinsics on its own views,
//! then the rig extrinsics), the recovery of tilted cameras those stages
//! run ([`tilted`]), and the rig-level helpers of the rig, hand-eye and
//! laser workflows: the export residuals, the hand-eye initialization and
//! solve options. The per-camera bootstrap and refinement each camera runs
//! live in [`crate::planar_family`]. Each rig camera has its own model
//! ([`CameraModelSpec`]): the rig's `camera` config, or a per-camera
//! override, so one rig can mix lenses.
//!
//! The step sequences beyond the rig stages, the residual blocks and the
//! export shapes stay per problem module.

mod stages;
mod tilted;

pub use stages::{IntrinsicsInitResult, IntrinsicsManualInit, RigInitResult, RigManualInit};
pub(crate) use stages::{
    RigStages, RigStagesState, init_intrinsics, init_rig, optimize_intrinsics, optimize_rig,
    rig_ba_notes,
};

use crate::Error;
use crate::common::config::{CameraConfig, IntrinsicsInitConfig, RobotPoseConfig};
use crate::planar_family::{
    CameraSeed, Provenance, bootstrap_camera, check_distortion_seed, check_sensor_seed,
};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraModel, CameraModelSpec, CameraParams, DistortionParams,
    FeatureResidualHistogram, FxFyCxCy, Iso3, NoMeta, PerFeatureResiduals, PlanarDataset,
    ProjectionParams, Real, RigDataset, RigView, ScheimpflugParams, SensorKind, SensorParams, View,
    build_feature_histogram, compute_rig_target_residuals,
};
use vision_calibration_linear::handeye::{
    HandeyeMethod, estimate_eye_in_hand, estimate_eye_to_hand,
};
use vision_calibration_optim::{
    HandeyeMode, HandeyeSolveOptions, PlanarIntrinsicsParams, RobustLoss,
};

/// Bootstrap result for the per-camera intrinsics stage.
#[derive(Debug, Clone)]
pub(crate) struct RigIntrinsicsBootstrap {
    /// Per-camera initial parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// `[view][cam] -> Option<Iso3>`, `cam_se3_target` (T_C_T).
    pub per_cam_target_poses: Vec<Vec<Option<Iso3>>>,
    /// Which seeds were given, for the session log.
    pub provenance: Provenance,
}

/// Per-camera rig intrinsics bootstrap shared by all rig workflows.
///
/// Loops over cameras, extracts per-camera planar views via the supplied
/// closure, and either uses seeded intrinsics or runs the iterative auto-fit
/// (Zhang's method; for a Scheimpflug sensor, the tilted variant that also
/// estimates the tilt). The linear stage fits Brown-Conrady coefficients,
/// embedded into each camera's distortion model (extra coefficients start
/// at zero) unless that camera's distortion is seeded in its own model, with
/// `init`'s options resolved for each camera
/// ([`IntrinsicsInitConfig::iterative_opts`]: only the terms the camera's
/// solves estimate are fitted; `iterations` overrides `init.iterations`).
/// Per-camera target poses are recovered from per-camera
/// homographies using whichever intrinsics were chosen.
///
/// # Errors
///
/// - Any per-camera seed disagrees with the camera count, or a
///   non-zero tilt seed is given for an untilted camera.
/// - A camera has fewer than 3 valid views.
/// - DLT homography or pose recovery fails on any view.
/// - Iterative intrinsics estimation fails (auto-fit only).
pub(crate) fn bootstrap_rig_intrinsics<F>(
    camera_configs: &[CameraConfig],
    num_views: usize,
    extract_views: F,
    seeds: IntrinsicsManualInit,
    init: &IntrinsicsInitConfig,
    iterations: Option<usize>,
) -> Result<RigIntrinsicsBootstrap, Error>
where
    F: Fn(usize) -> Vec<Option<View<NoMeta>>>,
{
    let models: Vec<CameraModelSpec> = camera_configs.iter().map(|c| c.model).collect();
    let num_cameras = models.len();
    validate_seeds(&seeds, &models)?;
    let provenance = seed_provenance(&seeds, &models);

    let mut cameras = Vec::with_capacity(num_cameras);
    let mut per_cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];

    for (cam_idx, model) in models.iter().enumerate() {
        let cam_views = extract_views(cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let seed = CameraSeed {
            intrinsics: seeds.per_cam_intrinsics.as_ref().map(|s| s[cam_idx]),
            distortion: seeds
                .per_cam_distortion
                .as_ref()
                .map(|s| s[cam_idx].clone()),
            sensor: seeds.per_cam_sensors.as_ref().map(|s| s[cam_idx]),
        };
        let init_opts = init.camera_init_opts(iterations, &camera_configs[cam_idx]);
        let camera_bootstrap =
            bootstrap_camera(cam_idx, *model, &planar_dataset, &seed, init_opts)?;

        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            per_cam_target_poses[global_idx][cam_idx] = Some(camera_bootstrap.poses[local_idx]);
        }

        cameras.push(camera_bootstrap.camera);
    }

    Ok(RigIntrinsicsBootstrap {
        cameras,
        per_cam_target_poses,
        provenance,
    })
}

fn validate_seeds(seeds: &IntrinsicsManualInit, models: &[CameraModelSpec]) -> Result<(), Error> {
    let num_cameras = models.len();
    if let Some(s) = &seeds.per_cam_intrinsics
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_intrinsics length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }
    if let Some(s) = &seeds.per_cam_distortion
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_distortion length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }
    if let Some(s) = &seeds.per_cam_distortion {
        for (cam_idx, (seed, model)) in s.iter().zip(models).enumerate() {
            check_distortion_seed(seed, *model, &format!("per_cam_distortion[{cam_idx}]"))?;
        }
    }
    if let Some(s) = &seeds.per_cam_sensors {
        if s.len() != num_cameras {
            return Err(Error::invalid_input(format!(
                "per_cam_sensors length ({}) != num_cameras ({})",
                s.len(),
                num_cameras
            )));
        }
        for (cam_idx, (seed, model)) in s.iter().zip(models).enumerate() {
            check_sensor_seed(*model, seed, &format!("per_cam_sensors[{cam_idx}]"))?;
        }
    }
    Ok(())
}

fn seed_provenance(seeds: &IntrinsicsManualInit, models: &[CameraModelSpec]) -> Provenance {
    let mut provenance = Provenance::default();
    provenance.note("per_cam_intrinsics", seeds.per_cam_intrinsics.is_some());
    provenance.note("per_cam_distortion", seeds.per_cam_distortion.is_some());
    if models.iter().any(|m| m.sensor == SensorKind::Scheimpflug) {
        provenance.note("per_cam_sensors", seeds.per_cam_sensors.is_some());
    }
    provenance
}

/// Build a planar dataset from per-view per-camera observations, returning
/// the dataset and the original (global) view indices it covers.
///
/// `views[i] = None` means the camera does not observe the target in view `i`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if fewer than 3 views have observations.
pub(crate) fn views_to_planar_dataset(
    views: &[Option<View<NoMeta>>],
) -> Result<(PlanarDataset, Vec<usize>), Error> {
    let (valid_views, indices): (Vec<_>, Vec<_>) = views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|view| (view.clone(), i)))
        .unzip();

    if valid_views.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: valid_views.len(),
        });
    }
    let dataset = PlanarDataset::new(valid_views)?;
    Ok((dataset, indices))
}

/// Check the cameras in state still have the models the config names.
///
/// # Errors
///
/// [`Error::InvalidInput`] naming the first camera whose model changed.
pub(crate) fn check_camera_models(
    cameras: &[CameraParams],
    model_for: impl Fn(usize) -> CameraModelSpec,
) -> Result<(), Error> {
    for (i, camera) in cameras.iter().enumerate() {
        let configured = model_for(i);
        if camera.spec() != Some(configured) {
            return Err(Error::invalid_input(format!(
                "camera {i}'s model changed since step_intrinsics_init_all (now {configured:?}); \
                 run it again"
            )));
        }
    }
    Ok(())
}

/// The fixed parameters of a rig camera in the rig bundle adjustment.
///
/// Intrinsics, projection and distortion follow the camera's `fix` lists
/// when `refine_intrinsics`, and are all held otherwise; a Scheimpflug
/// tilt follows `fix.sensor` when `refine_sensor`, and is held otherwise.
pub(crate) fn rig_ba_fix(
    camera: &CameraConfig,
    refine_intrinsics: bool,
    refine_sensor: bool,
) -> CameraFix {
    let all = CameraFix::all(camera.model);
    let mut fix = if refine_intrinsics {
        camera.fix.clone()
    } else {
        all.clone()
    };
    fix.sensor = if refine_sensor {
        camera.fix.sensor.clone()
    } else {
        all.sensor
    };
    fix
}

/// The runtime models of exported rig cameras.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a camera whose sensor homography is singular
/// (reachable only from a hand-edited export).
pub(crate) fn rig_camera_models(cameras: &[CameraParams]) -> Result<Vec<CameraModel>, Error> {
    cameras
        .iter()
        .enumerate()
        .map(|(i, c)| {
            c.build()
                .map_err(|e| Error::invalid_input(format!("camera {i}: {e}")))
        })
        .collect()
}

/// Planar-solve parameters of a tilted rig camera.
///
/// # Errors
///
/// [`Error`] when `camera_se3_target` is empty.
pub(crate) fn tilted_planar_params(
    intrinsics: FxFyCxCy<Real>,
    distortion: BrownConrady5<Real>,
    sensor: ScheimpflugParams,
    camera_se3_target: Vec<Iso3>,
) -> Result<PlanarIntrinsicsParams, Error> {
    Ok(PlanarIntrinsicsParams::new(
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 { params: distortion },
            sensor: SensorParams::Scheimpflug { params: sensor },
            intrinsics,
        },
        camera_se3_target,
    )?)
}

/// The Brown-Conrady coefficients of a camera (zero for any other
/// distortion).
pub(crate) fn rig_distortion(camera: &CameraParams) -> BrownConrady5<Real> {
    match &camera.distortion {
        DistortionParams::BrownConrady5 { params } => *params,
        _ => BrownConrady5::default(),
    }
}

/// The sensor tilt of a camera (zero for a sensor without a tilt).
pub(crate) fn rig_tilt(camera: &CameraParams) -> ScheimpflugParams {
    match camera.sensor {
        SensorParams::Scheimpflug { params } => params,
        _ => ScheimpflugParams::default(),
    }
}

/// Maximum per-camera mean reprojection error (px) tolerated before a camera's
/// intrinsics solve is treated as diverged. Healthy per-camera intrinsics land
/// well under a pixel; this generous ceiling only trips on a genuinely broken
/// solve — non-finite, or the `fx → 0` runaway that reaches astronomical
/// reprojection error.
pub(crate) const MAX_PERCAM_REPROJ_ERROR_PX: f64 = 50.0;

/// Fail fast if any camera's per-camera intrinsics solve diverged, naming the
/// offenders. Without this guard a single garbage camera (non-finite or runaway
/// reprojection error) silently poisons the linear rig init and the joint rig +
/// hand-eye bundle adjustment downstream.
///
/// # Errors
///
/// Returns [`Error::Numerical`] listing every camera whose error is non-finite
/// or exceeds [`MAX_PERCAM_REPROJ_ERROR_PX`].
pub(crate) fn guard_percam_reproj_errors(errors: &[f64]) -> Result<(), Error> {
    let bad: Vec<String> = errors
        .iter()
        .enumerate()
        .filter(|(_, e)| !e.is_finite() || **e > MAX_PERCAM_REPROJ_ERROR_PX)
        .map(|(i, e)| format!("camera {i}: {e:.3} px"))
        .collect();
    if !bad.is_empty() {
        return Err(Error::numerical(format!(
            "per-camera intrinsics diverged for {} of {} cameras ({}); \
             refusing to poison the rig solve — check detections/init for these cameras",
            bad.len(),
            errors.len(),
            bad.join(", "),
        )));
    }
    Ok(())
}

/// The views of camera `cam_idx` of a rig dataset, one per rig view: `None`
/// where the camera does not observe the target.
pub(crate) fn camera_views<M>(
    dataset: &RigDataset<M>,
    cam_idx: usize,
) -> Vec<Option<View<NoMeta>>> {
    dataset
        .views
        .iter()
        .map(|view| {
            view.obs
                .cameras
                .get(cam_idx)
                .and_then(|opt_obs| opt_obs.as_ref())
                .map(|obs| View::without_meta(obs.clone()))
        })
        .collect()
}

/// A rig dataset's observations without their per-view metadata.
pub(crate) fn without_meta<M>(dataset: &RigDataset<M>) -> RigDataset<NoMeta> {
    RigDataset {
        num_cameras: dataset.num_cameras,
        views: dataset
            .views
            .iter()
            .map(|v| RigView {
                meta: NoMeta,
                obs: v.obs.clone(),
            })
            .collect(),
    }
}

/// The inverse of each pose: `cam_to_rig` from `cam_se3_rig` and back.
pub(crate) fn inverse_poses(poses: &[Iso3]) -> Vec<Iso3> {
    poses.iter().map(|t| t.inverse()).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Export diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// One residual histogram per camera, of the values `values(camera)`.
pub(crate) fn per_camera_histograms<I>(
    num_cameras: usize,
    values: impl Fn(usize) -> I,
) -> Vec<FeatureResidualHistogram>
where
    I: IntoIterator<Item = f64>,
{
    (0..num_cameras)
        .map(|cam_idx| build_feature_histogram(values(cam_idx)))
        .collect()
}

/// The per-feature target residuals of a rig calibration, with their
/// per-camera histograms: each camera projects the target at
/// `cam_se3_rig[cam] * rig_se3_target[view]` through its model.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a camera whose model does not build, and
/// any error of [`compute_rig_target_residuals`].
pub(crate) fn rig_target_residuals<M>(
    cameras: &[CameraParams],
    dataset: &RigDataset<M>,
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> Result<PerFeatureResiduals, Error> {
    let target = compute_rig_target_residuals(
        &rig_camera_models(cameras)?,
        dataset,
        cam_se3_rig,
        rig_se3_target,
    )?;
    let mut residuals = PerFeatureResiduals::default();
    residuals.target_hist_per_camera = Some(per_camera_histograms(dataset.num_cameras, |cam| {
        target
            .iter()
            .filter(move |r| r.camera == cam)
            .filter_map(|r| r.error_px)
    }));
    residuals.target = target;
    Ok(residuals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hand-eye
// ─────────────────────────────────────────────────────────────────────────────

/// The closed-form hand-eye estimate of `mode` by `method`: the hand-eye
/// transform and the mode's fixed target pose, from the robot poses and
/// the observer's (a camera's, or a rig's) target poses.
///
/// - `EyeInHand`: (`gripper_se3_observer`, `base_se3_target`).
/// - `EyeToHand`: (`observer_se3_base`, `gripper_se3_target`).
///
/// # Errors
///
/// [`Error::Numerical`] when the linear solve fails.
pub(crate) fn linear_handeye(
    mode: HandeyeMode,
    method: HandeyeMethod,
    base_se3_gripper: &[Iso3],
    observer_se3_target: &[Iso3],
    min_motion_angle_deg: f64,
) -> Result<(Iso3, Iso3), Error> {
    match mode {
        HandeyeMode::EyeInHand => estimate_eye_in_hand(
            method,
            base_se3_gripper,
            observer_se3_target,
            min_motion_angle_deg,
        )
        .map(|e| (e.gripper_se3_camera, e.base_se3_target))
        .map_err(|e| Error::numerical(format!("linear hand-eye estimation failed: {e}"))),
        HandeyeMode::EyeToHand => estimate_eye_to_hand(
            method,
            base_se3_gripper,
            observer_se3_target,
            min_motion_angle_deg,
        )
        .map(|e| (e.camera_se3_base, e.gripper_se3_target))
        .map_err(|e| Error::numerical(format!("linear gripper->target estimation failed: {e}"))),
    }
}

/// The hand-eye bundle adjustment of one fixed target with the hand-eye
/// transform free, each camera holding `fix`, the extrinsics held where
/// `fix_extrinsics`, and the robot poses refined as `robot_poses` says.
pub(crate) fn handeye_solve_opts(
    robust_loss: RobustLoss,
    fix: Vec<CameraFix>,
    fix_extrinsics: Vec<bool>,
    robot_poses: &RobotPoseConfig,
) -> HandeyeSolveOptions {
    HandeyeSolveOptions {
        robust_loss,
        fix,
        fix_extrinsics,
        fix_handeye: false,
        fix_target_poses: Vec::new(),
        relax_target_poses: false,
        refine_robot_poses: robot_poses.refine,
        robot_rot_sigma: robot_poses.rot_sigma,
        robot_trans_sigma: robot_poses.trans_sigma,
        estimate_robot_sigma: robot_poses.estimate_sigma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, Camera, DistortionKind, FxFyCxCy, Pinhole, ScheimpflugParams, View,
        make_pinhole_camera, synthetic::planar,
    };

    /// `[cam][view] -> Option<View<NoMeta>>`.
    type PerCamViews = Vec<Vec<Option<View<NoMeta>>>>;

    fn pinhole_two_camera_views() -> (PerCamViews, FxFyCxCy<Real>, usize) {
        let cam_gt = make_pinhole_camera(
            FxFyCxCy {
                fx: 900.0,
                fy: 880.0,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5::default(),
        );
        let board = planar::grid_points(6, 5, 0.04);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.1, 0.6, 0.05);
        let views_a = planar::project_views_all(&cam_gt, &board, &poses).expect("views a");
        let views_b = planar::project_views_all(&cam_gt, &board, &poses).expect("views b");
        let num_views = views_a.len();
        let per_cam_views: PerCamViews = vec![
            views_a
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
            views_b
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
        ];
        (per_cam_views, cam_gt.k, num_views)
    }

    fn scheimpflug_two_camera_views(
        sensor: ScheimpflugParams,
    ) -> (PerCamViews, FxFyCxCy<Real>, usize) {
        let base = make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5::default(),
        );
        let camera = Camera::new(Pinhole, base.dist, sensor.compile(), base.k);
        let board = planar::grid_points(6, 5, 0.03);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.08, 0.55, 0.03);
        let views_a = planar::project_views_all(&camera, &board, &poses).expect("views a");
        let views_b = planar::project_views_all(&camera, &board, &poses).expect("views b");
        let num_views = views_a.len();
        let per_cam_views: PerCamViews = vec![
            views_a
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
            views_b
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
        ];
        (per_cam_views, base.k, num_views)
    }

    fn untilted() -> CameraConfig {
        CameraConfig::new(CameraModelSpec::pinhole(
            DistortionKind::BrownConrady5,
            SensorKind::Identity,
        ))
    }

    fn tilted() -> CameraConfig {
        CameraConfig::new(CameraModelSpec::pinhole(
            DistortionKind::BrownConrady5,
            SensorKind::Scheimpflug,
        ))
    }

    #[test]
    fn bootstrap_pinhole_recovers_synthetic_intrinsics() {
        let (per_cam_views, k_gt, num_views) = pinhole_two_camera_views();
        let result = bootstrap_rig_intrinsics(
            &[untilted(), untilted()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            IntrinsicsManualInit::default(),
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect("bootstrap");

        assert_eq!(result.cameras.len(), 2);
        assert!(
            result
                .cameras
                .iter()
                .all(|c| c.spec() == Some(untilted().model))
        );
        assert_eq!(result.per_cam_target_poses.len(), num_views);
        for cam in &result.cameras {
            let k = cam.intrinsics;
            assert!((k.fx - k_gt.fx).abs() / k_gt.fx < 0.05, "fx off");
            assert!((k.fy - k_gt.fy).abs() / k_gt.fy < 0.05, "fy off");
        }
        assert!(result.provenance.all_auto());
        assert_eq!(
            result.provenance.auto(),
            ["per_cam_intrinsics", "per_cam_distortion"]
        );
    }

    #[test]
    fn bootstrap_scheimpflug_recovers_synthetic_intrinsics() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let (per_cam_views, k_gt, num_views) = scheimpflug_two_camera_views(sensor_gt);
        let result = bootstrap_rig_intrinsics(
            &[tilted(), tilted()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            IntrinsicsManualInit::default(),
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect("bootstrap");

        assert_eq!(result.cameras.len(), 2);
        let sensors: Vec<ScheimpflugParams> = result.cameras.iter().map(rig_tilt).collect();
        assert!(
            result
                .cameras
                .iter()
                .all(|c| c.spec() == Some(tilted().model))
        );
        // Auto-fitted sensors are estimated from the Scheimpflug-aware bootstrap.
        for s in &sensors {
            assert!(s.tilt_x.is_finite());
            assert!(s.tilt_y.is_finite());
            assert!(s.tilt_x.abs() < 0.05);
            assert!(s.tilt_y.abs() < 0.05);
        }
        for cam in &result.cameras {
            // Scheimpflug-projected views are still close to pinhole at small tilts.
            let k = cam.intrinsics;
            assert!(k.fx.is_finite());
            assert!(k.fy.is_finite());
            assert!((k.cx - k_gt.cx).abs() < k_gt.cx * 0.2);
        }
        assert_eq!(
            result.provenance.auto(),
            [
                "per_cam_intrinsics",
                "per_cam_distortion",
                "per_cam_sensors"
            ]
        );
    }

    #[test]
    fn bootstrap_holds_a_tilted_cameras_tangential_terms_by_default() {
        let sensor = ScheimpflugParams {
            tilt_x: 0.05,
            tilt_y: -0.03,
        };
        let (per_cam_views, _, num_views) = scheimpflug_two_camera_views(sensor);
        let tangential = |camera: CameraConfig| -> Vec<(Real, Real)> {
            bootstrap_rig_intrinsics(
                &[camera.clone(), camera],
                num_views,
                |cam_idx| per_cam_views[cam_idx].clone(),
                IntrinsicsManualInit {
                    per_cam_sensors: Some(vec![sensor, sensor]),
                    ..Default::default()
                },
                &IntrinsicsInitConfig::default(),
                None,
            )
            .expect("bootstrap")
            .cameras
            .iter()
            .map(|c| match c.distortion {
                DistortionParams::BrownConrady5 { params } => (params.p1, params.p2),
                _ => unreachable!("Brown-Conrady camera"),
            })
            .collect()
        };
        // Held at zero: the camera's default fix list keeps them there in
        // every later solve, so a fitted value would stay frozen.
        assert!(tangential(tilted()).iter().all(|&p| p == (0.0, 0.0)));
        // Once the tilt is held (here at its seed), a camera whose solves
        // estimate them gets them fitted.
        let mut free = tilted();
        free.fix.distortion = Some(Vec::new());
        free.fix.sensor = Some(vec!["tilt_x".into(), "tilt_y".into()]);
        assert!(tangential(free).iter().any(|&p| p != (0.0, 0.0)));
    }

    #[test]
    fn bootstrap_with_intrinsics_seed_skips_autofit() {
        let (per_cam_views, k_gt, num_views) = pinhole_two_camera_views();
        let seeds = IntrinsicsManualInit {
            per_cam_intrinsics: Some(vec![k_gt, k_gt]),
            ..Default::default()
        };
        let result = bootstrap_rig_intrinsics(
            &[untilted(), untilted()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            seeds,
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect("bootstrap");

        for cam in &result.cameras {
            assert_eq!(cam.intrinsics.fx, k_gt.fx);
            assert_eq!(cam.intrinsics.fy, k_gt.fy);
        }
        assert_eq!(result.provenance.manual(), ["per_cam_intrinsics"]);
        assert_eq!(result.provenance.auto(), ["per_cam_distortion"]);
    }

    #[test]
    fn bootstrap_scheimpflug_seeded_sensors_used_verbatim() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let (per_cam_views, _, num_views) = scheimpflug_two_camera_views(sensor_gt);
        let seeds = IntrinsicsManualInit {
            per_cam_sensors: Some(vec![sensor_gt, sensor_gt]),
            ..Default::default()
        };
        let result = bootstrap_rig_intrinsics(
            &[tilted(), tilted()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            seeds,
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect("bootstrap");

        for s in result.cameras.iter().map(rig_tilt) {
            assert_eq!(s.tilt_x, sensor_gt.tilt_x);
            assert_eq!(s.tilt_y, sensor_gt.tilt_y);
        }
        assert!(result.provenance.manual().contains(&"per_cam_sensors"));
    }

    #[test]
    fn bootstrap_rejects_seed_length_mismatch() {
        let (per_cam_views, k_gt, num_views) = pinhole_two_camera_views();
        let seeds = IntrinsicsManualInit {
            per_cam_intrinsics: Some(vec![k_gt]), // wrong length: 1 vs 2 cameras
            ..Default::default()
        };
        let err = bootstrap_rig_intrinsics(
            &[untilted(), untilted()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            seeds,
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect_err("seed length mismatch should error");
        let msg = err.to_string();
        assert!(msg.contains("per_cam_intrinsics"), "got: {msg}");
    }

    #[test]
    fn bootstrap_rejects_insufficient_views() {
        let (per_cam_views, _, num_views) = pinhole_two_camera_views();
        // Only present the first view; cam_idx 0 will have <3 valid views.
        let truncated: PerCamViews = vec![
            vec![per_cam_views[0][0].clone(), None, None, None, None],
            per_cam_views[1].clone(),
        ];
        let err = bootstrap_rig_intrinsics(
            &[untilted(), untilted()],
            num_views,
            |cam_idx| truncated[cam_idx].clone(),
            IntrinsicsManualInit::default(),
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect_err("insufficient views should error");
        let msg = err.to_string();
        assert!(msg.contains("camera 0"), "got: {msg}");
    }

    #[test]
    fn bootstrap_mixes_lenses_per_camera() {
        let (per_cam_views, _, num_views) = pinhole_two_camera_views();
        let division = CameraConfig::new(CameraModelSpec::pinhole(
            DistortionKind::Division1,
            SensorKind::Identity,
        ));
        let result = bootstrap_rig_intrinsics(
            &[untilted(), division.clone()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            IntrinsicsManualInit::default(),
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect("bootstrap");
        assert_eq!(result.cameras[0].spec(), Some(untilted().model));
        assert_eq!(result.cameras[1].spec(), Some(division.model));
        // Only a Scheimpflug camera reports a tilt seed.
        assert!(!result.provenance.auto().contains(&"per_cam_sensors"));
    }

    #[test]
    fn tilt_seed_for_an_untilted_camera_must_be_zero() {
        let (per_cam_views, _, num_views) = pinhole_two_camera_views();
        let zero = IntrinsicsManualInit {
            per_cam_sensors: Some(vec![ScheimpflugParams::default(); 2]),
            ..Default::default()
        };
        assert!(
            bootstrap_rig_intrinsics(
                &[untilted(), untilted()],
                num_views,
                |cam_idx| per_cam_views[cam_idx].clone(),
                zero,
                &IntrinsicsInitConfig::default(),
                None,
            )
            .is_ok()
        );
        let tilted_seed = IntrinsicsManualInit {
            per_cam_sensors: Some(vec![
                ScheimpflugParams::default(),
                ScheimpflugParams {
                    tilt_x: 0.05,
                    tilt_y: 0.0,
                },
            ]),
            ..Default::default()
        };
        let err = bootstrap_rig_intrinsics(
            &[untilted(), untilted()],
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            tilted_seed,
            &IntrinsicsInitConfig::default(),
            None,
        )
        .expect_err("a tilt seed for an untilted camera is rejected");
        assert!(err.to_string().contains("per_cam_sensors[1]"), "{err}");
    }

    #[test]
    fn rig_ba_fix_follows_the_stage_flags() {
        let camera = tilted();
        let held = rig_ba_fix(&camera, false, false);
        let resolved = held.resolve(camera.model).unwrap();
        assert_eq!(resolved.intrinsics, vec![0, 1, 2, 3]);
        assert_eq!(resolved.sensor, vec![0, 1]);
        let refined = rig_ba_fix(&camera, true, true)
            .resolve(camera.model)
            .unwrap();
        assert!(refined.intrinsics.is_empty());
        assert_eq!(refined.distortion, vec![2, 3, 4]);
        assert!(refined.sensor.is_empty());
    }

    #[test]
    fn percam_guard_passes_healthy_cameras() {
        assert!(guard_percam_reproj_errors(&[0.31, 0.28, 0.27, 0.30]).is_ok());
    }

    #[test]
    fn percam_guard_rejects_nonfinite_camera_naming_it() {
        let err = guard_percam_reproj_errors(&[0.3, f64::INFINITY, 0.2])
            .expect_err("a non-finite camera must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("camera 1"),
            "error should name camera 1: {msg}"
        );
    }

    #[test]
    fn percam_guard_rejects_runaway_camera() {
        // The classic `fx → 0` divergence reaches astronomical reprojection error.
        let err = guard_percam_reproj_errors(&[0.3, 0.28, 6.1e21])
            .expect_err("a runaway camera must be rejected");
        assert!(err.to_string().contains("camera 2"));
    }
}
