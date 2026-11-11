//! The rig stages every rig problem type opens with: each camera's
//! intrinsics on its own views (initialize, then refine), then the rig
//! extrinsics (initialize, then the rig bundle adjustment).
//!
//! A problem type opts in by implementing [`RigStages`]; its `step_*`
//! functions delegate here and keep their own documentation, state beyond
//! [`RigStagesState`] and the later stages.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, DistortionParams, FxFyCxCy, Iso3, Real, RigDataset, ScheimpflugParams, SensorKind,
};
use vision_calibration_linear::extrinsics::estimate_extrinsics_from_cam_target_poses;
use vision_calibration_optim::{
    PlanarIntrinsicsParams, RigExtrinsicsEstimate, RigExtrinsicsParams, RigExtrinsicsSolveOptions,
    optimize_rig_extrinsics,
};

use super::tilted::TiltedRecovery;
use super::{
    bootstrap_rig_intrinsics, camera_views, check_camera_models, guard_percam_reproj_errors,
    inverse_poses, rig_ba_fix, views_to_planar_dataset, without_meta,
};
use crate::Error;
use crate::common::config::{CameraConfig, IntrinsicsInitConfig, RigConfig, SolverConfig};
use crate::common::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, RigInitOptions, RigOptimizeOptions,
};
use crate::planar_family::{Provenance, optimize_camera};
use crate::session::{CalibrationSession, ProblemType};

// ─────────────────────────────────────────────────────────────────────────────
// Manual seeds and step results
// ─────────────────────────────────────────────────────────────────────────────

/// Manual seeds for the **per-camera intrinsics stage** of a rig
/// calibration.
///
/// All fields are `Option<T>`. When `per_cam_intrinsics` is `Some`, Zhang's
/// auto-fit is skipped for *every* camera; per-camera target poses are then
/// recovered from per-camera homographies using the seeded intrinsics. When
/// `None`, each camera is auto-initialized.
///
/// When intrinsics are seeded but distortion is not, distortion starts at
/// zero; when intrinsics are auto-initialized, the fitted distortion is used
/// unless distortion is seeded. Each distortion seed is a value in its
/// camera's own distortion model: its kind must equal that camera's
/// configured distortion kind (a typed error otherwise) and it is used as
/// given.
///
/// `per_cam_sensors` seeds the tilt of each camera with a Scheimpflug
/// sensor (a seeded tilt skips the tilt sweep of the linear stage); a camera
/// with an untilted sensor takes only a zero tilt.
///
/// Per-camera vectors must have one entry per camera.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct IntrinsicsManualInit {
    /// Per-camera intrinsics seeds. `None` runs Zhang's method per camera.
    pub per_cam_intrinsics: Option<Vec<FxFyCxCy<Real>>>,
    /// Per-camera distortion seeds, each of the kind its camera's model names.
    pub per_cam_distortion: Option<Vec<DistortionParams>>,
    /// Per-camera sensor tilt seeds (Scheimpflug cameras; zero for the
    /// others).
    #[serde(default)]
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
}

/// Manual seeds for the **rig extrinsics stage**.
///
/// The camera-from-rig and rig-from-target poses are geometrically coupled,
/// so they are seeded together as [`RigPoses`] or not at all.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigManualInit {
    /// The seeded poses. `None` runs the linear extrinsics fit.
    pub poses: Option<RigPoses>,
}

/// Seeded rig poses: one `T_C_R` per camera and one `T_R_T` per view.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigPoses {
    /// Per-camera `T_C_R` (camera-from-rig); one per camera.
    pub cam_se3_rig: Vec<Iso3>,
    /// Per-view `T_R_T` (rig-from-target); one per view.
    pub rig_se3_target: Vec<Iso3>,
}

impl RigPoses {
    /// Seed the rig with `cam_se3_rig` (one pose per camera) and
    /// `rig_se3_target` (one pose per view).
    pub fn new(cam_se3_rig: Vec<Iso3>, rig_se3_target: Vec<Iso3>) -> Self {
        Self {
            cam_se3_rig,
            rig_se3_target,
        }
    }
}

/// Typed return value of the per-camera intrinsics initialization of a rig
/// (`step_intrinsics_init_all`, `step_intrinsics_init_all_with_seed`).
///
/// The inner `Option<Iso3>` in `cam_se3_target` reflects real
/// per-(view, camera) detection coverage — a camera may not have observed the
/// target in a given view.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsInitResult {
    /// Per-camera initial parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera target poses: `[view][cam] -> Option<Iso3>` (`cam_se3_target`).
    /// Inner `None` marks views where that camera did not observe the target.
    pub cam_se3_target: Vec<Vec<Option<Iso3>>>,
}

/// Typed return value of the rig extrinsics initialization (`step_rig_init`,
/// `step_rig_init_with_seed`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigInitResult {
    /// Initial per-camera `T_C_R` — camera-from-rig (reference camera is identity).
    pub initial_cam_se3_rig: Vec<Iso3>,
    /// Initial per-view `T_R_T` — rig-from-target.
    pub initial_rig_se3_target: Vec<Iso3>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Problem types with rig stages
// ─────────────────────────────────────────────────────────────────────────────

/// The session state of the rig stages, embedded (flattened) in the state of
/// each rig problem type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RigStagesState {
    /// Per-camera parameters, each in its camera's model.
    pub cameras: Option<Vec<CameraParams>>,
    /// Per-camera target poses: `[view][cam] -> Option<Iso3>`.
    /// `cam_se3_target` (T_C_T) for each camera in each view.
    pub cam_se3_target: Option<Vec<Vec<Option<Iso3>>>>,
    /// Per-camera mean reprojection error from intrinsics calibration.
    pub per_cam_reproj_errors: Option<Vec<f64>>,
    /// True when the current per-camera intrinsics came from full auto-init.
    #[serde(default)]
    pub per_cam_intrinsics_auto: bool,
    /// Initial camera-from-rig transforms: `cam_se3_rig` (T_C_R) per camera.
    /// Reference camera has identity.
    pub initial_cam_se3_rig: Option<Vec<Iso3>>,
    /// Initial rig-from-target poses: `rig_se3_target` (T_R_T) per view.
    pub initial_rig_se3_target: Option<Vec<Iso3>>,
    /// Mean reprojection error after rig BA.
    pub rig_ba_reproj_error: Option<f64>,
    /// Mean reprojection error after rig BA, per camera.
    pub rig_ba_per_cam_reproj_errors: Option<Vec<f64>>,
}

impl RigStagesState {
    /// Check if per-camera intrinsics have been computed.
    pub fn has_per_cam_intrinsics(&self) -> bool {
        self.cameras.is_some()
    }

    /// Check if rig initialization has been run.
    pub fn has_rig_init(&self) -> bool {
        self.initial_cam_se3_rig.is_some() && self.initial_rig_se3_target.is_some()
    }
}

/// A problem type whose pipeline opens with the rig stages: its input is a
/// rig dataset, its config names each camera's model and the stage
/// settings, and its state embeds a [`RigStagesState`].
pub(crate) trait RigStages: ProblemType {
    /// The per-view metadata of the input rig dataset.
    type Meta;

    /// The input as a rig dataset.
    fn dataset(input: &Self::Input) -> &RigDataset<Self::Meta>;
    /// The configuration of camera `index`.
    fn camera_for(config: &Self::Config, index: usize) -> CameraConfig;
    /// The per-camera linear-initialization settings.
    fn intrinsics_config(config: &Self::Config) -> &IntrinsicsInitConfig;
    /// The rig frame options.
    fn rig_config(config: &Self::Config) -> &RigConfig;
    /// The non-linear solve settings.
    fn solver_config(config: &Self::Config) -> &SolverConfig;
    /// The rig stages' state.
    fn stages(state: &Self::State) -> &RigStagesState;
    /// The rig stages' state, mutably.
    fn stages_mut(state: &mut Self::State) -> &mut RigStagesState;
}

/// The refined cameras of the intrinsics stage.
#[derive(Debug, Clone)]
pub(crate) struct RigCamerasEstimate {
    /// Per-camera refined parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera mean reprojection error in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-camera target poses refined alongside: `[view][cam]`.
    pub cam_se3_target: Vec<Vec<Option<Iso3>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Stages
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize every camera from its seeds or its own views
/// ([`bootstrap_rig_intrinsics`]), each in its model.
///
/// # Errors
///
/// Those of [`bootstrap_rig_intrinsics`], after the session's validation.
pub(crate) fn init_intrinsics<P: RigStages>(
    session: &mut CalibrationSession<P>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    session.validate()?;
    let input = P::dataset(session.require_input()?);
    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let num_cameras = input.num_cameras;
    let camera_configs: Vec<CameraConfig> =
        (0..num_cameras).map(|i| P::camera_for(config, i)).collect();

    let bootstrap = match bootstrap_rig_intrinsics(
        &camera_configs,
        input.num_views(),
        |cam_idx| camera_views(input, cam_idx),
        manual,
        P::intrinsics_config(config),
        opts.iterations,
    ) {
        Ok(b) => b,
        Err(e) => {
            session.log_failure("intrinsics_init_all", e.to_string());
            return Err(e);
        }
    };

    let stages = P::stages_mut(&mut session.state);
    stages.cameras = Some(bootstrap.cameras.clone());
    stages.cam_se3_target = Some(bootstrap.cam_se3_target.clone());
    stages.per_cam_intrinsics_auto = bootstrap.provenance.all_auto();

    session.log_success_with_notes(
        "intrinsics_init_all",
        format!(
            "initialized {num_cameras} cameras {}",
            bootstrap.provenance.source()
        ),
    );

    Ok(IntrinsicsInitResult {
        cameras: bootstrap.cameras,
        cam_se3_target: bootstrap.cam_se3_target,
    })
}

/// Refine each camera on its own views, in its model, holding its `fix`
/// lists, with every pose free: [`optimize_camera`], or after a full
/// auto-init the [`TiltedRecovery`] of the tilted cameras.
///
/// # Errors
///
/// - The intrinsics stage was not initialized, or the camera models changed
///   since.
/// - A camera's solve fails or diverges ([`guard_percam_reproj_errors`]).
pub(crate) fn optimize_intrinsics<P: RigStages>(
    session: &mut CalibrationSession<P>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<RigCamerasEstimate, Error> {
    session.validate()?;
    let input = P::dataset(session.require_input()?);
    let stages = P::stages(&session.state);
    if !stages.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_init_all first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let solver = P::solver_config(config);

    let cameras = stages
        .cameras
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    check_camera_models(&cameras, |i| P::camera_for(config, i).model)?;
    let mut cam_se3_target = stages
        .cam_se3_target
        .clone()
        .ok_or_else(|| Error::not_available("per-camera target poses"))?;
    // A tilted camera's own staged solve can settle in a wrong tilt /
    // principal-point basin when nothing was seeded; seeds anchor it.
    let recover_tilted = stages.per_cam_intrinsics_auto;

    let backend_opts =
        solver.step_backend_options(session.control(), opts.max_iters, opts.verbosity);

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);
    let mut tilted = TiltedRecovery::default();

    for (cam_idx, camera) in cameras.iter().enumerate() {
        let camera_config = P::camera_for(config, cam_idx);
        let cam_views = camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let initial_poses: Vec<Iso3> = valid_indices
            .iter()
            .map(|&global_idx| {
                cam_se3_target[global_idx][cam_idx]
                    .ok_or_else(|| Error::not_available("initial pose for camera view"))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let initial_params =
            PlanarIntrinsicsParams::new(camera.clone(), initial_poses).map_err(|e| {
                Error::numerical(format!("failed to build params for camera {cam_idx}: {e}"))
            })?;

        let result = if recover_tilted && camera_config.model.sensor == SensorKind::Scheimpflug {
            tilted.solve(
                cam_idx,
                &planar_dataset,
                &valid_indices,
                &initial_params,
                &camera_config,
                solver,
                P::intrinsics_config(config),
                backend_opts.clone(),
            )?
        } else {
            optimize_camera(
                cam_idx,
                &planar_dataset,
                &initial_params,
                &camera_config,
                solver,
                backend_opts.clone(),
            )?
        };

        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            cam_se3_target[global_idx][cam_idx] = Some(result.params.cam_se3_target[local_idx]);
        }
        optimized_cameras.push(result.params.camera);
        per_cam_reproj_errors.push(result.mean_reproj_error);
    }

    tilted.recover(
        &mut optimized_cameras,
        &mut per_cam_reproj_errors,
        &mut cam_se3_target,
        backend_opts,
        P::rig_config(config).reference_camera_idx,
    )?;

    // Fail fast on any diverged camera before its garbage intrinsics poison
    // the linear rig init and the joint bundle adjustments.
    guard_percam_reproj_errors(&per_cam_reproj_errors)?;

    let stages = P::stages_mut(&mut session.state);
    stages.cameras = Some(optimized_cameras.clone());
    stages.cam_se3_target = Some(cam_se3_target.clone());
    stages.per_cam_reproj_errors = Some(per_cam_reproj_errors.clone());

    let avg_error: f64 =
        per_cam_reproj_errors.iter().sum::<f64>() / per_cam_reproj_errors.len() as f64;
    session.log_success_with_notes(
        "intrinsics_optimize_all",
        format!("avg_reproj_err={avg_error:.3}px"),
    );

    Ok(RigCamerasEstimate {
        cameras: optimized_cameras,
        per_cam_reproj_errors,
        cam_se3_target,
    })
}

/// Initialize the rig extrinsics from the seeds, or by the linear fit of
/// the per-camera target poses with the reference camera as the rig frame.
///
/// # Errors
///
/// - The intrinsics stage was not run.
/// - The seeds break the coupling rule (one of `cam_se3_rig`,
///   `rig_se3_target` without the other) or their lengths differ from the
///   camera or view count.
/// - The linear fit fails.
pub(crate) fn init_rig<P: RigStages>(
    session: &mut CalibrationSession<P>,
    manual: RigManualInit,
    opts: Option<RigInitOptions>,
) -> Result<RigInitResult, Error> {
    let RigInitOptions {} = opts.unwrap_or_default();
    session.validate()?;
    let input = P::dataset(session.require_input()?);
    if !P::stages(&session.state).has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_optimize_all first)",
        ));
    }

    let num_views = input.num_views();
    let num_cameras = input.num_cameras;
    let reference_camera_idx = P::rig_config(&session.config).reference_camera_idx;

    let mut provenance = Provenance::default();
    let seeded = match manual.poses {
        Some(RigPoses {
            cam_se3_rig,
            rig_se3_target,
        }) => {
            if cam_se3_rig.len() != num_cameras {
                Err(format!(
                    "manual cam_se3_rig length ({}) does not match num_cameras ({num_cameras})",
                    cam_se3_rig.len()
                ))
            } else if rig_se3_target.len() != num_views {
                Err(format!(
                    "manual rig_se3_target length ({}) does not match num_views ({num_views})",
                    rig_se3_target.len()
                ))
            } else {
                Ok(Some((cam_se3_rig, rig_se3_target)))
            }
        }
        None => Ok(None),
    };
    let seeded = match seeded {
        Ok(seeded) => seeded,
        Err(msg) => {
            session.log_failure("rig_init", msg.clone());
            return Err(Error::invalid_input(msg));
        }
    };
    provenance.note("cam_se3_rig", seeded.is_some());
    provenance.note("rig_se3_target", seeded.is_some());

    let (cam_se3_rig, rig_se3_target) = match seeded {
        Some(seeded) => seeded,
        None => {
            let cam_se3_target = P::stages(&session.state)
                .cam_se3_target
                .as_ref()
                .ok_or_else(|| Error::not_available("per-camera target poses"))?;
            let extrinsic_result =
                estimate_extrinsics_from_cam_target_poses(cam_se3_target, reference_camera_idx)
                    .map_err(|e| {
                        Error::numerical(format!("rig extrinsics initialization failed: {e}"))
                    })?;
            (
                inverse_poses(&extrinsic_result.rig_se3_cam),
                extrinsic_result.rig_se3_target,
            )
        }
    };

    let stages = P::stages_mut(&mut session.state);
    stages.initial_cam_se3_rig = Some(cam_se3_rig.clone());
    stages.initial_rig_se3_target = Some(rig_se3_target.clone());

    session.log_success_with_notes(
        "rig_init",
        format!(
            "ref_cam={reference_camera_idx}, {num_views} views {}",
            provenance.source()
        ),
    );

    Ok(RigInitResult {
        initial_cam_se3_rig: cam_se3_rig,
        initial_rig_se3_target: rig_se3_target,
    })
}

/// The rig bundle adjustment from the initial rig: per-camera extrinsics
/// and per-view rig poses, and the cameras as `rig.refine_intrinsics` /
/// `rig.refine_sensor` say. The reference camera alone removes the rig's
/// 6-DOF gauge; no rig-from-target pose is fixed.
///
/// Records the mean and per-camera reprojection errors in the stage state;
/// the caller stores the estimate and logs [`rig_ba_notes`].
///
/// # Errors
///
/// - The rig was not initialized, or the camera models changed since.
/// - The solve fails.
pub(crate) fn optimize_rig<P: RigStages>(
    session: &mut CalibrationSession<P>,
    opts: Option<RigOptimizeOptions>,
) -> Result<RigExtrinsicsEstimate, Error> {
    session.validate()?;
    let input = P::dataset(session.require_input()?);
    let stages = P::stages(&session.state);
    if !stages.has_rig_init() {
        return Err(Error::not_available(
            "rig initialization (call step_rig_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let rig = P::rig_config(config);
    let solver = P::solver_config(config);

    let cameras = stages
        .cameras
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    check_camera_models(&cameras, |i| P::camera_for(config, i).model)?;
    let cam_se3_rig = stages
        .initial_cam_se3_rig
        .as_ref()
        .ok_or_else(|| Error::not_available("initial cam_se3_rig"))?;
    let rig_se3_target = stages
        .initial_rig_se3_target
        .clone()
        .ok_or_else(|| Error::not_available("initial rig_se3_target"))?;

    let num_cameras = input.num_cameras;
    let fix = (0..num_cameras)
        .map(|i| {
            rig_ba_fix(
                &P::camera_for(config, i),
                rig.refine_intrinsics,
                rig.refine_sensor,
            )
        })
        .collect();
    let fix_extrinsics: Vec<bool> = (0..num_cameras)
        .map(|i| i == rig.reference_camera_idx)
        .collect();

    let initial = RigExtrinsicsParams {
        cameras,
        rig_se3_cam: inverse_poses(cam_se3_rig),
        rig_se3_target,
    };
    let solve_opts = RigExtrinsicsSolveOptions {
        robust_loss: solver.robust_loss,
        fix,
        fix_extrinsics,
        // The reference-camera fix removes the full 6-DOF rig gauge on its
        // own; pinning a rig-from-target pose as well is redundant and
        // slightly worsens the fit (`docs/notes/rig-extrinsics.md` §Gauge).
        fix_rig_poses: Vec::new(),
    };
    let backend_opts =
        solver.step_backend_options(session.control(), opts.max_iters, opts.verbosity);
    let output =
        match optimize_rig_extrinsics(without_meta(input), initial, solve_opts, backend_opts) {
            Ok(r) => r,
            Err(e) => {
                session.log_failure("rig_optimize", e.to_string());
                return Err(Error::from(e));
            }
        };

    let stages = P::stages_mut(&mut session.state);
    stages.rig_ba_reproj_error = Some(output.mean_reproj_error);
    stages.rig_ba_per_cam_reproj_errors = Some(output.per_cam_reproj_errors.clone());
    Ok(output)
}

/// The log note of the rig bundle adjustment.
pub(crate) fn rig_ba_notes(output: &RigExtrinsicsEstimate) -> String {
    format!(
        "final_cost={:.2e}, mean_reproj_err={:.3}px",
        output.report.final_cost, output.mean_reproj_error
    )
}
