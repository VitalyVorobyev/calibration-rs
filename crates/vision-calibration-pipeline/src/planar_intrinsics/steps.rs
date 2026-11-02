//! Step functions for planar intrinsics calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<PlanarIntrinsicsProblem>` to perform calibration.
//! They serve every camera model the config names; a tilted (Scheimpflug)
//! sensor adds its own initialization and a staged solve, described at
//! [`step_init_with_seed`] and [`step_optimize`].
//!
//! # Example
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, step_init, step_optimize, step_filter,
//!     IntrinsicsFilterOptions,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//!
//! // Run initialization
//! step_init(&mut session, None)?;
//!
//! // Run optimization
//! step_optimize(&mut session, None)?;
//!
//! // Optionally filter outliers and re-run
//! step_filter(&mut session, None)?;
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Export result
//! let export = session.export()?;
//! # Ok(())
//! # }
//! ```

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, CameraProject, CorrespondenceView, DistortionKind,
    DistortionParams, FxFyCxCy, Iso3, Mat3, PlanarDataset, ProjectionKind, ProjectionParams, Real,
    ScheimpflugParams, SensorKind, SensorParams, View,
};
use vision_calibration_linear::scheimpflug_init::{
    ScheimpflugIntrinsicsInitOptions as LinearScheimpflugIntrinsicsInitOptions,
    estimate_scheimpflug_intrinsics_iterative,
};
use vision_calibration_optim::{
    BackendSolveOptions, CameraBounds, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RobustLoss, ScheimpflugStagedInitOptions,
    optimize_planar_intrinsics, optimize_scheimpflug_intrinsics_staged, with_leading_radial,
};

use crate::planar_family::{
    CameraSeed, bootstrap_camera, bootstrap_planar_intrinsics, check_distortion_seed,
    estimate_view_homographies, recover_planar_poses_from_homographies, unsupported_projection,
    unsupported_sensor,
};
use crate::rig_family::format_init_source;
use crate::session::CalibrationSession;

use super::problem::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

use crate::common::{IntrinsicsInitOptions, IntrinsicsOptimizeOptions};

/// Manual initialization seeds for planar intrinsics calibration.
///
/// All fields are `Option<T>`:
/// - `None` means *auto-initialize this group* (same path as plain `step_init`).
/// - `Some(value)` means *use this value*; do not auto-initialize.
///
/// Partial-seed semantics:
/// - When `intrinsics` is `Some` but `poses` is `None`, poses are recovered from
///   homographies using the **manual** intrinsics (not auto-estimated ones). This keeps
///   the geometric chain consistent with the seed.
/// - When `intrinsics` is `Some` and `distortion` is `None`, distortion defaults to
///   `BrownConrady5::default()` (zeros) — the auto distortion fit is coupled with the
///   iterative intrinsics estimator and does not run when intrinsics are seeded.
/// - When `intrinsics` is `None`, the bootstrap auto-fit runs and any `Some` field
///   overrides the corresponding bootstrap output.
///
/// The distortion seed is a value in the camera's own distortion model: its
/// kind must equal `camera.model.distortion` (a typed error otherwise). It is
/// used as given — no coefficient is converted, and the tilted sensor's
/// tangential zeroing applies only to the auto-fit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsManualInit {
    /// Manual intrinsics seed. `None` means auto-init via Zhang's method.
    pub intrinsics: Option<FxFyCxCy<Real>>,
    /// Manual distortion seed, of the kind `camera.model.distortion` names. `None`
    /// means auto-init (or zeros when intrinsics are also seeded — see struct
    /// docs).
    pub distortion: Option<DistortionParams>,
    /// Manual sensor tilt, for a camera whose model has a Scheimpflug sensor
    /// (a typed error otherwise). `None` starts from the auto-fit tilt (zero
    /// when intrinsics are seeded). A seeded tilt is trusted: `step_optimize`
    /// refines it directly instead of running the multi-start tilt sweep.
    pub sensor: Option<ScheimpflugParams>,
    /// Manual per-view poses (camera-from-target). `None` means recover from
    /// homographies using whichever intrinsics are in effect.
    pub poses: Option<Vec<Iso3>>,
}

/// Options for filtering observations based on reprojection error.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct IntrinsicsFilterOptions {
    /// Maximum reprojection error threshold (pixels).
    /// Points with error above this are removed.
    pub max_reproj_error: f64,

    /// Minimum number of points per view after filtering.
    pub min_points_per_view: usize,

    /// If true, remove entire views that have fewer than `min_points_per_view`.
    pub remove_sparse_views: bool,
}

impl Default for IntrinsicsFilterOptions {
    fn default() -> Self {
        Self {
            max_reproj_error: 2.0,
            min_points_per_view: 4,
            remove_sparse_views: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

/// Typed return value of [`step_init`] / [`step_init_with_seed`].
///
/// Carries the seeded-or-fitted initial estimates. The same values are also
/// written into `session.state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsInitResult {
    /// Initial camera, in the configured model.
    pub camera: CameraParams,
    /// Initial per-view target poses (`camera_se3_target`).
    pub poses: Vec<Iso3>,
}

/// Typed return value of [`step_optimize`].
///
/// Aggregates the optimization metrics of the planar-intrinsics solve; the same
/// values are also written into `session.state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsOptimizeResult {
    /// Final cost reported by the non-linear solver.
    pub final_cost: f64,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
    /// Number of solver iterations executed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics, distortion, sensor tilt and per-view poses from any
/// combination of manual seeds and auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate that
/// passes `IntrinsicsManualInit::default()` (all-`None`, full auto path).
///
/// # Field-by-field behavior
///
/// - `intrinsics`: `Some` skips the auto-fit and uses the seed; `None` runs the
///   iterative auto-fit (Zhang's method; for a Scheimpflug sensor, the tilted
///   variant that also estimates the tilt; for an equidistant camera, the
///   bearing-based fisheye initializer).
/// - `distortion`: `Some` overrides the bootstrap's fitted distortion; `None` keeps
///   the bootstrap value (or zeros when intrinsics are seeded and the bootstrap
///   doesn't run). The seed's kind must equal the camera's distortion kind.
/// - `sensor`: Scheimpflug sensor only — see [`IntrinsicsManualInit::sensor`].
/// - `poses`: `Some` uses the seed verbatim (count must match view count); `None`
///   recovers poses from homographies (bearings for an equidistant camera) using
///   the in-effect intrinsics — manual when seeded, auto otherwise.
///
/// For a Scheimpflug sensor, an unseeded `intrinsics` (from-scratch auto-init) is
/// **experimental**: under the tilt↔focal↔distortion degeneracy it can settle in a
/// wrong tilt/focal basin, and the init log says so. Seed a coarse focal plus the
/// nominal mount tilt for stable results.
///
/// Updates `session.state` and clears any previous optimization results. Same
/// postcondition as `step_init`.
///
/// # Errors
///
/// - Input not set, or fewer than 3 views.
/// - `manual.sensor` is `Some` but the camera's sensor is not Scheimpflug.
/// - `manual.distortion` is `Some` and of another kind than the camera's
///   distortion model.
/// - `init.iterations == 0` when running the bootstrap auto-fit.
/// - Homography or auto-init computation fails.
/// - `manual.poses` is `Some` but its length does not match the view count.
pub fn step_init_with_seed(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    session.validate()?;
    let model = session.config.camera.model;
    if manual.sensor.is_some() && model.sensor != SensorKind::Scheimpflug {
        let msg = format!(
            "a sensor tilt seed needs a Scheimpflug camera; the configured sensor is {:?}",
            model.sensor
        );
        session.log_failure("init", msg.clone());
        return Err(Error::invalid_input(msg));
    }
    if let Some(distortion) = &manual.distortion
        && let Err(e) = check_distortion_seed(distortion, model, "distortion")
    {
        session.log_failure("init", e.to_string());
        return Err(e);
    }
    match (model.projection, model.sensor) {
        (ProjectionKind::Pinhole, SensorKind::Identity) => init_untilted(session, manual, opts),
        (ProjectionKind::Pinhole, SensorKind::Scheimpflug) => init_tilted(session, manual, opts),
        (ProjectionKind::Equidistant, _) => init_fisheye(session, manual),
        (ProjectionKind::Pinhole, other) => Err(unsupported_sensor(other)),
        (other, _) => Err(unsupported_projection(other)),
    }
}

/// Equidistant (fisheye) camera: the bearing-based initializer, or the
/// seeds, through the per-camera bootstrap every workflow shares.
fn init_fisheye(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    manual: IntrinsicsManualInit,
) -> Result<IntrinsicsInitResult, Error> {
    let dataset = session.require_input()?.clone();
    let model = session.config.camera.model;
    if let Some(p) = &manual.poses
        && p.len() != dataset.num_views()
    {
        let msg = format!(
            "manual poses count ({}) does not match view count ({})",
            p.len(),
            dataset.num_views()
        );
        session.log_failure("init", msg.clone());
        return Err(Error::invalid_input(msg));
    }
    let seed = CameraSeed {
        intrinsics: manual.intrinsics,
        distortion: manual.distortion,
        sensor: None,
    };
    let bootstrap = match bootstrap_camera(0, model, &dataset, &seed, session.config.init_opts()) {
        Ok(b) => b,
        Err(e) => {
            session.log_failure("init", e.to_string());
            return Err(e);
        }
    };
    let (mut manual_fields, mut auto_fields) = (Vec::new(), Vec::new());
    for (name, seeded) in [
        ("intrinsics", seed.intrinsics.is_some()),
        ("distortion", seed.distortion.is_some()),
        ("poses", manual.poses.is_some()),
    ] {
        if seeded {
            manual_fields.push(name);
        } else {
            auto_fields.push(name);
        }
    }
    let poses = manual.poses.unwrap_or(bootstrap.poses);
    let k = bootstrap.camera.intrinsics;
    record_init(session, bootstrap.camera, poses, false, None);
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1} {}",
            k.fx,
            k.fy,
            k.cx,
            k.cy,
            format_init_source(&manual_fields, &auto_fields)
        ),
    );
    init_result(session)
}

/// Store an initialization in the session state, clearing any previous
/// optimization.
fn record_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    camera: CameraParams,
    poses: Vec<Iso3>,
    sensor_manual: bool,
    homographies: Option<Vec<Mat3>>,
) {
    session.state.homographies = homographies;
    session.state.initial_camera = Some(camera);
    session.state.initial_sensor_manual = sensor_manual;
    session.state.initial_poses = Some(poses);
    session.state.clear_optimization();
}

/// The initialization just recorded, in the configured model.
fn init_result(
    session: &CalibrationSession<PlanarIntrinsicsProblem>,
) -> Result<IntrinsicsInitResult, Error> {
    let params = session
        .state
        .initial_params(session.config.camera.model)
        .ok_or_else(|| Error::numerical("the initial camera does not fit its poses"))?;
    Ok(IntrinsicsInitResult {
        camera: params.camera,
        poses: params.camera_se3_target,
    })
}

/// The distortion of a camera with seeded intrinsics and no distortion seed:
/// zero (state conversion expresses it in the configured kind).
fn zero_distortion() -> DistortionParams {
    fitted_distortion(BrownConrady5::default())
}

/// The linear stage's fitted Brown-Conrady coefficients; state conversion
/// embeds them in the configured kind.
fn fitted_distortion(fitted: BrownConrady5<Real>) -> DistortionParams {
    DistortionParams::BrownConrady5 { params: fitted }
}

/// Untilted camera: Zhang's method with iterative distortion.
fn init_untilted(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    let input = session.require_input()?;

    let opts = opts.unwrap_or_default();
    let mut init_opts = session.config.init_opts().pinhole;
    if let Some(iters) = opts.iterations {
        init_opts.iterations = iters;
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    let (intrinsics, distortion, homographies, poses) = if let Some(k) = manual.intrinsics {
        manual_fields.push("intrinsics");

        let dist = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                zero_distortion()
            }
        };

        let homographies = match estimate_view_homographies(input) {
            Ok(hs) => hs,
            Err(e) => {
                session.log_failure("init", e.to_string());
                return Err(e);
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != input.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        input.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                p
            }
            None => {
                auto_fields.push("poses");
                match recover_planar_poses_from_homographies(&homographies, &k) {
                    Ok(ps) => ps,
                    Err(e) => {
                        session.log_failure("init", e.to_string());
                        return Err(e);
                    }
                }
            }
        };

        (k, dist, homographies, poses)
    } else {
        auto_fields.push("intrinsics");

        let bootstrap = match bootstrap_planar_intrinsics(input, init_opts) {
            Ok(b) => b,
            Err(e) => {
                session.log_failure("init", e.to_string());
                return Err(e);
            }
        };

        let dist = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                fitted_distortion(bootstrap.camera.dist)
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != input.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        input.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                p
            }
            None => {
                auto_fields.push("poses");
                bootstrap.poses
            }
        };

        (bootstrap.camera.k, dist, bootstrap.homographies, poses)
    };

    let camera = CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion,
        sensor: SensorParams::Identity,
        intrinsics,
    };
    record_init(session, camera, poses, false, Some(homographies));

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1} {}",
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, source
        ),
    );

    init_result(session)
}

/// Tilted (Scheimpflug) camera: the tilted linear bootstrap, or the seeds.
fn init_tilted(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    let dataset = session.require_input()?.clone();

    let opts = opts.unwrap_or_default();
    let mut init_iterations = session.config.init.iterations;
    if let Some(iterations) = opts.iterations {
        init_iterations = iterations;
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    // A user-provided tilt seed is trusted directly by `step_optimize`.
    let sensor_manual = manual.sensor.is_some();
    // From-scratch (no intrinsics seed) Scheimpflug auto-init is experimental — it
    // can land in a wrong tilt/focal basin under the tilt↔focal↔distortion
    // degeneracy. The supported path seeds a coarse focal + nominal mount tilt.
    let intrinsics_seeded = manual.intrinsics.is_some();

    let (intrinsics, distortion, sensor, poses) = if let Some(k) = manual.intrinsics {
        manual_fields.push("intrinsics");

        let manual_dist = manual.distortion;

        let sensor = match manual.sensor {
            Some(s) => {
                manual_fields.push("sensor");
                s
            }
            None => {
                auto_fields.push("sensor");
                ScheimpflugParams::default()
            }
        };

        let (dist, poses) = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != dataset.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        dataset.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                // Manual poses: use manual or default distortion unchanged.
                if manual_dist.is_some() {
                    manual_fields.push("distortion");
                } else {
                    auto_fields.push("distortion");
                }
                (manual_dist.unwrap_or_else(zero_distortion), p)
            }
            None => {
                auto_fields.push("poses");
                // Recover initial poses from board→pixel homographies. Use the
                // seeded intrinsics as the K matrix for decomposition. Distortion
                // and tilt effects in the homography are absorbed by the biased
                // pose estimate; the joint optimizer in step_optimize corrects
                // them.
                let raw_homographies = estimate_view_homographies(&dataset).map_err(|e| {
                    Error::numerical(format!(
                        "scheimpflug intrinsics homography estimation failed: {e}"
                    ))
                })?;
                let poses =
                    recover_planar_poses_from_homographies(&raw_homographies, &k).map_err(|e| {
                        Error::numerical(format!(
                            "scheimpflug intrinsics pose recovery failed: {e}"
                        ))
                    })?;
                let dist = match manual_dist {
                    Some(d) => {
                        manual_fields.push("distortion");
                        d
                    }
                    None => {
                        auto_fields.push("distortion");
                        zero_distortion()
                    }
                };
                (dist, poses)
            }
        };

        (k, dist, sensor, poses)
    } else {
        if init_iterations == 0 {
            return Err(Error::invalid_input("init.iterations must be positive"));
        }
        auto_fields.push("intrinsics");

        let bootstrap = estimate_scheimpflug_intrinsics_iterative(
            &dataset,
            LinearScheimpflugIntrinsicsInitOptions {
                iterations: init_iterations,
                distortion_opts: session
                    .config
                    .init
                    .distortion_fit_opts(&session.config.camera),
                ..Default::default()
            },
        )
        .map_err(|e| {
            Error::numerical(format!("scheimpflug intrinsics initialization failed: {e}"))
        })?;

        let dist = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                fitted_distortion(bootstrap.camera.dist)
            }
        };

        let sensor = match manual.sensor {
            Some(s) => {
                manual_fields.push("sensor");
                s
            }
            None => {
                auto_fields.push("sensor");
                bootstrap.sensor
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != dataset.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        dataset.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                p
            }
            None => {
                auto_fields.push("poses");
                bootstrap.poses
            }
        };

        (bootstrap.camera.k, dist, sensor, poses)
    };

    let camera = CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion,
        sensor: SensorParams::Scheimpflug { params: sensor },
        intrinsics,
    };
    record_init(session, camera, poses, sensor_manual, None);

    let source = format_init_source(&manual_fields, &auto_fields);
    let experimental = if intrinsics_seeded {
        ""
    } else {
        " — WARNING: from-scratch Scheimpflug auto-init is experimental and may \
         converge to a wrong tilt/focal basin; seed a coarse focal + nominal mount \
         tilt via step_init_with_seed for stable results"
    };
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, views={} {}{}",
            intrinsics.fx,
            intrinsics.fy,
            dataset.num_views(),
            source,
            experimental
        ),
    );

    init_result(session)
}

/// Initialize intrinsics and poses from observations using full auto-init.
///
/// Convenience wrapper around [`step_init_with_seed`] with `IntrinsicsManualInit::default()`
/// (all-`None`). Auto-fits intrinsics + distortion via Zhang's method with iterative
/// distortion (and, for a Scheimpflug sensor, the tilt), then recovers poses from
/// homographies.
///
/// For a Scheimpflug sensor this from-scratch init is **experimental**; prefer
/// [`step_init_with_seed`] with a coarse focal plus the nominal mount tilt.
///
/// Updates `session.state` with intermediate results (homographies, initial estimates).
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Optional overrides for initialization parameters
///
/// # Errors
///
/// See [`step_init_with_seed`].
pub fn step_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    step_init_with_seed(session, IntrinsicsManualInit::default(), opts)
}

/// Optimize camera parameters using non-linear least squares.
///
/// This step refines the initial estimates by minimizing reprojection error
/// using bundle adjustment.
///
/// An untilted camera is one joint solve. A tilted (Scheimpflug) camera is
/// solved in stages that keep the tilt out of its degenerate basins:
/// - with a **trusted tilt seed** (a `sensor` given to
///   [`step_init_with_seed`]): a sweep of the leading radial coefficient over
///   `{0, -0.20, -0.40}`, each start first adapting the poses (A0) and then
///   freeing the low-order radial terms (A1) with intrinsics and tilt held;
///   the best start feeds a joint solve bounded around the seed (focals
///   `[0.75, 1.5]×`, principal point `±0.15·f`, tilt `±0.35` rad);
/// - otherwise (**cold start**): the staged multi-start tilt sweep of
///   `optimize_scheimpflug_intrinsics_staged`.
///
/// Requires that [`step_init`] has been run first, with the camera sensor the
/// config still names.
///
/// Updates `session.state` with optimization metrics and sets `session.output`.
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Optional overrides for optimization parameters
///
/// # Errors
///
/// - Input not set
/// - Initialization not run
/// - The camera's projection or sensor changed since initialization
/// - Optimization fails
pub fn step_optimize(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<IntrinsicsOptimizeResult, Error> {
    // Validate preconditions
    session.validate()?;
    let input = session.require_input()?.clone();
    let config = session.config.clone();

    let initial = session
        .state
        .initial_params(config.camera.model)
        .ok_or_else(|| Error::not_available("initial params (call step_init first)"))?;
    if !session.state.initialized_for(config.camera.model) {
        let msg = format!(
            "the camera projection ({:?}) or sensor ({:?}) changed since step_init; run \
             step_init again",
            config.camera.model.projection, config.camera.model.sensor
        );
        session.log_failure("optimize", msg.clone());
        return Err(Error::invalid_input(msg));
    }

    // Get effective options
    let opts = opts.unwrap_or_default();
    let mut backend_opts = config.solver.backend_options();
    if let Some(max_iters) = opts.max_iters {
        backend_opts.max_iters = max_iters;
    }
    if let Some(verbosity) = opts.verbosity {
        backend_opts.verbosity = verbosity;
    }

    // Run optimization
    let solved = match config.camera.model.sensor {
        SensorKind::Identity => {
            optimize_planar_intrinsics(&input, &initial, config.solve_opts(), backend_opts)
                .map_err(Error::from)
        }
        SensorKind::Scheimpflug => optimize_tilted(
            &input,
            &initial,
            &config,
            session.state.initial_sensor_manual,
            backend_opts,
        ),
        other => Err(unsupported_sensor(other)),
    };
    let result = match solved {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("optimize", e.to_string());
            return Err(e);
        }
    };

    // Update state
    let final_cost = result.report.final_cost;
    let mean_reproj_error = result.mean_reproj_error;
    let iterations = result.report.num_iters;
    session.state.final_cost = Some(final_cost);
    session.state.mean_reproj_error = Some(mean_reproj_error);
    session.state.iterations = Some(iterations);

    // Set output
    session.set_output(result);

    // Log success
    session.log_success_with_notes(
        "optimize",
        format!("cost={final_cost:.2e}, reproj_err={mean_reproj_error:.3}px"),
    );

    Ok(IntrinsicsOptimizeResult {
        final_cost,
        mean_reproj_error,
        iterations,
    })
}

/// Every distortion coefficient of `kind` except the low-order radial terms
/// (`k1`, `k2`; the division model's `lambda`).
fn all_but_low_order_radial(kind: DistortionKind) -> Vec<String> {
    kind.param_names()
        .iter()
        .filter(|n| !matches!(**n, "k1" | "k2" | "lambda"))
        .map(|n| n.to_string())
        .collect()
}

/// The staged solve of a tilted (Scheimpflug) camera; see [`step_optimize`].
fn optimize_tilted(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    config: &PlanarIntrinsicsConfig,
    trust_seed_tilt: bool,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    let max_iters = backend_opts.max_iters;
    if max_iters == 0 {
        return Err(Error::invalid_input("max_iters must be positive"));
    }
    // Planar intrinsics has no global pose gauge to remove, so fixing view 0 is
    // only safe when its seed pose is exact. On a trusted warm start the seed pose
    // comes from a homography fit to *distorted* points, so it is biased under
    // strong distortion; pinning it there would hold the solve off the true
    // optimum. Free all poses in that case; the cold path keeps the configured
    // behaviour.
    let fix_poses = if trust_seed_tilt {
        Vec::new()
    } else {
        config.fix_poses.clone()
    };
    let solve_opts = PlanarIntrinsicsSolveOptions {
        robust_loss: config.solver.robust_loss,
        fix: config.camera.fix.clone(),
        fix_poses,
        bounds: None,
    };

    if !trust_seed_tilt {
        // Cold start: the staged multi-start init breaks the
        // tilt/principal-point/distortion degeneracy that a `tilt = 0` solve falls
        // into. This path is experimental.
        return Ok(optimize_scheimpflug_intrinsics_staged(
            dataset,
            initial,
            solve_opts,
            &ScheimpflugStagedInitOptions::default(),
            backend_opts,
        )?);
    }

    // Trusted warm start: k1 multi-start sweep (Phase A) followed by a bounded
    // joint refine (Phase B).
    //
    // Problem: starting from k1=0 (the default when only a mechanical-spec tilt
    // and nominal focal are provided), the Phase A sub-problem (fixed intrinsics
    // + fixed tilt, free k1 + poses, L2) can converge to a spurious k1≈0 local
    // minimum because poses adapt to absorb the radial distortion when k1 is
    // small. This is a genuine local minimum — more iterations do not escape it.
    //
    // Solution: sweep k1 starting points over a coarse negative grid
    // {0, -0.20, -0.40}. Tilt is held throughout Phase A, so the sweep is not
    // confounded with it; the Phase A cost (with fixed tilt+intrinsics)
    // cleanly distinguishes the correct k1 basin (low reproj) from the k1≈0
    // local minimum (high reproj). The winner feeds Phase B for the final
    // joint refine.
    let model = config.camera.model;
    let k1_seeds = [0.0_f64, -0.20, -0.40];
    // A0: everything but the poses held.
    let pose_only = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::None,
        fix: CameraFix::all(model),
        fix_poses: Vec::new(),
        bounds: None,
    };
    // A1: intrinsics and tilt held; the low-order radial terms free.
    let free_low_order_radial = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::None,
        fix: CameraFix {
            distortion: Some(all_but_low_order_radial(model.distortion)),
            ..CameraFix::all(model)
        },
        fix_poses: Vec::new(),
        bounds: None,
    };
    let pre_iters = (max_iters / 2).max(60);
    // Short pose-only pre-adapt budget: just enough to orient poses to each k1
    // seed without spending too much on an intermediate sub-problem.
    let pose_adapt_iters = pre_iters.min(20);
    let mut best_prefit: Option<PlanarIntrinsicsEstimate> = None;
    for &k1_seed in &k1_seeds {
        // Sweep the leading barrel coefficient (k1 for BC5/Rational8/ThinPrism9,
        // lambda for Division1).
        let seed_initial = PlanarIntrinsicsParams {
            camera: CameraParams {
                distortion: with_leading_radial(&initial.camera.distortion, k1_seed),
                ..initial.camera.clone()
            },
            camera_se3_target: initial.camera_se3_target.clone(),
        };
        // Sub-step A0: adapt poses to the seeded k1, holding k1 and intrinsics
        // fixed. Without this, the Gauss-Newton gradient in A1 is computed with
        // poses biased for k1=0, which can push k1 back toward zero even when
        // the correct basin is at k1≈-0.43.
        let Ok(a0) = optimize_planar_intrinsics(
            dataset,
            &seed_initial,
            pose_only.clone(),
            BackendSolveOptions {
                max_iters: pose_adapt_iters,
                ..backend_opts.clone()
            },
        ) else {
            continue;
        };
        // Sub-step A1: free k1 (and k2) from the pose-adapted starting point.
        if let Ok(candidate) = optimize_planar_intrinsics(
            dataset,
            &a0.params,
            free_low_order_radial.clone(),
            BackendSolveOptions {
                max_iters: pre_iters,
                ..backend_opts.clone()
            },
        ) && candidate.mean_reproj_error.is_finite()
        {
            let better = best_prefit
                .as_ref()
                .is_none_or(|b| candidate.mean_reproj_error < b.mean_reproj_error);
            if better {
                best_prefit = Some(candidate);
            }
        }
    }
    let prefit = best_prefit
        .ok_or_else(|| Error::numerical("all k1 Phase-A seeds failed to converge".to_string()))?;

    // Phase B: full bounded joint solve from the best Phase A starting point.
    // The bounds only keep the solve from wandering off on weakly
    // conditioned data: a mount's nominal tilt can be off by well over 0.10
    // rad (a spec that says untilted for a sensor tilted 16°), and a solve
    // that ends on its bound has a wrong tilt and no covariance, so the tilt
    // may move ±0.35 rad (20°) from the seed.
    let k_seed = initial.camera.intrinsics;
    let s_seed = match initial.camera.sensor {
        SensorParams::Scheimpflug { params } => params,
        _ => unreachable!("step_optimize routes only Scheimpflug cameras here"),
    };
    let (pp_x, pp_y) = (0.15 * k_seed.fx, 0.15 * k_seed.fy);
    let tilt_margin = 0.35_f64;
    let joint_bounds = CameraBounds {
        fx: Some((0.75 * k_seed.fx, 1.5 * k_seed.fx)),
        fy: Some((0.75 * k_seed.fy, 1.5 * k_seed.fy)),
        cx: Some((k_seed.cx - pp_x, k_seed.cx + pp_x)),
        cy: Some((k_seed.cy - pp_y, k_seed.cy + pp_y)),
        tilt_x: Some((s_seed.tilt_x - tilt_margin, s_seed.tilt_x + tilt_margin)),
        tilt_y: Some((s_seed.tilt_y - tilt_margin, s_seed.tilt_y + tilt_margin)),
    };
    let solve_opts_b = PlanarIntrinsicsSolveOptions {
        bounds: Some(joint_bounds),
        ..solve_opts
    };
    Ok(optimize_planar_intrinsics(
        dataset,
        &prefit.params,
        solve_opts_b,
        backend_opts,
    )?)
}

/// Filter observations based on reprojection error.
///
/// This step removes outlier points (and optionally entire views) based on
/// reprojection error computed from the current output.
///
/// Requires that [`step_optimize`] has been run first.
///
/// **Important**: This modifies `session.input` in-place, which triggers the
/// invalidation policy and clears state and output. You should run [`step_init`]
/// and [`step_optimize`] again after filtering.
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Filtering parameters; `None` uses [`IntrinsicsFilterOptions::default`]
///
/// # Errors
///
/// - Output not computed (need reprojection errors)
/// - Filtering would remove all observations
/// - Not enough points remaining per view
pub fn step_filter(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsFilterOptions>,
) -> Result<(), Error> {
    let opts = opts.unwrap_or_default();
    if opts.min_points_per_view < 4 {
        return Err(Error::invalid_input(
            "min_points_per_view must be >= 4 for homography",
        ));
    }

    let output = session.require_output()?.clone();
    let input = session.require_input()?.clone();

    let poses = &output.params.camera_se3_target;
    if input.views.len() != poses.len() {
        return Err(Error::invalid_input(format!(
            "pose count ({}) must match view count ({})",
            poses.len(),
            input.views.len()
        )));
    }

    let camera = output.params.build_camera()?;

    let mut filtered_views = Vec::new();
    let mut total_removed = 0usize;

    for (view, pose) in input.views.iter().zip(poses) {
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();
        let mut cov_px = Vec::new();

        for (i, (p3d, p2d)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_camera_point(&p_cam.coords) else {
                total_removed += 1;
                continue;
            };

            let error = (projected - *p2d).norm();
            if error <= opts.max_reproj_error {
                points_3d.push(*p3d);
                points_2d.push(*p2d);
                if let Some(cov) = view.obs.covariance(i) {
                    cov_px.push(cov);
                }
            } else {
                total_removed += 1;
            }
        }

        if points_3d.len() >= opts.min_points_per_view {
            let obs = if view.obs.covariances().is_some() {
                CorrespondenceView::new_with_covariances(points_3d, points_2d, cov_px)?
            } else {
                CorrespondenceView::new(points_3d, points_2d)?
            };
            filtered_views.push(View::without_meta(obs));
        } else if !opts.remove_sparse_views {
            // Keep sparse views if requested
            filtered_views.push(view.clone());
        }
        // else: view is removed entirely
    }

    let views_removed = input.views.len() - filtered_views.len();
    if filtered_views.is_empty() {
        return Err(Error::invalid_input("filtering would remove all views"));
    }

    let filtered_dataset = vision_calibration_core::PlanarDataset::new(filtered_views)
        .map_err(|e| Error::invalid_input(format!("failed to create filtered dataset: {e}")))?;

    // This will clear state and output per invalidation policy
    session.set_input(filtered_dataset)?;

    session.log_success_with_notes(
        "filter",
        format!("removed {} points, {} views", total_removed, views_removed),
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full calibration pipeline: init -> optimize.
///
/// Convenience function that runs initialization and optimization in sequence,
/// with the session's config (set it with `session.set_config` first). Both
/// steps run their auto paths; for a Scheimpflug sensor, seed the tilt with
/// [`step_init_with_seed`] instead when a nominal mount tilt is known.
///
/// # Errors
///
/// Any error from [`step_init`] or [`step_optimize`].
pub fn run_calibration(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
) -> Result<(), Error> {
    let _ = step_init(session, None)?;
    let _ = step_optimize(session, None)?;
    Ok(())
}

/// Run calibration with outlier filtering: init -> optimize -> filter -> init -> optimize.
///
/// Performs an initial calibration pass, filters outliers based on reprojection error,
/// then re-runs calibration on the cleaned data.
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Filtering parameters; `None` uses [`IntrinsicsFilterOptions::default`]
///
/// # Errors
///
/// Any error from the constituent steps.
pub fn run_calibration_with_filtering(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsFilterOptions>,
) -> Result<(), Error> {
    // First pass
    let _ = step_init(session, None)?;
    let _ = step_optimize(session, None)?;

    // Filter outliers
    step_filter(session, opts)?;

    // Second pass on cleaned data
    let _ = step_init(session, None)?;
    let _ = step_optimize(session, None)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn manual_init_distortion_seed_round_trips_as_a_tagged_payload() {
        let seed = IntrinsicsManualInit {
            distortion: Some(
                DistortionParams::from_values(
                    DistortionKind::KannalaBrandt4,
                    &[0.1, 0.0, 0.0, 0.0],
                )
                .unwrap(),
            ),
            ..Default::default()
        };
        let json = serde_json::to_value(&seed).unwrap();
        assert_eq!(json["distortion"]["type"], "kannala_brandt4");
        let back: IntrinsicsManualInit = serde_json::from_value(json).unwrap();
        assert_eq!(
            back.distortion.unwrap().values(),
            seed.distortion.unwrap().values()
        );
    }

    use super::*;
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCy, Pinhole, make_pinhole_camera, synthetic::planar,
    };

    fn make_test_camera() -> vision_calibration_core::PinholeCamera {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        )
    }

    fn make_test_dataset() -> vision_calibration_core::PlanarDataset {
        let cam_gt = make_test_camera();
        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();
        vision_calibration_core::PlanarDataset::new(
            views.into_iter().map(View::without_meta).collect(),
        )
        .unwrap()
    }

    #[test]
    fn step_init_computes_initial_estimate() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        step_init(&mut session, None).unwrap();

        assert!(session.state.is_initialized());
        assert!(session.state.homographies.is_some());
        assert_eq!(session.state.initial_poses.as_ref().unwrap().len(), 4);

        // Check intrinsics are reasonable
        let k = session.state.initial_camera.as_ref().unwrap().intrinsics;
        assert!((k.fx - 800.0).abs() < 100.0); // Within 12.5%
        assert!((k.fy - 780.0).abs() < 100.0);
    }

    #[test]
    fn step_optimize_requires_init() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let result = step_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("init"));
    }

    #[test]
    fn step_optimize_sets_output() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        step_init(&mut session, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        assert!(session.has_output());
        assert!(session.state.is_optimized());

        let output = session.output().unwrap();
        // Synthetic data should have very low reprojection error
        assert!(output.mean_reproj_error < 1.0);
    }

    #[test]
    fn step_filter_modifies_input() {
        let cam_gt = make_test_camera();
        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let mut views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        // Add an outlier
        if let Some(first_view) = views.first_mut()
            && let Some(p) = first_view.points_2d.first_mut()
        {
            p.x += 50.0;
        }

        let dataset = vision_calibration_core::PlanarDataset::new(
            views.into_iter().map(View::without_meta).collect(),
        )
        .unwrap();
        let original_points: usize = dataset.views.iter().map(|v| v.obs.len()).sum();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(dataset).unwrap();

        step_init(&mut session, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        step_filter(&mut session, None).unwrap();

        // Should have removed at least the outlier
        let filtered_points: usize = session
            .require_input()
            .unwrap()
            .views
            .iter()
            .map(|v| v.obs.len())
            .sum();
        assert!(filtered_points < original_points);

        // State and output should be cleared
        assert!(!session.state.is_initialized());
        assert!(!session.has_output());
    }

    #[test]
    fn run_calibration_full_pipeline() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        run_calibration(&mut session).unwrap();

        assert!(session.has_output());
        let output = session.output().unwrap();

        // Check accuracy on synthetic data (no noise)
        let k = output.params.camera.intrinsics;
        assert!((k.fx - 800.0).abs() < 10.0);
        assert!((k.fy - 780.0).abs() < 10.0);
        assert!((k.cx - 640.0).abs() < 10.0);
        assert!((k.cy - 360.0).abs() < 10.0);
    }

    #[test]
    fn session_json_checkpoint() {
        let mut session =
            CalibrationSession::<PlanarIntrinsicsProblem>::with_description("Test calibration");
        session.set_input(make_test_dataset()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        // Checkpoint
        let json = session.to_json().unwrap();

        // Restore
        let restored = CalibrationSession::<PlanarIntrinsicsProblem>::from_json(&json).unwrap();

        assert_eq!(
            restored.metadata.description,
            Some("Test calibration".to_string())
        );
        assert!(restored.has_input());
        assert!(restored.has_output());
        assert_eq!(restored.exports.len(), 1);
        assert!(restored.state.is_initialized());
        assert!(restored.state.is_optimized());
    }

    #[test]
    fn log_entries_recorded_through_pipeline() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        run_calibration(&mut session).unwrap();

        // Should have log entries for init and optimize
        assert!(session.log.len() >= 2);
        assert!(session.log.iter().any(|e| e.operation == "init"));
        assert!(session.log.iter().any(|e| e.operation == "optimize"));
        assert!(session.log.iter().all(|e| e.success));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Manual init (ADR 0011) tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn step_set_init_default_matches_step_init() {
        let mut session_a = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session_a.set_input(make_test_dataset()).unwrap();
        step_init(&mut session_a, None).unwrap();

        let mut session_b = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session_b.set_input(make_test_dataset()).unwrap();
        step_init_with_seed(&mut session_b, IntrinsicsManualInit::default(), None).unwrap();

        // Both paths should produce identical state — step_init delegates to
        // step_init_with_seed with default fields.
        let k_a = session_a.state.initial_camera.as_ref().unwrap().intrinsics;
        let k_b = session_b.state.initial_camera.as_ref().unwrap().intrinsics;
        assert!((k_a.fx - k_b.fx).abs() < 1e-9);
        assert!((k_a.fy - k_b.fy).abs() < 1e-9);
        assert!((k_a.cx - k_b.cx).abs() < 1e-9);
        assert!((k_a.cy - k_b.cy).abs() < 1e-9);
    }

    #[test]
    fn step_set_init_with_intrinsics_seed_converges() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let manual = IntrinsicsManualInit {
            intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            }),
            ..Default::default()
        };
        step_init_with_seed(&mut session, manual, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        let output = session.output().unwrap();
        // Synthetic data with GT-matching intrinsics seed should converge tightly.
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    #[test]
    fn step_set_init_with_full_seeds_converges_tightly() {
        // Run auto-init first to harvest a plausible pose set.
        let mut session_auto = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session_auto.set_input(make_test_dataset()).unwrap();
        step_init(&mut session_auto, None).unwrap();
        let auto_poses = session_auto.state.initial_poses.clone().unwrap();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let manual = IntrinsicsManualInit {
            intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            }),
            distortion: Some(DistortionParams::BrownConrady5 {
                params: BrownConrady5::default(),
            }),
            sensor: None,
            poses: Some(auto_poses),
        };
        step_init_with_seed(&mut session, manual, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        let output = session.output().unwrap();
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    #[test]
    fn step_set_init_rejects_wrong_pose_count() {
        use vision_calibration_core::Iso3;

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        // Test dataset has 4 views; supply only 1 pose.
        let manual = IntrinsicsManualInit {
            poses: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        let err = step_init_with_seed(&mut session, manual, None).unwrap_err();
        assert!(
            err.to_string().contains("manual poses count"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn step_set_init_logs_manual_and_auto_sources() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let manual = IntrinsicsManualInit {
            intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            }),
            ..Default::default()
        };
        step_init_with_seed(&mut session, manual, None).unwrap();

        let init_entry = session
            .log
            .iter()
            .find(|e| e.operation == "init")
            .expect("init log entry");
        let notes = init_entry.notes.as_deref().unwrap_or("");
        assert!(notes.contains("manual: intrinsics"), "notes: {}", notes);
        assert!(notes.contains("auto: distortion"), "notes: {}", notes);
        assert!(notes.contains("poses"), "notes: {}", notes);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scheimpflug sensor
    // ─────────────────────────────────────────────────────────────────────────

    fn scheimpflug_config() -> super::super::PlanarIntrinsicsConfig {
        let mut config = super::super::PlanarIntrinsicsConfig::default();
        config.camera.model.sensor = SensorKind::Scheimpflug;
        config
    }

    fn tilted_session(sensor: ScheimpflugParams) -> CalibrationSession<PlanarIntrinsicsProblem> {
        let base = make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5::default(),
        );
        let camera =
            vision_calibration_core::Camera::new(Pinhole, base.dist, sensor.compile(), base.k);
        let board_points = planar::grid_points(6, 5, 0.03);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.08, 0.55, 0.03);
        let views = planar::project_views_all(&camera, &board_points, &poses).expect("views");
        let dataset = vision_calibration_core::PlanarDataset::new(
            views.into_iter().map(View::without_meta).collect(),
        )
        .expect("dataset");
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_config(scheimpflug_config()).expect("config");
        session.set_input(dataset).expect("input");
        session
    }

    fn log_has_experimental_warning(session: &CalibrationSession<PlanarIntrinsicsProblem>) -> bool {
        session.log().iter().any(|e| {
            e.notes
                .as_deref()
                .is_some_and(|n| n.contains("experimental"))
        })
    }

    #[test]
    fn scheimpflug_auto_init_logs_experimental_warning() {
        let mut session = tilted_session(ScheimpflugParams::default());
        let init = step_init(&mut session, None).expect("step_init");
        assert!(matches!(
            init.camera.sensor,
            SensorParams::Scheimpflug { .. }
        ));
        assert!(session.state.homographies.is_none());
        assert!(
            log_has_experimental_warning(&session),
            "from-scratch auto-init should log the experimental warning"
        );
    }

    #[test]
    fn scheimpflug_seeded_init_does_not_log_experimental_warning() {
        let mut session = tilted_session(ScheimpflugParams::default());
        let manual = IntrinsicsManualInit {
            intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            }),
            ..Default::default()
        };
        step_init_with_seed(&mut session, manual, None).expect("seeded init");
        assert!(
            !log_has_experimental_warning(&session),
            "seeded init must not log the experimental warning"
        );
    }

    #[test]
    fn untilted_camera_never_logs_experimental_warning() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();
        step_init(&mut session, None).unwrap();
        assert!(!log_has_experimental_warning(&session));
    }

    #[test]
    fn scheimpflug_run_calibration_sets_output_and_state() {
        let mut session = tilted_session(ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        });
        run_calibration(&mut session).expect("run_calibration");

        assert!(session.has_output());
        assert!(session.state.is_initialized());
        assert!(session.state.is_optimized());

        let output = session.output().expect("output");
        assert!(output.mean_reproj_error.is_finite());
        assert!(output.mean_reproj_error < 1.0);
        assert_eq!(output.params.camera_se3_target.len(), 5);
        assert!(matches!(
            output.params.camera.sensor,
            SensorParams::Scheimpflug { .. }
        ));
        // One export kind for both sensors; the camera says what was calibrated.
        let export = session.export().expect("export");
        assert_eq!(export.kind, crate::common::ExportKind::PlanarIntrinsics);
        assert!(matches!(
            export.params.camera.sensor,
            SensorParams::Scheimpflug { .. }
        ));
    }

    #[test]
    fn scheimpflug_seeded_tilt_converges() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let mut session = tilted_session(sensor_gt);
        let manual = IntrinsicsManualInit {
            intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            }),
            distortion: Some(DistortionParams::BrownConrady5 {
                params: BrownConrady5::default(),
            }),
            sensor: Some(sensor_gt),
            poses: None,
        };
        step_init_with_seed(&mut session, manual, None).expect("seeded init");
        assert!(session.state.initial_sensor_manual);
        step_optimize(&mut session, None).expect("step_optimize");
        let output = session.output().expect("output");
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    /// A seed whose tilt is off by 16° (the mount spec says untilted, the
    /// sensor is tilted): the bounded joint solve still reaches the true
    /// tilt, inside its bounds.
    #[test]
    fn scheimpflug_seed_off_by_the_mount_tilt_converges_inside_its_bounds() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: -0.28,
            tilt_y: 0.0,
        };
        let mut session = tilted_session(sensor_gt);
        let manual = IntrinsicsManualInit {
            intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
            }),
            distortion: Some(DistortionParams::BrownConrady5 {
                params: BrownConrady5::default(),
            }),
            sensor: Some(ScheimpflugParams::default()),
            poses: None,
        };
        step_init_with_seed(&mut session, manual, None).expect("seeded init");
        step_optimize(&mut session, None).expect("step_optimize");
        let output = session.output().expect("output");
        let SensorParams::Scheimpflug { params } = output.params.camera.sensor else {
            panic!("a tilted camera");
        };
        assert!(
            (params.tilt_x - sensor_gt.tilt_x).abs() < 0.01,
            "tilt_x {:.4}",
            params.tilt_x
        );
        // This five-view scene leaves the principal point to the poses, so
        // only the bound is checked: the tilt ends inside it.
        assert!(
            !matches!(
                output.uncertainty,
                vision_calibration_core::Uncertainty::Unavailable {
                    reason: vision_calibration_core::UncertaintyUnavailable::AtBound { .. }
                }
            ),
            "{:?}",
            output.uncertainty
        );
    }

    #[test]
    fn scheimpflug_filtering_reruns_both_passes() {
        let mut session = tilted_session(ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        });
        run_calibration_with_filtering(&mut session, None).expect("filtered run");
        let output = session.output().expect("output");
        assert!(output.mean_reproj_error < 1.0);
        assert!(matches!(
            output.params.camera.sensor,
            SensorParams::Scheimpflug { .. }
        ));
    }

    #[test]
    fn sensor_seed_needs_a_scheimpflug_camera() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();
        let manual = IntrinsicsManualInit {
            sensor: Some(ScheimpflugParams::default()),
            ..Default::default()
        };
        let err = step_init_with_seed(&mut session, manual, None).unwrap_err();
        assert!(err.to_string().contains("Scheimpflug camera"), "{err}");
    }

    #[test]
    fn changing_the_sensor_after_init_is_rejected() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();
        step_init(&mut session, None).unwrap();
        session.set_config(scheimpflug_config()).unwrap();
        let err = step_optimize(&mut session, None).unwrap_err();
        assert!(err.to_string().contains("run step_init again"), "{err}");
    }

    #[test]
    fn distortion_seed_of_another_kind_is_rejected_naming_the_field() {
        let mut config = super::super::PlanarIntrinsicsConfig::default();
        config.camera.model.distortion = DistortionKind::Rational8;
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_config(config).unwrap();
        session.set_input(make_test_dataset()).unwrap();
        let manual = IntrinsicsManualInit {
            distortion: Some(DistortionParams::BrownConrady5 {
                params: BrownConrady5::default(),
            }),
            ..Default::default()
        };
        let err = step_init_with_seed(&mut session, manual, None).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("distortion"), "{msg}");
        assert!(msg.contains("Rational8"), "{msg}");
    }

    #[test]
    fn distortion_seed_of_the_cameras_kind_is_used_as_given() {
        let mut config = super::super::PlanarIntrinsicsConfig::default();
        config.camera.model.distortion = DistortionKind::Rational8;
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_config(config).unwrap();
        session.set_input(make_test_dataset()).unwrap();
        let seed = DistortionParams::from_values(
            DistortionKind::Rational8,
            &[-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let manual = IntrinsicsManualInit {
            distortion: Some(seed.clone()),
            ..Default::default()
        };
        let init = step_init_with_seed(&mut session, manual, None).unwrap();
        assert_eq!(init.camera.distortion.values(), seed.values());
    }
}
