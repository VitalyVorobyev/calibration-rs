//! Step functions for Scheimpflug intrinsics calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraParams, DistortionFixMask, DistortionParams, FxFyCxCySkew,
    IntrinsicsFixMask, IntrinsicsParams, Iso3, ProjectionParams, Real, ScheimpflugParams,
    SensorParams,
};
use vision_calibration_linear::distortion_fit::DistortionFitOptions;
use vision_calibration_linear::scheimpflug_init::{
    ScheimpflugIntrinsicsInitOptions as LinearScheimpflugIntrinsicsInitOptions,
    estimate_scheimpflug_intrinsics_iterative,
};
use vision_calibration_optim::{
    BackendSolveOptions, ScheimpflugBounds, ScheimpflugFixMask as OptimScheimpflugFixMask,
    ScheimpflugIntrinsicsEstimate, ScheimpflugIntrinsicsParams as OptimScheimpflugIntrinsicsParams,
    ScheimpflugIntrinsicsSolveOptions as OptimScheimpflugIntrinsicsSolveOptions,
    ScheimpflugStagedInitOptions, optimize_scheimpflug_intrinsics,
    optimize_scheimpflug_intrinsics_staged,
};

use crate::planar_family::{estimate_view_homographies, recover_planar_poses_from_homographies};
use crate::session::CalibrationSession;

use super::problem::{
    ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
    ScheimpflugIntrinsicsResult,
};

pub use crate::common::{IntrinsicsInitOptions, IntrinsicsOptimizeOptions};

/// Manual initialization seeds for Scheimpflug intrinsics calibration.
///
/// All fields are `Option<T>`:
/// - `None` means *auto-initialize this group* (same path as plain `step_init`).
/// - `Some(value)` means *use this value*; do not auto-initialize.
///
/// Partial-seed semantics mirror `PlanarManualInit`:
/// - `intrinsics: Some` skips the planar bootstrap; distortion defaults to zeros and
///   sensor to identity tilt unless also seeded; poses recover from homographies using
///   the manual intrinsics.
/// - `intrinsics: None` runs the bootstrap; any seeded field overrides the
///   corresponding bootstrap output. The auto path's tangential-distortion zeroing
///   (workflow invariant — Scheimpflug pipelines fix tangential) is **not** applied
///   when the user supplies a manual `distortion` — they get exactly what they pass.
///
/// See ADR 0011 for the design rationale.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScheimpflugManualInit {
    /// Manual intrinsics seed. `None` means auto-init via Zhang's method.
    pub intrinsics: Option<FxFyCxCySkew<Real>>,
    /// Manual distortion seed. `None` means auto-init (or zeros when intrinsics are
    /// seeded).
    pub distortion: Option<BrownConrady5<Real>>,
    /// Manual Scheimpflug sensor tilts. `None` means start at identity tilt
    /// (`ScheimpflugParams::default()`); the optimizer refines from there.
    pub sensor: Option<ScheimpflugParams>,
    /// Manual per-view poses (`camera_se3_target`). `None` means recover from
    /// homographies using whichever intrinsics are in effect.
    pub poses: Option<Vec<Iso3>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

/// Typed return value of [`step_init`] / [`step_init_with_seed`].
///
/// Carries the seeded-or-fitted initial estimates. The same values continue to
/// be written into `session.state` for backwards compatibility — see ADR 0011.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScheimpflugIntrinsicsInitResult {
    /// Initial pinhole intrinsics (fx, fy, cx, cy, skew).
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Initial Brown–Conrady distortion coefficients.
    pub distortion: BrownConrady5<Real>,
    /// Initial Scheimpflug sensor tilt parameters.
    pub sensor: ScheimpflugParams,
    /// Initial per-view target poses (`camera_se3_target`).
    pub poses: Vec<Iso3>,
}

/// Typed return value of [`step_optimize`].
///
/// Aggregates the optimization metrics that examples used to read out of
/// `session.state` after the Scheimpflug-intrinsics solve.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScheimpflugIntrinsicsOptimizeResult {
    /// Final cost reported by the non-linear solver.
    pub final_cost: f64,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
}

/// Initialize Scheimpflug intrinsics, distortion, sensor tilt, and per-view poses
/// from any combination of manual seeds and auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate that
/// passes `ScheimpflugManualInit::default()` (all-`None`, full auto path).
///
/// See [`ScheimpflugManualInit`] for partial-seed semantics.
///
/// # Errors
///
/// - Input not set, or fewer than 3 views.
/// - `init_iterations == 0` when running the bootstrap auto-fit.
/// - Homography or auto-init computation fails.
/// - `manual.poses` is `Some` but its length does not match the view count.
pub fn step_init_with_seed(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    manual: ScheimpflugManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<ScheimpflugIntrinsicsInitResult, Error> {
    session.validate()?;
    let dataset = session.require_input()?.clone();

    let opts = opts.unwrap_or_default();
    let mut init_iterations = session.config.init_iterations;
    if let Some(iterations) = opts.iterations {
        init_iterations = iterations;
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    // Capture before `manual` is destructured below: a user-provided tilt seed is
    // trusted directly by `step_optimize` (ADR 0022).
    let sensor_manual = manual.sensor.is_some();
    // From-scratch (no intrinsics seed) Scheimpflug auto-init is experimental — it
    // can land in a wrong tilt/focal basin under the tilt↔focal↔distortion
    // degeneracy. The supported path seeds a coarse focal + nominal mount tilt.
    let intrinsics_seeded = manual.intrinsics.is_some();

    let (intrinsics, distortion, sensor, poses) = if let Some(k) = manual.intrinsics {
        manual_fields.push("intrinsics");

        let manual_dist = manual.distortion;
        let dist_is_manual = manual_dist.is_some();

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
                if dist_is_manual {
                    manual_fields.push("distortion");
                } else {
                    auto_fields.push("distortion");
                }
                (manual_dist.unwrap_or_default(), p)
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

                let (dist, final_poses) = if !dist_is_manual {
                    auto_fields.push("distortion");
                    let poses = recover_planar_poses_from_homographies(&raw_homographies, &k)
                        .map_err(|e| {
                            Error::numerical(format!(
                                "scheimpflug intrinsics pose recovery failed: {e}"
                            ))
                        })?;
                    (BrownConrady5::default(), poses)
                } else {
                    // Manual distortion: recover poses from raw homographies.
                    manual_fields.push("distortion");
                    let poses = recover_planar_poses_from_homographies(&raw_homographies, &k)
                        .map_err(|e| {
                            Error::numerical(format!(
                                "scheimpflug intrinsics pose recovery failed: {e}"
                            ))
                        })?;
                    (manual_dist.unwrap(), poses)
                };
                (dist, final_poses)
            }
        };

        (k, dist, sensor, poses)
    } else {
        if init_iterations == 0 {
            return Err(Error::invalid_input("init_iterations must be positive"));
        }
        auto_fields.push("intrinsics");

        let bootstrap = estimate_scheimpflug_intrinsics_iterative(
            &dataset,
            LinearScheimpflugIntrinsicsInitOptions {
                iterations: init_iterations,
                distortion_opts: DistortionFitOptions {
                    fix_k3: session.config.fix_k3_in_init,
                    fix_tangential: true,
                    iters: 8,
                },
                zero_skew: session.config.zero_skew,
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
                bootstrap.camera.dist
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

    session.state.initial_intrinsics = Some(intrinsics);
    session.state.initial_distortion = Some(distortion);
    session.state.initial_sensor = Some(sensor);
    session.state.initial_sensor_manual = sensor_manual;
    session.state.initial_poses = Some(poses.clone());
    session.state.clear_optimization();

    let source = format_init_source(&manual_fields, &auto_fields);
    let experimental = if intrinsics_seeded {
        ""
    } else {
        " — WARNING: from-scratch Scheimpflug auto-init is experimental and may \
         converge to a wrong tilt/focal basin; seed a coarse focal + nominal mount \
         tilt via step_init_with_seed for stable results (ADR 0022)"
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

    Ok(ScheimpflugIntrinsicsInitResult {
        intrinsics,
        distortion,
        sensor,
        poses,
    })
}

fn format_init_source(manual: &[&str], auto: &[&str]) -> String {
    match (manual.is_empty(), auto.is_empty()) {
        (false, false) => format!("(manual: {}; auto: {})", manual.join(", "), auto.join(", ")),
        (false, true) => format!("(manual: {})", manual.join(", ")),
        (true, false) => format!("(auto: {})", auto.join(", ")),
        (true, true) => "(empty)".to_string(),
    }
}

/// Initialize intrinsics, distortion, sensor tilt, and poses from observations
/// using full auto-init.
///
/// Convenience wrapper around [`step_init_with_seed`] with `ScheimpflugManualInit::default()`.
///
/// **Experimental.** From-scratch (unseeded) Scheimpflug auto-init is *not stable*
/// under the tilt↔focal↔distortion degeneracy: on strong-distortion + tilted data
/// it can settle into a wrong tilt/focal basin. The **supported** workflow is
/// [`step_init_with_seed`] with a coarse focal seed plus the nominal Scheimpflug
/// mount tilt — see ADR 0022.
pub fn step_init(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<ScheimpflugIntrinsicsInitResult, Error> {
    step_init_with_seed(session, ScheimpflugManualInit::default(), opts)
}

/// Optimize Scheimpflug intrinsics, distortion, sensor tilt, and target poses.
pub fn step_optimize(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<ScheimpflugIntrinsicsOptimizeResult, Error> {
    session.validate()?;
    let dataset = session.require_input()?.clone();

    let (initial_intrinsics, initial_distortion, initial_sensor, initial_poses) = session
        .state
        .initial_values()
        .ok_or_else(|| Error::not_available("initial params (call step_init first)"))?;
    let trust_seed_tilt = session.state.initial_sensor_manual;

    let opts = opts.unwrap_or_default();
    let mut max_iters = session.config.max_iters;
    let mut verbosity = session.config.verbosity;
    if let Some(v) = opts.max_iters {
        max_iters = v;
    }
    if let Some(v) = opts.verbosity {
        verbosity = v;
    }
    if max_iters == 0 {
        return Err(Error::invalid_input("max_iters must be positive"));
    }

    let initial = OptimScheimpflugIntrinsicsParams::new(
        initial_intrinsics,
        initial_distortion,
        initial_sensor,
        initial_poses,
    )?;
    // Planar intrinsics has no global pose gauge to remove, so fixing view 0 is
    // only safe when its seed pose is exact. On a trusted warm start the seed pose
    // comes from a homography fit to *distorted* points, so it is biased under
    // strong distortion; pinning it there would hold the solve off the true
    // optimum (see ADR 0022). Free all poses in that case; the cold path keeps the
    // configured behaviour.
    let fix_poses = if trust_seed_tilt {
        Vec::new()
    } else if session.config.fix_first_pose {
        vec![0]
    } else {
        Vec::new()
    };
    let solve_opts = OptimScheimpflugIntrinsicsSolveOptions {
        robust_loss: session.config.robust_loss,
        fix_intrinsics: session.config.fix_intrinsics,
        fix_distortion: session.config.fix_distortion,
        fix_scheimpflug: to_optim_scheimpflug_fix_mask(session.config.fix_scheimpflug),
        fix_poses,
        bounds: None,
    };

    let backend_opts = BackendSolveOptions {
        max_iters,
        verbosity,
        ..Default::default()
    };
    let estimate = if trust_seed_tilt {
        // Trusted warm start (ADR 0022): k1 multi-start sweep (Phase A) followed by
        // a bounded joint refine (Phase B).
        //
        // Problem: starting from k1=0 (the default when only a mechanical-spec tilt
        // and nominal focal are provided), the Phase A sub-problem (fixed intrinsics
        // + fixed tilt, free k1 + poses, L2) can converge to a spurious k1≈0 local
        // minimum because poses adapt to absorb the radial distortion when k1 is
        // small. This is a genuine local minimum — more iterations do not escape it.
        //
        // Solution: sweep k1 starting points over a coarse negative grid
        // {0, -0.20, -0.40}. Tilt is FIXED throughout Phase A, so the degenerate
        // high-tilt basin (which requires tilt to move to ≈-13°) is inaccessible.
        // The Phase A cost (with fixed tilt+intrinsics) cleanly distinguishes the
        // correct k1 basin (low reproj) from the k1≈0 local minimum (high reproj).
        // The winner feeds Phase B (Huber, bounded tilt ±0.10 rad) for the final
        // joint refine.
        let k1_seeds = [0.0_f64, -0.20, -0.40];
        let fix_intrinsics_and_tilt_and_dist = OptimScheimpflugIntrinsicsSolveOptions {
            robust_loss: vision_calibration_optim::RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::all_fixed(),
            fix_distortion: DistortionFixMask {
                k1: true,
                k2: true,
                k3: true,
                p1: true,
                p2: true,
            },
            fix_scheimpflug: to_optim_scheimpflug_fix_mask(super::problem::ScheimpflugFixMask {
                tilt_x: true,
                tilt_y: true,
            }),
            fix_poses: Vec::new(),
            bounds: None,
        };
        let prefit_opts_free_k1 = OptimScheimpflugIntrinsicsSolveOptions {
            robust_loss: vision_calibration_optim::RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::all_fixed(),
            fix_distortion: DistortionFixMask {
                k1: false,
                k2: false,
                k3: true,
                p1: true,
                p2: true,
            },
            fix_scheimpflug: to_optim_scheimpflug_fix_mask(super::problem::ScheimpflugFixMask {
                tilt_x: true,
                tilt_y: true,
            }),
            fix_poses: Vec::new(),
            bounds: None,
        };
        let pre_iters = (max_iters / 2).max(60);
        // Short pose-only pre-adapt budget: just enough to orient poses to each k1
        // seed without spending too much on an intermediate sub-problem.
        let pose_adapt_iters = pre_iters.min(20);
        let mut best_prefit: Option<ScheimpflugIntrinsicsEstimate> = None;
        for &k1_seed in &k1_seeds {
            let seed_dist = BrownConrady5 {
                k1: k1_seed,
                ..initial.distortion
            };
            let seed_initial = OptimScheimpflugIntrinsicsParams::new(
                initial.intrinsics,
                seed_dist,
                initial.sensor,
                initial.camera_se3_target.clone(),
            )?;
            // Sub-step A0: adapt poses to the seeded k1, holding k1 and intrinsics
            // fixed. Without this, the Gauss-Newton gradient in A1 is computed with
            // poses biased for k1=0, which can push k1 back toward zero even when
            // the correct basin is at k1≈-0.43.
            let a0 = match optimize_scheimpflug_intrinsics(
                &dataset,
                &seed_initial,
                fix_intrinsics_and_tilt_and_dist.clone(),
                BackendSolveOptions {
                    max_iters: pose_adapt_iters,
                    verbosity,
                    ..Default::default()
                },
            ) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let a0_initial = match OptimScheimpflugIntrinsicsParams::new(
                a0.params.intrinsics,
                a0.params.distortion,
                a0.params.sensor,
                a0.params.camera_se3_target,
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            // Sub-step A1: free k1 (and k2) from the pose-adapted starting point.
            if let Ok(candidate) = optimize_scheimpflug_intrinsics(
                &dataset,
                &a0_initial,
                prefit_opts_free_k1.clone(),
                BackendSolveOptions {
                    max_iters: pre_iters,
                    verbosity,
                    ..Default::default()
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
        let prefit = best_prefit.ok_or_else(|| {
            crate::Error::numerical("all k1 Phase-A seeds failed to converge".to_string())
        })?;

        // Phase B: full bounded joint solve from the best Phase A starting point.
        // Tilt bounds ±0.10 rad around the seed keep the optimizer in the correct
        // tilt basin while letting focal, pp, tilt, k1, and poses refine jointly.
        let k_seed = initial_intrinsics;
        let s_seed = initial_sensor;
        let (pp_x, pp_y) = (0.15 * k_seed.fx, 0.15 * k_seed.fy);
        let tilt_margin = 0.10_f64;
        let joint_bounds = ScheimpflugBounds {
            fx: Some((0.75 * k_seed.fx, 1.5 * k_seed.fx)),
            fy: Some((0.75 * k_seed.fy, 1.5 * k_seed.fy)),
            cx: Some((k_seed.cx - pp_x, k_seed.cx + pp_x)),
            cy: Some((k_seed.cy - pp_y, k_seed.cy + pp_y)),
            tilt_x: Some((s_seed.tilt_x - tilt_margin, s_seed.tilt_x + tilt_margin)),
            tilt_y: Some((s_seed.tilt_y - tilt_margin, s_seed.tilt_y + tilt_margin)),
        };
        let solve_opts_b = OptimScheimpflugIntrinsicsSolveOptions {
            bounds: Some(joint_bounds),
            ..solve_opts
        };
        let joint_initial = OptimScheimpflugIntrinsicsParams::new(
            prefit.params.intrinsics,
            prefit.params.distortion,
            prefit.params.sensor,
            prefit.params.camera_se3_target,
        )?;
        optimize_scheimpflug_intrinsics(&dataset, &joint_initial, solve_opts_b, backend_opts)?
    } else {
        // Cold start: the staged multi-start init breaks the
        // tilt/principal-point/distortion degeneracy that a `tilt = 0` solve falls
        // into. This path is experimental — see ADR 0022.
        optimize_scheimpflug_intrinsics_staged(
            &dataset,
            &initial,
            solve_opts,
            &ScheimpflugStagedInitOptions::default(),
            backend_opts,
        )?
    };

    let result = ScheimpflugIntrinsicsResult {
        params: ScheimpflugIntrinsicsParams {
            camera: scheimpflug_camera_params(
                estimate.params.intrinsics,
                estimate.params.distortion,
                estimate.params.sensor,
            ),
            camera_se3_target: estimate.params.camera_se3_target,
        },
        report: estimate.report,
        mean_reproj_error: estimate.mean_reproj_error,
    };

    let final_cost = result.report.final_cost;
    let mean_reproj_error = result.mean_reproj_error;
    session.state.final_cost = Some(final_cost);
    session.state.mean_reproj_error = Some(mean_reproj_error);
    session.set_output(result);

    session.log_success_with_notes(
        "optimize",
        format!("cost={final_cost:.2e}, reproj_err={mean_reproj_error:.3}px"),
    );

    Ok(ScheimpflugIntrinsicsOptimizeResult {
        final_cost,
        mean_reproj_error,
    })
}

/// Run full Scheimpflug calibration pipeline on a session: init -> optimize.
pub fn run_calibration(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    config: Option<ScheimpflugIntrinsicsConfig>,
) -> Result<(), Error> {
    if let Some(cfg) = config {
        session.set_config(cfg)?;
    }
    let _ = step_init(session, None)?;
    let _ = step_optimize(session, None)?;
    Ok(())
}

fn to_optim_scheimpflug_fix_mask(
    mask: super::problem::ScheimpflugFixMask,
) -> OptimScheimpflugFixMask {
    OptimScheimpflugFixMask {
        tilt_x: mask.tilt_x,
        tilt_y: mask.tilt_y,
    }
}

fn scheimpflug_camera_params(
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 { params: distortion },
        sensor: SensorParams::Scheimpflug { params: sensor },
        intrinsics: IntrinsicsParams::FxFyCxCySkew { params: intrinsics },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        Camera, FxFyCxCySkew, Pinhole, PlanarDataset, View, make_pinhole_camera, synthetic::planar,
    };

    fn make_dataset(sensor: ScheimpflugParams) -> PlanarDataset {
        let base = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        let camera = Camera::new(Pinhole, base.dist, sensor.compile(), base.k);

        let board_points = planar::grid_points(6, 5, 0.03);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.08, 0.55, 0.03);
        let views = planar::project_views_all(&camera, &board_points, &poses).expect("views");
        PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
    }

    fn log_has_experimental_warning(
        session: &CalibrationSession<ScheimpflugIntrinsicsProblem>,
    ) -> bool {
        session.log().iter().any(|e| {
            e.notes
                .as_deref()
                .is_some_and(|n| n.contains("experimental"))
        })
    }

    #[test]
    fn auto_init_logs_experimental_warning() {
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session
            .set_input(make_dataset(ScheimpflugParams::default()))
            .expect("input");
        step_init(&mut session, None).expect("step_init");
        assert!(
            log_has_experimental_warning(&session),
            "from-scratch auto-init should log the experimental warning (ADR 0022)"
        );
    }

    #[test]
    fn seeded_init_does_not_log_experimental_warning() {
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session
            .set_input(make_dataset(ScheimpflugParams::default()))
            .expect("input");
        let manual = ScheimpflugManualInit {
            intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
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
    fn step_optimize_requires_initialization() {
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session
            .set_input(make_dataset(ScheimpflugParams::default()))
            .expect("input");

        let err = step_optimize(&mut session, None).expect_err("init should be required");
        assert!(err.to_string().contains("step_init"));
    }

    #[test]
    fn run_calibration_sets_output_and_state() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session.set_input(make_dataset(sensor_gt)).expect("input");

        run_calibration(&mut session, None).expect("run_calibration");

        assert!(session.has_output());
        assert!(session.state.is_initialized());
        assert!(session.state.is_optimized());

        let output = session.output().expect("output");
        assert!(output.mean_reproj_error.is_finite());
        assert!(output.mean_reproj_error < 1.0);
        assert_eq!(output.params.camera_se3_target.len(), 5);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Manual init (ADR 0011) tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn step_set_init_default_matches_step_init() {
        let sensor_gt = ScheimpflugParams::default();
        let mut session_a = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session_a.set_input(make_dataset(sensor_gt)).expect("input");
        step_init(&mut session_a, None).expect("step_init");

        let mut session_b = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session_b.set_input(make_dataset(sensor_gt)).expect("input");
        step_init_with_seed(&mut session_b, ScheimpflugManualInit::default(), None)
            .expect("step_set_init");

        let k_a = session_a.state.initial_intrinsics.unwrap();
        let k_b = session_b.state.initial_intrinsics.unwrap();
        assert!((k_a.fx - k_b.fx).abs() < 1e-9);
        assert!((k_a.fy - k_b.fy).abs() < 1e-9);
    }

    #[test]
    fn step_set_init_with_intrinsics_and_sensor_seed_converges() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session.set_input(make_dataset(sensor_gt)).expect("input");

        let manual = ScheimpflugManualInit {
            intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            distortion: Some(BrownConrady5::default()),
            sensor: Some(sensor_gt),
            poses: None,
        };
        step_init_with_seed(&mut session, manual, None).expect("step_set_init");
        step_optimize(&mut session, None).expect("step_optimize");

        let output = session.output().expect("output");
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    #[test]
    fn step_set_init_rejects_wrong_pose_count() {
        use vision_calibration_core::Iso3;

        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session
            .set_input(make_dataset(ScheimpflugParams::default()))
            .expect("input");

        // Test dataset has 5 views; supply only 1 pose.
        let manual = ScheimpflugManualInit {
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
}
