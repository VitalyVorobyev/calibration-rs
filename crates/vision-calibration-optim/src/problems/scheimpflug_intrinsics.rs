//! Scheimpflug planar intrinsics optimization using the backend-agnostic IR.

use crate::Error;
use crate::backend::{BackendKind, BackendSolveOptions, SolveReport, solve_with_backend};
use crate::ir::{Bound, RobustLoss};
use crate::params::distortion::unpack_distortion;
use crate::params::intrinsics::unpack_intrinsics;
use crate::params::pose_se3::se3_dvec_to_iso3;
use crate::problems::planar_family_shared::{
    PlanarReprojectionIrOptions, PlanarSensorIrOptions, build_planar_reprojection_ir,
};
use anyhow::{Result as AnyhowResult, anyhow, ensure};
use nalgebra::DVectorView;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, DistortionFixMask, FxFyCxCySkew, IntrinsicsFixMask,
    Iso3, Pinhole, PlanarDataset, Real, ScheimpflugParams, View,
};

/// Mask for Scheimpflug tilt parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ScheimpflugFixMask {
    /// Keep `tilt_x` fixed during optimization.
    pub tilt_x: bool,
    /// Keep `tilt_y` fixed during optimization.
    pub tilt_y: bool,
}

impl ScheimpflugFixMask {
    fn as_flags(self) -> [bool; 2] {
        [self.tilt_x, self.tilt_y]
    }
}

/// Optional per-parameter box bounds for a Scheimpflug intrinsics solve.
///
/// Each `Some((lower, upper))` constrains that parameter; `None` leaves it
/// unbounded. Bounds are the safety net that converts the cold-start runaway
/// (`fx → 0`, `cx → -1e19`, reproj → 1e21 px) into a finite, recoverable solve.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ScheimpflugBounds {
    /// Bounds on `fx`.
    pub fx: Option<(f64, f64)>,
    /// Bounds on `fy`.
    pub fy: Option<(f64, f64)>,
    /// Bounds on `cx`.
    pub cx: Option<(f64, f64)>,
    /// Bounds on `cy`.
    pub cy: Option<(f64, f64)>,
    /// Bounds on `tilt_x` (radians).
    pub tilt_x: Option<(f64, f64)>,
    /// Bounds on `tilt_y` (radians).
    pub tilt_y: Option<(f64, f64)>,
}

impl ScheimpflugBounds {
    /// Per-index bounds for the `cam` block (`[fx, fy, cx, cy]`).
    fn intrinsics_bounds(&self) -> Option<Vec<Bound>> {
        let mut v = Vec::new();
        for (idx, b) in [self.fx, self.fy, self.cx, self.cy].into_iter().enumerate() {
            if let Some((lower, upper)) = b {
                v.push(Bound { idx, lower, upper });
            }
        }
        (!v.is_empty()).then_some(v)
    }

    /// Per-index bounds for the `sensor` block (`[tilt_x, tilt_y]`).
    fn sensor_bounds(&self) -> Option<Vec<Bound>> {
        let mut v = Vec::new();
        for (idx, b) in [self.tilt_x, self.tilt_y].into_iter().enumerate() {
            if let Some((lower, upper)) = b {
                v.push(Bound { idx, lower, upper });
            }
        }
        (!v.is_empty()).then_some(v)
    }

    /// Generous bounds centered on a converged intrinsics estimate: focals
    /// within `[lo, hi]×` the estimate, principal point within a focal-scaled
    /// window, and tilt within `tilt_center ± tilt_bound`. Wide enough not to
    /// bias a healthy solve, tight enough to forbid the degenerate runaway.
    fn around(k: &FxFyCxCySkew<Real>, opts: &ScheimpflugStagedInitOptions) -> Self {
        let pp_x = opts.principal_point_bound_focal_fraction * k.fx;
        let pp_y = opts.principal_point_bound_focal_fraction * k.fy;
        Self {
            fx: Some((
                opts.focal_bound_factor.0 * k.fx,
                opts.focal_bound_factor.1 * k.fx,
            )),
            fy: Some((
                opts.focal_bound_factor.0 * k.fy,
                opts.focal_bound_factor.1 * k.fy,
            )),
            cx: Some((k.cx - pp_x, k.cx + pp_x)),
            cy: Some((k.cy - pp_y, k.cy + pp_y)),
            tilt_x: Some((
                opts.tilt_center_x - opts.tilt_bound,
                opts.tilt_center_x + opts.tilt_bound,
            )),
            tilt_y: Some((
                opts.tilt_center_y - opts.tilt_bound,
                opts.tilt_center_y + opts.tilt_bound,
            )),
        }
    }
}

/// Initial/refined parameters for Scheimpflug intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsParams {
    /// Camera intrinsics.
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Brown-Conrady distortion parameters.
    pub distortion: BrownConrady5<Real>,
    /// Scheimpflug sensor tilt parameters.
    pub sensor: ScheimpflugParams,
    /// Target poses per view (`camera_se3_target`).
    pub camera_se3_target: Vec<Iso3>,
}

impl ScheimpflugIntrinsicsParams {
    /// Construct parameter pack with validation.
    pub fn new(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        sensor: ScheimpflugParams,
        camera_se3_target: Vec<Iso3>,
    ) -> Result<Self, Error> {
        if camera_se3_target.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        Ok(Self {
            intrinsics,
            distortion,
            sensor,
            camera_se3_target,
        })
    }
}

/// Solve options for Scheimpflug intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsSolveOptions {
    /// Robust loss applied per observation.
    pub robust_loss: RobustLoss,
    /// Mask for fixing intrinsics parameters.
    pub fix_intrinsics: IntrinsicsFixMask,
    /// Mask for fixing distortion parameters.
    pub fix_distortion: DistortionFixMask,
    /// Mask for fixing Scheimpflug tilt parameters.
    pub fix_scheimpflug: ScheimpflugFixMask,
    /// Indices of poses to keep fixed.
    pub fix_poses: Vec<usize>,
    /// Optional box bounds on intrinsics/tilt parameters. `None` ⇒ unbounded.
    #[serde(default)]
    pub bounds: Option<ScheimpflugBounds>,
}

impl Default for ScheimpflugIntrinsicsSolveOptions {
    fn default() -> Self {
        Self {
            robust_loss: RobustLoss::None,
            fix_intrinsics: IntrinsicsFixMask::default(),
            fix_distortion: DistortionFixMask::radial_only(),
            fix_scheimpflug: ScheimpflugFixMask::default(),
            fix_poses: vec![0],
            bounds: None,
        }
    }
}

/// Optimization result for Scheimpflug intrinsics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsEstimate {
    /// Refined parameters.
    pub params: ScheimpflugIntrinsicsParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
}

fn build_scheimpflug_intrinsics_ir(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    opts: &ScheimpflugIntrinsicsSolveOptions,
) -> AnyhowResult<(
    crate::ir::ProblemIR,
    std::collections::HashMap<String, nalgebra::DVector<f64>>,
)> {
    build_planar_reprojection_ir(
        dataset,
        &initial.intrinsics,
        &initial.distortion,
        &initial.camera_se3_target,
        &PlanarReprojectionIrOptions {
            robust_loss: opts.robust_loss,
            fix_intrinsics_indices: opts.fix_intrinsics.to_indices(),
            fix_distortion_indices: opts.fix_distortion.to_indices(),
            fix_pose_indices: opts.fix_poses.clone(),
            sensor: Some(PlanarSensorIrOptions {
                params: initial.sensor,
                fix_indices: opts.fix_scheimpflug.as_flags(),
            }),
            model: crate::ir::CameraModelDesc::PINHOLE4_DIST5_SCHEIMPFLUG2,
            intrinsics_bounds: opts
                .bounds
                .as_ref()
                .and_then(ScheimpflugBounds::intrinsics_bounds),
            sensor_bounds: opts
                .bounds
                .as_ref()
                .and_then(ScheimpflugBounds::sensor_bounds),
        },
    )
}

/// Optimize Scheimpflug intrinsics using the default tiny-solver backend.
///
/// # Errors
///
/// Returns [`Error`] if IR construction or solver backend fails.
pub fn optimize_scheimpflug_intrinsics(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    opts: ScheimpflugIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
) -> Result<ScheimpflugIntrinsicsEstimate, Error> {
    optimize_scheimpflug_intrinsics_with_backend(
        dataset,
        initial,
        opts,
        BackendKind::TinySolver,
        backend_opts,
    )
}

/// Optimize Scheimpflug intrinsics using the selected backend.
///
/// # Errors
///
/// Returns [`Error`] if IR construction or solver backend fails.
pub fn optimize_scheimpflug_intrinsics_with_backend(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    opts: ScheimpflugIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: BackendSolveOptions,
) -> Result<ScheimpflugIntrinsicsEstimate, Error> {
    let (ir, initial_map) = build_scheimpflug_intrinsics_ir(dataset, initial, &opts)?;
    let solution = solve_with_backend(backend, &ir, &initial_map, &backend_opts)?;

    let intrinsics = unpack_intrinsics(
        solution
            .params
            .get("cam")
            .ok_or_else(|| anyhow!("missing intrinsics solution block"))?
            .as_view(),
    )?;
    let distortion = unpack_distortion(
        solution
            .params
            .get("dist")
            .ok_or_else(|| anyhow!("missing distortion solution block"))?
            .as_view(),
    )?;
    let sensor = unpack_scheimpflug(
        solution
            .params
            .get("sensor")
            .ok_or_else(|| anyhow!("missing sensor solution block"))?
            .as_view(),
    )?;

    let mut optimized_poses = Vec::with_capacity(dataset.num_views());
    for view_idx in 0..dataset.num_views() {
        let key = format!("pose/{view_idx}");
        let pose = solution
            .params
            .get(&key)
            .ok_or_else(|| anyhow!("missing {key} solution block"))?;
        optimized_poses.push(se3_dvec_to_iso3(pose.as_view())?);
    }

    let mean_reproj_error =
        compute_mean_reproj_error(dataset, intrinsics, distortion, sensor, &optimized_poses);

    Ok(ScheimpflugIntrinsicsEstimate {
        params: ScheimpflugIntrinsicsParams {
            intrinsics,
            distortion,
            sensor,
            camera_se3_target: optimized_poses,
        },
        report: solution.solve_report,
        mean_reproj_error,
    })
}

/// Options controlling the staged, multi-start Scheimpflug intrinsics init.
///
/// A cold start from `tilt = 0` lands in the classic Scheimpflug degeneracy —
/// tilt trades off against principal point and radial distortion — so a plain
/// joint solve either settles biased or runs `fx → 0`. This procedure breaks
/// it by:
/// 1. a **cheap sweep**: for each tilt seed, one short solve with the principal
///    point **frozen** (so the off-axis residual is attributed to tilt), run on
///    a corner-subsampled copy of the data — enough to *rank* the tilt basins
///    without paying full-convergence cost on the dense target;
/// 2. keeping the lowest-reprojection seed; and
/// 3. refining the winning basin on the **full** data: first with the principal
///    point still frozen (L2, full budget), then a final **bounded** joint
///    refine that frees the principal point, with box bounds forbidding the
///    `fx → 0 / cx → -1e19` runaway.
///
/// This is the library form of the recipe previously prototyped in the
/// `vision-calibration-bench` multi-start staging path; the cheap-sweep split
/// keeps the cost near a single full solve even with a dense target and 5 seeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ScheimpflugStagedInitOptions {
    /// `tilt_x` seeds (radians) tried as multi-start basins. Default spans
    /// ±10° in ~5° steps — symmetric about zero, so no knowledge of the true
    /// tilt sign leaks into the procedure.
    pub tilt_x_seeds: Vec<f64>,
    /// `tilt_y` seeds (radians). Usually just `[0.0]` — `tilt_y` is small and is
    /// recovered by the free solve once `tilt_x` is in the right basin.
    pub tilt_y_seeds: Vec<f64>,
    /// Iteration budget for each cheap sweep solve (basin selection only).
    /// Clamped to the caller's `max_iters`.
    pub sweep_max_iters: usize,
    /// Corners per view kept for the cheap sweep (uniform stride). `0` ⇒ use all
    /// corners. The winning basin is always refined on the full, unsubsampled
    /// data, so this only trades sweep speed against basin-ranking fidelity.
    pub sweep_max_corners_per_view: usize,
    /// Run the final bounded refine that frees the principal point. When `false`
    /// the principal point stays at its seed value (more robust, mildly biased).
    pub refine_principal_point: bool,
    /// Focal bounds for the final refine as `[lo, hi]` multiples of the focal.
    pub focal_bound_factor: (f64, f64),
    /// Principal-point bound half-width as a fraction of the focal length.
    pub principal_point_bound_focal_fraction: f64,
    /// Tilt magnitude bound (radians) for the final refine, measured from
    /// `tilt_center_x` / `tilt_center_y`. With the defaults (`tilt_center_*
    /// = 0.0`) this produces the symmetric `(−tilt_bound, +tilt_bound)`
    /// window used by the cold-start path. Set to the seeded mount angle for
    /// warm-start use to keep the optimizer out of the degenerate high-tilt
    /// basin that lies outside the physically realistic mount-angle range.
    pub tilt_bound: f64,
    /// Center of the `tilt_x` bound window (radians). Default `0.0` gives
    /// the cold-start symmetric window; set to the seeded `tilt_x` for the
    /// warm-start path.
    pub tilt_center_x: f64,
    /// Center of the `tilt_y` bound window (radians). Default `0.0`.
    pub tilt_center_y: f64,
}

impl Default for ScheimpflugStagedInitOptions {
    fn default() -> Self {
        Self {
            tilt_x_seeds: vec![-0.175, -0.087, 0.0, 0.087, 0.175],
            tilt_y_seeds: vec![0.0],
            sweep_max_iters: 20,
            sweep_max_corners_per_view: 60,
            refine_principal_point: true,
            focal_bound_factor: (0.75, 1.5),
            principal_point_bound_focal_fraction: 0.15,
            tilt_bound: 0.30,
            tilt_center_x: 0.0,
            tilt_center_y: 0.0,
        }
    }
}

/// Run one staged sub-solve at a chosen iteration budget.
fn solve_stage(
    dataset: &PlanarDataset,
    params: &ScheimpflugIntrinsicsParams,
    opts: ScheimpflugIntrinsicsSolveOptions,
    backend: BackendKind,
    backend_opts: &BackendSolveOptions,
    iters: usize,
) -> Result<ScheimpflugIntrinsicsEstimate, Error> {
    optimize_scheimpflug_intrinsics_with_backend(
        dataset,
        params,
        opts,
        backend,
        BackendSolveOptions {
            max_iters: iters,
            ..backend_opts.clone()
        },
    )
}

/// Uniformly stride each view down to at most `max_corners` correspondences.
/// Used to make the multi-start tilt sweep cheap; the winning basin is always
/// refined on the full, unsubsampled data. `max_corners == 0` keeps every corner.
fn subsample_dataset(dataset: &PlanarDataset, max_corners: usize) -> PlanarDataset {
    let views = dataset
        .views
        .iter()
        .map(|v| {
            let n = v.obs.points_3d.len();
            if max_corners == 0 || n <= max_corners {
                return v.clone();
            }
            let stride = n.div_ceil(max_corners);
            let p3 = v.obs.points_3d.iter().step_by(stride).copied().collect();
            let p2 = v.obs.points_2d.iter().step_by(stride).copied().collect();
            View::without_meta(
                CorrespondenceView::new(p3, p2).expect("subsampled view keeps ≥4 corners"),
            )
        })
        .collect();
    PlanarDataset::new(views).expect("subsampled dataset is non-empty")
}

/// `[fx, fy]` free, principal point frozen, tilt free, distortion per the final
/// mask — the stage shape used by both the sweep and the frozen-pp refine.
fn frozen_pp_opts(
    loss: RobustLoss,
    fix_distortion: DistortionFixMask,
    fix_poses: Vec<usize>,
    bounds: Option<ScheimpflugBounds>,
) -> ScheimpflugIntrinsicsSolveOptions {
    ScheimpflugIntrinsicsSolveOptions {
        robust_loss: loss,
        fix_intrinsics: IntrinsicsFixMask {
            fx: false,
            fy: false,
            cx: true,
            cy: true,
        },
        fix_distortion,
        fix_scheimpflug: ScheimpflugFixMask::default(),
        fix_poses,
        bounds,
    }
}

/// Robustly estimate Scheimpflug intrinsics from a cold start via a staged,
/// multi-start procedure that breaks the tilt/principal-point/distortion
/// degeneracy. `final_opts` describes the final joint refine (its masks,
/// distortion fix set, robust loss and fixed poses); the earlier stages are
/// derived from it. See [`ScheimpflugStagedInitOptions`] for the algorithm.
///
/// # Errors
///
/// Returns [`Error`] if every tilt seed fails to solve.
pub fn optimize_scheimpflug_intrinsics_staged(
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    final_opts: ScheimpflugIntrinsicsSolveOptions,
    staged_opts: &ScheimpflugStagedInitOptions,
    backend_opts: BackendSolveOptions,
) -> Result<ScheimpflugIntrinsicsEstimate, Error> {
    let backend = BackendKind::TinySolver;

    // No tilt to estimate ⇒ no degeneracy ⇒ a single direct solve suffices.
    if final_opts.fix_scheimpflug.tilt_x && final_opts.fix_scheimpflug.tilt_y {
        return optimize_scheimpflug_intrinsics_with_backend(
            dataset,
            initial,
            final_opts,
            backend,
            backend_opts,
        );
    }

    let max_iters = backend_opts.max_iters.max(1);
    let sweep_iters = staged_opts.sweep_max_iters.clamp(1, max_iters);
    let fix_poses = final_opts.fix_poses.clone();
    let fix_distortion = final_opts.fix_distortion;
    let robust_loss = final_opts.robust_loss;
    let init_bounds = ScheimpflugBounds::around(&initial.intrinsics, staged_opts);

    // ── Stage 1: cheap basin sweep on corner-subsampled data ────────────────
    // One short, principal-point-frozen solve per tilt seed — enough to rank the
    // tilt basins without paying full-convergence cost on the dense target.
    let sweep_dataset = subsample_dataset(dataset, staged_opts.sweep_max_corners_per_view);
    let mut best: Option<ScheimpflugIntrinsicsEstimate> = None;
    let mut last_err: Option<Error> = None;

    for &tilt_x in &staged_opts.tilt_x_seeds {
        for &tilt_y in &staged_opts.tilt_y_seeds {
            let seed = ScheimpflugIntrinsicsParams {
                sensor: ScheimpflugParams { tilt_x, tilt_y },
                ..initial.clone()
            };
            match solve_stage(
                &sweep_dataset,
                &seed,
                frozen_pp_opts(
                    robust_loss,
                    fix_distortion,
                    fix_poses.clone(),
                    Some(init_bounds),
                ),
                backend,
                &backend_opts,
                sweep_iters,
            ) {
                Ok(candidate) if candidate.mean_reproj_error.is_finite() => {
                    let better = best
                        .as_ref()
                        .is_none_or(|b| candidate.mean_reproj_error < b.mean_reproj_error);
                    if better {
                        best = Some(candidate);
                    }
                }
                Ok(_) => {}
                Err(e) => last_err = Some(e),
            }
        }
    }

    let best = best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            Error::Numerical("all Scheimpflug tilt seeds failed to converge".to_string())
        })
    })?;

    // ── Stage 2: refine the winning basin on the FULL data ──────────────────
    // Principal point still frozen; focal, tilt, distortion, and poses free.
    // Use the caller's robust loss (not None) so that corners that are badly
    // explained by the current (biased, zero-distortion) model are
    // down-weighted. L2 here would let a few strongly-distorted edge corners
    // pull the tilt and focal toward a degenerate basin; robust loss keeps the
    // optimization within the correct basin until distortion can absorb the
    // large residuals.
    let r1 = solve_stage(
        dataset,
        &best.params,
        frozen_pp_opts(
            robust_loss,
            fix_distortion,
            fix_poses.clone(),
            Some(init_bounds),
        ),
        backend,
        &backend_opts,
        max_iters,
    )?;

    if !staged_opts.refine_principal_point {
        return Ok(r1);
    }

    // ── Stage 3: bounded joint refine that frees the principal point ─────────
    // Box bounds forbid the `fx → 0 / cx → -1e19` runaway near the solution.
    //
    // Crucially, we anchor the focal bounds to the *initial* seed (init_bounds),
    // NOT to the Stage 2 result. If Stage 2 drifted toward its lower focal
    // bound (due to a frozen pp bias or a poorly conditioned frozen-pp
    // landscape), re-centering on Stage 2 would cascade the erosion further —
    // Stage 3 would start from the wrong focal and its bounds would exclude the
    // true focal. Anchoring to init_bounds keeps the correct focal range
    // accessible regardless of Stage 2's behavior, letting Stage 3 (with the pp
    // free) find the true optimum. For cameras where Stage 2 correctly converges
    // near the seed focal, init_bounds and the old r1-centered bounds are nearly
    // identical, so this change is a no-op for well-behaved cameras.
    let refine_opts = ScheimpflugIntrinsicsSolveOptions {
        robust_loss: RobustLoss::None,
        bounds: Some(init_bounds),
        ..final_opts
    };
    match solve_stage(
        dataset,
        &r1.params,
        refine_opts,
        backend,
        &backend_opts,
        max_iters,
    ) {
        // Accept only if freeing the principal point did not regress the fit.
        Ok(refined)
            if refined.mean_reproj_error.is_finite()
                && refined.mean_reproj_error <= r1.mean_reproj_error + 1e-9 =>
        {
            Ok(refined)
        }
        _ => Ok(r1),
    }
}

fn unpack_scheimpflug(values: DVectorView<'_, f64>) -> AnyhowResult<ScheimpflugParams> {
    ensure!(values.len() == 2, "scheimpflug block must have 2 entries");
    Ok(ScheimpflugParams {
        tilt_x: values[0],
        tilt_y: values[1],
    })
}

fn compute_mean_reproj_error(
    dataset: &PlanarDataset,
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    poses: &[Iso3],
) -> f64 {
    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
    let mut sum = 0.0;
    let mut count = 0usize;

    for (view, pose) in dataset.views.iter().zip(poses.iter()) {
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2d).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }

    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use vision_calibration_core::{CorrespondenceView, Pt2, Pt3, View};

    fn gt_camera() -> (FxFyCxCySkew<f64>, BrownConrady5<f64>, ScheimpflugParams) {
        // Principal point intentionally off image center, a large ~-5.7° tilt
        // (between two sweep seeds, so the sweep cannot trivially land on it),
        // and non-trivial radial distortion.
        let k = FxFyCxCySkew {
            fx: 1150.0,
            fy: 1155.0,
            cx: 372.0,
            cy: 258.0,
            skew: 0.0,
        };
        let dist = BrownConrady5 {
            k1: -0.12,
            k2: 0.03,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let sensor = ScheimpflugParams {
            tilt_x: -0.10,
            tilt_y: 0.008,
        };
        (k, dist, sensor)
    }

    fn make_pose(rx: f64, ry: f64, rz: f64, tx: f64, ty: f64, tz: f64) -> Iso3 {
        Iso3::from_parts(
            Translation3::new(tx, ty, tz),
            UnitQuaternion::from_euler_angles(rx, ry, rz),
        )
    }

    /// Varied viewpoints — perspective + off-axis coverage make the tilt observable.
    fn gt_poses() -> Vec<Iso3> {
        vec![
            make_pose(0.0, 0.0, 0.0, -0.01, -0.01, 0.45),
            make_pose(0.20, 0.0, 0.0, -0.02, 0.0, 0.50),
            make_pose(-0.20, 0.0, 0.0, 0.01, -0.02, 0.48),
            make_pose(0.0, 0.20, 0.0, -0.01, 0.01, 0.52),
            make_pose(0.0, -0.20, 0.0, 0.02, -0.01, 0.47),
            make_pose(0.15, 0.15, 0.0, -0.02, -0.02, 0.55),
            make_pose(-0.15, 0.15, 0.05, 0.01, 0.0, 0.49),
            make_pose(0.15, -0.15, -0.05, -0.01, 0.02, 0.53),
            make_pose(-0.15, -0.15, 0.0, 0.0, -0.01, 0.46),
            make_pose(0.10, -0.10, 0.10, -0.02, 0.01, 0.51),
            make_pose(-0.10, 0.10, -0.10, 0.01, -0.02, 0.50),
            make_pose(0.25, -0.05, 0.05, -0.01, -0.01, 0.58),
        ]
    }

    fn build_dataset(
        k: FxFyCxCySkew<f64>,
        dist: BrownConrady5<f64>,
        sensor: ScheimpflugParams,
        poses: &[Iso3],
    ) -> PlanarDataset {
        let camera = Camera::new(Pinhole, dist, sensor.compile(), k);
        // 9×9 planar target, 2 cm spacing, centered on the origin (~16 cm board).
        let mut object = Vec::new();
        for iy in 0..9 {
            for ix in 0..9 {
                object.push(Pt3::new(
                    (ix as f64 - 4.0) * 0.02,
                    (iy as f64 - 4.0) * 0.02,
                    0.0,
                ));
            }
        }
        let mut views = Vec::new();
        for pose in poses {
            let mut p3: Vec<Pt3> = Vec::new();
            let mut p2: Vec<Pt2> = Vec::new();
            for o in &object {
                let p_cam = pose.transform_point(o);
                if let Some(uv) = camera.project_point(&p_cam) {
                    p3.push(*o);
                    p2.push(uv);
                }
            }
            views.push(View::without_meta(CorrespondenceView::new(p3, p2).unwrap()));
        }
        PlanarDataset::new(views).unwrap()
    }

    #[test]
    fn staged_init_recovers_large_tilt_from_cold_start() {
        let (gt_k, gt_dist, gt_sensor) = gt_camera();
        let poses = gt_poses();
        let dataset = build_dataset(gt_k, gt_dist, gt_sensor, &poses);

        // Cold start: focal 5% off, principal point at a nominal center (not the
        // true off-center pp), zero distortion, and crucially tilt = 0.
        let initial = ScheimpflugIntrinsicsParams::new(
            FxFyCxCySkew {
                fx: 1150.0 * 1.05,
                fy: 1155.0 * 1.05,
                cx: 360.0,
                cy: 270.0,
                skew: 0.0,
            },
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
            ScheimpflugParams {
                tilt_x: 0.0,
                tilt_y: 0.0,
            },
            poses.clone(),
        )
        .unwrap();

        let final_opts = ScheimpflugIntrinsicsSolveOptions::default();
        let est = optimize_scheimpflug_intrinsics_staged(
            &dataset,
            &initial,
            final_opts,
            &ScheimpflugStagedInitOptions::default(),
            BackendSolveOptions {
                max_iters: 80,
                verbosity: 0,
                ..Default::default()
            },
        )
        .unwrap();

        let r = &est.params;
        eprintln!(
            "recovered: fx={:.1} fy={:.1} cx={:.1} cy={:.1} k1={:.3} k2={:.3} tau=({:.4},{:.4}) reproj={:.4}px",
            r.intrinsics.fx,
            r.intrinsics.fy,
            r.intrinsics.cx,
            r.intrinsics.cy,
            r.distortion.k1,
            r.distortion.k2,
            r.sensor.tilt_x,
            r.sensor.tilt_y,
            est.mean_reproj_error,
        );

        assert!(
            (r.sensor.tilt_x - (-0.10)).abs() < 0.01,
            "tilt_x not recovered: {} (want -0.10)",
            r.sensor.tilt_x
        );
        assert!(
            (r.sensor.tilt_y - 0.008).abs() < 0.01,
            "tilt_y not recovered: {} (want 0.008)",
            r.sensor.tilt_y
        );
        assert!(
            (r.intrinsics.fx - 1150.0).abs() / 1150.0 < 0.02,
            "fx not recovered: {}",
            r.intrinsics.fx
        );
        assert!(
            (r.intrinsics.fy - 1155.0).abs() / 1155.0 < 0.02,
            "fy not recovered: {}",
            r.intrinsics.fy
        );
        assert!(
            (r.intrinsics.cx - 372.0).abs() < 8.0,
            "cx not recovered: {}",
            r.intrinsics.cx
        );
        assert!(
            (r.intrinsics.cy - 258.0).abs() < 8.0,
            "cy not recovered: {}",
            r.intrinsics.cy
        );
        assert!(
            est.mean_reproj_error < 0.1,
            "reprojection error too high: {}",
            est.mean_reproj_error
        );
    }
}
