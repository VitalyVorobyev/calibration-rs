//! Staged, multi-start planar solve for a tilted (Scheimpflug) sensor.
//!
//! A tilted sensor's tilt trades off against the principal point and radial
//! distortion, so a plain solve from `tilt = 0` settles biased or runs
//! `fx → 0`. [`optimize_scheimpflug_intrinsics_staged`] breaks that
//! degeneracy with a cheap multi-start tilt sweep followed by staged
//! refinement, every stage a [`optimize_planar_intrinsics`] solve.

use crate::Error;
use crate::backend::BackendSolveOptions;
use crate::ir::SensorKind;
use crate::problems::planar_intrinsics::{
    CameraBounds, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
    optimize_planar_intrinsics,
};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraFix, CameraParams, CorrespondenceView, FxFyCxCy, PlanarDataset, Real, ScheimpflugParams,
    SensorParams, View,
};

/// Generous bounds centered on an intrinsics estimate: focals within
/// `[lo, hi]×` the estimate, principal point within a focal-scaled window,
/// and tilt within `tilt_center ± tilt_bound`. Wide enough not to bias a
/// healthy solve, tight enough to forbid the degenerate runaway.
fn bounds_around(k: &FxFyCxCy<Real>, opts: &ScheimpflugStagedInitOptions) -> CameraBounds {
    let pp_x = opts.principal_point_bound_focal_fraction * k.fx;
    let pp_y = opts.principal_point_bound_focal_fraction * k.fy;
    CameraBounds {
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
///    point still frozen (full budget), then a final **bounded** joint refine
///    that frees the principal point, with box bounds forbidding the
///    `fx → 0 / cx → -1e19` runaway. Both refines use the caller's robust
///    loss, and the last is kept unless it raises that objective.
///
/// This is the library form of the multi-start staging recipe also used by the
/// `vision-calibration-bench` staging path; the cheap-sweep split keeps the cost near a single full solve even with a dense target and 5 seeds.
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
/// Relative slack when Stage 3 is accepted on the objective: it starts at
/// Stage 2's solution, so it can only lower the objective up to the
/// backend's floating-point noise.
const OBJECTIVE_REL_TOL: f64 = 1e-12;

fn solve_stage(
    dataset: &PlanarDataset,
    params: &PlanarIntrinsicsParams,
    opts: PlanarIntrinsicsSolveOptions,
    backend_opts: &BackendSolveOptions,
    iters: usize,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    optimize_planar_intrinsics(
        dataset,
        params,
        opts,
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
/// fix list — the stage shape used by both the sweep and the frozen-pp refine.
fn frozen_pp_opts(
    final_opts: &PlanarIntrinsicsSolveOptions,
    bounds: CameraBounds,
) -> PlanarIntrinsicsSolveOptions {
    PlanarIntrinsicsSolveOptions {
        robust_loss: final_opts.robust_loss,
        fix: CameraFix {
            intrinsics: Some(vec!["cx".to_string(), "cy".to_string()]),
            sensor: Some(Vec::new()),
            ..final_opts.fix.clone()
        },
        fix_poses: final_opts.fix_poses.clone(),
        bounds: Some(bounds),
    }
}

/// `initial` with its sensor tilt replaced.
fn with_tilt(initial: &PlanarIntrinsicsParams, tilt: ScheimpflugParams) -> PlanarIntrinsicsParams {
    PlanarIntrinsicsParams {
        camera: CameraParams {
            sensor: SensorParams::Scheimpflug { params: tilt },
            ..initial.camera.clone()
        },
        camera_se3_target: initial.camera_se3_target.clone(),
    }
}

/// Robustly estimate a tilted camera from a cold start via a staged,
/// multi-start procedure that breaks the tilt/principal-point/distortion
/// degeneracy. `final_opts` describes the final joint refine (its fix lists,
/// robust loss and fixed poses); the earlier stages are derived from it. See
/// [`ScheimpflugStagedInitOptions`] for the algorithm. When `final_opts`
/// fixes both tilts there is no degeneracy to break, and this is a single
/// direct solve.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when `initial.camera` does not have a
///   Scheimpflug sensor, or a fix list names a parameter its model does not
///   have.
/// - The last solver error when every tilt seed fails to solve.
pub fn optimize_scheimpflug_intrinsics_staged(
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    final_opts: PlanarIntrinsicsSolveOptions,
    staged_opts: &ScheimpflugStagedInitOptions,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    let spec = initial
        .camera
        .spec()
        .filter(|spec| spec.sensor == SensorKind::Scheimpflug)
        .ok_or_else(|| {
            Error::invalid_input("the staged Scheimpflug solve needs a Scheimpflug sensor")
        })?;

    // No tilt to estimate ⇒ no degeneracy ⇒ a single direct solve suffices.
    if final_opts.fix.resolve(spec)?.sensor.len() == spec.sensor.dim() {
        return optimize_planar_intrinsics(dataset, initial, final_opts, backend_opts);
    }

    let max_iters = backend_opts.max_iters.max(1);
    let sweep_iters = staged_opts.sweep_max_iters.clamp(1, max_iters);
    let init_bounds = bounds_around(&initial.camera.intrinsics, staged_opts);

    // ── Stage 1: cheap basin sweep on corner-subsampled data ────────────────
    // One short, principal-point-frozen solve per tilt seed — enough to rank the
    // tilt basins without paying full-convergence cost on the dense target.
    let sweep_dataset = subsample_dataset(dataset, staged_opts.sweep_max_corners_per_view);
    let mut best: Option<PlanarIntrinsicsEstimate> = None;
    let mut last_err: Option<Error> = None;

    for &tilt_x in &staged_opts.tilt_x_seeds {
        for &tilt_y in &staged_opts.tilt_y_seeds {
            let seed = with_tilt(initial, ScheimpflugParams { tilt_x, tilt_y });
            match solve_stage(
                &sweep_dataset,
                &seed,
                frozen_pp_opts(&final_opts, init_bounds),
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
        frozen_pp_opts(&final_opts, init_bounds),
        &backend_opts,
        max_iters,
    )?;

    if !staged_opts.refine_principal_point {
        return Ok(r1);
    }

    // ── Stage 3: bounded joint refine that frees the principal point ─────────
    // Box bounds forbid the `fx → 0 / cx → -1e19` runaway near the solution.
    //
    // The focal bounds are anchored to the *initial* seed (init_bounds), not
    // to the Stage 2 result. If Stage 2 drifted toward its lower focal bound
    // (a frozen-pp bias or a poorly conditioned frozen-pp landscape),
    // re-centering on Stage 2 would cascade the erosion: Stage 3 would start
    // from the wrong focal and its bounds would exclude the true focal.
    // Anchoring to init_bounds keeps the correct focal range reachable whatever
    // Stage 2 did, letting Stage 3 (with the pp free) find the true optimum.
    //
    // It solves under the caller's robust loss, as Stage 2 did, and is
    // accepted on that objective: both stages minimize it over the same
    // residuals, so it is the one measure that compares them. The mean
    // reprojection error would count outliers at full weight and reject a
    // solve that explains the inliers better.
    let refine_opts = PlanarIntrinsicsSolveOptions {
        bounds: Some(init_bounds),
        ..final_opts
    };
    match solve_stage(dataset, &r1.params, refine_opts, &backend_opts, max_iters) {
        Ok(refined)
            if refined.report.final_cost.is_finite()
                && refined.report.final_cost
                    <= r1.report.final_cost * (1.0 + OBJECTIVE_REL_TOL) =>
        {
            Ok(refined)
        }
        _ => Ok(r1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use vision_calibration_core::{
        BrownConrady5, Camera, CorrespondenceView, DistortionParams, Iso3, Pinhole,
        ProjectionParams, Pt2, Pt3, View,
    };

    fn gt_camera() -> (FxFyCxCy<f64>, BrownConrady5<f64>, ScheimpflugParams) {
        // Principal point intentionally off image center, a large ~-5.7° tilt
        // (between two sweep seeds, so the sweep cannot trivially land on it),
        // and non-trivial radial distortion.
        let k = FxFyCxCy {
            fx: 1150.0,
            fy: 1155.0,
            cx: 372.0,
            cy: 258.0,
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
        k: FxFyCxCy<f64>,
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

    /// The cold start of `staged_init_recovers_large_tilt_from_cold_start`.
    fn cold_start(poses: &[Iso3]) -> PlanarIntrinsicsParams {
        PlanarIntrinsicsParams::new(
            CameraParams {
                projection: ProjectionParams::Pinhole,
                distortion: DistortionParams::BrownConrady5 {
                    params: BrownConrady5::default(),
                },
                sensor: SensorParams::Scheimpflug {
                    params: ScheimpflugParams::default(),
                },
                intrinsics: FxFyCxCy {
                    fx: 1150.0 * 1.05,
                    fy: 1155.0 * 1.05,
                    cx: 360.0,
                    cy: 270.0,
                },
            },
            poses.to_vec(),
        )
        .unwrap()
    }

    /// Under a robust loss with outliers, the last stage, which frees the
    /// principal point, is kept: the principal point and tilt end within 3σ
    /// of the truth. Judged on the mean reprojection error, outliers at full
    /// weight, that stage was rejected and the principal point stayed at its
    /// seed, about 17 px off.
    #[test]
    fn staged_init_frees_the_principal_point_under_a_robust_loss() {
        use crate::ir::RobustLoss;
        use vision_calibration_core::synthetic::noise::UniformPixelNoise;

        let (gt_k, gt_dist, gt_sensor) = gt_camera();
        let poses = gt_poses();
        let clean = build_dataset(gt_k, gt_dist, gt_sensor, &poses);
        // σ ≈ 0.1 px of noise, and every 20th corner displaced by 10–30 px.
        let noise = UniformPixelNoise {
            seed: 7,
            max_abs_px: 0.17,
        };
        let views = clean
            .views
            .iter()
            .enumerate()
            .map(|(v, view)| {
                let p2 = view
                    .obs
                    .points_2d
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let mut q = Pt2::from(noise.apply(v, i, p.coords));
                        if (v * 81 + i) % 20 == 0 {
                            let d = 10.0 + ((v + i) % 5) as f64 * 5.0;
                            q += nalgebra::Vector2::new(d, -0.5 * d);
                        }
                        q
                    })
                    .collect();
                View::without_meta(CorrespondenceView::new(view.obs.points_3d.clone(), p2).unwrap())
            })
            .collect();
        let dataset = PlanarDataset::new(views).unwrap();

        for loss in [
            RobustLoss::Huber { scale: 1.0 },
            RobustLoss::Cauchy { scale: 1.0 },
        ] {
            let est = optimize_scheimpflug_intrinsics_staged(
                &dataset,
                &cold_start(&poses),
                PlanarIntrinsicsSolveOptions {
                    robust_loss: loss,
                    ..Default::default()
                },
                &ScheimpflugStagedInitOptions::default(),
                BackendSolveOptions {
                    max_iters: 80,
                    ..Default::default()
                },
            )
            .unwrap();
            let cov = est
                .uncertainty
                .covariance()
                .unwrap_or_else(|| panic!("{loss:?}: a covariance"));
            let r = &est.params.camera;
            let SensorParams::Scheimpflug { params: tilt } = r.sensor else {
                panic!("Scheimpflug sensor expected");
            };
            for (label, error) in [
                ("cx", r.intrinsics.cx - gt_k.cx),
                ("cy", r.intrinsics.cy - gt_k.cy),
                ("tilt_x", tilt.tilt_x - gt_sensor.tilt_x),
                ("tilt_y", tilt.tilt_y - gt_sensor.tilt_y),
            ] {
                let sigma = cov
                    .std_dev(label)
                    .unwrap_or_else(|| panic!("{loss:?}: {label} was not estimated"));
                assert!(
                    error.abs() <= 3.0 * sigma,
                    "{loss:?} {label}: error {error:.3e}, σ {sigma:.3e}"
                );
            }
        }
    }

    #[test]
    fn staged_init_recovers_large_tilt_from_cold_start() {
        let (gt_k, gt_dist, gt_sensor) = gt_camera();
        let poses = gt_poses();
        let dataset = build_dataset(gt_k, gt_dist, gt_sensor, &poses);

        // Cold start: focal 5% off, principal point at a nominal center (not the
        // true off-center pp), zero distortion, and crucially tilt = 0.
        let initial = PlanarIntrinsicsParams::new(
            CameraParams {
                projection: ProjectionParams::Pinhole,
                distortion: DistortionParams::BrownConrady5 {
                    params: BrownConrady5 {
                        k1: 0.0,
                        k2: 0.0,
                        k3: 0.0,
                        p1: 0.0,
                        p2: 0.0,
                        iters: 8,
                    },
                },
                sensor: SensorParams::Scheimpflug {
                    params: ScheimpflugParams {
                        tilt_x: 0.0,
                        tilt_y: 0.0,
                    },
                },
                intrinsics: FxFyCxCy {
                    fx: 1150.0 * 1.05,
                    fy: 1155.0 * 1.05,
                    cx: 360.0,
                    cy: 270.0,
                },
            },
            poses.clone(),
        )
        .unwrap();

        let final_opts = PlanarIntrinsicsSolveOptions::default();
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

        let r = &est.params.camera;
        let DistortionParams::BrownConrady5 { params: r_dist } = est.params.camera.distortion
        else {
            panic!("Brown-Conrady result")
        };
        let SensorParams::Scheimpflug { params: tilt } = r.sensor else {
            panic!("Scheimpflug sensor expected, got {:?}", r.sensor);
        };
        eprintln!(
            "recovered: fx={:.1} fy={:.1} cx={:.1} cy={:.1} k1={:.3} k2={:.3} tau=({:.4},{:.4}) reproj={:.4}px",
            r.intrinsics.fx,
            r.intrinsics.fy,
            r.intrinsics.cx,
            r.intrinsics.cy,
            r_dist.k1,
            r_dist.k2,
            tilt.tilt_x,
            tilt.tilt_y,
            est.mean_reproj_error,
        );

        assert!(
            (tilt.tilt_x - (-0.10)).abs() < 0.01,
            "tilt_x not recovered: {} (want -0.10)",
            tilt.tilt_x
        );
        assert!(
            (tilt.tilt_y - 0.008).abs() < 0.01,
            "tilt_y not recovered: {} (want 0.008)",
            tilt.tilt_y
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

    #[test]
    fn staged_solve_rejects_an_untilted_camera() {
        let (gt_k, gt_dist, gt_sensor) = gt_camera();
        let poses = gt_poses();
        let dataset = build_dataset(gt_k, gt_dist, gt_sensor, &poses);
        let initial = PlanarIntrinsicsParams::new_from_components(gt_k, gt_dist, poses).unwrap();
        let err = optimize_scheimpflug_intrinsics_staged(
            &dataset,
            &initial,
            PlanarIntrinsicsSolveOptions::default(),
            &ScheimpflugStagedInitOptions::default(),
            BackendSolveOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("Scheimpflug sensor"), "{err}");
    }
}
