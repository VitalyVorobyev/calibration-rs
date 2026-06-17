//! Tilt-aware initialization for planar Scheimpflug intrinsics.
//!
//! A tilted sensor is a projective transform between distorted normalized
//! coordinates and the physical sensor plane:
//!
//! `pixel = K * T_scheimpflug * distortion(project(point))`.
//!
//! Plain Zhang initialization sees only the combined projective camera and can
//! trade sensor tilt against focal length and radial distortion. This module
//! performs a deterministic bounded search over plausible Scheimpflug tilts,
//! removes each candidate tilt in normalized coordinates, reuses the existing
//! iterative pinhole/distortion initializer, and scores the complete seed
//! through the full Scheimpflug projection model.

use crate::{
    Error,
    distortion_fit::DistortionFitOptions,
    homography::dlt_homography,
    iterative_intrinsics::{IterativeIntrinsicsOptions, estimate_intrinsics_iterative},
    math::null_space,
    planar_pose::estimate_planar_pose_from_h,
    zhang_intrinsics::PlanarIntrinsicsLinearInit,
};
use nalgebra::{DMatrix, SMatrix, SVector, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, DistortionModel, FxFyCxCySkew, IntrinsicsModel,
    Iso3, Mat3, NoMeta, Pinhole, PinholeCamera, PlanarDataset, Pt2, Real, ScheimpflugParams,
    SensorModel, View,
};

/// Options controlling tilt-aware Scheimpflug planar intrinsics initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsInitOptions {
    /// Number of iterative K/distortion refinement rounds per tilt candidate.
    pub iterations: usize,
    /// Brown-Conrady coefficients estimated during the linear initialization.
    pub distortion_opts: DistortionFitOptions,
    /// Force skew to zero after each candidate K estimate.
    pub zero_skew: bool,
    /// Initial `tilt_x` candidates, in radians.
    pub tilt_x_seeds: Vec<Real>,
    /// Initial `tilt_y` candidates, in radians.
    pub tilt_y_seeds: Vec<Real>,
    /// Number of local grid-refinement rounds around the best candidate.
    pub refine_rounds: usize,
    /// First local-refinement half-step, in radians.
    pub refine_step: Real,
    /// Maximum absolute tilt accepted for either axis, in radians.
    pub max_abs_tilt: Real,
}

impl Default for ScheimpflugIntrinsicsInitOptions {
    fn default() -> Self {
        Self {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_tangential: true,
                fix_k3: true,
                iters: 8,
            },
            zero_skew: true,
            // Covers roughly +/-9 degrees, including the common +/-5 degree
            // Scheimpflug region, without assuming the sign.
            tilt_x_seeds: vec![-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16],
            tilt_y_seeds: vec![-0.04, -0.02, 0.0, 0.02, 0.04],
            refine_rounds: 2,
            refine_step: 0.02,
            max_abs_tilt: 0.30,
        }
    }
}

/// Result of tilt-aware Scheimpflug planar intrinsics initialization.
#[derive(Debug, Clone)]
pub struct ScheimpflugIntrinsicsLinearInit {
    /// Initial pinhole intrinsics and Brown-Conrady distortion.
    pub camera: PinholeCamera,
    /// Initial Scheimpflug sensor tilt.
    pub sensor: ScheimpflugParams,
    /// Initial target-to-camera poses (`camera_se3_target`).
    pub poses: Vec<Iso3>,
    /// Mean full-model reprojection error of the initialized seed, in pixels.
    pub mean_reproj_error: Real,
}

#[derive(Debug, Clone)]
struct Candidate {
    init: ScheimpflugIntrinsicsLinearInit,
    score: Real,
}

#[derive(Debug, Clone, Copy)]
struct CandidateSeed {
    reference_k: FxFyCxCySkew<Real>,
    sensor: ScheimpflugParams,
}

/// Estimate Scheimpflug intrinsics, distortion, tilt, and per-view poses.
///
/// The input observations are pixel coordinates in the image frame. The
/// returned poses are `camera_se3_target`, matching the rest of the planar
/// calibration pipeline.
///
/// # Errors
///
/// Returns [`Error`] if there are fewer than three views, the tilt search has no
/// valid candidates, homographies are degenerate, or pose recovery fails.
pub fn estimate_scheimpflug_intrinsics_iterative(
    dataset: &PlanarDataset,
    opts: ScheimpflugIntrinsicsInitOptions,
) -> Result<ScheimpflugIntrinsicsLinearInit, Error> {
    validate_options(dataset, &opts)?;

    let pinhole_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations,
        distortion_opts: opts.distortion_opts,
        zero_skew: opts.zero_skew,
    };
    let base = estimate_intrinsics_iterative(dataset, pinhole_opts)?;
    let observed_center = observed_pixel_center(dataset);
    let mut search = initial_seed_grid(dataset, &base.k, &opts)?;
    let mut global_best: Option<Candidate> = None;
    let mut step = opts.refine_step;

    for round in 0..=opts.refine_rounds {
        let mut round_best: Option<Candidate> = None;
        for seed in search.drain(..) {
            let candidate = match evaluate_candidate(
                dataset,
                &seed.reference_k,
                seed.sensor,
                pinhole_opts,
                observed_center,
            ) {
                Ok(candidate) => candidate,
                Err(_) => continue,
            };
            if is_better(&candidate, round_best.as_ref()) {
                round_best = Some(candidate.clone());
            }
            if is_better(&candidate, global_best.as_ref()) {
                global_best = Some(candidate);
            }
        }

        let Some(best) = round_best.as_ref().or(global_best.as_ref()) else {
            return Err(Error::numerical(
                "all Scheimpflug initialization tilt candidates failed",
            ));
        };

        if round < opts.refine_rounds {
            let reference_k = best.init.camera.k;
            search = local_seed_grid(reference_k, best.init.sensor, step, &opts);
            step *= 0.5;
        }
    }

    global_best
        .map(|candidate| candidate.init)
        .ok_or_else(|| Error::numerical("Scheimpflug initialization produced no candidates"))
}

fn validate_options(
    dataset: &PlanarDataset,
    opts: &ScheimpflugIntrinsicsInitOptions,
) -> Result<(), Error> {
    if dataset.num_views() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: dataset.num_views(),
        });
    }
    if opts.tilt_x_seeds.is_empty() || opts.tilt_y_seeds.is_empty() {
        return Err(Error::invalid_input(
            "Scheimpflug init requires at least one tilt_x and tilt_y seed",
        ));
    }
    if opts.max_abs_tilt <= 0.0 || !opts.max_abs_tilt.is_finite() {
        return Err(Error::invalid_input("max_abs_tilt must be positive"));
    }
    if opts.refine_rounds > 0 && (opts.refine_step <= 0.0 || !opts.refine_step.is_finite()) {
        return Err(Error::invalid_input(
            "refine_step must be positive when refine_rounds > 0",
        ));
    }
    Ok(())
}

fn initial_seed_grid(
    dataset: &PlanarDataset,
    fallback_k: &FxFyCxCySkew<Real>,
    opts: &ScheimpflugIntrinsicsInitOptions,
) -> Result<Vec<CandidateSeed>, Error> {
    let mut out = homography_decomposition_seeds(dataset, opts).unwrap_or_default();
    for sensor in seed_grid(opts)? {
        push_candidate_seed(&mut out, *fallback_k, sensor);
    }
    Ok(out)
}

fn seed_grid(opts: &ScheimpflugIntrinsicsInitOptions) -> Result<Vec<ScheimpflugParams>, Error> {
    let mut out = Vec::with_capacity(opts.tilt_x_seeds.len() * opts.tilt_y_seeds.len());
    for &tilt_x in &opts.tilt_x_seeds {
        for &tilt_y in &opts.tilt_y_seeds {
            push_seed(&mut out, tilt_x, tilt_y, opts.max_abs_tilt)?;
        }
    }
    Ok(out)
}

fn local_seed_grid(
    reference_k: FxFyCxCySkew<Real>,
    center: ScheimpflugParams,
    step: Real,
    opts: &ScheimpflugIntrinsicsInitOptions,
) -> Vec<CandidateSeed> {
    let mut out = Vec::with_capacity(9);
    for dx in [-step, 0.0, step] {
        for dy in [-step, 0.0, step] {
            let tilt_x = center.tilt_x + dx;
            let tilt_y = center.tilt_y + dy;
            if tilt_x.abs() <= opts.max_abs_tilt && tilt_y.abs() <= opts.max_abs_tilt {
                push_candidate_seed(&mut out, reference_k, ScheimpflugParams { tilt_x, tilt_y });
            }
        }
    }
    out
}

fn push_candidate_seed(
    seeds: &mut Vec<CandidateSeed>,
    reference_k: FxFyCxCySkew<Real>,
    sensor: ScheimpflugParams,
) {
    if seeds.iter().any(|s| {
        (s.sensor.tilt_x - sensor.tilt_x).abs() < 1e-12
            && (s.sensor.tilt_y - sensor.tilt_y).abs() < 1e-12
            && (s.reference_k.fx - reference_k.fx).abs() < 1e-9
            && (s.reference_k.fy - reference_k.fy).abs() < 1e-9
            && (s.reference_k.cx - reference_k.cx).abs() < 1e-9
            && (s.reference_k.cy - reference_k.cy).abs() < 1e-9
    }) {
        return;
    }
    seeds.push(CandidateSeed {
        reference_k,
        sensor,
    });
}

fn push_seed(
    seeds: &mut Vec<ScheimpflugParams>,
    tilt_x: Real,
    tilt_y: Real,
    max_abs_tilt: Real,
) -> Result<(), Error> {
    if !tilt_x.is_finite() || !tilt_y.is_finite() {
        return Err(Error::invalid_input("tilt seeds must be finite"));
    }
    if tilt_x.abs() > max_abs_tilt || tilt_y.abs() > max_abs_tilt {
        return Ok(());
    }
    let seed = ScheimpflugParams { tilt_x, tilt_y };
    if !seeds
        .iter()
        .any(|s| (s.tilt_x - tilt_x).abs() < 1e-12 && (s.tilt_y - tilt_y).abs() < 1e-12)
    {
        seeds.push(seed);
    }
    Ok(())
}

fn homography_decomposition_seeds(
    dataset: &PlanarDataset,
    opts: &ScheimpflugIntrinsicsInitOptions,
) -> Result<Vec<CandidateSeed>, Error> {
    let homographies = view_homographies(dataset)?;
    let effective_k = PlanarIntrinsicsLinearInit::from_homographies(&homographies)?;
    let omega = estimate_absolute_conic(&homographies)?;
    let mut out = Vec::new();
    for &tilt_x in &opts.tilt_x_seeds {
        for &tilt_y in &opts.tilt_y_seeds {
            if tilt_x.abs() > opts.max_abs_tilt || tilt_y.abs() > opts.max_abs_tilt {
                continue;
            }
            if let Some(seed) = fit_zero_skew_k_and_tilt(
                &omega,
                effective_k,
                ScheimpflugParams { tilt_x, tilt_y },
                opts.max_abs_tilt,
            ) {
                push_candidate_seed(&mut out, seed.reference_k, seed.sensor);
            }
        }
    }
    if out.is_empty() {
        return Err(Error::numerical(
            "homography Scheimpflug decomposition produced no candidates",
        ));
    }
    Ok(out)
}

fn view_homographies(dataset: &PlanarDataset) -> Result<Vec<Mat3>, Error> {
    let mut homographies = Vec::with_capacity(dataset.num_views());
    for (idx, view) in dataset.views.iter().enumerate() {
        let board = view.obs.planar_points();
        let h = dlt_homography(&board, &view.obs.points_2d).map_err(|e| {
            Error::numerical(format!(
                "Scheimpflug init homography failed for view {idx}: {e}"
            ))
        })?;
        homographies.push(h);
    }
    Ok(homographies)
}

fn v_ij(h: &Mat3, i: usize, j: usize) -> SVector<Real, 6> {
    let hi = h.column(i);
    let hj = h.column(j);
    SVector::<Real, 6>::from_row_slice(&[
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])
}

fn estimate_absolute_conic(homographies: &[Mat3]) -> Result<Mat3, Error> {
    if homographies.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: homographies.len(),
        });
    }
    let mut design = DMatrix::<Real>::zeros(2 * homographies.len(), 6);
    for (idx, h) in homographies.iter().enumerate() {
        let v11 = v_ij(h, 0, 0);
        let v22 = v_ij(h, 1, 1);
        let v12 = v_ij(h, 0, 1);
        design.row_mut(2 * idx).copy_from(&v12.transpose());
        design
            .row_mut(2 * idx + 1)
            .copy_from(&(v11 - v22).transpose());
    }
    let b = null_space(&design)?.vector;
    let sign = if b[5] < 0.0 { -1.0 } else { 1.0 };
    let omega = Mat3::new(
        sign * b[0],
        sign * b[1],
        sign * b[3],
        sign * b[1],
        sign * b[2],
        sign * b[4],
        sign * b[3],
        sign * b[4],
        sign * b[5],
    );
    normalize_symmetric_matrix(omega)
        .ok_or_else(|| Error::numerical("Scheimpflug init absolute conic estimate is non-finite"))
}

fn fit_zero_skew_k_and_tilt(
    target_omega: &Mat3,
    effective_k: FxFyCxCySkew<Real>,
    sensor_seed: ScheimpflugParams,
    max_abs_tilt: Real,
) -> Option<CandidateSeed> {
    let f_scale = (0.5 * (effective_k.fx.abs() + effective_k.fy.abs())).max(1.0);
    let mut p = SVector::<Real, 6>::new(
        effective_k.fx.max(1.0).ln(),
        effective_k.fy.max(1.0).ln(),
        effective_k.cx / f_scale,
        effective_k.cy / f_scale,
        sensor_seed.tilt_x.clamp(-max_abs_tilt, max_abs_tilt),
        sensor_seed.tilt_y.clamp(-max_abs_tilt, max_abs_tilt),
    );
    let mut lambda = 1e-6;
    let mut best_res = conic_residual(&decode_params(&p, f_scale, max_abs_tilt)?, target_omega)?;

    for _ in 0..24 {
        let mut jac = SMatrix::<Real, 6, 6>::zeros();
        let eps = 1e-5;
        for col in 0..6 {
            let mut pp = p;
            pp[col] += eps;
            let rp = conic_residual(&decode_params(&pp, f_scale, max_abs_tilt)?, target_omega)?;
            let mut pm = p;
            pm[col] -= eps;
            let rm = conic_residual(&decode_params(&pm, f_scale, max_abs_tilt)?, target_omega)?;
            jac.set_column(col, &((rp - rm) / (2.0 * eps)));
        }
        let mut lhs = jac.transpose() * jac;
        for i in 0..6 {
            lhs[(i, i)] += lambda;
        }
        let rhs = -(jac.transpose() * best_res);
        let mut step = lhs.lu().solve(&rhs)?;
        let step_norm = step.norm();
        if step_norm > 0.25 {
            step *= 0.25 / step_norm;
        }

        let trial = p + step;
        let trial_seed = decode_params(&trial, f_scale, max_abs_tilt)?;
        let trial_res = conic_residual(&trial_seed, target_omega)?;
        if trial_res.norm_squared() < best_res.norm_squared() {
            p = trial;
            best_res = trial_res;
            lambda = (lambda * 0.3).max(1e-12);
            if step.norm() < 1e-8 {
                break;
            }
        } else {
            lambda = (lambda * 10.0).min(1e6);
        }
    }

    decode_params(&p, f_scale, max_abs_tilt)
}

fn decode_params(p: &SVector<Real, 6>, f_scale: Real, max_abs_tilt: Real) -> Option<CandidateSeed> {
    let fx = p[0].exp();
    let fy = p[1].exp();
    let cx = p[2] * f_scale;
    let cy = p[3] * f_scale;
    let sensor = ScheimpflugParams {
        tilt_x: p[4].clamp(-max_abs_tilt, max_abs_tilt),
        tilt_y: p[5].clamp(-max_abs_tilt, max_abs_tilt),
    };
    if [fx, fy, cx, cy, sensor.tilt_x, sensor.tilt_y]
        .iter()
        .any(|v| !v.is_finite())
        || fx <= 0.0
        || fy <= 0.0
    {
        return None;
    }
    Some(CandidateSeed {
        reference_k: FxFyCxCySkew {
            fx,
            fy,
            cx,
            cy,
            skew: 0.0,
        },
        sensor,
    })
}

fn conic_residual(seed: &CandidateSeed, target_omega: &Mat3) -> Option<SVector<Real, 6>> {
    let predicted = conic_from_k_and_tilt(&seed.reference_k, seed.sensor)?;
    let residual = predicted - *target_omega;
    Some(SVector::<Real, 6>::new(
        residual[(0, 0)],
        2.0_f64.sqrt() * residual[(0, 1)],
        residual[(1, 1)],
        2.0_f64.sqrt() * residual[(0, 2)],
        2.0_f64.sqrt() * residual[(1, 2)],
        residual[(2, 2)],
    ))
}

fn conic_from_k_and_tilt(
    intrinsics: &FxFyCxCySkew<Real>,
    sensor: ScheimpflugParams,
) -> Option<Mat3> {
    let effective = intrinsics.k_matrix() * sensor.compile().h;
    let inv = effective.try_inverse()?;
    normalize_symmetric_matrix(inv.transpose() * inv)
}

fn normalize_symmetric_matrix(m: Mat3) -> Option<Mat3> {
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let norm = Vector3::new(m[(0, 0)], m[(1, 1)], m[(2, 2)]).norm()
        + 2.0 * Vector3::new(m[(0, 1)], m[(0, 2)], m[(1, 2)]).norm();
    if norm <= Real::EPSILON || !norm.is_finite() {
        return None;
    }
    Some(m / norm)
}

fn evaluate_candidate(
    dataset: &PlanarDataset,
    reference_k: &FxFyCxCySkew<Real>,
    sensor: ScheimpflugParams,
    pinhole_opts: IterativeIntrinsicsOptions,
    observed_center: Pt2,
) -> Result<Candidate, Error> {
    let untilted = untilt_dataset(dataset, reference_k, sensor)?;
    let mut camera = estimate_intrinsics_iterative(&untilted, pinhole_opts)?;
    if pinhole_opts.zero_skew {
        camera.k.skew = 0.0;
    }
    camera.dist = damp_seed_distortion(camera.dist);
    validate_camera(&camera)?;
    let poses = recover_tilt_aware_poses(dataset, &camera.k, &camera.dist, sensor)?;
    let mean_reproj_error = mean_reproj_error(dataset, &camera, sensor, &poses);
    if !mean_reproj_error.is_finite() {
        return Err(Error::numerical(
            "non-finite Scheimpflug initialization reprojection error",
        ));
    }
    let score = mean_reproj_error
        + distortion_complexity(&camera.dist)
        + tilt_complexity(sensor)
        + principal_point_complexity(&camera.k, observed_center);
    Ok(Candidate {
        init: ScheimpflugIntrinsicsLinearInit {
            camera,
            sensor,
            poses,
            mean_reproj_error,
        },
        score,
    })
}

fn distortion_complexity(distortion: &BrownConrady5<Real>) -> Real {
    const DISTORTION_PRIOR_WEIGHT: Real = 0.25;
    let radial = distortion.k1 * distortion.k1
        + distortion.k2 * distortion.k2
        + 0.25 * distortion.k3 * distortion.k3;
    let tangential = distortion.p1 * distortion.p1 + distortion.p2 * distortion.p2;
    DISTORTION_PRIOR_WEIGHT * (radial + tangential)
}

fn tilt_complexity(sensor: ScheimpflugParams) -> Real {
    const TILT_PRIOR_WEIGHT: Real = 0.25;
    TILT_PRIOR_WEIGHT * (sensor.tilt_x * sensor.tilt_x + sensor.tilt_y * sensor.tilt_y)
}

fn principal_point_complexity(intrinsics: &FxFyCxCySkew<Real>, center: Pt2) -> Real {
    const PRINCIPAL_POINT_PRIOR_WEIGHT: Real = 0.0;
    let dx = (intrinsics.cx - center.x) / intrinsics.fx.max(1.0);
    let dy = (intrinsics.cy - center.y) / intrinsics.fy.max(1.0);
    PRINCIPAL_POINT_PRIOR_WEIGHT * (dx * dx + dy * dy)
}

fn observed_pixel_center(dataset: &PlanarDataset) -> Pt2 {
    let mut min_x = Real::INFINITY;
    let mut min_y = Real::INFINITY;
    let mut max_x = Real::NEG_INFINITY;
    let mut max_y = Real::NEG_INFINITY;
    for view in &dataset.views {
        for p in &view.obs.points_2d {
            if p.x.is_finite() && p.y.is_finite() {
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
    }
    if min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite() {
        if min_x >= 0.0 && min_y >= 0.0 {
            Pt2::new(0.5 * max_x, 0.5 * max_y)
        } else {
            Pt2::new(0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
        }
    } else {
        Pt2::new(0.0, 0.0)
    }
}

fn damp_seed_distortion(mut distortion: BrownConrady5<Real>) -> BrownConrady5<Real> {
    distortion.k1 = 0.0;
    distortion.k2 = 0.0;
    distortion.k3 = 0.0;
    distortion.p1 = distortion.p1.clamp(-0.05, 0.05);
    distortion.p2 = distortion.p2.clamp(-0.05, 0.05);
    distortion
}

fn untilt_dataset(
    dataset: &PlanarDataset,
    reference_k: &FxFyCxCySkew<Real>,
    sensor: ScheimpflugParams,
) -> Result<PlanarDataset, Error> {
    let sensor_model = sensor.compile();
    let views = dataset
        .views
        .iter()
        .map(|view| {
            let points_2d: Vec<Pt2> = view
                .obs
                .points_2d
                .iter()
                .map(|p| {
                    let sensor_pt = reference_k.pixel_to_sensor(p);
                    let normalized = sensor_model.sensor_to_normalized(&sensor_pt);
                    reference_k.sensor_to_pixel(&normalized)
                })
                .collect();
            make_view_like(&view.obs, points_2d)
        })
        .collect::<Result<Vec<_>, _>>()?;
    PlanarDataset::new(views).map_err(Error::from)
}

fn recover_tilt_aware_poses(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCySkew<Real>,
    distortion: &BrownConrady5<Real>,
    sensor: ScheimpflugParams,
) -> Result<Vec<Iso3>, Error> {
    let ideal = ideal_dataset(dataset, intrinsics, distortion, sensor)?;
    let k_matrix = intrinsics.k_matrix();
    let mut poses = Vec::with_capacity(ideal.num_views());
    for (idx, view) in ideal.views.iter().enumerate() {
        let board = view.obs.planar_points();
        let h = dlt_homography(&board, &view.obs.points_2d).map_err(|e| {
            Error::numerical(format!("tilt-aware homography failed for view {idx}: {e}"))
        })?;
        let pose = estimate_planar_pose_from_h(&k_matrix, &h).map_err(|e| {
            Error::numerical(format!(
                "tilt-aware pose recovery failed for view {idx}: {e}"
            ))
        })?;
        poses.push(pose);
    }
    Ok(poses)
}

fn ideal_dataset(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCySkew<Real>,
    distortion: &BrownConrady5<Real>,
    sensor: ScheimpflugParams,
) -> Result<PlanarDataset, Error> {
    let sensor_model = sensor.compile();
    let views = dataset
        .views
        .iter()
        .map(|view| {
            let points_2d: Vec<Pt2> = view
                .obs
                .points_2d
                .iter()
                .map(|p| {
                    let sensor_pt = intrinsics.pixel_to_sensor(p);
                    let distorted = sensor_model.sensor_to_normalized(&sensor_pt);
                    let undistorted = distortion.undistort(&distorted);
                    intrinsics.sensor_to_pixel(&undistorted)
                })
                .collect();
            make_view_like(&view.obs, points_2d)
        })
        .collect::<Result<Vec<_>, _>>()?;
    PlanarDataset::new(views).map_err(Error::from)
}

fn make_view_like(obs: &CorrespondenceView, points_2d: Vec<Pt2>) -> Result<View<NoMeta>, Error> {
    let converted = if obs.weights.is_empty() {
        CorrespondenceView::new(obs.points_3d.clone(), points_2d)?
    } else {
        CorrespondenceView::new_with_weights(obs.points_3d.clone(), points_2d, obs.weights.clone())?
    };
    Ok(View::without_meta(converted))
}

fn validate_camera(camera: &PinholeCamera) -> Result<(), Error> {
    let values = [
        camera.k.fx,
        camera.k.fy,
        camera.k.cx,
        camera.k.cy,
        camera.k.skew,
        camera.dist.k1,
        camera.dist.k2,
        camera.dist.k3,
        camera.dist.p1,
        camera.dist.p2,
    ];
    if values.iter().any(|v| !v.is_finite()) || camera.k.fx <= 0.0 || camera.k.fy <= 0.0 {
        return Err(Error::numerical(
            "Scheimpflug initialization produced invalid camera parameters",
        ));
    }
    Ok(())
}

fn mean_reproj_error(
    dataset: &PlanarDataset,
    camera: &PinholeCamera,
    sensor: ScheimpflugParams,
    poses: &[Iso3],
) -> Real {
    let camera = Camera::new(Pinhole, camera.dist, sensor.compile(), camera.k);
    let mut sum = 0.0;
    let mut count = 0usize;
    for (view, pose) in dataset.views.iter().zip(poses.iter()) {
        for (p3, p2) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }
    if count == 0 {
        Real::INFINITY
    } else {
        sum / count as Real
    }
}

fn is_better(candidate: &Candidate, current: Option<&Candidate>) -> bool {
    current.is_none_or(|best| candidate.score < best.score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};
    use vision_calibration_core::{Pt3, synthetic::planar};

    fn make_pose(rx: f64, ry: f64, rz: f64, tx: f64, ty: f64, tz: f64) -> Iso3 {
        Iso3::from_parts(
            Translation3::new(tx, ty, tz),
            UnitQuaternion::from_euler_angles(rx, ry, rz),
        )
    }

    fn varied_poses() -> Vec<Iso3> {
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

    fn grid_points(cols: usize, rows: usize, spacing: Real) -> Vec<Pt3> {
        let mut points = Vec::with_capacity(cols * rows);
        for iy in 0..rows {
            for ix in 0..cols {
                points.push(Pt3::new(
                    (ix as Real - (cols as Real - 1.0) * 0.5) * spacing,
                    (iy as Real - (rows as Real - 1.0) * 0.5) * spacing,
                    0.0,
                ));
            }
        }
        points
    }

    fn dataset(
        intrinsics: FxFyCxCySkew<Real>,
        distortion: BrownConrady5<Real>,
        sensor: ScheimpflugParams,
        cols: usize,
        rows: usize,
    ) -> PlanarDataset {
        let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
        let board = grid_points(cols, rows, 0.02);
        let views = planar::project_views_all(&camera, &board, &varied_poses())
            .expect("synthetic views")
            .into_iter()
            .map(View::without_meta)
            .collect();
        PlanarDataset::new(views).expect("dataset")
    }

    #[test]
    fn init_recovers_small_tilt_without_distortion() {
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let sensor = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let dataset = dataset(intrinsics, BrownConrady5::default(), sensor, 7, 6);
        let init = estimate_scheimpflug_intrinsics_iterative(
            &dataset,
            ScheimpflugIntrinsicsInitOptions::default(),
        )
        .expect("tilt-aware init");
        assert!(
            (init.sensor.tilt_x - sensor.tilt_x).abs() < 0.02,
            "tilt_x={} want {}",
            init.sensor.tilt_x,
            sensor.tilt_x
        );
        assert!(
            (init.sensor.tilt_y - sensor.tilt_y).abs() < 0.02,
            "tilt_y={} want {}",
            init.sensor.tilt_y,
            sensor.tilt_y
        );
        assert!(init.mean_reproj_error < 0.5, "{}", init.mean_reproj_error);
        assert_eq!(init.poses.len(), dataset.num_views());
    }

    #[test]
    fn init_recovers_rtv3d_like_tilt_with_strong_radial_distortion() {
        let intrinsics = FxFyCxCySkew {
            fx: 1155.0,
            fy: 1165.0,
            cx: 365.0,
            cy: 265.0,
            skew: 0.0,
        };
        let distortion = BrownConrady5 {
            k1: -0.43,
            k2: 0.25,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let sensor = ScheimpflugParams {
            tilt_x: -0.087,
            tilt_y: 0.004,
        };
        let dataset = dataset(intrinsics, distortion, sensor, 13, 13);
        let init = estimate_scheimpflug_intrinsics_iterative(
            &dataset,
            ScheimpflugIntrinsicsInitOptions::default(),
        )
        .expect("tilt-aware init");

        assert!(
            (init.sensor.tilt_x - sensor.tilt_x).abs() < 0.04,
            "tilt_x={} want {}",
            init.sensor.tilt_x,
            sensor.tilt_x
        );
        assert!(init.mean_reproj_error < 3.0, "{}", init.mean_reproj_error);
        assert!(
            (init.camera.k.fx - intrinsics.fx).abs() / intrinsics.fx < 0.2,
            "fx={} want {}",
            init.camera.k.fx,
            intrinsics.fx
        );
    }

    #[test]
    fn init_rejects_too_few_views() {
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dataset = dataset(
            intrinsics,
            BrownConrady5::default(),
            ScheimpflugParams::default(),
            6,
            5,
        );
        let short = PlanarDataset::new(dataset.views.into_iter().take(2).collect())
            .expect("two valid views");
        let err = estimate_scheimpflug_intrinsics_iterative(
            &short,
            ScheimpflugIntrinsicsInitOptions::default(),
        )
        .expect_err("too few views should fail");
        assert!(matches!(err, Error::InsufficientData { need: 3, got: 2 }));
    }

    #[test]
    fn init_rejects_empty_seed_grid() {
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dataset = dataset(
            intrinsics,
            BrownConrady5::default(),
            ScheimpflugParams::default(),
            6,
            5,
        );
        let opts = ScheimpflugIntrinsicsInitOptions {
            tilt_x_seeds: Vec::new(),
            ..Default::default()
        };
        let err = estimate_scheimpflug_intrinsics_iterative(&dataset, opts)
            .expect_err("empty grid should fail");
        assert!(err.to_string().contains("at least one tilt_x"));
    }
}
