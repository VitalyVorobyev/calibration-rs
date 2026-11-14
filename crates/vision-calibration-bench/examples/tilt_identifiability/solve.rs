//! Solves through the library: the direct planar solve, the pipeline's
//! seeded and cold routes, and the four parameterizations as a user
//! residual on the library's Levenberg–Marquardt loop.

use anyhow::{Result, anyhow};
use nalgebra::{DVector, RealField, Vector3};
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init, step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, DistortionKind, DistortionParams, FxFyCxCy, Iso3,
    PlanarDataset, ProjectionParams, ScheimpflugParams, SensorKind, SensorParams, Uncertainty,
};
use vision_calibration_optim::problem::{Manifold, Noise, Problem, Residual, SolveOptions, se3};
use vision_calibration_optim::{
    BackendSolveOptions, CameraBounds, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
    RobustLoss, SolverBackend, optimize_planar_intrinsics,
};

use crate::model::{Cam, FreeSet, NCAM, Param, RAW_NAMES, TX, TY};
use crate::scene::camera_params;

/// A solve's result in raw coordinates.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub cam: Cam,
    pub iters: usize,
    /// `½Σ‖r‖²`.
    pub cost: f64,
    /// Stated σ per raw coordinate name (`fx`, `tilt_x`, …) of the free
    /// coordinates, in the solve's own parameterization for the intrinsics.
    pub sigma: Vec<(String, f64)>,
    /// Stated correlations of the solve's coordinates.
    pub cov: Option<LabelledCovariance>,
    /// Why the covariance is unavailable, when it is.
    pub unavailable: Option<String>,
    /// The view-scatter factor the covariance was scaled by, when it was.
    pub view_scatter: Option<f64>,
    /// The fitted `cam_se3_target` of each view, in the camera's own frame.
    pub poses: Vec<Iso3>,
}

impl Outcome {
    pub fn sigma_of(&self, name: &str) -> Option<f64> {
        self.sigma.iter().find(|(n, _)| n == name).map(|&(_, s)| s)
    }
}

/// The library's last label segment (`intrinsics.fx` → `fx`).
fn short_label(label: &str) -> String {
    label.rsplit('.').next().unwrap_or(label).to_string()
}

/// Labels and the covariance matrix of a solve's coordinates.
type LabelledCovariance = (Vec<String>, Vec<Vec<f64>>);

/// What a solve's uncertainty says: stated σ per label, the covariance,
/// why it is unavailable, and the view-scatter factor.
type ReadUncertainty = (
    Vec<(String, f64)>,
    Option<LabelledCovariance>,
    Option<String>,
    Option<f64>,
);

/// Stated σ per label, the covariance, why it is unavailable, and the
/// view-scatter factor it was scaled by.
fn read_uncertainty(u: &Uncertainty) -> ReadUncertainty {
    match u.covariance() {
        Some(c) => {
            let labels: Vec<String> = c.labels.iter().map(|l| short_label(l)).collect();
            let sigma = labels
                .iter()
                .enumerate()
                .map(|(i, l)| (l.clone(), c.covariance[i][i].max(0.0).sqrt()))
                .collect();
            (
                sigma,
                Some((labels, c.covariance.clone())),
                None,
                c.view_scatter,
            )
        }
        None => {
            let reason = match u {
                Uncertainty::Unavailable { reason } => format!("{reason:?}"),
                Uncertainty::Available(_) => unreachable!("covariance() is Some for Available"),
            };
            (Vec::new(), None, Some(reason), None)
        }
    }
}

/// The distortion kind a free set needs: Brown-Conrady unless a thin-prism
/// term is free.
fn kind_for(free: FreeSet) -> DistortionKind {
    match free {
        FreeSet::Prism | FreeSet::TangentialPrism | FreeSet::PrismNoTilt => {
            DistortionKind::ThinPrism9
        }
        _ => DistortionKind::BrownConrady5,
    }
}

/// The library camera of `cam` in distortion `kind`.
pub fn library_camera(cam: &Cam, kind: DistortionKind) -> CameraParams {
    match kind {
        DistortionKind::ThinPrism9 => camera_params(cam),
        _ => {
            let d = cam.d;
            CameraParams::new(
                ProjectionParams::Pinhole,
                DistortionParams::BrownConrady5 {
                    params: BrownConrady5 {
                        k1: d[0],
                        k2: d[1],
                        k3: d[2],
                        p1: d[3],
                        p2: d[4],
                        iters: 20,
                    },
                },
                SensorParams::Scheimpflug {
                    params: ScheimpflugParams {
                        tilt_x: cam.tilt[0],
                        tilt_y: cam.tilt[1],
                    },
                },
                FxFyCxCy {
                    fx: cam.k[0],
                    fy: cam.k[1],
                    cx: cam.k[2],
                    cy: cam.k[3],
                },
            )
        }
    }
}

/// A library camera back in raw coordinates.
pub fn cam_of(params: &CameraParams) -> Cam {
    let i = params.intrinsics;
    let mut d = [0.0; 9];
    let values = params.distortion.values();
    d[..values.len().min(9)].copy_from_slice(&values[..values.len().min(9)]);
    let tilt = match params.sensor {
        SensorParams::Scheimpflug { params } => [params.tilt_x, params.tilt_y],
        _ => [0.0, 0.0],
    };
    Cam {
        k: [i.fx, i.fy, i.cx, i.cy],
        d,
        tilt,
    }
}

/// The fix lists of a free set, for distortion `kind`.
fn camera_fix(free: FreeSet, kind: DistortionKind) -> CameraFix {
    let idx = free.indices();
    let dist: Vec<String> = kind
        .param_names()
        .iter()
        .enumerate()
        .filter(|(i, _)| !idx.contains(&(4 + i)))
        .map(|(_, n)| n.to_string())
        .collect();
    let sensor: Vec<String> = [(TX, "tilt_x"), (TY, "tilt_y")]
        .iter()
        .filter(|(i, _)| !idx.contains(i))
        .map(|(_, n)| n.to_string())
        .collect();
    CameraFix {
        intrinsics: Some(Vec::new()),
        projection: Some(Vec::new()),
        distortion: Some(dist),
        sensor: Some(sensor),
    }
}

/// Backend options; `tight` runs to a relative decrease of 1e-14.
pub fn backend_opts(backend: SolverBackend, max_iters: usize, tight: bool) -> BackendSolveOptions {
    let mut o = BackendSolveOptions {
        backend,
        max_iters,
        ..BackendSolveOptions::default()
    };
    if tight {
        o.min_abs_decrease = Some(0.0);
        o.min_rel_decrease = Some(1e-14);
        o.min_error = Some(0.0);
    }
    o
}

/// `optimize_planar_intrinsics` from `cam`, `poses`.
pub fn solve_direct(
    dataset: &PlanarDataset,
    cam: &Cam,
    poses: &[Iso3],
    free: FreeSet,
    bounds: Option<CameraBounds>,
    opts: BackendSolveOptions,
) -> Result<Outcome> {
    let kind = kind_for(free);
    let initial = PlanarIntrinsicsParams::new(library_camera(cam, kind), poses.to_vec())?;
    let solve_opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::None,
        fix: camera_fix(free, kind),
        fix_poses: Vec::new(),
        bounds,
    };
    let est = optimize_planar_intrinsics(dataset, &initial, solve_opts, opts)?;
    let (sigma, cov, unavailable, view_scatter) = read_uncertainty(&est.uncertainty);
    Ok(Outcome {
        cam: cam_of(&est.params.camera),
        iters: est.report.num_iters,
        cost: est.report.final_cost,
        sigma,
        cov,
        unavailable,
        view_scatter,
        poses: est.params.cam_se3_target.clone(),
    })
}

/// The pipeline config of a tilted Brown-Conrady camera on `backend`.
fn pipeline_config(backend: SolverBackend) -> PlanarIntrinsicsConfig {
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model.sensor = SensorKind::Scheimpflug;
    config.solver.backend = backend;
    config
}

fn pipeline_outcome(session: &CalibrationSession<PlanarIntrinsicsProblem>) -> Result<Outcome> {
    let est = session
        .output()
        .ok_or_else(|| anyhow!("no output after step_optimize"))?;
    let (sigma, cov, unavailable, view_scatter) = read_uncertainty(&est.uncertainty);
    Ok(Outcome {
        cam: cam_of(&est.params.camera),
        iters: est.report.num_iters,
        cost: est.report.final_cost,
        sigma,
        cov,
        unavailable,
        view_scatter,
        poses: est.params.cam_se3_target.clone(),
    })
}

/// The seeded route (`step_init_with_seed` with intrinsics and tilt, then
/// `step_optimize`); also returns the initial camera and poses.
pub fn route_seeded(
    dataset: &PlanarDataset,
    seed_k: [f64; 4],
    seed_tilt: [f64; 2],
    backend: SolverBackend,
) -> Result<(Outcome, Cam, Vec<Iso3>)> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset.clone())?;
    session.set_config(pipeline_config(backend))?;
    let mut seed = IntrinsicsManualInit::default();
    seed.intrinsics = Some(FxFyCxCy {
        fx: seed_k[0],
        fy: seed_k[1],
        cx: seed_k[2],
        cy: seed_k[3],
    });
    seed.sensor = Some(ScheimpflugParams {
        tilt_x: seed_tilt[0],
        tilt_y: seed_tilt[1],
    });
    let init = step_init_with_seed(&mut session, seed, None)?;
    let init_cam = cam_of(&init.camera);
    step_optimize(&mut session, None)?;
    Ok((pipeline_outcome(&session)?, init_cam, init.cam_se3_target))
}

/// Step 1 of the determination procedure: the plain camera (untilted
/// sensor, Brown-Conrady with the default fix list, so `p1, p2` free) through
/// the pipeline's cold route (`step_init`, `step_optimize`).
pub fn route_plain(dataset: &PlanarDataset, backend: SolverBackend) -> Result<Outcome> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset.clone())?;
    let mut config = PlanarIntrinsicsConfig::default();
    config.solver.backend = backend;
    session.set_config(config)?;
    step_init(&mut session, None)?;
    step_optimize(&mut session, None)?;
    pipeline_outcome(&session)
}

/// The cold route: `step_init` (experimental for a tilted sensor) then
/// `step_optimize`.
pub fn route_cold(dataset: &PlanarDataset, backend: SolverBackend) -> Result<Outcome> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset.clone())?;
    session.set_config(pipeline_config(backend))?;
    step_init(&mut session, None)?;
    step_optimize(&mut session, None)?;
    pipeline_outcome(&session)
}

/// One corner's residual in parameterization `param`; blocks: the four
/// intrinsic coordinates, the free distortion terms, the tilt, the pose.
struct Corner {
    param: Param,
    /// Coordinates held at these values (15, raw for distortion and tilt).
    template: [f64; NCAM],
    /// Indices (4..13) of the free distortion terms, in block order.
    dist: Vec<usize>,
    tilt_free: bool,
    pw: [f64; 3],
    uv: [f64; 2],
}

impl Residual for Corner {
    fn dim(&self) -> usize {
        2
    }

    fn block_dims(&self) -> Vec<usize> {
        vec![4, self.dist.len(), 2, 7]
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let mut v: Vec<T> = self
            .template
            .iter()
            .map(|&x| T::from_f64(x).unwrap())
            .collect();
        for i in 0..4 {
            v[i] = blocks[0][i].clone();
        }
        for (j, &i) in self.dist.iter().enumerate() {
            v[i] = blocks[1][j].clone();
        }
        if self.tilt_free {
            v[TX] = blocks[2][0].clone();
            v[TY] = blocks[2][1].clone();
        }
        let pose = se3(&blocks[3]);
        let pw = Vector3::new(
            T::from_f64(self.pw[0]).unwrap(),
            T::from_f64(self.pw[1]).unwrap(),
            T::from_f64(self.pw[2]).unwrap(),
        );
        match self.param.project(&v, &pose, &pw) {
            Some([u, w]) => DVector::from_vec(vec![
                u - T::from_f64(self.uv[0]).unwrap(),
                w - T::from_f64(self.uv[1]).unwrap(),
            ]),
            None => DVector::from_element(2, T::from_f64(f64::NAN).unwrap()),
        }
    }
}

/// The joint solve in `param` from `cam`, `poses` (lens frame), through the
/// library's solver (tiny-solver; a user residual runs there only).
pub fn solve_param(
    dataset: &PlanarDataset,
    cam: &Cam,
    poses: &[Iso3],
    param: Param,
    free: FreeSet,
    max_iters: usize,
) -> Result<Outcome> {
    solve_param_with(dataset, cam, poses, param, free, max_iters, None)
}

/// A Gaussian prior on the tilt, `tilt ~ N(mean, sigma²)` per axis, weighed
/// against the corners at a stated pixel noise.
#[derive(Clone, Copy, Debug)]
pub struct TiltPrior {
    /// Prior mean `[tilt_x, tilt_y]`, radians.
    pub mean: [f64; 2],
    /// Prior σ per axis, radians.
    pub sigma: f64,
    /// The corners' noise, pixels: the prior is information in units of its
    /// own σ, so the corners are stated at this σ too (the noise of the
    /// untilted fit, as the library weighs robot-pose priors).
    pub pixel_sigma: f64,
}

/// The prior's residual `tilt − mean`, stated at the prior σ.
struct TiltPriorResidual {
    mean: [f64; 2],
}

impl Residual for TiltPriorResidual {
    fn dim(&self) -> usize {
        2
    }

    fn block_dims(&self) -> Vec<usize> {
        vec![2]
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        DVector::from_vec(vec![
            blocks[0][0].clone() - T::from_f64(self.mean[0]).unwrap(),
            blocks[0][1].clone() - T::from_f64(self.mean[1]).unwrap(),
        ])
    }
}

/// [`solve_param`], optionally with a tilt prior (then the tilt must be in
/// the free set). With a prior the solve is restarted from its own result
/// until the cost stops falling (relative decrease under 1e-12, at most 30
/// restarts): a weak prior leaves a long valley in which the default
/// stopping rule halts near the start, and the builder's solve has no
/// tolerance option.
pub fn solve_param_with(
    dataset: &PlanarDataset,
    cam: &Cam,
    poses: &[Iso3],
    param: Param,
    free: FreeSet,
    max_iters: usize,
    prior: Option<TiltPrior>,
) -> Result<Outcome> {
    let mut out = solve_param_once(dataset, cam, poses, param, free, max_iters, prior)?;
    if prior.is_some() {
        for _ in 0..30 {
            let next =
                solve_param_once(dataset, &out.cam, &out.poses, param, free, max_iters, prior)?;
            let rel = (out.cost - next.cost) / out.cost.max(1e-300);
            let iters = out.iters + next.iters;
            out = next;
            out.iters = iters;
            if rel < 1e-12 {
                break;
            }
        }
    }
    Ok(out)
}

fn solve_param_once(
    dataset: &PlanarDataset,
    cam: &Cam,
    poses: &[Iso3],
    param: Param,
    free: FreeSet,
    max_iters: usize,
    prior: Option<TiltPrior>,
) -> Result<Outcome> {
    let idx = free.indices();
    let coords = param.coords(cam);
    let dist: Vec<usize> = (4..13).filter(|i| idx.contains(i)).collect();
    let tilt_free = idx.contains(&TX);
    let mut problem = Problem::new();
    let k = problem.add_block("k", &coords[..4], Manifold::Euclidean)?;
    problem.set_labels(
        k,
        &param.intrinsic_names().map(|n| n.trim_end_matches('\'')),
    )?;
    let dvals: Vec<f64> = dist.iter().map(|&i| coords[i]).collect();
    let d = problem.add_block("d", &dvals, Manifold::Euclidean)?;
    let dnames: Vec<&str> = dist.iter().map(|&i| RAW_NAMES[i]).collect();
    problem.set_labels(d, &dnames)?;
    let t = problem.add_block("t", &coords[TX..=TY], Manifold::Euclidean)?;
    problem.set_labels(t, &["tilt_x", "tilt_y"])?;
    if !tilt_free {
        problem.fix(t)?;
    }
    let mut pose_ids = Vec::new();
    for (i, pose) in poses.iter().enumerate() {
        pose_ids.push(problem.add_pose(format!("pose{i}"), &param.pose_to(cam, pose))?);
    }
    for (vi, view) in dataset.views.iter().enumerate() {
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            problem.add_residual(
                Corner {
                    param,
                    template: coords,
                    dist: dist.clone(),
                    tilt_free,
                    pw: [p3.x, p3.y, p3.z],
                    uv: [p2.x, p2.y],
                },
                &[k, d, t, pose_ids[vi]],
                RobustLoss::None,
                match prior {
                    Some(p) => Noise::Stated(p.pixel_sigma),
                    None => Noise::Estimated("px".into()),
                },
            )?;
        }
    }
    if let Some(p) = prior {
        anyhow::ensure!(tilt_free, "a tilt prior needs the tilt free");
        problem.add_residual(
            TiltPriorResidual { mean: p.mean },
            &[t],
            RobustLoss::None,
            Noise::Stated(p.sigma),
        )?;
    }
    let mut opts = SolveOptions::default();
    opts.max_iters = max_iters;
    let sol = vision_calibration_optim::problem::solve(&problem, &opts)?;
    let mut v = coords;
    v[..4].copy_from_slice(sol.value(k).expect("own block"));
    for (j, &i) in dist.iter().enumerate() {
        v[i] = sol.value(d).expect("own block")[j];
    }
    v[TX..=TY].copy_from_slice(sol.value(t).expect("own block"));
    let out_cam = param.camera(&v);
    let (sigma, cov, unavailable, view_scatter) = read_uncertainty(&sol.uncertainty);
    // Back to lens-frame poses: the gauge parameterization's are in the
    // sensor-normal frame.
    let r = crate::model::tilt_rotation(out_cam.tilt[0], out_cam.tilt[1]);
    let q = nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(r),
    );
    let out_poses = pose_ids
        .iter()
        .map(|&id| {
            let p = sol.pose(id).expect("own pose block");
            match param {
                Param::Gauge => {
                    Iso3::from_parts(nalgebra::Translation3::identity(), q.inverse()) * p
                }
                _ => p,
            }
        })
        .collect();
    Ok(Outcome {
        cam: out_cam,
        iters: sol.report.num_iters,
        cost: sol.report.final_cost,
        sigma,
        cov,
        unavailable,
        view_scatter,
        poses: out_poses,
    })
}
