//! Is the tilt determined, and what does each answer cost?
//!
//! Four fits of the same data, each a model a tilt procedure can end in:
//!
//! - **M0**: tilt held at zero, `p1, p2` held (the tilted camera's default
//!   fix list with the tilt fixed);
//! - **M0p**: tilt held at zero, `p1, p2` free (the plain Brown-Conrady
//!   camera: decentring is the tilt's first-order equivalent);
//! - **M1**: tilt free, `p1, p2` held (today's tilted fit);
//! - **M2**: tilt and `p1, p2` free.
//!
//! The decision statistics are read from M1's (and M2's) covariance as the
//! library states it, view-scatter scaling included; the labels they are
//! judged against come from the Fisher information at the truth (or, with
//! per-view errors, from the Monte-Carlo scatter). What a fit costs is
//! measured downstream ([`crate::downstream`]) on noise-free held-out data:
//! the bias of a fit (its noise-free value) against the noise floor (the
//! correct model M1's error under noise).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use nalgebra::{Matrix2, Vector2};
use vision_calibration_core::{Iso3, PlanarDataset};
use vision_calibration_optim::SolverBackend;

use crate::downstream::{Bench, Metrics, evaluate};
use crate::model::{Cam, FreeSet, Param, TX, TY, fisher};
use crate::scene::{CamSpec, Coverage, ViewDesign};
use crate::solve::{Outcome, backend_opts, solve_direct};

/// Map `items` over the machine's threads, results in input order.
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(items.len().max(1));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let r = f(&items[i]);
                    *out[i].lock().expect("no poisoned slot") = Some(r);
                }
            });
        }
    });
    out.into_iter()
        .map(|m| {
            m.into_inner()
                .expect("no poisoned slot")
                .expect("every item mapped")
        })
        .collect()
}

/// The four models, in table order.
pub const MODELS: [&str; 4] = [
    "M0 (τ=0, p held)",
    "M0p (τ=0, p free)",
    "M1 (τ free, p held)",
    "M2 (τ, p free)",
];
pub const M0: usize = 0;
pub const M0P: usize = 1;
pub const M1: usize = 2;
pub const M2: usize = 3;

fn free_set(model: usize) -> FreeSet {
    match model {
        M0 => FreeSet::NoTilt,
        M0P => FreeSet::TangentialNoTilt,
        M1 => FreeSet::Default,
        _ => FreeSet::Tangential,
    }
}

/// The start of a model's solve: the truth for the tilted models (with the
/// tangential terms the model holds at zero), and for the untilted ones the
/// sensor-normal pinhole `K'` with poses in its frame, which is where an
/// untilted fit of a tilted camera goes without distortion.
pub fn model_start(model: usize, cam: &Cam, poses: &[Iso3]) -> (Cam, Vec<Iso3>) {
    let mut start = *cam;
    if matches!(model, M0 | M1) {
        start.d[3] = 0.0;
        start.d[4] = 0.0;
    }
    if matches!(model, M0 | M0P) {
        start.k = Param::Effective.intrinsics_of_raw(&cam.k, &cam.tilt);
        start.tilt = [0.0, 0.0];
        let poses = poses.iter().map(|p| Param::Gauge.pose_to(cam, p)).collect();
        (start, poses)
    } else {
        (start, poses.to_vec())
    }
}

/// Fit one model from its start, run to a tight stop so the valley's early
/// stop does not blur the comparison.
pub fn fit_model(model: usize, data: &PlanarDataset, cam: &Cam, poses: &[Iso3]) -> Result<Outcome> {
    let (start, start_poses) = model_start(model, cam, poses);
    solve_direct(
        data,
        &start,
        &start_poses,
        free_set(model),
        None,
        backend_opts(SolverBackend::TinySolver, 500, true),
    )
}

/// The tilt's stated uncertainty in a fit.
#[derive(Clone, Copy, Debug)]
pub struct TiltStat {
    /// The larger semi-axis of the tilt's 2×2 covariance, radians (σ along
    /// the worst-determined tilt direction).
    pub sigma: f64,
    /// The same without the view-scatter scaling.
    pub sigma_unscaled: f64,
    /// `τ̂ᵀ Σ⁻¹ τ̂`: the Wald statistic of "tilt = 0" (χ² with 2 dof).
    pub wald: f64,
}

/// The tilt's 2×2 covariance block of a fit, if it has one.
pub fn tilt_cov(o: &Outcome) -> Option<Matrix2<f64>> {
    let (labels, cov) = o.cov.as_ref()?;
    let ix = labels.iter().position(|l| l == "tilt_x")?;
    let iy = labels.iter().position(|l| l == "tilt_y")?;
    Some(Matrix2::new(
        cov[ix][ix],
        cov[ix][iy],
        cov[iy][ix],
        cov[iy][iy],
    ))
}

pub fn tilt_stat(o: &Outcome) -> Option<TiltStat> {
    let c = tilt_cov(o)?;
    let sigma = c.symmetric_eigen().eigenvalues.max().max(0.0).sqrt();
    let k = o.view_scatter.unwrap_or(1.0);
    let t = Vector2::new(o.cam.tilt[0], o.cam.tilt[1]);
    let wald = c
        .try_inverse()
        .map_or(f64::NAN, |ci| (t.transpose() * ci * t)[(0, 0)]);
    Some(TiltStat {
        sigma,
        sigma_unscaled: sigma / k.sqrt(),
        wald,
    })
}

/// The Fisher σ (larger semi-axis of the tilt's covariance) at the truth,
/// radians, at pixel noise `noise`, for a free set.
pub fn fisher_sigma(bench: &Bench, free: FreeSet, noise: f64) -> f64 {
    let f = fisher(
        &bench.train,
        &bench.train.cam,
        &bench.train.poses,
        Param::Raw,
        &free.indices(),
    );
    match (f.cov_of(TX, TX), f.cov_of(TX, TY), f.cov_of(TY, TY)) {
        (Some(xx), Some(xy), Some(yy)) => {
            let c = Matrix2::new(xx, xy, xy, yy);
            noise * c.symmetric_eigen().eigenvalues.max().max(0.0).sqrt()
        }
        _ => f64::INFINITY,
    }
}

/// Scalar residuals minus estimated parameters of a fit.
pub fn dof(data: &PlanarDataset, model: usize) -> f64 {
    let m: usize = data.views.iter().map(|v| 2 * v.obs.points_2d.len()).sum();
    let n = free_set(model).indices().len() + 6 * data.views.len();
    (m - n) as f64
}

/// One noise draw of a cell: every model's fit and what it costs.
#[derive(Clone, Debug)]
pub struct Draw {
    pub stat_m1: Option<TiltStat>,
    pub stat_m2: Option<TiltStat>,
    /// Likelihood ratio of M0 against M1 (`2Δcost/σ̂²`, σ̂ from M1).
    pub lr_m0: f64,
    /// Of M0p against M2.
    pub lr_m0p: f64,
    /// M1's tilt error, radians per axis.
    pub tilt_err: [f64; 2],
    pub metrics: [Metrics; 4],
    /// `k1, k2` error per model.
    pub k_err: [[f64; 2]; 4],
}

/// M1 alone on one dataset: its tilt statistic and tilt error (radians).
pub fn run_stat_draw(
    bench: &Bench,
    noise: f64,
    view_error: f64,
    seed: u64,
) -> Result<(Option<TiltStat>, [f64; 2])> {
    let data = bench.train.dataset_with_view_error(noise, view_error, seed);
    let truth = bench.train.cam;
    let fit = fit_model(M1, &data, &truth, &bench.train.poses)?;
    Ok((
        tilt_stat(&fit),
        [
            fit.cam.tilt[0] - truth.tilt[0],
            fit.cam.tilt[1] - truth.tilt[1],
        ],
    ))
}

/// Fit every model to one dataset and evaluate it.
pub fn run_draw(bench: &Bench, noise: f64, view_error: f64, seed: u64) -> Result<Draw> {
    let data = bench.train.dataset_with_view_error(noise, view_error, seed);
    let data2 = bench.partner_data(noise, seed);
    let truth = bench.train.cam;
    let mut fits = Vec::new();
    for model in 0..4 {
        fits.push(fit_model(model, &data, &truth, &bench.train.poses)?);
    }
    let mut metrics = [Metrics::default(); 4];
    let mut k_err = [[0.0; 2]; 4];
    for (m, fit) in fits.iter().enumerate() {
        metrics[m] = evaluate(bench, fit, &data, &data2)?;
        k_err[m] = [fit.cam.d[0] - truth.d[0], fit.cam.d[1] - truth.d[1]];
    }
    let lr = |a: usize, b: usize| {
        let d = dof(&data, b);
        if fits[b].cost > 0.0 {
            (fits[a].cost - fits[b].cost) * d / fits[b].cost
        } else {
            f64::NAN
        }
    };
    Ok(Draw {
        stat_m1: tilt_stat(&fits[M1]),
        stat_m2: tilt_stat(&fits[M2]),
        lr_m0: lr(M0, M1),
        lr_m0p: lr(M0P, M2),
        tilt_err: [
            fits[M1].cam.tilt[0] - truth.tilt[0],
            fits[M1].cam.tilt[1] - truth.tilt[1],
        ],
        metrics,
        k_err,
    })
}

/// A view design of the sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Design {
    /// 20 views over the whole image.
    Rich,
    /// 8 views over the central half of the image.
    Poor,
}

impl Design {
    pub fn name(self) -> &'static str {
        match self {
            Design::Rich => "20 views, full",
            Design::Poor => "8 views, central",
        }
    }
    pub fn view_design(self) -> ViewDesign {
        match self {
            Design::Rich => ViewDesign::baseline(),
            Design::Poor => ViewDesign {
                views: 8,
                coverage: Coverage::Central,
                ..ViewDesign::baseline()
            },
        }
    }
}

/// A camera of the sweep: field of view, corner distortion, tilt.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub hfov: f64,
    pub dist: f64,
    pub tilt: (f64, f64),
    pub design: Design,
}

impl Cell {
    pub fn spec(&self) -> CamSpec {
        CamSpec {
            hfov_deg: self.hfov,
            corner_distortion: self.dist,
            tilt_deg: self.tilt,
            ..CamSpec::baseline()
        }
    }
}

/// A cell's results: the Fisher σ, the noise-free fits, the draws.
#[derive(Clone, Debug)]
pub struct CellResult {
    pub cell: Cell,
    /// Fisher σ of the tilt at 1 px of noise (M1 and M2), radians.
    pub fisher_m1: f64,
    pub fisher_m2: f64,
    /// Noise-free fits: what each model costs with no noise (its bias).
    pub bias: [Metrics; 4],
    pub bias_k: [[f64; 2]; 4],
    /// Noise-free `2·cost` of each model (divide by σ² for the likelihood
    /// ratio its bias costs at noise σ).
    pub cost2: [f64; 4],
    /// Draws per noise level `(σ px, per-view error px)`.
    pub draws: Vec<((f64, f64), Vec<Draw>)>,
}

/// Run a cell: the Fisher σ, the noise-free fits, then `draws` noise draws
/// at each `(noise, view_error)` level.
pub fn run_cell(
    cell: &Cell,
    levels: &[(f64, f64)],
    draws: usize,
    seed0: u64,
) -> Result<CellResult> {
    let bench = Bench::new(cell.spec(), &cell.design.view_design())?;
    let fisher_m1 = fisher_sigma(&bench, FreeSet::Default, 1.0);
    let fisher_m2 = fisher_sigma(&bench, FreeSet::Tangential, 1.0);
    let data = bench.train.dataset(0.0, 1);
    let data2 = bench.partner_data(0.0, 1);
    let truth = bench.train.cam;
    let mut bias = [Metrics::default(); 4];
    let mut bias_k = [[0.0; 2]; 4];
    let mut cost2 = [0.0; 4];
    for model in 0..4 {
        let fit = fit_model(model, &data, &truth, &bench.train.poses)?;
        bias[model] = evaluate(&bench, &fit, &data, &data2)?;
        bias_k[model] = [fit.cam.d[0] - truth.d[0], fit.cam.d[1] - truth.d[1]];
        cost2[model] = 2.0 * fit.cost;
    }
    let mut out = Vec::new();
    for (li, &(noise, view_error)) in levels.iter().enumerate() {
        let mut ds = Vec::new();
        for d in 0..draws {
            let seed = seed0 + 7919 * d as u64 + 104_729 * li as u64;
            ds.push(run_draw(&bench, noise, view_error, seed)?);
        }
        out.push(((noise, view_error), ds));
    }
    Ok(CellResult {
        cell: *cell,
        fisher_m1,
        fisher_m2,
        bias,
        bias_k,
        cost2,
        draws: out,
    })
}
