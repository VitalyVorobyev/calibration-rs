//! `--parts field`: is the camera-fixed residual field a camera property,
//! and does a compact (parametric or empirical) description of it exist?
//!
//! Everything here is a linearized fit about the route's solution
//! (`linfield.rs`), held out by view folds exactly like the nonlinear runs,
//! with the field always evaluated at the model's own projection. Per rig
//! (cameras grouped by the id before `_cam`):
//!
//! - **check**: the linearized route, polynomial and spline fields against
//!   their nonlinear held-out numbers;
//! - **Q2**: each camera's field in an orthonormal basis `φ` (complex
//!   monomials of degree ≤ 6 about the image centre, the camera's own
//!   directions removed, orthonormal over the covered image), its spectrum,
//!   the cameras' similarity and principal fields, and compact bases (the
//!   `k` strongest `φ`, the `k` principal fields, leave-one-camera-out)
//!   with per-camera or rig-shared coefficients, every basis derived from
//!   training views only;
//! - **Q1**: what else could make an image-fixed field: blur, depth,
//!   orientation of the board, pixel phase, the perspective bias of a
//!   circle's centre — each as a regressor next to the field (held-out gain,
//!   and how much of the field it takes over), the field of sharp against
//!   blurred markers, and the two rigs' fields against each other;
//! - **Q3**: lens-side terms derived from the decentred-surface expansion
//!   (`model::LensTerm`), alone, with stability and correlations.

use std::collections::BTreeMap;

use anyhow::Result;
use nalgebra::{DMatrix, DVector};

use crate::linfield::{
    self, Basis, BoardPoints, BoardPoly, CMono, CamSol, Combo, Concat, Extra, LinPoint, NCAM, NCF,
    Phi, RawPoly, Scaled, Spec, Spline,
};
use crate::model::{self, Free, GenCfg, GenParams, LensTerm, project_point};
use crate::{CamCtx, CameraData, gen_init};

/// Lens-side candidates (`model::LensTerm`), two columns each.
pub const LENS: &[(&str, LensTerm)] = &[
    ("w̄ (anamorphic)", LensTerm::Mono(0, 1)),
    ("w̄² m-3", LensTerm::Mono(0, 2)),
    ("w³ m+2", LensTerm::Mono(3, 0)),
    ("ww̄² m-2", LensTerm::Mono(1, 2)),
    ("w̄³ m-4", LensTerm::Mono(0, 3)),
    ("w²w̄ m0", LensTerm::Mono(2, 1)),
    ("w³w̄ m+1", LensTerm::Mono(3, 1)),
    ("w²w̄² m-1", LensTerm::Mono(2, 2)),
    ("w⁴ m+3", LensTerm::Mono(4, 0)),
    ("ww̄³ m-3", LensTerm::Mono(1, 3)),
    ("w̄⁴ m-5", LensTerm::Mono(0, 4)),
    ("NAT B5", LensTerm::NatB5),
    ("NAT A5", LensTerm::NatA5),
];
/// Column layout of `LinPoint::extra`.
pub const X_LENS: usize = 0;
const X_ELL: usize = 2 * LENS.len();
const X_PIX: usize = X_ELL + 1;
const X_ORI: usize = X_PIX + 8;
const N_EXTRA: usize = X_ORI + 24;

/// Lens columns of the named terms (`Some(true)`: real part only,
/// `Some(false)`: imaginary part only).
pub fn lens_cols(names: &[(&str, Option<bool>)]) -> Vec<usize> {
    let mut out = Vec::new();
    for (n, part) in names {
        let k = LENS
            .iter()
            .position(|(m, _)| m == n)
            .unwrap_or_else(|| panic!("no lens term {n}"));
        match part {
            None => out.extend([X_LENS + 2 * k, X_LENS + 2 * k + 1]),
            Some(true) => out.push(X_LENS + 2 * k),
            Some(false) => out.push(X_LENS + 2 * k + 1),
        }
    }
    out
}

/// The centre of the image conic of a circle of radius `r_m` about board
/// point `b`, minus the projection of `b` (the perspective bias of a
/// circle's centre), through the full camera model.
fn ellipse_bias(
    params: &GenParams,
    size: [u32; 2],
    pose: &vision_calibration_core::Iso3,
    b: [f64; 2],
    r_m: f64,
) -> [f64; 2] {
    let plain = GenCfg::base("plain");
    let c = project_point(&plain, params, size, pose, b);
    let n = 32;
    let pts: Vec<[f64; 2]> = (0..n)
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n as f64;
            project_point(
                &plain,
                params,
                size,
                pose,
                [b[0] + r_m * t.cos(), b[1] + r_m * t.sin()],
            )
        })
        .collect();
    let m = [
        pts.iter().map(|p| p[0]).sum::<f64>() / n as f64,
        pts.iter().map(|p| p[1]).sum::<f64>() / n as f64,
    ];
    let s = (pts
        .iter()
        .map(|p| (p[0] - m[0]).powi(2) + (p[1] - m[1]).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let a = DMatrix::from_fn(n, 6, |i, j| {
        let (x, y) = ((pts[i][0] - m[0]) / s, (pts[i][1] - m[1]) / s);
        [x * x, x * y, y * y, x, y, 1.0][j]
    });
    let svd = (a.transpose() * &a).symmetric_eigen();
    let k = (0..6)
        .min_by(|&i, &j| svd.eigenvalues[i].total_cmp(&svd.eigenvalues[j]))
        .unwrap_or(0);
    let v = svd.eigenvectors.column(k);
    let (qa, qb, qc, qd, qe) = (v[0], v[1], v[2], v[3], v[4]);
    let det = 4.0 * qa * qc - qb * qb;
    let x0 = (-2.0 * qc * qd + qb * qe) / det;
    let y0 = (qb * qd - 2.0 * qa * qe) / det;
    [m[0] + s * x0 - c[0], m[1] + s * y0 - c[1]]
}

/// Fill `p.extra` ([`N_EXTRA`] columns) at camera `params`.
pub fn add_extras(pts: &mut [LinPoint], params: &GenParams, size: [u32; 2]) {
    let lens_terms: Vec<LensTerm> = LENS.iter().map(|x| x.1).collect();
    let lens_static: &'static [LensTerm] = Box::leak(lens_terms.into_boxed_slice());
    let cfg = GenCfg {
        lens: lens_static,
        ..GenCfg::base("lens")
    };
    let mut q = params.clone();
    q.lens = vec![0.0; 2 * LENS.len()];
    for p in pts.iter_mut() {
        let mut ex = vec![[0.0; 2]; N_EXTRA];
        for k in 0..2 * LENS.len() {
            let h = 1e-6;
            let (mut a, mut b) = (q.clone(), q.clone());
            a.lens[k] = h;
            b.lens[k] = -h;
            let pa = project_point(&cfg, &a, size, &p.pose, p.board);
            let pb = project_point(&cfg, &b, size, &p.pose, p.board);
            ex[X_LENS + k] = [(pa[0] - pb[0]) / (2.0 * h), (pa[1] - pb[1]) / (2.0 * h)];
        }
        ex[X_ELL] = ellipse_bias(params, size, &p.pose, p.board, 1e-3);
        let tau = std::f64::consts::TAU;
        let (su, cu) = (tau * p.u[0]).sin_cos();
        let (sv, cv) = (tau * p.u[1]).sin_cos();
        for (k, v) in [su, cu, sv, cv].iter().enumerate() {
            ex[X_PIX + 2 * k] = [*v, 0.0];
            ex[X_PIX + 2 * k + 1] = [0.0, *v];
        }
        for k in 1..=6 {
            let (s, c) = (k as f64 * p.roll).sin_cos();
            let at = X_ORI + 4 * (k - 1);
            ex[at] = [c, 0.0];
            ex[at + 1] = [0.0, c];
            ex[at + 2] = [s, 0.0];
            ex[at + 3] = [0.0, s];
        }
        p.extra = ex;
    }
}

/// Regressors of the defocus model `σ² = s₀² + (a + b·x + c·y + d/z)²`:
/// a full quadratic in `(x, y, 1/z)` (`x, y` half-size-normalized pixels,
/// `z` in units of 0.25 m).
fn blur_row(p: &LinPoint, size: [u32; 2]) -> [f64; 10] {
    let (w, h) = (f64::from(size[0]) / 2.0, f64::from(size[1]) / 2.0);
    let (x, y, z) = ((p.u[0] - w) / w, (p.u[1] - h) / h, 0.25 / p.depth);
    [1.0, x, y, z, x * x, x * y, y * y, x * z, y * z, z * z]
}

/// The geometric blur model fitted to the detector's blur (least squares on
/// `σ²`), and the share of the blur variance it explains.
fn blur_model(pts: &[LinPoint], size: [u32; 2]) -> Option<(DVector<f64>, f64)> {
    let with: Vec<&LinPoint> = pts.iter().filter(|p| p.blur.is_some()).collect();
    if with.len() < 50 {
        return None;
    }
    let a = DMatrix::from_fn(with.len(), 10, |i, j| blur_row(with[i], size)[j]);
    let b = DVector::from_iterator(
        with.len(),
        with.iter().map(|p| p.blur.unwrap_or(0.0).powi(2)),
    );
    let coef = (a.transpose() * &a).cholesky()?.solve(&(a.transpose() * b));
    let pred: Vec<f64> = with.iter().map(|p| blur_predict(&coef, p, size)).collect();
    let obs: Vec<f64> = with.iter().map(|p| p.blur.unwrap_or(0.0)).collect();
    let (m, sd) = mean_std(&obs);
    let rss: f64 = obs.iter().zip(&pred).map(|(o, q)| (o - q).powi(2)).sum();
    let _ = m;
    Some((coef, 1.0 - rss / (sd * sd * (obs.len() as f64 - 1.0))))
}

fn blur_predict(coef: &DVector<f64>, p: &LinPoint, size: [u32; 2]) -> f64 {
    let r = blur_row(p, size);
    (0..10).map(|j| r[j] * coef[j]).sum::<f64>().max(0.0).sqrt()
}

pub struct CamLin {
    pub id: String,
    pub size: [u32; 2],
    pub base: GenParams,
    pub full: Vec<LinPoint>,
    /// (train, test) per fold.
    pub folds: Vec<(Vec<LinPoint>, Vec<LinPoint>)>,
    /// Cells of the [`GRID`] this camera's detections cover.
    pub cover: Vec<bool>,
    /// The geometric blur model ([`blur_model`]), when the detector fits a
    /// blur.
    pub blur_coef: Option<DVector<f64>>,
}

pub struct Rig {
    pub name: String,
    pub cams: Vec<CamLin>,
    pub typical: GenParams,
    /// Covered cells: centre and the typical camera's directions there.
    pub grid: Vec<([f64; 2], [[f64; 2]; NCF])>,
    /// [`GRID`] index of each `grid` cell.
    pub grid_idx: Vec<usize>,
    pub phi: Phi,
}

const GRID: [usize; 2] = [48, 36];

fn typical(ps: &[GenParams]) -> GenParams {
    let mut t = ps[0].clone();
    let n = ps.len() as f64;
    for k in 0..NCAM {
        let m = ps.iter().map(|p| linfield::cam_get(p, k)).sum::<f64>() / n;
        let cur = linfield::cam_get(&t, k);
        linfield::cam_add(&mut t, k, m - cur);
    }
    t
}

/// Linearize every camera of every rig (full data and per fold).
pub fn build(cams: &[CameraData], ctxs: &[CamCtx]) -> Result<Vec<Rig>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, c) in cams.iter().enumerate() {
        let rig =
            c.id.rsplit_once("_cam")
                .map_or(c.id.clone(), |(a, _)| a.to_string());
        groups.entry(rig).or_default().push(i);
    }
    let mut rigs = Vec::new();
    for (name, idx) in groups {
        let bases: Vec<GenParams> = idx
            .iter()
            .map(|&i| gen_init(&ctxs[i].base, ctxs[i].centre))
            .collect::<Result<_>>()?;
        let typ = typical(&bases);
        let mut lins = Vec::new();
        let mut covered = vec![false; GRID[0] * GRID[1]];
        for (&i, base) in idx.iter().zip(&bases) {
            let (cam, ctx) = (&cams[i], &ctxs[i]);
            let size = cam.size;
            let n = cam.dataset.num_views();
            let all: Vec<usize> = (0..n).collect();
            let views = |ids: &[usize]| -> Vec<_> {
                ids.iter().map(|&k| cam.dataset.views[k].clone()).collect()
            };
            let blur_of = |p: &mut LinPoint| {
                p.blur = cam.blur.get(&crate::blur::key(p.obs)).copied();
            };
            let mut full = linfield::linearize(
                base,
                &typ,
                size,
                &all,
                &cam.dataset.views,
                &ctx.base.params.cam_se3_target,
            );
            full.iter_mut().for_each(blur_of);
            add_extras(&mut full, base, size);
            let mut cover = vec![false; GRID[0] * GRID[1]];
            for p in &full {
                let gi =
                    ((p.obs[0] / f64::from(size[0]) * GRID[0] as f64) as usize).min(GRID[0] - 1);
                let gj =
                    ((p.obs[1] / f64::from(size[1]) * GRID[1] as f64) as usize).min(GRID[1] - 1);
                covered[gj * GRID[0] + gi] = true;
                cover[gj * GRID[0] + gi] = true;
            }
            let mut folds = Vec::new();
            for ((train, test), fr) in ctx.folds.iter().zip(&ctx.fold_routes) {
                let th = gen_init(fr, ctx.centre)?;
                let mut tr = linfield::linearize(
                    &th,
                    &typ,
                    size,
                    train,
                    &views(train),
                    &fr.params.cam_se3_target,
                );
                let init: Vec<_> = test
                    .iter()
                    .map(|&k| ctx.base.params.cam_se3_target[k])
                    .collect();
                let po = model::fit(
                    &GenCfg::base("plain"),
                    test,
                    &views(test),
                    &th,
                    &init,
                    size,
                    Free::PosesOnly,
                    200,
                )?;
                let mut te = linfield::linearize(&th, &typ, size, test, &views(test), &po.poses);
                tr.iter_mut().for_each(blur_of);
                te.iter_mut().for_each(blur_of);
                add_extras(&mut tr, &th, size);
                add_extras(&mut te, &th, size);
                folds.push((tr, te));
            }
            let blur_fit = blur_model(&full, size);
            let blur_coef = blur_fit.as_ref().map(|b| b.0.clone());
            if let Some((coef, r2)) = blur_fit {
                eprintln!(
                    "  {}: blur² from (x, y, 1/z) explains {:.0} % of the blur variance",
                    cam.id,
                    100.0 * r2
                );
                let apply = |p: &mut LinPoint| p.gblur = Some(blur_predict(&coef, p, size));
                full.iter_mut().for_each(apply);
                for (tr, te) in &mut folds {
                    tr.iter_mut().for_each(apply);
                    te.iter_mut().for_each(apply);
                }
            }
            eprintln!("  linearized {} ({} points)", cam.id, full.len());
            lins.push(CamLin {
                id: cam.id.clone(),
                size,
                base: base.clone(),
                full,
                folds,
                cover,
                blur_coef,
            });
        }
        let size = lins[0].size;
        let grid_idx: Vec<usize> = (0..GRID[0] * GRID[1]).filter(|&k| covered[k]).collect();
        let grid: Vec<([f64; 2], [[f64; 2]; NCF])> = grid_idx
            .iter()
            .map(|&k| {
                let u = [
                    (k % GRID[0]) as f64 * f64::from(size[0]) / GRID[0] as f64
                        + f64::from(size[0]) / GRID[0] as f64 / 2.0,
                    (k / GRID[0]) as f64 * f64::from(size[1]) / GRID[1] as f64
                        + f64::from(size[1]) / GRID[1] as f64 / 2.0,
                ];
                (u, linfield::cam_fields(&typ, size, u))
            })
            .collect();
        let half = f64::from(size[0]) / 2.0;
        let mono = CMono::degrees(0, 6, [half, f64::from(size[1]) / 2.0], half);
        let phi = Phi::build(mono, &grid);
        rigs.push(Rig {
            name,
            cams: lins,
            typical: typ,
            grid,
            grid_idx,
            phi,
        });
    }
    Ok(rigs)
}

// ───────────────────────── evaluation ─────────────────────────

/// One camera's term for one data set: built per fold (`Some(f)`) or for
/// the full data (`None`).
pub(crate) struct Local {
    camera_free: bool,
    basis: Option<Box<dyn Basis>>,
    penalty: Option<DMatrix<f64>>,
    ortho: bool,
    held: Option<(Box<dyn Basis>, DVector<f64>)>,
}

impl Local {
    pub(crate) fn free(basis: Option<Box<dyn Basis>>) -> Self {
        Self {
            camera_free: true,
            basis,
            penalty: None,
            ortho: false,
            held: None,
        }
    }
    fn spec(&self) -> Spec<'_> {
        Spec {
            camera_free: self.camera_free,
            local: self.basis.as_deref(),
            penalty: self.penalty.as_ref(),
            ortho: self.ortho,
            shared: None,
            held: self.held.as_ref().map(|(b, c)| (b.as_ref(), c)),
        }
    }
}

/// A model's held-out result.
pub(crate) struct Row {
    name: String,
    /// Held-out residuals per camera (all folds).
    pub(crate) held: Vec<Vec<[f64; 2]>>,
    /// Full-data solution per camera.
    pub(crate) full: Vec<CamSol>,
    /// Fold solutions per camera.
    pub(crate) folds: Vec<Vec<CamSol>>,
}

impl Row {
    pub(crate) fn pooled(&self) -> [f64; 3] {
        let all: Vec<[f64; 2]> = self.held.iter().flatten().copied().collect();
        linfield::stats(&all)
    }
    pub(crate) fn line(&self, route: Option<f64>) -> String {
        let s = self.pooled();
        let per: Vec<String> = self
            .held
            .iter()
            .map(|h| format!("{:.3}", linfield::stats(h)[1]))
            .collect();
        let gain = route.map_or(String::new(), |r| {
            format!(" | {:+.1} %", 100.0 * (s[0] * s[0] - r * r) / (r * r))
        });
        format!(
            "| {} | {:.4} / {:.4} / {:.4}{} | {} |",
            self.name,
            s[0],
            s[1],
            s[2],
            gain,
            per.join(" | ")
        )
    }
}

pub(crate) fn eval_local(
    rig: &Rig,
    name: &str,
    make: &dyn Fn(Option<usize>, usize) -> Local,
) -> Row {
    let mut held = Vec::new();
    let mut full = Vec::new();
    let mut folds = Vec::new();
    for (c, cam) in rig.cams.iter().enumerate() {
        let mut h = Vec::new();
        let mut fs = Vec::new();
        for (f, (tr, te)) in cam.folds.iter().enumerate() {
            let l = make(Some(f), c);
            let sol = linfield::fit(&[tr], &l.spec());
            let cs = sol.cams[0].clone();
            h.extend(linfield::predict(
                te,
                &cs.dtheta,
                l.basis.as_deref().map(|b| (b, &cs.local)),
                None,
                l.held.as_ref().map(|(b, x)| (b.as_ref(), x)),
            ));
            fs.push(cs);
        }
        let l = make(None, c);
        full.push(linfield::fit(&[&cam.full], &l.spec()).cams[0].clone());
        held.push(h);
        folds.push(fs);
    }
    Row {
        name: name.to_string(),
        held,
        full,
        folds,
    }
}

/// A rig-shared term (one coefficient set for every camera), cameras free.
pub(crate) fn eval_shared(
    rig: &Rig,
    name: &str,
    make: &dyn Fn(Option<usize>) -> Box<dyn Basis>,
) -> (Row, Vec<DVector<f64>>) {
    eval_shared_local(rig, name, None, make)
}

/// [`eval_shared`] with a per-camera term `local` next to the shared one.
fn eval_shared_local(
    rig: &Rig,
    name: &str,
    local: Option<&dyn Basis>,
    make: &dyn Fn(Option<usize>) -> Box<dyn Basis>,
) -> (Row, Vec<DVector<f64>>) {
    let nf = rig.cams[0].folds.len();
    let mut held = vec![Vec::new(); rig.cams.len()];
    let mut folds = vec![Vec::new(); rig.cams.len()];
    let mut shared = Vec::new();
    for f in 0..nf {
        let b = make(Some(f));
        let spec = Spec {
            camera_free: true,
            local,
            penalty: None,
            ortho: false,
            shared: Some(b.as_ref()),
            held: None,
        };
        let sets: Vec<&[LinPoint]> = rig.cams.iter().map(|c| c.folds[f].0.as_slice()).collect();
        let sol = linfield::fit(&sets, &spec);
        for (c, cam) in rig.cams.iter().enumerate() {
            held[c].extend(linfield::predict(
                &cam.folds[f].1,
                &sol.cams[c].dtheta,
                local.map(|l| (l, &sol.cams[c].local)),
                Some((b.as_ref(), &sol.shared)),
                None,
            ));
            folds[c].push(sol.cams[c].clone());
        }
        shared.push(sol.shared.clone());
    }
    let b = make(None);
    let spec = Spec {
        camera_free: true,
        local,
        penalty: None,
        ortho: false,
        shared: Some(b.as_ref()),
        held: None,
    };
    let sets: Vec<&[LinPoint]> = rig.cams.iter().map(|c| c.full.as_slice()).collect();
    let sol = linfield::fit(&sets, &spec);
    shared.push(sol.shared.clone());
    (
        Row {
            name: name.to_string(),
            held,
            full: sol.cams,
            folds,
        },
        shared,
    )
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    crate::mean_std(v)
}

/// Jackknife sd over the folds of the local coefficient `k`, divided by the
/// full-data σ; RMS over cameras.
pub(crate) fn jack_ratio(row: &Row, k: usize, n_theta: usize) -> f64 {
    let mut acc = 0.0;
    let mut n = 0.0;
    for (full, folds) in row.full.iter().zip(&row.folds) {
        let v: Vec<f64> = folds.iter().map(|s| s.local[k]).collect();
        let m = v.len() as f64;
        let mu = v.iter().sum::<f64>() / m;
        let jk = ((m - 1.0) / m * v.iter().map(|x| (x - mu).powi(2)).sum::<f64>()).sqrt();
        let sig = (full.sigma2 * full.cov[(n_theta + k, n_theta + k)]).sqrt();
        acc += (jk / sig).powi(2);
        n += 1.0;
    }
    (acc / n).sqrt()
}

/// Largest |correlation| of local coefficient `k` with the camera's
/// parameters (full data), per camera; returns the parameter name and the
/// value with the largest magnitude over cameras.
fn max_corr(row: &Row, k: usize) -> (String, f64) {
    let mut best = (String::new(), 0.0f64);
    for s in &row.full {
        let i = NCAM + k;
        for j in 0..NCAM {
            let r = s.cov[(i, j)] / (s.cov[(i, i)] * s.cov[(j, j)]).sqrt();
            if r.abs() > best.1.abs() {
                best = (linfield::CAM_NAMES[j].to_string(), r);
            }
        }
    }
    best
}

/// Per camera φ-field: camera free, every φ.
fn phi_fields(rig: &Rig, pts: &dyn Fn(&CamLin) -> &[LinPoint]) -> Vec<CamSol> {
    rig.cams
        .iter()
        .map(|c| {
            linfield::fit(
                &[pts(c)],
                &Spec {
                    camera_free: true,
                    local: Some(&rig.phi),
                    penalty: None,
                    ortho: false,
                    shared: None,
                    held: None,
                },
            )
            .cams[0]
                .clone()
        })
        .collect()
}

fn cos_sim(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    a.dot(b) / (a.norm() * b.norm())
}

/// Energy of the coefficient noise of a φ-field (Σ var), px².
fn noise_energy(s: &CamSol, n_theta: usize, n: usize) -> f64 {
    (0..n)
        .map(|k| s.sigma2 * s.cov[(n_theta + k, n_theta + k)])
        .sum()
}

/// Indices of the `k` largest entries of `|v|`.
fn top_k(v: &DVector<f64>, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].abs().total_cmp(&v[a].abs()));
    idx.truncate(k);
    idx
}

fn selection(phi: &Phi, idx: &[usize]) -> Combo {
    let mut w = DMatrix::zeros(idx.len(), phi.n());
    for (r, &k) in idx.iter().enumerate() {
        w[(r, k)] = 1.0;
    }
    Combo {
        phi: phi.clone(),
        w,
    }
}

/// The Gram matrix of `φ` over the rig cells all of `cams` cover: the
/// metric in which fields are compared where every camera saw them.
fn metric(rig: &Rig, cams: &[usize]) -> DMatrix<f64> {
    let n = rig.phi.n();
    let mut w = DMatrix::<f64>::zeros(n, n);
    let mut buf = vec![[0.0; 2]; n];
    let mut cnt: f64 = 0.0;
    for (g, &k) in rig.grid.iter().zip(&rig.grid_idx) {
        if !cams.iter().all(|&c| rig.cams[c].cover[k]) {
            continue;
        }
        rig.phi.value(g.0, &g.1, &mut buf);
        for a in 0..n {
            for b in 0..n {
                w[(a, b)] += buf[a][0] * buf[b][0] + buf[a][1] * buf[b][1];
            }
        }
        cnt += 1.0;
    }
    w / cnt.max(1.0)
}

/// `⟨a, b⟩ / (‖a‖‖b‖)` in metric `w`.
fn cos_w(w: &DMatrix<f64>, a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    let ab = a.dot(&(w * b));
    ab / (a.dot(&(w * a)) * b.dot(&(w * b))).sqrt()
}

/// The top `k` principal fields (uncentered) of the rows, in metric `w`:
/// returned as `φ` coefficient rows of unit `w`-norm, with each
/// component's share of the energy.
fn principal_w(rows: &[DVector<f64>], k: usize, w: &DMatrix<f64>) -> (DMatrix<f64>, Vec<f64>) {
    let n = rows[0].len();
    let e = w.clone().symmetric_eigen();
    let mx = e.eigenvalues.max();
    let mut half = DMatrix::<f64>::zeros(n, n);
    let mut ihalf = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let l = e.eigenvalues[i];
        if l > 1e-9 * mx {
            let v = e.eigenvectors.column(i);
            half += (v * v.transpose()) * l.sqrt();
            ihalf += (v * v.transpose()) / l.sqrt();
        }
    }
    let b: Vec<DVector<f64>> = rows.iter().map(|r| &half * r).collect();
    let (wv, frac) = principal(&b, k);
    let out = DMatrix::from_fn(wv.nrows(), n, |r, j| (&ihalf * wv.row(r).transpose())[j]);
    (out, frac)
}

/// The top `k` right singular vectors (uncentered) of the rows `a`.
fn principal(rows: &[DVector<f64>], k: usize) -> (DMatrix<f64>, Vec<f64>) {
    let n = rows[0].len();
    let m = DMatrix::from_fn(rows.len(), n, |i, j| rows[i][j]);
    let svd = m.svd(false, true);
    let vt = svd.v_t.expect("v_t");
    let mut order: Vec<usize> = (0..svd.singular_values.len()).collect();
    order.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let tot: f64 = svd.singular_values.iter().map(|s| s * s).sum();
    let frac: Vec<f64> = order
        .iter()
        .map(|&i| svd.singular_values[i].powi(2) / tot)
        .collect();
    let k = k.min(order.len());
    let w = DMatrix::from_fn(k, n, |r, j| vt[(order[r], j)]);
    (w, frac)
}

fn mean_vec(rows: &[DVector<f64>]) -> DVector<f64> {
    let mut m = DVector::zeros(rows[0].len());
    for r in rows {
        m += r;
    }
    m / rows.len() as f64
}

// ───────────────────────── the study ─────────────────────────

pub fn run(cams: &[CameraData], ctxs: &[CamCtx]) -> Result<()> {
    let mut rigs = build(cams, ctxs)?;
    // A rig whose detector fits no blur takes the same unit's geometric blur
    // model from a rig whose detector does (same cameras, same focus).
    let models: BTreeMap<String, DVector<f64>> = rigs
        .iter()
        .flat_map(|r| r.cams.iter())
        .filter_map(|c| {
            let unit = c.id.rsplit_once("_cam").map(|x| x.1.to_string())?;
            c.blur_coef.clone().map(|m| (unit, m))
        })
        .collect();
    for rig in &mut rigs {
        for c in &mut rig.cams {
            if c.blur_coef.is_some() {
                continue;
            }
            let Some(m) = c.id.rsplit_once("_cam").and_then(|x| models.get(x.1)) else {
                continue;
            };
            eprintln!(
                "  {}: geometric blur from the same unit's other capture",
                c.id
            );
            let size = c.size;
            let apply = |p: &mut LinPoint| p.gblur = Some(blur_predict(m, p, size));
            c.full.iter_mut().for_each(apply);
            for (tr, te) in &mut c.folds {
                tr.iter_mut().for_each(apply);
                te.iter_mut().for_each(apply);
            }
        }
    }
    let mut rig_means = Vec::new();
    for rig in &rigs {
        let fields = study_rig(rig)?;
        rig_means.push(fields);
    }
    if rigs.len() == 2 {
        cross_rig(&rigs, &rig_means);
    }
    Ok(())
}

/// Each camera's full-data φ-field, for the cross-rig comparison.
struct RigFields {
    full: Vec<CamSol>,
    /// Per-camera full-data coefficients of named lens-side sets.
    lens: BTreeMap<String, Vec<DVector<f64>>>,
}

fn study_rig(rig: &Rig) -> Result<RigFields> {
    let nphi = rig.phi.n();
    let size = rig.cams[0].size;
    let (w, h) = (f64::from(size[0]), f64::from(size[1]));
    println!(
        "\n# Rig {} — {} cameras, {} covered cells of {}, {} φ fields (degree ≤ 6)",
        rig.name,
        rig.cams.len(),
        rig.grid.len(),
        GRID[0] * GRID[1],
        nphi
    );
    println!(
        "typical camera: fx {:.1} fy {:.1} cx {:.1} cy {:.1} k1 {:.3} k2 {:.3} tilt {:.4} {:.4}",
        rig.typical.intr[0],
        rig.typical.intr[1],
        rig.typical.intr[2],
        rig.typical.intr[3],
        rig.typical.dist[0],
        rig.typical.dist[1],
        rig.typical.tilt[0],
        rig.typical.tilt[1]
    );
    for c in &rig.cams {
        let b = &c.base;
        println!(
            "  {}: fx {:.1} fy {:.1} cx {:.1} cy {:.1} k1 {:.4} k2 {:.3} tilt {:+.4} {:+.4}",
            c.id,
            b.intr[0],
            b.intr[1],
            b.intr[2],
            b.intr[3],
            b.dist[0],
            b.dist[1],
            b.tilt[0],
            b.tilt[1]
        );
    }

    // ── check: the linearized harness against the nonlinear numbers ──
    println!(
        "\n## Check: linearized held-out (rms / mean / median px, Δ variance vs route, per-camera mean)"
    );
    let route = eval_local(rig, "route", &|_, _| Local::free(None));
    let r0 = route.pooled()[0];
    println!("{}", route.line(None));
    let poly = eval_local(rig, "poly 2-4 camera held", &|_, _| Local {
        camera_free: false,
        basis: Some(Box::new(RawPoly {
            terms: model::monomials(2, 4),
            half: [w / 2.0, h / 2.0],
        })),
        penalty: None,
        ortho: false,
        held: None,
    });
    println!("{}", poly.line(Some(r0)));
    for (cells, ortho) in [([6usize, 5usize], false), ([10, 8], true), ([10, 8], false)] {
        let row = eval_local(
            rig,
            &format!(
                "spline {}x{} {} s10",
                cells[0],
                cells[1],
                if ortho { "ortho" } else { "camera held" }
            ),
            &|_, _| Local {
                camera_free: ortho,
                basis: Some(Box::new(Spline {
                    cells,
                    size: [w, h],
                })),
                penalty: Some(linfield::spline_penalty(cells, 10.0)),
                ortho,
                held: None,
            },
        );
        println!("{}", row.line(Some(r0)));
    }
    let phi_row = eval_local(rig, "φ≤6 per camera (camera free)", &|_, _| {
        Local::free(Some(Box::new(rig.phi.clone())))
    });
    println!("{}", phi_row.line(Some(r0)));

    // ── Q2: spectrum, similarity, principal fields ──
    let full = phi_fields(rig, &|c| &c.full);
    let nt = NCAM;
    println!("\n## Q2 spectrum: each camera's φ-field (full data)");
    println!(
        "| camera | field RMS px | noise RMS px | by degree % (1,2,3,4,5,6) | largest |m| groups % |"
    );
    for (c, s) in rig.cams.iter().zip(&full) {
        let e: f64 = s.local.iter().map(|v| v * v).sum();
        let ne = noise_energy(s, nt, nphi);
        let mut by_deg = [0.0; 7];
        let mut by_m: BTreeMap<i32, f64> = BTreeMap::new();
        for j in 0..nphi {
            let (d, m) = rig.phi.order(j);
            by_deg[d as usize] += s.local[j].powi(2);
            *by_m.entry(m).or_default() += s.local[j].powi(2);
        }
        let mut ms: Vec<(i32, f64)> = by_m.into_iter().collect();
        ms.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!(
            "| {} | {:.4} | {:.4} | {} | {} |",
            c.id,
            e.sqrt(),
            ne.sqrt(),
            (1..7)
                .map(|d| format!("{:.0}", 100.0 * by_deg[d] / e))
                .collect::<Vec<_>>()
                .join(" "),
            ms.iter()
                .take(4)
                .map(|(m, v)| format!("m{m:+}:{:.0}", 100.0 * v / e))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    let a_full: Vec<DVector<f64>> = full.iter().map(|s| s.local.clone()).collect();
    let mean = mean_vec(&a_full);
    let em: f64 = mean.norm_squared();
    println!(
        "rig mean field: RMS {:.4} px; top φ of the mean (label, share of its energy):",
        em.sqrt()
    );
    let order = top_k(&mean, nphi);
    let mut cum = 0.0;
    for (r, &j) in order.iter().take(12).enumerate() {
        let (p, q, re) = rig.phi.labels[j];
        cum += mean[j].powi(2) / em;
        println!(
            "  {:>2}. z^{p} z̄^{q} {} (deg {}, m {:+}) {:+.4} px  {:.1} % (cum {:.1} %)",
            r + 1,
            if re { "re" } else { "im" },
            p + q,
            p - q - 1,
            mean[j],
            100.0 * mean[j].powi(2) / em,
            100.0 * cum
        );
    }
    println!(
        "cumulative share of each camera's field energy in the rig mean's top-k φ (k = 1, 2, 4, 6, 10):"
    );
    for (c, a) in rig.cams.iter().zip(&a_full) {
        let e = a.norm_squared();
        let fr: Vec<String> = [1, 2, 4, 6, 10]
            .iter()
            .map(|&k| {
                let s: f64 = order.iter().take(k).map(|&j| a[j].powi(2)).sum();
                format!("{:.0}", 100.0 * s / e)
            })
            .collect();
        let own: Vec<String> = [1, 2, 4, 6, 10]
            .iter()
            .map(|&k| {
                let s: f64 = top_k(a, k).iter().map(|&j| a[j].powi(2)).sum();
                format!("{:.0}", 100.0 * s / e)
            })
            .collect();
        println!(
            "  {}: rig-mean top-k {} %; own top-k {} %",
            c.id,
            fr.join(" / "),
            own.join(" / ")
        );
    }
    let all_cams: Vec<usize> = (0..rig.cams.len()).collect();
    let w_all = metric(rig, &all_cams);
    let n_common = rig
        .grid_idx
        .iter()
        .filter(|&&k| rig.cams.iter().all(|c| c.cover[k]))
        .count();
    let own_rms: Vec<String> = a_full
        .iter()
        .enumerate()
        .map(|(c, a)| format!("{:.3}", a.dot(&(metric(rig, &[c]) * a)).sqrt()))
        .collect();
    println!(
        "field RMS over each camera's own coverage: {} px; {} cells covered by every camera",
        own_rms.join(" "),
        n_common
    );
    println!("field correlation of camera pairs over the cells both cover:");
    let mut pair_sims = Vec::new();
    for i in 0..a_full.len() {
        let row: Vec<String> = (0..a_full.len())
            .map(|j| {
                let c = cos_w(&metric(rig, &[i, j]), &a_full[i], &a_full[j]);
                if j > i {
                    pair_sims.push(c);
                }
                format!("{c:+.2}")
            })
            .collect();
        println!("  cam{i} {}", row.join(" "));
    }
    let (ps_m, ps_sd) = mean_std(&pair_sims);
    println!("  pairs: mean {ps_m:+.2} ± {ps_sd:.2}");
    // Reliability: the same camera's fold fields against each other.
    let folds: Vec<Vec<CamSol>> = (0..rig.cams[0].folds.len())
        .map(|f| phi_fields(rig, &|c| &c.folds[f].0))
        .collect();
    let mut rel = Vec::new();
    for c in 0..rig.cams.len() {
        let mut v = Vec::new();
        for a in 0..folds.len() {
            for b in a + 1..folds.len() {
                v.push(cos_w(
                    &metric(rig, &[c]),
                    &folds[a][c].local,
                    &folds[b][c].local,
                ));
            }
        }
        rel.push(mean_std(&v).0);
    }
    println!(
        "same camera, training folds against each other (reliability ceiling): {}",
        rel.iter()
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let noise_frac: Vec<String> = full
        .iter()
        .map(|s| {
            format!(
                "{:.2}",
                1.0 - noise_energy(s, nt, nphi) / s.local.norm_squared()
            )
        })
        .collect();
    println!(
        "signal fraction of each field's energy (1 − noise/total): {}",
        noise_frac.join(" ")
    );
    let (_, frac) = principal_w(&a_full, a_full.len(), &w_all);
    let wn = |a: &DVector<f64>, b: &DVector<f64>| a.dot(&(&w_all * b));
    let along_mean: f64 = a_full
        .iter()
        .map(|a| wn(a, &mean).powi(2) / wn(&mean, &mean))
        .sum::<f64>()
        / a_full.iter().map(|a| wn(a, a)).sum::<f64>();
    println!(
        "principal fields over the common cells (uncentered): energy fractions {}; along the rig mean {:.0} %",
        frac.iter()
            .map(|f| format!("{:.0}%", 100.0 * f))
            .collect::<Vec<_>>()
            .join(" "),
        100.0 * along_mean
    );

    // Mirror symmetry about the vertical through the principal point (the
    // plane of the tilt): `(dx, dy)(x, y) = (−dx, dy)(2cx − x, y)` for a
    // field the tilted system makes; a target seen at six rolls has no
    // reason to keep it. Share of each field's energy that is symmetric,
    // over the cells whose mirror image the camera also covers (0.5 for
    // an unstructured field).
    let sym: Vec<String> = rig
        .cams
        .iter()
        .zip(&a_full)
        .map(|(c, a)| {
            let size = c.size;
            let cx = c.base.intr[2];
            let gw = f64::from(size[0]) / GRID[0] as f64;
            let gh = f64::from(size[1]) / GRID[1] as f64;
            let mut buf = vec![[0.0; 2]; nphi];
            let field = |u: [f64; 2], buf: &mut Vec<[f64; 2]>| {
                let cf = linfield::cam_fields(&rig.typical, size, u);
                rig.phi.value(u, &cf, buf);
                let mut f = [0.0; 2];
                for (k, b) in buf.iter().enumerate() {
                    f[0] += a[k] * b[0];
                    f[1] += a[k] * b[1];
                }
                f
            };
            let (mut s2, mut t2) = (0.0, 0.0);
            for k in 0..GRID[0] * GRID[1] {
                if !c.cover[k] {
                    continue;
                }
                let u = [
                    (k % GRID[0]) as f64 * gw + gw / 2.0,
                    (k / GRID[0]) as f64 * gh + gh / 2.0,
                ];
                let um = [2.0 * cx - u[0], u[1]];
                let (gi, gj) = ((um[0] / gw).floor(), (um[1] / gh).floor());
                if gi < 0.0 || gi >= GRID[0] as f64 || !c.cover[gj as usize * GRID[0] + gi as usize]
                {
                    continue;
                }
                let f = field(u, &mut buf);
                let g = field(um, &mut buf);
                let sym = [(f[0] - g[0]) / 2.0, (f[1] + g[1]) / 2.0];
                s2 += sym[0] * sym[0] + sym[1] * sym[1];
                t2 += f[0] * f[0] + f[1] * f[1];
            }
            format!("{:.2}", s2 / t2)
        })
        .collect();
    println!(
        "mirror-symmetric share of each field (about the vertical through cx; 0.5 = none): {}",
        sym.join(" ")
    );

    // ── Q2: compact bases, held-out ──
    let nf = rig.cams[0].folds.len();
    let fold_means: Vec<DVector<f64>> = (0..nf)
        .map(|f| mean_vec(&folds[f].iter().map(|s| s.local.clone()).collect::<Vec<_>>()))
        .collect();
    let fold_rows: Vec<Vec<DVector<f64>>> = (0..nf)
        .map(|f| folds[f].iter().map(|s| s.local.clone()).collect())
        .collect();
    let mean_of = |f: Option<usize>| match f {
        Some(f) => fold_means[f].clone(),
        None => mean.clone(),
    };
    let rows_of = |f: Option<usize>| match f {
        Some(f) => fold_rows[f].clone(),
        None => a_full.clone(),
    };
    println!("\n## Q2 compact bases, held-out (bases from training views only)");
    println!("| model | rms / mean / median | Δ var vs route | per-camera mean |");
    println!("{}", route.line(None));
    println!("{}", phi_row.line(Some(r0)));
    let mut compact: Vec<(String, Row, usize)> = Vec::new();
    for k in [1usize, 2, 4, 6, 10] {
        let row = eval_local(
            rig,
            &format!("φ top-{k} of rig mean, per camera"),
            &|f, _| Local::free(Some(Box::new(selection(&rig.phi, &top_k(&mean_of(f), k))))),
        );
        println!("{}", row.line(Some(r0)));
        compact.push((format!("φ top-{k}"), row, k));
        let (row, _) = eval_shared(rig, &format!("φ top-{k} of rig mean, shared"), &|f| {
            Box::new(selection(&rig.phi, &top_k(&mean_of(f), k)))
        });
        println!("{}", row.line(Some(r0)));
    }
    for k in [1usize, 2, 4] {
        let row = eval_local(rig, &format!("principal {k}, per camera"), &|f, _| {
            Local::free(Some(Box::new(Combo {
                phi: rig.phi.clone(),
                w: principal_w(&rows_of(f), k, &w_all).0,
            })))
        });
        println!("{}", row.line(Some(r0)));
        compact.push((format!("principal {k}"), row, k));
        let (row, _) = eval_shared(rig, &format!("principal {k}, shared"), &|f| {
            Box::new(Combo {
                phi: rig.phi.clone(),
                w: principal_w(&rows_of(f), k, &w_all).0,
            })
        });
        println!("{}", row.line(Some(r0)));
    }
    // Leave one camera out: the other cameras' full-data fields.
    let others = |c: usize| -> Vec<DVector<f64>> {
        a_full
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != c)
            .map(|(_, a)| a.clone())
            .collect()
    };
    let row = eval_local(rig, "LOCO: others' mean field, held", &|_, c| {
        let m = mean_vec(&others(c));
        Local {
            camera_free: true,
            basis: None,
            penalty: None,
            ortho: false,
            held: Some((Box::new(rig.phi.clone()), m)),
        }
    });
    println!("{}", row.line(Some(r0)));
    for k in [1usize, 2, 4] {
        let row = eval_local(
            rig,
            &format!("LOCO: others' principal {k}, per camera"),
            &|_, c| {
                Local::free(Some(Box::new(Combo {
                    phi: rig.phi.clone(),
                    w: principal_w(&others(c), k, &w_all).0,
                })))
            },
        );
        println!("{}", row.line(Some(r0)));
    }

    // Stability of the compact per-camera terms with a fixed basis.
    println!("\n## Q2 stability: per-camera coefficients of a fixed basis (from all views)");
    println!(
        "| basis | coef | across cameras mean ± sd px | jackknife sd / σ (RMS over cameras) | largest |corr| with the camera |"
    );
    for (label, basis) in [
        (
            "φ top-4",
            Box::new(selection(&rig.phi, &top_k(&mean, 4))) as Box<dyn Basis>,
        ),
        (
            "principal 2",
            Box::new(Combo {
                phi: rig.phi.clone(),
                w: principal_w(&a_full, 2, &w_all).0,
            }),
        ),
    ] {
        let n = basis.len();
        let row = eval_local(rig, label, &|_, _| {
            Local::free(Some(match label {
                "φ top-4" => Box::new(selection(&rig.phi, &top_k(&mean, 4))) as Box<dyn Basis>,
                _ => Box::new(Combo {
                    phi: rig.phi.clone(),
                    w: principal_w(&a_full, 2, &w_all).0,
                }),
            }))
        });
        for k in 0..n {
            let v: Vec<f64> = row.full.iter().map(|s| s.local[k]).collect();
            let (m, sd) = mean_std(&v);
            let (pn, pc) = max_corr(&row, k);
            println!(
                "| {label} | {k} | {m:+.4} ± {sd:.4} | {:.2} | {pn} {pc:+.2} |",
                jack_ratio(&row, k, NCAM)
            );
        }
    }
    let _ = compact;

    // ── Q1: alternatives to a camera property ──
    q1(rig, &route, &phi_row, &full)?;

    // ── Q3: lens-side terms ──
    let lens = q3(rig, &route, &phi_row);
    Ok(RigFields { full, lens })
}

/// The φ-field of each camera fitted next to an extra term.
pub(crate) fn with_term(rig: &Rig, name: &str, extra: &dyn Fn() -> Box<dyn Basis>) -> Row {
    eval_local(rig, name, &|_, _| {
        Local::free(Some(Box::new(Concat(vec![
            Box::new(rig.phi.clone()),
            extra(),
        ]))))
    })
}

/// Share of the φ-field's energy that moves when the term is added.
fn share(rig: &Rig, base: &[CamSol], with: &Row) -> f64 {
    let n = rig.phi.n();
    let (mut num, mut den) = (0.0, 0.0);
    for (c, (a, b)) in base.iter().zip(&with.full).enumerate() {
        let w = metric(rig, &[c]);
        let a = a.local.rows(0, n).into_owned();
        let d = &a - b.local.rows(0, n);
        num += d.dot(&(&w * &d));
        den += a.dot(&(&w * &a));
    }
    num / den
}

fn q1(rig: &Rig, route: &Row, phi_row: &Row, full: &[CamSol]) -> Result<()> {
    let r0 = route.pooled()[0];
    let size = rig.cams[0].size;
    let (w, h) = (f64::from(size[0]), f64::from(size[1]));
    let centre = [w / 2.0, h / 2.0];
    let has_blur = rig
        .cams
        .iter()
        .all(|c| c.full.iter().filter(|p| p.blur.is_some()).count() > c.full.len() / 2);
    println!("\n## Q1 alternatives: each next to the camera field (φ ≤ 6), and alone");
    println!("| model | rms / mean / median | Δ var vs route | per-camera mean |");
    println!("{}", route.line(None));
    println!("{}", phi_row.line(Some(r0)));
    // Lever arms the capture offers.
    for c in &rig.cams {
        let mut cells: BTreeMap<(usize, usize), Vec<f64>> = BTreeMap::new();
        for p in &c.full {
            cells
                .entry(((p.u[0] / 60.0) as usize, (p.u[1] / 60.0) as usize))
                .or_default()
                .push(p.depth);
        }
        let mut ratio: Vec<f64> = cells
            .values()
            .filter(|v| v.len() >= 10)
            .map(|v| {
                v.iter().fold(0.0f64, |m, x| m.max(*x)) / v.iter().fold(f64::MAX, |m, x| m.min(*x))
            })
            .collect();
        ratio.sort_by(f64::total_cmp);
        let mut by_view: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for p in &c.full {
            by_view.entry(p.view).or_default().push(p.roll);
        }
        let means: Vec<f64> = by_view
            .values()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .collect();
        let within: f64 =
            by_view.values().map(|v| mean_std(v).1).sum::<f64>() / by_view.len() as f64;
        let blurs: Vec<f64> = c.full.iter().filter_map(|p| p.blur).collect();
        let (bm, bs) = if blurs.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            mean_std(&blurs)
        };
        println!(
            "  lever {}: depth max/min per 60-px cell median {:.2}; board roll over views {:.0}°..{:.0}°, within a view sd {:.1}°; blur {:.2} ± {:.2} px",
            c.id,
            ratio.get(ratio.len() / 2).copied().unwrap_or(f64::NAN),
            means.iter().fold(f64::MAX, |m, x| m.min(*x)).to_degrees(),
            means.iter().fold(f64::MIN, |m, x| m.max(*x)).to_degrees(),
            within.to_degrees(),
            bm,
            bs
        );
    }
    let deg = move |lo, hi| CMono::degrees(lo, hi, centre, w / 2.0);
    // Complex monomials of degree lo..=hi whose rotation order passes `keep`.
    let deg_m = move |lo, hi, keep: fn(i32) -> bool| {
        let mut c = CMono::degrees(lo, hi, centre, w / 2.0);
        c.terms.retain(|&(p, q)| keep(p - q - 1));
        c
    };
    let zbar: Vec<f64> = rig
        .cams
        .iter()
        .map(|c| c.full.iter().map(|p| p.depth).sum::<f64>() / c.full.len() as f64)
        .collect();
    let zmean = zbar.iter().sum::<f64>() / zbar.len() as f64;
    let scale_mean: f64 = rig
        .cams
        .iter()
        .flat_map(|c| c.full.iter().map(|p| p.scale))
        .sum::<f64>()
        / rig.cams.iter().map(|c| c.full.len()).sum::<usize>() as f64;
    let has_gblur = rig
        .cams
        .iter()
        .all(|c| c.full.iter().all(|p| p.gblur.is_some()));
    let mut blur_med = f64::NAN;
    if has_gblur {
        let mut b: Vec<f64> = rig
            .cams
            .iter()
            .flat_map(|c| c.full.iter().filter_map(|p| p.gblur))
            .collect();
        b.sort_by(f64::total_cmp);
        blur_med = b[b.len() / 2];
    }
    type Alt = (&'static str, Box<dyn Fn() -> Box<dyn Basis>>);
    let mut alts: Vec<Alt> = vec![
        (
            "depth (z̄/z − 1) × deg 1-4",
            Box::new(move || {
                Box::new(Scaled {
                    inner: deg(1, 4),
                    f: Box::new(move |p: &LinPoint| zmean / p.depth - 1.0),
                })
            }),
        ),
        (
            "board orientation cos/sin kθ, k 1-6",
            Box::new(|| Box::new(Extra((X_ORI..X_ORI + 24).collect()))),
        ),
        (
            "pixel phase sin/cos 2πu, 2πv",
            Box::new(|| Box::new(Extra((X_PIX..X_PIX + 8).collect()))),
        ),
        (
            "circle-centre perspective bias (R² free)",
            Box::new(|| Box::new(Extra(vec![X_ELL]))),
        ),
        (
            "local scale (s/s̄ − 1) × deg 0-2",
            Box::new(move || {
                Box::new(Scaled {
                    inner: deg(0, 2),
                    f: Box::new(move |p: &LinPoint| p.scale / scale_mean - 1.0),
                })
            }),
        ),
    ];
    let depth_f = move |p: &LinPoint| zmean / p.depth - 1.0;
    alts.push((
        "depth × radial only (m 0, deg 1-4)",
        Box::new(move || {
            Box::new(Scaled {
                inner: deg_m(1, 4, |m| m == 0),
                f: Box::new(depth_f),
            })
        }),
    ));
    alts.push((
        "depth × m ±1 only (deg 1-4)",
        Box::new(move || {
            Box::new(Scaled {
                inner: deg_m(1, 4, |m| m.abs() == 1),
                f: Box::new(depth_f),
            })
        }),
    ));
    alts.push((
        "depth × |m| ≥ 2 only (deg 1-4)",
        Box::new(move || {
            Box::new(Scaled {
                inner: deg_m(1, 4, |m| m.abs() >= 2),
                f: Box::new(depth_f),
            })
        }),
    ));
    if has_gblur {
        let mut add_blur = |name: &'static str, f: fn(&LinPoint, f64) -> f64| {
            alts.insert(
                0,
                (
                    name,
                    Box::new(move || {
                        Box::new(Scaled {
                            inner: deg(0, 3),
                            f: Box::new(move |p: &LinPoint| f(p, blur_med)),
                        })
                    }),
                ),
            );
        };
        if has_blur {
            add_blur(
                "blur residual (measured − geometric) × deg 0-3",
                |p, _| match (p.blur, p.gblur) {
                    (Some(b), Some(g)) => b - g,
                    _ => 0.0,
                },
            );
        }
        add_blur("geometric blur (σ_geo − median) × deg 0-3", |p, m| {
            p.gblur.map_or(0.0, |g| g - m)
        });
        if has_blur {
            add_blur("blur (σ − median) × deg 0-3", |p, m| {
                p.blur.map_or(0.0, |b| b - m)
            });
        }
        if has_blur {
            alts.push((
                "blur × deg 0-3 + depth × deg 1-4",
                Box::new(move || {
                    Box::new(Concat(vec![
                        Box::new(Scaled {
                            inner: deg(0, 3),
                            f: Box::new(move |p: &LinPoint| p.blur.map_or(0.0, |b| b - blur_med)),
                        }),
                        Box::new(Scaled {
                            inner: deg(1, 4),
                            f: Box::new(depth_f),
                        }),
                    ]))
                }),
            ));
        }
        alts.push((
            "geometric blur × deg 0-3 + depth × deg 1-4",
            Box::new(move || {
                Box::new(Concat(vec![
                    Box::new(Scaled {
                        inner: deg(0, 3),
                        f: Box::new(move |p: &LinPoint| p.gblur.map_or(0.0, |b| b - blur_med)),
                    }),
                    Box::new(Scaled {
                        inner: deg(1, 4),
                        f: Box::new(depth_f),
                    }),
                ]))
            }),
        ));
    }
    println!(
        "| alternative | with field: rms / mean / median, Δ var vs field | share of the field it moves | alone: rms, Δ var vs route |"
    );
    let fr = phi_row.pooled()[0];
    let mut all_alts: Vec<Box<dyn Fn() -> Box<dyn Basis>>> = Vec::new();
    for (name, make) in &alts {
        let with = with_term(rig, name, make.as_ref());
        let alone = eval_local(rig, name, &|_, _| Local::free(Some(make())));
        let s = with.pooled();
        let a = alone.pooled()[0];
        println!(
            "| {name} | {:.4} / {:.4} / {:.4}, {:+.1} % | {:.1} % | {:.4}, {:+.1} % |",
            s[0],
            s[1],
            s[2],
            100.0 * (s[0] * s[0] - fr * fr) / (fr * fr),
            100.0 * share(rig, full, &with),
            a,
            100.0 * (a * a - r0 * r0) / (r0 * r0)
        );
        if *name == "circle-centre perspective bias (R² free)" {
            let v: Vec<f64> = alone.full.iter().map(|s| s.local[0]).collect();
            let vw: Vec<f64> = with.full.iter().map(|s| s.local[rig.phi.n()]).collect();
            println!(
                "|   coefficient R² (mm²) per camera: alone {} ; with field {} |",
                v.iter()
                    .map(|x| format!("{x:.2}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                vw.iter()
                    .map(|x| format!("{x:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        let _ = &mut all_alts;
    }
    if has_blur {
        blur_subsets(rig, full, blur_med);
    }
    board_frame(rig, route, phi_row, full);
    Ok(())
}

/// A target shown on a display is fixed in the board frame, a camera term
/// in the image frame. The six cameras see one board: a board-frame term is
/// shared by them, an image-frame field is per camera. How far the capture
/// separates the two frames: how much a board point moves over the image
/// between views, and how well image position predicts board position.
fn board_frame(rig: &Rig, route: &Row, phi_row: &Row, full: &[CamSol]) {
    let r0 = route.pooled()[0];
    let fr = phi_row.pooled()[0];
    println!("\n## Q1 board frame (display) against image frame (camera)");
    for c in &rig.cams {
        let mut by_pt: BTreeMap<(i64, i64), Vec<[f64; 2]>> = BTreeMap::new();
        for p in &c.full {
            by_pt
                .entry(BoardPoints::key(p.board))
                .or_default()
                .push(p.u);
        }
        let mut sd: Vec<f64> = by_pt
            .values()
            .filter(|v| v.len() >= 3)
            .map(|v| {
                let (_, a) = mean_std(&v.iter().map(|x| x[0]).collect::<Vec<_>>());
                let (_, b) = mean_std(&v.iter().map(|x| x[1]).collect::<Vec<_>>());
                a.hypot(b)
            })
            .collect();
        sd.sort_by(f64::total_cmp);
        // Share of the image position explained by a quadratic in board
        // position (pooled over views).
        let rows: Vec<[f64; 6]> = c
            .full
            .iter()
            .map(|p| {
                let (x, y) = (p.board[0] * 10.0, p.board[1] * 10.0);
                [1.0, x, y, x * x, x * y, y * y]
            })
            .collect();
        let a = DMatrix::from_fn(rows.len(), 6, |i, j| rows[i][j]);
        let mut r2 = [0.0; 2];
        for (d, r) in r2.iter_mut().enumerate() {
            let y = DVector::from_iterator(c.full.len(), c.full.iter().map(|p| p.u[d]));
            if let Some(ch) = (a.transpose() * &a).cholesky() {
                let fit = &a * ch.solve(&(a.transpose() * &y));
                let m = y.mean();
                *r = 1.0
                    - (&y - fit).norm_squared() / y.iter().map(|v| (v - m).powi(2)).sum::<f64>();
            }
        }
        println!(
            "  lever {}: a board point's image position over views, sd median {:.0} px (10th–90th {:.0}–{:.0}); image position explained by board position: u {:.0} %, v {:.0} %",
            c.id,
            sd.get(sd.len() / 2).copied().unwrap_or(f64::NAN),
            sd.get(sd.len() / 10).copied().unwrap_or(f64::NAN),
            sd.get(9 * sd.len() / 10).copied().unwrap_or(f64::NAN),
            100.0 * r2[0],
            100.0 * r2[1]
        );
    }
    let all: Vec<&LinPoint> = rig.cams.iter().flat_map(|c| c.full.iter()).collect();
    let n = all.len() as f64;
    let centre = [
        all.iter().map(|p| p.board[0]).sum::<f64>() / n,
        all.iter().map(|p| p.board[1]).sum::<f64>() / n,
    ];
    let mut count: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for p in &all {
        *count.entry(BoardPoints::key(p.board)).or_default() += 1;
    }
    let index: BTreeMap<(i64, i64), usize> = count
        .iter()
        .filter(|(_, c)| **c >= 8)
        .enumerate()
        .map(|(i, (k, _))| (*k, i))
        .collect();
    println!(
        "  {} board points with ≥ 8 observations over the rig (per-point offsets)",
        index.len()
    );
    let poly = |deg: usize| BoardPoly {
        terms: model::monomials(1, deg),
        centre,
        half: 0.06,
    };
    println!(
        "| model | rms / mean / median | Δ var vs route | Δ var vs field | share of the field it moves | per-camera mean |"
    );
    let line = |row: &Row, moved: Option<f64>| {
        let s = row.pooled();
        let per: Vec<String> = row
            .held
            .iter()
            .map(|h| format!("{:.3}", linfield::stats(h)[1]))
            .collect();
        println!(
            "| {} | {:.4} / {:.4} / {:.4} | {:+.1} % | {:+.1} % | {} | {} |",
            row.name,
            s[0],
            s[1],
            s[2],
            100.0 * (s[0] * s[0] - r0 * r0) / (r0 * r0),
            100.0 * (s[0] * s[0] - fr * fr) / (fr * fr),
            moved.map_or("—".into(), |m| format!("{:.1} %", 100.0 * m)),
            per.join(" | ")
        );
    };
    for deg in [2usize, 4, 6] {
        let (row, _) = eval_shared(
            rig,
            &format!("board in-plane poly 1-{deg}, shared"),
            &|_| Box::new(poly(deg)),
        );
        line(&row, None);
        let (row, _) = eval_shared_local(
            rig,
            &format!("field φ≤6 per camera + board poly 1-{deg} shared"),
            Some(&rig.phi),
            &|_| Box::new(poly(deg)),
        );
        line(&row, Some(share(rig, full, &row)));
    }
    let (row, _) = eval_shared(rig, "board per-point offsets, shared", &|_| {
        Box::new(BoardPoints {
            index: index.clone(),
        })
    });
    line(&row, None);
    let (row, _) = eval_shared_local(
        rig,
        "field φ≤6 per camera + board per-point offsets shared",
        Some(&rig.phi),
        &|_| {
            Box::new(BoardPoints {
                index: index.clone(),
            })
        },
    );
    line(&row, Some(share(rig, full, &row)));
}

/// The φ-field fitted on halves of each camera's markers: sharp against
/// blurred (relative to the blur typical of the position), and a parity
/// split as the control.
fn blur_subsets(rig: &Rig, full: &[CamSol], _median: f64) {
    println!(
        "\n## Q1a blur: the field of sharp vs blurred markers at the same image positions (full data, camera free)"
    );
    println!(
        "| camera | field RMS | ‖F_sharp − F_blurred‖ | ‖F_even − F_odd‖ (control) | blur-dependent part / field | relative-blur split: same |"
    );
    let nphi = rig.phi.n();
    let spec = Spec {
        camera_free: true,
        local: Some(&rig.phi),
        penalty: None,
        ortho: false,
        shared: None,
        held: None,
    };
    let (mut num, mut den) = (0.0, 0.0);
    for (c, base) in rig.cams.iter().zip(full) {
        let (w, h) = (f64::from(c.size[0]), f64::from(c.size[1]));
        // Quadratic blur surface over the image.
        let pts: Vec<&LinPoint> = c.full.iter().filter(|p| p.blur.is_some()).collect();
        let row = |p: &LinPoint| {
            let (x, y) = (p.u[0] / w - 0.5, p.u[1] / h - 0.5);
            [1.0, x, y, x * x, x * y, y * y]
        };
        let a = DMatrix::from_fn(pts.len(), 6, |i, j| row(pts[i])[j]);
        let b = DVector::from_iterator(pts.len(), pts.iter().map(|p| p.blur.unwrap_or(0.0)));
        let coef = (a.transpose() * &a)
            .cholesky()
            .map(|ch| ch.solve(&(a.transpose() * b)))
            .unwrap_or_else(|| DVector::zeros(6));
        let typ = |p: &LinPoint| (0..6).map(|j| row(p)[j] * coef[j]).sum::<f64>();
        let mut rel: Vec<f64> = pts.iter().map(|p| p.blur.unwrap_or(0.0) - typ(p)).collect();
        rel.sort_by(f64::total_cmp);
        let med = rel[rel.len() / 2];
        let split = |keep: &dyn Fn(usize, &LinPoint) -> bool| -> Vec<LinPoint> {
            c.full
                .iter()
                .enumerate()
                .filter(|(i, p)| p.blur.is_some() && keep(*i, p))
                .map(|(_, p)| p.clone())
                .collect()
        };
        let sharp = split(&|_, p| p.blur.unwrap_or(0.0) - typ(p) <= med);
        let blurred = split(&|_, p| p.blur.unwrap_or(0.0) - typ(p) > med);
        let even = split(&|i, _| i % 2 == 0);
        let odd = split(&|i, _| i % 2 == 1);
        let f = |pts: &[LinPoint]| linfield::fit(&[pts], &spec).cams[0].local.clone();
        let (fs, fb, fe, fo) = (f(&sharp), f(&blurred), f(&even), f(&odd));
        let d = (&fs - &fb).norm_squared();
        let dc = (&fe - &fo).norm_squared();
        let fe2 = base.local.norm_squared();
        num += (d - dc).max(0.0);
        den += fe2;
        println!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.2} | cos(F_sharp, F_blurred) {:+.2}, control {:+.2} |",
            c.id,
            fe2.sqrt(),
            d.sqrt(),
            dc.sqrt(),
            ((d - dc).max(0.0) / fe2).sqrt(),
            cos_sim(&fs, &fb),
            cos_sim(&fe, &fo)
        );
        let _ = nphi;
    }
    println!(
        "pooled: noise-corrected ‖F_sharp − F_blurred‖ / ‖F‖ = {:.2}",
        (num / den).sqrt()
    );
}

/// Lens-side terms: alone (no field), with stability.
fn q3(rig: &Rig, route: &Row, phi_row: &Row) -> BTreeMap<String, Vec<DVector<f64>>> {
    let mut out = BTreeMap::new();
    let r0 = route.pooled()[0];
    let fr = phi_row.pooled()[0];
    println!("\n## Q3 lens-side terms (no field; Δ var vs route; share of the field's gain)");
    println!(
        "| term | cols | rms / mean / median | Δ var vs route | of the field's gain | coef across cameras mean ± sd | jackknife sd / σ | largest |corr| |"
    );
    // (name, [(lens term, part: None both, Some(true) real, Some(false) imaginary)])
    type LensSet<'a> = (&'a str, Vec<(&'a str, Option<bool>)>);
    let sets: Vec<LensSet> = vec![
        ("skew (Im w̄)", vec![("w̄ (anamorphic)", Some(false))]),
        ("w̄² (m −3)", vec![("w̄² m-3", None)]),
        ("NAT B5: c̄w³ + 3c ww̄²", vec![("NAT B5", None)]),
        (
            "w³ + ww̄² free (m ±2)",
            vec![("w³ m+2", None), ("ww̄² m-2", None)],
        ),
        ("w̄³ (m −4, not NAT)", vec![("w̄³ m-4", None)]),
        ("radial twist Im w²w̄", vec![("w²w̄ m0", Some(false))]),
        ("NAT A5: deg-4 m ±1 linked", vec![("NAT A5", None)]),
        ("w³w̄ (m +1)", vec![("w³w̄ m+1", None)]),
        ("w²w̄² (thin prism r⁴)", vec![("w²w̄² m-1", None)]),
        (
            "w̄² + w̄³ (m −3, −4)",
            vec![("w̄² m-3", None), ("w̄³ m-4", None)],
        ),
        (
            "NAT set: skew + B5 + A5",
            vec![
                ("w̄ (anamorphic)", Some(false)),
                ("NAT B5", None),
                ("NAT A5", None),
            ],
        ),
        (
            "all lens deg ≤ 4 outside the model",
            vec![
                ("w̄ (anamorphic)", Some(false)),
                ("w̄² m-3", None),
                ("w³ m+2", None),
                ("ww̄² m-2", None),
                ("w̄³ m-4", None),
                ("w²w̄ m0", Some(false)),
                ("w³w̄ m+1", None),
                ("w⁴ m+3", None),
                ("ww̄³ m-3", None),
                ("w̄⁴ m-5", None),
            ],
        ),
    ];
    for (name, cols) in &sets {
        let idx = lens_cols(cols);
        let n = idx.len();
        let row = eval_local(rig, name, &|_, _| {
            Local::free(Some(Box::new(Extra(idx.clone()))))
        });
        let s = row.pooled();
        let gain = (r0 * r0 - s[0] * s[0]) / (r0 * r0 - fr * fr);
        let coefs: Vec<String> = (0..n.min(5))
            .map(|k| {
                let v: Vec<f64> = row.full.iter().map(|x| x.local[k]).collect();
                let (m, sd) = mean_std(&v);
                format!("{m:+.2e}±{sd:.1e}")
            })
            .collect();
        let jk: Vec<String> = (0..n.min(5))
            .map(|k| format!("{:.1}", jack_ratio(&row, k, NCAM)))
            .collect();
        let cr: Vec<String> = (0..n.min(5))
            .map(|k| {
                let (p, c) = max_corr(&row, k);
                format!("{p} {c:+.2}")
            })
            .collect();
        println!(
            "| {name} | {n} | {:.4} / {:.4} / {:.4} | {:+.1} % | {:.0} % | {} | {} | {} |",
            s[0],
            s[1],
            s[2],
            100.0 * (s[0] * s[0] - r0 * r0) / (r0 * r0),
            100.0 * gain,
            coefs.join(" "),
            jk.join(" "),
            cr.join(", ")
        );
        out.insert(
            name.to_string(),
            row.full.iter().map(|x| x.local.clone()).collect(),
        );
    }
    // The image-frame lens terms next to the defocus- and depth-coupled
    // terms: does the compact term survive once those are modelled?
    if rig
        .cams
        .iter()
        .all(|c| c.full.iter().all(|p| p.gblur.is_some()))
    {
        let size = rig.cams[0].size;
        let (w, h) = (f64::from(size[0]), f64::from(size[1]));
        let centre = [w / 2.0, h / 2.0];
        let mut g: Vec<f64> = rig
            .cams
            .iter()
            .flat_map(|c| c.full.iter().filter_map(|p| p.gblur))
            .collect();
        g.sort_by(f64::total_cmp);
        let gmed = g[g.len() / 2];
        let zmean = rig
            .cams
            .iter()
            .flat_map(|c| c.full.iter().map(|p| p.depth))
            .sum::<f64>()
            / rig.cams.iter().map(|c| c.full.len()).sum::<usize>() as f64;
        let coupled = move || -> Vec<Box<dyn Basis>> {
            vec![
                Box::new(Scaled {
                    inner: CMono::degrees(0, 3, centre, w / 2.0),
                    f: Box::new(move |p: &LinPoint| p.gblur.map_or(0.0, |b| b - gmed)),
                }),
                Box::new(Scaled {
                    inner: CMono::degrees(1, 4, centre, w / 2.0),
                    f: Box::new(move |p: &LinPoint| zmean / p.depth - 1.0),
                }),
            ]
        };
        let base = eval_local(rig, "defocus + depth", &|_, _| {
            Local::free(Some(Box::new(Concat(coupled()))))
        });
        let b0 = base.pooled()[0];
        let idx = lens_cols(&[("w̄² m-3", None), ("w̄³ m-4", None)]);
        let with = eval_local(rig, "w̄² + w̄³ + defocus + depth", &|_, _| {
            let mut v: Vec<Box<dyn Basis>> = vec![Box::new(Extra(idx.clone()))];
            v.extend(coupled());
            Local::free(Some(Box::new(Concat(v))))
        });
        let wf = with.pooled()[0];
        let coefs: Vec<String> = (0..4)
            .map(|k| {
                let v: Vec<f64> = with.full.iter().map(|x| x.local[k]).collect();
                let (m, sd) = mean_std(&v);
                format!("{m:+.2e}±{sd:.1e}")
            })
            .collect();
        let alone: Vec<String> = (0..4)
            .map(|k| {
                let v: Vec<f64> = out
                    .get("w̄² + w̄³ (m −3, −4)")
                    .map(|r: &Vec<DVector<f64>>| r.iter().map(|x| x[k]).collect())
                    .unwrap_or_default();
                let (m, sd) = mean_std(&v);
                format!("{m:+.2e}±{sd:.1e}")
            })
            .collect();
        println!(
            "w̄² + w̄³ next to defocus + depth: {b0:.4} → {wf:.4} px ({:+.1} % var); coefficients {} (alone {})",
            100.0 * (wf * wf - b0 * b0) / (b0 * b0),
            coefs.join(" "),
            alone.join(" ")
        );
        let _ = h;
    }
    out
}

/// The two rigs' fields on the cells both cover.
fn cross_rig(rigs: &[Rig], fields: &[RigFields]) {
    println!(
        "\n# Cross-rig: the camera fields of {} against {} (same units by index)",
        rigs[0].name, rigs[1].name
    );
    // A field on the GRID cells of `mask`, with both typical cameras'
    // directions there projected out.
    let pos_a: BTreeMap<usize, usize> = rigs[0]
        .grid_idx
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i))
        .collect();
    let pos_b: BTreeMap<usize, usize> = rigs[1]
        .grid_idx
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i))
        .collect();
    let pair = |i: usize, j: usize| -> Option<f64> {
        let cells: Vec<usize> = (0..GRID[0] * GRID[1])
            .filter(|&k| rigs[0].cams[i].cover[k] && rigs[1].cams[j].cover[k])
            .collect();
        if cells.len() < 30 {
            return None;
        }
        let nc = cells.len();
        let mut cmat = DMatrix::<f64>::zeros(2 * nc, 2 * NCF);
        let mut va = DVector::<f64>::zeros(2 * nc);
        let mut vb = DVector::<f64>::zeros(2 * nc);
        let mut ba = vec![[0.0; 2]; rigs[0].phi.n()];
        let mut bb = vec![[0.0; 2]; rigs[1].phi.n()];
        for (r, k) in cells.iter().enumerate() {
            let ga = &rigs[0].grid[pos_a[k]];
            let gb = &rigs[1].grid[pos_b[k]];
            for c in 0..NCF {
                for d in 0..2 {
                    cmat[(2 * r + d, c)] = ga.1[c][d];
                    cmat[(2 * r + d, NCF + c)] = gb.1[c][d];
                }
            }
            rigs[0].phi.value(ga.0, &ga.1, &mut ba);
            rigs[1].phi.value(gb.0, &gb.1, &mut bb);
            for d in 0..2 {
                va[2 * r + d] = (0..ba.len())
                    .map(|t| fields[0].full[i].local[t] * ba[t][d])
                    .sum();
                vb[2 * r + d] = (0..bb.len())
                    .map(|t| fields[1].full[j].local[t] * bb[t][d])
                    .sum();
            }
        }
        let svd = cmat.svd(true, false);
        let u = svd.u.expect("u");
        let smax = svd.singular_values.max();
        for k in 0..svd.singular_values.len() {
            if svd.singular_values[k] > 1e-6 * smax {
                let q = u.column(k);
                va -= q * q.dot(&va);
                vb -= q * q.dot(&vb);
            }
        }
        Some(cos_sim(&va, &vb))
    };
    let n = rigs[0].cams.len().min(rigs[1].cams.len());
    println!(
        "camera i of {} (rows) against camera j of {} (columns), over the cells both cover:",
        rigs[0].name, rigs[1].name
    );
    let (mut same, mut other) = (Vec::new(), Vec::new());
    for i in 0..n {
        let row: Vec<String> = (0..n)
            .map(|j| match pair(i, j) {
                Some(c) => {
                    if i == j {
                        same.push(c);
                    } else {
                        other.push(c);
                    }
                    format!("{c:+.2}")
                }
                None => "  .  ".into(),
            })
            .collect();
        println!("  cam{i} {}", row.join(" "));
    }
    let (sm, ss) = mean_std(&same);
    let (om, os) = mean_std(&other);
    println!(
        "same unit {sm:+.2} ± {ss:.2} (n {}), different units {om:+.2} ± {os:.2} (n {})",
        same.len(),
        other.len()
    );
    for (name, cols) in [("w̄² (m −3)", 1usize), ("w̄³ (m −4, not NAT)", 0usize)] {
        let (Some(a), Some(b)) = (fields[0].lens.get(name), fields[1].lens.get(name)) else {
            continue;
        };
        let va: Vec<f64> = a.iter().map(|v| v[cols]).collect();
        let vb: Vec<f64> = b.iter().map(|v| v[cols]).collect();
        println!(
            "{name} component {cols} per unit: {} | {} ; corr across units {:+.2}",
            va.iter()
                .map(|x| format!("{x:+.2e}"))
                .collect::<Vec<_>>()
                .join(" "),
            vb.iter()
                .map(|x| format!("{x:+.2e}"))
                .collect::<Vec<_>>()
                .join(" "),
            crate::pearson(&va, &vb)
        );
    }
}
