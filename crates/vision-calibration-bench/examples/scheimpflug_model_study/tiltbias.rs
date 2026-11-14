//! `--parts tiltbias`: does the distortion-derived tilt absorb the
//! defocus- and depth-coupled displacement of the residual?
//!
//! The route's camera is refitted jointly with a displacement law (a
//! per-point covariate — signed defocus, unsigned blur, depth — times
//! complex monomials of the image position), every pose free, by
//! Gauss-Newton iterated to convergence: each iteration re-linearizes the
//! model at the current camera and poses, re-fits the covariates (the
//! geometric blur model and the plane of sharp focus in the current `K'`
//! frame, both to the detector's blur, never to the residual) and evaluates
//! every term at the model's current projection.
//!
//! Camera parameters' pixel directions depend on the image position only,
//! the poses' on depth too; so the route's camera absorbs, of an
//! unmodelled displacement `d`, essentially its image-fixed average
//! `E[d | u]` over the capture's depths. A law ties that average to the
//! depth-varying part, so a joint fit's camera change is a statement about
//! the law as much as about the data: the laws below differ exactly in what
//! they assume of the image-fixed part (`Var` keeps none of it).
//!
//! Then: the focus-against-distortion comparison of the focus note with
//! the refitted tilt, and the reverse experiment (the tilt held at the
//! focus-derived value, the rest refitted with and without the law).

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use nalgebra::{DMatrix, DVector, Point3, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{Iso3, NoMeta, View};

use crate::field_study::{blur_model, blur_predict_at};
use crate::focus::{BlurObs, PofFit, fit_pof, tilt_from_pof, tilt_rotation};
use crate::focus_study::{CamResult, cross_camera, implied, real_obs};
use crate::linfield::{self, Basis, CMono, LinPoint, NCAM, Spec};
use crate::model::{self, Free, GenCfg, GenParams};
use crate::{CamCtx, CameraData, gen_init};

/// Detector blur by observed pixel (`blur::key`).
pub type BlurMap = BTreeMap<(i64, i64), f64>;

/// Gauss-Newton iterations of one inner solve.
const MAX_ITERS: usize = 40;
/// Ridge on the law's coefficients, relative to unit-weight pixel
/// residuals: a prior σ of 1 on every coefficient at a residual σ of
/// 0.1 px. A coefficient is the displacement (px) per px of the covariate
/// at the image's side edge, so the prior binds only the near-gauge
/// combinations (`a·κ/z` is a pose translation), not the term.
const COEF_PRIOR: f64 = 0.01;
/// Convergence: every free camera step below this fraction of its σ.
const STEP_TOL: f64 = 1e-3;
/// A stall (no decrease along the step) counts as converged when the
/// step was below this fraction of σ.
const STALL_TOL: f64 = 0.05;
/// A law whose stated σ(tilt_x) exceeds the plain fit's by this factor
/// does not leave the tilt identified: its tilt is reported, not used.
const IDENT_RATIO: f64 = 3.0;
/// Degree of the image-position regression that defines the image-fixed
/// part of a covariate (`Var` law).
const FIX_DEG: usize = 4;

// ───────────────────────── laws ─────────────────────────

/// A per-point covariate of a displacement law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cov {
    /// Geometric blur (blur² modelled from image position and inverse
    /// depth) minus its median, px.
    Geo,
    /// `z̄/z − 1`.
    Depth,
    /// Signed defocus `g = κ·(1/z − π·(x, y, 1))` about the fitted plane of
    /// sharp focus, px.
    Signed,
    /// `|g|` minus its median, px.
    Abs,
    /// `|g|`, px (no centring: a different image-fixed part).
    AbsRaw,
    /// `g` minus its regression on the image position.
    SignedVar,
    /// `|g|` minus its regression on the image position.
    AbsVar,
    /// `1`: an image-fixed field (synthetic control).
    One,
}

/// A displacement law: covariates times complex monomials of the image
/// position (degree ranges), coefficients per camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// `g × deg 0–3`: a centroid shift proportional to signed defocus.
    Signed,
    /// `g × deg 0–3 + |g| × deg 0–3`: the best held-out row of the focus
    /// note (linearized at the route).
    SignedAbs,
    /// `g × deg 0–1 + |g| × deg 0–3`: the centroid shift of a pupil whose
    /// centroid offset is affine in the image position, and the unsigned
    /// blur.
    Signed1Abs,
    /// `(|g| − median) × deg 0–3`.
    Abs,
    /// `|g| × deg 0–3`.
    AbsRaw,
    /// Geometric blur × deg 0–3 + depth × deg 1–4: the parametric note's.
    GeoDepth,
    /// Depth × deg 1–4.
    Depth,
    /// The depth-varying parts of `g` and `|g|` only (each minus its
    /// regression on the image position) × deg 0–3: no image-fixed part.
    Var,
    /// An image-fixed field, deg 0–4 (synthetic control only).
    Image,
}

impl Law {
    pub const ALL: [Law; 8] = [
        Law::Signed,
        Law::SignedAbs,
        Law::Signed1Abs,
        Law::Abs,
        Law::AbsRaw,
        Law::GeoDepth,
        Law::Depth,
        Law::Var,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Law::Signed => "g × deg 0-3",
            Law::SignedAbs => "g + |g| × deg 0-3",
            Law::Signed1Abs => "g × deg 0-1 + |g| × deg 0-3",
            Law::Abs => "(|g| − med) × deg 0-3",
            Law::AbsRaw => "|g| × deg 0-3",
            Law::GeoDepth => "geo blur 0-3 + depth 1-4",
            Law::Depth => "depth × deg 1-4",
            Law::Var => "g, |g| depth-varying × deg 0-3",
            Law::Image => "image field deg 0-4",
        }
    }

    fn parts(self) -> &'static [(Cov, usize, usize)] {
        match self {
            Law::Signed => &[(Cov::Signed, 0, 3)],
            Law::SignedAbs => &[(Cov::Signed, 0, 3), (Cov::Abs, 0, 3)],
            Law::Signed1Abs => &[(Cov::Signed, 0, 1), (Cov::Abs, 0, 3)],
            Law::Abs => &[(Cov::Abs, 0, 3)],
            Law::AbsRaw => &[(Cov::AbsRaw, 0, 3)],
            Law::GeoDepth => &[(Cov::Geo, 0, 3), (Cov::Depth, 1, 4)],
            Law::Depth => &[(Cov::Depth, 1, 4)],
            Law::Var => &[(Cov::SignedVar, 0, 3), (Cov::AbsVar, 0, 3)],
            Law::Image => &[(Cov::One, 0, 4)],
        }
    }
}

/// The constants a law's covariates are centred with, fixed per camera so
/// every fit of that camera (full data and folds) uses one basis.
#[derive(Debug, Clone, Copy, Default)]
pub struct Consts {
    pub gblur_med: f64,
    pub g_abs_med: f64,
    pub zbar: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

impl Consts {
    /// From annotated points ([`Covariates::annotate`]).
    pub fn of(pts: &[LinPoint]) -> Self {
        Self {
            gblur_med: median(pts.iter().filter_map(|p| p.gblur).collect()),
            g_abs_med: median(pts.iter().filter_map(|p| p.defocus.map(f64::abs)).collect()),
            zbar: pts.iter().map(|p| p.depth).sum::<f64>() / pts.len().max(1) as f64,
        }
    }
}

/// The covariate values of one point.
#[derive(Debug, Clone, Copy, Default)]
pub struct PointCov {
    pub gblur: f64,
    pub depth: f64,
    pub g: f64,
    /// The image-position regressions of `g` and `|g|` at the point.
    pub g_fix: f64,
    pub a_fix: f64,
}

/// A [`Law`]'s pixel-displacement columns: the monomials about the image
/// centre (scale: half the width), two real columns per term.
pub struct LawBasis {
    parts: Vec<(Cov, CMono)>,
    consts: Consts,
}

impl LawBasis {
    pub fn new(law: Law, size: [u32; 2], consts: Consts) -> Self {
        let (w, h) = (f64::from(size[0]), f64::from(size[1]));
        let parts = law
            .parts()
            .iter()
            .map(|&(c, lo, hi)| (c, CMono::degrees(lo, hi, [w / 2.0, h / 2.0], w / 2.0)))
            .collect();
        Self { parts, consts }
    }

    fn covariate(&self, c: Cov, v: &PointCov) -> f64 {
        let k = &self.consts;
        match c {
            Cov::Geo => v.gblur - k.gblur_med,
            Cov::Depth => k.zbar / v.depth - 1.0,
            Cov::Signed => v.g,
            Cov::Abs => v.g.abs() - k.g_abs_med,
            Cov::AbsRaw => v.g.abs(),
            Cov::SignedVar => v.g - v.g_fix,
            Cov::AbsVar => v.g.abs() - v.a_fix,
            Cov::One => 1.0,
        }
    }

    /// The columns at pixel `u` with covariates `v`.
    pub fn eval_at(&self, u: [f64; 2], v: &PointCov, out: &mut [[f64; 2]]) {
        let mut at = 0;
        for (c, m) in &self.parts {
            let n = 2 * m.terms.len();
            m.value(u, &mut out[at..at + n]);
            let s = self.covariate(*c, v);
            out[at..at + n]
                .iter_mut()
                .for_each(|o| *o = [o[0] * s, o[1] * s]);
            at += n;
        }
    }

    /// The degree of each column's monomial.
    pub fn col_degrees(&self) -> Vec<usize> {
        self.parts
            .iter()
            .flat_map(|(_, m)| m.terms.iter().flat_map(|&(p, q)| [(p + q) as usize; 2]))
            .collect()
    }

    /// The displacement `Σ coef·column` at `u`.
    pub fn displacement(&self, u: [f64; 2], v: &PointCov, coef: &DVector<f64>) -> [f64; 2] {
        let mut buf = vec![[0.0; 2]; self.len()];
        self.eval_at(u, v, &mut buf);
        buf.iter()
            .zip(coef.iter())
            .fold([0.0; 2], |s, (c, a)| [s[0] + a * c[0], s[1] + a * c[1]])
    }
}

impl Basis for LawBasis {
    fn len(&self) -> usize {
        self.parts.iter().map(|(_, m)| 2 * m.terms.len()).sum()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        self.eval_at(p.u, &PointCov::of(p), out);
    }
}

impl PointCov {
    /// The covariates a point carries (`LinPoint::extra`, set by
    /// [`PointCov::attach`]).
    pub fn of(p: &LinPoint) -> Self {
        let e = |i: usize| p.extra.get(i).copied().unwrap_or([0.0; 2]);
        Self {
            gblur: e(0)[0],
            depth: e(0)[1],
            g: e(1)[0],
            g_fix: e(1)[1],
            a_fix: e(2)[0],
        }
    }

    /// Store the covariates on a point (and mirror `gblur`, `defocus`).
    pub fn attach(&self, p: &mut LinPoint) {
        p.gblur = Some(self.gblur);
        p.defocus = Some(self.g);
        p.extra = vec![
            [self.gblur, self.depth],
            [self.g, self.g_fix],
            [self.a_fix, 0.0],
        ];
    }
}

// ───────────────────────── covariates ─────────────────────────

/// Real monomials `x^i·y^j`, `i + j ≤ FIX_DEG`, of the half-size-normalized
/// pixel: the regression that defines a covariate's image-fixed part.
fn fix_row(u: [f64; 2], size: [u32; 2]) -> Vec<f64> {
    let (w, h) = (f64::from(size[0]) / 2.0, f64::from(size[1]) / 2.0);
    let (x, y) = ((u[0] - w) / w, (u[1] - h) / h);
    let mut r = Vec::new();
    for d in 0..=FIX_DEG as i32 {
        for i in 0..=d {
            r.push(x.powi(i) * y.powi(d - i));
        }
    }
    r
}

fn regress(rows: &[Vec<f64>], y: &[f64]) -> Option<DVector<f64>> {
    let n = rows.first()?.len();
    let a = DMatrix::from_fn(rows.len(), n, |i, j| rows[i][j]);
    let b = DVector::from_column_slice(y);
    (a.transpose() * &a)
        .cholesky()
        .map(|c| c.solve(&(a.transpose() * b)))
}

/// The image-fixed parts of `g` and `|g|` at each point: their regressions
/// on the image position (degree [`FIX_DEG`]).
pub fn image_fixed_parts(
    u: &[[f64; 2]],
    g: &[f64],
    size: [u32; 2],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let rows: Vec<Vec<f64>> = u.iter().map(|x| fix_row(*x, size)).collect();
    let a: Vec<f64> = g.iter().map(|v| v.abs()).collect();
    let (cg, ca) = (regress(&rows, g)?, regress(&rows, &a)?);
    let dot = |r: &Vec<f64>, c: &DVector<f64>| r.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
    Some((
        rows.iter().map(|r| dot(r, &cg)).collect(),
        rows.iter().map(|r| dot(r, &ca)).collect(),
    ))
}

/// The covariate models of one fit, fitted to the detector's blur only
/// (never to the residual): the geometric blur, the plane of sharp focus
/// in the `K'` frame of the fit's tilt, and the image-position regressions
/// of `g` and `|g|`.
#[derive(Debug, Clone)]
pub struct Covariates {
    pub blur_coef: DVector<f64>,
    pub pof: PofFit,
    g_fix: DVector<f64>,
    a_fix: DVector<f64>,
}

/// A point's position in the frame of the plain pinhole `K'`.
fn kprime(p: &LinPoint, tilt: [f64; 2]) -> Vector3<f64> {
    tilt_rotation(tilt) * (p.pose * Point3::new(p.board[0], p.board[1], 0.0)).coords
}

impl Covariates {
    pub fn fit(pts: &[LinPoint], size: [u32; 2], tilt: [f64; 2]) -> Option<Self> {
        let (blur_coef, _) = blur_model(pts, size)?;
        let obs: Vec<BlurObs> = pts
            .iter()
            .filter_map(|p| {
                p.blur.map(|b| BlurObs {
                    x: kprime(p, tilt),
                    blur: b,
                })
            })
            .collect();
        let pof = fit_pof(&obs, None)?;
        let rows: Vec<Vec<f64>> = pts.iter().map(|p| fix_row(p.u, size)).collect();
        let g: Vec<f64> = pts.iter().map(|p| pof.g(&kprime(p, tilt))).collect();
        let a: Vec<f64> = g.iter().map(|v| v.abs()).collect();
        Some(Self {
            blur_coef,
            g_fix: regress(&rows, &g)?,
            a_fix: regress(&rows, &a)?,
            pof,
        })
    }

    /// The covariates of the lens-frame point `x` projected at `u`, for a
    /// camera with tilt `tilt`.
    pub fn point(&self, u: [f64; 2], x: &Vector3<f64>, tilt: [f64; 2], size: [u32; 2]) -> PointCov {
        let r = fix_row(u, size);
        let dot = |c: &DVector<f64>| r.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        PointCov {
            gblur: blur_predict_at(&self.blur_coef, u, x.z, size),
            depth: x.z,
            g: self.pof.g(&(tilt_rotation(tilt) * x)),
            g_fix: dot(&self.g_fix),
            a_fix: dot(&self.a_fix),
        }
    }

    /// The covariates of every point at its current geometry.
    pub fn at(&self, pts: &[LinPoint], size: [u32; 2], tilt: [f64; 2]) -> Vec<PointCov> {
        pts.iter()
            .map(|p| {
                let x = (p.pose * Point3::new(p.board[0], p.board[1], 0.0)).coords;
                self.point(p.u, &x, tilt, size)
            })
            .collect()
    }
}

/// Add to each point's camera and pose columns the derivative of the law's
/// displacement `B(u)·coef` through the monomials' dependence on the
/// projection `u` (central differences); the covariates are held per
/// point.
fn add_term_jacobian(
    pts: &mut [LinPoint],
    params: &GenParams,
    law: &LawBasis,
    coef: &DVector<f64>,
    size: [u32; 2],
) {
    if coef.iter().all(|c| *c == 0.0) {
        return;
    }
    let plain = GenCfg::base("plain");
    for p in pts {
        let v = PointCov::of(p);
        let disp = |prm: &GenParams, pose: &Iso3| {
            let u = model::project_point(&plain, prm, size, pose, p.board);
            law.displacement(u, &v, coef)
        };
        for k in 0..NCAM {
            let h = 1e-6 * linfield::cam_get(params, k).abs().max(1.0);
            let (mut a, mut b) = (params.clone(), params.clone());
            linfield::cam_add(&mut a, k, h);
            linfield::cam_add(&mut b, k, -h);
            let (da, db) = (disp(&a, &p.pose), disp(&b, &p.pose));
            p.jcam[k][0] += (da[0] - db[0]) / (2.0 * h);
            p.jcam[k][1] += (da[1] - db[1]) / (2.0 * h);
        }
        for m in 0..6 {
            let h = 1e-6;
            let mut w = [0.0; 6];
            w[m] = h;
            let da = disp(params, &(linfield::twist(w) * p.pose));
            w[m] = -h;
            let db = disp(params, &(linfield::twist(w) * p.pose));
            p.jpose[m][0] += (da[0] - db[0]) / (2.0 * h);
            p.jpose[m][1] += (da[1] - db[1]) / (2.0 * h);
        }
    }
}

// ───────────────────────── the joint fit ─────────────────────────

/// What a fit sees: the views, the detector's blur and the image size.
#[derive(Clone, Copy)]
pub struct Data<'a> {
    pub views: &'a [View<NoMeta>],
    pub blur: &'a BlurMap,
    pub size: [u32; 2],
}

/// A converged joint fit.
#[derive(Debug, Clone)]
pub struct Joint {
    pub params: GenParams,
    /// Poses of the fitted views, in their order.
    pub poses: Vec<Iso3>,
    pub coef: DVector<f64>,
    pub covs: Covariates,
    /// Stated σ of the eight camera parameters (`NaN` when held).
    pub sigma: [f64; NCAM],
    /// The same at the start (one Gauss-Newton step from it).
    pub sigma_start: [f64; NCAM],
    /// Correlation of each camera parameter with `tilt_x` (its full
    /// correlation row).
    pub corr_tx: [f64; NCAM],
    pub iters: usize,
    pub converged: bool,
    /// Weighted RMS of the training residual, px.
    pub rms: f64,
    /// The camera's largest move (in σ) when the covariates are refitted
    /// once at the solution and the fit repeated: a sensitivity, not a
    /// step of the fit.
    pub refit_move: f64,
}

/// Linearize `ids` at `params`, `poses`, with the detector's blur.
fn lin(data: Data<'_>, ids: &[usize], params: &GenParams, poses: &[Iso3]) -> Vec<LinPoint> {
    let vs: Vec<View<NoMeta>> = ids.iter().map(|&k| data.views[k].clone()).collect();
    let mut pts = linfield::linearize_plain(params, data.size, ids, &vs, poses);
    for p in &mut pts {
        p.blur = data.blur.get(&crate::blur::key(p.obs)).copied();
    }
    pts
}

fn cost(pts: &[LinPoint], basis: Option<&LawBasis>, coef: &DVector<f64>) -> f64 {
    let mut buf = vec![[0.0; 2]; basis.map_or(0, Basis::len)];
    pts.iter()
        .map(|p| {
            let mut r = p.r;
            if let Some(b) = basis {
                b.eval(p, &mut buf);
                for (c, a) in buf.iter().zip(coef.iter()) {
                    r[0] -= a * c[0];
                    r[1] -= a * c[1];
                }
            }
            p.w * (r[0] * r[0] + r[1] * r[1])
        })
        .sum()
}

type State = (GenParams, Vec<Iso3>, DVector<f64>);
type Step = ([f64; NCAM], BTreeMap<usize, [f64; 6]>, DVector<f64>);

fn apply(base: &State, step: &Step, ids: &[usize], alpha: f64) -> State {
    let mut params = base.0.clone();
    for (k, d) in step.0.iter().enumerate() {
        linfield::cam_add(&mut params, k, alpha * d);
    }
    let poses = ids
        .iter()
        .zip(&base.1)
        .map(|(id, pose)| match step.1.get(id) {
            Some(dp) => linfield::twist(std::array::from_fn(|m| alpha * dp[m])) * pose,
            None => *pose,
        })
        .collect();
    (params, poses, &base.2 + alpha * &step.2)
}

/// The result of one inner solve at fixed covariate models.
struct Inner {
    state: State,
    sigma: [f64; NCAM],
    /// σ at the first linearization (the start).
    sigma_start: [f64; NCAM],
    corr_tx: [f64; NCAM],
    iters: usize,
    converged: bool,
    rms: f64,
}

/// Gauss-Newton with step halving: the objective
/// `Σ w·|obs − u(θ, P) − B(u, v)·c|² + λ|c|²`, the monomials evaluated at the
/// current projection `u`, the covariates `v` held per point (`frozen`, in
/// the points' order), the law's coefficients under a weak ridge
/// ([`COEF_PRIOR`]) that pins only its near-gauge directions.
fn inner(
    data: Data<'_>,
    ids: &[usize],
    start: State,
    frozen: &[PointCov],
    law: Option<&LawBasis>,
    fixed: &[usize],
) -> Result<Inner> {
    let nb = law.map_or(0, Basis::len);
    let ridge = DMatrix::<f64>::identity(nb, nb) * COEF_PRIOR;
    let mut base = start.clone();
    let mut cur = start;
    let mut step: Option<Step> = None;
    let mut alpha = 1.0;
    let mut best = f64::INFINITY;
    let mut out = None;
    let mut sigma_start = None;
    let mut converged = false;
    let mut last_step = f64::INFINITY;
    let mut iters = 0;
    for it in 0..MAX_ITERS {
        iters = it + 1;
        let mut pts = lin(data, ids, &cur.0, &cur.1);
        anyhow::ensure!(pts.len() == frozen.len(), "point count changed");
        for (p, v) in pts.iter_mut().zip(frozen) {
            v.attach(p);
        }
        let c = cost(&pts, law, &cur.2) + COEF_PRIOR * cur.2.norm_squared();
        if c > best * (1.0 + 1e-12) {
            if let (Some(st), true) = (&step, alpha > 1.0 / 64.0) {
                alpha *= 0.5;
                cur = apply(&base, st, ids, alpha);
                continue;
            }
            // No decrease along the last step: the minimum, up to rounding,
            // when that step was already small.
            cur = base.clone();
            converged = last_step < STALL_TOL;
            break;
        }
        base = cur.clone();
        best = c;
        alpha = 1.0;
        if let Some(b) = law {
            add_term_jacobian(&mut pts, &cur.0, b, &cur.2, data.size);
        }
        for p in &mut pts {
            for &k in fixed {
                p.jcam[k] = [0.0; 2];
            }
        }
        let spec = Spec {
            camera_free: true,
            local: law.map(|b| b as &dyn Basis),
            penalty: (nb > 0).then_some(&ridge),
            ortho: false,
            shared: None,
            held: None,
        };
        let sol = linfield::fit(&[&pts], &spec);
        let cs = &sol.cams[0];
        let sigma: [f64; NCAM] = std::array::from_fn(|k| {
            if fixed.contains(&k) {
                f64::NAN
            } else {
                (cs.sigma2 * cs.cov[(k, k)]).sqrt()
            }
        });
        let corr_tx: [f64; NCAM] =
            std::array::from_fn(|k| cs.cov[(k, 6)] / (cs.cov[(k, k)] * cs.cov[(6, 6)]).sqrt());
        sigma_start.get_or_insert(sigma);
        let (_, dp) = linfield::predict_with_poses(
            &pts,
            &cs.dtheta,
            law.map(|b| (b as &dyn Basis, &cs.local)),
            None,
            None,
        );
        let st: Step = (cs.dtheta, dp, &cs.local - &base.2);
        last_step = (0..NCAM)
            .filter(|k| !fixed.contains(k))
            .map(|k| (cs.dtheta[k] / sigma[k]).abs())
            .fold(0.0, f64::max);
        let small = last_step < STEP_TOL;
        cur = apply(&base, &st, ids, 1.0);
        step = Some(st);
        let n = pts.len().max(1) as f64;
        out = Some((sigma, corr_tx, (cost(&pts, law, &base.2) / n).sqrt()));
        if small {
            converged = true;
            break;
        }
    }
    let (sigma, corr_tx, rms) = out.context("no iteration of the joint fit")?;
    Ok(Inner {
        state: cur,
        sigma,
        sigma_start: sigma_start.unwrap_or(sigma),
        corr_tx,
        iters,
        converged,
        rms,
    })
}

/// The covariate models fitted at `params`, `poses` and every point's
/// covariates there.
fn covariates_at(
    data: Data<'_>,
    ids: &[usize],
    params: &GenParams,
    poses: &[Iso3],
) -> Result<(Covariates, Vec<PointCov>)> {
    let pts = lin(data, ids, params, poses);
    let covs = Covariates::fit(&pts, data.size, params.tilt).context("the covariate fit failed")?;
    let v = covs.at(&pts, data.size, params.tilt);
    Ok((covs, v))
}

/// Fit the camera (all but `fixed`), every pose and the law's coefficients
/// to `ids`, from `init` and `poses` ([`inner`]), the covariates fitted
/// at the start (the blur model and the plane of sharp focus to the
/// detector's blur, in the start's `K'` frame) and held per point. Then,
/// as a sensitivity only, the covariates are refitted at the solution and
/// the fit repeated from it ([`Joint::refit_move`]).
pub fn joint_fit(
    data: Data<'_>,
    ids: &[usize],
    init: &GenParams,
    poses: &[Iso3],
    law: Option<&LawBasis>,
    fixed: &[usize],
) -> Result<Joint> {
    let nb = law.map_or(0, Basis::len);
    let start: State = (init.clone(), poses.to_vec(), DVector::zeros(nb));
    let (covs, frozen) = covariates_at(data, ids, init, poses)?;
    let r = inner(data, ids, start, &frozen, law, fixed)?;
    let refit_move = if law.is_some() {
        let (_, v2) = covariates_at(data, ids, &r.state.0, &r.state.1)?;
        let r2 = inner(data, ids, r.state.clone(), &v2, law, fixed)?;
        (0..NCAM)
            .filter(|k| !fixed.contains(k))
            .map(|k| {
                ((linfield::cam_get(&r2.state.0, k) - linfield::cam_get(&r.state.0, k))
                    / r.sigma[k])
                    .abs()
            })
            .fold(0.0, f64::max)
    } else {
        0.0
    };
    Ok(Joint {
        params: r.state.0,
        poses: r.state.1,
        coef: r.state.2,
        covs,
        sigma: r.sigma,
        sigma_start: r.sigma_start,
        corr_tx: r.corr_tx,
        iters: r.iters,
        converged: r.converged,
        rms: r.rms,
        refit_move,
    })
}

/// Held-out residuals of `test` views: poses fitted with the camera held
/// (the plain model, then the law applied with the fit's coefficients and
/// the poses refitted linearly), the covariates from the training fit's
/// models at the test views' geometry.
pub fn held_out(
    data: Data<'_>,
    test: &[usize],
    fit: &Joint,
    init_poses: &[Iso3],
    law: Option<&LawBasis>,
) -> Result<Vec<[f64; 2]>> {
    let vs: Vec<View<NoMeta>> = test.iter().map(|&k| data.views[k].clone()).collect();
    let po = model::fit(
        &GenCfg::base("plain"),
        test,
        &vs,
        &fit.params,
        init_poses,
        data.size,
        Free::PosesOnly,
        200,
    )?;
    let mut pts = lin(data, test, &fit.params, &po.poses);
    let v = fit.covs.at(&pts, data.size, fit.params.tilt);
    for (p, c) in pts.iter_mut().zip(&v) {
        c.attach(p);
    }
    if let Some(b) = law {
        add_term_jacobian(&mut pts, &fit.params, b, &fit.coef, data.size);
    }
    Ok(linfield::predict(
        &pts,
        &[0.0; NCAM],
        law.map(|b| (b as &dyn Basis, &fit.coef)),
        None,
        None,
    ))
}

// ───────────────────────── the tilt gauge ─────────────────────────

/// [`tilt_rotation`] as a quaternion, built from its two axis rotations
/// (`R = Ry(−τy)·Rx(−τx)` in the right-handed convention).
pub fn tilt_quat(tau: [f64; 2]) -> UnitQuaternion<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -tau[1])
        * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -tau[0])
}

/// `params`, `poses` re-expressed at tilt `tau` with the same plain
/// pinhole `K'` and the same `K'`-frame poses: the projection is unchanged
/// without distortion (`K·H(τ) = K'·R(τ)`), and only the distortion, which
/// acts in the lens frame, tells the two apart.
pub fn transport(params: &GenParams, poses: &[Iso3], tau: [f64; 2]) -> (GenParams, Vec<Iso3>) {
    let (r0, r1) = (tilt_rotation(params.tilt), tilt_rotation(tau));
    let [fx, fy, cx, cy] = params.intr;
    let (fxp, fyp) = (fx * r0[(2, 2)], fy * r0[(2, 2)]);
    let (cxp, cyp) = (cx - fx * r0[(0, 2)], cy - fy * r0[(1, 2)]);
    let (fx1, fy1) = (fxp / r1[(2, 2)], fyp / r1[(2, 2)]);
    let mut out = params.clone();
    out.intr = [fx1, fy1, cxp + fx1 * r1[(0, 2)], cyp + fy1 * r1[(1, 2)]];
    out.tilt = tau;
    let q = tilt_quat(tau).inverse() * tilt_quat(params.tilt);
    let g = Iso3::from_parts(Translation3::identity(), q);
    (out, poses.iter().map(|p| g * p).collect())
}

// ───────────────────────── the real-data study ─────────────────────────

/// Parameters reported, as camera indices: `tilt_x tilt_y cx cy k1 fx`.
const REPORT: [(usize, &str); 6] = [
    (6, "tilt_x"),
    (7, "tilt_y"),
    (2, "cx"),
    (3, "cy"),
    (4, "k1"),
    (0, "fx"),
];

fn jack(v: &[f64]) -> f64 {
    let k = v.len() as f64;
    let m = v.iter().sum::<f64>() / k;
    ((k - 1.0) / k * v.iter().map(|x| (x - m).powi(2)).sum::<f64>()).sqrt()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    crate::pearson(a, b)
}

fn rms(e: &[[f64; 2]]) -> f64 {
    linfield::stats(e)[0]
}

/// One configuration (law, tilt free or held) on one camera.
struct Run {
    full: Joint,
    folds: Vec<Joint>,
    held: Vec<[f64; 2]>,
}

/// The focus comparison inputs of a run: `π` in its `K'` frame, `f'`, the
/// implied `pitch/m_p` and its fold jackknife, the tilt's fold jackknife.
fn focus_result(cam: &CameraData, ctx: &CamCtx, run: &Run) -> Option<CamResult> {
    let n = cam.dataset.num_views();
    let all: Vec<usize> = (0..n).collect();
    let pof = |ids: &[usize], j: &Joint| -> Option<(Vector3<f64>, f64, f64)> {
        let obs: Vec<BlurObs> = real_obs(cam, ids, &j.poses, j.params.tilt)
            .iter()
            .map(|o| o.o)
            .collect();
        let fit = fit_pof(&obs, None)?;
        let fpx = j.params.intr[0] * tilt_rotation(j.params.tilt)[(2, 2)];
        Some((fit.pi, fpx, implied(&fit.pi, j.params.tilt, fpx).0))
    };
    let (pi, fpx, pitch) = pof(&all, &run.full)?;
    let mut fp = Vec::new();
    for ((train, _), j) in ctx.folds.iter().zip(&run.folds) {
        fp.push(pof(train, j)?.2);
    }
    let tj = |k: usize| {
        jack(
            &run.folds
                .iter()
                .map(|j| j.params.tilt[k])
                .collect::<Vec<_>>(),
        )
    };
    Some(CamResult {
        id: cam.id.clone(),
        tau: run.full.params.tilt,
        tau_jk: [tj(0), tj(1)],
        fpx,
        pi,
        pitch,
        pitch_jk: jack(&fp),
    })
}

/// Fit one configuration on a camera: full data from `start`, each fold
/// from `fold_start(f)`, held out by view folds.
fn run_config(
    cam: &CameraData,
    ctx: &CamCtx,
    law: Option<&LawBasis>,
    fixed: &[usize],
    start: &(GenParams, Vec<Iso3>),
    fold_start: &dyn Fn(usize) -> Result<(GenParams, Vec<Iso3>)>,
) -> Result<Run> {
    let data = Data {
        views: &cam.dataset.views,
        blur: &cam.blur,
        size: cam.size,
    };
    let all: Vec<usize> = (0..cam.dataset.num_views()).collect();
    let full = joint_fit(data, &all, &start.0, &start.1, law, fixed)?;
    let mut folds = Vec::new();
    let mut held = Vec::new();
    for (f, (train, test)) in ctx.folds.iter().enumerate() {
        let (p0, q0) = fold_start(f)?;
        let j = joint_fit(data, train, &p0, &q0, law, fixed)?;
        let init: Vec<Iso3> = test
            .iter()
            .map(|&k| ctx.base.params.cam_se3_target[k])
            .collect();
        held.extend(held_out(data, test, &j, &init, law)?);
        folds.push(j);
    }
    Ok(Run { full, folds, held })
}

pub fn run(cams: &[CameraData], ctxs: &[CamCtx]) -> Result<()> {
    println!("\n# Tilt bias: the route refitted jointly with a defocus/depth law");
    let configs: Vec<Option<Law>> = std::iter::once(None)
        .chain(Law::ALL.iter().copied().map(Some))
        .collect();
    let label = |l: Option<Law>| l.map_or("plain (route model)", Law::name);
    // runs[camera][config]
    let mut runs: Vec<Vec<Run>> = Vec::new();
    let mut consts_of = Vec::new();
    for (cam, ctx) in cams.iter().zip(ctxs) {
        let data = Data {
            views: &cam.dataset.views,
            blur: &cam.blur,
            size: cam.size,
        };
        let all: Vec<usize> = (0..cam.dataset.num_views()).collect();
        let init = gen_init(&ctx.base, ctx.centre)?;
        let poses0 = ctx.base.params.cam_se3_target.clone();
        let mut pts = lin(data, &all, &init, &poses0);
        let (_, v) = covariates_at(data, &all, &init, &poses0)?;
        for (p, c) in pts.iter_mut().zip(&v) {
            c.attach(p);
        }
        let consts = Consts::of(&pts);
        consts_of.push(consts);
        let fold_start = |f: usize| -> Result<(GenParams, Vec<Iso3>)> {
            let fr = &ctx.fold_routes[f];
            Ok((gen_init(fr, ctx.centre)?, fr.params.cam_se3_target.clone()))
        };
        let mut per = Vec::new();
        for law in &configs {
            let basis = law.map(|l| LawBasis::new(l, cam.size, consts));
            let r = run_config(
                cam,
                ctx,
                basis.as_ref(),
                &[],
                &(init.clone(), poses0.clone()),
                &fold_start,
            )?;
            eprintln!(
                "  {} {}: iters {} (converged {}, covariate refit moves {:.2} σ), train rms {:.4}, held-out {:.4}",
                cam.id,
                label(*law),
                r.full.iters,
                r.full.converged,
                r.full.refit_move,
                r.full.rms,
                rms(&r.held)
            );
            per.push(r);
        }
        runs.push(per);
    }
    let cid = |c: &CameraData| c.id.rsplit('_').next().unwrap_or("").to_string();

    // The measured form of the synthetic sweep: the rig mean of the
    // `g + |g|` coefficients.
    for (ci, law) in configs.iter().enumerate() {
        if !matches!(law, Some(Law::Signed1Abs | Law::Abs)) {
            continue;
        }
        let n = runs.len() as f64;
        let nb = runs[0][ci].full.coef.len();
        let mean: Vec<f64> = (0..nb)
            .map(|k| runs.iter().map(|p| p[ci].full.coef[k]).sum::<f64>() / n)
            .collect();
        println!(
            "\nrig-mean coefficients of {} (for tiltbias_synth): [{}]",
            label(*law),
            mean.iter()
                .map(|v| format!("{v:.4e}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // The route against its re-convergence by this fit.
    println!("\n## The plain model re-converged by the iterated fit, minus the library route");
    println!("camera|Δtilt_x|Δtilt_y|Δcx|Δcy|Δk1|Δfx (stated σ units)|iters");
    for ((cam, ctx), per) in cams.iter().zip(ctxs).zip(&runs) {
        let r0 = gen_init(&ctx.base, ctx.centre)?;
        let j = &per[0].full;
        let txt: Vec<String> = REPORT
            .iter()
            .map(|&(k, _)| {
                format!(
                    "{:+.2}",
                    (linfield::cam_get(&j.params, k) - linfield::cam_get(&r0, k)) / j.sigma[k]
                )
            })
            .collect();
        println!("{}|{}|{}", cid(cam), txt.join("|"), j.iters);
    }

    // Held-out.
    println!("\n## Held-out pooled RMS (px; 4 folds by view), Δ variance vs plain");
    let pooled = |ci: usize| -> f64 {
        let all: Vec<[f64; 2]> = runs.iter().flat_map(|p| p[ci].held.clone()).collect();
        rms(&all)
    };
    let p0 = pooled(0);
    for (ci, law) in configs.iter().enumerate() {
        let p = pooled(ci);
        let per: Vec<String> = runs
            .iter()
            .map(|r| format!("{:.4}", rms(&r[ci].held)))
            .collect();
        println!(
            "{}|{:.4}|{:+.1} %|{}",
            label(*law),
            p,
            100.0 * (p * p - p0 * p0) / (p0 * p0),
            per.join("|")
        );
    }

    // The camera's change per law.
    println!(
        "\n## Camera change, law − plain (full data), ± fold jackknife of the change; σ ratio = stated σ with the law / without"
    );
    for (ci, law) in configs.iter().enumerate().skip(1) {
        println!("\n### {}", label(*law));
        println!(
            "camera|{}|σ(tilt_x) ratio|corr(tilt_x, cy) with law|",
            REPORT
                .iter()
                .map(|(_, n)| format!("Δ{n}"))
                .collect::<Vec<_>>()
                .join("|")
        );
        for (cam, per) in cams.iter().zip(&runs) {
            let (a, b) = (&per[ci], &per[0]);
            let txt: Vec<String> = REPORT
                .iter()
                .map(|&(k, _)| {
                    let d =
                        linfield::cam_get(&a.full.params, k) - linfield::cam_get(&b.full.params, k);
                    let df: Vec<f64> = a
                        .folds
                        .iter()
                        .zip(&b.folds)
                        .map(|(x, y)| {
                            linfield::cam_get(&x.params, k) - linfield::cam_get(&y.params, k)
                        })
                        .collect();
                    format!("{d:+.5} ± {:.5}", jack(&df))
                })
                .collect();
            let ratio = a.full.sigma[6] / b.full.sigma[6];
            println!(
                "{}|{}|{:.2} (at the route {:.2}){}|{:+.2}|{}",
                cid(cam),
                txt.join("|"),
                ratio,
                a.full.sigma_start[6] / b.full.sigma[6],
                if ratio > IDENT_RATIO {
                    " (not identified)"
                } else {
                    ""
                },
                a.full.corr_tx[3],
                if a.full.converged {
                    ""
                } else {
                    "not converged"
                }
            );
        }
    }

    // The focus comparison per configuration.
    println!("\n## Focus against distortion with each law's tilt");
    let mut cross = Vec::new();
    for (ci, law) in configs.iter().enumerate() {
        println!("\n### {}", label(*law));
        let res: Vec<CamResult> = cams
            .iter()
            .zip(ctxs)
            .zip(&runs)
            .filter_map(|((cam, ctx), per)| focus_result(cam, ctx, &per[ci]))
            .collect();
        if res.len() < cams.len() {
            println!("  (a focus fit failed)");
            continue;
        }
        let x = cross_camera(&res, label(*law));
        cross.push((ci, x, res));
    }
    let base_diff: Vec<f64> = cross
        .first()
        .map(|(_, x, _)| x.diff.iter().map(|d| d[0]).collect())
        .unwrap_or_default();
    let base_tau: Vec<f64> = runs.iter().map(|p| p[0].full.params.tilt[0]).collect();
    println!(
        "\nlaw|pitch/m_p mean µm|sd|χ² (5 dof)|focus − dist τx: RMS, sd|τy: RMS|sd focus τx|sd dist τx|corr|Δτx vs plain's (focus − routed): slope, corr"
    );
    for (ci, x, _) in &cross {
        let dx: Vec<f64> = x.diff.iter().map(|d| d[0]).collect();
        let dy: Vec<f64> = x.diff.iter().map(|d| d[1]).collect();
        let rmsv = |v: &[f64]| (v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64).sqrt();
        let dtau: Vec<f64> = runs
            .iter()
            .zip(&base_tau)
            .map(|(p, t)| p[*ci].full.params.tilt[0] - t)
            .collect();
        let slope = {
            let sxy: f64 = dtau.iter().zip(&base_diff).map(|(a, b)| a * b).sum();
            let sxx: f64 = base_diff.iter().map(|b| b * b).sum();
            sxy / sxx
        };
        let worst = runs
            .iter()
            .map(|p| p[*ci].full.sigma[6] / p[0].full.sigma[6])
            .fold(0.0, f64::max);
        println!(
            "{}{}|{:.2}|{:.2}|{:.1}|{:.4}, {:.4}|{:.4}|{:.4}|{:.4}|{:+.2}|{:+.2}, {:+.2}",
            label(configs[*ci]),
            if worst > IDENT_RATIO {
                " (tilt not identified)"
            } else {
                ""
            },
            x.pitch_mean,
            x.pitch_sd,
            x.chi2,
            rmsv(&dx),
            crate::mean_std(&dx).1,
            rmsv(&dy),
            x.sd_focus,
            x.sd_dist,
            x.corr,
            slope,
            if *ci == 0 {
                f64::NAN
            } else {
                pearson(&dtau, &base_diff)
            }
        );
    }

    // Reverse: the tilt held at the focus-derived value.
    let Some((_, x0, res0)) = cross.first() else {
        return Ok(());
    };
    let pitch = x0.pitch_mean;
    println!(
        "\n## Reverse: tilt held at the focus-derived value (common pitch/m_p {pitch:.2} µm), the rest refitted"
    );
    let rev_laws: [Option<Law>; 5] = [
        None,
        Some(Law::Signed1Abs),
        Some(Law::Abs),
        Some(Law::GeoDepth),
        Some(Law::Var),
    ];
    let mut rev: Vec<Vec<Run>> = Vec::new();
    for (c, (((cam, ctx), per), r0)) in cams.iter().zip(ctxs).zip(&runs).zip(res0).enumerate() {
        let tau_f = tilt_from_pof(&r0.pi, r0.fpx * pitch * 1e-6);
        let plain = &per[0];
        let start = transport(&plain.full.params, &plain.full.poses, tau_f);
        let fold_start = |f: usize| -> Result<(GenParams, Vec<Iso3>)> {
            let j = &plain.folds[f];
            Ok(transport(&j.params, &j.poses, tau_f))
        };
        let mut row = Vec::new();
        for law in rev_laws {
            let basis = law.map(|l| LawBasis::new(l, cam.size, consts_of[c]));
            let r = run_config(cam, ctx, basis.as_ref(), &[6, 7], &start, &fold_start)?;
            eprintln!(
                "  {} held tilt, {}: iters {}, held-out {:.4}",
                cam.id,
                label(law),
                r.full.iters,
                rms(&r.held)
            );
            row.push(r);
        }
        println!(
            "  {}: focus tilt ({:+.4}, {:+.4}) against routed ({:+.4}, {:+.4}) rad",
            cid(cam),
            tau_f[0],
            tau_f[1],
            plain.full.params.tilt[0],
            plain.full.params.tilt[1]
        );
        rev.push(row);
    }
    println!("configuration|held-out pooled|Δ var vs free-tilt plain|per camera");
    for (i, law) in rev_laws.iter().enumerate() {
        let ci = configs.iter().position(|c| c == law).unwrap_or(0);
        for (tag, src) in [("free tilt", 0usize), ("focus tilt", 1)] {
            let held: Vec<&Vec<[f64; 2]>> = if src == 0 {
                runs.iter().map(|p| &p[ci].held).collect()
            } else {
                rev.iter().map(|p| &p[i].held).collect()
            };
            let all: Vec<[f64; 2]> = held.iter().flat_map(|h| h.iter().copied()).collect();
            let p = rms(&all);
            println!(
                "{tag}, {}|{:.4}|{:+.1} %|{}",
                label(*law),
                p,
                100.0 * (p * p - p0 * p0) / (p0 * p0),
                held.iter()
                    .map(|h| format!("{:.4}", rms(h)))
                    .collect::<Vec<_>>()
                    .join("|")
            );
        }
    }
    println!("\nheld tilt: camera change against the free-tilt fit of the same law (full data)");
    println!("camera|law|Δcx|Δcy|Δk1|Δfx|train rms free → held");
    for ((cam, per), row) in cams.iter().zip(&runs).zip(&rev) {
        for (i, law) in rev_laws.iter().enumerate() {
            let ci = configs.iter().position(|c| c == law).unwrap_or(0);
            let (a, b) = (&row[i].full, &per[ci].full);
            let d = |k| linfield::cam_get(&a.params, k) - linfield::cam_get(&b.params, k);
            println!(
                "{}|{}|{:+.2}|{:+.2}|{:+.4}|{:+.2}|{:.4} → {:.4}",
                cid(cam),
                label(*law),
                d(2),
                d(3),
                d(4),
                d(0),
                b.rms,
                a.rms
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::project_point;
    use crate::synth;

    /// The quaternion and the matrix form of the tilt rotation agree.
    #[test]
    fn tilt_quat_is_tilt_rotation() {
        for tau in [[-0.09, 0.0], [0.05, -0.12], [0.2, 0.15]] {
            let q = tilt_quat(tau).to_rotation_matrix();
            let r = tilt_rotation(tau);
            assert!((q.matrix() - r).abs().max() < 1e-14);
        }
    }

    /// Without distortion, moving the tilt along its gauge leaves every
    /// projection unchanged.
    #[test]
    fn transport_is_a_gauge_without_distortion() {
        let mut p = synth::typical();
        p.dist = [0.0; 9];
        let poses = synth::poses(6, 3);
        let (q, qposes) = transport(&p, &poses, [-0.07, 0.012]);
        let plain = GenCfg::base("plain");
        let mut worst: f64 = 0.0;
        for (a, b) in poses.iter().zip(&qposes) {
            for bx in [0.0, 0.03, 0.08] {
                for by in [0.0, 0.04, 0.07] {
                    let u = project_point(&plain, &p, synth::SIZE, a, [bx, by]);
                    let v = project_point(&plain, &q, synth::SIZE, b, [bx, by]);
                    worst = worst.max((u[0] - v[0]).hypot(u[1] - v[1]));
                }
            }
        }
        assert!(worst < 1e-8, "{worst} px");
    }
}
