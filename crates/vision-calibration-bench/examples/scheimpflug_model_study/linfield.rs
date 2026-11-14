//! Linearized fits of image-displacement terms about the route's solution.
//!
//! The route's camera and poses are the linearization point. Every candidate
//! term — an image field, a lens-side term, a detector regressor — is a set of
//! pixel-displacement columns per point; the camera's eight parameters
//! (`fx fy cx cy k1 k2 tilt_x tilt_y`) and each view's pose are columns too
//! (central differences of the model's projection). One Gauss-Newton step from
//! the route's optimum, poses eliminated per view (Schur complement), solves a
//! candidate exactly to second order in its displacement (a field of 0.1 px
//! moves the camera by a pixel at most). Held-out views are linearized at
//! their training fold's camera with their own poses fitted, then refitted
//! (poses only) after the candidate is applied.
//!
//! Every basis is evaluated at the model's own projection `u` (never at the
//! detection), so a field cannot contract the measurement noise.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Point3, Vector3};
use vision_calibration_core::{Iso3, NoMeta, View};

use crate::model::{GenCfg, GenParams, bspline_axis, project_point, spline_len};

/// Camera parameters a linearized fit can move.
pub const NCAM: usize = 8;
pub const CAM_NAMES: [&str; NCAM] = ["fx", "fy", "cx", "cy", "k1", "k2", "tilt_x", "tilt_y"];
/// The camera's pixel directions: its eight parameters and a rotation common
/// to every view (a gauge the poses undo).
pub const NCF: usize = NCAM + 3;

pub fn cam_get(p: &GenParams, k: usize) -> f64 {
    match k {
        0..=3 => p.intr[k],
        4 => p.dist[0],
        5 => p.dist[1],
        6 => p.tilt[0],
        _ => p.tilt[1],
    }
}

pub fn cam_add(p: &mut GenParams, k: usize, d: f64) {
    match k {
        0..=3 => p.intr[k] += d,
        4 => p.dist[0] += d,
        5 => p.dist[1] += d,
        6 => p.tilt[0] += d,
        _ => p.tilt[1] += d,
    }
}

/// Central-difference step of camera parameter `k` at `p`.
fn cam_step(p: &GenParams, k: usize) -> f64 {
    1e-6 * cam_get(p, k).abs().max(1.0)
}

/// A rigid motion `[ω, t]` applied on the left.
fn twist(v: [f64; 6]) -> Iso3 {
    Iso3::new(
        Vector3::new(v[3], v[4], v[5]),
        Vector3::new(v[0], v[1], v[2]),
    )
}

/// One observation at the linearization point.
#[derive(Debug, Clone)]
pub struct LinPoint {
    pub view: usize,
    pub board: [f64; 2],
    pub pose: Iso3,
    pub obs: [f64; 2],
    /// The model's projection, px.
    pub u: [f64; 2],
    /// `obs − u`.
    pub r: [f64; 2],
    /// Huber (1 px) weight at the linearization point.
    pub w: f64,
    pub jcam: [[f64; 2]; NCAM],
    /// Left rotation (3) then translation (3).
    pub jpose: [[f64; 2]; 6],
    /// The rig-typical camera's pixel directions at `u` ([`NCF`]).
    pub cfield: [[f64; 2]; NCF],
    /// Camera-frame depth, metres.
    pub depth: f64,
    /// Detector's fitted blur σ, px.
    pub blur: Option<f64>,
    /// The blur predicted from the geometry alone (image position and
    /// depth), px.
    pub gblur: Option<f64>,
    /// Image direction of the board's x axis at the point, rad.
    pub roll: f64,
    /// Local board-to-image scale, px per mm.
    pub scale: f64,
    /// Board-to-image Jacobian, px per mm: columns `∂u/∂b_x`, `∂u/∂b_y`.
    pub jboard: [[f64; 2]; 2],
    /// Precomputed columns (see the caller's catalog).
    pub extra: Vec<[f64; 2]>,
}

/// The rig-typical camera's pixel directions at pixel `u`: the ray through
/// `u` (at depth 1, identity pose) and the projection's derivative along
/// each camera parameter and each axis of a common rotation.
pub fn cam_fields(typ: &GenParams, size: [u32; 2], u: [f64; 2]) -> [[f64; 2]; NCF] {
    let plain = GenCfg::base("plain");
    let at_z1 = Iso3::translation(0.0, 0.0, 1.0);
    let mut x = [
        (u[0] - typ.intr[2]) / typ.intr[0],
        (u[1] - typ.intr[3]) / typ.intr[1],
    ];
    for _ in 0..30 {
        let f = project_point(&plain, typ, size, &at_z1, x);
        let e = [f[0] - u[0], f[1] - u[1]];
        if e[0].hypot(e[1]) < 1e-10 {
            break;
        }
        let d = 1e-7;
        let fx = project_point(&plain, typ, size, &at_z1, [x[0] + d, x[1]]);
        let fy = project_point(&plain, typ, size, &at_z1, [x[0], x[1] + d]);
        let (a, b) = ((fx[0] - f[0]) / d, (fy[0] - f[0]) / d);
        let (c, dd) = ((fx[1] - f[1]) / d, (fy[1] - f[1]) / d);
        let det = a * dd - b * c;
        x[0] -= (dd * e[0] - b * e[1]) / det;
        x[1] -= (-c * e[0] + a * e[1]) / det;
    }
    let mut out = [[0.0; 2]; NCF];
    for (k, o) in out.iter_mut().enumerate().take(NCAM) {
        let h = cam_step(typ, k);
        let (mut a, mut b) = (typ.clone(), typ.clone());
        cam_add(&mut a, k, h);
        cam_add(&mut b, k, -h);
        let pa = project_point(&plain, &a, size, &at_z1, x);
        let pb = project_point(&plain, &b, size, &at_z1, x);
        *o = [(pa[0] - pb[0]) / (2.0 * h), (pa[1] - pb[1]) / (2.0 * h)];
    }
    for m in 0..3 {
        let h = 1e-6;
        let mut v = [0.0; 6];
        v[m] = h;
        let pa = project_point(&plain, typ, size, &(twist(v) * at_z1), x);
        v[m] = -h;
        let pb = project_point(&plain, typ, size, &(twist(v) * at_z1), x);
        out[NCAM + m] = [(pa[0] - pb[0]) / (2.0 * h), (pa[1] - pb[1]) / (2.0 * h)];
    }
    out
}

/// Linearize `views` (dataset indices `ids`) at camera `params` and `poses`.
pub fn linearize(
    params: &GenParams,
    typical: &GenParams,
    size: [u32; 2],
    ids: &[usize],
    views: &[View<NoMeta>],
    poses: &[Iso3],
) -> Vec<LinPoint> {
    let plain = GenCfg::base("plain");
    let proj = |p: &GenParams, pose: &Iso3, b: [f64; 2]| project_point(&plain, p, size, pose, b);
    let mut out = Vec::new();
    for ((&id, view), pose) in ids.iter().zip(views).zip(poses) {
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            let b = [p3.x, p3.y];
            let u = proj(params, pose, b);
            let r = [p2.x - u[0], p2.y - u[1]];
            let n = r[0].hypot(r[1]);
            let mut jcam = [[0.0; 2]; NCAM];
            for (k, j) in jcam.iter_mut().enumerate() {
                let h = cam_step(params, k);
                let (mut a, mut c) = (params.clone(), params.clone());
                cam_add(&mut a, k, h);
                cam_add(&mut c, k, -h);
                let (pa, pc) = (proj(&a, pose, b), proj(&c, pose, b));
                *j = [(pa[0] - pc[0]) / (2.0 * h), (pa[1] - pc[1]) / (2.0 * h)];
            }
            let mut jpose = [[0.0; 2]; 6];
            for (m, j) in jpose.iter_mut().enumerate() {
                let h = 1e-6;
                let mut v = [0.0; 6];
                v[m] = h;
                let pa = proj(params, &(twist(v) * pose), b);
                v[m] = -h;
                let pc = proj(params, &(twist(v) * pose), b);
                *j = [(pa[0] - pc[0]) / (2.0 * h), (pa[1] - pc[1]) / (2.0 * h)];
            }
            // Board-to-image Jacobian (per metre).
            let hb = 1e-5;
            let bx = proj(params, pose, [b[0] + hb, b[1]]);
            let bxm = proj(params, pose, [b[0] - hb, b[1]]);
            let by = proj(params, pose, [b[0], b[1] + hb]);
            let bym = proj(params, pose, [b[0], b[1] - hb]);
            let jx = [(bx[0] - bxm[0]) / (2.0 * hb), (bx[1] - bxm[1]) / (2.0 * hb)];
            let jy = [(by[0] - bym[0]) / (2.0 * hb), (by[1] - bym[1]) / (2.0 * hb)];
            let det = (jx[0] * jy[1] - jx[1] * jy[0]).abs();
            let pc: Point3<f64> = pose * Point3::new(b[0], b[1], 0.0);
            out.push(LinPoint {
                view: id,
                board: b,
                pose: *pose,
                obs: [p2.x, p2.y],
                u,
                r,
                w: if n <= crate::HUBER_PX {
                    1.0
                } else {
                    crate::HUBER_PX / n
                },
                jcam,
                jpose,
                cfield: cam_fields(typical, size, u),
                depth: pc.z,
                blur: None,
                gblur: None,
                roll: jx[1].atan2(jx[0]),
                scale: det.sqrt() * 1e-3,
                jboard: [[jx[0] * 1e-3, jx[1] * 1e-3], [jy[0] * 1e-3, jy[1] * 1e-3]],
                extra: Vec::new(),
            });
        }
    }
    out
}

// ───────────────────────── bases ─────────────────────────

/// Pixel-displacement columns of a term, per point.
pub trait Basis {
    fn len(&self) -> usize;
    /// Columns at `p` into `out[..len]`.
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]);
}

/// The uniform bicubic B-spline of the study (`model::spline_eval`).
pub struct Spline {
    pub cells: [usize; 2],
    pub size: [f64; 2],
}

impl Basis for Spline {
    fn len(&self) -> usize {
        2 * spline_len(self.cells)
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        let n = spline_len(self.cells);
        let nx = self.cells[0] + 3;
        out[..2 * n].iter_mut().for_each(|o| *o = [0.0; 2]);
        let (ix, wx) = bspline_axis(p.u[0], self.size[0], self.cells[0]);
        let (iy, wy) = bspline_axis(p.u[1], self.size[1], self.cells[1]);
        for (b, wyb) in wy.iter().enumerate() {
            for (a, wxa) in wx.iter().enumerate() {
                let k = (iy + b) * nx + ix + a;
                out[k] = [wxa * wyb, 0.0];
                out[n + k] = [0.0, wxa * wyb];
            }
        }
    }
}

/// Second-difference smoothness penalty of a [`Spline`] block, weight
/// `1/σ²` relative to unit-weight reprojection residuals.
pub fn spline_penalty(cells: [usize; 2], sigma: f64) -> DMatrix<f64> {
    let n = spline_len(cells);
    let (nx, ny) = (cells[0] + 3, cells[1] + 3);
    let mut h = DMatrix::zeros(2 * n, 2 * n);
    let lam = 1.0 / (sigma * sigma);
    let mut add = |t: [(usize, f64); 3]| {
        for &(i, a) in &t {
            for &(j, b) in &t {
                h[(i, j)] += lam * a * b;
            }
        }
    };
    for comp in 0..2 {
        let at = |i: usize, j: usize| comp * n + j * nx + i;
        for j in 0..ny {
            for i in 0..nx {
                if i + 2 < nx {
                    add([(at(i, j), 1.0), (at(i + 1, j), -2.0), (at(i + 2, j), 1.0)]);
                }
                if j + 2 < ny {
                    add([(at(i, j), 1.0), (at(i, j + 1), -2.0), (at(i, j + 2), 1.0)]);
                }
            }
        }
    }
    h
}

/// The study's polynomial field (`GenCfg::field_deg`): `x^i y^j` in
/// half-size-normalized pixel coordinates, `u` components then `v`.
pub struct RawPoly {
    pub terms: Vec<(i32, i32)>,
    pub half: [f64; 2],
}

impl Basis for RawPoly {
    fn len(&self) -> usize {
        2 * self.terms.len()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        let x = (p.u[0] - self.half[0]) / self.half[0];
        let y = (p.u[1] - self.half[1]) / self.half[1];
        let n = self.terms.len();
        for (k, &(i, j)) in self.terms.iter().enumerate() {
            let m = x.powi(i) * y.powi(j);
            out[k] = [m, 0.0];
            out[n + k] = [0.0, m];
        }
    }
}

/// Complex monomials `z^p·z̄^q`, `z = (u − centre)/scale` in pixels, two
/// real columns per term: coefficient `1` then `i`.
#[derive(Clone)]
pub struct CMono {
    pub terms: Vec<(i32, i32)>,
    pub centre: [f64; 2],
    pub scale: f64,
}

impl CMono {
    /// Terms of degree `lo..=hi`, ordered by degree, then `|m|`, then `m`.
    pub fn degrees(lo: usize, hi: usize, centre: [f64; 2], scale: f64) -> Self {
        let mut terms = Vec::new();
        for d in lo as i32..=hi as i32 {
            let mut t: Vec<(i32, i32)> = (0..=d).map(|p| (p, d - p)).collect();
            t.sort_by_key(|&(p, q)| ((p - q - 1).abs(), p - q - 1));
            terms.extend(t);
        }
        Self {
            terms,
            centre,
            scale,
        }
    }
    pub fn value(&self, u: [f64; 2], out: &mut [[f64; 2]]) {
        let z = nalgebra::Complex::new(
            (u[0] - self.centre[0]) / self.scale,
            (u[1] - self.centre[1]) / self.scale,
        );
        for (k, &(p, q)) in self.terms.iter().enumerate() {
            let m = z.powi(p) * z.conj().powi(q);
            out[2 * k] = [m.re, m.im];
            out[2 * k + 1] = [-m.im, m.re];
        }
    }
}

impl Basis for CMono {
    fn len(&self) -> usize {
        2 * self.terms.len()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        self.value(p.u, out);
    }
}

/// An in-plane displacement of the target, fixed in the board frame: a
/// polynomial in board coordinates (normalized by `half`, metres, about
/// `centre`), in mm, mapped to the image by each point's board-to-image
/// Jacobian. Columns: `x` displacement per monomial, then `y`.
pub struct BoardPoly {
    pub terms: Vec<(i32, i32)>,
    pub centre: [f64; 2],
    pub half: f64,
}

impl Basis for BoardPoly {
    fn len(&self) -> usize {
        2 * self.terms.len()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        let x = (p.board[0] - self.centre[0]) / self.half;
        let y = (p.board[1] - self.centre[1]) / self.half;
        let n = self.terms.len();
        for (k, &(i, j)) in self.terms.iter().enumerate() {
            let m = x.powi(i) * y.powi(j);
            out[k] = [p.jboard[0][0] * m, p.jboard[0][1] * m];
            out[n + k] = [p.jboard[1][0] * m, p.jboard[1][1] * m];
        }
    }
}

/// A free in-plane offset (mm) per board point, mapped to the image by the
/// board-to-image Jacobian: the target's rasterisation and print error.
pub struct BoardPoints {
    /// Board point (µm key) → index; points not in the map get no column.
    pub index: BTreeMap<(i64, i64), usize>,
}

impl BoardPoints {
    pub fn key(b: [f64; 2]) -> (i64, i64) {
        ((b[0] * 1e6).round() as i64, (b[1] * 1e6).round() as i64)
    }
}

impl Basis for BoardPoints {
    fn len(&self) -> usize {
        2 * self.index.len()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        out[..2 * self.index.len()]
            .iter_mut()
            .for_each(|o| *o = [0.0; 2]);
        if let Some(&k) = self.index.get(&Self::key(p.board)) {
            out[2 * k] = p.jboard[0];
            out[2 * k + 1] = p.jboard[1];
        }
    }
}

/// A basis multiplied by a per-point scalar.
pub struct Scaled<B: Basis> {
    pub inner: B,
    pub f: Box<dyn Fn(&LinPoint) -> f64>,
}

impl<B: Basis> Basis for Scaled<B> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        self.inner.eval(p, out);
        let s = (self.f)(p);
        out[..self.inner.len()]
            .iter_mut()
            .for_each(|o| *o = [o[0] * s, o[1] * s]);
    }
}

/// Precomputed per-point columns `p.extra[idx]`.
pub struct Extra(pub Vec<usize>);

impl Basis for Extra {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        for (o, &k) in out.iter_mut().zip(&self.0) {
            *o = p.extra[k];
        }
    }
}

/// Several bases side by side.
pub struct Concat(pub Vec<Box<dyn Basis>>);

impl Basis for Concat {
    fn len(&self) -> usize {
        self.0.iter().map(|b| b.len()).sum()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        let mut at = 0;
        for b in &self.0 {
            b.eval(p, &mut out[at..at + b.len()]);
            at += b.len();
        }
    }
}

/// An orthonormal field basis `φ`: complex monomials with the rig-typical
/// camera's directions removed, Gram-Schmidt over a coverage grid (uniform
/// weight per covered cell), so a coefficient is the px RMS of its field
/// over the covered image. `φ_j = Σ coef_mono·mono + Σ coef_cam·cfield`.
#[derive(Clone)]
pub struct Phi {
    pub mono: CMono,
    /// `n_phi × (2·n_mono + NCF)`.
    pub coef: DMatrix<f64>,
    /// The monomial each `φ_j` was generated from: `(p, q, re|im)`.
    pub labels: Vec<(i32, i32, bool)>,
}

impl Phi {
    /// Build from the covered cells' centres and the typical camera's
    /// directions there.
    pub fn build(mono: CMono, cells: &[([f64; 2], [[f64; 2]; NCF])]) -> Self {
        let nm = 2 * mono.terms.len();
        let ncol = nm + NCF;
        let nc = cells.len();
        // Columns on the grid, inner product (1/nc) Σ <a, b>.
        let mut g = DMatrix::<f64>::zeros(2 * nc, ncol);
        let mut buf = vec![[0.0; 2]; nm];
        for (i, (u, cf)) in cells.iter().enumerate() {
            mono.value(*u, &mut buf);
            for k in 0..nm {
                g[(2 * i, k)] = buf[k][0];
                g[(2 * i + 1, k)] = buf[k][1];
            }
            for k in 0..NCF {
                g[(2 * i, nm + k)] = cf[k][0];
                g[(2 * i + 1, nm + k)] = cf[k][1];
            }
        }
        g /= (nc as f64).sqrt();
        // Modified Gram-Schmidt with coefficient tracking: the camera's
        // directions first (deflation only), then the monomials.
        let mut q: Vec<DVector<f64>> = Vec::new();
        let mut t: Vec<DVector<f64>> = Vec::new();
        let mut is_phi = Vec::new();
        let mut labels = Vec::new();
        let order: Vec<usize> = (nm..ncol).chain(0..nm).collect();
        for &j in &order {
            let col = g.column(j).into_owned();
            let n0 = col.norm();
            if n0 == 0.0 {
                continue;
            }
            let mut v = col.clone();
            let mut c = DVector::<f64>::zeros(ncol);
            c[j] = 1.0;
            for _ in 0..2 {
                for (qk, tk) in q.iter().zip(&t) {
                    let a = qk.dot(&v);
                    v -= qk * a;
                    c -= tk * a;
                }
            }
            let n = v.norm();
            if n < 1e-4 * n0 {
                continue;
            }
            q.push(v / n);
            t.push(c / n);
            let mono_col = j < nm;
            is_phi.push(mono_col);
            if mono_col {
                let (p, qq) = mono.terms[j / 2];
                labels.push((p, qq, j % 2 == 0));
            }
        }
        let rows: Vec<&DVector<f64>> = t
            .iter()
            .zip(&is_phi)
            .filter(|x| *x.1)
            .map(|x| x.0)
            .collect();
        let coef = DMatrix::from_fn(rows.len(), ncol, |i, j| rows[i][j]);
        Self { mono, coef, labels }
    }

    pub fn n(&self) -> usize {
        self.coef.nrows()
    }

    /// The fields at pixel `u` with camera directions `cf`.
    pub fn value(&self, u: [f64; 2], cf: &[[f64; 2]; NCF], out: &mut [[f64; 2]]) {
        let nm = 2 * self.mono.terms.len();
        let mut buf = vec![[0.0; 2]; nm];
        self.mono.value(u, &mut buf);
        for (j, o) in out.iter_mut().enumerate().take(self.n()) {
            let mut s = [0.0; 2];
            for (k, col) in buf.iter().chain(cf.iter()).enumerate() {
                let a = self.coef[(j, k)];
                s[0] += a * col[0];
                s[1] += a * col[1];
            }
            *o = s;
        }
    }

    /// Degree and rotation order of `φ_j`'s generating monomial.
    pub fn order(&self, j: usize) -> (i32, i32) {
        let (p, q, _) = self.labels[j];
        (p + q, p - q - 1)
    }
}

impl Basis for Phi {
    fn len(&self) -> usize {
        self.n()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        self.value(p.u, &p.cfield, out);
    }
}

/// Fields `Σ_l w_jl·φ_l` (rows of `w`): an empirical basis in `φ`
/// coordinates.
pub struct Combo {
    pub phi: Phi,
    pub w: DMatrix<f64>,
}

impl Basis for Combo {
    fn len(&self) -> usize {
        self.w.nrows()
    }
    fn eval(&self, p: &LinPoint, out: &mut [[f64; 2]]) {
        let mut buf = vec![[0.0; 2]; self.phi.n()];
        self.phi.eval(p, &mut buf);
        for (j, o) in out.iter_mut().enumerate().take(self.w.nrows()) {
            let mut s = [0.0; 2];
            for (l, b) in buf.iter().enumerate() {
                s[0] += self.w[(j, l)] * b[0];
                s[1] += self.w[(j, l)] * b[1];
            }
            *o = s;
        }
    }
}

// ───────────────────────── fits ─────────────────────────

/// What a linearized fit frees besides the poses.
pub struct Spec<'a> {
    /// The eight camera parameters.
    pub camera_free: bool,
    /// Per-camera coefficients.
    pub local: Option<&'a dyn Basis>,
    /// Penalty on the local block (added to the normal matrix).
    pub penalty: Option<&'a DMatrix<f64>>,
    /// Hold the local block orthogonal (over the training points) to the
    /// camera's directions and a common rotation.
    pub ortho: bool,
    /// Coefficients shared by every camera of the fit.
    pub shared: Option<&'a dyn Basis>,
    /// A term held at given coefficients (subtracted from the residual).
    pub held: Option<(&'a dyn Basis, &'a DVector<f64>)>,
}

#[cfg(test)]
impl Spec<'_> {
    pub fn route() -> Spec<'static> {
        Spec {
            camera_free: true,
            local: None,
            penalty: None,
            ortho: false,
            shared: None,
            held: None,
        }
    }
}

/// One camera's part of a solution.
#[derive(Debug, Clone)]
pub struct CamSol {
    pub dtheta: [f64; NCAM],
    pub local: DVector<f64>,
    /// Covariance (unit σ) of `[θ | local | shared]` over the free entries
    /// of this camera, and σ² of its residual.
    pub cov: DMatrix<f64>,
    pub sigma2: f64,
}

#[derive(Debug, Clone)]
pub struct Sol {
    pub cams: Vec<CamSol>,
    pub shared: DVector<f64>,
}

fn sym_solve(h: &DMatrix<f64>, b: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let n = h.nrows();
    if n == 0 {
        return (DVector::zeros(0), DMatrix::zeros(0, 0));
    }
    let tr = (0..n).map(|i| h[(i, i)].abs()).sum::<f64>() / n as f64;
    let mut hr = h.clone();
    for i in 0..n {
        hr[(i, i)] += 1e-12 * tr.max(1e-300);
    }
    if let Some(ch) = hr.clone().cholesky() {
        let inv = ch.inverse();
        (&inv * b, inv)
    } else {
        let e = nalgebra::SymmetricEigen::new(hr);
        let mx = e.eigenvalues.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut inv = DMatrix::zeros(n, n);
        for k in 0..n {
            let l = e.eigenvalues[k];
            if l > 1e-12 * mx {
                let v = e.eigenvectors.column(k);
                inv += (v * v.transpose()) / l;
            }
        }
        (&inv * b, inv)
    }
}

/// A camera's reduced normal equations over `[θ | local | shared]`.
struct Reduced {
    h: DMatrix<f64>,
    b: DVector<f64>,
    /// Local block = `n · z` when orthogonal.
    null: Option<DMatrix<f64>>,
    n_theta: usize,
    n_local: usize,
    n_obs: usize,
    n_views: usize,
}

fn columns(p: &LinPoint, spec: &Spec<'_>, buf: &mut [[f64; 2]]) -> usize {
    let mut at = 0;
    if spec.camera_free {
        buf[..NCAM].copy_from_slice(&p.jcam);
        at = NCAM;
    }
    if let Some(l) = spec.local {
        l.eval(p, &mut buf[at..at + l.len()]);
        at += l.len();
    }
    if let Some(s) = spec.shared {
        s.eval(p, &mut buf[at..at + s.len()]);
        at += s.len();
    }
    at
}

fn held_shift(p: &LinPoint, spec: &Spec<'_>, buf: &mut Vec<[f64; 2]>) -> [f64; 2] {
    let Some((b, c)) = spec.held else {
        return [0.0; 2];
    };
    buf.resize(b.len(), [0.0; 2]);
    b.eval(p, buf);
    let mut s = [0.0; 2];
    for (col, a) in buf.iter().zip(c.iter()) {
        s[0] += a * col[0];
        s[1] += a * col[1];
    }
    s
}

fn by_view(pts: &[LinPoint]) -> BTreeMap<usize, Vec<usize>> {
    let mut m: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, p) in pts.iter().enumerate() {
        m.entry(p.view).or_default().push(i);
    }
    m
}

fn reduce(pts: &[LinPoint], spec: &Spec<'_>) -> Reduced {
    let n_theta = if spec.camera_free { NCAM } else { 0 };
    let n_local = spec.local.map_or(0, |l| l.len());
    let n_shared = spec.shared.map_or(0, |s| s.len());
    let ng = n_theta + n_local + n_shared;
    let mut h = DMatrix::<f64>::zeros(ng, ng);
    let mut b = DVector::<f64>::zeros(ng);
    let mut buf = vec![[0.0; 2]; ng];
    let mut hbuf = Vec::new();
    let views = by_view(pts);
    let mut nz: Vec<usize> = Vec::with_capacity(ng);
    // Per view: the pose block against the columns the view touches.
    let mut slot = vec![usize::MAX; ng];
    for idx in views.values() {
        let mut hpp = nalgebra::Matrix6::<f64>::zeros();
        let mut cols: Vec<usize> = Vec::new();
        let mut hpg: Vec<[f64; 6]> = Vec::new();
        let mut bp = nalgebra::Vector6::<f64>::zeros();
        for &i in idx {
            let p = &pts[i];
            columns(p, spec, &mut buf);
            nz.clear();
            nz.extend((0..ng).filter(|&a| buf[a] != [0.0, 0.0]));
            let sh = held_shift(p, spec, &mut hbuf);
            let r = [p.r[0] - sh[0], p.r[1] - sh[1]];
            let w = p.w;
            for &a in &nz {
                if slot[a] == usize::MAX {
                    slot[a] = cols.len();
                    cols.push(a);
                    hpg.push([0.0; 6]);
                }
            }
            for c in 0..2 {
                for (ia, &a) in nz.iter().enumerate() {
                    let ja = buf[a][c];
                    b[a] += w * ja * r[c];
                    for &bb in &nz[ia..] {
                        let (lo, hi) = if a <= bb { (a, bb) } else { (bb, a) };
                        h[(lo, hi)] += w * ja * buf[bb][c];
                    }
                    for (g, jp) in hpg[slot[a]].iter_mut().zip(&p.jpose) {
                        *g += w * jp[c] * ja;
                    }
                }
                for m in 0..6 {
                    bp[m] += w * p.jpose[m][c] * r[c];
                    for n in 0..6 {
                        hpp[(m, n)] += w * p.jpose[m][c] * p.jpose[n][c];
                    }
                }
            }
        }
        for &a in &cols {
            slot[a] = usize::MAX;
        }
        let Some(k) = hpp.try_inverse() else {
            continue;
        };
        let kb = k * bp;
        let kg: Vec<nalgebra::Vector6<f64>> = hpg
            .iter()
            .map(|g| k * nalgebra::Vector6::from_column_slice(g))
            .collect();
        for (ia, &a) in cols.iter().enumerate() {
            let ga = nalgebra::Vector6::from_column_slice(&hpg[ia]);
            b[a] -= ga.dot(&kb);
            for (ib, &bb) in cols.iter().enumerate() {
                if bb >= a {
                    h[(a, bb)] -= ga.dot(&kg[ib]);
                }
            }
        }
    }
    for a in 0..ng {
        for bb in 0..a {
            h[(a, bb)] = h[(bb, a)];
        }
    }
    if let (Some(pen), true) = (spec.penalty, n_local > 0) {
        let mut blk = h.view_mut((n_theta, n_theta), (n_local, n_local));
        blk += pen;
    }
    let mut null = None;
    if spec.ortho && n_local > 0 {
        // Rows A_j = Σ w J_jᵀ B over the points.
        let l = spec.local.expect("a local block");
        let mut a = DMatrix::<f64>::zeros(NCF, n_local);
        let mut lb = vec![[0.0; 2]; n_local];
        for p in pts {
            l.eval(p, &mut lb);
            let dirs: Vec<[f64; 2]> = p.jcam.iter().chain(&p.jpose[..3]).copied().collect();
            for (j, d) in dirs.iter().enumerate() {
                for (k, bk) in lb.iter().enumerate() {
                    a[(j, k)] += p.w * (d[0] * bk[0] + d[1] * bk[1]);
                }
            }
        }
        for mut row in a.row_iter_mut() {
            let n = row.norm();
            if n > 0.0 {
                row /= n;
            }
        }
        let e = nalgebra::SymmetricEigen::new(a.transpose() * &a);
        let mx = e.eigenvalues.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let keep: Vec<usize> = (0..n_local)
            .filter(|&k| e.eigenvalues[k] < 1e-10 * mx)
            .collect();
        let nmat = DMatrix::from_fn(n_local, keep.len(), |i, j| e.eigenvectors[(i, keep[j])]);
        let nz = keep.len();
        let ng2 = n_theta + nz + n_shared;
        let mut t = DMatrix::<f64>::zeros(ng, ng2);
        for i in 0..n_theta {
            t[(i, i)] = 1.0;
        }
        t.view_mut((n_theta, n_theta), (n_local, nz))
            .copy_from(&nmat);
        for i in 0..n_shared {
            t[(n_theta + n_local + i, n_theta + nz + i)] = 1.0;
        }
        h = t.transpose() * &h * &t;
        b = t.transpose() * &b;
        null = Some(nmat);
    }
    Reduced {
        h,
        b,
        null,
        n_theta,
        n_local,
        n_obs: 2 * pts.len(),
        n_views: views.len(),
    }
}

/// Fit `spec` jointly over the cameras' point sets (shared coefficients
/// coupled, everything else per camera).
pub fn fit(sets: &[&[LinPoint]], spec: &Spec<'_>) -> Sol {
    let n_shared = spec.shared.map_or(0, |s| s.len());
    let reds: Vec<Reduced> = sets.iter().map(|pts| reduce(pts, spec)).collect();
    let own: Vec<usize> = reds.iter().map(|r| r.h.nrows() - n_shared).collect();
    let n_tot = own.iter().sum::<usize>() + n_shared;
    let mut h = DMatrix::<f64>::zeros(n_tot, n_tot);
    let mut b = DVector::<f64>::zeros(n_tot);
    let mut off = 0;
    let s0 = n_tot - n_shared;
    for (r, &n) in reds.iter().zip(&own) {
        let map = |i: usize| if i < n { off + i } else { s0 + i - n };
        for i in 0..r.h.nrows() {
            b[map(i)] += r.b[i];
            for j in 0..r.h.ncols() {
                h[(map(i), map(j))] += r.h[(i, j)];
            }
        }
        off += n;
    }
    let (x, inv) = sym_solve(&h, &b);
    let shared = x.rows(s0, n_shared).into_owned();
    let mut cams = Vec::new();
    let mut off = 0;
    for (pts, (r, &n)) in sets.iter().zip(reds.iter().zip(&own)) {
        let mut dtheta = [0.0; NCAM];
        for (k, d) in dtheta.iter_mut().enumerate().take(r.n_theta) {
            *d = x[off + k];
        }
        let z = x.rows(off + r.n_theta, n - r.n_theta).into_owned();
        let local = match &r.null {
            Some(nm) => nm * &z,
            None => z,
        };
        // Covariance over [θ | local | shared], local mapped from z.
        let idx: Vec<usize> = (off..off + n).chain(s0..s0 + n_shared).collect();
        let sub = DMatrix::from_fn(idx.len(), idx.len(), |i, j| inv[(idx[i], idx[j])]);
        let cov = match &r.null {
            Some(nm) => {
                let nz = nm.ncols();
                let dim = r.n_theta + r.n_local + n_shared;
                let mut t = DMatrix::<f64>::zeros(dim, idx.len());
                for i in 0..r.n_theta {
                    t[(i, i)] = 1.0;
                }
                t.view_mut((r.n_theta, r.n_theta), (r.n_local, nz))
                    .copy_from(nm);
                for i in 0..n_shared {
                    t[(r.n_theta + r.n_local + i, r.n_theta + nz + i)] = 1.0;
                }
                &t * sub * t.transpose()
            }
            None => sub,
        };
        let res = predict(
            pts,
            &dtheta,
            spec.local.map(|l| (l, &local)),
            spec.shared.map(|s| (s, &shared)),
            spec.held,
        );
        let rss: f64 = res
            .iter()
            .zip(pts.iter())
            .map(|(e, p)| p.w * (e[0] * e[0] + e[1] * e[1]))
            .sum();
        let dof = (r.n_obs as f64 - 6.0 * r.n_views as f64 - (n + n_shared) as f64).max(1.0);
        cams.push(CamSol {
            dtheta,
            local,
            cov,
            sigma2: rss / dof,
        });
        off += n;
    }
    Sol { cams, shared }
}

/// Residuals of `pts` after the camera change, the terms, and a pose-only
/// refit per view.
pub fn predict(
    pts: &[LinPoint],
    dtheta: &[f64; NCAM],
    local: Option<(&dyn Basis, &DVector<f64>)>,
    shared: Option<(&dyn Basis, &DVector<f64>)>,
    held: Option<(&dyn Basis, &DVector<f64>)>,
) -> Vec<[f64; 2]> {
    let mut e: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    let mut buf = Vec::new();
    for p in pts {
        let mut r = p.r;
        for (k, d) in dtheta.iter().enumerate() {
            r[0] -= p.jcam[k][0] * d;
            r[1] -= p.jcam[k][1] * d;
        }
        for (b, c) in [local, shared, held].into_iter().flatten() {
            buf.resize(b.len(), [0.0; 2]);
            b.eval(p, &mut buf);
            for (col, a) in buf.iter().zip(c.iter()) {
                r[0] -= a * col[0];
                r[1] -= a * col[1];
            }
        }
        e.push(r);
    }
    for idx in by_view(pts).values() {
        let mut hpp = nalgebra::Matrix6::<f64>::zeros();
        let mut bp = nalgebra::Vector6::<f64>::zeros();
        for &i in idx {
            let p = &pts[i];
            for (c, ec) in e[i].iter().enumerate() {
                for m in 0..6 {
                    bp[m] += p.w * p.jpose[m][c] * ec;
                    for n in 0..6 {
                        hpp[(m, n)] += p.w * p.jpose[m][c] * p.jpose[n][c];
                    }
                }
            }
        }
        let Some(inv) = hpp.try_inverse() else {
            continue;
        };
        let dp = inv * bp;
        for &i in idx {
            let p = &pts[i];
            for (c, ec) in e[i].iter_mut().enumerate() {
                *ec -= (0..6).map(|m| p.jpose[m][c] * dp[m]).sum::<f64>();
            }
        }
    }
    e
}

/// RMS, mean and median of the residual norms.
pub fn stats(e: &[[f64; 2]]) -> [f64; 3] {
    let mut n: Vec<f64> = e.iter().map(|r| r[0].hypot(r[1])).collect();
    let k = n.len().max(1) as f64;
    let rms = (n.iter().map(|v| v * v).sum::<f64>() / k).sqrt();
    let mean = n.iter().sum::<f64>() / k;
    n.sort_by(f64::total_cmp);
    [rms, mean, n.get(n.len() / 2).copied().unwrap_or(f64::NAN)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{self, Free};
    use crate::synth::{self, SIZE};

    /// The linearized fit of the plain model at a perturbed camera lands
    /// where the nonlinear fit does, and a camera-held linearized
    /// prediction reproduces the nonlinear pose-only residual.
    #[test]
    fn linearized_fit_matches_the_nonlinear_one() {
        let plain = GenCfg::base("plain");
        let truth = synth::typical();
        let (cam, poses) =
            synth::scene_with_poses("t", &plain, &truth, 8, (4, 4), 0.07, &|_| [0.0; 2]).unwrap();
        let ids: Vec<usize> = (0..cam.dataset.num_views()).collect();
        let exact = model::fit(
            &plain,
            &ids,
            &cam.dataset.views,
            &truth,
            &poses,
            SIZE,
            Free::All,
            100,
        )
        .unwrap();
        // Start a little off the optimum: one linear step must recover it.
        let mut start = exact.params.clone();
        start.intr[0] += 0.3;
        start.tilt[0] += 2e-4;
        let pose_fit = model::fit(
            &plain,
            &ids,
            &cam.dataset.views,
            &start,
            &poses,
            SIZE,
            Free::PosesOnly,
            100,
        )
        .unwrap();
        let pts = linearize(
            &start,
            &truth,
            SIZE,
            &ids,
            &cam.dataset.views,
            &pose_fit.poses,
        );
        let sol = fit(&[&pts], &Spec::route());
        let d = sol.cams[0].dtheta;
        assert!(
            (start.intr[0] + d[0] - exact.params.intr[0]).abs() < 0.02,
            "fx {} vs {}",
            start.intr[0] + d[0],
            exact.params.intr[0]
        );
        assert!((start.tilt[0] + d[6] - exact.params.tilt[0]).abs() < 2e-5);
        let res = predict(&pts, &d, None, None, None);
        let lin = stats(&res)[0];
        let nl = crate::rms_median(&exact.residuals).0;
        assert!((lin - nl).abs() < 2e-3 * nl, "{lin} vs {nl}");
    }

    /// `φ` is orthonormal over its grid and orthogonal to the camera's
    /// directions there.
    #[test]
    fn phi_is_orthonormal_and_free_of_the_camera() {
        let typ = synth::typical();
        let mut cells = Vec::new();
        for j in 0..18 {
            for i in 0..24 {
                let u = [15.0 + 30.0 * i as f64, 15.0 + 30.0 * j as f64];
                cells.push((u, cam_fields(&typ, SIZE, u)));
            }
        }
        let mono = CMono::degrees(0, 4, [360.0, 270.0], 360.0);
        let phi = Phi::build(mono, &cells);
        let n = phi.n();
        assert!(n > 10 && n < 30, "{n}");
        let mut g = DMatrix::<f64>::zeros(n, n);
        let mut c = DMatrix::<f64>::zeros(n, NCF);
        let mut buf = vec![[0.0; 2]; n];
        for (u, cf) in &cells {
            phi.value(*u, cf, &mut buf);
            for a in 0..n {
                for b in 0..n {
                    g[(a, b)] += buf[a][0] * buf[b][0] + buf[a][1] * buf[b][1];
                }
                for k in 0..NCF {
                    c[(a, k)] += buf[a][0] * cf[k][0] + buf[a][1] * cf[k][1];
                }
            }
        }
        g /= cells.len() as f64;
        assert!((g - DMatrix::identity(n, n)).abs().max() < 1e-8);
        assert!(c.abs().max() < 1e-6 * cells.len() as f64);
    }
}
