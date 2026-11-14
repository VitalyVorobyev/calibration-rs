//! An experimental tilted-camera model written as a user residual, so terms
//! the library does not have can be tried against the data: a distortion
//! centre apart from the principal point, the tilt applied before the
//! distortion, a depth-dependent `k1`, an out-of-plane board shape, a free
//! image-space field (an upper bound for any central-camera refinement), and
//! a principal point measured from the tilt's valley.
//!
//! The baseline configuration reproduces the library's Brown-Conrady +
//! Scheimpflug projection exactly (checked against it in `main`).

use std::collections::BTreeMap;

use anyhow::Result;
use nalgebra::{DVector, Matrix3, Point3, RealField, Vector3};
use vision_calibration_core::{Iso3, NoMeta, Uncertainty, View};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{
    BlockId, Manifold, Noise, Problem, Residual, SolveOptions, se3, solve,
};

use crate::PointResidual;

/// Distortion coefficient names, in block order.
pub const DIST_NAMES: [&str; 9] = ["k1", "k2", "k3", "p1", "p2", "s1", "s2", "s3", "s4"];
/// Reference depth of the depth-dependent `k1`, metres.
const Z0_M: f64 = 0.2;
/// Half-size of the board's normalized coordinates, metres.
const BOARD_HALF_M: f64 = 0.06;

/// Which terms a fit has.
#[derive(Debug, Clone)]
pub struct GenCfg {
    pub name: &'static str,
    /// Free distortion coefficients (names from [`DIST_NAMES`]).
    pub free_dist: &'static [&'static str],
    /// A distortion centre offset from the principal point (normalized).
    pub dc: bool,
    /// The tilt homography applied to the undistorted point.
    pub tilt_first: bool,
    /// `k1 + kz·(z0/Z − 1)`.
    pub depth: bool,
    /// The tilt scaled by `1 + η·r_d²` over the field (a field-dependent
    /// exit-pupil distance).
    pub pupil: bool,
    /// Lowest degree of the free image-space field.
    pub field_min_deg: usize,
    /// Hold the tilt at its initial value.
    pub fix_tilt: bool,
    /// Hold `fx fy cx cy` at their initial values.
    pub fix_intr: bool,
    /// Hold every distortion coefficient at its initial value.
    pub fix_dist: bool,
    /// Degree of the out-of-plane board shape (terms of degree 2..=deg).
    pub board_deg: usize,
    /// Degree of the free image-space field (terms of degree 2..=deg).
    pub field_deg: usize,
    /// Evaluate the field's basis at the observed pixel rather than at the
    /// model's own (pre-field) projection. Not a camera model: the cost then
    /// depends on the observation through the basis too, and a field whose
    /// Jacobian approaches the identity contracts the measurement noise
    /// instead of explaining it (errors in variables). Kept to measure that.
    pub field_at_obs: bool,
    /// Store `c̃ = c − s·f·tan(tilt)` in place of the principal point; the
    /// signs `(s_x, s_y)` pair `cx` with `tilt_y` and `cy` with `tilt_x`.
    pub reparam: Option<(f64, f64)>,
    /// A displacement in the sensor plane, radial from a centre: what a
    /// sensor surface that is not flat (or a cover glass parallel to it)
    /// does to a ray arriving from the exit pupil.
    pub sensor: SensorTerms,
    /// The centre of the sensor-plane terms.
    pub sensor_centre: Centre,
    /// An elliptic radial metric for `k1`:
    /// `r² + e1·(x² − y²) + e2·2xy` (astigmatic distortion).
    pub elliptic: bool,
    /// A bicubic B-spline displacement over the image, `[cells_x, cells_y]`
    /// uniform cells (`[0, 0]`: none), evaluated at the parametric
    /// projection and held at the border value outside the image.
    pub spline: [usize; 2],
    /// Smoothness prior of the spline: σ (px) of each control point's
    /// second difference along x and along y; `0` for none.
    pub spline_sigma: f64,
    /// Keep the spline orthogonal, on the fitted points, to the pixel
    /// directions of the free camera parameters and of a rotation common to
    /// every view, at the starting solution (a stiff linear penalty).
    pub spline_ortho: bool,
    /// Pass the smoothness prior to the covariance as a residual family
    /// whose scatter is estimated (a penalty), not as known information:
    /// the sandwich then counts only the measurement noise, not the prior's
    /// stated σ, in the estimate's scatter.
    pub spline_penalty: bool,
    /// Measure the orthogonality over the covered image (one sample per
    /// covered cell of a 24 × 18 grid, uniform weight) instead of over the
    /// fitted points, so the camera/field split does not follow the
    /// points' density.
    pub spline_ortho_cover: bool,
    /// Lens-side terms: complex displacements of the distorted normalized
    /// point before the tilt, functions of the undistorted point `w` about
    /// the distortion centre (see [`LensTerm`]), two parameters each.
    pub lens: &'static [LensTerm],
    /// Components of the lens block held at their initial value.
    pub lens_fix: &'static [usize],
}

/// A lens-side displacement `Δ(w)` in the normalized plane, `w = x + iy` the
/// undistorted point about the distortion centre, with one complex
/// coefficient `c` (parameters `Re c, Im c`).
///
/// A term `w^p·w̄^q` turns by `m = p − q − 1` under a rotation of the image
/// about the axis. A system of rotationally symmetric surfaces, each
/// decentred by a small vector `σ_j` (a tilted surface is a decentred
/// centre of curvature), can only produce terms `w^p w̄^q σ^a σ̄^b` with
/// `p − q + a − b = 1`: the derivation of the linked forms below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensTerm {
    /// `c·w^p·w̄^q`.
    Mono(i32, i32),
    /// Fifth-order distortion at second order in the decentrations:
    /// `|w − s|⁴(w − s)` expanded gives `s̄²w³ + 6|s|²w²w̄ + 3s²ww̄²`; the
    /// radial middle term is `k1`, the rest is `c̄·w³ + 3c·w·w̄²`
    /// (`m = ±2`, one complex coefficient `c = Σ W511_j s_j²`).
    NatB5,
    /// Fifth-order distortion at first order in the decentrations:
    /// `−(3c·w²w̄² + 2c̄·w³w̄)` (`m = ∓1` at degree 4, `c = Σ W511_j s_j`).
    NatA5,
}

/// `(a + ib)·(c + id)`.
fn cmul<T: RealField>(a: (T, T), b: (T, T)) -> (T, T) {
    (
        a.0.clone() * b.0.clone() - a.1.clone() * b.1.clone(),
        a.0 * b.1 + a.1 * b.0,
    )
}

/// `w^p·w̄^q`.
fn cmono<T: RealField>(w: &(T, T), p: i32, q: i32) -> (T, T) {
    let mut out = (T::one(), T::zero());
    for _ in 0..p {
        out = cmul(out, w.clone());
    }
    let wb = (w.0.clone(), -w.1.clone());
    for _ in 0..q {
        out = cmul(out, wb.clone());
    }
    out
}

/// The displacement of `term` with coefficient `c` at `w`.
pub fn lens_shift<T: RealField>(term: LensTerm, c: (T, T), w: &(T, T)) -> (T, T) {
    let k = |v: f64| T::from_f64(v).unwrap();
    let cb = (c.0.clone(), -c.1.clone());
    match term {
        LensTerm::Mono(p, q) => cmul(c, cmono(w, p, q)),
        LensTerm::NatB5 => {
            let a = cmul(cb, cmono(w, 3, 0));
            let b = cmul(c, cmono(w, 1, 2));
            (a.0 + k(3.0) * b.0, a.1 + k(3.0) * b.1)
        }
        LensTerm::NatA5 => {
            let a = cmul(c, cmono(w, 2, 2));
            let b = cmul(cb, cmono(w, 3, 1));
            (
                -(k(3.0) * a.0 + k(2.0) * b.0),
                -(k(3.0) * a.1 + k(2.0) * b.1),
            )
        }
    }
}

/// The sensor-plane term: `Δs = −h(ρ̃)·ρ̃·ρ_ref·1e-3`, with `ρ = s − s₀` the
/// position on the (tilted) sensor from the centre, `ρ̃ = ρ / ρ_ref`, and
/// `h` a height in thousandths of the pupil-to-sensor distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorTerms {
    Off,
    /// `h = Σ_{k=1..n} c_k·|ρ̃|^{2k}`: radial about the centre (a bowl; a
    /// cover glass gives `n = 1` to third order).
    Radial(usize),
    /// `h` a polynomial of degree 2..=d in `ρ̃` (constant and linear
    /// heights are the sensor distance and the tilt).
    Poly(usize),
}

impl SensorTerms {
    fn len(self) -> usize {
        match self {
            SensorTerms::Off => 0,
            SensorTerms::Radial(n) => n,
            SensorTerms::Poly(d) => monomials(2, d).len(),
        }
    }
}

/// Where the sensor-plane terms are centred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Centre {
    /// The foot of the perpendicular from the projection centre onto the
    /// tilted sensor: set by the tilt, no parameter.
    Foot,
    /// The principal point (`s = 0`).
    Principal,
    /// Free, started at the foot.
    Free,
}

/// Normalizing radius of the sensor-plane terms (normalized units, about
/// the half-width of a 720 px tile at `f ≈ 1150 px`).
pub const RHO_REF: f64 = 0.3;

/// The foot of the perpendicular from the projection centre onto the
/// sensor of [`tilt_h`], in its sensor coordinates: `−(r₀₂, r₁₂)` of
/// `R = Ry(τy)·Rx(τx)`.
pub fn foot<T: RealField>(tx: T, ty: T) -> (T, T) {
    let r02 = -(ty.clone().sin()) * tx.clone().cos();
    let r12 = tx.sin();
    (-r02, -r12)
}

/// The sensor-plane displacement at sensor position `(x, y)` from centre
/// `(cx0, cy0)`, coefficients `c`.
pub fn sensor_shift<T: RealField>(
    terms: SensorTerms,
    c: &[T],
    x: T,
    y: T,
    cx0: T,
    cy0: T,
) -> (T, T) {
    let k = |v: f64| T::from_f64(v).unwrap();
    let (rx, ry) = ((x - cx0) / k(RHO_REF), (y - cy0) / k(RHO_REF));
    let mut h = T::zero();
    match terms {
        SensorTerms::Off => {}
        SensorTerms::Radial(n) => {
            let r2 = rx.clone() * rx.clone() + ry.clone() * ry.clone();
            let mut p = r2.clone();
            for ci in c.iter().take(n) {
                h += ci.clone() * p.clone();
                p *= r2.clone();
            }
        }
        SensorTerms::Poly(d) => {
            for (ci, (i, j)) in c.iter().zip(monomials(2, d)) {
                h += ci.clone() * rx.clone().powi(i) * ry.clone().powi(j);
            }
        }
    }
    let s = -(h * k(RHO_REF * 1e-3));
    (s.clone() * rx, s * ry)
}

impl GenCfg {
    pub const fn base(name: &'static str) -> Self {
        Self {
            name,
            free_dist: &["k1", "k2"],
            dc: false,
            tilt_first: false,
            depth: false,
            pupil: false,
            field_min_deg: 2,
            fix_tilt: false,
            fix_intr: false,
            fix_dist: false,
            board_deg: 0,
            field_deg: 0,
            field_at_obs: false,
            reparam: None,
            sensor: SensorTerms::Off,
            sensor_centre: Centre::Foot,
            elliptic: false,
            spline: [0, 0],
            spline_sigma: 0.0,
            spline_ortho: false,
            spline_penalty: false,
            spline_ortho_cover: false,
            lens: &[],
            lens_fix: &[],
        }
    }
}

pub fn monomials(min: usize, deg: usize) -> Vec<(i32, i32)> {
    let mut m = Vec::new();
    for d in min as i32..=deg as i32 {
        for i in (0..=d).rev() {
            m.push((i, d - i));
        }
    }
    m
}

/// Model values, physical (the principal point is the real one).
#[derive(Debug, Clone)]
pub struct GenParams {
    pub intr: [f64; 4],
    pub dist: [f64; 9],
    pub tilt: [f64; 2],
    pub dc: [f64; 2],
    pub depth: f64,
    pub pupil: f64,
    pub board: Vec<f64>,
    pub field: Vec<f64>,
    /// Sensor-plane coefficients ([`SensorTerms`]).
    pub sensor: Vec<f64>,
    /// `e1, e2` (elliptic metric) and the sensor centre's offset from the
    /// foot (normalized), in that order.
    pub misc: [f64; 4],
    /// Spline control points (px): every `u` component, then every `v`.
    pub spline: Vec<f64>,
    /// Lens-side coefficients ([`LensTerm`]), two per term.
    pub lens: Vec<f64>,
    /// Board centre (metres) the board shape is expanded about.
    pub board_centre: [f64; 2],
}

/// OpenCV's tilted-sensor homography `proj_z · Ry(τy) · Rx(τx)`.
pub fn tilt_h<T: RealField>(tx: T, ty: T) -> Matrix3<T> {
    let (sx, cx) = (tx.clone().sin(), tx.cos());
    let (sy, cy) = (ty.clone().sin(), ty.cos());
    let (z, o) = (T::zero(), T::one());
    let rx = Matrix3::new(
        o.clone(),
        z.clone(),
        z.clone(),
        z.clone(),
        cx.clone(),
        sx.clone(),
        z.clone(),
        -sx,
        cx,
    );
    let ry = Matrix3::new(
        cy.clone(),
        z.clone(),
        -sy.clone(),
        z.clone(),
        o.clone(),
        z.clone(),
        sy,
        z.clone(),
        cy,
    );
    let r = ry * rx;
    let r22 = r[(2, 2)].clone();
    let p = Matrix3::new(
        r22.clone(),
        z.clone(),
        -r[(0, 2)].clone(),
        z.clone(),
        r22,
        -r[(1, 2)].clone(),
        z.clone(),
        z,
        o,
    );
    p * r
}

pub fn apply_h<T: RealField>(h: &Matrix3<T>, x: T, y: T) -> (T, T) {
    let q = h * Vector3::new(x, y, T::one());
    (q.x.clone() / q.z.clone(), q.y.clone() / q.z.clone())
}

/// The projection of one board point; see [`GenCfg`].
#[derive(Clone)]
pub struct GenResidual {
    board: [f64; 2],
    observed: [f64; 2],
    /// Normalized observed position, for the field basis.
    obs_norm: [f64; 2],
    /// Half the image size: the field basis is in `(u − w/2)/(w/2)`.
    half: [f64; 2],
    field_at_obs: bool,
    board_terms: Vec<(i32, i32)>,
    field_terms: Vec<(i32, i32)>,
    board_centre: [f64; 2],
    dc: bool,
    tilt_first: bool,
    depth: bool,
    pupil: bool,
    reparam: Option<(f64, f64)>,
    sensor: SensorTerms,
    sensor_centre: Centre,
    elliptic: bool,
    spline: [usize; 2],
    size: [f64; 2],
    lens: &'static [LensTerm],
}

/// Control points per axis of a uniform cubic B-spline over `cells` cells.
pub fn spline_len(cells: [usize; 2]) -> usize {
    if cells[0] == 0 || cells[1] == 0 {
        0
    } else {
        (cells[0] + 3) * (cells[1] + 3)
    }
}

/// The uniform cubic B-spline weights of `x` over `[0, len]` in `cells`
/// cells: the first control index and the four weights. Outside the range
/// the border value holds (the argument is clamped).
pub fn bspline_axis<T: RealField>(x: T, len: f64, cells: usize) -> (usize, [T; 4]) {
    let k = |v: f64| T::from_f64(v).unwrap();
    let mut t = x / k(len / cells as f64);
    if t < T::zero() {
        t = T::zero();
    }
    if t > k(cells as f64) {
        t = k(cells as f64);
    }
    let mut i = 0usize;
    while i + 1 < cells && t >= k((i + 1) as f64) {
        i += 1;
    }
    let f = t - k(i as f64);
    let f2 = f.clone() * f.clone();
    let f3 = f2.clone() * f.clone();
    let g = T::one() - f.clone();
    let w = [
        g.clone() * g.clone() * g / k(6.0),
        (k(3.0) * f3.clone() - k(6.0) * f2.clone() + k(4.0)) / k(6.0),
        (k(-3.0) * f3.clone() + k(3.0) * f2 + k(3.0) * f + T::one()) / k(6.0),
        f3 / k(6.0),
    ];
    (i, w)
}

/// The spline's displacement at pixel `(u, v)`.
pub fn spline_eval<T: RealField>(cells: [usize; 2], size: [f64; 2], c: &[T], u: T, v: T) -> (T, T) {
    let nx = cells[0] + 3;
    let n = spline_len(cells);
    let (ix, wx) = bspline_axis(u, size[0], cells[0]);
    let (iy, wy) = bspline_axis(v, size[1], cells[1]);
    let (mut du, mut dv) = (T::zero(), T::zero());
    for (b, wyb) in wy.iter().enumerate() {
        for (a, wxa) in wx.iter().enumerate() {
            let w = wxa.clone() * wyb.clone();
            let idx = (iy + b) * nx + ix + a;
            du += w.clone() * c[idx].clone();
            dv += w * c[n + idx].clone();
        }
    }
    (du, dv)
}

/// Block order: intr(4), dist(9), tilt(2), dc(2), depth(2), board(nb|1),
/// field(2·nf|1), sensor(ns|1), misc(4), spline(2·nc|1), lens(2·nl|1),
/// pose(7).
const N_BLOCKS: usize = 12;

impl GenResidual {
    fn dims(&self) -> Vec<usize> {
        vec![
            4,
            9,
            2,
            2,
            2,
            self.board_terms.len().max(1),
            (2 * self.field_terms.len()).max(1),
            self.sensor.len().max(1),
            4,
            (2 * spline_len(self.spline)).max(1),
            (2 * self.lens.len()).max(1),
            7,
        ]
    }

    /// Projected pixel at the blocks' values.
    fn project<T: RealField>(&self, b: &[DVector<T>]) -> (T, T) {
        let c = |v: f64| T::from_f64(v).unwrap();
        let (fx, fy) = (b[0][0].clone(), b[0][1].clone());
        let (tx, ty) = (b[2][0].clone(), b[2][1].clone());
        let (mut cx, mut cy) = (b[0][2].clone(), b[0][3].clone());
        if let Some((sx, sy)) = self.reparam {
            cx += c(sx) * fx.clone() * ty.clone().tan();
            cy += c(sy) * fy.clone() * tx.clone().tan();
        }
        // Board point, with the out-of-plane shape.
        let (bx, by) = (self.board[0], self.board[1]);
        let mut w = T::zero();
        if !self.board_terms.is_empty() {
            let xn = (bx - self.board_centre[0]) / BOARD_HALF_M;
            let yn = (by - self.board_centre[1]) / BOARD_HALF_M;
            for (k, &(i, j)) in self.board_terms.iter().enumerate() {
                w += b[5][k].clone() * c(1e-3 * xn.powi(i) * yn.powi(j));
            }
        }
        let pose = se3(&b[11]);
        let pc = pose * Point3::new(c(bx), c(by), w);
        let (mut x, mut y) = (pc.x.clone() / pc.z.clone(), pc.y.clone() / pc.z.clone());
        let h = tilt_h(tx.clone(), ty.clone());
        if self.tilt_first {
            (x, y) = apply_h(&h, x, y);
        }
        let d = &b[1];
        let mut k1 = d[0].clone();
        if self.depth {
            k1 += b[4][0].clone() * (c(Z0_M) / pc.z.clone() - T::one());
        }
        let (ox, oy) = if self.dc {
            (b[3][0].clone(), b[3][1].clone())
        } else {
            (T::zero(), T::zero())
        };
        let (ux, uy) = (x - ox.clone(), y - oy.clone());
        let r2 = ux.clone() * ux.clone() + uy.clone() * uy.clone();
        let r4 = r2.clone() * r2.clone();
        let r2e = if self.elliptic {
            r2.clone()
                + b[8][0].clone() * (ux.clone() * ux.clone() - uy.clone() * uy.clone())
                + c(2.0) * b[8][1].clone() * ux.clone() * uy.clone()
        } else {
            r2.clone()
        };
        let radial = T::one()
            + k1 * r2e
            + d[1].clone() * r4.clone()
            + d[2].clone() * r4.clone() * r2.clone();
        let two = c(2.0);
        let xy = ux.clone() * uy.clone();
        let mut xd = ux.clone() * radial.clone()
            + two.clone() * d[3].clone() * xy.clone()
            + d[4].clone() * (r2.clone() + two.clone() * ux.clone() * ux.clone())
            + d[5].clone() * r2.clone()
            + d[6].clone() * r4.clone()
            + ox;
        let mut yd = uy.clone() * radial
            + d[3].clone() * (r2.clone() + two.clone() * uy.clone() * uy.clone())
            + two * d[4].clone() * xy
            + d[7].clone() * r2
            + d[8].clone() * r4
            + oy;
        if !self.lens.is_empty() {
            let w = (ux.clone(), uy.clone());
            for (k, term) in self.lens.iter().enumerate() {
                let c = (b[10][2 * k].clone(), b[10][2 * k + 1].clone());
                let (dx, dy) = lens_shift(*term, c, &w);
                xd += dx;
                yd += dy;
            }
        }
        if !self.tilt_first {
            let h = if self.pupil {
                let s = T::one()
                    + b[4][1].clone() * (xd.clone() * xd.clone() + yd.clone() * yd.clone());
                tilt_h(tx.clone() * s.clone(), ty.clone() * s)
            } else {
                h
            };
            (xd, yd) = apply_h(&h, xd, yd);
            if self.sensor != SensorTerms::Off {
                let (mut sx0, mut sy0) = match self.sensor_centre {
                    Centre::Principal => (T::zero(), T::zero()),
                    Centre::Foot | Centre::Free => foot(tx.clone(), ty.clone()),
                };
                if self.sensor_centre == Centre::Free {
                    sx0 += b[8][2].clone();
                    sy0 += b[8][3].clone();
                }
                let (dx, dy) = sensor_shift(
                    self.sensor,
                    b[7].as_slice(),
                    xd.clone(),
                    yd.clone(),
                    sx0,
                    sy0,
                );
                xd += dx;
                yd += dy;
            }
        }
        let mut u = fx * xd + cx;
        let mut v = fy * yd + cy;
        let nf = self.field_terms.len();
        if nf > 0 {
            let (xn, yn) = if self.field_at_obs {
                (c(self.obs_norm[0]), c(self.obs_norm[1]))
            } else {
                (
                    (u.clone() - c(self.half[0])) / c(self.half[0]),
                    (v.clone() - c(self.half[1])) / c(self.half[1]),
                )
            };
            let (mut du, mut dv) = (T::zero(), T::zero());
            for (k, &(i, j)) in self.field_terms.iter().enumerate() {
                let m = xn.clone().powi(i) * yn.clone().powi(j);
                du += b[6][k].clone() * m.clone();
                dv += b[6][nf + k].clone() * m;
            }
            u += du;
            v += dv;
        }
        if spline_len(self.spline) > 0 {
            let (du, dv) = spline_eval(
                self.spline,
                self.size,
                b[9].as_slice(),
                u.clone(),
                v.clone(),
            );
            u += du;
            v += dv;
        }
        (u, v)
    }
}

impl Residual for GenResidual {
    fn dim(&self) -> usize {
        2
    }
    fn block_dims(&self) -> Vec<usize> {
        self.dims()
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let (u, v) = self.project(blocks);
        DVector::from_vec(vec![
            u - T::from_f64(self.observed[0]).unwrap(),
            v - T::from_f64(self.observed[1]).unwrap(),
        ])
    }
}

/// A fit of the experimental model.
#[derive(Debug, Clone)]
pub struct GenFit {
    pub params: GenParams,
    /// The spline's cells and the image size it spans.
    pub spline_cells: [usize; 2],
    pub size: [f64; 2],
    pub residuals: Vec<PointResidual>,
    /// The fitted pose of each view, in the order of `ids`.
    pub poses: Vec<Iso3>,
    pub iters: usize,
    pub cost: f64,
    pub covariance: String,
    pub sigma: BTreeMap<String, f64>,
    pub corr: BTreeMap<String, f64>,
}

/// Which model blocks a solve frees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Free {
    /// The model's free terms and every pose.
    All,
    /// Only the poses.
    PosesOnly,
    /// The model's free terms but the field, held, and every pose.
    AllButField,
}

/// Fit `cfg` to `views` (indices `ids`) from `init` and `poses`.
// A study helper: the arguments are the fit's independent inputs, and a
// struct to carry them would only be built at each call.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    cfg: &GenCfg,
    ids: &[usize],
    views: &[View<NoMeta>],
    init: &GenParams,
    poses: &[Iso3],
    image_size: [u32; 2],
    free: Free,
    max_iters: usize,
) -> Result<GenFit> {
    let board_terms = monomials(2, cfg.board_deg);
    let field_terms = monomials(cfg.field_min_deg, cfg.field_deg);
    let mut p = Problem::new();
    // The reparameterized principal point.
    let mut intr = init.intr;
    if let Some((sx, sy)) = cfg.reparam {
        intr[2] -= sx * intr[0] * init.tilt[1].tan();
        intr[3] -= sy * intr[1] * init.tilt[0].tan();
    }
    let b_intr = p.add_block("intr", &intr, Manifold::Euclidean)?;
    p.set_labels(b_intr, &["fx", "fy", "cx", "cy"])?;
    let b_dist = p.add_block("dist", &init.dist, Manifold::Euclidean)?;
    p.set_labels(b_dist, &DIST_NAMES)?;
    let b_tilt = p.add_block("tilt", &init.tilt, Manifold::Euclidean)?;
    p.set_labels(b_tilt, &["tilt_x", "tilt_y"])?;
    let b_dc = p.add_block("dc", &init.dc, Manifold::Euclidean)?;
    let b_depth = p.add_block("depth", &[init.depth, init.pupil], Manifold::Euclidean)?;
    let pad = |v: &[f64], n: usize| -> Vec<f64> {
        if n == 0 {
            vec![0.0]
        } else {
            let mut out = v.to_vec();
            out.resize(n, 0.0);
            out
        }
    };
    let b_board = p.add_block(
        "board",
        &pad(&init.board, board_terms.len()),
        Manifold::Euclidean,
    )?;
    let b_field = p.add_block(
        "field",
        &pad(&init.field, 2 * field_terms.len()),
        Manifold::Euclidean,
    )?;
    let n_sensor = cfg.sensor.len();
    let b_sensor = p.add_block("sensor", &pad(&init.sensor, n_sensor), Manifold::Euclidean)?;
    let b_misc = p.add_block("misc", &init.misc, Manifold::Euclidean)?;
    p.set_labels(b_misc, &["e1", "e2", "sc_dx", "sc_dy"])?;
    let n_spline = spline_len(cfg.spline);
    let b_spline = p.add_block(
        "spline",
        &pad(&init.spline, 2 * n_spline),
        Manifold::Euclidean,
    )?;
    let n_lens = cfg.lens.len();
    let b_lens = p.add_block("lens", &pad(&init.lens, 2 * n_lens), Manifold::Euclidean)?;
    let model_blocks = [
        b_intr, b_dist, b_tilt, b_dc, b_depth, b_board, b_field, b_sensor, b_misc, b_spline, b_lens,
    ];
    if free == Free::PosesOnly {
        for b in model_blocks {
            p.fix(b)?;
        }
    } else {
        let fixed: Vec<usize> = DIST_NAMES
            .iter()
            .enumerate()
            .filter(|(_, n)| !cfg.free_dist.contains(n))
            .map(|(i, _)| i)
            .collect();
        if !cfg.fix_dist {
            p.fix_components(b_dist, &fixed)?;
        }
        if !cfg.dc {
            p.fix(b_dc)?;
        }
        let depth_fixed: Vec<usize> = [(0, cfg.depth), (1, cfg.pupil)]
            .iter()
            .filter(|(_, on)| !on)
            .map(|(i, _)| *i)
            .collect();
        p.fix_components(b_depth, &depth_fixed)?;
        if cfg.fix_tilt {
            p.fix(b_tilt)?;
        }
        if cfg.fix_intr {
            p.fix(b_intr)?;
        }
        if cfg.fix_dist {
            p.fix(b_dist)?;
        }
        if free == Free::AllButField {
            p.fix(b_field)?;
            p.fix(b_spline)?;
        }
        if n_lens == 0 {
            p.fix(b_lens)?;
        } else if !cfg.lens_fix.is_empty() {
            p.fix_components(b_lens, cfg.lens_fix)?;
        }
        if n_spline == 0 {
            p.fix(b_spline)?;
        }
        if board_terms.is_empty() {
            p.fix(b_board)?;
        }
        if field_terms.is_empty() {
            p.fix(b_field)?;
        }
        if n_sensor == 0 {
            p.fix(b_sensor)?;
        }
        let mut misc_fixed = Vec::new();
        if !cfg.elliptic {
            misc_fixed.extend([0, 1]);
        }
        if cfg.sensor_centre != Centre::Free || n_sensor == 0 {
            misc_fixed.extend([2, 3]);
        }
        p.fix_components(b_misc, &misc_fixed)?;
    }
    let (w, h) = (f64::from(image_size[0]), f64::from(image_size[1]));
    let mut pose_blocks: Vec<BlockId> = Vec::new();
    let mut residuals: Vec<(usize, GenResidual)> = Vec::new();
    for ((&id, view), pose) in ids.iter().zip(views).zip(poses) {
        let pb = p.add_pose(format!("pose{id}"), pose)?;
        pose_blocks.push(pb);
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            let r = GenResidual {
                board: [p3.x, p3.y],
                observed: [p2.x, p2.y],
                obs_norm: [(p2.x - w / 2.0) / (w / 2.0), (p2.y - h / 2.0) / (h / 2.0)],
                half: [w / 2.0, h / 2.0],
                field_at_obs: cfg.field_at_obs,
                board_terms: board_terms.clone(),
                field_terms: field_terms.clone(),
                board_centre: init.board_centre,
                dc: cfg.dc,
                tilt_first: cfg.tilt_first,
                depth: cfg.depth,
                pupil: cfg.pupil,
                reparam: cfg.reparam,
                sensor: cfg.sensor,
                sensor_centre: cfg.sensor_centre,
                elliptic: cfg.elliptic,
                spline: cfg.spline,
                size: [w, h],
                lens: cfg.lens,
            };
            let mut blocks = model_blocks.to_vec();
            blocks.push(pb);
            debug_assert_eq!(blocks.len(), N_BLOCKS);
            p.add_residual(
                r.clone(),
                &blocks,
                RobustLoss::Huber {
                    scale: crate::HUBER_PX,
                },
                Noise::Estimated("px".into()),
            )?;
            residuals.push((id, r));
        }
    }
    if n_spline > 0 && free != Free::PosesOnly {
        spline_priors(&mut p, b_spline, cfg, ids, views, init, poses, [w, h])?;
    }
    let mut opts = SolveOptions::default();
    opts.max_iters = max_iters;
    let sol = solve(&p, &opts)?;
    let val = |b: BlockId| sol.value(b).map(<[f64]>::to_vec).unwrap_or_default();
    let mut intr_out = [0.0; 4];
    intr_out.copy_from_slice(&val(b_intr));
    let mut dist = [0.0; 9];
    dist.copy_from_slice(&val(b_dist));
    let tilt_v = val(b_tilt);
    let tilt = [tilt_v[0], tilt_v[1]];
    let mut phys = intr_out;
    if let Some((sx, sy)) = cfg.reparam {
        phys[2] += sx * phys[0] * tilt[1].tan();
        phys[3] += sy * phys[1] * tilt[0].tan();
    }
    let dc_v = val(b_dc);
    let params = GenParams {
        intr: phys,
        dist,
        tilt,
        dc: [dc_v[0], dc_v[1]],
        depth: val(b_depth)[0],
        pupil: val(b_depth)[1],
        board: if board_terms.is_empty() {
            Vec::new()
        } else {
            val(b_board)
        },
        field: if field_terms.is_empty() {
            Vec::new()
        } else {
            val(b_field)
        },
        sensor: if n_sensor == 0 {
            Vec::new()
        } else {
            val(b_sensor)
        },
        misc: {
            let m = val(b_misc);
            [m[0], m[1], m[2], m[3]]
        },
        spline: if n_spline == 0 {
            Vec::new()
        } else {
            val(b_spline)
        },
        lens: if n_lens == 0 { Vec::new() } else { val(b_lens) },
        board_centre: init.board_centre,
    };
    let poses_out: Vec<Iso3> = pose_blocks
        .iter()
        .map(|&b| sol.pose(b).expect("a pose block of this problem"))
        .collect();

    // Residuals at the solution.
    let model_vals: Vec<DVector<f64>> = model_blocks
        .iter()
        .map(|&b| DVector::from_vec(val(b)))
        .collect();
    let mut pts = Vec::with_capacity(residuals.len());
    let pose_of: BTreeMap<usize, usize> = ids.iter().enumerate().map(|(k, &id)| (id, k)).collect();
    for (id, r) in &residuals {
        let pose = &poses_out[pose_of[id]];
        let mut blocks = model_vals.clone();
        let q = pose.rotation.coords;
        let t = pose.translation.vector;
        blocks.push(DVector::from_vec(vec![q.x, q.y, q.z, q.w, t.x, t.y, t.z]));
        let (u, v) = r.project(&blocks);
        let pc = pose * Point3::new(r.board[0], r.board[1], 0.0);
        pts.push(PointResidual {
            view: *id,
            board_mm: [r.board[0] * 1e3, r.board[1] * 1e3],
            observed_px: r.observed,
            residual_px: [r.observed[0] - u, r.observed[1] - v],
            cam_mm: [pc.x * 1e3, pc.y * 1e3, pc.z * 1e3],
            blur_px: None,
        });
    }

    let mut sigma = BTreeMap::new();
    let mut corr = BTreeMap::new();
    let covariance = match &sol.uncertainty {
        Uncertainty::Available(cov) => {
            let idx = |name: &str| {
                cov.labels
                    .iter()
                    .position(|l| l.ends_with(&format!(".{name}")))
            };
            for (i, l) in cov.labels.iter().enumerate() {
                if !l.starts_with("pose") {
                    sigma.insert(l.clone(), cov.covariance[i][i].sqrt());
                }
            }
            for (a, b) in [
                ("cy", "tilt_x"),
                ("cx", "tilt_y"),
                ("fx", "k1"),
                ("fx", "tilt_x"),
            ] {
                if let (Some(i), Some(j)) = (idx(a), idx(b)) {
                    corr.insert(
                        format!("{a}/{b}"),
                        cov.covariance[i][j] / (cov.covariance[i][i] * cov.covariance[j][j]).sqrt(),
                    );
                }
            }
            "available".to_string()
        }
        Uncertainty::Unavailable { reason } => format!("{reason:?}"),
    };
    Ok(GenFit {
        params,
        spline_cells: cfg.spline,
        size: [w, h],
        residuals: pts,
        poses: poses_out,
        iters: sol.report.num_iters,
        cost: sol.report.final_cost,
        covariance,
        sigma,
        corr,
    })
}

/// The model's projection of board point `board` (metres) seen at `pose`,
/// in pixels: the forward map a synthetic scene is drawn from.
pub fn project_point(
    cfg: &GenCfg,
    params: &GenParams,
    size: [u32; 2],
    pose: &Iso3,
    board: [f64; 2],
) -> [f64; 2] {
    let (w, h) = (f64::from(size[0]), f64::from(size[1]));
    let r = GenResidual {
        board,
        observed: [0.0; 2],
        obs_norm: [0.0; 2],
        half: [w / 2.0, h / 2.0],
        field_at_obs: false,
        board_terms: monomials(2, cfg.board_deg),
        field_terms: monomials(cfg.field_min_deg, cfg.field_deg),
        board_centre: params.board_centre,
        dc: cfg.dc,
        tilt_first: cfg.tilt_first,
        depth: cfg.depth,
        pupil: cfg.pupil,
        reparam: None,
        sensor: cfg.sensor,
        sensor_centre: cfg.sensor_centre,
        elliptic: cfg.elliptic,
        spline: cfg.spline,
        size: [w, h],
        lens: cfg.lens,
    };
    let pad = |v: &[f64], n: usize| -> DVector<f64> {
        let mut out = v.to_vec();
        out.resize(n.max(1), 0.0);
        DVector::from_vec(out)
    };
    let q = pose.rotation.coords;
    let t = pose.translation.vector;
    let blocks = vec![
        DVector::from_row_slice(&params.intr),
        DVector::from_row_slice(&params.dist),
        DVector::from_row_slice(&params.tilt),
        DVector::from_row_slice(&params.dc),
        DVector::from_vec(vec![params.depth, params.pupil]),
        pad(&params.board, r.board_terms.len()),
        pad(&params.field, 2 * r.field_terms.len()),
        pad(&params.sensor, cfg.sensor.len()),
        DVector::from_row_slice(&params.misc),
        pad(&params.spline, 2 * spline_len(cfg.spline)),
        pad(&params.lens, 2 * cfg.lens.len()),
        DVector::from_vec(vec![q.x, q.y, q.z, q.w, t.x, t.y, t.z]),
    ];
    let (u, v) = r.project(&blocks);
    [u, v]
}

/// A linear residual on the spline block: `Σ row_k·c_k`.
#[derive(Clone)]
struct Linear {
    n: usize,
    terms: Vec<(usize, f64)>,
}

impl Linear {
    fn scaled(mut self, s: f64) -> Self {
        self.terms.iter_mut().for_each(|t| t.1 *= s);
        self
    }
}

impl Residual for Linear {
    fn dim(&self) -> usize {
        1
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![self.n]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let mut r = T::zero();
        for &(k, a) in &self.terms {
            r += blocks[0][k].clone() * T::from_f64(a).unwrap();
        }
        DVector::from_vec(vec![r])
    }
}

/// Stated σ of the orthogonality penalty (px). The covariance reads a
/// stated residual as known information with this σ, which leaks into the
/// camera's σ in proportion to it: at 1e-4 `cx, cy` state about three
/// times their Monte-Carlo scatter and at 1e-6 they are honest, but at
/// 1e-6 a solve started away from the optimum accepts no step (the
/// normal equations are too stiff). A hard constraint has neither problem.
const ORTHO_SIGMA: f64 = 1e-4;
/// [`ORTHO_SIGMA`], or `STUDY_ORTHO_SIGMA` from the environment (a study
/// knob).
fn ortho_sigma() -> f64 {
    std::env::var("STUDY_ORTHO_SIGMA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ORTHO_SIGMA)
}

/// Relative finite-difference step of the camera-parameter Jacobian the
/// orthogonality constraint is built from (central differences).
const FD_REL: f64 = 1e-6;

/// The spline's smoothness prior and orthogonality penalty.
#[allow(clippy::too_many_arguments)]
fn spline_priors(
    p: &mut Problem,
    b_spline: BlockId,
    cfg: &GenCfg,
    ids: &[usize],
    views: &[View<NoMeta>],
    init: &GenParams,
    poses: &[Iso3],
    size: [f64; 2],
) -> Result<()> {
    let n = spline_len(cfg.spline);
    let dim = 2 * n;
    let (nx, ny) = (cfg.spline[0] + 3, cfg.spline[1] + 3);
    if cfg.spline_sigma > 0.0 {
        for comp in 0..2 {
            for j in 0..ny {
                for i in 0..nx {
                    let at = |a: usize, b: usize| comp * n + b * nx + a;
                    let mut add = |terms: Vec<(usize, f64)>| -> Result<()> {
                        let r = Linear { n: dim, terms };
                        if cfg.spline_penalty {
                            p.add_residual(
                                r.scaled(1.0 / cfg.spline_sigma),
                                &[b_spline],
                                RobustLoss::None,
                                Noise::Estimated("smooth".into()),
                            )?;
                        } else {
                            p.add_residual(
                                r,
                                &[b_spline],
                                RobustLoss::None,
                                Noise::Stated(cfg.spline_sigma),
                            )?;
                        }
                        Ok(())
                    };
                    if i + 2 < nx {
                        add(vec![
                            (at(i, j), 1.0),
                            (at(i + 1, j), -2.0),
                            (at(i + 2, j), 1.0),
                        ])?;
                    }
                    if j + 2 < ny {
                        add(vec![
                            (at(i, j), 1.0),
                            (at(i, j + 1), -2.0),
                            (at(i, j + 2), 1.0),
                        ])?;
                    }
                }
            }
        }
    }
    if cfg.spline_ortho {
        for row in ortho_rows(cfg, ids, views, init, poses, size) {
            let terms: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, a)| **a != 0.0)
                .map(|(k, a)| (k, *a))
                .collect();
            p.add_residual(
                Linear { n: dim, terms },
                &[b_spline],
                RobustLoss::None,
                Noise::Stated(ortho_sigma()),
            )?;
        }
    }
    let _ = ids;
    Ok(())
}

/// Rows `J_jᵀ·B` (unit norm) of the orthogonality constraint: for each free
/// camera parameter and each common rigid motion `j`, the pixel direction
/// `J_j` it moves every fitted point in, against the spline basis `B` at
/// the point's projection, both at the starting solution.
pub fn ortho_rows(
    cfg: &GenCfg,
    _ids: &[usize],
    views: &[View<NoMeta>],
    init: &GenParams,
    poses: &[Iso3],
    size: [f64; 2],
) -> Vec<Vec<f64>> {
    let n = spline_len(cfg.spline);
    let nx = cfg.spline[0] + 3;
    let plain = GenCfg {
        spline: [0, 0],
        field_deg: 0,
        ..cfg.clone()
    };
    let isize = [size[0] as u32, size[1] as u32];
    // Perturbations: (parameter setter) as closures over a params copy.
    type Setter = Box<dyn Fn(&mut GenParams, f64)>;
    let mut dirs: Vec<(f64, Setter)> = Vec::new();
    if !cfg.fix_intr {
        for k in 0..4 {
            dirs.push((
                init.intr[k].abs().max(1.0),
                Box::new(move |q, d| q.intr[k] += d),
            ));
        }
    }
    if !cfg.fix_dist {
        for (k, name) in DIST_NAMES.iter().enumerate() {
            if cfg.free_dist.contains(name) {
                dirs.push((1.0, Box::new(move |q, d| q.dist[k] += d)));
            }
        }
    }
    if !cfg.fix_tilt {
        for k in 0..2 {
            dirs.push((1.0, Box::new(move |q, d| q.tilt[k] += d)));
        }
    }
    let n_param = dirs.len();
    let mut rows = vec![vec![0.0; 2 * n]; n_param + 3];
    // Samples: (pose, board point) whose projection the constraint is
    // measured at.
    let mut samples: Vec<(Iso3, [f64; 2])> = Vec::new();
    if cfg.spline_ortho_cover {
        const GX: usize = 24;
        const GY: usize = 18;
        let mut covered = vec![false; GX * GY];
        for v in views {
            for q in &v.obs.points_2d {
                let i = ((q.x / size[0] * GX as f64) as usize).min(GX - 1);
                let j = ((q.y / size[1] * GY as f64) as usize).min(GY - 1);
                covered[j * GX + i] = true;
            }
        }
        // A point on the plane z = 1 whose projection is the cell centre.
        let at_z1 = Iso3::translation(0.0, 0.0, 1.0);
        for j in 0..GY {
            for i in 0..GX {
                if !covered[j * GX + i] {
                    continue;
                }
                let target = [
                    (i as f64 + 0.5) / GX as f64 * size[0],
                    (j as f64 + 0.5) / GY as f64 * size[1],
                ];
                let mut x = [
                    (target[0] - init.intr[2]) / init.intr[0],
                    (target[1] - init.intr[3]) / init.intr[1],
                ];
                for _ in 0..20 {
                    let f = project_point(&plain, init, isize, &at_z1, x);
                    let e = [f[0] - target[0], f[1] - target[1]];
                    if e[0].hypot(e[1]) < 1e-10 {
                        break;
                    }
                    let d = 1e-7;
                    let fx = project_point(&plain, init, isize, &at_z1, [x[0] + d, x[1]]);
                    let fy = project_point(&plain, init, isize, &at_z1, [x[0], x[1] + d]);
                    let (a, b) = ((fx[0] - f[0]) / d, (fy[0] - f[0]) / d);
                    let (c, dd) = ((fx[1] - f[1]) / d, (fy[1] - f[1]) / d);
                    let det = a * dd - b * c;
                    x[0] -= (dd * e[0] - b * e[1]) / det;
                    x[1] -= (-c * e[0] + a * e[1]) / det;
                }
                samples.push((at_z1, x));
            }
        }
    } else {
        for (view, pose) in views.iter().zip(poses) {
            for p3 in &view.obs.points_3d {
                samples.push((*pose, [p3.x, p3.y]));
            }
        }
    }
    {
        for (pose, b) in &samples {
            let (pose, b) = (pose, *b);
            let u0 = project_point(&plain, init, isize, pose, b);
            // Basis at the projection.
            let (ix, wx) = bspline_axis(u0[0], size[0], cfg.spline[0]);
            let (iy, wy) = bspline_axis(u0[1], size[1], cfg.spline[1]);
            let mut jac: Vec<[f64; 2]> = Vec::with_capacity(n_param + 3);
            for (scale, set) in &dirs {
                let h = FD_REL * scale;
                let (mut qp, mut qm) = (init.clone(), init.clone());
                set(&mut qp, h);
                set(&mut qm, -h);
                let a = project_point(&plain, &qp, isize, pose, b);
                let c = project_point(&plain, &qm, isize, pose, b);
                jac.push([(a[0] - c[0]) / (2.0 * h), (a[1] - c[1]) / (2.0 * h)]);
            }
            // A rotation of the camera common to every view: its pixel
            // pattern depends on the image position only, so a field can
            // copy it and the poses undo it (a gauge). A common translation
            // moves each point by its inverse depth, which no image field
            // can copy, and is left free.
            for m in 0..3 {
                let h = 1e-6;
                let delta = |s: f64| {
                    let mut v = [0.0; 6];
                    v[m] = s;
                    Iso3::new(
                        Vector3::new(v[3], v[4], v[5]),
                        Vector3::new(v[0], v[1], v[2]),
                    )
                };
                let a = project_point(&plain, init, isize, &(delta(h) * pose), b);
                let c = project_point(&plain, init, isize, &(delta(-h) * pose), b);
                jac.push([(a[0] - c[0]) / (2.0 * h), (a[1] - c[1]) / (2.0 * h)]);
            }
            for (row, j) in rows.iter_mut().zip(&jac) {
                for (bb, wyb) in wy.iter().enumerate() {
                    for (aa, wxa) in wx.iter().enumerate() {
                        let k = (iy + bb) * nx + ix + aa;
                        row[k] += j[0] * wxa * wyb;
                        row[n + k] += j[1] * wxa * wyb;
                    }
                }
            }
        }
    }
    for row in &mut rows {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;

    /// The linked lens terms are the decentred fifth-order distortion
    /// `|w − s|⁴(w − s)` expanded in `s`: the first-order part is
    /// `NatA5(c = s)`, the second-order part `NatB5(c = s²)` plus the
    /// radial `6|s|²w²w̄` (which `k1` absorbs).
    #[test]
    fn linked_terms_are_the_decentred_expansion() {
        let w = Complex::new(0.23, -0.17);
        let s = Complex::new(0.011, 0.007);
        let f = |e: f64| {
            let d = w - s * e;
            d * d * d * d.conj() * d.conj()
        };
        let e = 1e-3;
        let first = (f(e) - f(-e)) / (2.0 * e);
        let second = (f(e) + f(-e)) / 2.0 - f(0.0);
        let shift = |t: LensTerm, c: Complex<f64>| {
            let (x, y) = lens_shift(t, (c.re, c.im), &(w.re, w.im));
            Complex::new(x, y)
        };
        let a5 = shift(LensTerm::NatA5, s);
        assert!((first - a5).norm() < 1e-9, "{first} vs {a5}");
        let radial = w * w * w.conj() * (6.0 * s.norm_sqr());
        let b5 = shift(LensTerm::NatB5, s * s) + radial;
        assert!(
            (second / (e * e) - b5).norm() < 1e-8,
            "{} vs {b5}",
            second / (e * e)
        );
        // A monomial term is `c·w^p·w̄^q`.
        let m = shift(LensTerm::Mono(0, 3), Complex::new(0.5, -0.2));
        let want = Complex::new(0.5, -0.2) * w.conj() * w.conj() * w.conj();
        assert!((m - want).norm() < 1e-15);
    }
}
