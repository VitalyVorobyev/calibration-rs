//! The residual-field slot: a smooth, camera-fixed displacement in pixels,
//! applied after the intrinsics.
//!
//! ```text
//! pixel = F(K(sensor(distortion(projection(p_c)))))      F(u) = u + Σ_k β_k(u)·c_k
//! ```
//!
//! The field is a uniform bicubic B-spline over the image rectangle
//! `[0, w] × [0, h]`, evaluated at the pixel the rest of the model produces.
//! Outside the rectangle the argument is clamped, so a point beyond the
//! border takes the border's displacement and nothing grows. Its inverse is
//! the fixed point of `u_par = u − F(u_par)`, a contraction whenever the
//! field's Lipschitz bound is at most [`FIELD_MAX_LIPSCHITZ`], which
//! [`BSplineField::validate`] checks; the bound also rules out a field that
//! folds the image over itself.
//!
//! Every evaluation, the optimizer's residuals and Jacobians included, goes
//! through the generic kernel of [`FieldKind::displace`] and
//! [`FieldKind::undisplace`].

use nalgebra::{Point2, RealField};
use serde::{Deserialize, Serialize};

use super::FieldKind;
use crate::{Error, Real};

/// The largest Lipschitz bound a residual field may have.
///
/// At or below it the inverse map is a contraction that halves its error
/// every step, and the field cannot fold the image over itself. Real
/// residual fields are far below it: a few pixels of displacement over a
/// knot spacing of a hundred pixels is a bound of a few hundredths.
pub const FIELD_MAX_LIPSCHITZ: Real = 0.5;

/// The most cells a residual field may have along either axis.
pub const FIELD_MAX_CELLS: u32 = 1024;

/// Cap on the fixed-point iterations of the inverse map. At the Lipschitz
/// bound of 0.5 the error halves every step, so 100 steps reach any
/// displacement's rounding level; a typical field converges in a handful.
const UNDISPLACE_MAX_ITERS: usize = 100;

/// Fixed-point step (pixels) below which the inverse map stops iterating;
/// one Newton step then polishes the value to rounding level.
const UNDISPLACE_STEP_TOL_PX: f64 = 1e-11;

/// A pixel-space displacement applied after the intrinsics, and its inverse.
///
/// Generic over `T: RealField` with no `Copy` bound, so the optimizer's
/// dual-number autodiff evaluates the same code.
pub trait FieldModel<T: RealField> {
    /// The displaced pixel `u + F(u)`.
    fn displace(&self, pixel: &Point2<T>) -> Point2<T>;
    /// The pixel `u_par` with `u_par + F(u_par) = pixel`.
    fn undisplace(&self, pixel: &Point2<T>) -> Point2<T>;
}

/// No residual field (identity mapping).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct NoField;

impl<T: RealField> FieldModel<T> for NoField {
    fn displace(&self, pixel: &Point2<T>) -> Point2<T> {
        pixel.clone()
    }

    fn undisplace(&self, pixel: &Point2<T>) -> Point2<T> {
        pixel.clone()
    }
}

/// A uniform bicubic B-spline displacement over the image.
///
/// `cells = [nx, ny]` uniform cells span the image rectangle
/// `[0, width] × [0, height]`; the spline has `(nx + 3) × (ny + 3)` control
/// displacements per component, stored row by row (`v` index outer, `u`
/// index inner). A pixel outside the rectangle takes the displacement of
/// the nearest border point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct BSplineField {
    /// Cells along `u` and `v`, each in `1..=1024` and at most the image
    /// size along that axis.
    pub cells: [u32; 2],
    /// The image size `[width, height]` in pixels: the rectangle the spline
    /// spans.
    pub image_size: [u32; 2],
    /// Control displacements of the `u` component, in pixels, row-major over
    /// the `(ny + 3) × (nx + 3)` control grid.
    pub du: Vec<Real>,
    /// Control displacements of the `v` component, in pixels, laid out like
    /// [`du`](Self::du).
    pub dv: Vec<Real>,
}

impl BSplineField {
    /// A field of `cells` over an image of `image_size` with every control
    /// displacement zero.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when the cells or the image size are out of
    /// range (see [`BSplineField::cells`]).
    pub fn zeros(cells: [u32; 2], image_size: [u32; 2]) -> Result<Self, Error> {
        let kind = FieldKind::BSpline { cells, image_size };
        kind.validate()?;
        let n = kind.dim() / 2;
        Ok(Self {
            cells,
            image_size,
            du: vec![0.0; n],
            dv: vec![0.0; n],
        })
    }

    /// The field's kind: its cells and image size.
    pub fn kind(&self) -> FieldKind {
        FieldKind::BSpline {
            cells: self.cells,
            image_size: self.image_size,
        }
    }

    /// An upper bound on the field's Lipschitz constant, `max ‖∂F/∂u‖`.
    ///
    /// Along each axis a cubic B-spline's derivative is a convex combination
    /// of its control differences over the knot spacing, so every entry of
    /// the 2×2 Jacobian is bounded by the largest such difference; the
    /// bound is the Frobenius norm of those entry bounds.
    pub fn lipschitz_bound(&self) -> Real {
        let [nx, ny] = control_grid(self.cells);
        let hx = Real::from(self.image_size[0]) / Real::from(self.cells[0]);
        let hy = Real::from(self.image_size[1]) / Real::from(self.cells[1]);
        let slopes = |c: &[Real]| -> (Real, Real) {
            let (mut gx, mut gy) = (0.0_f64, 0.0_f64);
            for j in 0..ny {
                for i in 0..nx {
                    let here = c[j * nx + i];
                    if i + 1 < nx {
                        gx = gx.max((c[j * nx + i + 1] - here).abs());
                    }
                    if j + 1 < ny {
                        gy = gy.max((c[(j + 1) * nx + i] - here).abs());
                    }
                }
            }
            (gx / hx, gy / hy)
        };
        let (ux, uy) = slopes(&self.du);
        let (vx, vy) = slopes(&self.dv);
        (ux * ux + uy * uy + vx * vx + vy * vy).sqrt()
    }

    /// Check the field is well formed and invertible: valid cells and image
    /// size, one finite control displacement per grid node in each
    /// component, and a Lipschitz bound of at most
    /// [`FIELD_MAX_LIPSCHITZ`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the problem.
    pub fn validate(&self) -> Result<(), Error> {
        let kind = self.kind();
        kind.validate()?;
        let n = kind.dim() / 2;
        if self.du.len() != n || self.dv.len() != n {
            return Err(Error::invalid_input(format!(
                "a residual field of {:?} cells has {n} control displacements per \
                 component; got {} (u) and {} (v)",
                self.cells,
                self.du.len(),
                self.dv.len()
            )));
        }
        if !self.du.iter().chain(&self.dv).all(|c| c.is_finite()) {
            return Err(Error::invalid_input(
                "a residual field's control displacements must be finite",
            ));
        }
        let lipschitz = self.lipschitz_bound();
        if lipschitz > FIELD_MAX_LIPSCHITZ {
            return Err(Error::invalid_input(format!(
                "the residual field's Lipschitz bound is {lipschitz:.3}, above \
                 {FIELD_MAX_LIPSCHITZ}: it may fold the image over itself and its inverse \
                 is not guaranteed to converge"
            )));
        }
        Ok(())
    }

    /// The control displacements as one vector: every `u` component, then
    /// every `v` component (the order of [`FieldKind::param_names`]).
    pub fn values(&self) -> Vec<Real> {
        self.du.iter().chain(&self.dv).copied().collect()
    }
}

impl FieldModel<Real> for BSplineField {
    fn displace(&self, pixel: &Point2<Real>) -> Point2<Real> {
        bspline_displace(self.cells, self.image_size, &self.du, &self.dv, pixel)
    }

    fn undisplace(&self, pixel: &Point2<Real>) -> Point2<Real> {
        bspline_undisplace(self.cells, self.image_size, &self.du, &self.dv, pixel)
    }
}

/// Serializable residual-field parameters.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum FieldParams {
    /// No residual field.
    #[default]
    None,
    /// A uniform bicubic B-spline displacement over the image.
    BSpline {
        /// The spline's grid and control displacements.
        #[serde(flatten)]
        params: BSplineField,
    },
}

impl FieldParams {
    /// Whether there is no field.
    pub fn is_none(&self) -> bool {
        matches!(self, FieldParams::None)
    }

    /// The field's kind.
    pub fn kind(&self) -> FieldKind {
        match self {
            FieldParams::None => FieldKind::None,
            FieldParams::BSpline { params } => params.kind(),
        }
    }

    /// The field's parameters in the order of [`FieldKind::param_names`]
    /// (empty without a field).
    pub fn values(&self) -> Vec<Real> {
        match self {
            FieldParams::None => Vec::new(),
            FieldParams::BSpline { params } => params.values(),
        }
    }

    /// Parameters of a `kind` field from values in the order of
    /// [`FieldKind::param_names`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when the kind is out of range or
    /// `values.len() != kind.dim()`.
    pub fn from_values(kind: FieldKind, values: &[Real]) -> Result<Self, Error> {
        kind.validate()?;
        if values.len() != kind.dim() {
            return Err(Error::invalid_input(format!(
                "expected {} field values, got {}",
                kind.dim(),
                values.len()
            )));
        }
        Ok(match kind {
            FieldKind::None => FieldParams::None,
            FieldKind::BSpline { cells, image_size } => {
                let (du, dv) = values.split_at(values.len() / 2);
                FieldParams::BSpline {
                    params: BSplineField {
                        cells,
                        image_size,
                        du: du.to_vec(),
                        dv: dv.to_vec(),
                    },
                }
            }
        })
    }

    /// Check the field is well formed and invertible (see
    /// [`BSplineField::validate`]); no field is always valid.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the problem.
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            FieldParams::None => Ok(()),
            FieldParams::BSpline { params } => params.validate(),
        }
    }
}

impl FieldModel<Real> for FieldParams {
    fn displace(&self, pixel: &Point2<Real>) -> Point2<Real> {
        match self {
            FieldParams::None => *pixel,
            FieldParams::BSpline { params } => params.displace(pixel),
        }
    }

    fn undisplace(&self, pixel: &Point2<Real>) -> Point2<Real> {
        match self {
            FieldParams::None => *pixel,
            FieldParams::BSpline { params } => params.undisplace(pixel),
        }
    }
}

/// Control points per axis of a spline of `cells` cells.
pub(crate) fn control_grid(cells: [u32; 2]) -> [usize; 2] {
    [cells[0] as usize + 3, cells[1] as usize + 3]
}

/// The uniform cubic B-spline basis along one axis at `x`, over `[0, len]`
/// in `cells` cells: the first of the four control indices, the four
/// weights, and their derivatives with respect to `x`. Outside the range the
/// argument is clamped, so the border value holds and the derivatives are
/// zero.
fn bspline_axis<T: RealField>(x: T, len: u32, cells: u32) -> (usize, [T; 4], [T; 4]) {
    let k = |v: f64| T::from_f64(v).expect("a RealField holds every f64 constant");
    let spacing = f64::from(len) / f64::from(cells);
    let top = k(f64::from(cells));
    let mut t = x / k(spacing);
    // Clamped: the argument becomes a constant, so the derivative is zero.
    let clamped = t < T::zero() || t > top;
    if t < T::zero() {
        t = T::zero();
    } else if t > top {
        t = top;
    }
    let mut i = 0usize;
    while i + 1 < cells as usize && t >= k((i + 1) as f64) {
        i += 1;
    }
    let f = t - k(i as f64);
    let f2 = f.clone() * f.clone();
    let f3 = f2.clone() * f.clone();
    let g = T::one() - f.clone();
    let sixth = k(1.0 / 6.0);
    let w = [
        g.clone() * g.clone() * g.clone() * sixth.clone(),
        (k(3.0) * f3.clone() - k(6.0) * f2.clone() + k(4.0)) * sixth.clone(),
        (k(-3.0) * f3.clone() + k(3.0) * f2.clone() + k(3.0) * f.clone() + T::one())
            * sixth.clone(),
        f3 * sixth,
    ];
    let dw = if clamped {
        [T::zero(), T::zero(), T::zero(), T::zero()]
    } else {
        // d w / d t, then d t / d x = 1 / spacing.
        let half_h = k(0.5 / spacing);
        [
            -(g.clone() * g) * half_h.clone(),
            (k(3.0) * f2.clone() - k(4.0) * f.clone()) * half_h.clone(),
            (k(-3.0) * f2.clone() + k(2.0) * f.clone() + T::one()) * half_h.clone(),
            f2 * half_h,
        ]
    };
    (i, w, dw)
}

/// The displacement `F(u)` and, when `jacobian` is set, its 2×2 Jacobian
/// `[[∂Fu/∂x, ∂Fu/∂y], [∂Fv/∂x, ∂Fv/∂y]]`.
fn bspline_eval<T: RealField>(
    cells: [u32; 2],
    image_size: [u32; 2],
    du: &[T],
    dv: &[T],
    pixel: &Point2<T>,
    jacobian: bool,
) -> ([T; 2], [[T; 2]; 2]) {
    let [nx, _] = control_grid(cells);
    let (ix, wx, dwx) = bspline_axis(pixel.x.clone(), image_size[0], cells[0]);
    let (iy, wy, dwy) = bspline_axis(pixel.y.clone(), image_size[1], cells[1]);
    let (mut fu, mut fv) = (T::zero(), T::zero());
    let mut jac = [[T::zero(), T::zero()], [T::zero(), T::zero()]];
    for b in 0..4 {
        for a in 0..4 {
            let idx = (iy + b) * nx + ix + a;
            let w = wx[a].clone() * wy[b].clone();
            fu += w.clone() * du[idx].clone();
            fv += w * dv[idx].clone();
            if jacobian {
                let wdx = dwx[a].clone() * wy[b].clone();
                let wdy = wx[a].clone() * dwy[b].clone();
                jac[0][0] += wdx.clone() * du[idx].clone();
                jac[0][1] += wdy.clone() * du[idx].clone();
                jac[1][0] += wdx * dv[idx].clone();
                jac[1][1] += wdy * dv[idx].clone();
            }
        }
    }
    ([fu, fv], jac)
}

/// `u + F(u)` for a B-spline field with control displacements `du`, `dv`.
pub(crate) fn bspline_displace<T: RealField>(
    cells: [u32; 2],
    image_size: [u32; 2],
    du: &[T],
    dv: &[T],
    pixel: &Point2<T>,
) -> Point2<T> {
    let ([fu, fv], _) = bspline_eval(cells, image_size, du, dv, pixel, false);
    Point2::new(pixel.x.clone() + fu, pixel.y.clone() + fv)
}

/// The pixel `u_par` with `u_par + F(u_par) = pixel`: the fixed point of
/// `u_par = pixel − F(u_par)` from `u_par = pixel`, then one Newton step.
///
/// The fixed-point iteration converges whenever the field's Lipschitz bound
/// is below one ([`BSplineField::validate`] holds it at
/// [`FIELD_MAX_LIPSCHITZ`]). The closing Newton step leaves the value at
/// rounding level and, under dual-number evaluation, gives the converged
/// point the derivative of the exact inverse, `−(I + ∂F/∂u)⁻¹ ∂F/∂θ`,
/// however few fixed-point steps were taken.
pub(crate) fn bspline_undisplace<T: RealField>(
    cells: [u32; 2],
    image_size: [u32; 2],
    du: &[T],
    dv: &[T],
    pixel: &Point2<T>,
) -> Point2<T> {
    let tol = T::from_f64(UNDISPLACE_STEP_TOL_PX).expect("a RealField holds every f64 constant");
    let tol2 = tol.clone() * tol;
    let mut q = pixel.clone();
    for _ in 0..UNDISPLACE_MAX_ITERS {
        let ([fu, fv], _) = bspline_eval(cells, image_size, du, dv, &q, false);
        let next = Point2::new(pixel.x.clone() - fu, pixel.y.clone() - fv);
        let (sx, sy) = (next.x.clone() - q.x.clone(), next.y.clone() - q.y.clone());
        q = next;
        if sx.clone() * sx + sy.clone() * sy <= tol2 {
            break;
        }
    }
    // Newton: r = q + F(q) − pixel, J = I + ∂F/∂u, q ← q − J⁻¹ r.
    let ([fu, fv], j) = bspline_eval(cells, image_size, du, dv, &q, true);
    let rx = q.x.clone() + fu - pixel.x.clone();
    let ry = q.y.clone() + fv - pixel.y.clone();
    let [[j00, j01], [j10, j11]] = j;
    let (a, b, c, d) = (T::one() + j00, j01, j10, T::one() + j11);
    let det = a.clone() * d.clone() - b.clone() * c.clone();
    let step_x = (d * rx.clone() - b * ry.clone()) / det.clone();
    let step_y = (a * ry - c * rx) / det;
    Point2::new(q.x.clone() - step_x, q.y.clone() - step_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(cells: [u32; 2], image_size: [u32; 2], seed: u64, scale: Real) -> BSplineField {
        // A deterministic pseudo-random control grid (splitmix64).
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as Real / (1u64 << 53) as Real * 2.0 - 1.0
        };
        let mut f = BSplineField::zeros(cells, image_size).unwrap();
        f.du.iter_mut().for_each(|c| *c = scale * next());
        f.dv.iter_mut().for_each(|c| *c = scale * next());
        f
    }

    #[test]
    fn basis_is_a_partition_of_unity() {
        for &x in &[-50.0, 0.0, 1e-9, 13.7, 99.999, 100.0, 333.3, 640.0, 700.0] {
            let (_, w, dw) = bspline_axis::<f64>(x, 640, 7);
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-15, "x = {x}: Σw = {s}");
            let ds: f64 = dw.iter().sum();
            assert!(ds.abs() < 1e-15, "x = {x}: Σw' = {ds}");
        }
    }

    /// A cubic B-spline reproduces every polynomial of degree at most 3
    /// inside its domain: the controls of `p` are its blossom values, here
    /// fitted by interpolation at the knots and checked between them.
    #[test]
    fn reproduces_cubic_polynomials() {
        // In knot units t (spacing h), the control of index i for the
        // monomials 1, t, t², t³ is 1, (i − 1), (i − 1)² − 1/3, (i − 1)³ − (i − 1).
        let (cells, size) = ([5u32, 4u32], [500u32, 400u32]);
        let [nx, ny] = control_grid(cells);
        let ctrl = |i: usize| -> [f64; 4] {
            let s = i as f64 - 1.0;
            [1.0, s, s * s - 1.0 / 3.0, s * s * s - s]
        };
        let poly = |t: f64| -> [f64; 4] { [1.0, t, t * t, t * t * t] };
        let coeffs_u = [0.3, -0.7, 0.05, 0.002];
        let coeffs_v = [-1.1, 0.4, -0.03, 0.001];
        let mut f = BSplineField::zeros(cells, size).unwrap();
        for j in 0..ny {
            for i in 0..nx {
                let (cx, cy) = (ctrl(i), ctrl(j));
                // u component: a polynomial in tx only; v component: in ty.
                f.du[j * nx + i] = (0..4).map(|d| coeffs_u[d] * cx[d]).sum();
                f.dv[j * nx + i] = (0..4).map(|d| coeffs_v[d] * cy[d]).sum();
            }
        }
        for &(x, y) in &[(0.0, 0.0), (37.5, 210.0), (250.0, 399.0), (499.9, 0.1)] {
            let (tx, ty) = (x / 100.0, y / 100.0);
            let want_u: f64 = (0..4).map(|d| coeffs_u[d] * poly(tx)[d]).sum();
            let want_v: f64 = (0..4).map(|d| coeffs_v[d] * poly(ty)[d]).sum();
            let got = f.displace(&Point2::new(x, y));
            assert!((got.x - x - want_u).abs() < 1e-12, "u at ({x}, {y})");
            assert!((got.y - y - want_v).abs() < 1e-12, "v at ({x}, {y})");
        }
    }

    #[test]
    fn inverse_round_trips_inside_and_outside() {
        let f = field([8, 6], [1280, 1024], 7, 2.0);
        f.validate().unwrap();
        assert!(f.lipschitz_bound() < FIELD_MAX_LIPSCHITZ);
        for &(x, y) in &[
            (0.0, 0.0),
            (640.0, 512.0),
            (1279.5, 3.0),
            (1280.0, 1024.0),
            (-40.0, 500.0),
            (1400.0, -30.0),
            (17.25, 1100.0),
        ] {
            let u = Point2::new(x, y);
            let back = f.displace(&f.undisplace(&u));
            assert!(
                (back - u).norm() < 1e-9,
                "({x}, {y}): {}",
                (back - u).norm()
            );
            let fwd = f.undisplace(&f.displace(&u));
            assert!((fwd - u).norm() < 1e-9, "({x}, {y}): {}", (fwd - u).norm());
        }
    }

    #[test]
    fn inverse_converges_at_the_lipschitz_bound() {
        let mut f = field([4, 3], [400, 300], 3, 1.0);
        // Scale to the bound itself.
        let s = FIELD_MAX_LIPSCHITZ / f.lipschitz_bound();
        f.du.iter_mut().chain(f.dv.iter_mut()).for_each(|c| *c *= s);
        f.validate().unwrap();
        for &(x, y) in &[(10.0, 10.0), (200.0, 150.0), (390.0, 290.0)] {
            let u = Point2::new(x, y);
            let back = f.displace(&f.undisplace(&u));
            assert!((back - u).norm() < 1e-9, "({x}, {y})");
        }
    }

    #[test]
    fn a_folding_field_is_rejected() {
        let mut f = BSplineField::zeros([4, 4], [400, 400]).unwrap();
        // A 60 px jump over a 100 px knot spacing: slope 0.6.
        f.du[5] = 60.0;
        let err = f.validate().unwrap_err().to_string();
        assert!(err.contains("Lipschitz"), "{err}");
        let params = FieldParams::BSpline { params: f };
        assert!(params.validate().is_err());
    }

    #[test]
    fn malformed_fields_are_rejected() {
        assert!(BSplineField::zeros([0, 3], [100, 100]).is_err());
        assert!(BSplineField::zeros([3, 3], [0, 100]).is_err());
        assert!(BSplineField::zeros([200, 3], [100, 100]).is_err());
        assert!(BSplineField::zeros([FIELD_MAX_CELLS + 1, 3], [1 << 20, 100]).is_err());
        let mut f = BSplineField::zeros([2, 2], [100, 100]).unwrap();
        f.dv.pop();
        assert!(f.validate().is_err());
        let mut f = BSplineField::zeros([2, 2], [100, 100]).unwrap();
        f.du[0] = f64::NAN;
        assert!(f.validate().is_err());
    }

    #[test]
    fn spatial_jacobian_matches_finite_differences() {
        let f = field([5, 4], [640, 480], 11, 3.0);
        for &(x, y) in &[(100.3, 77.7), (320.0, 240.0), (600.1, 470.2)] {
            let p = Point2::new(x, y);
            let (_, j) = bspline_eval(f.cells, f.image_size, &f.du, &f.dv, &p, true);
            let h = 1e-4;
            let at = |dx: f64, dy: f64| {
                bspline_eval(
                    f.cells,
                    f.image_size,
                    &f.du,
                    &f.dv,
                    &Point2::new(x + dx, y + dy),
                    false,
                )
                .0
            };
            let (px, mx, py, my) = (at(h, 0.0), at(-h, 0.0), at(0.0, h), at(0.0, -h));
            for r in 0..2 {
                let fd_x = (px[r] - mx[r]) / (2.0 * h);
                let fd_y = (py[r] - my[r]) / (2.0 * h);
                assert!((j[r][0] - fd_x).abs() < 1e-8, "∂F{r}/∂x at ({x}, {y})");
                assert!((j[r][1] - fd_y).abs() < 1e-8, "∂F{r}/∂y at ({x}, {y})");
            }
        }
    }

    #[test]
    fn values_round_trip_through_the_kind() {
        let f = field([3, 2], [300, 200], 5, 1.0);
        let p = FieldParams::BSpline { params: f.clone() };
        let back = FieldParams::from_values(p.kind(), &p.values()).unwrap();
        assert_eq!(back, p);
        assert!(FieldParams::from_values(p.kind(), &p.values()[1..]).is_err());
        assert_eq!(
            FieldParams::from_values(FieldKind::None, &[]).unwrap(),
            FieldParams::None
        );
    }

    #[test]
    fn serde_shape() {
        let f = BSplineField::zeros([1, 1], [10, 10]).unwrap();
        let json = serde_json::to_value(FieldParams::BSpline { params: f }).unwrap();
        assert_eq!(json["type"], "b_spline");
        assert_eq!(json["cells"], serde_json::json!([1, 1]));
        assert_eq!(json["image_size"], serde_json::json!([10, 10]));
        assert_eq!(json["du"].as_array().unwrap().len(), 16);
        let back: FieldParams = serde_json::from_value(json).unwrap();
        assert_eq!(back.kind().dim(), 32);
        let none: FieldParams = serde_json::from_str(r#"{"type":"none"}"#).unwrap();
        assert!(none.is_none());
    }
}
