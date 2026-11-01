use nalgebra::{Point2, RealField};
use serde::{Deserialize, Serialize};

/// Distortion model mapping between ideal and distorted normalized coordinates.
///
/// Implementations are generic over `T: RealField` with no `Copy` bound, so
/// the same code serves `f64` evaluation and the optimizer's dual-number
/// autodiff.
pub trait DistortionModel<T: RealField> {
    /// Apply distortion to undistorted normalized coordinates.
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T>;
    /// Remove distortion from distorted normalized coordinates.
    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T>;
}

/// No distortion (identity mapping).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct NoDistortion;

impl<T: RealField> DistortionModel<T> for NoDistortion {
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T> {
        n_undist.clone()
    }

    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T> {
        n_dist.clone()
    }
}

/// Solve `distort(n_u) = n_dist` for `n_u` by Newton's method, starting at
/// `n_dist`.
///
/// `eval(x, y)` returns the distorted point and its 2×2 Jacobian
/// `[[∂x_d/∂x, ∂x_d/∂y], [∂y_d/∂x, ∂y_d/∂y]]`. Convergence is quadratic:
/// wide-angle Brown-Conrady corners (k1 ≈ −0.35 at a normalized radius of
/// ~0.8) reach machine precision in four steps, where a fixed-point
/// iteration needs about 25.
///
/// Each step is halved (up to 30 times) until it reduces the residual, so a
/// Newton step never makes the estimate worse. The iteration stops when the
/// step is below a few ULPs of the point, when the residual is exactly zero,
/// when the Jacobian is singular, or after `max_iters` steps. Under
/// dual-number evaluation the tests compare real parts, and the converged
/// point carries the derivative of the exact inverse.
fn newton_undistort<T, F>(n_dist: &Point2<T>, max_iters: u32, eval: F) -> Point2<T>
where
    T: RealField,
    F: Fn(T, T) -> ((T, T), [[T; 2]; 2]),
{
    let (xd, yd) = (n_dist.x.clone(), n_dist.y.clone());
    let half = T::from_f64(0.5).unwrap();
    let tol = T::from_f64(4.0).unwrap() * T::default_epsilon();
    let tol2 = tol.clone() * tol;

    let mut x = xd.clone();
    let mut y = yd.clone();
    let ((fx, fy), mut jac) = eval(x.clone(), y.clone());
    let mut ex = fx - xd.clone();
    let mut ey = fy - yd.clone();
    let mut res2 = ex.clone() * ex.clone() + ey.clone() * ey.clone();

    for _ in 0..max_iters {
        if res2 == T::zero() {
            break;
        }
        let [[a, b], [c, d]] = jac.clone();
        let det = a.clone() * d.clone() - b.clone() * c.clone();
        if det == T::zero() || !det.is_finite() {
            break;
        }
        // δ = J⁻¹·e
        let mut dx = (d * ex.clone() - b * ey.clone()) / det.clone();
        let mut dy = (a * ey.clone() - c * ex.clone()) / det;

        let mut accepted = None;
        for _ in 0..30 {
            let (nx, ny) = (x.clone() - dx.clone(), y.clone() - dy.clone());
            let ((gx, gy), njac) = eval(nx.clone(), ny.clone());
            let (nex, ney) = (gx - xd.clone(), gy - yd.clone());
            let nres2 = nex.clone() * nex.clone() + ney.clone() * ney.clone();
            if nres2.is_finite() && nres2 < res2 {
                accepted = Some((nx, ny, nex, ney, nres2, njac));
                break;
            }
            dx *= half.clone();
            dy *= half.clone();
        }
        let Some((nx, ny, nex, ney, nres2, njac)) = accepted else {
            break;
        };
        let step2 = dx.clone() * dx + dy.clone() * dy;
        (x, y, ex, ey, res2, jac) = (nx, ny, nex, ney, nres2, njac);
        if step2 <= tol2.clone() * (T::one() + x.clone() * x.clone() + y.clone() * y.clone()) {
            break;
        }
    }
    // One more Newton step, without a line search. At a converged point it
    // moves the value by at most rounding. Under dual-number evaluation the
    // stopping tests above see only the real part, so the loop can exit
    // (e.g. at zero distortion, where the residual is exactly zero) before
    // the derivative part has moved; one step at the converged point sets it
    // to the derivative of the exact inverse.
    let [[a, b], [c, d]] = jac;
    let det = a.clone() * d.clone() - b.clone() * c.clone();
    if det != T::zero() && det.is_finite() {
        x -= (d * ex.clone() - b * ey.clone()) / det.clone();
        y -= (a * ey - c * ex) / det;
    }
    Point2::new(x, y)
}

/// Jacobian of `(x·R + x_tan, y·R + y_tan)` for a radial factor `R(r²)` with
/// derivative `dR = dR/d(r²)` and Brown-Conrady tangential terms.
fn radial_tangential_jacobian<T: RealField>(
    x: T,
    y: T,
    radial: T,
    d_radial: T,
    p1: T,
    p2: T,
) -> [[T; 2]; 2] {
    let two = T::one() + T::one();
    let six = T::from_f64(6.0).unwrap();
    let cross = two.clone() * x.clone() * y.clone() * d_radial.clone()
        + two.clone() * p1.clone() * x.clone()
        + two.clone() * p2.clone() * y.clone();
    [
        [
            radial.clone()
                + two.clone() * x.clone() * x.clone() * d_radial.clone()
                + two.clone() * p1.clone() * y.clone()
                + six.clone() * p2.clone() * x.clone(),
            cross.clone(),
        ],
        [
            cross,
            radial + two.clone() * y.clone() * y.clone() * d_radial + six * p1 * y + two * p2 * x,
        ],
    ]
}

/// Brown-Conrady radial and tangential terms of `(x, y)`: the radial factor
/// `1 + k1·r² + k2·r⁴ + k3·r⁶` and the tangential offsets.
fn brown_conrady_terms<T: RealField>(x: &T, y: &T, k: [&T; 3], p1: &T, p2: &T) -> (T, T, T, T) {
    let r2 = x.clone() * x.clone() + y.clone() * y.clone();
    let r4 = r2.clone() * r2.clone();
    let r6 = r4.clone() * r2.clone();

    let radial = T::one() + k[0].clone() * r2.clone() + k[1].clone() * r4 + k[2].clone() * r6;

    let two = T::one() + T::one();
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let xy = x.clone() * y.clone();

    let x_tan =
        two.clone() * p1.clone() * xy.clone() + p2.clone() * (r2.clone() + two.clone() * x2);
    let y_tan = p1.clone() * (r2.clone() + two.clone() * y2) + two * p2.clone() * xy;
    (radial, x_tan, y_tan, r2)
}

/// Brown-Conrady 5-parameter radial-tangential distortion model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct BrownConrady5<S: RealField> {
    /// Radial coefficient k1.
    pub k1: S,
    /// Radial coefficient k2.
    pub k2: S,
    /// Radial coefficient k3.
    pub k3: S,
    /// Tangential coefficient p1.
    pub p1: S,
    /// Tangential coefficient p2.
    pub p2: S,
    /// Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
    pub iters: u32,
}

impl<T: RealField> BrownConrady5<T> {
    /// The forward map and its Jacobian, for Newton undistortion.
    fn distort_with_jacobian(&self, x: T, y: T) -> ((T, T), [[T; 2]; 2]) {
        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let r4 = r2.clone() * r2.clone();
        let two = T::one() + T::one();
        let three = two.clone() + T::one();
        let radial = T::one()
            + self.k1.clone() * r2.clone()
            + self.k2.clone() * r4.clone()
            + self.k3.clone() * r4.clone() * r2.clone();
        let d_radial = self.k1.clone() + two * self.k2.clone() * r2 + three * self.k3.clone() * r4;
        (
            self.distort_impl(x.clone(), y.clone()),
            radial_tangential_jacobian(x, y, radial, d_radial, self.p1.clone(), self.p2.clone()),
        )
    }

    fn distort_impl(&self, x: T, y: T) -> (T, T) {
        let (radial, x_tan, y_tan, _) =
            brown_conrady_terms(&x, &y, [&self.k1, &self.k2, &self.k3], &self.p1, &self.p2);
        (x * radial.clone() + x_tan, y * radial + y_tan)
    }
}

impl<T: RealField> DistortionModel<T> for BrownConrady5<T> {
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T> {
        let (xd, yd) = self.distort_impl(n_undist.x.clone(), n_undist.y.clone());
        Point2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T> {
        let iters = if self.iters == 0 { 20 } else { self.iters };
        newton_undistort(n_dist, iters, |x, y| self.distort_with_jacobian(x, y))
    }
}

/// OpenCV rational polynomial 8-parameter distortion model.
///
/// Parameters: `[k1, k2, k3, k4, k5, k6, p1, p2]`.
///
/// Forward map: `x_d = x * (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶) + x_tan`,
/// where the tangential correction `x_tan` follows the Brown-Conrady convention.
///
/// Undistortion solves the forward map by Newton's method (at most `iters`
/// steps, default 20).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RationalPolynomial<S: RealField> {
    /// Numerator radial coefficient k1.
    pub k1: S,
    /// Numerator radial coefficient k2.
    pub k2: S,
    /// Numerator radial coefficient k3.
    pub k3: S,
    /// Denominator radial coefficient k4.
    pub k4: S,
    /// Denominator radial coefficient k5.
    pub k5: S,
    /// Denominator radial coefficient k6.
    pub k6: S,
    /// Tangential coefficient p1.
    pub p1: S,
    /// Tangential coefficient p2.
    pub p2: S,
    /// Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
    pub iters: u32,
}

impl<T: RealField> RationalPolynomial<T> {
    /// The forward map and its Jacobian, for Newton undistortion.
    fn distort_with_jacobian(&self, x: T, y: T) -> ((T, T), [[T; 2]; 2]) {
        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let r4 = r2.clone() * r2.clone();
        let two = T::one() + T::one();
        let three = two.clone() + T::one();
        let num = T::one()
            + self.k1.clone() * r2.clone()
            + self.k2.clone() * r4.clone()
            + self.k3.clone() * r4.clone() * r2.clone();
        let den = T::one()
            + self.k4.clone() * r2.clone()
            + self.k5.clone() * r4.clone()
            + self.k6.clone() * r4.clone() * r2.clone();
        let d_num = self.k1.clone()
            + two.clone() * self.k2.clone() * r2.clone()
            + three.clone() * self.k3.clone() * r4.clone();
        let d_den = self.k4.clone() + two * self.k5.clone() * r2 + three * self.k6.clone() * r4;
        let radial = num.clone() / den.clone();
        let d_radial = (d_num * den.clone() - num * d_den) / (den.clone() * den);
        (
            self.distort_impl(x.clone(), y.clone()),
            radial_tangential_jacobian(x, y, radial, d_radial, self.p1.clone(), self.p2.clone()),
        )
    }

    fn distort_impl(&self, x: T, y: T) -> (T, T) {
        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let r4 = r2.clone() * r2.clone();
        let r6 = r4.clone() * r2.clone();

        let num = T::one()
            + self.k1.clone() * r2.clone()
            + self.k2.clone() * r4.clone()
            + self.k3.clone() * r6.clone();
        let den =
            T::one() + self.k4.clone() * r2.clone() + self.k5.clone() * r4 + self.k6.clone() * r6;
        let radial = num / den;

        let two = T::one() + T::one();
        let x2 = x.clone() * x.clone();
        let y2 = y.clone() * y.clone();
        let xy = x.clone() * y.clone();

        let x_tan = two.clone() * self.p1.clone() * xy.clone()
            + self.p2.clone() * (r2.clone() + two.clone() * x2);
        let y_tan = self.p1.clone() * (r2 + two.clone() * y2) + two * self.p2.clone() * xy;

        (x * radial.clone() + x_tan, y * radial + y_tan)
    }
}

impl<T: RealField> DistortionModel<T> for RationalPolynomial<T> {
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T> {
        let (xd, yd) = self.distort_impl(n_undist.x.clone(), n_undist.y.clone());
        Point2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T> {
        // Newton on the forward map. It converges within the invertible part
        // of the field; beyond the fold of a strong wide-FOV model (where the
        // forward map stops being monotone in r) no inverse exists and the
        // result is the last step that reduced the residual.
        let iters = if self.iters == 0 { 20 } else { self.iters };
        newton_undistort(n_dist, iters, |x, y| self.distort_with_jacobian(x, y))
    }
}

/// Brown-Conrady + thin-prism 9-parameter distortion model.
///
/// Parameters: `[k1, k2, k3, p1, p2, s1, s2, s3, s4]`.
///
/// Extends the standard Brown-Conrady model with four thin-prism coefficients
/// that add higher-order sensor shift terms:
/// `x_d += s1·r² + s2·r⁴`,  `y_d += s3·r² + s4·r⁴`.
///
/// Undistortion solves the forward map by Newton's method (at most `iters`
/// steps, default 20).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ThinPrism<S: RealField> {
    /// Radial coefficient k1.
    pub k1: S,
    /// Radial coefficient k2.
    pub k2: S,
    /// Radial coefficient k3.
    pub k3: S,
    /// Tangential coefficient p1.
    pub p1: S,
    /// Tangential coefficient p2.
    pub p2: S,
    /// Thin-prism coefficient s1 (x correction, r²).
    pub s1: S,
    /// Thin-prism coefficient s2 (x correction, r⁴).
    pub s2: S,
    /// Thin-prism coefficient s3 (y correction, r²).
    pub s3: S,
    /// Thin-prism coefficient s4 (y correction, r⁴).
    pub s4: S,
    /// Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
    pub iters: u32,
}

impl<T: RealField> ThinPrism<T> {
    /// The forward map and its Jacobian, for Newton undistortion.
    fn distort_with_jacobian(&self, x: T, y: T) -> ((T, T), [[T; 2]; 2]) {
        let r2 = x.clone() * x.clone() + y.clone() * y.clone();
        let r4 = r2.clone() * r2.clone();
        let two = T::one() + T::one();
        let three = two.clone() + T::one();
        let radial = T::one()
            + self.k1.clone() * r2.clone()
            + self.k2.clone() * r4.clone()
            + self.k3.clone() * r4.clone() * r2.clone();
        let d_radial = self.k1.clone()
            + two.clone() * self.k2.clone() * r2.clone()
            + three * self.k3.clone() * r4;
        let mut jac = radial_tangential_jacobian(
            x.clone(),
            y.clone(),
            radial,
            d_radial,
            self.p1.clone(),
            self.p2.clone(),
        );
        // Thin-prism terms: s1·r² + s2·r⁴ in x, s3·r² + s4·r⁴ in y.
        let dpx = two.clone() * (self.s1.clone() + two.clone() * self.s2.clone() * r2.clone());
        let dpy = two.clone() * (self.s3.clone() + two * self.s4.clone() * r2);
        jac[0][0] += dpx.clone() * x.clone();
        jac[0][1] += dpx * y.clone();
        jac[1][0] += dpy.clone() * x.clone();
        jac[1][1] += dpy * y.clone();
        (self.distort_impl(x, y), jac)
    }

    fn distort_impl(&self, x: T, y: T) -> (T, T) {
        let (radial, x_tan, y_tan, r2) =
            brown_conrady_terms(&x, &y, [&self.k1, &self.k2, &self.k3], &self.p1, &self.p2);
        let r4 = r2.clone() * r2.clone();

        // Thin-prism terms.
        let prism_x = self.s1.clone() * r2.clone() + self.s2.clone() * r4.clone();
        let prism_y = self.s3.clone() * r2 + self.s4.clone() * r4;

        (
            x * radial.clone() + x_tan + prism_x,
            y * radial + y_tan + prism_y,
        )
    }
}

impl<T: RealField> DistortionModel<T> for ThinPrism<T> {
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T> {
        let (xd, yd) = self.distort_impl(n_undist.x.clone(), n_undist.y.clone());
        Point2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T> {
        let iters = if self.iters == 0 { 20 } else { self.iters };
        newton_undistort(n_dist, iters, |x, y| self.distort_with_jacobian(x, y))
    }
}

/// Fitzgibbon single-parameter division (fisheye) distortion model.
///
/// Parameter: `[lambda]`.
///
/// Both distortion and undistortion have **closed-form** solutions:
///
/// - **Undistort** (distorted → undistorted, natural direction):
///   `(x_u, y_u) = (x_d, y_d) / (1 + lambda · r_d²)`
///
/// - **Distort** (undistorted → distorted, via quadratic):
///   `scale = (1 − sqrt(1 − 4·lambda·r_u²)) / (2·lambda·r_u²)`,
///   `(x_d, y_d) = (x_u · scale, y_u · scale)`. Evaluated in the rationalized
///   form `scale = 2 / (1 + sqrt(1 − 4·lambda·r_u²))`, which has no `lambda` in
///   the denominator and stays analytic at `lambda = 0` (`scale → 1`,
///   `∂scale/∂lambda → r_u²`), so the parameter remains observable from a
///   zero seed under autodiff.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct Division<S: RealField> {
    /// Division distortion coefficient lambda.
    pub lambda: S,
}

impl<T: RealField> DistortionModel<T> for Division<T> {
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T> {
        let x = n_undist.x.clone();
        let y = n_undist.y.clone();
        let r_u2 = x.clone() * x.clone() + y.clone() * y.clone();

        let four = T::from_f64(4.0).unwrap();
        // disc = 1 - 4·λ·r_u²; clamp ≥ 0 for inputs beyond the invertible range.
        let disc = T::one() - four * self.lambda.clone() * r_u2;
        let disc = if disc < T::zero() { T::zero() } else { disc };

        // scale = (1 - √disc)/(2·λ·r_u²), rationalized to 2/(1 + √disc): no λ in
        // the denominator, analytic at λ = 0 (scale → 1, ∂scale/∂λ → r_u²).
        let two = T::one() + T::one();
        let scale = two / (T::one() + disc.sqrt());

        Point2::new(x * scale.clone(), y * scale)
    }

    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T> {
        let x = n_dist.x.clone();
        let y = n_dist.y.clone();
        let r_d2 = x.clone() * x.clone() + y.clone() * y.clone();
        let factor = T::one() / (T::one() + self.lambda.clone() * r_d2);
        Point2::new(x * factor.clone(), y * factor)
    }
}

/// Kannala-Brandt radial distortion: `r_d = r · (1 + k1·r² + k2·r⁴ + k3·r⁶ +
/// k4·r⁸)`, an odd polynomial in the radius of the normalized point.
///
/// Parameters: `[k1, k2, k3, k4]`. On [`Equidistant`](super::Equidistant)
/// coordinates the radius is the incidence angle `θ`, which makes the camera
/// the Kannala-Brandt fisheye model (OpenCV `fisheye`, Kalibr
/// `pinhole-equi` on a ray-angle basis); on pinhole coordinates it is a
/// radial-only Brown-Conrady model with a fourth term.
///
/// Undistortion solves the forward map by Newton's method (at most `iters`
/// steps, default 20).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct KannalaBrandt4<S: RealField> {
    /// Coefficient of `r²`.
    pub k1: S,
    /// Coefficient of `r⁴`.
    pub k2: S,
    /// Coefficient of `r⁶`.
    pub k3: S,
    /// Coefficient of `r⁸`.
    pub k4: S,
    /// Maximum Newton iterations of [`DistortionModel::undistort`] (0 → 20).
    pub iters: u32,
}

impl<T: RealField> KannalaBrandt4<T> {
    /// The radial factor `1 + k1·r² + … + k4·r⁸` and its derivative with
    /// respect to `r²`.
    fn radial(&self, r2: T) -> (T, T) {
        let two = T::one() + T::one();
        let three = two.clone() + T::one();
        let four = three.clone() + T::one();
        let r4 = r2.clone() * r2.clone();
        let r6 = r4.clone() * r2.clone();
        let radial = T::one()
            + self.k1.clone() * r2.clone()
            + self.k2.clone() * r4.clone()
            + self.k3.clone() * r6.clone()
            + self.k4.clone() * r6.clone() * r2.clone();
        let d_radial = self.k1.clone()
            + two * self.k2.clone() * r2
            + three * self.k3.clone() * r4
            + four * self.k4.clone() * r6;
        (radial, d_radial)
    }

    fn distort_with_jacobian(&self, x: T, y: T) -> ((T, T), [[T; 2]; 2]) {
        let (radial, d_radial) = self.radial(x.clone() * x.clone() + y.clone() * y.clone());
        let jac = radial_tangential_jacobian(
            x.clone(),
            y.clone(),
            radial.clone(),
            d_radial,
            T::zero(),
            T::zero(),
        );
        ((x * radial.clone(), y * radial), jac)
    }
}

impl<T: RealField> DistortionModel<T> for KannalaBrandt4<T> {
    fn distort(&self, n_undist: &Point2<T>) -> Point2<T> {
        let (x, y) = (n_undist.x.clone(), n_undist.y.clone());
        let (radial, _) = self.radial(x.clone() * x.clone() + y.clone() * y.clone());
        Point2::new(x * radial.clone(), y * radial)
    }

    fn undistort(&self, n_dist: &Point2<T>) -> Point2<T> {
        let iters = if self.iters == 0 { 20 } else { self.iters };
        newton_undistort(n_dist, iters, |x, y| self.distort_with_jacobian(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    fn grid() -> Vec<(f64, f64)> {
        let vals = [-0.4, -0.2, 0.0, 0.2, 0.4];
        vals.iter()
            .flat_map(|&x| vals.iter().map(move |&y| (x, y)))
            .collect()
    }

    // ── KannalaBrandt4 ──────────────────────────────────────────────────────

    /// OpenCV `fisheye::distortPoints` maps the equidistant point of angle
    /// `θ` to `θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)`; the model
    /// reproduces it, and inverts it wherever `θ_d(θ)` is increasing: the
    /// fitted range of a real lens (OpenCV's test fisheye reaches about
    /// 1.2 rad), and beyond 90° for a milder polynomial.
    #[test]
    fn kannala_brandt_matches_the_fisheye_polynomial_and_inverts() {
        let cases = [
            ([-0.0015, -0.0033, 0.0061, -0.0037], 1.3),
            ([-0.02, 0.002, -0.0001, 0.0], 2.6),
        ];
        for (k, max_theta) in cases {
            let m = KannalaBrandt4 {
                k1: k[0],
                k2: k[1],
                k3: k[2],
                k4: k[3],
                iters: 0,
            };
            for i in 0..=20 {
                let theta = max_theta * f64::from(i) / 20.0;
                let phi: f64 = 0.7;
                let p = Point2::new(theta * phi.cos(), theta * phi.sin());
                let d = m.distort(&p);
                let t2 = theta * theta;
                let expected = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + k[3] * t2))));
                assert!((d.coords.norm() - expected).abs() < 1e-14, "θ = {theta}");
                let u = m.undistort(&d);
                assert!((u - p).norm() < 1e-12, "θ = {theta}: {u:?}");
            }
        }
    }

    // ── RationalPolynomial ──────────────────────────────────────────────────

    #[test]
    fn rational_zero_params_is_identity() {
        let m = RationalPolynomial::<f64>::default();
        for (x, y) in grid() {
            let p = Point2::new(x, y);
            let d = m.distort(&p);
            assert!(
                (d.x - x).abs() < 1e-14 && (d.y - y).abs() < 1e-14,
                "zero rational distort not identity at ({x},{y}): {d:?}"
            );
            let u = m.undistort(&p);
            assert!(
                (u.x - x).abs() < 1e-14 && (u.y - y).abs() < 1e-14,
                "zero rational undistort not identity at ({x},{y}): {u:?}"
            );
        }
    }

    #[test]
    fn rational_distort_undistort_roundtrip() {
        let m = RationalPolynomial {
            k1: -0.3,
            k2: 0.1,
            k3: 0.0,
            k4: 0.01,
            k5: 0.0,
            k6: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 10,
        };
        for (x, y) in grid() {
            let p = Point2::new(x, y);
            let d = m.distort(&p);
            let u = m.undistort(&d);
            assert!(
                (u.x - x).abs() < 1e-4 && (u.y - y).abs() < 1e-4,
                "rational roundtrip failed at ({x},{y}): d={d:?} u={u:?}"
            );
        }
    }

    /// A strong positive-k1 pincushion at normalized radius ~1
    /// must round-trip through the iterative inverse without diverging or NaN.
    /// The previous `x -= distort(x) - x_d` update overshot badly here.
    #[test]
    fn rational_undistort_stable_strong_pincushion() {
        let m = RationalPolynomial {
            k1: 0.3,
            k2: 0.05,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 20,
        };
        let vals: [f64; 5] = [-0.7, -0.4, 0.0, 0.4, 0.7]; // radius up to ~1.0
        for &x in &vals {
            for &y in &vals {
                let p = Point2::new(x, y);
                let d = m.distort(&p);
                let u = m.undistort(&d);
                assert!(
                    u.x.is_finite() && u.y.is_finite(),
                    "non-finite undistort at ({x},{y}): {u:?}"
                );
                assert!(
                    (u.x - x).abs() < 1e-3 && (u.y - y).abs() < 1e-3,
                    "strong-pincushion roundtrip failed at ({x},{y}): d={d:?} u={u:?}"
                );
            }
        }
    }

    // ── ThinPrism ───────────────────────────────────────────────────────────

    #[test]
    fn thin_prism_zero_params_is_identity() {
        let m = ThinPrism::<f64>::default();
        for (x, y) in grid() {
            let p = Point2::new(x, y);
            let d = m.distort(&p);
            assert!(
                (d.x - x).abs() < 1e-14 && (d.y - y).abs() < 1e-14,
                "zero thin_prism distort not identity at ({x},{y}): {d:?}"
            );
            let u = m.undistort(&p);
            assert!(
                (u.x - x).abs() < 1e-14 && (u.y - y).abs() < 1e-14,
                "zero thin_prism undistort not identity at ({x},{y}): {u:?}"
            );
        }
    }

    #[test]
    fn thin_prism_distort_undistort_roundtrip() {
        let m = ThinPrism {
            k1: -0.3,
            k2: 0.1,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            s1: 0.001,
            s2: -0.0005,
            s3: 0.0008,
            s4: -0.0003,
            iters: 10,
        };
        for (x, y) in grid() {
            let p = Point2::new(x, y);
            let d = m.distort(&p);
            let u = m.undistort(&d);
            assert!(
                (u.x - x).abs() < 1e-4 && (u.y - y).abs() < 1e-4,
                "thin_prism roundtrip failed at ({x},{y}): d={d:?} u={u:?}"
            );
        }
    }

    // ── Division ────────────────────────────────────────────────────────────

    #[test]
    fn division_zero_lambda_is_identity() {
        let m = Division::<f64> { lambda: 0.0 };
        for (x, y) in grid() {
            let p = Point2::new(x, y);
            let d = m.distort(&p);
            assert!(
                (d.x - x).abs() < 1e-14 && (d.y - y).abs() < 1e-14,
                "zero division distort not identity at ({x},{y}): {d:?}"
            );
            let u = m.undistort(&p);
            assert!(
                (u.x - x).abs() < 1e-14 && (u.y - y).abs() < 1e-14,
                "zero division undistort not identity at ({x},{y}): {u:?}"
            );
        }
    }

    #[test]
    fn division_distort_undistort_roundtrip() {
        let m = Division { lambda: -0.2_f64 };
        for (x, y) in grid() {
            let p = Point2::new(x, y);
            let d = m.distort(&p);
            let u = m.undistort(&d);
            assert!(
                (u.x - x).abs() < 1e-9 && (u.y - y).abs() < 1e-9,
                "division roundtrip failed at ({x},{y}): d={d:?} u={u:?}"
            );
        }
    }
    // ── Newton undistortion ─────────────────────────────────────────────────

    /// Wide-angle lenses at whose image corners a plain fixed-point
    /// undistortion fails to converge.
    fn wide_angle_corner_models() -> Vec<(&'static str, Box<dyn DistortionModel<f64>>)> {
        vec![
            (
                "brown mild",
                Box::new(BrownConrady5 {
                    k1: -0.08,
                    k2: 0.02,
                    ..Default::default()
                }),
            ),
            (
                "brown wide-angle barrel",
                Box::new(BrownConrady5 {
                    k1: -0.35,
                    k2: 0.15,
                    k3: -0.03,
                    p1: 5e-4,
                    p2: -3e-4,
                    iters: 0,
                }),
            ),
            (
                "brown pincushion",
                Box::new(BrownConrady5 {
                    k1: 0.15,
                    k2: 0.05,
                    ..Default::default()
                }),
            ),
            (
                "rational",
                Box::new(RationalPolynomial {
                    k1: 0.8,
                    k2: 0.2,
                    k3: 0.01,
                    k4: 1.1,
                    k5: 0.35,
                    k6: 0.02,
                    p1: 1e-4,
                    p2: 1e-4,
                    iters: 0,
                }),
            ),
            (
                "thin prism",
                Box::new(ThinPrism {
                    k1: -0.1,
                    k2: 0.03,
                    s1: 1e-3,
                    s2: -5e-4,
                    s3: 8e-4,
                    s4: 2e-4,
                    ..Default::default()
                }),
            ),
        ]
    }

    /// `distort(undistort(d)) = d` to ~1e-15 over the distorted normalized
    /// field of a 2048×1536, f = 1800 px image (corners at |x| 0.57, |y| 0.43),
    /// extended to ±0.65 × ±0.5 to cover a 4°-tilted Scheimpflug sensor. (The
    /// wide-angle barrel folds at a distorted radius of ~0.945: beyond it
    /// there is no inverse.) At f = 1800 the bound is ~2e-12 px.
    #[test]
    fn newton_undistort_converges_at_the_corners() {
        let n = 41;
        for (name, m) in wide_angle_corner_models() {
            let mut worst = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    let x = -0.65 + 1.3 * i as f64 / (n - 1) as f64;
                    let y = -0.5 + 1.0 * j as f64 / (n - 1) as f64;
                    let d = Point2::new(x, y);
                    let back = m.distort(&m.undistort(&d));
                    worst = worst.max((back - d).norm());
                }
            }
            assert!(worst < 1e-15, "{name}: worst residual {worst:e}");
        }
    }

    /// The analytic Jacobians agree with central differences.
    #[test]
    fn newton_jacobians_match_finite_differences() {
        fn check(name: &str, f: impl Fn(f64, f64) -> ((f64, f64), [[f64; 2]; 2])) {
            let h = 1e-6;
            for &(x, y) in &[(0.31, -0.22), (-0.55, 0.41), (0.02, 0.6), (0.7, 0.5)] {
                let (_, jac) = f(x, y);
                let d = |dx: f64, dy: f64| {
                    let ((px, py), _) = f(x + dx, y + dy);
                    let ((mx, my), _) = f(x - dx, y - dy);
                    ((px - mx) / (2.0 * h), (py - my) / (2.0 * h))
                };
                let (dxx, dyx) = d(h, 0.0);
                let (dxy, dyy) = d(0.0, h);
                let fd = [[dxx, dxy], [dyx, dyy]];
                for r in 0..2 {
                    for c in 0..2 {
                        assert!(
                            (jac[r][c] - fd[r][c]).abs() < 1e-8,
                            "{name}: J[{r}][{c}] = {} vs {} at ({x},{y})",
                            jac[r][c],
                            fd[r][c]
                        );
                    }
                }
            }
        }
        let bc = BrownConrady5 {
            k1: -0.35,
            k2: 0.15,
            k3: -0.03,
            p1: 5e-4,
            p2: -3e-4,
            iters: 0,
        };
        check("brown", |x, y| bc.distort_with_jacobian(x, y));
        let rp = RationalPolynomial {
            k1: 0.8,
            k2: 0.2,
            k3: 0.01,
            k4: 1.1,
            k5: 0.35,
            k6: 0.02,
            p1: 1e-3,
            p2: -2e-3,
            iters: 0,
        };
        check("rational", |x, y| rp.distort_with_jacobian(x, y));
        let tp = ThinPrism {
            k1: -0.1,
            k2: 0.03,
            k3: 0.01,
            p1: 1e-3,
            p2: -1e-3,
            s1: 2e-3,
            s2: -5e-4,
            s3: 8e-4,
            s4: 2e-4,
            iters: 0,
        };
        check("thin prism", |x, y| tp.distort_with_jacobian(x, y));
    }
}
