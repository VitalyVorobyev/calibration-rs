use nalgebra::{Point2, RealField};
use serde::{Deserialize, Serialize};

/// Distortion model mapping between ideal and distorted normalized coordinates.
pub trait DistortionModel<S: RealField + Copy> {
    /// Apply distortion to undistorted normalized coordinates.
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S>;
    /// Remove distortion from distorted normalized coordinates.
    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S>;
}

/// No distortion (identity mapping).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct NoDistortion;

impl<S: RealField + Copy> DistortionModel<S> for NoDistortion {
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S> {
        *n_undist
    }

    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S> {
        *n_dist
    }
}

/// Solve `distort(n_u) = n_dist` for `n_u` by Newton's method, starting at
/// `n_dist`.
///
/// `eval(x, y)` returns the distorted point and its 2×2 Jacobian
/// `[[∂x_d/∂x, ∂x_d/∂y], [∂y_d/∂x, ∂y_d/∂y]]`. Convergence is quadratic:
/// wide-angle Brown-Conrady corners (k1 ≈ −0.35 at a normalized radius of
/// ~0.8) reach machine precision in four steps, where the former
/// fixed-point iteration needed 25 (calibration-rs#120).
///
/// Each step is halved (up to 30 times) until it reduces the residual, so a
/// Newton step never makes the estimate worse. The iteration stops when the
/// step is below a few ULPs of the point, when the residual is exactly zero,
/// when the Jacobian is singular, or after `max_iters` steps.
fn newton_undistort<S, F>(n_dist: &Point2<S>, max_iters: u32, eval: F) -> Point2<S>
where
    S: RealField + Copy,
    F: Fn(S, S) -> ((S, S), [[S; 2]; 2]),
{
    let (xd, yd) = (n_dist.x, n_dist.y);
    let half = S::from_f64(0.5).unwrap();
    let tol = S::from_f64(4.0).unwrap() * S::default_epsilon();
    let tol2 = tol * tol;

    let mut x = xd;
    let mut y = yd;
    let ((fx, fy), mut jac) = eval(x, y);
    let mut ex = fx - xd;
    let mut ey = fy - yd;
    let mut res2 = ex * ex + ey * ey;

    for _ in 0..max_iters {
        if res2 == S::zero() {
            break;
        }
        let [[a, b], [c, d]] = jac;
        let det = a * d - b * c;
        if det == S::zero() || !det.is_finite() {
            break;
        }
        // δ = J⁻¹·e
        let mut dx = (d * ex - b * ey) / det;
        let mut dy = (a * ey - c * ex) / det;

        let mut accepted = None;
        for _ in 0..30 {
            let (nx, ny) = (x - dx, y - dy);
            let ((gx, gy), njac) = eval(nx, ny);
            let (nex, ney) = (gx - xd, gy - yd);
            let nres2 = nex * nex + ney * ney;
            if nres2.is_finite() && nres2 < res2 {
                accepted = Some((nx, ny, nex, ney, nres2, njac));
                break;
            }
            dx *= half;
            dy *= half;
        }
        let Some((nx, ny, nex, ney, nres2, njac)) = accepted else {
            break;
        };
        let step2 = dx * dx + dy * dy;
        (x, y, ex, ey, res2, jac) = (nx, ny, nex, ney, nres2, njac);
        if step2 <= tol2 * (S::one() + x * x + y * y) {
            break;
        }
    }
    Point2::new(x, y)
}

/// Jacobian of `(x·R + x_tan, y·R + y_tan)` for a radial factor `R(r²)` with
/// derivative `dR = dR/d(r²)` and Brown-Conrady tangential terms.
fn radial_tangential_jacobian<S: RealField + Copy>(
    x: S,
    y: S,
    radial: S,
    d_radial: S,
    p1: S,
    p2: S,
) -> [[S; 2]; 2] {
    let two = S::one() + S::one();
    let six = S::from_f64(6.0).unwrap();
    let cross = two * x * y * d_radial + two * p1 * x + two * p2 * y;
    [
        [
            radial + two * x * x * d_radial + two * p1 * y + six * p2 * x,
            cross,
        ],
        [
            cross,
            radial + two * y * y * d_radial + six * p1 * y + two * p2 * x,
        ],
    ]
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

impl<S: RealField + Copy> BrownConrady5<S> {
    /// The forward map and its Jacobian, for Newton undistortion.
    fn distort_with_jacobian(&self, x: S, y: S) -> ((S, S), [[S; 2]; 2]) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let two = S::one() + S::one();
        let three = two + S::one();
        let radial = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r4 * r2;
        let d_radial = self.k1 + two * self.k2 * r2 + three * self.k3 * r4;
        (
            self.distort_impl(x, y),
            radial_tangential_jacobian(x, y, radial, d_radial, self.p1, self.p2),
        )
    }

    fn distort_impl(&self, x: S, y: S) -> (S, S) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let two = S::one() + S::one();
        let x2 = x * x;
        let y2 = y * y;
        let xy = x * y;

        let x_tan = two * self.p1 * xy + self.p2 * (r2 + two * x2);
        let y_tan = self.p1 * (r2 + two * y2) + two * self.p2 * xy;

        (x * radial + x_tan, y * radial + y_tan)
    }
}

impl<S: RealField + Copy> DistortionModel<S> for BrownConrady5<S> {
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S> {
        let (xd, yd) = self.distort_impl(n_undist.x, n_undist.y);
        Point2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S> {
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

impl<S: RealField + Copy> RationalPolynomial<S> {
    /// The forward map and its Jacobian, for Newton undistortion.
    fn distort_with_jacobian(&self, x: S, y: S) -> ((S, S), [[S; 2]; 2]) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let two = S::one() + S::one();
        let three = two + S::one();
        let num = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r4 * r2;
        let den = S::one() + self.k4 * r2 + self.k5 * r4 + self.k6 * r4 * r2;
        let d_num = self.k1 + two * self.k2 * r2 + three * self.k3 * r4;
        let d_den = self.k4 + two * self.k5 * r2 + three * self.k6 * r4;
        let radial = num / den;
        let d_radial = (d_num * den - num * d_den) / (den * den);
        (
            self.distort_impl(x, y),
            radial_tangential_jacobian(x, y, radial, d_radial, self.p1, self.p2),
        )
    }

    fn distort_impl(&self, x: S, y: S) -> (S, S) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let num = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let den = S::one() + self.k4 * r2 + self.k5 * r4 + self.k6 * r6;
        let radial = num / den;

        let two = S::one() + S::one();
        let x2 = x * x;
        let y2 = y * y;
        let xy = x * y;

        let x_tan = two * self.p1 * xy + self.p2 * (r2 + two * x2);
        let y_tan = self.p1 * (r2 + two * y2) + two * self.p2 * xy;

        (x * radial + x_tan, y * radial + y_tan)
    }
}

impl<S: RealField + Copy> DistortionModel<S> for RationalPolynomial<S> {
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S> {
        let (xd, yd) = self.distort_impl(n_undist.x, n_undist.y);
        Point2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S> {
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

impl<S: RealField + Copy> ThinPrism<S> {
    /// The forward map and its Jacobian, for Newton undistortion.
    fn distort_with_jacobian(&self, x: S, y: S) -> ((S, S), [[S; 2]; 2]) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let two = S::one() + S::one();
        let three = two + S::one();
        let radial = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r4 * r2;
        let d_radial = self.k1 + two * self.k2 * r2 + three * self.k3 * r4;
        let mut jac = radial_tangential_jacobian(x, y, radial, d_radial, self.p1, self.p2);
        // Thin-prism terms: s1·r² + s2·r⁴ in x, s3·r² + s4·r⁴ in y.
        let dpx = two * (self.s1 + two * self.s2 * r2);
        let dpy = two * (self.s3 + two * self.s4 * r2);
        jac[0][0] += dpx * x;
        jac[0][1] += dpx * y;
        jac[1][0] += dpy * x;
        jac[1][1] += dpy * y;
        (self.distort_impl(x, y), jac)
    }

    fn distort_impl(&self, x: S, y: S) -> (S, S) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = S::one() + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let two = S::one() + S::one();
        let x2 = x * x;
        let y2 = y * y;
        let xy = x * y;

        let x_tan = two * self.p1 * xy + self.p2 * (r2 + two * x2);
        let y_tan = self.p1 * (r2 + two * y2) + two * self.p2 * xy;

        // Add thin-prism terms
        let prism_x = self.s1 * r2 + self.s2 * r4;
        let prism_y = self.s3 * r2 + self.s4 * r4;

        (x * radial + x_tan + prism_x, y * radial + y_tan + prism_y)
    }
}

impl<S: RealField + Copy> DistortionModel<S> for ThinPrism<S> {
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S> {
        let (xd, yd) = self.distort_impl(n_undist.x, n_undist.y);
        Point2::new(xd, yd)
    }

    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S> {
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

impl<S: RealField + Copy> DistortionModel<S> for Division<S> {
    fn distort(&self, n_undist: &Point2<S>) -> Point2<S> {
        let x = n_undist.x;
        let y = n_undist.y;
        let r_u2 = x * x + y * y;

        let four = S::from_f64(4.0).unwrap();
        // disc = 1 - 4·λ·r_u²; clamp ≥ 0 for inputs beyond the invertible range.
        let disc = S::one() - four * self.lambda * r_u2;
        let disc = if disc < S::zero() { S::zero() } else { disc };

        // scale = (1 - √disc)/(2·λ·r_u²), rationalized to 2/(1 + √disc): no λ in
        // the denominator, analytic at λ = 0 (scale → 1, ∂scale/∂λ → r_u²).
        let two = S::one() + S::one();
        let scale = two / (S::one() + disc.sqrt());

        Point2::new(x * scale, y * scale)
    }

    fn undistort(&self, n_dist: &Point2<S>) -> Point2<S> {
        let x = n_dist.x;
        let y = n_dist.y;
        let r_d2 = x * x + y * y;
        let factor = S::one() / (S::one() + self.lambda * r_d2);
        Point2::new(x * factor, y * factor)
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
    // ── Newton undistortion (calibration-rs#120) ────────────────────────────

    /// The lenses of calibration-rs#120 (found by etendue gate G3.1), each of
    /// which the former fixed-point iteration failed at the image corners.
    fn issue_120_models() -> Vec<(&'static str, Box<dyn DistortionModel<f64>>)> {
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
        for (name, m) in issue_120_models() {
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
