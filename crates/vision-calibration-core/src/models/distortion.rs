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
    /// Iterations for undistortion.
    pub iters: u32,
}

impl<S: RealField + Copy> BrownConrady5<S> {
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
        let mut x = n_dist.x;
        let mut y = n_dist.y;

        let iters = if self.iters == 0 { 8 } else { self.iters };
        for _ in 0..iters {
            let (xd, yd) = self.distort_impl(x, y);
            let ex = xd - n_dist.x;
            let ey = yd - n_dist.y;
            x -= ex;
            y -= ey;
        }
        Point2::new(x, y)
    }
}

/// OpenCV rational polynomial 8-parameter distortion model.
///
/// Parameters: `[k1, k2, k3, k4, k5, k6, p1, p2]`.
///
/// Forward map: `x_d = x * (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶) + x_tan`,
/// where the tangential correction `x_tan` follows the Brown-Conrady convention.
///
/// Undistortion uses fixed-point iteration (default 10 iterations).
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
    /// Iterations for undistortion (0 → 10).
    pub iters: u32,
}

impl<S: RealField + Copy> RationalPolynomial<S> {
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
        let mut x = n_dist.x;
        let mut y = n_dist.y;

        let iters = if self.iters == 0 { 10 } else { self.iters };
        for _ in 0..iters {
            let (xd, yd) = self.distort_impl(x, y);
            let ex = xd - n_dist.x;
            let ey = yd - n_dist.y;
            x -= ex;
            y -= ey;
        }
        Point2::new(x, y)
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
/// Undistortion uses fixed-point iteration (default 10 iterations).
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
    /// Iterations for undistortion (0 → 10).
    pub iters: u32,
}

impl<S: RealField + Copy> ThinPrism<S> {
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
        let mut x = n_dist.x;
        let mut y = n_dist.y;

        let iters = if self.iters == 0 { 10 } else { self.iters };
        for _ in 0..iters {
            let (xd, yd) = self.distort_impl(x, y);
            let ex = xd - n_dist.x;
            let ey = yd - n_dist.y;
            x -= ex;
            y -= ey;
        }
        Point2::new(x, y)
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
}
