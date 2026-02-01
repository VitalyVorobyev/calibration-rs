use std::num::NonZero;

use nalgebra as na;

use tiny_solver::manifold::{AutoDiffManifold, Manifold};

/// Unit vector manifold in 3D: S² ⊂ R³.
///
/// State (ambient): 3
/// Tangent: 2
///
/// Tangent coords are expressed in a local orthonormal basis (b1, b2) at x.
/// `plus` is the S² exponential map; `minus` is the S² logarithm map.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnitVector3Manifold;

impl UnitVector3Manifold {
    #[inline]
    fn x3<T: na::RealField>(x: na::DVectorView<T>) -> na::Vector3<T> {
        debug_assert_eq!(x.len(), 3);
        na::Vector3::new(x[0].clone(), x[1].clone(), x[2].clone())
    }

    #[inline]
    fn d2<T: na::RealField>(d: na::DVectorView<T>) -> na::Vector2<T> {
        debug_assert_eq!(d.len(), 2);
        na::Vector2::new(d[0].clone(), d[1].clone())
    }

    /// Build an orthonormal tangent basis (b1, b2) at x (assumed unit-ish).
    ///
    /// Deterministic: chooses a reference axis far enough from x to avoid degeneracy,
    /// then Gram–Schmidt and cross product.
    #[inline]
    fn basis<T: na::RealField>(x_unit: &na::Vector3<T>) -> (na::Vector3<T>, na::Vector3<T>) {
        // Pick an axis not too aligned with x. This uses ordering; your SO3 exp already does.
        let ax_x = na::Vector3::new(T::one(), T::zero(), T::zero());
        let ax_z = na::Vector3::new(T::zero(), T::zero(), T::one());

        // If |x.z| is small, z-axis is safe; otherwise use x-axis.
        let a = if x_unit[2].clone().abs() < T::from_f64(0.9).unwrap() {
            ax_z
        } else {
            ax_x
        };

        // b1 = normalize( a - x (x·a) )
        let proj = x_unit.clone() * x_unit.dot(&a);
        let mut b1 = (a - proj).normalize();
        // b2 = x × b1 (already unit if x,b1 unit and orthogonal)
        let b2 = x_unit.cross(&b1);

        // Extra guard (rare): if normalization produced NaNs due to a bad input x, fall back.
        // (We don't try too hard here; state is expected to be a unit vector.)
        if b1.norm_squared() == T::zero() {
            b1 = na::Vector3::new(T::zero(), T::one(), T::zero());
        }

        (b1, b2)
    }
}

impl<T: na::RealField> AutoDiffManifold<T> for UnitVector3Manifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        const EPS: f64 = 1e-6;
        debug_assert_eq!(x.len(), 3);
        debug_assert_eq!(delta.len(), 2);

        let x0 = Self::x3(x).normalize();
        let d = Self::d2(delta);

        let (b1, b2) = Self::basis(&x0);

        // Tangent vector v in R³ at x0.
        let v = b1 * d[0].clone() + b2 * d[1].clone();
        let theta2 = v.norm_squared();

        let (cos_t, sin_over_t) = if theta2 < T::from_f64(EPS * EPS).unwrap() {
            // series:
            // cos θ ≈ 1 - θ²/2
            // sin θ / θ ≈ 1 - θ²/6
            (
                T::one() - theta2.clone() / T::from_f64(2.0).unwrap(),
                T::one() - theta2 / T::from_f64(6.0).unwrap(),
            )
        } else {
            let theta = theta2.sqrt();
            let (sin_t, cos_t) = theta.clone().sin_cos();
            let sin_over_t = sin_t / theta;
            (cos_t, sin_over_t)
        };

        let x1 = (x0 * cos_t) + (v * sin_over_t);

        // Re-normalize to stay on the sphere even with numeric drift.
        let x1 = x1.normalize();

        na::dvector![x1[0].clone(), x1[1].clone(), x1[2].clone()]
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        const EPS: f64 = 1e-6;
        debug_assert_eq!(y.len(), 3);
        debug_assert_eq!(x.len(), 3);

        let x0 = Self::x3(x).normalize();
        let y0 = Self::x3(y).normalize();

        let (b1, b2) = Self::basis(&x0);

        // Log map on S² at x0.
        //
        // sinθ = ||x×y||, cosθ = x·y, θ = atan2(sinθ, cosθ)
        // u = y - (x·y)x is the tangent direction with ||u|| = sinθ
        // w = (θ / sinθ) u   (if sinθ != 0)
        let dot = x0.dot(&y0);
        let cross = x0.cross(&y0);
        let sin_t = cross.norm();

        let theta = sin_t.clone().atan2(dot.clone());

        let u = y0 - x0.clone() * dot.clone();

        let w = if sin_t < T::from_f64(EPS).unwrap() {
            // Near-parallel or near-antipodal.
            // Parallel (dot>0): log is ~0.
            // Antipodal (dot<0): direction is not unique; pick deterministic b1 * π.
            if dot > T::zero() {
                na::Vector3::zeros()
            } else {
                b1.clone() * T::from_f64(std::f64::consts::PI).unwrap()
            }
        } else {
            u * (theta / sin_t)
        };

        let d0 = b1.dot(&w);
        let d1 = b2.dot(&w);

        na::dvector![d0, d1]
    }
}

impl Manifold for UnitVector3Manifold {
    fn tangent_size(&self) -> NonZero<usize> {
        NonZero::new(2).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    fn dv(v: &[f64]) -> na::DVector<f64> {
        na::DVector::from_column_slice(v)
    }

    #[test]
    fn s2_plus_zero_is_identity() {
        let m = UnitVector3Manifold;
        let x = dv(&[0.2, -0.3, 0.932]).normalize();
        let d0 = dv(&[0.0, 0.0]);

        let y = m.plus_f64(x.as_view(), d0.as_view());
        assert!((y.clone() - x).norm() < 1e-12);
        assert!((y.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn s2_minus_same_is_zero() {
        let m = UnitVector3Manifold;
        let x = dv(&[0.4, 0.1, 0.91]).normalize();

        let d = m.minus_f64(x.as_view(), x.as_view());
        assert!(d.norm() < 1e-12);
    }

    #[test]
    fn s2_plus_minus_roundtrip_small() {
        let m = UnitVector3Manifold;
        let x = dv(&[0.4, 0.1, 0.91]).normalize();
        let d = dv(&[0.01, -0.02]);

        let y = m.plus_f64(x.as_view(), d.as_view());
        assert!((y.norm() - 1.0).abs() < 1e-12);

        let d2 = m.minus_f64(y.as_view(), x.as_view());
        assert!((d2 - d).norm() < 1e-9);
    }
}
