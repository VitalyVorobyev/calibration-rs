use std::num::NonZero;

use nalgebra as na;

use tiny_solver::manifold::{AutoDiffManifold, Manifold};

use crate::backend::s2;

/// Unit vector manifold in 3D: S² ⊂ R³.
///
/// State (ambient): 3
/// Tangent: 2
///
/// Tangent coords are expressed in a local orthonormal basis (b1, b2) at x;
/// `plus` / `minus` are the shared S² maps in [`s2`].
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
}

impl<T: na::RealField> AutoDiffManifold<T> for UnitVector3Manifold {
    fn plus(&self, x: na::DVectorView<T>, delta: na::DVectorView<T>) -> na::DVector<T> {
        debug_assert_eq!(x.len(), 3);
        debug_assert_eq!(delta.len(), 2);
        let x1 = s2::plus(&Self::x3(x), &Self::d2(delta));
        na::dvector![x1[0].clone(), x1[1].clone(), x1[2].clone()]
    }

    fn minus(&self, y: na::DVectorView<T>, x: na::DVectorView<T>) -> na::DVector<T> {
        debug_assert_eq!(y.len(), 3);
        debug_assert_eq!(x.len(), 3);
        let d = s2::minus(&Self::x3(y), &Self::x3(x));
        na::dvector![d[0].clone(), d[1].clone()]
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
