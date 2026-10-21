use nalgebra::{Point2, RealField, Vector3};
use serde::{Deserialize, Serialize};

/// A camera ray: the points `origin + s · direction`, `s ≥ 0`.
///
/// Central projections put the origin at the camera centre; a telecentric
/// projection's rays are parallel to the optical axis and start on the
/// `z = 0` plane. `direction` is not normalized: for a pinhole camera it is
/// the ray's point on the `z = 1` plane.
#[derive(Clone, Debug, PartialEq)]
pub struct Ray<T: RealField> {
    /// Ray origin in camera coordinates.
    pub origin: Vector3<T>,
    /// Ray direction in camera coordinates (not necessarily unit length).
    pub direction: Vector3<T>,
}

impl<T: RealField> Ray<T> {
    /// The ray's point on the plane `z = 1`, or `None` when the ray runs
    /// parallel to that plane or meets it behind its origin.
    pub fn point_at_unit_depth(&self) -> Option<Vector3<T>> {
        self.point_at_depth(T::one())
    }

    /// The ray's point on the plane `z = depth`, or `None` when the ray runs
    /// parallel to that plane or meets it behind its origin.
    pub fn point_at_depth(&self, depth: T) -> Option<Vector3<T>> {
        let dz = self.direction.z.clone();
        if dz == T::zero() {
            return None;
        }
        let s = (depth - self.origin.z.clone()) / dz;
        if s < T::zero() {
            return None;
        }
        Some(&self.origin + &self.direction * s)
    }
}

/// Projection slot: a camera-frame point to normalized coordinates.
///
/// The projection takes the point, not only its direction, so a
/// non-central (telecentric) projection fits the same contract. Generic over
/// `T: RealField` with no `Copy` bound, so the optimizer's dual-number
/// autodiff evaluates the same code.
pub trait ProjectionModel<T: RealField> {
    /// Whether the camera-frame point lies in the projection's domain
    /// (e.g. in front of a pinhole camera).
    fn in_domain(&self, p_c: &Vector3<T>) -> bool;

    /// Normalized coordinates of a camera-frame point.
    ///
    /// Total: outside [`in_domain`](Self::in_domain) the result stays finite
    /// (guarded) but has no meaning; callers that need validity check the
    /// domain first.
    fn project(&self, p_c: &Vector3<T>) -> Point2<T>;

    /// The ray of all camera-frame points with normalized coordinates `n`.
    fn unproject(&self, n: &Point2<T>) -> Ray<T>;

    /// Whether every ray passes through the camera centre.
    fn is_central(&self) -> bool;
}

/// Classic pinhole projection model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct Pinhole;

impl<T: RealField> ProjectionModel<T> for Pinhole {
    fn in_domain(&self, p_c: &Vector3<T>) -> bool {
        p_c.z > T::zero()
    }

    /// `(x / z, y / z)` with the depth clamped to at least `1e-12`.
    fn project(&self, p_c: &Vector3<T>) -> Point2<T> {
        let eps = T::from_f64(1e-12).unwrap();
        let z = if p_c.z.clone() > eps.clone() {
            p_c.z.clone()
        } else {
            eps
        };
        Point2::new(p_c.x.clone() / z.clone(), p_c.y.clone() / z)
    }

    fn unproject(&self, n: &Point2<T>) -> Ray<T> {
        Ray {
            origin: Vector3::zeros(),
            direction: Vector3::new(n.x.clone(), n.y.clone(), T::one()),
        }
    }

    fn is_central(&self) -> bool {
        true
    }
}
