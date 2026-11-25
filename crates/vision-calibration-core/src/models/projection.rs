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

/// Equidistant (fisheye) projection: the normalized point lies in the
/// direction of the point's azimuth at a radius equal to its incidence angle,
/// `n = θ · (x, y) / ρ` with `ρ = √(x² + y²)` and `θ = atan2(ρ, z)`.
///
/// Normalized coordinates are in radians, so `fx`, `fy` are pixels per
/// radian. The domain is every direction but straight back (`θ < π`): points
/// beyond 90° from the optical axis (`z ≤ 0`) project. Combined with
/// [`KannalaBrandt4`](super::KannalaBrandt4) distortion it is the
/// Kannala-Brandt fisheye model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct Equidistant;

/// Below this radius (relative to `|p|`) the equidistant map uses the
/// first-order form `(x, y) / z`, where `θ / ρ → 1 / z`.
const EQUIDISTANT_AXIS: f64 = 1e-8;

impl<T: RealField> ProjectionModel<T> for Equidistant {
    fn in_domain(&self, p_c: &Vector3<T>) -> bool {
        let rho2 = p_c.x.clone() * p_c.x.clone() + p_c.y.clone() * p_c.y.clone();
        rho2 > T::zero() || p_c.z > T::zero()
    }

    fn project(&self, p_c: &Vector3<T>) -> Point2<T> {
        let (x, y, z) = (p_c.x.clone(), p_c.y.clone(), p_c.z.clone());
        let rho2 = x.clone() * x.clone() + y.clone() * y.clone();
        let norm2 = rho2.clone() + z.clone() * z.clone();
        let axis = T::from_f64(EQUIDISTANT_AXIS).unwrap();
        let axis2 = axis.clone() * axis * norm2;

        // Near the +z axis (θ ≈ 0), θ/ρ → 1/z * (1 - ρ²/(3z²)), avoiding sqrt(0)
        // and 0/0 atan2 singularities under dual numbers.
        if z > T::zero() && rho2 <= axis2 {
            let three = T::from_f64(3.0).unwrap();
            let z2 = z.clone() * z.clone();
            let scale = (T::one() - rho2 / (three * z2)) / z;
            return Point2::new(x * scale.clone(), y * scale);
        }
        // Straight backward (x = y = 0, z <= 0) is outside the domain; keep finite
        // without calling sqrt(0) under dual numbers.
        if rho2 <= T::zero() {
            return Point2::origin();
        }

        let rho = rho2.sqrt();
        // Split around |z| < rho so the quotient fed to atan() never divides by
        // zero (avoiding num-dual's Dual::atan2(y, 0.0) NaN derivative at z = 0).
        let theta = if z.clone().abs() < rho {
            T::from_f64(std::f64::consts::FRAC_PI_2).unwrap() - (z / rho.clone()).atan()
        } else if z > T::zero() {
            (rho.clone() / z).atan()
        } else {
            T::from_f64(std::f64::consts::PI).unwrap() - (rho.clone() / (-z)).atan()
        };
        let scale = theta / rho;
        Point2::new(x * scale.clone(), y * scale)
    }

    fn unproject(&self, n: &Point2<T>) -> Ray<T> {
        let theta2 = n.x.clone() * n.x.clone() + n.y.clone() * n.y.clone();
        let axis = T::from_f64(EQUIDISTANT_AXIS).unwrap();
        let axis2 = axis.clone() * axis;
        // Avoid sqrt(0) at (0, 0) under dual numbers via second-order Taylor series.
        let (sinc, cos_theta) = if theta2 > axis2 {
            let theta = theta2.sqrt();
            (theta.clone().sin() / theta.clone(), theta.cos())
        } else {
            let half = T::from_f64(0.5).unwrap();
            let sixth = T::from_f64(1.0 / 6.0).unwrap();
            (T::one() - theta2.clone() * sixth, T::one() - theta2 * half)
        };
        Ray {
            origin: Vector3::zeros(),
            direction: Vector3::new(n.x.clone() * sinc.clone(), n.y.clone() * sinc, cos_theta),
        }
    }

    fn is_central(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equidistant_radius_is_the_incidence_angle_beyond_ninety_degrees() {
        for theta in [
            0.0_f64,
            1e-10,
            0.3,
            1.2,
            std::f64::consts::FRAC_PI_2,
            2.0,
            3.0,
        ] {
            let phi = 2.1_f64;
            let dir = Vector3::new(
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            );
            let p = dir * 3.5;
            assert!(Equidistant.in_domain(&p), "θ = {theta}");
            let n = Equidistant.project(&p);
            assert!(
                (n.coords.norm() - theta).abs() < 1e-12,
                "θ = {theta}: {n:?}"
            );
            if theta > 0.0 {
                assert!(
                    (n.y.atan2(n.x) - phi).abs() < 1e-9,
                    "azimuth at θ = {theta}"
                );
            }
            let ray = Equidistant.unproject(&n);
            assert!(
                (ray.direction.normalize() - dir).norm() < 1e-12,
                "θ = {theta}: {:?}",
                ray.direction
            );
        }
        assert!(!Equidistant.in_domain(&Vector3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn equidistant_near_negative_z_axis_stays_on_pi_circle() {
        let p = Vector3::new(1e-10_f64, 0.0, -1.0);
        assert!(Equidistant.in_domain(&p));
        let n: Point2<f64> = Equidistant.project(&p);
        assert!((n.x - std::f64::consts::PI).abs() < 1e-9, "got {n:?}");
        assert!(n.y.abs() < 1e-15);

        let behind = Vector3::new(0.0_f64, 0.0, -1.0);
        let n_behind: Point2<f64> = Equidistant.project(&behind);
        assert!(n_behind.x.is_finite() && n_behind.y.is_finite());
        assert!(n_behind.coords.norm() < 1e-12);
    }
}
