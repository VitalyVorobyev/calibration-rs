//! Laser plane parameters for optimization.
//!
//! A laser plane is parameterized as (n̂, d) where:
//! - n̂ is a unit normal vector (3D)
//! - d is the signed distance from the camera origin
//!
//! In optimization, the normal is optimized on the S2 manifold (unit sphere),
//! while the distance is a separate Euclidean scalar.

use crate::Error;
use nalgebra::{DVector, DVectorView, Unit, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{Iso3, Pt3};

/// Laser plane in camera frame: unit normal + signed distance.
///
/// The plane equation is: n̂ · p + d = 0
/// where p is a point in camera coordinates.
///
/// A plane is valid by construction: its normal has unit length (within
/// `1e-9`) and every value is finite, and a plane read from JSON is checked
/// the same way.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(try_from = "LaserPlaneRaw")]
#[non_exhaustive]
pub struct LaserPlane {
    /// Unit normal vector in camera frame. Serializes as `[nx, ny, nz]`.
    #[cfg_attr(feature = "schemars", schemars(with = "[f64; 3]"))]
    normal: Unit<Vector3<f64>>,
    /// Signed distance from camera origin
    distance: f64,
}

/// The wire form of [`LaserPlane`], validated on conversion; it also
/// describes its JSON Schema.
#[derive(Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars", schemars(rename = "LaserPlane"))]
struct LaserPlaneRaw {
    /// Unit normal vector in camera frame. Serializes as `[nx, ny, nz]`.
    #[cfg_attr(feature = "schemars", schemars(with = "[f64; 3]"))]
    normal: Vector3<f64>,
    /// Signed distance from camera origin
    distance: f64,
}

impl TryFrom<LaserPlaneRaw> for LaserPlane {
    type Error = Error;

    fn try_from(raw: LaserPlaneRaw) -> Result<Self, Self::Error> {
        Self::new(raw.normal, raw.distance)
    }
}

/// How far from 1 a unit normal's length may be.
const UNIT_NORMAL_TOLERANCE: f64 = 1e-9;

impl LaserPlane {
    /// The plane `n̂ · p + d = 0` with unit normal `normal` and signed
    /// distance `distance` from the camera origin.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] unless every value is finite and the normal
    /// has unit length (within `1e-9`). `distance` is a distance only for a
    /// unit normal, so a normal of another length is rejected rather than
    /// rescaled: rescaling the normal alone would move the plane. For a plane
    /// equation `n · p + d = 0` with any nonzero `n`, use
    /// [`LaserPlane::from_coefficients`].
    pub fn new(normal: Vector3<f64>, distance: f64) -> Result<Self, Error> {
        if !(distance.is_finite() && normal.iter().all(|c| c.is_finite())) {
            return Err(Error::invalid_input(format!(
                "a laser plane must be finite, got normal [{}, {}, {}] and distance {distance}",
                normal.x, normal.y, normal.z
            )));
        }
        let length = normal.norm();
        if (length - 1.0).abs() > UNIT_NORMAL_TOLERANCE {
            return Err(Error::invalid_input(format!(
                "a laser plane's normal must have unit length, got {length}; divide the \
                 normal and the distance by its length (LaserPlane::from_coefficients)"
            )));
        }
        Ok(Self {
            normal: Unit::new_normalize(normal),
            distance,
        })
    }

    /// The plane `n · p + d = 0` for any nonzero, finite `n`: both are
    /// divided by `|n|`, which leaves the plane where it is.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a value that is not finite or a zero
    /// normal.
    pub fn from_coefficients(n: Vector3<f64>, d: f64) -> Result<Self, Error> {
        let length = n.norm();
        if !(length.is_finite() && length > 0.0 && d.is_finite()) {
            return Err(Error::invalid_input(format!(
                "a plane equation needs a finite, nonzero normal and a finite offset, got \
                 [{}, {}, {}] and {d}",
                n.x, n.y, n.z
            )));
        }
        Self::new(n / length, d / length)
    }

    /// The unit normal, in the camera frame.
    pub fn normal(&self) -> Unit<Vector3<f64>> {
        self.normal
    }

    /// The signed distance from the camera origin, metres: `n̂ · p + d = 0`.
    pub fn distance(&self) -> f64 {
        self.distance
    }

    /// Convert the unit normal to a 3D parameter vector [nx, ny, nz].
    pub(crate) fn normal_to_dvec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.normal.x, self.normal.y, self.normal.z])
    }

    /// Convert the distance to a 1D parameter vector \[d\].
    pub(crate) fn distance_to_dvec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.distance])
    }

    /// Parse from split normal + distance vectors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `normal` is not length 3 or
    /// `distance` is not length 1.
    pub(crate) fn from_split_dvec(
        normal: DVectorView<f64>,
        distance: DVectorView<f64>,
    ) -> Result<Self, Error> {
        if normal.len() != 3 {
            return Err(Error::invalid_input(format!(
                "LaserPlane normal requires 3D vector, got {}",
                normal.len()
            )));
        }
        if distance.len() != 1 {
            return Err(Error::invalid_input(format!(
                "LaserPlane distance requires 1D vector, got {}",
                distance.len()
            )));
        }
        let normal = Unit::new_normalize(Vector3::new(normal[0], normal[1], normal[2]));
        Ok(Self {
            normal,
            distance: distance[0],
        })
    }

    /// Compute signed distance from a point to the plane.
    ///
    /// Returns positive if the point is on the side the normal points to,
    /// negative otherwise.
    pub fn point_distance(&self, point: &Pt3) -> f64 {
        self.normal.dot(&point.coords) + self.distance
    }

    /// Transform this plane from frame A to frame B.
    ///
    /// Given `T_B_A` (a pose that maps points from frame A to frame B, i.e.
    /// `p_B = T_B_A * p_A`), compute the plane equation in frame B:
    /// `n_B · p_B + d_B = 0`.
    ///
    /// Derivation: `p_A = R^T (p_B - t)` where `(R, t) = T_B_A`, so
    /// `n_A · p_A + d_A = (R n_A) · p_B + (d_A - (R n_A) · t) = 0`.
    pub fn transform_by(&self, t_b_a: &Iso3) -> Self {
        let r = t_b_a.rotation.to_rotation_matrix();
        let t = t_b_a.translation.vector;
        let n_b = r * self.normal.into_inner();
        let d_b = self.distance - n_b.dot(&t);
        Self {
            normal: Unit::new_normalize(n_b),
            distance: d_b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plane_needs_a_unit_normal_and_finite_values() {
        let plane = LaserPlane::new(Vector3::new(0.0, 0.6, 0.8), -0.5).unwrap();
        assert!((plane.normal().norm() - 1.0).abs() < 1e-15);
        assert_eq!(plane.distance(), -0.5);
        assert!(LaserPlane::new(Vector3::new(0.0, 0.0, 2.0), -0.5).is_err());
        assert!(LaserPlane::new(Vector3::new(0.0, 0.0, 1.0), f64::NAN).is_err());
        assert!(LaserPlane::new(Vector3::new(f64::INFINITY, 0.0, 1.0), 0.0).is_err());
    }

    #[test]
    fn coefficients_are_divided_by_the_normal_length() {
        // 2z − 1 = 0 is the plane z = 0.5: both parts are halved.
        let plane = LaserPlane::from_coefficients(Vector3::new(0.0, 0.0, 2.0), -1.0).unwrap();
        assert!((plane.normal().z - 1.0).abs() < 1e-15);
        assert!((plane.distance() + 0.5).abs() < 1e-15);
        assert!(plane.point_distance(&Pt3::new(3.0, -1.0, 0.5)).abs() < 1e-15);
        assert!(LaserPlane::from_coefficients(Vector3::zeros(), 1.0).is_err());
    }

    #[test]
    fn deserialization_validates_and_round_trips() {
        let plane = LaserPlane::new(Vector3::new(0.0, 0.6, 0.8), -0.25).unwrap();
        let json = serde_json::to_string(&plane).unwrap();
        let back: LaserPlane = serde_json::from_str(&json).unwrap();
        assert_eq!(back.normal(), plane.normal());
        assert_eq!(back.distance(), plane.distance());
        // A normal of another length, as hand-written JSON can carry: rejected,
        // not silently renormalized (which would move the plane).
        let err = serde_json::from_str::<LaserPlane>(r#"{"normal":[0.0,0.0,2.0],"distance":-0.5}"#)
            .unwrap_err();
        assert!(err.to_string().contains("unit length"), "{err}");
        assert!(
            serde_json::from_str::<LaserPlane>(r#"{"normal":[0.0,0.0,1.0],"distance":null}"#)
                .is_err()
        );
    }

    #[test]
    fn plane_point_distance() {
        // Plane: z = 0.5 (parallel to XY plane)
        // Normal: [0, 0, 1], distance: -0.5
        let plane = LaserPlane::new((Vector3::new(0.0, 0.0, 1.0)).normalize(), -0.5).unwrap();

        // Point on the plane
        let p1 = Pt3::new(1.0, 2.0, 0.5);
        assert!(plane.point_distance(&p1).abs() < 1e-10);

        // Point above the plane
        let p2 = Pt3::new(1.0, 2.0, 0.7);
        assert!((plane.point_distance(&p2) - 0.2).abs() < 1e-10);

        // Point below the plane
        let p3 = Pt3::new(1.0, 2.0, 0.3);
        assert!((plane.point_distance(&p3) + 0.2).abs() < 1e-10);
    }

    #[test]
    fn plane_transform_identity_is_noop() {
        let plane = LaserPlane::new((Vector3::new(0.1, -0.2, 0.9)).normalize(), -0.4).unwrap();
        let t = Iso3::identity();
        let out = plane.transform_by(&t);
        assert!((plane.normal.x - out.normal.x).abs() < 1e-12);
        assert!((plane.normal.y - out.normal.y).abs() < 1e-12);
        assert!((plane.normal.z - out.normal.z).abs() < 1e-12);
        assert!((plane.distance - out.distance).abs() < 1e-12);
    }

    #[test]
    fn plane_transform_preserves_geometry() {
        use nalgebra::{Rotation3, Translation3};
        let plane = LaserPlane::new((Vector3::new(0.2, 0.3, 0.95)).normalize(), -0.5).unwrap();
        let t_b_a = Iso3::from_parts(
            Translation3::new(0.1, -0.05, 0.2),
            Rotation3::from_euler_angles(0.2, -0.1, 0.15).into(),
        );
        let plane_b = plane.transform_by(&t_b_a);

        // Pick a point on the original plane and verify its image is on the new plane.
        let n_a = plane.normal.into_inner();
        let p_a_on_plane = Pt3::from(-plane.distance * n_a);
        let p_b = t_b_a.transform_point(&p_a_on_plane);
        assert!(
            plane_b.point_distance(&p_b).abs() < 1e-10,
            "transformed point not on transformed plane: {}",
            plane_b.point_distance(&p_b)
        );
    }

    #[test]
    fn plane_split_roundtrip() {
        let plane = LaserPlane::new((Vector3::new(1.0, 2.0, 3.0)).normalize(), -0.4).unwrap();

        let normal = plane.normal_to_dvec();
        let distance = plane.distance_to_dvec();
        let plane2 = LaserPlane::from_split_dvec(normal.as_view(), distance.as_view()).unwrap();

        assert!((plane.normal.x - plane2.normal.x).abs() < 1e-10);
        assert!((plane.normal.y - plane2.normal.y).abs() < 1e-10);
        assert!((plane.normal.z - plane2.normal.z).abs() < 1e-10);
        assert!((plane.distance - plane2.distance).abs() < 1e-10);
    }
}
