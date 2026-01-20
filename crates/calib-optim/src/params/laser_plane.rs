//! Laser plane parameters for optimization.
//!
//! A laser plane is parameterized as (n̂, d) where:
//! - n̂ is a unit normal vector (3D)
//! - d is the signed distance from the camera origin
//!
//! In optimization, the normal is optimized on the S2 manifold (unit sphere),
//! while the distance is a separate Euclidean scalar.

use anyhow::{ensure, Result};
use calib_core::Pt3;
use nalgebra::{DVector, DVectorView, Unit, Vector3};
use serde::{Deserialize, Serialize};

/// Laser plane in camera frame: unit normal + signed distance.
///
/// The plane equation is: n̂ · p + d = 0
/// where p is a point in camera coordinates.
///
/// # Example
///
/// ```ignore
/// // Internal module: not part of the stable public API.
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserPlane {
    /// Unit normal vector in camera frame
    pub normal: Unit<Vector3<f64>>,
    /// Signed distance from camera origin
    pub distance: f64,
}

impl LaserPlane {
    /// Create a new laser plane from normal and distance.
    ///
    /// The normal will be normalized to unit length.
    pub fn new(normal: Vector3<f64>, distance: f64) -> Self {
        Self {
            normal: Unit::new_normalize(normal),
            distance,
        }
    }

    /// Convert to 4D parameter vector [nx, ny, nz, d].
    ///
    /// This is a compact serialization form. For S2 optimization, prefer
    /// [`normal_to_dvec`](Self::normal_to_dvec) and
    /// [`distance_to_dvec`](Self::distance_to_dvec).
    pub fn to_dvec(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.normal.x,
            self.normal.y,
            self.normal.z,
            self.distance,
        ])
    }

    /// Convert the unit normal to a 3D parameter vector [nx, ny, nz].
    pub fn normal_to_dvec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.normal.x, self.normal.y, self.normal.z])
    }

    /// Convert the distance to a 1D parameter vector \[d\].
    pub fn distance_to_dvec(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.distance])
    }

    /// Parse from 4D vector [nx, ny, nz, d].
    ///
    /// The first 3 components are normalized to create a unit normal.
    pub fn from_dvec(v: DVectorView<f64>) -> Result<Self> {
        ensure!(
            v.len() == 4,
            "LaserPlane requires 4D vector, got {}",
            v.len()
        );
        let normal = Unit::new_normalize(Vector3::new(v[0], v[1], v[2]));
        Ok(Self {
            normal,
            distance: v[3],
        })
    }

    /// Parse from split normal + distance vectors.
    pub fn from_split_dvec(normal: DVectorView<f64>, distance: DVectorView<f64>) -> Result<Self> {
        ensure!(
            normal.len() == 3,
            "LaserPlane normal requires 3D vector, got {}",
            normal.len()
        );
        ensure!(
            distance.len() == 1,
            "LaserPlane distance requires 1D vector, got {}",
            distance.len()
        );
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_from_normal_distance() {
        let normal = Vector3::new(0.0, 0.0, 2.0); // Will be normalized to [0,0,1]
        let plane = LaserPlane::new(normal, -0.5);

        assert!((plane.normal.z - 1.0).abs() < 1e-10);
        assert!((plane.normal.x).abs() < 1e-10);
        assert!((plane.normal.y).abs() < 1e-10);
        assert!((plane.distance + 0.5).abs() < 1e-10);
    }

    #[test]
    fn plane_roundtrip_conversion() {
        let plane = LaserPlane::new(Vector3::new(1.0, 0.0, 1.0), -0.3);

        let v = plane.to_dvec();
        let plane2 = LaserPlane::from_dvec(v.as_view()).unwrap();

        assert!((plane.normal.x - plane2.normal.x).abs() < 1e-10);
        assert!((plane.normal.y - plane2.normal.y).abs() < 1e-10);
        assert!((plane.normal.z - plane2.normal.z).abs() < 1e-10);
        assert!((plane.distance - plane2.distance).abs() < 1e-10);
    }

    #[test]
    fn plane_point_distance() {
        // Plane: z = 0.5 (parallel to XY plane)
        // Normal: [0, 0, 1], distance: -0.5
        let plane = LaserPlane::new(Vector3::new(0.0, 0.0, 1.0), -0.5);

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
    fn plane_from_dvec_wrong_size() {
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]); // Only 3D
        assert!(LaserPlane::from_dvec(v.as_view()).is_err());
    }

    #[test]
    fn plane_split_roundtrip() {
        let plane = LaserPlane::new(Vector3::new(1.0, 2.0, 3.0), -0.4);

        let normal = plane.normal_to_dvec();
        let distance = plane.distance_to_dvec();
        let plane2 = LaserPlane::from_split_dvec(normal.as_view(), distance.as_view()).unwrap();

        assert!((plane.normal.x - plane2.normal.x).abs() < 1e-10);
        assert!((plane.normal.y - plane2.normal.y).abs() < 1e-10);
        assert!((plane.normal.z - plane2.normal.z).abs() < 1e-10);
        assert!((plane.distance - plane2.distance).abs() < 1e-10);
    }
}
