//! Mathematical utilities and type definitions.
//!
//! This module provides fundamental types used throughout the library
//! and utility functions for coordinate transformations.

use nalgebra::{Isometry3, Matrix3, Matrix4, Point2, Point3, Vector2, Vector3};

pub mod coordinate_utils;

// Re-export coordinate utilities for convenience
pub use coordinate_utils::{
    distort_to_pixel, normalized_to_pixel, pixel_to_normalized, undistort_pixel,
};

/// Scalar type used throughout the library (currently `f64`).
pub type Real = f64;

/// 2D vector with [`Real`] components.
pub type Vec2 = Vector2<Real>;
/// 3D vector with [`Real`] components.
pub type Vec3 = Vector3<Real>;
/// 2D point with [`Real`] coordinates.
pub type Pt2 = Point2<Real>;
/// 3D point with [`Real`] coordinates.
pub type Pt3 = Point3<Real>;
/// 3×3 matrix with [`Real`] entries.
pub type Mat3 = Matrix3<Real>;
/// 4×4 matrix with [`Real`] entries.
pub type Mat4 = Matrix4<Real>;
/// 3D rigid transform (SE(3)) using [`Real`].
pub type Iso3 = Isometry3<Real>;

/// Convert a 2D point in Euclidean coordinates into homogeneous coordinates.
///
/// Given a point `p = (x, y)`, returns the homogeneous vector `(x, y, 1)`.
pub fn to_homogeneous(p: &Pt2) -> Vec3 {
    Vec3::new(p.x, p.y, 1.0)
}

/// Convert a 3D homogeneous vector back to a 2D point.
///
/// The input is interpreted as `(x, y, w)` and the result is `(x / w, y / w)`.
/// The caller is responsible for ensuring that `w != 0`.
pub fn from_homogeneous(v: &Vec3) -> Pt2 {
    Pt2::new(v.x / v.z, v.y / v.z)
}
