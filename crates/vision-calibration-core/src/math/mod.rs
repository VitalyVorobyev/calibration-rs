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

/// JSON Schema proxy for [`Iso3`] (`nalgebra::Isometry3<f64>`).
///
/// `nalgebra` does not implement [`schemars::JsonSchema`], so `*Export` and
/// parameter types that embed [`Iso3`] annotate the field with
/// `#[cfg_attr(feature = "schemars", schemars(with = "Iso3Schema"))]`
/// (or `Vec<Iso3Schema>` / `Option<Iso3Schema>`). This proxy mirrors the exact
/// serde wire format of `Isometry3`:
/// `{ "rotation": [qx, qy, qz, qw], "translation": [tx, ty, tz] }`.
///
/// It exists only to describe that shape to `schemars`; it is never
/// constructed at runtime.
#[cfg(feature = "schemars")]
#[derive(schemars::JsonSchema)]
pub struct Iso3Schema {
    /// Unit quaternion `[qx, qy, qz, qw]` (i, j, k, w order).
    pub rotation: [Real; 4],
    /// Translation `[tx, ty, tz]` in meters.
    pub translation: [Real; 3],
}

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

#[cfg(all(test, feature = "schemars"))]
mod schema_tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion};

    /// [`Iso3Schema`] exists purely to describe [`Iso3`]'s serde wire shape
    /// to `schemars` — nothing checks at compile time that the two stay in
    /// sync. Pin the real `Iso3` serialization (a non-identity pose, so a
    /// field getting silently dropped or reordered can't hide behind
    /// zero/identity defaults) to exactly the `{rotation, translation}`
    /// layout `Iso3Schema` declares, so a future `nalgebra` upgrade that
    /// changes the wire format fails loudly here instead of only showing up
    /// as a schema/runtime mismatch downstream.
    #[test]
    fn iso3_json_shape_matches_iso3_schema() {
        let rotation = UnitQuaternion::from_euler_angles(0.3, -0.5, 1.1);
        let translation = Translation3::new(1.0, -2.0, 0.5);
        let iso = Iso3::from_parts(translation, rotation);

        let value = serde_json::to_value(iso).expect("Iso3 serializes to JSON");
        let obj = value.as_object().expect("Iso3 serializes as a JSON object");

        assert_eq!(
            obj.keys().collect::<Vec<_>>(),
            vec!["rotation", "translation"],
            "Iso3's serde field set/order must match Iso3Schema exactly",
        );

        // `rotation.coords` is the quaternion's underlying `Vector4` in
        // `[i, j, k, w]` (i.e. `[qx, qy, qz, qw]`) storage order; `.vector`
        // is the translation's underlying `Vector3`.
        let rotation_arr = obj["rotation"].as_array().expect("rotation is an array");
        let rotation_vals: Vec<f64> = rotation_arr
            .iter()
            .map(|v| v.as_f64().expect("rotation entry is a number"))
            .collect();
        assert_eq!(
            rotation_vals,
            vec![
                rotation.coords[0],
                rotation.coords[1],
                rotation.coords[2],
                rotation.coords[3],
            ],
            "rotation must serialize as [qx, qy, qz, qw]",
        );

        let translation_arr = obj["translation"]
            .as_array()
            .expect("translation is an array");
        let translation_vals: Vec<f64> = translation_arr
            .iter()
            .map(|v| v.as_f64().expect("translation entry is a number"))
            .collect();
        assert_eq!(
            translation_vals,
            vec![
                translation.vector[0],
                translation.vector[1],
                translation.vector[2],
            ],
            "translation must serialize as [tx, ty, tz]",
        );
    }
}
