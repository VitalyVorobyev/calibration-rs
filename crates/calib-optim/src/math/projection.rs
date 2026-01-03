//! Minimal projection helpers shared by factors.

use nalgebra::{RealField, Vector2, Vector3};

/// Default epsilon added to depth for numerical stability.
pub const PROJECTION_EPS: f64 = 1.0e-9;

/// Project a 3D point in camera coordinates using a pinhole model.
pub fn project_pinhole<T: RealField>(fx: T, fy: T, cx: T, cy: T, pc: Vector3<T>) -> Vector2<T> {
    let eps = T::from_f64(PROJECTION_EPS).unwrap();
    let z = pc.z.clone() + eps;
    let x = pc.x.clone() / z.clone();
    let y = pc.y.clone() / z;
    Vector2::new(fx * x + cx, fy * y + cy)
}
