use nalgebra::{RealField, Vector2, Vector3};
use serde::{Deserialize, Serialize};

/// Projection model from a camera direction to normalized coordinates.
pub trait ProjectionModel<S: RealField + Copy> {
    /// Project a direction in camera coordinates to normalized coordinates.
    ///
    /// Returns `None` when the direction is not projectable (e.g. behind camera).
    fn project_dir(&self, dir_c: &Vector3<S>) -> Option<Vector2<S>>;
    /// Unproject normalized coordinates to a direction in camera coordinates.
    fn unproject_dir(&self, n: &Vector2<S>) -> Vector3<S>;
}

/// Classic pinhole projection model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Pinhole;

impl<S: RealField + Copy> ProjectionModel<S> for Pinhole {
    fn project_dir(&self, dir_c: &Vector3<S>) -> Option<Vector2<S>> {
        if dir_c.z <= S::zero() {
            return None;
        }
        Some(Vector2::new(dir_c.x / dir_c.z, dir_c.y / dir_c.z))
    }

    fn unproject_dir(&self, n: &Vector2<S>) -> Vector3<S> {
        Vector3::new(n.x, n.y, S::one())
    }
}
