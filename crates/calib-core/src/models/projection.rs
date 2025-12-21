use nalgebra::{RealField, Vector2, Vector3};
use serde::{Deserialize, Serialize};

pub trait ProjectionModel<S: RealField + Copy> {
    fn project_dir(&self, dir_c: &Vector3<S>) -> Option<Vector2<S>>;
    fn unproject_dir(&self, n: &Vector2<S>) -> Vector3<S>;
}

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
