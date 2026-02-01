use nalgebra::{Point2, Point3, RealField, Vector3};
use serde::{Deserialize, Serialize};

use super::{DistortionModel, IntrinsicsModel, ProjectionModel, SensorModel};

/// A camera ray represented by its intersection with the z = 1 plane.
#[derive(Clone, Copy, Debug)]
pub struct Ray<S: RealField + Copy> {
    /// Point on the z = 1 plane in camera coordinates.
    pub point: Vector3<S>,
}

/// A composable camera model: projection -> distortion -> sensor -> intrinsics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera<S, P, D, Sm, K>
where
    S: RealField + Copy,
    P: ProjectionModel<S>,
    D: DistortionModel<S>,
    Sm: SensorModel<S>,
    K: IntrinsicsModel<S>,
{
    /// Projection model (e.g. pinhole).
    pub proj: P,
    /// Distortion model (e.g. Brown-Conrady).
    pub dist: D,
    /// Sensor model (e.g. identity or tilted homography).
    pub sensor: Sm,
    /// Intrinsics model (K).
    pub k: K,
    _phantom: core::marker::PhantomData<S>,
}

impl<S, P, D, Sm, K> Camera<S, P, D, Sm, K>
where
    S: RealField + Copy,
    P: ProjectionModel<S>,
    D: DistortionModel<S>,
    Sm: SensorModel<S>,
    K: IntrinsicsModel<S>,
{
    /// Build a camera from its component models.
    pub fn new(proj: P, dist: D, sensor: Sm, k: K) -> Self {
        Self {
            proj,
            dist,
            sensor,
            k,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Project a 3D point in camera coordinates into pixel coordinates.
    ///
    /// Returns `None` if the point is behind the camera or not projectable.
    pub fn project_point_c(&self, p_c: &Vector3<S>) -> Option<Point2<S>> {
        if p_c.z <= S::zero() {
            return None;
        }
        let dir = *p_c;
        let n_u = self.proj.project_dir(&dir)?;
        let n_d = self.dist.distort(&n_u);
        let s = self.sensor.normalized_to_sensor(&n_d);
        Some(self.k.sensor_to_pixel(&s))
    }

    /// Project a 3D point in camera coordinates into pixel coordinates.
    ///
    /// This is a convenience wrapper around `project_point_c`.
    pub fn project_point(&self, p_c: &Point3<S>) -> Option<Point2<S>> {
        self.project_point_c(&p_c.coords)
    }

    /// Backproject a pixel to a point on the z = 1 plane in camera coordinates.
    pub fn backproject_pixel(&self, px: &Point2<S>) -> Ray<S> {
        let s = self.k.pixel_to_sensor(px);
        let n_d = self.sensor.sensor_to_normalized(&s);
        let n_u = self.dist.undistort(&n_d);
        let dir = self.proj.unproject_dir(&n_u);
        debug_assert!(dir.z != S::zero());
        let point = dir / dir.z;
        Ray { point }
    }
}
