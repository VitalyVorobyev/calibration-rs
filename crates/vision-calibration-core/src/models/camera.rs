use nalgebra::{Point2, Point3, RealField, Vector3};
use serde::{Deserialize, Serialize};

use super::{DistortionModel, IntrinsicsModel, ProjectionModel, Ray, SensorModel};

/// Tiny abstraction over "anything that can project a 3D point in camera
/// frame to a 2D pixel".
///
/// Implemented by [`Camera<f64, _, _, _, _>`] (and therefore by
/// [`crate::PinholeCamera`] and the Scheimpflug variants used elsewhere in
/// the workspace), this trait lets per-feature residual helpers in
/// `vision_calibration_core` operate on any pinhole-style camera regardless
/// of distortion or sensor configuration.
pub trait CameraProject {
    /// Project a 3D point in camera coordinates into pixel coordinates.
    /// Returns `None` if the point is behind the camera or projection fails.
    fn project_camera_point(&self, p_c: &Vector3<f64>) -> Option<Point2<f64>>;
}

impl<P, D, Sm, K> CameraProject for Camera<f64, P, D, Sm, K>
where
    P: ProjectionModel<f64>,
    D: DistortionModel<f64>,
    Sm: SensorModel<f64>,
    K: IntrinsicsModel<f64>,
{
    fn project_camera_point(&self, p_c: &Vector3<f64>) -> Option<Point2<f64>> {
        self.project_point_c(p_c)
    }
}

/// A composable camera model: projection -> distortion -> sensor -> intrinsics.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
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
    /// Returns `None` if the point is outside the projection's domain (for a
    /// pinhole camera: not in front of it).
    pub fn project_point_c(&self, p_c: &Vector3<S>) -> Option<Point2<S>> {
        if !self.proj.in_domain(p_c) {
            return None;
        }
        let n_u = self.proj.project(p_c);
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

    /// The camera ray through a pixel.
    pub fn backproject_pixel(&self, px: &Point2<S>) -> Ray<S> {
        let s = self.k.pixel_to_sensor(px);
        let n_d = self.sensor.sensor_to_normalized(&s);
        let n_u = self.dist.undistort(&n_d);
        self.proj.unproject(&n_u)
    }
}
