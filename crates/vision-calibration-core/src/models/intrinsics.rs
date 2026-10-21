use nalgebra::{Matrix3, Point2, RealField};
use serde::{Deserialize, Serialize};

/// Intrinsics that map sensor-plane coordinates to pixel coordinates.
pub trait IntrinsicsModel<S: RealField + Copy> {
    /// Convert sensor-plane coordinates into pixel coordinates.
    fn sensor_to_pixel(&self, sensor: &Point2<S>) -> Point2<S>;
    /// Convert pixel coordinates into sensor-plane coordinates.
    fn pixel_to_sensor(&self, pixel: &Point2<S>) -> Point2<S>;
}

/// Standard pinhole intrinsics: focal lengths and principal point in pixels.
///
/// The pixel grid is assumed orthogonal, so `K` carries no skew term.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct FxFyCxCy<S: RealField + Copy> {
    /// Focal length in pixels along X.
    pub fx: S,
    /// Focal length in pixels along Y.
    pub fy: S,
    /// Principal point X coordinate in pixels.
    pub cx: S,
    /// Principal point Y coordinate in pixels.
    pub cy: S,
}

impl<S: RealField + Copy> FxFyCxCy<S> {
    /// Return the 3x3 camera intrinsics matrix K.
    pub fn k_matrix(&self) -> Matrix3<S> {
        Matrix3::new(
            self.fx,
            S::zero(),
            self.cx,
            S::zero(),
            self.fy,
            self.cy,
            S::zero(),
            S::zero(),
            S::one(),
        )
    }
}

impl<S: RealField + Copy> IntrinsicsModel<S> for FxFyCxCy<S> {
    fn sensor_to_pixel(&self, sensor: &Point2<S>) -> Point2<S> {
        let u = self.fx * sensor.x + self.cx;
        let v = self.fy * sensor.y + self.cy;
        Point2::new(u, v)
    }

    fn pixel_to_sensor(&self, pixel: &Point2<S>) -> Point2<S> {
        Point2::new((pixel.x - self.cx) / self.fx, (pixel.y - self.cy) / self.fy)
    }
}
