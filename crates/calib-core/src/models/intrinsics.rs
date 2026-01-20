use nalgebra::{Matrix3, Point2, RealField};
use serde::{Deserialize, Serialize};

/// Intrinsics that map sensor-plane coordinates to pixel coordinates.
pub trait IntrinsicsModel<S: RealField + Copy> {
    /// Convert sensor-plane coordinates into pixel coordinates.
    fn sensor_to_pixel(&self, sensor: &Point2<S>) -> Point2<S>;
    /// Convert pixel coordinates into sensor-plane coordinates.
    fn pixel_to_sensor(&self, pixel: &Point2<S>) -> Point2<S>;
}

/// Standard pinhole intrinsics with optional skew.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FxFyCxCySkew<S: RealField + Copy> {
    /// Focal length in pixels along X.
    pub fx: S,
    /// Focal length in pixels along Y.
    pub fy: S,
    /// Principal point X coordinate in pixels.
    pub cx: S,
    /// Principal point Y coordinate in pixels.
    pub cy: S,
    /// Skew term (typically 0).
    pub skew: S,
}

impl<S: RealField + Copy> FxFyCxCySkew<S> {
    /// Return the 3x3 camera intrinsics matrix K.
    pub fn k_matrix(&self) -> Matrix3<S> {
        Matrix3::new(
            self.fx,
            self.skew,
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

impl<S: RealField + Copy> IntrinsicsModel<S> for FxFyCxCySkew<S> {
    fn sensor_to_pixel(&self, sensor: &Point2<S>) -> Point2<S> {
        let u = self.fx * sensor.x + self.skew * sensor.y + self.cx;
        let v = self.fy * sensor.y + self.cy;
        Point2::new(u, v)
    }

    fn pixel_to_sensor(&self, pixel: &Point2<S>) -> Point2<S> {
        let sy = (pixel.y - self.cy) / self.fy;
        let sx = (pixel.x - self.cx - self.skew * sy) / self.fx;
        Point2::new(sx, sy)
    }
}
