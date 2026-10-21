//! The camera blocks at the head of a factor's parameter list, evaluated
//! through the core camera models.
//!
//! A factor's camera is data (a [`CameraModelSpec`]); its parameter blocks
//! are `[intrinsics, projection?, distortion?, sensor?]`, the optional blocks
//! present exactly when their kind has parameters. Evaluation dispatches per
//! slot through the kind's methods in `vision_calibration_core`, so this
//! module holds no camera math and a new model needs no change here.

use nalgebra::{DVector, Point2, RealField, Vector3};
use vision_calibration_core::{CameraModelSpec, Ray};

/// A factor's camera: its spec and its parameter blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CameraBlocks<'a, T: RealField> {
    spec: CameraModelSpec,
    intrinsics: &'a [T],
    projection: &'a [T],
    distortion: &'a [T],
    sensor: &'a [T],
}

impl<'a, T: RealField> CameraBlocks<'a, T> {
    /// A camera from its blocks; a slot without parameters takes `&[]`.
    pub(crate) fn new(
        spec: CameraModelSpec,
        intrinsics: &'a [T],
        projection: &'a [T],
        distortion: &'a [T],
        sensor: &'a [T],
    ) -> Self {
        debug_assert!(intrinsics.len() >= 4, "intrinsics must have 4 params");
        debug_assert_eq!(projection.len(), spec.projection.dim());
        debug_assert_eq!(distortion.len(), spec.distortion.dim());
        debug_assert_eq!(sensor.len(), spec.sensor.dim());
        Self {
            spec,
            intrinsics,
            projection,
            distortion,
            sensor,
        }
    }

    /// Split an IR-ordered block list into the camera implied by `spec` and
    /// the remaining (chain) blocks.
    pub(crate) fn split(
        spec: CameraModelSpec,
        params: &'a [DVector<T>],
    ) -> (Self, &'a [DVector<T>]) {
        let mut rest = &params[1..];
        let mut take = |dim: usize| -> &'a [T] {
            if dim == 0 {
                return &[];
            }
            let (head, tail) = rest.split_first().expect("camera block missing");
            rest = tail;
            head.as_slice()
        };
        let projection = take(spec.projection.dim());
        let distortion = take(spec.distortion.dim());
        let sensor = take(spec.sensor.dim());
        let camera = Self::new(spec, params[0].as_slice(), projection, distortion, sensor);
        (camera, rest)
    }

    /// The intrinsics block `[fx, fy, cx, cy]`.
    pub(crate) fn intrinsics(&self) -> &'a [T] {
        self.intrinsics
    }

    /// Pixel of a camera-frame point.
    pub(crate) fn project(&self, p_c: &Vector3<T>) -> (T, T) {
        let n = self.spec.projection.project(self.projection, p_c);
        let d = self.spec.distortion.distort(self.distortion, &n);
        let [x, y]: [T; 2] = self.spec.sensor.to_sensor(self.sensor, &d).coords.into();
        let k = self.intrinsics;
        (
            k[0].clone() * x + k[2].clone(),
            k[1].clone() * y + k[3].clone(),
        )
    }

    /// Undistorted normalized coordinates of a pixel.
    pub(crate) fn pixel_to_normalized(&self, pixel: [f64; 2]) -> Point2<T> {
        let k = self.intrinsics;
        let u = T::from_f64(pixel[0]).unwrap();
        let v = T::from_f64(pixel[1]).unwrap();
        let s = Point2::new(
            (u - k[2].clone()) / k[0].clone(),
            (v - k[3].clone()) / k[1].clone(),
        );
        let d = self.spec.sensor.to_normalized(self.sensor, &s);
        self.spec.distortion.undistort(self.distortion, &d)
    }

    /// The camera ray through a pixel.
    pub(crate) fn pixel_ray(&self, pixel: [f64; 2]) -> Ray<T> {
        let n = self.pixel_to_normalized(pixel);
        self.spec.projection.unproject(self.projection, &n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use num_dual::Dual64;
    use vision_calibration_core::{DistortionKind, SensorKind};

    /// At zero distortion the undistortion's real part converges before the
    /// first step, but its derivative with respect to the coefficients must
    /// still be the exact one: `∂x_u/∂k1 = −x·r²` (first order in `k`).
    #[test]
    fn undistort_derivative_at_zero_distortion() {
        let (x, y) = (0.3_f64, -0.2_f64);
        let r2 = x * x + y * y;
        let intr = [1.0, 1.0, 0.0, 0.0].map(Dual64::from);
        let mut d = DVector::from_element(5, Dual64::from(0.0));
        d[0] = Dual64::new(0.0, 1.0);
        let spec = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
        let camera = CameraBlocks::new(spec, &intr, &[], d.as_slice(), &[]);
        let n = camera.pixel_to_normalized([x, y]);
        assert!((n.x.re - x).abs() < 1e-15 && (n.y.re - y).abs() < 1e-15);
        assert!((n.x.eps + x * r2).abs() < 1e-12, "∂x_u/∂k1 = {}", n.x.eps);
        assert!((n.y.eps + y * r2).abs() < 1e-12, "∂y_u/∂k1 = {}", n.y.eps);
    }
}
