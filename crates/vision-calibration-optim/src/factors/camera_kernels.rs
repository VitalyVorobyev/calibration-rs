//! Zero-sized camera-model kernels for descriptor-based factors.
//!
//! Each kernel type implements one slot of a
//! [`CameraModelDesc`](crate::ir::CameraModelDesc) as static generic methods
//! (no `dyn`, no `self`): the backend matches the descriptor once per factor
//! and monomorphizes the residual over the selected kernel types, so the
//! autodiff path stays generic over `T: RealField` with zero per-evaluation
//! dispatch on the camera-model axis.
//!
//! A kernel holds no math: it reads its parameter block into the core model
//! of its kind (`vision_calibration_core`) and calls that model, so the
//! optimizer and every other consumer evaluate one implementation.

use nalgebra::{DVectorView, Point2, RealField, Vector3};
use vision_calibration_core::{
    BrownConrady5, DistortionModel, Division, Pinhole, ProjectionModel, RationalPolynomial,
    ScheimpflugParams, SensorModel, ThinPrism,
};

/// Split a point into its coordinates without cloning them.
fn coords<T: RealField>(p: Point2<T>) -> (T, T) {
    let [x, y]: [T; 2] = p.coords.into();
    (x, y)
}

/// Projection slot: normalizes a camera-frame point.
pub(crate) trait ProjectionKernel: 'static {
    /// Normalized coordinates of a camera-frame point.
    fn normalize<T: RealField>(p_c: &Vector3<T>) -> (T, T);
}

/// Pinhole projection (core [`Pinhole`], depth clamped at `1e-12`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PinholeKernel;

impl ProjectionKernel for PinholeKernel {
    fn normalize<T: RealField>(p_c: &Vector3<T>) -> (T, T) {
        coords(Pinhole.project(p_c))
    }
}

/// Distortion slot: maps normalized coordinates to distorted normalized
/// coordinates (and back, for the laser back-projection path).
///
/// `params` is `None` exactly when `DIM == 0`.
pub(crate) trait DistortionKernel: 'static {
    /// Dimension of the distortion parameter block (0 = no block).
    const DIM: usize;

    /// Apply distortion to normalized coordinates.
    fn distort<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T);

    /// Invert distortion (distorted normalized -> undistorted normalized).
    fn undistort<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T);
}

/// Distortion kernel backed by a core model built from the parameter block.
macro_rules! distortion_kernel {
    ($(#[$doc:meta])* $kernel:ident, $dim:expr, |$d:ident| $model:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy)]
        pub(crate) struct $kernel;

        impl DistortionKernel for $kernel {
            const DIM: usize = $dim;

            fn distort<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
                let $d = params.expect(concat!(stringify!($kernel), " requires a distortion block"));
                debug_assert!($d.len() >= $dim, "distortion block too short");
                coords($model.distort(&Point2::new(x, y)))
            }

            fn undistort<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
                let $d = params.expect(concat!(stringify!($kernel), " requires a distortion block"));
                debug_assert!($d.len() >= $dim, "distortion block too short");
                coords($model.undistort(&Point2::new(x, y)))
            }
        }
    };
}

/// Identity distortion (no parameter block).
#[derive(Debug, Clone, Copy)]
pub(crate) struct NoDistortionKernel;

impl DistortionKernel for NoDistortionKernel {
    const DIM: usize = 0;

    fn distort<T: RealField>(_params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        (x, y)
    }

    fn undistort<T: RealField>(_params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        (x, y)
    }
}

distortion_kernel!(
    /// Brown-Conrady distortion `[k1, k2, k3, p1, p2]`.
    BrownConrady5Kernel,
    5,
    |d| BrownConrady5 {
        k1: d[0].clone(),
        k2: d[1].clone(),
        k3: d[2].clone(),
        p1: d[3].clone(),
        p2: d[4].clone(),
        iters: 0,
    }
);

distortion_kernel!(
    /// Rational polynomial distortion `[k1, k2, k3, k4, k5, k6, p1, p2]`.
    RationalKernel,
    8,
    |d| RationalPolynomial {
        k1: d[0].clone(),
        k2: d[1].clone(),
        k3: d[2].clone(),
        k4: d[3].clone(),
        k5: d[4].clone(),
        k6: d[5].clone(),
        p1: d[6].clone(),
        p2: d[7].clone(),
        iters: 0,
    }
);

distortion_kernel!(
    /// Thin-prism distortion `[k1, k2, k3, p1, p2, s1, s2, s3, s4]`.
    ThinPrismKernel,
    9,
    |d| ThinPrism {
        k1: d[0].clone(),
        k2: d[1].clone(),
        k3: d[2].clone(),
        p1: d[3].clone(),
        p2: d[4].clone(),
        s1: d[5].clone(),
        s2: d[6].clone(),
        s3: d[7].clone(),
        s4: d[8].clone(),
        iters: 0,
    }
);

distortion_kernel!(
    /// Division (Fitzgibbon) distortion `[lambda]`, closed form both ways.
    DivisionKernel,
    1,
    |d| Division {
        lambda: d[0].clone(),
    }
);

/// Sensor slot: maps distorted normalized coordinates to the sensor plane
/// (and back, for the laser back-projection path).
///
/// `params` is `None` exactly when `DIM == 0`.
pub(crate) trait SensorKernel: 'static {
    /// Dimension of the sensor parameter block (0 = no block).
    const DIM: usize;

    /// Map distorted normalized coordinates onto the sensor plane.
    fn to_sensor<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T);

    /// Map sensor-plane coordinates back to distorted normalized coordinates.
    fn to_normalized<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T);
}

/// Identity sensor (no parameter block).
#[derive(Debug, Clone, Copy)]
pub(crate) struct IdentitySensorKernel;

impl SensorKernel for IdentitySensorKernel {
    const DIM: usize = 0;

    fn to_sensor<T: RealField>(_params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        (x, y)
    }

    fn to_normalized<T: RealField>(_params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        (x, y)
    }
}

/// Scheimpflug tilted-sensor homography `[tau_x, tau_y]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Scheimpflug2Kernel;

impl Scheimpflug2Kernel {
    fn model<T: RealField>(params: Option<DVectorView<'_, T>>) -> ScheimpflugParams<T> {
        let s = params.expect("Scheimpflug2 requires a sensor block");
        debug_assert!(s.len() >= 2, "sensor must have 2 params");
        ScheimpflugParams {
            tilt_x: s[0].clone(),
            tilt_y: s[1].clone(),
        }
    }
}

impl SensorKernel for Scheimpflug2Kernel {
    const DIM: usize = 2;

    fn to_sensor<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        coords(Self::model(params).normalized_to_sensor(&Point2::new(x, y)))
    }

    fn to_normalized<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        coords(Self::model(params).sensor_to_normalized(&Point2::new(x, y)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use num_dual::Dual64;

    /// At zero distortion the undistortion's real part converges before the
    /// first step, but its derivative with respect to the coefficients must
    /// still be the exact one: `∂x_u/∂k1 = −x·r²` (first order in `k`).
    #[test]
    fn undistort_derivative_at_zero_distortion() {
        let (x, y) = (0.3_f64, -0.2_f64);
        let r2 = x * x + y * y;
        let mut d = DVector::from_element(5, Dual64::from(0.0));
        d[0] = Dual64::new(0.0, 1.0);
        let (xu, yu) =
            BrownConrady5Kernel::undistort(Some(d.as_view()), Dual64::from(x), Dual64::from(y));
        assert!((xu.re - x).abs() < 1e-15 && (yu.re - y).abs() < 1e-15);
        let (dx, dy) = (xu.eps, yu.eps);
        assert!((dx + x * r2).abs() < 1e-12, "∂x_u/∂k1 = {dx}");
        assert!((dy + y * r2).abs() < 1e-12, "∂y_u/∂k1 = {dy}");
    }
}
