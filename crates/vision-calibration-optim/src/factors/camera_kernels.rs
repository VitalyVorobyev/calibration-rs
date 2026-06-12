//! Zero-sized camera-model kernels for descriptor-based factors.
//!
//! Each kernel type implements one slot of a
//! [`CameraModelDesc`](crate::ir::CameraModelDesc) as static generic methods
//! (no `dyn`, no `self`): the backend matches the descriptor once per factor
//! and monomorphizes the residual over the selected kernel types, so the
//! autodiff path stays generic over `T: RealField` with zero per-evaluation
//! dispatch on the camera-model axis.
//!
//! Adding a camera model = one kernel type here + one descriptor enum variant
//! + one row in the backend dispatch table.

use nalgebra::{DVectorView, RealField, Vector3};

use super::reprojection_model::{
    apply_scheimpflug_generic, apply_scheimpflug_inverse_generic, distort_brown_conrady_generic,
};

/// Projection slot: normalizes a camera-frame point onto the z=1 plane.
pub(crate) trait ProjectionKernel: 'static {
    /// Normalize a camera-frame point to `(x, y)` on the z=1 plane.
    fn normalize<T: RealField>(p_c: &Vector3<T>) -> (T, T);
}

/// Pinhole projection with a `max(z, 1e-12)` depth guard.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PinholeKernel;

impl ProjectionKernel for PinholeKernel {
    fn normalize<T: RealField>(p_c: &Vector3<T>) -> (T, T) {
        let eps = T::from_f64(1e-12).unwrap();
        let z_safe = if p_c.z.clone() > eps.clone() {
            p_c.z.clone()
        } else {
            eps
        };
        (p_c.x.clone() / z_safe.clone(), p_c.y.clone() / z_safe)
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

/// Brown-Conrady distortion `[k1, k2, k3, p1, p2]`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BrownConrady5Kernel;

impl DistortionKernel for BrownConrady5Kernel {
    const DIM: usize = 5;

    fn distort<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        let d = params.expect("BrownConrady5 requires a distortion block");
        debug_assert!(d.len() >= 5, "distortion must have 5 params");
        distort_brown_conrady_generic(
            x,
            y,
            d[0].clone(),
            d[1].clone(),
            d[2].clone(),
            d[3].clone(),
            d[4].clone(),
        )
    }

    /// Fixed-point inversion (5 iterations), matching the forward model well
    /// inside the calibrated field of view.
    fn undistort<T: RealField>(params: Option<DVectorView<'_, T>>, x_d: T, y_d: T) -> (T, T) {
        let d = params.expect("BrownConrady5 requires a distortion block");
        debug_assert!(d.len() >= 5, "distortion must have 5 params");
        let k1 = d[0].clone();
        let k2 = d[1].clone();
        let k3 = d[2].clone();
        let p1 = d[3].clone();
        let p2 = d[4].clone();

        let mut x_u = x_d.clone();
        let mut y_u = y_d.clone();
        for _ in 0..5 {
            let r2 = x_u.clone() * x_u.clone() + y_u.clone() * y_u.clone();
            let r4 = r2.clone() * r2.clone();
            let r6 = r4.clone() * r2.clone();

            let radial =
                T::one() + k1.clone() * r2.clone() + k2.clone() * r4.clone() + k3.clone() * r6;
            let xy = x_u.clone() * y_u.clone();
            let two = T::from_f64(2.0).unwrap();
            let dx_t = two.clone() * p1.clone() * xy.clone()
                + p2.clone() * (r2.clone() + two.clone() * x_u.clone() * x_u.clone());
            let dy_t = p1.clone() * (r2.clone() + two.clone() * y_u.clone() * y_u.clone())
                + two.clone() * p2.clone() * xy;

            x_u = (x_d.clone() - dx_t) / radial.clone();
            y_u = (y_d.clone() - dy_t) / radial;
        }
        (x_u, y_u)
    }
}

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

impl SensorKernel for Scheimpflug2Kernel {
    const DIM: usize = 2;

    fn to_sensor<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        let s = params.expect("Scheimpflug2 requires a sensor block");
        debug_assert!(s.len() >= 2, "sensor must have 2 params");
        apply_scheimpflug_generic(x, y, s[0].clone(), s[1].clone())
    }

    fn to_normalized<T: RealField>(params: Option<DVectorView<'_, T>>, x: T, y: T) -> (T, T) {
        let s = params.expect("Scheimpflug2 requires a sensor block");
        debug_assert!(s.len() >= 2, "sensor must have 2 params");
        apply_scheimpflug_inverse_generic(x, y, s[0].clone(), s[1].clone())
    }
}
