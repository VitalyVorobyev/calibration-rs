use nalgebra::{Matrix3, Point2, RealField, Vector3};
use serde::{Deserialize, Serialize};

/// Sensor model mapping between normalized and sensor-plane coordinates.
///
/// Generic over `T: RealField` with no `Copy` bound, so the optimizer's
/// dual-number autodiff evaluates the same code.
pub trait SensorModel<T: RealField> {
    /// Map normalized coordinates to the sensor plane.
    fn normalized_to_sensor(&self, n: &Point2<T>) -> Point2<T>;
    /// Map sensor-plane coordinates back to normalized coordinates.
    fn sensor_to_normalized(&self, s: &Point2<T>) -> Point2<T>;
}

/// Identity sensor model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct IdentitySensor;

impl<T: RealField> SensorModel<T> for IdentitySensor {
    fn normalized_to_sensor(&self, n: &Point2<T>) -> Point2<T> {
        n.clone()
    }

    fn sensor_to_normalized(&self, s: &Point2<T>) -> Point2<T> {
        s.clone()
    }
}

/// Sensor model represented by a 3x3 homography.
#[derive(Clone, Debug)]
pub struct HomographySensor<T: RealField> {
    /// Homography matrix (normalized -> sensor).
    pub h: Matrix3<T>,
    /// Inverse homography matrix (sensor -> normalized).
    pub h_inv: Matrix3<T>,
}

impl<T: RealField> HomographySensor<T> {
    /// Build a homography sensor if the matrix is invertible.
    pub fn new(h: Matrix3<T>) -> Option<Self> {
        let h_inv = h.clone().try_inverse()?;
        Some(Self { h, h_inv })
    }
}

impl<T: RealField> SensorModel<T> for HomographySensor<T> {
    fn normalized_to_sensor(&self, n: &Point2<T>) -> Point2<T> {
        apply_homography(&self.h, n)
    }

    fn sensor_to_normalized(&self, s: &Point2<T>) -> Point2<T> {
        apply_homography_inverse(&self.h_inv, s)
    }
}

/// Scheimpflug tilt parameters (OpenCV-compatible).
// Generic over the scalar so the optimizer evaluates the tilted sensor
// through this type under autodiff; `f64` is the default.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ScheimpflugParams<T: RealField = f64> {
    /// Tilt around X axis in radians (alias: tau_x).
    #[serde(alias = "tau_x")]
    pub tilt_x: T,
    /// Tilt around Y axis in radians (alias: tau_y).
    #[serde(alias = "tau_y")]
    pub tilt_y: T,
}

impl Default for ScheimpflugParams {
    fn default() -> Self {
        Self {
            tilt_x: 0.0,
            tilt_y: 0.0,
        }
    }
}

impl ScheimpflugParams {
    /// Build a homography matching OpenCV's tilted sensor model.
    ///
    /// Panics if the generated homography is not invertible.
    pub fn compile(&self) -> HomographySensor<f64> {
        let h = tilt_projection_matrix(self.tilt_x, self.tilt_y);
        let h_inv = h
            .try_inverse()
            .expect("Scheimpflug homography not invertible");
        HomographySensor { h, h_inv }
    }
}

impl<T: RealField> SensorModel<T> for ScheimpflugParams<T> {
    fn normalized_to_sensor(&self, n: &Point2<T>) -> Point2<T> {
        let h = tilt_projection_matrix(self.tilt_x.clone(), self.tilt_y.clone());
        apply_homography(&h, n)
    }

    /// Inverts the tilt homography; a singular homography (not reachable for
    /// physical tilts) returns the input unchanged.
    fn sensor_to_normalized(&self, s: &Point2<T>) -> Point2<T> {
        let h = tilt_projection_matrix(self.tilt_x.clone(), self.tilt_y.clone());
        match h.try_inverse() {
            Some(h_inv) => apply_homography_inverse(&h_inv, s),
            None => s.clone(),
        }
    }
}

/// `H·[p, 1]` dehomogenized, with the depth clamped to at least `1e-12` so a
/// point at or behind the sensor plane stays finite (and differentiable).
fn apply_homography<T: RealField>(h: &Matrix3<T>, p: &Point2<T>) -> Point2<T> {
    let q = h * Vector3::new(p.x.clone(), p.y.clone(), T::one());
    let eps = T::from_f64(1e-12).unwrap();
    let z = if q.z.clone() > eps.clone() {
        q.z.clone()
    } else {
        eps
    };
    Point2::new(q.x.clone() / z.clone(), q.y.clone() / z)
}

/// `H⁻¹·[p, 1]` dehomogenized, with `|z|` clamped to at least `1e-12`.
fn apply_homography_inverse<T: RealField>(h_inv: &Matrix3<T>, p: &Point2<T>) -> Point2<T> {
    let q = h_inv * Vector3::new(p.x.clone(), p.y.clone(), T::one());
    let eps = T::from_f64(1e-12).unwrap();
    let z = if q.z.clone().abs() > eps.clone() {
        q.z.clone()
    } else {
        eps
    };
    Point2::new(q.x.clone() / z.clone(), q.y.clone() / z)
}

/// OpenCV's tilted-sensor homography `proj_z · Ry(τy) · Rx(τx)`.
pub(crate) fn tilt_projection_matrix<T: RealField>(tau_x: T, tau_y: T) -> Matrix3<T> {
    let s_tx = tau_x.clone().sin();
    let c_tx = tau_x.cos();
    let s_ty = tau_y.clone().sin();
    let c_ty = tau_y.cos();
    let zero = T::zero();
    let one = T::one();

    let rot_x = Matrix3::new(
        one.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        c_tx.clone(),
        s_tx.clone(),
        zero.clone(),
        -s_tx,
        c_tx,
    );
    let rot_y = Matrix3::new(
        c_ty.clone(),
        zero.clone(),
        -s_ty.clone(),
        zero.clone(),
        one.clone(),
        zero.clone(),
        s_ty,
        zero.clone(),
        c_ty,
    );
    let rot_xy = rot_y * rot_x;

    let r22 = rot_xy[(2, 2)].clone();
    let proj_z = Matrix3::new(
        r22.clone(),
        zero.clone(),
        -rot_xy[(0, 2)].clone(),
        zero.clone(),
        r22,
        -rot_xy[(1, 2)].clone(),
        zero.clone(),
        zero,
        one,
    );

    proj_z * rot_xy
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3};

    /// Independent re-derivation of OpenCV's `computeTiltProjectionMatrix`.
    ///
    /// OpenCV builds the tilt rotation as `R = Ry(τy) · Rx(τx)` with the
    /// *sensor-frame* sign convention
    /// `Rx = [[1,0,0],[0,c,s],[0,-s,c]]`, `Ry = [[c,0,-s],[0,1,0],[s,0,c]]`,
    /// which equal nalgebra's right-handed `from_axis_angle` evaluated at the
    /// *negated* angle. We deliberately rebuild them through `Rotation3`
    /// (a different code path than the impl's literal `Matrix3::new`) so the
    /// test cross-checks the hand-written matrices rather than copying them.
    fn opencv_tilt_reference(tau_x: f64, tau_y: f64) -> Matrix3<f64> {
        let rx = *Rotation3::from_axis_angle(&Vector3::x_axis(), -tau_x).matrix();
        let ry = *Rotation3::from_axis_angle(&Vector3::y_axis(), -tau_y).matrix();
        let r = ry * rx;
        let proj_z = Matrix3::new(
            r[(2, 2)],
            0.0,
            -r[(0, 2)],
            0.0,
            r[(2, 2)],
            -r[(1, 2)],
            0.0,
            0.0,
            1.0,
        );
        proj_z * r
    }

    #[test]
    fn tilt_matrix_matches_opencv_convention() {
        // Synthetic τ pairs (radians) — not the private dataset's values.
        for &(tx, ty) in &[
            (0.0, 0.0),
            (0.05, -0.03),
            (0.1, 0.1),
            (-0.12, 0.07),
            (0.2, -0.25),
        ] {
            let got = tilt_projection_matrix(tx, ty);
            let want = opencv_tilt_reference(tx, ty);
            let diff = (got - want).abs().max();
            assert!(
                diff < 1e-12,
                "tilt matrix mismatch at (τx={tx}, τy={ty}): max abs diff {diff:e}\n got = {got}\nwant = {want}"
            );
        }
    }

    #[test]
    fn tilt_identity_at_zero() {
        let h = tilt_projection_matrix(0.0, 0.0);
        assert!((h - Matrix3::<f64>::identity()).abs().max() < 1e-15);
    }

    #[test]
    fn optical_axis_maps_to_sensor_origin() {
        // The optical-axis point (normalized origin) must land on the sensor
        // origin for any tilt — an invariant independent of the matrix internals.
        for &(tx, ty) in &[(0.05, -0.03), (0.1, 0.1), (-0.12, 0.07)] {
            let sensor = ScheimpflugParams {
                tilt_x: tx,
                tilt_y: ty,
            }
            .compile();
            let s = sensor.normalized_to_sensor(&Point2::new(0.0, 0.0));
            assert!(
                s.x.abs() < 1e-12 && s.y.abs() < 1e-12,
                "(τx={tx}, τy={ty}) → {s:?}"
            );
        }
    }

    #[test]
    fn sensor_roundtrip_is_identity() {
        let sensor = ScheimpflugParams {
            tilt_x: 0.08,
            tilt_y: -0.06,
        }
        .compile();
        for &(x, y) in &[(0.0, 0.0), (0.3, -0.2), (-0.45, 0.5)] {
            let n = Point2::new(x, y);
            let back = sensor.sensor_to_normalized(&sensor.normalized_to_sensor(&n));
            assert!((back.x - x).abs() < 1e-12 && (back.y - y).abs() < 1e-12);
        }
    }
}
