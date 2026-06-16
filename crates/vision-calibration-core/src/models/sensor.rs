use nalgebra::{Matrix3, Point2, Point3, RealField};
use serde::{Deserialize, Serialize};

/// Sensor model mapping between normalized and sensor-plane coordinates.
pub trait SensorModel<S: RealField + Copy> {
    /// Map normalized coordinates to the sensor plane.
    fn normalized_to_sensor(&self, n: &Point2<S>) -> Point2<S>;
    /// Map sensor-plane coordinates back to normalized coordinates.
    fn sensor_to_normalized(&self, s: &Point2<S>) -> Point2<S>;
}

/// Identity sensor model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct IdentitySensor;

impl<S: RealField + Copy> SensorModel<S> for IdentitySensor {
    fn normalized_to_sensor(&self, n: &Point2<S>) -> Point2<S> {
        *n
    }

    fn sensor_to_normalized(&self, s: &Point2<S>) -> Point2<S> {
        *s
    }
}

/// Sensor model represented by a 3x3 homography.
#[derive(Clone, Debug)]
pub struct HomographySensor<S: RealField + Copy> {
    /// Homography matrix (normalized -> sensor).
    pub h: Matrix3<S>,
    /// Inverse homography matrix (sensor -> normalized).
    pub h_inv: Matrix3<S>,
}

impl<S: RealField + Copy> HomographySensor<S> {
    /// Build a homography sensor if the matrix is invertible.
    pub fn new(h: Matrix3<S>) -> Option<Self> {
        let h_inv = h.try_inverse()?;
        Some(Self { h, h_inv })
    }
}

impl<S: RealField + Copy> SensorModel<S> for HomographySensor<S> {
    fn normalized_to_sensor(&self, n: &Point2<S>) -> Point2<S> {
        dehomogenize(&(self.h * homogenize(n)))
    }

    fn sensor_to_normalized(&self, s: &Point2<S>) -> Point2<S> {
        dehomogenize(&(self.h_inv * homogenize(s)))
    }
}

/// Scheimpflug tilt parameters (OpenCV-compatible).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct ScheimpflugParams {
    /// Tilt around X axis in radians (alias: tau_x).
    #[serde(alias = "tau_x")]
    pub tilt_x: f64,
    /// Tilt around Y axis in radians (alias: tau_y).
    #[serde(alias = "tau_y")]
    pub tilt_y: f64,
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

fn homogenize<S: RealField + Copy>(p: &Point2<S>) -> Point3<S> {
    Point3::new(p.x, p.y, S::one())
}

fn dehomogenize<S: RealField + Copy>(p: &Point3<S>) -> Point2<S> {
    Point2::new(p.x / p.z, p.y / p.z)
}

fn tilt_projection_matrix(tau_x: f64, tau_y: f64) -> Matrix3<f64> {
    let (s_tx, c_tx) = tau_x.sin_cos();
    let (s_ty, c_ty) = tau_y.sin_cos();

    let rot_x = Matrix3::new(1.0, 0.0, 0.0, 0.0, c_tx, s_tx, 0.0, -s_tx, c_tx);
    let rot_y = Matrix3::new(c_ty, 0.0, -s_ty, 0.0, 1.0, 0.0, s_ty, 0.0, c_ty);
    let rot_xy = rot_y * rot_x;

    let proj_z = Matrix3::new(
        rot_xy[(2, 2)],
        0.0,
        -rot_xy[(0, 2)],
        0.0,
        rot_xy[(2, 2)],
        -rot_xy[(1, 2)],
        0.0,
        0.0,
        1.0,
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
