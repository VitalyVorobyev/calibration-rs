use nalgebra::{Matrix3, RealField, Vector2, Vector3};
use serde::{Deserialize, Serialize};

/// Sensor model mapping between normalized and sensor-plane coordinates.
pub trait SensorModel<S: RealField + Copy> {
    /// Map normalized coordinates to the sensor plane.
    fn normalized_to_sensor(&self, n: &Vector2<S>) -> Vector2<S>;
    /// Map sensor-plane coordinates back to normalized coordinates.
    fn sensor_to_normalized(&self, s: &Vector2<S>) -> Vector2<S>;
}

/// Identity sensor model.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct IdentitySensor;

impl<S: RealField + Copy> SensorModel<S> for IdentitySensor {
    fn normalized_to_sensor(&self, n: &Vector2<S>) -> Vector2<S> {
        *n
    }

    fn sensor_to_normalized(&self, s: &Vector2<S>) -> Vector2<S> {
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
    fn normalized_to_sensor(&self, n: &Vector2<S>) -> Vector2<S> {
        dehomogenize(&(self.h * homogenize(n)))
    }

    fn sensor_to_normalized(&self, s: &Vector2<S>) -> Vector2<S> {
        dehomogenize(&(self.h_inv * homogenize(s)))
    }
}

/// Scheimpflug tilt parameters (OpenCV-compatible).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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

fn homogenize<S: RealField + Copy>(p: &Vector2<S>) -> Vector3<S> {
    Vector3::new(p.x, p.y, S::one())
}

fn dehomogenize<S: RealField + Copy>(p: &Vector3<S>) -> Vector2<S> {
    Vector2::new(p.x / p.z, p.y / p.z)
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
