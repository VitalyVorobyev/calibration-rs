use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

use super::{
    BrownConrady5, Camera, FxFyCxCySkew, HomographySensor, IdentitySensor, NoDistortion, Pinhole,
    ProjectionModel, ScheimpflugParams, SensorModel,
};

/// Serializable projection model configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProjectionConfig {
    /// Classic pinhole model.
    Pinhole,
}

/// Serializable distortion model configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistortionConfig {
    /// No distortion.
    None,
    /// Brown-Conrady 5-parameter radial-tangential model.
    BrownConrady5 {
        /// Radial distortion k1.
        k1: f64,
        /// Radial distortion k2.
        k2: f64,
        /// Radial distortion k3.
        k3: f64,
        /// Tangential distortion p1.
        p1: f64,
        /// Tangential distortion p2.
        p2: f64,
        /// Iterations for undistortion (if None, default is used).
        iters: Option<u32>,
    },
}

/// Serializable sensor model configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorConfig {
    /// Identity sensor model.
    Identity,
    /// Homography applied in the sensor plane.
    Homography { h: [[f64; 3]; 3] },
    /// Scheimpflug/tilted sensor model.
    Scheimpflug(ScheimpflugParams),
}

/// Serializable intrinsics configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IntrinsicsConfig {
    /// Pinhole intrinsics with optional skew.
    FxFyCxCySkew {
        /// Focal length in pixels along X.
        fx: f64,
        /// Focal length in pixels along Y.
        fy: f64,
        /// Principal point X coordinate in pixels.
        cx: f64,
        /// Principal point Y coordinate in pixels.
        cy: f64,
        /// Skew term (typically 0).
        skew: f64,
    },
}

/// Type-erased projection model for configs.
#[derive(Clone, Debug)]
pub enum AnyProj {
    /// Pinhole projection.
    Pinhole(Pinhole),
}

impl ProjectionModel<f64> for AnyProj {
    fn project_dir(&self, dir_c: &nalgebra::Vector3<f64>) -> Option<nalgebra::Vector2<f64>> {
        match self {
            AnyProj::Pinhole(m) => m.project_dir(dir_c),
        }
    }

    fn unproject_dir(&self, n: &nalgebra::Vector2<f64>) -> nalgebra::Vector3<f64> {
        match self {
            AnyProj::Pinhole(m) => m.unproject_dir(n),
        }
    }
}

/// Type-erased distortion model for configs.
#[derive(Clone, Debug)]
pub enum AnyDist {
    /// No distortion.
    None(NoDistortion),
    /// Brown-Conrady distortion.
    BrownConrady5(BrownConrady5<f64>),
}

impl super::DistortionModel<f64> for AnyDist {
    fn distort(&self, n: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnyDist::None(m) => m.distort(n),
            AnyDist::BrownConrady5(m) => m.distort(n),
        }
    }

    fn undistort(&self, n: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnyDist::None(m) => m.undistort(n),
            AnyDist::BrownConrady5(m) => m.undistort(n),
        }
    }
}

/// Type-erased sensor model for configs.
#[derive(Clone, Debug)]
pub enum AnySensor {
    /// Identity sensor.
    Identity(IdentitySensor),
    /// Homography sensor.
    Homography(HomographySensor<f64>),
}

impl SensorModel<f64> for AnySensor {
    fn normalized_to_sensor(&self, n: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnySensor::Identity(m) => m.normalized_to_sensor(n),
            AnySensor::Homography(m) => m.normalized_to_sensor(n),
        }
    }

    fn sensor_to_normalized(&self, s: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnySensor::Identity(m) => m.sensor_to_normalized(s),
            AnySensor::Homography(m) => m.sensor_to_normalized(s),
        }
    }
}

/// Type-erased intrinsics model for configs.
#[derive(Clone, Debug)]
pub enum AnyK {
    /// Pinhole intrinsics with skew.
    FxFyCxCySkew(FxFyCxCySkew<f64>),
}

impl super::IntrinsicsModel<f64> for AnyK {
    fn sensor_to_pixel(&self, sensor: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnyK::FxFyCxCySkew(m) => m.sensor_to_pixel(sensor),
        }
    }

    fn pixel_to_sensor(&self, pixel: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnyK::FxFyCxCySkew(m) => m.pixel_to_sensor(pixel),
        }
    }
}

/// Serializable camera configuration for building a runtime model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Projection model configuration.
    pub projection: ProjectionConfig,
    /// Distortion model configuration.
    pub distortion: DistortionConfig,
    /// Sensor model configuration.
    pub sensor: SensorConfig,
    /// Intrinsics model configuration.
    pub intrinsics: IntrinsicsConfig,
}

/// Concrete camera type built from configs (f64).
pub type CameraF64 = Camera<f64, AnyProj, AnyDist, AnySensor, AnyK>;

impl CameraConfig {
    /// Build a concrete camera model from this configuration.
    ///
    /// Panics if a provided homography is not invertible.
    pub fn build(&self) -> CameraF64 {
        let proj = match self.projection {
            ProjectionConfig::Pinhole => AnyProj::Pinhole(Pinhole),
        };

        let dist = match self.distortion {
            DistortionConfig::None => AnyDist::None(NoDistortion),
            DistortionConfig::BrownConrady5 {
                k1,
                k2,
                k3,
                p1,
                p2,
                iters,
            } => AnyDist::BrownConrady5(BrownConrady5 {
                k1,
                k2,
                k3,
                p1,
                p2,
                iters: iters.unwrap_or(8),
            }),
        };

        let sensor = match &self.sensor {
            SensorConfig::Identity => AnySensor::Identity(IdentitySensor),
            SensorConfig::Homography { h } => {
                let h = Matrix3::from_row_slice(&[
                    h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
                ]);
                let h_inv = h.try_inverse().expect("Homography not invertible");
                AnySensor::Homography(HomographySensor { h, h_inv })
            }
            SensorConfig::Scheimpflug(p) => AnySensor::Homography(p.compile()),
        };

        let k = match self.intrinsics {
            IntrinsicsConfig::FxFyCxCySkew {
                fx,
                fy,
                cx,
                cy,
                skew,
            } => AnyK::FxFyCxCySkew(FxFyCxCySkew {
                fx,
                fy,
                cx,
                cy,
                skew,
            }),
        };

        Camera::new(proj, dist, sensor, k)
    }
}
