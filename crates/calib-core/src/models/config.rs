use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

use super::{
    BrownConrady5, Camera, FxFyCxCySkew, HomographySensor, IdentitySensor, NoDistortion, Pinhole,
    ProjectionModel, ScheimpflugParams, SensorModel,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProjectionConfig {
    Pinhole,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistortionConfig {
    None,
    BrownConrady5 {
        k1: f64,
        k2: f64,
        k3: f64,
        p1: f64,
        p2: f64,
        iters: Option<u32>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorConfig {
    Identity,
    Homography { h: [[f64; 3]; 3] },
    Scheimpflug(ScheimpflugParams),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IntrinsicsConfig {
    FxFyCxCySkew {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        skew: f64,
    },
}

#[derive(Clone, Debug)]
pub enum AnyProj {
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

#[derive(Clone, Debug)]
pub enum AnyDist {
    None(NoDistortion),
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

#[derive(Clone, Debug)]
pub enum AnySensor {
    Identity(IdentitySensor),
    Homography(HomographySensor<f64>),
}

impl SensorModel<f64> for AnySensor {
    fn to_sensor(&self, n: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnySensor::Identity(m) => m.to_sensor(n),
            AnySensor::Homography(m) => m.to_sensor(n),
        }
    }

    fn from_sensor(&self, s: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnySensor::Identity(m) => m.from_sensor(s),
            AnySensor::Homography(m) => m.from_sensor(s),
        }
    }
}

#[derive(Clone, Debug)]
pub enum AnyK {
    FxFyCxCySkew(FxFyCxCySkew<f64>),
}

impl super::IntrinsicsModel<f64> for AnyK {
    fn to_pixel(&self, s: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnyK::FxFyCxCySkew(m) => m.to_pixel(s),
        }
    }

    fn from_pixel(&self, px: &nalgebra::Vector2<f64>) -> nalgebra::Vector2<f64> {
        match self {
            AnyK::FxFyCxCySkew(m) => m.from_pixel(px),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraConfig {
    pub projection: ProjectionConfig,
    pub distortion: DistortionConfig,
    pub sensor: SensorConfig,
    pub intrinsics: IntrinsicsConfig,
}

pub type CameraF64 = Camera<f64, AnyProj, AnyDist, AnySensor, AnyK>;

impl CameraConfig {
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
                    h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1],
                    h[2][2],
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
