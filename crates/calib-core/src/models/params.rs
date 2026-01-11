use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

use super::{
    BrownConrady5, Camera, FxFyCxCySkew, HomographySensor, IdentitySensor, NoDistortion, Pinhole,
    ProjectionModel, ScheimpflugParams, SensorModel,
};
use crate::Real;

/// Serializable projection model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProjectionParams {
    /// Classic pinhole model.
    Pinhole,
}

/// Serializable distortion model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DistortionParams {
    /// No distortion.
    None,
    /// Brown-Conrady 5-parameter radial-tangential model.
    BrownConrady5 {
        #[serde(flatten)]
        params: BrownConrady5<Real>,
    },
}

/// Serializable sensor model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SensorParams {
    /// Identity sensor model.
    Identity,
    /// Homography applied in the sensor plane.
    Homography { h: [[Real; 3]; 3] },
    /// Scheimpflug/tilted sensor model.
    Scheimpflug {
        #[serde(flatten)]
        params: ScheimpflugParams,
    },
}

/// Serializable intrinsics parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IntrinsicsParams {
    /// Pinhole intrinsics with optional skew.
    FxFyCxCySkew {
        #[serde(flatten)]
        params: FxFyCxCySkew<Real>,
    },
}

/// Serializable camera parameters for building a runtime model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraParams {
    /// Projection model parameters.
    pub projection: ProjectionParams,
    /// Distortion model parameters.
    pub distortion: DistortionParams,
    /// Sensor model parameters.
    pub sensor: SensorParams,
    /// Intrinsics model parameters.
    pub intrinsics: IntrinsicsParams,
}

/// Concrete camera type built from parameters (f64).
pub type CameraModel = Camera<Real, AnyProjection, AnyDistortion, AnySensor, AnyIntrinsics>;

impl CameraParams {
    /// Build a concrete camera model from this parameter set.
    ///
    /// Panics if a provided homography is not invertible.
    pub fn build(&self) -> CameraModel {
        let proj = match self.projection {
            ProjectionParams::Pinhole => AnyProjection::Pinhole(Pinhole),
        };

        let dist = match self.distortion {
            DistortionParams::None => AnyDistortion::None(NoDistortion),
            DistortionParams::BrownConrady5 { params } => AnyDistortion::BrownConrady5(params),
        };

        let sensor = match &self.sensor {
            SensorParams::Identity => AnySensor::Identity(IdentitySensor),
            SensorParams::Homography { h } => {
                let h = Matrix3::from_row_slice(&[
                    h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
                ]);
                let h_inv = h.try_inverse().expect("Homography not invertible");
                AnySensor::Homography(HomographySensor { h, h_inv })
            }
            SensorParams::Scheimpflug { params } => AnySensor::Homography(params.compile()),
        };

        let k = match self.intrinsics {
            IntrinsicsParams::FxFyCxCySkew { params } => AnyIntrinsics::FxFyCxCySkew(params),
        };

        Camera::new(proj, dist, sensor, k)
    }
}

// Internal type-erased model wrappers to produce a single concrete Camera type.
// These are intentionally doc-hidden from the public API surface.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum AnyProjection {
    Pinhole(Pinhole),
}

impl ProjectionModel<Real> for AnyProjection {
    fn project_dir(&self, dir_c: &nalgebra::Vector3<Real>) -> Option<nalgebra::Vector2<Real>> {
        match self {
            AnyProjection::Pinhole(m) => m.project_dir(dir_c),
        }
    }

    fn unproject_dir(&self, n: &nalgebra::Vector2<Real>) -> nalgebra::Vector3<Real> {
        match self {
            AnyProjection::Pinhole(m) => m.unproject_dir(n),
        }
    }
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum AnyDistortion {
    None(NoDistortion),
    BrownConrady5(BrownConrady5<Real>),
}

impl super::DistortionModel<Real> for AnyDistortion {
    fn distort(&self, n: &nalgebra::Vector2<Real>) -> nalgebra::Vector2<Real> {
        match self {
            AnyDistortion::None(m) => m.distort(n),
            AnyDistortion::BrownConrady5(m) => m.distort(n),
        }
    }

    fn undistort(&self, n: &nalgebra::Vector2<Real>) -> nalgebra::Vector2<Real> {
        match self {
            AnyDistortion::None(m) => m.undistort(n),
            AnyDistortion::BrownConrady5(m) => m.undistort(n),
        }
    }
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum AnySensor {
    Identity(IdentitySensor),
    Homography(HomographySensor<Real>),
}

impl SensorModel<Real> for AnySensor {
    fn normalized_to_sensor(&self, n: &nalgebra::Vector2<Real>) -> nalgebra::Vector2<Real> {
        match self {
            AnySensor::Identity(m) => m.normalized_to_sensor(n),
            AnySensor::Homography(m) => m.normalized_to_sensor(n),
        }
    }

    fn sensor_to_normalized(&self, s: &nalgebra::Vector2<Real>) -> nalgebra::Vector2<Real> {
        match self {
            AnySensor::Identity(m) => m.sensor_to_normalized(s),
            AnySensor::Homography(m) => m.sensor_to_normalized(s),
        }
    }
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum AnyIntrinsics {
    FxFyCxCySkew(FxFyCxCySkew<Real>),
}

impl super::IntrinsicsModel<Real> for AnyIntrinsics {
    fn sensor_to_pixel(&self, sensor: &nalgebra::Vector2<Real>) -> nalgebra::Vector2<Real> {
        match self {
            AnyIntrinsics::FxFyCxCySkew(m) => m.sensor_to_pixel(sensor),
        }
    }

    fn pixel_to_sensor(&self, pixel: &nalgebra::Vector2<Real>) -> nalgebra::Vector2<Real> {
        match self {
            AnyIntrinsics::FxFyCxCySkew(m) => m.pixel_to_sensor(pixel),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_build_camera() {
        let params = CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::None,
            sensor: SensorParams::Identity,
            intrinsics: IntrinsicsParams::FxFyCxCySkew {
                params: FxFyCxCySkew {
                    fx: 800.0,
                    fy: 810.0,
                    cx: 640.0,
                    cy: 360.0,
                    skew: 0.0,
                },
            },
        };
        let cam = params.build();
        let px = cam.project_point_c(&nalgebra::Vector3::new(0.1, 0.2, 1.0));
        assert!(px.is_some());
    }

    #[test]
    fn distortion_params_serde_shape() {
        let json = r#"{
            "type": "brown_conrady5",
            "k1": 0.1,
            "k2": 0.01,
            "k3": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "iters": 4
        }"#;
        let cfg: DistortionParams = serde_json::from_str(json).expect("serde should succeed");
        match cfg {
            DistortionParams::BrownConrady5 { params } => {
                assert!((params.k1 - 0.1).abs() < 1e-12);
                assert!((params.k2 - 0.01).abs() < 1e-12);
                assert!((params.k3 - 0.0).abs() < 1e-12);
                assert!((params.p1 - 0.0).abs() < 1e-12);
                assert!((params.p2 - 0.0).abs() < 1e-12);
                assert_eq!(params.iters, 4);
            }
            _ => panic!("expected BrownConrady5 params"),
        }
    }
}
