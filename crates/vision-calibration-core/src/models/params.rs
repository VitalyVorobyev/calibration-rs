use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

use super::{
    BrownConrady5, Camera, CameraModelSpec, DistortionKind, Division, FxFyCxCy, HomographySensor,
    IdentitySensor, NoDistortion, Pinhole, ProjectionKind, ProjectionModel, RationalPolynomial,
    ScheimpflugParams, SensorKind, SensorModel, ThinPrism,
};
use crate::{Error, Real};

/// Serializable projection model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProjectionParams {
    /// Classic pinhole model.
    Pinhole,
}

/// Serializable distortion model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DistortionParams {
    /// No distortion.
    None,
    /// Brown-Conrady 5-parameter radial-tangential model.
    BrownConrady5 {
        /// Flattened Brown-Conrady coefficients.
        #[serde(flatten)]
        params: BrownConrady5<Real>,
    },
    /// OpenCV rational polynomial 8-parameter model.
    Rational {
        /// Flattened rational polynomial coefficients.
        #[serde(flatten)]
        params: RationalPolynomial<Real>,
    },
    /// Brown-Conrady + thin-prism 9-parameter model.
    ThinPrism {
        /// Flattened thin-prism coefficients.
        #[serde(flatten)]
        params: ThinPrism<Real>,
    },
    /// Fitzgibbon single-parameter division model.
    Division {
        /// Division distortion coefficient.
        lambda: Real,
    },
}

/// Serializable sensor model parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SensorParams {
    /// Identity sensor model.
    Identity,
    /// Homography applied in the sensor plane.
    Homography {
        /// Row-major homography matrix mapping normalized to sensor coordinates.
        h: [[Real; 3]; 3],
    },
    /// Scheimpflug/tilted sensor model.
    Scheimpflug {
        /// Flattened Scheimpflug tilt parameters.
        #[serde(flatten)]
        params: ScheimpflugParams,
    },
}

/// Newton iterations of the inverse map in a distortion built by
/// [`DistortionParams::from_values`].
const UNDISTORT_ITERS: u32 = 8;

fn check_len(slot: &str, expected: usize, values: &[Real]) -> Result<(), Error> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(Error::invalid_input(format!(
            "expected {expected} {slot} values, got {}",
            values.len()
        )))
    }
}

impl ProjectionParams {
    /// The projection model of these parameters.
    pub fn kind(&self) -> ProjectionKind {
        match self {
            ProjectionParams::Pinhole => ProjectionKind::Pinhole,
        }
    }

    /// The projection's own parameters in the order of
    /// [`ProjectionKind::param_names`] (empty for a pinhole).
    pub fn values(&self) -> Vec<Real> {
        match self {
            ProjectionParams::Pinhole => Vec::new(),
        }
    }

    /// Parameters of a `kind` projection from values in the order of
    /// [`ProjectionKind::param_names`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `values.len() != kind.dim()`.
    pub fn from_values(kind: ProjectionKind, values: &[Real]) -> Result<Self, Error> {
        check_len("projection", kind.dim(), values)?;
        Ok(match kind {
            ProjectionKind::Pinhole => ProjectionParams::Pinhole,
        })
    }
}

impl DistortionParams {
    /// The distortion model of these parameters.
    pub fn kind(&self) -> DistortionKind {
        match self {
            DistortionParams::None => DistortionKind::None,
            DistortionParams::BrownConrady5 { .. } => DistortionKind::BrownConrady5,
            DistortionParams::Rational { .. } => DistortionKind::Rational8,
            DistortionParams::ThinPrism { .. } => DistortionKind::ThinPrism9,
            DistortionParams::Division { .. } => DistortionKind::Division1,
        }
    }

    /// The coefficients in the order of [`DistortionKind::param_names`]
    /// (empty without distortion).
    pub fn values(&self) -> Vec<Real> {
        match self {
            DistortionParams::None => Vec::new(),
            DistortionParams::BrownConrady5 { params: p } => vec![p.k1, p.k2, p.k3, p.p1, p.p2],
            DistortionParams::Rational { params: p } => {
                vec![p.k1, p.k2, p.k3, p.k4, p.k5, p.k6, p.p1, p.p2]
            }
            DistortionParams::ThinPrism { params: p } => {
                vec![p.k1, p.k2, p.k3, p.p1, p.p2, p.s1, p.s2, p.s3, p.s4]
            }
            DistortionParams::Division { lambda } => vec![*lambda],
        }
    }

    /// Parameters of a `kind` distortion from coefficients in the order of
    /// [`DistortionKind::param_names`]. Iterative inverses get 8 Newton steps.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `values.len() != kind.dim()`.
    pub fn from_values(kind: DistortionKind, v: &[Real]) -> Result<Self, Error> {
        check_len("distortion", kind.dim(), v)?;
        let iters = UNDISTORT_ITERS;
        Ok(match kind {
            DistortionKind::None => DistortionParams::None,
            DistortionKind::BrownConrady5 => DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: v[0],
                    k2: v[1],
                    k3: v[2],
                    p1: v[3],
                    p2: v[4],
                    iters,
                },
            },
            DistortionKind::Rational8 => DistortionParams::Rational {
                params: RationalPolynomial {
                    k1: v[0],
                    k2: v[1],
                    k3: v[2],
                    k4: v[3],
                    k5: v[4],
                    k6: v[5],
                    p1: v[6],
                    p2: v[7],
                    iters,
                },
            },
            DistortionKind::ThinPrism9 => DistortionParams::ThinPrism {
                params: ThinPrism {
                    k1: v[0],
                    k2: v[1],
                    k3: v[2],
                    p1: v[3],
                    p2: v[4],
                    s1: v[5],
                    s2: v[6],
                    s3: v[7],
                    s4: v[8],
                    iters,
                },
            },
            DistortionKind::Division1 => DistortionParams::Division { lambda: v[0] },
        })
    }
}

impl DistortionParams {
    /// A copy with the coefficients replaced by `v` (in the order of
    /// [`DistortionKind::param_names`]), keeping the model and its
    /// non-parameter settings (the Newton iteration count of the inverse).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `v.len()` differs from the model's
    /// dimension.
    pub fn with_values(&self, v: &[Real]) -> Result<Self, Error> {
        check_len("distortion", self.kind().dim(), v)?;
        Ok(match self {
            DistortionParams::None => DistortionParams::None,
            DistortionParams::BrownConrady5 { params: p } => DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: v[0],
                    k2: v[1],
                    k3: v[2],
                    p1: v[3],
                    p2: v[4],
                    ..*p
                },
            },
            DistortionParams::Rational { params: p } => DistortionParams::Rational {
                params: RationalPolynomial {
                    k1: v[0],
                    k2: v[1],
                    k3: v[2],
                    k4: v[3],
                    k5: v[4],
                    k6: v[5],
                    p1: v[6],
                    p2: v[7],
                    ..*p
                },
            },
            DistortionParams::ThinPrism { params: p } => DistortionParams::ThinPrism {
                params: ThinPrism {
                    k1: v[0],
                    k2: v[1],
                    k3: v[2],
                    p1: v[3],
                    p2: v[4],
                    s1: v[5],
                    s2: v[6],
                    s3: v[7],
                    s4: v[8],
                    ..*p
                },
            },
            DistortionParams::Division { .. } => DistortionParams::Division { lambda: v[0] },
        })
    }
}

impl SensorParams {
    /// The sensor model of these parameters, or `None` for a general
    /// [`SensorParams::Homography`], which no kind describes.
    pub fn kind(&self) -> Option<SensorKind> {
        match self {
            SensorParams::Identity => Some(SensorKind::Identity),
            SensorParams::Scheimpflug { .. } => Some(SensorKind::Scheimpflug),
            SensorParams::Homography { .. } => None,
        }
    }

    /// The sensor's parameters in the order of [`SensorKind::param_names`]
    /// (empty for an identity sensor and for a general homography, which is
    /// never estimated).
    pub fn values(&self) -> Vec<Real> {
        match self {
            SensorParams::Identity | SensorParams::Homography { .. } => Vec::new(),
            SensorParams::Scheimpflug { params } => vec![params.tilt_x, params.tilt_y],
        }
    }

    /// Parameters of a `kind` sensor from values in the order of
    /// [`SensorKind::param_names`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `values.len() != kind.dim()`.
    pub fn from_values(kind: SensorKind, v: &[Real]) -> Result<Self, Error> {
        check_len("sensor", kind.dim(), v)?;
        Ok(match kind {
            SensorKind::Identity => SensorParams::Identity,
            SensorKind::Scheimpflug => SensorParams::Scheimpflug {
                params: ScheimpflugParams {
                    tilt_x: v[0],
                    tilt_y: v[1],
                },
            },
        })
    }
}

/// Serializable camera parameters for building a runtime model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct CameraParams {
    /// Projection model parameters.
    pub projection: ProjectionParams,
    /// Distortion model parameters.
    pub distortion: DistortionParams,
    /// Sensor model parameters.
    pub sensor: SensorParams,
    /// Pinhole intrinsics (`fx, fy, cx, cy` in pixels).
    pub intrinsics: FxFyCxCy<Real>,
}

/// Concrete camera type built from parameters (f64).
pub type CameraModel = Camera<Real, AnyProjection, AnyDistortion, AnySensor, FxFyCxCy<Real>>;

impl CameraParams {
    /// The model spec of these parameters, or `None` for a general
    /// [`SensorParams::Homography`] sensor, which no spec describes.
    pub fn spec(&self) -> Option<CameraModelSpec> {
        Some(CameraModelSpec::new(
            self.projection.kind(),
            self.distortion.kind(),
            self.sensor.kind()?,
        ))
    }

    /// The camera's parameter blocks in solver order, each in the order of
    /// its kind's `param_names`: intrinsics (`fx, fy, cx, cy`), then the
    /// projection, distortion and sensor values. A slot without parameters
    /// gives an empty block.
    pub fn blocks(&self) -> [Vec<Real>; 4] {
        let k = &self.intrinsics;
        [
            vec![k.fx, k.fy, k.cx, k.cy],
            self.projection.values(),
            self.distortion.values(),
            self.sensor.values(),
        ]
    }

    /// Build a concrete camera model from this parameter set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Singular`] if a
    /// [`SensorParams::Homography`] is not invertible. `CameraParams`
    /// deserializes straight from an export file, so this is reachable with
    /// hand-edited or corrupted input, not only with a programming mistake.
    pub fn build(&self) -> Result<CameraModel, Error> {
        let proj = match self.projection {
            ProjectionParams::Pinhole => AnyProjection::Pinhole(Pinhole),
        };

        let dist = match self.distortion {
            DistortionParams::None => AnyDistortion::None(NoDistortion),
            DistortionParams::BrownConrady5 { params } => AnyDistortion::BrownConrady5(params),
            DistortionParams::Rational { params } => AnyDistortion::Rational(params),
            DistortionParams::ThinPrism { params } => AnyDistortion::ThinPrism(params),
            DistortionParams::Division { lambda } => AnyDistortion::Division(Division { lambda }),
        };

        let sensor = match &self.sensor {
            SensorParams::Identity => AnySensor::Identity(IdentitySensor),
            SensorParams::Homography { h } => {
                let h = Matrix3::from_row_slice(&[
                    h[0][0], h[0][1], h[0][2], h[1][0], h[1][1], h[1][2], h[2][0], h[2][1], h[2][2],
                ]);
                let h_inv = h.try_inverse().ok_or(Error::Singular)?;
                AnySensor::Homography(HomographySensor { h, h_inv })
            }
            SensorParams::Scheimpflug { params } => AnySensor::Homography(params.compile()),
        };

        Ok(Camera::new(proj, dist, sensor, self.intrinsics))
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
    fn in_domain(&self, p_c: &nalgebra::Vector3<Real>) -> bool {
        match self {
            AnyProjection::Pinhole(m) => m.in_domain(p_c),
        }
    }

    fn project(&self, p_c: &nalgebra::Vector3<Real>) -> nalgebra::Point2<Real> {
        match self {
            AnyProjection::Pinhole(m) => m.project(p_c),
        }
    }

    fn unproject(&self, n: &nalgebra::Point2<Real>) -> super::Ray<Real> {
        match self {
            AnyProjection::Pinhole(m) => m.unproject(n),
        }
    }

    fn is_central(&self) -> bool {
        match self {
            AnyProjection::Pinhole(m) => ProjectionModel::<Real>::is_central(m),
        }
    }
}

#[derive(Clone, Debug)]
#[doc(hidden)]
pub enum AnyDistortion {
    /// Identity (no distortion).
    None(NoDistortion),
    /// Brown-Conrady 5-parameter model.
    BrownConrady5(BrownConrady5<Real>),
    /// Rational polynomial 8-parameter model.
    Rational(RationalPolynomial<Real>),
    /// Thin-prism 9-parameter model.
    ThinPrism(ThinPrism<Real>),
    /// Division 1-parameter model.
    Division(Division<Real>),
}

impl super::DistortionModel<Real> for AnyDistortion {
    fn distort(&self, n: &nalgebra::Point2<Real>) -> nalgebra::Point2<Real> {
        match self {
            AnyDistortion::None(m) => m.distort(n),
            AnyDistortion::BrownConrady5(m) => m.distort(n),
            AnyDistortion::Rational(m) => m.distort(n),
            AnyDistortion::ThinPrism(m) => m.distort(n),
            AnyDistortion::Division(m) => m.distort(n),
        }
    }

    fn undistort(&self, n: &nalgebra::Point2<Real>) -> nalgebra::Point2<Real> {
        match self {
            AnyDistortion::None(m) => m.undistort(n),
            AnyDistortion::BrownConrady5(m) => m.undistort(n),
            AnyDistortion::Rational(m) => m.undistort(n),
            AnyDistortion::ThinPrism(m) => m.undistort(n),
            AnyDistortion::Division(m) => m.undistort(n),
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
    fn normalized_to_sensor(&self, n: &nalgebra::Point2<Real>) -> nalgebra::Point2<Real> {
        match self {
            AnySensor::Identity(m) => m.normalized_to_sensor(n),
            AnySensor::Homography(m) => m.normalized_to_sensor(n),
        }
    }

    fn sensor_to_normalized(&self, s: &nalgebra::Point2<Real>) -> nalgebra::Point2<Real> {
        match self {
            AnySensor::Identity(m) => m.sensor_to_normalized(s),
            AnySensor::Homography(m) => m.sensor_to_normalized(s),
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
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 810.0,
                cx: 640.0,
                cy: 360.0,
            },
        };
        let cam = params.build().unwrap();
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
            DistortionParams::None => panic!("expected BrownConrady5 params, got None"),
            DistortionParams::Rational { .. } => {
                panic!("expected BrownConrady5 params, got Rational")
            }
            DistortionParams::ThinPrism { .. } => {
                panic!("expected BrownConrady5 params, got ThinPrism")
            }
            DistortionParams::Division { .. } => {
                panic!("expected BrownConrady5 params, got Division")
            }
        }
    }

    #[test]
    fn every_slot_round_trips_through_its_values() {
        for kind in ProjectionKind::ALL {
            let values: Vec<Real> = (0..kind.dim()).map(|i| 0.1 + i as Real).collect();
            let p = ProjectionParams::from_values(kind, &values).unwrap();
            assert_eq!((p.kind(), p.values()), (kind, values));
        }
        for kind in DistortionKind::ALL {
            let values: Vec<Real> = (0..kind.dim()).map(|i| 0.01 * (i + 1) as Real).collect();
            let d = DistortionParams::from_values(kind, &values).unwrap();
            assert_eq!((d.kind(), d.values()), (kind, values));
        }
        for kind in SensorKind::ALL {
            let values: Vec<Real> = (0..kind.dim()).map(|i| 0.02 * (i + 1) as Real).collect();
            let s = SensorParams::from_values(kind, &values).unwrap();
            assert_eq!((s.kind(), s.values()), (Some(kind), values));
        }
    }

    #[test]
    fn from_values_rejects_a_wrong_length() {
        assert!(DistortionParams::from_values(DistortionKind::BrownConrady5, &[0.1; 4]).is_err());
        assert!(SensorParams::from_values(SensorKind::Scheimpflug, &[0.1]).is_err());
        assert!(ProjectionParams::from_values(ProjectionKind::Pinhole, &[0.1]).is_err());
        let d = DistortionParams::Division { lambda: 0.1 };
        assert!(d.with_values(&[0.1, 0.2]).is_err());
    }

    #[test]
    fn with_values_keeps_the_model_settings() {
        let d = DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                iters: 20,
                ..Default::default()
            },
        };
        let out = d.with_values(&[0.1, 0.2, 0.3, 0.4, 0.5]).unwrap();
        let DistortionParams::BrownConrady5 { params } = out else {
            panic!("model changed");
        };
        assert_eq!(params.iters, 20);
        assert_eq!(params.k1, 0.1);
        assert_eq!(params.p2, 0.5);
    }

    #[test]
    fn a_general_homography_has_no_kind() {
        let s = SensorParams::Homography {
            h: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert_eq!(s.kind(), None);
        assert!(s.values().is_empty());
    }
}
