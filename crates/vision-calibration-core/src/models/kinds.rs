//! Camera-model kinds: which model fills each slot of a camera, with the
//! parameter metadata every consumer derives from (block layout, covariance
//! labels, fix-list validation).

use nalgebra::{Point2, RealField, Vector3};
use serde::{Deserialize, Serialize};

use super::{
    BrownConrady5, DistortionModel, Division, Pinhole, ProjectionModel, RationalPolynomial, Ray,
    ScheimpflugParams, SensorModel, ThinPrism,
};

/// Names of the intrinsics parameters, in block order.
pub const INTRINSICS_PARAM_NAMES: [&str; 4] = ["fx", "fy", "cx", "cy"];

/// Which projection model a camera uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ProjectionKind {
    /// Pinhole (perspective) projection; no parameters of its own.
    #[default]
    Pinhole,
}

impl ProjectionKind {
    /// Every projection model, for iterating over the supported models.
    pub const ALL: [ProjectionKind; 1] = [ProjectionKind::Pinhole];

    /// Names of the projection's own parameters, in block order (empty when
    /// the projection has none).
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            ProjectionKind::Pinhole => &[],
        }
    }

    /// Dimension of the projection parameter block (0 = no block).
    pub fn dim(self) -> usize {
        self.param_names().len()
    }

    /// Whether every ray of this projection passes through the camera
    /// centre.
    pub fn is_central(self) -> bool {
        match self {
            ProjectionKind::Pinhole => true,
        }
    }

    /// Whether a 3D line images as a straight line in normalized
    /// coordinates (true for pinhole; not for fisheye projections).
    pub fn preserves_lines(self) -> bool {
        match self {
            ProjectionKind::Pinhole => true,
        }
    }

    /// Normalized coordinates of a camera-frame point under this projection
    /// with parameters `params` (in [`param_names`](Self::param_names)
    /// order). Total, like [`ProjectionModel::project`].
    pub fn project<T: RealField>(self, _params: &[T], p_c: &Vector3<T>) -> Point2<T> {
        match self {
            ProjectionKind::Pinhole => Pinhole.project(p_c),
        }
    }

    /// The ray of normalized coordinates `n` under this projection.
    pub fn unproject<T: RealField>(self, _params: &[T], n: &Point2<T>) -> Ray<T> {
        match self {
            ProjectionKind::Pinhole => Pinhole.unproject(n),
        }
    }
}

/// Which distortion model a camera carries.
///
/// `dim() == 0` means the camera has no distortion parameter block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DistortionKind {
    /// No distortion; no parameter block.
    None,
    /// Brown-Conrady `[k1, k2, k3, p1, p2]`.
    BrownConrady5,
    /// Rational polynomial `[k1, k2, k3, k4, k5, k6, p1, p2]`.
    Rational8,
    /// Brown-Conrady + thin-prism `[k1, k2, k3, p1, p2, s1, s2, s3, s4]`.
    ThinPrism9,
    /// Fitzgibbon division model `[lambda]`.
    Division1,
}

impl DistortionKind {
    /// Every distortion model, for iterating over the supported models.
    pub const ALL: [DistortionKind; 5] = [
        DistortionKind::None,
        DistortionKind::BrownConrady5,
        DistortionKind::Rational8,
        DistortionKind::ThinPrism9,
        DistortionKind::Division1,
    ];

    /// Dimension of the distortion parameter block (0 = no block).
    pub fn dim(self) -> usize {
        self.param_names().len()
    }

    /// Names of the distortion coefficients, in block order.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            DistortionKind::None => &[],
            DistortionKind::BrownConrady5 => &["k1", "k2", "k3", "p1", "p2"],
            DistortionKind::Rational8 => &["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"],
            DistortionKind::ThinPrism9 => &["k1", "k2", "k3", "p1", "p2", "s1", "s2", "s3", "s4"],
            DistortionKind::Division1 => &["lambda"],
        }
    }

    /// Name of the leading radial (barrel) coefficient: `k1`, or `lambda` for
    /// the division model; `None` without distortion.
    pub fn leading_radial(self) -> Option<&'static str> {
        match self {
            DistortionKind::None => None,
            DistortionKind::BrownConrady5
            | DistortionKind::Rational8
            | DistortionKind::ThinPrism9 => Some("k1"),
            DistortionKind::Division1 => Some("lambda"),
        }
    }

    /// Names of the coefficients beyond the low-order radial terms (`k1`,
    /// `k2`; the division model's `lambda`), in block order: the tangential,
    /// thin-prism and higher-order radial terms, which trade off against a
    /// sensor tilt and against the extrinsics of a joint solve.
    pub fn higher_order_names(self) -> Vec<&'static str> {
        self.param_names()
            .iter()
            .copied()
            .filter(|n| !matches!(*n, "k1" | "k2" | "lambda"))
            .collect()
    }

    /// Distort normalized coordinates with coefficients `c` (in
    /// [`param_names`](Self::param_names) order).
    pub fn distort<T: RealField>(self, c: &[T], n: &Point2<T>) -> Point2<T> {
        debug_assert!(c.len() >= self.dim(), "distortion block too short");
        match self {
            DistortionKind::None => n.clone(),
            DistortionKind::BrownConrady5 => brown_conrady5(c).distort(n),
            DistortionKind::Rational8 => rational8(c).distort(n),
            DistortionKind::ThinPrism9 => thin_prism9(c).distort(n),
            DistortionKind::Division1 => division1(c).distort(n),
        }
    }

    /// Invert [`distort`](Self::distort).
    pub fn undistort<T: RealField>(self, c: &[T], n: &Point2<T>) -> Point2<T> {
        debug_assert!(c.len() >= self.dim(), "distortion block too short");
        match self {
            DistortionKind::None => n.clone(),
            DistortionKind::BrownConrady5 => brown_conrady5(c).undistort(n),
            DistortionKind::Rational8 => rational8(c).undistort(n),
            DistortionKind::ThinPrism9 => thin_prism9(c).undistort(n),
            DistortionKind::Division1 => division1(c).undistort(n),
        }
    }
}

fn brown_conrady5<T: RealField>(c: &[T]) -> BrownConrady5<T> {
    BrownConrady5 {
        k1: c[0].clone(),
        k2: c[1].clone(),
        k3: c[2].clone(),
        p1: c[3].clone(),
        p2: c[4].clone(),
        iters: 0,
    }
}

fn rational8<T: RealField>(c: &[T]) -> RationalPolynomial<T> {
    RationalPolynomial {
        k1: c[0].clone(),
        k2: c[1].clone(),
        k3: c[2].clone(),
        k4: c[3].clone(),
        k5: c[4].clone(),
        k6: c[5].clone(),
        p1: c[6].clone(),
        p2: c[7].clone(),
        iters: 0,
    }
}

fn thin_prism9<T: RealField>(c: &[T]) -> ThinPrism<T> {
    ThinPrism {
        k1: c[0].clone(),
        k2: c[1].clone(),
        k3: c[2].clone(),
        p1: c[3].clone(),
        p2: c[4].clone(),
        s1: c[5].clone(),
        s2: c[6].clone(),
        s3: c[7].clone(),
        s4: c[8].clone(),
        iters: 0,
    }
}

fn division1<T: RealField>(c: &[T]) -> Division<T> {
    Division {
        lambda: c[0].clone(),
    }
}

/// Which sensor model a camera uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SensorKind {
    /// Sensor plane perpendicular to the optical axis; no parameters.
    #[default]
    Identity,
    /// Scheimpflug tilted sensor `[tilt_x, tilt_y]` (radians).
    Scheimpflug,
}

impl SensorKind {
    /// Every sensor model, for iterating over the supported models.
    pub const ALL: [SensorKind; 2] = [SensorKind::Identity, SensorKind::Scheimpflug];

    /// Names of the sensor parameters, in block order.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            SensorKind::Identity => &[],
            SensorKind::Scheimpflug => &["tilt_x", "tilt_y"],
        }
    }

    /// Dimension of the sensor parameter block (0 = no block).
    pub fn dim(self) -> usize {
        self.param_names().len()
    }

    /// Map distorted normalized coordinates onto the sensor plane with
    /// parameters `s` (in [`param_names`](Self::param_names) order).
    pub fn to_sensor<T: RealField>(self, s: &[T], n: &Point2<T>) -> Point2<T> {
        match self {
            SensorKind::Identity => n.clone(),
            SensorKind::Scheimpflug => scheimpflug(s).normalized_to_sensor(n),
        }
    }

    /// Invert [`to_sensor`](Self::to_sensor).
    pub fn to_normalized<T: RealField>(self, s: &[T], p: &Point2<T>) -> Point2<T> {
        match self {
            SensorKind::Identity => p.clone(),
            SensorKind::Scheimpflug => scheimpflug(s).sensor_to_normalized(p),
        }
    }
}

fn scheimpflug<T: RealField>(s: &[T]) -> ScheimpflugParams<T> {
    debug_assert!(s.len() >= 2, "sensor block too short");
    ScheimpflugParams {
        tilt_x: s[0].clone(),
        tilt_y: s[1].clone(),
    }
}

/// A camera model: the kind in each slot. The values live in
/// [`CameraParams`](super::CameraParams); [`CameraParams::spec`](super::CameraParams::spec)
/// recovers the spec from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CameraModelSpec {
    /// Projection slot.
    pub projection: ProjectionKind,
    /// Distortion slot.
    pub distortion: DistortionKind,
    /// Sensor slot.
    pub sensor: SensorKind,
}

impl CameraModelSpec {
    /// A spec from its three slots.
    pub const fn new(
        projection: ProjectionKind,
        distortion: DistortionKind,
        sensor: SensorKind,
    ) -> Self {
        Self {
            projection,
            distortion,
            sensor,
        }
    }

    /// A pinhole camera with the given distortion and sensor.
    pub const fn pinhole(distortion: DistortionKind, sensor: SensorKind) -> Self {
        Self::new(ProjectionKind::Pinhole, distortion, sensor)
    }
}

#[cfg(all(test, feature = "schemars"))]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The variant names a kind's JSON schema lists (`enum` values or
    /// per-variant `const`s).
    fn schema_names<T: schemars::JsonSchema>() -> BTreeSet<String> {
        fn collect(v: &serde_json::Value, out: &mut BTreeSet<String>) {
            match v {
                serde_json::Value::Object(map) => {
                    for (key, value) in map {
                        match (key.as_str(), value) {
                            ("const", serde_json::Value::String(s)) => {
                                out.insert(s.clone());
                            }
                            ("enum", serde_json::Value::Array(items)) => {
                                out.extend(
                                    items.iter().filter_map(|i| i.as_str().map(String::from)),
                                );
                            }
                            _ => collect(value, out),
                        }
                    }
                }
                serde_json::Value::Array(items) => items.iter().for_each(|i| collect(i, out)),
                _ => {}
            }
        }
        let mut out = BTreeSet::new();
        collect(
            &serde_json::to_value(schemars::schema_for!(T)).unwrap(),
            &mut out,
        );
        out
    }

    fn all_names<T: serde::Serialize>(all: &[T]) -> BTreeSet<String> {
        all.iter()
            .map(|k| {
                serde_json::to_value(k)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect()
    }

    /// `ALL` lists every variant: the schema is derived from the enum, so a
    /// variant missing from `ALL` fails here.
    #[test]
    fn all_lists_every_kind() {
        assert_eq!(
            all_names(&ProjectionKind::ALL),
            schema_names::<ProjectionKind>()
        );
        assert_eq!(
            all_names(&DistortionKind::ALL),
            schema_names::<DistortionKind>()
        );
        assert_eq!(all_names(&SensorKind::ALL), schema_names::<SensorKind>());
        assert_eq!(
            ProjectionKind::ALL.len(),
            all_names(&ProjectionKind::ALL).len()
        );
        assert_eq!(
            DistortionKind::ALL.len(),
            all_names(&DistortionKind::ALL).len()
        );
        assert_eq!(SensorKind::ALL.len(), all_names(&SensorKind::ALL).len());
    }
}
