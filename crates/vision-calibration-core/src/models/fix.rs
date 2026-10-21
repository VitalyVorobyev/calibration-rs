//! Which camera parameters a solve holds fixed, named per slot.

use serde::{Deserialize, Serialize};

use super::{CameraModelSpec, DistortionKind, INTRINSICS_PARAM_NAMES, SensorKind};
use crate::Error;

/// Camera parameters held fixed during a solve, named per slot.
///
/// Each list names parameters of the model in that slot (`fx`, `cy`, `k3`,
/// `tilt_y`, …). `None` means the model's default
/// ([`CameraModelSpec::default_fix`]); an empty list fixes nothing in that
/// slot. Names are checked against the model by [`CameraFix::resolve`], before
/// any work.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CameraFix {
    /// Fixed intrinsics (`fx`, `fy`, `cx`, `cy`); default: none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsics: Option<Vec<String>>,
    /// Fixed projection parameters; default: none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<Vec<String>>,
    /// Fixed distortion coefficients; default: `k3` for Brown-Conrady, and
    /// everything but the low-order radial terms under a Scheimpflug sensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distortion: Option<Vec<String>>,
    /// Fixed sensor parameters (`tilt_x`, `tilt_y`); default: none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor: Option<Vec<String>>,
}

impl CameraFix {
    /// Every parameter of `spec` fixed.
    pub fn all(spec: CameraModelSpec) -> Self {
        let names = |n: &[&str]| Some(n.iter().map(|s| s.to_string()).collect());
        Self {
            intrinsics: names(&INTRINSICS_PARAM_NAMES),
            projection: names(spec.projection.param_names()),
            distortion: names(spec.distortion.param_names()),
            sensor: names(spec.sensor.param_names()),
        }
    }

    /// No parameter fixed (every list explicitly empty).
    pub fn none() -> Self {
        Self {
            intrinsics: Some(Vec::new()),
            projection: Some(Vec::new()),
            distortion: Some(Vec::new()),
            sensor: Some(Vec::new()),
        }
    }

    /// The fixed parameters as indices into each slot's block, with `None`
    /// lists replaced by the model's default.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] naming the slot and the model when a list holds
    /// a name the model does not have.
    pub fn resolve(&self, spec: CameraModelSpec) -> Result<FixedParams, Error> {
        let default = spec.default_fix();
        let slot = |names: &Option<Vec<String>>,
                    default: &Option<Vec<String>>,
                    valid: &[&str],
                    slot: &str|
         -> Result<Vec<usize>, Error> {
            let names = names.as_ref().or(default.as_ref()).map_or(&[][..], |v| v);
            let mut idx = names
                .iter()
                .map(|n| {
                    valid.iter().position(|v| v == n).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "unknown {slot} parameter `{n}` for camera model {spec:?}; \
                             expected one of {valid:?}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            idx.sort_unstable();
            idx.dedup();
            Ok(idx)
        };
        Ok(FixedParams {
            intrinsics: slot(
                &self.intrinsics,
                &default.intrinsics,
                &INTRINSICS_PARAM_NAMES,
                "intrinsics",
            )?,
            projection: slot(
                &self.projection,
                &default.projection,
                spec.projection.param_names(),
                "projection",
            )?,
            distortion: slot(
                &self.distortion,
                &default.distortion,
                spec.distortion.param_names(),
                "distortion",
            )?,
            sensor: slot(
                &self.sensor,
                &default.sensor,
                spec.sensor.param_names(),
                "sensor",
            )?,
        })
    }
}

/// Fixed camera parameters, as sorted indices into each slot's parameter
/// block (the order of the kinds' `param_names`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FixedParams {
    /// Fixed indices into `[fx, fy, cx, cy]`.
    pub intrinsics: Vec<usize>,
    /// Fixed indices into the projection block.
    pub projection: Vec<usize>,
    /// Fixed indices into the distortion block.
    pub distortion: Vec<usize>,
    /// Fixed indices into the sensor block.
    pub sensor: Vec<usize>,
}

impl CameraModelSpec {
    /// The parameters this model holds fixed unless a configuration says
    /// otherwise, every slot present.
    ///
    /// - Brown-Conrady holds `k3`: its sixth-order term overfits typical
    ///   data. The extended models are chosen for their extra terms and hold
    ///   nothing.
    /// - A Scheimpflug sensor leaves only the low-order radial terms free
    ///   (`k1`, `k2`; the division model's `lambda`): tangential and
    ///   higher-order terms trade off against the tilt.
    pub fn default_fix(&self) -> CameraFix {
        let names = self.distortion.param_names();
        let distortion: Vec<&str> = if self.sensor == SensorKind::Scheimpflug {
            names
                .iter()
                .copied()
                .filter(|n| !matches!(*n, "k1" | "k2" | "lambda"))
                .collect()
        } else if self.distortion == DistortionKind::BrownConrady5 {
            vec!["k3"]
        } else {
            Vec::new()
        };
        let owned = |v: Vec<&str>| Some(v.into_iter().map(str::to_string).collect());
        CameraFix {
            intrinsics: Some(Vec::new()),
            projection: Some(Vec::new()),
            distortion: owned(distortion),
            sensor: Some(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(d: DistortionKind, s: SensorKind) -> CameraModelSpec {
        CameraModelSpec::pinhole(d, s)
    }

    #[test]
    fn defaults_resolve_to_the_conventional_masks() {
        let bc = spec(DistortionKind::BrownConrady5, SensorKind::Identity);
        assert_eq!(
            CameraFix::default().resolve(bc).unwrap().distortion,
            vec![2]
        );
        let tilted = spec(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
        assert_eq!(
            CameraFix::default().resolve(tilted).unwrap().distortion,
            vec![2, 3, 4]
        );
        let rational = spec(DistortionKind::Rational8, SensorKind::Identity);
        assert!(
            CameraFix::default()
                .resolve(rational)
                .unwrap()
                .distortion
                .is_empty()
        );
        let tilted_rational = spec(DistortionKind::Rational8, SensorKind::Scheimpflug);
        assert_eq!(
            CameraFix::default()
                .resolve(tilted_rational)
                .unwrap()
                .distortion,
            vec![2, 3, 4, 5, 6, 7]
        );
        let prism = spec(DistortionKind::ThinPrism9, SensorKind::Scheimpflug);
        assert_eq!(
            CameraFix::default().resolve(prism).unwrap().distortion,
            vec![2, 3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn explicit_lists_replace_the_default_per_slot() {
        let bc = spec(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
        let fix = CameraFix {
            intrinsics: Some(vec!["cy".into(), "cx".into()]),
            distortion: Some(Vec::new()),
            ..CameraFix::default()
        };
        let r = fix.resolve(bc).unwrap();
        assert_eq!(r.intrinsics, vec![2, 3]);
        assert!(r.distortion.is_empty());
        assert!(r.sensor.is_empty());
    }

    #[test]
    fn unknown_names_are_rejected() {
        let bc = spec(DistortionKind::BrownConrady5, SensorKind::Identity);
        let fix = CameraFix {
            sensor: Some(vec!["tilt_x".into()]),
            ..CameraFix::default()
        };
        let err = fix.resolve(bc).unwrap_err().to_string();
        assert!(err.contains("tilt_x"), "{err}");
    }

    #[test]
    fn json_omits_default_slots() {
        let fix: CameraFix = serde_json::from_str(r#"{"distortion": ["k3", "p1"]}"#).unwrap();
        assert_eq!(fix.intrinsics, None);
        assert_eq!(fix.distortion, Some(vec!["k3".into(), "p1".into()]));
        assert!(serde_json::from_str::<CameraFix>(r#"{"skew": []}"#).is_err());
    }
}
