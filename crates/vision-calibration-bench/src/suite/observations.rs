//! The image-free observation file of a regression-suite entry.
//!
//! An [`ObservationSet`] is the detector output a calibration consumes,
//! frozen once: the problem's input exactly as `CalibrationSession::set_input`
//! takes it, plus the camera image sizes that tools without a camera model
//! of their own (OpenCV's `calibrateCamera`) need. The `input` payload is the
//! problem type's `Input`, so the file format has one definition — the one
//! the pipeline and the Python package already read.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use vision_calibration_pipeline::Error as PipelineError;
use vision_calibration_pipeline::laserline_device::{self, LaserlineDeviceProblem};
use vision_calibration_pipeline::planar_intrinsics::{self, PlanarIntrinsicsProblem};
use vision_calibration_pipeline::rig_extrinsics::{self, RigExtrinsicsProblem};
use vision_calibration_pipeline::rig_handeye::{self, RigHandeyeProblem};
use vision_calibration_pipeline::rig_handeye_laserline::{self, RigHandeyeLaserlineProblem};
use vision_calibration_pipeline::rig_laserline_device::{self, RigLaserlineDeviceProblem};
use vision_calibration_pipeline::scheimpflug_intrinsics::{self, ScheimpflugIntrinsicsProblem};
use vision_calibration_pipeline::session::{CalibrationSession, ProblemType};
use vision_calibration_pipeline::single_cam_handeye::{self, SingleCamHandeyeProblem};

/// The observation-file format version this build reads and writes.
pub const OBSERVATION_SET_VERSION: u32 = 1;

/// Image-free observations for one calibration problem.
///
/// Serialized flat: `{"version", "cameras", "problem", "input"}`, where
/// `problem` names the problem type and `input` is its input payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSet {
    /// Format version; [`OBSERVATION_SET_VERSION`] for files this build
    /// writes.
    pub version: u32,
    /// The cameras, in the order the input indexes them.
    pub cameras: Vec<ObservedCamera>,
    /// The problem type and its input.
    #[serde(flatten)]
    pub input: ProblemInput,
}

/// One camera of an [`ObservationSet`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedCamera {
    /// Camera id from the dataset manifest.
    pub id: String,
    /// Image `[width, height]` in pixels.
    pub image_size: [u32; 2],
}

/// A problem type together with its input payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "problem", content = "input", rename_all = "snake_case")]
pub enum ProblemInput {
    /// Planar intrinsics of one camera.
    PlanarIntrinsics(<PlanarIntrinsicsProblem as ProblemType>::Input),
    /// Scheimpflug intrinsics of one camera.
    ScheimpflugIntrinsics(<ScheimpflugIntrinsicsProblem as ProblemType>::Input),
    /// Hand-eye calibration of one camera.
    SingleCamHandeye(<SingleCamHandeyeProblem as ProblemType>::Input),
    /// One camera plus a laser plane.
    LaserlineDevice(<LaserlineDeviceProblem as ProblemType>::Input),
    /// Extrinsics of a camera rig.
    RigExtrinsics(<RigExtrinsicsProblem as ProblemType>::Input),
    /// Hand-eye calibration of a camera rig.
    RigHandeye(<RigHandeyeProblem as ProblemType>::Input),
    /// Per-camera laser planes on a calibrated rig.
    RigLaserlineDevice(<RigLaserlineDeviceProblem as ProblemType>::Input),
    /// Joint rig hand-eye and laser-plane calibration.
    RigHandeyeLaserline(<RigHandeyeLaserlineProblem as ProblemType>::Input),
}

impl ProblemInput {
    /// The problem's name in the file format (the `problem` tag).
    pub fn problem_name(&self) -> &'static str {
        match self {
            Self::PlanarIntrinsics(_) => "planar_intrinsics",
            Self::ScheimpflugIntrinsics(_) => "scheimpflug_intrinsics",
            Self::SingleCamHandeye(_) => "single_cam_handeye",
            Self::LaserlineDevice(_) => "laserline_device",
            Self::RigExtrinsics(_) => "rig_extrinsics",
            Self::RigHandeye(_) => "rig_handeye",
            Self::RigLaserlineDevice(_) => "rig_laserline_device",
            Self::RigHandeyeLaserline(_) => "rig_handeye_laserline",
        }
    }

    /// Run the problem's full calibration and return its export as JSON.
    ///
    /// `overrides` is merged into the problem's default config, key by key
    /// and recursively; `None` runs the default config.
    ///
    /// # Errors
    ///
    /// Fails when the overrides do not form a valid config, or when the
    /// input is rejected or the calibration fails.
    pub fn calibrate(&self, overrides: Option<&Value>) -> Result<Value> {
        let name = self.problem_name();
        match self {
            Self::PlanarIntrinsics(input) => calibrate::<PlanarIntrinsicsProblem>(
                name,
                input,
                overrides,
                planar_intrinsics::run_calibration,
            ),
            Self::ScheimpflugIntrinsics(input) => {
                calibrate::<ScheimpflugIntrinsicsProblem>(name, input, overrides, |session| {
                    scheimpflug_intrinsics::run_calibration(session, None)
                })
            }
            Self::SingleCamHandeye(input) => calibrate::<SingleCamHandeyeProblem>(
                name,
                input,
                overrides,
                single_cam_handeye::run_calibration,
            ),
            Self::LaserlineDevice(input) => {
                calibrate::<LaserlineDeviceProblem>(name, input, overrides, |session| {
                    laserline_device::run_calibration(session, None)
                })
            }
            Self::RigExtrinsics(input) => calibrate::<RigExtrinsicsProblem>(
                name,
                input,
                overrides,
                rig_extrinsics::run_calibration,
            ),
            Self::RigHandeye(input) => {
                calibrate::<RigHandeyeProblem>(name, input, overrides, rig_handeye::run_calibration)
            }
            Self::RigLaserlineDevice(input) => calibrate::<RigLaserlineDeviceProblem>(
                name,
                input,
                overrides,
                rig_laserline_device::run_calibration,
            ),
            Self::RigHandeyeLaserline(input) => calibrate::<RigHandeyeLaserlineProblem>(
                name,
                input,
                overrides,
                rig_handeye_laserline::run_calibration,
            ),
        }
    }
}

fn calibrate<P>(
    name: &str,
    input: &P::Input,
    overrides: Option<&Value>,
    run: impl FnOnce(&mut CalibrationSession<P>) -> Result<(), PipelineError>,
) -> Result<Value>
where
    P: ProblemType,
{
    let config: P::Config = match overrides {
        None => P::Config::default(),
        Some(overrides) => {
            let mut config = serde_json::to_value(P::Config::default())?;
            merge(&mut config, overrides);
            serde_json::from_value(config)
                .with_context(|| format!("invalid {name} config overrides"))?
        }
    };
    let mut session = CalibrationSession::<P>::new();
    session.set_config(config)?;
    session.set_input(input.clone())?;
    run(&mut session).with_context(|| format!("{name} calibration failed"))?;
    Ok(serde_json::to_value(session.export()?)?)
}

/// Merge `overrides` into `base`: objects merge key by key, recursively;
/// any other value replaces the base value.
fn merge(base: &mut Value, overrides: &Value) {
    match (base, overrides) {
        (Value::Object(base), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match base.get_mut(key) {
                    Some(slot) => merge(slot, value),
                    None => {
                        base.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (slot, value) => *slot = value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use vision_calibration_core::{CorrespondenceView, PlanarDataset, Pt2, Pt3, View};

    fn planar_set() -> ObservationSet {
        let obs = CorrespondenceView::new(
            vec![
                Pt3::new(0.0, 0.0, 0.0),
                Pt3::new(0.01, 0.0, 0.0),
                Pt3::new(0.0, 0.01, 0.0),
                Pt3::new(0.01, 0.01, 0.0),
            ],
            vec![
                Pt2::new(100.0, 100.0),
                Pt2::new(110.0, 100.0),
                Pt2::new(100.0, 110.0),
                Pt2::new(110.0, 110.0),
            ],
        )
        .unwrap();
        ObservationSet {
            version: OBSERVATION_SET_VERSION,
            cameras: vec![ObservedCamera {
                id: "cam0".into(),
                image_size: [640, 480],
            }],
            input: ProblemInput::PlanarIntrinsics(
                PlanarDataset::new(vec![View::without_meta(obs)]).unwrap(),
            ),
        }
    }

    #[test]
    fn serializes_flat_with_problem_tag() {
        let value = serde_json::to_value(planar_set()).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["problem"], "planar_intrinsics");
        assert_eq!(value["cameras"][0]["image_size"], json!([640, 480]));
        assert!(value["input"]["views"].is_array());
    }

    #[test]
    fn roundtrips_through_json() {
        let json = serde_json::to_string(&planar_set()).unwrap();
        let back: ObservationSet = serde_json::from_str(&json).unwrap();
        assert_eq!(back.input.problem_name(), "planar_intrinsics");
        assert_eq!(serde_json::to_string(&back).unwrap(), json);
    }

    #[test]
    fn merge_overrides_nested_keys_only() {
        let mut base =
            json!({"solver": {"max_iters": 50, "backend": "tiny_solver"}, "fix_k3": true});
        merge(&mut base, &json!({"solver": {"max_iters": 200}}));
        assert_eq!(
            base,
            json!({"solver": {"max_iters": 200, "backend": "tiny_solver"}, "fix_k3": true})
        );
    }

    #[test]
    fn unknown_override_key_is_an_error() {
        let set = planar_set();
        let err = set
            .input
            .calibrate(Some(&json!({"no_such_field": 1})))
            .unwrap_err();
        assert!(format!("{err:#}").contains("invalid planar_intrinsics config"));
    }
}
