//! A run's camera parameters with their uncertainty, and its projection
//! uncertainty, from the library's [`CalibratedCameras`] view of the
//! export: every camera model's parameters come from the library's own
//! list (`optim::camera_parameters`), so no camera model needs code here.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{
    CameraParams, DistortionKind, ParameterCovariance, ProjectionKind, SensorKind, Uncertainty,
    UncertaintyUnavailable,
};
use vision_calibration::optim::{ProjectionUncertainty, camera_parameters};

/// Which stage of the camera model a parameter belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ParameterGroup {
    /// `fx`, `fy`, `cx`, `cy`.
    Intrinsics,
    /// The projection's own parameters.
    Projection,
    /// The distortion coefficients.
    Distortion,
    /// The sensor's (a Scheimpflug tilt).
    Sensor,
}

/// Why a parameter (or a camera's projection) has no σ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SigmaUnavailable {
    /// The run has no covariance, for the library's reason.
    NoCovariance { reason: UncertaintyUnavailable },
    /// The final solve held it fixed (for a projection: every parameter of
    /// the camera): exact by assumption.
    Fixed,
    /// The covariance lists it neither as estimated nor as fixed.
    NotListed,
    /// A camera model the solver does not estimate (a general sensor
    /// homography).
    UnsupportedModel { message: String },
}

/// A parameter's standard deviation, or why it has none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParameterSigma {
    /// Estimated: its σ, in the parameter's unit.
    Estimated {
        sigma: f64,
    },
    Unavailable {
        reason: SigmaUnavailable,
    },
}

/// One parameter of a camera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraParameter {
    /// Its label in the export's uncertainty (`fx`, `cam1.k1`).
    pub label: String,
    /// Its name in the camera model (`fx`, `k1`, `tilt_x`).
    pub name: String,
    pub group: ParameterGroup,
    pub value: f64,
    /// `px` (intrinsics), `rad` (sensor tilt) or empty (a coefficient).
    pub unit: String,
    pub sigma: ParameterSigma,
    /// The solve ended with it on a box bound (the reason the run has no
    /// covariance). The bounds themselves are the workflow's own and not
    /// recorded in the export.
    pub at_bound: bool,
}

/// One camera's model and parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraParameterSet {
    pub camera: usize,
    pub projection: ProjectionKind,
    pub distortion: DistortionKind,
    /// `null` for a general sensor homography.
    pub sensor: Option<SensorKind>,
    /// Its parameters in the solver's order; empty for a model the solver
    /// does not estimate (`unsupported` says why).
    pub parameters: Vec<CameraParameter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported: Option<String>,
}

/// Every camera's parameters, in camera order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CameraParameters {
    pub cameras: Vec<CameraParameterSet>,
}

/// The prefix of camera `camera`'s labels in the export's uncertainty:
/// `cam<i>.` in a multi-camera result, none in a one-camera one (the
/// library's rule).
fn label_prefix(uncertainty: &Uncertainty, cameras: usize, camera: usize) -> String {
    let prefixed = format!("cam{camera}.");
    let bare = match uncertainty.covariance() {
        Some(cov) => {
            camera == 0
                && !cov
                    .labels
                    .iter()
                    .chain(&cov.fixed)
                    .any(|l| l.starts_with(&prefixed))
        }
        None => cameras == 1,
    };
    if bare { String::new() } else { prefixed }
}

fn group(params: &CameraParams, name: &str) -> ParameterGroup {
    if params.projection.kind().param_names().contains(&name) {
        ParameterGroup::Projection
    } else if params.distortion.kind().param_names().contains(&name) {
        ParameterGroup::Distortion
    } else if params
        .sensor
        .kind()
        .is_some_and(|k| k.param_names().contains(&name))
    {
        ParameterGroup::Sensor
    } else {
        ParameterGroup::Intrinsics
    }
}

fn sigma(uncertainty: &Uncertainty, label: &str) -> ParameterSigma {
    let unavailable = |reason| ParameterSigma::Unavailable { reason };
    match uncertainty {
        Uncertainty::Available(cov) => match cov.std_dev(label) {
            Some(sigma) => ParameterSigma::Estimated { sigma },
            None if cov.fixed.iter().any(|l| l == label) => unavailable(SigmaUnavailable::Fixed),
            None => unavailable(SigmaUnavailable::NotListed),
        },
        Uncertainty::Unavailable { reason } => unavailable(SigmaUnavailable::NoCovariance {
            reason: reason.clone(),
        }),
    }
}

fn at_bound(uncertainty: &Uncertainty, label: &str) -> bool {
    matches!(
        uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::AtBound { labels }
        } if labels.iter().any(|l| l == label)
    )
}

/// Every camera's parameters with their σ.
pub fn camera_parameters_of(cameras: &dyn CalibratedCameras) -> CameraParameters {
    let all = cameras.camera_params();
    let uncertainty = cameras.parameter_uncertainty();
    let sets = all
        .iter()
        .enumerate()
        .map(|(camera, params)| {
            let prefix = label_prefix(uncertainty, all.len(), camera);
            let (parameters, unsupported) = match camera_parameters(params) {
                Ok(list) => (
                    list.into_iter()
                        .map(|(name, value)| {
                            let label = format!("{prefix}{name}");
                            let group = group(params, name);
                            CameraParameter {
                                sigma: sigma(uncertainty, &label),
                                at_bound: at_bound(uncertainty, &label),
                                label,
                                name: name.to_owned(),
                                unit: match group {
                                    ParameterGroup::Intrinsics => "px",
                                    ParameterGroup::Sensor => "rad",
                                    _ => "",
                                }
                                .to_owned(),
                                group,
                                value,
                            }
                        })
                        .collect(),
                    None,
                ),
                Err(e) => (Vec::new(), Some(e.to_string())),
            };
            CameraParameterSet {
                camera,
                projection: params.projection.kind(),
                distortion: params.distortion.kind(),
                sensor: params.sensor.kind(),
                parameters,
                unsupported,
            }
        })
        .collect();
    CameraParameters { cameras: sets }
}

/// Why camera `camera` has no projection σ, if it has none: no covariance,
/// a model the solver does not estimate, or none of its parameters
/// estimated.
pub fn projection_sigma_unavailable(
    cameras: &dyn CalibratedCameras,
    camera: usize,
) -> Option<SigmaUnavailable> {
    let all = cameras.camera_params();
    let params = all.get(camera)?;
    let cov: &ParameterCovariance = match cameras.parameter_uncertainty() {
        Uncertainty::Available(cov) => cov,
        Uncertainty::Unavailable { reason } => {
            return Some(SigmaUnavailable::NoCovariance {
                reason: reason.clone(),
            });
        }
    };
    let names = match camera_parameters(params) {
        Ok(list) => list,
        Err(e) => {
            return Some(SigmaUnavailable::UnsupportedModel {
                message: e.to_string(),
            });
        }
    };
    let prefix = label_prefix(cameras.parameter_uncertainty(), all.len(), camera);
    let estimated = names
        .iter()
        .any(|(name, _)| cov.index(&format!("{prefix}{name}")).is_some());
    (!estimated).then_some(SigmaUnavailable::Fixed)
}

/// The projection uncertainty of `camera` over an image of `image_size`.
pub fn projection_uncertainty(
    cameras: &dyn CalibratedCameras,
    camera: usize,
    image_size: [u32; 2],
) -> Result<ProjectionUncertainty, String> {
    cameras
        .projection_uncertainty(camera, image_size)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use vision_calibration::common::calibrated_cameras;

    fn identity() -> serde_json::Value {
        json!({"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.5]})
    }

    /// A planar-intrinsics export with a covariance over `cx` and `cy`,
    /// `fx` and `fy` held fixed.
    fn planar_export(uncertainty: serde_json::Value) -> serde_json::Value {
        json!({
            "kind": "planar_intrinsics",
            "schema_version": 1,
            "camera": {
                "projection": {"type": "pinhole"},
                "distortion": {"type": "brown_conrady5",
                    "k1": -0.1, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0, "iters": 8},
                "sensor": {"type": "identity"},
                "intrinsics": {
                    "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0
                }
            },
            "cam_se3_target": [identity()],
            "report": {"final_cost": 0.0, "num_iters": 1},
            "mean_reproj_error": 0.1,
            "per_cam_reproj_errors": [0.1],
            "per_feature_residuals": {"target": []},
            "uncertainty": uncertainty
        })
    }

    fn available() -> serde_json::Value {
        json!({
            "status": "available",
            "labels": ["cx", "cy", "cam_se3_target.rx"],
            "covariance": [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1e-6]],
            "fixed": ["fx", "fy"],
            "noise": [{"kind": "reprojection", "sigma": 0.1, "count": 100}],
            "num_residuals": 100,
            "dof": 90
        })
    }

    fn cameras(export: serde_json::Value) -> Box<dyn CalibratedCameras> {
        calibrated_cameras(&export.to_string()).unwrap()
    }

    #[test]
    fn every_parameter_has_a_value_and_a_sigma_or_its_reason() {
        let set = camera_parameters_of(cameras(planar_export(available())).as_ref());
        let camera = &set.cameras[0];
        assert_eq!(camera.distortion, DistortionKind::BrownConrady5);
        let row = |name: &str| camera.parameters.iter().find(|p| p.name == name).unwrap();
        assert_eq!(row("fx").value, 800.0);
        assert_eq!(row("fx").group, ParameterGroup::Intrinsics);
        assert_eq!(row("fx").unit, "px");
        assert_eq!(
            row("fx").sigma,
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::Fixed
            }
        );
        assert_eq!(row("cx").sigma, ParameterSigma::Estimated { sigma: 0.5 });
        assert_eq!(row("k1").group, ParameterGroup::Distortion);
        assert_eq!(row("k1").value, -0.1);
        assert_eq!(
            row("k1").sigma,
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::NotListed
            }
        );
        assert_eq!(
            projection_sigma_unavailable(cameras(planar_export(available())).as_ref(), 0),
            None
        );
    }

    #[test]
    fn a_run_without_covariance_says_why_and_flags_parameters_on_a_bound() {
        let export = planar_export(json!({
            "status": "unavailable",
            "reason": {"kind": "at_bound", "labels": ["cx"]}
        }));
        let set = camera_parameters_of(cameras(export.clone()).as_ref());
        let cx = set.cameras[0]
            .parameters
            .iter()
            .find(|p| p.name == "cx")
            .unwrap();
        assert!(cx.at_bound);
        assert!(matches!(
            cx.sigma,
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::NoCovariance { .. }
            }
        ));
        assert!(matches!(
            projection_sigma_unavailable(cameras(export).as_ref(), 0),
            Some(SigmaUnavailable::NoCovariance { .. })
        ));
    }

    #[test]
    fn principal_point_sigma_is_the_whole_map() {
        let cameras = cameras(planar_export(available()));
        let map = projection_uncertainty(cameras.as_ref(), 0, [640, 480]).unwrap();
        assert!((map.center - 0.5).abs() < 1e-9);
        assert!((map.max - 0.5).abs() < 1e-9);
        assert!(projection_uncertainty(cameras.as_ref(), 1, [640, 480]).is_err());
    }
}
