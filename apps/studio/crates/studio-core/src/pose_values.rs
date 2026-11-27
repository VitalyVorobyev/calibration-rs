//! The poses and laser planes a run estimates besides the cameras' own
//! parameters, with their values and their σ from the export's
//! uncertainty: each camera's pose in a rig of several, the hand-eye, each
//! camera's laser plane. Values and labels come from the library's
//! [`CalibratedCameras`] view of the export.
//!
//! A pose's covariance is over its right perturbation `pose·Exp(δ)`, `δ =
//! [rx, ry, rz, tx, ty, tz]`, not over the components of its translation or
//! rotation vector, so a pose's σ is given per `δ` component; a plane's is
//! over its normal's components and its distance, its values' own.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{Iso3, RobotFrame, Uncertainty};

use crate::uncertainty::{ParameterSigma, SigmaUnavailable, label_prefix, sigma};

/// A rigid pose the run estimates: a camera's pose in the rig, or the
/// hand-eye.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PoseParameter {
    /// Its name, `<to>_se3_<from>` (`cam1.cam_se3_rig`, `gripper_se3_rig`,
    /// `rig_se3_base`): its labels in the export's uncertainty are
    /// `<label>.rx … <label>.tz`.
    pub label: String,
    /// The camera whose pose in the rig it is; `null` for the hand-eye.
    pub camera: Option<usize>,
    /// The translation, metres: the `from` frame's origin in the `to`
    /// frame.
    pub translation_m: [f64; 3],
    /// The rotation as an axis-angle vector, degrees: along the axis, as
    /// long as the angle.
    pub rotation_deg: [f64; 3],
    /// The σ of `[rx, ry, rz, tx, ty, tz]`, the right perturbation
    /// `pose·Exp(δ)` the covariance is over: small rotations about the
    /// `from` frame's axes in degrees, then small translations along them
    /// in metres. Not the σ of `rotation_deg`'s or `translation_m`'s
    /// components, which the covariance does not give.
    pub sigma: Vec<ParameterSigma>,
}

/// A camera's laser plane, in the camera's frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlaneParameter {
    /// Its name (`plane`, `cam1.plane`): its labels in the export's
    /// uncertainty are `<label>.nx`, `.ny`, `.nz` and `.d`.
    pub label: String,
    pub camera: usize,
    /// The unit normal.
    pub normal: [f64; 3],
    /// The signed distance from the camera centre, metres.
    pub distance_m: f64,
    /// The σ of `nx`, `ny`, `nz` (unitless; the covariance of a unit normal
    /// has rank 2) and of the distance (metres).
    pub sigma: Vec<ParameterSigma>,
}

/// The pose components, in the covariance's order.
const POSE_COMPONENTS: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];
const PLANE_COMPONENTS: [&str; 4] = ["nx", "ny", "nz", "d"];

/// The σ of the coordinate `label` of the parameter block `block`:
/// [`sigma`], and held fixed when the covariance lists the whole block as
/// fixed, as it lists a pose (`gripper_se3_cam`) or a plane's normal
/// (`plane.normal`).
fn component_sigma(uncertainty: &Uncertainty, block: &str, label: &str) -> ParameterSigma {
    let found = sigma(uncertainty, label);
    let block_fixed = uncertainty
        .covariance()
        .is_some_and(|cov| cov.fixed.iter().any(|l| l == block));
    match found {
        ParameterSigma::Unavailable {
            reason: SigmaUnavailable::NotListed,
        } if block_fixed => ParameterSigma::Unavailable {
            reason: SigmaUnavailable::Fixed,
        },
        found => found,
    }
}

/// `sigma` times `scale` (radians to degrees), when there is one.
fn scaled(sigma: ParameterSigma, scale: f64) -> ParameterSigma {
    match sigma {
        ParameterSigma::Estimated { sigma } => ParameterSigma::Estimated {
            sigma: sigma * scale,
        },
        other => other,
    }
}

fn pose(
    uncertainty: &Uncertainty,
    label: String,
    camera: Option<usize>,
    value: &Iso3,
) -> PoseParameter {
    let t = value.translation.vector;
    let r = value.rotation.scaled_axis().map(f64::to_degrees);
    let sigma = POSE_COMPONENTS
        .iter()
        .map(|c| {
            let s = component_sigma(uncertainty, &label, &format!("{label}.{c}"));
            if c.starts_with('r') {
                scaled(s, 1f64.to_degrees())
            } else {
                s
            }
        })
        .collect();
    PoseParameter {
        translation_m: [t.x, t.y, t.z],
        rotation_deg: [r.x, r.y, r.z],
        sigma,
        label,
        camera,
    }
}

/// The name of the hand-eye's pose, as the export's uncertainty labels it:
/// `gripper_se3_rig` eye-in-hand, `rig_se3_base` eye-to-hand, the camera in
/// place of the rig for one camera.
fn handeye_label(mount: RobotFrame, cameras: usize) -> String {
    let rig = if cameras > 1 { "rig" } else { "cam" };
    match mount {
        RobotFrame::Gripper => format!("gripper_se3_{rig}"),
        RobotFrame::Base => format!("{rig}_se3_base"),
    }
}

/// Each camera's pose in a rig of several cameras, then the hand-eye.
pub fn pose_parameters_of(cameras: &dyn CalibratedCameras) -> Vec<PoseParameter> {
    let uncertainty = cameras.parameter_uncertainty();
    let poses = cameras.cam_se3_rig();
    let count = poses.len();
    let mut out: Vec<PoseParameter> = if count > 1 {
        poses
            .iter()
            .enumerate()
            .map(|(i, value)| pose(uncertainty, format!("cam{i}.cam_se3_rig"), Some(i), value))
            .collect()
    } else {
        Vec::new()
    };
    if let Some(handeye) = cameras.handeye() {
        let label = handeye_label(handeye.mount_frame(), count);
        out.push(pose(uncertainty, label, None, handeye.pose()));
    }
    out
}

/// Each camera's laser plane, for a laser run.
pub fn plane_parameters_of(cameras: &dyn CalibratedCameras) -> Vec<PlaneParameter> {
    let uncertainty = cameras.parameter_uncertainty();
    let planes = cameras.laser_planes_cam();
    let count = cameras.camera_params().len();
    planes
        .iter()
        .enumerate()
        .map(|(camera, plane)| {
            let label = format!("{}plane", label_prefix(uncertainty, count, camera));
            let n = plane.normal();
            PlaneParameter {
                normal: [n.x, n.y, n.z],
                distance_m: plane.distance(),
                sigma: PLANE_COMPONENTS
                    .iter()
                    .map(|c| {
                        // The normal is one block, the distance another.
                        let block = match *c {
                            "d" => format!("{label}.d"),
                            _ => format!("{label}.normal"),
                        };
                        component_sigma(uncertainty, &block, &format!("{label}.{c}"))
                    })
                    .collect(),
                label,
                camera,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use vision_calibration::common::calibrated_cameras;

    /// The library's export shape fixture of a rig hand-eye laser
    /// calibration (`rig`) or a one-camera hand-eye (`single`), with
    /// `uncertainty`. The fixture pins key paths, not values (its
    /// histograms do not parse): its per-camera statistics, robot
    /// refinement and diagnostics, which no pose or plane reads, are left
    /// out.
    fn export(fixture: &str, uncertainty: Value) -> Box<dyn CalibratedCameras> {
        let text = match fixture {
            "rig" => include_str!(
                "../../../../../crates/vision-calibration-bench/tests/golden/rig_handeye_laserline.export.json"
            ),
            _ => include_str!(
                "../../../../../crates/vision-calibration-bench/tests/golden/single_cam_handeye.export.json"
            ),
        };
        let mut export: Value = serde_json::from_str(text).unwrap();
        export["uncertainty"] = uncertainty;
        let object = export.as_object_mut().unwrap();
        object.remove("robot_refinement");
        if object.contains_key("per_cam_stats") {
            object.insert("per_cam_stats".into(), json!([]));
        }
        object.insert(
            "diagnostics".into(),
            json!({"per_feature_residuals": {"target": []}}),
        );
        calibrated_cameras(&export.to_string()).unwrap()
    }

    fn not_computed() -> Value {
        json!({"status": "unavailable", "reason": {"kind": "not_computed"}})
    }

    fn covariance(labels: &[&str], fixed: &[&str]) -> Value {
        let n = labels.len();
        let diagonal: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 0.0004 } else { 0.0 }).collect())
            .collect();
        json!({
            "status": "available",
            "labels": labels,
            "covariance": diagonal,
            "fixed": fixed,
            "noise": [{"kind": "reprojection", "sigma": 0.1, "count": 100}],
            "num_residuals": 100,
            "dof": 90
        })
    }

    #[test]
    fn poses_and_planes_carry_their_values_and_the_sigma_of_their_components() {
        let pose_labels = |name: &str| POSE_COMPONENTS.map(|c| format!("{name}.{c}"));
        let handeye = export("rig", not_computed())
            .calibrated_rig()
            .unwrap()
            .handeye_labels()
            .unwrap();
        // The library's name for the hand-eye is the one listed here.
        let name = handeye[0].strip_suffix(".rx").unwrap().to_owned();
        let mut labels: Vec<String> = pose_labels("cam1.cam_se3_rig").to_vec();
        labels.extend(handeye.iter().cloned());
        labels.extend(PLANE_COMPONENTS.map(|c| format!("cam1.plane.{c}")));
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let cameras = export(
            "rig",
            covariance(&labels, &["cam0.cam_se3_rig", "cam0.plane.normal"]),
        );

        let poses = pose_parameters_of(cameras.as_ref());
        let names: Vec<&str> = poses.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(
            names,
            ["cam0.cam_se3_rig", "cam1.cam_se3_rig", name.as_str()]
        );
        assert_eq!(
            poses[0].sigma[0],
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::Fixed
            }
        );
        let cam1 = &poses[1];
        assert_eq!(cam1.camera, Some(1));
        let value = cameras.cam_se3_rig()[1];
        assert_eq!(cam1.translation_m[0], value.translation.vector.x);
        let angle = value.rotation.angle().to_degrees();
        let length = cam1.rotation_deg.iter().map(|r| r * r).sum::<f64>().sqrt();
        assert!((length - angle).abs() < 1e-12);
        // σ 0.02: rotations in degrees, translations in metres.
        let estimated = |s: &ParameterSigma| match s {
            ParameterSigma::Estimated { sigma } => *sigma,
            other => panic!("{other:?}"),
        };
        assert!((estimated(&cam1.sigma[0]) - 0.02f64.to_degrees()).abs() < 1e-12);
        assert!((estimated(&cam1.sigma[5]) - 0.02).abs() < 1e-15);
        assert_eq!(poses[2].camera, None);
        assert_eq!(
            poses[2].translation_m[2],
            cameras.handeye().unwrap().pose().translation.vector.z
        );

        let planes = plane_parameters_of(cameras.as_ref());
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[1].label, "cam1.plane");
        assert_eq!(
            planes[1].distance_m,
            cameras.laser_planes_cam()[1].distance()
        );
        assert!((estimated(&planes[1].sigma[3]) - 0.02).abs() < 1e-15);
        // Camera 0's plane normal was held; its distance is not listed.
        assert_eq!(
            planes[0].sigma[0],
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::Fixed
            }
        );
        assert_eq!(
            planes[0].sigma[3],
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::NotListed
            }
        );
    }

    #[test]
    fn one_camera_has_its_hand_eye_alone_and_no_covariance_says_why() {
        let cameras = export("single", not_computed());
        let poses = pose_parameters_of(cameras.as_ref());
        let names: Vec<&str> = poses.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(names, ["gripper_se3_cam"]);
        assert!(poses[0].sigma.iter().all(|s| matches!(
            s,
            ParameterSigma::Unavailable {
                reason: SigmaUnavailable::NoCovariance { .. }
            }
        )));
        assert!(plane_parameters_of(cameras.as_ref()).is_empty());
        assert_eq!(handeye_label(RobotFrame::Base, 1), "cam_se3_base");
        assert_eq!(handeye_label(RobotFrame::Base, 2), "rig_se3_base");
    }
}
