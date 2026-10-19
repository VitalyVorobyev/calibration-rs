//! A result's parameter changes in units of the reference's standard
//! deviations.
//!
//! The reference export's `uncertainty` names its estimated parameters by
//! label (`fx`, `cam1.k1`, `cam1.cam_se3_rig.tx`, per view
//! `rig_se3_target.rz`). Each label is resolved in both exports, and its
//! change is measured in the coordinates the covariance uses: the value's
//! difference for a scalar, the right perturbation `T_ref⁻¹ · T_new` for a
//! pose coordinate.

use anyhow::{Context, Result, anyhow, bail};
use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use serde_json::Value;
use vision_calibration_core::{Iso3, Uncertainty};
use vision_calibration_pipeline::common::{CalibratedCameras, calibrated_cameras};

/// One estimated parameter's change from the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterChange {
    /// The label, with `view<i>.` in front of a per-view one.
    pub label: String,
    /// New value minus reference value, in the covariance's coordinates.
    pub delta: f64,
    /// The reference's standard deviation of the parameter.
    pub sigma: f64,
}

impl ParameterChange {
    /// The change in standard deviations.
    pub fn z(&self) -> f64 {
        self.delta / self.sigma
    }
}

/// The change of every parameter the reference's covariance reports;
/// empty when the reference has no covariance.
///
/// # Errors
///
/// Fails when the reference's `uncertainty` does not parse, or a label
/// cannot be found in either export.
pub fn parameter_changes(reference: &Value, actual: &Value) -> Result<Vec<ParameterChange>> {
    let Some(uncertainty) = export_uncertainty(reference) else {
        return Ok(Vec::new());
    };
    let uncertainty: Uncertainty =
        serde_json::from_value(uncertainty.clone()).context("parse the reference uncertainty")?;
    let Uncertainty::Available(cov) = uncertainty else {
        return Ok(Vec::new());
    };
    let reference = Resolver::new(reference).context("read the reference")?;
    let actual = Resolver::new(actual).context("read the result")?;
    let blocks = std::iter::once((None, &cov.labels, &cov.covariance)).chain(
        cov.views
            .iter()
            .map(|v| (Some(v.view), &v.labels, &v.covariance)),
    );
    let mut out = Vec::new();
    for (view, labels, matrix) in blocks {
        for (i, label) in labels.iter().enumerate() {
            let delta = delta(&reference, &actual, label, view)
                .with_context(|| format!("resolve {}", qualified(label, view)))?;
            out.push(ParameterChange {
                label: qualified(label, view),
                delta,
                sigma: matrix[i][i].sqrt(),
            });
        }
    }
    Ok(out)
}

/// An export's `uncertainty`: top-level, or with the laserline device's
/// estimate.
pub(super) fn export_uncertainty(export: &Value) -> Option<&Value> {
    export
        .get("uncertainty")
        .or_else(|| export.pointer("/estimate/uncertainty"))
}

fn qualified(label: &str, view: Option<usize>) -> String {
    match view {
        Some(v) => format!("view{v}.{label}"),
        None => label.to_string(),
    }
}

/// Reads labelled values from one export.
pub(super) struct Resolver<'a> {
    export: &'a Value,
    cameras: Box<dyn CalibratedCameras>,
}

impl<'a> Resolver<'a> {
    pub(super) fn new(export: &'a Value) -> Result<Self> {
        Ok(Self {
            export,
            cameras: calibrated_cameras(export.clone())?,
        })
    }

    /// A scalar parameter by its label: a camera parameter (`fx`,
    /// `cam1.k1`), or a laser-plane component (`plane.nx`, `cam1.plane.d`)
    /// signed so its normal points along `align`.
    pub(super) fn scalar(&self, label: &str, align: Option<[f64; 3]>) -> Result<f64> {
        if let Some((camera, component)) = plane_component(label) {
            let (normal, distance) = plane(self.export, camera)?;
            let sign = match align {
                Some(n) if normal.iter().zip(n).map(|(a, b)| a * b).sum::<f64>() < 0.0 => -1.0,
                _ => 1.0,
            };
            return Ok(sign * normal.get(component).copied().unwrap_or(distance));
        }
        self.cameras
            .camera_parameter(label)
            .ok_or_else(|| anyhow!("no camera parameter `{label}` in the export"))
    }

    /// The pose `name` (`gripper_se3_camera`, `cam1.cam_se3_rig`), or view
    /// `view`'s pose `name` (`camera_se3_target`).
    pub(super) fn pose(&self, name: &str, view: Option<usize>) -> Result<Iso3> {
        let (camera, local) = split_camera(name);
        pose(field(self.export, local, view.or(camera))?)
    }

    /// The laser plane normal of `camera`, for aligning another export's.
    fn plane_normal(&self, camera: Option<usize>) -> Option<[f64; 3]> {
        plane(self.export, camera).ok().map(|(n, _)| n)
    }
}

/// The change of one labelled coordinate.
fn delta(
    reference: &Resolver<'_>,
    actual: &Resolver<'_>,
    label: &str,
    view: Option<usize>,
) -> Result<f64> {
    if let Some((name, index)) = pose_coordinate(label) {
        let (camera, local) = split_camera(name);
        if local == "robot_delta" {
            let component = |export: &Value| -> Result<f64> {
                number(&field(export, "robot_deltas", view.or(camera))?[index])
            };
            return Ok(component(actual.export)? - component(reference.export)?);
        }
        return Ok(
            right_perturbation(&reference.pose(name, view)?, &actual.pose(name, view)?)[index],
        );
    }
    let align = plane_component(label).and_then(|(camera, _)| reference.plane_normal(camera));
    Ok(actual.scalar(label, align)? - reference.scalar(label, None)?)
}

pub(super) const POSE_COORDINATES: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];

/// `cam1.cam_se3_rig.tz` → `("cam1.cam_se3_rig", 5)`.
pub(super) fn pose_coordinate(label: &str) -> Option<(&str, usize)> {
    let (name, coordinate) = label.rsplit_once('.')?;
    Some((
        name,
        POSE_COORDINATES.iter().position(|c| *c == coordinate)?,
    ))
}

/// The laser-plane component labels, `n · p + d = 0`.
const PLANE_COMPONENTS: [&str; 4] = ["nx", "ny", "nz", "d"];

/// `plane.nx` → `(None, 0)`, `cam1.plane.d` → `(Some(1), 3)`.
pub(super) fn plane_component(label: &str) -> Option<(Option<usize>, usize)> {
    let (camera, local) = split_camera(label);
    let component = local.strip_prefix("plane.")?;
    Some((
        camera,
        PLANE_COMPONENTS.iter().position(|c| *c == component)?,
    ))
}

/// `cam1.fx` → `(Some(1), "fx")`; `fx` → `(None, "fx")`.
pub(super) fn split_camera(label: &str) -> (Option<usize>, &str) {
    label
        .strip_prefix("cam")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(index, local)| Some((Some(index.parse().ok()?), local)))
        .unwrap_or((None, label))
}

/// Export field `name`, element `index` when given. Looked up at the top
/// level, under `params`, and under the laserline device's
/// `estimate.params`, whose per-view poses are named `poses`.
pub(super) fn field<'a>(export: &'a Value, name: &str, index: Option<usize>) -> Result<&'a Value> {
    let containers = [
        Some(export),
        export.get("params"),
        export.pointer("/estimate/params"),
    ];
    let alias = (name == "camera_se3_target").then_some("poses");
    let value = containers
        .into_iter()
        .flatten()
        .find_map(|c| {
            c.get(name)
                .or_else(|| alias.and_then(|a| c.get(a)))
                .filter(|v| !v.is_null())
        })
        .ok_or_else(|| anyhow!("no `{name}` in the export"))?;
    match index {
        None => Ok(value),
        Some(i) => value
            .get(i)
            .ok_or_else(|| anyhow!("`{name}` has no element {i}")),
    }
}

/// The laser plane `(normal, distance)` of `camera`: the laserline device's
/// one plane, or a rig's per-camera plane in that camera's frame.
fn plane(export: &Value, camera: Option<usize>) -> Result<([f64; 3], f64)> {
    let plane = match camera {
        None => field(export, "plane", None)?,
        Some(i) => field(export, "laser_planes_cam", Some(i))?,
    };
    let normal = plane
        .get("normal")
        .and_then(Value::as_array)
        .filter(|n| n.len() == 3)
        .ok_or_else(|| anyhow!("a laser plane needs a 3-vector `normal`"))?;
    let distance = plane
        .get("distance")
        .ok_or_else(|| anyhow!("a laser plane needs `distance`"))?;
    Ok((
        [
            number(&normal[0])?,
            number(&normal[1])?,
            number(&normal[2])?,
        ],
        number(distance)?,
    ))
}

pub(super) fn number(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| anyhow!("expected a number, found {value}"))
}

/// An exported pose `{"rotation": [qx, qy, qz, qw], "translation": [x, y, z]}`.
pub(super) fn pose(value: &Value) -> Result<Iso3> {
    let numbers = |key: &str, n: usize| -> Result<Vec<f64>> {
        let items = value
            .get(key)
            .and_then(Value::as_array)
            .filter(|a| a.len() == n)
            .ok_or_else(|| anyhow!("a pose needs `{key}` with {n} numbers"))?;
        items.iter().map(number).collect()
    };
    let q = numbers("rotation", 4)?;
    let t = numbers("translation", 3)?;
    if q.iter().all(|v| *v == 0.0) {
        bail!("a pose rotation must not be zero");
    }
    Ok(Iso3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_quaternion(Quaternion::new(q[3], q[0], q[1], q[2])),
    ))
}

/// `[ω, v]` with `actual = reference · Exp([ω, v])`, to first order.
pub(super) fn right_perturbation(reference: &Iso3, actual: &Iso3) -> [f64; 6] {
    let rel = reference.inverse() * actual;
    let w = rel.rotation.scaled_axis();
    let v = rel.translation.vector;
    [w.x, w.y, w.z, v.x, v.y, v.z]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::test_exports::rig_export;
    use serde_json::json;
    use vision_calibration_core::synthetic::poses::make_iso;

    #[test]
    fn changes_are_measured_in_the_reference_sigma() {
        let reference = rig_export(900.0, &make_iso((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)), 0.0);
        let actual = rig_export(
            901.0,
            &make_iso((0.0, 0.0, 0.002), (0.1005, 0.0, 0.0)),
            0.002,
        );
        let changes = parameter_changes(&reference, &actual).unwrap();
        let z: Vec<_> = changes.iter().map(|c| (c.label.as_str(), c.z())).collect();
        assert_eq!(z[0], ("cam1.fx", 0.5));
        assert_eq!(z[1].0, "cam1.cam_se3_rig.tx");
        assert!((z[1].1 - 0.5).abs() < 1e-3, "{z:?}");
        assert!((z[2].1 - 2.0).abs() < 1e-6, "{z:?}");
        assert_eq!(z[3].0, "view1.robot_delta.rx");
        assert!((z[3].1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn an_unresolvable_label_is_an_error() {
        let mut reference = rig_export(900.0, &Iso3::identity(), 0.0);
        reference["uncertainty"]["labels"][0] = json!("cam1.no_such");
        let err = parameter_changes(&reference, &reference).unwrap_err();
        assert!(format!("{err:#}").contains("cam1.no_such"), "{err:#}");
    }

    #[test]
    fn no_covariance_means_no_changes() {
        let reference =
            json!({"uncertainty": {"status": "unavailable", "reason": {"kind": "not_converged"}}});
        assert!(
            parameter_changes(&reference, &reference)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn laser_planes_are_read_where_each_export_keeps_them() {
        let single = json!({
            "estimate": {"params": {"plane": {"normal": [0.0, 0.0, -1.0], "distance": 0.5}}}
        });
        assert_eq!(plane(&single, None).unwrap(), ([0.0, 0.0, -1.0], 0.5));
        let rig = json!({"laser_planes_cam": [
            {"normal": [1.0, 0.0, 0.0], "distance": 0.1},
            {"normal": [0.0, 1.0, 0.0], "distance": 0.2}
        ]});
        assert_eq!(plane(&rig, Some(1)).unwrap(), ([0.0, 1.0, 0.0], 0.2));
        assert_eq!(plane_component("cam1.plane.d"), Some((Some(1), 3)));
        assert_eq!(plane_component("plane.nx"), Some((None, 0)));
    }
}
