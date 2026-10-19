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
    let Some(uncertainty) = reference.get("uncertainty") else {
        return Ok(Vec::new());
    };
    let uncertainty: Uncertainty =
        serde_json::from_value(uncertainty.clone()).context("parse the reference uncertainty")?;
    let Uncertainty::Available(cov) = uncertainty else {
        return Ok(Vec::new());
    };
    let blocks = std::iter::once((None, &cov.labels, &cov.covariance)).chain(
        cov.views
            .iter()
            .map(|v| (Some(v.view), &v.labels, &v.covariance)),
    );
    let mut out = Vec::new();
    for (view, labels, matrix) in blocks {
        for (i, label) in labels.iter().enumerate() {
            let delta = delta(reference, actual, label, view)
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

fn qualified(label: &str, view: Option<usize>) -> String {
    match view {
        Some(v) => format!("view{v}.{label}"),
        None => label.to_string(),
    }
}

/// The change of one labelled coordinate.
fn delta(reference: &Value, actual: &Value, label: &str, view: Option<usize>) -> Result<f64> {
    let (camera, local) = split_camera(label);
    if let Some((name, coordinate)) = local.rsplit_once('.')
        && let Some(index) = POSE_COORDINATES.iter().position(|c| *c == coordinate)
    {
        let at = view.or(camera);
        if name == "robot_delta" {
            let component = |export: &Value| -> Result<f64> {
                number(&field(export, "robot_deltas", at)?[index])
            };
            return Ok(component(actual)? - component(reference)?);
        }
        let pose = |export: &Value| pose(field(export, name, at)?);
        return Ok(right_perturbation(&pose(reference)?, &pose(actual)?)[index]);
    }
    let scalar = |export: &Value| camera_parameter(export, camera, local);
    Ok(scalar(actual)? - scalar(reference)?)
}

const POSE_COORDINATES: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];

/// `cam1.fx` → `(Some(1), "fx")`; `fx` → `(None, "fx")`.
fn split_camera(label: &str) -> (Option<usize>, &str) {
    label
        .strip_prefix("cam")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(index, local)| Some((Some(index.parse().ok()?), local)))
        .unwrap_or((None, label))
}

/// Export field `name` (at the top level or under `params`), element
/// `index` when given.
fn field<'a>(export: &'a Value, name: &str, index: Option<usize>) -> Result<&'a Value> {
    let value = export
        .get(name)
        .or_else(|| export.get("params").and_then(|p| p.get(name)))
        .filter(|v| !v.is_null())
        .ok_or_else(|| anyhow!("no `{name}` in the export"))?;
    match index {
        None => Ok(value),
        Some(i) => value
            .get(i)
            .ok_or_else(|| anyhow!("`{name}` has no element {i}")),
    }
}

/// An intrinsics, distortion or sensor parameter of camera `index` (the
/// only camera when `None`).
fn camera_parameter(export: &Value, index: Option<usize>, name: &str) -> Result<f64> {
    let mut objects = Vec::new();
    match index {
        Some(i) => {
            objects.push(field(export, "cameras", Some(i))?);
            if let Ok(sensor) = field(export, "sensors", Some(i)) {
                objects.push(sensor);
            }
        }
        None => objects.push(field(export, "camera", None)?),
    }
    objects
        .into_iter()
        .find_map(|object| find_key(object, name))
        .map(number)
        .ok_or_else(|| anyhow!("no camera parameter `{name}` in the export"))?
}

/// The first value under key `key`, depth first.
fn find_key<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    let object = value.as_object()?;
    object
        .get(key)
        .or_else(|| object.values().find_map(|v| find_key(v, key)))
}

fn number(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| anyhow!("expected a number, found {value}"))
}

/// An exported pose `{"rotation": [qx, qy, qz, qw], "translation": [x, y, z]}`.
fn pose(value: &Value) -> Result<Iso3> {
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
fn right_perturbation(reference: &Iso3, actual: &Iso3) -> [f64; 6] {
    let rel = reference.inverse() * actual;
    let w = rel.rotation.scaled_axis();
    let v = rel.translation.vector;
    [w.x, w.y, w.z, v.x, v.y, v.z]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity() -> Value {
        json!({"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.0, 0.0, 0.0]})
    }

    fn export(fx: f64, tz: f64, delta_rx: f64) -> Value {
        json!({
            "cameras": [
                {"k": {"fx": 800.0}, "dist": {"k1": 0.1}},
                {"k": {"fx": fx}, "dist": {"k1": 0.2}}
            ],
            "sensors": null,
            "cam_se3_rig": [identity(), {"rotation": [0.0, 0.0, 0.0, 1.0], "translation": [0.1, 0.0, tz]}],
            "robot_deltas": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [delta_rx, 0.0, 0.0, 0.0, 0.0, 0.0]],
            "uncertainty": {
                "status": "available",
                "labels": ["cam1.fx", "cam1.cam_se3_rig.tz"],
                "covariance": [[4.0, 0.0], [0.0, 1e-6]],
                "views": [{"view": 1, "labels": ["robot_delta.rx"], "covariance": [[1e-6]]}],
                "fixed": [],
                "noise": [],
                "num_residuals": 100,
                "dof": 90
            }
        })
    }

    #[test]
    fn changes_are_measured_in_the_reference_sigma() {
        let reference = export(900.0, 0.0, 0.0);
        let actual = export(901.0, 0.0005, 0.002);
        let changes = parameter_changes(&reference, &actual).unwrap();
        let z: Vec<_> = changes.iter().map(|c| (c.label.as_str(), c.z())).collect();
        assert_eq!(z[0], ("cam1.fx", 0.5));
        assert_eq!(z[1].0, "cam1.cam_se3_rig.tz");
        assert!((z[1].1 - 0.5).abs() < 1e-9);
        assert_eq!(z[2].0, "view1.robot_delta.rx");
        assert!((z[2].1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn an_unresolvable_label_is_an_error() {
        let mut reference = export(900.0, 0.0, 0.0);
        reference["uncertainty"]["labels"] = json!(["cam1.no_such", "cam1.cam_se3_rig.tz"]);
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
}
