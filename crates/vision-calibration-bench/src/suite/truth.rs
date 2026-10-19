//! The true parameters of a synthetic entry, and a result's errors against
//! them in units of its own standard deviations.
//!
//! `truth/<id>.json` names each true value by the label the result's
//! covariance uses: scalars (`fx`, `cam1.k1`, `plane.nx`, `cam1.plane.d`)
//! under `parameters`, poses (`gripper_se3_camera`, `cam1.cam_se3_rig`)
//! under `poses`, so a parameter's error is read off by its label. A
//! label the truth does not name is not checked.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use vision_calibration_core::{Iso3, Uncertainty};

use super::significance::{
    ParameterChange, Resolver, export_uncertainty, plane_component, pose_coordinate,
    right_perturbation,
};

/// A synthetic entry's true parameters.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Truth {
    /// Scalar parameters by label: intrinsics, distortion, sensor tilt and
    /// laser-plane components.
    pub parameters: BTreeMap<String, f64>,
    /// Poses by name, as the export stores them.
    #[cfg_attr(
        feature = "schema",
        schemars(with = "BTreeMap<String, vision_calibration_core::Iso3Schema>")
    )]
    pub poses: BTreeMap<String, Iso3>,
}

impl Truth {
    /// The true normal of the laser plane of `camera`, when the truth
    /// names it.
    fn plane_normal(&self, camera: Option<usize>) -> Option<[f64; 3]> {
        let prefix = camera.map_or(String::new(), |i| format!("cam{i}."));
        let n = |c: &str| self.parameters.get(&format!("{prefix}plane.{c}")).copied();
        Some([n("nx")?, n("ny")?, n("nz")?])
    }
}

/// The error of every estimated parameter of `export` that `truth` names,
/// as a [`ParameterChange`] from the truth: `delta` is estimate minus truth
/// (for a pose, the right perturbation from the true pose; a plane signed
/// like the true one), `sigma` the export's own standard deviation. Empty
/// when the export has no covariance.
///
/// # Errors
///
/// Fails when the export's covariance does not parse, or a labelled value
/// the truth names cannot be found in the export.
pub fn truth_errors(export: &Value, truth: &Truth) -> Result<Vec<ParameterChange>> {
    let Some(uncertainty) = export_uncertainty(export) else {
        return Ok(Vec::new());
    };
    let uncertainty: Uncertainty =
        serde_json::from_value(uncertainty.clone()).context("parse the result's uncertainty")?;
    let Uncertainty::Available(cov) = uncertainty else {
        return Ok(Vec::new());
    };
    let result = Resolver::new(export).context("read the result")?;
    let mut out = Vec::new();
    for (i, label) in cov.labels.iter().enumerate() {
        let delta = if let Some(&true_value) = truth.parameters.get(label) {
            let align = plane_component(label).and_then(|(camera, _)| truth.plane_normal(camera));
            result.scalar(label, align)? - true_value
        } else if let Some((name, index)) = pose_coordinate(label)
            && let Some(true_pose) = truth.poses.get(name)
        {
            right_perturbation(true_pose, &result.pose(name, None)?)[index]
        } else {
            continue;
        };
        out.push(ParameterChange {
            label: label.clone(),
            delta,
            sigma: cov.covariance[i][i].max(0.0).sqrt(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::test_exports::rig_export;
    use vision_calibration_core::synthetic::poses::make_iso;

    #[test]
    fn errors_are_in_units_of_the_result_sigma() {
        let true_pose = make_iso((0.0, 0.0, 0.0), (0.1, 0.0, 0.0));
        let estimate = make_iso((0.0, 0.0, 0.002), (0.1005, 0.0, 0.0));
        let truth = Truth {
            parameters: BTreeMap::from([("cam1.fx".into(), 820.0)]),
            poses: BTreeMap::from([("cam1.cam_se3_rig".into(), true_pose)]),
        };
        let errors = truth_errors(&rig_export(821.0, &estimate, 0.0), &truth).unwrap();
        let z: Vec<_> = errors.iter().map(|c| (c.label.as_str(), c.z())).collect();
        assert_eq!(z[0], ("cam1.fx", 0.5));
        assert!((z[1].1 - 0.5).abs() < 1e-3, "{z:?}");
        assert!((z[2].1 - 2.0).abs() < 1e-6, "{z:?}");
    }

    #[test]
    fn labels_without_truth_are_not_checked() {
        let truth = Truth::default();
        let errors = truth_errors(&rig_export(821.0, &Iso3::identity(), 0.0), &truth).unwrap();
        assert!(errors.is_empty());
    }
}
