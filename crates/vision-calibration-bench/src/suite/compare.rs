//! Other libraries' results on the suite, against calibration-rs's.
//!
//! A tool's result for entry `id` is `results/<id>/<tool>.json`, written by
//! its harness (`tools/compare/<tool>/`): the parameters it estimated under
//! the labels calibration-rs's covariance uses, the poses under their export
//! names, the standard deviations it reports, and its mean reprojection
//! error. Each comparison measures the difference from the calibration-rs
//! reference in units of the reference's σ, sets the two libraries' σ side
//! by side, and for a synthetic entry measures both against the truth.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use vision_calibration_core::{Iso3, ParameterCovariance, Uncertainty};

use super::significance::{
    ParameterChange, export_uncertainty, pose_coordinate, right_perturbation,
};
use super::truth::{Truth, truth_errors};
use super::{ReferenceResult, Suite, SuiteEntry, load_truth};

/// A result another library produced for a suite entry.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalResult {
    /// Library and version, e.g. `opencv 4.12.0`.
    pub producer: String,
    /// The functions and flags that produced it.
    pub method: String,
    /// Scalar parameters by calibration-rs label.
    pub parameters: BTreeMap<String, f64>,
    /// Poses by calibration-rs export name.
    #[cfg_attr(
        feature = "schema",
        schemars(with = "BTreeMap<String, vision_calibration_core::Iso3Schema>")
    )]
    pub poses: BTreeMap<String, Iso3>,
    /// The library's own standard deviations, by label.
    #[serde(default)]
    pub sigma: BTreeMap<String, f64>,
    /// Mean Euclidean reprojection error, pixels, when the library's
    /// procedure yields one comparable with calibration-rs's.
    #[serde(default)]
    pub mean_reproj_error: Option<f64>,
}

impl ExternalResult {
    fn values(&self) -> Truth {
        Truth {
            parameters: self.parameters.clone(),
            poses: self.poses.clone(),
        }
    }
}

/// One entry, one external tool.
#[derive(Debug, Clone)]
pub struct Comparison {
    /// Entry id.
    pub id: String,
    /// The external result.
    pub external: ExternalResult,
    /// External value minus calibration-rs's, per label both estimate, in
    /// units of calibration-rs's σ.
    pub differences: Vec<ParameterChange>,
    /// calibration-rs's σ over the external tool's, per label both report.
    pub sigma_ratios: Vec<(String, f64)>,
    /// calibration-rs's mean reprojection error, pixels.
    pub mean_reproj_error: Option<f64>,
    /// For a synthetic entry, the largest error against the truth in units
    /// of calibration-rs's σ: `(calibration-rs, external)`.
    pub truth: Option<(f64, f64)>,
}

/// Compare every external result stored for `entry` with its reference.
///
/// # Errors
///
/// Fails on an unreadable reference, external result or truth.
pub fn compare_entry(suite: &Suite, entry: &SuiteEntry) -> Result<Vec<Comparison>> {
    let reference_path = suite.reference_path(&entry.id);
    let Some(dir) = reference_path.parent() else {
        return Ok(Vec::new());
    };
    let mut externals: Vec<_> = match std::fs::read_dir(dir) {
        Ok(read) => read
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json") && p.as_path() != reference_path)
            .collect(),
        Err(_) => return Ok(Vec::new()),
    };
    externals.sort();
    if externals.is_empty() {
        return Ok(Vec::new());
    }
    let reference: ReferenceResult = read_json(&reference_path)?;
    let export = &reference.export;
    let cov = covariance(export)?;
    let truth = match entry.synthetic {
        Some(_) => Some(load_truth(suite, &entry.id)?),
        None => None,
    };
    externals
        .iter()
        .map(|path| {
            let external: ExternalResult = read_json(path)?;
            let values = external.values();
            // `truth_errors` measures the export against a set of values:
            // reference minus external, so negate for external minus ours.
            let differences = truth_errors(export, &values)
                .with_context(|| format!("compare {}", path.display()))?
                .into_iter()
                .map(|c| ParameterChange {
                    delta: -c.delta,
                    ..c
                })
                .collect();
            let sigma_ratios = external
                .sigma
                .iter()
                .filter_map(|(label, theirs)| {
                    let ours = cov.as_ref()?.std_dev(label)?;
                    Some((label.clone(), ours / theirs))
                })
                .collect();
            let truth = match (&truth, &cov) {
                (Some(truth), Some(cov)) => Some((
                    largest(&truth_errors(export, truth)?),
                    largest(&against(&values, truth, cov)),
                )),
                _ => None,
            };
            Ok(Comparison {
                id: entry.id.clone(),
                external,
                differences,
                sigma_ratios,
                mean_reproj_error: export.get("mean_reproj_error").and_then(Value::as_f64),
                truth,
            })
        })
        .collect()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn covariance(export: &Value) -> Result<Option<ParameterCovariance>> {
    let Some(value) = export_uncertainty(export) else {
        return Ok(None);
    };
    let uncertainty: Uncertainty = serde_json::from_value(value.clone())?;
    Ok(uncertainty.covariance().cloned())
}

/// `values` against `truth` over the labels of `cov` both name, in units of
/// `cov`'s σ.
fn against(values: &Truth, truth: &Truth, cov: &ParameterCovariance) -> Vec<ParameterChange> {
    cov.labels
        .iter()
        .enumerate()
        .filter_map(|(i, label)| {
            let delta = match (values.parameters.get(label), truth.parameters.get(label)) {
                (Some(v), Some(t)) => v - t,
                _ => {
                    let (name, index) = pose_coordinate(label)?;
                    right_perturbation(truth.poses.get(name)?, values.poses.get(name)?)[index]
                }
            };
            Some(ParameterChange {
                label: label.clone(),
                delta,
                sigma: cov.covariance[i][i].max(0.0).sqrt(),
            })
        })
        .collect()
}

fn largest(changes: &[ParameterChange]) -> f64 {
    changes.iter().map(|c| c.z().abs()).fold(0.0, f64::max)
}

/// A Markdown table of the comparisons.
pub fn render_markdown(comparisons: &[Comparison]) -> String {
    let mut out = String::from(
        "| Entry | Library | Parameters | Largest difference | Median | σ ratio (ours / theirs) | Mean reprojection (ours / theirs) | Truth (ours / theirs) |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for c in comparisons {
        let z: Vec<f64> = c.differences.iter().map(|d| d.z().abs()).collect();
        let worst = c
            .differences
            .iter()
            .max_by(|a, b| a.z().abs().total_cmp(&b.z().abs()));
        let largest = worst.map_or("—".to_string(), |d| {
            format!("{:.2}σ (`{}`)", d.z().abs(), d.label)
        });
        let ratios: Vec<f64> = c.sigma_ratios.iter().map(|(_, r)| *r).collect();
        let ratio = match (
            ratios.iter().copied().reduce(f64::min),
            ratios.iter().copied().reduce(f64::max),
        ) {
            (Some(lo), Some(hi)) => format!("{lo:.3}–{hi:.3}"),
            _ => "—".to_string(),
        };
        let reproj = match (c.mean_reproj_error, c.external.mean_reproj_error) {
            (Some(ours), Some(theirs)) => format!("{ours:.4} / {theirs:.4} px"),
            _ => "—".to_string(),
        };
        let truth = c.truth.map_or("—".to_string(), |(ours, theirs)| {
            format!("{ours:.2}σ / {theirs:.2}σ")
        });
        out.push_str(&format!(
            "| `{}` | {} | {} | {largest} | {} | {ratio} | {reproj} | {truth} |\n",
            c.id,
            c.external.producer,
            z.len(),
            median(z).map_or("—".to_string(), |m| format!("{m:.2}σ")),
        ));
    }
    out
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cov(labels: &[&str], var: f64) -> ParameterCovariance {
        let n = labels.len();
        ParameterCovariance {
            labels: labels.iter().map(|l| l.to_string()).collect(),
            covariance: (0..n)
                .map(|i| (0..n).map(|j| if i == j { var } else { 0.0 }).collect())
                .collect(),
            views: Vec::new(),
            fixed: Vec::new(),
            noise: Vec::new(),
            num_residuals: 10,
            dof: 5,
        }
    }

    #[test]
    fn values_are_measured_against_the_truth_in_sigma() {
        let truth = Truth {
            parameters: BTreeMap::from([("fx".into(), 1000.0)]),
            poses: BTreeMap::from([("handeye".into(), Iso3::identity())]),
        };
        let mut moved = Iso3::identity();
        moved.translation.vector.x = 0.003;
        let values = Truth {
            parameters: BTreeMap::from([("fx".into(), 1002.0)]),
            poses: BTreeMap::from([("handeye".into(), moved)]),
        };
        let changes = against(&values, &truth, &cov(&["fx", "handeye.tx", "cx"], 1.0));
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].z(), 2.0);
        assert!((changes[1].z() - 0.003).abs() < 1e-12);
    }

    #[test]
    fn median_of_even_and_odd_lists() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(Vec::new()), None);
    }
}
