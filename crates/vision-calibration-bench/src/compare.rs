//! Tolerance model for pass/fail judgement.
//!
//! This module defines *only* the tolerance types — how strict a comparison is.
//! The comparison logic that consumes them (record-vs-baseline diffing) arrives
//! in a later phase. Keeping tolerances separate from the metric
//! [`crate::record::BenchRecord`] is a core design tenet: one surface says what
//! a run produced, the other says how to judge it.
//!
//! A [`BTreeMap`] (not `HashMap`) backs per-dataset overrides so serialized
//! tolerances are deterministic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// An absolute/relative tolerance pair.
///
/// A value typically passes when it is within `abs` *or* within `rel` (a
/// fraction) of its baseline; the exact predicate is defined by the comparison
/// logic in a later phase.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tol {
    /// Absolute tolerance, in the metric's native units.
    pub abs: f64,
    /// Relative tolerance, as a fraction of the baseline value.
    pub rel: f64,
}

/// The full tolerance set used to judge a benchmark record against a baseline.
///
/// The named fields are the global defaults per metric class; `overrides` lets
/// individual datasets relax or tighten any class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tolerances {
    /// Tolerance on reprojection-fit metrics.
    pub reproj: Tol,
    /// Tolerance on cross-validation generalization metrics.
    pub crossval: Tol,
    /// Tolerance on parameter-stability metrics.
    pub stability: Tol,
    /// Tolerance on detection-coverage metrics.
    pub detection: Tol,
    /// Tolerance on laser-plane metrics.
    pub laser: Tol,
    /// Tolerance on delta-to-prior metrics.
    pub delta: Tol,
    /// Per-dataset overrides, keyed by dataset id (deterministic ordering).
    #[serde(default)]
    pub overrides: BTreeMap<String, PerDatasetTol>,
}

/// Per-dataset tolerance overrides.
///
/// Every field is optional; a `None` falls back to the corresponding global
/// [`Tolerances`] field.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PerDatasetTol {
    /// Override for reprojection-fit tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reproj: Option<Tol>,
    /// Override for cross-validation tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossval: Option<Tol>,
    /// Override for stability tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<Tol>,
    /// Override for detection tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detection: Option<Tol>,
    /// Override for laser tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laser: Option<Tol>,
    /// Override for delta-to-prior tolerance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<Tol>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerances_roundtrip() {
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "puzzle_130x130".to_string(),
            PerDatasetTol {
                reproj: Some(Tol {
                    abs: 0.05,
                    rel: 0.1,
                }),
                laser: Some(Tol {
                    abs: 1e-4,
                    rel: 0.2,
                }),
                ..Default::default()
            },
        );
        let tol = Tolerances {
            reproj: Tol {
                abs: 0.02,
                rel: 0.05,
            },
            crossval: Tol {
                abs: 0.05,
                rel: 0.1,
            },
            stability: Tol {
                abs: 0.01,
                rel: 0.05,
            },
            detection: Tol {
                abs: 1.0,
                rel: 0.02,
            },
            laser: Tol {
                abs: 1e-4,
                rel: 0.1,
            },
            delta: Tol {
                abs: 0.0,
                rel: 0.05,
            },
            overrides,
        };
        let json = serde_json::to_string(&tol).expect("serialize");
        let back: Tolerances = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tol, back);
    }

    #[test]
    fn per_dataset_tol_default_all_none() {
        let p = PerDatasetTol::default();
        let json = serde_json::to_string(&p).expect("serialize");
        // All-None serializes to an empty object thanks to skip_serializing_if.
        assert_eq!(json, "{}");
        let back: PerDatasetTol = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(p, back);
    }
}
