//! Serializable records of a solver benchmark run.
//!
//! New optional fields (the solver `backend`, for example) are added with
//! `#[serde(default)]`, so reports written by older builds keep loading.

use serde::{Deserialize, Serialize};
use vision_calibration_optim::{SolveReport, SolverBackend};

use super::metrics::QualityMetrics;
use super::scenes::SceneSpec;

/// Report schema version; bumped on any incompatible change.
pub const SOLVER_SCHEMA_VERSION: u32 = 1;

/// One timed run of the init and optimize phases, milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingSample {
    /// Initialization phase (`None` when the problem only exposes a
    /// whole-pipeline entry point).
    pub init_ms: Option<f64>,
    /// Optimization phase; the whole pipeline when `init_ms` is `None`.
    pub optimize_ms: f64,
}

/// Wall-clock timing of a scene: medians over the timed repeats plus the raw
/// samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverTiming {
    /// Median init time.
    pub init_ms: Option<f64>,
    /// Median optimize time.
    pub optimize_ms: f64,
    /// Median of `init + optimize` per sample.
    pub total_ms: f64,
    /// Every timed repeat, in run order.
    pub samples: Vec<TimingSample>,
}

impl SolverTiming {
    /// Summarize timed repeats. `None` for an empty sample set.
    pub fn from_samples(samples: Vec<TimingSample>) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let init: Vec<f64> = samples.iter().filter_map(|s| s.init_ms).collect();
        let init_ms = (init.len() == samples.len()).then(|| median(init));
        let optimize_ms = median(samples.iter().map(|s| s.optimize_ms).collect());
        let total_ms = median(
            samples
                .iter()
                .map(|s| s.init_ms.unwrap_or(0.0) + s.optimize_ms)
                .collect(),
        );
        Some(Self {
            init_ms,
            optimize_ms,
            total_ms,
            samples,
        })
    }
}

/// Median of a non-empty list (mean of the middle pair for even lengths).
pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Outcome of a scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// The scene solved and was scored.
    Ok,
    /// Scene generation or the solve failed; the message says why.
    Error(String),
}

/// Everything recorded for one scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverRunRecord {
    /// The matrix cell.
    pub scene: SceneSpec,
    /// Timing, present when the scene solved.
    pub timing: Option<SolverTiming>,
    /// Backend report, when the problem exposes one.
    pub solve_report: Option<SolveReport>,
    /// Quality metrics, present when the scene solved.
    pub metrics: Option<QualityMetrics>,
    /// Outcome.
    pub status: RunStatus,
}

impl SolverRunRecord {
    /// A failed scene.
    pub fn failed(scene: SceneSpec, message: impl Into<String>) -> Self {
        Self {
            scene,
            timing: None,
            solve_report: None,
            metrics: None,
            status: RunStatus::Error(message.into()),
        }
    }

    /// The same record with every wall-clock field cleared, for exact
    /// comparison of two runs.
    pub fn without_timing(mut self) -> Self {
        self.timing = None;
        self
    }
}

/// A solver benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverBenchReport {
    /// [`SOLVER_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Solver backend every scene ran on.
    #[serde(default)]
    pub backend: SolverBackend,
    /// Git SHA the binary ran from.
    pub git_sha: String,
    /// Run start, Unix epoch seconds.
    pub timestamp_unix_secs: String,
    /// Preset name (`quick` / `full`).
    pub preset: String,
    /// Timed repeats per scene.
    pub repeats: usize,
    /// One record per scene, in preset order.
    pub records: Vec<SolverRunRecord>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_even_and_unsorted() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn timing_summarizes_samples() {
        let samples = vec![
            TimingSample {
                init_ms: Some(1.0),
                optimize_ms: 10.0,
            },
            TimingSample {
                init_ms: Some(3.0),
                optimize_ms: 30.0,
            },
            TimingSample {
                init_ms: Some(2.0),
                optimize_ms: 20.0,
            },
        ];
        let t = SolverTiming::from_samples(samples).unwrap();
        assert_eq!(
            (t.init_ms, t.optimize_ms, t.total_ms),
            (Some(2.0), 20.0, 22.0)
        );
        let whole = SolverTiming::from_samples(vec![TimingSample {
            init_ms: None,
            optimize_ms: 5.0,
        }])
        .unwrap();
        assert_eq!((whole.init_ms, whole.total_ms), (None, 5.0));
        assert!(SolverTiming::from_samples(vec![]).is_none());
    }
}
