//! Session infrastructure types.
//!
//! Defines artifact identifiers, run records, and options for session operations.

use serde::{Deserialize, Serialize};

/// Opaque artifact identifier.
///
/// Used to reference observations, initial values, and optimized results
/// stored in a [`CalibrationSession`](super::CalibrationSession).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ArtifactId(pub(crate) u64);

impl ArtifactId {
    /// Get the raw ID value (for display/debugging).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArtifactId({})", self.0)
    }
}

/// Run record identifier for audit trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RunId(pub(crate) u64);

impl RunId {
    /// Get the raw ID value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RunId({})", self.0)
    }
}

/// Classification of artifact types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactKind {
    /// Calibration observations (image points, correspondences).
    Observations,
    /// Initial parameter estimates from linear methods.
    InitialValues,
    /// Optimized results from non-linear refinement.
    OptimizedResults,
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactKind::Observations => write!(f, "Observations"),
            ArtifactKind::InitialValues => write!(f, "InitialValues"),
            ArtifactKind::OptimizedResults => write!(f, "OptimizedResults"),
        }
    }
}

/// Classification of run operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunKind {
    /// Adding observations to the session.
    AddObservations,
    /// Running linear initialization.
    Init,
    /// Running non-linear optimization.
    Optimize,
    /// Filtering observations based on residuals.
    FilterObs,
}

impl std::fmt::Display for RunKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunKind::AddObservations => write!(f, "AddObservations"),
            RunKind::Init => write!(f, "Init"),
            RunKind::Optimize => write!(f, "Optimize"),
            RunKind::FilterObs => write!(f, "FilterObs"),
        }
    }
}

/// Run record for audit trail and reproducibility.
///
/// Each operation on a session creates a run record tracking:
/// - What operation was performed
/// - Which artifacts were inputs
/// - Which artifacts were produced as outputs
/// - The options used (serialized as JSON)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique identifier for this run.
    pub id: RunId,
    /// Type of operation performed.
    pub kind: RunKind,
    /// Unix timestamp when the run started.
    pub started_at: u64,
    /// Unix timestamp when the run completed.
    pub finished_at: u64,
    /// Artifact IDs used as inputs.
    pub inputs: Vec<ArtifactId>,
    /// Artifact IDs produced as outputs.
    pub outputs: Vec<ArtifactId>,
    /// Serialized options for reproducibility.
    pub options_json: serde_json::Value,
    /// Optional notes about this run.
    pub notes: Option<String>,
}

/// Options for filtering observations based on residuals.
///
/// Used by [`CalibrationSession::run_filter_obs`](super::CalibrationSession::run_filter_obs)
/// to remove outlier observations after an optimization pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterOptions {
    /// Maximum reprojection error (pixels) for a point to be kept.
    /// Points with error exceeding this threshold are removed.
    /// If `None`, no threshold filtering is applied.
    pub max_reproj_error: Option<f64>,

    /// Minimum number of points per view after filtering.
    /// Views with fewer remaining points may be removed (see `remove_sparse_views`).
    pub min_points_per_view: usize,

    /// If true, remove entire views that have fewer than `min_points_per_view`
    /// remaining after filtering. If false, keep sparse views.
    pub remove_sparse_views: bool,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            max_reproj_error: Some(2.0), // 2 pixel threshold
            min_points_per_view: 4,      // need at least 4 points for homography
            remove_sparse_views: true,
        }
    }
}

/// Options for exporting results.
///
/// Controls what information is included in the export report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Include per-view residual statistics in the export.
    pub include_residuals: bool,

    /// Include estimated poses in the export.
    pub include_poses: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_id_display() {
        let id = ArtifactId(42);
        assert_eq!(format!("{}", id), "ArtifactId(42)");
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn run_id_display() {
        let id = RunId(7);
        assert_eq!(format!("{}", id), "RunId(7)");
        assert_eq!(id.raw(), 7);
    }

    #[test]
    fn filter_options_default() {
        let opts = FilterOptions::default();
        assert_eq!(opts.max_reproj_error, Some(2.0));
        assert_eq!(opts.min_points_per_view, 4);
        assert!(opts.remove_sparse_views);
    }

    #[test]
    fn run_record_serialization() {
        let record = RunRecord {
            id: RunId(1),
            kind: RunKind::Init,
            started_at: 1000,
            finished_at: 1005,
            inputs: vec![ArtifactId(0)],
            outputs: vec![ArtifactId(1)],
            options_json: serde_json::json!({"iterations": 10}),
            notes: Some("test run".to_string()),
        };

        let json = serde_json::to_string(&record).unwrap();
        let restored: RunRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.id, RunId(1));
        assert_eq!(restored.kind, RunKind::Init);
        assert_eq!(restored.inputs.len(), 1);
        assert_eq!(restored.outputs.len(), 1);
    }
}
