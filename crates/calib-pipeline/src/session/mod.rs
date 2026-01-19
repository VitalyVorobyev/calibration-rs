//! Session-based calibration framework with artifact management.
//!
//! This module provides a generic calibration session infrastructure that supports
//! branching workflows with artifact-based state management. Sessions store observations,
//! initialization seeds, and optimized results as artifacts, allowing multiple
//! initialization attempts and optimization paths from the same data.
//!
//! # Architecture
//!
//! The session system is built around two key abstractions:
//!
//! - [`ProblemType`]: A trait defining the interface for a calibration problem,
//!   including observation types, linear initialization, non-linear optimization,
//!   and residual-based filtering.
//! - [`CalibrationSession`]: A generic session container parameterized over a problem type,
//!   managing artifacts and providing an API for branching calibration workflows.
//!
//! # Example
//!
//! ```ignore
//! use calib_pipeline::session::{CalibrationSession, PlanarIntrinsicsProblem};
//! use calib_pipeline::session::types::FilterOptions;
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//!
//! // Add observations
//! let obs0 = session.add_observations(observations);
//!
//! // Try two different initialization strategies
//! let seed_a = session.run_init(obs0, init_opts_a)?;
//! let seed_b = session.run_init(obs0, init_opts_b)?;
//!
//! // Optimize from the better seed
//! let sol1 = session.run_optimize(obs0, seed_a, optim_opts)?;
//!
//! // Filter observations based on residuals
//! let obs1 = session.run_filter_obs(obs0, sol1, FilterOptions::default())?;
//!
//! // Re-initialize and optimize on filtered data
//! let seed_c = session.run_init(obs1, init_opts)?;
//! let sol2 = session.run_optimize(obs1, seed_c, optim_opts)?;
//!
//! // Export final results
//! let report = session.run_export(sol2, Default::default())?;
//! ```

pub mod types;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::SystemTime;

pub use types::{ArtifactId, ArtifactKind, ExportOptions, FilterOptions, RunId, RunKind, RunRecord};

/// Trait defining the interface for a calibration problem.
///
/// Each problem type (e.g., planar intrinsics, hand-eye, linescan) implements
/// this trait to provide its specific observation types, initialization logic,
/// optimization routines, and filtering capabilities.
///
/// The contract separates stages:
/// - `initialize`: Linear or closed-form seeding only (no nonlinear BA)
/// - `optimize`: Consumes observations and seeds to run nonlinear refinement
/// - `filter_observations`: Removes outliers based on optimization residuals
/// - `export`: Converts optimized results to a user-facing report
pub trait ProblemType: Sized {
    /// Type holding observations (e.g., image points, correspondences).
    type Observations: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>;

    /// Type holding initial parameter estimates from linear methods.
    type InitialValues: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>;

    /// Type holding optimized results from non-linear refinement.
    type OptimizedResults: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>;

    /// Type for the export report (user-facing, may differ from OptimizedResults).
    type ExportReport: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>;

    /// Options for initialization (linear solver configuration).
    type InitOptions: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de> + Default;

    /// Options for optimization (non-linear solver configuration).
    type OptimOptions: Clone + std::fmt::Debug + Serialize + for<'de> Deserialize<'de> + Default;

    /// Human-readable problem name (e.g., "planar_intrinsics").
    fn problem_name() -> &'static str;

    /// Run linear initialization to compute initial parameter estimates.
    fn initialize(obs: &Self::Observations, opts: &Self::InitOptions)
        -> Result<Self::InitialValues>;

    /// Run non-linear optimization to refine parameters.
    ///
    /// Implementations must consume both the observations and the initial values
    /// produced by [`initialize`](ProblemType::initialize).
    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults>;

    /// Filter observations based on optimization residuals.
    ///
    /// Returns a new observations set with outliers removed according to the
    /// filtering options (e.g., maximum reprojection error threshold).
    fn filter_observations(
        obs: &Self::Observations,
        result: &Self::OptimizedResults,
        opts: &FilterOptions,
    ) -> Result<Self::Observations>;

    /// Export optimized results to a user-facing report format.
    fn export(result: &Self::OptimizedResults, opts: &ExportOptions) -> Result<Self::ExportReport>;
}

/// Stored artifact variants (problem-specific types).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "P: ProblemType")]
pub enum Artifact<P: ProblemType> {
    /// Calibration observations.
    Observations(P::Observations),
    /// Initial parameter estimates.
    InitialValues(P::InitialValues),
    /// Optimized results.
    OptimizedResults(P::OptimizedResults),
}

impl<P: ProblemType> Artifact<P> {
    /// Get the kind of this artifact.
    pub fn kind(&self) -> ArtifactKind {
        match self {
            Artifact::Observations(_) => ArtifactKind::Observations,
            Artifact::InitialValues(_) => ArtifactKind::InitialValues,
            Artifact::OptimizedResults(_) => ArtifactKind::OptimizedResults,
        }
    }
}

/// Metadata about a calibration session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Problem type identifier.
    pub problem_type: String,
    /// Timestamp when session was created (seconds since UNIX epoch).
    pub created_at: u64,
    /// Timestamp when session was last modified (seconds since UNIX epoch).
    pub last_updated: u64,
    /// Optional user-provided description.
    pub description: Option<String>,
}

impl SessionMetadata {
    fn new(problem_type: String) -> Self {
        let now = current_timestamp();
        Self {
            problem_type,
            created_at: now,
            last_updated: now,
            description: None,
        }
    }

    fn touch(&mut self) {
        self.last_updated = current_timestamp();
    }
}

/// A generic calibration session with artifact-based state management.
///
/// Supports branching workflows: multiple initializations from the same
/// observations, multiple optimizations from different seeds, and
/// residual-based observation filtering.
///
/// # Artifact Management
///
/// All data (observations, initial values, optimized results) are stored as
/// artifacts identified by [`ArtifactId`]. Operations return artifact IDs
/// that can be used in subsequent operations, enabling branching workflows.
///
/// # Audit Trail
///
/// Each operation creates a [`RunRecord`] capturing inputs, outputs, options,
/// and timestamps for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "P: ProblemType")]
pub struct CalibrationSession<P: ProblemType> {
    /// Session metadata.
    pub metadata: SessionMetadata,

    /// Next artifact ID to assign.
    next_artifact_id: u64,
    /// Stored artifacts.
    artifacts: BTreeMap<ArtifactId, Artifact<P>>,

    /// Next run ID to assign.
    next_run_id: u64,
    /// Run history for audit trail.
    runs: Vec<RunRecord>,
}

impl<P: ProblemType> CalibrationSession<P> {
    /// Create a new empty session.
    pub fn new() -> Self {
        Self {
            metadata: SessionMetadata::new(P::problem_name().to_string()),
            next_artifact_id: 0,
            artifacts: BTreeMap::new(),
            next_run_id: 0,
            runs: Vec::new(),
        }
    }

    /// Create a new session with a description.
    pub fn new_with_description(description: String) -> Self {
        let mut session = Self::new();
        session.metadata.description = Some(description);
        session
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Core Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Add observations to the session.
    ///
    /// Returns an artifact ID for referencing these observations in subsequent operations.
    pub fn add_observations(&mut self, obs: P::Observations) -> ArtifactId {
        let started_at = current_timestamp();
        let id = self.alloc_artifact_id();
        self.artifacts.insert(id, Artifact::Observations(obs));
        self.record_run(RunKind::AddObservations, vec![], vec![id], &(), started_at);
        id
    }

    /// Run linear initialization from observations.
    ///
    /// # Arguments
    /// * `obs_id` - Artifact ID of observations to initialize from
    /// * `opts` - Initialization options
    ///
    /// # Returns
    /// Artifact ID of the initial values on success.
    ///
    /// # Errors
    /// - If `obs_id` doesn't exist or isn't an Observations artifact
    /// - If initialization fails
    pub fn run_init(&mut self, obs_id: ArtifactId, opts: P::InitOptions) -> Result<ArtifactId> {
        let started_at = current_timestamp();
        let obs = self.require_observations(obs_id)?;
        let init = P::initialize(obs, &opts)?;

        let id = self.alloc_artifact_id();
        self.artifacts.insert(id, Artifact::InitialValues(init));
        self.record_run(RunKind::Init, vec![obs_id], vec![id], &opts, started_at);
        Ok(id)
    }

    /// Run non-linear optimization from observations and initial values.
    ///
    /// # Arguments
    /// * `obs_id` - Artifact ID of observations
    /// * `init_id` - Artifact ID of initial values (from `run_init`)
    /// * `opts` - Optimization options
    ///
    /// # Returns
    /// Artifact ID of the optimized results on success.
    ///
    /// # Errors
    /// - If `obs_id` doesn't exist or isn't an Observations artifact
    /// - If `init_id` doesn't exist or isn't an InitialValues artifact
    /// - If optimization fails
    pub fn run_optimize(
        &mut self,
        obs_id: ArtifactId,
        init_id: ArtifactId,
        opts: P::OptimOptions,
    ) -> Result<ArtifactId> {
        let started_at = current_timestamp();
        let obs = self.require_observations(obs_id)?;
        let init = self.require_initial_values(init_id)?;
        let result = P::optimize(obs, init, &opts)?;

        let id = self.alloc_artifact_id();
        self.artifacts
            .insert(id, Artifact::OptimizedResults(result));
        self.record_run(
            RunKind::Optimize,
            vec![obs_id, init_id],
            vec![id],
            &opts,
            started_at,
        );
        Ok(id)
    }

    /// Filter observations based on residuals from an optimized result.
    ///
    /// Returns a new observations artifact with outliers removed according
    /// to the filter options.
    ///
    /// # Arguments
    /// * `obs_id` - Artifact ID of observations to filter
    /// * `result_id` - Artifact ID of optimized results (for residual computation)
    /// * `opts` - Filtering options (threshold, minimum points, etc.)
    ///
    /// # Returns
    /// Artifact ID of the filtered observations on success.
    ///
    /// # Errors
    /// - If `obs_id` doesn't exist or isn't an Observations artifact
    /// - If `result_id` doesn't exist or isn't an OptimizedResults artifact
    /// - If filtering produces empty observations
    pub fn run_filter_obs(
        &mut self,
        obs_id: ArtifactId,
        result_id: ArtifactId,
        opts: FilterOptions,
    ) -> Result<ArtifactId> {
        let started_at = current_timestamp();
        let obs = self.require_observations(obs_id)?;
        let result = self.require_optimized_results(result_id)?;
        let filtered = P::filter_observations(obs, result, &opts)?;

        let id = self.alloc_artifact_id();
        self.artifacts.insert(id, Artifact::Observations(filtered));
        self.record_run(
            RunKind::FilterObs,
            vec![obs_id, result_id],
            vec![id],
            &opts,
            started_at,
        );
        Ok(id)
    }

    /// Export results without storing an artifact.
    ///
    /// Returns the export report struct directly for the caller to use.
    ///
    /// # Arguments
    /// * `result_id` - Artifact ID of optimized results to export
    /// * `opts` - Export options (what to include in report)
    ///
    /// # Returns
    /// The export report on success.
    ///
    /// # Errors
    /// - If `result_id` doesn't exist or isn't an OptimizedResults artifact
    pub fn run_export(
        &self,
        result_id: ArtifactId,
        opts: ExportOptions,
    ) -> Result<P::ExportReport> {
        let result = self.require_optimized_results(result_id)?;
        P::export(result, &opts)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Artifact Access
    // ─────────────────────────────────────────────────────────────────────────

    /// Get an artifact by ID.
    pub fn get_artifact(&self, id: ArtifactId) -> Option<&Artifact<P>> {
        self.artifacts.get(&id)
    }

    /// Get observations by ID, returning `None` if not found or wrong type.
    pub fn get_observations(&self, id: ArtifactId) -> Option<&P::Observations> {
        match self.artifacts.get(&id) {
            Some(Artifact::Observations(obs)) => Some(obs),
            _ => None,
        }
    }

    /// Get initial values by ID, returning `None` if not found or wrong type.
    pub fn get_initial_values(&self, id: ArtifactId) -> Option<&P::InitialValues> {
        match self.artifacts.get(&id) {
            Some(Artifact::InitialValues(init)) => Some(init),
            _ => None,
        }
    }

    /// Get optimized results by ID, returning `None` if not found or wrong type.
    pub fn get_optimized_results(&self, id: ArtifactId) -> Option<&P::OptimizedResults> {
        match self.artifacts.get(&id) {
            Some(Artifact::OptimizedResults(result)) => Some(result),
            _ => None,
        }
    }

    /// List all artifact IDs of a given kind.
    pub fn list_artifacts(&self, kind: ArtifactKind) -> Vec<ArtifactId> {
        self.artifacts
            .iter()
            .filter(|(_, a)| a.kind() == kind)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get the total number of artifacts in the session.
    pub fn artifact_count(&self) -> usize {
        self.artifacts.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Audit Trail
    // ─────────────────────────────────────────────────────────────────────────

    /// Get all run records.
    pub fn runs(&self) -> &[RunRecord] {
        &self.runs
    }

    /// Get runs that produced a specific artifact.
    pub fn runs_producing(&self, artifact_id: ArtifactId) -> Vec<&RunRecord> {
        self.runs
            .iter()
            .filter(|r| r.outputs.contains(&artifact_id))
            .collect()
    }

    /// Get runs that used a specific artifact as input.
    pub fn runs_consuming(&self, artifact_id: ArtifactId) -> Vec<&RunRecord> {
        self.runs
            .iter()
            .filter(|r| r.inputs.contains(&artifact_id))
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Serialization
    // ─────────────────────────────────────────────────────────────────────────

    /// Serialize session to JSON string for checkpointing.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Deserialize session from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal Helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn alloc_artifact_id(&mut self) -> ArtifactId {
        let id = ArtifactId(self.next_artifact_id);
        self.next_artifact_id += 1;
        id
    }

    fn require_observations(&self, id: ArtifactId) -> Result<&P::Observations> {
        match self.artifacts.get(&id) {
            Some(Artifact::Observations(obs)) => Ok(obs),
            Some(other) => bail!(
                "artifact {} is {}, expected Observations",
                id,
                other.kind()
            ),
            None => bail!("artifact {} not found", id),
        }
    }

    fn require_initial_values(&self, id: ArtifactId) -> Result<&P::InitialValues> {
        match self.artifacts.get(&id) {
            Some(Artifact::InitialValues(init)) => Ok(init),
            Some(other) => bail!(
                "artifact {} is {}, expected InitialValues",
                id,
                other.kind()
            ),
            None => bail!("artifact {} not found", id),
        }
    }

    fn require_optimized_results(&self, id: ArtifactId) -> Result<&P::OptimizedResults> {
        match self.artifacts.get(&id) {
            Some(Artifact::OptimizedResults(result)) => Ok(result),
            Some(other) => bail!(
                "artifact {} is {}, expected OptimizedResults",
                id,
                other.kind()
            ),
            None => bail!("artifact {} not found", id),
        }
    }

    fn record_run(
        &mut self,
        kind: RunKind,
        inputs: Vec<ArtifactId>,
        outputs: Vec<ArtifactId>,
        options: &impl Serialize,
        started_at: u64,
    ) {
        let run = RunRecord {
            id: RunId(self.next_run_id),
            kind,
            started_at,
            finished_at: current_timestamp(),
            inputs,
            outputs,
            options_json: serde_json::to_value(options).unwrap_or_default(),
            notes: None,
        };
        self.next_run_id += 1;
        self.runs.push(run);
        self.metadata.touch();
    }
}

impl<P: ProblemType> Default for CalibrationSession<P> {
    fn default() -> Self {
        Self::new()
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock problem type for testing
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockObservations {
        data: Vec<f64>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockInitial {
        value: f64,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockResult {
        optimized_value: f64,
        residuals: Vec<f64>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockReport {
        final_value: f64,
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    struct MockInitOptions {
        scale: f64,
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    struct MockOptimOptions {
        iterations: usize,
    }

    struct MockProblem;

    impl ProblemType for MockProblem {
        type Observations = MockObservations;
        type InitialValues = MockInitial;
        type OptimizedResults = MockResult;
        type ExportReport = MockReport;
        type InitOptions = MockInitOptions;
        type OptimOptions = MockOptimOptions;

        fn problem_name() -> &'static str {
            "mock_problem"
        }

        fn initialize(
            obs: &Self::Observations,
            opts: &Self::InitOptions,
        ) -> Result<Self::InitialValues> {
            let scale = if opts.scale == 0.0 { 1.0 } else { opts.scale };
            Ok(MockInitial {
                value: obs.data.iter().sum::<f64>() * scale,
            })
        }

        fn optimize(
            obs: &Self::Observations,
            init: &Self::InitialValues,
            _opts: &Self::OptimOptions,
        ) -> Result<Self::OptimizedResults> {
            // Residuals are difference from init value
            let residuals: Vec<f64> = obs.data.iter().map(|d| (d - init.value).abs()).collect();
            Ok(MockResult {
                optimized_value: init.value * 2.0,
                residuals,
            })
        }

        fn filter_observations(
            obs: &Self::Observations,
            result: &Self::OptimizedResults,
            opts: &FilterOptions,
        ) -> Result<Self::Observations> {
            let threshold = opts.max_reproj_error.unwrap_or(f64::INFINITY);
            let filtered: Vec<f64> = obs
                .data
                .iter()
                .zip(&result.residuals)
                .filter(|(_, r)| **r <= threshold)
                .map(|(d, _)| *d)
                .collect();

            if filtered.is_empty() {
                bail!("filtering removed all observations");
            }

            Ok(MockObservations { data: filtered })
        }

        fn export(
            result: &Self::OptimizedResults,
            _opts: &ExportOptions,
        ) -> Result<Self::ExportReport> {
            Ok(MockReport {
                final_value: result.optimized_value,
            })
        }
    }

    #[test]
    fn add_observations_returns_valid_id() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let id = session.add_observations(obs);
        assert_eq!(id.raw(), 0);
        assert!(session.get_observations(id).is_some());
    }

    #[test]
    fn run_init_creates_initial_values_artifact() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);

        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();
        assert_eq!(init_id.raw(), 1);

        let init = session.get_initial_values(init_id).unwrap();
        assert!((init.value - 6.0).abs() < 1e-12);
    }

    #[test]
    fn run_init_fails_on_wrong_artifact_type() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);
        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();

        // Try to init from an InitialValues artifact (should fail)
        let result = session.run_init(init_id, MockInitOptions::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("InitialValues"));
    }

    #[test]
    fn run_init_fails_on_nonexistent_artifact() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let result = session.run_init(ArtifactId(999), MockInitOptions::default());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn run_optimize_creates_optimized_results() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);
        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();

        let result_id = session
            .run_optimize(obs_id, init_id, MockOptimOptions::default())
            .unwrap();
        assert_eq!(result_id.raw(), 2);

        let result = session.get_optimized_results(result_id).unwrap();
        assert!((result.optimized_value - 12.0).abs() < 1e-12);
    }

    #[test]
    fn branching_multiple_inits_from_same_obs() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);

        // Two different init strategies
        let init_a = session
            .run_init(obs_id, MockInitOptions { scale: 1.0 })
            .unwrap();
        let init_b = session
            .run_init(obs_id, MockInitOptions { scale: 2.0 })
            .unwrap();

        let val_a = session.get_initial_values(init_a).unwrap().value;
        let val_b = session.get_initial_values(init_b).unwrap().value;

        assert!((val_a - 6.0).abs() < 1e-12);
        assert!((val_b - 12.0).abs() < 1e-12);
    }

    #[test]
    fn branching_multiple_optimizes_from_same_init() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);
        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();

        // Two optimizations from same init
        let result_a = session
            .run_optimize(obs_id, init_id, MockOptimOptions { iterations: 10 })
            .unwrap();
        let result_b = session
            .run_optimize(obs_id, init_id, MockOptimOptions { iterations: 20 })
            .unwrap();

        assert_ne!(result_a, result_b);
        assert!(session.get_optimized_results(result_a).is_some());
        assert!(session.get_optimized_results(result_b).is_some());
    }

    #[test]
    fn filter_obs_produces_new_artifact() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 100.0],
        }; // 100.0 is outlier
        let obs_id = session.add_observations(obs);
        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();
        let result_id = session
            .run_optimize(obs_id, init_id, MockOptimOptions::default())
            .unwrap();

        let filtered_id = session
            .run_filter_obs(
                obs_id,
                result_id,
                FilterOptions {
                    max_reproj_error: Some(10.0),
                    ..Default::default()
                },
            )
            .unwrap();

        let filtered = session.get_observations(filtered_id).unwrap();
        assert!(filtered.data.len() < 3); // Some points were filtered
    }

    #[test]
    fn run_export_returns_report() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);
        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();
        let result_id = session
            .run_optimize(obs_id, init_id, MockOptimOptions::default())
            .unwrap();

        let report = session.run_export(result_id, ExportOptions::default()).unwrap();
        assert!((report.final_value - 12.0).abs() < 1e-12);
    }

    #[test]
    fn session_json_roundtrip() {
        let mut session =
            CalibrationSession::<MockProblem>::new_with_description("Test session".to_string());
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);
        session.run_init(obs_id, MockInitOptions::default()).unwrap();

        let json = session.to_json().unwrap();
        let restored: CalibrationSession<MockProblem> =
            CalibrationSession::from_json(&json).unwrap();

        assert_eq!(restored.artifact_count(), 2);
        assert_eq!(restored.runs().len(), 2);
        assert_eq!(
            restored.metadata.description.as_ref().unwrap(),
            "Test session"
        );
    }

    #[test]
    fn list_artifacts_filters_by_kind() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs.clone());
        session.add_observations(obs);
        session.run_init(obs_id, MockInitOptions::default()).unwrap();

        let obs_list = session.list_artifacts(ArtifactKind::Observations);
        let init_list = session.list_artifacts(ArtifactKind::InitialValues);

        assert_eq!(obs_list.len(), 2);
        assert_eq!(init_list.len(), 1);
    }

    #[test]
    fn run_records_track_inputs_outputs() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        let obs_id = session.add_observations(obs);
        let init_id = session.run_init(obs_id, MockInitOptions::default()).unwrap();

        // Find run that produced init_id
        let producing_runs = session.runs_producing(init_id);
        assert_eq!(producing_runs.len(), 1);
        assert_eq!(producing_runs[0].kind, RunKind::Init);
        assert_eq!(producing_runs[0].inputs, vec![obs_id]);
        assert_eq!(producing_runs[0].outputs, vec![init_id]);
    }

    #[test]
    fn metadata_timestamps_update() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let created = session.metadata.created_at;

        // Small delay to ensure timestamp changes (in real usage)
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        session.add_observations(obs);

        // last_updated should be >= created
        assert!(session.metadata.last_updated >= created);
    }

    #[test]
    fn full_branching_workflow() {
        // Simulate the target API workflow
        let mut s = CalibrationSession::<MockProblem>::new();

        let obs0 = s.add_observations(MockObservations {
            data: vec![1.0, 2.0, 3.0, 100.0],
        });

        // Try two different seeds
        let seed_a = s.run_init(obs0, MockInitOptions { scale: 1.0 }).unwrap();
        let seed_b = s.run_init(obs0, MockInitOptions { scale: 0.5 }).unwrap();

        // Optimize from seed_a
        let sol1 = s
            .run_optimize(obs0, seed_a, MockOptimOptions::default())
            .unwrap();

        // Filter obs based on sol1 residuals
        let obs1 = s
            .run_filter_obs(
                obs0,
                sol1,
                FilterOptions {
                    max_reproj_error: Some(50.0),
                    ..Default::default()
                },
            )
            .unwrap();

        // Re-init and re-optimize on filtered data
        let seed_c = s.run_init(obs1, MockInitOptions { scale: 1.0 }).unwrap();
        let sol2 = s
            .run_optimize(obs1, seed_c, MockOptimOptions::default())
            .unwrap();

        // Export
        let report = s.run_export(sol2, ExportOptions::default()).unwrap();

        // Verify workflow completed
        assert!(report.final_value > 0.0);
        assert_eq!(s.artifact_count(), 7); // obs0, seed_a, seed_b, sol1, obs1, seed_c, sol2
        assert_eq!(s.runs().len(), 7);
    }
}
