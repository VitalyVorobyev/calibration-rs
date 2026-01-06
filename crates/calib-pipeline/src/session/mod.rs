//! Session-based calibration framework with state management.
//!
//! This module provides a generic calibration session infrastructure that tracks
//! progress through pipeline stages (Uninitialized → Initialized → Optimized → Exported)
//! and supports checkpointing via JSON serialization.
//!
//! # Architecture
//!
//! The session system is built around two key abstractions:
//!
//! - [`ProblemType`]: A trait that defines the interface for a calibration problem,
//!   including observation types, initialization, and optimization methods.
//! - [`CalibrationSession`]: A generic session container parameterized over a problem type,
//!   managing state transitions and providing a fluent API for running calibration pipelines.
//!
//! # Example
//!
//! ```ignore
//! use calib_pipeline::session::{CalibrationSession, PlanarIntrinsicsProblem};
//!
//! // Create a new session
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//!
//! // Add observations
//! session.add_observations(observations)?;
//!
//! // Run initialization
//! session.initialize(init_options)?;
//!
//! // Run optimization
//! session.optimize(optim_options)?;
//!
//! // Export results
//! let report = session.export()?;
//! ```

pub mod problem_types;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Trait defining the interface for a calibration problem.
///
/// Each problem type (e.g., planar intrinsics, hand-eye, linescan) implements
/// this trait to provide its specific observation types, initialization logic,
/// and optimization routines.
pub trait ProblemType: Sized {
    /// Type holding observations (e.g., image points, poses).
    type Observations: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Type holding initial parameter estimates from linear methods.
    type InitialValues: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Type holding optimized results from non-linear refinement.
    type OptimizedResults: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Options for initialization (linear solver configuration).
    type InitOptions: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Options for optimization (non-linear solver configuration).
    type OptimOptions: Clone + Serialize + for<'de> Deserialize<'de>;

    /// Human-readable problem name (e.g., "planar_intrinsics").
    fn problem_name() -> &'static str;

    /// Run linear initialization to compute initial parameter estimates.
    fn initialize(
        obs: &Self::Observations,
        opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues>;

    /// Run non-linear optimization to refine parameters.
    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults>;
}

/// Current stage of a calibration session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStage {
    /// Session created, no observations yet.
    Uninitialized,
    /// Linear initialization complete.
    Initialized,
    /// Non-linear optimization complete.
    Optimized,
    /// Results exported (terminal state).
    Exported,
}

/// Metadata about a calibration session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Problem type identifier.
    pub problem_type: String,
    /// Timestamp when session was created (seconds since UNIX epoch).
    pub created_at: u64,
    /// Timestamp when session last changed stage (seconds since UNIX epoch).
    pub last_updated: u64,
    /// Optional user-provided description.
    pub description: Option<String>,
}

impl SessionMetadata {
    /// Create new metadata for the given problem type.
    fn new(problem_type: String) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            problem_type,
            created_at: now,
            last_updated: now,
            description: None,
        }
    }

    /// Update the last_updated timestamp to current time.
    fn touch(&mut self) {
        self.last_updated = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// A generic calibration session parameterized over a problem type.
///
/// Tracks progress through pipeline stages and provides a stateful API for
/// running calibration workflows with checkpoint support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSession<P: ProblemType> {
    /// Current pipeline stage.
    stage: SessionStage,
    /// Problem-specific observations.
    observations: Option<P::Observations>,
    /// Initial parameter estimates (populated after initialization).
    initial_values: Option<P::InitialValues>,
    /// Optimized results (populated after optimization).
    optimized_results: Option<P::OptimizedResults>,
    /// Session metadata.
    metadata: SessionMetadata,
}

impl<P: ProblemType> CalibrationSession<P> {
    /// Create a new uninitialized session.
    pub fn new() -> Self {
        Self {
            stage: SessionStage::Uninitialized,
            observations: None,
            initial_values: None,
            optimized_results: None,
            metadata: SessionMetadata::new(P::problem_name().to_string()),
        }
    }

    /// Create a new session with a description.
    pub fn new_with_description(description: String) -> Self {
        let mut session = Self::new();
        session.metadata.description = Some(description);
        session
    }

    /// Get the current stage.
    pub fn stage(&self) -> SessionStage {
        self.stage
    }

    /// Get session metadata.
    pub fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    /// Get mutable session metadata.
    pub fn metadata_mut(&mut self) -> &mut SessionMetadata {
        &mut self.metadata
    }

    /// Get observations if present.
    pub fn observations(&self) -> Option<&P::Observations> {
        self.observations.as_ref()
    }

    /// Get initial values if present.
    pub fn initial_values(&self) -> Option<&P::InitialValues> {
        self.initial_values.as_ref()
    }

    /// Get optimized results if present.
    pub fn optimized_results(&self) -> Option<&P::OptimizedResults> {
        self.optimized_results.as_ref()
    }

    /// Set observations and transition to Uninitialized stage.
    ///
    /// This can be called multiple times to replace observations.
    pub fn set_observations(&mut self, obs: P::Observations) {
        self.observations = Some(obs);
        self.stage = SessionStage::Uninitialized;
        self.initial_values = None;
        self.optimized_results = None;
        self.metadata.touch();
    }

    /// Run linear initialization.
    ///
    /// Requires: stage == Uninitialized and observations present.
    /// On success: transitions to Initialized stage.
    pub fn initialize(&mut self, opts: P::InitOptions) -> Result<&P::InitialValues> {
        if self.stage != SessionStage::Uninitialized {
            bail!(
                "initialize() requires Uninitialized stage, but session is in {:?}",
                self.stage
            );
        }

        let obs = self
            .observations
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no observations present"))?;

        let init = P::initialize(obs, &opts)?;
        self.initial_values = Some(init);
        self.stage = SessionStage::Initialized;
        self.metadata.touch();

        Ok(self.initial_values.as_ref().unwrap())
    }

    /// Run non-linear optimization.
    ///
    /// Requires: stage == Initialized.
    /// On success: transitions to Optimized stage.
    pub fn optimize(&mut self, opts: P::OptimOptions) -> Result<&P::OptimizedResults> {
        if self.stage != SessionStage::Initialized {
            bail!(
                "optimize() requires Initialized stage, but session is in {:?}",
                self.stage
            );
        }

        let obs = self.observations.as_ref().unwrap();
        let init = self.initial_values.as_ref().unwrap();

        let result = P::optimize(obs, init, &opts)?;
        self.optimized_results = Some(result);
        self.stage = SessionStage::Optimized;
        self.metadata.touch();

        Ok(self.optimized_results.as_ref().unwrap())
    }

    /// Export results and transition to Exported stage.
    ///
    /// Requires: stage == Optimized.
    /// On success: transitions to Exported (terminal state).
    pub fn export(&mut self) -> Result<P::OptimizedResults> {
        if self.stage != SessionStage::Optimized {
            bail!(
                "export() requires Optimized stage, but session is in {:?}",
                self.stage
            );
        }

        self.stage = SessionStage::Exported;
        self.metadata.touch();

        Ok(self.optimized_results.clone().unwrap())
    }

    /// Serialize session to JSON string for checkpointing.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Deserialize session from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }
}

impl<P: ProblemType> Default for CalibrationSession<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock problem type for testing
    #[derive(Clone, Serialize, Deserialize)]
    struct MockObservations {
        data: Vec<f64>,
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct MockInitial {
        value: f64,
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct MockResult {
        optimized_value: f64,
    }

    #[derive(Clone, Serialize, Deserialize)]
    struct MockInitOptions;

    #[derive(Clone, Serialize, Deserialize)]
    struct MockOptimOptions;

    struct MockProblem;

    impl ProblemType for MockProblem {
        type Observations = MockObservations;
        type InitialValues = MockInitial;
        type OptimizedResults = MockResult;
        type InitOptions = MockInitOptions;
        type OptimOptions = MockOptimOptions;

        fn problem_name() -> &'static str {
            "mock_problem"
        }

        fn initialize(
            obs: &Self::Observations,
            _opts: &Self::InitOptions,
        ) -> Result<Self::InitialValues> {
            Ok(MockInitial {
                value: obs.data.iter().sum(),
            })
        }

        fn optimize(
            _obs: &Self::Observations,
            init: &Self::InitialValues,
            _opts: &Self::OptimOptions,
        ) -> Result<Self::OptimizedResults> {
            Ok(MockResult {
                optimized_value: init.value * 2.0,
            })
        }
    }

    #[test]
    fn session_starts_uninitialized() {
        let session = CalibrationSession::<MockProblem>::new();
        assert_eq!(session.stage(), SessionStage::Uninitialized);
        assert!(session.observations().is_none());
        assert!(session.initial_values().is_none());
        assert!(session.optimized_results().is_none());
    }

    #[test]
    fn session_accepts_observations() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        session.set_observations(obs.clone());
        assert_eq!(session.stage(), SessionStage::Uninitialized);
        assert_eq!(session.observations().unwrap().data, obs.data);
    }

    #[test]
    fn initialize_requires_uninitialized_stage() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        session.set_observations(obs);

        let result = session.initialize(MockInitOptions);
        assert!(result.is_ok());
        assert_eq!(session.stage(), SessionStage::Initialized);
        assert!((session.initial_values().unwrap().value - 6.0).abs() < 1e-12);
    }

    #[test]
    fn initialize_fails_without_observations() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let result = session.initialize(MockInitOptions);
        assert!(result.is_err());
    }

    #[test]
    fn optimize_requires_initialized_stage() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        session.set_observations(obs);
        session.initialize(MockInitOptions).unwrap();

        let result = session.optimize(MockOptimOptions);
        assert!(result.is_ok());
        assert_eq!(session.stage(), SessionStage::Optimized);
        assert!((session.optimized_results().unwrap().optimized_value - 12.0).abs() < 1e-12);
    }

    #[test]
    fn optimize_fails_if_not_initialized() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let result = session.optimize(MockOptimOptions);
        assert!(result.is_err());
    }

    #[test]
    fn export_requires_optimized_stage() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        session.set_observations(obs);
        session.initialize(MockInitOptions).unwrap();
        session.optimize(MockOptimOptions).unwrap();

        let result = session.export();
        assert!(result.is_ok());
        assert_eq!(session.stage(), SessionStage::Exported);
        assert!((result.unwrap().optimized_value - 12.0).abs() < 1e-12);
    }

    #[test]
    fn export_fails_if_not_optimized() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let result = session.export();
        assert!(result.is_err());
    }

    #[test]
    fn session_json_roundtrip() {
        let mut session =
            CalibrationSession::<MockProblem>::new_with_description("Test session".to_string());
        let obs = MockObservations {
            data: vec![1.0, 2.0, 3.0],
        };
        session.set_observations(obs);
        session.initialize(MockInitOptions).unwrap();

        let json = session.to_json().unwrap();
        let restored: CalibrationSession<MockProblem> =
            CalibrationSession::from_json(&json).unwrap();

        assert_eq!(restored.stage(), SessionStage::Initialized);
        assert_eq!(restored.observations().unwrap().data, vec![1.0, 2.0, 3.0]);
        assert!((restored.initial_values().unwrap().value - 6.0).abs() < 1e-12);
        assert_eq!(
            restored.metadata().description.as_ref().unwrap(),
            "Test session"
        );
    }

    #[test]
    fn metadata_tracks_timestamps() {
        let session = CalibrationSession::<MockProblem>::new();
        let created = session.metadata().created_at;
        let updated = session.metadata().last_updated;
        assert_eq!(created, updated);
        assert!(created > 0);
    }

    #[test]
    fn set_observations_resets_state() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let obs1 = MockObservations {
            data: vec![1.0, 2.0],
        };
        session.set_observations(obs1);
        session.initialize(MockInitOptions).unwrap();
        assert_eq!(session.stage(), SessionStage::Initialized);

        let obs2 = MockObservations {
            data: vec![3.0, 4.0],
        };
        session.set_observations(obs2);
        assert_eq!(session.stage(), SessionStage::Uninitialized);
        assert!(session.initial_values().is_none());
    }
}
