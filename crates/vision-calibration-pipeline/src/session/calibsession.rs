//! Calibration session container with mutable state.
//!
//! Provides a generic session container parameterized over a problem type.
//! Sessions store configuration, input data, intermediate state, and the
//! final output. Step functions mutate the session in-place.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use super::problem_type::ProblemType;
use super::types::{ExportRecord, LogEntry, SessionMetadata};

/// A calibration session container with mutable state.
///
/// The session stores configuration, input data, intermediate state,
/// and the final output. Step functions mutate the session in-place.
///
/// # Design Principles
///
/// - **Single output**: Only one final result is stored.
/// - **Embedded input**: Input data is stored directly in the session.
/// - **Config changes don't auto-clear**: Changing config doesn't invalidate output.
/// - **Exports collection**: Multiple exports can be generated and stored.
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::planar_intrinsics::{PlanarIntrinsicsProblem, step_init, step_optimize};
/// # fn main() -> anyhow::Result<()> {
/// # let dataset = unimplemented!();
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
///
/// step_init(&mut session, None)?;
/// step_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "P: ProblemType")]
pub struct CalibrationSession<P: ProblemType> {
    /// Session metadata (problem type, schema version, timestamps, description).
    pub metadata: SessionMetadata,

    /// Configuration parameters (always present, defaults if not explicitly set).
    pub config: P::Config,

    /// Input observations (embedded). `None` until set.
    input: Option<P::Input>,

    /// Problem-specific intermediate state (default until computed).
    pub state: P::State,

    /// Final calibration output. `None` until computed.
    output: Option<P::Output>,

    /// Collection of generated exports.
    pub exports: Vec<ExportRecord<P::Export>>,

    /// Operation log (lightweight audit trail).
    pub log: Vec<LogEntry>,
}

impl<P: ProblemType> CalibrationSession<P> {
    // ─────────────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a new empty session with default configuration.
    pub fn new() -> Self {
        Self {
            metadata: SessionMetadata::new(P::name(), P::schema_version()),
            config: P::Config::default(),
            input: None,
            state: P::State::default(),
            output: None,
            exports: Vec::new(),
            log: Vec::new(),
        }
    }

    /// Create a new session with a description.
    pub fn with_description(description: impl Into<String>) -> Self {
        Self {
            metadata: SessionMetadata::with_description(
                P::name(),
                P::schema_version(),
                description,
            ),
            config: P::Config::default(),
            input: None,
            state: P::State::default(),
            output: None,
            exports: Vec::new(),
            log: Vec::new(),
        }
    }

    /// Create a new session with input data.
    ///
    /// # Errors
    ///
    /// Returns an error if input validation fails.
    pub fn with_input(input: P::Input) -> Result<Self> {
        let mut session = Self::new();
        session.set_input(input)?;
        Ok(session)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Input Management
    // ─────────────────────────────────────────────────────────────────────────

    /// Set input data, applying validation and invalidation policy.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ProblemType::validate_input`] fails.
    pub fn set_input(&mut self, input: P::Input) -> Result<()> {
        P::validate_input(&input)?;

        let policy = P::on_input_change();
        if policy.clear_state {
            self.state = P::State::default();
        }
        if policy.clear_output {
            self.output = None;
        }
        if policy.clear_exports {
            self.exports.clear();
        }

        self.input = Some(input);
        self.metadata.touch();
        Ok(())
    }

    /// Get a reference to the input, if set.
    pub fn input(&self) -> Option<&P::Input> {
        self.input.as_ref()
    }

    /// Get a mutable reference to the input, if set.
    pub fn input_mut(&mut self) -> Option<&mut P::Input> {
        self.input.as_mut()
    }

    /// Get a reference to the input, or error if not set.
    ///
    /// # Errors
    ///
    /// Returns an error if input is not set.
    pub fn require_input(&self) -> Result<&P::Input> {
        self.input
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("input not set"))
    }

    /// Get a mutable reference to the input, or error if not set.
    ///
    /// # Errors
    ///
    /// Returns an error if input is not set.
    pub fn require_input_mut(&mut self) -> Result<&mut P::Input> {
        self.input
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("input not set"))
    }

    /// Check if input is set.
    pub fn has_input(&self) -> bool {
        self.input.is_some()
    }

    /// Clear input data, applying invalidation policy.
    pub fn clear_input(&mut self) {
        let policy = P::on_input_change();
        if policy.clear_state {
            self.state = P::State::default();
        }
        if policy.clear_output {
            self.output = None;
        }
        if policy.clear_exports {
            self.exports.clear();
        }
        self.input = None;
        self.metadata.touch();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration Management
    // ─────────────────────────────────────────────────────────────────────────

    /// Set configuration, applying validation and invalidation policy.
    ///
    /// # Errors
    ///
    /// Returns an error if [`ProblemType::validate_config`] fails.
    pub fn set_config(&mut self, config: P::Config) -> Result<()> {
        P::validate_config(&config)?;

        let policy = P::on_config_change();
        if policy.clear_state {
            self.state = P::State::default();
        }
        if policy.clear_output {
            self.output = None;
        }
        if policy.clear_exports {
            self.exports.clear();
        }

        self.config = config;
        self.metadata.touch();
        Ok(())
    }

    /// Update configuration with a closure.
    ///
    /// The closure receives a mutable reference to the current config.
    /// After the closure returns, validation and invalidation policy are applied.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails after the update.
    pub fn update_config<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&mut P::Config),
    {
        let mut new_config = self.config.clone();
        f(&mut new_config);
        self.set_config(new_config)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Output Management
    // ─────────────────────────────────────────────────────────────────────────

    /// Get a reference to the output, if computed.
    pub fn output(&self) -> Option<&P::Output> {
        self.output.as_ref()
    }

    /// Get a mutable reference to the output, if computed.
    pub fn output_mut(&mut self) -> Option<&mut P::Output> {
        self.output.as_mut()
    }

    /// Get a reference to the output, or error if not computed.
    ///
    /// # Errors
    ///
    /// Returns an error if output is not computed.
    pub fn require_output(&self) -> Result<&P::Output> {
        self.output
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("output not computed"))
    }

    /// Set the output (typically called by step functions).
    pub fn set_output(&mut self, output: P::Output) {
        self.output = Some(output);
        self.metadata.touch();
    }

    /// Check if output is computed.
    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    /// Clear output.
    pub fn clear_output(&mut self) {
        self.output = None;
        self.metadata.touch();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Export
    // ─────────────────────────────────────────────────────────────────────────

    /// Export the current output and add to exports collection.
    ///
    /// # Errors
    ///
    /// Returns an error if output is not computed or if export conversion fails.
    pub fn export(&mut self) -> Result<P::Export> {
        let output = self.require_output()?;
        let export = P::export(output, &self.config)?;
        self.exports.push(ExportRecord::new(export.clone()));
        self.metadata.touch();
        Ok(export)
    }

    /// Export the current output with notes and add to exports collection.
    ///
    /// # Errors
    ///
    /// Returns an error if output is not computed or if export conversion fails.
    pub fn export_with_notes(&mut self, notes: impl Into<String>) -> Result<P::Export> {
        let output = self.require_output()?;
        let export = P::export(output, &self.config)?;
        self.exports
            .push(ExportRecord::with_notes(export.clone(), notes));
        self.metadata.touch();
        Ok(export)
    }

    /// Export without adding to collection (peek).
    ///
    /// Useful for inspecting the export without modifying the session.
    ///
    /// # Errors
    ///
    /// Returns an error if output is not computed or if export conversion fails.
    pub fn export_peek(&self) -> Result<P::Export> {
        let output = self.require_output()?;
        P::export(output, &self.config)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Validation
    // ─────────────────────────────────────────────────────────────────────────

    /// Validate that the session is ready for processing.
    ///
    /// Checks:
    /// 1. Input is set and valid
    /// 2. Config is valid
    /// 3. Input and config are compatible (cross-validation)
    ///
    /// # Errors
    ///
    /// Returns an error if any validation check fails.
    pub fn validate(&self) -> Result<()> {
        let input = self.require_input()?;
        P::validate_input(input)?;
        P::validate_config(&self.config)?;
        P::validate_input_config(input, &self.config)?;
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Logging
    // ─────────────────────────────────────────────────────────────────────────

    /// Log a successful operation.
    pub fn log_success(&mut self, operation: impl Into<String>) {
        self.log.push(LogEntry::success(operation));
        self.metadata.touch();
    }

    /// Log a successful operation with notes.
    pub fn log_success_with_notes(
        &mut self,
        operation: impl Into<String>,
        notes: impl Into<String>,
    ) {
        self.log
            .push(LogEntry::success_with_notes(operation, notes));
        self.metadata.touch();
    }

    /// Log a failed operation.
    pub fn log_failure(&mut self, operation: impl Into<String>, error: impl Into<String>) {
        self.log.push(LogEntry::failure(operation, error));
        self.metadata.touch();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reset
    // ─────────────────────────────────────────────────────────────────────────

    /// Reset state to default, keeping input, config, and output.
    pub fn reset_state(&mut self) {
        self.state = P::State::default();
        self.metadata.touch();
    }

    /// Reset output, keeping input, config, and state.
    pub fn reset_output(&mut self) {
        self.output = None;
        self.metadata.touch();
    }

    /// Reset everything except config and description.
    pub fn reset(&mut self) {
        self.input = None;
        self.state = P::State::default();
        self.output = None;
        self.exports.clear();
        self.log.clear();
        self.metadata.touch();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Serialization
    // ─────────────────────────────────────────────────────────────────────────

    /// Serialize session to JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Deserialize session from JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Deserialization fails
    /// - Schema version is newer than supported
    pub fn from_json(json: &str) -> Result<Self> {
        let session: Self = serde_json::from_str(json)?;

        // Verify schema version compatibility
        if session.metadata.schema_version > P::schema_version() {
            bail!(
                "session schema version {} is newer than supported version {}",
                session.metadata.schema_version,
                P::schema_version()
            );
        }

        Ok(session)
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
    use serde::{Deserialize, Serialize};

    // ─────────────────────────────────────────────────────────────────────────
    // Mock Problem Type for Testing
    // ─────────────────────────────────────────────────────────────────────────

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    struct MockConfig {
        scale: f64,
        max_iters: usize,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockInput {
        data: Vec<f64>,
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    struct MockState {
        computed: Option<f64>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockOutput {
        result: f64,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockExport {
        value: f64,
    }

    #[derive(Debug)]
    struct MockProblem;

    impl ProblemType for MockProblem {
        type Config = MockConfig;
        type Input = MockInput;
        type State = MockState;
        type Output = MockOutput;
        type Export = MockExport;

        fn name() -> &'static str {
            "mock_problem"
        }

        fn schema_version() -> u32 {
            1
        }

        fn validate_input(input: &Self::Input) -> Result<()> {
            if input.data.is_empty() {
                bail!("input data cannot be empty");
            }
            Ok(())
        }

        fn validate_config(config: &Self::Config) -> Result<()> {
            if config.max_iters == 0 {
                bail!("max_iters must be positive");
            }
            Ok(())
        }

        fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
            Ok(MockExport {
                value: output.result,
            })
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn new_session_has_defaults() {
        let session = CalibrationSession::<MockProblem>::new();
        assert_eq!(session.metadata.problem_type, "mock_problem");
        assert_eq!(session.metadata.schema_version, 1);
        assert!(session.metadata.description.is_none());
        assert!(session.input().is_none());
        assert!(session.output().is_none());
        assert!(session.exports.is_empty());
        assert!(session.log.is_empty());
    }

    #[test]
    fn with_description() {
        let session =
            CalibrationSession::<MockProblem>::with_description("Test calibration session");
        assert_eq!(
            session.metadata.description,
            Some("Test calibration session".to_string())
        );
    }

    #[test]
    fn set_input_validates() {
        let mut session = CalibrationSession::<MockProblem>::new();

        // Empty input should fail
        let result = session.set_input(MockInput { data: vec![] });
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));

        // Valid input should succeed
        let result = session.set_input(MockInput {
            data: vec![1.0, 2.0],
        });
        assert!(result.is_ok());
        assert!(session.has_input());
    }

    #[test]
    fn set_input_clears_state_and_output() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.state.computed = Some(42.0);
        session.set_output(MockOutput { result: 100.0 });

        // Set new input - should clear state and output
        session.set_input(MockInput { data: vec![2.0] }).unwrap();

        assert!(session.state.computed.is_none());
        assert!(session.output().is_none());
    }

    #[test]
    fn set_config_keeps_output() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.set_output(MockOutput { result: 100.0 });

        // Change config - should keep output (per default policy)
        session
            .set_config(MockConfig {
                scale: 2.0,
                max_iters: 100,
            })
            .unwrap();

        assert!(session.has_output());
        assert_eq!(session.output().unwrap().result, 100.0);
    }

    #[test]
    fn set_config_validates() {
        let mut session = CalibrationSession::<MockProblem>::new();

        // Invalid config should fail
        let result = session.set_config(MockConfig {
            scale: 1.0,
            max_iters: 0,
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iters"));
    }

    #[test]
    fn update_config() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.config.scale = 1.0;
        session.config.max_iters = 10;

        session.update_config(|c| c.scale = 5.0).unwrap();

        assert_eq!(session.config.scale, 5.0);
        assert_eq!(session.config.max_iters, 10); // unchanged
    }

    #[test]
    fn require_input_errors_when_none() {
        let session = CalibrationSession::<MockProblem>::new();
        let result = session.require_input();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("input not set"));
    }

    #[test]
    fn require_output_errors_when_none() {
        let session = CalibrationSession::<MockProblem>::new();
        let result = session.require_output();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("output not computed")
        );
    }

    #[test]
    fn export_adds_to_collection() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.set_output(MockOutput { result: 42.0 });

        assert!(session.exports.is_empty());

        let export1 = session.export().unwrap();
        assert_eq!(export1.value, 42.0);
        assert_eq!(session.exports.len(), 1);

        let export2 = session.export().unwrap();
        assert_eq!(export2.value, 42.0);
        assert_eq!(session.exports.len(), 2);
    }

    #[test]
    fn export_peek_does_not_store() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.set_output(MockOutput { result: 42.0 });

        let export = session.export_peek().unwrap();
        assert_eq!(export.value, 42.0);
        assert!(session.exports.is_empty()); // Not added
    }

    #[test]
    fn export_with_notes() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.set_output(MockOutput { result: 42.0 });

        session.export_with_notes("final result").unwrap();

        assert_eq!(session.exports.len(), 1);
        assert_eq!(session.exports[0].notes, Some("final result".to_string()));
    }

    #[test]
    fn validate_checks_all_hooks() {
        let mut session = CalibrationSession::<MockProblem>::new();

        // No input - should fail
        let result = session.validate();
        assert!(result.is_err());

        // With valid input and config
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.config.max_iters = 10;
        let result = session.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn log_entries_recorded() {
        let mut session = CalibrationSession::<MockProblem>::new();

        session.log_success("init");
        session.log_success_with_notes("optimize", "converged");
        session.log_failure("filter", "too few points");

        assert_eq!(session.log.len(), 3);
        assert!(session.log[0].success);
        assert_eq!(session.log[0].operation, "init");
        assert!(session.log[1].success);
        assert_eq!(session.log[1].notes, Some("converged".to_string()));
        assert!(!session.log[2].success);
        assert_eq!(session.log[2].notes, Some("too few points".to_string()));
    }

    #[test]
    fn json_roundtrip_empty() {
        let session = CalibrationSession::<MockProblem>::new();
        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<MockProblem>::from_json(&json).unwrap();

        assert_eq!(restored.metadata.problem_type, "mock_problem");
        assert!(restored.input().is_none());
        assert!(restored.output().is_none());
    }

    #[test]
    fn json_roundtrip_with_input() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session
            .set_input(MockInput {
                data: vec![1.0, 2.0, 3.0],
            })
            .unwrap();

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<MockProblem>::from_json(&json).unwrap();

        let input = restored.input().unwrap();
        assert_eq!(input.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn json_roundtrip_with_output() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.set_output(MockOutput { result: 42.0 });

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<MockProblem>::from_json(&json).unwrap();

        assert_eq!(restored.output().unwrap().result, 42.0);
    }

    #[test]
    fn json_roundtrip_with_exports() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.set_output(MockOutput { result: 42.0 });
        session.export().unwrap();
        session.export_with_notes("second export").unwrap();

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<MockProblem>::from_json(&json).unwrap();

        assert_eq!(restored.exports.len(), 2);
        assert!(restored.exports[0].notes.is_none());
        assert_eq!(restored.exports[1].notes, Some("second export".to_string()));
    }

    #[test]
    fn schema_version_checked() {
        // Create a session and serialize it
        let session = CalibrationSession::<MockProblem>::new();
        let mut json = session.to_json().unwrap();

        // Manually bump the schema version in the JSON
        json = json.replace("\"schema_version\": 1", "\"schema_version\": 999");

        // Deserializing should fail
        let result = CalibrationSession::<MockProblem>::from_json(&json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("schema version"));
    }

    #[test]
    fn reset_state() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.state.computed = Some(42.0);
        session.set_output(MockOutput { result: 100.0 });

        session.reset_state();

        assert!(session.has_input()); // Kept
        assert!(session.state.computed.is_none()); // Cleared
        assert!(session.has_output()); // Kept
    }

    #[test]
    fn reset_output() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.state.computed = Some(42.0);
        session.set_output(MockOutput { result: 100.0 });

        session.reset_output();

        assert!(session.has_input()); // Kept
        assert!(session.state.computed.is_some()); // Kept
        assert!(!session.has_output()); // Cleared
    }

    #[test]
    fn reset_all() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.state.computed = Some(42.0);
        session.set_output(MockOutput { result: 100.0 });
        session.export().unwrap();
        session.log_success("test");

        session.reset();

        assert!(!session.has_input());
        assert!(session.state.computed.is_none());
        assert!(!session.has_output());
        assert!(session.exports.is_empty());
        assert!(session.log.is_empty());
    }

    #[test]
    fn clear_input() {
        let mut session = CalibrationSession::<MockProblem>::new();
        session.set_input(MockInput { data: vec![1.0] }).unwrap();
        session.state.computed = Some(42.0);
        session.set_output(MockOutput { result: 100.0 });

        session.clear_input();

        assert!(!session.has_input());
        assert!(session.state.computed.is_none()); // Cleared per policy
        assert!(!session.has_output()); // Cleared per policy
    }

    #[test]
    fn with_input_validates() {
        // Invalid input
        let result = CalibrationSession::<MockProblem>::with_input(MockInput { data: vec![] });
        assert!(result.is_err());

        // Valid input
        let result = CalibrationSession::<MockProblem>::with_input(MockInput { data: vec![1.0] });
        assert!(result.is_ok());
        assert!(result.unwrap().has_input());
    }

    #[test]
    fn metadata_timestamps_update() {
        let mut session = CalibrationSession::<MockProblem>::new();
        let created = session.metadata.created_at;
        let initial_modified = session.metadata.last_modified;

        // Operations should update last_modified
        session.set_input(MockInput { data: vec![1.0] }).unwrap();

        assert_eq!(session.metadata.created_at, created);
        assert!(session.metadata.last_modified >= initial_modified);
    }
}
