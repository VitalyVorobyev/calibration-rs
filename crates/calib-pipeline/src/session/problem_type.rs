//! Problem type trait for calibration sessions.
//!
//! Defines the minimal interface that calibration problems must implement
//! to work with [`CalibrationSession`](super::CalibrationSession).

use anyhow::Result;
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

/// Policy for invalidating session data when input or config changes.
///
/// Used by [`ProblemType::on_input_change`] and [`ProblemType::on_config_change`]
/// to specify what session fields should be cleared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidationPolicy {
    /// Clear the problem-specific state (intermediate results).
    pub clear_state: bool,
    /// Clear the final output.
    pub clear_output: bool,
    /// Clear the exports collection.
    pub clear_exports: bool,
}

impl InvalidationPolicy {
    /// Policy that clears nothing.
    pub const KEEP_ALL: Self = Self {
        clear_state: false,
        clear_output: false,
        clear_exports: false,
    };

    /// Policy that clears state and output but keeps exports.
    pub const CLEAR_COMPUTED: Self = Self {
        clear_state: true,
        clear_output: true,
        clear_exports: false,
    };

    /// Policy that clears everything.
    pub const CLEAR_ALL: Self = Self {
        clear_state: true,
        clear_output: true,
        clear_exports: true,
    };
}

impl Default for InvalidationPolicy {
    fn default() -> Self {
        Self::KEEP_ALL
    }
}

/// Trait defining the interface for a calibration problem.
///
/// Each problem type (e.g., planar intrinsics, hand-eye, laserline) implements
/// this trait to specify its configuration, input data, internal state,
/// output, and export formats.
///
/// # Design Philosophy
///
/// The trait is minimal by design. Problem-specific behavior is implemented
/// via step functions that operate on `&mut CalibrationSession<Self>` rather
/// than as trait methods. This allows:
///
/// - Arbitrary step signatures with step-specific options
/// - Easy composition of custom pipelines
/// - Direct access to intermediate state
///
/// # Associated Types
///
/// - **Config**: Calibration parameters (solver options, fix masks, thresholds)
/// - **Input**: Observation data (image points, correspondences)
/// - **State**: Problem-specific intermediate results (homographies, initial poses)
/// - **Output**: Final calibration result (camera parameters, errors)
/// - **Export**: User-facing export format (may be same as Output)
///
/// # Example
///
/// ```ignore
/// pub struct MyProblem;
///
/// impl ProblemType for MyProblem {
///     type Config = MyConfig;
///     type Input = MyDataset;
///     type State = MyState;
///     type Output = MyResult;
///     type Export = MyExport;
///
///     fn name() -> &'static str { "my_problem" }
///
///     fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
///         Ok(output.into())
///     }
/// }
/// ```
pub trait ProblemType: Sized + 'static {
    /// Configuration parameters that control calibration behavior.
    ///
    /// Examples: solver options, fix masks, convergence thresholds.
    /// Must have a sensible `Default` implementation.
    type Config: Clone + Default + Serialize + DeserializeOwned + Debug;

    /// Input observations (embedded in session).
    ///
    /// Examples: `PlanarDataset`, `HandEyeDataset`, `LaserlineDataset`.
    type Input: Clone + Serialize + DeserializeOwned + Debug;

    /// Problem-specific workspace for intermediate results.
    ///
    /// Examples: computed homographies, initial poses, per-view residuals.
    /// Use `()` if no intermediate state is needed.
    type State: Clone + Default + Serialize + DeserializeOwned + Debug;

    /// Final calibration output (single result).
    ///
    /// Examples: `PlanarIntrinsicsEstimate`, `HandEyeEstimate`.
    type Output: Clone + Serialize + DeserializeOwned + Debug;

    /// Export format for external consumption.
    ///
    /// May be the same as `Output`, or a simplified/transformed version
    /// suitable for downstream use.
    type Export: Clone + Serialize + DeserializeOwned + Debug;

    // ─────────────────────────────────────────────────────────────────────────
    // Identity
    // ─────────────────────────────────────────────────────────────────────────

    /// Unique identifier for this problem type.
    ///
    /// Used for serialization, logging, and identifying session files.
    /// Should be a stable, lowercase, snake_case string.
    ///
    /// Examples: `"planar_intrinsics"`, `"hand_eye"`, `"laserline_bundle"`.
    fn name() -> &'static str;

    /// Schema version for forward compatibility.
    ///
    /// Bump when the serialization format of any associated type changes
    /// in a way that breaks backward compatibility.
    ///
    /// The session loader will reject sessions with a schema version
    /// newer than the current implementation supports.
    fn schema_version() -> u32 {
        1
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Validation Hooks (optional)
    // ─────────────────────────────────────────────────────────────────────────

    /// Validate input data after setting.
    ///
    /// Called by [`CalibrationSession::set_input`](super::CalibrationSession::set_input).
    /// Return an error to reject the input.
    ///
    /// Default implementation: always valid.
    fn validate_input(_input: &Self::Input) -> Result<()> {
        Ok(())
    }

    /// Validate configuration.
    ///
    /// Called by [`CalibrationSession::set_config`](super::CalibrationSession::set_config).
    /// Return an error to reject the config.
    ///
    /// Default implementation: always valid.
    fn validate_config(_config: &Self::Config) -> Result<()> {
        Ok(())
    }

    /// Cross-validate input and config together.
    ///
    /// Called by [`CalibrationSession::validate`](super::CalibrationSession::validate)
    /// after individual validation passes.
    ///
    /// Default implementation: always valid.
    fn validate_input_config(_input: &Self::Input, _config: &Self::Config) -> Result<()> {
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reset/Invalidation Policy (optional)
    // ─────────────────────────────────────────────────────────────────────────

    /// Policy for what to clear when input changes.
    ///
    /// Default: clear state and output, keep exports.
    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    /// Policy for what to clear when config changes.
    ///
    /// Default: keep everything (config changes don't auto-invalidate).
    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Export (required)
    // ─────────────────────────────────────────────────────────────────────────

    /// Convert output to export format.
    ///
    /// Called by [`CalibrationSession::export`](super::CalibrationSession::export).
    /// The config is provided for cases where export format depends on settings.
    fn export(output: &Self::Output, config: &Self::Config) -> Result<Self::Export>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verify that InvalidationPolicy constants are correct
    #[test]
    fn invalidation_policy_constants() {
        assert_eq!(
            InvalidationPolicy::KEEP_ALL,
            InvalidationPolicy {
                clear_state: false,
                clear_output: false,
                clear_exports: false,
            }
        );
        assert_eq!(
            InvalidationPolicy::CLEAR_COMPUTED,
            InvalidationPolicy {
                clear_state: true,
                clear_output: true,
                clear_exports: false,
            }
        );
        assert_eq!(
            InvalidationPolicy::CLEAR_ALL,
            InvalidationPolicy {
                clear_state: true,
                clear_output: true,
                clear_exports: true,
            }
        );
    }

    #[test]
    fn invalidation_policy_default() {
        let policy = InvalidationPolicy::default();
        assert_eq!(policy, InvalidationPolicy::KEEP_ALL);
    }
}
