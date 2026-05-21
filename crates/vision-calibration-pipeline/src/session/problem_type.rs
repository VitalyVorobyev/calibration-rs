//! Problem type trait for calibration sessions.
//!
//! Defines the minimal interface that calibration problems must implement
//! to work with [`CalibrationSession`](super::CalibrationSession).

use crate::Error;
use serde::{Serialize, de::DeserializeOwned};
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

/// Crate-internal carrier for a problem's session scratch state.
///
/// Kept separate from the public [`ProblemType`] contract so the per-problem
/// `*State` scratch structs can remain `pub(crate)` — they are internal
/// pipeline plumbing, not part of the public API.
pub(crate) trait ProblemState {
    /// Problem-specific workspace for intermediate results.
    type State: Clone + Default + Serialize + DeserializeOwned + Debug;
}

/// Trait defining the interface for a calibration problem.
///
/// Each problem type (e.g., planar intrinsics, hand-eye, laserline) implements
/// this trait to specify its configuration, input data, output, and export
/// formats.
///
/// This trait is implemented only by the crate's built-in problem types. It is
/// not downstream-implementable: it requires a `pub(crate)` supertrait
/// (`ProblemState`), which carries the internal `State` associated type.
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
/// - **Output**: Final calibration result (camera parameters, errors)
/// - **Export**: User-facing export format (may be same as Output)
///
/// The internal `State` associated type (problem-specific intermediate results)
/// is carried by the crate-private `ProblemState` supertrait.
///
// The `pub(crate)` [`ProblemState`] supertrait is intentional: it seals
// `ProblemType` so only the crate's built-in problem types can implement it.
#[allow(private_bounds)]
pub trait ProblemType: ProblemState + Sized + 'static {
    /// Configuration parameters that control calibration behavior.
    ///
    /// Examples: solver options, fix masks, convergence thresholds.
    /// Must have a sensible `Default` implementation.
    type Config: Clone + Default + Serialize + DeserializeOwned + Debug;

    /// Input observations (embedded in session).
    ///
    /// Examples: `PlanarDataset`, `HandEyeDataset`, `LaserlineDataset`.
    type Input: Clone + Serialize + DeserializeOwned + Debug;

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
    /// The session loader validates this value strictly and rejects sessions
    /// whose metadata schema version differs from the current implementation.
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
    fn validate_input(_input: &Self::Input) -> Result<(), Error> {
        Ok(())
    }

    /// Validate configuration.
    ///
    /// Called by [`CalibrationSession::set_config`](super::CalibrationSession::set_config).
    /// Return an error to reject the config.
    ///
    /// Default implementation: always valid.
    fn validate_config(_config: &Self::Config) -> Result<(), Error> {
        Ok(())
    }

    /// Cross-validate input and config together.
    ///
    /// Called by [`CalibrationSession::validate`](super::CalibrationSession::validate)
    /// after individual validation passes.
    ///
    /// Default implementation: always valid.
    fn validate_input_config(_input: &Self::Input, _config: &Self::Config) -> Result<(), Error> {
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
    /// The input + config are provided so exports can attach per-feature
    /// reprojection residuals (ADR 0012) and other input-dependent diagnostic
    /// data.
    fn export(
        input: &Self::Input,
        output: &Self::Output,
        config: &Self::Config,
    ) -> Result<Self::Export, Error>;
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
