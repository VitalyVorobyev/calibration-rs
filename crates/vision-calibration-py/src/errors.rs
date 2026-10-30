//! Python exceptions raised by the extension and the mapping from the Rust
//! error types.
//!
//! The calibration error (`vision_calibration::Error`) keeps the package's
//! established style: invalid input → `ValueError`, everything else →
//! `RuntimeError`. The manifest runner's `RunError` and the folder sniffer's
//! `SniffError` are typed exceptions that carry the Rust variant as `kind`,
//! so a caller can branch on it; the runner's ask-user ambiguity is its own
//! subclass carrying the field, prompt and suggestions.

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use vision_calibration::dataset::SniffError as RustSniffError;
use vision_calibration::dataset_runner::RunError as RustRunError;

create_exception!(
    vision_calibration,
    RunError,
    PyRuntimeError,
    "A manifest-driven run failed. `kind` names the failure (`\"validation\"`, `\"empty_image_match\"`, `\"detection\"`, ...)."
);
create_exception!(
    vision_calibration,
    AskUserError,
    RunError,
    "The manifest is ambiguous and the runner will not guess: `field` names the manifest field to set, `prompt` asks for it, `suggestions` lists candidate values."
);
create_exception!(
    vision_calibration,
    SniffError,
    PyRuntimeError,
    "A dataset folder could not be sniffed. `kind` is `\"not_a_directory\"`, `\"no_cameras\"` or `\"io\"`."
);

/// Map a calibration error: input problems are `ValueError`, the rest
/// `RuntimeError`. `context` prefixes the message.
pub(crate) fn calib_err(context: &str, err: vision_calibration::Error) -> PyErr {
    use vision_calibration::Error;
    let message = format!("{context}: {err}");
    match err {
        Error::InvalidInput { .. }
        | Error::InsufficientData { .. }
        | Error::NotAvailable { .. } => PyValueError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}

/// The `kind` of a `RunError`: its Rust variant in snake case.
pub(crate) fn run_error_kind(err: &RustRunError) -> &'static str {
    match err {
        RustRunError::Validation(_) => "validation",
        RustRunError::UnsupportedTopology { .. } => "unsupported_topology",
        RustRunError::UnsupportedTarget { .. } => "unsupported_target",
        RustRunError::InvalidTargetConfig { .. } => "invalid_target_config",
        RustRunError::EmptyImageMatch { .. } => "empty_image_match",
        RustRunError::BadGlob { .. } => "bad_glob",
        RustRunError::ViewCountMismatch { .. } => "view_count_mismatch",
        RustRunError::PoseCountMismatch { .. } => "pose_count_mismatch",
        RustRunError::PoseParse { .. } => "pose_parse",
        RustRunError::BadPairingRegex { .. } => "bad_pairing_regex",
        RustRunError::PairingTokenMismatch { .. } => "pairing_token_mismatch",
        RustRunError::Io(_) => "io",
        RustRunError::Decode { .. } => "decode",
        RustRunError::Detection { .. } => "detection",
        RustRunError::CornerCovariance { .. } => "corner_covariance",
        RustRunError::CovarianceUnsupported { .. } => "covariance_unsupported",
        RustRunError::Cache(_) => "cache",
        RustRunError::AskUser { .. } => "ask_user",
        RustRunError::InsufficientUsableViews { .. } => "insufficient_usable_views",
        RustRunError::NoWholeChessboardViews { .. } => "no_whole_chessboard_views",
        RustRunError::LaserPairing { .. } => "laser_pairing",
        RustRunError::LaserExtraction { .. } => "laser_extraction",
        RustRunError::InsufficientLaserViews { .. } => "insufficient_laser_views",
        RustRunError::UpstreamCalibration { .. } => "upstream_calibration",
    }
}

/// Map a runner error to `RunError`, or `AskUserError` for the ask-user
/// ambiguity.
pub(crate) fn run_err(py: Python<'_>, err: RustRunError) -> PyErr {
    let kind = run_error_kind(&err);
    let message = err.to_string();
    let result = match err {
        RustRunError::AskUser {
            field,
            prompt,
            suggestions,
        } => {
            let exc = AskUserError::new_err(message);
            let value = exc.value(py);
            value
                .setattr("kind", kind)
                .and_then(|()| value.setattr("field", field))
                .and_then(|()| value.setattr("prompt", prompt))
                .and_then(|()| value.setattr("suggestions", suggestions))
                .map(|()| exc)
        }
        _ => {
            let exc = RunError::new_err(message);
            exc.value(py).setattr("kind", kind).map(|()| exc)
        }
    };
    result.unwrap_or_else(|err| err)
}

/// Map a sniffer error to `SniffError`.
pub(crate) fn sniff_err(py: Python<'_>, err: RustSniffError) -> PyErr {
    let kind = match &err {
        RustSniffError::NotADirectory(_) => "not_a_directory",
        RustSniffError::NoCameras(_) => "no_cameras",
        RustSniffError::Io { .. } => "io",
    };
    let exc = SniffError::new_err(err.to_string());
    match exc.value(py).setattr("kind", kind) {
        Ok(()) => exc,
        Err(err) => err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_error_kinds_are_the_snake_case_variant_names() {
        let err = RustRunError::AskUser {
            field: "pose_pairing".into(),
            prompt: "?".into(),
            suggestions: vec![],
        };
        assert_eq!(run_error_kind(&err), "ask_user");
        let err = RustRunError::InvalidTargetConfig {
            message: "bad".into(),
        };
        assert_eq!(run_error_kind(&err), "invalid_target_config");
        let err = RustRunError::InsufficientUsableViews {
            camera_id: "cam0".into(),
            usable: 1,
            total: 3,
        };
        assert_eq!(run_error_kind(&err), "insufficient_usable_views");
    }
}
