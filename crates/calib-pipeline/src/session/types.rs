//! Session infrastructure types for the session API.
//!
//! Defines metadata, logging, and export record types.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Metadata about a calibration session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Problem type identifier (from `ProblemType::name()`).
    pub problem_type: String,

    /// Schema version (from `ProblemType::schema_version()`).
    pub schema_version: u32,

    /// Unix timestamp when session was created (seconds since epoch).
    pub created_at: u64,

    /// Unix timestamp when session was last modified (seconds since epoch).
    pub last_modified: u64,

    /// Optional user-provided description.
    pub description: Option<String>,
}

impl SessionMetadata {
    /// Create new metadata for a problem type.
    pub fn new(problem_type: impl Into<String>, schema_version: u32) -> Self {
        let now = current_timestamp();
        Self {
            problem_type: problem_type.into(),
            schema_version,
            created_at: now,
            last_modified: now,
            description: None,
        }
    }

    /// Create new metadata with a description.
    pub fn with_description(
        problem_type: impl Into<String>,
        schema_version: u32,
        description: impl Into<String>,
    ) -> Self {
        let mut meta = Self::new(problem_type, schema_version);
        meta.description = Some(description.into());
        meta
    }

    /// Update the last_modified timestamp to now.
    pub fn touch(&mut self) {
        self.last_modified = current_timestamp();
    }
}

/// Lightweight operation log entry.
///
/// Captures basic information about operations performed on a session.
/// Intended for debugging and audit trail, not for replay/undo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Unix timestamp of the operation (seconds since epoch).
    pub timestamp: u64,

    /// Operation name (e.g., "init", "optimize", "filter").
    pub operation: String,

    /// Whether the operation succeeded.
    pub success: bool,

    /// Optional notes or error message.
    pub notes: Option<String>,
}

impl LogEntry {
    /// Create a success log entry.
    pub fn success(operation: impl Into<String>) -> Self {
        Self {
            timestamp: current_timestamp(),
            operation: operation.into(),
            success: true,
            notes: None,
        }
    }

    /// Create a success log entry with notes.
    pub fn success_with_notes(operation: impl Into<String>, notes: impl Into<String>) -> Self {
        Self {
            timestamp: current_timestamp(),
            operation: operation.into(),
            success: true,
            notes: Some(notes.into()),
        }
    }

    /// Create a failure log entry.
    pub fn failure(operation: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            timestamp: current_timestamp(),
            operation: operation.into(),
            success: false,
            notes: Some(error.into()),
        }
    }
}

/// Record of an exported result.
///
/// Each call to [`CalibrationSession::export`](super::CalibrationSession::export)
/// creates an export record that is stored in the session's exports collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRecord<E> {
    /// Unix timestamp when the export was created (seconds since epoch).
    pub timestamp: u64,

    /// The exported data.
    pub export: E,

    /// Optional user-provided notes about this export.
    pub notes: Option<String>,
}

impl<E> ExportRecord<E> {
    /// Create an export record.
    pub fn new(export: E) -> Self {
        Self {
            timestamp: current_timestamp(),
            export,
            notes: None,
        }
    }

    /// Create an export record with notes.
    pub fn with_notes(export: E, notes: impl Into<String>) -> Self {
        Self {
            timestamp: current_timestamp(),
            export,
            notes: Some(notes.into()),
        }
    }
}

/// Get the current Unix timestamp in seconds.
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_new() {
        let meta = SessionMetadata::new("test_problem", 1);
        assert_eq!(meta.problem_type, "test_problem");
        assert_eq!(meta.schema_version, 1);
        assert!(meta.created_at > 0);
        assert_eq!(meta.created_at, meta.last_modified);
        assert!(meta.description.is_none());
    }

    #[test]
    fn metadata_with_description() {
        let meta = SessionMetadata::with_description("test_problem", 2, "Test session");
        assert_eq!(meta.problem_type, "test_problem");
        assert_eq!(meta.schema_version, 2);
        assert_eq!(meta.description, Some("Test session".to_string()));
    }

    #[test]
    fn metadata_touch() {
        let mut meta = SessionMetadata::new("test", 1);
        let original = meta.last_modified;
        // Note: in fast execution, touch might not change the timestamp
        // since it's in seconds. This test just verifies it doesn't panic.
        meta.touch();
        assert!(meta.last_modified >= original);
    }

    #[test]
    fn log_entry_success() {
        let entry = LogEntry::success("init");
        assert_eq!(entry.operation, "init");
        assert!(entry.success);
        assert!(entry.notes.is_none());
        assert!(entry.timestamp > 0);
    }

    #[test]
    fn log_entry_success_with_notes() {
        let entry = LogEntry::success_with_notes("optimize", "converged in 10 iterations");
        assert_eq!(entry.operation, "optimize");
        assert!(entry.success);
        assert_eq!(entry.notes, Some("converged in 10 iterations".to_string()));
    }

    #[test]
    fn log_entry_failure() {
        let entry = LogEntry::failure("init", "not enough views");
        assert_eq!(entry.operation, "init");
        assert!(!entry.success);
        assert_eq!(entry.notes, Some("not enough views".to_string()));
    }

    #[test]
    fn export_record_new() {
        let record = ExportRecord::new(42);
        assert_eq!(record.export, 42);
        assert!(record.notes.is_none());
        assert!(record.timestamp > 0);
    }

    #[test]
    fn export_record_with_notes() {
        let record = ExportRecord::with_notes("result", "final calibration");
        assert_eq!(record.export, "result");
        assert_eq!(record.notes, Some("final calibration".to_string()));
    }

    #[test]
    fn export_record_serialization() {
        let record = ExportRecord::with_notes(vec![1.0, 2.0, 3.0], "test data");
        let json = serde_json::to_string(&record).unwrap();
        let restored: ExportRecord<Vec<f64>> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.export, vec![1.0, 2.0, 3.0]);
        assert_eq!(restored.notes, Some("test data".to_string()));
    }
}
