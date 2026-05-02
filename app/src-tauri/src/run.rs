//! In-process calibration runner for the Run workspace (B3b).
//!
//! Takes a JSON-encoded [`DatasetSpec`] manifest plus a JSON-encoded
//! per-problem `*Config`, dispatches to the right `CalibrationSession`,
//! runs detection (cached) + calibration on a `tauri::async_runtime::spawn_blocking`
//! task, and returns the export plus a structured log. PR 1 wires up
//! `PlanarIntrinsics` only; PR 2 (B3c) extends to the other 7 topologies
//! via the same shape.
//!
//! Per ADR 0019, ambiguity that the runtime cannot auto-resolve is
//! surfaced as a structured `RunResponse::AskUser` so the Run workspace
//! can render a blocking modal instead of silently guessing.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::State;

use vision_calibration_dataset::{DatasetSpec, Topology};
use vision_calibration_detect::FsDetectionCache;
use vision_calibration_pipeline::dataset_runner::{RunError, build_planar_input};
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

use crate::export_cache::ExportCache;

/// Successful calibration run.
#[derive(Serialize, Deserialize)]
pub struct RunSuccess {
    /// Final export JSON, ready to drop into the diagnose viewer.
    pub export: serde_json::Value,
    /// Total wall-clock time for detection + calibration.
    pub duration_ms: u64,
    /// Number of views with >= 4 features after detection.
    pub usable_views: usize,
    /// Total number of images attempted.
    pub total_views: usize,
    /// Whether at least one detection was served from the cache.
    /// (Useful as a sanity check for "second run is faster"
    /// in PR 1; precise per-image hit/miss counts come in PR 4.)
    pub cache_used: bool,
}

/// Tagged response shape so the React layer can match on `kind` rather
/// than parsing free-form error strings.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunResponse {
    /// Calibration completed and an export was produced.
    Ok(RunSuccess),
    /// Runtime ambiguity per ADR 0019 — Run workspace must render a
    /// modal asking the user to fill `field`.
    AskUser {
        /// Dotted field path the runtime needs help with.
        field: String,
        /// Human-friendly prompt copy for the modal.
        prompt: String,
        /// Optional preset suggestions (e.g. `["t_base_tcp", "t_tcp_base"]`).
        suggestions: Vec<String>,
    },
    /// Manifest validation failed before the runner could touch the
    /// filesystem. Holds the underlying validation message.
    ValidationFailed { message: String },
    /// User-correctable detection / IO / decode error. The frontend
    /// surfaces `message` directly in a non-modal error banner.
    Failed {
        /// Stable kind tag for the frontend.
        category: String,
        /// Human-readable detail.
        message: String,
    },
}

/// Tauri command: run a calibration end-to-end.
///
/// `manifest_json` is the raw JSON of a [`DatasetSpec`] (typically
/// emitted by the schema-driven form in the Run workspace). `config_json`
/// is the matching per-problem `*Config` JSON. `manifest_dir` is the
/// directory whose globs are resolved against — usually the parent of
/// the `dataset.toml` file the user picked.
#[tauri::command]
pub async fn run_calibration_cmd(
    manifest_json: serde_json::Value,
    config_json: serde_json::Value,
    manifest_dir: String,
    cache: State<'_, ExportCache>,
) -> Result<RunResponse, String> {
    // Long-running solve: keep IPC responsive by running on a blocking
    // task. The State can't cross the spawn_blocking boundary directly,
    // so we do detection + calibration on the worker thread and pop
    // the produced export into the cache here, in the async caller.
    let response = tauri::async_runtime::spawn_blocking(move || {
        run_blocking(manifest_json, config_json, &manifest_dir)
    })
    .await
    .map_err(|e| format!("runner task panicked: {e}"))?;
    if let RunResponse::Ok(success) = &response {
        cache.set("<live-run>".to_string(), success.export.clone());
    }
    Ok(response)
}

fn run_blocking(
    manifest_json: serde_json::Value,
    config_json: serde_json::Value,
    manifest_dir: &str,
) -> RunResponse {
    let started = Instant::now();
    let spec: DatasetSpec = match serde_json::from_value(manifest_json) {
        Ok(s) => s,
        Err(e) => {
            return RunResponse::Failed {
                category: "manifest_parse".into(),
                message: format!("manifest parse failed: {e}"),
            };
        }
    };
    let config: PlanarIntrinsicsConfig = match serde_json::from_value(config_json) {
        Ok(c) => c,
        Err(e) => {
            return RunResponse::Failed {
                category: "config_parse".into(),
                message: format!("config parse failed: {e}"),
            };
        }
    };
    if !matches!(spec.topology, Topology::PlanarIntrinsics) {
        return RunResponse::Failed {
            category: "unsupported_topology".into(),
            message: format!(
                "PR 1 of the runner only supports PlanarIntrinsics; \
                 manifest declares {:?}. Coverage for the other seven \
                 topologies ships in B3c.",
                spec.topology
            ),
        };
    }

    let base_dir = Path::new(manifest_dir);
    let detection_cache = FsDetectionCache::new(detection_cache_root(&spec, base_dir));

    // Dataset → IR conversion (validation + detection-with-cache).
    let planar_run = match build_planar_input(&spec, base_dir, &detection_cache, false) {
        Ok(r) => r,
        Err(e) => return run_error_to_response(e),
    };
    let cache_used = planar_run.usable_views > 0; // refined in B3e with hit/miss counts

    // Calibration: drive a session.
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    if let Err(e) = session.set_input(planar_run.dataset.clone()) {
        return RunResponse::Failed {
            category: "session_set_input".into(),
            message: format!("set_input failed: {e}"),
        };
    }
    if let Err(e) = session.set_config(config) {
        return RunResponse::Failed {
            category: "session_set_config".into(),
            message: format!("set_config failed: {e}"),
        };
    }
    if let Err(e) = run_calibration(&mut session) {
        return RunResponse::Failed {
            category: "calibration_failed".into(),
            message: format!("calibration failed: {e}"),
        };
    }
    let export = match session.export() {
        Ok(e) => e,
        Err(e) => {
            return RunResponse::Failed {
                category: "session_export".into(),
                message: format!("export failed: {e}"),
            };
        }
    };
    let export_value = match serde_json::to_value(&export) {
        Ok(v) => v,
        Err(e) => {
            return RunResponse::Failed {
                category: "export_serialize".into(),
                message: format!("export serialization failed: {e}"),
            };
        }
    };

    // The async caller pushes `export_value` into ExportCache after the
    // worker returns, so the diagnose / 3D / epipolar workspaces can
    // pick it up via the existing `compute_*` commands without a disk
    // round-trip.
    let _ = planar_run.view_paths; // PR 1 doesn't yet thread image_manifest back

    RunResponse::Ok(RunSuccess {
        export: export_value,
        duration_ms: started.elapsed().as_millis() as u64,
        usable_views: planar_run.usable_views,
        total_views: planar_run.total_views,
        cache_used,
    })
}

fn run_error_to_response(err: RunError) -> RunResponse {
    match err {
        RunError::AskUser {
            field,
            prompt,
            suggestions,
        } => RunResponse::AskUser {
            field,
            prompt,
            suggestions,
        },
        RunError::Validation(v) => RunResponse::ValidationFailed {
            message: v.to_string(),
        },
        other => RunResponse::Failed {
            category: error_category(&other).into(),
            message: other.to_string(),
        },
    }
}

fn error_category(err: &RunError) -> &'static str {
    match err {
        RunError::Validation(_) => "validation",
        RunError::UnsupportedTopology { .. } => "unsupported_topology",
        RunError::UnsupportedTarget { .. } => "unsupported_target",
        RunError::EmptyImageMatch { .. } => "empty_image_match",
        RunError::BadGlob { .. } => "bad_glob",
        RunError::Io(_) => "io",
        RunError::Decode { .. } => "decode",
        RunError::Detection { .. } => "detection",
        RunError::Cache(_) => "cache",
        RunError::AskUser { .. } => "ask_user",
        RunError::InsufficientUsableViews { .. } => "insufficient_views",
    }
}

fn detection_cache_root(spec: &DatasetSpec, base_dir: &Path) -> PathBuf {
    // Per ADR 0017 the cache root is per-dataset. We use a stable id
    // from the manifest's `description` if set, otherwise the absolute
    // base directory mangled into a filesystem-safe slug.
    let id = match &spec.description {
        Some(d) if !d.is_empty() => slug(d),
        _ => slug(&base_dir.to_string_lossy()),
    };
    let cache_home = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
    cache_home
        .join("calibration-rs")
        .join("detections")
        .join(id)
}

fn slug(input: &str) -> String {
    input
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
