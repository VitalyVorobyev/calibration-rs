//! In-process calibration runner for the Run workspace (B3b/B3c).
//!
//! Takes a JSON-encoded [`DatasetSpec`] manifest plus a JSON-encoded
//! per-problem `*Config`, dispatches on the manifest's topology to the
//! right `CalibrationSession`, runs detection (cached) + calibration on
//! a `tauri::async_runtime::spawn_blocking` task, and returns the
//! export plus run metrics. Supported topologies: `PlanarIntrinsics`,
//! `ScheimpflugIntrinsics`, `RigExtrinsics`, `RigHandeye`. The laser
//! topologies await the laser-frame manifest design; `SingleCamHandeye`
//! follows in B3c-2.
//!
//! Per ADR 0019, ambiguity that the runtime cannot auto-resolve is
//! surfaced as a structured `RunResponse::AskUser` so the Run workspace
//! can render a blocking modal instead of silently guessing.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::State;

use vision_calibration_core::{FrameRef, ImageManifest, PlanarDataset, RigDataset};
use vision_calibration_dataset::{DatasetSpec, Topology};
use vision_calibration_detect::FsDetectionCache;
use vision_calibration_pipeline::dataset_runner::{
    RunError, build_planar_input, build_rig_extrinsics_input, build_rig_handeye_input,
};
use vision_calibration_pipeline::laserline_device::LaserlineDeviceConfig;
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, run_calibration as run_planar_calibration,
};
use vision_calibration_pipeline::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsProblem, run_calibration as run_rig_extrinsics_calibration,
};
use vision_calibration_pipeline::rig_handeye::{
    RigHandeyeConfig, RigHandeyeProblem, run_calibration as run_rig_handeye_calibration,
};
use vision_calibration_pipeline::rig_laserline_device::RigLaserlineDeviceConfig;
use vision_calibration_pipeline::scheimpflug_intrinsics::{
    ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsProblem,
    run_calibration as run_scheimpflug_calibration,
};
use vision_calibration_pipeline::session::{CalibrationSession, ProblemType};
use vision_calibration_pipeline::single_cam_handeye::SingleCamHandeyeConfig;

use crate::export_cache::ExportCache;

/// Successful calibration run.
#[derive(Debug, Serialize, Deserialize)]
pub struct RunSuccess {
    /// Final export JSON, ready to drop into the diagnose viewer.
    pub export: serde_json::Value,
    /// Total wall-clock time for detection + calibration.
    pub duration_ms: u64,
    /// Number of usable views after detection. For single-camera
    /// topologies: views with >= 4 features. For rig topologies: views
    /// where at least one camera reached >= 4 features.
    pub usable_views: usize,
    /// Total number of views attempted (images for single-camera
    /// topologies, paired rig views for rig topologies).
    pub total_views: usize,
    /// Whether at least one detection was served from the cache.
    /// (Sanity check for "second run is faster"; precise per-image
    /// hit/miss counts come in B3e.)
    pub cache_used: bool,
}

/// Tagged response shape so the React layer can match on `kind` rather
/// than parsing free-form error strings.
#[derive(Debug, Serialize)]
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

/// Tauri command: default `*Config` JSON for a topology.
///
/// Single source of truth for the Run workspace's config defaults —
/// the TS side never hand-copies Rust default values.
#[tauri::command]
pub fn default_config_cmd(topology: String) -> Result<serde_json::Value, String> {
    let value = match topology.as_str() {
        "planar_intrinsics" => serde_json::to_value(PlanarIntrinsicsConfig::default()),
        "scheimpflug_intrinsics" => serde_json::to_value(ScheimpflugIntrinsicsConfig::default()),
        "single_cam_handeye" => serde_json::to_value(SingleCamHandeyeConfig::default()),
        "laserline_device" => serde_json::to_value(LaserlineDeviceConfig::default()),
        "rig_extrinsics" => serde_json::to_value(RigExtrinsicsConfig::default()),
        "rig_handeye" => serde_json::to_value(RigHandeyeConfig::default()),
        "rig_laserline_device" => serde_json::to_value(RigLaserlineDeviceConfig::default()),
        other => return Err(format!("unknown topology {other:?}")),
    };
    value.map_err(|e| format!("default config serialization failed: {e}"))
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
    let base_dir = Path::new(manifest_dir);
    let detection_cache = FsDetectionCache::new(detection_cache_root(&spec, base_dir));

    match spec.topology {
        Topology::PlanarIntrinsics => run_planar_topology::<PlanarIntrinsicsProblem>(
            &spec,
            config_json,
            base_dir,
            &detection_cache,
            started,
            run_planar_calibration,
        ),
        Topology::ScheimpflugIntrinsics => run_planar_topology::<ScheimpflugIntrinsicsProblem>(
            &spec,
            config_json,
            base_dir,
            &detection_cache,
            started,
            |session| run_scheimpflug_calibration(session, None),
        ),
        Topology::RigExtrinsics => run_rig_topology::<RigExtrinsicsProblem, _>(
            &spec,
            config_json,
            base_dir,
            &detection_cache,
            started,
            build_rig_extrinsics_input,
            run_rig_extrinsics_calibration,
        ),
        Topology::RigHandeye => run_rig_topology::<RigHandeyeProblem, _>(
            &spec,
            config_json,
            base_dir,
            &detection_cache,
            started,
            build_rig_handeye_input,
            run_rig_handeye_calibration,
        ),
        other @ (Topology::SingleCamHandeye
        | Topology::LaserlineDevice
        | Topology::RigLaserlineDevice) => RunResponse::Failed {
            category: "unsupported_topology".into(),
            message: format!(
                "topology {other:?} is not wired into the Run workspace yet \
                 (the laser topologies land with the laser-frame manifest \
                 design; single_cam_handeye follows in B3c-2)"
            ),
        },
    }
}

/// Drive one `CalibrationSession` from parsed input to serialized
/// export. Generic plumbing shared by every topology arm; the
/// per-problem differences (input building, `run_calibration` wrapper,
/// image-manifest shape) stay in the callers.
fn run_session<P>(
    input: P::Input,
    config_json: serde_json::Value,
    run: impl FnOnce(&mut CalibrationSession<P>) -> Result<(), vision_calibration_pipeline::Error>,
) -> Result<serde_json::Value, Box<RunResponse>>
where
    P: ProblemType,
{
    let config: P::Config = serde_json::from_value(config_json).map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "config_parse".into(),
            message: format!("config parse failed: {e}"),
        })
    })?;
    let mut session = CalibrationSession::<P>::new();
    session.set_input(input).map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "session_set_input".into(),
            message: format!("set_input failed: {e}"),
        })
    })?;
    session.set_config(config).map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "session_set_config".into(),
            message: format!("set_config failed: {e}"),
        })
    })?;
    run(&mut session).map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "calibration_failed".into(),
            message: format!("calibration failed: {e}"),
        })
    })?;
    let export = session.export().map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "session_export".into(),
            message: format!("export failed: {e}"),
        })
    })?;
    serde_json::to_value(&export).map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "export_serialize".into(),
            message: format!("export serialization failed: {e}"),
        })
    })
}

/// Splice an image manifest into a serialized export. Every supported
/// `*Export` carries an `image_manifest: Option<ImageManifest>` field
/// with `#[serde(default)]`, so setting it on the JSON value is
/// round-trip-safe and avoids a cross-export trait.
fn splice_image_manifest(
    export: &mut serde_json::Value,
    manifest: Result<ImageManifest, String>,
) -> Result<(), Box<RunResponse>> {
    let manifest = manifest.map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "image_manifest".into(),
            message: e,
        })
    })?;
    let value = serde_json::to_value(&manifest).map_err(|e| {
        Box::new(RunResponse::Failed {
            category: "image_manifest".into(),
            message: format!("image manifest serialization failed: {e}"),
        })
    })?;
    export["image_manifest"] = value;
    Ok(())
}

fn run_planar_topology<P>(
    spec: &DatasetSpec,
    config_json: serde_json::Value,
    base_dir: &Path,
    detection_cache: &FsDetectionCache,
    started: Instant,
    run: impl FnOnce(&mut CalibrationSession<P>) -> Result<(), vision_calibration_pipeline::Error>,
) -> RunResponse
where
    P: ProblemType<Input = PlanarDataset>,
{
    let planar_run = match build_planar_input(spec, base_dir, detection_cache, false) {
        Ok(r) => r,
        Err(e) => return run_error_to_response(e),
    };
    let cache_used = planar_run.usable_views > 0; // refined in B3e with hit/miss counts

    let mut export = match run_session::<P>(planar_run.dataset.clone(), config_json, run) {
        Ok(v) => v,
        Err(boxed) => return *boxed,
    };
    if let Err(boxed) = splice_image_manifest(
        &mut export,
        planar_image_manifest(&planar_run.view_paths, base_dir),
    ) {
        return *boxed;
    }

    RunResponse::Ok(RunSuccess {
        export,
        duration_ms: started.elapsed().as_millis() as u64,
        usable_views: planar_run.usable_views,
        total_views: planar_run.total_views,
        cache_used,
    })
}

type RigBuilder<Meta> =
    fn(
        &DatasetSpec,
        &Path,
        &dyn vision_calibration_detect::DetectionCache,
        bool,
    ) -> Result<vision_calibration_pipeline::dataset_runner::RigRunResult<Meta>, RunError>;

fn run_rig_topology<P, Meta>(
    spec: &DatasetSpec,
    config_json: serde_json::Value,
    base_dir: &Path,
    detection_cache: &FsDetectionCache,
    started: Instant,
    build: RigBuilder<Meta>,
    run: impl FnOnce(&mut CalibrationSession<P>) -> Result<(), vision_calibration_pipeline::Error>,
) -> RunResponse
where
    P: ProblemType<Input = RigDataset<Meta>>,
{
    let rig_run = match build(spec, base_dir, detection_cache, false) {
        Ok(r) => r,
        Err(e) => return run_error_to_response(e),
    };
    let cache_used = rig_run.usable_views > 0; // refined in B3e with hit/miss counts
    let usable_views = rig_run.usable_views;
    let total_views = rig_run.total_views;

    let mut export = match run_session::<P>(rig_run.dataset, config_json, run) {
        Ok(v) => v,
        Err(boxed) => return *boxed,
    };
    if let Err(boxed) = splice_image_manifest(
        &mut export,
        rig_image_manifest(&rig_run.view_paths, base_dir),
    ) {
        return *boxed;
    }

    RunResponse::Ok(RunSuccess {
        export,
        duration_ms: started.elapsed().as_millis() as u64,
        usable_views,
        total_views,
        cache_used,
    })
}

fn planar_image_manifest(view_paths: &[PathBuf], base_dir: &Path) -> Result<ImageManifest, String> {
    let base_abs = canonical_base(base_dir)?;
    let mut frames = Vec::with_capacity(view_paths.len());
    for (pose, path) in view_paths.iter().enumerate() {
        frames.push(frame_ref(pose, 0, path, &base_abs)?);
    }
    Ok(manifest_from_frames(frames))
}

/// Build a rig manifest from `view_paths[view][camera]` — one frame
/// per `(view, camera)` slot that contributed a usable observation.
fn rig_image_manifest(
    view_paths: &[Vec<Option<PathBuf>>],
    base_dir: &Path,
) -> Result<ImageManifest, String> {
    let base_abs = canonical_base(base_dir)?;
    let mut frames = Vec::new();
    for (pose, cameras) in view_paths.iter().enumerate() {
        for (camera, maybe_path) in cameras.iter().enumerate() {
            if let Some(path) = maybe_path {
                frames.push(frame_ref(pose, camera, path, &base_abs)?);
            }
        }
    }
    Ok(manifest_from_frames(frames))
}

fn canonical_base(base_dir: &Path) -> Result<PathBuf, String> {
    base_dir
        .canonicalize()
        .map_err(|e| format!("canonicalize manifest dir {}: {e}", base_dir.display()))
}

fn frame_ref(pose: usize, camera: usize, path: &Path, base_abs: &Path) -> Result<FrameRef, String> {
    let abs = path
        .canonicalize()
        .map_err(|e| format!("canonicalize image path {}: {e}", path.display()))?;
    let rel = abs.strip_prefix(base_abs).map_err(|_| {
        format!(
            "accepted image {} is not under manifest dir {}",
            abs.display(),
            base_abs.display()
        )
    })?;
    let mut frame = FrameRef::default();
    frame.pose = pose;
    frame.camera = camera;
    frame.path = rel.to_path_buf();
    Ok(frame)
}

fn manifest_from_frames(frames: Vec<FrameRef>) -> ImageManifest {
    let mut manifest = ImageManifest::default();
    manifest.root = PathBuf::from(".");
    manifest.frames = frames;
    manifest
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
        RunError::InvalidTargetConfig { .. } => "invalid_target_config",
        RunError::EmptyImageMatch { .. } => "empty_image_match",
        RunError::BadGlob { .. } => "bad_glob",
        RunError::ViewCountMismatch { .. } => "view_pairing",
        RunError::PoseCountMismatch { .. } => "view_pairing",
        RunError::PoseParse { .. } => "pose_parse",
        RunError::BadPairingRegex { .. } => "view_pairing",
        RunError::PairingTokenMismatch { .. } => "view_pairing",
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn unsupported_topologies_name_the_roadmap() {
        for topology in [
            "single_cam_handeye",
            "laserline_device",
            "rig_laserline_device",
        ] {
            let manifest = json!({
                "version": 1,
                "topology": topology,
                "cameras": [
                    {"id": "cam0", "images": {"kind": "glob", "pattern": "*.png"}},
                ],
                "target": {"kind": "chessboard", "rows": 9, "cols": 6, "square_size_m": 0.025},
            });
            let response = run_blocking(manifest, json!({}), "/tmp");
            match response {
                RunResponse::Failed { category, message } => {
                    assert_eq!(category, "unsupported_topology");
                    assert!(message.contains("not wired"), "got: {message}");
                }
                _ => panic!("expected Failed for {topology}"),
            }
        }
    }

    #[test]
    fn default_config_round_trips_per_topology() {
        for topology in [
            "planar_intrinsics",
            "scheimpflug_intrinsics",
            "single_cam_handeye",
            "laserline_device",
            "rig_extrinsics",
            "rig_handeye",
            "rig_laserline_device",
        ] {
            let value = default_config_cmd(topology.to_string())
                .unwrap_or_else(|e| panic!("default config for {topology}: {e}"));
            assert!(value.is_object(), "{topology} default must be an object");
        }
        assert!(default_config_cmd("not_a_topology".into()).is_err());
    }

    #[test]
    fn rig_manifest_skips_none_slots_and_indexes_cameras() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        std::fs::create_dir_all(base.join("cam0")).unwrap();
        std::fs::create_dir_all(base.join("cam1")).unwrap();
        std::fs::write(base.join("cam0/a.png"), b"x").unwrap();
        std::fs::write(base.join("cam1/a.png"), b"x").unwrap();
        std::fs::write(base.join("cam0/b.png"), b"x").unwrap();

        let view_paths = vec![
            vec![Some(base.join("cam0/a.png")), Some(base.join("cam1/a.png"))],
            vec![Some(base.join("cam0/b.png")), None], // cam1 missed view 1
        ];
        let manifest = rig_image_manifest(&view_paths, base).unwrap();
        assert_eq!(manifest.frames.len(), 3);
        let f = &manifest.frames[2];
        assert_eq!((f.pose, f.camera), (1, 0));
        assert_eq!(f.path, PathBuf::from("cam0/b.png"));
        // No frame for the None slot.
        assert!(
            !manifest.frames.iter().any(|f| f.pose == 1 && f.camera == 1),
            "None slots must not produce frames"
        );
    }

    #[test]
    fn image_manifest_splice_lands_in_export_json() {
        let mut export = json!({"params": {}, "report": {}});
        let mut manifest = ImageManifest::default();
        manifest.root = PathBuf::from(".");
        splice_image_manifest(&mut export, Ok(manifest)).unwrap();
        assert!(export["image_manifest"].is_object());
        assert_eq!(export["image_manifest"]["root"], ".");
    }
}
