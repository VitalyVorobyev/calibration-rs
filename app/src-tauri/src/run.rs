//! In-process calibration runner for the Run workspace (B3b/B3c).
//!
//! Takes a JSON-encoded [`DatasetSpec`] manifest plus a JSON-encoded
//! per-problem `*Config`, dispatches on the manifest's topology to the
//! right `CalibrationSession`, runs detection (cached) + calibration on
//! a `tauri::async_runtime::spawn_blocking` task, and returns the
//! export plus run metrics. All seven topologies are supported; the
//! laser topologies (ADR 0021) extract laser-line pixels through
//! [`crate::laser::VmLaserExtractor`].
//!
//! Per ADR 0019, ambiguity that the runtime cannot auto-resolve is
//! surfaced as a structured `RunResponse::AskUser` so the Run workspace
//! can render a blocking modal instead of silently guessing.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::State;

use vision_calibration_core::{FrameKind, FrameRef, ImageManifest, PlanarDataset, RigDataset};
use vision_calibration_dataset::{DatasetSpec, Topology};
use vision_calibration_detect::FsDetectionCache;
use vision_calibration_pipeline::dataset_runner::{
    RunError, build_laserline_device_input, build_planar_input, build_rig_extrinsics_input,
    build_rig_handeye_input, build_rig_laserline_device_input, build_single_cam_handeye_input,
};
use vision_calibration_pipeline::laserline_device::{
    LaserlineDeviceConfig, LaserlineDeviceProblem,
    run_calibration as run_laserline_device_calibration,
};
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, run_calibration as run_planar_calibration,
};
use vision_calibration_pipeline::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsProblem, run_calibration as run_rig_extrinsics_calibration,
};
use vision_calibration_pipeline::rig_handeye::{
    RigHandeyeConfig, RigHandeyeProblem, run_calibration as run_rig_handeye_calibration,
};
use vision_calibration_pipeline::rig_laserline_device::{
    RigLaserlineDeviceConfig, RigLaserlineDeviceProblem,
    run_calibration as run_rig_laserline_device_calibration,
};
use vision_calibration_pipeline::scheimpflug_intrinsics::{
    ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsProblem,
    run_calibration as run_scheimpflug_calibration,
};
use vision_calibration_pipeline::session::{CalibrationSession, ProblemType};
use vision_calibration_pipeline::single_cam_handeye::{
    SingleCamHandeyeConfig, SingleCamHandeyeProblem,
    run_calibration as run_single_cam_handeye_calibration,
};

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
        Topology::SingleCamHandeye => {
            run_single_cam_handeye_topology(&spec, config_json, base_dir, &detection_cache, started)
        }
        Topology::LaserlineDevice => {
            run_laserline_topology(&spec, config_json, base_dir, &detection_cache, started)
        }
        Topology::RigLaserlineDevice => {
            run_rig_laserline_topology(&spec, config_json, base_dir, &detection_cache, started)
        }
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

fn run_single_cam_handeye_topology(
    spec: &DatasetSpec,
    config_json: serde_json::Value,
    base_dir: &Path,
    detection_cache: &FsDetectionCache,
    started: Instant,
) -> RunResponse {
    let handeye_run = match build_single_cam_handeye_input(spec, base_dir, detection_cache, false) {
        Ok(r) => r,
        Err(e) => return run_error_to_response(e),
    };
    let cache_used = handeye_run.usable_views > 0; // refined in B3e with hit/miss counts
    let usable_views = handeye_run.usable_views;
    let total_views = handeye_run.total_views;

    let mut export = match run_session::<SingleCamHandeyeProblem>(
        handeye_run.input,
        config_json,
        run_single_cam_handeye_calibration,
    ) {
        Ok(v) => v,
        Err(boxed) => return *boxed,
    };
    // Single camera ⇒ the planar manifest shape (pose = kept-view
    // index, camera = 0) applies directly.
    if let Err(boxed) = splice_image_manifest(
        &mut export,
        planar_image_manifest(&handeye_run.view_paths, base_dir),
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

fn run_laserline_topology(
    spec: &DatasetSpec,
    config_json: serde_json::Value,
    base_dir: &Path,
    detection_cache: &FsDetectionCache,
    started: Instant,
) -> RunResponse {
    let laser_run = match build_laserline_device_input(
        spec,
        base_dir,
        detection_cache,
        &crate::laser::VmLaserExtractor,
        false,
    ) {
        Ok(r) => r,
        Err(e) => return run_error_to_response(e),
    };
    let cache_used = laser_run.usable_views > 0; // refined in B3e with hit/miss counts
    let usable_views = laser_run.usable_views;
    let total_views = laser_run.total_views;

    let mut export =
        match run_session::<LaserlineDeviceProblem>(laser_run.input, config_json, |session| {
            run_laserline_device_calibration(session, None)
        }) {
            Ok(v) => v,
            Err(boxed) => return *boxed,
        };
    // Single camera ⇒ planar manifest shape (pose = kept-view index,
    // camera = 0) for the target frames, plus the aligned laser frames
    // (ADR 0021 §5).
    if let Err(boxed) = splice_image_manifest(
        &mut export,
        laserline_image_manifest(&laser_run.view_paths, &laser_run.laser_paths, base_dir),
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

fn run_rig_laserline_topology(
    spec: &DatasetSpec,
    config_json: serde_json::Value,
    base_dir: &Path,
    detection_cache: &FsDetectionCache,
    started: Instant,
) -> RunResponse {
    let laser_run = match build_rig_laserline_device_input(
        spec,
        base_dir,
        detection_cache,
        &crate::laser::VmLaserExtractor,
        false,
    ) {
        Ok(r) => r,
        Err(e) => return run_error_to_response(e),
    };
    let cache_used = laser_run.usable_views > 0; // refined in B3e with hit/miss counts
    let usable_views = laser_run.usable_views;
    let total_views = laser_run.total_views;
    let view_paths = laser_run.view_paths.clone();
    let laser_paths = laser_run.laser_paths.clone();

    let mut export = match run_session::<RigLaserlineDeviceProblem>(
        laser_run.input,
        config_json,
        run_rig_laserline_device_calibration,
    ) {
        Ok(v) => v,
        Err(boxed) => return *boxed,
    };
    if let Err(boxed) = splice_image_manifest(
        &mut export,
        rig_laserline_image_manifest(&view_paths, &laser_paths, base_dir),
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

/// Single-camera laser manifest: target frames (kind omitted in JSON)
/// plus one laser frame per kept view (ADR 0021 §5). `view_paths` and
/// `laser_paths` are aligned by kept-view index.
fn laserline_image_manifest(
    view_paths: &[PathBuf],
    laser_paths: &[PathBuf],
    base_dir: &Path,
) -> Result<ImageManifest, String> {
    let base_abs = canonical_base(base_dir)?;
    let mut frames = Vec::with_capacity(view_paths.len() + laser_paths.len());
    for (pose, path) in view_paths.iter().enumerate() {
        frames.push(frame_ref(pose, 0, path, &base_abs)?);
    }
    for (pose, path) in laser_paths.iter().enumerate() {
        frames.push(frame_ref_of_kind(pose, 0, path, &base_abs, FrameKind::Laser)?);
    }
    Ok(manifest_from_frames(frames))
}

/// Rig laser manifest: target frames plus laser frames per
/// `(view, camera)` slot that contributed a usable laser observation.
fn rig_laserline_image_manifest(
    view_paths: &[Vec<Option<PathBuf>>],
    laser_paths: &[Vec<Option<PathBuf>>],
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
    for (pose, cameras) in laser_paths.iter().enumerate() {
        for (camera, maybe_path) in cameras.iter().enumerate() {
            if let Some(path) = maybe_path {
                frames.push(frame_ref_of_kind(
                    pose,
                    camera,
                    path,
                    &base_abs,
                    FrameKind::Laser,
                )?);
            }
        }
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

fn frame_ref_of_kind(
    pose: usize,
    camera: usize,
    path: &Path,
    base_abs: &Path,
    kind: FrameKind,
) -> Result<FrameRef, String> {
    let mut frame = frame_ref(pose, camera, path, base_abs)?;
    frame.kind = kind;
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
        RunError::LaserPairing { .. } => "view_pairing",
        RunError::LaserExtraction { .. } => "detection",
        RunError::InsufficientLaserViews { .. } => "insufficient_views",
        RunError::UpstreamCalibration { .. } => "upstream_calibration",
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
    fn laser_topologies_dispatch_and_fail_validation_without_laser_images() {
        // The laser arms are wired (ADR 0021); a manifest without
        // laser_images must now fail *validation*, not dispatch.
        for (topology, num_cams) in [("laserline_device", 1), ("rig_laserline_device", 2)] {
            let cameras: Vec<_> = (0..num_cams)
                .map(|i| json!({"id": format!("cam{i}"), "images": {"kind": "glob", "pattern": "*.png"}}))
                .collect();
            let manifest = json!({
                "version": 1,
                "topology": topology,
                "cameras": cameras,
                "target": {"kind": "chessboard", "rows": 9, "cols": 6, "square_size_m": 0.025},
                "robot_poses": {"path": "poses.txt", "format": "rowmajor4x4"},
                "pose_convention": {
                    "transform": "t_base_tcp",
                    "rotation_format": "matrix4x4_row_major",
                    "translation_units": "m",
                },
            });
            let response = run_blocking(manifest, json!({}), "/tmp");
            match response {
                RunResponse::ValidationFailed { message } => {
                    assert!(message.contains("laser_images"), "{topology}: {message}");
                }
                other => panic!("expected ValidationFailed for {topology}, got {other:?}"),
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
    fn laser_manifests_emit_both_frame_kinds() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        std::fs::create_dir_all(base.join("cam0")).unwrap();
        for name in ["cam0/t0.png", "cam0/t1.png", "cam0/l0.png", "cam0/l1.png"] {
            std::fs::write(base.join(name), b"x").unwrap();
        }

        // Single-camera: target + laser frames aligned by kept-view index.
        let manifest = laserline_image_manifest(
            &[base.join("cam0/t0.png"), base.join("cam0/t1.png")],
            &[base.join("cam0/l0.png"), base.join("cam0/l1.png")],
            base,
        )
        .unwrap();
        assert_eq!(manifest.target_frames().count(), 2);
        assert_eq!(manifest.laser_frames().count(), 2);
        let laser = manifest.frame_of_kind(1, 0, FrameKind::Laser).unwrap();
        assert_eq!(laser.path, PathBuf::from("cam0/l1.png"));
        // `frame()` stays target-only despite the laser entry for (1, 0).
        assert_eq!(
            manifest.frame(1, 0).unwrap().path,
            PathBuf::from("cam0/t1.png")
        );

        // Rig: None laser slots produce no frames.
        let manifest = rig_laserline_image_manifest(
            &[vec![Some(base.join("cam0/t0.png"))]],
            &[vec![None]],
            base,
        )
        .unwrap();
        assert_eq!(manifest.laser_frames().count(), 0);
        assert_eq!(manifest.target_frames().count(), 1);
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

    /// Run one local-data preset end-to-end through `run_blocking` and
    /// return the success payload. Skips (returns `None`) when the
    /// dataset is absent — `data/stereo*` are gitignored, local-only.
    fn run_local_preset(rel_manifest: &str) -> Option<RunSuccess> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifest_path = repo_root.join(rel_manifest);
        if !manifest_path.exists() {
            eprintln!("skipping: {} not present", manifest_path.display());
            return None;
        }
        let raw = std::fs::read_to_string(&manifest_path).unwrap();
        let manifest: serde_json::Value =
            serde_json::to_value(toml::from_str::<toml::Value>(&raw).unwrap()).unwrap();
        let topology = manifest["topology"].as_str().unwrap().to_string();
        let config = default_config_cmd(topology).unwrap();
        let dir = manifest_path
            .parent()
            .unwrap()
            .to_string_lossy()
            .to_string();
        match run_blocking(manifest, config, &dir) {
            RunResponse::Ok(success) => Some(success),
            other => panic!("{rel_manifest}: expected Ok, got {other:?}"),
        }
    }

    /// Local-only end-to-end acceptance over the gitignored bundled
    /// datasets. Run manually:
    /// `cargo test -p calibration-diagnose -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the local data/stereo + data/stereo_charuco datasets"]
    fn local_presets_end_to_end() {
        if let Some(s) = run_local_preset("data/stereo/dataset_rig.toml") {
            assert!(
                s.usable_views >= 10,
                "stereo rig: {} usable",
                s.usable_views
            );
            assert!(s.export["cameras"].is_array(), "rig export has cameras[]");
            assert!(s.export["image_manifest"]["frames"].is_array());
        }
        if let Some(s) = run_local_preset("data/stereo_charuco/dataset_cam1.toml") {
            assert!(
                s.usable_views >= 10,
                "charuco cam1: {} usable",
                s.usable_views
            );
            assert!(s.export["image_manifest"]["frames"].is_array());
        }
        if let Some(s) = run_local_preset("data/stereo_charuco/dataset_rig.toml") {
            assert!(
                s.usable_views >= 10,
                "charuco rig: {} usable",
                s.usable_views
            );
            assert!(s.export["cameras"].is_array(), "rig export has cameras[]");
        }
    }

    /// Two-stage laser acceptance over the local rtv3d dataset
    /// (6-camera Scheimpflug rig, tiled strips, ChArUco + laser
    /// frames): rig hand-eye first, then `RigLaserlineDevice` with the
    /// stage-1 export as the frozen upstream — exactly the user
    /// workflow the laser slice ships (ADR 0021). Skips when
    /// `privatedata/rtv3d` is absent (gitignored, local-only).
    #[test]
    #[ignore = "needs the local privatedata/rtv3d dataset; 6-cam detection + two solves"]
    fn rtv3d_laser_end_to_end() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let data_dir = repo_root.join("privatedata/rtv3d");
        if !data_dir.exists() {
            eprintln!("skipping: {} not present", data_dir.display());
            return;
        }
        let read_manifest = |name: &str| -> serde_json::Value {
            let raw = std::fs::read_to_string(data_dir.join(name)).unwrap();
            serde_json::to_value(toml::from_str::<toml::Value>(&raw).unwrap()).unwrap()
        };
        let dir = data_dir.to_string_lossy().to_string();

        // ── Stage 1: rig hand-eye (Scheimpflug, EyeToHand) ────────────────
        // Same overrides the rtv3d_rig example settled on; every one is
        // plain config JSON, i.e. reachable from the app's ConfigForm.
        let mut config = default_config_cmd("rig_handeye".into()).unwrap();
        config["sensor"] = json!({"kind": "Scheimpflug"});
        config["handeye_init"]["handeye_mode"] = json!("EyeToHand");
        config["solver"]["max_iters"] = json!(200);
        config["solver"]["robust_loss"] = json!({"Huber": {"scale": 1.0}});
        let handeye = match run_blocking(read_manifest("dataset_rig_handeye.toml"), config, &dir) {
            RunResponse::Ok(s) => s,
            other => panic!("rig handeye stage: expected Ok, got {other:?}"),
        };
        assert!(
            handeye.usable_views >= 10,
            "rig handeye: {} usable views",
            handeye.usable_views
        );
        // The hand-eye stage lands at ~1.5-2 px on rtv3d (the example's
        // own hand-eye stage shows 1.6-2.1 px per camera; sub-pixel
        // needs the downstream joint BA). The oracle is ~2.2 px.
        let mean_px = handeye.export["mean_reproj_error"].as_f64().unwrap();
        assert!(mean_px < 2.2, "rig handeye mean reproj {mean_px:.3} px");
        std::fs::write(
            data_dir.join("rig_handeye_export.json"),
            serde_json::to_string(&handeye.export).unwrap(),
        )
        .unwrap();

        // ── Stage 2: rig laserline over the frozen export ─────────────────
        // PointToPlane keeps the residual in metres (the example's
        // stage-3 choice), so the σ gate below is unit-meaningful.
        let mut laser_config = default_config_cmd("rig_laserline_device".into()).unwrap();
        laser_config["max_iters"] = json!(200);
        laser_config["laser_residual_type"] = json!("PointToPlane");
        let laser = match run_blocking(read_manifest("dataset_laser.toml"), laser_config, &dir) {
            RunResponse::Ok(s) => s,
            other => panic!("rig laserline stage: expected Ok, got {other:?}"),
        };
        // Persist the stage-2 export so the app can open it (laser
        // overlay + 3D planes). Stop-gap until B3e ships in-app
        // "save export to file".
        std::fs::write(
            data_dir.join("rig_laserline_export.json"),
            serde_json::to_string(&laser.export).unwrap(),
        )
        .unwrap();
        let planes = laser.export["laser_planes_cam"].as_array().unwrap();
        assert_eq!(planes.len(), 6, "one laser plane per camera");
        let stats = laser.export["per_camera_stats"].as_array().unwrap();
        assert_eq!(stats.len(), 6);
        for (cam, s) in stats.iter().enumerate() {
            // Plane-fit sanity gate, not an oracle-beating gate: with
            // the upstream frozen at ~1.5 px the plane fit lands near
            // ~1 mm; sub-0.1 mm needs the joint BA (V5, out of scope
            // here). 2 mm flags a broken plane / wrong chain.
            let laser_err = s["mean_laser_error"].as_f64().unwrap();
            eprintln!("cam {cam}: mean point-to-plane {:.4} mm", laser_err * 1e3);
            assert!(
                laser_err < 2e-3,
                "cam {cam}: mean laser residual {laser_err:.6} m"
            );
        }
        // The manifest must carry both frame kinds (ADR 0021 §5): the
        // target frames for the reprojection overlay and the laser
        // frames Diagnose's laser view plots residuals onto.
        let frames = laser.export["image_manifest"]["frames"].as_array().unwrap();
        let n_laser = frames.iter().filter(|f| f["kind"] == "laser").count();
        let n_target = frames.len() - n_laser;
        assert!(n_target > 0, "manifest carries target frames");
        assert!(
            n_laser > 0,
            "manifest carries laser-kind frames ({} total)",
            frames.len()
        );
        let laser_residuals = laser.export["per_feature_residuals"]["laser"]
            .as_array()
            .unwrap();
        assert!(
            !laser_residuals.is_empty(),
            "laser export carries per-pixel laser residuals"
        );
        eprintln!("manifest: {n_target} target + {n_laser} laser frames");
        eprintln!(
            "rtv3d laser E2E: handeye {mean_px:.3} px over {} views; laser stage {} views in {} ms",
            handeye.usable_views, laser.usable_views, laser.duration_ms
        );
    }

    /// End-to-end hand-eye run over the committed `data/kuka_1` dataset
    /// (30 chessboard views + rowmajor4x4 robot poses). Ignored for
    /// time, not data availability — kuka_1 ships with the repo.
    #[test]
    #[ignore = "full detection + solve takes tens of seconds; run with --ignored"]
    fn kuka_handeye_end_to_end() {
        let s = run_local_preset("data/kuka_1/dataset.toml")
            .expect("data/kuka_1 is committed; the manifest must be present");
        assert!(s.usable_views >= 20, "kuka_1: {} usable", s.usable_views);
        assert_eq!(s.total_views, 30);
        let mean_px = s.export["mean_reproj_error"].as_f64().unwrap();
        assert!(
            mean_px < 2.0,
            "kuka_1 mean reprojection error {mean_px:.3} px (expected ~1.2)"
        );
        assert!(s.export["gripper_se3_camera"].is_object(), "eye-in-hand");
        let frames = s.export["image_manifest"]["frames"].as_array().unwrap();
        assert_eq!(frames.len(), s.usable_views);
    }
}
