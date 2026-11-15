//! Most-recently-loaded export cache.
//!
//! `compute_epipolar_overlay` is invoked once per click in the epipolar
//! workspace; re-reading and re-parsing the export JSON on every call
//! would be wasteful. We hold the most recent export JSON in a
//! `tauri::State<ExportCache>` singleton (the desktop app) or in the
//! calib-viewer's server state, populated by `set_active_export` and
//! replaced when a new export is made active.

use std::sync::Mutex;

/// Snapshot of a loaded export. The JSON is kept as raw `Value` so the
/// command can pull whatever fields it needs (cameras, sensors,
/// cam_se3_rig, …) without committing the cache to a particular shape.
#[derive(Debug, Clone)]
pub struct CachedExport {
    /// Parsed export JSON.
    pub value: serde_json::Value,
}

/// Tauri-managed state holding the latest cached export.
#[derive(Default)]
pub struct ExportCache {
    inner: Mutex<Option<CachedExport>>,
}

impl ExportCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the cached export.
    pub fn set(&self, value: serde_json::Value) {
        let mut guard = self.inner.lock().expect("export cache mutex poisoned");
        *guard = Some(CachedExport { value });
    }

    /// Run `f` against the cached export under the lock. Returns `None`
    /// when no export has been loaded yet. Cloning the `serde_json::Value`
    /// inside `f` is fine — it's a few-KB to a few-hundred-KB document.
    pub fn read<R>(&self, f: impl FnOnce(&CachedExport) -> R) -> Option<R> {
        let guard = self.inner.lock().expect("export cache mutex poisoned");
        guard.as_ref().map(f)
    }

    /// Run the command body `f` against the cached export; an error when no
    /// export has been made active yet.
    pub fn with<R>(
        &self,
        f: impl FnOnce(&serde_json::Value) -> Result<R, String>,
    ) -> Result<R, String> {
        self.read(|cached| f(&cached.value))
            .ok_or_else(|| "no export loaded yet".to_string())?
    }
}
