//! Most-recently-loaded export cache.
//!
//! `compute_epipolar_overlay` is invoked once per click in the epipolar
//! workspace; re-reading and re-parsing the export JSON on every call
//! would be wasteful. We hold the most recent `(path, json)` pair in a
//! `tauri::State<ExportCache>` singleton, populated by `load_export` and
//! invalidated when a new export is loaded.

use std::sync::Mutex;

/// Snapshot of a loaded export. The JSON is kept as raw `Value` so the
/// command can pull whatever fields it needs (cameras, sensors,
/// cam_se3_rig, …) without committing the cache to a particular shape.
#[derive(Debug, Clone)]
pub struct CachedExport {
    /// Absolute path the export was loaded from. Kept around so future
    /// commands can disambiguate stale caches after a reload — read by
    /// debug logs but not yet by command code.
    #[allow(dead_code)]
    pub path: String,
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
    pub fn set(&self, path: String, value: serde_json::Value) {
        let mut guard = self.inner.lock().expect("export cache mutex poisoned");
        *guard = Some(CachedExport { path, value });
    }

    /// Run `f` against the cached export under the lock. Returns `None`
    /// when no export has been loaded yet. Cloning the `serde_json::Value`
    /// inside `f` is fine — it's a few-KB to a few-hundred-KB document.
    pub fn read<R>(&self, f: impl FnOnce(&CachedExport) -> R) -> Option<R> {
        let guard = self.inner.lock().expect("export cache mutex poisoned");
        guard.as_ref().map(f)
    }
}
