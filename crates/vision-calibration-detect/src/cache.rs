//! Content-addressed detection cache (ADR 0017).
//!
//! The cache is keyed on
//! `(image_content_hash, detector_name, canonical_config_hash)` where:
//!
//! * `image_content_hash` is the BLAKE-style 64-bit FNV-1a digest of
//!   the raw image bytes. We deliberately avoid file mtime — moving an
//!   image around shouldn't invalidate the cached detection of its
//!   contents.
//! * `detector_name` is the `Detector::name()` string (e.g.
//!   `"chessboard"`).
//! * `canonical_config_hash` is the hash of the detector config
//!   serialized to JSON with sorted keys, so semantically equal
//!   configs produce identical keys regardless of struct field order.
//!
//! The default [`FsDetectionCache`] persists entries as JSON files
//! under `<root>/<key>.json`. Tests can use it with a `tempfile::tempdir()`
//! root; production code points it at `~/.cache/calibration-rs/<dataset_id>/`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::Feature;

/// Errors raised by [`DetectionCache`] implementations.
#[derive(Debug, Error)]
pub enum CacheError {
    /// I/O failure inside the cache backend.
    #[error("cache I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Serde failure (de)serializing a cache entry.
    #[error("cache serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Composite cache key. Cheap to compute, hex-encoded for filesystem
/// safety.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// FNV-1a digest of the raw image bytes.
    pub image_hash: u64,
    /// Stable detector name (e.g. `"chessboard"`).
    pub detector: String,
    /// Hash of the canonical (sorted-key) JSON of the detector config.
    pub config_hash: u64,
}

impl CacheKey {
    /// Build a key from raw image bytes + detector name + config JSON.
    /// Uses canonical (sorted-key) serialization for the config so
    /// `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` collide as intended.
    pub fn from_inputs(image_bytes: &[u8], detector: &str, config: &Value) -> Self {
        Self {
            image_hash: hash_image_bytes(image_bytes),
            detector: detector.to_string(),
            config_hash: hash_canonical_json(config),
        }
    }

    /// Filename-safe encoding (`<image>-<detector>-<config>.json`).
    pub fn file_name(&self) -> String {
        format!(
            "{:016x}-{}-{:016x}.json",
            self.image_hash, self.detector, self.config_hash
        )
    }
}

/// Hash raw image bytes with FNV-1a (64-bit). Cheap, allocation-free,
/// good enough for content-addressed cache keys (we're not defending
/// against adversarial collisions, only against accidental ones).
pub fn hash_image_bytes(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn hash_canonical_json(value: &Value) -> u64 {
    let canonical = canonicalize(value.clone());
    let bytes = serde_json::to_vec(&canonical).expect("canonical JSON is serializable");
    hash_image_bytes(&bytes)
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted: std::collections::BTreeMap<String, Value> =
                std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k, canonicalize(v));
            }
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

/// Cached detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFeatures {
    /// The features that were originally produced by the detector.
    pub features: Vec<Feature>,
}

/// Detection cache backend.
///
/// Implementations need only be `Send + Sync` so the runner can share
/// one instance across the per-image detection loop on a tokio
/// blocking task.
pub trait DetectionCache: Send + Sync {
    /// Try to read a cache entry. `Ok(None)` means cache miss.
    fn get(&self, key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError>;

    /// Store a cache entry.
    fn put(&self, key: &CacheKey, value: &CachedFeatures) -> Result<(), CacheError>;
}

/// Filesystem-backed cache writing one JSON file per entry under
/// [`Self::root`]. Stable across crashes; safe under concurrent
/// readers/writers as long as the underlying filesystem provides
/// atomic rename (ext4, APFS, NTFS — all do).
#[derive(Debug, Clone)]
pub struct FsDetectionCache {
    root: PathBuf,
}

impl FsDetectionCache {
    /// Create a cache rooted at `root`. The directory is created on
    /// first write; reads from a missing directory return cache miss.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Return the on-disk root.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl DetectionCache for FsDetectionCache {
    fn get(&self, key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
        let path = self.root.join(key.file_name());
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn put(&self, key: &CacheKey, value: &CachedFeatures) -> Result<(), CacheError> {
        std::fs::create_dir_all(&self.root)?;
        let final_path = self.root.join(key.file_name());
        let tmp_path = self.root.join(format!("{}.tmp", key.file_name()));
        let bytes = serde_json::to_vec_pretty(value)?;
        std::fs::write(&tmp_path, bytes)?;
        std::fs::rename(tmp_path, final_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn key_is_deterministic_under_field_reordering() {
        let img = b"image bytes";
        let cfg_a = json!({ "rows": 9, "cols": 6 });
        let cfg_b = json!({ "cols": 6, "rows": 9 });
        let a = CacheKey::from_inputs(img, "chessboard", &cfg_a);
        let b = CacheKey::from_inputs(img, "chessboard", &cfg_b);
        assert_eq!(a, b);
    }

    #[test]
    fn key_changes_when_config_changes() {
        let img = b"image bytes";
        let cfg_a = json!({ "rows": 9, "cols": 6 });
        let cfg_b = json!({ "rows": 9, "cols": 7 });
        let a = CacheKey::from_inputs(img, "chessboard", &cfg_a);
        let b = CacheKey::from_inputs(img, "chessboard", &cfg_b);
        assert_ne!(a, b);
    }

    #[test]
    fn key_changes_when_image_changes() {
        let cfg = json!({ "rows": 9, "cols": 6 });
        let a = CacheKey::from_inputs(b"image A", "chessboard", &cfg);
        let b = CacheKey::from_inputs(b"image B", "chessboard", &cfg);
        assert_ne!(a, b);
    }

    #[test]
    fn fs_cache_roundtrip() {
        let dir = tempdir().unwrap();
        let cache = FsDetectionCache::new(dir.path());
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({"x": 1}));
        assert!(cache.get(&key).unwrap().is_none(), "expected miss");
        let entry = CachedFeatures {
            features: vec![Feature {
                image_xy: [10.0, 20.0],
                world_xyz: [0.0, 0.0, 0.0],
            }],
        };
        cache.put(&key, &entry).unwrap();
        let back = cache.get(&key).unwrap().expect("cache hit expected");
        assert_eq!(back.features.len(), 1);
        assert_eq!(back.features[0].image_xy, [10.0, 20.0]);
    }

    #[test]
    fn fs_cache_returns_miss_for_unknown_root() {
        let cache =
            FsDetectionCache::new(std::env::temp_dir().join("definitely-does-not-exist-xxxx"));
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({}));
        assert!(cache.get(&key).unwrap().is_none());
    }
}
