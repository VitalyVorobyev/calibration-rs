//! Content-addressed detection cache.
//!
//! The cache is keyed on
//! `(image_content_hash, detector_name, canonical_config_hash)` where:
//!
//! * `image_content_hash` is the 64-bit FNV-1a digest of
//!   the raw image bytes. We deliberately avoid file mtime — moving an
//!   image around shouldn't invalidate the cached detection of its
//!   contents.
//! * `detector_name` is the detector's name (e.g. `"chessboard"`).
//! * `canonical_config_hash` is the hash of the detector config
//!   serialized to JSON with sorted keys, so semantically equal
//!   configs produce identical keys regardless of struct field order.
//!   [`CacheKey::for_detection`] hashes the config together with this
//!   crate's release and the detector's output revision, so upgrading the
//!   detector stack recomputes cached detections.
//!
//! The default [`FsDetectionCache`] persists entries as JSON files
//! under `<root>/<key>.json`. Tests can use it with a `tempfile::tempdir()`
//! root; production code points it at `~/.cache/calibration-rs/<dataset_id>/`.
//!
//! A detection's [`DetectionDetail`] is a separate entry under the same key
//! ([`DetectionCache::get_detail`], `<root>/<key>.detail.json`), so a
//! calibration, which reads only the features, never parses it.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::{DetectionDetail, DetectorConfig, Feature};

/// Errors raised by [`DetectionCache`] implementations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CacheError {
    /// I/O failure inside the cache backend.
    #[error("cache I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A cache entry could not be (de)serialized; the message names why.
    #[error("cache serde error: {0}")]
    Serde(String),
}

/// Composite cache key. Cheap to compute, hex-encoded for filesystem
/// safety.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CacheKey {
    /// FNV-1a digest of the raw image bytes.
    pub image_hash: u64,
    /// Stable detector name (e.g. `"chessboard"`).
    pub detector: String,
    /// Hash of the canonical (sorted-key) JSON of the detector config.
    pub config_hash: u64,
}

impl CacheKey {
    /// Build a key from raw image bytes, a detector name and the config
    /// that produced the features. The config is hashed as canonical
    /// (sorted-key) JSON, so field order does not matter. A custom
    /// extractor (a laser-line one, say) keys its results this way.
    pub fn from_inputs(image_bytes: &[u8], detector: &str, config: &impl Serialize) -> Self {
        let config = serde_json::to_value(config)
            .expect("a detector config is a plain record of options; it serializes");
        Self {
            image_hash: hash_image_bytes(image_bytes),
            detector: detector.to_string(),
            config_hash: hash_canonical_json(&config),
        }
    }

    /// Build the key of one detection of `config` in an image whose file
    /// holds `image_bytes`, cropped to `roi` (`[x, y, width, height]`) when
    /// given: [`Self::from_inputs`] on the config, the crop, this crate's
    /// release, the version of the detection crate it wraps, and the
    /// detector's output revision.
    ///
    /// Keying on both versions means an upgrade of the detector stack — this
    /// crate or the target-detection crate it wraps — recomputes cached
    /// detections instead of serving the previous detector's output.
    pub fn for_detection(
        image_bytes: &[u8],
        config: &DetectorConfig,
        roi: Option<[u32; 4]>,
    ) -> Self {
        #[derive(Serialize)]
        struct Keyed<'a> {
            config: &'a DetectorConfig,
            release: &'static str,
            backend: &'static str,
            #[serde(skip_serializing_if = "Option::is_none")]
            revision: Option<u32>,
            /// The crop; features are in its pixel frame.
            #[serde(skip_serializing_if = "Option::is_none")]
            roi: Option<[u32; 4]>,
        }
        let keyed = Keyed {
            config,
            release: env!("CARGO_PKG_VERSION"),
            backend: config.backend(),
            revision: config.output_revision(),
            roi,
        };
        Self::from_inputs(image_bytes, config.name(), &keyed)
    }

    /// Filename-safe encoding (`<image>-<detector>-<config>.json`).
    ///
    /// The detector segment is sanitized so the result is a legal file
    /// name on every supported OS — notably Windows/NTFS, which rejects
    /// `< > : " / \ | ? *`. Laser namespaces use a `laser:<name>` form,
    /// so the `:` must be stripped or the on-disk cache
    /// write fails with `ERROR_INVALID_NAME` (OS error 87).
    pub fn file_name(&self) -> String {
        format!(
            "{:016x}-{}-{:016x}.json",
            self.image_hash,
            sanitize_detector(&self.detector),
            self.config_hash
        )
    }

    /// The file name of the key's detection detail entry
    /// (`<image>-<detector>-<config>.detail.json`), beside its features entry.
    fn detail_file_name(&self) -> String {
        format!(
            "{:016x}-{}-{:016x}.detail.json",
            self.image_hash,
            sanitize_detector(&self.detector),
            self.config_hash
        )
    }
}

/// Replace any character that is not alphanumeric, `_`, `-`, or `.`
/// with `_`, so the detector id is a portable file-name segment. The
/// detector ids are a controlled, in-tree set of stable strings, so
/// this mapping stays injective in practice — there is no
/// `laser:fake`/`laser_fake` pair to collide.
fn sanitize_detector(detector: &str) -> String {
    detector
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Hash raw image bytes with FNV-1a (64-bit). Cheap, allocation-free,
/// good enough for content-addressed cache keys (we're not defending
/// against adversarial collisions, only against accidental ones).
pub(crate) fn hash_image_bytes(bytes: &[u8]) -> u64 {
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
#[non_exhaustive]
pub struct CachedFeatures {
    /// The features that were originally produced by the detector.
    pub features: Vec<Feature>,
}

impl CachedFeatures {
    /// A cache entry holding `features`.
    pub fn new(features: Vec<Feature>) -> Self {
        Self { features }
    }
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

    /// The detection detail stored under `key`
    /// ([`DetectorConfig::detect_detailed`]), apart from the features entry
    /// of the same key; `Ok(None)` on a miss. A calibration never reads it.
    ///
    /// The default stores no detail, so this is always a miss.
    fn get_detail(&self, key: &CacheKey) -> Result<Option<DetectionDetail>, CacheError> {
        let _ = key;
        Ok(None)
    }

    /// Store the detail of the detection whose features are stored under
    /// `key`.
    ///
    /// The default stores nothing.
    fn put_detail(&self, key: &CacheKey, detail: &DetectionDetail) -> Result<(), CacheError> {
        let _ = (key, detail);
        Ok(())
    }
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

impl FsDetectionCache {
    /// The entry `file_name` under the root, parsed; `None` when absent.
    fn read<T: serde::de::DeserializeOwned>(
        &self,
        file_name: &str,
    ) -> Result<Option<T>, CacheError> {
        match std::fs::read(self.root.join(file_name)) {
            Ok(bytes) => Ok(Some(
                serde_json::from_slice(&bytes).map_err(|e| CacheError::Serde(e.to_string()))?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Write `bytes` as the entry `file_name`: to a temporary file, then
    /// renamed over the entry, so a reader sees the old entry or the new one
    /// whole.
    fn write(&self, file_name: &str, bytes: &[u8]) -> Result<(), CacheError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

        std::fs::create_dir_all(&self.root)?;
        let final_path = self.root.join(file_name);
        let tmp_id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let tmp_path = self
            .root
            .join(format!("{file_name}.{}.{tmp_id}.tmp", std::process::id()));
        if let Err(err) = std::fs::write(&tmp_path, bytes) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(err.into());
        }
        if let Err(err) = std::fs::rename(&tmp_path, final_path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(err.into());
        }
        Ok(())
    }
}

impl DetectionCache for FsDetectionCache {
    fn get(&self, key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
        self.read(&key.file_name())
    }

    fn put(&self, key: &CacheKey, value: &CachedFeatures) -> Result<(), CacheError> {
        let bytes =
            serde_json::to_vec_pretty(value).map_err(|e| CacheError::Serde(e.to_string()))?;
        self.write(&key.file_name(), &bytes)
    }

    /// Reads `<image>-<detector>-<config>.detail.json` under the root.
    fn get_detail(&self, key: &CacheKey) -> Result<Option<DetectionDetail>, CacheError> {
        self.read(&key.detail_file_name())
    }

    /// Writes `<image>-<detector>-<config>.detail.json` under the root, as
    /// compact JSON, the same atomic way as [`put`](DetectionCache::put).
    fn put_detail(&self, key: &CacheKey, detail: &DetectionDetail) -> Result<(), CacheError> {
        let bytes = serde_json::to_vec(detail).map_err(|e| CacheError::Serde(e.to_string()))?;
        self.write(&key.detail_file_name(), &bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    #[cfg(all(feature = "chessboard", feature = "puzzleboard"))]
    fn detector_keys_carry_the_release_and_output_revision() {
        let chessboard = DetectorConfig::Chessboard(crate::ChessboardConfig::new(17, 28, 0.02));
        let release = env!("CARGO_PKG_VERSION");
        assert_eq!(
            CacheKey::for_detection(b"img", &chessboard, None),
            CacheKey::from_inputs(
                b"img",
                "chessboard",
                &json!({
                    "config": { "detector": "chessboard", "rows": 17, "cols": 28, "square_size_m": 0.02 },
                    "release": release,
                    "backend": crate::backend::CALIB_TARGETS,
                    "revision": 2,
                })
            )
        );
        // A crop is part of the key.
        let puzzle = DetectorConfig::Puzzleboard(crate::PuzzleboardConfig::new(10, 10, 0.01));
        assert_eq!(
            CacheKey::for_detection(b"img", &puzzle, Some([1, 2, 3, 4])),
            CacheKey::from_inputs(
                b"img",
                "puzzleboard",
                &json!({
                    "config": { "detector": "puzzleboard", "rows": 10, "cols": 10, "cell_size_m": 0.01 },
                    "release": release,
                    "backend": crate::backend::CALIB_TARGETS,
                    "revision": 1,
                    "roi": [1, 2, 3, 4],
                })
            )
        );
        assert_ne!(
            CacheKey::for_detection(b"img", &puzzle, None),
            CacheKey::for_detection(b"img", &puzzle, Some([1, 2, 3, 4])),
        );
    }

    /// A detection that measured blur is keyed apart from one that did not,
    /// so a blur-less cache entry is never served to a run that asks for blur
    /// (and the default config keeps its key, checked above).
    #[test]
    #[cfg(all(feature = "chessboard", feature = "puzzleboard"))]
    fn measuring_blur_changes_the_key() {
        let chessboard = crate::ChessboardConfig::new(17, 28, 0.02);
        let puzzle = crate::PuzzleboardConfig::new(10, 10, 0.01);
        for (plain, with) in [
            (
                DetectorConfig::Chessboard(chessboard),
                DetectorConfig::Chessboard(chessboard.with_measure_blur(true)),
            ),
            (
                DetectorConfig::Puzzleboard(puzzle),
                DetectorConfig::Puzzleboard(puzzle.with_measure_blur(true)),
            ),
        ] {
            assert_ne!(
                CacheKey::for_detection(b"img", &plain, None),
                CacheKey::for_detection(b"img", &with, None),
            );
            // Turned off explicitly, the key is the default one.
            let off = match with {
                DetectorConfig::Chessboard(c) => {
                    DetectorConfig::Chessboard(c.with_measure_blur(false))
                }
                DetectorConfig::Puzzleboard(c) => {
                    DetectorConfig::Puzzleboard(c.with_measure_blur(false))
                }
                _ => unreachable!("a chessboard or PuzzleBoard config"),
            };
            assert_eq!(
                CacheKey::for_detection(b"img", &plain, None),
                CacheKey::for_detection(b"img", &off, None),
            );
        }
    }

    /// The backend versions in the cache key are the ones the workspace
    /// lockfile resolves, and every `calib-targets-*` crate resolves to the
    /// facade's version. Skipped outside the repository (a published package
    /// has no workspace lockfile).
    #[test]
    #[cfg(all(feature = "chessboard", feature = "ringgrid"))]
    fn backend_versions_match_the_lockfile() {
        let lock = concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock");
        let Ok(lock) = std::fs::read_to_string(lock) else {
            return;
        };
        let locked: Vec<(&str, &str)> = lock
            .split("[[package]]")
            .filter_map(|entry| {
                let field = |key: &str| {
                    entry.lines().find_map(|l| {
                        l.strip_prefix(key)?
                            .strip_prefix(" = \"")?
                            .strip_suffix('"')
                    })
                };
                Some((field("name")?, field("version")?))
            })
            .collect();
        let backends = [crate::backend::CALIB_TARGETS, crate::backend::RINGGRID];
        for pair in backends.iter().flat_map(|b| b.split(" / ")) {
            let (name, version) = pair.split_once(' ').expect("\"<crate> <version>\"");
            assert!(
                locked.contains(&(name, version)),
                "Cargo.lock does not resolve {pair}; update `crate::backend`"
            );
        }
        let facade = locked
            .iter()
            .find(|(name, _)| *name == "calib-targets")
            .expect("calib-targets is locked")
            .1;
        for (name, version) in locked
            .iter()
            .filter(|(n, _)| n.starts_with("calib-targets-"))
        {
            assert_eq!(
                *version, facade,
                "{name} {version} differs from calib-targets {facade}; run `cargo update` on the family"
            );
        }
    }

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
                cov_px: None,
                blur: None,
            }],
        };
        cache.put(&key, &entry).unwrap();
        let back = cache.get(&key).unwrap().expect("cache hit expected");
        assert_eq!(back.features.len(), 1);
        assert_eq!(back.features[0].image_xy, [10.0, 20.0]);
    }

    /// A cache hit must return exactly the features that were stored: a
    /// detection read back from disk must calibrate to the same result as
    /// the fresh one.
    #[test]
    fn fs_cache_roundtrip_is_bit_exact() {
        let dir = tempdir().unwrap();
        let cache = FsDetectionCache::new(dir.path());
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({}));
        // Sub-pixel positions as detectors produce them (`f32` widened), and
        // metric target points (grid index × pitch).
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let features: Vec<Feature> = (0..20_000)
            .map(|i| Feature {
                image_xy: [f64::from((next() * 1440.0) as f32), next() * 1080.0],
                world_xyz: [(i % 130) as f64 * 0.0052, (i / 130) as f64 * 0.001014, 0.0],
                cov_px: None,
                blur: None,
            })
            .collect();
        let entry = CachedFeatures { features };
        cache.put(&key, &entry).unwrap();
        let back = cache.get(&key).unwrap().expect("cache hit expected");
        let bits = |f: &Feature| (f.image_xy.map(f64::to_bits), f.world_xyz.map(f64::to_bits));
        for (stored, read) in entry.features.iter().zip(&back.features) {
            assert_eq!(bits(stored), bits(read), "{stored:?} read back as {read:?}");
        }
    }

    /// A feature's blur is stored and read back exactly, and an entry
    /// written before features carried a blur still reads, without one.
    #[test]
    fn fs_cache_carries_the_blur_and_reads_old_entries() {
        let dir = tempdir().unwrap();
        let cache = FsDetectionCache::new(dir.path());
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({}));
        let blur = crate::FeatureBlur::new(
            [2.25, -0.125, 1.75],
            [[4e-3, 1e-4, 0.0], [1e-4, 1e-3, 2e-5], [0.0, 2e-5, 3e-3]],
        );
        let entry = CachedFeatures::new(vec![
            Feature::new([1.5, 2.5], [0.0, 0.0, 0.0]).with_blur(Some(blur)),
            Feature::new([7.5, 2.5], [0.008, 0.0, 0.0]),
        ]);
        cache.put(&key, &entry).unwrap();
        let back = cache.get(&key).unwrap().expect("cache hit expected");
        assert_eq!(back.features, entry.features);

        let old = CacheKey::from_inputs(b"old", "chessboard", &json!({}));
        std::fs::write(
            dir.path().join(old.file_name()),
            r#"{"features":[{"image_xy":[1.5,2.5],"world_xyz":[0.0,0.0,0.0]}]}"#,
        )
        .unwrap();
        let back = cache.get(&old).unwrap().expect("cache hit expected");
        assert_eq!(back.features[0].blur, None);
    }

    #[test]
    fn file_name_is_windows_safe_for_laser_namespace() {
        // `laser:<name>` namespaces (ADR 0016) must not leak the colon
        // into the file name — NTFS rejects it (OS error 87).
        let key = CacheKey::from_inputs(b"img", "laser:fake", &json!({}));
        let name = key.file_name();
        assert!(
            !name.contains(':'),
            "detector segment must be sanitized, got {name}"
        );
        assert!(name.contains("laser_fake"), "got {name}");
    }

    #[test]
    fn fs_cache_roundtrip_with_colon_detector() {
        let dir = tempdir().unwrap();
        let cache = FsDetectionCache::new(dir.path());
        let key = CacheKey::from_inputs(b"img", "laser:fake", &json!({"x": 1}));
        let entry = CachedFeatures {
            features: vec![Feature {
                image_xy: [1.0, 2.0],
                world_xyz: [0.0, 0.0, 0.0],
                cov_px: None,
                blur: None,
            }],
        };
        cache.put(&key, &entry).unwrap();
        assert_eq!(cache.get(&key).unwrap().unwrap().features.len(), 1);
    }

    #[test]
    fn fs_cache_returns_miss_for_unknown_root() {
        let cache =
            FsDetectionCache::new(std::env::temp_dir().join("definitely-does-not-exist-xxxx"));
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({}));
        assert!(cache.get(&key).unwrap().is_none());
    }

    #[test]
    fn concurrent_cache_puts_are_atomic_and_clean() {
        let dir = tempdir().unwrap();
        let cache = std::sync::Arc::new(FsDetectionCache::new(dir.path()));
        let key = CacheKey::from_inputs(b"shared", "chessboard", &json!({"n": 1}));
        let entry = CachedFeatures::new(vec![Feature::new([12.5, 24.5], [0.02, 0.04, 0.0])]);

        std::thread::scope(|s| {
            for _ in 0..8 {
                let cache = &cache;
                let key = &key;
                let entry = &entry;
                s.spawn(move || {
                    for _ in 0..10 {
                        cache.put(key, entry).unwrap();
                    }
                });
            }
        });

        let back = cache.get(&key).unwrap().expect("cache hit expected");
        assert_eq!(back.features, entry.features);
        let leftover_tmps: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("tmp"))
            .collect();
        assert!(
            leftover_tmps.is_empty(),
            "leftover temp files: {leftover_tmps:?}"
        );
    }

    fn a_detail() -> DetectionDetail {
        serde_json::from_value(json!({
            "features": [{ "image_xy": [12.5, 24.5], "world_xyz": [0.02, 0.04, 0.0] }],
            "outcome": { "status": "detected" },
            "target": {
                "kind": "chessboard",
                "corners": [{ "grid": [1, 2], "score": 0.25 }],
                "quality": { "local_residual_count": 0 },
            },
        }))
        .unwrap()
    }

    #[test]
    fn fs_cache_keeps_the_detail_beside_the_features() {
        let dir = tempdir().unwrap();
        let cache = FsDetectionCache::new(dir.path());
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({"n": 1}));
        assert!(cache.get_detail(&key).unwrap().is_none(), "a miss");

        let detail = a_detail();
        cache.put_detail(&key, &detail).unwrap();
        assert_eq!(cache.get_detail(&key).unwrap(), Some(detail.clone()));
        // The features entry is its own: still a miss, and a hit once
        // stored, without touching the detail.
        assert!(cache.get(&key).unwrap().is_none());
        cache
            .put(&key, &CachedFeatures::new(detail.features.clone()))
            .unwrap();
        assert_eq!(cache.get(&key).unwrap().unwrap().features, detail.features);
        assert_eq!(cache.get_detail(&key).unwrap(), Some(detail));
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        let stem = key.file_name().trim_end_matches(".json").to_string();
        assert_eq!(
            names,
            [format!("{stem}.detail.json"), format!("{stem}.json")]
        );
    }

    #[test]
    fn concurrent_detail_puts_are_atomic_and_clean() {
        let dir = tempdir().unwrap();
        let cache = FsDetectionCache::new(dir.path());
        let key = CacheKey::from_inputs(b"shared", "chessboard", &json!({"n": 1}));
        let detail = a_detail();
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..10 {
                        cache.put_detail(&key, &detail).unwrap();
                    }
                });
            }
        });
        assert_eq!(cache.get_detail(&key).unwrap(), Some(detail));
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            1,
            "no temp files left"
        );
    }

    /// A cache written before the detail methods existed keeps compiling
    /// and stores no detail.
    #[test]
    fn a_cache_without_the_detail_methods_stores_no_detail() {
        struct FeaturesOnly;
        impl DetectionCache for FeaturesOnly {
            fn get(&self, _: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
                Ok(None)
            }
            fn put(&self, _: &CacheKey, _: &CachedFeatures) -> Result<(), CacheError> {
                Ok(())
            }
        }
        let key = CacheKey::from_inputs(b"img", "chessboard", &json!({}));
        FeaturesOnly.put_detail(&key, &a_detail()).unwrap();
        assert_eq!(FeaturesOnly.get_detail(&key).unwrap(), None);
    }
}
