//! File helpers of the project module: atomic writes, content digests and
//! canonical JSON.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::ProjectError;

/// A sequence number that makes this process's temporary names unique.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// A name unique to this process and call: `<prefix>-<pid>-<seq>`.
pub(crate) fn unique_name(prefix: &str) -> String {
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{seq}", std::process::id())
}

/// `sha256:<hex>` of `bytes`.
pub(crate) fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(7 + 64);
    text.push_str("sha256:");
    for byte in digest.iter() {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

/// `value` as compact JSON with the keys of every object sorted, whatever
/// order the map type keeps them in.
pub(crate) fn canonical_json(value: &impl Serialize) -> Result<String, serde_json::Error> {
    // Written by hand rather than through the map, so the order holds even
    // when serde_json keeps insertion order.
    fn write(value: &Value, out: &mut String) -> Result<(), serde_json::Error> {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, key) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key)?);
                    out.push(':');
                    write(&map[key.as_str()], out)?;
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(item, out)?;
                }
                out.push(']');
            }
            other => out.push_str(&serde_json::to_string(other)?),
        }
        Ok(())
    }
    let value = serde_json::to_value(value)?;
    let mut out = String::new();
    write(&value, &mut out)?;
    Ok(out)
}

/// `sha256:<hex>` of `value`'s canonical JSON.
pub(crate) fn digest_of(value: &impl Serialize) -> String {
    // Every value hashed here is a plain record of the library's own types,
    // which serialize to JSON without failing.
    let text = canonical_json(value).expect("the hashed records serialize to JSON");
    sha256(text.as_bytes())
}

/// `value` as pretty JSON with a final newline: how the project writes its
/// JSON files.
pub(crate) fn json_text(value: &impl Serialize, path: &Path) -> Result<String, ProjectError> {
    let mut text = serde_json::to_string_pretty(value).map_err(|e| ProjectError::parse(path, e))?;
    text.push('\n');
    Ok(text)
}

/// Read `path` whole.
pub(crate) fn read(path: &Path) -> Result<Vec<u8>, ProjectError> {
    std::fs::read(path).map_err(|e| ProjectError::io(path, e))
}

/// Replace `path` with `bytes` in one step: written to a temporary sibling,
/// then renamed over it, so a reader sees the old file or the new one.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), ProjectError> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let partial = path.with_file_name(unique_name(&format!(".{name}.partial")));
    if let Err(err) = std::fs::write(&partial, bytes) {
        let _ = std::fs::remove_file(&partial);
        return Err(ProjectError::io(path, err));
    }
    std::fs::rename(&partial, path).map_err(|err| {
        let _ = std::fs::remove_file(&partial);
        ProjectError::io(path, err)
    })
}

/// A folder that is removed when dropped, unless [`kept`](Self::kept).
pub(crate) struct TempDir {
    path: PathBuf,
    armed: bool,
}

impl TempDir {
    /// Create the new folder `path`.
    pub(crate) fn create(path: PathBuf) -> Result<Self, ProjectError> {
        std::fs::create_dir(&path).map_err(|e| ProjectError::io(&path, e))?;
        Ok(Self { path, armed: true })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Keep the folder: it was renamed into place.
    pub(crate) fn kept(mut self) {
        self.armed = false;
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_json_sorts_keys_at_every_level() {
        let value = json!({ "b": 1, "a": { "d": [ { "z": 1, "y": 2 } ], "c": 0.5 } });
        assert_eq!(
            canonical_json(&value).unwrap(),
            r#"{"a":{"c":0.5,"d":[{"y":2,"z":1}]},"b":1}"#
        );
        assert_eq!(
            digest_of(&value),
            digest_of(&json!({ "a": { "c": 0.5, "d": [ { "y": 2, "z": 1 } ] }, "b": 1 }))
        );
        assert!(digest_of(&value).starts_with("sha256:"));
        assert_eq!(digest_of(&value).len(), 7 + 64);
    }

    #[test]
    fn sha256_of_the_empty_input() {
        assert_eq!(
            sha256(b""),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn an_atomic_write_replaces_the_file_and_leaves_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a.json");
        write_atomic(&path, b"one").unwrap();
        write_atomic(&path, b"two").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
    }

    #[test]
    fn a_temp_dir_is_removed_unless_kept() {
        let tmp = tempfile::tempdir().unwrap();
        let gone = tmp.path().join("gone");
        drop(TempDir::create(gone.clone()).unwrap());
        assert!(!gone.exists());
        let kept = tmp.path().join("kept");
        TempDir::create(kept.clone()).unwrap().kept();
        assert!(kept.is_dir());
    }
}
