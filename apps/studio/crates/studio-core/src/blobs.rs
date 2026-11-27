//! Derived blobs — a disparity job's planes, point cloud and rectified
//! pair, a run's map planes — kept in memory and served by the `blob` resource. A blob's id is
//! the SHA-256 of what made it (the command, its canonical arguments, the
//! calibration's fingerprint, the part), so the same request finds its
//! blobs again: the store memoizes as well as it serves. Least recently
//! used blobs are evicted past the cap; a client asking for one answers
//! `blob_expired` and runs its command again.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::api::resource::hex;

/// The store's default cap, in bytes.
pub const BLOB_CAP: usize = 256 << 20;

/// A blob a command made: fetch it from the `blob` resource with its id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BlobRef {
    pub id: String,
    /// `application/x-vam1` (a value plane), `application/x-f32` (raw
    /// little-endian floats), `image/png`, …
    pub media_type: String,
    /// Its length in bytes.
    pub len: u64,
}

struct Entry {
    media_type: &'static str,
    bytes: Arc<Vec<u8>>,
    used: u64,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<String, Entry>,
    total: usize,
    clock: u64,
}

/// The in-memory blob store.
pub struct BlobStore {
    cap: usize,
    inner: Mutex<Inner>,
}

impl Default for BlobStore {
    fn default() -> Self {
        Self::new(BLOB_CAP)
    }
}

impl BlobStore {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: Mutex::new(Inner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // The map stays whole whatever panicked while it was held.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The id of the blob `parts` name.
    pub fn id(parts: &[&[u8]]) -> String {
        let mut hash = Sha256::new();
        for part in parts {
            hash.update((part.len() as u64).to_le_bytes());
            hash.update(part);
        }
        hex(&hash.finalize())
    }

    /// Keep `bytes` as blob `id`, evicting the least recently used others
    /// past the cap.
    pub fn put(&self, id: String, media_type: &'static str, bytes: Vec<u8>) -> BlobRef {
        let len = bytes.len();
        let mut inner = self.lock();
        inner.clock += 1;
        let used = inner.clock;
        if let Some(old) = inner.entries.insert(
            id.clone(),
            Entry {
                media_type,
                bytes: Arc::new(bytes),
                used,
            },
        ) {
            inner.total -= old.bytes.len();
        }
        inner.total += len;
        while inner.total > self.cap {
            let oldest = inner
                .entries
                .iter()
                .filter(|(key, _)| **key != id)
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| key.clone());
            let Some(oldest) = oldest else { break };
            if let Some(gone) = inner.entries.remove(&oldest) {
                inner.total -= gone.bytes.len();
            }
        }
        BlobRef {
            id,
            media_type: media_type.to_owned(),
            len: len as u64,
        }
    }

    /// Blob `id`, if it is still kept.
    pub fn get(&self, id: &str) -> Option<(&'static str, Arc<Vec<u8>>)> {
        let mut inner = self.lock();
        inner.clock += 1;
        let now = inner.clock;
        let entry = inner.entries.get_mut(id)?;
        entry.used = now;
        Some((entry.media_type, Arc::clone(&entry.bytes)))
    }

    /// Drop blob `id`, as the cap would (tests of expiry).
    #[cfg(test)]
    pub fn forget(&self, id: &str) {
        let mut inner = self.lock();
        if let Some(gone) = inner.entries.remove(id) {
            inner.total -= gone.bytes.len();
        }
    }

    /// Whether every blob of `refs` is still kept.
    pub fn has_all<'a>(&self, refs: impl IntoIterator<Item = &'a BlobRef>) -> bool {
        let inner = self.lock();
        refs.into_iter().all(|r| inner.entries.contains_key(&r.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_content_addresses_of_their_parts() {
        let a = BlobStore::id(&[b"disparity", b"{}"]);
        assert_eq!(a.len(), 64);
        assert_eq!(a, BlobStore::id(&[b"disparity", b"{}"]));
        // Part boundaries count: "ab"+"c" is not "a"+"bc".
        assert_ne!(BlobStore::id(&[b"ab", b"c"]), BlobStore::id(&[b"a", b"bc"]));
    }

    #[test]
    fn the_least_recently_used_blob_goes_first() {
        let store = BlobStore::new(10);
        let a = store.put("a".into(), "x", vec![0; 4]);
        store.put("b".into(), "x", vec![0; 4]);
        // Reading `a` makes `b` the oldest.
        assert!(store.get("a").is_some());
        store.put("c".into(), "x", vec![0; 4]);
        assert!(store.get("b").is_none());
        assert!(store.has_all([&a]));
        assert_eq!(a.len, 4);
        // A blob larger than the cap is kept alone.
        store.put("big".into(), "x", vec![0; 20]);
        assert!(store.get("big").is_some());
        assert!(store.get("a").is_none());
    }
}
