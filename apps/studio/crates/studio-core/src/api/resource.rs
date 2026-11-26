//! A resource's answer, with its HTTP semantics decided once here so both
//! shells only copy it out: a file (a source image, a cached transcode) or
//! bytes (a derived blob), its media type, an `ETag` and `Cache-Control`,
//! and the conditional and range requests on it (`If-None-Match` → 304, one
//! `Range` → 206 or 416).

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use super::error::StudioError;

/// What a resource function serves.
#[derive(Debug, Clone)]
pub enum Resource {
    /// A file, served as it is (and range-able).
    File {
        path: PathBuf,
        media_type: &'static str,
        cache_control: &'static str,
    },
    /// Bytes in memory, named by `etag`.
    Bytes {
        bytes: Arc<Vec<u8>>,
        media_type: &'static str,
        etag: String,
        cache_control: &'static str,
    },
}

/// `Cache-Control` of a resource whose source can change on disk:
/// revalidate every time (one `stat`).
pub const REVALIDATE: &str = "private, no-cache";
/// `Cache-Control` of content-addressed bytes, which never change.
pub const IMMUTABLE: &str = "private, max-age=31536000, immutable";

/// The request's conditions, as the client sent them.
#[derive(Debug, Clone, Default)]
pub struct Conditions {
    /// `If-None-Match`.
    pub if_none_match: Option<String>,
    /// `Range`.
    pub range: Option<String>,
}

/// The body of a [`ResourceReply`].
#[derive(Debug, Clone)]
pub enum Body {
    /// `len` bytes of the file at `path` from `start`.
    File {
        path: PathBuf,
        start: u64,
        len: u64,
    },
    Bytes(Arc<Vec<u8>>),
    /// No body (304, 416).
    Empty,
}

/// A resource's HTTP answer.
#[derive(Debug, Clone)]
pub struct ResourceReply {
    /// 200, 206, 304 or 416.
    pub status: u16,
    pub media_type: &'static str,
    pub etag: String,
    pub cache_control: &'static str,
    /// `Content-Range` of a 206 or 416.
    pub content_range: Option<String>,
    pub body: Body,
}

impl ResourceReply {
    /// The body's length in bytes.
    pub fn len(&self) -> u64 {
        match &self.body {
            Body::File { len, .. } => *len,
            Body::Bytes(bytes) => bytes.len() as u64,
            Body::Empty => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The body read into memory.
    pub fn read_body(&self) -> std::io::Result<Vec<u8>> {
        match &self.body {
            Body::File { path, start, len } => {
                let mut file = std::fs::File::open(path)?;
                file.seek(SeekFrom::Start(*start))?;
                let mut out = Vec::with_capacity(usize::try_from(*len).unwrap_or(0));
                file.take(*len).read_to_end(&mut out)?;
                Ok(out)
            }
            Body::Bytes(bytes) => Ok(bytes.as_ref().clone()),
            Body::Empty => Ok(Vec::new()),
        }
    }
}

/// A quoted strong entity tag of `parts`.
pub fn etag_of(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("\"{}\"", &hex(&hash.finalize())[..32])
}

/// Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// `resource` answered under `conditions`.
pub fn reply(resource: Resource, conditions: &Conditions) -> Result<ResourceReply, StudioError> {
    let (media_type, cache_control, etag, total, full) = match resource {
        Resource::File {
            path,
            media_type,
            cache_control,
        } => {
            let meta = std::fs::metadata(&path).map_err(|e| {
                StudioError::new(super::ErrorKind::Io, format!("{}: {e}", path.display()))
            })?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            let etag = etag_of(&[
                path.to_string_lossy().as_bytes(),
                &meta.len().to_le_bytes(),
                &mtime.to_le_bytes(),
            ]);
            let len = meta.len();
            (
                media_type,
                cache_control,
                etag,
                len,
                Body::File {
                    path,
                    start: 0,
                    len,
                },
            )
        }
        Resource::Bytes {
            bytes,
            media_type,
            etag,
            cache_control,
        } => {
            let len = bytes.len() as u64;
            (media_type, cache_control, etag, len, Body::Bytes(bytes))
        }
    };
    let base = ResourceReply {
        status: 200,
        media_type,
        etag,
        cache_control,
        content_range: None,
        body: Body::Empty,
    };
    if conditions
        .if_none_match
        .as_deref()
        .is_some_and(|tags| matches_etag(tags, &base.etag))
    {
        return Ok(ResourceReply {
            status: 304,
            ..base
        });
    }
    // Ranges are served on files; bytes in memory are small and whole.
    if let (Some(range), Body::File { path, .. }) = (conditions.range.as_deref(), &full) {
        match parse_range(range, total) {
            RangeAnswer::Ignore => {}
            RangeAnswer::Unsatisfiable => {
                return Ok(ResourceReply {
                    status: 416,
                    content_range: Some(format!("bytes */{total}")),
                    ..base
                });
            }
            RangeAnswer::Range { start, end } => {
                return Ok(ResourceReply {
                    status: 206,
                    content_range: Some(format!("bytes {start}-{}/{total}", end - 1)),
                    body: Body::File {
                        path: path.clone(),
                        start,
                        len: end - start,
                    },
                    ..base
                });
            }
        }
    }
    Ok(ResourceReply { body: full, ..base })
}

/// Whether an `If-None-Match` list names `etag` (or is `*`).
fn matches_etag(tags: &str, etag: &str) -> bool {
    tags.split(',').map(str::trim).any(|tag| {
        tag == "*" || tag == etag || tag.strip_prefix("W/").is_some_and(|weak| weak == etag)
    })
}

#[derive(Debug, PartialEq, Eq)]
enum RangeAnswer {
    /// Not a single byte range this server serves: the whole body.
    Ignore,
    /// Outside the body: 416.
    Unsatisfiable,
    /// Bytes `start..end`.
    Range { start: u64, end: u64 },
}

/// One `bytes=a-b`, `bytes=a-` or `bytes=-n` range of a body of `total`
/// bytes. Several ranges, other units and malformed ranges are ignored (the
/// whole body is served), as RFC 9110 allows.
fn parse_range(header: &str, total: u64) -> RangeAnswer {
    let Some(spec) = header.trim().strip_prefix("bytes=") else {
        return RangeAnswer::Ignore;
    };
    if spec.contains(',') {
        return RangeAnswer::Ignore;
    }
    let Some((first, last)) = spec.trim().split_once('-') else {
        return RangeAnswer::Ignore;
    };
    let number = |s: &str| s.trim().parse::<u64>().ok();
    let (start, end) = match (first.trim().is_empty(), last.trim().is_empty()) {
        // `-n`: the last n bytes.
        (true, false) => match number(last) {
            Some(0) => return RangeAnswer::Unsatisfiable,
            Some(n) => (total.saturating_sub(n), total),
            None => return RangeAnswer::Ignore,
        },
        // `a-`: from a to the end.
        (false, true) => match number(first) {
            Some(a) => (a, total),
            None => return RangeAnswer::Ignore,
        },
        (false, false) => match (number(first), number(last)) {
            (Some(a), Some(b)) if a <= b => (a, total.min(b.saturating_add(1))),
            _ => return RangeAnswer::Ignore,
        },
        (true, true) => return RangeAnswer::Ignore,
    };
    if start >= total {
        RangeAnswer::Unsatisfiable
    } else {
        RangeAnswer::Range { start, end }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(contents: &[u8]) -> (tempfile::TempDir, Resource) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("f.bin");
        std::fs::write(&path, contents).unwrap();
        let resource = Resource::File {
            path,
            media_type: "application/octet-stream",
            cache_control: REVALIDATE,
        };
        (tmp, resource)
    }

    fn with(range: Option<&str>, etag: Option<&str>) -> Conditions {
        Conditions {
            range: range.map(str::to_owned),
            if_none_match: etag.map(str::to_owned),
        }
    }

    #[test]
    fn ranges_parse_as_rfc_9110_has_them() {
        use RangeAnswer::*;
        assert_eq!(parse_range("bytes=0-3", 10), Range { start: 0, end: 4 });
        assert_eq!(parse_range("bytes=5-", 10), Range { start: 5, end: 10 });
        assert_eq!(parse_range("bytes=-3", 10), Range { start: 7, end: 10 });
        assert_eq!(parse_range("bytes=8-100", 10), Range { start: 8, end: 10 });
        assert_eq!(parse_range("bytes=10-", 10), Unsatisfiable);
        assert_eq!(parse_range("bytes=-0", 10), Unsatisfiable);
        assert_eq!(parse_range("bytes=0-1,4-5", 10), Ignore);
        assert_eq!(parse_range("items=0-1", 10), Ignore);
        assert_eq!(parse_range("bytes=4-2", 10), Ignore);
        assert_eq!(parse_range("bytes=x-", 10), Ignore);
    }

    #[test]
    fn a_file_answers_ranges_and_revalidation() {
        let (_tmp, resource) = file(b"0123456789");
        let whole = reply(resource.clone(), &Conditions::default()).unwrap();
        assert_eq!(whole.status, 200);
        assert_eq!(whole.read_body().unwrap(), b"0123456789");
        assert_eq!(whole.cache_control, REVALIDATE);

        let part = reply(resource.clone(), &with(Some("bytes=2-4"), None)).unwrap();
        assert_eq!(part.status, 206);
        assert_eq!(part.content_range.as_deref(), Some("bytes 2-4/10"));
        assert_eq!(part.read_body().unwrap(), b"234");

        let out = reply(resource.clone(), &with(Some("bytes=20-"), None)).unwrap();
        assert_eq!(out.status, 416);
        assert_eq!(out.content_range.as_deref(), Some("bytes */10"));
        assert!(out.is_empty());

        let same = reply(resource.clone(), &with(None, Some(&whole.etag))).unwrap();
        assert_eq!(same.status, 304);
        assert!(same.is_empty());
        let other = reply(resource, &with(None, Some("\"nope\""))).unwrap();
        assert_eq!(other.status, 200);
    }

    #[test]
    fn a_changed_file_has_a_new_etag() {
        let (_tmp, resource) = file(b"one");
        let before = reply(resource.clone(), &Conditions::default())
            .unwrap()
            .etag;
        let Resource::File { path, .. } = &resource else {
            unreachable!()
        };
        std::fs::write(path, b"three").unwrap();
        let after = reply(resource, &Conditions::default()).unwrap().etag;
        assert_ne!(before, after);
    }

    #[test]
    fn bytes_are_served_whole_with_their_tag() {
        let resource = Resource::Bytes {
            bytes: Arc::new(b"abc".to_vec()),
            media_type: "image/png",
            etag: "\"id\"".into(),
            cache_control: IMMUTABLE,
        };
        let r = reply(resource.clone(), &with(Some("bytes=0-0"), None)).unwrap();
        assert_eq!((r.status, r.len()), (200, 3));
        assert_eq!(
            reply(resource, &with(None, Some("W/\"id\"")))
                .unwrap()
                .status,
            304
        );
    }
}
