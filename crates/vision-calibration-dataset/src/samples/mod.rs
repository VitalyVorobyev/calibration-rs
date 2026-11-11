//! Sample datasets: real calibration images with a ready manifest.
//!
//! [`catalogue`] lists the samples. With the `fetch` feature,
//! `fetch` downloads one from its publisher, at a pinned
//! revision, verifies every file against its SHA-256 and writes the
//! sample's manifest next to the files, so the returned path runs as is:
//!
//! ```no_run
//! # #[cfg(feature = "fetch")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use vision_calibration_dataset::DatasetSpec;
//! use vision_calibration_dataset::samples;
//!
//! let manifest = samples::fetch("opencv_left", &samples::default_cache_dir()?)?;
//! let spec = DatasetSpec::from_path(&manifest)?;
//! println!("{:?} at {}", spec.topology, manifest.display());
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "fetch"))]
//! # fn main() {}
//! ```
//!
//! Files already in the cache and matching their checksum are not
//! downloaded again. Each sample states its licence and the citation its
//! publisher asks for.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::DatasetSpec;

#[cfg(feature = "fetch")]
mod fetch;
#[doc(hidden)]
#[cfg(feature = "fetch")]
pub use fetch::{
    FetchOutcome, extract_zip, fetch_files, sha256_file, summarize_files, unverified, verify,
};
#[cfg(feature = "fetch")]
pub use fetch::{default_cache_dir, fetch};

/// The environment variable that overrides `default_cache_dir`.
pub const CACHE_DIR_ENV: &str = "VISION_CALIBRATION_DATA";

/// One sample dataset of the [`catalogue`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Sample {
    /// The name `fetch` takes, also the sample's directory in
    /// the cache.
    pub id: String,
    /// What the sample holds: the workflow, the views and the target.
    pub description: String,
    /// SPDX licence identifier; `NOASSERTION` when the publisher states no
    /// licence.
    pub licence: String,
    /// The citation the publisher asks for.
    pub citation: String,
    /// The publisher's page for the data.
    pub source_url: String,
    source: FetchSource,
}

impl Sample {
    /// The sample's manifest, as `fetch` writes it.
    pub fn manifest(&self) -> DatasetSpec {
        toml::from_str(self.embedded().manifest)
            .expect("every sample manifest parses (checked by the crate's tests)")
    }

    /// Where the sample's files are downloaded from.
    #[doc(hidden)]
    pub fn source(&self) -> &FetchSource {
        &self.source
    }

    /// The sample's files, by path under the dataset root, with their
    /// SHA-256 (lowercase hex).
    #[doc(hidden)]
    pub fn files(&self) -> BTreeMap<String, String> {
        parse_checksums(self.embedded().files)
            .expect("every sample file list parses (checked by the crate's tests)")
    }

    fn embedded(&self) -> &'static Embedded {
        EMBEDDED
            .iter()
            .find(|e| e.id == self.id)
            .expect("every catalogue entry is embedded (checked by the crate's tests)")
    }
}

/// Where a dataset's files are published.
///
/// Every file of a dataset's file list is placed at its path relative to
/// the dataset root.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FetchSource {
    /// Each file downloaded on its own, from `base_url` followed by its
    /// path.
    Files {
        /// URL prefix, ending in `/`.
        base_url: String,
    },
    /// One ZIP archive holding the files.
    Zip {
        /// URL of the archive.
        url: String,
        /// Directory of the archive that is the dataset root; `""` when
        /// the files sit at the top of the archive.
        #[serde(default)]
        root: String,
    },
}

/// Why a sample could not be listed or fetched.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FetchError {
    /// No sample has this id.
    #[error("no sample named {id:?}; the samples are {}", known.join(", "))]
    UnknownSample {
        /// The id asked for.
        id: String,
        /// The ids of the catalogue.
        known: Vec<String>,
    },
    /// No cache directory could be determined: the platform has none and
    /// [`CACHE_DIR_ENV`] is not set.
    #[error("no cache directory: set {CACHE_DIR_ENV} or pass a directory")]
    NoCacheDir,
    /// A download failed.
    #[error("GET {url}: {message}")]
    Download {
        /// The URL.
        url: String,
        /// The HTTP client's message.
        message: String,
    },
    /// A downloaded archive could not be unpacked, or lacks listed files.
    #[error("unpack {url}: {message}")]
    Archive {
        /// The archive's URL.
        url: String,
        /// What went wrong.
        message: String,
    },
    /// Files are missing or do not match their checksum. After a download
    /// this means the publisher's files changed.
    #[error(
        "{} of {total} files under {} are missing or do not match their checksum: {}",
        files.len(), dir.display(), summarize_names(files)
    )]
    Checksum {
        /// The dataset directory.
        dir: PathBuf,
        /// How many files the list names.
        total: usize,
        /// The files that are missing or do not match.
        files: Vec<String>,
    },
    /// A file list is malformed or names an unsafe path.
    #[error("invalid file list: {0}")]
    InvalidList(String),
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The I/O failure.
        #[source]
        source: std::io::Error,
    },
}

/// How many files an error lists before summarizing the rest.
const MAX_LISTED_FILES: usize = 5;

/// The first few of `files`, then how many more.
fn summarize_names<S: AsRef<str>>(files: &[S]) -> String {
    let mut text = files
        .iter()
        .take(MAX_LISTED_FILES)
        .map(AsRef::as_ref)
        .collect::<Vec<_>>()
        .join(", ");
    if files.len() > MAX_LISTED_FILES {
        text.push_str(&format!(" and {} more", files.len() - MAX_LISTED_FILES));
    }
    text
}

/// A sample's manifest and file list, compiled into the crate.
struct Embedded {
    id: &'static str,
    manifest: &'static str,
    files: &'static str,
}

macro_rules! embed {
    ($id:literal) => {
        Embedded {
            id: $id,
            manifest: include_str!(concat!("../../samples/", $id, "/dataset.toml")),
            files: include_str!(concat!("../../samples/", $id, "/files.sha256")),
        }
    };
}

const EMBEDDED: &[Embedded] = &[
    embed!("opencv_left"),
    embed!("opencv_stereo"),
    embed!("opencv_fisheye_stereo"),
    embed!("kuka_1"),
    embed!("ds8"),
];

const CATALOGUE: &str = include_str!("../../samples/catalogue.toml");

/// The sample datasets, in catalogue order.
pub fn catalogue() -> &'static [Sample] {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Catalogue {
        sample: Vec<Sample>,
    }
    static SAMPLES: OnceLock<Vec<Sample>> = OnceLock::new();
    SAMPLES.get_or_init(|| {
        toml::from_str::<Catalogue>(CATALOGUE)
            .expect("the sample catalogue parses (checked by the crate's tests)")
            .sample
    })
}

/// The sample named `id`.
///
/// # Errors
///
/// [`FetchError::UnknownSample`] when the catalogue has no such sample.
pub fn sample(id: &str) -> Result<&'static Sample, FetchError> {
    let samples = catalogue();
    samples
        .iter()
        .find(|s| s.id == id)
        .ok_or_else(|| FetchError::UnknownSample {
            id: id.to_string(),
            known: samples.iter().map(|s| s.id.clone()).collect(),
        })
}

/// A relative path that stays inside the directory it is joined to: `/`
/// separators, no empty, `.` or `..` component, not absolute, no drive,
/// backslash or NUL.
///
/// # Errors
///
/// [`FetchError::InvalidList`] naming the offending path.
#[doc(hidden)]
pub fn safe_relative_path(name: &str) -> Result<PathBuf, FetchError> {
    let invalid = |why: &str| Err(FetchError::InvalidList(format!("{name:?} {why}")));
    if name.is_empty() {
        return invalid("is empty");
    }
    if name.contains(['\\', ':', '\0']) {
        return invalid("is not a `/`-separated path without a drive or NUL");
    }
    if name.starts_with('/') {
        return invalid("is an absolute path");
    }
    let mut path = PathBuf::new();
    for part in name.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return invalid("has an empty, `.` or `..` component");
        }
        path.push(part);
    }
    Ok(path)
}

/// Parse `sha256sum` output: `<64 hex>  <path>` (text mode) or
/// `<64 hex> *<path>` (binary mode) per line, into path → lowercase hex.
///
/// # Errors
///
/// [`FetchError::InvalidList`] on a malformed line.
#[doc(hidden)]
pub fn parse_checksums(text: &str) -> Result<BTreeMap<String, String>, FetchError> {
    let mut sums = BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let invalid = |why: &str| FetchError::InvalidList(format!("line {}: {why}", n + 1));
        let (hex, rest) = line
            .split_once(' ')
            .ok_or_else(|| invalid("expected `<sha256>  <path>`"))?;
        let path = rest
            .strip_prefix(' ')
            .or_else(|| rest.strip_prefix('*'))
            .ok_or_else(|| invalid("expected two spaces"))?;
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(invalid(&format!("{hex:?} is not a SHA-256 digest")));
        }
        sums.insert(path.to_string(), hex.to_ascii_lowercase());
    }
    Ok(sums)
}

/// Directory of the sample `id` inside the cache `cache`.
pub fn sample_dir(cache: &Path, id: &str) -> PathBuf {
    cache.join(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProblemKind;

    #[test]
    fn every_sample_is_embedded_with_a_valid_manifest_and_file_list() {
        let samples = catalogue();
        assert_eq!(samples.len(), EMBEDDED.len());
        for sample in samples {
            let spec = sample.manifest();
            assert!(spec.unresolved.is_empty(), "{}", sample.id);
            let files = sample.files();
            assert!(!files.is_empty(), "{}", sample.id);
            for rel in files.keys() {
                safe_relative_path(rel).unwrap();
            }
            if let FetchSource::Files { base_url } = sample.source() {
                assert!(base_url.ends_with('/'), "{}", sample.id);
            }
            assert!(!sample.licence.is_empty() && !sample.citation.is_empty());
        }
        assert_eq!(
            sample("kuka_1").unwrap().manifest().topology,
            ProblemKind::SingleCamHandeye
        );
    }

    #[test]
    fn unknown_samples_name_the_catalogue() {
        let err = sample("nope").unwrap_err().to_string();
        assert!(err.contains("opencv_left") && err.contains("ds8"), "{err}");
    }

    #[test]
    fn checksum_lists_parse_both_sha256sum_modes() {
        let hex = "a".repeat(64);
        let sums = parse_checksums(&format!("{hex}  images/a.png\n{hex} *b.json\n")).unwrap();
        assert_eq!(sums.len(), 2);
        assert!(parse_checksums("abc  x.json").is_err());
        assert!(parse_checksums(&format!("{}x.json", "a".repeat(64))).is_err());
    }

    #[test]
    fn unsafe_paths_are_rejected() {
        assert_eq!(
            safe_relative_path("images/camera0/a.png").unwrap(),
            Path::new("images").join("camera0").join("a.png")
        );
        for name in [
            "", "/abs", "../up", "a/../b", "a//b", "./a", "C:/x", "a\\b", "a\0b",
        ] {
            assert!(safe_relative_path(name).is_err(), "{name:?}");
        }
    }
}
