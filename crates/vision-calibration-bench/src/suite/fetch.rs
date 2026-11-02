//! Download an image entry's files from their publisher into a local cache.
//!
//! An entry's [`FetchSource`] (in `suite.toml`) says where the files are
//! published; its fetch list (`fetch/<id>.sha256`, in `sha256sum` format)
//! names every file the dataset manifest reads, by its path under the
//! dataset root, with its SHA-256. [`fetch_entry`] downloads the files into
//! [`entry_dir`] and verifies each one; files already there and matching
//! their checksum are not downloaded again. `suite freeze --images <cache>`
//! then reads the entry's images from the cache instead of the manifest's
//! own directory. The download and verification are the library's sample
//! fetcher (`vision_calibration_dataset::samples`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use vision_calibration_dataset::samples::{self, fetch_files};
pub use vision_calibration_dataset::samples::{sha256_file, unverified, verify};

use super::manifest::{parse_checksums, safe_relative_path};
use super::{Suite, SuiteEntry};

/// What [`fetch_entry`] did for one entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// Every listed file was already in the cache and matched its checksum.
    UpToDate {
        /// Number of listed files.
        files: usize,
    },
    /// The files that were missing or did not match were downloaded, and
    /// every listed file now matches its checksum.
    Fetched {
        /// Number of listed files.
        files: usize,
        /// How many of them were downloaded.
        downloaded: usize,
    },
    /// The entry's images cannot be downloaded automatically (the
    /// publisher needs an account, say); they are downloaded by hand from
    /// this page.
    Manual {
        /// The entry's `source_url`.
        source_url: Option<String>,
    },
    /// The entry has no images (a synthetic entry).
    NoImages,
}

/// Directory of an entry's files inside the cache `cache`.
pub fn entry_dir(cache: &Path, id: &str) -> PathBuf {
    cache.join(id)
}

/// Read an entry's fetch list: relative path → SHA-256 (lowercase hex).
///
/// # Errors
///
/// Fails when the file is missing or malformed, is empty, or names an
/// unsafe path (see [`safe_relative_path`]).
pub fn read_fetch_list(suite: &Suite, id: &str) -> Result<BTreeMap<String, String>> {
    let path = suite.fetch_list_path(id);
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let list = parse_checksums(&text).with_context(|| format!("parse {}", path.display()))?;
    ensure!(!list.is_empty(), "{} lists no files", path.display());
    for rel in list.keys() {
        safe_relative_path(rel).with_context(|| format!("{}", path.display()))?;
    }
    Ok(list)
}

/// Download an entry's files into `entry_dir(cache, id)` and verify them.
/// Files already there and matching their checksum are kept; nothing is
/// downloaded when all of them do.
///
/// # Errors
///
/// Fails on a missing or invalid fetch list, a download or archive error,
/// and when a downloaded file does not match its checksum (the publisher's
/// file changed). An entry without a fetch source is not an error: it
/// reports [`FetchOutcome::Manual`] or [`FetchOutcome::NoImages`].
pub fn fetch_entry(suite: &Suite, entry: &SuiteEntry, cache: &Path) -> Result<FetchOutcome> {
    let Some(source) = &entry.fetch else {
        return Ok(match entry.manifest {
            Some(_) => FetchOutcome::Manual {
                source_url: entry.source_url.clone(),
            },
            None => FetchOutcome::NoImages,
        });
    };
    let list = read_fetch_list(suite, &entry.id)?;
    let dir = entry_dir(cache, &entry.id);
    let outcome = fetch_files(source, &list, &dir).with_context(|| {
        format!(
            "fetch {:?} (files listed in {})",
            entry.id,
            suite.fetch_list_path(&entry.id).display()
        )
    })?;
    Ok(match outcome {
        samples::FetchOutcome::UpToDate { files } => FetchOutcome::UpToDate { files },
        samples::FetchOutcome::Fetched { files, downloaded } => {
            FetchOutcome::Fetched { files, downloaded }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::manifest::sha256_hex;

    /// A fresh, empty directory under the system temp dir.
    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("calib-suite-fetch-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A suite whose one entry fetches from an unroutable URL: any attempt
    /// to download fails, so a pass proves the cache was used.
    fn offline_suite(dir: &Path, list: &str) -> Suite {
        fs::create_dir_all(dir.join("fetch")).unwrap();
        fs::write(
            dir.join("suite.toml"),
            "suite_version = \"t\"\n[[entry]]\nid = \"e\"\ndescription = \"d\"\n\
             licence = \"MIT\"\nmanifest = \"data/dataset.toml\"\n\
             fetch = { kind = \"files\", base_url = \"http://127.0.0.1:9/\" }\n\
             [[entry]]\nid = \"manual\"\ndescription = \"d\"\nlicence = \"MIT\"\n\
             source_url = \"https://example.org/data\"\nmanifest = \"data/dataset.toml\"\n\
             [[entry]]\nid = \"synthetic\"\ndescription = \"d\"\nlicence = \"CC0-1.0\"\n\
             synthetic = \"scene\"\n",
        )
        .unwrap();
        fs::write(dir.join("fetch/e.sha256"), list).unwrap();
        Suite::load(&dir.join("suite.toml")).unwrap()
    }

    #[test]
    fn verified_files_are_not_downloaded_again() {
        let dir = temp_dir("skip");
        let list = format!(
            "{}  images/a.png\n{}  poses.txt\n",
            sha256_hex(b"image"),
            sha256_hex(b"poses")
        );
        let suite = offline_suite(&dir.join("suite"), &list);
        let cache = dir.join("cache");
        let files = entry_dir(&cache, "e");
        fs::create_dir_all(files.join("images")).unwrap();
        fs::write(files.join("images/a.png"), b"image").unwrap();
        fs::write(files.join("poses.txt"), b"poses").unwrap();

        let entry = &suite.manifest.entries[0];
        assert_eq!(
            fetch_entry(&suite, entry, &cache).unwrap(),
            FetchOutcome::UpToDate { files: 2 }
        );

        // A changed file is fetched again; the download fails here.
        fs::write(files.join("poses.txt"), b"changed").unwrap();
        let err = fetch_entry(&suite, entry, &cache).unwrap_err();
        assert!(format!("{err:#}").contains("poses.txt"), "{err:#}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn entries_without_a_fetch_source_are_not_errors() {
        let dir = temp_dir("manual");
        let suite = offline_suite(&dir, &format!("{}  a.png\n", sha256_hex(b"")));
        let cache = dir.join("cache");
        assert_eq!(
            fetch_entry(&suite, &suite.manifest.entries[1], &cache).unwrap(),
            FetchOutcome::Manual {
                source_url: Some("https://example.org/data".to_string())
            }
        );
        assert_eq!(
            fetch_entry(&suite, &suite.manifest.entries[2], &cache).unwrap(),
            FetchOutcome::NoImages
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn committed_fetch_lists_are_valid() {
        let suite =
            Suite::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../regression/suite.toml"))
                .unwrap();
        let mut fetchable = 0;
        for entry in suite.manifest.entries.iter().filter(|e| e.fetch.is_some()) {
            let list = read_fetch_list(&suite, &entry.id).unwrap();
            for file in entry.corners.iter().flat_map(|c| c.files()) {
                assert!(list.contains_key(file), "{}: {file} not fetched", entry.id);
            }
            fetchable += 1;
        }
        assert!(fetchable > 0);
    }

    #[test]
    fn fetch_lists_with_unsafe_paths_are_rejected() {
        let dir = temp_dir("unsafe-list");
        let suite = offline_suite(&dir, &format!("{}  ../a.png\n", sha256_hex(b"")));
        assert!(read_fetch_list(&suite, "e").is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// The suite's image entries that are also library samples fetch the
    /// same files from the same source, and run the same manifest. The
    /// sample's copies live in the published dataset crate, which cannot
    /// read files outside its package.
    #[test]
    fn suite_entries_match_their_library_samples() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let suite = Suite::load(&root.join("regression/suite.toml")).unwrap();
        for (id, manifest) in [
            ("kuka_1", "data/kuka_1/dataset.toml"),
            ("ds8", "data/DS8/dataset.toml"),
        ] {
            let sample = samples::sample(id).unwrap();
            let entry = suite.manifest.entries.iter().find(|e| e.id == id).unwrap();
            assert_eq!(entry.fetch.as_ref(), Some(sample.source()), "{id}");
            let files = sample.files();
            for (rel, sha) in read_fetch_list(&suite, id).unwrap() {
                assert_eq!(files.get(&rel), Some(&sha), "{id}: {rel}");
            }
            let local =
                vision_calibration_dataset::DatasetSpec::from_path(root.join(manifest)).unwrap();
            assert_eq!(
                serde_json::to_value(local).unwrap(),
                serde_json::to_value(sample.manifest()).unwrap(),
                "{id}"
            );
        }
    }
}
