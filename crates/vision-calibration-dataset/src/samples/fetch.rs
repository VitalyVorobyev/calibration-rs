//! Downloading a dataset's files and verifying them against their
//! checksums.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{
    CACHE_DIR_ENV, FetchError, FetchSource, safe_relative_path, sample, sample_dir, summarize_names,
};

/// The name of the manifest [`fetch`] writes into a sample's directory.
const MANIFEST_FILE: &str = "dataset.toml";

/// Download the sample `id` into `<cache>/<id>/`, verify every file
/// against its SHA-256 and write the sample's manifest there. Returns the
/// manifest's path.
///
/// Files already there and matching their checksum are kept; nothing is
/// downloaded when all of them do, so a second call works offline.
///
/// # Errors
///
/// [`FetchError::UnknownSample`] for an id not in the
/// [`catalogue`](super::catalogue); [`FetchError::Download`] and
/// [`FetchError::Archive`] when the publisher cannot be reached or sends
/// something else; [`FetchError::Checksum`] when a downloaded file does not
/// match (the publisher's files changed); [`FetchError::Io`] on a cache
/// that cannot be written.
pub fn fetch(id: &str, cache: &Path) -> Result<PathBuf, FetchError> {
    let sample = sample(id)?;
    let dir = sample_dir(cache, id);
    fetch_files(sample.source(), &sample.files(), &dir)?;
    let manifest = dir.join(MANIFEST_FILE);
    fs::write(&manifest, sample.embedded().manifest)
        .map_err(|source| io_error(&manifest, source))?;
    Ok(manifest)
}

/// The cache directory samples are fetched into when the caller names
/// none: [`CACHE_DIR_ENV`] when set, else `vision-calibration` in the
/// platform's cache directory (`~/Library/Caches` on macOS,
/// `$XDG_CACHE_HOME` or `~/.cache` elsewhere on Unix, `%LOCALAPPDATA%` on
/// Windows).
///
/// # Errors
///
/// [`FetchError::NoCacheDir`] when neither is set.
pub fn default_cache_dir() -> Result<PathBuf, FetchError> {
    let env = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(dir) = env(CACHE_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    let base = if cfg!(windows) {
        env("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|home| PathBuf::from(home).join("Library/Caches"))
    } else {
        env("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env("HOME").map(|home| PathBuf::from(home).join(".cache")))
    };
    base.map(|dir| dir.join("vision-calibration"))
        .ok_or(FetchError::NoCacheDir)
}

/// What [`fetch_files`] did.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchOutcome {
    /// Every listed file was already there and matched its checksum.
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
}

/// Download the files of `list` (path → SHA-256) that are missing from
/// `dir` or do not match, from `source`, and verify all of them.
///
/// # Errors
///
/// As [`fetch`], minus the unknown sample.
#[doc(hidden)]
pub fn fetch_files(
    source: &FetchSource,
    list: &BTreeMap<String, String>,
    dir: &Path,
) -> Result<FetchOutcome, FetchError> {
    let missing = unverified(dir, list)?;
    if missing.is_empty() {
        return Ok(FetchOutcome::UpToDate { files: list.len() });
    }
    match source {
        FetchSource::Files { base_url } => {
            for rel in &missing {
                download(
                    &format!("{base_url}{rel}"),
                    &dir.join(safe_relative_path(rel)?),
                )?;
            }
        }
        FetchSource::Zip { url, root } => {
            let mut archive = dir.as_os_str().to_owned();
            archive.push(".zip");
            let archive = PathBuf::from(archive);
            download(url, &archive)?;
            let file = fs::File::open(&archive).map_err(|source| io_error(&archive, source))?;
            let wanted: BTreeSet<&str> = missing.iter().copied().collect();
            let extracted = extract_zip(file, root, dir, &wanted).map_err(|err| match err {
                FetchError::Archive { message, .. } => FetchError::Archive {
                    url: url.clone(),
                    message,
                },
                other => other,
            })?;
            fs::remove_file(&archive).map_err(|source| io_error(&archive, source))?;
            if extracted.len() != wanted.len() {
                let lacking: Vec<&str> = wanted.difference(&extracted).copied().collect();
                return Err(FetchError::Archive {
                    url: url.clone(),
                    message: format!(
                        "lacks {} listed file(s): {}",
                        lacking.len(),
                        summarize_names(&lacking)
                    ),
                });
            }
        }
    }
    verify(dir, list)?;
    Ok(FetchOutcome::Fetched {
        files: list.len(),
        downloaded: missing.len(),
    })
}

fn io_error(path: &Path, source: io::Error) -> FetchError {
    FetchError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// SHA-256 (lowercase hex) of the file at `path`, read in chunks.
///
/// # Errors
///
/// [`FetchError::Io`] when the file cannot be read.
#[doc(hidden)]
pub fn sha256_file(path: &Path) -> Result<String, FetchError> {
    let mut file = fs::File::open(path).map_err(|source| io_error(path, source))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|source| io_error(path, source))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// The files of `list` that are missing under `dir` or do not match their
/// checksum, in list order.
///
/// # Errors
///
/// [`FetchError::InvalidList`] on an unsafe path; [`FetchError::Io`] on a
/// file that exists but cannot be read.
#[doc(hidden)]
pub fn unverified<'a>(
    dir: &Path,
    list: &'a BTreeMap<String, String>,
) -> Result<Vec<&'a str>, FetchError> {
    let mut bad = Vec::new();
    for (rel, expected) in list {
        let path = dir.join(safe_relative_path(rel)?);
        if !path.is_file() || &sha256_file(&path)? != expected {
            bad.push(rel.as_str());
        }
    }
    Ok(bad)
}

/// Check that every file of `list` is in `dir` and matches its checksum.
///
/// # Errors
///
/// [`FetchError::Checksum`] naming the files that are missing or do not
/// match.
#[doc(hidden)]
pub fn verify(dir: &Path, list: &BTreeMap<String, String>) -> Result<(), FetchError> {
    let bad = unverified(dir, list)?;
    if bad.is_empty() {
        Ok(())
    } else {
        Err(FetchError::Checksum {
            dir: dir.to_path_buf(),
            total: list.len(),
            files: bad.into_iter().map(str::to_string).collect(),
        })
    }
}

/// The first few of `files`, then how many more.
#[doc(hidden)]
pub fn summarize_files(files: &[&str]) -> String {
    summarize_names(files)
}

/// Download `url` to `dest`, through a `.part` file renamed on success so
/// an interrupted download never leaves a complete-looking file.
fn download(url: &str, dest: &Path) -> Result<(), FetchError> {
    let dir = dest.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|source| io_error(dir, source))?;
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let failed = |message: String| FetchError::Download {
        url: url.to_string(),
        message,
    };
    let mut response = ureq::get(url)
        .call()
        .map_err(|err| failed(err.to_string()))?;
    let mut file = fs::File::create(&part).map_err(|source| io_error(&part, source))?;
    io::copy(&mut response.body_mut().as_reader(), &mut file)
        .map_err(|err| failed(err.to_string()))?;
    drop(file);
    fs::rename(&part, dest).map_err(|source| io_error(dest, source))
}

/// Unpack the members of a ZIP archive under its directory `root` (`""`:
/// the whole archive) whose path below `root` is in `wanted`, into `dir`.
/// Returns the paths unpacked.
///
/// # Errors
///
/// [`FetchError::Archive`] on an invalid archive and on any member name
/// that is not a safe relative path (absolute, `..`, a drive): the archive
/// is rejected, not just the member. [`FetchError::Io`] on a file that
/// cannot be written.
#[doc(hidden)]
pub fn extract_zip<'w, R: Read + Seek>(
    reader: R,
    root: &str,
    dir: &Path,
    wanted: &BTreeSet<&'w str>,
) -> Result<BTreeSet<&'w str>, FetchError> {
    let invalid = |message: String| FetchError::Archive {
        url: String::new(),
        message,
    };
    let mut archive =
        zip::ZipArchive::new(reader).map_err(|err| invalid(format!("not a ZIP archive: {err}")))?;
    let prefix = match root.trim_end_matches('/') {
        "" => String::new(),
        root => format!("{root}/"),
    };
    let mut extracted = BTreeSet::new();
    for index in 0..archive.len() {
        let mut member = archive
            .by_index(index)
            .map_err(|err| invalid(format!("member {index}: {err}")))?;
        let name = member
            .name()
            .map_err(|err| invalid(format!("member {index}: {err}")))?
            .into_owned();
        let trimmed = name.trim_end_matches('/');
        if trimmed.is_empty() {
            return Err(invalid(format!("member {index} has an empty name")));
        }
        safe_relative_path(trimmed).map_err(|err| invalid(format!("unsafe path: {err}")))?;
        if member.is_dir() {
            continue;
        }
        let Some(rel) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(&rel) = wanted.get(rel) else {
            continue;
        };
        let path = dir.join(safe_relative_path(rel)?);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;
        }
        let mut file = fs::File::create(&path).map_err(|source| io_error(&path, source))?;
        io::copy(&mut member, &mut file).map_err(|source| io_error(&path, source))?;
        extracted.insert(rel);
    }
    Ok(extracted)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use super::*;

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// A fresh, empty directory under the system temp dir.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vision-calibration-samples-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn zip_of(members: &[(&str, &[u8])]) -> Cursor<Vec<u8>> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for (name, bytes) in members {
            if name.ends_with('/') {
                writer.add_directory(*name, options).unwrap();
            } else {
                writer.start_file(*name, options).unwrap();
                writer.write_all(bytes).unwrap();
            }
        }
        let mut cursor = writer.finish().unwrap();
        cursor.set_position(0);
        cursor
    }

    #[test]
    fn verification_reports_missing_and_changed_files() {
        let dir = temp_dir("verify");
        fs::write(dir.join("a.txt"), b"a").unwrap();
        fs::write(dir.join("b.txt"), b"tampered").unwrap();
        let list = BTreeMap::from([
            ("a.txt".to_string(), sha256_hex(b"a")),
            ("b.txt".to_string(), sha256_hex(b"b")),
            ("c.txt".to_string(), sha256_hex(b"c")),
        ]);
        assert_eq!(unverified(&dir, &list).unwrap(), ["b.txt", "c.txt"]);
        let err = verify(&dir, &list).unwrap_err().to_string();
        assert!(err.contains("2 of 3 files"), "{err}");
        assert_eq!(sha256_file(&dir.join("a.txt")).unwrap(), sha256_hex(b"a"));

        fs::write(dir.join("b.txt"), b"b").unwrap();
        fs::write(dir.join("c.txt"), b"c").unwrap();
        verify(&dir, &list).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn verified_files_are_not_downloaded_again() {
        let dir = temp_dir("skip");
        fs::create_dir_all(dir.join("images")).unwrap();
        fs::write(dir.join("images/a.png"), b"image").unwrap();
        fs::write(dir.join("poses.txt"), b"poses").unwrap();
        let list = BTreeMap::from([
            ("images/a.png".to_string(), sha256_hex(b"image")),
            ("poses.txt".to_string(), sha256_hex(b"poses")),
        ]);
        // An unroutable URL: any download fails, so a pass proves the files
        // on disk were used.
        let source = FetchSource::Files {
            base_url: "http://127.0.0.1:9/".to_string(),
        };
        assert_eq!(
            fetch_files(&source, &list, &dir).unwrap(),
            FetchOutcome::UpToDate { files: 2 }
        );

        fs::write(dir.join("poses.txt"), b"changed").unwrap();
        let err = fetch_files(&source, &list, &dir).unwrap_err();
        assert!(
            matches!(&err, FetchError::Download { url, .. } if url.ends_with("poses.txt")),
            "{err}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_cache_directory_follows_the_environment() {
        // Read-only on the process environment: the variable is not set in
        // the test environment, so the platform default applies.
        if std::env::var_os(CACHE_DIR_ENV).is_none()
            && let Ok(dir) = default_cache_dir()
        {
            assert!(dir.ends_with("vision-calibration"), "{}", dir.display());
        }
    }

    #[test]
    fn zip_extraction_takes_the_wanted_files_under_the_root() {
        let dir = temp_dir("zip");
        let archive = zip_of(&[
            ("DS/", b""),
            ("DS/images/", b""),
            ("DS/images/a.png", b"image"),
            ("DS/poses.txt", b"poses"),
            ("DS/notes.txt", b"not listed"),
            ("other/poses.txt", b"outside the root"),
        ]);
        let wanted = BTreeSet::from(["images/a.png", "poses.txt", "missing.txt"]);
        let extracted = extract_zip(archive, "DS", &dir, &wanted).unwrap();
        assert_eq!(extracted, BTreeSet::from(["images/a.png", "poses.txt"]));
        assert_eq!(fs::read(dir.join("images/a.png")).unwrap(), b"image");
        assert_eq!(fs::read(dir.join("poses.txt")).unwrap(), b"poses");
        assert!(!dir.join("notes.txt").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn zip_with_an_escaping_member_is_rejected() {
        for evil in ["DS/../../evil.txt", "/tmp/evil.txt"] {
            let dir = temp_dir("zip-evil");
            let archive = zip_of(&[("DS/ok.txt", b"ok"), (evil, b"evil")]);
            let wanted = BTreeSet::from(["ok.txt"]);
            let err = extract_zip(archive, "DS", &dir.join("out"), &wanted).unwrap_err();
            assert!(err.to_string().contains("unsafe path"), "{evil}: {err}");
            assert!(!dir.join("evil.txt").exists());
            fs::remove_dir_all(&dir).unwrap();
        }
    }
}
