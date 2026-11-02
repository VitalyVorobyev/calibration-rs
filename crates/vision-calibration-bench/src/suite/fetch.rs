//! Download an image entry's files from their publisher into a local cache.
//!
//! An entry's [`FetchSource`] (in `suite.toml`) says where the files are
//! published; its fetch list (`fetch/<id>.sha256`, in `sha256sum` format)
//! names every file the dataset manifest reads, by its path under the
//! dataset root, with its SHA-256. [`fetch_entry`] downloads the files into
//! [`entry_dir`] and verifies each one; files already there and matching
//! their checksum are not downloaded again. `suite freeze --images <cache>`
//! then reads the entry's images from the cache instead of the manifest's
//! own directory.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};

use super::manifest::{parse_checksums, safe_relative_path};
use super::{FetchSource, Suite, SuiteEntry};

/// How many files an error lists before summarizing the rest.
const MAX_LISTED_FILES: usize = 5;

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

/// SHA-256 (lowercase hex) of the file at `path`, read in chunks.
///
/// # Errors
///
/// Fails when the file cannot be read.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = io::Read::read(&mut file, &mut buf)
            .with_context(|| format!("read {}", path.display()))?;
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
/// Fails on an unsafe path in `list` or a file that exists but cannot be
/// read.
pub fn unverified<'a>(dir: &Path, list: &'a BTreeMap<String, String>) -> Result<Vec<&'a str>> {
    let mut bad = Vec::new();
    for (rel, expected) in list {
        let path = dir.join(safe_relative_path(rel)?);
        if !path.is_file() || &sha256_file(&path)? != expected {
            bad.push(rel.as_str());
        }
    }
    Ok(bad)
}

/// Check that every file of an entry's fetch list is in `dir` and matches
/// its checksum.
///
/// # Errors
///
/// Fails when a file is missing or does not match, naming the first few.
pub fn verify(dir: &Path, list: &BTreeMap<String, String>) -> Result<()> {
    let bad = unverified(dir, list)?;
    ensure!(
        bad.is_empty(),
        "{} of {} files under {} are missing or do not match their checksum: {}",
        bad.len(),
        list.len(),
        dir.display(),
        summarize(&bad)
    );
    Ok(())
}

fn summarize(files: &[&str]) -> String {
    let mut text = files
        .iter()
        .take(MAX_LISTED_FILES)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if files.len() > MAX_LISTED_FILES {
        text.push_str(&format!(" and {} more", files.len() - MAX_LISTED_FILES));
    }
    text
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
    let missing = unverified(&dir, &list)?;
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
            fs::create_dir_all(cache).with_context(|| format!("create {}", cache.display()))?;
            let archive = cache.join(format!("{}.zip", entry.id));
            download(url, &archive)?;
            let file =
                fs::File::open(&archive).with_context(|| format!("open {}", archive.display()))?;
            let wanted: BTreeSet<&str> = missing.iter().copied().collect();
            let extracted =
                extract_zip(file, root, &dir, &wanted).with_context(|| format!("unpack {url}"))?;
            fs::remove_file(&archive).with_context(|| format!("remove {}", archive.display()))?;
            ensure!(
                extracted.len() == wanted.len(),
                "{url} lacks {} listed file(s): {}",
                wanted.len() - extracted.len(),
                summarize(&wanted.difference(&extracted).copied().collect::<Vec<_>>())
            );
        }
    }
    let bad = unverified(&dir, &list)?;
    ensure!(
        bad.is_empty(),
        "{} downloaded file(s) of {:?} do not match {}; the published files changed: {}",
        bad.len(),
        entry.id,
        suite.fetch_list_path(&entry.id).display(),
        summarize(&bad)
    );
    Ok(FetchOutcome::Fetched {
        files: list.len(),
        downloaded: missing.len(),
    })
}

/// Download `url` to `dest`, through a `.part` file renamed on success so
/// an interrupted download never leaves a complete-looking file.
fn download(url: &str, dest: &Path) -> Result<()> {
    let dir = dest
        .parent()
        .with_context(|| format!("{} has no parent directory", dest.display()))?;
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let mut part = dest.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let mut response = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let mut file = fs::File::create(&part).with_context(|| format!("create {}", part.display()))?;
    io::copy(&mut response.body_mut().as_reader(), &mut file)
        .with_context(|| format!("download {url}"))?;
    drop(file);
    fs::rename(&part, dest).with_context(|| format!("write {}", dest.display()))
}

/// Unpack the members of a ZIP archive under its directory `root` (`""`:
/// the whole archive) whose path below `root` is in `wanted`, into `dir`.
/// Returns the paths unpacked.
///
/// # Errors
///
/// Fails on an invalid archive, on any member name that is not a safe
/// relative path (absolute, `..`, a drive) — the archive is rejected, not
/// just the member — and on I/O errors.
pub fn extract_zip<'w, R: Read + Seek>(
    reader: R,
    root: &str,
    dir: &Path,
    wanted: &BTreeSet<&'w str>,
) -> Result<BTreeSet<&'w str>> {
    let mut archive = zip::ZipArchive::new(reader).context("read ZIP archive")?;
    let prefix = match root.trim_end_matches('/') {
        "" => String::new(),
        root => format!("{root}/"),
    };
    let mut extracted = BTreeSet::new();
    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        let name = member.name()?.into_owned();
        let trimmed = name.trim_end_matches('/');
        if trimmed.is_empty() {
            bail!("archive member {index} has an empty name");
        }
        safe_relative_path(trimmed).context("unsafe path in archive")?;
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
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        let mut file =
            fs::File::create(&path).with_context(|| format!("create {}", path.display()))?;
        io::copy(&mut member, &mut file).with_context(|| format!("unpack {name}"))?;
        extracted.insert(rel);
    }
    Ok(extracted)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

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
        assert!(unverified(&dir, &list).unwrap().is_empty());
        verify(&dir, &list).unwrap();
        fs::remove_dir_all(&dir).unwrap();
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
            assert!(
                format!("{err:#}").contains("unsafe path"),
                "{evil}: {err:#}"
            );
            assert!(!dir.join("evil.txt").exists());
            fs::remove_dir_all(&dir).unwrap();
        }
    }
}
