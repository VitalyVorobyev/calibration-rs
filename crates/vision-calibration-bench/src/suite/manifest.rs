//! The suite manifest (`suite.toml`) and its checksum file (`SHA256SUMS`).
//!
//! Layout of a suite directory:
//!
//! ```text
//! suite.toml                          entries: provenance, licence, config
//! SHA256SUMS                          `sha256sum` format, one line per data file
//! observations/<id>.json              the entry's ObservationSet
//! truth/<id>.json                     a synthetic entry's true parameters
//! results/<id>/calibration-rs.json    the entry's reference result
//! fetch/<id>.sha256                   files `suite fetch` downloads, with checksums
//! ```
//!
//! File locations follow from the entry id, so the manifest names none of
//! them. `SHA256SUMS` is written by `suite freeze` and checks with the
//! standard tools (`sha256sum -c SHA256SUMS`, `shasum -a 256 -c SHA256SUMS`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub use super::corners::PublishedCorners;

/// Name of the checksum file next to `suite.toml`.
pub const CHECKSUM_FILE: &str = "SHA256SUMS";

/// A loaded suite: its manifest and the directory holding it.
#[derive(Debug, Clone)]
pub struct Suite {
    /// Directory of `suite.toml`; every suite path is relative to it.
    pub root: PathBuf,
    /// The parsed manifest.
    pub manifest: SuiteManifest,
}

/// `suite.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteManifest {
    /// Suite version, e.g. `"2026.1"`. Reference results record the
    /// version they were produced for.
    pub suite_version: String,
    /// The entries, in run order.
    #[serde(rename = "entry")]
    pub entries: Vec<SuiteEntry>,
}

/// One `[[entry]]` of `suite.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SuiteEntry {
    /// Unique id; names the observation and reference-result files.
    pub id: String,
    /// One line on what the entry exercises.
    pub description: String,
    /// Licence of the data (SPDX identifier where one exists).
    pub licence: String,
    /// How to cite the data, when its authors ask for it.
    #[serde(default)]
    pub citation: Option<String>,
    /// Where the original images are published.
    #[serde(default)]
    pub source_url: Option<String>,
    /// Dataset manifest the observations are frozen from, relative to the
    /// suite directory. `None` for entries that have no images.
    #[serde(default)]
    pub manifest: Option<PathBuf>,
    /// The corner files the publisher released with the images, which the
    /// observations are frozen from instead of from detection. Excludes
    /// `manifest` and `synthetic`; needs `fetch`.
    #[serde(default)]
    pub corners: Option<PublishedCorners>,
    /// Where `suite fetch` downloads the files the dataset manifest or the
    /// corner source reads. `None` when they must be downloaded by hand
    /// from `source_url`. The files and their checksums are listed in
    /// `fetch/<id>.sha256`.
    #[serde(default)]
    pub fetch: Option<FetchSource>,
    /// For a synthetic entry, the benchmark scene its observations and
    /// truth are frozen from (`calib-bench solver` scene id). Excludes
    /// `manifest`.
    #[serde(default)]
    pub synthetic: Option<String>,
    /// Overrides merged into the problem's default config.
    #[serde(default)]
    pub config: Option<Value>,
    /// Drift tolerance against the reference result.
    #[serde(default)]
    pub tolerance: Tolerance,
}

/// How the files of an image entry are downloaded from their publisher.
///
/// Every file named in the entry's `fetch/<id>.sha256` is placed at that
/// path, relative to the dataset root the manifest's paths resolve
/// against.
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

/// Allowed difference between a number in a result and in its reference:
/// `|a − b| ≤ abs + rel · max(|a|, |b|)`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Tolerance {
    /// Relative part.
    pub rel: f64,
    /// Absolute part, for values near zero.
    pub abs: f64,
}

impl Default for Tolerance {
    /// Tight enough to flag any change in the solution, loose enough for
    /// the last-bit differences between platforms and compilers.
    fn default() -> Self {
        Self {
            rel: 1e-6,
            abs: 1e-9,
        }
    }
}

impl Tolerance {
    /// Whether `a` and `b` agree within the tolerance.
    pub fn accepts(&self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.abs + self.rel * a.abs().max(b.abs())
    }
}

impl Suite {
    /// Load `suite.toml` from `path`.
    ///
    /// # Errors
    ///
    /// Fails on unreadable or invalid TOML and on duplicate entry ids.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let manifest: SuiteManifest =
            toml::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let mut seen = std::collections::BTreeSet::new();
        for entry in &manifest.entries {
            ensure!(
                seen.insert(entry.id.as_str()),
                "duplicate suite entry id {:?}",
                entry.id
            );
            let sources = [
                entry.manifest.is_some(),
                entry.corners.is_some(),
                entry.synthetic.is_some(),
            ];
            ensure!(
                sources.iter().filter(|&&s| s).count() <= 1,
                "suite entry {:?} names more than one of a dataset manifest, published \
                 corners and a synthetic scene",
                entry.id
            );
            ensure!(
                entry.fetch.is_none() || entry.manifest.is_some() || entry.corners.is_some(),
                "suite entry {:?} has a fetch source but no dataset manifest or corners",
                entry.id
            );
            if let Some(corners) = &entry.corners {
                ensure!(
                    entry.fetch.is_some(),
                    "suite entry {:?}: published corners are read from the fetched files, \
                     so the entry needs a fetch source",
                    entry.id
                );
                corners
                    .validate()
                    .with_context(|| format!("suite entry {:?}: corners", entry.id))?;
            }
            if let Some(FetchSource::Files { base_url }) = &entry.fetch {
                ensure!(
                    base_url.ends_with('/'),
                    "suite entry {:?}: fetch base_url must end in `/`",
                    entry.id
                );
            }
        }
        let root = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self { root, manifest })
    }

    /// The entries selected by `only` (all of them when it is empty).
    ///
    /// # Errors
    ///
    /// Fails when `only` names an id the suite does not have.
    pub fn select(&self, only: &[String]) -> Result<Vec<&SuiteEntry>> {
        for id in only {
            ensure!(
                self.manifest.entries.iter().any(|e| &e.id == id),
                "no suite entry {id:?}"
            );
        }
        Ok(self
            .manifest
            .entries
            .iter()
            .filter(|e| only.is_empty() || only.contains(&e.id))
            .collect())
    }

    /// Path of an entry's observation file, relative to [`Self::root`].
    pub fn observations_rel(id: &str) -> String {
        format!("observations/{id}.json")
    }

    /// Path of an entry's observation file.
    pub fn observations_path(&self, id: &str) -> PathBuf {
        self.root.join(Self::observations_rel(id))
    }

    /// Path of a synthetic entry's truth file, relative to [`Self::root`].
    pub fn truth_rel(id: &str) -> String {
        format!("truth/{id}.json")
    }

    /// Path of a synthetic entry's truth file.
    pub fn truth_path(&self, id: &str) -> PathBuf {
        self.root.join(Self::truth_rel(id))
    }

    /// Path of an image entry's fetch list: the files `suite fetch`
    /// downloads and their SHA-256 digests, in `sha256sum` format.
    pub fn fetch_list_path(&self, id: &str) -> PathBuf {
        self.root.join("fetch").join(format!("{id}.sha256"))
    }

    /// Path of an entry's reference result.
    pub fn reference_path(&self, id: &str) -> PathBuf {
        self.root
            .join("results")
            .join(id)
            .join("calibration-rs.json")
    }

    /// Read the observation file of `id` after checking it against
    /// `SHA256SUMS`.
    ///
    /// # Errors
    ///
    /// Fails when the file is missing, has no checksum line, or does not
    /// match it. A checksum is never skipped.
    pub fn read_verified_observations(&self, id: &str) -> Result<Vec<u8>> {
        self.read_verified(&Self::observations_rel(id))
    }

    /// Read the truth file of `id` after checking it against `SHA256SUMS`.
    ///
    /// # Errors
    ///
    /// As [`Self::read_verified_observations`].
    pub fn read_verified_truth(&self, id: &str) -> Result<Vec<u8>> {
        self.read_verified(&Self::truth_rel(id))
    }

    fn read_verified(&self, rel: &str) -> Result<Vec<u8>> {
        let sums = self.read_checksums()?;
        let expected = sums
            .get(rel)
            .with_context(|| format!("{CHECKSUM_FILE} has no line for {rel}"))?;
        let path = self.root.join(rel);
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let actual = sha256_hex(&bytes);
        if &actual != expected {
            bail!("{rel}: SHA-256 {actual} does not match {CHECKSUM_FILE} ({expected})");
        }
        Ok(bytes)
    }

    /// Parse `SHA256SUMS` into `relative path → hex digest`.
    ///
    /// # Errors
    ///
    /// Fails when the file is missing or a line is malformed.
    pub fn read_checksums(&self) -> Result<BTreeMap<String, String>> {
        let path = self.root.join(CHECKSUM_FILE);
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        parse_checksums(&text).with_context(|| format!("parse {}", path.display()))
    }

    /// Rewrite `SHA256SUMS` from the observation and truth files on disk,
    /// one line per file.
    ///
    /// # Errors
    ///
    /// Fails on I/O errors.
    pub fn write_checksums(&self) -> Result<()> {
        let mut text = String::new();
        let mut rels: Vec<String> = self
            .manifest
            .entries
            .iter()
            .flat_map(|e| [Self::observations_rel(&e.id), Self::truth_rel(&e.id)])
            .filter(|rel| self.root.join(rel).is_file())
            .collect();
        rels.sort();
        for rel in rels {
            let bytes = std::fs::read(self.root.join(&rel))?;
            writeln!(text, "{}  {rel}", sha256_hex(&bytes)).expect("writing to a String");
        }
        let path = self.root.join(CHECKSUM_FILE);
        std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))
    }
}

/// Turn a `/`-separated relative path from a fetch list, a corner source
/// or an archive into a path that stays inside the directory it is joined
/// to.
///
/// # Errors
///
/// Fails on an empty path, an absolute path, a drive or `\` separator, a
/// NUL byte, and an empty, `.` or `..` component.
pub fn safe_relative_path(name: &str) -> Result<PathBuf> {
    ensure!(!name.is_empty(), "empty path");
    ensure!(
        !name.contains(['\\', ':', '\0']),
        "{name:?}: paths use `/` and no drive or NUL"
    );
    ensure!(!name.starts_with('/'), "{name:?} is an absolute path");
    let mut path = PathBuf::new();
    for part in name.split('/') {
        ensure!(
            !part.is_empty() && part != "." && part != "..",
            "{name:?} has an empty, `.` or `..` component"
        );
        path.push(part);
    }
    Ok(path)
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(hex, "{byte:02x}").expect("writing to a String");
    }
    hex
}

/// Parse `sha256sum` output: `<64 hex>  <path>` (text mode) or
/// `<64 hex> *<path>` (binary mode) per line.
pub(crate) fn parse_checksums(text: &str) -> Result<BTreeMap<String, String>> {
    let mut sums = BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (hex, rest) = line
            .split_once(' ')
            .with_context(|| format!("line {}: expected `<sha256>  <path>`", n + 1))?;
        let path = rest.strip_prefix(' ').or_else(|| rest.strip_prefix('*'));
        let path = path.with_context(|| format!("line {}: expected two spaces", n + 1))?;
        ensure!(
            hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()),
            "line {}: {hex:?} is not a SHA-256 digest",
            n + 1
        );
        sums.insert(path.to_string(), hex.to_ascii_lowercase());
    }
    Ok(sums)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_empty_input_is_the_standard_digest() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn parses_text_and_binary_mode_lines() {
        let hex = "a".repeat(64);
        let sums =
            parse_checksums(&format!("{hex}  observations/a.json\n{hex} *b.json\n")).unwrap();
        assert_eq!(sums["observations/a.json"], hex);
        assert_eq!(sums["b.json"], hex);
    }

    #[test]
    fn rejects_malformed_checksum_lines() {
        assert!(parse_checksums("abc  x.json").is_err());
        assert!(parse_checksums(&format!("{}x.json", "a".repeat(64))).is_err());
    }

    #[test]
    fn safe_paths_stay_inside_their_directory() {
        assert_eq!(
            safe_relative_path("images/camera0/a.png").unwrap(),
            Path::new("images").join("camera0").join("a.png")
        );
        for unsafe_name in [
            "",
            "/etc/passwd",
            "../a.png",
            "images/../../a.png",
            "images/./a.png",
            "images//a.png",
            "images/",
            "C:/a.png",
            "images\\a.png",
            "a\0.png",
        ] {
            assert!(
                safe_relative_path(unsafe_name).is_err(),
                "{unsafe_name:?} accepted"
            );
        }
    }

    #[test]
    fn tolerance_mixes_relative_and_absolute() {
        let tol = Tolerance::default();
        assert!(tol.accepts(1000.0, 1000.0005));
        assert!(!tol.accepts(1000.0, 1000.01));
        assert!(tol.accepts(0.0, 5e-10));
        assert!(!tol.accepts(0.0, 1e-8));
    }

    #[test]
    fn manifest_parses_entries_with_config_tables() {
        let manifest: SuiteManifest = toml::from_str(
            r#"
            suite_version = "2026.1"

            [[entry]]
            id = "a"
            description = "planar"
            licence = "MIT"

            [entry.config.solver]
            max_iters = 200

            [entry.tolerance]
            rel = 1e-4
            "#,
        )
        .unwrap();
        let entry = &manifest.entries[0];
        assert_eq!(entry.config.as_ref().unwrap()["solver"]["max_iters"], 200);
        assert_eq!(entry.tolerance.rel, 1e-4);
        assert_eq!(entry.tolerance.abs, Tolerance::default().abs);
    }

    #[test]
    fn fetch_sources_parse_and_need_a_manifest() {
        let manifest: SuiteManifest = toml::from_str(
            r#"
            suite_version = "t"

            [[entry]]
            id = "a"
            description = "d"
            licence = "MIT"
            manifest = "a/dataset.toml"
            fetch = { kind = "zip", url = "https://example.org/a.zip", root = "A" }

            [[entry]]
            id = "b"
            description = "d"
            licence = "MIT"
            manifest = "b/dataset.toml"
            fetch = { kind = "files", base_url = "https://example.org/b/" }
            "#,
        )
        .unwrap();
        assert_eq!(
            manifest.entries[0].fetch,
            Some(FetchSource::Zip {
                url: "https://example.org/a.zip".into(),
                root: "A".into()
            })
        );
        assert_eq!(
            manifest.entries[1].fetch,
            Some(FetchSource::Files {
                base_url: "https://example.org/b/".into()
            })
        );

        let dir =
            std::env::temp_dir().join(format!("calib-suite-fetch-src-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let entry = "[[entry]]\nid = \"a\"\ndescription = \"d\"\nlicence = \"MIT\"\n";
        for (extra, message) in [
            (
                "fetch = { kind = \"zip\", url = \"https://example.org/a.zip\" }\n",
                "no dataset manifest",
            ),
            (
                "manifest = \"a.toml\"\nfetch = { kind = \"files\", base_url = \"https://example.org/a\" }\n",
                "must end in `/`",
            ),
        ] {
            std::fs::write(
                dir.join("suite.toml"),
                format!("suite_version = \"t\"\n{entry}{extra}"),
            )
            .unwrap();
            let err = Suite::load(&dir.join("suite.toml")).unwrap_err();
            assert!(err.to_string().contains(message), "{err}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corner_sources_parse_and_are_checked() {
        let corners = "[entry.corners]\nformat = \"opencv_storage\"\n\
            problem = \"planar_intrinsics\"\nobject = \"object.xml\"\n\
            cameras = [{ id = \"left\", points = \"left.xml\", image_size = [1280, 800] }]\n";
        let entry = "[[entry]]\nid = \"a\"\ndescription = \"d\"\nlicence = \"MIT\"\n";
        let fetch = "fetch = { kind = \"files\", base_url = \"https://example.org/a/\" }\n";

        let dir = std::env::temp_dir().join(format!("calib-suite-corners-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let load = |extra: &str| {
            std::fs::write(
                dir.join("suite.toml"),
                format!("suite_version = \"t\"\n{entry}{extra}{corners}"),
            )
            .unwrap();
            Suite::load(&dir.join("suite.toml"))
        };
        let suite = load(fetch).unwrap();
        let source = suite.manifest.entries[0].corners.as_ref().unwrap();
        assert_eq!(
            source.files().collect::<Vec<_>>(),
            ["object.xml", "left.xml"]
        );
        assert_eq!(source.cameras[0].image_size, [1280, 800]);
        for (extra, message) in [
            ("", "needs a fetch source"),
            (
                &format!("{fetch}manifest = \"a/dataset.toml\"\n") as &str,
                "more than one",
            ),
        ] {
            let err = load(extra).unwrap_err();
            assert!(format!("{err:#}").contains(message), "{err:#}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tampered_observations_fail_the_checksum() {
        let dir = std::env::temp_dir().join(format!("calib-suite-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("observations")).unwrap();
        std::fs::write(
            dir.join("suite.toml"),
            "suite_version = \"t\"\n[[entry]]\nid = \"a\"\ndescription = \"d\"\nlicence = \"MIT\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("observations/a.json"), b"{}").unwrap();
        let suite = Suite::load(&dir.join("suite.toml")).unwrap();
        suite.write_checksums().unwrap();
        assert!(suite.read_verified_observations("a").is_ok());

        std::fs::write(dir.join("observations/a.json"), b"{ }").unwrap();
        let err = suite.read_verified_observations("a").unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
