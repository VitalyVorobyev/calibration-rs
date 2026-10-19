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
            ensure!(
                entry.manifest.is_none() || entry.synthetic.is_none(),
                "suite entry {:?} names both a dataset manifest and a synthetic scene",
                entry.id
            );
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
fn parse_checksums(text: &str) -> Result<BTreeMap<String, String>> {
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
