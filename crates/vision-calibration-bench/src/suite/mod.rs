//! The public regression suite: image-free observations, frozen once, with
//! reference results that every change is checked against.
//!
//! - [`manifest`] — `suite.toml` and the `SHA256SUMS` checksum file.
//! - [`observations`] — the [`ObservationSet`] file format.
//! - [`reference`](mod@reference) — stored results and the drift check.
//! - `freeze` (feature `tier-b`) — dataset manifest + images →
//!   [`ObservationSet`].
//!
//! [`check_entry`] verifies an entry's checksum, calibrates from its
//! observations and compares the export with the reference result;
//! [`bless_entry`] writes that reference.

#[cfg(feature = "tier-b")]
pub mod freeze;
pub mod manifest;
pub mod observations;
pub mod reference;

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub use manifest::{Suite, SuiteEntry, Tolerance};
pub use observations::{OBSERVATION_SET_VERSION, ObservationSet, ObservedCamera, ProblemInput};
pub use reference::{Drift, ReferenceResult, drifts};

/// How many drifts a failed check lists before summarizing the rest.
const MAX_LISTED_DRIFTS: usize = 12;

/// The outcome of checking one entry.
#[derive(Debug, Clone)]
pub struct EntryReport {
    /// Entry id.
    pub id: String,
    /// Wall time of the check, in seconds.
    pub seconds: f64,
    /// Why the check failed; empty when it passed.
    pub failures: Vec<String>,
}

impl EntryReport {
    /// Whether the entry passed.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Read, checksum-verify and parse an entry's observations.
///
/// # Errors
///
/// Fails on a missing or mismatched checksum, invalid JSON or an
/// unsupported format version.
pub fn load_observations(suite: &Suite, id: &str) -> Result<ObservationSet> {
    let bytes = suite.read_verified_observations(id)?;
    let set: ObservationSet = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", Suite::observations_rel(id)))?;
    ensure!(
        set.version == OBSERVATION_SET_VERSION,
        "{}: observation format version {} (this build reads {OBSERVATION_SET_VERSION})",
        Suite::observations_rel(id),
        set.version
    );
    Ok(set)
}

/// Calibrate an entry from its verified observations; returns the export
/// as a reference stores it.
///
/// The per-feature residual lists are left out: they follow from the
/// parameters and the observations, and outweigh everything else in the
/// export a hundredfold. Their histograms and the reprojection statistics
/// stay.
///
/// # Errors
///
/// See [`load_observations`] and [`ProblemInput::calibrate`].
pub fn calibrate_entry(suite: &Suite, entry: &SuiteEntry) -> Result<Value> {
    let set = load_observations(suite, &entry.id)?;
    let mut export = set.input.calibrate(entry.config.as_ref())?;
    if let Some(Value::Object(residuals)) = export.get_mut("per_feature_residuals") {
        residuals.remove("target");
        residuals.remove("laser");
    }
    Ok(export)
}

/// Check one entry against its reference result. Every failure — checksum,
/// calibration, missing reference, drift — is reported, never raised.
pub fn check_entry(suite: &Suite, entry: &SuiteEntry) -> EntryReport {
    let start = Instant::now();
    let failures = match check_failures(suite, entry) {
        Ok(failures) => failures,
        Err(err) => vec![format!("{err:#}")],
    };
    EntryReport {
        id: entry.id.clone(),
        seconds: start.elapsed().as_secs_f64(),
        failures,
    }
}

fn check_failures(suite: &Suite, entry: &SuiteEntry) -> Result<Vec<String>> {
    let export = calibrate_entry(suite, entry)?;
    let path = suite.reference_path(&entry.id);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read reference {}", path.display()))?;
    let reference: ReferenceResult =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;

    let mut failures = Vec::new();
    if reference.suite_version != suite.manifest.suite_version {
        failures.push(format!(
            "reference is for suite {}, the suite is {}",
            reference.suite_version, suite.manifest.suite_version
        ));
    }
    let found = drifts(&reference.export, &export, entry.tolerance);
    for drift in found.iter().take(MAX_LISTED_DRIFTS) {
        failures.push(format!(
            "{}: {} → {}",
            drift.path, drift.expected, drift.actual
        ));
    }
    if found.len() > MAX_LISTED_DRIFTS {
        failures.push(format!(
            "… and {} more drifted values",
            found.len() - MAX_LISTED_DRIFTS
        ));
    }
    Ok(failures)
}

/// Calibrate an entry and store the export as its reference result.
///
/// # Errors
///
/// Fails when the calibration fails or the file cannot be written.
pub fn bless_entry(suite: &Suite, entry: &SuiteEntry) -> Result<()> {
    let reference = ReferenceResult {
        suite_version: suite.manifest.suite_version.clone(),
        producer: format!("calibration-rs {}", env!("CARGO_PKG_VERSION")),
        export: calibrate_entry(suite, entry)?,
    };
    let path = suite.reference_path(&entry.id);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = serde_json::to_string_pretty(&reference)?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))
}
