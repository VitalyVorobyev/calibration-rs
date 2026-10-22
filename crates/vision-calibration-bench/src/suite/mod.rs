//! The public regression suite: image-free observations, frozen once, with
//! reference results that every change is checked against.
//!
//! - [`manifest`] — `suite.toml` and the `SHA256SUMS` checksum file.
//! - [`observations`] — the [`ObservationSet`] file format.
//! - [`reference`](mod@reference) — stored results and the drift check.
//! - [`significance`] — parameter changes in units of the reference's σ.
//! - [`truth`] — a synthetic entry's true parameters, and a result's errors
//!   against them in units of its σ.
//! - [`synthetic`] — a benchmark scene → [`ObservationSet`] + [`Truth`].
//! - [`compare`] — other libraries' results against the references.
//! - `fetch` (feature `tier-b`) — an image entry's files, downloaded from
//!   their publisher into a local cache and checksum-verified.
//! - `freeze` (feature `tier-b`) — dataset manifest + images →
//!   [`ObservationSet`].
//!
//! [`check_entry`] verifies an entry's checksum, calibrates from its
//! observations and compares the export with the reference result, either
//! value for value or, with [`Gate::Sigma`], parameter by parameter in
//! units of the reference's standard deviations. A synthetic entry is also
//! checked against its truth: every parameter within [`TRUTH_SIGMA`] of its
//! true value, in units of the result's own σ. [`bless_entry`] writes the
//! reference; [`freeze_synthetic_entry`] writes a synthetic entry's data.

pub mod compare;
#[cfg(feature = "tier-b")]
pub mod fetch;
#[cfg(feature = "tier-b")]
pub mod freeze;
pub mod manifest;
pub mod observations;
pub mod reference;
pub mod significance;
pub mod synthetic;
#[cfg(test)]
mod test_exports;
pub mod truth;

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use vision_calibration_optim::SolverBackend;

pub use manifest::{FetchSource, Suite, SuiteEntry, Tolerance};
pub use observations::{OBSERVATION_SET_VERSION, ObservationSet, ObservedCamera, ProblemInput};
pub use reference::{Drift, ReferenceResult, drifts};
pub use significance::{ParameterChange, parameter_changes};
pub use truth::{Truth, truth_errors};

/// How many drifts a failed check lists before summarizing the rest.
const MAX_LISTED_DRIFTS: usize = 12;

/// A synthetic entry fails when a parameter is further than this many of
/// the result's standard deviations from its true value. Generous for
/// noise (the largest of 20 Gaussian errors exceeds 4σ about once in
/// 1500), so a failure means a bias or a σ that is far too small.
pub const TRUTH_SIGMA: f64 = 5.0;

/// The outcome of checking one entry.
#[derive(Debug, Clone)]
pub struct EntryReport {
    /// Entry id.
    pub id: String,
    /// Wall time of the check, in seconds.
    pub seconds: f64,
    /// Why the check failed; empty when it passed.
    pub failures: Vec<String>,
    /// The largest parameter change in units of the reference's σ, when
    /// the reference has a covariance.
    pub largest_change: Option<ParameterChange>,
    /// For a synthetic entry, the largest error against the truth in units
    /// of the result's σ.
    pub largest_truth_error: Option<ParameterChange>,
}

/// What a check compares against the reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gate {
    /// Every value of the export within the entry's tolerance.
    Exact,
    /// Every parameter the reference's covariance reports within this many
    /// of its standard deviations; nothing else is compared. For results
    /// that may legitimately differ in their last digits, such as those of
    /// another solver backend.
    Sigma(f64),
}

/// How an entry is checked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheckOptions {
    /// What is compared.
    pub gate: Gate,
    /// The solver backend for every stage; `None` keeps the entry's config.
    pub backend: Option<SolverBackend>,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            gate: Gate::Exact,
            backend: None,
        }
    }
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
pub fn calibrate_entry(
    suite: &Suite,
    entry: &SuiteEntry,
    backend: Option<SolverBackend>,
) -> Result<Value> {
    let set = load_observations(suite, &entry.id)?;
    let mut export = set.input.calibrate(entry.config.as_ref(), backend)?;
    if let Some(Value::Object(residuals)) = export.get_mut("per_feature_residuals") {
        residuals.remove("target");
        residuals.remove("laser");
        residuals.remove("laser_lines");
    }
    Ok(export)
}

/// Check one entry against its reference result. Every failure — checksum,
/// calibration, missing reference, drift — is reported, never raised.
pub fn check_entry(suite: &Suite, entry: &SuiteEntry, options: CheckOptions) -> EntryReport {
    let start = Instant::now();
    let outcome = check_failures(suite, entry, options).unwrap_or_else(|err| Outcome {
        failures: vec![format!("{err:#}")],
        ..Outcome::default()
    });
    EntryReport {
        id: entry.id.clone(),
        seconds: start.elapsed().as_secs_f64(),
        failures: outcome.failures,
        largest_change: outcome.largest_change,
        largest_truth_error: outcome.largest_truth_error,
    }
}

#[derive(Default)]
struct Outcome {
    failures: Vec<String>,
    largest_change: Option<ParameterChange>,
    largest_truth_error: Option<ParameterChange>,
}

fn largest(changes: &[ParameterChange]) -> Option<ParameterChange> {
    changes
        .iter()
        .max_by(|a, b| a.z().abs().total_cmp(&b.z().abs()))
        .cloned()
}

/// `changes` at or beyond `k` of their σ, as failure lines; a NaN change
/// fails too.
fn beyond(changes: &[ParameterChange], k: f64, what: &str) -> Vec<String> {
    changes
        .iter()
        .filter(|c| c.z().abs().partial_cmp(&k) != Some(std::cmp::Ordering::Less))
        .map(|c| format!("{}: {what} {:+.3}σ (σ = {:.3e})", c.label, c.z(), c.sigma))
        .collect()
}

fn check_failures(suite: &Suite, entry: &SuiteEntry, options: CheckOptions) -> Result<Outcome> {
    let export = calibrate_entry(suite, entry, options.backend)?;
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
    let changes = parameter_changes(&reference.export, &export)?;
    let mut listed: Vec<String> = match options.gate {
        Gate::Exact => drifts(&reference.export, &export, entry.tolerance)
            .into_iter()
            .map(|d| format!("{}: {} → {}", d.path, d.expected, d.actual))
            .collect(),
        Gate::Sigma(k) => beyond(&changes, k, "changed by"),
    };
    let truth_errors = match entry.synthetic {
        Some(_) => truth_errors(&export, &load_truth(suite, &entry.id)?)?,
        None => Vec::new(),
    };
    listed.extend(beyond(&truth_errors, TRUTH_SIGMA, "off the truth by"));
    failures.extend(listed.iter().take(MAX_LISTED_DRIFTS).cloned());
    if listed.len() > MAX_LISTED_DRIFTS {
        failures.push(format!(
            "… and {} more drifted values",
            listed.len() - MAX_LISTED_DRIFTS
        ));
    }
    Ok(Outcome {
        failures,
        largest_change: largest(&changes),
        largest_truth_error: largest(&truth_errors),
    })
}

/// Read, checksum-verify and parse a synthetic entry's truth.
///
/// # Errors
///
/// Fails on a missing or mismatched checksum or invalid JSON.
pub fn load_truth(suite: &Suite, id: &str) -> Result<Truth> {
    let bytes = suite.read_verified_truth(id)?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", Suite::truth_rel(id)))
}

/// Write a synthetic entry's observations and truth from its benchmark
/// scene, and refresh `SHA256SUMS`.
///
/// # Errors
///
/// Fails when the entry is not synthetic, its scene cannot be built, or a
/// file cannot be written.
pub fn freeze_synthetic_entry(suite: &Suite, entry: &SuiteEntry) -> Result<ObservationSet> {
    let scene = entry
        .synthetic
        .as_deref()
        .with_context(|| format!("entry {:?} is not synthetic", entry.id))?;
    let (set, truth) = synthetic::synthesize(scene)?;
    write_compact(&suite.observations_path(&entry.id), &set)?;
    write_compact(&suite.truth_path(&entry.id), &truth)?;
    suite.write_checksums()?;
    Ok(set)
}

/// Write `value` as compact JSON with a trailing newline.
pub(crate) fn write_compact(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = serde_json::to_string(value)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
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
        export: calibrate_entry(suite, entry, None)?,
    };
    let path = suite.reference_path(&entry.id);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = serde_json::to_string_pretty(&reference)?;
    text.push('\n');
    std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))
}
