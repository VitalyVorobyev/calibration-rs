//! A device's calibration history across the projects under the roots:
//! the library's `LibraryScan::devices` and `DeviceHistory`, one row per
//! run with what the Device screen lists (`device_history`). The series of
//! its parameters are `device_series`'s; seeds are `seed`'s.

use std::collections::BTreeMap;
use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::analysis::Verdict;
use vision_calibration::project::{
    DeviceHistory, DeviceIdentity, DeviceRun, LibraryScan, LibraryScanOptions, RunSource, RunState,
    ScanFailure, SeedSource, scan_library,
};

use crate::api::{Ctx, StudioError};
use crate::runs::slash;

/// One device, by what a project's `[device]` states: its product (none for
/// a device of no stated product) and its id, both compared as written.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceKey {
    #[serde(default)]
    pub product: Option<String>,
    pub id: String,
}

impl DeviceKey {
    /// Whether `device` is this one.
    pub(super) fn names(&self, device: &DeviceIdentity) -> bool {
        device.product == self.product && device.id == self.id
    }
}

/// A device, or every device of a product (a device type).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeviceSelector {
    /// The device of `product` (none for no stated product) and `id`.
    Device {
        #[serde(default)]
        product: Option<String>,
        id: String,
    },
    /// Every device whose product is `product`.
    Type { product: String },
}

impl DeviceSelector {
    fn selects(&self, device: &DeviceIdentity) -> bool {
        match self {
            Self::Device { product, id } => device.product == *product && device.id == *id,
            Self::Type { product } => device.product.as_deref() == Some(product.as_str()),
        }
    }
}

/// Every device's history, in the library's order (by product, then id),
/// from one scan of the roots; and the scan.
pub(super) fn all_histories(ctx: &Ctx) -> (Vec<DeviceHistory>, LibraryScan) {
    let scan = scan_library(&ctx.scope.roots(), &LibraryScanOptions::default());
    (scan.devices(), scan)
}

/// The histories of the devices `selector` names, in the library's order;
/// and the scan.
pub(super) fn histories(ctx: &Ctx, selector: &DeviceSelector) -> (Vec<DeviceHistory>, LibraryScan) {
    let (all, scan) = all_histories(ctx);
    let selected = all
        .into_iter()
        .filter(|history| selector.selects(&history.device))
        .collect();
    (selected, scan)
}

/// The calibration history of a device (or of every device of a product):
/// every run of every project under the roots that names it, in the
/// library's order (runs without a timestamp first, then by timestamp, then
/// by project folder and listing order), with the run that stands for each
/// device (its latest starred run, else its latest pass run calibrated in a
/// project). A device no project names has no history (`devices` is
/// empty); a folder or project the scan cannot read is a failure, not an
/// error.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceHistoryArgs {
    pub device: DeviceSelector,
}

/// The histories of the devices asked for.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DeviceHistories {
    /// One per device, by product, then id.
    pub devices: Vec<DeviceHistoryView>,
    /// What the library scan could not read, by path.
    pub failures: Vec<ScanFailure>,
}

/// One device's runs.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DeviceHistoryView {
    /// The device, as the first of its projects (by folder) states it.
    pub device: DeviceIdentity,
    /// Its runs, in the history's order.
    pub runs: Vec<DeviceRunRow>,
}

/// One run of a device's history.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DeviceRunRow {
    /// The project folder.
    pub project: String,
    /// The project's name, or its folder's.
    pub project_name: String,
    /// The run's id (`run-003`).
    pub id: String,
    /// Where its export came from, or why its record cannot be read.
    pub origin: RunOrigin,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub starred: bool,
    /// Its quality verdict; absent when it was not assessed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    /// Each camera's reprojection RMS, in camera order; empty when the run
    /// has no summary.
    pub cameras: Vec<CameraRms>,
    /// When the run was written, seconds since the Unix epoch; absent for a
    /// run of the app's earlier format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
    /// The seed the run started from, for a seeded calibrated run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<RunSeedUse>,
    /// Whether it is the run that stands for its device (the library's
    /// representative run).
    pub representative: bool,
}

/// Where a run's export came from.
#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunOrigin {
    /// The project calibrated it.
    Calibrated,
    /// It was imported from an export calibrated elsewhere.
    Imported,
    /// A run of the app's earlier format, without a record.
    Unrecorded,
    /// Its record cannot be read by this version.
    Unreadable { message: String },
}

/// A camera's reprojection RMS in a run.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CameraRms {
    /// The camera's id in the manifest.
    pub id: String,
    /// Pixels; absent without residuals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_px: Option<f64>,
}

/// The seed a run started from, as its record keeps it (its values are in
/// `run_info`'s record).
#[derive(Debug, Serialize, JsonSchema)]
pub struct RunSeedUse {
    /// Where it came from, as the project (or the run's options) named it.
    pub source: SeedSource,
    /// The seed's own description (`the calibrated run run-003`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// What the workflow took from it (`cam0.lens`, `cam1.cam_se3_rig`,
    /// `cam0.laser_plane`, `handeye`).
    pub applied: Vec<String>,
    /// What it holds that the workflow does not use.
    pub ignored: Vec<String>,
}

pub fn device_history(ctx: &Ctx, args: DeviceHistoryArgs) -> Result<DeviceHistories, StudioError> {
    let (histories, scan) = histories(ctx, &args.device);
    let names: BTreeMap<PathBuf, String> = scan
        .projects
        .iter()
        .map(|p| (p.dir.clone(), p.name.clone()))
        .collect();
    let devices = histories
        .iter()
        .map(|history| {
            let representative = history.representative_run();
            DeviceHistoryView {
                device: history.device.clone(),
                runs: history
                    .runs
                    .iter()
                    .map(|entry| {
                        let chosen = representative.is_some_and(|r| std::ptr::eq(r, entry));
                        row(entry, &names, chosen)
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(DeviceHistories {
        devices,
        failures: scan.failures,
    })
}

fn row(entry: &DeviceRun, names: &BTreeMap<PathBuf, String>, representative: bool) -> DeviceRunRow {
    let run = &entry.run;
    let (origin, timestamp, seed) = match &run.state {
        RunState::Recorded(record) => match &record.source {
            RunSource::Calibrated(calibrated) => (
                RunOrigin::Calibrated,
                Some(record.timestamp),
                calibrated.inputs.seed.as_ref().map(|seed| RunSeedUse {
                    source: seed.source.clone(),
                    description: seed.value.description.clone(),
                    applied: seed.applied.clone(),
                    ignored: seed.ignored.clone(),
                }),
            ),
            _ => (RunOrigin::Imported, Some(record.timestamp), None),
        },
        RunState::Unrecorded(_) => (RunOrigin::Unrecorded, None, None),
        RunState::Unreadable { message } => (
            RunOrigin::Unreadable {
                message: message.clone(),
            },
            None,
            None,
        ),
        // A state of a later library version.
        _ => (
            RunOrigin::Unreadable {
                message: "a run this app does not know".into(),
            },
            None,
            None,
        ),
    };
    let summary = run.summary();
    DeviceRunRow {
        project: slash(&entry.project),
        project_name: names.get(&entry.project).cloned().unwrap_or_default(),
        id: run.id.clone(),
        origin,
        label: run.notes.label.clone(),
        starred: run.notes.starred,
        verdict: summary.and_then(|s| s.verdict),
        cameras: summary
            .map(|s| {
                s.cameras
                    .iter()
                    .map(|c| CameraRms {
                        id: c.id.clone(),
                        rms_px: c.rms_px,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        timestamp,
        seed,
        representative,
    }
}
