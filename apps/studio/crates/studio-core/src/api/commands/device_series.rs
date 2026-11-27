//! The values of a device's parameters over its history, for a trend chart
//! (`device_series`): each run's camera parameters, mounts, hand-eye and
//! laser planes with their σ, read from its export as
//! `camera_parameter_values` reads them, and the spread of each across the
//! runs from the library's `DeviceSeed::aggregate` of their seeds.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::device_seed::{DeviceSeed, SeedSpread};
use vision_calibration::project::{DeviceHistory, DeviceIdentity, DeviceRun, ScanFailure};

use super::device::{DeviceSelector, histories};
use crate::api::{Ctx, ErrorKind, StudioError};
use crate::runs::{RunRef, slash};
use crate::scope::UserPath;
use crate::seeds::seed_of_run;
use crate::uncertainty::{CameraParameters, ParameterGroup, ParameterSigma, camera_parameters_of};

/// The values of a device's parameters (or of each device of a product)
/// over its history, for a trend chart: every camera parameter, each
/// camera's mount in a rig, the hand-eye and each laser plane, as each run
/// estimated them, with their σ. A parameter only some runs have is a gap
/// in the others (`null`), never dropped. `starred_only` keeps the starred
/// runs. Each series of a camera parameter names its spread across the
/// runs, the library's robust summary (median, 1.4826·MAD, min, max) of
/// the runs' seeds.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeviceSeriesArgs {
    pub device: DeviceSelector,
    #[serde(default)]
    pub starred_only: bool,
}

/// The series of the devices asked for.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DeviceSeriesSet {
    /// One per device, by product, then id.
    pub devices: Vec<DeviceSeries>,
    /// What the library scan could not read, by path.
    pub failures: Vec<ScanFailure>,
}

/// One device's parameters over its runs.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DeviceSeries {
    pub device: DeviceIdentity,
    /// The x axis: the runs, in the history's order.
    pub runs: Vec<SeriesRun>,
    /// Each parameter, in the order the runs first name them (each run's
    /// cameras in camera order, the solver's order within a camera, then
    /// mounts, the hand-eye and laser planes).
    pub parameters: Vec<ParameterSeries>,
    /// The spread of every value across the runs' seeds.
    pub statistics: SeriesStatistics,
}

/// One run on the x axis.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SeriesRun {
    /// Its position in the device's history (`device_history`).
    pub order: usize,
    /// The project folder.
    pub project: String,
    pub id: String,
    /// When it was written, seconds since the Unix epoch; absent for a run
    /// of the app's earlier format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<u64>,
    /// Why it has no values (its export cannot be read, its cameras are not
    /// named): every series has a gap there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<StudioError>,
}

/// What a parameter belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SeriesGroup {
    Intrinsics,
    Projection,
    Distortion,
    Sensor,
    /// A camera's pose in a rig of several (`cam_se3_rig`).
    Mount,
    /// The rig's (or the camera's) pose on the robot.
    Handeye,
    /// A camera's laser plane.
    LaserPlane,
}

/// What a series' σ is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SigmaBasis {
    /// The value itself: a band of `value ± σ` is its uncertainty.
    Value,
    /// The pose's right perturbation `pose·Exp(δ)` along this component
    /// (as `camera_parameter_values` gives it), not the value's own σ.
    Perturbation,
}

/// One parameter over the runs.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ParameterSeries {
    /// `<camera>.<name>` (`cam0.fx`), `<camera>.cam_se3_rig.<rx…tz>`,
    /// `<pose>.<rx…tz>` for the hand-eye (`gripper_se3_rig.tx`),
    /// `<camera>.laser_plane.<nx, ny, nz, distance>`; cameras by their id.
    pub key: String,
    /// The camera's id; absent for the hand-eye.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    pub group: SeriesGroup,
    /// The parameter or component (`fx`, `k1`, `tx`, `nx`).
    pub name: String,
    /// `px`, `rad`, `deg` (a rotation's axis-angle component), `m`, or
    /// empty (a coefficient, a normal's component).
    pub unit: String,
    pub sigma_of: SigmaBasis,
    /// One per run of `runs`, in its order; `null` where the run does not
    /// have the parameter.
    pub points: Vec<Option<SeriesPoint>>,
    /// The label of its spread in `statistics` (the same as `key`), when
    /// the library's spread is of these values: a camera parameter's. A
    /// mount's, hand-eye's or laser plane's spread is listed under the
    /// library's own labels, over its own parametrization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread: Option<String>,
}

/// A run's value of a parameter.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SeriesPoint {
    pub value: f64,
    pub sigma: ParameterSigma,
    /// The solve ended with it on a bound.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub at_bound: bool,
}

/// The spread of the values across the runs: the library's aggregate of
/// the seeds of the runs that make one.
#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum SeriesStatistics {
    Available {
        /// The runs aggregated, as positions in `runs`.
        runs: Vec<usize>,
        /// Each value's values, median, 1.4826·MAD, min and max, in the
        /// library's order and with its labels.
        spread: Vec<SeedSpread>,
        /// What only some of the runs have, left out of the spread.
        omitted: Vec<String>,
        /// The runs that make no seed, and why.
        left_out: Vec<LeftOutRun>,
    },
    /// No spread: the library's reason (no run makes a seed; the runs'
    /// cameras, models or image sizes differ).
    Unavailable {
        reason: StudioError,
        left_out: Vec<LeftOutRun>,
    },
}

/// A run left out of the statistics.
#[derive(Debug, Serialize, JsonSchema)]
pub struct LeftOutRun {
    /// Its position in `runs`.
    pub run: usize,
    pub reason: StudioError,
}

pub fn device_series(ctx: &Ctx, args: DeviceSeriesArgs) -> Result<DeviceSeriesSet, StudioError> {
    let (histories, scan) = histories(ctx, &args.device);
    Ok(DeviceSeriesSet {
        devices: histories
            .iter()
            .map(|history| series_of(ctx, history, args.starred_only))
            .collect(),
        failures: scan.failures,
    })
}

fn series_of(ctx: &Ctx, history: &DeviceHistory, starred_only: bool) -> DeviceSeries {
    let entries: Vec<(usize, &DeviceRun)> = history
        .runs
        .iter()
        .enumerate()
        .filter(|(_, entry)| !starred_only || entry.run.notes.starred)
        .collect();
    let mut table = Table::new(entries.len());
    let mut runs = Vec::with_capacity(entries.len());
    for (position, (order, entry)) in entries.iter().enumerate() {
        let error = match values_of(ctx, entry) {
            Ok((ids, values)) => {
                table.add(position, &ids, &values);
                None
            }
            Err(error) => Some(error),
        };
        runs.push(SeriesRun {
            order: *order,
            project: slash(&entry.project),
            id: entry.run.id.clone(),
            timestamp: entry.run.record().map(|r| r.timestamp),
            error,
        });
    }
    let statistics = statistics_of(ctx, &entries);
    if let SeriesStatistics::Available { spread, .. } = &statistics {
        for series in &mut table.series {
            let camera = !matches!(
                series.group,
                SeriesGroup::Mount | SeriesGroup::Handeye | SeriesGroup::LaserPlane
            );
            if camera && spread.iter().any(|s| s.label == series.key) {
                series.spread = Some(series.key.clone());
            }
        }
    }
    DeviceSeries {
        device: history.device.clone(),
        runs,
        parameters: table.series,
        statistics,
    }
}

/// The run's camera ids (from its summary) and its parameters with their
/// σ (from its export).
fn values_of(ctx: &Ctx, entry: &DeviceRun) -> Result<(Vec<String>, CameraParameters), StudioError> {
    let ids: Vec<String> = entry
        .run
        .summary()
        .map(|s| s.cameras.iter().map(|c| c.id.clone()).collect())
        .ok_or_else(|| StudioError::invalid("the run's files do not name its cameras"))?;
    let run = RunRef {
        dir: UserPath::new(slash(&entry.project)),
        id: entry.run.id.clone(),
    };
    let values = camera_parameters_of(ctx.runs.get(ctx, &run)?.cameras()?);
    if values.cameras.len() != ids.len() {
        return Err(StudioError::invalid(format!(
            "the run's record names {} cameras and its export has {}",
            ids.len(),
            values.cameras.len()
        )));
    }
    Ok((ids, values))
}

/// The library's aggregate of the seeds of `entries`' runs.
fn statistics_of(ctx: &Ctx, entries: &[(usize, &DeviceRun)]) -> SeriesStatistics {
    let mut seeds = Vec::new();
    let mut used = Vec::new();
    let mut left_out = Vec::new();
    for (position, (_, entry)) in entries.iter().enumerate() {
        match seed_of_run(&ctx.scope, &entry.run.dir) {
            Ok(seed) => {
                seeds.push(seed);
                used.push(position);
            }
            Err(reason) => left_out.push(LeftOutRun {
                run: position,
                reason,
            }),
        }
    }
    match DeviceSeed::aggregate(&seeds) {
        Ok(aggregate) => SeriesStatistics::Available {
            runs: used,
            spread: aggregate.spread,
            omitted: aggregate.omitted,
            left_out,
        },
        Err(e) => SeriesStatistics::Unavailable {
            reason: StudioError::new(ErrorKind::Seed, e.to_string()),
            left_out,
        },
    }
}

/// The series being built: one per key, in first-seen order.
struct Table {
    runs: usize,
    index: BTreeMap<String, usize>,
    series: Vec<ParameterSeries>,
}

/// A series' identity, without its points.
struct Key<'a> {
    camera: Option<&'a str>,
    group: SeriesGroup,
    prefix: String,
    name: &'a str,
    unit: &'a str,
    sigma_of: SigmaBasis,
}

const POSE: [(&str, &str); 6] = [
    ("rx", "deg"),
    ("ry", "deg"),
    ("rz", "deg"),
    ("tx", "m"),
    ("ty", "m"),
    ("tz", "m"),
];
const PLANE: [(&str, &str); 4] = [("nx", ""), ("ny", ""), ("nz", ""), ("distance", "m")];

impl Table {
    fn new(runs: usize) -> Self {
        Self {
            runs,
            index: BTreeMap::new(),
            series: Vec::new(),
        }
    }

    fn put(&mut self, position: usize, key: Key<'_>, point: SeriesPoint) {
        let name = if key.prefix.is_empty() {
            key.name.to_owned()
        } else {
            format!("{}.{}", key.prefix, key.name)
        };
        let at = *self.index.entry(name.clone()).or_insert_with(|| {
            self.series.push(ParameterSeries {
                key: name,
                camera: key.camera.map(str::to_owned),
                group: key.group,
                name: key.name.to_owned(),
                unit: key.unit.to_owned(),
                sigma_of: key.sigma_of,
                points: vec![None; self.runs],
                spread: None,
            });
            self.series.len() - 1
        });
        self.series[at].points[position] = Some(point);
    }

    fn add(&mut self, position: usize, ids: &[String], values: &CameraParameters) {
        for set in &values.cameras {
            let id = ids[set.camera].as_str();
            for p in &set.parameters {
                let key = Key {
                    camera: Some(id),
                    group: match p.group {
                        ParameterGroup::Intrinsics => SeriesGroup::Intrinsics,
                        ParameterGroup::Projection => SeriesGroup::Projection,
                        ParameterGroup::Distortion => SeriesGroup::Distortion,
                        ParameterGroup::Sensor => SeriesGroup::Sensor,
                    },
                    prefix: id.to_owned(),
                    name: &p.name,
                    unit: &p.unit,
                    sigma_of: SigmaBasis::Value,
                };
                let point = SeriesPoint {
                    value: p.value,
                    sigma: p.sigma.clone(),
                    at_bound: p.at_bound,
                };
                self.put(position, key, point);
            }
        }
        for pose in &values.poses {
            let (camera, group, prefix) = match pose.camera {
                Some(i) => (
                    Some(ids[i].as_str()),
                    SeriesGroup::Mount,
                    format!("{}.cam_se3_rig", ids[i]),
                ),
                None => (None, SeriesGroup::Handeye, pose.label.clone()),
            };
            let components = pose.rotation_deg.iter().chain(&pose.translation_m);
            for (k, (value, (name, unit))) in components.zip(POSE).enumerate() {
                let key = Key {
                    camera,
                    group,
                    prefix: prefix.clone(),
                    name,
                    unit,
                    sigma_of: SigmaBasis::Perturbation,
                };
                let point = SeriesPoint {
                    value: *value,
                    sigma: pose.sigma[k].clone(),
                    at_bound: false,
                };
                self.put(position, key, point);
            }
        }
        for plane in &values.planes {
            let id = ids[plane.camera].as_str();
            let components = plane
                .normal
                .iter()
                .chain(std::iter::once(&plane.distance_m));
            for (k, (value, (name, unit))) in components.zip(PLANE).enumerate() {
                let key = Key {
                    camera: Some(id),
                    group: SeriesGroup::LaserPlane,
                    prefix: format!("{id}.laser_plane"),
                    name,
                    unit,
                    sigma_of: SigmaBasis::Value,
                };
                let point = SeriesPoint {
                    value: *value,
                    sigma: plane.sigma[k].clone(),
                    at_bound: false,
                };
                self.put(position, key, point);
            }
        }
    }
}
