//! The pre-registered real-data study of detector feature covariance
//! (`docs/notes/feature-covariance-study.md`): `calib-bench feature-cov`.
//!
//! Every camera of the study's datasets is detected once per arm through the
//! dataset runner (the arms differ only in the detector's covariance
//! request, and their corners are checked identical) and calibrated on its
//! route's configuration:
//!
//! - E1–E3 ([`planar`]): held-out folds, split halves and the full set, per
//!   camera;
//! - E4 ([`downstream`]): the dataset's own rig or hand-eye route on split
//!   halves;
//! - E5 ([`timing`]): detection time, in a single-threaded child process.
//!
//! [`rule`] pools the cameras and applies the decision rule; [`report`]
//! writes `target/feature-cov/`. A private dataset contributes aggregates
//! only: no per-view or per-split values reach its files.

pub mod arms;
pub mod downstream;
pub mod geometry;
pub mod planar;
pub mod report;
pub mod rule;
pub mod stats;
pub mod timing;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use vision_calibration::planar_intrinsics::{IntrinsicsManualInit, PlanarIntrinsicsConfig};
use vision_calibration_dataset::DatasetSpec;

use crate::detect::{camera_spec, detect_planar, detection_cache};
use crate::manifest::dataset_spec;
use crate::registry::{BenchEntry, ProblemKind, Visibility};
use crate::run::{
    planar_intrinsics_config, rig_extrinsics_config, rig_handeye_config, single_cam_handeye_config,
};
use arms::{Arm, ensure_same_corners, with_noise};
use planar::{CameraRoute, Unit, UnitRaw, UnitSummary};

/// How the study runs.
#[derive(Debug, Clone)]
pub struct Options {
    /// Output directory.
    pub out: PathBuf,
    /// Seeded random half-splits besides the even/odd split (20 in the
    /// protocol).
    pub random_splits: usize,
    /// Bootstrap resamples (10 000 in the protocol).
    pub resamples: usize,
    /// Calibrations run in parallel.
    pub threads: usize,
    /// Time at most this many images per camera (all in the protocol).
    pub timing_images: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            out: default_out_dir(),
            random_splits: 20,
            resamples: 10_000,
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            timing_images: None,
        }
    }
}

impl Options {
    /// Whether these are the protocol's settings.
    pub fn is_protocol(&self) -> bool {
        self.random_splits == 20 && self.resamples == 10_000 && self.timing_images.is_none()
    }
}

/// `target/feature-cov` in the workspace.
pub fn default_out_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/feature-cov")
}

/// One dataset of the study.
#[derive(Debug, Clone, Copy)]
pub struct PlanEntry {
    /// Registry id.
    pub id: &'static str,
    /// The cameras E1–E3 run on, by manifest index.
    pub cameras: &'static [usize],
    /// Whether E4 runs the dataset's own route.
    pub downstream: bool,
}

/// The study's datasets (the note's §Datasets).
pub const PLAN: [PlanEntry; 7] = [
    PlanEntry {
        id: "stereo_left",
        cameras: &[0],
        downstream: false,
    },
    PlanEntry {
        id: "stereo_right",
        cameras: &[0],
        downstream: false,
    },
    PlanEntry {
        id: "kuka_1",
        cameras: &[0],
        downstream: true,
    },
    PlanEntry {
        id: "ds8",
        cameras: &[0, 1, 2],
        downstream: true,
    },
    PlanEntry {
        id: "stereo_rig",
        cameras: &[],
        downstream: true,
    },
    PlanEntry {
        id: "stereo_charuco",
        cameras: &[0, 1],
        downstream: true,
    },
    PlanEntry {
        id: "scheimpflug_rig",
        cameras: &[0, 1, 2, 3, 4, 5],
        downstream: false,
    },
];

/// The plan's datasets among `entries` (data roots resolved), restricted to
/// `only` when it is not empty.
///
/// # Errors
///
/// An `only` id that is not in the plan.
pub fn select(
    entries: &[BenchEntry],
    only: &[String],
) -> Result<Vec<(PlanEntry, Option<BenchEntry>)>> {
    for id in only {
        anyhow::ensure!(
            PLAN.iter().any(|p| p.id == id),
            "--only id `{id}` is not a study dataset (known: {})",
            PLAN.iter().map(|p| p.id).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(PLAN
        .iter()
        .filter(|p| only.is_empty() || only.iter().any(|id| id == p.id))
        .map(|p| (*p, entries.iter().find(|e| e.id == p.id).cloned()))
        .collect())
}

/// The calibrated cameras and downstream results of a study run.
#[derive(Debug, Default)]
pub struct Study {
    /// Per camera.
    pub units: Vec<(UnitSummary, UnitRaw)>,
    /// Per dataset with a downstream route.
    pub downstream: Vec<downstream::Downstream>,
    /// Plan datasets that are not registered or not on disk.
    pub unavailable: Vec<String>,
}

/// Run E1–E4 on the selected datasets, writing each camera's file as it
/// finishes.
pub fn study(selected: &[(PlanEntry, Option<BenchEntry>)], options: &Options) -> Result<Study> {
    let mut out = Study::default();
    for (plan, entry) in selected {
        let Some(entry) = entry.as_ref().filter(|e| e.data_root.is_dir()) else {
            progress(plan.id, "UNAVAILABLE (not registered or not on disk)");
            out.unavailable.push(plan.id.to_string());
            continue;
        };
        let spec = dataset_spec(entry)?;
        for &camera in plan.cameras {
            let unit = detect_unit(entry, &spec, camera)?;
            progress(
                &unit.label,
                format!(
                    "E1–E3: {} views, arms {}",
                    unit.arms[&Arm::Uniform].views.len(),
                    unit.arms
                        .keys()
                        .map(|a| a.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
            let raw = planar::measure(&unit, options)
                .with_context(|| format!("{}: E1–E3", unit.label))?;
            let summary = planar::summarize(&unit, &raw, options.resamples);
            report::write_unit(&options.out, &summary, &raw)?;
            out.units.push((summary, raw));
        }
        if plan.downstream {
            let result = downstream::measure(entry, &spec, options)?;
            report::write_downstream(&options.out, &result)?;
            out.downstream.push(result);
        }
    }
    Ok(out)
}

/// The images E5 times: each camera's images with a detected board.
pub fn timing_inputs(
    selected: &[(PlanEntry, Option<BenchEntry>)],
) -> Result<Vec<timing::TimingInput>> {
    let mut inputs = Vec::new();
    for (plan, entry) in selected {
        let Some(entry) = entry.as_ref().filter(|e| e.data_root.is_dir()) else {
            continue;
        };
        let spec = dataset_spec(entry)?;
        for &camera in plan.cameras {
            let spec = camera_spec(&spec, camera);
            let detected = detect_planar(
                &spec,
                &entry.data_root,
                entry.board.as_ref(),
                &detection_cache(),
            )?;
            inputs.push(timing::TimingInput {
                label: format!("{}/{}", entry.id, spec.cameras[0].id),
                images: detected.image_paths.into_iter().next().unwrap_or_default(),
                spec,
            });
        }
    }
    Ok(inputs)
}

/// Pool, decide and write the summary and report.
pub fn finish(
    study: &Study,
    timing: Option<&timing::Timing>,
    options: &Options,
) -> Result<rule::Decision> {
    let units: Vec<(&UnitSummary, &UnitRaw)> = study.units.iter().map(|(s, r)| (s, r)).collect();
    let decision = rule::decide(&units, timing, options.resamples);
    report::write_summary(&options.out, study, timing, &decision, options)?;
    Ok(decision)
}

/// Detect camera `camera` of `entry` for every arm.
fn detect_unit(entry: &BenchEntry, spec: &DatasetSpec, camera: usize) -> Result<Unit> {
    anyhow::ensure!(
        camera < spec.cameras.len(),
        "{}: camera {camera} out of range",
        entry.id
    );
    let unit_spec = camera_spec(spec, camera);
    let label = format!("{}/{}", entry.id, unit_spec.cameras[0].id);
    let cache = detection_cache();
    let mut arms = BTreeMap::new();
    let mut declined = BTreeMap::new();
    let mut image_size = None;
    for arm in Arm::DETECTED {
        progress(&label, format!("detection, arm {arm}"));
        let detected = arms::detect_arm(
            arm,
            |s| detect_planar(s, &entry.data_root, entry.board.as_ref(), &cache),
            &unit_spec,
        )
        .with_context(|| format!("{label}: detection"))?;
        match detected {
            Ok(d) => {
                image_size.get_or_insert(d.image_sizes[0]);
                arms.insert(arm, d.input);
            }
            Err(fault) => {
                declined.insert(arm, fault);
            }
        }
    }
    let reference = &arms[&Arm::Uniform];
    for (arm, dataset) in &arms {
        ensure_same_corners(
            reference.views.iter().map(|v| Some(&v.obs)),
            dataset.views.iter().map(|v| Some(&v.obs)),
            &format!("{label} {arm}"),
        )?;
    }
    let noisy = with_noise(reference, &label)?;
    arms.insert(Arm::UniformNoise, noisy);
    Ok(Unit {
        label,
        dataset: entry.id.clone(),
        camera: unit_spec.cameras[0].id.clone(),
        private: entry.visibility == Visibility::Private,
        image_size: image_size.context("the uniform arm detected nothing")?,
        arms,
        declined,
        route: camera_route(entry, spec, camera)?,
    })
}

/// Camera `camera`'s calibration on `entry`'s route: the route
/// configuration's camera model, intrinsics initialization and solver, and
/// for a seeded rig its seed.
fn camera_route(entry: &BenchEntry, spec: &DatasetSpec, camera: usize) -> Result<CameraRoute> {
    let planar = |camera_config, init, solver| {
        let mut config = PlanarIntrinsicsConfig::default();
        config.camera = camera_config;
        config.init = init;
        config.solver = solver;
        config
    };
    Ok(match entry.problem {
        ProblemKind::PlanarIntrinsics => {
            anyhow::ensure!(
                !entry.is_scheimpflug_planar(),
                "{}: the seeded single-camera Scheimpflug route is not in the study",
                entry.id
            );
            CameraRoute {
                config: planar_intrinsics_config(entry),
                seed: None,
            }
        }
        ProblemKind::SingleCamHandeye => {
            let c = single_cam_handeye_config(entry);
            CameraRoute {
                config: planar(c.camera, c.intrinsics, c.solver),
                seed: None,
            }
        }
        ProblemKind::RigExtrinsics => {
            let c = rig_extrinsics_config(entry);
            CameraRoute {
                config: planar(c.camera_for(camera), c.intrinsics.clone(), c.solver.clone()),
                seed: None,
            }
        }
        ProblemKind::RigHandeye => {
            let c = rig_handeye_config(entry, spec)?;
            let seed = c.manual_init.as_ref().map(|m| {
                let mut seed = IntrinsicsManualInit::default();
                seed.intrinsics = m
                    .per_cam_intrinsics
                    .as_ref()
                    .and_then(|v| v.get(camera).copied());
                seed.distortion = m
                    .per_cam_distortion
                    .as_ref()
                    .and_then(|v| v.get(camera).copied());
                seed.sensor = m
                    .per_cam_sensors
                    .as_ref()
                    .and_then(|v| v.get(camera).copied());
                seed
            });
            CameraRoute {
                config: planar(c.camera_for(camera), c.intrinsics.clone(), c.solver.clone()),
                seed,
            }
        }
        other => anyhow::bail!("{}: {other:?} is not a study route", entry.id),
    })
}

/// `f` over `items` on `threads` threads, results in item order.
pub fn par_map<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, R)>> = Mutex::new(Vec::with_capacity(items.len()));
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, items.len().max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let r = f(item);
                    results
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push((i, r));
                }
            });
        }
    });
    let mut results = results
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    results.sort_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, r)| r).collect()
}

/// Progress on stderr, with the seconds since the first message.
pub fn progress(label: &str, message: impl std::fmt::Display) {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
    eprintln!(
        "[feature-cov {:>7.1}s {label}] {message}",
        elapsed.as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map_keeps_item_order() {
        let items: Vec<usize> = (0..50).collect();
        assert_eq!(
            par_map(&items, 4, |i| i * 2),
            (0..50).map(|i| i * 2).collect::<Vec<_>>()
        );
        assert!(par_map(&Vec::<usize>::new(), 4, |i| *i).is_empty());
    }

    #[test]
    fn the_plan_names_the_suite_image_entries() {
        let suite = crate::suite::manifest::Suite::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../regression/suite.toml"),
        )
        .unwrap();
        let entries = crate::suite::datasets::bench_entries(&suite, None, &[]).unwrap();
        for id in ["stereo_left", "stereo_right", "kuka_1", "ds8", "stereo_rig"] {
            assert!(
                entries.iter().any(|e| e.id == id),
                "{id} is not a suite image entry"
            );
            assert!(PLAN.iter().any(|p| p.id == id));
        }
        assert!(select(&entries, &["nope".to_string()]).is_err());
    }
}
