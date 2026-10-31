//! E4: how the weighting moves the transforms downstream of the cameras —
//! a rig's camera-to-rig transforms and the hand-eye transform — as the
//! split-half difference of the dataset's own route.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{Iso3, RigDataset};
use vision_calibration::optim::{HandeyeMode, RobotPoseMeta};
use vision_calibration::rig_extrinsics::{RigExtrinsicsConfig, RigExtrinsicsProblem};
use vision_calibration::rig_handeye::{RigHandeyeConfig, RigHandeyeProblem};
use vision_calibration::session::CalibrationSession;
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeConfig, SingleCamHandeyeInput, SingleCamHandeyeProblem,
};
use vision_calibration_core::{NoMeta, Uncertainty};
use vision_calibration_dataset::DatasetSpec;

use super::Options;
use super::arms::{Arm, Declined, detect_arm, ensure_same_corners};
use super::par_map;
use super::planar::splits;
use super::stats::median;
use crate::detect::{
    detect_rig_extrinsics, detect_rig_handeye, detect_single_cam_handeye, detection_cache,
};
use crate::registry::{BenchEntry, ProblemKind};
use crate::run::{rig_extrinsics_config, rig_handeye_config, single_cam_handeye_config};

/// A downstream route's input and configuration.
#[derive(Debug, Clone)]
enum Route {
    SingleCamHandeye(SingleCamHandeyeConfig, SingleCamHandeyeInput),
    RigExtrinsics(RigExtrinsicsConfig, RigDataset<NoMeta>),
    RigHandeye(Box<RigHandeyeConfig>, RigDataset<RobotPoseMeta>),
}

/// The transforms a downstream calibration produced.
#[derive(Debug, Clone)]
struct Solved {
    /// `cam_se3_rig` per camera; empty for a single camera.
    cam_se3_rig: Vec<Iso3>,
    /// The hand-eye transform (`gripper_se3_<rig>` or `<rig>_se3_base`).
    handeye: Option<Iso3>,
    /// Its stated σ: rotation (deg) and translation (mm), each the root sum
    /// of its three components' variances.
    handeye_sigma: Option<[f64; 2]>,
}

impl Route {
    fn views(&self) -> usize {
        match self {
            Route::SingleCamHandeye(_, input) => input.views.len(),
            Route::RigExtrinsics(_, input) => input.views.len(),
            Route::RigHandeye(_, input) => input.views.len(),
        }
    }

    /// Calibrate the views `keep` selects.
    fn solve(&self, keep: &dyn Fn(usize) -> bool) -> Result<Solved> {
        fn subset<T: Clone>(views: &[T], keep: &dyn Fn(usize) -> bool) -> Vec<T> {
            views
                .iter()
                .enumerate()
                .filter(|(i, _)| keep(*i))
                .map(|(_, v)| v.clone())
                .collect()
        }
        match self {
            Route::SingleCamHandeye(config, input) => {
                let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
                session.set_config(config.clone())?;
                session.set_input(SingleCamHandeyeInput::new(subset(&input.views, keep))?)?;
                vision_calibration::single_cam_handeye::run_calibration(&mut session)?;
                let export = session.export()?;
                let (handeye, label) = match export.handeye_mode {
                    HandeyeMode::EyeInHand => (export.gripper_se3_camera, "gripper_se3_camera"),
                    HandeyeMode::EyeToHand => (export.camera_se3_base, "camera_se3_base"),
                };
                Ok(Solved {
                    cam_se3_rig: Vec::new(),
                    handeye,
                    handeye_sigma: pose_sigma(export.parameter_uncertainty(), label),
                })
            }
            Route::RigExtrinsics(config, input) => {
                let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
                session.set_config(config.clone())?;
                session.set_input(RigDataset::new(
                    subset(&input.views, keep),
                    input.num_cameras,
                )?)?;
                vision_calibration::rig_extrinsics::run_calibration(&mut session)?;
                let export = session.export()?;
                Ok(Solved {
                    cam_se3_rig: export.cam_se3_rig,
                    handeye: None,
                    handeye_sigma: None,
                })
            }
            Route::RigHandeye(config, input) => {
                let mut session = CalibrationSession::<RigHandeyeProblem>::new();
                session.set_config((**config).clone())?;
                session.set_input(RigDataset::new(
                    subset(&input.views, keep),
                    input.num_cameras,
                )?)?;
                vision_calibration::rig_handeye::run_calibration(&mut session)?;
                let export = session.export()?;
                let (handeye, label) = match export.handeye_mode {
                    HandeyeMode::EyeInHand => (export.gripper_se3_rig, "gripper_se3_rig"),
                    HandeyeMode::EyeToHand => (export.rig_se3_base, "rig_se3_base"),
                };
                Ok(Solved {
                    cam_se3_rig: export.cam_se3_rig.clone(),
                    handeye,
                    handeye_sigma: pose_sigma(export.parameter_uncertainty(), label),
                })
            }
        }
    }
}

/// The stated σ of pose `name` in a covariance: rotation (deg) and
/// translation (mm).
fn pose_sigma(uncertainty: &Uncertainty, name: &str) -> Option<[f64; 2]> {
    let c = uncertainty.covariance()?;
    let rss = |parts: [&str; 3]| -> Option<f64> {
        parts
            .iter()
            .map(|p| c.std_dev(&format!("{name}.{p}")).map(|s| s * s))
            .sum::<Option<f64>>()
            .map(f64::sqrt)
    };
    Some([
        rss(["rx", "ry", "rz"])?.to_degrees(),
        rss(["tx", "ty", "tz"])? * 1000.0,
    ])
}

/// Rotation (deg) and translation (mm) between two transforms.
fn difference(a: &Iso3, b: &Iso3) -> [f64; 2] {
    [
        (a.rotation * b.rotation.inverse()).angle().to_degrees(),
        (a.translation.vector - b.translation.vector).norm() * 1000.0,
    ]
}

/// Detect `entry`'s route input for `arm`.
fn detect_route(
    entry: &BenchEntry,
    spec: &DatasetSpec,
    arm: Arm,
) -> Result<Result<Route, Declined>> {
    let cache = detection_cache();
    let board = entry.board.as_ref();
    let root = &entry.data_root;
    match entry.problem {
        ProblemKind::SingleCamHandeye => detect_arm(
            arm,
            |s| detect_single_cam_handeye(s, root, board, &cache),
            spec,
        )
        .map(|r| r.map(|d| Route::SingleCamHandeye(single_cam_handeye_config(entry), d.input))),
        ProblemKind::RigExtrinsics => {
            detect_arm(arm, |s| detect_rig_extrinsics(s, root, board, &cache), spec)
                .map(|r| r.map(|d| Route::RigExtrinsics(rig_extrinsics_config(entry), d.input)))
        }
        ProblemKind::RigHandeye => {
            let config = rig_handeye_config(entry, spec)?;
            detect_arm(
                arm,
                |s| detect_rig_handeye(s, root, board, &cache).map(|(d, _)| d),
                spec,
            )
            .map(|r| r.map(|d| Route::RigHandeye(Box::new(config), d.input)))
        }
        other => anyhow::bail!("{}: no downstream route for {other:?}", entry.id),
    }
}

/// The corners of every view, camera by camera.
fn corners(route: &Route) -> Vec<Option<&vision_calibration_core::CorrespondenceView>> {
    match route {
        Route::SingleCamHandeye(_, input) => input.views.iter().map(|v| Some(&v.obs)).collect(),
        Route::RigExtrinsics(_, input) => input
            .views
            .iter()
            .flat_map(|v| v.obs.cameras.iter().map(Option::as_ref))
            .collect(),
        Route::RigHandeye(_, input) => input
            .views
            .iter()
            .flat_map(|v| v.obs.cameras.iter().map(Option::as_ref))
            .collect(),
    }
}

/// One arm's downstream figures: medians over splits.
#[derive(Debug, Clone, Serialize)]
pub struct DownstreamArm {
    /// Per non-reference camera: median split-half difference of
    /// `cam_se3_rig`, `[deg, mm]`.
    pub cam_se3_rig: BTreeMap<usize, [f64; 2]>,
    /// Median split-half difference of the hand-eye transform, `[deg, mm]`.
    pub handeye: Option<[f64; 2]>,
    /// The full calibration's stated hand-eye σ, `[deg, mm]`.
    pub handeye_sigma: Option<[f64; 2]>,
    /// Median over splits of the difference the halves' stated σ predict,
    /// `√(σ_A² + σ_B²)`, `[deg, mm]`.
    pub handeye_predicted: Option<[f64; 2]>,
    /// Splits where both halves calibrated.
    pub splits_used: usize,
    /// Calibrations that failed.
    pub failures: usize,
}

/// E4 of one dataset.
#[derive(Debug, Clone, Serialize)]
pub struct Downstream {
    /// Registry id.
    pub dataset: String,
    /// Rig views of the route.
    pub views: usize,
    /// The covariance arms that are absent, and why.
    pub declined: BTreeMap<Arm, Declined>,
    /// Per arm (`uniform_noise` is not run downstream).
    pub arms: BTreeMap<Arm, DownstreamArm>,
}

/// E4 for `entry`.
pub fn measure(entry: &BenchEntry, spec: &DatasetSpec, options: &Options) -> Result<Downstream> {
    let mut routes: BTreeMap<Arm, Route> = BTreeMap::new();
    let mut declined = BTreeMap::new();
    for arm in Arm::DETECTED {
        super::progress(&entry.id, format!("E4 detection, arm {arm}"));
        match detect_route(entry, spec, arm)
            .with_context(|| format!("{}: E4 detection", entry.id))?
        {
            Ok(route) => {
                routes.insert(arm, route);
            }
            Err(fault) => {
                declined.insert(arm, fault);
            }
        }
    }
    let reference = &routes[&Arm::Uniform];
    for (arm, route) in &routes {
        ensure_same_corners(
            corners(reference),
            corners(route),
            &format!("{} E4 {arm}", entry.id),
        )?;
    }
    let n = reference.views();
    let split_sets = splits(n, options.random_splits, &format!("{}/e4", entry.id));
    let jobs: Vec<(Arm, Option<(usize, bool)>)> = routes
        .keys()
        .flat_map(|&arm| {
            (0..split_sets.len())
                .flat_map(|s| [Some((s, true)), Some((s, false))])
                .chain([None])
                .map(move |half| (arm, half))
        })
        .collect();
    super::progress(&entry.id, format!("E4: {} route calibrations", jobs.len()));
    let solved = par_map(&jobs, options.threads, |&(arm, half)| {
        let keep = |i: usize| half.is_none_or(|(s, a)| split_sets[s][i] == a);
        routes[&arm].solve(&keep).ok()
    });
    let per_arm = 2 * split_sets.len() + 1;
    let arms = routes
        .keys()
        .enumerate()
        .map(|(position, &arm)| {
            let block = &solved[position * per_arm..(position + 1) * per_arm];
            let failures = block.iter().filter(|s| s.is_none()).count();
            let pairs: Vec<(&Solved, &Solved)> = block[..per_arm - 1]
                .chunks(2)
                .filter_map(|p| Some((p[0].as_ref()?, p[1].as_ref()?)))
                .collect();
            let cameras = pairs.first().map_or(0, |(a, _)| a.cam_se3_rig.len());
            let med = |values: Vec<[f64; 2]>| -> Option<[f64; 2]> {
                (!values.is_empty()).then(|| {
                    [
                        median(values.iter().map(|v| v[0])),
                        median(values.iter().map(|v| v[1])),
                    ]
                })
            };
            let cam_se3_rig = (0..cameras)
                .filter_map(|c| {
                    let d: Vec<[f64; 2]> = pairs
                        .iter()
                        .filter_map(|(a, b)| {
                            Some(difference(a.cam_se3_rig.get(c)?, b.cam_se3_rig.get(c)?))
                        })
                        .collect();
                    // The reference camera is the identity in both halves.
                    let moved = d.iter().any(|v| v[0] > 1e-9 || v[1] > 1e-9);
                    moved.then(|| med(d).map(|m| (c, m))).flatten()
                })
                .collect();
            let handeye = med(pairs
                .iter()
                .filter_map(|(a, b)| Some(difference(a.handeye.as_ref()?, b.handeye.as_ref()?)))
                .collect());
            let handeye_predicted = med(pairs
                .iter()
                .filter_map(|(a, b)| {
                    let (sa, sb) = (a.handeye_sigma?, b.handeye_sigma?);
                    Some([sa[0].hypot(sb[0]), sa[1].hypot(sb[1])])
                })
                .collect());
            let summary = DownstreamArm {
                cam_se3_rig,
                handeye,
                handeye_sigma: block[per_arm - 1].as_ref().and_then(|s| s.handeye_sigma),
                handeye_predicted,
                splits_used: pairs.len(),
                failures,
            };
            (arm, summary)
        })
        .collect();
    Ok(Downstream {
        dataset: entry.id.clone(),
        views: n,
        declined,
        arms,
    })
}
