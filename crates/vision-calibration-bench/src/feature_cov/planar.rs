//! E1–E3 on one camera: held-out accuracy, split-half efficiency and the
//! honesty of the stated projection uncertainty.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;
use vision_calibration::analysis::planar_intrinsics_report;
use vision_calibration::common::CalibratedCameras;
use vision_calibration::planar_intrinsics::{
    IntrinsicsManualInit, PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsProblem,
    step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration_core::{NoMeta, NoiseLevel, PlanarDataset, Pt2, View};

use super::Options;
use super::arms::{Arm, Declined};
use super::geometry::{
    SplitComparison, ZSums, compare, convex_hull, grid, homography_squared_errors, inside,
    undistort,
};
use super::par_map;
use super::stats::{
    Class, Interval, PairedSums, Rng, median, median_of, pooled, rms_difference, rms_of, seed_of,
};

/// Held-out folds: view `i` goes to fold `i mod FOLDS`.
pub const FOLDS: usize = 5;

/// How one camera is calibrated: its route's configuration and, for a
/// seeded route, its seed.
#[derive(Debug, Clone)]
pub struct CameraRoute {
    /// Planar-intrinsics configuration.
    pub config: PlanarIntrinsicsConfig,
    /// The route's seed (a Scheimpflug camera); `None` initializes from the
    /// data.
    pub seed: Option<IntrinsicsManualInit>,
}

impl CameraRoute {
    /// Calibrate `views` on this route.
    pub fn calibrate(&self, views: Vec<View<NoMeta>>) -> Result<PlanarIntrinsicsExport> {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_config(self.config.clone())?;
        session.set_input(PlanarDataset::new(views)?)?;
        step_init_with_seed(&mut session, self.seed.clone().unwrap_or_default(), None)?;
        step_optimize(&mut session, None)?;
        Ok(session.export()?)
    }
}

/// One camera of one dataset, detected for every arm.
#[derive(Debug, Clone)]
pub struct Unit {
    /// `<dataset>/<camera>`.
    pub label: String,
    /// Registry id of the dataset.
    pub dataset: String,
    /// Camera id.
    pub camera: String,
    /// Whether the dataset is private (aggregates only).
    pub private: bool,
    /// `[width, height]` of the camera's images.
    pub image_size: [u32; 2],
    /// Each present arm's views (identical corners).
    pub arms: BTreeMap<Arm, PlanarDataset>,
    /// The covariance arms that are absent, and why.
    pub declined: BTreeMap<Arm, Declined>,
    /// The camera's route.
    pub route: CameraRoute,
}

/// What one camera's calibrations measured, per arm.
#[derive(Debug, Clone, Default, Serialize)]
pub struct UnitRaw {
    /// Per view: `(Σe², n)` of the held-out reprojection residuals, px².
    pub heldout: BTreeMap<Arm, Vec<(f64, usize)>>,
    /// Per view: `(Σe², n)` of the held-out homography residuals, px².
    pub homography: BTreeMap<Arm, Vec<(f64, usize)>>,
    /// Per split: the split-half comparison; `None` where a half failed.
    pub splits: BTreeMap<Arm, Vec<Option<SplitComparison>>>,
    /// The noise level of the calibration on every view.
    pub noise: BTreeMap<Arm, Option<NoiseLevel>>,
    /// Calibrations that failed.
    pub failures: BTreeMap<Arm, usize>,
    /// Grid nodes inside the hull of the camera's corners.
    pub hull_nodes: usize,
}

/// The half-splits of `n` views: the even/odd split, then `random` seeded
/// random halves. `true` puts a view in half A.
pub fn splits(n: usize, random: usize, label: &str) -> Vec<Vec<bool>> {
    let mut rng = Rng::new(seed_of(&format!("{label}/splits")));
    let mut out = vec![(0..n).map(|i| i % 2 == 0).collect::<Vec<_>>()];
    for _ in 0..random {
        let mut order: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut order);
        let mut in_a = vec![false; n];
        for &i in &order[..n / 2] {
            in_a[i] = true;
        }
        out.push(in_a);
    }
    out
}

/// The views a calibration job uses.
#[derive(Debug, Clone, Copy)]
enum Subset {
    /// Every view but fold `f`.
    Train(usize),
    /// One half of split `s` (`true`: half A).
    Half(usize, bool),
    /// Every view.
    All,
}

/// Calibrate every fold, half and the full set for every arm of `unit`,
/// and measure E1–E3.
pub fn measure(unit: &Unit, options: &Options) -> Result<UnitRaw> {
    let reference = unit
        .arms
        .get(&Arm::Uniform)
        .context("the uniform arm is missing")?;
    let n = reference.views.len();
    let split_sets = splits(n, options.random_splits, &unit.label);
    // Every arm's jobs in one fixed layout: the folds, the two halves of
    // each split, then all views.
    let per_arm = FOLDS + 2 * split_sets.len() + 1;
    let slot = |subset: Subset| match subset {
        Subset::Train(f) => f,
        Subset::Half(s, a) => FOLDS + 2 * s + usize::from(!a),
        Subset::All => per_arm - 1,
    };
    let jobs: Vec<(Arm, Subset)> = unit
        .arms
        .keys()
        .flat_map(|&arm| {
            (0..FOLDS)
                .map(Subset::Train)
                .chain(
                    (0..split_sets.len())
                        .flat_map(|s| [Subset::Half(s, true), Subset::Half(s, false)]),
                )
                .chain([Subset::All])
                .map(move |subset| (arm, subset))
        })
        .collect();
    let exports = par_map(&jobs, options.threads, |&(arm, subset)| {
        let keep = |i: usize| match subset {
            Subset::Train(f) => i % FOLDS != f,
            Subset::Half(s, a) => split_sets[s][i] == a,
            Subset::All => true,
        };
        let chosen = unit.arms[&arm]
            .views
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(_, v)| v.clone())
            .collect();
        unit.route.calibrate(chosen).ok()
    });
    let mut raw = UnitRaw::default();
    for ((arm, _), export) in jobs.iter().zip(&exports) {
        if export.is_none() {
            *raw.failures.entry(*arm).or_default() += 1;
        }
    }
    let arm_position: BTreeMap<Arm, usize> = unit
        .arms
        .keys()
        .enumerate()
        .map(|(i, &arm)| (arm, i))
        .collect();
    let get =
        |arm: Arm, subset: Subset| exports[arm_position[&arm] * per_arm + slot(subset)].as_ref();
    let nodes = grid(unit.image_size);
    // The decisive grid: nodes inside the hull of every detected corner.
    let corners: Vec<Pt2> = reference
        .views
        .iter()
        .flat_map(|v| v.obs.points_2d.iter().copied())
        .collect();
    let hull = convex_hull(&corners);
    let in_hull: Vec<bool> = nodes
        .iter()
        .map(|n| inside(&hull, &Pt2::new(n[0], n[1])))
        .collect();
    raw.hull_nodes = in_hull.iter().filter(|&&x| x).count();
    for (&arm, dataset) in &unit.arms {
        // E1: each held-out view under the calibration of its fold.
        let mut heldout = vec![(0.0, 0usize); n];
        let mut homography = vec![(0.0, 0usize); n];
        for (i, view) in dataset.views.iter().enumerate() {
            let Some(export) = get(arm, Subset::Train(i % FOLDS)) else {
                continue;
            };
            heldout[i] = heldout_sums(export, view)?;
            homography[i] = homography_sums(export, view)?;
        }
        raw.heldout.insert(arm, heldout);
        raw.homography.insert(arm, homography);
        // E2/E3: the halves of each split.
        let comparisons = (0..split_sets.len())
            .map(|s| {
                let (Some(a), Some(b)) = (
                    get(arm, Subset::Half(s, true)),
                    get(arm, Subset::Half(s, false)),
                ) else {
                    return Ok(None);
                };
                compare_halves(a, b, &nodes, &in_hull).map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        raw.splits.insert(arm, comparisons);
        raw.noise.insert(
            arm,
            get(arm, Subset::All).and_then(|e| {
                e.uncertainty
                    .covariance()
                    .and_then(|c| c.noise.first().map(|n| n.level))
            }),
        );
    }
    Ok(raw)
}

/// `(Σe², n)` of `view`'s residuals under `export`'s camera, its pose fitted
/// unweighted with the camera fixed.
fn heldout_sums(export: &PlanarIntrinsicsExport, view: &View<NoMeta>) -> Result<(f64, usize)> {
    let mut camera_only = export.clone();
    camera_only.per_feature_residuals.target.clear();
    let report = planar_intrinsics_report(&camera_only, std::slice::from_ref(view))?;
    let errors: Vec<f64> = report.levels[0]
        .residuals
        .iter()
        .filter_map(|r| r.error_px)
        .collect();
    if errors.len() != view.obs.len() {
        // The pose fit failed or a corner did not project: no paired value.
        return Ok((0.0, 0));
    }
    Ok((errors.iter().map(|e| e * e).sum(), errors.len()))
}

/// `(Σe², n)` of the homography residuals of `view`'s corners undistorted by
/// `export`'s camera.
fn homography_sums(export: &PlanarIntrinsicsExport, view: &View<NoMeta>) -> Result<(f64, usize)> {
    let camera = &export.camera_params()[0];
    let Ok(ideal) = undistort(camera, &view.obs.points_2d) else {
        return Ok((0.0, 0));
    };
    let errors = homography_squared_errors(&view.obs.planar_points(), &ideal)?;
    Ok((errors.iter().sum(), errors.len()))
}

/// Compare two halves' cameras, with their stated projection covariances
/// when both have one.
fn compare_halves(
    a: &PlanarIntrinsicsExport,
    b: &PlanarIntrinsicsExport,
    nodes: &[[f64; 2]],
    in_hull: &[bool],
) -> Result<SplitComparison> {
    let covariances = a
        .projection_covariance(0, nodes)
        .ok()
        .zip(b.projection_covariance(0, nodes).ok());
    compare(
        &a.camera_params()[0],
        &b.camera_params()[0],
        nodes,
        in_hull,
        covariances
            .as_ref()
            .map(|(ca, cb)| (ca.as_slice(), cb.as_slice())),
    )
}

/// The per-view paired sums of `arm` against `uniform` on one measure:
/// views where both have every residual.
pub fn paired(per_view: &BTreeMap<Arm, Vec<(f64, usize)>>, arm: Arm) -> Vec<PairedSums> {
    let (Some(reference), Some(other)) = (per_view.get(&Arm::Uniform), per_view.get(&arm)) else {
        return Vec::new();
    };
    reference
        .iter()
        .zip(other)
        .filter(|((_, n), (_, m))| *n > 0 && n == m)
        .map(|(&(a, n), &(x, _))| (a, x, n))
        .collect()
}

/// Per split, `arm`'s decisive-grid projection difference minus
/// `uniform`'s, where both halves of both arms calibrated.
pub fn split_differences(raw: &UnitRaw, arm: Arm) -> Vec<f64> {
    let (Some(reference), Some(other)) = (raw.splits.get(&Arm::Uniform), raw.splits.get(&arm))
    else {
        return Vec::new();
    };
    reference
        .iter()
        .zip(other)
        .filter_map(|(a, x)| Some(x.as_ref()?.hull.rms_px() - a.as_ref()?.hull.rms_px()))
        .filter(|d| d.is_finite())
        .collect()
}

/// `arm`'s per-split honesty sums on the decisive grid, for the splits that
/// have any.
pub fn split_z(raw: &UnitRaw, arm: Arm) -> Vec<ZSums> {
    raw.splits
        .get(&arm)
        .map(|s| {
            s.iter()
                .filter_map(|c| c.map(|c| c.hull.z))
                .filter(|z| z.pairs > 0)
                .collect()
        })
        .unwrap_or_default()
}

/// The pooled ratio `r` of several splits' sums.
pub fn ratio_of(items: &[ZSums]) -> f64 {
    ZSums::total(items).ratio()
}

/// The pooled coverage below `z² = 2.30` of several splits' sums.
pub fn coverage_68_of(items: &[ZSums]) -> f64 {
    ZSums::total(items).coverage_68()
}

/// The pooled coverage below `z² = 6.18` of several splits' sums.
pub fn coverage_95_of(items: &[ZSums]) -> f64 {
    ZSums::total(items).coverage_95()
}

/// The E1 practical margin: `max(0.005 px, 1 %` of uniform's held-out RMS`)`.
pub fn e1_margin(uniform_heldout_rms_px: f64) -> f64 {
    E1_MIN_MARGIN_PX.max(E1_RELATIVE_MARGIN * uniform_heldout_rms_px)
}

/// The E2 practical margin: 2 % of uniform's split RMS.
pub fn e2_margin(uniform_split_rms_px: f64) -> f64 {
    E2_RELATIVE_MARGIN * uniform_split_rms_px
}

/// Smallest E1 margin, px.
const E1_MIN_MARGIN_PX: f64 = 0.005;
/// E1 margin relative to uniform's held-out RMS.
const E1_RELATIVE_MARGIN: f64 = 0.01;
/// E2 margin relative to uniform's split RMS.
const E2_RELATIVE_MARGIN: f64 = 0.02;

/// An effect, its practical margin and its class.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Effect {
    /// arm − `uniform` with its 95 % interval.
    pub interval: Interval,
    /// The size an effect must reach to count, px.
    pub margin: f64,
    /// Better, worse or neutral.
    pub class: Class,
}

impl Effect {
    /// Classify `interval` against `margin`.
    pub fn new(interval: Interval, margin: f64) -> Self {
        Self {
            interval,
            margin,
            class: Class::of(interval, margin),
        }
    }
}

/// E3 on one grid: the pooled ratio and coverages with their intervals.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Honesty {
    /// `r = Σ z² / (2 N)` over every `(split, node)` pair; 1 when honest.
    pub ratio: Interval,
    /// Pooled fraction of pairs below `z² = 2.30` (nominal 0.683).
    pub coverage_68: Interval,
    /// Pooled fraction of pairs below `z² = 6.18` (nominal 0.954).
    pub coverage_95: Interval,
    /// `(split, node)` pairs.
    pub pairs: usize,
}

impl Honesty {
    /// From per-split sums; intervals resample splits.
    pub fn of(splits: &[ZSums], resamples: usize, seed: u64) -> Option<Self> {
        (!splits.is_empty()).then(|| Self {
            ratio: pooled(&[splits], ratio_of, resamples, seed),
            coverage_68: pooled(&[splits], coverage_68_of, resamples, seed ^ 1),
            coverage_95: pooled(&[splits], coverage_95_of, resamples, seed ^ 2),
            pairs: ZSums::total(splits).pairs,
        })
    }
}

/// The full-grid E2 and E3 figures (reported only).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FullGrid {
    /// Median over splits of the split-half difference, px.
    pub split_rms_px: f64,
    /// Pooled `r`.
    pub ratio: Option<f64>,
    /// Pooled coverage below `z² = 2.30`.
    pub coverage_68: Option<f64>,
    /// Pooled coverage below `z² = 6.18`.
    pub coverage_95: Option<f64>,
}

/// One arm's figures on one camera.
#[derive(Debug, Clone, Serialize)]
pub struct ArmSummary {
    /// Pooled held-out reprojection RMS, px (over the views paired with
    /// `uniform`).
    pub heldout_rms_px: f64,
    /// Pooled held-out homography RMS, px.
    pub homography_rms_px: f64,
    /// E1 on the held-out RMS, arm − `uniform` (interval: views).
    pub heldout_effect: Option<Effect>,
    /// E1 on the homography RMS, arm − `uniform` (interval: views).
    pub homography_effect: Option<Effect>,
    /// Median over splits of the split-half projection difference on the
    /// decisive grid, px.
    pub split_rms_px: f64,
    /// E2: median over splits of arm − `uniform` on the decisive grid
    /// (interval: splits).
    pub split_effect: Option<Effect>,
    /// Splits where both halves calibrated (and, for an effect, `uniform`'s
    /// too).
    pub splits_used: usize,
    /// E3 on the decisive grid (intervals: splits).
    pub honesty: Option<Honesty>,
    /// E2 and E3 over the full grid, reported only.
    pub full_grid: FullGrid,
    /// σ̂ (px) or σ̂₀ of the calibration on every view.
    pub noise: Option<NoiseLevel>,
    /// Calibrations of this arm that failed.
    pub failures: usize,
}

/// One camera's figures.
#[derive(Debug, Clone, Serialize)]
pub struct UnitSummary {
    /// `<dataset>/<camera>`.
    pub unit: String,
    /// Registry id.
    pub dataset: String,
    /// Camera id.
    pub camera: String,
    /// Private dataset: aggregates only.
    pub private: bool,
    /// Views with a detected board.
    pub views: usize,
    /// Corners over all views.
    pub corners: usize,
    /// Image size, px.
    pub image_size: [u32; 2],
    /// Grid nodes inside the hull of the camera's corners (the decisive
    /// grid), of the full grid's.
    pub hull_nodes: [usize; 2],
    /// The covariance arms that are absent, and why.
    pub declined: BTreeMap<Arm, Declined>,
    /// Passes the power check: `uniform_noise` raises the held-out RMS with
    /// an interval above 0.
    pub decisive: bool,
    /// Per arm.
    pub arms: BTreeMap<Arm, ArmSummary>,
}

/// Summarize one camera's measurements.
pub fn summarize(unit: &Unit, raw: &UnitRaw, resamples: usize) -> UnitSummary {
    let seed = |what: &str, arm: Arm| seed_of(&format!("{}/{what}/{arm}", unit.label));
    let hull_rms = |arm: Arm| median(raw.splits[&arm].iter().flatten().map(|c| c.hull.rms_px()));
    let uniform_heldout = rms_of(&paired(&raw.heldout, Arm::Uniform), false);
    let (m1, m2) = (
        e1_margin(uniform_heldout),
        e2_margin(hull_rms(Arm::Uniform)),
    );
    let arms = unit
        .arms
        .keys()
        .map(|&arm| {
            let held = paired(&raw.heldout, arm);
            let homo = paired(&raw.homography, arm);
            let effect = |items: &[PairedSums], what: &str| {
                (arm != Arm::Uniform && !items.is_empty()).then(|| {
                    Effect::new(
                        pooled(&[items], rms_difference, resamples, seed(what, arm)),
                        m1,
                    )
                })
            };
            let used = raw.splits[&arm].iter().flatten().count();
            let diffs = split_differences(raw, arm);
            let full: Vec<ZSums> = raw.splits[&arm]
                .iter()
                .flatten()
                .map(|c| c.full.z)
                .filter(|z| z.pairs > 0)
                .collect();
            let full_total = ZSums::total(&full);
            let summary = ArmSummary {
                heldout_rms_px: rms_of(&held, true),
                homography_rms_px: rms_of(&homo, true),
                heldout_effect: effect(&held, "heldout"),
                homography_effect: effect(&homo, "homography"),
                split_rms_px: hull_rms(arm),
                split_effect: (arm != Arm::Uniform && !diffs.is_empty()).then(|| {
                    Effect::new(
                        pooled(&[&diffs], median_of, resamples, seed("split", arm)),
                        m2,
                    )
                }),
                splits_used: if arm == Arm::Uniform {
                    used
                } else {
                    diffs.len()
                },
                honesty: Honesty::of(&split_z(raw, arm), resamples, seed("z2", arm)),
                full_grid: FullGrid {
                    split_rms_px: median(
                        raw.splits[&arm].iter().flatten().map(|c| c.full.rms_px()),
                    ),
                    ratio: (full_total.pairs > 0).then(|| full_total.ratio()),
                    coverage_68: (full_total.pairs > 0).then(|| full_total.coverage_68()),
                    coverage_95: (full_total.pairs > 0).then(|| full_total.coverage_95()),
                },
                noise: raw.noise.get(&arm).copied().flatten(),
                failures: raw.failures.get(&arm).copied().unwrap_or(0),
            };
            (arm, summary)
        })
        .collect::<BTreeMap<_, _>>();
    let decisive = arms
        .get(&Arm::UniformNoise)
        .and_then(|a| a.heldout_effect)
        .is_some_and(|e| e.interval.above_zero());
    let reference = &unit.arms[&Arm::Uniform];
    UnitSummary {
        unit: unit.label.clone(),
        dataset: unit.dataset.clone(),
        camera: unit.camera.clone(),
        private: unit.private,
        views: reference.views.len(),
        corners: reference.views.iter().map(|v| v.obs.len()).sum(),
        image_size: unit.image_size,
        hull_nodes: [raw.hull_nodes, grid(unit.image_size).len()],
        declined: unit.declined.clone(),
        decisive,
        arms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_halves_and_deterministic() {
        let s = splits(11, 3, "x");
        assert_eq!(s.len(), 4);
        assert_eq!(s, splits(11, 3, "x"));
        assert_eq!(s[0].iter().filter(|&&a| a).count(), 6);
        for split in &s[1..] {
            assert_eq!(split.iter().filter(|&&a| a).count(), 5);
        }
        assert_ne!(s[1], s[2]);
    }

    /// A synthetic camera seen in 15 views with N(0, 0.1²) px corner noise;
    /// `known1` states the true isotropic covariance.
    fn synthetic_unit(seed: u64) -> Unit {
        use vision_calibration_core::synthetic::{planar as synth, poses};
        use vision_calibration_core::{
            BrownConrady5, CorrespondenceView, FxFyCxCy, PointCovariance, make_pinhole_camera,
        };
        let camera = make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 790.0,
                cx: 320.0,
                cy: 240.0,
            },
            BrownConrady5 {
                k1: -0.1,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        );
        let board = synth::grid_points(8, 6, 0.03);
        let specs = poses::seeded_board_pose_specs(15, 0.5, 11 + seed);
        let poses = poses::centered_board_poses(&specs, (0.105, 0.075), 0.5);
        let mut rng = Rng::new(5 + seed);
        let sigma = 0.1;
        let mut uniform = Vec::new();
        let mut known = Vec::new();
        for pose in &poses {
            let view = synth::project_view_all(&camera, pose, &board).unwrap();
            let px: Vec<_> = view
                .points_2d
                .iter()
                .map(|p| {
                    vision_calibration_core::Pt2::new(
                        p.x + sigma * rng.normal(),
                        p.y + sigma * rng.normal(),
                    )
                })
                .collect();
            let cov =
                vec![PointCovariance::new(sigma * sigma, 0.0, sigma * sigma).unwrap(); px.len()];
            uniform.push(View::without_meta(
                CorrespondenceView::new(board.clone(), px.clone()).unwrap(),
            ));
            known.push(View::without_meta(
                CorrespondenceView::new_with_covariances(board.clone(), px, cov).unwrap(),
            ));
        }
        let uniform = PlanarDataset::new(uniform).unwrap();
        let noisy = super::super::arms::with_noise(&uniform, "synthetic").unwrap();
        Unit {
            label: "synthetic/cam0".into(),
            dataset: "synthetic".into(),
            camera: "cam0".into(),
            private: false,
            image_size: [640, 480],
            arms: BTreeMap::from([
                (Arm::Uniform, uniform),
                (Arm::Known1, PlanarDataset::new(known).unwrap()),
                (Arm::UniformNoise, noisy),
            ]),
            declined: BTreeMap::from([(Arm::Estimated, Declined::Missing)]),
            route: CameraRoute {
                config: PlanarIntrinsicsConfig::default(),
                seed: None,
            },
        }
    }

    #[test]
    fn smoke_synthetic_camera_through_every_measure() {
        let unit = synthetic_unit(0);
        let options = Options {
            random_splits: 20,
            resamples: 200,
            threads: 4,
            ..Options::default()
        };
        let raw = measure(&unit, &options).unwrap();
        assert!(raw.failures.is_empty(), "{:?}", raw.failures);
        let summary = summarize(&unit, &raw, options.resamples);
        let uniform = &summary.arms[&Arm::Uniform];
        let known = &summary.arms[&Arm::Known1];
        assert!(
            uniform.heldout_rms_px > 0.05 && uniform.heldout_rms_px < 0.2,
            "{uniform:?}"
        );
        assert_eq!(uniform.splits_used, options.random_splits + 1);
        // A constant isotropic covariance weights every corner alike: the same
        // calibrations as `uniform`, and σ̂₀ ≈ 1 for the true σ.
        let e1 = known.heldout_effect.unwrap();
        assert!(e1.interval.point.abs() < 1e-6, "{e1:?}");
        assert_eq!(e1.class, Class::Neutral);
        assert!((e1.margin - 0.005).abs() < 1e-12, "the 0.005 px floor");
        assert!(known.split_effect.unwrap().interval.point.abs() < 1e-6);
        match known.noise {
            Some(NoiseLevel::Sigma0(s)) => assert!((0.8..1.25).contains(&s), "σ̂₀ {s}"),
            other => panic!("known1 reports {other:?}"),
        }
        assert!(matches!(uniform.noise, Some(NoiseLevel::Sigma(s)) if (0.08..0.125).contains(&s)));
        // The decisive grid is the part of the image the corners reach.
        let [hull, full] = summary.hull_nodes;
        assert!(hull > 10 && hull < full, "{hull} of {full}");
        assert!(uniform.split_rms_px < uniform.full_grid.split_rms_px);
        let honesty = uniform.honesty.unwrap();
        assert_eq!(honesty.pairs, hull * uniform.splits_used);
        // The power check sees its own noise.
        assert!(
            summary.arms[&Arm::UniformNoise]
                .heldout_effect
                .unwrap()
                .interval
                .point
                > 0.0
        );
        assert_eq!(summary.declined[&Arm::Estimated], Declined::Missing);
        // The rule runs on it.
        let decision = super::super::rule::decide(&[(&summary, &raw)], None, 200);
        assert!(!decision.arms[&Arm::Known1].flips, "no timing: never flips");
    }

    /// The synthetic truth gate (Amendment 1): with exactly true noise,
    /// pooled over independent scenes, every split and every decisive node,
    /// `r` lies in 0.85–1.15 and both coverages within ±5 points of nominal.
    /// Pooling over many scenes matters: one scene's splits share its data,
    /// so their `z²` move together.
    #[test]
    fn synthetic_truth_gate() {
        const SCENES: u64 = 24;
        let options = Options {
            random_splits: 20,
            resamples: 10,
            threads: std::thread::available_parallelism().map_or(4, usize::from),
            ..Options::default()
        };
        let mut sums: BTreeMap<Arm, Vec<ZSums>> = BTreeMap::new();
        for seed in 0..SCENES {
            let mut unit = synthetic_unit(seed);
            unit.arms.remove(&Arm::UniformNoise);
            let raw = measure(&unit, &options).unwrap();
            assert!(raw.failures.is_empty(), "{:?}", raw.failures);
            for arm in [Arm::Uniform, Arm::Known1] {
                sums.entry(arm).or_default().extend(split_z(&raw, arm));
            }
        }
        for (arm, s) in &sums {
            let total = ZSums::total(s);
            let (r, c68, c95) = (total.ratio(), total.coverage_68(), total.coverage_95());
            eprintln!(
                "synthetic truth gate {arm}: r {r:.3}, ≤2.30 {:.1} %, ≤6.18 {:.1} %, {} pairs",
                100.0 * c68,
                100.0 * c95,
                total.pairs
            );
            assert!((0.85..=1.15).contains(&r), "{arm}: r = {r}");
            assert!((c68 - 0.683).abs() <= 0.05, "{arm}: coverage 68 = {c68}");
            assert!((c95 - 0.954).abs() <= 0.05, "{arm}: coverage 95 = {c95}");
        }
    }

    #[test]
    fn pairing_drops_views_without_every_residual() {
        let mut per_view = BTreeMap::new();
        per_view.insert(Arm::Uniform, vec![(1.0, 2), (1.0, 2), (0.0, 0)]);
        per_view.insert(Arm::Known1, vec![(2.0, 2), (0.0, 0), (1.0, 2)]);
        assert_eq!(paired(&per_view, Arm::Known1), vec![(1.0, 2.0, 2)]);
    }
}
