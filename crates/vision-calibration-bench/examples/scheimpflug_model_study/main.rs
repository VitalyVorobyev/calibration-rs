//! What limits a tilted (Scheimpflug) camera's reprojection error: a model
//! study over registry datasets.
//!
//! For each planar Scheimpflug dataset it runs the seeded acceptance route
//! (Brown-Conrady `k1, k2` under a tilted sensor), then refines camera model
//! variants from it and scores each on
//!
//! - the training views (the residual of the fit itself), and
//! - held-out views: K-fold over views, the route and the variant refit on
//!   the other folds, then only the held-out views' poses fitted with the
//!   camera held,
//!
//! each through the calibration-free smooth-fit floor of
//! `vision_calibration_bench::floor`. `--parts` picks what runs:
//!
//! - `lib`: the library's distortion models under fix masks, and the
//!   solver's stopping rule (the route against a tightly converged refine on
//!   both backends);
//! - `gen`: terms the library does not have, written as a user residual
//!   (`model.rs`): a distortion centre, the tilt before the distortion, a
//!   depth-dependent `k1`, a board shape, a free image-space field;
//! - `basin`: the tilt perturbed from the optimum, with the principal point
//!   as is and measured from the tilt's valley;
//! - `blur`: train on sharp or blurred markers (the detector's fitted blur,
//!   `blur.rs`), score on either;
//! - `xcam`: one camera's image field held on another;
//! - `target`: a non-ideal target fitted jointly over every given camera
//!   (`target.rs`, `target_study.rs`): in-plane point offsets, an
//!   out-of-plane shape, cover-layer refraction, a view-angle shift, with
//!   or without an image field, scored on held-out captures; `--models`
//!   picks a subset by name (also for `lib` and `gen`);
//! - `mimic`: the camera-fixed field (route + a held polynomial field)
//!   copied without noise, its structure and what each candidate takes
//!   from it (`mimic.rs`);
//! - `synth`: synthetic scenes (`synth_plain_<seed>`, `synth_outside_<seed>`,
//!   `synth_spline_<seed>` as dataset names) scored like the datasets;
//!   `mc`: the Monte-Carlo of the B-spline field's σ (`synth.rs`).
//!
//! Image fields are evaluated at the model's own projection; the
//! observation-evaluated form (`@obs`) is kept to measure why it is not a
//! camera model.
//!
//! The study, and the per-point residuals for further analysis, are written
//! to `--out`. Findings: `docs/notes/scheimpflug-residual-model.md` and
//! `docs/notes/display-target-residual.md`, `docs/notes/camera-residual-field.md`.
//!
//! ```text
//! cargo run --release -p vision-calibration-bench --features tier-b \
//!     --example scheimpflug_model_study -- \
//!     --registry crates/vision-calibration-bench/registry/private.json \
//!     --out <dir> [--folds 4] [--parts lib,gen,basin,blur,xcam,target] \
//!     [--models <name>,...] <dataset-id>...
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use nalgebra::Point3;
use serde::Serialize;

use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration_bench::floor::{self, FloorCamera, FloorPoint};
use vision_calibration_bench::registry::{BenchEntry, load_registry};
use vision_calibration_bench::run::tier_b::detect_scheimpflug_seeded_input;
use vision_calibration_core::{
    CameraFix, CameraModelSpec, CameraParams, DistortionKind, DistortionParams, Iso3, NoMeta,
    PlanarDataset, RationalPolynomial, SensorKind, SensorParams, ThinPrism, Uncertainty, View,
};
mod blur;
mod mimic;
mod model;
mod synth;
mod target;
mod target_study;

use model::{Centre, Free, GenCfg, GenFit, GenParams, SensorTerms};
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RobustLoss, SolverBackend, optimize_planar_intrinsics,
};

const HUBER_PX: f64 = 1.0;

/// A camera model variant: a distortion kind and its free coefficients.
#[derive(Debug, Clone)]
struct Variant {
    name: &'static str,
    kind: DistortionKind,
    free: &'static [&'static str],
}

const VARIANTS: &[Variant] = &[
    Variant {
        name: "bc5 k1k2 (default)",
        kind: DistortionKind::BrownConrady5,
        free: &["k1", "k2"],
    },
    Variant {
        name: "bc5 +k3",
        kind: DistortionKind::BrownConrady5,
        free: &["k1", "k2", "k3"],
    },
    Variant {
        name: "bc5 +p1p2",
        kind: DistortionKind::BrownConrady5,
        free: &["k1", "k2", "p1", "p2"],
    },
    Variant {
        name: "bc5 all",
        kind: DistortionKind::BrownConrady5,
        free: &["k1", "k2", "k3", "p1", "p2"],
    },
    Variant {
        name: "tp9 k1k2+s",
        kind: DistortionKind::ThinPrism9,
        free: &["k1", "k2", "s1", "s2", "s3", "s4"],
    },
    Variant {
        name: "tp9 k1k2+s2s4",
        kind: DistortionKind::ThinPrism9,
        free: &["k1", "k2", "s2", "s4"],
    },
    Variant {
        name: "tp9 k1k2+p+s1s3",
        kind: DistortionKind::ThinPrism9,
        free: &["k1", "k2", "p1", "p2", "s1", "s3"],
    },
    Variant {
        name: "tp9 k1k2+p+s",
        kind: DistortionKind::ThinPrism9,
        free: &["k1", "k2", "p1", "p2", "s1", "s2", "s3", "s4"],
    },
    Variant {
        name: "tp9 all",
        kind: DistortionKind::ThinPrism9,
        free: &["k1", "k2", "k3", "p1", "p2", "s1", "s2", "s3", "s4"],
    },
    Variant {
        name: "rat8 k1k2k4",
        kind: DistortionKind::Rational8,
        free: &["k1", "k2", "k4"],
    },
    Variant {
        name: "rat8 all",
        kind: DistortionKind::Rational8,
        free: &["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"],
    },
];

struct Args {
    registry: PathBuf,
    out: PathBuf,
    folds: usize,
    /// Parts to run: `lib`, `gen`, `basin` (all by default).
    parts: Vec<String>,
    /// Target-study models to run (all when empty).
    models: Vec<String>,
    datasets: Vec<String>,
}

fn parse_args() -> Result<Args> {
    let mut registry = None;
    let mut out = None;
    let mut folds = 4;
    let mut parts = vec!["lib".to_string(), "gen".to_string(), "basin".to_string()];
    let mut datasets = Vec::new();
    let mut models = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--registry" => registry = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--folds" => folds = it.next().context("--folds needs a value")?.parse()?,
            "--parts" => {
                parts = it
                    .next()
                    .context("--parts needs a value")?
                    .split(',')
                    .map(str::to_string)
                    .collect();
            }
            "--models" => {
                models = it
                    .next()
                    .context("--models needs a value")?
                    .split(',')
                    .map(str::to_string)
                    .collect();
            }
            _ if a.starts_with("--") => bail!("unknown flag {a}"),
            _ => datasets.push(a),
        }
    }
    Ok(Args {
        registry: registry.context("--registry is required")?,
        out: out.context("--out is required")?,
        folds,
        parts,
        models,
        datasets,
    })
}

// ───────────────────────── solves ─────────────────────────

fn tight(backend: SolverBackend, max_iters: usize) -> BackendSolveOptions {
    BackendSolveOptions {
        backend,
        max_iters,
        min_abs_decrease: Some(1e-14),
        min_rel_decrease: Some(1e-14),
        min_error: Some(1e-20),
        ..BackendSolveOptions::default()
    }
}

/// The seeded acceptance route (as `calib-bench accept` runs it).
fn route(dataset: &PlanarDataset, seed: &IntrinsicsManualInit) -> Result<PlanarIntrinsicsEstimate> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset.clone())?;
    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model =
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
    config.solver.max_iters = 120;
    config.optimize.reprojection_loss = RobustLoss::Huber { scale: HUBER_PX };
    session.set_config(config)?;
    step_init_with_seed(&mut session, seed.clone(), None)?;
    step_optimize(&mut session, None)?;
    session
        .output()
        .cloned()
        .context("step_optimize left no estimate")
}

/// `camera` with its distortion re-expressed in `kind` (new terms zero).
fn convert(camera: &CameraParams, kind: DistortionKind) -> Result<CameraParams> {
    let (k1, k2, k3, p1, p2, iters) = match &camera.distortion {
        DistortionParams::BrownConrady5 { params: p } => (p.k1, p.k2, p.k3, p.p1, p.p2, p.iters),
        DistortionParams::ThinPrism { params: p } => (p.k1, p.k2, p.k3, p.p1, p.p2, p.iters),
        DistortionParams::Rational { params: p } => (p.k1, p.k2, p.k3, p.p1, p.p2, p.iters),
        other => bail!("unsupported distortion {other:?}"),
    };
    let distortion = match kind {
        DistortionKind::BrownConrady5 => {
            let mut p = match &camera.distortion {
                DistortionParams::BrownConrady5 { params } => *params,
                _ => bail!("cannot narrow a distortion model to bc5"),
            };
            p.k1 = k1;
            DistortionParams::BrownConrady5 { params: p }
        }
        DistortionKind::ThinPrism9 => DistortionParams::ThinPrism {
            params: ThinPrism {
                k1,
                k2,
                k3,
                p1,
                p2,
                s1: 0.0,
                s2: 0.0,
                s3: 0.0,
                s4: 0.0,
                iters,
            },
        },
        DistortionKind::Rational8 => DistortionParams::Rational {
            params: RationalPolynomial {
                k1,
                k2,
                k3,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
                p1,
                p2,
                iters,
            },
        },
        other => bail!("unsupported distortion kind {other:?}"),
    };
    Ok(camera.clone().with_distortion(distortion))
}

fn variant_fix(v: &Variant) -> CameraFix {
    CameraFix {
        distortion: Some(
            v.kind
                .param_names()
                .iter()
                .filter(|n| !v.free.contains(n))
                .map(|n| n.to_string())
                .collect(),
        ),
        ..CameraFix::none()
    }
}

/// Refine `start` as variant `v`, every pose free, Huber loss, converged
/// tightly.
fn refine(
    dataset: &PlanarDataset,
    start: &PlanarIntrinsicsParams,
    v: &Variant,
    backend: SolverBackend,
    max_iters: usize,
) -> Result<PlanarIntrinsicsEstimate> {
    let camera = convert(&start.camera, v.kind)?;
    let params = PlanarIntrinsicsParams::new(camera, start.cam_se3_target.clone())?;
    let opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::Huber { scale: HUBER_PX },
        fix: variant_fix(v),
        fix_poses: Vec::new(),
        bounds: None,
    };
    Ok(optimize_planar_intrinsics(
        dataset,
        &params,
        opts,
        tight(backend, max_iters),
    )?)
}

/// Fit only the poses of `dataset`'s views, the camera held.
fn pose_only(
    dataset: &PlanarDataset,
    camera: &CameraParams,
    poses: Vec<Iso3>,
) -> Result<PlanarIntrinsicsEstimate> {
    let spec = camera.spec().context("camera spec")?;
    let params = PlanarIntrinsicsParams::new(camera.clone(), poses)?;
    let opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::Huber { scale: HUBER_PX },
        fix: CameraFix::all(spec),
        fix_poses: Vec::new(),
        bounds: None,
    };
    Ok(optimize_planar_intrinsics(
        dataset,
        &params,
        opts,
        tight(SolverBackend::TinySolver, 200),
    )?)
}

fn subset(dataset: &PlanarDataset, views: &[usize]) -> Result<PlanarDataset> {
    Ok(PlanarDataset::new(
        views.iter().map(|&v| dataset.views[v].clone()).collect(),
    )?)
}

// ───────────────────────── residuals ─────────────────────────

/// One point's calibrated residual with its geometry.
#[derive(Debug, Clone, Serialize)]
struct PointResidual {
    view: usize,
    board_mm: [f64; 2],
    observed_px: [f64; 2],
    residual_px: [f64; 2],
    /// Camera-frame position of the board point, millimetres.
    cam_mm: [f64; 3],
    /// The detector's fitted blur σ, pixels.
    blur_px: Option<f64>,
}

fn residuals(
    view_ids: &[usize],
    views: &[View<vision_calibration_core::NoMeta>],
    camera: &CameraParams,
    poses: &[Iso3],
) -> Result<Vec<PointResidual>> {
    let model = camera.build()?;
    let mut out = Vec::new();
    for ((&id, view), pose) in view_ids.iter().zip(views).zip(poses) {
        for (p3, p2) in view.obs.points_3d.iter().zip(&view.obs.points_2d) {
            let pc: Point3<f64> = pose * p3;
            let Some(proj) = model.project_point(&pc) else {
                continue;
            };
            out.push(PointResidual {
                view: id,
                board_mm: [p3.x * 1e3, p3.y * 1e3],
                observed_px: [p2.x, p2.y],
                residual_px: [p2.x - proj.x, p2.y - proj.y],
                cam_mm: [pc.x * 1e3, pc.y * 1e3, pc.z * 1e3],
                blur_px: None,
            });
        }
    }
    Ok(out)
}

fn floor_camera(id: &str, size: [u32; 2], pts: &[PointResidual]) -> FloorCamera {
    FloorCamera {
        id: id.to_string(),
        image_size: Some([f64::from(size[0]), f64::from(size[1])]),
        points: pts
            .iter()
            .map(|p| FloorPoint {
                view: p.view,
                board_mm: p.board_mm,
                observed_px: p.observed_px,
                residual_px: Some(p.residual_px),
            })
            .collect(),
    }
}

fn rms_median(pts: &[PointResidual]) -> (f64, f64) {
    let mut m: Vec<f64> = pts
        .iter()
        .map(|p| p.residual_px[0].hypot(p.residual_px[1]))
        .collect();
    let rms = (m.iter().map(|v| v * v).sum::<f64>() / m.len().max(1) as f64).sqrt();
    m.sort_by(f64::total_cmp);
    let med = if m.is_empty() {
        f64::NAN
    } else {
        m[m.len() / 2]
    };
    (rms, med)
}

// ───────────────────────── parameters ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ParamSummary {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    tilt_x: f64,
    tilt_y: f64,
    k1: f64,
    k2: f64,
    extra: BTreeMap<String, f64>,
    /// Stated σ by label (camera parameters only).
    sigma: BTreeMap<String, f64>,
    /// Correlations of interest.
    corr: BTreeMap<String, f64>,
    covariance: String,
    iters: usize,
    cost: f64,
}

fn summarize(est: &PlanarIntrinsicsEstimate) -> ParamSummary {
    let c = &est.params.camera;
    let (tilt_x, tilt_y) = match c.sensor {
        SensorParams::Scheimpflug { params } => (params.tilt_x, params.tilt_y),
        _ => (0.0, 0.0),
    };
    let mut extra = BTreeMap::new();
    let (k1, k2) = match &c.distortion {
        DistortionParams::BrownConrady5 { params: p } => {
            extra.insert("k3".into(), p.k3);
            extra.insert("p1".into(), p.p1);
            extra.insert("p2".into(), p.p2);
            (p.k1, p.k2)
        }
        DistortionParams::ThinPrism { params: p } => {
            for (n, v) in [
                ("k3", p.k3),
                ("p1", p.p1),
                ("p2", p.p2),
                ("s1", p.s1),
                ("s2", p.s2),
                ("s3", p.s3),
                ("s4", p.s4),
            ] {
                extra.insert(n.into(), v);
            }
            (p.k1, p.k2)
        }
        DistortionParams::Rational { params: p } => {
            for (n, v) in [
                ("k3", p.k3),
                ("k4", p.k4),
                ("k5", p.k5),
                ("k6", p.k6),
                ("p1", p.p1),
                ("p2", p.p2),
            ] {
                extra.insert(n.into(), v);
            }
            (p.k1, p.k2)
        }
        _ => (f64::NAN, f64::NAN),
    };
    let mut sigma = BTreeMap::new();
    let mut corr = BTreeMap::new();
    let covariance = match &est.uncertainty {
        Uncertainty::Available(cov) => {
            let idx = |name: &str| {
                cov.labels
                    .iter()
                    .position(|l| l == name || l.ends_with(&format!(".{name}")))
            };
            for (i, l) in cov.labels.iter().enumerate() {
                sigma.insert(l.clone(), cov.covariance[i][i].sqrt());
            }
            for (a, b) in [
                ("cy", "tilt_x"),
                ("cx", "tilt_y"),
                ("fx", "fy"),
                ("fx", "k1"),
                ("fx", "tilt_x"),
                ("k1", "k2"),
            ] {
                if let (Some(i), Some(j)) = (idx(a), idx(b)) {
                    let r =
                        cov.covariance[i][j] / (cov.covariance[i][i] * cov.covariance[j][j]).sqrt();
                    corr.insert(format!("{a}/{b}"), r);
                }
            }
            "available".to_string()
        }
        Uncertainty::Unavailable { reason } => format!("{reason:?}"),
    };
    ParamSummary {
        fx: c.intrinsics.fx,
        fy: c.intrinsics.fy,
        cx: c.intrinsics.cx,
        cy: c.intrinsics.cy,
        tilt_x,
        tilt_y,
        k1,
        k2,
        extra,
        sigma,
        corr,
        covariance,
        iters: est.report.num_iters,
        cost: est.report.final_cost,
    }
}

// ───────────────────────── models ─────────────────────────

const GEN_MODELS: &[GenCfg] = &[
    GenCfg::base("gen base"),
    GenCfg {
        dc: true,
        ..GenCfg::base("gen +dist centre")
    },
    GenCfg {
        tilt_first: true,
        ..GenCfg::base("gen tilt first")
    },
    GenCfg {
        depth: true,
        ..GenCfg::base("gen +depth k1")
    },
    GenCfg {
        board_deg: 2,
        ..GenCfg::base("gen +board2")
    },
    GenCfg {
        board_deg: 3,
        ..GenCfg::base("gen +board3")
    },
    GenCfg {
        board_deg: 4,
        ..GenCfg::base("gen +board4")
    },
    GenCfg {
        field_deg: 4,
        ..GenCfg::base("gen +field4")
    },
    GenCfg {
        field_deg: 6,
        ..GenCfg::base("gen +field6")
    },
    GenCfg {
        field_deg: 4,
        field_at_obs: true,
        ..GenCfg::base("gen +field4 @obs")
    },
    GenCfg {
        field_deg: 4,
        fix_intr: true,
        fix_dist: true,
        fix_tilt: true,
        ..GenCfg::base("gen +field4 camera held")
    },
    GenCfg {
        field_deg: 4,
        field_at_obs: true,
        fix_intr: true,
        fix_dist: true,
        fix_tilt: true,
        ..GenCfg::base("gen +field4 @obs camera held")
    },
    GenCfg {
        board_deg: 4,
        field_deg: 4,
        ..GenCfg::base("gen +board4+field4")
    },
    GenCfg {
        pupil: true,
        ..GenCfg::base("gen +pupil")
    },
    GenCfg {
        field_deg: 2,
        ..GenCfg::base("gen +field2")
    },
    GenCfg {
        field_deg: 3,
        ..GenCfg::base("gen +field3")
    },
    GenCfg {
        field_min_deg: 3,
        field_deg: 3,
        ..GenCfg::base("gen +field3 only")
    },
    GenCfg {
        field_min_deg: 4,
        field_deg: 4,
        ..GenCfg::base("gen +field4 only")
    },
    GenCfg {
        free_dist: &["k1", "k2", "p1", "p2", "s1", "s2", "s3", "s4"],
        fix_tilt: true,
        ..GenCfg::base("gen p+s, tilt held")
    },
    GenCfg {
        board_deg: 4,
        depth: true,
        dc: true,
        ..GenCfg::base("gen +board4+depth+dc")
    },
    // Structured candidates for the camera-fixed field.
    GenCfg {
        tilt_first: true,
        dc: true,
        ..GenCfg::base("gen tilt first +dc")
    },
    GenCfg {
        free_dist: &["k1", "k2", "k3"],
        dc: true,
        ..GenCfg::base("gen +k3 +dc")
    },
    GenCfg {
        elliptic: true,
        ..GenCfg::base("gen +elliptic")
    },
    GenCfg {
        sensor: SensorTerms::Radial(1),
        sensor_centre: Centre::Principal,
        ..GenCfg::base("gen sensor bowl @pp")
    },
    GenCfg {
        sensor: SensorTerms::Radial(1),
        ..GenCfg::base("gen sensor bowl @foot")
    },
    GenCfg {
        sensor: SensorTerms::Radial(2),
        ..GenCfg::base("gen sensor bowl2 @foot")
    },
    GenCfg {
        sensor: SensorTerms::Radial(1),
        sensor_centre: Centre::Free,
        ..GenCfg::base("gen sensor bowl @free")
    },
    GenCfg {
        sensor: SensorTerms::Poly(2),
        ..GenCfg::base("gen sensor h2 @foot")
    },
    GenCfg {
        sensor: SensorTerms::Poly(3),
        ..GenCfg::base("gen sensor h3 @foot")
    },
    GenCfg {
        sensor: SensorTerms::Poly(3),
        sensor_centre: Centre::Principal,
        ..GenCfg::base("gen sensor h3 @pp")
    },
    GenCfg {
        sensor: SensorTerms::Poly(3),
        sensor_centre: Centre::Free,
        ..GenCfg::base("gen sensor h3 @free")
    },
    GenCfg {
        sensor: SensorTerms::Poly(4),
        ..GenCfg::base("gen sensor h4 @foot")
    },
    GenCfg {
        spline: [10, 8],
        spline_sigma: 10.0,
        spline_ortho: true,
        spline_ortho_cover: true,
        ..GenCfg::base("gen spline10x8 ortho-cover s10")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 30.0,
        spline_ortho: true,
        spline_ortho_cover: true,
        ..GenCfg::base("gen spline6x5 ortho-cover s30")
    },
    // The generic field done as a camera term: a B-spline over the image,
    // evaluated at the model's own projection, smoothness prior σ (px).
    GenCfg {
        spline: [6, 5],
        spline_sigma: 0.3,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s0.3")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 1.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s1")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 3.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s3")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 10.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s10")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 30.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s30")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 100.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline6x5 ortho s100")
    },
    GenCfg {
        spline: [8, 6],
        spline_sigma: 3.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline8x6 ortho s3")
    },
    GenCfg {
        spline: [4, 3],
        spline_sigma: 3.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline4x3 ortho s3")
    },
    GenCfg {
        spline: [8, 6],
        spline_sigma: 10.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline8x6 ortho s10")
    },
    GenCfg {
        spline: [4, 3],
        spline_sigma: 10.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline4x3 ortho s10")
    },
    GenCfg {
        spline: [8, 6],
        spline_sigma: 30.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline8x6 ortho s30")
    },
    GenCfg {
        spline: [4, 3],
        spline_sigma: 30.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline4x3 ortho s30")
    },
    GenCfg {
        spline: [10, 8],
        spline_sigma: 3.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline10x8 ortho s3")
    },
    GenCfg {
        spline: [10, 8],
        spline_sigma: 10.0,
        spline_ortho: true,
        ..GenCfg::base("gen spline10x8 ortho s10")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 1.0,
        fix_intr: true,
        fix_dist: true,
        fix_tilt: true,
        ..GenCfg::base("gen spline6x5 held s1")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 1.0,
        ..GenCfg::base("gen spline6x5 free s1")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 10.0,
        fix_intr: true,
        fix_dist: true,
        fix_tilt: true,
        ..GenCfg::base("gen spline6x5 held s10")
    },
    GenCfg {
        spline: [6, 5],
        spline_sigma: 10.0,
        ..GenCfg::base("gen spline6x5 free s10")
    },
];

#[derive(Clone, Copy)]
enum Model<'a> {
    Lib(&'a Variant),
    Gen(&'a GenCfg),
}

impl Model<'_> {
    fn name(&self) -> &'static str {
        match self {
            Model::Lib(v) => v.name,
            Model::Gen(g) => g.name,
        }
    }
}

// ───────────────────────── study ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct VariantResult {
    variant: String,
    camera: String,
    full: ParamSummary,
    train_rms: f64,
    train_median: f64,
    heldout_rms: f64,
    heldout_median: f64,
    heldout_mean: f64,
    /// Held-out residual norms, for pooling.
    #[serde(skip)]
    heldout_norms: Vec<f64>,
    /// Parameters of each fold's training fit.
    folds: Vec<ParamSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct StoppingCheck {
    camera: String,
    backend: String,
    route: ParamSummary,
    tight: ParamSummary,
}

#[derive(Debug, Clone, Serialize, Default)]
struct Study {
    variants: Vec<VariantResult>,
    stopping: Vec<StoppingCheck>,
    /// Floor reports pooled over cameras, per variant: training and held-out.
    floor_train: BTreeMap<String, floor::FloorReport>,
    floor_heldout: BTreeMap<String, floor::FloorReport>,
    basin: Vec<BasinRow>,
}

struct CameraData {
    id: String,
    dataset: PlanarDataset,
    seed: IntrinsicsManualInit,
    size: [u32; 2],
    blur: BTreeMap<(i64, i64), f64>,
    /// The capture (image file stem) of each view.
    captures: Vec<String>,
}

fn load(args: &Args) -> Result<Vec<CameraData>> {
    let registry = load_registry(&args.registry)?;
    let root = std::env::current_dir()?;
    let mut out = Vec::new();
    for id in &args.datasets {
        let mut entry: BenchEntry = registry
            .find(id)
            .with_context(|| format!("no dataset {id}"))?
            .clone();
        if entry.data_root.is_relative() {
            entry.data_root = root.join(&entry.data_root);
        }
        let d = detect_scheimpflug_seeded_input(&entry)?;
        // The blur map re-runs the photometric detector on every image; only
        // the blur part and the default parts' blur/depth summary read it.
        let blur = if args.parts.iter().any(|p| p == "blur" || p == "lib") {
            blur::blur_map(&entry)?
        } else {
            BTreeMap::new()
        };
        let captures = d
            .view_paths
            .iter()
            .map(|p| {
                p.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        out.push(CameraData {
            id: id.clone(),
            dataset: d.dataset,
            seed: d.seed,
            size: d.image_size,
            blur,
            captures,
        });
    }
    Ok(out)
}

struct CamCtx {
    base: PlanarIntrinsicsEstimate,
    folds: Vec<(Vec<usize>, Vec<usize>)>,
    fold_routes: Vec<PlanarIntrinsicsEstimate>,
    centre: [f64; 2],
}

/// The route's solution and the folds (by view index) with their routes.
fn make_ctx(cam: &CameraData, base: PlanarIntrinsicsEstimate, k: usize) -> Result<CamCtx> {
    let n = cam.dataset.num_views();
    let folds: Vec<(Vec<usize>, Vec<usize>)> = (0..k)
        .map(|f| {
            let test: Vec<usize> = (0..n).filter(|v| v % k == f).collect();
            let train: Vec<usize> = (0..n).filter(|v| v % k != f).collect();
            (train, test)
        })
        .collect();
    let mut fold_routes = Vec::new();
    for (train, _) in &folds {
        fold_routes.push(route(&subset(&cam.dataset, train)?, &cam.seed)?);
    }
    Ok(CamCtx {
        centre: board_centre(&cam.dataset),
        base,
        folds,
        fold_routes,
    })
}

fn board_centre(ds: &PlanarDataset) -> [f64; 2] {
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
    for v in &ds.views {
        for p in &v.obs.points_3d {
            sx += p.x;
            sy += p.y;
            n += 1.0;
        }
    }
    [sx / n, sy / n]
}

fn gen_init(est: &PlanarIntrinsicsEstimate, centre: [f64; 2]) -> Result<GenParams> {
    let c = &est.params.camera;
    let DistortionParams::BrownConrady5 { params: d } = &c.distortion else {
        bail!("the route returns Brown-Conrady");
    };
    let SensorParams::Scheimpflug { params: s } = c.sensor else {
        bail!("the route returns a Scheimpflug sensor");
    };
    Ok(GenParams {
        intr: [
            c.intrinsics.fx,
            c.intrinsics.fy,
            c.intrinsics.cx,
            c.intrinsics.cy,
        ],
        dist: [d.k1, d.k2, d.k3, d.p1, d.p2, 0.0, 0.0, 0.0, 0.0],
        tilt: [s.tilt_x, s.tilt_y],
        dc: [0.0; 2],
        depth: 0.0,
        pupil: 0.0,
        board: Vec::new(),
        field: Vec::new(),
        sensor: Vec::new(),
        misc: [0.0; 4],
        spline: Vec::new(),
        board_centre: centre,
    })
}

fn gen_summary(f: &GenFit) -> ParamSummary {
    let p = &f.params;
    let mut extra = BTreeMap::new();
    for (i, n) in model::DIST_NAMES.iter().enumerate().skip(2) {
        extra.insert(n.to_string(), p.dist[i]);
    }
    extra.insert("dc_x".into(), p.dc[0]);
    extra.insert("dc_y".into(), p.dc[1]);
    extra.insert("depth_k1".into(), p.depth);
    extra.insert("pupil_eta".into(), p.pupil);
    for (i, b) in p.board.iter().enumerate() {
        extra.insert(format!("board{i}_mm"), *b);
    }
    for (i, b) in p.sensor.iter().enumerate() {
        extra.insert(format!("sensor{i}"), *b);
    }
    for (n, v) in ["e1", "e2", "sc_dx", "sc_dy"].iter().zip(p.misc) {
        extra.insert(n.to_string(), v);
    }
    if !p.spline.is_empty() {
        // Largest control displacement and the Lipschitz bound of the
        // field (largest neighbouring control difference over the knot
        // spacing, in px/px): the unprojection contracts when it is < 1.
        let cells = f.spline_cells;
        let (nx, ny) = (cells[0] + 3, cells[1] + 3);
        let n = nx * ny;
        let h = (f.size[0] / cells[0] as f64).min(f.size[1] / cells[1] as f64);
        let mut lip: f64 = 0.0;
        for comp in 0..2 {
            for j in 0..ny {
                for i in 0..nx {
                    let c = p.spline[comp * n + j * nx + i];
                    if i + 1 < nx {
                        lip = lip.max((p.spline[comp * n + j * nx + i + 1] - c).abs() / h);
                    }
                    if j + 1 < ny {
                        lip = lip.max((p.spline[comp * n + (j + 1) * nx + i] - c).abs() / h);
                    }
                }
            }
        }
        let max = p.spline.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        extra.insert("spline_max_px".into(), max);
        extra.insert("spline_lipschitz".into(), lip);
    }
    ParamSummary {
        fx: p.intr[0],
        fy: p.intr[1],
        cx: p.intr[2],
        cy: p.intr[3],
        tilt_x: p.tilt[0],
        tilt_y: p.tilt[1],
        k1: p.dist[0],
        k2: p.dist[1],
        extra,
        sigma: f.sigma.clone(),
        corr: f.corr.clone(),
        covariance: f.covariance.clone(),
        iters: f.iters,
        cost: f.cost,
    }
}

struct Eval {
    full: ParamSummary,
    train: Vec<PointResidual>,
    held: Vec<PointResidual>,
    folds: Vec<ParamSummary>,
}

fn evaluate(model: Model<'_>, cam: &CameraData, ctx: &CamCtx) -> Result<Eval> {
    let n = cam.dataset.num_views();
    let all: Vec<usize> = (0..n).collect();
    let views = |ids: &[usize]| -> Vec<View<NoMeta>> {
        ids.iter().map(|&i| cam.dataset.views[i].clone()).collect()
    };
    let base_poses = &ctx.base.params.cam_se3_target;
    let mut held = Vec::new();
    let mut folds = Vec::new();
    let (full, mut train) = match model {
        Model::Lib(v) => {
            let full = refine(
                &cam.dataset,
                &ctx.base.params,
                v,
                SolverBackend::TinySolver,
                3000,
            )?;
            let pts = residuals(
                &all,
                &cam.dataset.views,
                &full.params.camera,
                &full.params.cam_se3_target,
            )?;
            for ((train, test), fr) in ctx.folds.iter().zip(&ctx.fold_routes) {
                let fit = refine(
                    &subset(&cam.dataset, train)?,
                    &fr.params,
                    v,
                    SolverBackend::TinySolver,
                    3000,
                )?;
                folds.push(summarize(&fit));
                let init: Vec<Iso3> = test.iter().map(|&i| base_poses[i]).collect();
                let po = pose_only(&subset(&cam.dataset, test)?, &fit.params.camera, init)?;
                held.extend(residuals(
                    test,
                    &views(test),
                    &fit.params.camera,
                    &po.params.cam_se3_target,
                )?);
            }
            (summarize(&full), pts)
        }
        Model::Gen(g) => {
            let init = gen_init(&ctx.base, ctx.centre)?;
            let full = model::fit(
                g,
                &all,
                &views(&all),
                &init,
                base_poses,
                cam.size,
                Free::All,
                500,
            )?;
            for ((train, test), fr) in ctx.folds.iter().zip(&ctx.fold_routes) {
                let init = gen_init(fr, ctx.centre)?;
                let fit = model::fit(
                    g,
                    train,
                    &views(train),
                    &init,
                    &fr.params.cam_se3_target,
                    cam.size,
                    Free::All,
                    500,
                )?;
                folds.push(gen_summary(&fit));
                let init_poses: Vec<Iso3> = test.iter().map(|&i| base_poses[i]).collect();
                let po = model::fit(
                    g,
                    test,
                    &views(test),
                    &fit.params,
                    &init_poses,
                    cam.size,
                    Free::PosesOnly,
                    200,
                )?;
                held.extend(po.residuals);
            }
            let s = gen_summary(&full);
            (s, full.residuals)
        }
    };
    held.sort_by_key(|p| p.view);
    for p in train.iter_mut().chain(held.iter_mut()) {
        p.blur_px = cam.blur.get(&blur::key(p.observed_px)).copied();
    }
    Ok(Eval {
        full,
        train,
        held,
        folds,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;
    let part = |p: &str| args.parts.iter().any(|x| x == p);
    let cams = if part("synth") {
        synth_cams(&args.datasets)?
    } else {
        load(&args)?
    };

    let mut models: Vec<Model<'_>> = Vec::new();
    if part("lib") {
        models.extend(VARIANTS.iter().map(Model::Lib));
    }
    if part("gen") {
        models.extend(GEN_MODELS.iter().map(Model::Gen));
    }
    if !args.models.is_empty() && !part("target") {
        models.retain(|m| args.models.iter().any(|n| n == m.name()));
    }

    let mut study = Study::default();
    let mut floor_train: BTreeMap<&str, Vec<FloorCamera>> = BTreeMap::new();
    let mut floor_heldout: BTreeMap<&str, Vec<FloorCamera>> = BTreeMap::new();
    let mut dumps: BTreeMap<String, BTreeMap<String, Vec<PointResidual>>> = BTreeMap::new();
    let mut ctxs: Vec<CamCtx> = Vec::new();
    let mut transfer_lines: Vec<String> = Vec::new();

    if part("target") {
        target_study::run(&cams, args.folds, &args.out, &args.models)?;
    }
    if part("mc") {
        for l in synth::monte_carlo(40, 7)? {
            println!("{l}");
        }
    }
    let per_camera =
        !models.is_empty() || part("basin") || part("blur") || part("xcam") || part("mimic");
    let mut mimic_lines: Vec<String> = Vec::new();
    let mut mimic_maps: Vec<(String, Vec<Option<[f64; 2]>>)> = Vec::new();
    for cam in cams.iter().filter(|_| per_camera) {
        let n = cam.dataset.num_views();
        let blurred = cam.blur.len();
        eprintln!("== {} ({} views, {} blur entries)", cam.id, n, blurred);
        let base = route(&cam.dataset, &cam.seed)?;
        print_cov_labels_once(&base);

        if part("lib") {
            for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
                let t = refine(&cam.dataset, &base.params, &VARIANTS[0], backend, 3000)?;
                study.stopping.push(StoppingCheck {
                    camera: cam.id.clone(),
                    backend: format!("{backend:?}"),
                    route: summarize(&base),
                    tight: summarize(&t),
                });
            }
        }

        let ctx = make_ctx(cam, base, args.folds)?;

        for m in &models {
            let e = evaluate(*m, cam, &ctx)?;
            let (tr, tm) = rms_median(&e.train);
            let (hr, hm) = rms_median(&e.held);
            eprintln!(
                "  {:<24} train {:.4}/{:.4}  held-out {:.4}/{:.4}  iters {}  cov {}",
                m.name(),
                tr,
                tm,
                hr,
                hm,
                e.full.iters,
                e.full.covariance
            );
            floor_train
                .entry(m.name())
                .or_default()
                .push(floor_camera(&cam.id, cam.size, &e.train));
            floor_heldout
                .entry(m.name())
                .or_default()
                .push(floor_camera(&cam.id, cam.size, &e.held));
            let norms: Vec<f64> = e
                .held
                .iter()
                .map(|p| p.residual_px[0].hypot(p.residual_px[1]))
                .collect();
            let d = dumps.entry(cam.id.clone()).or_default();
            d.insert(format!("{}|train", m.name()), e.train);
            d.insert(format!("{}|heldout", m.name()), e.held);
            study.variants.push(VariantResult {
                variant: m.name().to_string(),
                camera: cam.id.clone(),
                full: e.full,
                train_rms: tr,
                train_median: tm,
                heldout_rms: hr,
                heldout_median: hm,
                heldout_mean: norms.iter().sum::<f64>() / norms.len().max(1) as f64,
                heldout_norms: norms,
                folds: e.folds,
            });
        }

        if part("basin") {
            study.basin.extend(basin(cam, &ctx)?);
        }
        if part("mimic") {
            let m = mimic::run(cam, &ctx)?;
            for l in &m.lines {
                eprintln!("  {l}");
            }
            mimic_lines.extend(m.lines);
            mimic_maps.push((cam.id.clone(), m.map));
        }
        if part("blur") {
            let field4 = GEN_MODELS
                .iter()
                .find(|g| g.name == "gen +field4")
                .context("field4")?;
            transfer_lines.extend(blur_transfer(cam, &ctx, &[&GEN_MODELS[0], field4])?);
        }
        ctxs.push(ctx);
    }
    if part("xcam") {
        let field4 = GEN_MODELS
            .iter()
            .find(|g| g.name == "gen +field4")
            .context("field4")?;
        cross_camera(&cams, &ctxs, field4)?;
    }
    if !mimic_lines.is_empty() {
        println!("\n## Mimic: candidates fitted to the noise-free field4 mapping");
        println!("camera|model|rms px|parameters|iters");
        for l in &mimic_lines {
            println!("{l}");
        }
        for (id, m) in &mimic_maps {
            mimic::print_map(id, m);
        }
        println!("\ncell-map correlation between cameras (route Δ, noise-free):");
        for (a, (ia, ma)) in mimic_maps.iter().enumerate() {
            let row: Vec<String> = mimic_maps
                .iter()
                .map(|(_, mb)| format!("{:+.2}", mimic::map_corr(ma, mb).0))
                .collect();
            println!(
                "  {:>5} {}",
                ia.rsplit('_').next().unwrap_or(""),
                row.join(" ")
            );
            let _ = a;
        }
    }
    if !transfer_lines.is_empty() {
        println!("\n## Blur transfer: camera|split|model|train->eval|rms|points");
        for l in &transfer_lines {
            println!("{l}");
        }
    }

    for (name, fc) in &floor_train {
        study
            .floor_train
            .insert(name.to_string(), floor::analyze(fc)?);
    }
    for (name, fc) in &floor_heldout {
        study
            .floor_heldout
            .insert(name.to_string(), floor::analyze(fc)?);
    }

    std::fs::write(
        args.out.join("study.json"),
        serde_json::to_string_pretty(&study)?,
    )?;
    std::fs::write(
        args.out.join("residuals.json"),
        serde_json::to_string(&dumps)?,
    )?;
    if !models.is_empty() {
        print_summary(&study, &models);
    }
    if let Some(first) = models
        .first()
        .filter(|_| cams.iter().any(|c| !c.blur.is_empty()))
    {
        analyze_blur_depth(&cams, &study, &dumps, first.name());
    }
    if let Some(m) = models.iter().find(|m| m.name() == "tp9 all") {
        analyze_blur_depth(&cams, &study, &dumps, m.name());
    }
    print_basin(&study.basin);
    Ok(())
}

/// The synthetic scenes named on the command line (`synth_<truth>_<seed>`):
/// `plain` (the route's model), `outside` (plus a smooth field outside every
/// parametric term).
fn synth_cams(names: &[String]) -> Result<Vec<CameraData>> {
    let plain = GenCfg::base("truth");
    let truth = synth::typical();
    names
        .iter()
        .map(|name| {
            let mut it = name.split('_').skip(1);
            let kind = it.next().unwrap_or("");
            let seed: u64 = it.next().unwrap_or("0").parse()?;
            match kind {
                "plain" => {
                    synth::scene(name, &plain, &truth, 20, (seed, seed), 0.07, &|_| [0.0; 2])
                }
                "outside" => synth::scene(
                    name,
                    &plain,
                    &truth,
                    20,
                    (seed, seed),
                    0.07,
                    &synth::outside_field,
                ),
                "spline" => synth::spline_scene(name, seed, seed).map(|(c, _, _)| c),
                other => bail!("unknown synthetic scene {other}"),
            }
        })
        .collect()
}

fn print_cov_labels_once(est: &PlanarIntrinsicsEstimate) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    if let Uncertainty::Available(c) = &est.uncertainty {
        eprintln!("covariance labels: {:?}", c.labels);
    }
}

// ───────────────────────── basin / reparameterization ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct BasinRow {
    camera: String,
    parameterization: String,
    axis: String,
    delta_rad: f64,
    cost_excess: f64,
    iters: usize,
    tilt_err: f64,
    pp_err: f64,
    corr: Option<f64>,
}

/// Covariance entry of two labels of a library estimate.
fn lib_cov(est: &PlanarIntrinsicsEstimate, a: &str, b: &str) -> Option<f64> {
    let c = est.uncertainty.covariance()?;
    let find = |n: &str| {
        c.labels
            .iter()
            .position(|l| l == n || l.ends_with(&format!(".{n}")))
    };
    let (i, j) = (find(a)?, find(b)?);
    Some(c.covariance[i][j])
}

fn basin(cam: &CameraData, ctx: &CamCtx) -> Result<Vec<BasinRow>> {
    let n = cam.dataset.num_views();
    let all: Vec<usize> = (0..n).collect();
    let views: Vec<View<NoMeta>> = cam.dataset.views.clone();
    let poses = &ctx.base.params.cam_se3_target;
    let init0 = gen_init(&ctx.base, ctx.centre)?;
    // The valley's slope at the optimum, from the covariance.
    let slope_y = lib_cov(&ctx.base, "cy", "tilt_x").unwrap_or(f64::NAN)
        / lib_cov(&ctx.base, "tilt_x", "tilt_x").unwrap_or(f64::NAN);
    let slope_x = lib_cov(&ctx.base, "cx", "tilt_y").unwrap_or(f64::NAN)
        / lib_cov(&ctx.base, "tilt_y", "tilt_y").unwrap_or(f64::NAN);
    eprintln!(
        "  valley slope dcy/dtilt_x = {:.1} px/rad ({:.3} fy), dcx/dtilt_y = {:.1} px/rad ({:.3} fx)",
        slope_y,
        slope_y / init0.intr[1],
        slope_x,
        slope_x / init0.intr[0]
    );
    let signs = (slope_x.signum(), slope_y.signum());
    let plain = GenCfg::base("plain");
    let reparam = GenCfg {
        reparam: Some(signs),
        ..GenCfg::base("reparam")
    };
    // Optimum of each parameterization from the route's solution.
    let opt = model::fit(
        &plain,
        &all,
        &views,
        &init0,
        poses,
        cam.size,
        Free::All,
        500,
    )?;
    let mut rows = Vec::new();
    for (cfg, label) in [(&plain, "c"), (&reparam, "c - s f tan(tilt)")] {
        let at_opt = model::fit(cfg, &all, &views, &init0, poses, cam.size, Free::All, 500)?;
        rows.push(BasinRow {
            camera: cam.id.clone(),
            parameterization: label.to_string(),
            axis: "optimum".into(),
            delta_rad: 0.0,
            cost_excess: at_opt.cost - opt.cost,
            iters: at_opt.iters,
            tilt_err: 0.0,
            pp_err: 0.0,
            corr: at_opt.corr.get("cy/tilt_x").copied(),
        });
        for (axis, ai) in [("tilt_x", 0usize), ("tilt_y", 1)] {
            for delta in [-0.15, -0.08, -0.04, -0.02, 0.02, 0.04, 0.08, 0.15] {
                let mut start = init0.clone();
                start.tilt[ai] += delta;
                let f = model::fit(cfg, &all, &views, &start, poses, cam.size, Free::All, 500)?;
                let pp = if ai == 0 { 3 } else { 2 };
                rows.push(BasinRow {
                    camera: cam.id.clone(),
                    parameterization: label.to_string(),
                    axis: axis.into(),
                    delta_rad: delta,
                    cost_excess: f.cost - opt.cost,
                    iters: f.iters,
                    tilt_err: f.params.tilt[ai] - opt.params.tilt[ai],
                    pp_err: f.params.intr[pp] - opt.params.intr[pp],
                    corr: None,
                });
            }
        }
    }
    Ok(rows)
}

fn print_basin(rows: &[BasinRow]) {
    if rows.is_empty() {
        return;
    }
    println!("\n## Basin: start tilt perturbed, everything else at the optimum");
    println!(
        "| parameterization | axis | delta | ok/total | mean iters | max |tilt err| | max cost excess |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|");
    let mut groups: BTreeMap<(String, String, i64), Vec<&BasinRow>> = BTreeMap::new();
    for r in rows {
        groups
            .entry((
                r.parameterization.clone(),
                r.axis.clone(),
                (r.delta_rad * 1000.0) as i64,
            ))
            .or_default()
            .push(r);
    }
    for ((p, a, d), g) in &groups {
        let ok = g
            .iter()
            .filter(|r| {
                r.cost_excess.abs() < 1e-4 * (1.0 + r.cost_excess.abs()) && r.tilt_err.abs() < 1e-4
            })
            .count();
        let it = g.iter().map(|r| r.iters as f64).sum::<f64>() / g.len() as f64;
        let te = g.iter().map(|r| r.tilt_err.abs()).fold(0.0, f64::max);
        let ce = g.iter().map(|r| r.cost_excess).fold(f64::MIN, f64::max);
        println!(
            "| {p} | {a} | {:+.3} | {ok}/{} | {it:.1} | {te:.2e} | {ce:.3e} |",
            *d as f64 / 1000.0,
            g.len()
        );
    }
    let corr: Vec<String> = rows
        .iter()
        .filter(|r| r.axis == "optimum")
        .map(|r| {
            format!(
                "{}:{:?}",
                r.parameterization,
                r.corr.map(|c| (c * 1000.0).round() / 1000.0)
            )
        })
        .collect();
    println!("corr(cy, tilt_x) at the optimum: {}", corr.join(" "));
}

// ───────────────────────── blur and depth ─────────────────────────

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    (m, s)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (ma, _) = mean_std(a);
    let (mb, _) = mean_std(b);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

/// Least squares of stacked 2-vector residuals on 2-vector regressors:
/// coefficients and the fraction of the residual sum of squares explained.
fn regress(rows: &[([f64; 2], Vec<[f64; 2]>)]) -> (Vec<f64>, f64) {
    let k = rows.first().map_or(0, |r| r.1.len());
    let mut ata = nalgebra::DMatrix::<f64>::zeros(k, k);
    let mut atb = nalgebra::DVector::<f64>::zeros(k);
    let mut tss = 0.0;
    for (y, x) in rows {
        for c in 0..2 {
            tss += y[c] * y[c];
            for i in 0..k {
                atb[i] += x[i][c] * y[c];
                for j in 0..k {
                    ata[(i, j)] += x[i][c] * x[j][c];
                }
            }
        }
    }
    let Some(beta) = ata.clone().cholesky().map(|c| c.solve(&atb)) else {
        return (vec![f64::NAN; k], f64::NAN);
    };
    let mut rss = 0.0;
    for (y, x) in rows {
        for c in 0..2 {
            let pred: f64 = (0..k).map(|i| beta[i] * x[i][c]).sum();
            rss += (y[c] - pred).powi(2);
        }
    }
    (beta.iter().copied().collect(), 1.0 - rss / tss)
}

fn analyze_blur_depth(
    cams: &[CameraData],
    study: &Study,
    dumps: &BTreeMap<String, BTreeMap<String, Vec<PointResidual>>>,
    model: &str,
) {
    println!("\n## Blur and depth: model `{model}`, held-out residuals");
    let mut all_rows: Vec<([f64; 2], Vec<[f64; 2]>)> = Vec::new();
    let mut blur_bins: Vec<(f64, f64)> = Vec::new();
    let mut field_pairs = (Vec::new(), Vec::new());
    let mut random_pairs = (Vec::new(), Vec::new());
    let mut field_rms = (0.0, 0.0, 0usize);
    let mut per_view = Vec::new();
    for cam in cams {
        let Some(pts) = dumps
            .get(&cam.id)
            .and_then(|d| d.get(&format!("{model}|heldout")))
        else {
            continue;
        };
        let Some(vr) = study
            .variants
            .iter()
            .find(|v| v.camera == cam.id && v.variant == model)
        else {
            continue;
        };
        let p = &vr.full;
        let (w, h) = (f64::from(cam.size[0]), f64::from(cam.size[1]));
        let with_blur: Vec<&PointResidual> = pts.iter().filter(|q| q.blur_px.is_some()).collect();
        let mut blurs: Vec<f64> = with_blur.iter().map(|q| q.blur_px.unwrap_or(0.0)).collect();
        blurs.sort_by(f64::total_cmp);
        let med_blur = blurs.get(blurs.len() / 2).copied().unwrap_or(f64::NAN);
        let zs: Vec<f64> = pts.iter().map(|q| q.cam_mm[2]).collect();
        let zbar = zs.iter().sum::<f64>() / zs.len() as f64;
        for q in &with_blur {
            let r = q.residual_px;
            blur_bins.push((q.blur_px.unwrap_or(0.0), r[0].hypot(r[1])));
        }
        // Camera-fixed field of sharp and blurred markers, and of a random
        // split, per 6x4 cell.
        let cell = |q: &PointResidual| {
            let c = ((q.observed_px[0] / w * 6.0) as usize).min(5);
            let r = ((q.observed_px[1] / h * 4.0) as usize).min(3);
            r * 6 + c
        };
        let mut sums = vec![[0.0f64; 3]; 24 * 4];
        for (i, q) in with_blur.iter().enumerate() {
            let sharp = q.blur_px.unwrap_or(0.0) <= med_blur;
            let parity = i % 2 == 0;
            for (slot, on) in [(0usize, sharp), (1, !sharp), (2, parity), (3, !parity)] {
                if on {
                    let s = &mut sums[cell(q) * 4 + slot];
                    s[0] += q.residual_px[0];
                    s[1] += q.residual_px[1];
                    s[2] += 1.0;
                }
            }
        }
        for c in 0..24 {
            let get = |slot: usize| {
                let s = sums[c * 4 + slot];
                (s[2] >= 15.0).then(|| [s[0] / s[2], s[1] / s[2]])
            };
            if let (Some(a), Some(b)) = (get(0), get(1)) {
                field_pairs.0.extend(a);
                field_pairs.1.extend(b);
                field_rms.0 += a[0] * a[0] + a[1] * a[1];
                field_rms.1 += b[0] * b[0] + b[1] * b[1];
                field_rms.2 += 1;
            }
            if let (Some(a), Some(b)) = (get(2), get(3)) {
                random_pairs.0.extend(a);
                random_pairs.1.extend(b);
            }
        }
        // Regressors, per view.
        let mut by_view: BTreeMap<usize, Vec<&PointResidual>> = BTreeMap::new();
        for q in pts {
            by_view.entry(q.view).or_default().push(q);
        }
        for (view, vp) in &by_view {
            // Quadratic blur surface of the view.
            let bp: Vec<&&PointResidual> = vp.iter().filter(|q| q.blur_px.is_some()).collect();
            let grad = if bp.len() >= 12 {
                let rows: Vec<[f64; 6]> = bp
                    .iter()
                    .map(|q| {
                        let (u, v) = (q.observed_px[0] / w - 0.5, q.observed_px[1] / h - 0.5);
                        [1.0, u, v, u * u, u * v, v * v]
                    })
                    .collect();
                let a = nalgebra::DMatrix::from_fn(rows.len(), 6, |i, j| rows[i][j]);
                let b = nalgebra::DVector::from_iterator(
                    bp.len(),
                    bp.iter().map(|q| q.blur_px.unwrap_or(0.0)),
                );
                (a.transpose() * &a)
                    .cholesky()
                    .map(|c| c.solve(&(a.transpose() * b)))
            } else {
                None
            };
            let mut basis_rows = Vec::new();
            for q in vp {
                let d = [q.observed_px[0] - p.cx, q.observed_px[1] - p.cy];
                let rn2 = (d[0] * d[0] + d[1] * d[1]) / (p.fx * p.fy);
                let dz = zbar / q.cam_mm[2] - 1.0;
                let depth = [dz * rn2 * d[0], dz * rn2 * d[1]];
                let radial_unit = {
                    let n = d[0].hypot(d[1]).max(1e-9);
                    [d[0] / n, d[1] / n]
                };
                let (gb, sb) = match (&grad, q.blur_px) {
                    (Some(c), Some(b)) => {
                        let (u, v) = (q.observed_px[0] / w - 0.5, q.observed_px[1] / h - 0.5);
                        let gu = (c[1] + 2.0 * c[3] * u + c[4] * v) / w;
                        let gv = (c[2] + c[4] * u + 2.0 * c[5] * v) / h;
                        (
                            [gu * 100.0, gv * 100.0],
                            [
                                (b - med_blur) * radial_unit[0],
                                (b - med_blur) * radial_unit[1],
                            ],
                        )
                    }
                    _ => continue,
                };
                all_rows.push((q.residual_px, vec![gb, sb, depth]));
                basis_rows.push((q.residual_px, vec![[rn2 * d[0], rn2 * d[1]]]));
            }
            if basis_rows.len() >= 12 {
                let (beta, _) = regress(&basis_rows);
                let zv = vp.iter().map(|q| q.cam_mm[2]).sum::<f64>() / vp.len() as f64;
                per_view.push((cam.id.clone(), *view, zv, zbar, beta[0]));
            }
        }
    }
    // Blur bins.
    blur_bins.sort_by(|a, b| a.0.total_cmp(&b.0));
    let nb = 5;
    let mut line = String::from("blur quintiles (sigma: rms px):");
    for k in 0..nb {
        let s = &blur_bins[k * blur_bins.len() / nb..(k + 1) * blur_bins.len() / nb];
        let mb = s.iter().map(|x| x.0).sum::<f64>() / s.len() as f64;
        let r = (s.iter().map(|x| x.1 * x.1).sum::<f64>() / s.len() as f64).sqrt();
        line.push_str(&format!("  {mb:.2}: {r:.4}"));
    }
    println!("{line}");
    let n = field_rms.2.max(1) as f64;
    println!(
        "camera-fixed field, sharp vs blurred markers (cells with >=15 each): corr {:.3} (random split {:.3}), rms sharp {:.4} blurred {:.4} px over {} cells",
        pearson(&field_pairs.0, &field_pairs.1),
        pearson(&random_pairs.0, &random_pairs.1),
        (field_rms.0 / n).sqrt(),
        (field_rms.1 / n).sqrt(),
        field_rms.2
    );
    let names = [
        "blur gradient (px per 100 px)",
        "(blur - median) radial",
        "depth-dependent k1",
    ];
    for (i, name) in names.iter().enumerate() {
        let rows: Vec<_> = all_rows.iter().map(|(y, x)| (*y, vec![x[i]])).collect();
        let (b, r2) = regress(&rows);
        println!(
            "  {name:<32} coef {:+.4}  explains {:.2}% of the residual sum of squares",
            b[0],
            100.0 * r2
        );
    }
    let (b, r2) = regress(&all_rows);
    println!(
        "  joint: coefs {:?} explains {:.2}%",
        b.iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>(),
        100.0 * r2
    );
    let inv: Vec<f64> = per_view.iter().map(|v| v.3 / v.2 - 1.0).collect();
    let a: Vec<f64> = per_view.iter().map(|v| v.4).collect();
    println!(
        "per-view residual radial coefficient (px, on r^2·d) vs zbar/z - 1: corr {:.3} over {} views; range of view depth {:.0}..{:.0} mm",
        pearson(&inv, &a),
        per_view.len(),
        per_view.iter().map(|v| v.2).fold(f64::MAX, f64::min),
        per_view.iter().map(|v| v.2).fold(f64::MIN, f64::max),
    );
}

fn print_heldout_table(study: &Study, models: &[Model<'_>]) {
    println!("\n## Held-out table: pooled rms / mean / median px, per-camera mean px");
    for m in models {
        let rows: Vec<_> = study
            .variants
            .iter()
            .filter(|r| r.variant == m.name())
            .collect();
        let mut all: Vec<f64> = rows
            .iter()
            .flat_map(|r| r.heldout_norms.iter().copied())
            .collect();
        let n = all.len();
        let ms: f64 = all.iter().map(|v| v * v).sum();
        let mean: f64 = all.iter().sum();
        all.sort_by(f64::total_cmp);
        let median = all.get(n / 2).copied().unwrap_or(f64::NAN);
        let cams: Vec<String> = rows
            .iter()
            .map(|r| format!("{:.3}", r.heldout_mean))
            .collect();
        println!(
            "| {} | {:.3} / {:.3} / {:.3} | {} |",
            m.name(),
            (ms / n.max(1) as f64).sqrt(),
            mean / n.max(1) as f64,
            median,
            cams.join(" | ")
        );
    }
}

fn print_summary(study: &Study, models: &[Model<'_>]) {
    print_heldout_table(study, models);
    println!("## Variants (pooled floor: train | held-out)");
    println!("| variant | train rms | med | gap | camfield | held rms | med | gap | camfield |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    for m in models {
        let t = &study.floor_train[m.name()].pooled;
        let h = &study.floor_heldout[m.name()].pooled;
        println!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.1}% | {:.4} | {:.4} | {:.4} | {:.1}% |",
            m.name(),
            t.calibrated.rms_px,
            t.calibrated.median_px,
            t.calibrated.smooth_gap_rms_px,
            100.0 * t.camera_field.fraction,
            h.calibrated.rms_px,
            h.calibrated.median_px,
            h.calibrated.smooth_gap_rms_px,
            100.0 * h.camera_field.fraction,
        );
    }
    if let Some(m) = models.first() {
        let smooth = &study.floor_train[m.name()].pooled.quartic;
        println!(
            "smooth-fit floor: rms {:.4} median {:.4}",
            smooth.rms_px, smooth.median_px
        );
    }

    println!("\n## Per camera (all-point rms/median: train | held-out)");
    for m in models {
        let cells: Vec<String> = study
            .variants
            .iter()
            .filter(|r| r.variant == m.name())
            .map(|r| {
                format!(
                    "{:.3}/{:.3}|{:.3}/{:.3}",
                    r.train_rms, r.train_median, r.heldout_rms, r.heldout_median
                )
            })
            .collect();
        println!("{:<24} {}", m.name(), cells.join("  "));
    }

    println!("\n## Parameters across cameras (mean ± sd) and within-camera fold sd");
    for m in models {
        let rows: Vec<_> = study
            .variants
            .iter()
            .filter(|r| r.variant == m.name())
            .collect();
        let pick = |f: &dyn Fn(&ParamSummary) -> f64| -> String {
            let across: Vec<f64> = rows.iter().map(|r| f(&r.full)).collect();
            let (mu, s) = mean_std(&across);
            let within: Vec<f64> = rows
                .iter()
                .map(|r| mean_std(&r.folds.iter().map(f).collect::<Vec<_>>()).1)
                .collect();
            let w = within.iter().sum::<f64>() / within.len() as f64;
            format!("{mu:.4}±{s:.4} (fold sd {w:.4})")
        };
        println!(
            "{:<24} fx {}  fy-fx {}  cx {}  cy {}  tilt_x {}  k1 {}",
            m.name(),
            pick(&|p| p.fx),
            pick(&|p| p.fy - p.fx),
            pick(&|p| p.cx),
            pick(&|p| p.cy),
            pick(&|p| p.tilt_x),
            pick(&|p| p.k1),
        );
        let covs: Vec<String> = rows
            .iter()
            .map(|r| format!("{}:{}", r.full.iters, r.full.covariance))
            .collect();
        println!("{:<24} iters:cov {}", "", covs.join(" "));
    }

    if !study.stopping.is_empty() {
        println!("\n## Stopping rule: route vs tight refine (same model)");
    }
    for s in &study.stopping {
        println!(
            "{} {:<11} cost {:.6} -> {:.6} ({} -> {} iters)  dfx {:+.3} dcx {:+.3} dcy {:+.3} dtilt_x {:+.6} dk1 {:+.6}",
            s.camera,
            s.backend,
            s.route.cost,
            s.tight.cost,
            s.route.iters,
            s.tight.iters,
            s.tight.fx - s.route.fx,
            s.tight.cx - s.route.cx,
            s.tight.cy - s.route.cy,
            s.tight.tilt_x - s.route.tilt_x,
            s.tight.k1 - s.route.k1,
        );
    }
}

// ───────────────────────── transfer tests ─────────────────────────

/// Which points of a view a split keeps: by observed position and point
/// index.
type PointFilter<'a> = &'a dyn Fn([f64; 2], usize) -> bool;

/// The views restricted to points `keep` accepts (by observed position and
/// point index); views left with fewer than 8 points are dropped. Returns
/// the kept view ids and views.
fn split_views(
    cam: &CameraData,
    keep: &dyn Fn([f64; 2], usize) -> bool,
) -> Result<(Vec<usize>, Vec<View<NoMeta>>)> {
    let mut ids = Vec::new();
    let mut views = Vec::new();
    for (i, v) in cam.dataset.views.iter().enumerate() {
        let (mut p3, mut p2) = (Vec::new(), Vec::new());
        for (k, (a, b)) in v.obs.points_3d.iter().zip(&v.obs.points_2d).enumerate() {
            if keep([b.x, b.y], k) {
                p3.push(*a);
                p2.push(*b);
            }
        }
        if p3.len() >= 8 {
            ids.push(i);
            views.push(View {
                obs: vision_calibration_core::CorrespondenceView::new(p3, p2)?,
                meta: NoMeta,
            });
        }
    }
    Ok((ids, views))
}

/// A quadratic surface of the blur over the image, pooled over views: the
/// blur a marker at that position typically has.
fn blur_surface(cam: &CameraData) -> Option<nalgebra::DVector<f64>> {
    let (w, h) = (f64::from(cam.size[0]), f64::from(cam.size[1]));
    let mut ata = nalgebra::DMatrix::<f64>::zeros(6, 6);
    let mut atb = nalgebra::DVector::<f64>::zeros(6);
    for v in &cam.dataset.views {
        for b in &v.obs.points_2d {
            let Some(&s) = cam.blur.get(&blur::key([b.x, b.y])) else {
                continue;
            };
            let r = surface_row(b.x / w - 0.5, b.y / h - 0.5);
            for i in 0..6 {
                atb[i] += r[i] * s;
                for j in 0..6 {
                    ata[(i, j)] += r[i] * r[j];
                }
            }
        }
    }
    ata.cholesky().map(|c| c.solve(&atb))
}

fn surface_row(u: f64, v: f64) -> [f64; 6] {
    [1.0, u, v, u * u, u * v, v * v]
}

/// Train on one half of the markers, evaluate on either half (poses
/// refitted, camera held). Halves: by blur (`abs`), by blur relative to the
/// blur typical of the marker's image position (`rel`, same coverage), and
/// by point parity (`rand`, the control).
fn blur_transfer(cam: &CameraData, ctx: &CamCtx, models: &[&GenCfg]) -> Result<Vec<String>> {
    let (w, h) = (f64::from(cam.size[0]), f64::from(cam.size[1]));
    let surf = blur_surface(cam).context("blur surface")?;
    let typical = |px: [f64; 2]| {
        let r = surface_row(px[0] / w - 0.5, px[1] / h - 0.5);
        (0..6).map(|i| r[i] * surf[i]).sum::<f64>()
    };
    let blur_of = |px: [f64; 2]| cam.blur.get(&blur::key(px)).copied();
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let mut abs_v = Vec::new();
    let mut rel_v = Vec::new();
    for v in &cam.dataset.views {
        for b in &v.obs.points_2d {
            if let Some(s) = blur_of([b.x, b.y]) {
                abs_v.push(s);
                rel_v.push(s - typical([b.x, b.y]));
            }
        }
    }
    let (abs_med, rel_med) = (median(abs_v), median(rel_v));
    let abs_lo = |px: [f64; 2], _: usize| blur_of(px).is_some_and(|s| s <= abs_med);
    let abs_hi = |px: [f64; 2], _: usize| blur_of(px).is_some_and(|s| s > abs_med);
    let rel_lo = |px: [f64; 2], _: usize| blur_of(px).is_some_and(|s| s - typical(px) <= rel_med);
    let rel_hi = |px: [f64; 2], _: usize| blur_of(px).is_some_and(|s| s - typical(px) > rel_med);
    let even = |px: [f64; 2], k: usize| blur_of(px).is_some() && k.is_multiple_of(2);
    let odd = |px: [f64; 2], k: usize| blur_of(px).is_some() && !k.is_multiple_of(2);
    let splits: [(&str, PointFilter<'_>, PointFilter<'_>); 3] = [
        ("abs", &abs_lo, &abs_hi),
        ("rel", &rel_lo, &rel_hi),
        ("rand", &even, &odd),
    ];
    let poses = &ctx.base.params.cam_se3_target;
    let init = gen_init(&ctx.base, ctx.centre)?;
    let mut lines = Vec::new();
    for (split, lo, hi) in splits {
        let sets = [("lo", split_views(cam, lo)?), ("hi", split_views(cam, hi)?)];
        for cfg in models {
            for (tn, (tids, tviews)) in &sets {
                let tp: Vec<Iso3> = tids.iter().map(|&i| poses[i]).collect();
                let fit = model::fit(cfg, tids, tviews, &init, &tp, cam.size, Free::All, 500)?;
                for (en, (eids, eviews)) in &sets {
                    let ep: Vec<Iso3> = eids.iter().map(|&i| poses[i]).collect();
                    let ev = model::fit(
                        cfg,
                        eids,
                        eviews,
                        &fit.params,
                        &ep,
                        cam.size,
                        Free::PosesOnly,
                        200,
                    )?;
                    let (r, _) = rms_median(&ev.residuals);
                    lines.push(format!(
                        "{}|{}|{}|{}->{}|{:.4}|{}",
                        cam.id,
                        split,
                        cfg.name,
                        tn,
                        en,
                        r,
                        ev.residuals.len()
                    ));
                }
            }
        }
    }
    Ok(lines)
}

/// The free field of one camera, held, on another: is it a property of the
/// lens design or of the unit?
fn cross_camera(cams: &[CameraData], ctxs: &[CamCtx], cfg: &GenCfg) -> Result<()> {
    let mut fits = Vec::new();
    for (cam, ctx) in cams.iter().zip(ctxs) {
        let n = cam.dataset.num_views();
        let all: Vec<usize> = (0..n).collect();
        let init = gen_init(&ctx.base, ctx.centre)?;
        fits.push(model::fit(
            cfg,
            &all,
            &cam.dataset.views,
            &init,
            &ctx.base.params.cam_se3_target,
            cam.size,
            Free::All,
            500,
        )?);
    }
    println!(
        "\n## Cross-camera: `{}` field of camera A held on camera B (train rms px)",
        cfg.name
    );
    println!("rows A, columns B; diagonal = own fit; last column = baseline on B");
    let base_cfg = GenCfg::base("base");
    for (a, fa) in fits.iter().enumerate() {
        let mut row = format!("{:>5}", cams[a].id.rsplit('_').next().unwrap_or(""));
        for (b, (cam, ctx)) in cams.iter().zip(ctxs).enumerate() {
            let n = cam.dataset.num_views();
            let all: Vec<usize> = (0..n).collect();
            let r = if a == b {
                rms_median(&fits[b].residuals).0
            } else {
                let mut init = gen_init(&ctx.base, ctx.centre)?;
                init.dist = fa.params.dist;
                init.field = fa.params.field.clone();
                let f = model::fit(
                    cfg,
                    &all,
                    &cam.dataset.views,
                    &init,
                    &ctx.base.params.cam_se3_target,
                    cam.size,
                    Free::AllButField,
                    500,
                )?;
                rms_median(&f.residuals).0
            };
            row.push_str(&format!(" {r:.4}"));
        }
        println!("{row}");
    }
    let mut row = String::from(" base");
    for (cam, ctx) in cams.iter().zip(ctxs) {
        let n = cam.dataset.num_views();
        let all: Vec<usize> = (0..n).collect();
        let init = gen_init(&ctx.base, ctx.centre)?;
        let f = model::fit(
            &base_cfg,
            &all,
            &cam.dataset.views,
            &init,
            &ctx.base.params.cam_se3_target,
            cam.size,
            Free::All,
            500,
        )?;
        row.push_str(&format!(" {:.4}", rms_median(&f.residuals).0));
    }
    println!("{row}");
    Ok(())
}
