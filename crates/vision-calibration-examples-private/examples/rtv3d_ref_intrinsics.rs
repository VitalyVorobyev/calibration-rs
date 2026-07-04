//! Per-camera Scheimpflug **intrinsics** calibration on the `rtv3d_ref`
//! reference dataset from a *coarse, user-provided* seed — the supported
//! workflow (ADR 0022).
//!
//! Companion to `rtv3d_ref_rig` (full from-scratch rig) and `rtv3d_ref_reproj`
//! (frozen-intrinsics parity). This harness exercises the recommended path for
//! Scheimpflug intrinsics: the coarse prior comes from the dataset's device
//! spec (`spec.json`, ADR 0023) — lens focal + pixel pitch → `fx = fy`, and
//! the Scheimpflug mount angle (≈−5°) → the tilt seed — and bundle adjustment
//! refines it. Nothing from `artifacts.json` is seeded; the oracle is read
//! only for the final comparison.
//!
//! Why not from scratch? On this data (strong radial distortion, k1≈−0.43, plus
//! a ≈−5° tilt) Zhang-from-scratch underestimates the focal and the solve settles
//! into a wrong tilt/focal basin — see ADR 0022 and the P6 diagnosis. A coarse
//! seed removes that fragility.
//!
//! **Acceptance gate:** every camera must reach mean reprojection ≤ 0.5 px. The
//! process exits non-zero if any camera misses it (a reprojection error > 0.5 px
//! is never a success).
//!
//! Run:
//! `cargo run --release --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! rtv3d_ref_intrinsics`
//!
//! Env:
//! - `RTV3D_REF_DATA_DIR` (default `privatedata/rtv3d_ref`). Must contain a
//!   `spec.json` device spec (gitignored, like the rest of `privatedata/`).
//! - `RTV3D_REF_MAXITERS` (default `120`).
//! - `Q4_DISTORTION_SWEEP=1` — after the gated run, re-run the same seeded
//!   route once per distortion model (BrownConrady5, Rational8, ThinPrism9,
//!   Division1) and print a per-camera / per-model mean-reprojection table.
//!   Informational only: it never changes the output of the default run nor
//!   the process exit code.

use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;
use std::time::Instant;

use vision_calibration::device_seed::{DEVICE_SPEC_FILENAME, DeviceSpec, scheimpflug_seed};
use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsProblem,
    step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration_core::{
    BrownConrady5, FxFyCxCySkew, IntrinsicsParams, NoMeta, PlanarDataset, ScheimpflugParams,
    SensorParams, View,
};
use vision_calibration_examples_private::{
    RefArtifacts, RefIntrinsic, detect_target, load_gray, load_poses, load_ref_artifacts,
    split_horizontal, split_opencv_distortion,
};
use vision_calibration_optim::{DistortionKind, RobustLoss};

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 130;
const BOARD_COLS: u32 = 130;
const CELL_SIZE_MM: f64 = 5.0;
const GATE_PX: f64 = 0.5;

/// Recovered + reference summary for one camera.
struct CamResult {
    intr: FxFyCxCySkew<f64>,
    dist: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    mean_reproj: f64,
    corners: usize,
    views: usize,
}

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("RTV3D_REF_DATA_DIR").unwrap_or_else(|_| "privatedata/rtv3d_ref".to_string()),
    );
    let max_iters: usize = std::env::var("RTV3D_REF_MAXITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    println!("data dir = {}", data_dir.display());

    // Device spec (ADR 0023): lens focal + pixel pitch + mount tilt → seed.
    let spec_path = data_dir.join(DEVICE_SPEC_FILENAME);
    let spec = DeviceSpec::from_path(&spec_path)
        .with_context(|| format!("load device spec {}", spec_path.display()))?;
    if spec.cameras.len() != NUM_CAMERAS {
        return Err(anyhow!(
            "device spec has {} cameras, expected {NUM_CAMERAS}",
            spec.cameras.len()
        ));
    }
    {
        let seed0 = scheimpflug_seed(&spec, "cam0")?;
        let k = seed0.intrinsics.expect("spec-derived intrinsics");
        let s = seed0.sensor.expect("spec-derived sensor");
        println!(
            "spec seed (cam0): fx=fy={:.1}, pp=({:.0},{:.0}), tilt_x={:.4} rad, distortion=0",
            k.fx, k.cx, k.cy, s.tilt_x
        );
    }

    let art =
        load_ref_artifacts(&data_dir.join("artifacts.json")).context("load oracle artifacts")?;
    if art.num_cameras != NUM_CAMERAS || art.intrinsic.len() != NUM_CAMERAS {
        return Err(anyhow!(
            "expected {NUM_CAMERAS} cameras, got num_cameras={} intrinsics={}",
            art.num_cameras,
            art.intrinsic.len()
        ));
    }
    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!(
        "loaded {} poses, {} oracle cameras",
        poses.len(),
        art.num_cameras
    );

    // ── Detect puzzle_board in every camera tile of every pose ───────────────
    let t_det = Instant::now();
    let mut per_cam_views: Vec<Vec<View<NoMeta>>> = vec![Vec::new(); NUM_CAMERAS];
    for (i, pose) in poses.iter().enumerate() {
        let img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("pose {i} target"))?;
        let tiles = split_horizontal(&img, NUM_CAMERAS);
        for (c, tile) in tiles.iter().enumerate() {
            if let Ok(v) = detect_target(tile, BOARD_ROWS, BOARD_COLS, CELL_SIZE_MM) {
                per_cam_views[c].push(View::without_meta(v));
            }
        }
    }
    println!("detect: {:.2?}", t_det.elapsed());

    // Opt-in per-model sweep re-runs the seeded route on the same detections;
    // the gated loop below consumes `per_cam_views`, so snapshot them first
    // (only when the sweep is enabled — the default path is untouched).
    let sweep_views: Option<Vec<Vec<View<NoMeta>>>> =
        sweep_enabled().then(|| per_cam_views.clone());

    // ── Per-camera seeded intrinsics calibration ─────────────────────────────
    let t0 = Instant::now();
    let mut results: Vec<CamResult> = Vec::with_capacity(NUM_CAMERAS);
    for (c, views) in per_cam_views.into_iter().enumerate() {
        let num_views = views.len();
        let dataset = PlanarDataset::new(views)
            .with_context(|| format!("camera {c}: build planar dataset"))?;
        let corners = dataset.views.iter().map(|v| v.obs.points_2d.len()).sum();

        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session.set_input(dataset)?;

        let mut config = ScheimpflugIntrinsicsConfig::default();
        config.max_iters = max_iters;
        // Match the oracle lens config: k1,k2 free; k3 + tangential fixed (the
        // default `radial_only`); both Scheimpflug tilts free; robust loss to
        // down-weight the detector outlier tail.
        config.fix_scheimpflug = ScheimpflugFixMask {
            tilt_x: false,
            tilt_y: false,
        };
        config.robust_loss = RobustLoss::Huber { scale: 1.0 };
        session.set_config(config)?;

        // Coarse "datasheet" prior derived from the device spec (ADR 0023):
        // fx = fy from lens focal / pixel pitch, principal point at the tile
        // center, nominal mount tilt. Distortion + poses auto. The seeded
        // tilt is trusted directly (ADR 0022).
        let seed = scheimpflug_seed(&spec, &format!("cam{c}"))
            .with_context(|| format!("camera {c}: derive seed from device spec"))?;
        step_init_with_seed(&mut session, seed, None)
            .with_context(|| format!("camera {c}: seeded init"))?;

        step_optimize(&mut session, None).with_context(|| format!("camera {c}: optimize"))?;

        let out = session.output().expect("output after optimize");
        let intr = match &out.params.camera.intrinsics {
            IntrinsicsParams::FxFyCxCySkew { params } => *params,
        };
        let dist = match &out.params.camera.distortion {
            vision_calibration_core::DistortionParams::BrownConrady5 { params } => *params,
            other => {
                return Err(anyhow!(
                    "camera {c}: unexpected distortion params: {other:?}"
                ));
            }
        };
        let sensor = match &out.params.camera.sensor {
            SensorParams::Scheimpflug { params } => *params,
            other => return Err(anyhow!("camera {c}: unexpected sensor params: {other:?}")),
        };
        results.push(CamResult {
            intr,
            dist,
            sensor,
            mean_reproj: out.mean_reproj_error,
            corners,
            views: num_views,
        });
    }
    println!("calibrate (seeded) total: {:.2?}\n", t0.elapsed());

    report(&art, &results);

    // ── Opt-in distortion-model sweep (informational; never gates) ───────────
    if let Some(sweep_views) = &sweep_views {
        run_distortion_sweep(&spec, sweep_views, max_iters);
    }

    // ── Hard gate: every camera ≤ 0.5 px ─────────────────────────────────────
    let failed: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .filter(|(_, r)| r.mean_reproj > GATE_PX || r.mean_reproj.is_nan())
        .map(|(i, r)| (i, r.mean_reproj))
        .collect();
    if failed.is_empty() {
        println!("\nGATE PASS: all {NUM_CAMERAS} cameras ≤ {GATE_PX} px mean reprojection.");
        Ok(())
    } else {
        let list = failed
            .iter()
            .map(|(i, e)| format!("cam {i}: {e:.4} px"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(anyhow!(
            "GATE FAIL: {} of {NUM_CAMERAS} cameras exceed {GATE_PX} px ({list}). \
             A reprojection error > 0.5 px is not a success — investigate (focal prior, \
             richer distortion, detector tail), do not relax the gate.",
            failed.len()
        ))
    }
}

/// Whether the opt-in per-distortion-model sweep is requested.
fn sweep_enabled() -> bool {
    std::env::var("Q4_DISTORTION_SWEEP").as_deref() == Ok("1")
}

/// Run the seeded Scheimpflug intrinsics solve for one camera under a given
/// distortion model and return its mean reprojection error (px).
///
/// Mirrors the gated run's config exactly — same seed, same fix masks, same
/// robust loss — changing only `distortion_model`. Non-BC5 models produce a
/// different `DistortionParams` variant on output, but the sweep only reads
/// the scalar reprojection metric, so no variant matching is needed.
fn sweep_solve_camera(
    spec: &DeviceSpec,
    cam: usize,
    views: Vec<View<NoMeta>>,
    model: DistortionKind,
    max_iters: usize,
) -> Result<f64> {
    let dataset = PlanarDataset::new(views)?;
    let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.max_iters = max_iters;
    config.fix_scheimpflug = ScheimpflugFixMask {
        tilt_x: false,
        tilt_y: false,
    };
    config.robust_loss = RobustLoss::Huber { scale: 1.0 };
    config.distortion_model = model;
    session.set_config(config)?;

    let seed = scheimpflug_seed(spec, &format!("cam{cam}"))?;
    step_init_with_seed(&mut session, seed, None)?;
    step_optimize(&mut session, None)?;

    let out = session.output().expect("output after optimize");
    Ok(out.mean_reproj_error)
}

/// Median of the finite values (proper even-length average); `None` if empty.
fn median(mut xs: Vec<f64>) -> Option<f64> {
    xs.retain(|v| v.is_finite());
    if xs.is_empty() {
        return None;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = xs.len();
    Some(if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    })
}

/// Opt-in per-distortion-model sweep (env `Q4_DISTORTION_SWEEP=1`).
///
/// For each model it re-runs the same seeded route per camera and reports the
/// mean reprojection error (the gated metric), plus a per-model median across
/// cameras. Purely informational — it never affects the process exit code. A
/// solve that errors is reported as `ERR`; a non-finite reprojection as
/// `DIVERGED`.
fn run_distortion_sweep(spec: &DeviceSpec, per_cam_views: &[Vec<View<NoMeta>>], max_iters: usize) {
    let models = [
        DistortionKind::BrownConrady5,
        DistortionKind::Rational8,
        DistortionKind::ThinPrism9,
        DistortionKind::Division1,
    ];
    println!(
        "\n── Q4 distortion-model sweep (mean reprojection px; informational, does NOT affect gate) ──"
    );
    print!("  {:<13} |", "model");
    for c in 0..NUM_CAMERAS {
        print!(" {:>8} |", format!("cam{c}"));
    }
    println!(" {:>8}", "median");
    for model in models {
        let results: Vec<Result<f64>> = per_cam_views
            .iter()
            .enumerate()
            .map(|(c, views)| sweep_solve_camera(spec, c, views.clone(), model, max_iters))
            .collect();
        print!("  {:<13} |", format!("{model:?}"));
        for cell in &results {
            let s = match cell {
                Ok(v) if v.is_finite() => format!("{v:.4}"),
                Ok(_) => "DIVERGED".to_string(),
                Err(_) => "ERR".to_string(),
            };
            print!(" {s:>8} |");
        }
        let finite: Vec<f64> = results
            .iter()
            .filter_map(|c| c.as_ref().ok().copied())
            .collect();
        match median(finite) {
            Some(m) => println!(" {m:>8.4}"),
            None => println!(" {:>8}", "n/a"),
        }
    }
}

/// Reference (fx, fy, cx, cy, Brown-Conrady, Scheimpflug) for a camera.
fn ref_params(intr: &RefIntrinsic) -> (f64, f64, f64, f64, BrownConrady5<f64>, ScheimpflugParams) {
    let m = &intr.matrix;
    let (dist, tilt) = split_opencv_distortion(&intr.distortion).expect("oracle distortion");
    (m[0][0], m[1][1], m[0][2], m[1][2], dist, tilt)
}

fn report(art: &RefArtifacts, results: &[CamResult]) {
    println!("── Recovered intrinsics vs oracle (coarse seed) ──");
    println!(
        "  cam | views | corners |     fx (ref)      |     fy (ref)      |   cx (ref)    |   cy (ref)    |    k1 (ref)     |  tau_x° (ref)  |  tau_y° (ref)"
    );
    for (i, r) in results.iter().enumerate() {
        let rf = ref_params(&art.intrinsic[i]);
        println!(
            "  {i:>3} | {:>5} | {:>7} | {:8.1} ({:7.1}) | {:8.1} ({:7.1}) | {:6.1} ({:6.1}) | {:6.1} ({:6.1}) | {:+.4} ({:+.4}) | {:+.3} ({:+.3}) | {:+.3} ({:+.3})",
            r.views,
            r.corners,
            r.intr.fx,
            rf.0,
            r.intr.fy,
            rf.1,
            r.intr.cx,
            rf.2,
            r.intr.cy,
            rf.3,
            r.dist.k1,
            rf.4.k1,
            r.sensor.tilt_x.to_degrees(),
            rf.5.tilt_x.to_degrees(),
            r.sensor.tilt_y.to_degrees(),
            rf.5.tilt_y.to_degrees(),
        );
    }

    println!("\n── Per-camera mean reprojection: ours (seeded) vs oracle ──");
    println!("  cam | our_reproj | ref_reproj |    Δ    | gate");
    for (i, r) in results.iter().enumerate() {
        let refp = art.intrinsic[i].reprojection_error_pix;
        let gate = if r.mean_reproj <= GATE_PX {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "  {i:>3} | {:10.4} | {:10.4} | {:+.4} | {gate}",
            r.mean_reproj,
            refp,
            r.mean_reproj - refp
        );
    }
}
