//! Per-camera Scheimpflug **intrinsics** calibration on the `rtv3d_ref`
//! reference dataset from a *coarse, user-provided* seed — the supported
//! workflow (ADR 0022).
//!
//! Companion to `rtv3d_ref_rig` (full from-scratch rig) and `rtv3d_ref_reproj`
//! (frozen-intrinsics parity). This harness exercises the recommended path for
//! Scheimpflug intrinsics: the engineer supplies a coarse prior they actually
//! have — a nominal focal (from the lens spec) and the nominal sensor tilt (the
//! Scheimpflug mount angle, ≈−5°) — and bundle adjustment refines it. Nothing
//! from `artifacts.json` is seeded; the oracle is read only for the final
//! comparison.
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
//! - `RTV3D_REF_DATA_DIR` (default `privatedata/rtv3d_ref`).
//! - `RTV3D_REF_FOCAL` — coarse nominal focal seed in px (default `1150`). Probe
//!   the tolerance by varying it; the oracle focals are ≈1150–1166.
//! - `RTV3D_REF_MAXITERS` (default `120`).

use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;
use std::time::Instant;

use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsProblem,
    ScheimpflugManualInit, step_init_with_seed, step_optimize,
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
use vision_calibration_optim::RobustLoss;

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 130;
const BOARD_COLS: u32 = 130;
const CELL_SIZE_MM: f64 = 5.0;
const GATE_PX: f64 = 0.5;
/// Each pose strip is 4320×540 → six 720×540 tiles; the nominal principal point
/// is the tile center.
const TILE_CX: f64 = 360.0;
const TILE_CY: f64 = 270.0;
/// Nominal Scheimpflug mount tilt (≈−5°): a known mechanical spec, used as the
/// seed `tilt_x`. The actual per-camera tilt is recovered by the solve.
const NOMINAL_TILT_X: f64 = -0.087;

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
    let focal_seed: f64 = std::env::var("RTV3D_REF_FOCAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1150.0);
    let max_iters: usize = std::env::var("RTV3D_REF_MAXITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    println!("data dir = {}", data_dir.display());
    println!(
        "coarse seed: fx=fy={focal_seed:.0}, pp=({TILE_CX:.0},{TILE_CY:.0}), tilt_x={NOMINAL_TILT_X:.3} rad, distortion=0"
    );

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
    println!("loaded {} poses, {} oracle cameras", poses.len(), art.num_cameras);

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

        // Coarse "datasheet" prior: nominal focal, principal point at tile center,
        // nominal mount tilt. Distortion + poses auto. The seeded tilt is trusted
        // directly (ADR 0022).
        let mut seed = ScheimpflugManualInit::default();
        seed.intrinsics = Some(FxFyCxCySkew {
            fx: focal_seed,
            fy: focal_seed,
            cx: TILE_CX,
            cy: TILE_CY,
            skew: 0.0,
        });
        seed.sensor = Some(ScheimpflugParams {
            tilt_x: NOMINAL_TILT_X,
            tilt_y: 0.0,
        });
        step_init_with_seed(&mut session, seed, None)
            .with_context(|| format!("camera {c}: seeded init"))?;

        step_optimize(&mut session, None).with_context(|| format!("camera {c}: optimize"))?;

        let out = session.output().expect("output after optimize");
        let intr = match &out.params.camera.intrinsics {
            IntrinsicsParams::FxFyCxCySkew { params } => *params,
        };
        let dist = match &out.params.camera.distortion {
            vision_calibration_core::DistortionParams::BrownConrady5 { params } => *params,
            other => return Err(anyhow!("camera {c}: unexpected distortion params: {other:?}")),
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

    // ── Hard gate: every camera ≤ 0.5 px ─────────────────────────────────────
    let failed: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .filter(|(_, r)| !(r.mean_reproj <= GATE_PX))
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
        let gate = if r.mean_reproj <= GATE_PX { "PASS" } else { "FAIL" };
        println!(
            "  {i:>3} | {:10.4} | {:10.4} | {:+.4} | {gate}",
            r.mean_reproj,
            refp,
            r.mean_reproj - refp
        );
    }
}
