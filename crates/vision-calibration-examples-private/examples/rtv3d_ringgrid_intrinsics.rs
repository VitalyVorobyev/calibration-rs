//! Per-camera Scheimpflug **intrinsics** calibration on the private
//! `rtv3d_ringgrid` dataset (a 6-camera Scheimpflug rig captured against a
//! coded **ring-grid** target) from a coarse, user-provided seed — the
//! supported Scheimpflug workflow (ADR 0022).
//!
//! This is the fast per-camera debug loop for the ring-grid target: it detects
//! the ring-grid in every camera tile, then runs a seeded per-camera intrinsics
//! solve. Unlike the puzzleboard `rtv3d_ref_intrinsics` companion, the
//! `rtv3d_ringgrid` dataset ships **no `artifacts.json` oracle** (it is a
//! different physical rig), so there is nothing to seed from and no per-camera
//! reference intrinsics to compare against. For quality context only, the
//! puzzleboard `rtv3d_ref` oracle's per-camera reprojection error is loaded
//! read-only as a baseline column when available.
//!
//! Why seeded, not from scratch? On Scheimpflug data (strong radial distortion
//! plus a several-degree mount tilt) Zhang-from-scratch underestimates the focal
//! and settles in a wrong tilt/focal basin (ADR 0022). A coarse focal + tilt
//! seed removes that fragility. The nominal focal for this rig is **not known a
//! priori** — sweep `RTV3D_RINGGRID_FOCAL` to find the basin.
//!
//! Run:
//! `cargo run --release --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! rtv3d_ringgrid_intrinsics`
//!
//! Env:
//! - `RTV3D_RINGGRID_DATA_DIR` (default `privatedata/rtv3d_ringgrid`).
//! - `RTV3D_RINGGRID_FOCAL` — coarse nominal focal seed in px (default `1150`).
//! - `RTV3D_RINGGRID_TILT_X` — nominal mount tilt seed in rad (default `-0.087`).
//! - `RTV3D_RINGGRID_RING_WIDTH_MM` — coded-band width override (default: board
//!   manifest value, else `1.152`). A decode-tuning knob.
//! - `RTV3D_RINGGRID_MAXITERS` (default `120`).
//! - `RTV3D_RINGGRID_DETECT_ONLY` — set to `1` to print per-camera detection
//!   density and exit before solving (fast detection probe).
//! - `RTV3D_PUZZLE_REF_DIR` — puzzleboard oracle dir for the baseline column
//!   (default `privatedata/rtv3d_ref`; skipped if absent).

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
    BoardRinggridSpec, detect_ringgrid, load_gray, load_poses, load_ref_artifacts,
    load_ringgrid_board, split_horizontal,
};
use vision_calibration_optim::RobustLoss;

const NUM_CAMERAS: usize = 6;
const GATE_PX: f64 = 0.5;
/// Each pose strip is 4320×540 → six 720×540 tiles; the nominal principal point
/// is the tile center.
const TILE_CX: f64 = 360.0;
const TILE_CY: f64 = 270.0;
/// Nominal Scheimpflug mount tilt (≈−5°), used as the seed `tilt_x`.
const DEFAULT_TILT_X: f64 = -0.087;

/// Recovered summary for one camera.
struct CamResult {
    intr: FxFyCxCySkew<f64>,
    dist: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    mean_reproj: f64,
    markers: usize,
    views: usize,
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("RTV3D_RINGGRID_DATA_DIR")
            .unwrap_or_else(|_| "privatedata/rtv3d_ringgrid".to_string()),
    );
    let focal_seed = env_f64("RTV3D_RINGGRID_FOCAL", 1150.0);
    let tilt_x_seed = env_f64("RTV3D_RINGGRID_TILT_X", DEFAULT_TILT_X);
    let max_iters: usize = std::env::var("RTV3D_RINGGRID_MAXITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let detect_only = std::env::var("RTV3D_RINGGRID_DETECT_ONLY").as_deref() == Ok("1");

    let board: BoardRinggridSpec = load_ringgrid_board(&data_dir.join("board_ringgrid.json"))
        .context("load ring-grid board manifest")?;
    let ring_width_mm = env_f64("RTV3D_RINGGRID_RING_WIDTH_MM", board.ring_width_mm());

    println!("data dir = {}", data_dir.display());
    println!(
        "board: {} rows × {} long-row cols, pitch {:.1} mm, outer/inner {:.1}/{:.1} mm, ring width {:.3} mm",
        board.rows,
        board.long_row_cols,
        board.pitch_mm,
        board.marker_outer_radius_mm,
        board.marker_inner_radius_mm,
        ring_width_mm,
    );
    println!(
        "coarse seed: fx=fy={focal_seed:.0}, pp=({TILE_CX:.0},{TILE_CY:.0}), tilt_x={tilt_x_seed:.3} rad, distortion=0"
    );

    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!("loaded {} poses", poses.len());

    // ── Detect the ring-grid in every camera tile of every pose ──────────────
    let t_det = Instant::now();
    let mut per_cam_views: Vec<Vec<View<NoMeta>>> = vec![Vec::new(); NUM_CAMERAS];
    let mut det_markers = [0usize; NUM_CAMERAS];
    for (i, pose) in poses.iter().enumerate() {
        let img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("pose {i} target"))?;
        let tiles = split_horizontal(&img, NUM_CAMERAS);
        for (c, tile) in tiles.iter().enumerate() {
            if let Ok(v) = detect_ringgrid(tile, &board, ring_width_mm) {
                det_markers[c] += v.points_2d.len();
                per_cam_views[c].push(View::without_meta(v));
            }
        }
    }
    println!("detect: {:.2?}", t_det.elapsed());
    println!("  cam | views | markers | markers/view");
    for c in 0..NUM_CAMERAS {
        let v = per_cam_views[c].len();
        let m = det_markers[c];
        let per = if v > 0 { m as f64 / v as f64 } else { 0.0 };
        println!("  {c:>3} | {v:>5} | {m:>7} | {per:>7.1}");
    }
    if detect_only {
        println!("\nRTV3D_RINGGRID_DETECT_ONLY=1 → stopping after detection.");
        return Ok(());
    }

    // ── Per-camera seeded intrinsics calibration ─────────────────────────────
    let t0 = Instant::now();
    let mut results: Vec<Option<CamResult>> = Vec::with_capacity(NUM_CAMERAS);
    for (c, views) in per_cam_views.into_iter().enumerate() {
        if views.len() < 3 {
            println!("camera {c}: only {} views — skipping solve", views.len());
            results.push(None);
            continue;
        }
        let num_views = views.len();
        let dataset = PlanarDataset::new(views)
            .with_context(|| format!("camera {c}: build planar dataset"))?;
        let markers = dataset.views.iter().map(|v| v.obs.points_2d.len()).sum();

        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session.set_input(dataset)?;

        let mut config = ScheimpflugIntrinsicsConfig::default();
        config.max_iters = max_iters;
        // k1,k2 free; k3 + tangential fixed (default radial_only); both
        // Scheimpflug tilts free; robust loss for the detector outlier tail.
        config.fix_scheimpflug = ScheimpflugFixMask {
            tilt_x: false,
            tilt_y: false,
        };
        config.robust_loss = RobustLoss::Huber { scale: 1.0 };
        session.set_config(config)?;

        let mut seed = ScheimpflugManualInit::default();
        seed.intrinsics = Some(FxFyCxCySkew {
            fx: focal_seed,
            fy: focal_seed,
            cx: TILE_CX,
            cy: TILE_CY,
            skew: 0.0,
        });
        seed.sensor = Some(ScheimpflugParams {
            tilt_x: tilt_x_seed,
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
        results.push(Some(CamResult {
            intr,
            dist,
            sensor,
            mean_reproj: out.mean_reproj_error,
            markers,
            views: num_views,
        }));
    }
    println!("calibrate (seeded) total: {:.2?}\n", t0.elapsed());

    // Optional puzzleboard quality baseline (different rig — context only).
    let puzzle_ref_dir = PathBuf::from(
        std::env::var("RTV3D_PUZZLE_REF_DIR")
            .unwrap_or_else(|_| "privatedata/rtv3d_ref".to_string()),
    );
    let puzzle_reproj: Option<Vec<f64>> =
        load_ref_artifacts(&puzzle_ref_dir.join("artifacts.json"))
            .ok()
            .map(|a| {
                a.intrinsic
                    .iter()
                    .map(|i| i.reprojection_error_pix)
                    .collect()
            });

    report(&results, puzzle_reproj.as_deref());

    // ── Hard gate: every solved camera ≤ 0.5 px ──────────────────────────────
    let failed: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.as_ref().map(|r| (i, r.mean_reproj)))
        .filter(|(_, e)| *e > GATE_PX || e.is_nan())
        .collect();
    let solved = results.iter().filter(|r| r.is_some()).count();
    if failed.is_empty() && solved > 0 {
        println!("\nGATE PASS: all {solved} solved cameras ≤ {GATE_PX} px mean reprojection.");
        Ok(())
    } else if solved == 0 {
        Err(anyhow!(
            "no camera produced enough ring-grid views to solve — tune detection \
             (RTV3D_RINGGRID_RING_WIDTH_MM) or check the dataset"
        ))
    } else {
        let list = failed
            .iter()
            .map(|(i, e)| format!("cam {i}: {e:.4} px"))
            .collect::<Vec<_>>()
            .join(", ");
        Err(anyhow!(
            "GATE FAIL: {} of {solved} solved cameras exceed {GATE_PX} px ({list}). \
             Investigate focal prior (RTV3D_RINGGRID_FOCAL), detection density, or the \
             detector tail — do not relax the gate.",
            failed.len()
        ))
    }
}

fn report(results: &[Option<CamResult>], puzzle_reproj: Option<&[f64]>) {
    println!("── Recovered ring-grid intrinsics (coarse seed, no oracle for this rig) ──");
    println!(
        "  cam | views | markers |    fx    |    fy    |   cx   |   cy   |    k1     |    k2     |  tau_x° |  tau_y°"
    );
    for (i, r) in results.iter().enumerate() {
        match r {
            Some(r) => println!(
                "  {i:>3} | {:>5} | {:>7} | {:8.1} | {:8.1} | {:6.1} | {:6.1} | {:+.5} | {:+.5} | {:+.3} | {:+.3}",
                r.views,
                r.markers,
                r.intr.fx,
                r.intr.fy,
                r.intr.cx,
                r.intr.cy,
                r.dist.k1,
                r.dist.k2,
                r.sensor.tilt_x.to_degrees(),
                r.sensor.tilt_y.to_degrees(),
            ),
            None => println!("  {i:>3} |   (skipped — too few views)"),
        }
    }

    println!("\n── Per-camera mean reprojection: ring-grid (this rig) vs puzzleboard baseline ──");
    println!(
        "  (different rig — baseline is the rtv3d_ref puzzleboard oracle, quality context only)"
    );
    println!("  cam | ringgrid_reproj | puzzle_ref_reproj | gate");
    for (i, r) in results.iter().enumerate() {
        let ours = match r {
            Some(r) => format!("{:15.4}", r.mean_reproj),
            None => format!("{:>15}", "n/a"),
        };
        let base = match puzzle_reproj.and_then(|b| b.get(i)) {
            Some(b) => format!("{b:17.4}"),
            None => format!("{:>17}", "n/a"),
        };
        let gate = match r {
            Some(r) if r.mean_reproj <= GATE_PX => "PASS",
            Some(_) => "FAIL",
            None => "—",
        };
        println!("  {i:>3} | {ours} | {base} | {gate}");
    }
}
