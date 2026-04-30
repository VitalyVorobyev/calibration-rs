//! End-to-end proof that ADR 0011 manual init (PR #32) works on real data.
//!
//! Loads `data/stereo_charuco/`, then runs the rig-extrinsics pipeline three
//! ways and compares results:
//!
//! 1. **Run A — full auto-init**: today's behaviour. Captures per-cam intrinsics
//!    + cam-to-rig extrinsics + mean reprojection error.
//! 2. **Run B — replay manual seed**: seeds Run A's per-cam intrinsics +
//!    distortion via [`RigIntrinsicsManualInit`] AND seeds Run A's rig
//!    initialization via [`RigExtrinsicsManualInit`], then skips per-camera
//!    intrinsics re-optimization (since the seeds are already optimized) and
//!    runs only `step_rig_optimize`. Expectation: identical final
//!    reprojection error (Zhang + DLT bypassed entirely).
//! 3. **Run C — perturbed manual seed**: seeds intrinsics with a noticeable
//!    error (10 % focal-length perturbation, principal point offset by 50 px).
//!    Expectation: BA recovers a final reprojection error close to Run A.
//!
//! Each run also prints the stage-init log lines so the `(manual: …)` /
//! `(auto: …)` markers show the seed source.
//!
//! Run with: `cargo run -p vision-calibration --example manual_init_proof --release`

#[path = "support/stereo_charuco_io.rs"]
mod stereo_charuco_io;

use anyhow::{Result, ensure};
use chess_corners::ChessConfig;
use std::io::{self, Write};
use std::path::PathBuf;
use stereo_charuco_io::{
    BOARD_CELL_SIZE_MM, BOARD_COLS, BOARD_DICTIONARY_NAME, BOARD_MARKER_SIZE_REL, BOARD_ROWS,
    load_stereo_charuco_input_with_progress, make_charuco_detector_params,
};
use vision_calibration::core::{BrownConrady5, FxFyCxCySkew, Iso3, PinholeCamera};
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsInput, RigExtrinsicsManualInit, RigExtrinsicsProblem, RigIntrinsicsManualInit,
    step_intrinsics_optimize_all, step_rig_optimize, step_set_intrinsics_init_all,
    step_set_rig_init,
};
use vision_calibration::session::LogEntry;

fn main() -> Result<()> {
    let max_views = parse_max_views();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data_dir = repo_root.join("data/stereo_charuco");
    ensure!(
        data_dir.exists(),
        "stereo ChArUco dataset not found at {}",
        data_dir.display()
    );

    println!("=== Manual init proof on data/stereo_charuco ===\n");
    println!("Dataset: {}", data_dir.display());
    println!(
        "Board: {}x{}, cell {:.4}mm, marker scale {:.2}, dict {}",
        BOARD_ROWS, BOARD_COLS, BOARD_CELL_SIZE_MM, BOARD_MARKER_SIZE_REL, BOARD_DICTIONARY_NAME
    );
    println!("Max views: {max_views}\n");

    let chess_config = ChessConfig::default();
    let charuco_params = make_charuco_detector_params();

    let load = || -> Result<RigExtrinsicsInput> {
        let (input, summary) = load_stereo_charuco_input_with_progress(
            &data_dir,
            &chess_config,
            &charuco_params,
            Some(max_views),
            |idx, total, suffix| {
                print!("\r  Detect pair {idx}/{total} ({suffix})");
                let _ = io::stdout().flush();
            },
        )?;
        println!(
            "\n  Loaded {} views from {} pairs ({} skipped, left={} right={})",
            summary.used_views,
            summary.total_pairs,
            summary.skipped_views,
            summary.usable_left,
            summary.usable_right
        );
        Ok(input)
    };

    // ─── Run A: full auto pipeline ────────────────────────────────────────
    println!("--- Run A: full auto-init pipeline ---");
    let mut session_a = CalibrationSession::<RigExtrinsicsProblem>::new();
    session_a.set_input(load()?)?;
    step_set_intrinsics_init_all(&mut session_a, RigIntrinsicsManualInit::default(), None)?;
    step_intrinsics_optimize_all(&mut session_a, None)?;
    step_set_rig_init(&mut session_a, RigExtrinsicsManualInit::default())?;
    step_rig_optimize(&mut session_a, None)?;
    let a_summary = summarize(&session_a, "Run A")?;
    print_init_logs(&session_a.log, "Run A");
    print_per_feature_residuals_summary(&mut session_a, "Run A")?;
    println!();

    let cameras_a = session_a
        .state
        .per_cam_intrinsics
        .clone()
        .expect("Run A: per-cam intrinsics");
    let cam_se3_rig_a = session_a
        .state
        .initial_cam_se3_rig
        .clone()
        .expect("Run A: initial cam_se3_rig");
    let rig_se3_target_a = session_a
        .state
        .initial_rig_se3_target
        .clone()
        .expect("Run A: initial rig_se3_target");

    let per_cam_k: Vec<FxFyCxCySkew<f64>> = cameras_a.iter().map(|c| c.k).collect();
    let per_cam_dist: Vec<BrownConrady5<f64>> = cameras_a.iter().map(|c| c.dist).collect();

    // ─── Run B: replay Run A's intermediates as manual seeds ──────────────
    println!("--- Run B: replay Run A's seeds via ManualInit ---");
    let mut session_b = CalibrationSession::<RigExtrinsicsProblem>::new();
    session_b.set_input(load()?)?;
    step_set_intrinsics_init_all(
        &mut session_b,
        RigIntrinsicsManualInit {
            per_cam_intrinsics: Some(per_cam_k.clone()),
            per_cam_distortion: Some(per_cam_dist.clone()),
            per_cam_sensors: None,
        },
        None,
    )?;
    // Skip step_intrinsics_optimize_all: the seed is Run A's already-optimized
    // intrinsics, so re-running the per-camera optimizer would just shift them
    // by solver-noise levels and is not what a "load saved calibration" caller
    // would do. We still need state.per_cam_reproj_errors populated for the
    // rig stage, but rig_optimize does not depend on it directly — it only
    // needs per_cam_intrinsics + per_cam_target_poses, both already set by
    // step_set_intrinsics_init_all.
    step_set_rig_init(
        &mut session_b,
        RigExtrinsicsManualInit {
            cam_se3_rig: Some(cam_se3_rig_a.clone()),
            rig_se3_target: Some(rig_se3_target_a.clone()),
        },
    )?;
    step_rig_optimize(&mut session_b, None)?;
    let b_summary = summarize(&session_b, "Run B")?;
    print_init_logs(&session_b.log, "Run B");
    println!();

    // ─── Run C: rough perturbed seed (intrinsics only) ────────────────────
    println!("--- Run C: perturbed intrinsics seed (10% fx, 50px cx) ---");
    let perturbed_k: Vec<FxFyCxCySkew<f64>> = per_cam_k
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
            FxFyCxCySkew {
                fx: k.fx * (1.0 + sign * 0.10),
                fy: k.fy * (1.0 + sign * 0.10),
                cx: k.cx + sign * 50.0,
                cy: k.cy - sign * 50.0,
                skew: k.skew,
            }
        })
        .collect();
    let mut session_c = CalibrationSession::<RigExtrinsicsProblem>::new();
    session_c.set_input(load()?)?;
    step_set_intrinsics_init_all(
        &mut session_c,
        RigIntrinsicsManualInit {
            per_cam_intrinsics: Some(perturbed_k),
            per_cam_distortion: None, // let auto-fit it
            per_cam_sensors: None,
        },
        None,
    )?;
    step_intrinsics_optimize_all(&mut session_c, None)?;
    step_set_rig_init(&mut session_c, RigExtrinsicsManualInit::default())?;
    step_rig_optimize(&mut session_c, None)?;
    let c_summary = summarize(&session_c, "Run C")?;
    print_init_logs(&session_c.log, "Run C");
    println!();

    // ─── Verdict ──────────────────────────────────────────────────────────
    println!("=== Verdict ===");
    print_compare("auto vs replay (B vs A)", &a_summary, &b_summary);
    print_compare("auto vs perturbed (C vs A)", &a_summary, &c_summary);

    let replay_diff = (b_summary.mean_reproj - a_summary.mean_reproj).abs();
    let recover_diff = (c_summary.mean_reproj - a_summary.mean_reproj).abs();

    let replay_ok = replay_diff < 1.0e-4;
    let recover_ok = recover_diff < 0.05;

    println!(
        "\n  replay agrees with auto: {} (|Δ reproj| = {:.2e} px, threshold 1e-4)",
        if replay_ok { "PASS" } else { "FAIL" },
        replay_diff
    );
    println!(
        "  perturbed BA recovers : {} (|Δ reproj| = {:.4} px, threshold 0.05)",
        if recover_ok { "PASS" } else { "FAIL" },
        recover_diff
    );

    ensure!(
        replay_ok && recover_ok,
        "manual init proof failed: replay_ok={replay_ok}, recover_ok={recover_ok}"
    );
    println!("\nAll checks passed.");
    Ok(())
}

struct RunSummary {
    label: String,
    mean_reproj: f64,
    per_cam_reproj: Vec<f64>,
    cameras: Vec<PinholeCamera>,
    cam_se3_rig: Vec<Iso3>,
}

fn summarize(
    session: &CalibrationSession<RigExtrinsicsProblem>,
    label: &str,
) -> Result<RunSummary> {
    let mean_reproj = session
        .state
        .rig_ba_reproj_error
        .ok_or_else(|| anyhow::anyhow!("{label}: rig BA reproj missing"))?;
    let per_cam_reproj = session
        .state
        .rig_ba_per_cam_reproj_errors
        .clone()
        .unwrap_or_default();
    let cameras = session.state.per_cam_intrinsics.clone().unwrap_or_default();
    let output = session
        .require_output()
        .map_err(|e| anyhow::anyhow!("{label}: no output: {e}"))?;
    let cam_se3_rig: Vec<Iso3> = output.cam_to_rig().iter().map(|t| t.inverse()).collect();

    println!("  {label}: mean reproj = {:.4} px", mean_reproj);
    for (i, err) in per_cam_reproj.iter().enumerate() {
        println!("    cam {i}: reproj = {:.4} px", err);
    }
    if cam_se3_rig.len() >= 2 {
        let baseline =
            (cam_se3_rig[1].translation.vector - cam_se3_rig[0].translation.vector).norm();
        println!(
            "    baseline |t(cam0->cam1)| = {:.4} m ({:.2} mm)",
            baseline,
            baseline * 1000.0
        );
    }
    for (i, cam) in cameras.iter().enumerate() {
        let k = &cam.k;
        println!(
            "    cam {i}: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
            k.fx, k.fy, k.cx, k.cy
        );
    }

    Ok(RunSummary {
        label: label.to_string(),
        mean_reproj,
        per_cam_reproj,
        cameras,
        cam_se3_rig,
    })
}

fn print_compare(title: &str, a: &RunSummary, b: &RunSummary) {
    println!("\n  {title}:");
    println!(
        "    mean reproj: {:.4} → {:.4} px (Δ = {:+.2e})",
        a.mean_reproj,
        b.mean_reproj,
        b.mean_reproj - a.mean_reproj
    );
    for (i, (ea, eb)) in a
        .per_cam_reproj
        .iter()
        .zip(b.per_cam_reproj.iter())
        .enumerate()
    {
        println!(
            "    cam {i}: {:.4} → {:.4} px (Δ = {:+.2e})",
            ea,
            eb,
            eb - ea
        );
    }
    if a.cam_se3_rig.len() >= 2 && b.cam_se3_rig.len() >= 2 {
        let ba = (a.cam_se3_rig[1].translation.vector - a.cam_se3_rig[0].translation.vector).norm();
        let bb = (b.cam_se3_rig[1].translation.vector - b.cam_se3_rig[0].translation.vector).norm();
        println!(
            "    baseline: {:.2} → {:.2} mm (Δ = {:+.4} mm)",
            ba * 1000.0,
            bb * 1000.0,
            (bb - ba) * 1000.0
        );
    }
    for (i, (ka, kb)) in a.cameras.iter().zip(b.cameras.iter()).enumerate() {
        println!(
            "    cam {i}: fx Δ = {:+.4}, fy Δ = {:+.4}, cx Δ = {:+.4}, cy Δ = {:+.4}",
            kb.k.fx - ka.k.fx,
            kb.k.fy - ka.k.fy,
            kb.k.cx - ka.k.cx,
            kb.k.cy - ka.k.cy,
        );
    }
    let _ = b.label.as_str();
}

fn print_init_logs(log: &[LogEntry], label: &str) {
    let init_ops = ["intrinsics_init_all", "rig_init"];
    println!("  {label} init log:");
    for entry in log {
        if init_ops.contains(&entry.operation.as_str()) {
            let notes = entry.notes.as_deref().unwrap_or("");
            println!("    [{}] {}", entry.operation, notes);
        }
    }
}

/// Demonstrate the ADR 0012 per-feature residuals on the exported result.
///
/// Calls `session.export_peek()` (does not mutate the export collection),
/// prints per-camera histograms, head/tail of the residual vector, and
/// asserts that:
/// - the histogram counts sum to the number of records that produced a
///   `Some(error_px)`;
/// - the mean recomputed from the records matches the histogram's `mean`
///   field within float roundoff.
fn print_per_feature_residuals_summary(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    label: &str,
) -> Result<()> {
    let export = session.export_peek()?;
    let pf = &export.per_feature_residuals;
    let total = pf.target.len();
    let with_value = pf.target.iter().filter(|r| r.error_px.is_some()).count();
    println!(
        "  {label} per-feature residuals: {total} records ({with_value} with finite error_px, {} divergent)",
        total - with_value
    );

    if let Some(hists) = pf.target_hist_per_camera.as_ref() {
        for (i, h) in hists.iter().enumerate() {
            println!(
                "    cam {i}: count={} mean={:.4}px max={:.4}px buckets <=1={} <=2={} <=5={} <=10={} >10={}",
                h.count,
                h.mean,
                h.max,
                h.counts[0],
                h.counts[1],
                h.counts[2],
                h.counts[3],
                h.counts[4]
            );
            // Each histogram aggregates only that camera's records that
            // produced a finite error. Verify that invariant:
            let cam_with_value = pf
                .target
                .iter()
                .filter(|r| r.camera == i)
                .filter_map(|r| r.error_px)
                .count();
            ensure!(
                h.count == cam_with_value,
                "{label}: cam {i} histogram count {} != expected {}",
                h.count,
                cam_with_value
            );
            let mean_from_records: f64 = if cam_with_value == 0 {
                0.0
            } else {
                pf.target
                    .iter()
                    .filter(|r| r.camera == i)
                    .filter_map(|r| r.error_px)
                    .sum::<f64>()
                    / cam_with_value as f64
            };
            ensure!(
                (mean_from_records - h.mean).abs() < 1e-9,
                "{label}: cam {i} mean from records {mean_from_records} != histogram mean {}",
                h.mean
            );
        }
    }

    let head_n = 3.min(pf.target.len());
    if head_n > 0 {
        println!("    head:");
        for r in pf.target.iter().take(head_n) {
            println!(
                "      pose={} cam={} feat={} err={}",
                r.pose,
                r.camera,
                r.feature,
                r.error_px
                    .map(|e| format!("{:.4}px", e))
                    .unwrap_or_else(|| "None".to_string())
            );
        }
    }
    let tail_n = 3.min(pf.target.len().saturating_sub(head_n));
    if tail_n > 0 {
        println!("    tail:");
        for r in pf.target.iter().rev().take(tail_n).rev() {
            println!(
                "      pose={} cam={} feat={} err={}",
                r.pose,
                r.camera,
                r.feature,
                r.error_px
                    .map(|e| format!("{:.4}px", e))
                    .unwrap_or_else(|| "None".to_string())
            );
        }
    }
    Ok(())
}

fn parse_max_views() -> usize {
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        if let Some(raw) = arg.strip_prefix("--max-views=") {
            return raw.parse().expect("invalid --max-views");
        }
        if arg == "--max-views"
            && let Some(raw) = iter.next()
        {
            return raw.parse().expect("invalid --max-views");
        }
    }
    8
}
