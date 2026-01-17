//! Stereo (two-camera) calibration session on real images in `data/stereo/imgs`.
//!
//! This example demonstrates:
//! - chessboard corner detection on real images,
//! - multi-camera rig calibration (per-camera intrinsics + camera-to-rig extrinsics),
//! - per-step reprojection error reporting,
//! - optionally fixing intrinsics during rig BA (`--fix-intrinsics`).

use anyhow::{ensure, Context, Result};
use calib::core::{CameraParams, Iso3, Real};
use calib::optim::ir::RobustLoss;
use calib::rig::{
    rig_reprojection_errors, rig_reprojection_errors_from_report, RigCameraViewData,
    RigExtrinsicsInitOptions, RigExtrinsicsInput, RigExtrinsicsOptimOptions, RigReprojectionErrors,
    RigViewData,
};
use calib::session::{CalibrationSession, RigExtrinsicsProblem};
use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;
use image::ImageReader;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let fix_intrinsics = args.iter().any(|a| a == "--fix-intrinsics");
    let max_views = parse_max_views(&args)?;

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imgs_dir = repo_root.join("data/stereo/imgs");
    ensure!(
        imgs_dir.exists(),
        "stereo dataset not found at {}",
        imgs_dir.display()
    );

    // Dataset settings (matches `crates/calib-linear/tests/data/stereo_linear.json`).
    let board_rows = 7;
    let board_cols = 11;
    let square_size_m = 0.03;

    let chess_config = ChessConfig::default();
    let board_params = ChessboardParams {
        expected_rows: Some(board_rows),
        expected_cols: Some(board_cols),
        ..ChessboardParams::default()
    };

    println!("=== Stereo rig calibration session ===");
    println!(
        "Dataset: {}, board {}x{}, square {:.3} m",
        imgs_dir.display(),
        board_rows,
        board_cols,
        square_size_m
    );
    println!(
        "BA intrinsics refinement: {}",
        if fix_intrinsics { "OFF" } else { "ON" }
    );
    if let Some(n) = max_views {
        println!("Max views: {n}");
    }
    println!();

    let input = load_stereo_input(
        &imgs_dir,
        &chess_config,
        &board_params,
        square_size_m,
        max_views,
    )?;
    ensure!(
        input.views.len() >= 3,
        "need at least 3 usable views, got {}",
        input.views.len()
    );

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_observations(input.clone());

    // --- Init ---
    session.initialize(RigExtrinsicsInitOptions {
        ref_cam_idx: 0,
        ..Default::default()
    })?;
    let init = session.initial_values().context("missing init values")?;
    println!("--- Init ---");
    print_cameras("Per-camera init", &init.cameras);
    print_baseline("Init baseline", &init.cam_to_rig);
    let init_reproj = rig_reprojection_errors(
        &input,
        &init.cameras,
        &init.cam_to_rig,
        &init.rig_from_target,
    )?;
    print_reprojection("Init reprojection", &init_reproj);
    println!();

    // --- Optimize ---
    let mut optim_opts = RigExtrinsicsOptimOptions::default();
    optim_opts.solve_opts.robust_loss = RobustLoss::Huber { scale: 2.0 };
    optim_opts.backend_opts.max_iters = 200;
    if fix_intrinsics {
        let n = input.num_cameras;
        optim_opts.solve_opts.fix_intrinsics = vec![true; n];
        optim_opts.solve_opts.fix_distortion = vec![true; n];
    }

    session.optimize(optim_opts)?;
    let report = session.export()?;

    println!("--- Optimized ---");
    print_cameras("Per-camera optimized", &report.cameras);
    print_baseline("Optimized baseline", &report.cam_to_rig);
    let final_reproj = rig_reprojection_errors_from_report(&input, &report)?;
    print_reprojection("Optimized reprojection", &final_reproj);

    Ok(())
}

fn parse_max_views(args: &[String]) -> Result<Option<usize>> {
    for arg in args {
        let Some(raw) = arg.strip_prefix("--max-views=") else {
            continue;
        };
        let n: usize = raw.parse().context("invalid --max-views value")?;
        ensure!(n > 0, "--max-views must be > 0");
        return Ok(Some(n));
    }
    Ok(None)
}

fn load_stereo_input(
    imgs_dir: &Path,
    chess_config: &ChessConfig,
    board_params: &ChessboardParams,
    square_size_m: Real,
    max_views: Option<usize>,
) -> Result<RigExtrinsicsInput> {
    let left_dir = imgs_dir.join("leftcamera");
    let right_dir = imgs_dir.join("rightcamera");
    ensure!(
        left_dir.exists(),
        "left camera folder not found: {}",
        left_dir.display()
    );
    ensure!(
        right_dir.exists(),
        "right camera folder not found: {}",
        right_dir.display()
    );

    let mut views = Vec::new();
    let mut total_left = 0usize;
    let mut total_right = 0usize;
    let mut skipped = 0usize;

    for idx in 1usize.. {
        if let Some(max) = max_views {
            if views.len() >= max {
                break;
            }
        }

        let left_path = left_dir.join(format!("Im_L_{idx}.png"));
        let right_path = right_dir.join(format!("Im_R_{idx}.png"));

        if !left_path.exists() && !right_path.exists() {
            break;
        }
        ensure!(
            left_path.exists() && right_path.exists(),
            "missing stereo pair: left={}, right={}",
            left_path.display(),
            right_path.display()
        );

        let left = detect_view(
            &left_path,
            chess_config,
            board_params.clone(),
            square_size_m,
        )
        .with_context(|| format!("left detection failed for {}", left_path.display()))?;
        let right = detect_view(
            &right_path,
            chess_config,
            board_params.clone(),
            square_size_m,
        )
        .with_context(|| format!("right detection failed for {}", right_path.display()))?;

        if left.is_none() && right.is_none() {
            skipped += 1;
            continue;
        }
        if left.is_some() {
            total_left += 1;
        }
        if right.is_some() {
            total_right += 1;
        }

        views.push(RigViewData {
            cameras: vec![left, right],
        });
    }

    ensure!(
        total_left >= 3 && total_right >= 3,
        "need >=3 usable views per camera (left={}, right={})",
        total_left,
        total_right
    );

    println!(
        "Loaded {} views (skipped {} empty), usable left={}, right={}",
        views.len(),
        skipped,
        total_left,
        total_right
    );

    Ok(RigExtrinsicsInput {
        views,
        num_cameras: 2,
    })
}

fn detect_view(
    path: &Path,
    chess_config: &ChessConfig,
    board_params: ChessboardParams,
    square_size_m: Real,
) -> Result<Option<RigCameraViewData>> {
    let img = ImageReader::open(path)
        .with_context(|| format!("failed to read image {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_luma8();

    let detection = detect::detect_chessboard(&img, chess_config, board_params);
    let Some(detection) = detection else {
        return Ok(None);
    };
    Ok(Some(detection_to_view_data(detection, square_size_m)?))
}

fn detection_to_view_data(
    detection: ChessboardDetectionResult,
    square_size_m: Real,
) -> Result<RigCameraViewData> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();
    for corner in detection.detection.corners {
        let Some(grid) = corner.grid else {
            continue;
        };
        points_3d.push(calib::core::Pt3::new(
            grid.i as Real * square_size_m,
            grid.j as Real * square_size_m,
            0.0,
        ));
        points_2d.push(calib::core::Vec2::new(
            corner.position.x as Real,
            corner.position.y as Real,
        ));
    }

    ensure!(
        points_3d.len() >= 4,
        "insufficient corners after grid filtering"
    );

    Ok(RigCameraViewData {
        points_3d,
        points_2d,
        weights: None,
    })
}

fn print_cameras(label: &str, cams: &[CameraParams]) {
    println!("{label}:");
    for (idx, cam) in cams.iter().enumerate() {
        let k = match cam.intrinsics {
            calib::core::IntrinsicsParams::FxFyCxCySkew { params } => params,
        };
        let dist = match cam.distortion {
            calib::core::DistortionParams::BrownConrady5 { params } => Some(params),
            calib::core::DistortionParams::None => None,
        };

        print!(
            "  cam{idx}: fx={:.3} fy={:.3} cx={:.3} cy={:.3}",
            k.fx, k.fy, k.cx, k.cy
        );
        if let Some(d) = dist {
            print!(
                " | k1={:.3e} k2={:.3e} k3={:.3e} p1={:.3e} p2={:.3e}",
                d.k1, d.k2, d.k3, d.p1, d.p2
            );
        } else {
            print!(" | dist=None");
        }
        println!();
    }
}

fn print_baseline(label: &str, cam_to_rig: &[Iso3]) {
    if cam_to_rig.len() < 2 {
        return;
    }
    let baseline = cam_to_rig[1].translation.vector.norm();
    println!("{label}: |t(cam1->rig)| = {baseline:.6} m");
}

fn print_reprojection(label: &str, err: &RigReprojectionErrors) {
    println!("{label}:");
    if let (Some(mean), Some(rmse)) = (err.mean_px, err.rmse_px) {
        println!("  overall: mean={mean:.4} px, rmse={rmse:.4} px");
    } else {
        println!("  overall: (no projectable points)");
    }

    for (idx, (m, r)) in err
        .per_camera_mean_px
        .iter()
        .zip(err.per_camera_rmse_px.iter())
        .enumerate()
    {
        match (m, r) {
            (Some(m), Some(r)) => println!("  cam{idx}: mean={m:.4} px, rmse={r:.4} px"),
            _ => println!("  cam{idx}: (no projectable points)"),
        }
    }
    println!(
        "  points: used={}, skipped={}",
        err.num_points_used, err.num_points_skipped
    );
}
