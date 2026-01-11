//! Example: end-to-end hand-eye calibration with a single camera.
//!
//! This example:
//! 1) Detects chessboard corners in `data/kuka_1/*.png`
//! 2) Runs planar intrinsics init + refinement
//! 3) Initializes hand-eye with DLT and refines it non-linearly
//!
//! The dataset should include:
//! - `RobotPosesVec.txt` (base-to-gripper 4x4 matrices, one per image)
//! - `squaresize.txt` (e.g. "20mm")
//! - `01.png`..`30.png` images

use anyhow::{ensure, Context, Result};
use calib::core::{Iso3, Pt3, Vec2};
use calib::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
use calib::linear::distortion_fit::DistortionFitOptions;
use calib::linear::handeye::estimate_handeye_dlt;
use calib::linear::iterative_intrinsics::IterativeIntrinsicsOptions;
use calib::optim::backend::BackendSolveOptions;
use calib::optim::ir::RobustLoss;
use calib::optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;
use calib::pipeline::handeye::{
    optimize_handeye, CameraViewObservations, HandEyeDataset, HandEyeInit, HandEyeSolveOptions,
    RigViewObservations,
};
use calib::pipeline::{HandEyeMode, PlanarViewData};
use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;
use image::ImageReader;
use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use std::path::Path;

struct ViewSample {
    view: PlanarViewData,
    robot_pose: Iso3,
}

fn load_square_size_m(path: &Path) -> Result<f64> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read square size from {}", path.display()))?;
    let raw = text.trim().to_lowercase();
    ensure!(!raw.is_empty(), "square size file is empty");

    if let Some(mm) = raw.strip_suffix("mm") {
        let value: f64 = mm.trim().parse()?;
        return Ok(value / 1000.0);
    }

    if let Some(m) = raw.strip_suffix('m') {
        let value: f64 = m.trim().parse()?;
        return Ok(value);
    }

    Ok(raw.parse()?)
}

fn parse_pose_line(line: &str, idx: usize) -> Result<Iso3> {
    let values: Vec<f64> = line
        .split_whitespace()
        .map(|v| v.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("invalid float in robot pose line {}", idx + 1))?;
    ensure!(
        values.len() == 16,
        "robot pose line {} expected 16 values, got {}",
        idx + 1,
        values.len()
    );

    let r = Matrix3::new(
        values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
        values[10],
    );
    let t = Vector3::new(values[3], values[7], values[11]);
    let rot = Rotation3::from_matrix_unchecked(r);
    Ok(Iso3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_rotation_matrix(&rot),
    ))
}

fn load_robot_poses(path: &Path) -> Result<Vec<Iso3>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read robot poses from {}", path.display()))?;
    let mut poses = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        poses.push(parse_pose_line(line, idx)?);
    }
    ensure!(!poses.is_empty(), "robot pose file is empty");
    Ok(poses)
}

fn detection_to_view_data(
    detection: ChessboardDetectionResult,
    square_size_m: f64,
) -> Result<PlanarViewData> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();
    for corner in detection.detection.corners {
        let Some(grid) = corner.grid else {
            continue;
        };
        points_3d.push(Pt3::new(
            grid.i as f64 * square_size_m,
            grid.j as f64 * square_size_m,
            0.0,
        ));
        points_2d.push(Vec2::new(
            corner.position.x as f64,
            corner.position.y as f64,
        ));
    }

    ensure!(
        points_3d.len() >= 4,
        "insufficient corners after grid filtering"
    );

    Ok(PlanarViewData {
        points_3d,
        points_2d,
        weights: None,
    })
}

fn main() -> Result<()> {
    println!("=== Hand-Eye Single Camera Calibration Example ===\n");

    let base_path = Path::new("data/kuka_1");
    ensure!(
        base_path.exists(),
        "dataset not found at {}",
        base_path.display()
    );

    let square_size_m = load_square_size_m(&base_path.join("squaresize.txt"))?;
    let robot_poses = load_robot_poses(&base_path.join("RobotPosesVec.txt"))?;

    let chess_config = ChessConfig::default();
    let board_params = ChessboardParams::default();

    let mut samples = Vec::new();
    for (idx, robot_pose) in robot_poses.iter().enumerate() {
        let img_path = base_path.join(format!("{:02}.png", idx + 1));
        let img = ImageReader::open(&img_path)
            .with_context(|| format!("failed to read image {}", img_path.display()))?
            .decode()
            .with_context(|| format!("failed to decode {}", img_path.display()))?
            .to_luma8();

        let detection = match detect::detect_chessboard(&img, &chess_config, board_params.clone()) {
            Some(result) => result,
            None => {
                eprintln!("Skipping view {:02}: chessboard not detected", idx + 1);
                continue;
            }
        };

        let view = detection_to_view_data(detection, square_size_m)?;
        samples.push(ViewSample {
            view,
            robot_pose: robot_pose.clone(),
        });
    }

    ensure!(
        samples.len() >= 3,
        "need at least 3 valid views, got {}",
        samples.len()
    );

    let views: Vec<PlanarViewData> = samples.iter().map(|s| s.view.clone()).collect();

    let init_opts = IterativeIntrinsicsOptions {
        iterations: 2,
        distortion_opts: DistortionFitOptions {
            fix_k3: true,
            fix_tangential: false,
            iters: 8,
        },
    };

    let init = initialize_planar_intrinsics(&views, &init_opts)?;
    println!(
        "Init intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
        init.intrinsics.fx, init.intrinsics.fy, init.intrinsics.cx, init.intrinsics.cy
    );

    let solve_opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::Huber { scale: 2.0 },
        fix_k3: true,
        fix_poses: vec![0],
        ..Default::default()
    };
    let backend_opts = BackendSolveOptions {
        max_iters: 60,
        ..Default::default()
    };

    let optim = optimize_planar_intrinsics_from_init(&views, &init, &solve_opts, &backend_opts)?;
    println!(
        "Planar refinement: mean reproj error {:.3} px",
        optim.mean_reproj_error
    );

    let base_to_gripper: Vec<Iso3> = samples.iter().map(|s| s.robot_pose.clone()).collect();
    let cam_to_target = optim.poses.clone();

    // Hand-eye DLT expects camera poses in the target frame (T_T_C).
    let cam_in_target: Vec<Iso3> = cam_to_target.iter().map(|pose| pose.inverse()).collect();
    let handeye_dlt = estimate_handeye_dlt(&base_to_gripper, &cam_in_target, 1.0)
        .context("hand-eye DLT initialization failed")?;
    let handeye_init = handeye_dlt;

    let target_poses: Vec<Iso3> = base_to_gripper
        .iter()
        .zip(&cam_to_target)
        .map(|(base_to_gripper, cam_to_target)| {
            base_to_gripper.clone() * handeye_init * cam_to_target.clone()
        })
        .collect();

    let mut rig_views = Vec::new();
    for sample in &samples {
        let obs = CameraViewObservations::new(
            sample.view.points_3d.clone(),
            sample.view.points_2d.clone(),
        )?;
        rig_views.push(RigViewObservations {
            cameras: vec![Some(obs)],
            robot_pose: sample.robot_pose.clone(),
        });
    }

    let dataset = HandEyeDataset::new(rig_views, 1, HandEyeMode::EyeInHand)?;
    // Hand-eye solver packs 4 intrinsics parameters; skew must be zero.
    let mut handeye_intrinsics = optim.intrinsics;
    handeye_intrinsics.skew = 0.0;
    let init = HandEyeInit {
        intrinsics: vec![handeye_intrinsics],
        distortion: vec![optim.distortion],
        cam_to_rig: vec![Iso3::identity()],
        handeye: handeye_init,
        target_poses,
    };

    let mut handeye_opts = HandEyeSolveOptions::default();
    handeye_opts.robust_loss = RobustLoss::Huber { scale: 2.0 };
    handeye_opts.fix_fx = true;
    handeye_opts.fix_fy = true;
    handeye_opts.fix_cx = true;
    handeye_opts.fix_cy = true;
    handeye_opts.fix_k1 = true;
    handeye_opts.fix_k2 = true;
    handeye_opts.fix_k3 = true;
    handeye_opts.fix_p1 = true;
    handeye_opts.fix_p2 = true;
    handeye_opts.fix_extrinsics = vec![true];
    handeye_opts.fix_target_poses = vec![0];

    let result = optimize_handeye(dataset, init, handeye_opts, BackendSolveOptions::default())?;
    let t = result.handeye.translation.vector;
    let rot = result.handeye.rotation.to_rotation_matrix();

    println!(
        "Hand-eye (gripper-from-camera) t = [{:.4}, {:.4}, {:.4}]",
        t.x, t.y, t.z
    );
    println!("Hand-eye rotation:\n{}", rot.matrix());
    println!("Final cost: {:.6}", result.final_cost);

    Ok(())
}
