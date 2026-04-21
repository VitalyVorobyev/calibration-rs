//! End-to-end rig calibration on the 130x130 puzzleboard dataset.
//!
//! Runs the complete pipeline:
//! 1. Detect 130x130 puzzleboard (1.014 mm cells) in each of the 6 per-camera
//!    tiles for every pose.
//! 2. Detect laser lines in double-snap poses.
//! 3. `RigScheimpflugHandeyeProblem` session (intrinsics → rig → hand-eye).
//! 4. `RigLaserlineDeviceProblem` session consuming the frozen upstream
//!    calibration to recover 6 laser planes in rig frame.
//! 5. Demonstrate `pixel_to_gripper_point` on sample laser pixels.
//!
//! Set `PUZZLE_DATA_DIR` to the dataset directory and run with
//! `cargo run -p vision-calibration-examples-private --example
//! puzzle_130x130_rig --release`.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::time::Instant;

use vision_calibration::{
    pixel_to_gripper_point,
    rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
        RigUpstreamCalibration,
    },
    rig_scheimpflug_handeye::{RigScheimpflugHandeyeConfig, RigScheimpflugHandeyeProblem},
    session::CalibrationSession,
};
use vision_calibration_core::{CorrespondenceView, Pt2, RigDataset, RigView, RigViewObs};
use vision_calibration_examples_private::{
    PoseEntry, detect_laser, detect_target, load_gray, load_poses, split_horizontal,
};
use vision_calibration_optim::{RigLaserlineDataset, RigLaserlineView, RobotPoseMeta};

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 130;
const BOARD_COLS: u32 = 130;
const CELL_SIZE_MM: f64 = 1.014;

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("PUZZLE_DATA_DIR")
            .unwrap_or_else(|_| "privatedata/130x130_puzzle".to_string()),
    );
    println!("data dir = {}", data_dir.display());

    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!("loaded {} poses", poses.len());

    // ─── Stage 1: detect targets and laser lines ───────────────────────────
    let t0 = Instant::now();
    let (rig_handeye_views, rig_laserline_views) =
        build_datasets(&data_dir, &poses).context("detect stage")?;
    println!(
        "stage 1 (detect): {:.2?} → {} handeye views, {} laser views",
        t0.elapsed(),
        rig_handeye_views.len(),
        rig_laserline_views.len()
    );

    // ─── Stage 2: rig + scheimpflug + hand-eye ─────────────────────────────
    let handeye_dataset = RigDataset::new(rig_handeye_views, NUM_CAMERAS)?;
    let mut rig_session =
        CalibrationSession::<RigScheimpflugHandeyeProblem>::with_description("puzzle_130x130_rig");
    rig_session.set_input(handeye_dataset)?;

    let mut cfg = RigScheimpflugHandeyeConfig::default();
    cfg.solver.max_iters = 120;
    cfg.solver.verbosity = 1;
    rig_session.set_config(cfg)?;

    let t0 = Instant::now();
    vision_calibration::rig_scheimpflug_handeye::run_calibration(&mut rig_session)?;
    println!(
        "stage 2 (rig + scheimpflug + hand-eye): {:.2?}",
        t0.elapsed()
    );

    let rig_export = rig_session.export()?;
    println!(
        "  mean reproj error:   {:.4} px",
        rig_export.mean_reproj_error
    );
    for (i, err) in rig_export.per_cam_reproj_errors.iter().enumerate() {
        println!("    camera {i}: {err:.4} px");
    }
    println!(
        "  gripper_se3_rig: t={:?} |t|={:.4} m",
        rig_export.gripper_se3_rig.translation.vector.data.0[0],
        rig_export.gripper_se3_rig.translation.vector.norm()
    );

    // ─── Stage 3: rig laserline calibration ────────────────────────────────
    let laserline_dataset = RigLaserlineDataset::new(rig_laserline_views, NUM_CAMERAS)
        .context("build laserline dataset")?;
    let mut rig_se3_target = Vec::new();
    for pose in &poses {
        if !pose.has_laser() {
            continue;
        }
        // T_R_T = T_G_R^-1 * T_B_G^-1 * T_B_T
        let gripper_se3_rig = rig_export.gripper_se3_rig;
        let base_se3_target = rig_export.base_se3_target;
        let base_se3_gripper = pose.base_se3_gripper();
        rig_se3_target
            .push(gripper_se3_rig.inverse() * base_se3_gripper.inverse() * base_se3_target);
    }

    let upstream = RigUpstreamCalibration {
        intrinsics: rig_export.cameras.iter().map(|c| c.k).collect(),
        distortion: rig_export.cameras.iter().map(|c| c.dist).collect(),
        sensors: rig_export.sensors.clone(),
        cam_se3_rig: rig_export.cam_se3_rig.clone(),
        rig_se3_target,
    };
    let laserline_input = RigLaserlineDeviceInput {
        dataset: laserline_dataset,
        upstream,
        initial_planes_cam: None,
    };

    let mut laser_session =
        CalibrationSession::<RigLaserlineDeviceProblem>::with_description("puzzle_130x130_laser");
    laser_session.set_input(laserline_input)?;
    let mut laser_cfg = RigLaserlineDeviceConfig::default();
    laser_cfg.max_iters = Some(200);
    laser_cfg.verbosity = Some(1);
    laser_session.set_config(laser_cfg)?;

    let t0 = Instant::now();
    vision_calibration::rig_laserline_device::run_calibration(&mut laser_session)?;
    println!("stage 3 (rig laserline): {:.2?}", t0.elapsed());

    let laser_export = laser_session.export()?;
    for (i, stats) in laser_export.per_camera_stats.iter().enumerate() {
        println!(
            "  camera {i}: reproj={:.4}px, laser={:.4}px/m",
            stats.mean_reproj_error, stats.mean_laser_error
        );
    }
    for (i, p) in laser_export.laser_planes_rig.iter().enumerate() {
        let n = p.normal.into_inner();
        println!(
            "  plane (rig) {i}: n=({:.4}, {:.4}, {:.4}), d={:.4} m",
            n.x, n.y, n.z, p.distance
        );
    }

    // ─── Stage 4: pixel → gripper point demo ───────────────────────────────
    println!("\nsample pixel→gripper mappings:");
    for cam in 0..NUM_CAMERAS {
        let pixel = Pt2::new(320.0, 240.0); // center-ish
        match pixel_to_gripper_point(cam, pixel, &rig_export, &laser_export.laser_planes_rig) {
            Ok(p) => println!(
                "  cam{cam} ({:.1},{:.1}) → ({:.3}, {:.3}, {:.3}) m (gripper)",
                pixel.x, pixel.y, p.x, p.y, p.z
            ),
            Err(e) => println!("  cam{cam}: {e}"),
        }
    }

    Ok(())
}

fn build_datasets(
    data_dir: &Path,
    poses: &[PoseEntry],
) -> Result<(Vec<RigView<RobotPoseMeta>>, Vec<RigLaserlineView>)> {
    let mut handeye_views = Vec::new();
    let mut laser_views = Vec::new();

    for (i, pose) in poses.iter().enumerate() {
        let target_img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("pose {i} target"))?;
        let target_tiles = split_horizontal(&target_img, NUM_CAMERAS);

        let mut cam_obs: Vec<Option<CorrespondenceView>> = Vec::with_capacity(NUM_CAMERAS);
        for (cam_idx, tile) in target_tiles.iter().enumerate() {
            match detect_target(tile, BOARD_ROWS, BOARD_COLS, CELL_SIZE_MM) {
                Ok(view) => cam_obs.push(Some(view)),
                Err(e) => {
                    eprintln!("pose {i} cam {cam_idx}: target detection failed ({e})");
                    cam_obs.push(None);
                }
            }
        }

        if !cam_obs.iter().any(|c| c.is_some()) {
            return Err(anyhow!("pose {i}: no target detections in any camera"));
        }

        handeye_views.push(RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: pose.base_se3_gripper(),
            },
            obs: RigViewObs {
                cameras: cam_obs.clone(),
            },
        });

        if pose.has_laser() {
            let laser_img = load_gray(&data_dir.join(&pose.laser_image))
                .with_context(|| format!("pose {i} laser"))?;
            let laser_tiles = split_horizontal(&laser_img, NUM_CAMERAS);
            let laser_pixels: Vec<Option<Vec<Pt2>>> = laser_tiles
                .iter()
                .map(|tile| {
                    let pts = detect_laser(tile);
                    if pts.is_empty() { None } else { Some(pts) }
                })
                .collect();
            laser_views.push(RigLaserlineView {
                cameras: cam_obs,
                laser_pixels,
            });
        }

        // Drop per-pose images to limit peak memory.
        let _ = target_img;
    }

    Ok((handeye_views, laser_views))
}
