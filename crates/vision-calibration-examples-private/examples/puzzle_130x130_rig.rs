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
//! `cargo run --manifest-path crates/vision-calibration-examples-private/Cargo.toml
//! --example puzzle_130x130_rig --release`.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[path = "puzzle_130x130_rig/viewer.rs"]
mod puzzle_viewer;

use vision_calibration::{
    pixel_to_gripper_point,
    rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
        RigUpstreamCalibration,
    },
    rig_scheimpflug_handeye::{RigScheimpflugHandeyeConfig, RigScheimpflugHandeyeProblem},
    session::CalibrationSession,
};
use vision_calibration_core::{
    BrownConrady5, CameraFixMask, CorrespondenceView, DistortionFixMask, FxFyCxCySkew, Pt2,
    RigDataset, RigView, RigViewObs, ScheimpflugParams, make_pinhole_camera,
};
use vision_calibration_examples_private::{
    PoseEntry, detect_laser, detect_target, load_gray, load_poses, split_horizontal,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeMode, LaserlineResidualType, RigHandeyeLaserlineDataset,
    RigHandeyeLaserlineParams, RigHandeyeLaserlinePerCamStats, RigHandeyeLaserlineSolveOptions,
    RigHandeyeLaserlineView, RigLaserlineDataset, RigLaserlineView, RobotPoseMeta, RobustLoss,
    ScheimpflugFixMask, evaluate_rig_handeye_laserline, optimize_rig_handeye_laserline,
};

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 130;
const BOARD_COLS: u32 = 130;
const CELL_SIZE_MM: f64 = 1.014;

/// Homogeneous rig: all 6 cameras share the same optics. Provide a good
/// initial intrinsic guess that accounts for Scheimpflug (principal point at
/// image center, moderate focal length). Zhang's method applied to
/// Scheimpflug images "compensates" for the missing tilt by shifting cy,
/// yielding a wrong (but self-consistent) K. Non-linear BA with tilts as
/// free parameters should then find the correct tilt + refine fx/fy.
fn build_initial_intrinsics(tile_w: u32, tile_h: u32) -> FxFyCxCySkew<f64> {
    FxFyCxCySkew {
        fx: 1800.0,
        fy: 1800.0,
        cx: (tile_w as f64) * 0.5,
        cy: (tile_h as f64) * 0.5,
        skew: 0.0,
    }
}

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("PUZZLE_DATA_DIR")
            .unwrap_or_else(|_| "privatedata/130x130_puzzle".to_string()),
    );
    println!("data dir = {}", data_dir.display());

    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!("loaded {} poses", poses.len());

    // Probe tile size from the first target image.
    let first_target = load_gray(&data_dir.join(&poses[0].target_image))?;
    let tile_w = first_target.width() / NUM_CAMERAS as u32;
    let tile_h = first_target.height();
    println!("tile size: {tile_w}x{tile_h}");
    drop(first_target);

    // ─── Stage 1: detect targets and laser lines ───────────────────────────
    let t0 = Instant::now();
    let detected = build_datasets(&data_dir, &poses).context("detect stage")?;
    println!(
        "stage 1 (detect): {:.2?} → {} handeye views, {} laser views, {} joint views",
        t0.elapsed(),
        detected.handeye_views.len(),
        detected.laserline_views.len(),
        detected.joint_views.len()
    );
    print_detection_diagnostics(&detected);

    // ─── Stage 2: rig + scheimpflug + hand-eye ─────────────────────────────
    let handeye_dataset = RigDataset::new(detected.handeye_views.clone(), NUM_CAMERAS)?;
    let mut rig_session =
        CalibrationSession::<RigScheimpflugHandeyeProblem>::with_description("puzzle_130x130_rig");
    rig_session.set_input(handeye_dataset)?;

    // Provide a good initial intrinsic guess for all 6 cameras so BA recovers
    // the Scheimpflug tilts rather than drifting cy to compensate. Principal
    // point is set to the tile center; focal length is a rough prior.
    let initial_k = build_initial_intrinsics(tile_w, tile_h);
    let initial_dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let initial_camera = make_pinhole_camera(initial_k, initial_dist);
    let initial_cameras = vec![initial_camera; NUM_CAMERAS];
    let initial_sensors = vec![ScheimpflugParams::default(); NUM_CAMERAS];

    let mut cfg = RigScheimpflugHandeyeConfig::default();
    cfg.solver.max_iters = 200;
    cfg.solver.verbosity = 1;
    cfg.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
    // Sensor rig is rigidly mounted in the robot base frame; the puzzleboard
    // target is held by the robot end-effector → hand-to-eye / EyeToHand.
    cfg.handeye_init.handeye_mode = HandEyeMode::EyeToHand;
    // Refine per-view robot pose se(3) deltas — encoder noise otherwise
    // biases hand-eye. Stage 4 uses these as initial robot-delta parameters
    // rather than mutating the canonical detected dataset.
    cfg.handeye_ba.refine_robot_poses = true;
    cfg.intrinsics.initial_cameras = Some(initial_cameras);
    cfg.intrinsics.initial_sensors = Some(initial_sensors);
    // The seeded K (fx=1800, cx/cy at tile center) is only a rough prior.
    // Let per-camera BA refine fx/fy/cx/cy + k1, and the Scheimpflug tilts.
    // k2 is held fixed because with this narrow FOV (~22°) and limited
    // per-view corner coverage the solver overfits k2 (values ±4) and the
    // resulting per-camera models no longer share a consistent rigid rig.
    cfg.intrinsics.fix_intrinsics_when_overridden = false;
    // Refine k1 + tangential distortion (p1, p2); fix k2, k3. The narrow FOV
    // doesn't support k2; tangential terms can absorb subtle asymmetry in
    // sensor mounting that otherwise leaks into the rig extrinsics.
    cfg.intrinsics.fix_distortion_in_percam_ba = DistortionFixMask {
        k1: false,
        k2: true,
        k3: true,
        p1: false,
        p2: false,
    };
    cfg.intrinsics.fix_scheimpflug = ScheimpflugFixMask {
        tilt_x: false,
        tilt_y: true,
    };
    // Freeze intrinsics + Scheimpflug in rig BA. Allowing tilts to drift
    // during rig BA produced wildly inconsistent tilts (e.g. +13° tilt_y)
    // without improving residuals; per-camera BA already nails tilt values
    // in a narrow -8°…-10° x_tilt band, so keep them frozen.
    cfg.rig.refine_intrinsics_in_rig_ba = false;
    cfg.rig.refine_scheimpflug_in_rig_ba = false;
    let robot_rot_sigma = cfg.handeye_ba.robot_rot_sigma;
    let robot_trans_sigma = cfg.handeye_ba.robot_trans_sigma;
    rig_session.set_config(cfg)?;

    let t0 = Instant::now();
    {
        use vision_calibration::rig_scheimpflug_handeye as rh;
        let step_t = Instant::now();
        rh::step_intrinsics_init_all(&mut rig_session, None)?;
        println!("  step_intrinsics_init_all: {:.2?}", step_t.elapsed());
        if let Some(cams) = &rig_session.state.per_cam_intrinsics {
            for (i, c) in cams.iter().enumerate() {
                let fb = rig_session
                    .state
                    .per_cam_used_fallback
                    .as_ref()
                    .map(|v| v[i])
                    .unwrap_or(false);
                println!(
                    "    [zhang] cam {i}{}: fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
                    if fb { " (fallback)" } else { "" },
                    c.k.fx,
                    c.k.fy,
                    c.k.cx,
                    c.k.cy
                );
            }
        }
        let step_t = Instant::now();
        rh::step_intrinsics_optimize_all(&mut rig_session, None)?;
        println!("  step_intrinsics_optimize_all: {:.2?}", step_t.elapsed());
        if let Some(errs) = &rig_session.state.per_cam_reproj_errors {
            for (i, e) in errs.iter().enumerate() {
                println!("    cam {i} intrinsics reproj = {e:?}");
            }
        }
        if let Some(cams) = &rig_session.state.per_cam_intrinsics {
            for (i, c) in cams.iter().enumerate() {
                println!(
                    "    cam {i}: fx={:.1} fy={:.1} cx={:.1} cy={:.1} k1={:+.4} k2={:+.4}",
                    c.k.fx, c.k.fy, c.k.cx, c.k.cy, c.dist.k1, c.dist.k2
                );
            }
        }
        if let Some(sens) = &rig_session.state.per_cam_sensors {
            for (i, s) in sens.iter().enumerate() {
                println!(
                    "    cam {i} tilt: tilt_x={:+.4} rad ({:+.3}°) tilt_y={:+.4} rad ({:+.3}°)",
                    s.tilt_x,
                    s.tilt_x.to_degrees(),
                    s.tilt_y,
                    s.tilt_y.to_degrees()
                );
            }
        }
        let step_t = Instant::now();
        rh::step_rig_init(&mut rig_session)?;
        println!("  step_rig_init: {:.2?}", step_t.elapsed());
        let step_t = Instant::now();
        rh::step_rig_optimize(&mut rig_session, None)?;
        println!(
            "  step_rig_optimize: {:.2?}, rig_reproj={:?}",
            step_t.elapsed(),
            rig_session.state.rig_ba_reproj_error
        );
        if let Some(per) = &rig_session.state.rig_ba_per_cam_reproj_errors {
            for (i, e) in per.iter().enumerate() {
                println!("    cam {i} rig reproj = {e:.3} px");
            }
        }
        if let Some(cams) = &rig_session.state.per_cam_intrinsics {
            for (i, c) in cams.iter().enumerate() {
                println!(
                    "    [after rig] cam {i}: fx={:.1} fy={:.1} cx={:.1} cy={:.1} k1={:+.4} k2={:+.4}",
                    c.k.fx, c.k.fy, c.k.cx, c.k.cy, c.dist.k1, c.dist.k2
                );
            }
        }
        if let Some(sens) = &rig_session.state.per_cam_sensors {
            for (i, s) in sens.iter().enumerate() {
                println!(
                    "    [after rig] cam {i} tilt: tilt_x={:+.4} ({:+.3}°) tilt_y={:+.4} ({:+.3}°)",
                    s.tilt_x,
                    s.tilt_x.to_degrees(),
                    s.tilt_y,
                    s.tilt_y.to_degrees()
                );
            }
        }
        let step_t = Instant::now();
        rh::step_handeye_init(&mut rig_session, None)?;
        println!("  step_handeye_init: {:.2?}", step_t.elapsed());
        let step_t = Instant::now();
        rh::step_handeye_optimize(&mut rig_session, None)?;
        println!("  step_handeye_optimize: {:.2?}", step_t.elapsed());
    }
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
    match rig_export.handeye_mode {
        HandEyeMode::EyeInHand => {
            let he = rig_export
                .gripper_se3_rig
                .expect("EyeInHand missing gripper_se3_rig");
            println!(
                "  gripper_se3_rig: |t|={:.4} m",
                he.translation.vector.norm()
            );
        }
        HandEyeMode::EyeToHand => {
            let he = rig_export
                .rig_se3_base
                .expect("EyeToHand missing rig_se3_base");
            let gt = rig_export
                .gripper_se3_target
                .expect("EyeToHand missing gripper_se3_target");
            println!(
                "  rig_se3_base:         |t|={:.4} m",
                he.translation.vector.norm()
            );
            println!(
                "  gripper_se3_target:   |t|={:.4} m",
                gt.translation.vector.norm()
            );
        }
    }
    print_robot_delta_summary(
        "stage 2 exported robot deltas",
        rig_export.robot_deltas.as_deref(),
    );

    // ─── Stage 3: rig laserline calibration ────────────────────────────────
    let laserline_dataset = RigLaserlineDataset::new(detected.laserline_views.clone(), NUM_CAMERAS)
        .context("build laserline dataset")?;
    // Per-view rig_se3_target in either mode:
    //   EyeInHand:  T_R_T_i = T_G_R^-1 * T_B_G_i^-1 * T_B_T
    //   EyeToHand:  T_R_T_i = T_R_B    * T_B_G_i    * T_G_T
    let mut rig_se3_target = Vec::new();
    for pose in &poses {
        if !pose.has_laser() {
            continue;
        }
        let base_se3_gripper = pose.base_se3_gripper();
        let rt = match rig_export.handeye_mode {
            HandEyeMode::EyeInHand => {
                let gripper_se3_rig = rig_export
                    .gripper_se3_rig
                    .expect("EyeInHand missing gripper_se3_rig");
                let base_se3_target = rig_export
                    .base_se3_target
                    .expect("EyeInHand missing base_se3_target");
                gripper_se3_rig.inverse() * base_se3_gripper.inverse() * base_se3_target
            }
            HandEyeMode::EyeToHand => {
                let rig_se3_base = rig_export
                    .rig_se3_base
                    .expect("EyeToHand missing rig_se3_base");
                let gripper_se3_target = rig_export
                    .gripper_se3_target
                    .expect("EyeToHand missing gripper_se3_target");
                rig_se3_base * base_se3_gripper * gripper_se3_target
            }
        };
        rig_se3_target.push(rt);
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
    laser_cfg.laser_residual_type = LaserlineResidualType::PointToPlane;
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

    // ─── Stage 4: joint rig + hand-eye + laser-plane BA ────────────────────
    //
    // Warm-starts come from stages 2 (rig + Scheimpflug + hand-eye) and 3
    // (per-camera laser planes, upstream frozen). The joint BA relaxes all
    // upstream parameters and adds per-pixel laser residuals (laser line
    // distance in undistorted pixel space) to constrain intrinsics,
    // extrinsics, hand-eye, target reference pose, and laser planes all
    // together under a single cost.
    //
    // Residual type: `LineDistNormalized` — exactly the "undistort laser
    // pixel, intersect the laser plane with the target plane in camera
    // frame, project to z=1 plane, take perpendicular distance" formulation.
    print_frame_table();
    let joint_views: Vec<RigHandeyeLaserlineView> = detected.joint_views.clone();
    let joint_dataset =
        RigHandeyeLaserlineDataset::new(joint_views, NUM_CAMERAS, rig_export.handeye_mode)?;

    let handeye = match rig_export.handeye_mode {
        HandEyeMode::EyeInHand => rig_export
            .gripper_se3_rig
            .expect("EyeInHand missing gripper_se3_rig"),
        HandEyeMode::EyeToHand => rig_export
            .rig_se3_base
            .expect("EyeToHand missing rig_se3_base"),
    };
    let target_ref = match rig_export.handeye_mode {
        HandEyeMode::EyeInHand => rig_export
            .base_se3_target
            .expect("EyeInHand missing base_se3_target"),
        HandEyeMode::EyeToHand => rig_export
            .gripper_se3_target
            .expect("EyeToHand missing gripper_se3_target"),
    };
    // Convert cam_se3_rig to cam_to_rig (T_rig_from_cam) for the optim block.
    let cam_to_rig: Vec<_> = rig_export.cam_se3_rig.iter().map(|t| t.inverse()).collect();

    let joint_initial = RigHandeyeLaserlineParams {
        cameras: rig_export.cameras.clone(),
        sensors: rig_export.sensors.clone(),
        cam_to_rig,
        handeye,
        target_ref,
        planes_cam: laser_export.laser_planes_cam.clone(),
    };
    let initial_robot_deltas = rig_export.robot_deltas.clone();
    let (joint_initial_mean, joint_initial_stats) = evaluate_rig_handeye_laserline(
        &joint_dataset,
        &joint_initial,
        initial_robot_deltas.as_deref(),
    );
    println!(
        "  stage 4 initial eval on canonical observations: {:.4} px",
        joint_initial_mean
    );
    print_joint_stats("initial", &joint_initial_stats);

    // Fix distortion to radial-only (k1, k2); hold k3/p1/p2 at their warm-
    // start values. Fix Scheimpflug tilts (they were converged per camera).
    // Free everything else, except reference camera extrinsic for gauge.
    let cam_fix = CameraFixMask {
        intrinsics: vision_calibration_core::IntrinsicsFixMask::all_free(),
        distortion: DistortionFixMask {
            k1: false,
            k2: true, // narrow FOV; keep k2 at warm-start value
            k3: true,
            p1: true,
            p2: true,
        },
    };
    let mut joint_opts = RigHandeyeLaserlineSolveOptions {
        // PointToPlane residual (meters): a 1 mm error is a 0.001 residual;
        // target reprojection is ~1-20 px, so the calib term naturally
        // drives geometry while the laser term regularizes. Switching to
        // LineDistNormalized is fruitful only after the joint geometry is
        // already well-conditioned (reproj < 1 px).
        laser_residual_type: LaserlineResidualType::PointToPlane,
        fix_intrinsics: vec![cam_fix; NUM_CAMERAS],
        fix_extrinsics: (0..NUM_CAMERAS).map(|i| i == 0).collect(),
        ..Default::default()
    };
    // Balance calib (pixel-scale) against PointToPlane laser (meter-scale).
    // With ~10k calib residuals at a few-px reproj and ~2k laser residuals
    // at a few-cm initial residual, a laser_weight around 1e4 makes laser
    // influential without destabilizing the calib term. The residual
    // magnitudes on this particular real dataset (20 px rig reproj,
    // 70 mm laser) keep the joint BA from escaping the warm-start basin;
    // getting below 1 px reproj / 0.1 mm laser σ needs improvements
    // upstream (shared-intrinsics rig or sub-pixel detection).
    joint_opts.laser_weight = 1e4;
    joint_opts.calib_weight = 1.0;
    joint_opts.refine_robot_poses = true;
    joint_opts.robot_rot_sigma = robot_rot_sigma;
    joint_opts.robot_trans_sigma = robot_trans_sigma;
    joint_opts.initial_robot_deltas = initial_robot_deltas;
    // Scheimpflug tilts converge well in per-camera BA — don't refit here.
    joint_opts.fix_scheimpflug = vec![
        vision_calibration_optim::ScheimpflugFixMask {
            tilt_x: true,
            tilt_y: true,
        };
        NUM_CAMERAS
    ];

    let backend_opts = BackendSolveOptions {
        max_iters: 30,
        verbosity: 1,
        ..Default::default()
    };

    let t0 = Instant::now();
    let joint_est = optimize_rig_handeye_laserline(
        joint_dataset,
        joint_initial.clone(),
        joint_opts,
        backend_opts,
    )?;
    println!("stage 4 (joint BA): {:.2?}", t0.elapsed());
    println!("  solve final_cost = {:.3e}", joint_est.report.final_cost);
    println!(
        "  mean reproj after joint BA: {:.4} px",
        joint_est.mean_reproj_error_px
    );
    for (i, s) in joint_est.per_cam_stats.iter().enumerate() {
        println!(
            "    cam {i}: reproj={:.4}px max={:.2}px hist={:?}  laser={:.5}m ({:.1}μm) max={:.5}m hist={:?}  laser_px={:.4}px max={:.2}px",
            s.mean_reproj_error_px,
            s.max_reproj_error_px,
            s.reproj_histogram_px,
            s.mean_laser_err_m,
            s.mean_laser_err_m * 1e6,
            s.max_laser_err_m,
            s.laser_histogram_m,
            s.mean_laser_err_px,
            s.max_laser_err_px
        );
    }
    print_param_deltas(&joint_initial, &joint_est.params);
    print_robot_delta_summary("joint BA robot deltas", joint_est.robot_deltas.as_deref());
    for (i, p) in joint_est.planes_rig.iter().enumerate() {
        let n = p.normal.into_inner();
        println!(
            "  plane (rig) {i}: n=({:+.4}, {:+.4}, {:+.4}), d={:+.4} m",
            n.x, n.y, n.z, p.distance
        );
    }

    if let Ok(out_dir) = std::env::var("PUZZLE_VIEWER_OUT") {
        let out_dir = PathBuf::from(out_dir);
        puzzle_viewer::write_viewer_artifacts(puzzle_viewer::ViewerExportInput {
            out_dir: &out_dir,
            data_dir: &data_dir,
            poses: &poses,
            tile_w,
            tile_h,
            detected: &detected,
            rig_export: &rig_export,
            joint_initial: &joint_initial,
            joint_initial_stats: &joint_initial_stats,
            joint_est: &joint_est,
        })?;
        println!(
            "viewer artifacts: {}",
            out_dir.join("viewer_manifest.json").display()
        );
    }

    // ─── Stage 5: pixel → gripper point demo ───────────────────────────────
    // Use the first pose that carried laser observations to anchor the query
    // — in EyeToHand, the gripper-frame mapping depends on the robot pose.
    let ref_pose = poses
        .iter()
        .find(|p| p.has_laser())
        .map(|p| p.base_se3_gripper());
    println!("\nsample pixel→gripper mappings:");
    for cam in 0..NUM_CAMERAS {
        let pixel = Pt2::new(320.0, 240.0); // center-ish
        match pixel_to_gripper_point(
            cam,
            pixel,
            &rig_export,
            &laser_export.laser_planes_rig,
            ref_pose,
        ) {
            Ok(p) => println!(
                "  cam{cam} ({:.1},{:.1}) → ({:.3}, {:.3}, {:.3}) m (gripper)",
                pixel.x, pixel.y, p.x, p.y, p.z
            ),
            Err(e) => println!("  cam{cam}: {e}"),
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct DetectedDatasets {
    handeye_views: Vec<RigView<RobotPoseMeta>>,
    laserline_views: Vec<RigLaserlineView>,
    joint_views: Vec<RigHandeyeLaserlineView>,
}

fn print_detection_diagnostics(detected: &DetectedDatasets) {
    let mut target_views_per_cam = [0usize; NUM_CAMERAS];
    let mut target_points_per_cam = [0usize; NUM_CAMERAS];
    let mut laser_views_per_cam = [0usize; NUM_CAMERAS];
    let mut laser_pixels_per_cam = [0usize; NUM_CAMERAS];
    let mut per_pose_target_cams = Vec::with_capacity(detected.joint_views.len());
    let mut target_min_x = [f64::INFINITY; NUM_CAMERAS];
    let mut target_max_x = [f64::NEG_INFINITY; NUM_CAMERAS];
    let mut target_min_y = [f64::INFINITY; NUM_CAMERAS];
    let mut target_max_y = [f64::NEG_INFINITY; NUM_CAMERAS];

    for view in &detected.joint_views {
        let mut cams_this_pose = 0usize;
        for (cam_idx, obs) in view.obs.cameras.iter().enumerate() {
            if let Some(obs) = obs {
                cams_this_pose += 1;
                target_views_per_cam[cam_idx] += 1;
                target_points_per_cam[cam_idx] += obs.points_2d.len();
                for p in &obs.points_3d {
                    target_min_x[cam_idx] = target_min_x[cam_idx].min(p.x);
                    target_max_x[cam_idx] = target_max_x[cam_idx].max(p.x);
                    target_min_y[cam_idx] = target_min_y[cam_idx].min(p.y);
                    target_max_y[cam_idx] = target_max_y[cam_idx].max(p.y);
                }
            }
        }
        per_pose_target_cams.push(cams_this_pose);
        for (cam_idx, pixels) in view.obs.laser_pixels.iter().enumerate() {
            if let Some(pixels) = pixels {
                laser_views_per_cam[cam_idx] += 1;
                laser_pixels_per_cam[cam_idx] += pixels.len();
            }
        }
    }

    println!("  detection diagnostics:");
    for cam in 0..NUM_CAMERAS {
        println!(
            "    cam {cam}: target_views={} target_pts={} target_x=[{:.4},{:.4}]m target_y=[{:.4},{:.4}]m laser_views={} laser_pts={}",
            target_views_per_cam[cam],
            target_points_per_cam[cam],
            target_min_x[cam],
            target_max_x[cam],
            target_min_y[cam],
            target_max_y[cam],
            laser_views_per_cam[cam],
            laser_pixels_per_cam[cam]
        );
    }
    println!("    target cameras per pose: {per_pose_target_cams:?}");
}

fn print_frame_table() {
    println!("  frame table:");
    println!("    cam_se3_rig = T_C_R (exported by stage 2)");
    println!("    cam_to_rig  = T_R_C (joint optimizer parameter)");
    println!("    EyeToHand   = T_C_T = T_C_R * T_R_B * T_B_G * T_G_T");
    println!("    robot delta = T_B_G_corr = exp(delta) * T_B_G");
}

fn print_joint_stats(label: &str, stats: &[RigHandeyeLaserlinePerCamStats]) {
    for (i, s) in stats.iter().enumerate() {
        println!(
            "    [{label}] cam {i}: reproj={:.4}px max={:.2}px hist={:?} laser={:.5}m ({:.1}μm) max={:.5}m hist={:?} laser_px={:.4}px max={:.2}px counts=({}/{})",
            s.mean_reproj_error_px,
            s.max_reproj_error_px,
            s.reproj_histogram_px,
            s.mean_laser_err_m,
            s.mean_laser_err_m * 1e6,
            s.max_laser_err_m,
            s.laser_histogram_m,
            s.mean_laser_err_px,
            s.max_laser_err_px,
            s.reproj_count,
            s.laser_count
        );
    }
}

fn print_param_deltas(
    initial: &RigHandeyeLaserlineParams,
    final_params: &RigHandeyeLaserlineParams,
) {
    let mut max_fx = 0.0f64;
    let mut max_c = 0.0f64;
    let mut max_dist = 0.0f64;
    let mut max_tilt = 0.0f64;
    let mut max_extr_t = 0.0f64;
    let mut max_extr_rot = 0.0f64;
    let mut max_plane_angle = 0.0f64;
    let mut max_plane_d = 0.0f64;

    for (a, b) in initial.cameras.iter().zip(final_params.cameras.iter()) {
        max_fx = max_fx.max((a.k.fx - b.k.fx).abs().max((a.k.fy - b.k.fy).abs()));
        max_c = max_c.max((a.k.cx - b.k.cx).abs().max((a.k.cy - b.k.cy).abs()));
        max_dist = max_dist
            .max((a.dist.k1 - b.dist.k1).abs())
            .max((a.dist.k2 - b.dist.k2).abs())
            .max((a.dist.k3 - b.dist.k3).abs())
            .max((a.dist.p1 - b.dist.p1).abs())
            .max((a.dist.p2 - b.dist.p2).abs());
    }
    for (a, b) in initial.sensors.iter().zip(final_params.sensors.iter()) {
        max_tilt = max_tilt
            .max((a.tilt_x - b.tilt_x).abs())
            .max((a.tilt_y - b.tilt_y).abs());
    }
    for (a, b) in initial
        .cam_to_rig
        .iter()
        .zip(final_params.cam_to_rig.iter())
    {
        let d = a.inverse() * *b;
        max_extr_t = max_extr_t.max(d.translation.vector.norm());
        max_extr_rot = max_extr_rot.max(d.rotation.angle());
    }
    for (a, b) in initial
        .planes_cam
        .iter()
        .zip(final_params.planes_cam.iter())
    {
        let dot = a
            .normal
            .into_inner()
            .dot(&b.normal.into_inner())
            .clamp(-1.0, 1.0);
        max_plane_angle = max_plane_angle.max(dot.acos());
        max_plane_d = max_plane_d.max((a.distance - b.distance).abs());
    }
    let handeye_delta = initial.handeye.inverse() * final_params.handeye;
    let target_delta = initial.target_ref.inverse() * final_params.target_ref;
    println!(
        "  joint parameter deltas: max_f={max_fx:.4}px max_c={max_c:.4}px max_dist={max_dist:.3e} max_tilt={:.4}rad",
        max_tilt
    );
    println!(
        "    extrinsics max: rot={:.4}rad trans={:.5}m; handeye: rot={:.4}rad trans={:.5}m; target: rot={:.4}rad trans={:.5}m",
        max_extr_rot,
        max_extr_t,
        handeye_delta.rotation.angle(),
        handeye_delta.translation.vector.norm(),
        target_delta.rotation.angle(),
        target_delta.translation.vector.norm()
    );
    println!(
        "    laser planes max: normal_angle={:.4}rad distance={:.5}m",
        max_plane_angle, max_plane_d
    );
}

fn print_robot_delta_summary(label: &str, deltas: Option<&[[f64; 6]]>) {
    let Some(deltas) = deltas else {
        println!("  {label}: none");
        return;
    };
    let mut max_rot = 0.0f64;
    let mut max_trans = 0.0f64;
    for delta in deltas {
        let rot = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let trans = (delta[3] * delta[3] + delta[4] * delta[4] + delta[5] * delta[5]).sqrt();
        max_rot = max_rot.max(rot);
        max_trans = max_trans.max(trans);
    }
    println!(
        "  {label}: count={} max_rot={:.4}rad ({:.3}°) max_trans={:.5}m",
        deltas.len(),
        max_rot,
        max_rot.to_degrees(),
        max_trans
    );
}

fn build_datasets(data_dir: &Path, poses: &[PoseEntry]) -> Result<DetectedDatasets> {
    let mut handeye_views = Vec::new();
    let mut laser_views = Vec::new();
    let mut joint_views = Vec::new();

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

        let mut laser_pixels: Vec<Option<Vec<Pt2>>> = vec![None; NUM_CAMERAS];
        if pose.has_laser() {
            let laser_img = load_gray(&data_dir.join(&pose.laser_image))
                .with_context(|| format!("pose {i} laser"))?;
            let laser_tiles = split_horizontal(&laser_img, NUM_CAMERAS);
            laser_pixels = laser_tiles
                .iter()
                .map(|tile| {
                    let pts = detect_laser(tile);
                    if pts.is_empty() { None } else { Some(pts) }
                })
                .collect();
            laser_views.push(RigLaserlineView {
                cameras: cam_obs,
                laser_pixels: laser_pixels.clone(),
            });
        }
        joint_views.push(RigHandeyeLaserlineView {
            obs: RigLaserlineView {
                cameras: handeye_views
                    .last()
                    .expect("handeye view just pushed")
                    .obs
                    .cameras
                    .clone(),
                laser_pixels,
            },
            meta: RobotPoseMeta {
                base_se3_gripper: pose.base_se3_gripper(),
            },
        });

        // Drop per-pose images to limit peak memory.
        let _ = target_img;
    }

    Ok(DetectedDatasets {
        handeye_views,
        laserline_views: laser_views,
        joint_views,
    })
}
