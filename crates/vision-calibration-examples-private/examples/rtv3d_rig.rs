//! End-to-end rig calibration on the rtv3d dataset.
//!
//! The rtv3d sensor is six laser-plane-triangulation devices: each device is a
//! Scheimpflug-optics camera plus a laser plane projector. A pose stores all
//! six camera views as one 4320×540 horizontal strip (6 tiles of 720×540).
//! The calibration target is a ChArUco 22×22 board (DICT_4X4_1000,
//! marker_size_rel 0.75). Hand-eye mode is EyeToHand — the empirical
//! convention check (3× lower rigid-robot hand-eye residual) overrules the
//! EyeInHand claim in the dataset's own `dataset.json`.
//!
//! Pipeline:
//! 1. Detect the ChArUco board in each of the 6 per-camera tiles per pose;
//!    detect laser lines on `double_snap` poses (skipped when the dataset
//!    ships no laser images).
//! 2. `RigHandeyeProblem` with `SensorMode::Scheimpflug`, EyeToHand
//!    (intrinsics → rig → hand-eye).
//! 3. `RigLaserlineDeviceProblem` consuming the frozen upstream rig
//!    calibration to recover the 6 laser planes (skipped without laser data).
//! 4. Joint rig + hand-eye + laser-plane BA (skipped without laser data).
//! 5. Compare everything against the legacy-system oracle in
//!    `artifacts.json` (camera 5 is degenerate there: fx=51, 127 px reproj).
//!
//! Environment:
//! - `RTV3D_DATA_DIR`  — dataset directory (default `privatedata/rtv3d`).
//! - `CELL_SIZE_MM`    — ChArUco cell pitch (default 5.2). The dataset ships
//!   conflicting specs (4.8 vs 5.2); run both and let the extrinsic
//!   translation norms / plane distances against the oracle discriminate.
//! - `RTV3D_SEED`      — `generic` (default; fx=2000, centered pp, zero tilt)
//!   or `oracle` (seed K/dist/tilt from artifacts.json; cam 5 falls back to
//!   the generic seed because the oracle entry is degenerate).
//!
//! Run:
//! `RTV3D_DATA_DIR=privatedata/rtv3d cargo run --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example rtv3d_rig
//! --release`

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vision_calibration::{
    rig_handeye::{
        self as rh, RigHandeyeConfig, RigHandeyeIntrinsicsManualInit, RigHandeyeProblem, SensorMode,
    },
    rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
    },
    session::CalibrationSession,
};
use vision_calibration_core::{
    BrownConrady5, CameraFixMask, CorrespondenceView, DistortionFixMask, FrameRef, FxFyCxCySkew,
    ImageManifest, Iso3, PixelRect, Pt2, RigDataset, RigView, RigViewObs, ScheimpflugParams,
};
use vision_calibration_examples_private::{
    PoseEntry, detect_charuco, detect_laser, load_gray, load_poses, split_horizontal,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeMode, LaserPlane, LaserlineResidualType,
    RigHandeyeLaserlineDataset, RigHandeyeLaserlineParams, RigHandeyeLaserlineView,
    RigLaserlineDataset, RigLaserlineView, RobotPoseMeta, RobustLoss, ScheimpflugFixMask,
    evaluate_rig_handeye_laserline, optimize_rig_handeye_laserline,
};

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 22;
const BOARD_COLS: u32 = 22;
const MARKER_SIZE_REL: f32 = 0.75;
const DICTIONARY: &str = "DICT_4X4_1000";

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("RTV3D_DATA_DIR").unwrap_or_else(|_| "privatedata/rtv3d".to_string()),
    );
    let cell_size_mm: f64 = std::env::var("CELL_SIZE_MM")
        .ok()
        .map(|s| s.parse())
        .transpose()
        .context("CELL_SIZE_MM must be a number")?
        .unwrap_or(5.2);
    let seed_mode = std::env::var("RTV3D_SEED").unwrap_or_else(|_| "generic".to_string());
    println!("data dir = {}", data_dir.display());
    println!("cell size = {cell_size_mm} mm, seed = {seed_mode}");

    let oracle = load_oracle(&data_dir.join("artifacts.json"))
        .context("load artifacts.json oracle")
        .map_err(|e| {
            eprintln!("warning: {e:#}");
            e
        })
        .ok();

    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!("loaded {} poses", poses.len());

    let first_target = load_gray(&data_dir.join(&poses[0].target_image))?;
    let tile_w = first_target.width() / NUM_CAMERAS as u32;
    let tile_h = first_target.height();
    println!("tile size: {tile_w}x{tile_h}");
    drop(first_target);

    // ─── Stage 1: detect targets and laser lines ───────────────────────────
    let t0 = Instant::now();
    let detected = build_datasets(&data_dir, &poses, cell_size_mm).context("detect stage")?;
    println!(
        "stage 1 (detect): {:.2?} → {} handeye views, {} laser views",
        t0.elapsed(),
        detected.handeye_views.len(),
        detected.laserline_views.len(),
    );
    print_detection_diagnostics(&detected);

    // ─── Stage 2: rig + scheimpflug + hand-eye ─────────────────────────────
    let handeye_dataset = RigDataset::new(detected.handeye_views.clone(), NUM_CAMERAS)?;
    let mut rig_session = CalibrationSession::<RigHandeyeProblem>::with_description("rtv3d_rig");
    rig_session.set_input(handeye_dataset)?;

    let manual_init = build_intrinsics_seed(&seed_mode, oracle.as_ref(), tile_w, tile_h)?;

    let mut cfg = RigHandeyeConfig::default();
    cfg.solver.max_iters = 200;
    cfg.solver.verbosity = 1;
    cfg.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
    // EyeToHand: established by the empirical convention check (3× lower
    // rigid-robot hand-eye residual than EyeInHand, overruling the EyeInHand
    // claim in `dataset.json`). `RTV3D_HANDEYE=eye_in_hand` re-runs that
    // comparison.
    cfg.handeye_init.handeye_mode = match std::env::var("RTV3D_HANDEYE").as_deref() {
        Ok("eye_in_hand") => HandEyeMode::EyeInHand,
        _ => HandEyeMode::EyeToHand,
    };
    // `RTV3D_REFINE_ROBOT=0` disables robot-pose deltas. With rigid robot
    // poses the hand-eye residual is directly sensitive to the board scale,
    // which makes the cell-size A/B check sharp (robot translations anchor
    // absolute scale).
    cfg.handeye_ba.refine_robot_poses =
        std::env::var("RTV3D_REFINE_ROBOT").map_or(true, |v| v != "0");
    // Scheimpflug: both tilts free (the legacy system also frees tau_x/tau_y;
    // oracle tilts reach 0.27 rad). Distortion: k1, k2 free; k3 + tangential
    // fixed, matching the legacy config (`fix_k3: true, fix_tangent: true`).
    cfg.sensor = SensorMode::Scheimpflug {
        init_tilt_x: 0.0,
        init_tilt_y: 0.0,
        fix_scheimpflug_in_intrinsics: ScheimpflugFixMask {
            tilt_x: false,
            tilt_y: false,
        },
        distortion_mask_in_percam_ba: DistortionFixMask {
            k1: false,
            k2: false,
            k3: true,
            p1: true,
            p2: true,
        },
        refine_scheimpflug_in_rig_ba: false,
    };
    cfg.rig.refine_intrinsics_in_rig_ba = false;
    let robot_rot_sigma = cfg.handeye_ba.robot_rot_sigma;
    let robot_trans_sigma = cfg.handeye_ba.robot_trans_sigma;
    rig_session.set_config(cfg)?;

    let t0 = Instant::now();
    let intr_opt;
    {
        let step_t = Instant::now();
        let _ = rh::step_intrinsics_init_all_with_seed(&mut rig_session, manual_init, None)?;
        println!(
            "  step_intrinsics_init_all_with_seed: {:.2?}",
            step_t.elapsed()
        );
        let step_t = Instant::now();
        intr_opt = rh::step_intrinsics_optimize_all(&mut rig_session, None)?;
        println!("  step_intrinsics_optimize_all: {:.2?}", step_t.elapsed());
        for (i, e) in intr_opt.per_cam_reproj_errors.iter().enumerate() {
            println!("    cam {i} intrinsics reproj = {e:?}");
        }
        for (i, c) in intr_opt.per_cam_intrinsics.iter().enumerate() {
            println!(
                "    cam {i}: fx={:.1} fy={:.1} cx={:.1} cy={:.1} k1={:+.4} k2={:+.4}",
                c.k.fx, c.k.fy, c.k.cx, c.k.cy, c.dist.k1, c.dist.k2
            );
        }
        if let Some(sens) = &intr_opt.per_cam_sensors {
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
        let _rig_init = rh::step_rig_init(&mut rig_session)?;
        println!("  step_rig_init: {:.2?}", step_t.elapsed());
        let step_t = Instant::now();
        let rig_opt = rh::step_rig_optimize(&mut rig_session, None)?;
        println!(
            "  step_rig_optimize: {:.2?}, rig_reproj={:.4}",
            step_t.elapsed(),
            rig_opt.mean_reproj_error
        );
        for (i, e) in rig_opt.per_cam_reproj_errors.iter().enumerate() {
            println!("    cam {i} rig reproj = {e:.3} px");
        }
        let step_t = Instant::now();
        let _he_init = rh::step_handeye_init(&mut rig_session, None)?;
        println!("  step_handeye_init: {:.2?}", step_t.elapsed());
        let step_t = Instant::now();
        let _he_opt = rh::step_handeye_optimize(&mut rig_session, None)?;
        println!("  step_handeye_optimize: {:.2?}", step_t.elapsed());
    }
    println!(
        "stage 2 (rig + scheimpflug + hand-eye): {:.2?}",
        t0.elapsed()
    );

    let mut rig_export = rig_session.export()?;
    rig_export.image_manifest = Some(build_image_manifest(&poses, tile_w, tile_h));
    let export_path = data_dir.join("export.json");
    std::fs::write(
        &export_path,
        serde_json::to_string_pretty(&rig_export).context("serialize rig export")?,
    )
    .with_context(|| format!("write {}", export_path.display()))?;
    println!("  wrote {}", export_path.display());
    println!(
        "  mean reproj error:   {:.4} px",
        rig_export.mean_reproj_error
    );
    for (i, err) in rig_export.per_cam_reproj_errors.iter().enumerate() {
        println!("    camera {i}: {err:.4} px");
    }
    if let Some(he) = &rig_export.gripper_se3_rig {
        println!(
            "  gripper_se3_rig: |t|={:.4} m",
            he.translation.vector.norm()
        );
    }
    if let Some(he) = &rig_export.rig_se3_base {
        println!("  rig_se3_base: |t|={:.4} m", he.translation.vector.norm());
    }

    // ─── Stage 3: rig laserline calibration (needs laser images) ───────────
    let mut laser_planes_cam: Option<Vec<LaserPlane>> = None;
    let mut joint_stats_sigma_mm: Option<Vec<f64>> = None;
    let mut joint_reproj_px: Option<Vec<f64>> = None;
    if detected.laserline_views.is_empty() {
        println!("stage 3 (rig laserline): skipped — no laser observations in this dataset");
    } else {
        let laserline_dataset =
            RigLaserlineDataset::new(detected.laserline_views.clone(), NUM_CAMERAS)
                .context("build laserline dataset")?;
        // Per-view rig_se3_target:
        //   EyeInHand:  T_R_T_i = T_G_R^-1 * T_B_G_i^-1 * T_B_T
        //   EyeToHand:  T_R_T_i = T_R_B    * T_B_G_i    * T_G_T
        let mut rig_se3_target = Vec::new();
        for &pose_idx in &detected.laser_pose_indices {
            let base_se3_gripper = poses[pose_idx].base_se3_gripper();
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

        let upstream = rig_export
            .to_upstream_calibration(rig_se3_target)
            .context("rig handeye export must be Scheimpflug for laserline upstream")?;
        let laserline_input = RigLaserlineDeviceInput {
            dataset: laserline_dataset,
            upstream,
            initial_planes_cam: None,
        };

        let mut laser_session =
            CalibrationSession::<RigLaserlineDeviceProblem>::with_description("rtv3d_laser");
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
                "  camera {i}: reproj={:.4}px, laser={:.4}",
                stats.mean_reproj_error, stats.mean_laser_error
            );
        }
        for (i, p) in laser_export.laser_planes_cam.iter().enumerate() {
            let n = p.normal.into_inner();
            println!(
                "  plane (cam) {i}: n=({:+.4}, {:+.4}, {:+.4}), d={:+.4} m",
                n.x, n.y, n.z, p.distance
            );
        }
        // ─── Stage 4: joint rig + hand-eye + laser-plane BA ────────────────
        let joint_dataset = RigHandeyeLaserlineDataset::new(
            detected.joint_views.clone(),
            NUM_CAMERAS,
            rig_export.handeye_mode,
        )?;
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
        let cam_to_rig: Vec<_> = rig_export.cam_se3_rig.iter().map(|t| t.inverse()).collect();
        let scheimpflug_sensors = rig_export
            .sensors
            .clone()
            .context("rig handeye export missing Scheimpflug sensors")?;
        let joint_initial = RigHandeyeLaserlineParams {
            cameras: rig_export.cameras.clone(),
            sensors: scheimpflug_sensors,
            cam_to_rig,
            handeye,
            target_ref,
            planes_cam: laser_export.laser_planes_cam.clone(),
        };
        let initial_robot_deltas = rig_export.robot_deltas.clone();
        let (joint_initial_mean, _) = evaluate_rig_handeye_laserline(
            &joint_dataset,
            &joint_initial,
            initial_robot_deltas.as_deref(),
        );
        println!("  stage 4 initial eval: {:.4} px", joint_initial_mean);

        let cam_fix = CameraFixMask {
            intrinsics: vision_calibration_core::IntrinsicsFixMask::all_free(),
            distortion: DistortionFixMask {
                k1: false,
                k2: false,
                k3: true,
                p1: true,
                p2: true,
            },
        };
        let mut joint_opts = vision_calibration_optim::RigHandeyeLaserlineSolveOptions {
            laser_residual_type: LaserlineResidualType::PointToPlane,
            fix_intrinsics: vec![cam_fix; NUM_CAMERAS],
            fix_extrinsics: (0..NUM_CAMERAS).map(|i| i == 0).collect(),
            ..Default::default()
        };
        joint_opts.laser_weight = 1e4;
        joint_opts.calib_weight = 1.0;
        joint_opts.refine_robot_poses = true;
        joint_opts.robot_rot_sigma = robot_rot_sigma;
        joint_opts.robot_trans_sigma = robot_trans_sigma;
        joint_opts.initial_robot_deltas = initial_robot_deltas;
        // Tilts converged in per-camera BA; hold them in the joint solve.
        joint_opts.fix_scheimpflug = vec![
            ScheimpflugFixMask {
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
        println!(
            "  mean reproj after joint BA: {:.4} px",
            joint_est.mean_reproj_error_px
        );
        let mut sigmas = Vec::with_capacity(NUM_CAMERAS);
        let mut reprojs = Vec::with_capacity(NUM_CAMERAS);
        for (i, s) in joint_est.per_cam_stats.iter().enumerate() {
            println!(
                "    cam {i}: reproj={:.4}px max={:.2}px  laser={:.4}mm max={:.4}mm (n={})",
                s.mean_reproj_error_px,
                s.max_reproj_error_px,
                s.mean_laser_err_m * 1e3,
                s.max_laser_err_m * 1e3,
                s.laser_count
            );
            sigmas.push(s.mean_laser_err_m * 1e3);
            reprojs.push(s.mean_reproj_error_px);
        }
        joint_stats_sigma_mm = Some(sigmas);
        joint_reproj_px = Some(reprojs);
        laser_planes_cam = Some(joint_est.params.planes_cam.clone());
    }

    // ─── Oracle comparison ──────────────────────────────────────────────────
    if let Some(oracle) = &oracle {
        compare_to_oracle(
            oracle,
            &rig_export,
            &intr_opt.per_cam_reproj_errors,
            joint_reproj_px.as_deref(),
            laser_planes_cam.as_deref(),
            joint_stats_sigma_mm.as_deref(),
        );
    } else {
        println!("\nno oracle (artifacts.json) — skipping comparison");
    }

    Ok(())
}

// ───────────────────────────── detection ─────────────────────────────────

#[derive(Debug, Clone)]
struct DetectedDatasets {
    handeye_views: Vec<RigView<RobotPoseMeta>>,
    laserline_views: Vec<RigLaserlineView>,
    joint_views: Vec<RigHandeyeLaserlineView>,
    laser_pose_indices: Vec<usize>,
}

fn build_datasets(
    data_dir: &Path,
    poses: &[PoseEntry],
    cell_size_mm: f64,
) -> Result<DetectedDatasets> {
    let mut handeye_views = Vec::new();
    let mut laser_views = Vec::new();
    let mut joint_views = Vec::new();
    let mut laser_pose_indices = Vec::new();

    for (i, pose) in poses.iter().enumerate() {
        let target_img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("pose {i} target"))?;
        let target_tiles = split_horizontal(&target_img, NUM_CAMERAS);

        let mut cam_obs: Vec<Option<CorrespondenceView>> = Vec::with_capacity(NUM_CAMERAS);
        for (cam_idx, tile) in target_tiles.iter().enumerate() {
            match detect_charuco(
                tile,
                BOARD_ROWS,
                BOARD_COLS,
                cell_size_mm,
                MARKER_SIZE_REL,
                DICTIONARY,
            ) {
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
        // poses.json may list laser_image names for poses whose laser files
        // were never shipped — guard on file existence, not just the snap type.
        let laser_path = data_dir.join(&pose.laser_image);
        if pose.has_laser() && laser_path.is_file() {
            let laser_img = load_gray(&laser_path).with_context(|| format!("pose {i} laser"))?;
            let laser_tiles = split_horizontal(&laser_img, NUM_CAMERAS);
            laser_pixels = laser_tiles
                .iter()
                .map(|tile| {
                    let pts = detect_laser(tile);
                    if pts.is_empty() { None } else { Some(pts) }
                })
                .collect();
            laser_views.push(RigLaserlineView {
                cameras: cam_obs.clone(),
                laser_pixels: laser_pixels.clone(),
            });
            laser_pose_indices.push(i);
        }
        joint_views.push(RigHandeyeLaserlineView {
            obs: RigLaserlineView {
                cameras: cam_obs,
                laser_pixels,
            },
            meta: RobotPoseMeta {
                base_se3_gripper: pose.base_se3_gripper(),
            },
        });
    }

    Ok(DetectedDatasets {
        handeye_views,
        laserline_views: laser_views,
        joint_views,
        laser_pose_indices,
    })
}

fn print_detection_diagnostics(detected: &DetectedDatasets) {
    let mut target_views = [0usize; NUM_CAMERAS];
    let mut target_points = [0usize; NUM_CAMERAS];
    let mut laser_views = [0usize; NUM_CAMERAS];
    let mut laser_pixels = [0usize; NUM_CAMERAS];
    for view in &detected.joint_views {
        for (cam, obs) in view.obs.cameras.iter().enumerate() {
            if let Some(obs) = obs {
                target_views[cam] += 1;
                target_points[cam] += obs.points_2d.len();
            }
        }
        for (cam, px) in view.obs.laser_pixels.iter().enumerate() {
            if let Some(px) = px {
                laser_views[cam] += 1;
                laser_pixels[cam] += px.len();
            }
        }
    }
    println!("  detection diagnostics:");
    for cam in 0..NUM_CAMERAS {
        println!(
            "    cam {cam}: target_views={} target_pts={} laser_views={} laser_pts={}",
            target_views[cam], target_points[cam], laser_views[cam], laser_pixels[cam]
        );
    }
}

// ───────────────────────────── seeding ───────────────────────────────────

fn generic_intrinsics(tile_w: u32, tile_h: u32) -> FxFyCxCySkew<f64> {
    FxFyCxCySkew {
        fx: 2000.0,
        fy: 2000.0,
        cx: (tile_w as f64) * 0.5,
        cy: (tile_h as f64) * 0.5,
        skew: 0.0,
    }
}

fn build_intrinsics_seed(
    seed_mode: &str,
    oracle: Option<&Oracle>,
    tile_w: u32,
    tile_h: u32,
) -> Result<RigHandeyeIntrinsicsManualInit> {
    let zero_dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let mut intrinsics = vec![generic_intrinsics(tile_w, tile_h); NUM_CAMERAS];
    let mut distortion = vec![zero_dist; NUM_CAMERAS];
    let mut sensors = vec![ScheimpflugParams::default(); NUM_CAMERAS];

    match seed_mode {
        "generic" => {}
        "oracle" => {
            let oracle =
                oracle.ok_or_else(|| anyhow!("RTV3D_SEED=oracle requires artifacts.json"))?;
            for (i, cam) in oracle.cameras.iter().enumerate().take(NUM_CAMERAS) {
                // The oracle's cam 5 is degenerate (fx=51) — keep the generic
                // seed there.
                if cam.fx < 500.0 {
                    eprintln!(
                        "oracle cam {i} looks degenerate (fx={}); generic seed",
                        cam.fx
                    );
                    continue;
                }
                intrinsics[i] = FxFyCxCySkew {
                    fx: cam.fx,
                    fy: cam.fy,
                    cx: cam.cx,
                    cy: cam.cy,
                    skew: 0.0,
                };
                distortion[i] = BrownConrady5 {
                    k1: cam.k1,
                    k2: cam.k2,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                };
                sensors[i] = ScheimpflugParams {
                    tilt_x: cam.tau_x,
                    tilt_y: cam.tau_y,
                };
            }
        }
        other => return Err(anyhow!("unknown RTV3D_SEED '{other}'")),
    }

    let mut manual_init = RigHandeyeIntrinsicsManualInit::default();
    manual_init.per_cam_intrinsics = Some(intrinsics);
    manual_init.per_cam_distortion = Some(distortion);
    manual_init.per_cam_sensors = Some(sensors);
    Ok(manual_init)
}

// ───────────────────────────── oracle ─────────────────────────────────────

/// Per-camera reference calibration parsed from the legacy `artifacts.json`.
#[derive(Debug)]
struct OracleCamera {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    tau_x: f64,
    tau_y: f64,
    reproj_px: f64,
    /// `camera_se3_sensor` with translation converted to meters. The sensor
    /// frame is camera 0 (identity there), matching our rig frame.
    cam_se3_rig: Iso3,
    /// Laser plane in camera frame, Hessian form `n·x = d` (meters).
    plane_normal: nalgebra::Vector3<f64>,
    plane_distance: f64,
    plane_sigma_mm: f64,
}

#[derive(Debug)]
struct Oracle {
    cameras: Vec<OracleCamera>,
}

#[derive(Deserialize)]
struct ArtifactsJson {
    intrinsic: Vec<ArtifactsIntrinsic>,
    extrinsic: Vec<ArtifactsExtrinsic>,
    laserplanes: Vec<ArtifactsPlane>,
    num_cameras: usize,
}

#[derive(Deserialize)]
struct ArtifactsIntrinsic {
    /// `[k1, k2, k3, p1, p2, tau_x, tau_y]`
    distortion: Vec<f64>,
    matrix: [[f64; 3]; 3],
    reprojection_error_pix: f64,
}

#[derive(Deserialize)]
struct ArtifactsExtrinsic {
    camera_se3_sensor: [[f64; 4]; 4],
}

#[derive(Deserialize)]
struct ArtifactsPlane {
    /// Column vectors `[[x],[y],[z]]` in millimeters (origin) / unit (axes).
    origin: [[f64; 1]; 3],
    xaxis: [[f64; 1]; 3],
    yaxis: [[f64; 1]; 3],
    standard_deviation_mm: f64,
}

fn iso3_from_rowmajor_mm(m: &[[f64; 4]; 4]) -> Iso3 {
    use nalgebra::{Matrix3, Translation3, UnitQuaternion};
    let rot = Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );
    const MM_TO_M: f64 = 1.0e-3;
    let trans = Translation3::new(m[0][3] * MM_TO_M, m[1][3] * MM_TO_M, m[2][3] * MM_TO_M);
    Iso3::from_parts(trans, UnitQuaternion::from_matrix(&rot))
}

fn load_oracle(path: &Path) -> Result<Oracle> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: ArtifactsJson =
        serde_json::from_str(&s).with_context(|| format!("parse {}", path.display()))?;
    if raw.intrinsic.len() != raw.num_cameras
        || raw.extrinsic.len() != raw.num_cameras
        || raw.laserplanes.len() != raw.num_cameras
    {
        return Err(anyhow!("artifacts.json arrays disagree with num_cameras"));
    }
    let mut cameras = Vec::with_capacity(raw.num_cameras);
    for ((intr, extr), plane) in raw
        .intrinsic
        .iter()
        .zip(raw.extrinsic.iter())
        .zip(raw.laserplanes.iter())
    {
        if intr.distortion.len() < 7 {
            return Err(anyhow!("oracle distortion vector shorter than 7"));
        }
        let xaxis = nalgebra::Vector3::new(plane.xaxis[0][0], plane.xaxis[1][0], plane.xaxis[2][0]);
        let yaxis = nalgebra::Vector3::new(plane.yaxis[0][0], plane.yaxis[1][0], plane.yaxis[2][0]);
        let origin_m =
            nalgebra::Vector3::new(plane.origin[0][0], plane.origin[1][0], plane.origin[2][0])
                * 1.0e-3;
        let normal = xaxis.cross(&yaxis).normalize();
        cameras.push(OracleCamera {
            fx: intr.matrix[0][0],
            fy: intr.matrix[1][1],
            cx: intr.matrix[0][2],
            cy: intr.matrix[1][2],
            k1: intr.distortion[0],
            k2: intr.distortion[1],
            tau_x: intr.distortion[5],
            tau_y: intr.distortion[6],
            reproj_px: intr.reprojection_error_pix,
            cam_se3_rig: iso3_from_rowmajor_mm(&extr.camera_se3_sensor),
            plane_normal: normal,
            plane_distance: normal.dot(&origin_m),
            plane_sigma_mm: plane.standard_deviation_mm,
        });
    }
    Ok(Oracle { cameras })
}

// ───────────────────────────── comparison ────────────────────────────────

/// Compare against the legacy oracle.
///
/// Hard criteria (the "beat the oracle" verdict):
/// - per-camera reprojection: our best stage (joint BA when laser data is
///   present, hand-eye export otherwise) < oracle per-camera reproj for cams
///   0–4; cam 5 must merely be sane (< 10 px; oracle cam 5 is broken).
/// - extrinsic scale: per-camera |t| within 10 % of the oracle norms
///   (cams 1–4). Full pose deltas are NOT a criterion: the tilt ↔ principal
///   point ↔ rotation valley means our parameterization and the legacy one
///   describe the same optics with different splits, and the legacy camera
///   frame convention differs (rotated AOI), so parameter-by-parameter pose
///   comparison is ill-posed. The tables are still printed as information.
/// - laser σ: per-camera point-to-plane RMS < oracle `standard_deviation_mm`.
///   Plane normal/distance deltas are informational only (frame conventions
///   differ; σ is the physically meaningful fit quality).
fn compare_to_oracle(
    oracle: &Oracle,
    rig_export: &vision_calibration::rig_handeye::RigHandeyeExport,
    intr_reproj: &[f64],
    joint_reproj: Option<&[f64]>,
    laser_planes_cam: Option<&[LaserPlane]>,
    laser_sigma_mm: Option<&[f64]>,
) {
    println!("\n══════════ oracle comparison (artifacts.json) ══════════");
    println!("note: oracle cam 5 is degenerate (fx=51, reproj 127 px) — its deltas are nominal");

    let sensors = rig_export.sensors.as_deref().unwrap_or(&[]);
    println!("\nintrinsics (ours vs oracle):");
    println!(
        "  cam |     fx (Δ)      |     fy (Δ)      |    cx (Δ)    |    cy (Δ)    |  tau_x ours/orc  |  tau_y ours/orc"
    );
    for (i, cam) in rig_export.cameras.iter().enumerate() {
        let o = &oracle.cameras[i];
        let (tx, ty) = sensors
            .get(i)
            .map(|s| (s.tilt_x, s.tilt_y))
            .unwrap_or((0.0, 0.0));
        println!(
            "   {i}  | {:7.1} ({:+6.1}) | {:7.1} ({:+6.1}) | {:6.1} ({:+5.1}) | {:6.1} ({:+5.1}) | {:+.4}/{:+.4} | {:+.4}/{:+.4}",
            cam.k.fx,
            cam.k.fx - o.fx,
            cam.k.fy,
            cam.k.fy - o.fy,
            cam.k.cx,
            cam.k.cx - o.cx,
            cam.k.cy,
            cam.k.cy - o.cy,
            tx,
            o.tau_x,
            ty,
            o.tau_y,
        );
    }

    println!("\nreprojection error (px):");
    println!("  cam | ours (intrinsics) | ours (handeye) | ours (joint BA) | oracle | beat?");
    let mut reproj_pass = true;
    for i in 0..NUM_CAMERAS {
        let ours_intr = intr_reproj.get(i).copied();
        let ours_he = rig_export.per_cam_reproj_errors.get(i).copied();
        let ours_joint = joint_reproj.and_then(|r| r.get(i)).copied();
        let ours_best = ours_joint.or(ours_he);
        let o = &oracle.cameras[i];
        let sane_fx = {
            let fx = rig_export.cameras[i].k.fx;
            (1500.0..=2500.0).contains(&fx)
        };
        let beat = match ours_best {
            Some(r) if i < 5 => r < o.reproj_px && sane_fx,
            Some(r) => r < 10.0 && sane_fx, // cam 5: oracle is broken; sane = < 10 px
            None => false,
        };
        if !beat {
            reproj_pass = false;
        }
        println!(
            "   {i}  | {:>17} | {:>14} | {:>15} | {:6.2} | {}",
            ours_intr.map_or("-".into(), |v| format!("{v:.3}")),
            ours_he.map_or("-".into(), |v| format!("{v:.3}")),
            ours_joint.map_or("-".into(), |v| format!("{v:.3}")),
            o.reproj_px,
            if beat { "YES" } else { "no" }
        );
    }

    println!("\nextrinsics vs oracle camera_se3_sensor (rig frame = cam 0):");
    println!("note: rot/trans deltas are informational — the legacy frame convention and");
    println!("      the tilt↔pose parameter split differ; |t| norms carry the scale criterion");
    println!("  cam | rot Δ (deg) | trans Δ (mm) | |t| ours (mm) | |t| oracle (mm) | scale ok?");
    let mut extr_pass = true;
    for (i, ours) in rig_export.cam_se3_rig.iter().enumerate() {
        let o = &oracle.cameras[i].cam_se3_rig;
        let delta = ours.inverse() * *o;
        let rot_deg = delta.rotation.angle().to_degrees();
        let trans_mm = (ours.translation.vector - o.translation.vector).norm() * 1e3;
        let t_ours = ours.translation.vector.norm() * 1e3;
        let t_orc = o.translation.vector.norm() * 1e3;
        // cams 1–4 carry the scale criterion; cam 0 is gauge, cam 5 oracle is
        // broken
        let scale_ok = if (1..5).contains(&i) {
            (t_ours - t_orc).abs() / t_orc <= 0.10
        } else {
            true
        };
        if !scale_ok {
            extr_pass = false;
        }
        println!(
            "   {i}  | {:11.3} | {:12.2} | {:13.2} | {:14.2} | {}",
            rot_deg,
            trans_mm,
            t_ours,
            t_orc,
            if (1..5).contains(&i) {
                if scale_ok { "YES" } else { "no" }
            } else {
                "-"
            },
        );
    }

    let mut plane_pass = None;
    if let Some(planes) = laser_planes_cam {
        println!("\nlaser planes vs oracle (camera frame):");
        println!("note: normal/dist deltas are informational — frames differ; σ is the criterion");
        println!("  cam | normal Δ (deg) | dist Δ (mm) | σ ours (mm) | σ oracle (mm) | beat σ?");
        let mut pass = true;
        for (i, p) in planes.iter().enumerate() {
            let o = &oracle.cameras[i];
            let mut n_o = o.plane_normal;
            let mut d_o = o.plane_distance;
            let n = p.normal.into_inner();
            if n.dot(&n_o) < 0.0 {
                n_o = -n_o;
                d_o = -d_o;
            }
            let angle_deg = n.dot(&n_o).clamp(-1.0, 1.0).acos().to_degrees();
            let dist_mm = (p.distance - d_o).abs() * 1e3;
            let sigma = laser_sigma_mm.and_then(|s| s.get(i)).copied();
            let beat_sigma = sigma.map(|s| s < o.plane_sigma_mm);
            if beat_sigma != Some(true) {
                pass = false;
            }
            println!(
                "   {i}  | {:14.3} | {:11.3} | {:>11} | {:13.3} | {}",
                angle_deg,
                dist_mm,
                sigma.map_or("-".into(), |s| format!("{s:.4}")),
                o.plane_sigma_mm,
                beat_sigma.map_or("-", |b| if b { "YES" } else { "no" }),
            );
        }
        plane_pass = Some(pass);
    }

    println!("\nverdict:");
    println!(
        "  reprojection (cams 0-4 < oracle, cam 5 sane): {}",
        if reproj_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  extrinsic scale (cams 1-4, |t| within 10%): {}",
        if extr_pass { "PASS" } else { "FAIL" }
    );
    if let Some(p) = plane_pass {
        println!(
            "  laser plane fit (σ < oracle, all cams): {}",
            if p { "PASS" } else { "FAIL" }
        );
    }
}

// ───────────────────────────── manifest ──────────────────────────────────

/// One `FrameRef` per (pose, camera): the strip PNG with a 720×540 ROI tile.
fn build_image_manifest(poses: &[PoseEntry], tile_w: u32, tile_h: u32) -> ImageManifest {
    let mut frames = Vec::with_capacity(poses.len() * NUM_CAMERAS);
    for (pose_idx, pose) in poses.iter().enumerate() {
        for cam_idx in 0..NUM_CAMERAS {
            let mut roi = PixelRect::default();
            roi.x = (cam_idx as u32) * tile_w;
            roi.w = tile_w;
            roi.h = tile_h;
            let mut frame = FrameRef::default();
            frame.pose = pose_idx;
            frame.camera = cam_idx;
            frame.path = PathBuf::from(&pose.target_image);
            frame.roi = Some(roi);
            frames.push(frame);
        }
    }
    let mut manifest = ImageManifest::default();
    manifest.root = PathBuf::from(".");
    manifest.frames = frames;
    manifest
}
