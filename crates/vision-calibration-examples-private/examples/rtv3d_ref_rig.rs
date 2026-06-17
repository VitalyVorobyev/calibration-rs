//! From-scratch rig calibration on the `rtv3d_ref` reference dataset.
//!
//! Companion to `rtv3d_ref_reproj`. That harness *froze* the reference
//! intrinsics and asked whether our reprojection reproduces the oracle. This
//! one asks the harder question: **starting only from the images and the robot
//! poses — no reference values seeded — does our pipeline recover the same
//! calibration?**
//!
//! Inputs used: detected puzzle_board corners (130×130, 5 mm) per camera tile
//! per pose, and the per-pose robot `tcp2base`. Nothing from `artifacts.json`
//! is fed in; intrinsics are bootstrapped by Zhang's method per camera.
//!
//! Pipeline (`RigHandeyeProblem`, `SensorMode::Scheimpflug`, EyeInHand):
//!   intrinsics init (Zhang) → intrinsics BA (+ tilt) → rig init → rig BA →
//!   hand-eye init → hand-eye BA.
//!
//! Then the recovered intrinsics / extrinsics / hand-eye and per-camera
//! reprojection error are compared against the oracle `artifacts.json`. The
//! `artifacts.json` is read *only* for this final comparison.
//!
//! Run:
//! `cargo run --release --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! rtv3d_ref_rig`
//!
//! Env:
//! - `RTV3D_REF_DATA_DIR` (default `privatedata/rtv3d_ref`).
//! - `RTV3D_REF_HANDEYE` = `eye_in_hand` (default) | `eye_to_hand`.

use anyhow::{Context, Result};
use nalgebra::{Matrix3, Translation3, UnitQuaternion};
use std::path::PathBuf;
use std::time::Instant;

use vision_calibration::{
    rig_handeye::{self as rh, RigHandeyeConfig, RigHandeyeProblem, SensorMode},
    session::CalibrationSession,
};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, DistortionFixMask, Iso3, RigDataset, RigView, RigViewObs,
    ScheimpflugParams,
};
use vision_calibration_examples_private::{
    RefArtifacts, RefIntrinsic, load_gray, load_poses, load_ref_artifacts, split_horizontal,
    split_opencv_distortion,
};
use vision_calibration_optim::{HandEyeMode, RobotPoseMeta, RobustLoss, ScheimpflugFixMask};

const NUM_CAMERAS: usize = 6;
const BOARD_ROWS: u32 = 130;
const BOARD_COLS: u32 = 130;
const CELL_SIZE_MM: f64 = 5.0;

fn main() -> Result<()> {
    let data_dir = PathBuf::from(
        std::env::var("RTV3D_REF_DATA_DIR").unwrap_or_else(|_| "privatedata/rtv3d_ref".to_string()),
    );
    let handeye_mode = match std::env::var("RTV3D_REF_HANDEYE").as_deref() {
        Ok("eye_to_hand") => HandEyeMode::EyeToHand,
        _ => HandEyeMode::EyeInHand,
    };
    println!("data dir = {}", data_dir.display());
    println!("hand-eye mode = {handeye_mode:?} (config says eye_in_hand_static_target)");

    let art =
        load_ref_artifacts(&data_dir.join("artifacts.json")).context("load oracle artifacts")?;
    let poses = load_poses(&data_dir.join("poses.json"))?;
    println!(
        "loaded {} poses, {} oracle cameras",
        poses.len(),
        art.num_cameras
    );

    // ── Detect puzzle_board in every camera tile of every pose ───────────────
    // The linear-init homography DLT now skips the unused U factor in its SVD
    // (see `HomographySolver::dlt`), so dense (~200 corners/view) detections no
    // longer hang the init — we feed every detected corner straight through.
    let t_det = Instant::now();
    let mut views: Vec<RigView<RobotPoseMeta>> = Vec::new();
    let mut det_views = [0usize; NUM_CAMERAS];
    let mut det_pts = [0usize; NUM_CAMERAS];
    for (i, pose) in poses.iter().enumerate() {
        let img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("pose {i} target"))?;
        let tiles = split_horizontal(&img, NUM_CAMERAS);
        let mut cams: Vec<Option<CorrespondenceView>> = Vec::with_capacity(NUM_CAMERAS);
        for (c, tile) in tiles.iter().enumerate() {
            match vision_calibration_examples_private::detect_target(
                tile,
                BOARD_ROWS,
                BOARD_COLS,
                CELL_SIZE_MM,
            ) {
                Ok(v) => {
                    det_views[c] += 1;
                    det_pts[c] += v.points_2d.len();
                    cams.push(Some(v));
                }
                Err(_) => cams.push(None),
            }
        }
        views.push(RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: pose.base_se3_gripper(),
            },
            obs: RigViewObs { cameras: cams },
        });
    }
    println!("detect: {:.2?}", t_det.elapsed());
    for c in 0..NUM_CAMERAS {
        println!("  cam {c}: {} views, {} corners", det_views[c], det_pts[c]);
    }

    // ── From-scratch RigHandeye(Scheimpflug) calibration ─────────────────────
    let dataset = RigDataset::new(views, NUM_CAMERAS)?;
    let mut session = CalibrationSession::<RigHandeyeProblem>::with_description("rtv3d_ref_rig");
    session.set_input(dataset)?;

    let mut cfg = RigHandeyeConfig::default();
    cfg.solver.max_iters = std::env::var("RTV3D_REF_MAXITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    cfg.solver.verbosity = 0;
    cfg.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
    cfg.handeye_init.handeye_mode = handeye_mode;
    // Match the oracle's lens config: k1,k2 free; k3 + tangential fixed; both
    // Scheimpflug tilts free.
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
    session.set_config(cfg)?;

    let t0 = Instant::now();
    // No seed → Zhang's method per camera (genuinely from scratch).
    let s = Instant::now();
    rh::step_intrinsics_init_all(&mut session, None)?;
    println!("  intrinsics init (Zhang): {:.2?}", s.elapsed());
    let s = Instant::now();
    let intr = rh::step_intrinsics_optimize_all(&mut session, None)?;
    println!(
        "  intrinsics BA: {:.2?}  per-cam reproj={:?}",
        s.elapsed(),
        intr.per_cam_reproj_errors
            .iter()
            .map(|e| (e * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    let s = Instant::now();
    rh::step_rig_init(&mut session)?;
    println!("  rig init: {:.2?}", s.elapsed());
    let s = Instant::now();
    let rig = rh::step_rig_optimize(&mut session, None)?;
    println!(
        "  rig BA: {:.2?}  rig_reproj={:.4}px",
        s.elapsed(),
        rig.mean_reproj_error
    );
    let s = Instant::now();
    rh::step_handeye_init(&mut session, None)?;
    println!("  hand-eye init: {:.2?}", s.elapsed());
    let s = Instant::now();
    rh::step_handeye_optimize(&mut session, None)?;
    println!("  hand-eye BA: {:.2?}", s.elapsed());
    let export = session.export()?;
    println!(
        "calibrate (from scratch) total: {:.2?}  final_mean_reproj={:.4}px",
        t0.elapsed(),
        export.mean_reproj_error
    );

    compare(&art, &export, handeye_mode);
    Ok(())
}

/// Build an `Iso3` from a row-major 4x4 transform whose translation is in mm.
fn iso_from_4x4_mm(m: &[[f64; 4]; 4]) -> Iso3 {
    let rot = Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );
    let q = UnitQuaternion::from_matrix(&rot);
    let t = Translation3::new(m[0][3] * 1e-3, m[1][3] * 1e-3, m[2][3] * 1e-3);
    Iso3::from_parts(t, q)
}

/// Reference (fx,fy,cx,cy, Brown-Conrady, Scheimpflug) for a camera.
fn ref_params(intr: &RefIntrinsic) -> (f64, f64, f64, f64, BrownConrady5<f64>, ScheimpflugParams) {
    let m = &intr.matrix;
    let (dist, tilt) = split_opencv_distortion(&intr.distortion).expect("oracle distortion");
    (m[0][0], m[1][1], m[0][2], m[1][2], dist, tilt)
}

fn compare(
    art: &RefArtifacts,
    export: &vision_calibration::rig_handeye::RigHandeyeExport,
    handeye_mode: HandEyeMode,
) {
    println!("\n── Recovered intrinsics vs oracle (from scratch) ──");
    println!(
        "  cam |     fx (ref)      |     fy (ref)      |   cx (ref)    |   cy (ref)    |    k1 (ref)     |    k2 (ref)     |  tau_x° (ref)  |  tau_y° (ref)"
    );
    let sensors = export.sensors.as_ref();
    for i in 0..NUM_CAMERAS {
        let our = &export.cameras[i];
        let our_s = sensors.map(|s| s[i]).unwrap_or_default();
        let r = ref_params_from(art, i);
        println!(
            "  {i:>3} | {:8.1} ({:7.1}) | {:8.1} ({:7.1}) | {:6.1} ({:6.1}) | {:6.1} ({:6.1}) | {:+.4} ({:+.4}) | {:+.4} ({:+.4}) | {:+.3} ({:+.3}) | {:+.3} ({:+.3})",
            our.k.fx,
            r.0,
            our.k.fy,
            r.1,
            our.k.cx,
            r.2,
            our.k.cy,
            r.3,
            our.dist.k1,
            r.4.k1,
            our.dist.k2,
            r.4.k2,
            our_s.tilt_x.to_degrees(),
            r.5.tilt_x.to_degrees(),
            our_s.tilt_y.to_degrees(),
            r.5.tilt_y.to_degrees(),
        );
    }

    println!("\n── Recovered extrinsics (cam_se3_rig) vs oracle camera_se3_sensor ──");
    println!("  cam | our |t| mm | ref |t| mm | Δt mm | Δrot deg");
    for i in 0..NUM_CAMERAS {
        let our = export.cam_se3_rig[i];
        let refe = iso_from_4x4_mm(&art.extrinsic[i].camera_se3_sensor);
        let our_t_mm = our.translation.vector.norm() * 1e3;
        let ref_t_mm = refe.translation.vector.norm() * 1e3;
        let dt_mm = (our.translation.vector - refe.translation.vector).norm() * 1e3;
        let drot = (our.rotation.inverse() * refe.rotation)
            .angle()
            .to_degrees();
        println!("  {i:>3} | {our_t_mm:9.2} | {ref_t_mm:9.2} | {dt_mm:6.2} | {drot:7.3}");
    }

    println!("\n── Recovered hand-eye vs oracle tcp_se3_sensor ──");
    let our_he = match handeye_mode {
        HandEyeMode::EyeInHand => export.gripper_se3_rig,
        HandEyeMode::EyeToHand => export.rig_se3_base,
    };
    let ref_he = iso_from_4x4_mm(&art.handeye.tcp_se3_sensor);
    match our_he {
        Some(he) => {
            let dt_mm = (he.translation.vector - ref_he.translation.vector).norm() * 1e3;
            let drot = (he.rotation.inverse() * ref_he.rotation)
                .angle()
                .to_degrees();
            println!(
                "  our |t|={:.2} mm  ref |t|={:.2} mm  Δt={:.2} mm  Δrot={:.3} deg",
                he.translation.vector.norm() * 1e3,
                ref_he.translation.vector.norm() * 1e3,
                dt_mm,
                drot
            );
            println!(
                "  (note: hand-eye absolute frame depends on topology; Δrot is the meaningful term)"
            );
        }
        None => println!("  no hand-eye of the expected topology in the export"),
    }

    println!("\n── Per-camera reprojection: ours (from scratch) vs oracle ──");
    println!("  cam | our_reproj | ref_reproj |   Δ");
    for i in 0..NUM_CAMERAS {
        let our = export
            .per_cam_reproj_errors
            .get(i)
            .copied()
            .unwrap_or(f64::NAN);
        let refp = art.intrinsic[i].reprojection_error_pix;
        println!("  {i:>3} | {our:10.4} | {refp:10.4} | {:+.4}", our - refp);
    }
}

fn ref_params_from(
    art: &RefArtifacts,
    i: usize,
) -> (f64, f64, f64, f64, BrownConrady5<f64>, ScheimpflugParams) {
    ref_params(&art.intrinsic[i])
}
