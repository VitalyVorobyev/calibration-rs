//! Stereo rectification of a Scheimpflug rig validated against the `scheimpflug_rig_ref`
//! reference oracle.
//!
//! The `scheimpflug_rig_ref` `artifacts.json` (the customer's "QUICK" oracle) calibrates a
//! 6-camera Scheimpflug rig to sub-pixel reprojection error. Each camera carries
//! a genuine tilted sensor (`τx ≈ -5°`, small `τy`) and the cameras sit on a ring
//! converging on a central puzzle_board — a real, asymmetric multi-view geometry.
//!
//! This harness validates [`vision_mvg::rectification::rectify`] on
//! that real calibration. For each pair `(cam0, cam_b)` it:
//!
//! 1. Reads the oracle `K`, Scheimpflug tilt, and `camera_se3_sensor` extrinsics.
//! 2. Builds the two-camera rig (pinhole + Brown-Conrady + Scheimpflug tilt) and
//!    rectifies it from the calibration.
//! 3. Projects a synthetic 3-D cloud through **both** cameras' full models.
//! 4. Rectifies the raw pixels and measures how far apart the rectified rows
//!    of corresponding points are (`|Δv|`, pixels).
//!
//! A correct rectification drives `|Δv|` to numerical zero for every
//! correspondence — corresponding points share a row, the precondition for 1-D
//! disparity search. The verdict gate is `max |Δv| < 1e-3 px`.
//!
//! This is a calibration-only check (no detector, no images): the oracle *is* the
//! ground truth, so the rectification is exercised against the device's actual
//! intrinsics, tilts, and inter-camera geometry rather than synthetic stand-ins.
//!
//! Run:
//! `cargo run --manifest-path
//! crates/vision-calibration-examples-private/Cargo.toml --example
//! scheimpflug_rig_ref_rectify --release`
//!
//! Env: `SCHEIMPFLUG_RIG_REF_DATA_DIR` (default `privatedata/scheimpflug_rig_ref`).

use anyhow::{Context, Result};
use nalgebra::{Matrix3, Translation3, UnitQuaternion};
use std::path::PathBuf;

use vision_calibration_core::{
    CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, ProjectionParams, Pt2, Pt3,
    SensorParams, Uncertainty,
};
use vision_calibration_examples_private::{
    RefIntrinsic, load_ref_artifacts, split_opencv_distortion,
};
use vision_mvg::rectification::{RectifyOptions, rectify};

/// Per-camera tile size (the 4320×540 strip splits into 6 × 720×540).
const TILE_W: f64 = 720.0;
const TILE_H: f64 = 540.0;
/// Verdict tolerance on rectified row disagreement (pixels).
const ROW_TOL_PX: f64 = 1e-3;
/// Minimum overlapping correspondences a pair must contribute to be meaningful.
/// Wide-baseline pairs (cameras on opposite sides of the ring) share only a
/// small field of view, so this is kept modest.
const MIN_POINTS: usize = 8;

fn main() -> Result<()> {
    let dir = std::env::var("SCHEIMPFLUG_RIG_REF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("privatedata/scheimpflug_rig_ref"));
    let art_path = dir.join("artifacts.json");
    let art = load_ref_artifacts(&art_path)
        .with_context(|| format!("load oracle {}", art_path.display()))?;

    println!(
        "scheimpflug_rig_ref Scheimpflug rectification check — {} cameras\n",
        art.num_cameras
    );
    println!(
        "{:<8} {:>10} {:>8} {:>12} {:>12}",
        "pair", "baseline", "points", "mean|Δv|px", "max|Δv|px"
    );
    println!("{}", "─".repeat(54));

    let mut worst: f64 = 0.0;
    let mut any_failed = false;

    // Camera 0 is the reference (left); rectify it against every other camera.
    let cam0_ext = iso_from_4x4_mm(&art.extrinsic[0].camera_se3_sensor);
    let left_params = camera_params(&art.intrinsic[0])?;

    for b in 1..art.num_cameras {
        let cam_b_ext = iso_from_4x4_mm(&art.extrinsic[b].camera_se3_sensor);
        // cam_b_se3_cam0 = T_Cb_S · T_C0_S⁻¹
        let cam_b_se3_cam0 = cam_b_ext * cam0_ext.inverse();

        let right_params = camera_params(&art.intrinsic[b])?;
        let rig = CalibratedRig::new(
            vec![left_params.clone(), right_params],
            vec![Iso3::identity(), cam_b_se3_cam0],
            Uncertainty::default(),
        )?;
        let rect = rectify(&rig, [0, 1], &RectifyOptions::default())
            .with_context(|| format!("rectify pair (0, {b})"))?;

        // Project a cloud at the working distance through both cameras.
        let mut sum_dv = 0.0;
        let mut max_dv: f64 = 0.0;
        let mut n = 0usize;
        for p0 in working_cloud() {
            let p_b = cam_b_se3_cam0 * p0;
            let (Some(px0), Some(pxb)) = (
                rig.model(0).and_then(|m| m.project_point(&p0)),
                rig.model(1).and_then(|m| m.project_point(&p_b)),
            ) else {
                continue;
            };
            if !in_tile(&px0) || !in_tile(&pxb) {
                continue;
            }
            let (Some(rl), Some(rr)) =
                (rect.rectified_pixel(0, &px0), rect.rectified_pixel(1, &pxb))
            else {
                continue;
            };
            let dv = (rl.y - rr.y).abs();
            sum_dv += dv;
            max_dv = max_dv.max(dv);
            n += 1;
        }

        let mean_dv = if n > 0 { sum_dv / n as f64 } else { f64::NAN };
        let ok = n >= MIN_POINTS && max_dv < ROW_TOL_PX;
        any_failed |= !ok;
        worst = worst.max(max_dv);
        println!(
            "{:<8} {:>9.3}m {:>8} {:>12.2e} {:>12.2e}{}",
            format!("0–{b}"),
            rect.baseline(),
            n,
            mean_dv,
            max_dv,
            if ok { "" } else { "  <-- FAIL" }
        );
    }

    println!("{}", "─".repeat(54));
    if any_failed {
        println!(
            "\nVERDICT: FAIL — some pair exceeds {ROW_TOL_PX:e} px row tolerance \
             (worst {worst:.2e} px)."
        );
        std::process::exit(1);
    }
    println!(
        "\nVERDICT: Scheimpflug stereo rectification VALIDATED on scheimpflug_rig_ref \
         (worst row disagreement {worst:.2e} px over all pairs)."
    );
    Ok(())
}

/// The camera (K, lens distortion and Scheimpflug tilt) of an oracle intrinsics
/// block. Raw pixels go through the whole model, distortion included.
fn camera_params(intr: &RefIntrinsic) -> Result<CameraParams> {
    let m = &intr.matrix;
    let (distortion, tilt) = split_opencv_distortion(&intr.distortion)?;
    Ok(CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 { params: distortion },
        sensor: SensorParams::Scheimpflug { params: tilt },
        intrinsics: FxFyCxCy {
            fx: m[0][0],
            fy: m[1][1],
            cx: m[0][2],
            cy: m[1][2],
        },
    })
}

/// A spread of 3-D points (camera-0 frame, metres) near the rig's working
/// distance, where the converging cameras' fields of view overlap.
fn working_cloud() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for i in 0..9 {
        for j in 0..9 {
            let x = -0.04 + 0.01 * i as f64;
            let y = -0.04 + 0.01 * j as f64;
            for &z in &[0.26, 0.30, 0.34, 0.38] {
                pts.push(Pt3::new(x, y, z));
            }
        }
    }
    pts
}

fn in_tile(px: &Pt2) -> bool {
    px.x >= 0.0 && px.x < TILE_W && px.y >= 0.0 && px.y < TILE_H
}

/// Build an `Iso3` from a row-major 4×4 transform whose translation is in mm.
fn iso_from_4x4_mm(m: &[[f64; 4]; 4]) -> Iso3 {
    let rot = Matrix3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );
    let q = UnitQuaternion::from_matrix(&rot);
    let t = Translation3::new(m[0][3] * 1e-3, m[1][3] * 1e-3, m[2][3] * 1e-3);
    Iso3::from_parts(t, q)
}
