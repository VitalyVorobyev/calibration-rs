//! Scheimpflug-aware stereo rectification validated against the `rtv3d_ref`
//! reference oracle (C4 / D4 gate).
//!
//! The `rtv3d_ref` `artifacts.json` (the customer's "QUICK" oracle) calibrates a
//! 6-camera Scheimpflug rig to sub-pixel reprojection error. Each camera carries
//! a genuine tilted sensor (`τx ≈ -5°`, small `τy`) and the cameras sit on a ring
//! converging on a central puzzle_board — a real, asymmetric multi-view geometry.
//!
//! This harness validates [`vision_mvg::rectification::rectify_stereo_pair`] on
//! that real calibration. For each pair `(cam0, cam_b)` it:
//!
//! 1. Reads the oracle `K`, Scheimpflug tilt, and `camera_se3_sensor` extrinsics.
//! 2. Builds the Scheimpflug-aware rectifying homographies.
//! 3. Projects a synthetic 3-D cloud through **both** cameras using the real
//!    pinhole + Scheimpflug-tilt model at the oracle's K and tilt, with lens
//!    distortion omitted — the rectification contract takes *undistorted*
//!    pixels (the Brown-Conrady model itself is validated separately by
//!    `rtv3d_ref_reproj`).
//! 4. Applies the rectifying maps and measures how far apart the rectified rows
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
//! rtv3d_ref_rectify --release`
//!
//! Env: `RTV3D_REF_DATA_DIR` (default `privatedata/rtv3d_ref`).

use anyhow::{Context, Result};
use nalgebra::{Matrix3, Translation3, UnitQuaternion};
use std::path::PathBuf;

use vision_calibration_core::{BrownConrady5, Camera, FxFyCxCySkew, Iso3, Mat3, Pinhole, Pt2, Pt3};
use vision_calibration_examples_private::{
    RefIntrinsic, ScheimpflugCamera, load_ref_artifacts, split_opencv_distortion,
};
use vision_mvg::rectification::{RectifyCamera, RectifyOptions, rectify_stereo_pair};

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
    let dir = std::env::var("RTV3D_REF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("privatedata/rtv3d_ref"));
    let art_path = dir.join("artifacts.json");
    let art = load_ref_artifacts(&art_path)
        .with_context(|| format!("load oracle {}", art_path.display()))?;

    println!(
        "rtv3d_ref Scheimpflug rectification check — {} cameras\n",
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
    let left_rect = rectify_camera(&art.intrinsic[0])?;
    let left_cam = ideal_camera(&art.intrinsic[0])?;

    for b in 1..art.num_cameras {
        let cam_b_ext = iso_from_4x4_mm(&art.extrinsic[b].camera_se3_sensor);
        // cam_b_se3_cam0 = T_Cb_S · T_C0_S⁻¹
        let cam_b_se3_cam0 = cam_b_ext * cam0_ext.inverse();

        let right_rect = rectify_camera(&art.intrinsic[b])?;
        let right_cam = ideal_camera(&art.intrinsic[b])?;

        let rect = rectify_stereo_pair(
            &left_rect,
            &right_rect,
            &cam_b_se3_cam0,
            &RectifyOptions::default(),
        )
        .with_context(|| format!("rectify pair (0, {b})"))?;

        // Project a cloud at the working distance through both cameras.
        let mut sum_dv = 0.0;
        let mut max_dv: f64 = 0.0;
        let mut n = 0usize;
        for p0 in working_cloud() {
            let p_b = cam_b_se3_cam0 * p0;
            let (Some(px0), Some(pxb)) = (
                left_cam.project_point_c(&p0.coords),
                right_cam.project_point_c(&p_b.coords),
            ) else {
                continue;
            };
            if !in_tile(&px0) || !in_tile(&pxb) {
                continue;
            }
            let dv = (rect.rectify_left(&px0).y - rect.rectify_right(&pxb).y).abs();
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
            rect.baseline,
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
        "\nVERDICT: Scheimpflug stereo rectification VALIDATED on rtv3d_ref \
         (worst row disagreement {worst:.2e} px over all pairs)."
    );
    Ok(())
}

/// Build a [`RectifyCamera`] (K + Scheimpflug tilt) from an oracle intrinsics block.
fn rectify_camera(intr: &RefIntrinsic) -> Result<RectifyCamera> {
    let m = &intr.matrix;
    let k = Mat3::new(
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    );
    let (_dist, tilt) = split_opencv_distortion(&intr.distortion)?;
    Ok(RectifyCamera::scheimpflug(k, tilt))
}

/// Build a distortion-free pinhole + Scheimpflug-tilt camera at the oracle's K
/// and tilt. Projecting through this yields the **undistorted** pixels that the
/// rectification maps expect; lens distortion is intentionally omitted here.
fn ideal_camera(intr: &RefIntrinsic) -> Result<ScheimpflugCamera> {
    let m = &intr.matrix;
    let k = FxFyCxCySkew {
        fx: m[0][0],
        fy: m[1][1],
        cx: m[0][2],
        cy: m[1][2],
        skew: m[0][1],
    };
    let (_dist, tilt) = split_opencv_distortion(&intr.distortion)?;
    let no_distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    Ok(Camera::new(Pinhole, no_distortion, tilt.compile(), k))
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
