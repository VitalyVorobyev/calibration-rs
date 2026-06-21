//! Two-view multiple-view geometry through the facade `mvg` module.
//!
//! Companion runnable example for
//! [`docs/tutorials/multiple-view-geometry.md`]. It walks the
//! facade-exposed MVG surface end-to-end on a synthetic calibrated stereo pair
//! with known ground truth:
//!
//! 1. relative pose recovery (5-point + cheirality) — `mvg::pose_recovery`
//! 2. two-view triangulation (returned by step 1)   — `mvg::types`
//! 3. bundle adjustment (with `--features refine`)   — `mvg::bundle_adjust`
//! 4. Scheimpflug-aware stereo rectification         — `mvg::rectification`
//!
//! Run:
//! ```bash
//! cargo run -p vision-calibration --example mvg_two_view --features refine
//! ```

use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use vision_calibration::core::{Iso3, Pt2, Pt3};
use vision_calibration::mvg::{
    pose_recovery::recover_relative_pose,
    rectification::{RectifyCamera, RectifyOptions, rectify_stereo_pair},
    types::Correspondence2D,
};

/// Shared pinhole intrinsics for both cameras.
fn intrinsics() -> Matrix3<f64> {
    Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0)
}

/// Ground-truth relative pose `T_C1_C0` (maps a point from camera 0 to camera 1).
fn ground_truth_pose() -> Iso3 {
    let r = Rotation3::from_euler_angles(0.02, 0.06, -0.01);
    Iso3::from_parts(Translation3::new(-0.6, 0.04, 0.10), UnitQuaternion::from(r))
}

/// A genuinely 3-D point cloud in front of both cameras (world == camera-0
/// frame). Depth is decorrelated from `(x, y)` and spans a wide range so the
/// scene is non-planar — a near-planar cloud is degenerate for the 5-point
/// essential-matrix estimator.
fn scene() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            let x = -0.6 + 0.24 * i as f64;
            let y = -0.6 + 0.24 * j as f64;
            let h = ((i * 5 + j * 11 + 7) % 17) as f64 / 16.0; // pseudo-random in [0, 1]
            let z = 3.5 + 3.0 * h; // depth in [3.5, 6.5], decorrelated from x/y
            pts.push(Pt3::new(x, y, z));
        }
    }
    pts
}

/// Normalized image coordinates of a camera-frame point.
fn normalize(p: &Vector3<f64>) -> Pt2 {
    Pt2::new(p.x / p.z, p.y / p.z)
}

/// Apply (zero-skew) pinhole intrinsics to a normalized point → pixel.
fn to_pixel(k: &Matrix3<f64>, n: &Pt2) -> Pt2 {
    Pt2::new(k[(0, 0)] * n.x + k[(0, 2)], k[(1, 1)] * n.y + k[(1, 2)])
}

fn main() {
    let k = intrinsics();
    let gt = ground_truth_pose();
    let points = scene();

    // Project the scene into both cameras. Camera 0 is the world frame; a point
    // in camera 1 is `gt * p`. Pose recovery wants normalized coordinates;
    // rectification wants pixels.
    let mut corrs = Vec::new();
    let mut px0 = Vec::new();
    let mut px1 = Vec::new();
    for p in &points {
        let c0 = p.coords;
        let c1 = (gt * p).coords;
        if c0.z <= 0.0 || c1.z <= 0.0 {
            continue;
        }
        let (n0, n1) = (normalize(&c0), normalize(&c1));
        corrs.push(Correspondence2D::new(n0, n1));
        px0.push(to_pixel(&k, &n0));
        px1.push(to_pixel(&k, &n1));
    }
    println!(
        "scene: {} correspondences visible in both views",
        corrs.len()
    );

    // 1 + 2. Relative pose recovery (5-point + cheirality) also triangulates the
    // inlier correspondences. Translation is recovered up to scale (unit length).
    let rel = recover_relative_pose(&corrs).expect("recover relative pose");
    let rot_err = (rel.r - gt.rotation.to_rotation_matrix().into_inner()).norm();
    let t_dot = rel.t.dot(&gt.translation.vector.normalize()).abs();
    println!(
        "1. pose recovery : rotation Δ = {rot_err:.2e}, |t·t_gt| = {t_dot:.6}, \
         {} points triangulated",
        rel.points.len()
    );
    assert!(rot_err < 1e-3, "rotation not recovered: {rot_err}");
    assert!(
        t_dot > 0.9999,
        "translation direction not recovered: {t_dot}"
    );

    // 3. Bundle adjustment jointly refines poses + structure (frozen intrinsics).
    #[cfg(feature = "refine")]
    bundle_adjust_demo(&k, &gt, &points);
    #[cfg(not(feature = "refine"))]
    println!("3. bundle adjust : skipped — re-run with `--features refine`");

    // 4. Rectify the calibrated pair so corresponding points share an image row.
    let rect = rectify_stereo_pair(
        &RectifyCamera::pinhole(k),
        &RectifyCamera::pinhole(k),
        &gt,
        &RectifyOptions::default(),
    )
    .expect("rectify stereo pair");
    let max_dv = px0
        .iter()
        .zip(&px1)
        .map(|(a, b)| (rect.rectify_left(a).y - rect.rectify_right(b).y).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "4. rectification : baseline = {:.3}, max row disagreement = {max_dv:.2e} px",
        rect.baseline
    );
    assert!(max_dv < 1e-9, "rectified rows not aligned: {max_dv}");

    println!("\nAll stages recovered ground truth. ✔");
}

/// Perturb the second camera + the structure, then bundle-adjust back to ground
/// truth. Camera 0 (the reference) anchors the rigid gauge.
#[cfg(feature = "refine")]
fn bundle_adjust_demo(k: &Matrix3<f64>, gt: &Iso3, points: &[Pt3]) {
    use vision_calibration::mvg::bundle_adjust::{
        BundleAdjustmentOptions, BundleObservation, bundle_adjust,
    };

    let poses_gt = [Iso3::identity(), *gt];
    let intr = [*k, *k];

    // Every point seen by both cameras (pixel observations).
    let mut obs = Vec::new();
    for (j, p) in points.iter().enumerate() {
        for (cam, pose) in poses_gt.iter().enumerate() {
            let c = (pose * p).coords;
            if c.z > 0.0 {
                obs.push(BundleObservation::new(cam, j, to_pixel(k, &normalize(&c))));
            }
        }
    }

    // Perturb camera 1 and the points; camera 0 stays at ground truth as the anchor.
    let mut init_poses = poses_gt;
    init_poses[1] = *gt
        * Iso3::from_parts(
            Translation3::new(0.05, -0.03, 0.04),
            UnitQuaternion::from(Rotation3::from_euler_angles(0.01, -0.015, 0.008)),
        );
    let init_points: Vec<Pt3> = points
        .iter()
        .enumerate()
        .map(|(i, p)| p + 0.1 * Vector3::new((i % 3) as f64 - 1.0, (i % 2) as f64 - 0.5, 0.2))
        .collect();

    let res = bundle_adjust(
        &intr,
        &obs,
        &init_poses,
        &init_points,
        &BundleAdjustmentOptions::default(),
    )
    .expect("bundle adjust");
    println!(
        "3. bundle adjust : RMS {:.4} → {:.4} px over {} cameras, {} points",
        res.initial_rms,
        res.final_rms,
        res.poses.len(),
        res.points.len()
    );
    assert!(
        res.final_rms < 0.1 && res.final_rms < res.initial_rms,
        "bundle adjustment did not converge: {} → {}",
        res.initial_rms,
        res.final_rms
    );
}
