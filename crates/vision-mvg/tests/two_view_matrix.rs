//! Synthetic-GT matrix test for the two-view / triangulation family (Q8,
//! `docs/notes/README.md`; math note: `docs/notes/two-view-triangulation.md`).
//!
//! Ground-truth grid (baseline / parallax regimes) × pixel-noise levels. Every
//! cell builds an exact synthetic stereo pair with a known intrinsic `K` and a
//! known relative pose `(R, t)`, projects a non-coplanar 3D point cloud to
//! pixels, adds deterministic per-pixel noise, and pushes the result through:
//!
//! 1. **Relative pose recovery** (`recover_relative_pose`, the 5-point +
//!    cheirality pipeline on normalized coordinates) — asserts rotation error
//!    (deg) and translation-direction error (deg).
//! 2. **Metric triangulation** (`triangulate_nview`, DLT + Gauss-Newton) with
//!    the *known* metric projection matrices — asserts reprojection error (px)
//!    and 3D Euclidean error against ground truth, scaled to the injected
//!    noise floor.
//!
//! Two-view geometry is observable only up to a similarity scale, so (1) uses
//! only the scale-invariant quantities (rotation, translation *direction*).
//! (2) resolves the scale gauge by handing the solver the true metric cameras,
//! which is exactly the frozen-extrinsics regime the triangulation solver runs
//! in downstream.

#![allow(missing_docs)]

use nalgebra::Rotation3;
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::{FxFyCxCySkew, Mat3, Pt2, Pt3, Vec3, pixel_to_normalized};
use vision_geometry::camera_matrix::Mat34;
use vision_mvg::triangulation::triangulate_nview;
use vision_mvg::{Correspondence2D, recover_relative_pose};

/// Shared intrinsics for both views (a 1280×720-ish pinhole).
fn intrinsics() -> Mat3 {
    FxFyCxCySkew {
        fx: 800.0,
        fy: 790.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    }
    .k_matrix()
}

/// A non-coplanar 3D point cloud spread across the shared frustum. Depth varies
/// non-linearly with `(ix, iy)` so the points do not lie near any plane — a
/// planar cloud would make the essential matrix / metric-DLT rank deficient
/// (see the math note, §Identifiability and degeneracies).
fn world_points() -> Vec<Pt3> {
    let mut world = Vec::new();
    for iy in 0..6 {
        for ix in 0..7 {
            let x = -1.2 + 0.4 * ix as f64;
            let y = -0.9 + 0.36 * iy as f64;
            let z = 3.0
                + 0.6 * ((ix as f64) * 0.7).sin()
                + 0.5 * ((iy as f64) * 1.1).cos()
                + 0.12 * (ix + iy) as f64;
            world.push(Pt3::new(x, y, z));
        }
    }
    world
}

/// One ground-truth relative pose (a baseline / parallax regime).
struct GtCell {
    name: &'static str,
    /// Euler angles (rad) for the camera-1 → camera-2 rotation.
    rot: (f64, f64, f64),
    /// Metric translation camera-1 → camera-2 (`pc2 = R·pc1 + t`).
    t: Vec3,
    /// Tolerance scale: rotation-error bound per pixel of noise (deg/px).
    rot_deg_per_px: f64,
    /// Tolerance scale: translation-direction-error bound per pixel of noise.
    tdir_deg_per_px: f64,
    /// Tolerance scale: median 3D triangulation error bound per pixel of noise.
    err3d_per_px: f64,
}

fn cells() -> Vec<GtCell> {
    vec![
        // Tolerance scales are ~2× the observed deterministic errors, so the
        // assertions are meaningful yet stable (the noise is seeded). The
        // translation-direction scale climbs sharply as the baseline narrows —
        // the two-view parallax degeneracy made quantitative.
        GtCell {
            name: "wide-sideways",
            rot: (0.04, -0.03, 0.02),
            t: Vec3::new(0.8, 0.05, 0.10),
            rot_deg_per_px: 0.5,
            tdir_deg_per_px: 1.0,
            err3d_per_px: 0.04,
        },
        GtCell {
            name: "wide-rotated",
            rot: (0.18, -0.12, 0.06),
            t: Vec3::new(0.7, 0.15, 0.10),
            rot_deg_per_px: 0.5,
            tdir_deg_per_px: 1.5,
            err3d_per_px: 0.04,
        },
        GtCell {
            name: "medium",
            rot: (0.06, -0.04, 0.02),
            t: Vec3::new(0.30, 0.03, 0.04),
            rot_deg_per_px: 0.4,
            tdir_deg_per_px: 2.5,
            err3d_per_px: 0.1,
        },
        GtCell {
            name: "narrow",
            rot: (0.03, -0.02, 0.00),
            t: Vec3::new(0.10, 0.01, 0.01),
            rot_deg_per_px: 0.4,
            tdir_deg_per_px: 8.0,
            err3d_per_px: 0.3,
        },
    ]
}

/// Rotation error (deg) between two rotation matrices.
fn rot_err_deg(r_est: &Mat3, r_gt: &Mat3) -> f64 {
    let d = r_est.transpose() * r_gt;
    let cos = ((d.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

/// Translation-direction error (deg). Two-view `t` is recovered up to scale;
/// the sign is fixed by cheirality but the reported metric is the acute angle
/// between the estimated and GT translation *lines* (`|cos|`).
fn tdir_err_deg(t_est: &Vec3, t_gt: &Vec3) -> f64 {
    let cos = t_est
        .normalize()
        .dot(&t_gt.normalize())
        .abs()
        .clamp(0.0, 1.0);
    cos.acos().to_degrees()
}

/// Build the metric pixel-domain projection matrices `P₁ = K[I|0]`,
/// `P₂ = K[R|t]`.
fn projection_pair(k: &Mat3, r: &Mat3, t: &Vec3) -> (Mat34, Mat34) {
    let mut p1 = Mat34::zeros();
    p1.fixed_view_mut::<3, 3>(0, 0).copy_from(k);
    let mut p2 = Mat34::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&(k * r));
    p2.set_column(3, &(k * t));
    (p1, p2)
}

fn project(p: &Mat34, pw: &Pt3) -> Option<Pt2> {
    let x = p * nalgebra::Vector4::new(pw.x, pw.y, pw.z, 1.0);
    if x.z <= 1e-6 {
        return None;
    }
    Some(Pt2::new(x.x / x.z, x.y / x.z))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
fn two_view_recovers_pose_and_triangulation_across_grid_and_noise() {
    let k = intrinsics();
    let world = world_points();
    let noise_levels = [0.0_f64, 0.2, 0.5];

    for cell in cells() {
        let r_gt = *Rotation3::from_euler_angles(cell.rot.0, cell.rot.1, cell.rot.2).matrix();
        let (p1, p2) = projection_pair(&k, &r_gt, &cell.t);

        for &noise_px in &noise_levels {
            let noise = UniformPixelNoise {
                seed: 0xB1AB,
                max_abs_px: noise_px,
            };

            // Project the cloud to noisy pixels in both views.
            let mut corrs = Vec::new();
            let mut px1 = Vec::new();
            let mut px2 = Vec::new();
            let mut kept_world = Vec::new();
            for (j, pw) in world.iter().enumerate() {
                let (Some(u1), Some(u2)) = (project(&p1, pw), project(&p2, pw)) else {
                    continue;
                };
                let n1 = noise.sample(0, j);
                let n2 = noise.sample(1, j);
                let q1 = Pt2::new(u1.x + n1.x, u1.y + n1.y);
                let q2 = Pt2::new(u2.x + n2.x, u2.y + n2.y);
                corrs.push(Correspondence2D::new(
                    pixel_to_normalized(q1, &k),
                    pixel_to_normalized(q2, &k),
                ));
                px1.push(q1);
                px2.push(q2);
                kept_world.push(*pw);
            }
            assert!(corrs.len() >= 30, "too few visible correspondences");

            let label = format!("{} noise={noise_px}px", cell.name);

            // --- (1) Relative pose recovery ---
            let pose = recover_relative_pose(&corrs)
                .unwrap_or_else(|e| panic!("{label}: pose recovery failed: {e:?}"));
            let rerr = rot_err_deg(&pose.r, &r_gt);
            let terr = tdir_err_deg(&pose.t, &cell.t);

            // --- (2) Metric triangulation with the true cameras ---
            let mut reproj_px = Vec::new();
            let mut err3d = Vec::new();
            for ((u1, u2), pw) in px1.iter().zip(px2.iter()).zip(kept_world.iter()) {
                let tp = triangulate_nview(&[p1, p2], &[*u1, *u2])
                    .unwrap_or_else(|e| panic!("{label}: triangulation failed: {e:?}"));
                assert!(tp.in_front, "{label}: triangulated point behind a camera");
                reproj_px.push(tp.reprojection_error);
                err3d.push((tp.point - pw).norm());
            }
            let med_reproj = median(reproj_px);
            let med_err3d = median(err3d);

            eprintln!(
                "{label:24} | rot {rerr:7.4} deg | tdir {terr:8.4} deg | \
                 reproj {med_reproj:.3e} px | 3Derr {med_err3d:.3e}"
            );

            // Tolerances: exact data must be tight; noisy cells scale with the
            // injected amplitude (empirical bounds with margin).
            if noise_px == 0.0 {
                assert!(rerr < 0.02, "{label}: rotation {rerr} deg (exact)");
                assert!(terr < 0.05, "{label}: t-dir {terr} deg (exact)");
                assert!(med_reproj < 1e-6, "{label}: reproj {med_reproj} px (exact)");
                assert!(med_err3d < 1e-5, "{label}: 3D err {med_err3d} (exact)");
            } else {
                let rot_bound = cell.rot_deg_per_px * noise_px;
                let tdir_bound = cell.tdir_deg_per_px * noise_px;
                // Reprojection floor: post-fit reprojection tracks the pixel
                // noise (uniform [-a, a] per axis ⇒ error-norm mean ≈ 0.765·a).
                let reproj_bound = noise_px * 1.0 + 0.02;
                let err3d_bound = cell.err3d_per_px * noise_px;
                assert!(
                    rerr < rot_bound,
                    "{label}: rotation {rerr} deg > bound {rot_bound}"
                );
                assert!(
                    terr < tdir_bound,
                    "{label}: t-dir {terr} deg > bound {tdir_bound}"
                );
                assert!(
                    med_reproj < reproj_bound,
                    "{label}: reproj {med_reproj} px > bound {reproj_bound}"
                );
                assert!(
                    med_err3d < err3d_bound,
                    "{label}: 3D err {med_err3d} > bound {err3d_bound}"
                );
            }
        }
    }
}
