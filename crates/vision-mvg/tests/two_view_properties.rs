//! Property tests for two-view / triangulation invariants (Q8,
//! `docs/notes/two-view-triangulation.md`).
//!
//! These are deterministic parameter sweeps (no external proptest dependency —
//! the workspace has none) that check *structural* invariants which must hold
//! for **any** valid configuration, over a grid of rotations, translation
//! directions, and scene points. They extend the single-configuration unit
//! tests in `vision-geometry` / `vision-mvg` `#[cfg(test)]` modules to a broad
//! sweep. Each checks a real invariant, not a restatement of the implementation:
//!
//! 1. `E = [t]× R` consistency: the GT essential matrix annihilates GT
//!    calibrated correspondences (`x₂ᵀ E x₁ = 0`), and `decompose_essential`
//!    recovers `(R, t̂)` up to the known sign fourfold.
//! 2. Estimated-`F` epipolar residual: `fundamental_8point` on noiseless pixel
//!    correspondences yields `|x₂ᵀ F x₁| / ‖F‖ ≈ 0`.
//! 3. Triangulate → reproject round trip: `triangulate_nview` on noiseless
//!    metric cameras reprojects back onto the observations.

#![allow(missing_docs)]

use nalgebra::Rotation3;
use vision_calibration_core::{FxFyCxCySkew, Mat3, Pt2, Pt3, Vec3};
use vision_geometry::camera_matrix::Mat34;
use vision_geometry::epipolar::{decompose_essential, fundamental_8point};
use vision_mvg::triangulation::triangulate_nview;

fn skew(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// A small, deterministic sweep of relative poses (Euler angles rad, metric t).
fn pose_grid() -> Vec<(Rotation3<f64>, Vec3)> {
    let rots = [
        (0.00, 0.00, 0.00),
        (0.10, -0.05, 0.02),
        (-0.15, 0.08, -0.04),
        (0.22, 0.14, 0.06),
        (-0.05, -0.18, 0.10),
    ];
    let ts = [
        Vec3::new(0.6, 0.02, 0.05),
        Vec3::new(-0.4, 0.30, -0.10),
        Vec3::new(0.10, -0.50, 0.20),
        Vec3::new(0.25, 0.10, 0.40),
    ];
    let mut out = Vec::new();
    for r in &rots {
        for t in &ts {
            out.push((Rotation3::from_euler_angles(r.0, r.1, r.2), *t));
        }
    }
    out
}

/// A non-coplanar calibrated point cloud in front of camera 1.
fn scene() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for iy in 0..5 {
        for ix in 0..5 {
            let x = -0.5 + 0.25 * ix as f64;
            let y = -0.4 + 0.2 * iy as f64;
            let z = 2.0 + 0.5 * ((ix as f64) * 0.9).sin() + 0.3 * (ix + iy) as f64;
            pts.push(Pt3::new(x, y, z));
        }
    }
    pts
}

/// Invariant 1: `E = [t]× R` annihilates GT calibrated correspondences, and
/// `decompose_essential(E)` contains the GT `(R, t̂)`.
#[test]
fn essential_skew_t_r_is_consistent_and_decomposes() {
    let world = scene();

    for (rot, t) in pose_grid() {
        // Pure rotation (t = 0) yields E = 0, which is correctly rejected by
        // decompose_essential; skip it here (covered by the degeneracy tests).
        if t.norm() < 1e-9 {
            continue;
        }
        let r = *rot.matrix();
        let e = skew(&t) * r;

        // Epipolar constraint for every GT correspondence: x₂ᵀ E x₁ = 0.
        let e_norm = e.norm();
        assert!(e_norm > 0.0);
        for pw in &world {
            let pc1 = pw.coords;
            let pc2 = r * pw.coords + t;
            if pc1.z <= 1e-6 || pc2.z <= 1e-6 {
                continue;
            }
            let x1 = nalgebra::Vector3::new(pc1.x / pc1.z, pc1.y / pc1.z, 1.0);
            let x2 = nalgebra::Vector3::new(pc2.x / pc2.z, pc2.y / pc2.z, 1.0);
            let res = (x2.transpose() * e * x1)[0].abs() / e_norm;
            assert!(res < 1e-12, "epipolar residual {res:.3e} for GT E");
        }

        // The GT (R, t̂) must appear among the four decomposition candidates.
        let candidates = decompose_essential(&e).unwrap();
        let mut found = false;
        for (r_est, t_est) in &candidates {
            let d = r_est.transpose() * r;
            let cos = ((d.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
            let ang = cos.acos().to_degrees();
            let cos_t = t_est.normalize().dot(&t.normalize()).abs();
            if ang < 1e-6 && (1.0 - cos_t) < 1e-9 {
                found = true;
            }
        }
        assert!(found, "decompose_essential did not recover the GT pose");
    }
}

/// Invariant 2: `fundamental_8point` on noiseless pixel correspondences yields a
/// matrix that satisfies the epipolar constraint on the GT correspondences.
#[test]
fn fundamental_8point_epipolar_residual_is_tight() {
    let k = FxFyCxCySkew {
        fx: 820.0,
        fy: 800.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    }
    .k_matrix();
    let world = scene();

    for (rot, t) in pose_grid() {
        if t.norm() < 1e-9 {
            continue; // pure rotation ⇒ F is not defined by a baseline
        }
        let r = *rot.matrix();

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        for pw in &world {
            let pc1 = pw.coords;
            let pc2 = r * pw.coords + t;
            if pc1.z <= 1e-6 || pc2.z <= 1e-6 {
                continue;
            }
            let u1 = k * pc1;
            let u2 = k * pc2;
            pts1.push(Pt2::new(u1.x / u1.z, u1.y / u1.z));
            pts2.push(Pt2::new(u2.x / u2.z, u2.y / u2.z));
        }
        assert!(pts1.len() >= 8);

        let f = fundamental_8point(&pts1, &pts2).unwrap();
        let f_norm = f.norm();
        assert!(f_norm > 0.0);
        let mut max_res = 0.0_f64;
        for (a, b) in pts1.iter().zip(pts2.iter()) {
            let x1 = nalgebra::Vector3::new(a.x, a.y, 1.0);
            let x2 = nalgebra::Vector3::new(b.x, b.y, 1.0);
            let res = (x2.transpose() * f * x1)[0].abs() / f_norm;
            max_res = max_res.max(res);
        }
        assert!(
            max_res < 1e-6,
            "estimated-F epipolar residual {max_res:.3e}"
        );
    }
}

/// Invariant 3: triangulate → reproject round trip. For noiseless metric
/// cameras the recovered 3D point reprojects onto the observations and matches
/// the GT point.
#[test]
fn triangulate_reproject_roundtrip() {
    let k = FxFyCxCySkew {
        fx: 820.0,
        fy: 800.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    }
    .k_matrix();
    let world = scene();

    for (rot, t) in pose_grid() {
        let r = *rot.matrix();
        let mut p1 = Mat34::zeros();
        p1.fixed_view_mut::<3, 3>(0, 0).copy_from(&k);
        let mut p2 = Mat34::zeros();
        p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&(k * r));
        p2.set_column(3, &(k * t));

        for pw in &world {
            let x1 = p1 * nalgebra::Vector4::new(pw.x, pw.y, pw.z, 1.0);
            let x2 = p2 * nalgebra::Vector4::new(pw.x, pw.y, pw.z, 1.0);
            if x1.z <= 1e-6 || x2.z <= 1e-6 {
                continue;
            }
            let u1 = Pt2::new(x1.x / x1.z, x1.y / x1.z);
            let u2 = Pt2::new(x2.x / x2.z, x2.y / x2.z);

            let tp = triangulate_nview(&[p1, p2], &[u1, u2]).unwrap();
            // Round-trip: reprojection error is ~0 and the 3D point matches GT.
            assert!(
                tp.reprojection_error < 1e-6,
                "reprojection error {:.3e}",
                tp.reprojection_error
            );
            assert!(tp.in_front);
            assert!(
                (tp.point - pw).norm() < 1e-6,
                "3D round-trip error {:.3e}",
                (tp.point - pw).norm()
            );
        }
    }
}
