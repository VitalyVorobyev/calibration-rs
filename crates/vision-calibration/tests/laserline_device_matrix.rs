//! Synthetic-GT matrix test for the laserline device family
//! (math note: `docs/notes/laserline-bundle.md`).
//!
//! Ground-truth laser-plane grid (2 orientations × 2 working distances) ×
//! pixel-noise levels. Every cell builds an exact synthetic laserline dataset
//! (target corners + laser-stripe pixels), runs the standard
//! `step_init → step_optimize` pipeline (`run_calibration`), and asserts
//! laser-plane recovery — normal in degrees, signed distance in millimetres —
//! plus a laser/reprojection residual floor consistent with the injected noise.
//!
//! No cargo feature gates the laserline math in the facade or pipeline (only
//! the *bench* crate's `laser` feature pulls in the vision-metrology
//! extractor), so this test runs under both plain `cargo test` and
//! `cargo test --workspace --all-features`.

#![allow(missing_docs)]

mod common;

use std::f64::consts::PI;

use common::laser::{board, build_dataset, poses};
use nalgebra::{Rotation3, Translation3, Vector3};
use vision_calibration::core::LaserPlane;
use vision_calibration::core::{FxFyCxCy, Iso3, Pt3};
use vision_calibration::laserline_device::{
    LaserlineDeviceConfig, LaserlineDeviceExport, LaserlineDeviceInput, LaserlineDeviceProblem,
    run_calibration,
};
use vision_calibration::optim::SolverBackend;
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::noise::UniformPixelNoise;

/// Run the pipeline on both solver backends, check that they agree, and
/// return the default backend's export.
fn solve(dataset: LaserlineDeviceInput) -> LaserlineDeviceExport {
    let run = |backend| {
        let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
        session.set_input(dataset.clone()).expect("input");
        let mut config = LaserlineDeviceConfig::default();
        config.solver.backend = backend;
        session.set_config(config).expect("config");
        run_calibration(&mut session).expect("calibration");
        session.export().expect("export")
    };
    let tiny = run(SolverBackend::TinySolver);
    let factrs = run(SolverBackend::Factrs);
    let (a, b) = (&tiny, &factrs);
    for (name, x, y) in [
        ("fx", a.camera.intrinsics.fx, b.camera.intrinsics.fx),
        ("fy", a.camera.intrinsics.fy, b.camera.intrinsics.fy),
        ("cx", a.camera.intrinsics.cx, b.camera.intrinsics.cx),
        ("cy", a.camera.intrinsics.cy, b.camera.intrinsics.cy),
    ] {
        assert!(
            (x - y).abs() <= 1e-2,
            "backends disagree on {name}: {x} vs {y}"
        );
    }
    let dn = a.laser_plane.normal().angle(&b.laser_plane.normal());
    let dd = (a.laser_plane.distance() - b.laser_plane.distance()).abs();
    assert!(dn <= 1e-5 && dd <= 1e-5, "plane {dn} rad / {dd} m");
    tiny
}

#[test]
fn laserline_device_recovers_plane_across_grid_and_noise() {
    let gt_intr = FxFyCxCy {
        fx: 900.0,
        fy: 900.0,
        cx: 640.0,
        cy: 360.0,
    };
    let board = board();

    // GT plane grid: 2 orientations × 2 working distances (the anchor depth
    // z0 where the plane crosses the optical axis; the signed distance is
    // d = -n_z · z0).
    let orientations = [
        Vector3::new(0.30, 0.0, 1.0), // ~16.7° tilt about the y-axis
        Vector3::new(0.0, 0.45, 1.0), // ~24.2° tilt about the x-axis
    ];
    let depths = [0.50_f64, 0.62_f64];
    let noise_levels = [0.0_f64, 0.15, 0.30];

    for (oi, n_raw) in orientations.iter().enumerate() {
        let n_gt = n_raw.normalize();
        for &z0 in &depths {
            let d_gt = -n_gt.z * z0;
            let plane_gt = LaserPlane::new((n_gt).normalize(), d_gt).unwrap();
            let poses = poses(z0);

            for &noise_px in &noise_levels {
                let noise = UniformPixelNoise {
                    seed: 7,
                    max_abs_px: noise_px,
                };
                let dataset = build_dataset(&plane_gt, &gt_intr, &board, &poses, &noise);
                for (i, v) in dataset.iter().enumerate() {
                    assert!(
                        v.meta.laser_pixels.len() >= 8,
                        "orient={oi} z0={z0}: view {i} has only {} laser pixels",
                        v.meta.laser_pixels.len()
                    );
                }

                let export = solve(dataset);
                let plane = &export.laser_plane;

                // Align to the GT sign (the S2 double cover: (n, d) ≡ (−n, −d)).
                let mut n_est = plane.normal().into_inner();
                let mut d_est = plane.distance();
                if n_est.dot(&n_gt) < 0.0 {
                    n_est = -n_est;
                    d_est = -d_est;
                }
                let angle_deg = n_est.dot(&n_gt).clamp(-1.0, 1.0).acos().to_degrees();
                let dist_err_mm = (d_est - d_gt).abs() * 1000.0;
                let mean_laser = export.stats.mean_laser_error; // px (LineDistNormalized)
                let mean_reproj = export.stats.mean_reproj_error; // px

                let label = format!("orient={oi} z0={z0} noise={noise_px}px");
                eprintln!(
                    "{label}: angle={angle_deg:.4}° dist_err={dist_err_mm:.4}mm \
                     laser={mean_laser:.4}px reproj={mean_reproj:.4}px"
                );

                // Recovery tolerances: exact-data cells must be tight; noisy
                // cells scale with the injected amplitude.
                // Exact-data cells recover to solver tolerance; noisy cells
                // scale with the injected amplitude (empirical bounds with
                // ~2× headroom over the observed recovery — see the math
                // note's Evidence table).
                let (angle_tol, dist_tol_mm, laser_bound, reproj_bound) = if noise_px == 0.0 {
                    (0.05, 0.10, 5.0e-3, 1.0e-3)
                } else {
                    (
                        0.6 * noise_px + 0.03,  // deg
                        6.0 * noise_px + 0.30,  // mm
                        noise_px + 0.04,        // px (line-distance residual)
                        0.95 * noise_px + 0.05, // px (reprojection floor ≈ 0.765·a)
                    )
                };

                assert!(
                    angle_deg < angle_tol,
                    "{label}: normal angle {angle_deg:.4}° > {angle_tol}°"
                );
                assert!(
                    dist_err_mm < dist_tol_mm,
                    "{label}: distance error {dist_err_mm:.4} mm > {dist_tol_mm} mm"
                );
                assert!(
                    mean_laser < laser_bound,
                    "{label}: mean laser residual {mean_laser:.4} px > {laser_bound:.4} px"
                );
                assert!(
                    mean_reproj < reproj_bound,
                    "{label}: mean reproj {mean_reproj:.4} px > {reproj_bound:.4} px"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property tests — real invariants of the (n̂, d) plane parameterization.
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic LCG in `[0, 1)` (no external RNG crate; keeps the property
/// sweep reproducible across platforms).
fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * lcg_unit(state)
}

/// The point-to-plane signed distance is invariant to the frame in which the
/// point and plane are expressed: for any SE(3) `T`, `n·p + d` equals
/// `n_B·(T·p) + d_B` where `(n_B, d_B) = T·plane`. This is the geometric fact
/// the laser point-to-plane residual (`optim::factors::laserline`) relies on —
/// the residual is unchanged by any in-plane rotation / rigid re-framing of the
/// plane. `R` preserves the dot product, so equality is exact.
#[test]
fn point_to_plane_residual_is_se3_invariant() {
    let mut s = 0x00C0_FFEE_u64;
    for _ in 0..5000 {
        let n = Vector3::new(
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
        );
        if n.norm() < 1e-3 {
            continue;
        }
        let d = uniform(&mut s, -2.0, 2.0);
        let plane = LaserPlane::new((n).normalize(), d).unwrap();
        let p = Pt3::new(
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
        );
        let t = Iso3::from_parts(
            Translation3::new(
                uniform(&mut s, -1.0, 1.0),
                uniform(&mut s, -1.0, 1.0),
                uniform(&mut s, -1.0, 1.0),
            ),
            Rotation3::from_euler_angles(
                uniform(&mut s, -PI, PI),
                uniform(&mut s, -PI, PI),
                uniform(&mut s, -PI, PI),
            )
            .into(),
        );

        let before = plane.point_distance(&p);
        let after = plane
            .transform_by(&t)
            .point_distance(&t.transform_point(&p));
        assert!(
            (before - after).abs() < 1e-10,
            "SE(3) invariance broken: {before} vs {after}"
        );
    }
}

/// The global sign flip `(n, d) → (−n, −d)` is a gauge of the `(n̂, d)`
/// parameterization: it names the *same* plane (the S² double cover). The
/// signed point-to-plane distance therefore only flips sign; its magnitude —
/// which is what the residual squares — is invariant. The matrix test relies
/// on this to sign-align the recovered normal to the ground truth.
#[test]
fn plane_sign_flip_is_a_gauge() {
    let mut s = 0x0000_1234_u64;
    for _ in 0..5000 {
        let n = Vector3::new(
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
        );
        if n.norm() < 1e-3 {
            continue;
        }
        let d = uniform(&mut s, -2.0, 2.0);
        let p = Pt3::new(
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
            uniform(&mut s, -1.0, 1.0),
        );
        let plane = LaserPlane::new((n).normalize(), d).unwrap();
        let flipped = LaserPlane::new((-n).normalize(), -d).unwrap();
        let a = plane.point_distance(&p);
        let b = flipped.point_distance(&p);
        assert!(
            (a + b).abs() < 1e-10,
            "sign flip is not a gauge: {a} vs {b} (should be negatives)"
        );
    }
}
