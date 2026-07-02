//! Synthetic-GT matrix test for planar intrinsics — the template proof-pack
//! test (Q1, `docs/notes/README.md`; math note:
//! `docs/notes/planar-intrinsics.md`).
//!
//! Ground-truth grid × noise levels: every cell builds an exact synthetic
//! dataset via `vision_calibration::synthetic`, runs the standard
//! `step_init → step_optimize` pipeline, and asserts parameter recovery and
//! a reprojection error consistent with the injected noise floor.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3};
use vision_calibration::core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, IntrinsicsParams, Iso3, Pinhole,
    PlanarDataset, Pt3, View,
};
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsProblem, step_init, step_optimize};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::{noise::UniformPixelNoise, planar};

/// One cell of the ground-truth grid.
struct GtCell {
    fx: f64,
    fy: f64,
    k1: f64,
    k2: f64,
}

/// Board + pose set shared by every cell: an 11×9 grid of 25 mm cells seen
/// from 8 diverse orientations (mixed pitch/yaw, varying distance). Pose
/// diversity matters — near-coplanar plane normals leave the image of the
/// absolute conic underconstrained (see the math note, §identifiability).
fn board() -> Vec<Pt3> {
    planar::grid_points(11, 9, 0.025)
}

fn poses() -> Vec<Iso3> {
    let angles: [(f64, f64); 8] = [
        (0.00, 0.00),
        (0.18, -0.06),
        (-0.15, 0.10),
        (0.08, 0.20),
        (-0.06, -0.18),
        (0.22, 0.14),
        (-0.20, -0.10),
        (0.12, -0.22),
    ];
    angles
        .iter()
        .enumerate()
        .map(|(i, (pitch, yaw))| {
            // Center the 0.25 × 0.2 m board on the optical axis, distance
            // ramping 0.55 → 0.90 m.
            Iso3::from_parts(
                Translation3::new(-0.125, -0.1, 0.55 + 0.05 * i as f64),
                Rotation3::from_euler_angles(*pitch, *yaw, 0.0).into(),
            )
        })
        .collect()
}

/// Run the standard planar pipeline on one synthetic dataset; return
/// (recovered intrinsics, recovered distortion, mean reprojection px).
fn solve(dataset: PlanarDataset) -> (FxFyCxCySkew<f64>, BrownConrady5<f64>, f64) {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset).expect("input");
    step_init(&mut session, None).expect("init");
    step_optimize(&mut session, None).expect("optimize");
    let out = session.output().expect("output");
    let IntrinsicsParams::FxFyCxCySkew { params: intr } = out.params.camera.intrinsics;
    let dist = match &out.params.camera.distortion {
        vision_calibration::core::DistortionParams::BrownConrady5 { params } => *params,
        other => panic!("unexpected distortion params: {other:?}"),
    };
    (intr, dist, out.mean_reproj_error)
}

#[test]
fn planar_intrinsics_recovers_gt_across_grid_and_noise() {
    // Ground-truth grid: two focal regimes × three distortion strengths.
    let grid: Vec<GtCell> = [
        (800.0, 810.0, 0.0, 0.0),
        (800.0, 810.0, -0.20, 0.08),
        (800.0, 810.0, -0.40, 0.18),
        (1400.0, 1385.0, 0.0, 0.0),
        (1400.0, 1385.0, -0.20, 0.08),
        (1400.0, 1385.0, -0.40, 0.18),
    ]
    .iter()
    .map(|&(fx, fy, k1, k2)| GtCell { fx, fy, k1, k2 })
    .collect();
    // Uniform per-axis pixel noise amplitudes (deterministic, seeded).
    let noise_levels = [0.0, 0.15, 0.30];

    let board = board();
    let poses = poses();

    for cell in &grid {
        for &noise_px in &noise_levels {
            let gt_intr = FxFyCxCySkew {
                fx: cell.fx,
                fy: cell.fy,
                cx: 366.0,
                cy: 263.0,
                skew: 0.0,
            };
            let gt_dist = BrownConrady5 {
                k1: cell.k1,
                k2: cell.k2,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            };
            let camera = Camera::new(Pinhole, gt_dist, IdentitySensor, gt_intr);
            let noise = UniformPixelNoise {
                seed: 42,
                max_abs_px: noise_px,
            };
            let views = planar::project_views_noisy(&camera, &board, &poses, &noise)
                .expect("all board points projectable")
                .into_iter()
                .map(View::without_meta)
                .collect();
            let dataset = PlanarDataset::new(views).expect("dataset");

            let (intr, dist, mean_reproj) = solve(dataset);

            let label = format!("cell fx={} k1={} noise={noise_px}px", cell.fx, cell.k1);
            // Focal recovery: exact-data cells must be tight; noisy cells
            // scale with the noise amplitude (empirical bound with margin).
            let focal_tol = if noise_px == 0.0 { 0.002 } else { 0.02 };
            assert!(
                (intr.fx - gt_intr.fx).abs() / gt_intr.fx < focal_tol,
                "{label}: fx {} vs gt {}",
                intr.fx,
                gt_intr.fx
            );
            assert!(
                (intr.fy - gt_intr.fy).abs() / gt_intr.fy < focal_tol,
                "{label}: fy {} vs gt {}",
                intr.fy,
                gt_intr.fy
            );
            // Principal point within ~noise-scaled pixels.
            let pp_tol = if noise_px == 0.0 { 0.5 } else { 8.0 };
            assert!(
                (intr.cx - gt_intr.cx).abs() < pp_tol && (intr.cy - gt_intr.cy).abs() < pp_tol,
                "{label}: pp ({}, {}) vs gt ({}, {})",
                intr.cx,
                intr.cy,
                gt_intr.cx,
                gt_intr.cy
            );
            // Distortion recovery (k3/p1/p2 fixed at zero by default config,
            // matching the GT).
            let k_tol = if noise_px == 0.0 { 0.005 } else { 0.05 };
            assert!(
                (dist.k1 - gt_dist.k1).abs() < k_tol,
                "{label}: k1 {} vs gt {}",
                dist.k1,
                gt_dist.k1
            );
            // Residual floor: uniform noise in [-a, a] per axis has expected
            // error-norm mean ≈ 0.765·a; allow generous headroom, and demand
            // near-zero residuals on exact data.
            // (1e-4 px on exact data: comfortably above the solver's
            // convergence tolerance, far below any physical noise floor.)
            let reproj_bound = if noise_px == 0.0 {
                1e-4
            } else {
                noise_px * 0.9 + 0.02
            };
            assert!(
                mean_reproj <= reproj_bound,
                "{label}: mean reproj {mean_reproj} > bound {reproj_bound}"
            );
        }
    }
}
