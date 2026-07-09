//! Synthetic-GT matrix test for the multi-camera rig extrinsics family (Q8,
//! `docs/notes/README.md`; math note: `docs/notes/rig-extrinsics.md`).
//!
//! Ground-truth grid × noise levels: every cell builds an exact synthetic
//! two-camera rig via `vision_calibration::synthetic`, runs the *standard*
//! `step_intrinsics_init_all → step_intrinsics_optimize_all → step_rig_init →
//! step_rig_optimize` pipeline, and asserts inter-camera rotation/translation
//! (baseline) recovery plus a reprojection error consistent with the injected
//! noise floor. It also covers:
//!
//! - the reference-camera gauge: shifting which camera pins the rig frame
//!   re-expresses every extrinsic but leaves the relative inter-camera pose
//!   and the reprojection error invariant (§gauge of the math note);
//! - the minimum-overlap requirement: a disconnected co-visibility graph
//!   (no view where a camera and the reference co-observe the target) is
//!   rejected by rig init (§identifiability of the math note).

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration::core::{
    BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, NoMeta, PinholeCamera, Pt3, RigDataset,
    RigView, RigViewObs, make_pinhole_camera,
};
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsInput, RigExtrinsicsProblem,
    RigIntrinsicsManualInit, run_calibration, step_intrinsics_init_all_with_seed, step_rig_init,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::{noise::UniformPixelNoise, planar};

// ─────────────────────────────────────────────────────────────────────────────
// Ground-truth fixtures
// ─────────────────────────────────────────────────────────────────────────────

/// One cell of the ground-truth grid: an inter-camera extrinsic (`cam1_se3_rig`,
/// i.e. the relative pose T_C1_C0 because camera 0 pins the rig frame).
struct GtCell {
    /// `cam1_se3_rig` = T_C1_R (rig = camera 0): the observable relative pose.
    cam1_se3_rig: Iso3,
    label: &'static str,
}

fn make_iso(angles: (f64, f64, f64), t: (f64, f64, f64)) -> Iso3 {
    let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
    Iso3::from_parts(
        Translation3::new(t.0, t.1, t.2),
        UnitQuaternion::from_rotation_matrix(&rot),
    )
}

/// Two rig cameras with distinct intrinsics (exercises per-camera intrinsics
/// handling). Zero distortion isolates the extrinsics geometry — the distortion
/// sub-model is covered by the planar pack.
fn rig_cameras() -> [PinholeCamera; 2] {
    [
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 850.0,
                fy: 845.0,
                cx: 640.0,
                cy: 480.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        ),
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 820.0,
                fy: 815.0,
                cx: 650.0,
                cy: 470.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        ),
    ]
}

/// Board shared by every cell: an 8×6 grid of 30 mm cells (48 points).
fn board() -> Vec<Pt3> {
    planar::grid_points(8, 6, 0.03)
}

/// Rig poses (`rig_se3_target`, T_R_T) with strongly diverse pitch/yaw and a
/// distance ramp. Orientation diversity is required for the per-camera Zhang
/// init to be well posed *and* for the inter-camera extrinsics to be observable
/// from the shared views (see the math note, §identifiability). The board is
/// re-centred on the optical axis so the rotations stay near fronto-parallel.
fn rig_poses() -> Vec<Iso3> {
    let angles: [(f64, f64); 8] = [
        (0.00, 0.00),
        (0.16, -0.06),
        (-0.14, 0.10),
        (0.08, 0.18),
        (-0.06, -0.16),
        (0.20, 0.12),
        (-0.18, -0.10),
        (0.11, -0.20),
    ];
    angles
        .iter()
        .enumerate()
        .map(|(i, (pitch, yaw))| {
            Iso3::from_parts(
                // Board spans [0, 0.21] × [0, 0.15] m; recentre and ramp 0.60→0.81 m.
                Translation3::new(-0.105, -0.075, 0.60 + 0.03 * i as f64),
                Rotation3::from_euler_angles(*pitch, *yaw, 0.0).into(),
            )
        })
        .collect()
}

/// (Δtranslation in metres, Δrotation angle in radians) between two poses.
fn pose_error(a: &Iso3, b: &Iso3) -> (f64, f64) {
    let dt = (a.translation.vector - b.translation.vector).norm();
    let r_diff = a.rotation.inverse() * b.rotation;
    (dt, r_diff.angle())
}

/// Relative pose T_C1_C0 = `cam_se3_rig[1] · cam_se3_rig[0]⁻¹` — the
/// gauge-invariant observable (independent of which camera pins the rig frame).
fn relative_pose(cam_se3_rig: &[Iso3]) -> Iso3 {
    cam_se3_rig[1] * cam_se3_rig[0].inverse()
}

/// Build a synthetic two-camera rig dataset. `cam_se3_rig[c]` is the ground-truth
/// camera-from-rig transform (camera 0 is the rig frame ⇒ identity). Each camera
/// gets an independent deterministic noise stream (`seed + cam_idx`) so the two
/// views of a pose are not correlated. Every camera observes every view (full
/// co-visibility).
fn build_rig_dataset(
    cameras: &[PinholeCamera],
    cam_se3_rig: &[Iso3],
    board: &[Pt3],
    rig_poses: &[Iso3],
    seed: u64,
    noise_px: f64,
) -> RigExtrinsicsInput {
    let num_cameras = cameras.len();
    // Per-camera projected views: `[cam][view]`.
    let per_cam_views: Vec<Vec<CorrespondenceView>> = cameras
        .iter()
        .enumerate()
        .map(|(cam_idx, cam)| {
            let cam_poses: Vec<Iso3> = rig_poses
                .iter()
                .map(|rig_se3_target| cam_se3_rig[cam_idx] * rig_se3_target)
                .collect();
            let noise = UniformPixelNoise {
                seed: seed + cam_idx as u64,
                max_abs_px: noise_px,
            };
            planar::project_views_noisy(cam, board, &cam_poses, &noise)
                .expect("all board points projectable in every camera/view")
        })
        .collect();

    let views: Vec<RigView<NoMeta>> = (0..rig_poses.len())
        .map(|view_idx| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: (0..num_cameras)
                    .map(|cam_idx| Some(per_cam_views[cam_idx][view_idx].clone()))
                    .collect(),
            },
        })
        .collect();

    RigDataset::new(views, num_cameras).expect("valid rig dataset")
}

/// Run the standard four-step rig pipeline with the given reference camera and
/// return the export.
fn solve(input: RigExtrinsicsInput, reference_camera_idx: usize) -> RigExtrinsicsExport {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    let mut config = RigExtrinsicsConfig::default();
    config.rig.reference_camera_idx = reference_camera_idx;
    session.set_config(config).expect("config");
    session.set_input(input).expect("input");
    run_calibration(&mut session).expect("calibration");
    session.export().expect("export")
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix test
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rig_extrinsics_recovers_gt_across_grid_and_noise() {
    // Two rig geometries: a narrow-baseline near-parallel stereo pair and a
    // wide-baseline strongly converged pair.
    let grid = [
        GtCell {
            cam1_se3_rig: make_iso((0.0, 0.05, 0.0), (-0.06, 0.0, 0.0)),
            label: "narrow_baseline",
        },
        GtCell {
            cam1_se3_rig: make_iso((0.02, 0.14, 0.01), (-0.25, 0.0, 0.02)),
            label: "wide_baseline",
        },
    ];
    let noise_levels = [0.0, 0.15, 0.30];

    let cameras = rig_cameras();
    let board = board();
    let rig_poses = rig_poses();

    for cell in &grid {
        // Camera 0 pins the rig frame (identity); camera 1 carries the GT pose.
        let cam_se3_rig_gt = [Iso3::identity(), cell.cam1_se3_rig];
        let gt_baseline = cell.cam1_se3_rig.translation.vector.norm();

        for &noise_px in &noise_levels {
            let input = build_rig_dataset(
                &cameras,
                &cam_se3_rig_gt,
                &board,
                &rig_poses,
                /*seed=*/ 20,
                noise_px,
            );
            let export = solve(input, /*reference_camera_idx=*/ 0);
            let label = format!("{} noise={noise_px}px", cell.label);

            assert_eq!(export.cam_se3_rig.len(), 2, "{label}: camera count");
            assert!(
                export.sensors.is_none(),
                "{label}: pinhole rig has no sensors"
            );

            // Reference camera pins the rig frame ⇒ its extrinsic is identity.
            let (dt_ref, dang_ref) = pose_error(&export.cam_se3_rig[0], &Iso3::identity());
            assert!(
                dt_ref < 1e-9 && dang_ref < 1e-9,
                "{label}: reference camera extrinsic not identity (dt={dt_ref}, dang={dang_ref})"
            );

            // Inter-camera pose recovery. With camera 0 as reference, the
            // observable relative pose T_C1_C0 equals cam_se3_rig[1].
            let rel_est = relative_pose(&export.cam_se3_rig);
            let (dt, dang) = pose_error(&rel_est, &cell.cam1_se3_rig);

            // Rotation recovery (radians). Exact data hits solver tolerance;
            // noisy cells scale linearly in the injected amplitude
            // (≈ 2e-2 rad/px — see the math note, §noise sensitivity).
            let ang_tol = if noise_px == 0.0 {
                5.0e-4
            } else {
                noise_px * 0.03 + 5.0e-4
            };
            assert!(
                dang < ang_tol,
                "{label}: inter-camera rotation err {dang} rad > {ang_tol}"
            );

            // Baseline / translation recovery (metres). The inter-camera
            // translation is the most noise-sensitive DOF: the strongly
            // converged wide pair couples ≈ 6e-3 rad of rotation error into
            // ≈ 2 cm of translation error at 0.3 px, roughly 3× the near-parallel
            // narrow pair. Both scale linearly in the injected amplitude.
            let trans_tol = if noise_px == 0.0 {
                5.0e-4
            } else {
                noise_px * 0.085 + 1.0e-3
            };
            assert!(
                dt < trans_tol,
                "{label}: inter-camera translation err {dt} m > {trans_tol} (gt baseline {gt_baseline} m)"
            );

            // Residual floor: uniform noise in [−a, a] per axis has expected
            // error-norm mean ≈ 0.765·a; near-zero on exact data. The rig cost
            // couples both cameras, so allow generous headroom on noisy cells.
            let reproj_bound = if noise_px == 0.0 {
                1e-3
            } else {
                noise_px * 0.9 + 0.05
            };
            assert!(
                export.mean_reproj_error <= reproj_bound,
                "{label}: mean reproj {} > bound {reproj_bound}",
                export.mean_reproj_error
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property test — reference-camera gauge invariance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rig_extrinsics_reference_camera_gauge_invariance() {
    // The rig frame is a gauge: which camera pins it (identity extrinsic) only
    // re-expresses every `cam_se3_rig`, it does not change the observable
    // relative inter-camera pose or the reprojection error. Solving the *same*
    // pixels with reference camera 0 vs. camera 1 must therefore agree on the
    // relative pose T_C1_C0 and on the mean reprojection error, while each
    // solve's reference camera lands at identity (see the math note, §gauge).
    let cameras = rig_cameras();
    let board = board();
    let rig_poses = rig_poses();
    let cam_se3_rig_gt = [
        Iso3::identity(),
        make_iso((0.02, 0.12, 0.01), (-0.20, 0.01, 0.02)),
    ];

    // A single noisy dataset drives both solves.
    let input = build_rig_dataset(
        &cameras,
        &cam_se3_rig_gt,
        &board,
        &rig_poses,
        /*seed=*/ 31,
        /*noise_px=*/ 0.15,
    );

    let export_ref0 = solve(input.clone(), 0);
    let export_ref1 = solve(input, 1);

    // Each solve pins its own reference camera to identity.
    let (dt0, dang0) = pose_error(&export_ref0.cam_se3_rig[0], &Iso3::identity());
    let (dt1, dang1) = pose_error(&export_ref1.cam_se3_rig[1], &Iso3::identity());
    assert!(
        dt0 < 1e-9 && dang0 < 1e-9,
        "ref0 solve: camera 0 not identity (dt={dt0}, dang={dang0})"
    );
    assert!(
        dt1 < 1e-9 && dang1 < 1e-9,
        "ref1 solve: camera 1 not identity (dt={dt1}, dang={dang1})"
    );

    // The absolute extrinsics differ between the two gauges …
    let rel0 = relative_pose(&export_ref0.cam_se3_rig);
    let rel1 = relative_pose(&export_ref1.cam_se3_rig);
    let expressed_differently = (export_ref0.cam_se3_rig[1].translation.vector
        - Iso3::identity().translation.vector)
        .norm()
        > 1e-3;
    assert!(
        expressed_differently,
        "sanity: camera-1 extrinsic under ref0 should be non-identity"
    );

    // … but the observable relative pose is gauge-invariant.
    let (dt, dang) = pose_error(&rel0, &rel1);
    assert!(
        dt < 5e-4 && dang < 5e-4,
        "relative inter-camera pose not gauge-invariant under reference shift: dt={dt} m, dang={dang} rad"
    );

    // The reprojection error is a property of the reconstructed geometry, not
    // of the frame choice — it is invariant to the reference camera. The two
    // solves converge to the same minimum from different gauge fixings, so the
    // residual gap is bounded by solver tolerance, not by the gauge.
    assert!(
        (export_ref0.mean_reproj_error - export_ref1.mean_reproj_error).abs() < 5e-3,
        "reprojection error not gauge-invariant: {} vs {}",
        export_ref0.mean_reproj_error,
        export_ref1.mean_reproj_error
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Degeneracy test — disconnected co-visibility graph
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rig_extrinsics_rejects_disconnected_covisibility_graph() {
    // Linear rig init recovers camera 1's extrinsic from views where BOTH
    // camera 1 and the reference co-observe the target. If the co-visibility
    // graph is disconnected — camera 0 sees views {0..3}, camera 1 sees views
    // {4..7}, no shared view — the relative pose is unobservable and rig init
    // must fail with an overlap error (see the math note, §identifiability).
    let cameras = rig_cameras();
    let board = board();
    let rig_poses = rig_poses();
    let cam_se3_rig_gt = [
        Iso3::identity(),
        make_iso((0.0, 0.1, 0.0), (-0.15, 0.0, 0.0)),
    ];

    // Project every camera in every view (exact), then blank out the overlap.
    let per_cam_views: Vec<Vec<CorrespondenceView>> = cameras
        .iter()
        .enumerate()
        .map(|(cam_idx, cam)| {
            let cam_poses: Vec<Iso3> = rig_poses
                .iter()
                .map(|rig_se3_target| cam_se3_rig_gt[cam_idx] * rig_se3_target)
                .collect();
            planar::project_views_all(cam, &board, &cam_poses).expect("projectable")
        })
        .collect();

    let views: Vec<RigView<NoMeta>> = (0..rig_poses.len())
        .map(|view_idx| {
            // Camera 0 owns the first half of the views, camera 1 the second —
            // each keeps ≥ 3 views (enough for per-camera Zhang) but they never
            // share a view.
            let cam0 = (view_idx < 4).then(|| per_cam_views[0][view_idx].clone());
            let cam1 = (view_idx >= 4).then(|| per_cam_views[1][view_idx].clone());
            RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: vec![cam0, cam1],
                },
            }
        })
        .collect();
    let input = RigDataset::new(views, 2).expect("valid rig dataset");

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(input).expect("input");

    // Seed both cameras with their GT intrinsics so the per-camera stage is
    // trivially well posed and the test isolates the connectivity failure.
    let mut seed = RigIntrinsicsManualInit::default();
    seed.per_cam_intrinsics = Some(cameras.iter().map(|c| c.k).collect());
    step_intrinsics_init_all_with_seed(&mut session, seed, None).expect("intrinsics init");

    let err = step_rig_init(&mut session).expect_err("disconnected graph must fail rig init");
    let msg = err.to_string();
    assert!(
        msg.contains("overlap"),
        "error should report the missing co-visibility overlap, got: {msg}"
    );
}
