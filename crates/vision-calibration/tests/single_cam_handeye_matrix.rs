//! Synthetic-GT matrix test for the single-camera hand-eye family (Q8,
//! `docs/notes/README.md`; math note: `docs/notes/hand-eye.md`).
//!
//! Ground-truth grid × noise levels: every cell builds an exact synthetic
//! dataset via `vision_calibration::synthetic`, runs the *standard*
//! `step_intrinsics_init → step_intrinsics_optimize → step_handeye_init →
//! step_handeye_optimize` pipeline, and asserts hand-eye rotation/translation
//! recovery plus a reprojection error consistent with the injected noise
//! floor. It also covers the EyeToHand convention and the base-frame gauge
//! invariance of the recovered hand-eye transform.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration::core::{
    BrownConrady5, FxFyCxCySkew, Iso3, PinholeCamera, Pt3, make_pinhole_camera,
};
use vision_calibration::optim::HandEyeMode;
use vision_calibration::session::CalibrationSession;
use vision_calibration::single_cam_handeye::{
    HandeyeMeta, SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeInput,
    SingleCamHandeyeProblem, SingleCamHandeyeView, step_handeye_init, step_handeye_optimize,
    step_intrinsics_init, step_intrinsics_optimize,
};
use vision_calibration::synthetic::{noise::UniformPixelNoise, planar};

/// One cell of the ground-truth grid: a focal regime and a hand-eye transform.
struct GtCell {
    fx: f64,
    fy: f64,
    /// Hand-eye ground truth (`gripper_se3_camera`, `T_G_C`) for EyeInHand.
    handeye: Iso3,
    label: &'static str,
}

/// Board shared by every cell: a 9×7 grid of 30 mm cells (63 points).
fn board() -> Vec<Pt3> {
    planar::grid_points(9, 7, 0.03)
}

/// Robot stations (`base_se3_gripper`, `T_B_G`) with strongly diverse rotation
/// axes. Hand-eye identifiability *requires* ≥ 2 relative motions with
/// non-parallel rotation axes (see the math note, §identifiability): the
/// roll/pitch/yaw mix below supplies that, and the translation ramp spreads
/// the target distance so Zhang's intrinsics init is well posed too.
fn robot_poses() -> Vec<Iso3> {
    // (euler angles rad, translation m) per station.
    type Station = ((f64, f64, f64), (f64, f64, f64));
    let specs: [Station; 9] = [
        ((0.00, 0.00, 0.00), (0.00, 0.00, 0.00)),
        ((0.30, 0.00, 0.00), (0.10, 0.00, 0.00)), // roll
        ((0.00, 0.30, 0.00), (0.00, 0.10, 0.00)), // pitch
        ((0.00, 0.00, 0.30), (0.00, 0.00, 0.08)), // yaw
        ((0.22, 0.20, 0.00), (0.05, -0.05, 0.02)),
        ((-0.24, 0.00, 0.20), (-0.05, 0.05, 0.03)),
        ((0.16, -0.18, 0.12), (0.02, -0.04, 0.05)),
        ((-0.15, 0.22, -0.10), (-0.03, 0.03, 0.04)),
        ((0.20, -0.10, 0.24), (0.04, 0.02, 0.01)),
    ];
    specs
        .iter()
        .map(|&(angles, t)| make_iso(angles, t))
        .collect()
}

fn make_iso(angles: (f64, f64, f64), t: (f64, f64, f64)) -> Iso3 {
    let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
    Iso3::from_parts(
        Translation3::new(t.0, t.1, t.2),
        UnitQuaternion::from_rotation_matrix(&rot),
    )
}

fn camera(fx: f64, fy: f64) -> PinholeCamera {
    make_pinhole_camera(
        FxFyCxCySkew {
            fx,
            fy,
            cx: 512.0,
            cy: 384.0,
            skew: 0.0,
        },
        // Zero distortion: this family's proof pack isolates the hand-eye
        // geometry; the distortion sub-model is covered by the planar pack.
        BrownConrady5::default(),
    )
}

/// (Δtranslation in metres, Δrotation angle in radians) between two poses.
fn pose_error(a: &Iso3, b: &Iso3) -> (f64, f64) {
    let dt = (a.translation.vector - b.translation.vector).norm();
    let r_diff = a.rotation.inverse() * b.rotation;
    (dt, r_diff.angle())
}

/// Build the per-view EyeInHand observations for a ground-truth
/// `(camera, handeye X = T_G_C, target Y = T_B_T)`, with deterministic pixel
/// noise. The camera pose per view is `T_C_T_i = (T_B_G_i · X)^-1 · Y`.
fn eye_in_hand_views(
    camera: &PinholeCamera,
    board: &[Pt3],
    robot_poses: &[Iso3],
    handeye: &Iso3,
    target_in_base: &Iso3,
    noise: &UniformPixelNoise,
) -> Vec<SingleCamHandeyeView> {
    let cam_poses: Vec<Iso3> = robot_poses
        .iter()
        .map(|tbg| (tbg * handeye).inverse() * target_in_base)
        .collect();
    let obs = planar::project_views_noisy(camera, board, &cam_poses, noise)
        .expect("all board points projectable in every view");
    obs.into_iter()
        .zip(robot_poses.iter())
        .map(|(obs, tbg)| SingleCamHandeyeView {
            obs,
            meta: HandeyeMeta {
                base_se3_gripper: *tbg,
            },
        })
        .collect()
}

/// Run the standard four-step single-cam hand-eye pipeline and export.
fn solve(input: SingleCamHandeyeInput, mode: HandEyeMode) -> SingleCamHandeyeExport {
    solve_with(input, mode, true)
}

/// Same, with control over robot-pose refinement (the default is `true`).
fn solve_with(
    input: SingleCamHandeyeInput,
    mode: HandEyeMode,
    refine_robot_poses: bool,
) -> SingleCamHandeyeExport {
    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    let mut config = SingleCamHandeyeConfig::default();
    config.handeye_mode = mode;
    config.refine_robot_poses = refine_robot_poses;
    session.set_config(config).expect("config");
    session.set_input(input).expect("input");
    step_intrinsics_init(&mut session, None).expect("intrinsics init");
    step_intrinsics_optimize(&mut session, None).expect("intrinsics optimize");
    step_handeye_init(&mut session, None).expect("handeye init");
    step_handeye_optimize(&mut session, None).expect("handeye optimize");
    session.export().expect("export")
}

#[test]
fn single_cam_handeye_recovers_gt_across_grid_and_noise() {
    // Two hand-eye regimes: a modest offset and a larger rotated/translated one.
    let x_small = make_iso((0.10, -0.05, 0.02), (0.05, -0.03, 0.08));
    let x_large = make_iso((0.25, 0.18, -0.12), (-0.09, 0.06, 0.12));

    let grid = [
        GtCell {
            fx: 800.0,
            fy: 780.0,
            handeye: x_small,
            label: "f800/X_small",
        },
        GtCell {
            fx: 800.0,
            fy: 780.0,
            handeye: x_large,
            label: "f800/X_large",
        },
        GtCell {
            fx: 1200.0,
            fy: 1200.0,
            handeye: x_small,
            label: "f1200/X_small",
        },
    ];
    let noise_levels = [0.0, 0.15, 0.30];

    // Target fixed in the base frame, ~1 m in front, mildly tilted so the
    // near-identity robot station is not degenerately fronto-parallel.
    let target_in_base = make_iso((0.05, -0.08, 0.0), (-0.10, -0.08, 1.0));

    let board = board();
    let robot_poses = robot_poses();

    for cell in &grid {
        let cam_gt = camera(cell.fx, cell.fy);
        for &noise_px in &noise_levels {
            let noise = UniformPixelNoise {
                seed: 7,
                max_abs_px: noise_px,
            };
            let views = eye_in_hand_views(
                &cam_gt,
                &board,
                &robot_poses,
                &cell.handeye,
                &target_in_base,
                &noise,
            );
            let input = SingleCamHandeyeInput::new(views).expect("input");
            let export = solve(input, HandEyeMode::EyeInHand);

            let x_est = export
                .gripper_se3_camera
                .expect("EyeInHand populates gripper_se3_camera");
            let (dt, dang) = pose_error(&x_est, &cell.handeye);
            let label = format!("{} noise={noise_px}px", cell.label);

            // Focal recovery (relative). Exact data must be tight; noisy cells
            // scale with the injected amplitude.
            let focal_tol = if noise_px == 0.0 { 0.003 } else { 0.03 };
            assert!(
                (export.camera.k.fx - cell.fx).abs() / cell.fx < focal_tol
                    && (export.camera.k.fy - cell.fy).abs() / cell.fy < focal_tol,
                "{label}: focal ({}, {}) vs gt ({}, {})",
                export.camera.k.fx,
                export.camera.k.fy,
                cell.fx,
                cell.fy
            );

            // Hand-eye rotation recovery (radians). 1e-3 rad ≈ 0.057°.
            let ang_tol = if noise_px == 0.0 { 2.0e-3 } else { 0.02 };
            assert!(
                dang < ang_tol,
                "{label}: hand-eye rotation err {dang} rad > {ang_tol}"
            );

            // Hand-eye translation recovery (metres). The translation DOF is
            // the most noise-sensitive part of the chain (see the math note,
            // §noise sensitivity): it is the ratio of pose residuals to the
            // rank-deficient (R_A − I) blocks.
            let trans_tol = if noise_px == 0.0 { 2.0e-3 } else { 8.0e-3 };
            assert!(
                dt < trans_tol,
                "{label}: hand-eye translation err {dt} m > {trans_tol}"
            );

            // Residual floor: uniform noise in [−a, a] per axis has expected
            // error-norm mean ≈ 0.765·a; near-zero on exact data.
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

#[test]
fn single_cam_handeye_eye_to_hand_recovers_gt() {
    // EyeToHand: a scene-fixed camera observes a gripper-mounted target.
    //   handeye = camera_se3_base (T_C_B), target = gripper_se3_target (T_G_T)
    //   T_C_T_i = T_C_B · T_B_G_i · T_G_T
    let cam_gt = camera(950.0, 940.0);
    let camera_se3_base = make_iso((0.08, -0.06, 0.03), (0.02, 0.04, 1.2));
    let gripper_se3_target = make_iso((0.03, 0.05, -0.02), (0.03, -0.02, 0.02));

    let board = board();
    let robot_poses = robot_poses();

    for &noise_px in &[0.0, 0.2] {
        let noise = UniformPixelNoise {
            seed: 11,
            max_abs_px: noise_px,
        };
        let cam_poses: Vec<Iso3> = robot_poses
            .iter()
            .map(|tbg| camera_se3_base * tbg * gripper_se3_target)
            .collect();
        let obs =
            planar::project_views_noisy(&cam_gt, &board, &cam_poses, &noise).expect("projectable");
        let views: Vec<SingleCamHandeyeView> = obs
            .into_iter()
            .zip(robot_poses.iter())
            .map(|(obs, tbg)| SingleCamHandeyeView {
                obs,
                meta: HandeyeMeta {
                    base_se3_gripper: *tbg,
                },
            })
            .collect();
        let input = SingleCamHandeyeInput::new(views).expect("input");
        let export = solve(input, HandEyeMode::EyeToHand);

        let x_est = export
            .camera_se3_base
            .expect("EyeToHand populates camera_se3_base");
        let g_est = export
            .gripper_se3_target
            .expect("EyeToHand populates gripper_se3_target");
        let (dt_x, dang_x) = pose_error(&x_est, &camera_se3_base);
        let (dt_g, dang_g) = pose_error(&g_est, &gripper_se3_target);

        // `camera_se3_base` is a *derived*, distant pose (target ~1.2 m away),
        // so its translation is the least-constrained DOF; allow it a little
        // more headroom than the direct DLT output `gripper_se3_target`.
        let (ang_tol, trans_tol, reproj_bound) = if noise_px == 0.0 {
            (3.0e-3, 3.0e-3, 1e-3)
        } else {
            (0.02, 1.2e-2, noise_px * 0.9 + 0.05)
        };
        assert!(
            dang_x < ang_tol && dang_g < ang_tol,
            "noise={noise_px}px: rotation err camera_se3_base={dang_x} gripper_se3_target={dang_g} > {ang_tol}"
        );
        assert!(
            dt_x < trans_tol && dt_g < trans_tol,
            "noise={noise_px}px: translation err camera_se3_base={dt_x} gripper_se3_target={dt_g} > {trans_tol}"
        );
        assert!(
            export.mean_reproj_error <= reproj_bound,
            "noise={noise_px}px: mean reproj {} > {reproj_bound}",
            export.mean_reproj_error
        );
    }
}

#[test]
fn single_cam_handeye_base_frame_gauge_invariance() {
    // Rebasing the robot base frame (T_B_G_i → G · T_B_G_i, T_B_T → G · T_B_T)
    // leaves every camera-frame observation identical:
    //   (G·T_B_G · X)^-1 · (G·T_B_T) = X^-1 · T_B_G^-1 · T_B_T.
    // So the recovered hand-eye X and the reprojection error must be invariant
    // to G, while only the fixed target pose absorbs it. This is a genuine
    // gauge of the joint reprojection cost (see the math note, §gauge).
    //
    // Robot-pose refinement is disabled here: the zero-mean se(3) priors are
    // written in the base frame via a left-multiplicative correction, so under
    // rebasing the penalty transforms by Ad_G and is *not* invariant (it mildly
    // breaks this gauge — a real, documented subtlety). With the priors off the
    // reprojection cost is exactly gauge-invariant.
    let cam_gt = camera(880.0, 870.0);
    let handeye = make_iso((0.12, -0.07, 0.04), (0.06, -0.02, 0.09));
    let target_in_base = make_iso((0.04, -0.06, 0.0), (-0.08, -0.05, 1.0));
    let board = board();
    let robot_poses = robot_poses();
    let noise = UniformPixelNoise {
        seed: 5,
        max_abs_px: 0.1,
    };

    let solve_for = |poses: &[Iso3], target: &Iso3| -> SingleCamHandeyeExport {
        let views = eye_in_hand_views(&cam_gt, &board, poses, &handeye, target, &noise);
        let input = SingleCamHandeyeInput::new(views).expect("input");
        solve_with(input, HandEyeMode::EyeInHand, false)
    };

    let base = solve_for(&robot_poses, &target_in_base);

    // Arbitrary rigid rebasing of the base frame.
    let g = make_iso((0.20, -0.35, 0.15), (0.3, -0.4, 0.2));
    let rebased_poses: Vec<Iso3> = robot_poses.iter().map(|p| g * p).collect();
    let rebased_target = g * target_in_base;
    let rebased = solve_for(&rebased_poses, &rebased_target);

    let x_base = base.gripper_se3_camera.unwrap();
    let x_rebased = rebased.gripper_se3_camera.unwrap();
    let (dt, dang) = pose_error(&x_base, &x_rebased);

    // The hand-eye X is gauge-invariant; both solves converge from the same
    // noisy pixels to the same X up to solver tolerance.
    assert!(
        dt < 1e-6 && dang < 1e-6,
        "hand-eye not gauge-invariant under base rebasing: dt={dt} m, dang={dang} rad"
    );
    assert!(
        (base.mean_reproj_error - rebased.mean_reproj_error).abs() < 1e-9,
        "reprojection error not gauge-invariant: {} vs {}",
        base.mean_reproj_error,
        rebased.mean_reproj_error
    );
}
