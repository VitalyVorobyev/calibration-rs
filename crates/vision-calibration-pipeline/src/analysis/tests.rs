//! Synthetic-ground-truth tests for the multi-level reprojection report.
//!
//! The diagnostic value of the report rests on two claims, validated here on
//! synthetic data with a known camera and known board poses:
//!
//! 1. the **intrinsic floor** recovers the injected corner-detection noise
//!    level (so the floor reflects detection, not the constraints), and
//! 2. constraining the board pose **adds a known error** on top of that floor
//!    (so the level delta recovers the injected error budget).

use super::*;

use crate::common::{ProblemKind, VersionedExport};
use nalgebra::{SVector, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCy, NoMeta, PerFeatureResiduals, Pinhole,
    PinholeCamera, RigView, RigViewObs, ScheimpflugParams, compute_rig_target_residuals,
    make_pinhole_camera, pinhole_camera_params,
};

/// Tiny deterministic xorshift64* RNG — keeps the synthetic noise reproducible
/// without a `rand` dependency.
struct Lcg(u64);
impl Lcg {
    /// Next sample, uniform in `[-1, 1)` (std ≈ 1/√3).
    fn next_unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn test_camera() -> PinholeCamera {
    let k = FxFyCxCy {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
    };
    let dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    make_pinhole_camera(k, dist)
}

/// A planar chessboard grid of 3D target points (z = 0).
fn board(rows: usize, cols: usize, pitch: f64) -> Vec<Pt3> {
    let mut pts = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            pts.push(Pt3::new(j as f64 * pitch, i as f64 * pitch, 0.0));
        }
    }
    pts
}

/// A board pose in front of the camera: shifted to be visible, tilted a bit.
fn pose_in_front(tx: f64, ty: f64, tz: f64, tilt: f64) -> Iso3 {
    let rot = UnitQuaternion::from_scaled_axis(Vector3::new(tilt, tilt * 0.5, 0.0));
    Iso3::from_parts(Translation3::new(tx, ty, tz), rot)
}

fn project_all(camera: &impl CameraProject, pose: &Iso3, pts: &[Pt3]) -> Vec<Pt2> {
    pts.iter()
        .map(|p| {
            let pc = pose * p;
            camera
                .project_camera_point(&pc.coords)
                .expect("point projects in front of camera")
        })
        .collect()
}

#[test]
fn intrinsic_floor_uses_scheimpflug_sensor_when_present() {
    let pinhole = test_camera();
    let sensor = ScheimpflugParams {
        tilt_x: 0.08,
        tilt_y: -0.01,
    };
    let scheimpflug = Camera::new(Pinhole, pinhole.dist, sensor.compile(), pinhole.k);
    let pts3 = board(7, 9, 0.03);
    let true_pose = pose_in_front(-0.05, -0.04, 0.62, 0.09);
    let observed = project_all(&scheimpflug, &true_pose, &pts3);

    let pose_pinhole = intrinsic_floor_view(&pinhole, &pinhole.k, &pts3, &observed)
        .expect("pinhole floor pose solves");
    let pose_scheimpflug = intrinsic_floor_view(&scheimpflug, &pinhole.k, &pts3, &observed)
        .expect("scheimpflug floor pose solves");

    let mut pinhole_floor = Vec::new();
    push_view_residuals(
        &mut pinhole_floor,
        &pinhole,
        &pose_pinhole,
        &pts3,
        &observed,
        0,
        0,
    );
    let mut scheimpflug_floor = Vec::new();
    push_view_residuals(
        &mut scheimpflug_floor,
        &scheimpflug,
        &pose_scheimpflug,
        &pts3,
        &observed,
        0,
        0,
    );

    let pinhole_stats = LevelStats::from_residuals(&pinhole_floor);
    let scheimpflug_stats = LevelStats::from_residuals(&scheimpflug_floor);
    assert!(
        scheimpflug_stats.mean < 1e-3,
        "tilted-sensor synthetic data should have near-zero Scheimpflug floor, got {}",
        scheimpflug_stats.mean
    );
    assert!(
        pinhole_stats.mean > scheimpflug_stats.mean + 0.1,
        "pinhole-only floor should disagree on tilted data: pinhole {} vs scheimpflug {}",
        pinhole_stats.mean,
        scheimpflug_stats.mean
    );
}

/// The intrinsic floor recovers the injected pixel-noise level: with a correct
/// camera model and a free per-view pose, the residual RMS should be close to
/// the corner-detection noise σ.
#[test]
fn intrinsic_floor_recovers_noise_level() {
    let camera = test_camera();
    let pts3 = board(7, 9, 0.03); // 63 corners — well over the 6 DOF.
    let sigma = 0.25_f64;
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);

    let mut errs = Vec::new();
    for (v, pose) in [
        pose_in_front(-0.10, -0.08, 0.55, 0.10),
        pose_in_front(0.05, -0.05, 0.70, -0.08),
        pose_in_front(-0.02, 0.06, 0.60, 0.05),
    ]
    .iter()
    .enumerate()
    {
        let clean = project_all(&camera, pose, &pts3);
        let noisy: Vec<Pt2> = clean
            .iter()
            .map(|p| {
                // Uniform[-1,1) has std 1/√3; scale by σ·√3 to get std ≈ σ.
                Pt2::new(
                    p.x + rng.next_unit() * sigma * 1.7320508,
                    p.y + rng.next_unit() * sigma * 1.7320508,
                )
            })
            .collect();
        let pose_hat =
            intrinsic_floor_view(&camera, &camera.k, &pts3, &noisy).expect("floor pose solves");
        let mut recs = Vec::new();
        push_view_residuals(&mut recs, &camera, &pose_hat, &pts3, &noisy, v, 0);
        errs.extend(recs.iter().filter_map(|r| r.error_px));
    }

    let stats = LevelStats::from_errors(&mut errs);
    // `error_px` is the 2D magnitude √(du²+dv²). For independent per-axis noise
    // of std σ, its RMS is ≈ σ·√2 (each axis contributes σ²), shrunk only
    // slightly by the 6 pose DOF the per-view floor absorbs over many corners.
    // So the floor RMS should land near σ·√2, NOT σ — a floor near σ·√2 with a
    // near-zero clean-data floor (see `constraint_adds_known_error`) is exactly
    // the detector-noise floor we want the diagnostic to report.
    let expected = sigma * std::f64::consts::SQRT_2;
    assert!(
        (stats.rms - expected).abs() < 0.08,
        "floor rms {} should be ≈ σ·√2 = {}",
        stats.rms,
        expected
    );
}

/// A pose perturbation away from the per-view optimum raises the error by a
/// known amount: the level delta recovers the injected budget. Clean data → the
/// intrinsic floor is ~0; projecting through a perturbed pose yields a clearly
/// larger error.
#[test]
fn constraint_adds_known_error() {
    let camera = test_camera();
    let pts3 = board(7, 9, 0.03);
    let true_pose = pose_in_front(0.0, 0.0, 0.6, 0.07);
    let clean = project_all(&camera, &true_pose, &pts3);

    // Floor: recompute the pose freely from clean data → near-zero error.
    let pose_hat =
        intrinsic_floor_view(&camera, &camera.k, &pts3, &clean).expect("floor pose solves");
    let mut floor = Vec::new();
    push_view_residuals(&mut floor, &camera, &pose_hat, &pts3, &clean, 0, 0);
    let floor_stats = LevelStats::from_residuals(&floor);
    assert!(
        floor_stats.mean < 1e-3,
        "clean-data floor should be ~0, got {}",
        floor_stats.mean
    );

    // Constrained: project through a pose rotated ~0.6° about x.
    let perturbed = apply_left_tangent(
        &true_pose,
        &SVector::<f64, 6>::from_column_slice(&[0.010, 0.0, 0.0, 0.0, 0.0, 0.0]),
    );
    let mut constrained = Vec::new();
    push_view_residuals(&mut constrained, &camera, &perturbed, &pts3, &clean, 0, 0);
    let report = LevelReport::from_residuals(ReprojLevel::Handeye, constrained, 1, 1);
    assert!(
        report.overall.mean > floor_stats.mean + 0.5,
        "perturbed pose should add clear error: constrained {} vs floor {}",
        report.overall.mean,
        floor_stats.mean
    );
}

#[test]
fn level_stats_percentiles_linear() {
    let mut errs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let s = LevelStats::from_errors(&mut errs);
    assert_eq!(s.count, 5);
    assert!((s.mean - 3.0).abs() < 1e-12);
    assert!((s.median - 3.0).abs() < 1e-12);
    assert!((s.max - 5.0).abs() < 1e-12);
    // p95 with linear interpolation: rank = 0.95*4 = 3.8 → 4 + 0.8*(5-4) = 4.8.
    assert!((s.p95 - 4.8).abs() < 1e-9, "p95={}", s.p95);

    let empty = LevelStats::from_errors(&mut []);
    assert_eq!(empty.count, 0);
    assert_eq!(empty.mean, 0.0);
}

#[test]
fn report_serde_roundtrip() {
    let mut r = TargetFeatureResidual::default();
    r.observed_px = [1.0, 2.0];
    r.projected_px = Some([1.1, 2.1]);
    r.error_px = Some(0.1414);
    let level = LevelReport::from_residuals(ReprojLevel::Intrinsic, vec![r], 1, 1);
    let report = ReprojReport::from_levels(vec![level]);

    let json = serde_json::to_string(&report).unwrap();
    let back: ReprojReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.levels.len(), 1);
    assert_eq!(back.levels[0].level, ReprojLevel::Intrinsic);
    assert!((back.headline_px - report.headline_px).abs() < 1e-12);
}

#[test]
fn rig_handeye_report_with_rig_stage_has_three_levels() {
    let camera = test_camera();
    let pts3 = board(3, 3, 0.03);
    let rig_se3_target = vec![pose_in_front(0.0, 0.0, 0.7, 0.04)];
    let cam_se3_rig = vec![Iso3::identity(), Iso3::identity()];
    let observed = project_all(&camera, &rig_se3_target[0], &pts3);
    let obs = CorrespondenceView::new(pts3.clone(), observed).unwrap();
    let dataset = RigDataset::new(
        vec![RigView::new(
            RigViewObs::new(vec![Some(obs.clone()), Some(obs)]),
            NoMeta,
        )],
        2,
    )
    .unwrap();
    let cameras = vec![camera.clone(), camera.clone()];
    let target = compute_rig_target_residuals(&cameras, &dataset, &cam_se3_rig, &rig_se3_target)
        .expect("residuals");
    let mut per_feature_residuals = PerFeatureResiduals::default();
    per_feature_residuals.target = target;
    let export = RigHandeyeExport {
        kind: ProblemKind::RigHandeye,
        schema_version: RigHandeyeExport::SCHEMA_VERSION,
        cameras: cameras.iter().map(pinhole_camera_params).collect(),
        cam_se3_rig: cam_se3_rig.clone(),
        rig_se3_target: rig_se3_target.clone(),
        handeye: crate::common::HandeyeSolution::EyeToHand {
            rig_se3_base: Iso3::identity(),
            gripper_se3_target: Iso3::identity(),
        },
        robot_refinement: None,
        mean_reproj_error: 0.0,
        per_cam_reproj_errors: vec![0.0, 0.0],
        diagnostics: crate::common::ExportDiagnostics::new(per_feature_residuals),
        uncertainty: Default::default(),
    };
    let rig_stage = RigStageReprojection::new(export.cameras.clone(), cam_se3_rig, rig_se3_target);

    let report = rig_handeye_report_with_rig_stage(&export, &dataset, &rig_stage).unwrap();
    let levels: Vec<_> = report.levels.iter().map(|level| level.level).collect();
    assert_eq!(
        levels,
        vec![
            ReprojLevel::Intrinsic,
            ReprojLevel::RigExtrinsic,
            ReprojLevel::Handeye
        ]
    );
    assert_eq!(report.headline_px, report.levels[2].overall.mean);
}

/// A two-camera eye-in-hand rig scene with corner noise whose *measured* robot
/// poses carry a small error: the true poses project the corners, the
/// perturbed ones go into the view metadata.
fn rig_handeye_input_with_robot_error() -> crate::rig_handeye::RigHandeyeInput {
    use crate::common::RobotPoseMeta;
    use nalgebra::Rotation3;

    let iso = |r: (f64, f64, f64), t: (f64, f64, f64)| {
        Iso3::from_parts(
            Translation3::new(t.0, t.1, t.2),
            Rotation3::from_euler_angles(r.0, r.1, r.2).into(),
        )
    };
    let camera = |offset: f64| {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0 + offset,
                fy: 780.0 + offset,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5::default(),
        )
    };
    let cameras = [camera(0.0), camera(10.0)];
    let cam_se3_rig = [Iso3::identity(), iso((0.0, 0.0, 0.1), (0.2, 0.0, 0.0))];
    let gripper_se3_rig = iso((0.05, -0.03, 0.02), (0.03, -0.02, 0.08));
    let base_se3_target = iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.2));
    let pts3 = board(5, 6, 0.05);
    let robot_poses = [
        iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        iso((0.3, 0.0, 0.0), (0.0, -0.3, 0.0)),
        iso((0.0, 0.3, 0.0), (0.3, 0.0, 0.0)),
        iso((0.2, 0.2, 0.0), (0.2, -0.2, 0.0)),
        iso((-0.3, 0.0, 0.1), (0.0, 0.3, 0.05)),
        iso((0.0, -0.3, -0.1), (-0.3, 0.0, -0.05)),
        iso((0.25, -0.2, 0.2), (-0.2, -0.25, 0.0)),
        iso((-0.2, 0.25, -0.2), (0.25, 0.2, 0.05)),
    ];

    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let views = robot_poses
        .iter()
        .map(|base_se3_gripper| {
            let rig_se3_target = (base_se3_gripper * gripper_se3_rig).inverse() * base_se3_target;
            let obs = cameras
                .iter()
                .zip(&cam_se3_rig)
                .map(|(camera, cam_se3_rig)| {
                    let pts2 = project_all(camera, &(cam_se3_rig * rig_se3_target), &pts3)
                        .into_iter()
                        .map(|p| {
                            Pt2::new(p.x + 0.15 * rng.next_unit(), p.y + 0.15 * rng.next_unit())
                        })
                        .collect();
                    Some(CorrespondenceView::new(pts3.clone(), pts2).unwrap())
                })
                .collect();
            // About 0.05° and 0.5 mm of robot error per view.
            let delta = SVector::<f64, 6>::from_fn(|i, _| {
                let scale = if i < 3 { 0.001 } else { 0.0005 };
                scale * rng.next_unit()
            });
            let measured = apply_left_tangent(base_se3_gripper, &delta);
            RigView::new(RigViewObs::new(obs), RobotPoseMeta::new(measured))
        })
        .collect();
    RigDataset::new(views, 2).unwrap()
}

/// With rigid robot poses the hand-eye BA moves the intrinsics away from the
/// rig stage's to absorb the robot error. The rig-extrinsic level must still
/// read the rig stage's own fit — its cameras, rig extrinsics and target
/// poses — at the noise level, not the rig-stage poses seen through the final
/// cameras.
#[test]
fn rig_extrinsic_level_reads_the_rig_stage_cameras() {
    use crate::common::config::RobotPoseConfig;
    use crate::rig_handeye::{
        RigHandeyeConfig, RigHandeyeProblem, step_handeye_init, step_handeye_optimize,
        step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
    };
    use crate::session::CalibrationSession;

    let dataset = rig_handeye_input_with_robot_error();
    let mut config = RigHandeyeConfig::default();
    config.handeye_optimize.robot_poses = RobotPoseConfig::Exact;
    config.handeye_optimize.refine_intrinsics = true;
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_config(config).unwrap();
    session.set_input(dataset.clone()).unwrap();
    step_intrinsics_init_all(&mut session, None).unwrap();
    step_intrinsics_optimize_all(&mut session, None).unwrap();
    step_rig_init(&mut session, None).unwrap();
    let rig_stage_fit = step_rig_optimize(&mut session, None).unwrap();
    step_handeye_init(&mut session, None).unwrap();
    step_handeye_optimize(&mut session, None).unwrap();
    let export = session.export().unwrap();

    // The rig stage's own residuals, recomputed from its state.
    let stage_cameras = session.state.rig.cameras.clone().unwrap();
    let stage_cam_se3_rig = session.state.rig_ba_cam_se3_rig.clone().unwrap();
    let stage_rig_se3_target = session.state.rig_ba_rig_se3_target.clone().unwrap();
    let stage_residuals = compute_rig_target_residuals(
        &rig_camera_models(&stage_cameras).unwrap(),
        &dataset,
        &stage_cam_se3_rig,
        &stage_rig_se3_target,
    )
    .unwrap();
    let stage_mean = LevelStats::from_residuals(&stage_residuals).mean;
    assert!(
        (stage_mean - rig_stage_fit.mean_reproj_error).abs() < 1e-9,
        "recomputed rig-stage mean {stage_mean} vs the step's {}",
        rig_stage_fit.mean_reproj_error
    );

    // The setup: the final cameras moved away from the rig stage's.
    let dfx = (export.cameras[0].intrinsics.fx - stage_cameras[0].intrinsics.fx).abs();
    assert!(
        dfx > 0.5,
        "final fx should move off the rig stage's, Δfx = {dfx}"
    );

    let rig_stage = RigStageReprojection::from_session(&session).unwrap();
    assert_eq!(
        serde_json::to_value(&rig_stage.cameras).unwrap(),
        serde_json::to_value(&stage_cameras).unwrap()
    );
    assert_eq!(rig_stage.cam_se3_rig, stage_cam_se3_rig);
    assert_eq!(rig_stage.rig_se3_target, stage_rig_se3_target);
    let report = rig_handeye_report_with_rig_stage(&export, &dataset, &rig_stage).unwrap();
    let rig_level = report.level(ReprojLevel::RigExtrinsic).unwrap();
    let handeye_level = report.level(ReprojLevel::Handeye).unwrap();
    assert!(
        (rig_level.overall.mean - stage_mean).abs() < 1e-9,
        "rig-extrinsic level {} px should equal the rig stage's own {} px",
        rig_level.overall.mean,
        stage_mean
    );
    assert!(
        stage_mean < 0.2,
        "rig stage at the noise level, got {stage_mean} px"
    );
    assert!(
        handeye_level.overall.mean > 2.0 * rig_level.overall.mean,
        "hand-eye level {} px should exceed the rig-extrinsic level {} px",
        handeye_level.overall.mean,
        rig_level.overall.mean
    );
}
