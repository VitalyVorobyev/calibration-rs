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

use nalgebra::{SVector, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCySkew, NoMeta, PerFeatureResiduals, RigView,
    RigViewObs, compute_rig_target_residuals, make_pinhole_camera,
};
use vision_calibration_optim::HandEyeMode;

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
    let k = FxFyCxCySkew {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
        skew: 0.0,
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
    let report = LevelReport::from_residuals(ReprojLevel::HandEye, constrained, 1, 1);
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
        vec![RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![Some(obs.clone()), Some(obs)],
            },
        }],
        2,
    )
    .unwrap();
    let cameras = vec![camera.clone(), camera.clone()];
    let target = compute_rig_target_residuals(&cameras, &dataset, &cam_se3_rig, &rig_se3_target)
        .expect("residuals");
    let mut per_feature_residuals = PerFeatureResiduals::default();
    per_feature_residuals.target = target;
    let export = RigHandeyeExport {
        cameras,
        sensors: None,
        cam_se3_rig: cam_se3_rig.clone(),
        rig_se3_target: rig_se3_target.clone(),
        handeye_mode: HandEyeMode::EyeToHand,
        gripper_se3_rig: None,
        rig_se3_base: Some(Iso3::identity()),
        base_se3_target: None,
        gripper_se3_target: Some(Iso3::identity()),
        robot_deltas: None,
        mean_reproj_error: 0.0,
        per_cam_reproj_errors: vec![0.0, 0.0],
        per_feature_residuals,
        image_manifest: None,
    };
    let rig_stage = RigStageReprojection {
        cam_se3_rig,
        rig_se3_target,
    };

    let report = rig_handeye_report_with_rig_stage(&export, &dataset, &rig_stage).unwrap();
    let levels: Vec<_> = report.levels.iter().map(|level| level.level).collect();
    assert_eq!(
        levels,
        vec![
            ReprojLevel::Intrinsic,
            ReprojLevel::RigExtrinsic,
            ReprojLevel::HandEye
        ]
    );
    assert_eq!(report.headline_px, report.levels[2].overall.mean);
}
