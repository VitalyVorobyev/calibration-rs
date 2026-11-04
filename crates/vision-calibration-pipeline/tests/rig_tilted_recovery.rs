//! Integration tests: the recovery of the tilted cameras of a rig
//! auto-initialized from scratch, which the rig intrinsics stage of every
//! rig problem type runs.
//!
//! - A synthetic EyeInHand rig of three tilted cameras; the third has corner
//!   noise of about a pixel, so after its own solve it fits worse than the
//!   recovery threshold and the intrinsics stage re-solves it from a nominal
//!   seed of the two well-fitted cameras, then against their rig. The
//!   calibration must keep every camera at its noise floor and recover the
//!   hand-eye transform.
//! - A synthetic rig of four strongly distorted, tilted cameras seen in a
//!   few nearly fronto-parallel views (no robot): one camera's own staged
//!   solve settles in a wrong tilt / principal-point basin at twice the
//!   noise floor; the wide tilt multi-start must bring every camera's own
//!   solve below that, and the rig solve to the noise floor and the truth,
//!   on both solver backends.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::synthetic::{noise::UniformPixelNoise, planar, poses};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    FxFyCxCy, Iso3, NoMeta, Pinhole, Pt3, Real, RigDataset, RigView, RigViewObs, ScheimpflugParams,
    SensorKind, SensorParams,
};
use vision_calibration_optim::{RobotPoseMeta, SolverBackend};
use vision_calibration_pipeline::common::config::CameraConfig;
use vision_calibration_pipeline::rig_extrinsics::{
    self, RigExtrinsicsConfig, RigExtrinsicsProblem, step_intrinsics_init_all,
    step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
};
use vision_calibration_pipeline::rig_handeye::{
    RigHandeyeConfig, RigHandeyeProblem, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

fn make_dataset(
    noise_px: [Real; 3],
    tilt2: ScheimpflugParams,
) -> (
    RigDataset<RobotPoseMeta>,
    FxFyCxCy<Real>,
    Isometry3<Real>, // handeye_gt
) {
    let intrinsics = FxFyCxCy {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 5,
    };
    let sensors = [
        ScheimpflugParams::default(),
        ScheimpflugParams {
            tilt_x: 0.05,
            tilt_y: -0.03,
        },
        tilt2,
    ];
    let cameras: Vec<Camera<Real, Pinhole, BrownConrady5<Real>, _, FxFyCxCy<Real>>> = sensors
        .iter()
        .map(|s| Camera::new(Pinhole, distortion, s.compile(), intrinsics))
        .collect();

    let cam0_to_rig = Isometry3::identity();
    let cam1_to_rig = Isometry3::from_parts(
        Translation3::new(0.12, 0.0, 0.0),
        Rotation3::identity().into(),
    );
    let cam2_to_rig = Isometry3::from_parts(
        Translation3::new(-0.10, 0.02, 0.0),
        Rotation3::from_euler_angles(0.0, 0.05, 0.0).into(),
    );
    let cam_to_rig = [cam0_to_rig, cam1_to_rig, cam2_to_rig];

    let handeye_gt = Isometry3::from_parts(
        Translation3::new(0.05, 0.03, 0.15),
        Rotation3::from_euler_angles(0.1, -0.05, 0.15).into(),
    );

    let target_pose_gt = Isometry3::identity();

    let robot_poses = vec![
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, -1.0),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.05, 0.05, -1.05),
            Rotation3::from_euler_angles(0.0, 0.0, 0.1).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, -0.05, -0.95),
            Rotation3::from_euler_angles(0.0, 0.0, -0.1).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.03, 0.03, -1.02),
            Rotation3::from_euler_angles(0.0, 0.0, 0.05).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.02, 0.04, -1.05),
            Rotation3::from_euler_angles(0.02, 0.03, -0.05).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.04, -0.02, -0.98),
            Rotation3::from_euler_angles(-0.05, 0.04, 0.08).into(),
        ),
    ];

    let mut views = Vec::new();
    for robot_pose in &robot_poses {
        let mut cam_obs = Vec::new();
        for (cam_idx, c2r) in cam_to_rig.iter().enumerate() {
            let mut pts3 = Vec::new();
            let mut pts2 = Vec::new();
            for y in -2..=2i32 {
                for x in -3..=3i32 {
                    let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                    let p_base = target_pose_gt.transform_point(&pw);
                    let p_gripper = robot_pose.inverse_transform_point(&p_base);
                    let p_rig = handeye_gt.inverse_transform_point(&p_gripper);
                    let p_cam = c2r.inverse_transform_point(&p_rig);
                    if let Some(px) = cameras[cam_idx].project_point(&p_cam) {
                        let k = (pts2.len() * 7 + cam_idx * 13 + views.len() * 31) as Real;
                        let n = noise_px[cam_idx];
                        pts3.push(pw);
                        pts2.push(
                            px + nalgebra::Vector2::new(n * (k * 1.37).sin(), n * (k * 2.11).cos()),
                        );
                    }
                }
            }
            cam_obs.push(Some(CorrespondenceView::new(pts3, pts2).unwrap()));
        }
        views.push(RigView {
            obs: RigViewObs { cameras: cam_obs },
            meta: RobotPoseMeta {
                base_se3_gripper: *robot_pose,
            },
        });
    }

    (RigDataset::new(views, 3).unwrap(), intrinsics, handeye_gt)
}

fn scheimpflug_handeye_config() -> RigHandeyeConfig {
    // `RigHandeyeConfig` is `#[non_exhaustive]` — populate via Default + field
    // assignment from outside the crate.
    let mut cfg = RigHandeyeConfig::default();
    cfg.camera = CameraConfig::new(CameraModelSpec::pinhole(
        DistortionKind::BrownConrady5,
        SensorKind::Scheimpflug,
    ));
    cfg
}

#[test]
fn a_noisy_tilted_camera_goes_through_the_nominal_recovery() {
    let noise = [0.05, 0.05, 1.0];
    let (dataset, intrinsics_gt, handeye_gt) = make_dataset(
        noise,
        ScheimpflugParams {
            tilt_x: -0.04,
            tilt_y: 0.02,
        },
    );
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_config(scheimpflug_handeye_config()).unwrap();
    session.set_input(dataset).unwrap();
    run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();

    // Each camera at its noise floor: the corner offsets have a mean norm
    // of about the noise amplitude.
    for (i, (err, n)) in export.per_cam_reproj_errors.iter().zip(noise).enumerate() {
        assert!(*err < 1.1 * n, "camera {i}: {err:.3} px for noise {n} px");
    }
    for (i, cam) in export.cameras.iter().enumerate() {
        let fx_err = (cam.intrinsics.fx - intrinsics_gt.fx).abs() / intrinsics_gt.fx;
        assert!(fx_err < 0.05, "cam{i} fx error {:.3}%", fx_err * 100.0);
    }
    let handeye = export.gripper_se3_rig.expect("EyeInHand export");
    let dt = (handeye.translation.vector - handeye_gt.translation.vector).norm();
    assert!(dt < 0.02, "hand-eye translation error {dt:.4} m");
}

/// Per-axis pixel noise of the four-camera rig, px.
const RIG_NOISE_PX: Real = 0.2;
/// Mean reprojection error at the noise floor of the rig solve: the mean
/// norm of the injected noise is about 0.26 px.
const RIG_FLOOR_MEAN_PX: Real = 0.30;
/// Bound on each camera's own solve: one camera stops at about 0.30 px in
/// a shallow neighbour of its minimum (its principal point some 40 px off),
/// which the rig solve refines; the wrong basin sits at about 0.56 px.
const RIG_STAGE_MEAN_PX: Real = 0.40;
/// Principal-point error bound, px: about 5σ of the rig solve (σ ≈ 1.1 to
/// 1.4 px); the wrong basin is about 300 px off.
const RIG_PP_TOL_PX: Real = 7.5;
/// Tilt error bound, degrees; the wrong basin is about 14° off.
const RIG_TILT_TOL_DEG: Real = 0.5;

/// One camera of the four-camera rig: strong barrel distortion and a mount
/// tilt of about −5°, each camera a little different.
struct TiltedTruth {
    k: FxFyCxCy<Real>,
    k1: Real,
    tilt: ScheimpflugParams,
}

fn tilted_rig_truth() -> [TiltedTruth; 4] {
    let camera = |fx: Real, cx: Real, cy: Real, k1: Real, tilt_deg: (Real, Real)| TiltedTruth {
        k: FxFyCxCy { fx, fy: fx, cx, cy },
        k1,
        tilt: ScheimpflugParams {
            tilt_x: tilt_deg.0.to_radians(),
            tilt_y: tilt_deg.1.to_radians(),
        },
    };
    [
        camera(1160.0, 367.5, 265.0, -0.38, (-4.5, 0.25)),
        camera(1142.0, 354.0, 274.0, -0.416, (-5.4, -0.2)),
        camera(1146.0, 357.0, 272.0, -0.408, (-5.2, -0.1)),
        camera(1150.0, 360.0, 270.0, -0.40, (-5.0, 0.0)),
    ]
}

/// Six views of a 13 × 10 board (10 mm cells), tilted at most 0.1 rad, at
/// about 0.3 m from a rig of four cameras 20 mm apart, toed in on the board.
fn tilted_rig_dataset() -> RigDataset<NoMeta> {
    let num_views = 6;
    let board = planar::grid_points(13, 10, 0.01);
    let rig_se3_target = poses::tilted_board_poses(
        &poses::seeded_tilts(num_views, 0.10, 4),
        (0.06, 0.045),
        0.30,
        0.01,
    );
    let layout = poses::rig_layout(4, 0.02, 0.02 / 0.30);
    let per_cam: Vec<Vec<CorrespondenceView>> = tilted_rig_truth()
        .iter()
        .zip(&layout)
        .enumerate()
        .map(|(c, (truth, cam_se3_rig))| {
            let distortion = BrownConrady5 {
                k1: truth.k1,
                k2: 0.15,
                ..BrownConrady5::default()
            };
            let camera = Camera::new(Pinhole, distortion, truth.tilt.compile(), truth.k);
            let cam_se3_target: Vec<Iso3> =
                rig_se3_target.iter().map(|t| cam_se3_rig * t).collect();
            let noise = UniformPixelNoise {
                seed: 404 + c as u64,
                max_abs_px: RIG_NOISE_PX * 3.0_f64.sqrt(),
            };
            planar::project_views_noisy(&camera, &board, &cam_se3_target, &noise)
                .expect("every corner projects")
        })
        .collect();
    let views = (0..num_views)
        .map(|v| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: per_cam.iter().map(|views| Some(views[v].clone())).collect(),
            },
        })
        .collect();
    RigDataset::new(views, 4).expect("rig dataset")
}

fn assert_at_truth(label: &str, cameras: &[CameraParams]) {
    for (i, (camera, truth)) in cameras.iter().zip(tilted_rig_truth()).enumerate() {
        let k = camera.intrinsics;
        let pp_err = (k.cx - truth.k.cx).hypot(k.cy - truth.k.cy);
        let SensorParams::Scheimpflug { params: tilt } = camera.sensor else {
            panic!("{label}: camera {i} lost its Scheimpflug sensor");
        };
        let tilt_err_deg = (tilt.tilt_x - truth.tilt.tilt_x)
            .hypot(tilt.tilt_y - truth.tilt.tilt_y)
            .to_degrees();
        assert!(
            pp_err < RIG_PP_TOL_PX,
            "{label}: camera {i} principal point {pp_err:.1} px off"
        );
        assert!(
            tilt_err_deg < RIG_TILT_TOL_DEG,
            "{label}: camera {i} tilt {tilt_err_deg:.2}° off"
        );
    }
}

#[test]
fn rig_extrinsics_recovers_a_tilted_camera_in_a_wrong_basin() {
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let mut config = RigExtrinsicsConfig::default();
        config.camera = CameraConfig::new(CameraModelSpec::pinhole(
            DistortionKind::BrownConrady5,
            SensorKind::Scheimpflug,
        ));
        config.solver.backend = backend;
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_config(config).unwrap();
        session.set_input(tilted_rig_dataset()).unwrap();

        step_intrinsics_init_all(&mut session, None).unwrap();
        let stage = step_intrinsics_optimize_all(&mut session, None).unwrap();
        for (i, err) in stage.per_cam_reproj_errors.iter().enumerate() {
            assert!(
                *err < RIG_STAGE_MEAN_PX,
                "{backend:?}: camera {i} own solve at {err:.3} px"
            );
        }

        step_rig_init(&mut session, None).unwrap();
        step_rig_optimize(&mut session, None).unwrap();
        let export: rig_extrinsics::RigExtrinsicsExport = session.export().unwrap();
        assert!(
            export.mean_reproj_error < RIG_FLOOR_MEAN_PX,
            "{backend:?}: rig mean reprojection error {:.3} px",
            export.mean_reproj_error
        );
        assert_at_truth(&format!("{backend:?}"), &export.cameras);
    }
}
