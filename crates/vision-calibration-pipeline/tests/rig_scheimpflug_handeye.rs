//! Integration test: Scheimpflug-flavoured rig hand-eye via the unified
//! `RigHandeyeProblem` (post A6.3 collapse).
//!
//! Synthetic EyeInHand rig with 2 cameras and Scheimpflug tilt. Runs all 6
//! steps and asserts intrinsics, hand-eye, and JSON export round-trip.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Pinhole, Pt3, Real, RigDataset,
    RigView, RigViewObs, ScheimpflugParams,
};
use vision_calibration_optim::RobotPoseMeta;
use vision_calibration_pipeline::rig_handeye::{
    RigHandeyeConfig, RigHandeyeExport, RigHandeyeProblem, SensorMode, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

fn make_dataset() -> (
    RigDataset<RobotPoseMeta>,
    FxFyCxCySkew<Real>,
    Isometry3<Real>, // handeye_gt
) {
    let intrinsics = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
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
    ];
    let cameras: Vec<Camera<Real, Pinhole, BrownConrady5<Real>, _, FxFyCxCySkew<Real>>> = sensors
        .iter()
        .map(|s| Camera::new(Pinhole, distortion, s.compile(), intrinsics))
        .collect();

    let cam0_to_rig = Isometry3::identity();
    let cam1_to_rig = Isometry3::from_parts(
        Translation3::new(0.12, 0.0, 0.0),
        Rotation3::identity().into(),
    );
    let cam_to_rig = [cam0_to_rig, cam1_to_rig];

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
                        pts3.push(pw);
                        pts2.push(px);
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

    (RigDataset::new(views, 2).unwrap(), intrinsics, handeye_gt)
}

fn scheimpflug_handeye_config() -> RigHandeyeConfig {
    // `RigHandeyeConfig` is `#[non_exhaustive]` — populate via Default + field
    // assignment from outside the crate.
    let mut cfg = RigHandeyeConfig::default();
    cfg.sensor = SensorMode::Scheimpflug {
        init_tilt_x: 0.0,
        init_tilt_y: 0.0,
        fix_scheimpflug_in_intrinsics: Default::default(),
        refine_scheimpflug_in_rig_ba: false,
    };
    cfg
}

#[test]
fn pipeline_converges_scheimpflug_rig_handeye() {
    let (dataset, intrinsics_gt, handeye_gt) = make_dataset();

    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_config(scheimpflug_handeye_config()).unwrap();
    session.set_input(dataset).unwrap();
    run_calibration(&mut session).unwrap();

    let export = session.export().unwrap();

    // JSON round-trip + sensor field populated.
    let json = serde_json::to_string(&export).unwrap();
    let restored: RigHandeyeExport = serde_json::from_str(&json).unwrap();
    assert!(
        restored.sensors.is_some(),
        "Scheimpflug handeye export must carry sensors"
    );

    // Intrinsics convergence (~5% relative error).
    for (i, cam) in export.cameras.iter().enumerate() {
        let fx_err = (cam.k.fx - intrinsics_gt.fx).abs() / intrinsics_gt.fx;
        let fy_err = (cam.k.fy - intrinsics_gt.fy).abs() / intrinsics_gt.fy;
        assert!(fx_err < 0.05, "cam{i} fx error {:.3}%", fx_err * 100.0);
        assert!(fy_err < 0.05, "cam{i} fy error {:.3}%", fy_err * 100.0);
    }

    // Hand-eye recovery.
    let handeye_final = export
        .gripper_se3_rig
        .expect("EyeInHand should have gripper_se3_rig");
    let dt = (handeye_final.translation.vector - handeye_gt.translation.vector).norm();
    let r_diff = handeye_final.rotation.to_rotation_matrix().transpose()
        * handeye_gt.rotation.to_rotation_matrix();
    let cos_theta = ((r_diff.matrix().trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    let ang = cos_theta.acos();
    assert!(dt < 0.02, "hand-eye translation error {dt:.4}");
    assert!(ang < 0.02, "hand-eye rotation error {ang:.4} rad");

    // Reprojection.
    assert!(
        export.mean_reproj_error < 1.0,
        "mean reproj error too large: {}",
        export.mean_reproj_error
    );
}

#[test]
fn pipeline_rejects_insufficient_views() {
    let (mut dataset, _, _) = make_dataset();
    dataset.views.truncate(2);
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_config(scheimpflug_handeye_config()).unwrap();
    let err = session.set_input(dataset).unwrap_err().to_string();
    assert!(err.contains("need 3"), "unexpected error: {err}");
}
