//! Integration test: the recovery of a badly fitted tilted camera in a rig
//! hand-eye calibration.
//!
//! Synthetic EyeInHand rig of three tilted cameras, auto-initialized; the
//! third has corner noise of about a pixel, so after its own solve it fits
//! worse than the recovery threshold and the intrinsics stage re-solves it
//! from a nominal seed of the two well-fitted cameras, then against their
//! rig. The calibration must keep every camera at its noise floor and
//! recover the hand-eye transform.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraModelSpec, CorrespondenceView, DistortionKind, FxFyCxCy, Pinhole,
    Pt3, Real, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorKind,
};
use vision_calibration_optim::RobotPoseMeta;
use vision_calibration_pipeline::common::config::CameraConfig;
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
