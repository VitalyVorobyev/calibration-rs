//! Pipeline integration test for `rig_scheimpflug_extrinsics`.
//!
//! Mirrors `laserline_device.rs` in structure: synthetic 2-camera rig with
//! Scheimpflug tilt, 4 views, runs all 4 steps, checks convergence and JSON
//! export round-trip.

use nalgebra::{Isometry3, Rotation3, Translation3};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, NoMeta, Pinhole, Pt3, Real,
    RigDataset, RigView, RigViewObs, ScheimpflugParams, make_pinhole_camera,
};
use vision_calibration_pipeline::rig_scheimpflug_extrinsics::{
    RigScheimpflugExtrinsicsExport, RigScheimpflugExtrinsicsProblem, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

fn make_dataset() -> (
    RigDataset<NoMeta>,
    [FxFyCxCySkew<Real>; 2],
    [ScheimpflugParams; 2],
    Isometry3<Real>,
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
            tilt_x: 0.06,
            tilt_y: -0.04,
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

    let rig_from_target = vec![
        Isometry3::from_parts(
            Translation3::new(0.1, -0.05, 1.0),
            Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.08, 0.03, 1.1),
            Rotation3::from_euler_angles(-0.08, 0.06, -0.15).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, 0.08, 0.95),
            Rotation3::from_euler_angles(0.05, -0.08, 0.12).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.02, 0.06, 1.05),
            Rotation3::from_euler_angles(-0.05, 0.10, 0.08).into(),
        ),
    ];

    let mut views = Vec::new();
    for rig_pose in &rig_from_target {
        let mut cam_obs = Vec::new();
        for (cam_idx, c2r) in cam_to_rig.iter().enumerate() {
            let mut pts3 = Vec::new();
            let mut pts2 = Vec::new();
            for y in -2..=2i32 {
                for x in -3..=3i32 {
                    let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                    let p_rig = rig_pose.transform_point(&pw);
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
            meta: NoMeta,
        });
    }

    let pinhole_k = [
        make_pinhole_camera(intrinsics, distortion),
        make_pinhole_camera(intrinsics, distortion),
    ];
    let _ = pinhole_k; // used for reference

    let intrinsics_arr = [intrinsics, intrinsics];
    (
        RigDataset::new(views, 2).unwrap(),
        intrinsics_arr,
        sensors,
        cam1_to_rig,
    )
}

#[test]
fn pipeline_converges_scheimpflug_rig_extrinsics() {
    let (dataset, intrinsics_gt, _sensors_gt, cam1_to_rig_gt) = make_dataset();

    let mut session = CalibrationSession::<RigScheimpflugExtrinsicsProblem>::new();
    session.set_input(dataset).unwrap();
    run_calibration(&mut session).unwrap();

    let export = session.export().unwrap();

    // JSON round-trip.
    let json = serde_json::to_string(&export).unwrap();
    let _: RigScheimpflugExtrinsicsExport = serde_json::from_str(&json).unwrap();

    // Intrinsics convergence (~5% relative error).
    for (i, cam) in export.cameras.iter().enumerate() {
        let gt = intrinsics_gt[i];
        let fx_err = (cam.k.fx - gt.fx).abs() / gt.fx;
        let fy_err = (cam.k.fy - gt.fy).abs() / gt.fy;
        assert!(
            fx_err < 0.05,
            "cam{i} fx relative error {:.3}%",
            fx_err * 100.0
        );
        assert!(
            fy_err < 0.05,
            "cam{i} fy relative error {:.3}%",
            fy_err * 100.0
        );
    }

    // Camera-1 extrinsics recovery.
    let cam1_final = export.cam_se3_rig[1].inverse(); // cam_se3_rig is T_C_R; we need T_R_C
    let dt = (cam1_final.translation.vector - cam1_to_rig_gt.translation.vector).norm();
    assert!(dt < 0.02, "cam1 translation error {dt:.4}");

    // Reprojection error below 1 px.
    assert!(
        export.mean_reproj_error < 1.0,
        "mean reproj error too large: {}",
        export.mean_reproj_error
    );
}

#[test]
fn pipeline_rejects_insufficient_views() {
    let (mut dataset, _, _, _) = make_dataset();
    dataset.views.truncate(1);
    let mut session = CalibrationSession::<RigScheimpflugExtrinsicsProblem>::new();
    let err = session.set_input(dataset).unwrap_err().to_string();
    assert!(err.contains("need 3"), "unexpected error: {err}");
}

#[test]
fn json_roundtrip_for_input() {
    let (dataset, _, _, _) = make_dataset();
    let json = serde_json::to_string(&dataset).unwrap();
    let _: vision_calibration_core::RigDataset<NoMeta> = serde_json::from_str(&json).unwrap();
}
