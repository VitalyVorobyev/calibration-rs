//! Synthetic multi-camera rig intrinsics + extrinsics calibration.
//!
//! Demonstrates the session API and the pipeline convenience wrapper.

use anyhow::{ensure, Result};
use calib::core::{
    synthetic::planar, BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole,
};
use calib::pipeline::{run_rig_extrinsics, RigExtrinsicsConfig, RigExtrinsicsInput, RigViewData};
use calib::session::{CalibrationSession, RigExtrinsicsInitOptions, RigExtrinsicsProblem};
use nalgebra::{UnitQuaternion, Vector3};

fn main() -> Result<()> {
    let intrinsics_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let distortion_gt = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };

    let camera_gt = Camera::new(Pinhole, distortion_gt, IdentitySensor, intrinsics_gt);

    // Ground truth rig extrinsics: camera->rig (T_R_C)
    let cam0_to_rig = Iso3::identity();
    let cam1_to_rig = Iso3::from_parts(
        Vector3::new(0.10, 0.0, 0.0).into(),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.05, 0.0)),
    );

    // Planar target points (Z=0)
    let board_points = planar::grid_points(6, 5, 0.05);

    // Views: target->rig (T_R_T), with a missing observation for camera 1 in one view
    let mut views = Vec::new();
    let rig_from_targets = planar::poses_yaw_y_z(4, 0.0, 0.08, 0.7, 0.05);
    for (view_idx, rig_from_target) in rig_from_targets.iter().enumerate() {
        let mut cameras = Vec::with_capacity(2);
        for cam_idx in 0..2 {
            let cam_to_rig = if cam_idx == 0 {
                cam0_to_rig
            } else {
                cam1_to_rig
            };

            // Drop cam1 in view 0 to demonstrate missing observations
            if cam_idx == 1 && view_idx == 0 {
                cameras.push(None);
                continue;
            }

            let cam_from_target = cam_to_rig.inverse() * *rig_from_target;
            let view = planar::project_view_all(&camera_gt, &cam_from_target, &board_points)?;
            cameras.push(Some(view));
        }

        views.push(RigViewData { cameras });
    }

    let input = RigExtrinsicsInput {
        views,
        num_cameras: 2,
    };

    // 1) Session API
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_observations(input.clone());
    session.initialize(RigExtrinsicsInitOptions::default())?;
    session.optimize(Default::default())?;
    let report_session = session.export()?;
    ensure!(
        report_session.final_cost.is_finite(),
        "final cost is not finite"
    );

    // 2) Pipeline convenience wrapper
    let report_pipeline = run_rig_extrinsics(&input, &RigExtrinsicsConfig::default())?;
    ensure!(
        report_pipeline.final_cost.is_finite(),
        "final cost is not finite"
    );

    println!(
        "rig extrinsics final cost (session):  {}",
        report_session.final_cost
    );
    println!(
        "rig extrinsics final cost (pipeline): {}",
        report_pipeline.final_cost
    );
    println!(
        "cam1 baseline (rig frame): {:?}",
        report_pipeline.cam_to_rig[1].translation.vector
    );

    Ok(())
}
