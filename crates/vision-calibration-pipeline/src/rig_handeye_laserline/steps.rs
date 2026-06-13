//! Step runner for joint rig hand-eye laserline calibration.

use crate::Error;
use crate::rig_handeye::{
    RigHandeyeExport, RigHandeyeProblem, run_calibration as run_rig_handeye_calibration,
};
use crate::rig_laserline_device::{
    RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
    run_calibration as run_rig_laserline_calibration,
};
use crate::session::CalibrationSession;
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeMode, RigHandeyeLaserlineParams, RigHandeyeLaserlineSolveOptions,
    optimize_rig_handeye_laserline,
};

use super::problem::{
    RigHandeyeLaserlineOutput, RigHandeyeLaserlineProblem, joint_fix_extrinsics,
    joint_fix_intrinsics, joint_fix_scheimpflug,
};

/// Run the full joint rig hand-eye laserline pipeline.
///
/// # Errors
///
/// Returns [`Error`] if any warm-start or joint optimization stage fails.
pub fn run_calibration(
    session: &mut CalibrationSession<RigHandeyeLaserlineProblem>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?.clone();
    let config = session.config.clone();
    let mode = config.handeye.handeye_init.handeye_mode;

    let mut handeye_session =
        CalibrationSession::<RigHandeyeProblem>::with_description("joint_warmstart_handeye");
    handeye_session.set_input(input.handeye_dataset())?;
    handeye_session.set_config(config.handeye.clone())?;
    run_rig_handeye_calibration(&mut handeye_session)?;
    let rig_export = handeye_session.export()?;

    let laserline_dataset = input.laserline_dataset()?;
    let upstream = rig_export.to_upstream_calibration(rig_export.rig_se3_target.clone())?;
    let mut laser_session =
        CalibrationSession::<RigLaserlineDeviceProblem>::with_description("joint_warmstart_laser");
    laser_session.set_input(RigLaserlineDeviceInput {
        dataset: laserline_dataset.clone(),
        upstream,
        initial_planes_cam: None,
    })?;
    laser_session.set_config(config.laserline_init.clone())?;
    run_rig_laserline_calibration(&mut laser_session)?;
    let laser_export = laser_session.export()?;

    let joint_dataset = input.joint_dataset(mode)?;
    let joint_initial = joint_initial_from_exports(&rig_export, &laser_export, mode)?;
    let initial_robot_deltas = rig_export
        .robot_deltas
        .clone()
        .filter(|deltas| deltas.len() == input.num_views());
    let n = input.num_cameras;
    let joint_opts = RigHandeyeLaserlineSolveOptions {
        laser_residual_type: config.joint_ba.laser_residual_type,
        calib_loss: config.joint_ba.calib_loss,
        laser_loss: config.joint_ba.laser_loss,
        calib_weight: config.joint_ba.calib_weight,
        laser_weight: config.joint_ba.laser_weight,
        fix_intrinsics: joint_fix_intrinsics(&config.joint_ba, n),
        fix_scheimpflug: joint_fix_scheimpflug(&config.joint_ba, n),
        fix_extrinsics: joint_fix_extrinsics(&config.joint_ba, n),
        fix_handeye: config.joint_ba.fix_handeye,
        fix_target_ref: config.joint_ba.fix_target_ref,
        fix_planes: vec![false; n],
        refine_robot_poses: config.joint_ba.refine_robot_poses,
        robot_rot_sigma: config.joint_ba.robot_rot_sigma,
        robot_trans_sigma: config.joint_ba.robot_trans_sigma,
        initial_robot_deltas,
    };
    let backend_opts = BackendSolveOptions {
        max_iters: config.joint_ba.max_iters,
        verbosity: config.joint_ba.verbosity,
        ..Default::default()
    };

    let estimate =
        optimize_rig_handeye_laserline(joint_dataset, joint_initial, joint_opts, backend_opts)?;
    session.set_output(RigHandeyeLaserlineOutput {
        estimate: estimate.clone(),
        handeye_mode: mode,
    });
    session.log_success_with_notes(
        "joint_optimize",
        format!(
            "mean_reproj={:.3}px, final_cost={:.3e}",
            estimate.mean_reproj_error_px, estimate.report.final_cost
        ),
    );
    Ok(())
}

fn joint_initial_from_exports(
    rig_export: &RigHandeyeExport,
    laser_export: &crate::rig_laserline_device::RigLaserlineDeviceExport,
    mode: HandEyeMode,
) -> Result<RigHandeyeLaserlineParams, Error> {
    let handeye = match mode {
        HandEyeMode::EyeInHand => rig_export
            .gripper_se3_rig
            .ok_or_else(|| Error::invalid_input("EyeInHand export missing gripper_se3_rig"))?,
        HandEyeMode::EyeToHand => rig_export
            .rig_se3_base
            .ok_or_else(|| Error::invalid_input("EyeToHand export missing rig_se3_base"))?,
    };
    let target_ref = match mode {
        HandEyeMode::EyeInHand => rig_export
            .base_se3_target
            .ok_or_else(|| Error::invalid_input("EyeInHand export missing base_se3_target"))?,
        HandEyeMode::EyeToHand => rig_export
            .gripper_se3_target
            .ok_or_else(|| Error::invalid_input("EyeToHand export missing gripper_se3_target"))?,
    };
    let sensors = rig_export.sensors.clone().unwrap_or_else(|| {
        vec![vision_calibration_core::ScheimpflugParams::default(); rig_export.cameras.len()]
    });
    Ok(RigHandeyeLaserlineParams {
        cameras: rig_export.cameras.clone(),
        sensors,
        cam_to_rig: rig_export.cam_se3_rig.iter().map(|t| t.inverse()).collect(),
        handeye,
        target_ref,
        planes_cam: laser_export.laser_planes_cam.clone(),
    })
}
