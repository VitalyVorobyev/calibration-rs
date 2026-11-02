//! Step runner for joint rig hand-eye laserline calibration.

use crate::Error;
use crate::common::ExportKind;
use crate::rig_handeye::{
    RigHandeyeExport, RigHandeyeProblem, run_calibration as run_rig_handeye_calibration,
};
use crate::rig_laserline_device::{
    RigLaserlineDeviceInput, RigLaserlineDeviceProblem,
    run_calibration as run_rig_laserline_calibration,
};
use crate::session::CalibrationSession;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{CameraModelSpec, CameraParams};
use vision_calibration_optim::{
    BackendSolveOptions, HandeyeMode, LaserPlane, RigHandeyeLaserlineParams,
    RigHandeyeLaserlineSolveOptions, optimize_rig_handeye_laserline,
};

use super::problem::{
    RigHandeyeLaserlineConfig, RigHandeyeLaserlineInput, RigHandeyeLaserlineProblem, joint_fix,
    joint_fix_extrinsics,
};

/// Manual seeds for the warm start of the joint solve.
///
/// The joint solve starts from a rig hand-eye calibration (cameras,
/// `cam_se3_rig`, the hand-eye transform and the target reference pose) and
/// one laser plane per camera. Each field replaces one warm-start stage:
///
/// - `rig_handeye` — a rig hand-eye result of the same views, for example
///   from an earlier [`crate::rig_handeye`] run on this input. The `handeye`
///   stage is skipped and its export used as is: cameras, extrinsics,
///   hand-eye transform, target reference pose, per-view `rig_se3_target`
///   and robot-pose corrections. It must have one camera per input camera,
///   each in the model `handeye.camera` / `handeye.camera_overrides`
///   configure, one `rig_se3_target` per view (and one robot-pose correction
///   per view, if it has corrections), and the configured
///   `handeye_init.handeye_mode` with its hand-eye transform and target
///   reference pose.
/// - `planes_cam` — one laser plane per camera, each in its camera's frame
///   (a design value, say). The `laserline_init` stage refines these
///   instead of fitting its starting planes linearly.
///
/// A field left `None` runs its stage as [`run_calibration`] does.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct JointManualInit {
    /// Rig hand-eye result to start from instead of running the `handeye`
    /// stage.
    pub rig_handeye: Option<RigHandeyeExport>,
    /// Per-camera laser planes (camera frame) the `laserline_init` stage
    /// starts from.
    pub planes_cam: Option<Vec<LaserPlane>>,
}

/// Run the full joint rig hand-eye laserline pipeline.
///
/// The same as [`run_calibration_with_seed`] with an empty seed: runs rig
/// hand-eye, fits the laser planes against its frozen geometry, then refines
/// everything jointly.
///
/// # Errors
///
/// Returns [`Error`] if any warm-start or joint optimization stage fails.
pub fn run_calibration(
    session: &mut CalibrationSession<RigHandeyeLaserlineProblem>,
) -> Result<(), Error> {
    run_calibration_with_seed(session, JointManualInit::default())
}

/// Run the joint rig hand-eye laserline pipeline from manual seeds.
///
/// A seeded field replaces its warm-start stage (see [`JointManualInit`]);
/// the others run as in [`run_calibration`]. The joint solve is the same.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when the seed does not fit the input or the
///   config (camera count or models, view count, hand-eye mode, plane
///   count), raised before any solve.
/// - Any error of a warm-start stage or of the joint optimization.
pub fn run_calibration_with_seed(
    session: &mut CalibrationSession<RigHandeyeLaserlineProblem>,
    manual: JointManualInit,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?.clone();
    let config = session.config.clone();
    let mode = config.handeye.handeye_init.handeye_mode;

    if let Err(err) = check_manual_init(&input, &config, &manual) {
        session.log_failure("joint_optimize", err.to_string());
        return Err(err);
    }
    let mut seeded = Vec::new();
    if manual.rig_handeye.is_some() {
        seeded.push("rig_handeye");
    }
    if manual.planes_cam.is_some() {
        seeded.push("planes_cam");
    }

    let rig_export = match manual.rig_handeye {
        Some(export) => export,
        None => {
            let mut handeye_session = CalibrationSession::<RigHandeyeProblem>::with_description(
                "joint_warmstart_handeye",
            );
            handeye_session.set_input(input.handeye_dataset())?;
            handeye_session.set_config(config.handeye.clone())?;
            run_rig_handeye_calibration(&mut handeye_session)?;
            handeye_session.export()?
        }
    };

    let laserline_dataset = input.laserline_dataset()?;
    let upstream = rig_export.to_upstream_calibration(rig_export.rig_se3_target.clone())?;
    let mut laser_session =
        CalibrationSession::<RigLaserlineDeviceProblem>::with_description("joint_warmstart_laser");
    laser_session.set_input(RigLaserlineDeviceInput {
        dataset: laserline_dataset.clone(),
        upstream,
        initial_planes_cam: manual.planes_cam,
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
        laser_residual_type: config.joint_optimize.laser_residual_type,
        calib_loss: config.joint_optimize.calib_loss,
        laser_loss: config.joint_optimize.laser_loss,
        calib_weight: config.joint_optimize.calib_weight,
        laser_weight: config.joint_optimize.laser_weight,
        fix: joint_fix(
            &config.joint_optimize,
            &joint_models(&joint_initial.cameras)?,
        )?,
        fix_extrinsics: joint_fix_extrinsics(config.handeye.rig.reference_camera_idx, n),
        fix_handeye: config.joint_optimize.fix_handeye,
        fix_target_ref: config.joint_optimize.fix_target_ref,
        fix_planes: vec![false; n],
        refine_robot_poses: config.joint_optimize.robot_poses.refine,
        robot_rot_sigma: config.joint_optimize.robot_poses.rot_sigma,
        robot_trans_sigma: config.joint_optimize.robot_poses.trans_sigma,
        estimate_robot_sigma: config.joint_optimize.robot_poses.estimate_sigma,
        initial_robot_deltas,
    };
    let backend_opts = BackendSolveOptions {
        max_iters: config.joint_optimize.solver.max_iters,
        verbosity: config.joint_optimize.solver.verbosity,
        ..config.joint_optimize.solver.backend_options()
    };

    let estimate =
        optimize_rig_handeye_laserline(joint_dataset, joint_initial, joint_opts, backend_opts)?;
    session.set_output(estimate.clone());
    let mut notes = format!(
        "mean_reproj={:.3}px, final_cost={:.3e}",
        estimate.mean_reproj_error_px, estimate.report.final_cost
    );
    if !seeded.is_empty() {
        notes.push_str(&format!(" (manual: {})", seeded.join(", ")));
    }
    session.log_success_with_notes("joint_optimize", notes);
    Ok(())
}

/// Check that a seed fits the input and the config.
///
/// # Errors
///
/// [`Error::InvalidInput`] naming the first mismatch.
fn check_manual_init(
    input: &RigHandeyeLaserlineInput,
    config: &RigHandeyeLaserlineConfig,
    manual: &JointManualInit,
) -> Result<(), Error> {
    let n_cams = input.num_cameras;
    let n_views = input.num_views();
    if let Some(export) = &manual.rig_handeye {
        let fail = |what: String| Err(Error::invalid_input(format!("seed rig_handeye: {what}")));
        if export.kind != ExportKind::RigHandeye {
            return fail(format!(
                "export kind {:?}, expected RigHandeye",
                export.kind
            ));
        }
        if export.cameras.len() != n_cams || export.cam_se3_rig.len() != n_cams {
            return fail(format!(
                "{} cameras / {} cam_se3_rig, expected {n_cams} (the input's cameras)",
                export.cameras.len(),
                export.cam_se3_rig.len()
            ));
        }
        for (i, camera) in export.cameras.iter().enumerate() {
            let configured = config.handeye.camera_for(i).model;
            if camera.spec() != Some(configured) {
                return fail(format!(
                    "camera {i} is a {:?}, the config's handeye.camera is {configured:?}",
                    camera.spec()
                ));
            }
        }
        if export.rig_se3_target.len() != n_views {
            return fail(format!(
                "{} rig_se3_target poses, expected {n_views} (the input's views)",
                export.rig_se3_target.len()
            ));
        }
        if let Some(deltas) = &export.robot_deltas
            && deltas.len() != n_views
        {
            return fail(format!(
                "{} robot_deltas, expected {n_views} (the input's views)",
                deltas.len()
            ));
        }
        let mode = config.handeye.handeye_init.handeye_mode;
        if export.handeye_mode != mode {
            return fail(format!(
                "handeye_mode {:?}, the config's handeye.handeye_init.handeye_mode is {mode:?}",
                export.handeye_mode
            ));
        }
        let (handeye, target_ref) = match mode {
            HandeyeMode::EyeInHand => (export.gripper_se3_rig, export.base_se3_target),
            HandeyeMode::EyeToHand => (export.rig_se3_base, export.gripper_se3_target),
        };
        if handeye.is_none() || target_ref.is_none() {
            return fail(format!(
                "{mode:?} export lacks its hand-eye transform or target reference pose"
            ));
        }
    }
    if let Some(planes) = &manual.planes_cam {
        if planes.len() != n_cams {
            return Err(Error::invalid_input(format!(
                "seed planes_cam: {} planes, expected {n_cams} (one per camera)",
                planes.len()
            )));
        }
        if let Some(i) = planes
            .iter()
            .position(|p| !(p.distance.is_finite() && p.normal.iter().all(|c| c.is_finite())))
        {
            return Err(Error::invalid_input(format!(
                "seed planes_cam: plane {i} is not finite"
            )));
        }
    }
    Ok(())
}

/// The models of the warm-started cameras.
fn joint_models(cameras: &[CameraParams]) -> Result<Vec<CameraModelSpec>, Error> {
    cameras
        .iter()
        .enumerate()
        .map(|(i, camera)| {
            camera.spec().ok_or_else(|| {
                Error::invalid_input(format!("camera {i} has a general homography sensor"))
            })
        })
        .collect()
}

fn joint_initial_from_exports(
    rig_export: &RigHandeyeExport,
    laser_export: &crate::rig_laserline_device::RigLaserlineDeviceExport,
    mode: HandeyeMode,
) -> Result<RigHandeyeLaserlineParams, Error> {
    let handeye = match mode {
        HandeyeMode::EyeInHand => rig_export
            .gripper_se3_rig
            .ok_or_else(|| Error::invalid_input("EyeInHand export missing gripper_se3_rig"))?,
        HandeyeMode::EyeToHand => rig_export
            .rig_se3_base
            .ok_or_else(|| Error::invalid_input("EyeToHand export missing rig_se3_base"))?,
    };
    let target_ref = match mode {
        HandeyeMode::EyeInHand => rig_export
            .base_se3_target
            .ok_or_else(|| Error::invalid_input("EyeInHand export missing base_se3_target"))?,
        HandeyeMode::EyeToHand => rig_export
            .gripper_se3_target
            .ok_or_else(|| Error::invalid_input("EyeToHand export missing gripper_se3_target"))?,
    };
    Ok(RigHandeyeLaserlineParams {
        cameras: rig_export.cameras.clone(),
        cam_to_rig: rig_export.cam_se3_rig.iter().map(|t| t.inverse()).collect(),
        handeye,
        target_ref,
        planes_cam: laser_export.laser_planes_cam.clone(),
    })
}
