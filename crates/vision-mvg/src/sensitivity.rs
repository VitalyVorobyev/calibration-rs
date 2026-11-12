//! What the closed-form measurement covariances share: where a rig camera's
//! calibration coordinates sit in the calibration's covariance.
//!
//! A measurement whose point is an implicit function of its pixels and the
//! calibration (a laser point, a dense stereo point) is propagated in closed
//! form, `∂x/∂c = −A⁻¹ ∂F/∂c`, over the coordinates `c` the calibration's
//! covariance lists. This module finds those coordinates for one camera: its
//! intrinsic, distortion and sensor parameters in the order of
//! [`CameraJacobian::labels`], and its `cam_se3_rig` (the right perturbation
//! `cam_se3_rig · Exp(ω, v)`), with the labels the exports use.

use crate::{Error, Result};
use nalgebra::Matrix3;
use vision_calibration_core::{CalibratedRig, ParameterCovariance, Real, Vec3};
use vision_calibration_optim::CameraJacobian;

/// The tangent coordinates of a pose, in the exports' order.
pub(crate) const POSE_AXES: [&str; 6] = ["rx", "ry", "rz", "tx", "ty", "tz"];

/// Where one camera's estimated parameters are in the calibration's
/// covariance.
pub(crate) struct CameraCoordinates {
    /// `(column of ∂π/∂θ, label, position in the covariance)` of each camera
    /// parameter the covariance lists, in the order of
    /// [`CameraJacobian::labels`].
    pub theta: Vec<(usize, String, usize)>,
    /// The labels and positions of the six coordinates of the camera's
    /// `cam_se3_rig` (`rx ry rz tx ty tz`), when the calibration estimated
    /// the pose; `None` when it held it (the reference camera, a single
    /// camera).
    pub extrinsic: Option<Vec<(String, usize)>>,
}

/// The label prefix of camera `k`'s parameters: `cam<k>.` in a rig of two
/// or more cameras, none for a single camera.
pub(crate) fn camera_prefix(rig: &CalibratedRig, k: usize) -> String {
    if rig.num_cameras() > 1 {
        format!("cam{k}.")
    } else {
        String::new()
    }
}

/// Where camera `k`'s parameters (those of `jacobian`) and its extrinsic are
/// in `covariance`. Parameters it does not list were held fixed by the
/// calibration and count as exact.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a covariance that lists part of the
/// camera's `cam_se3_rig`: a pose is estimated whole or held.
pub(crate) fn camera_coordinates(
    rig: &CalibratedRig,
    covariance: &ParameterCovariance,
    k: usize,
    jacobian: &CameraJacobian,
) -> Result<CameraCoordinates> {
    let prefix = camera_prefix(rig, k);
    let theta = jacobian
        .labels()
        .iter()
        .enumerate()
        .filter_map(|(j, label)| {
            let label = format!("{prefix}{label}");
            covariance.index(&label).map(|at| (j, label, at))
        })
        .collect();
    let pose: Vec<(String, Option<usize>)> = POSE_AXES
        .iter()
        .map(|c| {
            let label = format!("{prefix}cam_se3_rig.{c}");
            let at = covariance.index(&label);
            (label, at)
        })
        .collect();
    let extrinsic = if rig.num_cameras() > 1 && pose.iter().all(|(_, at)| at.is_some()) {
        Some(
            pose.into_iter()
                .map(|(label, at)| (label, at.expect("checked")))
                .collect(),
        )
    } else if pose.iter().any(|(_, at)| at.is_some()) {
        return Err(Error::invalid_input(format!(
            "camera {k}: the covariance lists part of its cam_se3_rig; a pose is \
             estimated whole or held"
        )));
    } else {
        None
    };
    Ok(CameraCoordinates { theta, extrinsic })
}

/// `[v]ₓ`, the cross-product matrix: `[v]ₓ w = v × w`.
pub(crate) fn skew(v: &Vec3) -> Matrix3<Real> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}
