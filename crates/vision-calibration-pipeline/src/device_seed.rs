//! Device-spec → manual-init seed derivation (ADR 0023).
//!
//! Maps a [`DeviceSpec`] — datasheet values plus the nominal mechanical
//! layout — onto the ADR 0011 manual-init structs consumed by
//! `step_init_with_seed` and friends. All unit conversion between the
//! datasheet-natural spec (`mm`, `µm`, degrees) and the internal
//! representation (pixels, radians, millimetres) happens here and nowhere
//! else:
//!
//! - `fx = fy = focal_mm · 1000 / pixel_pitch_um`, principal point
//!   defaulting to the resolution center, `skew = 0`;
//! - mount tilt degrees → `ScheimpflugParams` radians;
//! - `rig_se3_cam` mount poses → inverted into the `cam_se3_rig` (`T_C_R`)
//!   the rig problems expect.
//!
//! The derivations are per-camera-id, never positional: the caller states
//! the camera order it needs and mismatches surface as typed errors.

use vision_calibration_core::{FxFyCxCySkew, Iso3, Real, ScheimpflugParams};
use vision_calibration_optim::HandEyeMode;

use crate::rig_handeye::{RigHandeyeHandeyeManualInit, RigHandeyeIntrinsicsManualInit};
use crate::scheimpflug_intrinsics::ScheimpflugManualInit;

// The schema types every consumer of this module needs, re-exported so the
// facade can expose the full device-spec surface from one place.
pub use vision_calibration_dataset::{
    CameraDeviceSpec, CameraMountSpec, DEVICE_SPEC_FILENAME, DEVICE_SPEC_VERSION, DeviceSpec,
    DeviceSpecError, HandeyeMountSpec, LaserPlaneSpec, NominalPoseSpec, RigLayoutSpec,
    ScheimpflugMountSpec,
};

/// Failures when deriving seeds from a [`DeviceSpec`].
#[derive(Debug, thiserror::Error)]
pub enum DeviceSeedError {
    /// The spec has no camera with the requested id.
    #[error("device spec has no camera with id `{id}`")]
    UnknownCameraId {
        /// The unmatched id.
        id: String,
    },

    /// The spec has no rig mechanical layout.
    #[error("device spec has no rig layout")]
    MissingRigLayout,

    /// The rig layout has no hand-eye mount.
    #[error("device spec rig layout has no hand-eye mount")]
    MissingHandeye,

    /// The rig layout has no mount for the requested camera.
    #[error("device spec rig layout has no mount for camera `{id}`")]
    MissingCameraMount {
        /// The unmatched id.
        id: String,
    },

    /// The spec's hand-eye mount mode contradicts the session's configured
    /// mode — seeding across modes would silently seed the wrong transform.
    #[error("device spec hand-eye mount is {spec:?} but the session expects {expected:?}")]
    HandeyeModeMismatch {
        /// Mode stated by the spec.
        spec: HandEyeMode,
        /// Mode the caller's config expects.
        expected: HandEyeMode,
    },
}

/// Seed for a single-camera Scheimpflug intrinsics session.
///
/// Intrinsics and sensor tilt are always seeded (ADR 0022: both are
/// load-bearing). A camera without a `scheimpflug` entry is frontal *by
/// design*, so it seeds an identity tilt — the spec asserts knowledge of
/// the mount, absence is not ignorance. Distortion and poses stay `None`
/// (auto).
pub fn scheimpflug_seed(
    spec: &DeviceSpec,
    camera_id: &str,
) -> Result<ScheimpflugManualInit, DeviceSeedError> {
    let cam = lookup(spec, camera_id)?;
    Ok(ScheimpflugManualInit {
        intrinsics: Some(intrinsics_from_camera(cam)),
        sensor: Some(sensor_from_camera(cam)),
        ..Default::default()
    })
}

/// Per-camera intrinsics + sensor seeds for a rig hand-eye session, in the
/// caller's `camera_ids` order.
///
/// `per_cam_sensors` is always populated (identity tilt for frontal
/// cameras); it is ignored by the pipeline under `SensorMode::Pinhole`.
/// `per_cam_distortion` stays `None` (auto).
pub fn rig_intrinsics_seed(
    spec: &DeviceSpec,
    camera_ids: &[&str],
) -> Result<RigHandeyeIntrinsicsManualInit, DeviceSeedError> {
    let cams = camera_ids
        .iter()
        .map(|id| lookup(spec, id))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RigHandeyeIntrinsicsManualInit {
        per_cam_intrinsics: Some(cams.iter().map(|c| intrinsics_from_camera(c)).collect()),
        per_cam_sensors: Some(cams.iter().map(|c| sensor_from_camera(c)).collect()),
        ..Default::default()
    })
}

/// Nominal `cam_se3_rig` (`T_C_R`) poses from the mechanical layout, in the
/// caller's `camera_ids` order.
///
/// This is deliberately *not* a `RigHandeyeRigManualInit`: ADR 0011 couples
/// `cam_se3_rig` with the data-dependent per-view `rig_se3_target`
/// (both-or-neither), so combining the nominals with per-view estimates is
/// the caller's job (Track S3).
pub fn nominal_cam_se3_rig(
    spec: &DeviceSpec,
    camera_ids: &[&str],
) -> Result<Vec<Iso3>, DeviceSeedError> {
    let rig = spec.rig.as_ref().ok_or(DeviceSeedError::MissingRigLayout)?;
    camera_ids
        .iter()
        .map(|id| {
            // Spec cameras and mounts are validated id-consistent at load;
            // distinguish "unknown camera" from "camera without a mount".
            lookup(spec, id)?;
            let mount = rig.cameras.iter().find(|m| m.id == *id).ok_or_else(|| {
                DeviceSeedError::MissingCameraMount {
                    id: (*id).to_string(),
                }
            })?;
            Ok(iso3_from_nominal(&mount.rig_se3_cam).inverse())
        })
        .collect()
}

/// Hand-eye stage seed from the mechanical mount.
///
/// The spec's mount mode must match the session's configured
/// [`HandEyeMode`]; a mismatch is a typed error rather than a silently
/// wrong transform. `mode_target_pose` stays `None` (data-dependent).
pub fn handeye_seed(
    spec: &DeviceSpec,
    expected_mode: HandEyeMode,
) -> Result<RigHandeyeHandeyeManualInit, DeviceSeedError> {
    let rig = spec.rig.as_ref().ok_or(DeviceSeedError::MissingRigLayout)?;
    let mount = rig
        .handeye
        .as_ref()
        .ok_or(DeviceSeedError::MissingHandeye)?;
    let (spec_mode, pose) = match mount {
        HandeyeMountSpec::EyeInHand { gripper_se3_rig } => {
            (HandEyeMode::EyeInHand, gripper_se3_rig)
        }
        HandeyeMountSpec::EyeToHand { rig_se3_base } => (HandEyeMode::EyeToHand, rig_se3_base),
    };
    if spec_mode != expected_mode {
        return Err(DeviceSeedError::HandeyeModeMismatch {
            spec: spec_mode,
            expected: expected_mode,
        });
    }
    Ok(RigHandeyeHandeyeManualInit {
        handeye: Some(iso3_from_nominal(pose)),
        ..Default::default()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

fn lookup<'a>(spec: &'a DeviceSpec, id: &str) -> Result<&'a CameraDeviceSpec, DeviceSeedError> {
    spec.camera(id)
        .ok_or_else(|| DeviceSeedError::UnknownCameraId { id: id.to_string() })
}

fn intrinsics_from_camera(cam: &CameraDeviceSpec) -> FxFyCxCySkew<Real> {
    let f_px = cam.focal_mm * 1000.0 / cam.pixel_pitch_um;
    let [cx, cy] = cam.principal_point_px.unwrap_or([
        f64::from(cam.resolution_px[0]) * 0.5,
        f64::from(cam.resolution_px[1]) * 0.5,
    ]);
    FxFyCxCySkew {
        fx: f_px,
        fy: f_px,
        cx,
        cy,
        skew: 0.0,
    }
}

fn sensor_from_camera(cam: &CameraDeviceSpec) -> ScheimpflugParams {
    match &cam.scheimpflug {
        Some(mount) => ScheimpflugParams {
            tilt_x: mount.tilt_x_deg.to_radians(),
            tilt_y: mount.tilt_y_deg.to_radians(),
        },
        None => ScheimpflugParams::default(),
    }
}

fn iso3_from_nominal(pose: &NominalPoseSpec) -> Iso3 {
    let [roll, pitch, yaw] = pose.rpy_deg.map(f64::to_radians);
    let rotation = nalgebra::UnitQuaternion::from_euler_angles(roll, pitch, yaw);
    let translation = nalgebra::Translation3::new(
        pose.translation_mm[0],
        pose.translation_mm[1],
        pose.translation_mm[2],
    );
    Iso3::from_parts(translation, rotation)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{CameraMountSpec, RigLayoutSpec, ScheimpflugMountSpec};

    fn camera(id: &str) -> CameraDeviceSpec {
        CameraDeviceSpec {
            id: id.to_string(),
            focal_mm: 8.0,
            pixel_pitch_um: 7.0,
            resolution_px: [720, 540],
            principal_point_px: None,
            scheimpflug: Some(ScheimpflugMountSpec {
                tilt_x_deg: -5.0,
                tilt_y_deg: 0.0,
            }),
        }
    }

    fn rig_spec() -> DeviceSpec {
        DeviceSpec {
            version: 1,
            cameras: vec![camera("cam0"), camera("cam1")],
            rig: Some(RigLayoutSpec {
                cameras: vec![
                    CameraMountSpec {
                        id: "cam0".to_string(),
                        rig_se3_cam: NominalPoseSpec {
                            rpy_deg: [0.0, 0.0, 90.0],
                            translation_mm: [100.0, 0.0, 0.0],
                        },
                    },
                    CameraMountSpec {
                        id: "cam1".to_string(),
                        rig_se3_cam: NominalPoseSpec::default(),
                    },
                ],
                handeye: Some(HandeyeMountSpec::EyeToHand {
                    rig_se3_base: NominalPoseSpec {
                        rpy_deg: [0.0, 0.0, 0.0],
                        translation_mm: [0.0, 250.0, 0.0],
                    },
                }),
                laser_planes: None,
            }),
            description: None,
        }
    }

    #[test]
    fn focal_px_from_datasheet() {
        // The S1 acceptance formula: f_px = f_mm / pixel_pitch.
        let mut cam = camera("cam0");
        cam.focal_mm = 16.0;
        cam.pixel_pitch_um = 4.8;
        let k = intrinsics_from_camera(&cam);
        assert!((k.fx - 16.0e-3 / 4.8e-6).abs() < 1e-9);
        assert!((k.fx - 3_333.333_333_333).abs() < 1e-6);
        assert_eq!(k.fx, k.fy);
        assert_eq!(k.skew, 0.0);

        let k = intrinsics_from_camera(&camera("cam0"));
        assert!((k.fx - 8.0e-3 / 7.0e-6).abs() < 1e-9); // ≈ 1142.857
    }

    #[test]
    fn principal_point_defaults_to_center() {
        let k = intrinsics_from_camera(&camera("cam0"));
        assert_eq!((k.cx, k.cy), (360.0, 270.0));

        let mut cam = camera("cam0");
        cam.principal_point_px = Some([355.5, 275.25]);
        let k = intrinsics_from_camera(&cam);
        assert_eq!((k.cx, k.cy), (355.5, 275.25));
    }

    #[test]
    fn tilt_degrees_to_radians_and_frontal_identity() {
        let seed = scheimpflug_seed(&rig_spec(), "cam0").unwrap();
        let sensor = seed.sensor.unwrap();
        assert!((sensor.tilt_x - (-5.0f64).to_radians()).abs() < 1e-12);
        assert!((sensor.tilt_x - (-0.087_266_46)).abs() < 1e-7);
        assert_eq!(sensor.tilt_y, 0.0);
        assert!(seed.intrinsics.is_some());
        assert!(seed.distortion.is_none());
        assert!(seed.poses.is_none());

        let mut spec = rig_spec();
        spec.cameras[0].scheimpflug = None;
        let seed = scheimpflug_seed(&spec, "cam0").unwrap();
        let sensor = seed.sensor.unwrap();
        assert_eq!((sensor.tilt_x, sensor.tilt_y), (0.0, 0.0));
    }

    #[test]
    fn rig_intrinsics_follow_caller_order() {
        let spec = rig_spec();
        let seed = rig_intrinsics_seed(&spec, &["cam1", "cam0"]).unwrap();
        let ks = seed.per_cam_intrinsics.unwrap();
        assert_eq!(ks.len(), 2);
        let sensors = seed.per_cam_sensors.unwrap();
        assert_eq!(sensors.len(), 2);
        assert!(seed.per_cam_distortion.is_none());

        assert!(matches!(
            rig_intrinsics_seed(&spec, &["cam0", "cam9"]),
            Err(DeviceSeedError::UnknownCameraId { .. })
        ));
    }

    #[test]
    fn rpy_convention_fixed_axes_xyz() {
        // roll = 90° about x maps ŷ → ẑ.
        let pose = NominalPoseSpec {
            rpy_deg: [90.0, 0.0, 0.0],
            translation_mm: [0.0; 3],
        };
        let iso = iso3_from_nominal(&pose);
        let mapped = iso * nalgebra::Vector3::y();
        assert!((mapped - nalgebra::Vector3::z()).norm() < 1e-12);
    }

    #[test]
    fn cam_se3_rig_is_inverted_mount() {
        // Mount: Rz(90°), t = [100, 0, 0] (camera pose in rig frame).
        // Inverse: R = Rz(−90°), t = −Rz(−90°)·[100,0,0] = [0, 100, 0].
        let spec = rig_spec();
        let poses = nominal_cam_se3_rig(&spec, &["cam0", "cam1"]).unwrap();
        let t = poses[0].translation.vector;
        assert!((t - nalgebra::Vector3::new(0.0, 100.0, 0.0)).norm() < 1e-9);
        // Round-trip: inverse of the inverse is the mount pose.
        let back = poses[0].inverse();
        assert!((back.translation.vector.x - 100.0).abs() < 1e-9);
        assert!(poses[1].translation.vector.norm() < 1e-12); // identity mount

        assert!(matches!(
            nominal_cam_se3_rig(&spec, &["cam9"]),
            Err(DeviceSeedError::UnknownCameraId { .. })
        ));

        let mut no_mount = rig_spec();
        no_mount.rig.as_mut().unwrap().cameras.pop();
        assert!(matches!(
            nominal_cam_se3_rig(&no_mount, &["cam1"]),
            Err(DeviceSeedError::MissingCameraMount { .. })
        ));

        let mut no_rig = rig_spec();
        no_rig.rig = None;
        assert!(matches!(
            nominal_cam_se3_rig(&no_rig, &["cam0"]),
            Err(DeviceSeedError::MissingRigLayout)
        ));
    }

    #[test]
    fn handeye_seed_mode_checked() {
        let spec = rig_spec();
        let seed = handeye_seed(&spec, HandEyeMode::EyeToHand).unwrap();
        let handeye = seed.handeye.unwrap();
        assert!((handeye.translation.vector.y - 250.0).abs() < 1e-12);
        assert!(seed.mode_target_pose.is_none());

        assert!(matches!(
            handeye_seed(&spec, HandEyeMode::EyeInHand),
            Err(DeviceSeedError::HandeyeModeMismatch { .. })
        ));

        let mut no_he = rig_spec();
        no_he.rig.as_mut().unwrap().handeye = None;
        assert!(matches!(
            handeye_seed(&no_he, HandEyeMode::EyeToHand),
            Err(DeviceSeedError::MissingHandeye)
        ));
    }
}
