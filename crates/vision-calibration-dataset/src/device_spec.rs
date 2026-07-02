//! Device-specification sidecar (`spec.json`) types.
//!
//! [`DeviceSpec`] describes the *hardware* a dataset was captured with —
//! lens focal lengths, sensor pixel pitch, Scheimpflug mount tilt, and the
//! rig's nominal mechanical layout — as transcribed from datasheets and
//! mechanical drawings. It is the structured source for the seeded
//! initialization route (ADR 0022): derivation functions in the pipeline
//! crate turn a `DeviceSpec` into the ADR 0011 manual-init seeds.
//!
//! Units are datasheet-natural and encoded in field names (`_mm`, `_um`,
//! `_px`, `_deg`); unit conversion happens exactly once, in the derivation
//! layer. Poses follow the ADR 0009 `frame_se3_frame` naming. See ADR 0023
//! for the design rationale.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

/// Conventional sidecar filename, next to the dataset manifest.
pub const DEVICE_SPEC_FILENAME: &str = "spec.json";

/// Current device-spec format version.
pub const DEVICE_SPEC_VERSION: u32 = 1;

fn default_device_spec_version() -> u32 {
    DEVICE_SPEC_VERSION
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level spec
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level device specification. Describes the capture hardware, not the
/// captured data (that is [`DatasetSpec`](crate::DatasetSpec)'s job).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DeviceSpec {
    /// Format-version sentinel. Always `1` for now; bumped on breaking
    /// schema revisions.
    #[serde(default = "default_device_spec_version")]
    pub version: u32,

    /// Per-camera hardware specifications. Non-empty; ids unique and
    /// matching the dataset manifest's camera ids (e.g. `"cam0"`).
    pub cameras: Vec<CameraDeviceSpec>,

    /// Nominal rig mechanical layout. `None` for single-camera devices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig: Option<RigLayoutSpec>,

    /// Free-form provenance note (device model, drawing revision, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Datasheet values for one camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CameraDeviceSpec {
    /// Stable camera identifier matching the dataset manifest
    /// (`CameraSource::id`).
    pub id: String,

    /// Lens focal length in millimetres (datasheet value).
    pub focal_mm: f64,

    /// Sensor pixel pitch in micrometres (datasheet value).
    pub pixel_pitch_um: f64,

    /// `[width, height]` of the images actually calibrated, in pixels.
    /// For tiled multi-camera sensors this is the tile size, not the
    /// full sensor.
    pub resolution_px: [u32; 2],

    /// Nominal principal point in pixels. `None` defaults to the
    /// resolution center at derivation time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub principal_point_px: Option<[f64; 2]>,

    /// Scheimpflug mount tilt. `None` means a frontal (untilted) sensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheimpflug: Option<ScheimpflugMountSpec>,
}

/// Mechanically-set Scheimpflug sensor tilt, in degrees.
///
/// Maps onto `ScheimpflugParams { tilt_x, tilt_y }` (radians) at
/// derivation time.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ScheimpflugMountSpec {
    /// Tilt about the sensor x axis, degrees.
    #[serde(default)]
    pub tilt_x_deg: f64,

    /// Tilt about the sensor y axis, degrees.
    #[serde(default)]
    pub tilt_y_deg: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Rig mechanical layout
// ─────────────────────────────────────────────────────────────────────────────

/// Nominal rig mechanical layout from the drawing: camera mounts, the
/// hand-eye mounting, and laser-plane nominals.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct RigLayoutSpec {
    /// Nominal camera mount poses, keyed by camera id. Every id must
    /// exist in [`DeviceSpec::cameras`]; cameras without a known mount
    /// may be omitted.
    pub cameras: Vec<CameraMountSpec>,

    /// How the rig relates to the robot. `None` when the dataset has no
    /// robot (pure intrinsics/extrinsics).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handeye: Option<HandeyeMountSpec>,

    /// Nominal laser planes, one per camera-laser pair.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laser_planes: Option<Vec<LaserPlaneSpec>>,
}

/// Nominal mount pose of one camera in the rig frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CameraMountSpec {
    /// Camera id, matching [`DeviceSpec::cameras`].
    pub id: String,

    /// Camera pose in the rig frame (`T_R_C` — maps camera-frame points
    /// into the rig frame). This is what a mechanical drawing states;
    /// derivation inverts it into the `cam_se3_rig` the manual-init
    /// surface expects.
    pub rig_se3_cam: NominalPoseSpec,
}

/// How the rig is mounted relative to the robot. The tag fixes the frame
/// meaning of the nominal pose, so the two hand-eye modes cannot be
/// confused.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum HandeyeMountSpec {
    /// Rig mounted on the gripper: the nominal is `gripper_se3_rig`
    /// (`T_G_R`).
    EyeInHand {
        /// Rig pose in the gripper frame.
        gripper_se3_rig: NominalPoseSpec,
    },
    /// Rig fixed in the workcell, target on the gripper: the nominal is
    /// `rig_se3_base` (`T_R_B`).
    EyeToHand {
        /// Robot-base pose in the rig frame.
        rig_se3_base: NominalPoseSpec,
    },
}

/// Human-friendly nominal SE(3) pose: fixed-axes XYZ roll-pitch-yaw in
/// degrees (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`, nalgebra
/// `from_euler_angles` convention) plus a translation in millimetres.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NominalPoseSpec {
    /// `[roll, pitch, yaw]` in degrees, fixed-axes XYZ.
    #[serde(default)]
    pub rpy_deg: [f64; 3],

    /// Translation in millimetres.
    #[serde(default)]
    pub translation_mm: [f64; 3],
}

/// Nominal laser plane `{p : n·p = d}` in the **rig** frame, keyed to its
/// paired camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct LaserPlaneSpec {
    /// The camera this laser is paired with, matching
    /// [`DeviceSpec::cameras`].
    pub camera_id: String,

    /// Unit plane normal in the rig frame.
    pub normal: [f64; 3],

    /// Signed plane distance `d` in millimetres.
    pub distance_mm: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors + validation
// ─────────────────────────────────────────────────────────────────────────────

/// Loading/validation failures for [`DeviceSpec`].
#[derive(Debug, thiserror::Error)]
pub enum DeviceSpecError {
    /// The sidecar file could not be read.
    #[error("failed to read device spec: {0}")]
    Io(#[from] std::io::Error),

    /// The sidecar file is not valid `DeviceSpec` JSON.
    #[error("failed to parse device spec JSON: {0}")]
    Json(#[from] serde_json::Error),

    /// The file declares a newer format version than this build supports.
    #[error("unsupported device-spec version {got} (max supported {max})")]
    UnsupportedVersion {
        /// Version found in the file.
        got: u32,
        /// Highest version this build understands.
        max: u32,
    },

    /// `cameras` is empty.
    #[error("device spec has no cameras")]
    NoCameras,

    /// Two cameras share an id.
    #[error("duplicate camera id `{id}`")]
    DuplicateCameraId {
        /// The offending id.
        id: String,
    },

    /// A camera datasheet value is not strictly positive.
    #[error("camera `{id}`: {field} must be positive")]
    NonPositiveCameraField {
        /// The offending camera id.
        id: String,
        /// The offending field name.
        field: &'static str,
    },

    /// The rig layout references a camera id absent from `cameras`.
    #[error("rig layout references unknown camera id `{id}`")]
    UnknownRigCameraId {
        /// The unmatched id.
        id: String,
    },

    /// Two rig mounts reference the same camera.
    #[error("duplicate rig mount for camera id `{id}`")]
    DuplicateMountId {
        /// The offending id.
        id: String,
    },

    /// A laser-plane normal is not unit length.
    #[error("laser plane for camera `{camera_id}`: normal must be unit length (norm {norm})")]
    NonUnitLaserNormal {
        /// The paired camera id.
        camera_id: String,
        /// The actual norm found.
        norm: f64,
    },
}

impl DeviceSpec {
    /// Load and validate a device spec from a JSON sidecar file.
    pub fn from_path(path: &Path) -> Result<Self, DeviceSpecError> {
        let raw = std::fs::read_to_string(path)?;
        let spec: Self = serde_json::from_str(&raw)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Look up a camera spec by id.
    pub fn camera(&self, id: &str) -> Option<&CameraDeviceSpec> {
        self.cameras.iter().find(|c| c.id == id)
    }

    /// Structural validation: fail-fast on transcription errors
    /// (ADR 0019). Called by [`DeviceSpec::from_path`].
    pub fn validate(&self) -> Result<(), DeviceSpecError> {
        if self.version > DEVICE_SPEC_VERSION {
            return Err(DeviceSpecError::UnsupportedVersion {
                got: self.version,
                max: DEVICE_SPEC_VERSION,
            });
        }
        if self.cameras.is_empty() {
            return Err(DeviceSpecError::NoCameras);
        }
        let mut seen = std::collections::BTreeSet::new();
        for cam in &self.cameras {
            if !seen.insert(cam.id.as_str()) {
                return Err(DeviceSpecError::DuplicateCameraId { id: cam.id.clone() });
            }
            let positive: [(&'static str, f64); 4] = [
                ("focal_mm", cam.focal_mm),
                ("pixel_pitch_um", cam.pixel_pitch_um),
                ("resolution_px[0]", f64::from(cam.resolution_px[0])),
                ("resolution_px[1]", f64::from(cam.resolution_px[1])),
            ];
            for (field, value) in positive {
                if !value.is_finite() || value <= 0.0 {
                    return Err(DeviceSpecError::NonPositiveCameraField {
                        id: cam.id.clone(),
                        field,
                    });
                }
            }
        }
        if let Some(rig) = &self.rig {
            let mut mounted = std::collections::BTreeSet::new();
            for mount in &rig.cameras {
                if self.camera(&mount.id).is_none() {
                    return Err(DeviceSpecError::UnknownRigCameraId {
                        id: mount.id.clone(),
                    });
                }
                if !mounted.insert(mount.id.as_str()) {
                    return Err(DeviceSpecError::DuplicateMountId {
                        id: mount.id.clone(),
                    });
                }
            }
            for plane in rig.laser_planes.as_deref().unwrap_or_default() {
                if self.camera(&plane.camera_id).is_none() {
                    return Err(DeviceSpecError::UnknownRigCameraId {
                        id: plane.camera_id.clone(),
                    });
                }
                let norm = plane.normal.iter().map(|c| c * c).sum::<f64>().sqrt();
                if (norm - 1.0).abs() > 1e-6 {
                    return Err(DeviceSpecError::NonUnitLaserNormal {
                        camera_id: plane.camera_id.clone(),
                        norm,
                    });
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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

    fn full_spec() -> DeviceSpec {
        DeviceSpec {
            version: DEVICE_SPEC_VERSION,
            cameras: vec![camera("cam0"), camera("cam1")],
            rig: Some(RigLayoutSpec {
                cameras: vec![CameraMountSpec {
                    id: "cam0".to_string(),
                    rig_se3_cam: NominalPoseSpec {
                        rpy_deg: [0.0, 30.0, 0.0],
                        translation_mm: [100.0, 0.0, -50.0],
                    },
                }],
                handeye: Some(HandeyeMountSpec::EyeToHand {
                    rig_se3_base: NominalPoseSpec {
                        rpy_deg: [0.0, 0.0, 90.0],
                        translation_mm: [0.0, 250.0, 0.0],
                    },
                }),
                laser_planes: Some(vec![LaserPlaneSpec {
                    camera_id: "cam1".to_string(),
                    normal: [0.0, 0.0, 1.0],
                    distance_mm: 120.0,
                }]),
            }),
            description: Some("test rig".to_string()),
        }
    }

    #[test]
    fn json_roundtrip_full() {
        let spec = full_spec();
        let json = serde_json::to_string_pretty(&spec).unwrap();
        assert!(json.contains("\"focal_mm\""));
        assert!(json.contains("\"pixel_pitch_um\""));
        assert!(json.contains("\"eye_to_hand\""));
        assert!(json.contains("\"rig_se3_base\""));
        let back: DeviceSpec = serde_json::from_str(&json).unwrap();
        back.validate().unwrap();
        assert_eq!(back.cameras.len(), 2);
        let rig = back.rig.unwrap();
        assert_eq!(rig.cameras[0].rig_se3_cam.rpy_deg, [0.0, 30.0, 0.0]);
        match rig.handeye.unwrap() {
            HandeyeMountSpec::EyeToHand { rig_se3_base } => {
                assert_eq!(rig_se3_base.translation_mm, [0.0, 250.0, 0.0]);
            }
            other => panic!("wrong handeye mode: {other:?}"),
        }
        assert_eq!(rig.laser_planes.unwrap()[0].distance_mm, 120.0);
    }

    #[test]
    fn json_roundtrip_minimal_defaults() {
        let json = r#"{
            "cameras": [{
                "id": "cam0",
                "focal_mm": 16.0,
                "pixel_pitch_um": 4.8,
                "resolution_px": [1920, 1200]
            }]
        }"#;
        let spec: DeviceSpec = serde_json::from_str(json).unwrap();
        spec.validate().unwrap();
        assert_eq!(spec.version, DEVICE_SPEC_VERSION);
        assert!(spec.cameras[0].principal_point_px.is_none());
        assert!(spec.cameras[0].scheimpflug.is_none());
        assert!(spec.rig.is_none());
        let json2 = serde_json::to_string(&spec).unwrap();
        assert!(!json2.contains("principal_point_px"));
        assert!(!json2.contains("\"rig\""));
    }

    #[test]
    fn unknown_field_rejected() {
        let json = r#"{
            "cameras": [{
                "id": "cam0",
                "focal_mm": 16.0,
                "pixel_pitch_mm": 0.0048,
                "resolution_px": [1920, 1200]
            }]
        }"#;
        // `pixel_pitch_mm` is a unit typo — must not parse.
        assert!(serde_json::from_str::<DeviceSpec>(json).is_err());
    }

    #[test]
    fn version_gate() {
        let mut spec = full_spec();
        spec.version = DEVICE_SPEC_VERSION + 1;
        assert!(matches!(
            spec.validate(),
            Err(DeviceSpecError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn validation_failure_modes() {
        let empty = DeviceSpec {
            version: DEVICE_SPEC_VERSION,
            cameras: vec![],
            rig: None,
            description: None,
        };
        assert!(matches!(empty.validate(), Err(DeviceSpecError::NoCameras)));

        let mut dup = full_spec();
        dup.cameras[1].id = "cam0".to_string();
        assert!(matches!(
            dup.validate(),
            Err(DeviceSpecError::DuplicateCameraId { .. })
        ));

        let mut bad_focal = full_spec();
        bad_focal.cameras[0].focal_mm = 0.0;
        assert!(matches!(
            bad_focal.validate(),
            Err(DeviceSpecError::NonPositiveCameraField {
                field: "focal_mm",
                ..
            })
        ));

        let mut bad_pitch = full_spec();
        bad_pitch.cameras[0].pixel_pitch_um = -4.8;
        assert!(matches!(
            bad_pitch.validate(),
            Err(DeviceSpecError::NonPositiveCameraField {
                field: "pixel_pitch_um",
                ..
            })
        ));

        let mut bad_res = full_spec();
        bad_res.cameras[0].resolution_px = [0, 540];
        assert!(matches!(
            bad_res.validate(),
            Err(DeviceSpecError::NonPositiveCameraField { .. })
        ));

        let mut ghost_mount = full_spec();
        ghost_mount.rig.as_mut().unwrap().cameras[0].id = "cam9".to_string();
        assert!(matches!(
            ghost_mount.validate(),
            Err(DeviceSpecError::UnknownRigCameraId { .. })
        ));

        let mut dup_mount = full_spec();
        let mount = dup_mount.rig.as_ref().unwrap().cameras[0].clone();
        dup_mount.rig.as_mut().unwrap().cameras.push(mount);
        assert!(matches!(
            dup_mount.validate(),
            Err(DeviceSpecError::DuplicateMountId { .. })
        ));

        let mut bad_normal = full_spec();
        bad_normal
            .rig
            .as_mut()
            .unwrap()
            .laser_planes
            .as_mut()
            .unwrap()[0]
            .normal = [0.0, 0.0, 2.0];
        assert!(matches!(
            bad_normal.validate(),
            Err(DeviceSpecError::NonUnitLaserNormal { .. })
        ));
    }

    #[test]
    fn from_path_loads_and_validates() {
        let dir = std::env::temp_dir().join(format!("device-spec-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(DEVICE_SPEC_FILENAME);
        std::fs::write(&path, serde_json::to_string(&full_spec()).unwrap()).unwrap();
        let spec = DeviceSpec::from_path(&path).unwrap();
        assert_eq!(spec.cameras.len(), 2);
        assert!(spec.camera("cam1").is_some());
        assert!(spec.camera("cam9").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
