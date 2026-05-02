//! Manifest validation. Catches structural errors that would
//! otherwise surface as a fail-fast `AskUser` event downstream.

use crate::spec::{DatasetSpec, ImagePattern, RobotPoseSource, RotationFormat, Topology};
use thiserror::Error;

/// Validation failure modes.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError {
    /// The manifest still has unresolved fields. Per ADR 0019 the
    /// runner refuses to dispatch until they are filled in.
    #[error("manifest has unresolved fields: {0:?}")]
    Unresolved(Vec<String>),

    /// A topology requires robot poses but `robot_poses` is `None`.
    #[error("topology {topology:?} requires robot_poses but none was given")]
    MissingRobotPoses {
        /// The offending topology.
        topology: Topology,
    },

    /// The number of cameras declared doesn't match what the topology
    /// expects — either too few (single-cam topology with extras) or
    /// too many (rig topology with insufficient cameras for stereo).
    #[error("topology {topology:?} expects {expected_min}..={expected_max} camera(s), got {got}")]
    WrongCameraCount {
        /// The offending topology.
        topology: Topology,
        /// How many cameras were declared.
        got: usize,
        /// Inclusive minimum count for the topology.
        expected_min: usize,
        /// Inclusive maximum count (`usize::MAX` for unbounded rigs).
        expected_max: usize,
    },

    /// `pose_convention` is missing while `robot_poses` is set.
    #[error("robot_poses is set but pose_convention is missing")]
    MissingPoseConvention,

    /// Two cameras share the same id, breaking pairing assumptions.
    #[error("duplicate camera id {0:?}")]
    DuplicateCameraId(String),

    /// A glob pattern is empty or a path list is empty.
    #[error("camera {0:?} has no images")]
    EmptyImagePattern(String),

    /// The pose-column rotation list has the wrong length for the
    /// declared rotation format.
    #[error("rotation columns: format {format:?} expects {expected} columns, got {got}")]
    BadRotationColumnCount {
        /// The declared rotation format.
        format: RotationFormat,
        /// How many rotation columns the user supplied.
        got: usize,
        /// How many were expected.
        expected: usize,
    },

    /// An ROI rectangle has zero width or height.
    #[error("camera {camera_id:?} has degenerate ROI {roi:?}")]
    DegenerateRoi {
        /// The offending camera id.
        camera_id: String,
        /// The ROI as declared in the manifest (`[x, y, w, h]`).
        roi: [u32; 4],
    },
}

/// Validate the structural invariants of a manifest. Does not touch
/// the filesystem (e.g. doesn't expand globs). Per ADR 0019, callers
/// should reject the manifest as soon as any error fires.
pub fn validate(spec: &DatasetSpec) -> Result<(), ValidationError> {
    if !spec.unresolved.is_empty() {
        return Err(ValidationError::Unresolved(spec.unresolved.clone()));
    }

    let needs_robot = topology_needs_robot(spec.topology);
    if needs_robot && spec.robot_poses.is_none() {
        return Err(ValidationError::MissingRobotPoses {
            topology: spec.topology,
        });
    }

    let (min_cams, max_cams) = topology_camera_range(spec.topology);
    let got = spec.cameras.len();
    if got < min_cams || got > max_cams {
        return Err(ValidationError::WrongCameraCount {
            topology: spec.topology,
            got,
            expected_min: min_cams,
            expected_max: max_cams,
        });
    }

    if spec.robot_poses.is_some() && spec.pose_convention.is_none() {
        return Err(ValidationError::MissingPoseConvention);
    }

    let mut seen_ids = std::collections::HashSet::new();
    for cam in &spec.cameras {
        if !seen_ids.insert(cam.id.clone()) {
            return Err(ValidationError::DuplicateCameraId(cam.id.clone()));
        }
        match &cam.images {
            ImagePattern::Glob { pattern } if pattern.is_empty() => {
                return Err(ValidationError::EmptyImagePattern(cam.id.clone()));
            }
            ImagePattern::List { paths } if paths.is_empty() => {
                return Err(ValidationError::EmptyImagePattern(cam.id.clone()));
            }
            _ => {}
        }
        if let Some(roi) = cam.roi_xywh
            && (roi[2] == 0 || roi[3] == 0)
        {
            return Err(ValidationError::DegenerateRoi {
                camera_id: cam.id.clone(),
                roi,
            });
        }
    }

    if let Some(robot) = &spec.robot_poses {
        validate_pose_columns(
            robot,
            spec.pose_convention.as_ref().map(|c| c.rotation_format),
        )?;
    }

    Ok(())
}

fn topology_needs_robot(t: Topology) -> bool {
    matches!(t, Topology::SingleCamHandeye | Topology::RigHandeye)
}

fn topology_camera_range(t: Topology) -> (usize, usize) {
    match t {
        // Single-camera topologies must have exactly one camera; an
        // extra camera in the manifest is almost always a user mistake
        // (wrong topology selected, leftover entry from a copy-paste).
        Topology::PlanarIntrinsics
        | Topology::ScheimpflugIntrinsics
        | Topology::SingleCamHandeye
        | Topology::LaserlineDevice => (1, 1),
        // Rig topologies need at least two cameras; the upper bound is
        // unconstrained (puzzle 130×130 ships with 6).
        Topology::RigExtrinsics | Topology::RigHandeye | Topology::RigLaserlineDevice => {
            (2, usize::MAX)
        }
    }
}

fn validate_pose_columns(
    robot: &RobotPoseSource,
    format: Option<RotationFormat>,
) -> Result<(), ValidationError> {
    let Some(format) = format else {
        // Caller has already produced a `MissingPoseConvention` error
        // if `pose_convention` is `None` while `robot_poses` is set.
        return Ok(());
    };
    let expected = expected_rotation_columns(format);
    let got = robot.columns.rotation.len();
    if got != expected {
        return Err(ValidationError::BadRotationColumnCount {
            format,
            got,
            expected,
        });
    }
    Ok(())
}

fn expected_rotation_columns(format: RotationFormat) -> usize {
    match format {
        RotationFormat::QuatXyzw | RotationFormat::QuatWxyz => 4,
        RotationFormat::EulerXyzDeg
        | RotationFormat::EulerXyzRad
        | RotationFormat::EulerZyxDeg
        | RotationFormat::EulerZyxRad
        | RotationFormat::AxisAngleRad => 3,
        RotationFormat::Matrix4x4RowMajor => 16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{
        CameraSource, DatasetSpec, ImagePattern, PoseColumnMap, PoseConvention, RobotPoseFormat,
        RobotPoseSource, RotationFormat, TargetSpec, Topology, TransformConvention,
        TranslationUnits,
    };

    fn planar_chessboard_minimal() -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::Glob {
                    pattern: "cam0/*.png".into(),
                },
                roi_xywh: None,
            }],
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            robot_poses: None,
            topology: Topology::PlanarIntrinsics,
            pose_pairing: None,
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
    }

    #[test]
    fn minimal_planar_chessboard_validates() {
        let spec = planar_chessboard_minimal();
        validate(&spec).unwrap();
    }

    #[test]
    fn unresolved_fields_block_validation() {
        let mut spec = planar_chessboard_minimal();
        spec.unresolved.push("pose_convention.transform".into());
        let err = validate(&spec).unwrap_err();
        match err {
            ValidationError::Unresolved(fields) => {
                assert_eq!(fields, vec!["pose_convention.transform".to_string()]);
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn rig_handeye_requires_robot_poses() {
        let mut spec = planar_chessboard_minimal();
        spec.topology = Topology::RigHandeye;
        spec.cameras.push(CameraSource {
            id: "cam1".into(),
            images: ImagePattern::Glob {
                pattern: "cam1/*.png".into(),
            },
            roi_xywh: None,
        });
        let err = validate(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::MissingRobotPoses { .. }));
    }

    #[test]
    fn planar_topology_rejects_extra_camera() {
        let mut spec = planar_chessboard_minimal();
        spec.cameras.push(CameraSource {
            id: "cam1".into(),
            images: ImagePattern::Glob {
                pattern: "cam1/*.png".into(),
            },
            roi_xywh: None,
        });
        let err = validate(&spec).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::WrongCameraCount {
                expected_min: 1,
                expected_max: 1,
                got: 2,
                ..
            }
        ));
    }

    #[test]
    fn duplicate_camera_id_rejected() {
        let mut spec = planar_chessboard_minimal();
        spec.topology = Topology::RigExtrinsics;
        spec.cameras.push(CameraSource {
            id: "cam0".into(),
            images: ImagePattern::Glob {
                pattern: "cam_dup/*.png".into(),
            },
            roi_xywh: None,
        });
        let err = validate(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::DuplicateCameraId(_)));
    }

    #[test]
    fn rotation_column_count_must_match_format() {
        let robot = RobotPoseSource {
            path: "poses.csv".into(),
            format: RobotPoseFormat::Csv,
            columns: PoseColumnMap {
                pose_id: None,
                tx: "x".into(),
                ty: "y".into(),
                tz: "z".into(),
                // QuatXyzw needs 4 columns; supply 3 to trigger.
                rotation: vec!["rx".into(), "ry".into(), "rz".into()],
            },
        };
        let convention = PoseConvention {
            transform: TransformConvention::TBaseTcp,
            rotation_format: RotationFormat::QuatXyzw,
            translation_units: TranslationUnits::M,
        };
        let mut spec = planar_chessboard_minimal();
        spec.topology = Topology::SingleCamHandeye;
        spec.robot_poses = Some(robot);
        spec.pose_convention = Some(convention);
        let err = validate(&spec).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::BadRotationColumnCount {
                expected: 4,
                got: 3,
                ..
            }
        ));
    }

    #[test]
    fn degenerate_roi_rejected() {
        let mut spec = planar_chessboard_minimal();
        spec.cameras[0].roi_xywh = Some([0, 0, 0, 480]);
        let err = validate(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::DegenerateRoi { .. }));
    }

    #[test]
    fn json_roundtrip_minimal() {
        let spec = planar_chessboard_minimal();
        let s = serde_json::to_string(&spec).unwrap();
        let back: DatasetSpec = serde_json::from_str(&s).unwrap();
        assert_eq!(back.cameras.len(), 1);
        assert!(back.unresolved.is_empty());
    }
}
