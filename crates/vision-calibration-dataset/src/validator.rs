//! Manifest validation. Catches structural errors that would
//! otherwise surface as a fail-fast `AskUser` event downstream.

use crate::spec::{
    DatasetSpec, ImagePattern, RobotPoseFormat, RobotPoseSource, RotationFormat, Topology,
};
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

    /// A tabular pose format (csv / json / jsonl) needs a `columns`
    /// mapping to find the pose fields, but none was given.
    #[error("pose format {format:?} requires a columns mapping but none was given")]
    MissingPoseColumns {
        /// The declared pose-file format.
        format: RobotPoseFormat,
    },

    /// The headerless `rowmajor4x4` format has a fixed 16-values-per-line
    /// layout; a `columns` mapping or a non-matrix rotation format would
    /// be silently ignored, so both are rejected up front (ADR 0019).
    #[error("pose format rowmajor4x4 {problem}")]
    BadMatrixPoseConfig {
        /// What is inconsistent (human-readable).
        problem: String,
    },

    /// An ROI rectangle has zero width or height.
    #[error("camera {camera_id:?} has degenerate ROI {roi:?}")]
    DegenerateRoi {
        /// The offending camera id.
        camera_id: String,
        /// The ROI as declared in the manifest (`[x, y, w, h]`).
        roi: [u32; 4],
    },

    /// A laser-only manifest field is set on a topology that does not
    /// use it. Per ADR 0019 an ignored field is a fail-fast event —
    /// the user almost certainly selected the wrong topology.
    #[error("topology {topology:?} does not use {field}")]
    FieldUnusedByTopology {
        /// The offending topology.
        topology: Topology,
        /// Dotted path of the unused field (e.g. `"cameras[cam0].laser_images"`).
        field: String,
    },

    /// A laser topology camera is missing its laser-frame source.
    #[error("camera {0:?} needs laser_images for a laser topology")]
    MissingLaserImages(String),

    /// A camera's `laser_images` glob pattern or path list is empty.
    #[error("camera {0:?} has an empty laser_images pattern")]
    EmptyLaserImagePattern(String),

    /// `RigLaserlineDevice` needs a frozen upstream rig hand-eye export.
    #[error(
        "topology {topology:?} requires upstream_calibration \
         (path to a frozen rig hand-eye export JSON) but none was given"
    )]
    MissingUpstreamCalibration {
        /// The offending topology.
        topology: Topology,
    },

    /// Laser extraction parameters are out of range.
    #[error("laser extraction: {0}")]
    BadLaserExtraction(String),
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

    validate_laser_fields(spec)?;

    if let Some(robot) = &spec.robot_poses {
        validate_pose_columns(
            robot,
            spec.pose_convention.as_ref().map(|c| c.rotation_format),
        )?;
    }

    Ok(())
}

/// Laser-field rules (ADR 0021): laser topologies require laser
/// sources everywhere; non-laser topologies must not carry any of the
/// laser-only fields (an ignored field is a fail-fast event).
fn validate_laser_fields(spec: &DatasetSpec) -> Result<(), ValidationError> {
    let uses_laser = matches!(
        spec.topology,
        Topology::LaserlineDevice | Topology::RigLaserlineDevice
    );

    if !uses_laser {
        if spec.laser.is_some() {
            return Err(ValidationError::FieldUnusedByTopology {
                topology: spec.topology,
                field: "laser".into(),
            });
        }
        if spec.upstream_calibration.is_some() {
            return Err(ValidationError::FieldUnusedByTopology {
                topology: spec.topology,
                field: "upstream_calibration".into(),
            });
        }
        if let Some(cam) = spec.cameras.iter().find(|c| c.laser_images.is_some()) {
            return Err(ValidationError::FieldUnusedByTopology {
                topology: spec.topology,
                field: format!("cameras[{}].laser_images", cam.id),
            });
        }
        return Ok(());
    }

    // Every camera in a laser topology needs a laser source — a rig
    // camera with no laser views would leave its plane unconstrained.
    for cam in &spec.cameras {
        match &cam.laser_images {
            None => return Err(ValidationError::MissingLaserImages(cam.id.clone())),
            Some(ImagePattern::Glob { pattern }) if pattern.is_empty() => {
                return Err(ValidationError::EmptyLaserImagePattern(cam.id.clone()));
            }
            Some(ImagePattern::List { paths }) if paths.is_empty() => {
                return Err(ValidationError::EmptyLaserImagePattern(cam.id.clone()));
            }
            Some(_) => {}
        }
    }

    if let Some(laser) = &spec.laser {
        if laser.sigma <= 0.0 {
            return Err(ValidationError::BadLaserExtraction(format!(
                "sigma must be positive, got {}",
                laser.sigma
            )));
        }
        if laser.pos_thresh <= 0.0 || laser.neg_thresh <= 0.0 {
            return Err(ValidationError::BadLaserExtraction(format!(
                "edge thresholds must be positive, got pos={} neg={}",
                laser.pos_thresh, laser.neg_thresh
            )));
        }
        // `min_points == 0` makes the runner's `pixels.len() <
        // min_points` drop-bar vacuously false, so an empty extraction
        // counts as a usable view (rig path even increments
        // `per_camera_usable`) and the laser plane is left
        // unconstrained — a misleading "success". Require at least the
        // two points a line needs.
        if laser.min_points < 2 {
            return Err(ValidationError::BadLaserExtraction(format!(
                "min_points must be at least 2, got {}",
                laser.min_points
            )));
        }
    }

    match spec.topology {
        Topology::RigLaserlineDevice if spec.upstream_calibration.is_none() => {
            Err(ValidationError::MissingUpstreamCalibration {
                topology: spec.topology,
            })
        }
        Topology::LaserlineDevice if spec.upstream_calibration.is_some() => {
            // Single-camera laserline calibrates its own intrinsics —
            // an upstream export here means the wrong topology.
            Err(ValidationError::FieldUnusedByTopology {
                topology: spec.topology,
                field: "upstream_calibration".into(),
            })
        }
        _ => Ok(()),
    }
}

fn topology_needs_robot(t: Topology) -> bool {
    matches!(
        t,
        Topology::SingleCamHandeye | Topology::RigHandeye | Topology::RigLaserlineDevice
    )
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

    if robot.format == RobotPoseFormat::Rowmajor4x4 {
        if robot.columns.is_some() {
            return Err(ValidationError::BadMatrixPoseConfig {
                problem: "is headerless (16 values per line); remove the columns mapping".into(),
            });
        }
        if robot.matrix_field.is_some() {
            return Err(ValidationError::BadMatrixPoseConfig {
                problem: "is headerless (16 values per line); remove matrix_field".into(),
            });
        }
        if format != RotationFormat::Matrix4x4RowMajor {
            return Err(ValidationError::BadMatrixPoseConfig {
                problem: format!(
                    "requires pose_convention.rotation_format = matrix4x4_row_major, got {format:?}"
                ),
            });
        }
        return Ok(());
    }

    if let Some(field) = &robot.matrix_field {
        if robot.format == RobotPoseFormat::Csv {
            return Err(ValidationError::BadMatrixPoseConfig {
                problem: format!(
                    "matrix_field {field:?} requires a json or jsonl pose file \
                     (CSV rows cannot hold a nested matrix)"
                ),
            });
        }
        if robot.columns.is_some() {
            return Err(ValidationError::BadMatrixPoseConfig {
                problem: format!(
                    "matrix_field {field:?} and a columns mapping are mutually exclusive"
                ),
            });
        }
        if format != RotationFormat::Matrix4x4RowMajor {
            return Err(ValidationError::BadMatrixPoseConfig {
                problem: format!(
                    "matrix_field {field:?} requires pose_convention.rotation_format = \
                     matrix4x4_row_major, got {format:?}"
                ),
            });
        }
        return Ok(());
    }

    let Some(columns) = &robot.columns else {
        return Err(ValidationError::MissingPoseColumns {
            format: robot.format,
        });
    };
    let expected = expected_rotation_columns(format);
    let got = columns.rotation.len();
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
                laser_images: None,
            }],
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
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
            laser_images: None,
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
            laser_images: None,
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
            laser_images: None,
        });
        let err = validate(&spec).unwrap_err();
        assert!(matches!(err, ValidationError::DuplicateCameraId(_)));
    }

    #[test]
    fn rotation_column_count_must_match_format() {
        let robot = RobotPoseSource {
            path: "poses.csv".into(),
            format: RobotPoseFormat::Csv,
            columns: Some(PoseColumnMap {
                pose_id: None,
                tx: "x".into(),
                ty: "y".into(),
                tz: "z".into(),
                // QuatXyzw needs 4 columns; supply 3 to trigger.
                rotation: vec!["rx".into(), "ry".into(), "rz".into()],
            }),
            matrix_field: None,
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

    fn handeye_spec_with(robot: RobotPoseSource, rotation_format: RotationFormat) -> DatasetSpec {
        let mut spec = planar_chessboard_minimal();
        spec.topology = Topology::SingleCamHandeye;
        spec.robot_poses = Some(robot);
        spec.pose_convention = Some(PoseConvention {
            transform: TransformConvention::TBaseTcp,
            rotation_format,
            translation_units: TranslationUnits::M,
        });
        spec
    }

    #[test]
    fn tabular_format_requires_columns() {
        let robot = RobotPoseSource {
            path: "poses.csv".into(),
            format: RobotPoseFormat::Csv,
            columns: None,
            matrix_field: None,
        };
        let err = validate(&handeye_spec_with(robot, RotationFormat::QuatXyzw)).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::MissingPoseColumns {
                format: RobotPoseFormat::Csv
            }
        ));
    }

    #[test]
    fn rowmajor4x4_validates_without_columns() {
        let robot = RobotPoseSource {
            path: "RobotPosesVec.txt".into(),
            format: RobotPoseFormat::Rowmajor4x4,
            columns: None,
            matrix_field: None,
        };
        validate(&handeye_spec_with(robot, RotationFormat::Matrix4x4RowMajor)).unwrap();
    }

    #[test]
    fn rowmajor4x4_rejects_columns_mapping() {
        let robot = RobotPoseSource {
            path: "RobotPosesVec.txt".into(),
            format: RobotPoseFormat::Rowmajor4x4,
            columns: Some(PoseColumnMap {
                pose_id: None,
                tx: "x".into(),
                ty: "y".into(),
                tz: "z".into(),
                rotation: vec![],
            }),
            matrix_field: None,
        };
        let err =
            validate(&handeye_spec_with(robot, RotationFormat::Matrix4x4RowMajor)).unwrap_err();
        match err {
            ValidationError::BadMatrixPoseConfig { problem } => {
                assert!(problem.contains("columns"), "got: {problem}");
            }
            other => panic!("expected BadMatrixPoseConfig, got {other:?}"),
        }
    }

    #[test]
    fn rowmajor4x4_rejects_non_matrix_rotation_format() {
        let robot = RobotPoseSource {
            path: "RobotPosesVec.txt".into(),
            format: RobotPoseFormat::Rowmajor4x4,
            columns: None,
            matrix_field: None,
        };
        let err = validate(&handeye_spec_with(robot, RotationFormat::QuatXyzw)).unwrap_err();
        match err {
            ValidationError::BadMatrixPoseConfig { problem } => {
                assert!(
                    problem.contains("matrix4x4_row_major"),
                    "error must name the required format, got: {problem}"
                );
            }
            other => panic!("expected BadMatrixPoseConfig, got {other:?}"),
        }
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

    // ── Laser fields (ADR 0021) ─────────────────────────────────────────

    /// Minimal valid single-camera laserline manifest.
    fn laserline_minimal() -> DatasetSpec {
        let mut spec = planar_chessboard_minimal();
        spec.topology = Topology::LaserlineDevice;
        spec.cameras[0].laser_images = Some(ImagePattern::Glob {
            pattern: "cam0/laser_*.png".into(),
        });
        spec
    }

    /// Minimal valid rig laserline manifest (2 cameras, rowmajor poses).
    fn rig_laserline_minimal() -> DatasetSpec {
        let mut spec = laserline_minimal();
        spec.topology = Topology::RigLaserlineDevice;
        spec.cameras.push(CameraSource {
            id: "cam1".into(),
            images: ImagePattern::Glob {
                pattern: "cam1/*.png".into(),
            },
            roi_xywh: None,
            laser_images: Some(ImagePattern::Glob {
                pattern: "cam1/laser_*.png".into(),
            }),
        });
        spec.robot_poses = Some(RobotPoseSource {
            path: "poses.txt".into(),
            format: RobotPoseFormat::Rowmajor4x4,
            columns: None,
            matrix_field: None,
        });
        spec.pose_convention = Some(PoseConvention {
            transform: TransformConvention::TBaseTcp,
            rotation_format: RotationFormat::Matrix4x4RowMajor,
            translation_units: TranslationUnits::M,
        });
        spec.upstream_calibration = Some("rig_handeye_export.json".into());
        spec
    }

    #[test]
    fn laserline_manifests_validate() {
        validate(&laserline_minimal()).unwrap();
        validate(&rig_laserline_minimal()).unwrap();
    }

    #[test]
    fn laser_fields_rejected_on_non_laser_topology() {
        let mut spec = planar_chessboard_minimal();
        spec.laser = Some(crate::spec::LaserExtractionSpec::default());
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::FieldUnusedByTopology { .. }
        ));

        let mut spec = planar_chessboard_minimal();
        spec.upstream_calibration = Some("export.json".into());
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::FieldUnusedByTopology { .. }
        ));

        let mut spec = planar_chessboard_minimal();
        spec.cameras[0].laser_images = Some(ImagePattern::Glob {
            pattern: "laser_*.png".into(),
        });
        match validate(&spec).unwrap_err() {
            ValidationError::FieldUnusedByTopology { field, .. } => {
                assert_eq!(field, "cameras[cam0].laser_images");
            }
            other => panic!("expected FieldUnusedByTopology, got {other:?}"),
        }
    }

    #[test]
    fn laser_topology_requires_laser_images_on_every_camera() {
        let mut spec = laserline_minimal();
        spec.cameras[0].laser_images = None;
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::MissingLaserImages(_)
        ));

        let mut spec = rig_laserline_minimal();
        spec.cameras[1].laser_images = None;
        match validate(&spec).unwrap_err() {
            ValidationError::MissingLaserImages(id) => assert_eq!(id, "cam1"),
            other => panic!("expected MissingLaserImages, got {other:?}"),
        }
    }

    #[test]
    fn rig_laserline_requires_upstream_and_robot_poses() {
        let mut spec = rig_laserline_minimal();
        spec.upstream_calibration = None;
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::MissingUpstreamCalibration { .. }
        ));

        let mut spec = rig_laserline_minimal();
        spec.robot_poses = None;
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::MissingRobotPoses { .. }
        ));
    }

    #[test]
    fn single_cam_laserline_rejects_upstream() {
        let mut spec = laserline_minimal();
        spec.upstream_calibration = Some("export.json".into());
        match validate(&spec).unwrap_err() {
            ValidationError::FieldUnusedByTopology { field, .. } => {
                assert_eq!(field, "upstream_calibration");
            }
            other => panic!("expected FieldUnusedByTopology, got {other:?}"),
        }
    }

    #[test]
    fn bad_laser_extraction_params_rejected() {
        let mut spec = laserline_minimal();
        spec.laser = Some(crate::spec::LaserExtractionSpec {
            sigma: 0.0,
            ..Default::default()
        });
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::BadLaserExtraction(_)
        ));
    }

    #[test]
    fn zero_min_points_rejected() {
        // A zero (or one) drop-bar would let empty extractions pass as
        // usable views, leaving the laser plane unconstrained.
        for min_points in [0, 1] {
            let mut spec = laserline_minimal();
            spec.laser = Some(crate::spec::LaserExtractionSpec {
                min_points,
                ..Default::default()
            });
            assert!(
                matches!(
                    validate(&spec).unwrap_err(),
                    ValidationError::BadLaserExtraction(_)
                ),
                "min_points={min_points} should be rejected"
            );
        }
    }

    #[test]
    fn empty_laser_pattern_rejected() {
        let mut spec = laserline_minimal();
        spec.cameras[0].laser_images = Some(ImagePattern::List { paths: vec![] });
        assert!(matches!(
            validate(&spec).unwrap_err(),
            ValidationError::EmptyLaserImagePattern(_)
        ));
    }

    // ── matrix_field (ADR 0021) ─────────────────────────────────────────

    fn matrix_field_robot() -> RobotPoseSource {
        RobotPoseSource {
            path: "poses.json".into(),
            format: RobotPoseFormat::Json,
            columns: None,
            matrix_field: Some("tcp2base".into()),
        }
    }

    #[test]
    fn matrix_field_json_validates() {
        validate(&handeye_spec_with(
            matrix_field_robot(),
            RotationFormat::Matrix4x4RowMajor,
        ))
        .unwrap();
    }

    #[test]
    fn matrix_field_rejects_csv_and_columns_and_non_matrix_rotation() {
        let mut robot = matrix_field_robot();
        robot.format = RobotPoseFormat::Csv;
        assert!(matches!(
            validate(&handeye_spec_with(robot, RotationFormat::Matrix4x4RowMajor)).unwrap_err(),
            ValidationError::BadMatrixPoseConfig { .. }
        ));

        let mut robot = matrix_field_robot();
        robot.columns = Some(PoseColumnMap {
            pose_id: None,
            tx: "x".into(),
            ty: "y".into(),
            tz: "z".into(),
            rotation: vec![],
        });
        assert!(matches!(
            validate(&handeye_spec_with(robot, RotationFormat::Matrix4x4RowMajor)).unwrap_err(),
            ValidationError::BadMatrixPoseConfig { .. }
        ));

        assert!(matches!(
            validate(&handeye_spec_with(
                matrix_field_robot(),
                RotationFormat::QuatXyzw
            ))
            .unwrap_err(),
            ValidationError::BadMatrixPoseConfig { .. }
        ));
    }

    #[test]
    fn laser_spec_json_roundtrip_with_defaults() {
        // An omitted `[laser]` table and an explicit default must
        // deserialize identically; partial tables fill from defaults.
        let spec = rig_laserline_minimal();
        let s = serde_json::to_string(&spec).unwrap();
        assert!(!s.contains("\"laser\":"), "None laser stays off the wire");
        let partial: crate::spec::LaserExtractionSpec =
            serde_json::from_str(r#"{"min_points": 120}"#).unwrap();
        assert_eq!(partial.min_points, 120);
        assert_eq!(partial.sigma, 1.2);
    }
}
