//! The dataset manifest a registry entry describes.
//!
//! Every Tier-B run goes through the pipeline's `dataset_runner`, which reads
//! a [`DatasetSpec`]. An entry either carries that manifest (`spec`, inline or
//! by path) or describes its data with the registry's own fields — `board`,
//! `cameras`, `robot_poses` and `detector` — which [`dataset_spec`] lowers to
//! the same manifest here, in one place.

use std::path::Path;

use anyhow::{Context, Result};
use vision_calibration_dataset::{
    CameraSource, DatasetSpec, DetectorSpec, ImagePattern, PoseConvention, PosePairing,
    RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec, Topology, TransformConvention,
    TranslationUnits,
};

use crate::registry::{BenchEntry, BoardGeometry, CameraLayout, PoseSource, ProblemKind, SpecRef};

/// Pairing token of a rig image: its filename after up to two leading
/// `<alphanumeric run><separator>` segments, the per-camera tag. `Im_L_2.png`
/// and `Im_R_2.png` share `2.png`; `Cam1_<pose>.png` and `Cam2_<pose>.png`
/// share `<pose>.png`.
const RIG_PAIRING_REGEX: &str = r"^(?:[A-Za-z0-9]+[_-]){0,2}(?<view>.+)$";

/// Floor on decoded PuzzleBoard corners per view: sparser decodes are too
/// unreliable to keep.
const PUZZLEBOARD_MIN_FEATURES: usize = 8;

/// Field of a snap-list pose file holding the row-major `T_base_tcp` matrix.
const SNAP_LIST_MATRIX_FIELD: &str = "tcp2base";

/// The manifest of `entry`'s dataset: its `spec` when it has one, otherwise
/// the manifest its `board`, `cameras`, `robot_poses` and `detector` fields
/// describe. Paths in it are relative to `entry.data_root`.
///
/// # Errors
///
/// A `spec` path that cannot be read or parsed, or registry fields that do
/// not describe a dataset (no board, an unknown layout, pose format or
/// convention, or a ChArUco board without a dictionary).
pub fn dataset_spec(entry: &BenchEntry) -> Result<DatasetSpec> {
    match &entry.spec {
        Some(SpecRef::Inline(spec)) => Ok((**spec).clone()),
        Some(SpecRef::Path { path }) => {
            let path = resolve_against(&entry.data_root, path);
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("read dataset manifest {}", path.display()))?;
            serde_json::from_str(&raw)
                .with_context(|| format!("parse dataset manifest {}", path.display()))
        }
        None => lower_registry_fields(entry)
            .with_context(|| format!("entry `{}`: registry fields", entry.id)),
    }
}

/// Join `path` onto `base` unless it is already absolute.
pub fn resolve_against(base: &Path, path: &Path) -> std::path::PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn lower_registry_fields(entry: &BenchEntry) -> Result<DatasetSpec> {
    let board = entry
        .board
        .as_ref()
        .context("an entry without a `spec` needs a `board`")?;
    anyhow::ensure!(!entry.cameras.is_empty(), "no `cameras`");
    let target = target(board)?;

    let min_features_per_view = match &target {
        // A strict grid keeps only complete boards.
        TargetSpec::Chessboard { rows, cols, .. } if board.strict_grid => {
            Some((*rows * *cols) as usize)
        }
        TargetSpec::Puzzleboard { .. } => Some(PUZZLEBOARD_MIN_FEATURES),
        _ => None,
    };
    let chess_corners = entry.detector.as_ref().and_then(|d| d.chess_corners);
    let detector =
        (chess_corners.is_some() || min_features_per_view.is_some()).then_some(DetectorSpec {
            chess_corners,
            min_features_per_view,
        });

    let topology = entry.problem.topology();
    let hand_eye = matches!(
        topology,
        Topology::SingleCamHandeye | Topology::RigHandeye | Topology::RigHandeyeLaserline
    );
    let (robot_poses, pose_convention) = if hand_eye {
        let source = entry
            .robot_poses
            .as_ref()
            .context("a hand-eye entry needs `robot_poses`")?;
        let (source, convention) = robot_pose_source(source)?;
        (Some(source), Some(convention))
    } else {
        (None, None)
    };
    let pose_pairing = match topology {
        Topology::PlanarIntrinsics | Topology::LaserlineDevice => None,
        Topology::RigExtrinsics => Some(PosePairing::SharedFilenameToken {
            regex: RIG_PAIRING_REGEX.to_string(),
            group: "view".to_string(),
        }),
        _ => Some(PosePairing::ByIndex),
    };

    Ok(DatasetSpec {
        version: 1,
        cameras: entry.cameras.iter().map(camera_source).collect(),
        target,
        detector,
        robot_poses,
        laser: None,
        upstream_calibration: None,
        topology,
        pose_pairing,
        pose_convention,
        unresolved: Vec::new(),
        description: entry.notes.clone(),
    })
}

/// The detection target a board describes, selected by its `layout`:
/// `charuco`, `puzzleboard`, or a chessboard for anything else.
fn target(board: &BoardGeometry) -> Result<TargetSpec> {
    let rows = u32::try_from(board.rows).context("board rows")?;
    let cols = u32::try_from(board.cols).context("board cols")?;
    let layout = board.layout.as_deref().unwrap_or("checkerboard");
    Ok(if layout.eq_ignore_ascii_case("charuco") {
        let dictionary = board
            .dictionary
            .clone()
            .context("a charuco board needs a `dictionary`")?;
        // The OpenCV ChArUco default marker scale.
        let marker_size_rel = f64::from(board.marker_size_rel.unwrap_or(0.75));
        TargetSpec::Charuco {
            rows,
            cols,
            square_size_m: board.cell_size_m,
            marker_size_m: board.cell_size_m * marker_size_rel,
            dictionary,
        }
    } else if layout.eq_ignore_ascii_case("puzzleboard") {
        TargetSpec::Puzzleboard {
            layout: format!("{rows}x{cols}"),
            cell_size_m: board.cell_size_m,
        }
    } else {
        TargetSpec::Chessboard {
            rows,
            cols,
            square_size_m: board.cell_size_m,
        }
    })
}

fn camera_source(camera: &CameraLayout) -> CameraSource {
    let pattern = match camera.folder.as_str() {
        "" | "." => camera.filename_glob.clone(),
        folder => Path::new(folder)
            .join(&camera.filename_glob)
            .to_string_lossy()
            .into_owned(),
    };
    CameraSource {
        id: camera.id.clone(),
        images: ImagePattern::Glob { pattern },
        roi_xywh: camera.tile,
        laser_images: None,
    }
}

fn robot_pose_source(source: &PoseSource) -> Result<(RobotPoseSource, PoseConvention)> {
    let (format, matrix_field) = match source.format.as_str() {
        "rowmajor4x4" => (RobotPoseFormat::Rowmajor4x4, None),
        "counted4x4" => (RobotPoseFormat::Counted4x4, None),
        "snap_list_json" => (
            RobotPoseFormat::Json,
            Some(SNAP_LIST_MATRIX_FIELD.to_string()),
        ),
        other => anyhow::bail!(
            "unsupported robot-pose format '{other}' \
             (use 'rowmajor4x4', 'counted4x4', or 'snap_list_json')"
        ),
    };
    let transform = match source.convention.as_str() {
        "base_se3_gripper" => TransformConvention::TBaseTcp,
        "gripper_se3_base" => TransformConvention::TTcpBase,
        other => anyhow::bail!(
            "unsupported robot-pose convention '{other}' \
             (use 'base_se3_gripper' or 'gripper_se3_base')"
        ),
    };
    let translation_units = match source.units.as_deref() {
        Some("mm") => TranslationUnits::Mm,
        Some("m") | None => TranslationUnits::M,
        Some(other) => anyhow::bail!("unsupported pose units '{other}' (use 'mm' or 'm')"),
    };
    Ok((
        RobotPoseSource {
            path: source.path.clone(),
            format,
            columns: None,
            matrix_field,
        },
        PoseConvention {
            transform,
            rotation_format: RotationFormat::Matrix4x4RowMajor,
            translation_units,
        },
    ))
}

impl ProblemKind {
    /// The manifest topology of this problem kind.
    pub fn topology(self) -> Topology {
        match self {
            ProblemKind::PlanarIntrinsics => Topology::PlanarIntrinsics,
            ProblemKind::SingleCamHandeye => Topology::SingleCamHandeye,
            ProblemKind::RigExtrinsics => Topology::RigExtrinsics,
            ProblemKind::RigHandeye => Topology::RigHandeye,
            ProblemKind::RigHandeyeLaserline => Topology::RigHandeyeLaserline,
            ProblemKind::LaserlineDevice => Topology::LaserlineDevice,
            ProblemKind::RigLaserlineDevice => Topology::RigLaserlineDevice,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vision_calibration_dataset::validate;

    use crate::registry::DetectorOverride;

    fn entry(problem: ProblemKind, board: BoardGeometry, cameras: Vec<CameraLayout>) -> BenchEntry {
        let json = serde_json::json!({
            "id": "e",
            "visibility": "public",
            "tiers": ["b"],
            "problem": problem,
            "data_root": "/data",
            "board": board,
            "cameras": cameras,
        });
        serde_json::from_value(json).expect("entry")
    }

    fn board(layout: &str) -> BoardGeometry {
        BoardGeometry {
            rows: 7,
            cols: 11,
            cell_size_m: 0.03,
            dictionary: Some("DICT_4X4_1000".into()),
            layout: Some(layout.into()),
            marker_size_rel: None,
            strict_grid: false,
        }
    }

    fn camera(id: &str, folder: &str, tile: Option<[u32; 4]>) -> CameraLayout {
        CameraLayout {
            id: id.into(),
            folder: folder.into(),
            filename_glob: "*.png".into(),
            tile,
            expected_size: None,
            color: false,
        }
    }

    fn pose_source(format: &str, convention: &str, units: Option<&str>) -> PoseSource {
        PoseSource {
            path: PathBuf::from("poses.txt"),
            format: format.into(),
            convention: convention.into(),
            units: units.map(str::to_string),
        }
    }

    #[test]
    fn rig_pairing_token_strips_the_camera_tag() {
        let re = regex_lite(RIG_PAIRING_REGEX);
        for (name, token) in [
            ("Im_L_2.png", "2.png"),
            ("Im_R_10.png", "10.png"),
            ("Cam1_-14.0Rx+0.0Ry.png", "-14.0Rx+0.0Ry.png"),
            ("Cam2_-14.0Rx+0.0Ry.png", "-14.0Rx+0.0Ry.png"),
            ("image7.png", "image7.png"),
        ] {
            assert_eq!(re(name), token, "{name}");
        }
    }

    /// Applies `regex` the way the pipeline's filename-token pairing does.
    fn regex_lite(regex: &str) -> impl Fn(&str) -> String {
        let re = regex::Regex::new(regex).expect("regex");
        move |name| re.captures(name).expect("match")["view"].to_string()
    }

    #[test]
    fn chessboard_rig_lowers_to_a_valid_token_paired_manifest() {
        let e = entry(
            ProblemKind::RigExtrinsics,
            board("checkerboard"),
            vec![
                camera("left", "imgs/left", None),
                camera("right", "imgs/right", None),
            ],
        );
        let spec = dataset_spec(&e).expect("spec");
        validate(&spec).expect("valid");
        assert!(matches!(
            spec.target,
            TargetSpec::Chessboard { rows: 7, cols: 11, square_size_m } if square_size_m == 0.03
        ));
        assert!(spec.detector.is_none());
        assert!(matches!(
            spec.pose_pairing,
            Some(PosePairing::SharedFilenameToken { .. })
        ));
        assert!(matches!(
            &spec.cameras[0].images,
            ImagePattern::Glob { pattern } if pattern == "imgs/left/*.png"
        ));
    }

    #[test]
    fn strict_grid_keeps_only_complete_boards() {
        let mut b = board("checkerboard");
        b.strict_grid = true;
        let spec = dataset_spec(&entry(
            ProblemKind::PlanarIntrinsics,
            b,
            vec![camera("cam0", ".", None)],
        ))
        .expect("spec");
        assert_eq!(
            spec.detector.and_then(|d| d.min_features_per_view),
            Some(77)
        );
        assert!(matches!(
            &spec.cameras[0].images,
            ImagePattern::Glob { pattern } if pattern == "*.png"
        ));
    }

    #[test]
    fn charuco_board_takes_the_marker_scale() {
        let mut b = board("charuco");
        b.cell_size_m = 0.0052;
        b.marker_size_rel = Some(0.75);
        let spec = dataset_spec(&entry(
            ProblemKind::PlanarIntrinsics,
            b,
            vec![camera("cam0", "", Some([720, 0, 720, 540]))],
        ))
        .expect("spec");
        let TargetSpec::Charuco {
            square_size_m,
            marker_size_m,
            ..
        } = spec.target
        else {
            panic!("expected a charuco target");
        };
        assert_eq!((marker_size_m / square_size_m) as f32, 0.75);
        assert_eq!(spec.cameras[0].roi_xywh, Some([720, 0, 720, 540]));
    }

    #[test]
    fn puzzleboard_names_its_grid_and_keeps_dense_views() {
        let mut b = board("puzzleboard");
        b.rows = 130;
        b.cols = 130;
        let spec = dataset_spec(&entry(
            ProblemKind::PlanarIntrinsics,
            b,
            vec![camera("cam0", "", None)],
        ))
        .expect("spec");
        assert!(matches!(
            &spec.target,
            TargetSpec::Puzzleboard { layout, .. } if layout == "130x130"
        ));
        assert_eq!(
            spec.detector.and_then(|d| d.min_features_per_view),
            Some(PUZZLEBOARD_MIN_FEATURES)
        );
    }

    #[test]
    fn detector_override_reaches_the_manifest() {
        let mut e = entry(
            ProblemKind::PlanarIntrinsics,
            board("checkerboard"),
            vec![camera("cam0", "", None)],
        );
        e.detector = Some(DetectorOverride {
            chess_corners: Some(crate::registry::ChessCornersDetectorSpec {
                threshold_value: Some(30.0),
                ..Default::default()
            }),
            extra: Default::default(),
        });
        let spec = dataset_spec(&e).expect("spec");
        let chess = spec.detector.and_then(|d| d.chess_corners).expect("chess");
        assert_eq!(chess.threshold_value, Some(30.0));
    }

    #[test]
    fn robot_pose_sources_map_to_manifest_formats() {
        let cases = [
            (
                pose_source("rowmajor4x4", "base_se3_gripper", None),
                RobotPoseFormat::Rowmajor4x4,
                TransformConvention::TBaseTcp,
                TranslationUnits::M,
            ),
            (
                pose_source("counted4x4", "gripper_se3_base", Some("mm")),
                RobotPoseFormat::Counted4x4,
                TransformConvention::TTcpBase,
                TranslationUnits::Mm,
            ),
            (
                pose_source("snap_list_json", "base_se3_gripper", Some("mm")),
                RobotPoseFormat::Json,
                TransformConvention::TBaseTcp,
                TranslationUnits::Mm,
            ),
        ];
        for (source, format, transform, units) in cases {
            let mut e = entry(
                ProblemKind::RigHandeye,
                board("checkerboard"),
                vec![camera("a", "a", None), camera("b", "b", None)],
            );
            e.robot_poses = Some(source);
            let spec = dataset_spec(&e).expect("spec");
            validate(&spec).expect("valid");
            let poses = spec.robot_poses.expect("poses");
            let convention = spec.pose_convention.expect("convention");
            assert_eq!(poses.format, format);
            assert_eq!(convention.transform, transform);
            assert_eq!(convention.translation_units, units);
            assert!(matches!(spec.pose_pairing, Some(PosePairing::ByIndex)));
        }
    }

    #[test]
    fn unknown_pose_format_is_an_error() {
        let mut e = entry(
            ProblemKind::SingleCamHandeye,
            board("checkerboard"),
            vec![camera("cam0", "", None)],
        );
        e.robot_poses = Some(pose_source("csv", "base_se3_gripper", None));
        let err = dataset_spec(&e).unwrap_err();
        assert!(format!("{err:#}").contains("unsupported robot-pose format"));
    }
}
