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
    CameraSource, DatasetSpec, DetectorSpec, ImagePattern, LaserExtractionSpec, LaserScanAxis,
    PoseConvention, PosePairing, RobotPoseFormat, RobotPoseSource, RotationFormat, TargetSpec,
    TransformConvention, TranslationUnits,
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

/// The manifest of `entry`'s joint rig hand-eye and laser-plane calibration:
/// its rig hand-eye manifest `spec` as a `rig_handeye_laserline` dataset.
///
/// Every camera reads its region of the laser frame each row of the
/// snap-list pose file names, in row order, so laser frame `i` pairs with
/// target view `i` and robot pose `i` (`by_index`, as the poses already do).
/// A frame whose line the extractor does not find leaves its camera's slot
/// empty. The extraction parameters are the entry's `laser.params`
/// (`axis`, `sigma`, `pos_thresh`, `neg_thresh`, `min_points`) over the
/// manifest defaults.
///
/// # Errors
///
/// An entry without `laser`, without a `snap_list_json` pose file, a pose
/// file that cannot be read, a row that names no laser frame, or a laser
/// parameter of the wrong type.
pub fn joint_laserline_spec(entry: &BenchEntry, spec: &DatasetSpec) -> Result<DatasetSpec> {
    let laser = entry
        .laser
        .as_ref()
        .with_context(|| format!("entry `{}` declares no laser data", entry.id))?;
    let pose_src = entry
        .robot_poses
        .as_ref()
        .filter(|src| src.format == "snap_list_json")
        .with_context(|| {
            format!(
                "entry `{}` declares laser data, which pairs with target frames through \
                 a snap_list_json pose file",
                entry.id
            )
        })?;
    let path = resolve_against(&entry.data_root, &pose_src.path);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("read laser pose list {}", path.display()))?;
    let rows: Vec<SnapListRow> = serde_json::from_str(&text)
        .with_context(|| format!("parse laser pose list {}", path.display()))?;
    let laser_frames = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.laser_image.map(Into::into).with_context(|| {
                format!("pose list {} row {i} names no laser_image", path.display())
            })
        })
        .collect::<Result<Vec<std::path::PathBuf>>>()?;

    let mut spec = spec.clone();
    spec.topology = ProblemKind::RigHandeyeLaserline;
    spec.laser = Some(laser_extraction(&laser.params)?);
    for camera in &mut spec.cameras {
        camera.laser_images = Some(ImagePattern::List {
            paths: laser_frames.clone(),
        });
    }
    Ok(spec)
}

/// The laser frame a snap-list row names.
#[derive(serde::Deserialize)]
struct SnapListRow {
    laser_image: Option<String>,
}

/// The manifest's laser extraction with a registry entry's `laser.params`
/// over its defaults.
fn laser_extraction(params: &serde_json::Value) -> Result<LaserExtractionSpec> {
    let mut spec = LaserExtractionSpec::default();
    let number = |key: &str| -> Result<Option<f64>> {
        params
            .get(key)
            .map(|v| {
                v.as_f64()
                    .with_context(|| format!("laser.params.{key} is not a number"))
            })
            .transpose()
    };
    if let Some(axis) = params.get("axis") {
        spec.scan_axis = match axis.as_str() {
            Some("cols") => LaserScanAxis::Cols,
            Some("rows") => LaserScanAxis::Rows,
            _ => anyhow::bail!("laser.params.axis must be \"cols\" or \"rows\", got {axis}"),
        };
    }
    if let Some(sigma) = number("sigma")? {
        spec.sigma = sigma;
    }
    if let Some(pos_thresh) = number("pos_thresh")? {
        spec.pos_thresh = pos_thresh;
    }
    if let Some(neg_thresh) = number("neg_thresh")? {
        spec.neg_thresh = neg_thresh;
    }
    if let Some(min_points) = params.get("min_points") {
        spec.min_points = min_points
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .context("laser.params.min_points is not a count")?;
    }
    Ok(spec)
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

    let topology = entry.problem;
    let hand_eye = matches!(
        topology,
        ProblemKind::SingleCamHandeye | ProblemKind::RigHandeye | ProblemKind::RigHandeyeLaserline
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
        ProblemKind::PlanarIntrinsics | ProblemKind::LaserlineDevice => None,
        ProblemKind::RigExtrinsics => Some(PosePairing::SharedFilenameToken {
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
    // Manifest globs use `/` on every OS.
    let pattern = match camera.folder.as_str() {
        "" | "." => camera.filename_glob.clone(),
        folder => format!(
            "{}/{}",
            folder.trim_end_matches(['/', '\\']),
            camera.filename_glob
        ),
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

    /// A rig hand-eye entry with laser data over a snap-list pose file of
    /// `rows` in a fresh directory.
    fn laser_entry(name: &str, rows: serde_json::Value) -> BenchEntry {
        let root = std::env::temp_dir().join(format!(
            "calib-bench-manifest-{name}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).expect("temp dir");
        std::fs::write(root.join("poses.json"), rows.to_string()).expect("pose list");
        let mut e = entry(
            ProblemKind::RigHandeye,
            board("checkerboard"),
            vec![
                camera("a", "", Some([0, 0, 10, 10])),
                camera("b", "", Some([10, 0, 10, 10])),
            ],
        );
        e.data_root = root;
        let mut source = pose_source("snap_list_json", "base_se3_gripper", Some("mm"));
        source.path = PathBuf::from("poses.json");
        e.robot_poses = Some(source);
        e.laser = Some(
            serde_json::from_value(serde_json::json!({
                "method": "vision-metrology-edge1d",
                "params": { "axis": "rows", "sigma": 2.0, "pos_thresh": 5.0, "min_points": 50 },
                "residual_type": "extraction_only",
            }))
            .expect("laser config"),
        );
        e
    }

    #[test]
    fn laser_entry_lowers_to_a_valid_joint_manifest() {
        let e = laser_entry(
            "valid",
            serde_json::json!([
                { "type": "double_snap", "target_image": "target_0.png", "laser_image": "laser_0.png" },
                { "type": "target_snap", "target_image": "target_1.png", "laser_image": "laser_1.png" },
            ]),
        );
        let spec = joint_laserline_spec(&e, &dataset_spec(&e).expect("spec")).expect("joint");
        validate(&spec).expect("valid");
        assert_eq!(spec.topology, ProblemKind::RigHandeyeLaserline);
        assert!(matches!(spec.pose_pairing, Some(PosePairing::ByIndex)));
        let laser = spec.laser.expect("laser");
        assert_eq!(laser.scan_axis, LaserScanAxis::Rows);
        assert_eq!((laser.sigma, laser.pos_thresh), (2.0, 5.0));
        assert_eq!(laser.neg_thresh, LaserExtractionSpec::default().neg_thresh);
        assert_eq!(laser.min_points, 50);
        for camera in &spec.cameras {
            match &camera.laser_images {
                Some(ImagePattern::List { paths }) => assert_eq!(
                    paths,
                    &[PathBuf::from("laser_0.png"), PathBuf::from("laser_1.png")]
                ),
                other => panic!("camera {}: {other:?}", camera.id),
            }
        }
    }

    #[test]
    fn a_pose_row_without_a_laser_frame_is_an_error() {
        let e = laser_entry(
            "missing",
            serde_json::json!([{ "type": "target_snap", "target_image": "target_0.png" }]),
        );
        let err = joint_laserline_spec(&e, &dataset_spec(&e).expect("spec")).unwrap_err();
        assert!(
            format!("{err:#}").contains("row 0 names no laser_image"),
            "{err:#}"
        );
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
