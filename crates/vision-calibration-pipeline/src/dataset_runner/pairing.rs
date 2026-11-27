//! View pairing: align per-camera image lists into rig views, and
//! (for hand-eye) align those views with robot poses.
//!
//! Two strategies, per the manifest's `pose_pairing` field:
//! - `by_index` — every camera contributes image `i` to view `i`;
//!   poses pair by row index. Requires equal counts everywhere.
//! - `shared_filename_token` — a regex with a named capture group
//!   extracts a view token from each filename; views are the sorted
//!   intersection of the per-camera token sets; poses pair by their
//!   `pose_id` column equalling the token.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use vision_calibration_dataset::{
    DatasetSpec, PosePairing, ValidationError, validate_robot_pose_source,
};

use super::poses::{load_robot_poses, poses_by_id, require_pose_id_column};
use super::{RunError, expand_camera_images, expand_image_pattern, require_pose_pairing};

/// Per-camera image lists aligned into rig views.
///
/// [`Project::views`](crate::project::Project::views) lists them;
/// [`Project::view_poses`](crate::project::Project::view_poses) and
/// [`Project::view_laser_frames`](crate::project::Project::view_laser_frames)
/// pair them with the manifest's robot poses and laser frames.
#[derive(Debug)]
#[non_exhaustive]
pub struct PairedViews {
    /// `paths[view][camera]` — `None` when a camera has no image for
    /// this view (only possible under token pairing).
    pub paths: Vec<Vec<Option<PathBuf>>>,
    /// View token per view (token pairing) or stringified index
    /// (`by_index`). Used to pair robot poses by `pose_id`.
    pub tokens: Vec<String>,
}

/// One pose of a manifest's robot pose file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoseRow {
    /// The pose's zero-based position among the file's poses: its data row
    /// for CSV (the header excluded) and JSON Lines, its element for JSON,
    /// its line for `rowmajor4x4`, its matrix for `counted4x4`. Blank lines
    /// are not counted. A pose parse error names its row the same way.
    pub index: usize,
    /// Its `pose_id` value, when the manifest maps a `pose_id` column.
    pub id: Option<String>,
}

/// How a dataset's views pair with its robot poses, by the manifest's
/// `pose_pairing`: pose `i` with view `i` under `by_index`, the pose whose
/// `pose_id` is the view's token under `shared_filename_token`.
///
/// What the pairing leaves unmatched is data here rather than an error: a
/// view without a pose, a pose that pairs with no view. A run refuses such
/// a dataset (for instance with [`RunError::PoseCountMismatch`] under
/// `by_index` pairing); this shows which views and rows are at fault.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewPoses {
    /// Per view, in the order of [`PairedViews::tokens`], the pose it pairs
    /// with; `None` for a view no pose pairs with.
    pub views: Vec<Option<PoseRow>>,
    /// The poses that pair with no view, in file order.
    pub unpaired: Vec<PoseRow>,
}

/// How a dataset's laser frames pair with its views, by the manifest's
/// `pose_pairing`: each camera's laser image `i` with view `i` under
/// `by_index`, the laser image whose file name yields the view's token
/// under `shared_filename_token`.
///
/// What the pairing leaves unmatched is data here rather than an error: a
/// view without a camera's laser frame, a laser frame that pairs with no
/// view.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewLaserFrames {
    /// `paths[view][camera]`, shaped like [`PairedViews::paths`]: the laser
    /// frame of that camera in that view; `None` where it has none (a
    /// camera without `laser_images` has none in any view).
    pub paths: Vec<Vec<Option<PathBuf>>>,
    /// The laser frames that pair with no view, by camera, each camera's in
    /// the order its images sort.
    pub unpaired: Vec<UnpairedLaserFrame>,
}

/// A laser frame that pairs with no view.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct UnpairedLaserFrame {
    /// Index of the camera in the manifest's `cameras`.
    pub camera: usize,
    /// The laser image.
    pub path: PathBuf,
    /// The pairing token the frame carries: its index among the camera's
    /// laser images under `by_index`, the token its file name yields under
    /// `shared_filename_token`; `None` when the file name yields none. A
    /// token that is one of the views' belongs to a view another frame of
    /// the camera already pairs with (the first in sort order pairs).
    pub token: Option<String>,
}

/// The views a manifest's images pair into, without reading any image:
/// each camera's images listed (relative paths resolved against
/// `base_dir`) and aligned by the manifest's `pose_pairing`; one view per
/// image for a topology that pairs nothing.
///
/// # Errors
///
/// As the converters fail before detection: an image pattern that does not
/// expand ([`RunError::BadGlob`]), an open pairing
/// ([`RunError::AskUser`]) and the pairing's own errors.
pub(crate) fn pair_dataset_views(
    spec: &DatasetSpec,
    base_dir: &Path,
) -> Result<PairedViews, RunError> {
    let images = spec
        .cameras
        .iter()
        .map(|camera| expand_camera_images(camera, base_dir))
        .collect::<Result<Vec<_>, _>>()?;
    let camera_ids: Vec<String> = spec.cameras.iter().map(|c| c.id.clone()).collect();
    match require_pose_pairing(spec)? {
        Some(pairing) => pair_views(&images, &camera_ids, pairing),
        None => pair_views(&images, &camera_ids, &PosePairing::ByIndex),
    }
}

/// The robot pose of each of the views `tokens`, by the manifest's
/// pairing; `None` when the manifest declares no `robot_poses`. Reads the
/// robot pose file.
///
/// # Errors
///
/// What keeps the pose file from being listed: it cannot be read or parsed
/// ([`RunError::Io`], [`RunError::PoseParse`]), its `pose_convention` or
/// column mapping is missing or does not fit ([`RunError::Validation`]),
/// token pairing has no `pose_id` column ([`RunError::AskUser`]) or a
/// repeated `pose_id` ([`RunError::PairingTokenMismatch`]).
pub(crate) fn list_view_poses(
    spec: &DatasetSpec,
    base_dir: &Path,
    tokens: &[String],
) -> Result<Option<ViewPoses>, RunError> {
    if spec.robot_poses.is_none() {
        return Ok(None);
    }
    let pairing = spec.pose_pairing.as_ref().unwrap_or(&PosePairing::ByIndex);
    pair_pose_rows(spec, base_dir, pairing, tokens).map(Some)
}

/// Each camera's laser frame in each of the views `tokens`, by the
/// manifest's pairing; `None` when no camera declares `laser_images`. Lists
/// the laser image folders.
///
/// # Errors
///
/// A laser image pattern that does not expand ([`RunError::BadGlob`]), a
/// pairing regex that does not compile ([`RunError::BadPairingRegex`]).
pub(crate) fn list_laser_frames(
    spec: &DatasetSpec,
    base_dir: &Path,
    tokens: &[String],
) -> Result<Option<ViewLaserFrames>, RunError> {
    if spec
        .cameras
        .iter()
        .all(|camera| camera.laser_images.is_none())
    {
        return Ok(None);
    }
    let pairing = spec.pose_pairing.as_ref().unwrap_or(&PosePairing::ByIndex);
    pair_laser_frames(spec, base_dir, pairing, tokens).map(Some)
}

/// The pose rows of a manifest that declares robot poses, paired with the
/// views `tokens` by `pairing`.
fn pair_pose_rows(
    spec: &DatasetSpec,
    base_dir: &Path,
    pairing: &PosePairing,
    tokens: &[String],
) -> Result<ViewPoses, RunError> {
    validate_robot_pose_source(spec)?;
    // The check above refuses robot poses without a convention, and the
    // caller pairs poses only when the manifest declares them.
    let (Some(source), Some(convention)) = (&spec.robot_poses, &spec.pose_convention) else {
        return Err(ValidationError::MissingPoseConvention.into());
    };
    let poses = load_robot_poses(source, convention, base_dir)?;
    let rows: Vec<PoseRow> = poses
        .iter()
        .enumerate()
        .map(|(index, pose)| PoseRow {
            index,
            id: pose.id.clone(),
        })
        .collect();
    let views: Vec<Option<usize>> = match pairing {
        PosePairing::ByIndex => (0..tokens.len())
            .map(|view| (view < rows.len()).then_some(view))
            .collect(),
        PosePairing::SharedFilenameToken { .. } => {
            require_pose_id_column(spec)?;
            let by_id = poses_by_id(&poses)?;
            tokens
                .iter()
                .map(|token| by_id.get(token.as_str()).copied())
                .collect()
        }
        other => {
            return Err(RunError::UnsupportedOption {
                option: "pairing",
                value: format!("{other:?}"),
            });
        }
    };
    let mut paired = vec![false; rows.len()];
    for &index in views.iter().flatten() {
        paired[index] = true;
    }
    Ok(ViewPoses {
        views: views
            .into_iter()
            .map(|index| index.map(|i| rows[i].clone()))
            .collect(),
        unpaired: rows
            .into_iter()
            .zip(paired)
            .filter_map(|(row, paired)| (!paired).then_some(row))
            .collect(),
    })
}

/// Every camera's laser frames, paired with the views `tokens` by
/// `pairing`: each frame pairs with the view whose token it carries, the
/// first in sort order where two carry one.
fn pair_laser_frames(
    spec: &DatasetSpec,
    base_dir: &Path,
    pairing: &PosePairing,
    tokens: &[String],
) -> Result<ViewLaserFrames, RunError> {
    let token_regex = match pairing {
        PosePairing::ByIndex => None,
        PosePairing::SharedFilenameToken { regex, group } => {
            let re = regex::Regex::new(regex).map_err(|e| RunError::BadPairingRegex {
                regex: regex.clone(),
                message: e.to_string(),
            })?;
            Some((re, group.as_str()))
        }
        other => {
            return Err(RunError::UnsupportedOption {
                option: "pairing",
                value: format!("{other:?}"),
            });
        }
    };
    let view_of: HashMap<&str, usize> = tokens
        .iter()
        .enumerate()
        .map(|(view, token)| (token.as_str(), view))
        .collect();
    let mut paths = vec![vec![None; spec.cameras.len()]; tokens.len()];
    let mut unpaired = Vec::new();
    for (camera, source) in spec.cameras.iter().enumerate() {
        let Some(pattern) = &source.laser_images else {
            continue;
        };
        for (i, path) in expand_image_pattern(pattern, &source.id, base_dir)?
            .into_iter()
            .enumerate()
        {
            let token = match &token_regex {
                None => Some(i.to_string()),
                Some((re, group)) => filename_token(re, group, &path),
            };
            let free_view = token
                .as_deref()
                .and_then(|t| view_of.get(t).copied())
                .filter(|&view| paths[view][camera].is_none());
            match free_view {
                Some(view) => paths[view][camera] = Some(path),
                None => unpaired.push(UnpairedLaserFrame {
                    camera,
                    path,
                    token,
                }),
            }
        }
    }
    Ok(ViewLaserFrames { paths, unpaired })
}

/// The pairing token `re`'s group `group` captures from `path`'s file name.
fn filename_token(re: &regex::Regex, group: &str, path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy();
    re.captures(&name)
        .and_then(|c| c.name(group))
        .map(|m| m.as_str().to_string())
}

/// Align per-camera image lists into views using `pairing`.
///
/// `images[c]` is camera `c`'s expanded, sorted image list and
/// `camera_ids[c]` its manifest id (for error messages).
pub(crate) fn pair_views(
    images: &[Vec<PathBuf>],
    camera_ids: &[String],
    pairing: &PosePairing,
) -> Result<PairedViews, RunError> {
    match pairing {
        PosePairing::ByIndex => {
            let counts: Vec<(String, usize)> = camera_ids
                .iter()
                .cloned()
                .zip(images.iter().map(Vec::len))
                .collect();
            let first = counts[0].1;
            if counts.iter().any(|(_, n)| *n != first) {
                return Err(RunError::ViewCountMismatch { counts });
            }
            let paths = (0..first)
                .map(|view| images.iter().map(|cam| Some(cam[view].clone())).collect())
                .collect();
            let tokens = (0..first).map(|i| i.to_string()).collect();
            Ok(PairedViews { paths, tokens })
        }
        PosePairing::SharedFilenameToken { regex, group } => {
            let re = regex::Regex::new(regex).map_err(|e| RunError::BadPairingRegex {
                regex: regex.clone(),
                message: e.to_string(),
            })?;
            if !re.capture_names().flatten().any(|name| name == group) {
                return Err(RunError::BadPairingRegex {
                    regex: regex.clone(),
                    message: format!("regex has no named capture group {group:?}"),
                });
            }

            // token → path, per camera.
            let mut per_camera: Vec<HashMap<String, PathBuf>> = Vec::with_capacity(images.len());
            for (cam_images, cam_id) in images.iter().zip(camera_ids) {
                let mut map = HashMap::new();
                for path in cam_images {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let token = re
                        .captures(&name)
                        .and_then(|c| c.name(group))
                        .map(|m| m.as_str().to_string())
                        .ok_or_else(|| RunError::PairingTokenMismatch {
                            message: format!(
                                "camera {cam_id:?}: filename {name:?} does not match the \
                                 pairing regex (or group {group:?} did not participate)"
                            ),
                        })?;
                    if let Some(previous) = map.insert(token.clone(), path.clone()) {
                        return Err(RunError::PairingTokenMismatch {
                            message: format!(
                                "camera {cam_id:?}: token {token:?} extracted from two files \
                                 ({} and {})",
                                previous.display(),
                                path.display()
                            ),
                        });
                    }
                }
                per_camera.push(map);
            }

            // Views = sorted union of tokens; cameras missing a token
            // contribute `None` for that view. (Intersection would
            // silently drop views where one camera blinked — the rig
            // IR supports per-camera gaps, so keep them.)
            let mut all_tokens: BTreeSet<String> = BTreeSet::new();
            for map in &per_camera {
                all_tokens.extend(map.keys().cloned());
            }
            if all_tokens.is_empty() {
                return Err(RunError::PairingTokenMismatch {
                    message: "no view tokens extracted from any camera".into(),
                });
            }

            let tokens: Vec<String> = all_tokens.into_iter().collect();
            let paths = tokens
                .iter()
                .map(|t| per_camera.iter().map(|map| map.get(t).cloned()).collect())
                .collect();
            Ok(PairedViews { paths, tokens })
        }
        other => Err(RunError::UnsupportedOption {
            option: "pairing",
            value: format!("{other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn paths(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(PathBuf::from).collect()
    }

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("cam{i}")).collect()
    }

    #[test]
    fn by_index_pairs_in_order() {
        let images = vec![
            paths(&["a/1.png", "a/2.png"]),
            paths(&["b/1.png", "b/2.png"]),
        ];
        let paired = pair_views(&images, &ids(2), &PosePairing::ByIndex).unwrap();
        assert_eq!(paired.paths.len(), 2);
        assert_eq!(paired.paths[1][0].as_deref(), Some(Path::new("a/2.png")));
        assert_eq!(paired.tokens, vec!["0", "1"]);
    }

    #[test]
    fn by_index_rejects_unequal_counts() {
        let images = vec![paths(&["a/1.png", "a/2.png"]), paths(&["b/1.png"])];
        let err = pair_views(&images, &ids(2), &PosePairing::ByIndex).unwrap_err();
        match err {
            RunError::ViewCountMismatch { counts } => {
                assert_eq!(counts, vec![("cam0".into(), 2), ("cam1".into(), 1)]);
            }
            other => panic!("expected ViewCountMismatch, got {other:?}"),
        }
    }

    fn token_pairing() -> PosePairing {
        PosePairing::SharedFilenameToken {
            regex: r"^Cam\d+_(?<view>\d+)\.png$".into(),
            group: "view".into(),
        }
    }

    #[test]
    fn token_pairing_aligns_and_fills_gaps() {
        let images = vec![
            paths(&["x/Cam0_001.png", "x/Cam0_002.png", "x/Cam0_003.png"]),
            paths(&["y/Cam1_001.png", "y/Cam1_003.png"]), // 002 missing
        ];
        let paired = pair_views(&images, &ids(2), &token_pairing()).unwrap();
        assert_eq!(paired.tokens, vec!["001", "002", "003"]);
        assert!(paired.paths[1][1].is_none(), "cam1 has no 002 frame");
        assert!(paired.paths[0][1].is_some());
        assert_eq!(
            paired.paths[2][1].as_deref().unwrap().to_str().unwrap(),
            "y/Cam1_003.png"
        );
    }

    #[test]
    fn token_regex_compile_error_is_actionable() {
        let pairing = PosePairing::SharedFilenameToken {
            regex: "([unclosed".into(),
            group: "view".into(),
        };
        let images = vec![paths(&["a/1.png"])];
        let err = pair_views(&images, &ids(1), &pairing).unwrap_err();
        assert!(matches!(err, RunError::BadPairingRegex { .. }));
    }

    #[test]
    fn missing_group_is_actionable() {
        let pairing = PosePairing::SharedFilenameToken {
            regex: r"^Cam\d+_(\d+)\.png$".into(), // unnamed group
            group: "view".into(),
        };
        let images = vec![paths(&["a/Cam0_001.png"])];
        let err = pair_views(&images, &ids(1), &pairing).unwrap_err();
        match err {
            RunError::BadPairingRegex { message, .. } => {
                assert!(message.contains("view"), "got: {message}")
            }
            other => panic!("expected BadPairingRegex, got {other:?}"),
        }
    }

    /// A two-camera rig hand-eye dataset in `dir` paired by the token of
    /// `img_<token>.png`: cam0 sees views 1, 2, 3 and cam1 views 1, 3.
    /// `pose_ids` are the rows of `poses.csv`; `laser` writes laser frames
    /// `laser_<token>.png` of cam0 for those tokens.
    fn token_handeye(dir: &Path, pose_ids: &[&str], laser: Option<&[&str]>) -> DatasetSpec {
        for (cam, views) in [(0, vec![1, 2, 3]), (1, vec![1, 3])] {
            std::fs::create_dir_all(dir.join(format!("cam{cam}"))).unwrap();
            for view in views {
                std::fs::write(dir.join(format!("cam{cam}/img_{view}.png")), b"-").unwrap();
            }
        }
        let mut csv = String::from("id,x,y,z,qx,qy,qz,qw\n");
        for id in pose_ids {
            csv.push_str(&format!("{id},0,0,0,0,0,0,1\n"));
        }
        std::fs::write(dir.join("poses.csv"), csv).unwrap();
        let mut cam0 = serde_json::json!({
            "id": "cam0", "images": { "kind": "glob", "pattern": "cam0/img_*.png" }
        });
        if let Some(tokens) = laser {
            std::fs::create_dir_all(dir.join("laser")).unwrap();
            for token in tokens {
                std::fs::write(dir.join(format!("laser/laser_{token}.png")), b"-").unwrap();
            }
            cam0["laser_images"] = serde_json::json!({ "kind": "glob", "pattern": "laser/*.png" });
        }
        serde_json::from_value(serde_json::json!({
            "version": 1,
            "topology": "rig_handeye",
            "cameras": [
                cam0,
                { "id": "cam1", "images": { "kind": "glob", "pattern": "cam1/img_*.png" } },
            ],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
            "robot_poses": {
                "path": "poses.csv",
                "format": "csv",
                "columns": {
                    "pose_id": "id", "tx": "x", "ty": "y", "tz": "z",
                    "rotation": ["qx", "qy", "qz", "qw"]
                }
            },
            "pose_convention": {
                "transform": "t_base_tcp", "rotation_format": "quat_xyzw", "translation_units": "m"
            },
            "pose_pairing": {
                "kind": "shared_filename_token", "regex": "_(?P<id>\\d+)\\.png$", "group": "id"
            },
        }))
        .unwrap()
    }

    fn row(index: usize, id: &str) -> PoseRow {
        PoseRow {
            index,
            id: Some(id.into()),
        }
    }

    /// The views of `spec` and their robot poses.
    fn poses(spec: &DatasetSpec, dir: &Path) -> Result<Option<ViewPoses>, RunError> {
        let views = pair_dataset_views(spec, dir).unwrap();
        list_view_poses(spec, dir, &views.tokens)
    }

    /// The views of `spec` and their laser frames.
    fn lasers(spec: &DatasetSpec, dir: &Path) -> ViewLaserFrames {
        let views = pair_dataset_views(spec, dir).unwrap();
        list_laser_frames(spec, dir, &views.tokens)
            .unwrap()
            .expect("the manifest declares laser frames")
    }

    #[test]
    fn views_pair_with_their_robot_pose_and_the_rows_no_view_pairs_with_are_listed() {
        let tmp = tempfile::tempdir().unwrap();
        // View 2 has no row; row "7" has no view.
        let spec = token_handeye(tmp.path(), &["3", "7", "1"], None);
        let views = pair_dataset_views(&spec, tmp.path()).unwrap();
        assert_eq!(views.tokens, ["1", "2", "3"]);
        let listed = poses(&spec, tmp.path()).unwrap().unwrap();
        assert_eq!(listed.views, [Some(row(2, "1")), None, Some(row(0, "3"))]);
        assert_eq!(listed.unpaired, [row(1, "7")]);
        assert_eq!(
            list_laser_frames(&spec, tmp.path(), &views.tokens).unwrap(),
            None,
            "no camera declares laser images"
        );
        // A manifest without robot poses has none to pair.
        let mut planar = spec.clone();
        planar.robot_poses = None;
        assert_eq!(poses(&planar, tmp.path()).unwrap(), None);
    }

    #[test]
    fn by_index_pairs_pose_rows_by_position_and_lists_the_extra_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = token_handeye(tmp.path(), &["a", "b", "c", "d"], None);
        // cam1 gets the third image by_index needs.
        std::fs::write(tmp.path().join("cam1/img_2.png"), b"-").unwrap();
        spec.pose_pairing = Some(PosePairing::ByIndex);
        let listed = poses(&spec, tmp.path()).unwrap().unwrap();
        assert_eq!(
            listed.views,
            [Some(row(0, "a")), Some(row(1, "b")), Some(row(2, "c"))]
        );
        assert_eq!(listed.unpaired, [row(3, "d")]);
        // Fewer rows than views: the last views have none.
        std::fs::write(
            tmp.path().join("poses.csv"),
            "id,x,y,z,qx,qy,qz,qw\na,0,0,0,0,0,0,1\n",
        )
        .unwrap();
        let listed = poses(&spec, tmp.path()).unwrap().unwrap();
        assert_eq!(listed.views, [Some(row(0, "a")), None, None]);
        assert!(listed.unpaired.is_empty());
    }

    #[test]
    fn a_pose_file_that_cannot_be_listed_is_the_run_s_error() {
        let tmp = tempfile::tempdir().unwrap();
        // A repeated pose_id: the run's own error.
        let spec = token_handeye(tmp.path(), &["1", "1"], None);
        let err = poses(&spec, tmp.path()).unwrap_err();
        assert!(
            matches!(err, RunError::PairingTokenMismatch { ref message } if message.contains("duplicate pose_id")),
            "{err:?}"
        );
        // No pose_id column under token pairing: the run's question.
        let mut spec = token_handeye(tmp.path(), &["1"], None);
        spec.robot_poses
            .as_mut()
            .unwrap()
            .columns
            .as_mut()
            .unwrap()
            .pose_id = None;
        assert!(matches!(
            poses(&spec, tmp.path()).unwrap_err(),
            RunError::AskUser { ref field, .. } if field == "robot_poses.columns.pose_id"
        ));
        // A missing pose file, and a missing convention.
        let mut spec = token_handeye(tmp.path(), &["1"], None);
        std::fs::remove_file(tmp.path().join("poses.csv")).unwrap();
        assert!(matches!(
            poses(&spec, tmp.path()).unwrap_err(),
            RunError::Io(_)
        ));
        spec.pose_convention = None;
        assert!(matches!(
            poses(&spec, tmp.path()).unwrap_err(),
            RunError::Validation(ValidationError::MissingPoseConvention)
        ));
    }

    #[test]
    fn laser_frames_pair_by_token_and_the_unpaired_ones_are_listed() {
        let tmp = tempfile::tempdir().unwrap();
        // Frame 9 has no view; view 2 has no frame.
        let spec = token_handeye(tmp.path(), &["1", "2", "3"], Some(&["1", "3", "9"]));
        std::fs::write(tmp.path().join("laser/notes.png"), b"-").unwrap();
        let frames = lasers(&spec, tmp.path());
        let laser = |name: &str| Some(tmp.path().join("laser").join(name));
        assert_eq!(
            frames.paths,
            [
                vec![laser("laser_1.png"), None],
                vec![None, None],
                vec![laser("laser_3.png"), None],
            ]
        );
        assert_eq!(
            frames.unpaired,
            [
                UnpairedLaserFrame {
                    camera: 0,
                    path: tmp.path().join("laser/laser_9.png"),
                    token: Some("9".into()),
                },
                UnpairedLaserFrame {
                    camera: 0,
                    path: tmp.path().join("laser/notes.png"),
                    token: None,
                },
            ]
        );
    }

    #[test]
    fn by_index_pairs_laser_frames_by_position() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = token_handeye(tmp.path(), &["1", "2", "3"], Some(&["1", "2", "3", "4"]));
        std::fs::write(tmp.path().join("cam1/img_2.png"), b"-").unwrap();
        spec.pose_pairing = Some(PosePairing::ByIndex);
        let frames = lasers(&spec, tmp.path());
        assert_eq!(frames.paths.len(), 3);
        assert_eq!(
            frames.paths[2][0].as_deref(),
            Some(tmp.path().join("laser/laser_3.png").as_path())
        );
        assert_eq!(frames.unpaired.len(), 1);
        assert_eq!(frames.unpaired[0].token.as_deref(), Some("3"));
    }

    #[test]
    fn unmatched_filename_is_actionable() {
        let images = vec![paths(&["a/notes.txt"])];
        let err = pair_views(&images, &ids(1), &token_pairing()).unwrap_err();
        match err {
            RunError::PairingTokenMismatch { message } => {
                assert!(message.contains("notes.txt"), "got: {message}")
            }
            other => panic!("expected PairingTokenMismatch, got {other:?}"),
        }
    }
}
