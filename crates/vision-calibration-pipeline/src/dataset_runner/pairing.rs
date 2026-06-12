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
use std::path::PathBuf;

use vision_calibration_dataset::PosePairing;

use super::RunError;

/// Per-camera image lists aligned into rig views.
#[derive(Debug)]
pub struct PairedViews {
    /// `paths[view][camera]` — `None` when a camera has no image for
    /// this view (only possible under token pairing).
    pub paths: Vec<Vec<Option<PathBuf>>>,
    /// View token per view (token pairing) or stringified index
    /// (`by_index`). Used to pair robot poses by `pose_id`.
    pub tokens: Vec<String>,
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
