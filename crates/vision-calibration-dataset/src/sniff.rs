//! Heuristic dataset folder → [`DatasetSpec`] skeleton (ADR 0016 / 0019).
//!
//! [`sniff_folder`] walks a dataset directory and infers **only what is
//! structurally unambiguous**: camera directories, image globs, robot-pose
//! file presence and format, and `by_index` pairing. Everything that needs
//! domain knowledge — board geometry, target kind, sensor mode, frame
//! convention, and the topology when it is genuinely ambiguous — is left at
//! a best-guess placeholder and its dotted field path is recorded in
//! [`DatasetSpec::unresolved`]. Per ADR 0019 the runner refuses to proceed
//! until that list is cleared, so placeholders are never trusted
//! numerically; they exist only so the manifest deserializes and the
//! schema-driven form has something to edit.
//!
//! This is heuristic-only (no LLM, no documentation scraping); see the B3d
//! entry in `docs/ROADMAP.md`. The companion `generate-manifest` binary
//! (the `cli` feature) and the app's `sniff_folder` Tauri command both call
//! this function so all three paths share one inference.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::spec::{
    CameraSource, DatasetSpec, ImagePattern, PoseConvention, PosePairing, RobotPoseFormat,
    RobotPoseSource, RotationFormat, TargetSpec, Topology, TransformConvention, TranslationUnits,
};

/// Errors raised while sniffing a dataset folder.
#[derive(Debug, thiserror::Error)]
pub enum SniffError {
    /// The root path is not a readable directory.
    #[error("{0:?} is not a readable directory")]
    NotADirectory(PathBuf),

    /// No image files were found anywhere under the root, so there is no
    /// camera to describe.
    #[error("no image files (png/jpg/jpeg/tif/tiff/bmp) found under {0:?}")]
    NoCameras(PathBuf),

    /// A filesystem read failed.
    #[error("reading {path:?}: {source}")]
    Io {
        /// The path being read when the error occurred.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },
}

/// Image file extensions recognised as camera frames (case-insensitive).
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "tif", "tiff", "bmp"];

/// Inspect a dataset folder and emit a best-effort [`DatasetSpec`] skeleton.
///
/// The returned spec always carries a non-empty
/// [`unresolved`](DatasetSpec::unresolved) list unless every field could be
/// inferred (which, for board geometry, never happens in v0). Callers are
/// expected to surface the unresolved paths to the user and block execution
/// until they are filled and cleared (ADR 0019).
pub fn sniff_folder(root: &Path) -> Result<DatasetSpec, SniffError> {
    if !root.is_dir() {
        return Err(SniffError::NotADirectory(root.to_path_buf()));
    }

    let mut images: Vec<ImageFile> = Vec::new();
    let mut others: Vec<PathBuf> = Vec::new();
    walk(root, root, &mut images, &mut others)?;

    let cameras = group_cameras(&images);
    if cameras.is_empty() {
        return Err(SniffError::NoCameras(root.to_path_buf()));
    }

    let mut unresolved: Vec<String> = Vec::new();

    // Robot poses — inferred only from a recognisable pose file.
    let pose = detect_pose_file(root, &others);

    // Topology: choose the most likely problem type from camera count and
    // pose presence; flag `topology` unresolved when the choice is genuinely
    // ambiguous (single-cam intrinsics can be pinhole or Scheimpflug; N
    // cameras without poses can be a rig or N independent mono calibrations).
    let n = cameras.len();
    let has_poses = pose.is_some();
    let (topology, topology_ambiguous) = match (n, has_poses) {
        (1, true) => (Topology::SingleCamHandeye, false),
        (1, false) => (Topology::PlanarIntrinsics, true),
        (_, true) => (Topology::RigHandeye, false),
        (_, false) => (Topology::RigExtrinsics, true),
    };

    // Build camera sources.
    let camera_sources: Vec<CameraSource> = cameras
        .iter()
        .map(|c| CameraSource {
            id: c.id.clone(),
            images: ImagePattern::Glob {
                pattern: c.glob.clone(),
            },
            roi_xywh: None,
            laser_images: None,
        })
        .collect();

    // Pose pairing — only meaningful for multi-camera or hand-eye datasets.
    let pairing_relevant = n > 1 || has_poses;
    let pose_pairing = if pairing_relevant {
        let first = cameras[0].count;
        let consistent = cameras.iter().all(|c| c.count == first);
        let poses_match = pose.as_ref().is_none_or(|p| p.count == first);
        if consistent && poses_match {
            Some(PosePairing::ByIndex)
        } else {
            unresolved.push("pose_pairing".to_string());
            None
        }
    } else {
        None
    };

    // Assemble pose source + convention.
    let (robot_poses, pose_convention) = match pose {
        None => (None, None),
        Some(p) => {
            let source = RobotPoseSource {
                path: p.rel,
                format: p.format,
                columns: None,
                matrix_field: None,
            };
            match p.format {
                RobotPoseFormat::Rowmajor4x4 => {
                    // `rowmajor4x4` fixes the rotation format; the transform
                    // direction and translation units are not inferable from
                    // the numbers, so pre-fill the KUKA/UR defaults but flag
                    // them for confirmation.
                    unresolved.push("pose_convention.transform".to_string());
                    unresolved.push("pose_convention.translation_units".to_string());
                    let conv = PoseConvention {
                        transform: TransformConvention::TBaseTcp,
                        rotation_format: RotationFormat::Matrix4x4RowMajor,
                        translation_units: TranslationUnits::M,
                    };
                    (Some(source), Some(conv))
                }
                RobotPoseFormat::Csv | RobotPoseFormat::Json | RobotPoseFormat::Jsonl => {
                    // Tabular formats need a column mapping and a full
                    // convention the sniffer cannot read off the structure.
                    unresolved.push("robot_poses.columns".to_string());
                    unresolved.push("pose_convention".to_string());
                    (Some(source), None)
                }
            }
        }
    };

    // Target geometry is never inferable from images alone in v0.
    let target = TargetSpec::Chessboard {
        rows: 0,
        cols: 0,
        square_size_m: 0.0,
    };
    unresolved.push("target".to_string());

    if topology_ambiguous {
        unresolved.push("topology".to_string());
    }

    Ok(DatasetSpec {
        version: 1,
        cameras: camera_sources,
        target,
        detector: None,
        robot_poses,
        laser: None,
        upstream_calibration: None,
        topology,
        pose_pairing,
        pose_convention,
        unresolved,
        description: Some(
            "auto-generated by sniff_folder — review and clear _unresolved before running"
                .to_string(),
        ),
    })
}

/// One discovered image file, with its path relative to the sniff root.
struct ImageFile {
    rel: PathBuf,
    ext: String,
}

/// A camera candidate: all images sharing one parent directory.
struct CameraGroup {
    id: String,
    glob: String,
    count: usize,
}

/// Recursively collect image files (into `images`) and every other regular
/// file (into `others`), skipping dotfiles and dot-directories. Paths are
/// stored relative to `root`.
fn walk(
    dir: &Path,
    root: &Path,
    images: &mut Vec<ImageFile>,
    others: &mut Vec<PathBuf>,
) -> Result<(), SniffError> {
    let entries = std::fs::read_dir(dir).map_err(|source| SniffError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| SniffError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue; // skip .DS_Store, .git, hidden dirs, etc.
        }
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| SniffError::Io {
            path: path.clone(),
            source,
        })?;
        if file_type.is_dir() {
            walk(&path, root, images, others)?;
        } else if file_type.is_file() {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            match path.extension().and_then(|e| e.to_str()) {
                Some(ext) if is_image_ext(ext) => images.push(ImageFile {
                    rel,
                    ext: ext.to_ascii_lowercase(),
                }),
                _ => others.push(rel),
            }
        }
    }
    Ok(())
}

fn is_image_ext(ext: &str) -> bool {
    let lower = ext.to_ascii_lowercase();
    IMAGE_EXTS.contains(&lower.as_str())
}

/// Group images by parent directory into camera candidates. Single-group
/// datasets get the id `cam0`; multi-group datasets use the leaf directory
/// name (deduplicated). Groups are ordered by directory path for
/// determinism.
fn group_cameras(images: &[ImageFile]) -> Vec<CameraGroup> {
    let mut by_dir: BTreeMap<PathBuf, Vec<&ImageFile>> = BTreeMap::new();
    for img in images {
        let parent = img.rel.parent().map(Path::to_path_buf).unwrap_or_default();
        by_dir.entry(parent).or_default().push(img);
    }

    let single = by_dir.len() == 1;
    let mut used_ids: Vec<String> = Vec::new();
    let mut groups = Vec::with_capacity(by_dir.len());
    for (dir, imgs) in by_dir {
        let ext = dominant_ext(&imgs);
        let glob = glob_for(&dir, &ext);
        // Count only the images the emitted `*.{ext}` glob will actually
        // expand to, not every image in the directory. Otherwise a mixed-
        // extension group (e.g. 3 PNG + 1 JPG) would report 4 and could be
        // paired `by_index` against a 4-long stream the glob only fills with
        // 3 — a count mismatch the runner hits later.
        let count = imgs.iter().filter(|img| img.ext == ext).count();
        let id = if single {
            "cam0".to_string()
        } else {
            unique_id(leaf_name(&dir), &mut used_ids)
        };
        groups.push(CameraGroup { id, glob, count });
    }
    groups
}

/// Most common extension in a group (ties broken alphabetically for
/// determinism).
fn dominant_ext(imgs: &[&ImageFile]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for img in imgs {
        *counts.entry(img.ext.as_str()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(a.0)))
        .map(|(ext, _)| ext.to_string())
        .unwrap_or_else(|| "png".to_string())
}

/// Build a forward-slash glob (`dir/*.ext`) for a camera directory.
fn glob_for(dir: &Path, ext: &str) -> String {
    let dir_str = dir
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/");
    if dir_str.is_empty() {
        format!("*.{ext}")
    } else {
        format!("{dir_str}/*.{ext}")
    }
}

fn leaf_name(dir: &Path) -> String {
    dir.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "cam".to_string())
}

fn unique_id(base: String, used: &mut Vec<String>) -> String {
    let mut candidate = base.clone();
    let mut i = 1;
    while used.contains(&candidate) {
        candidate = format!("{base}_{i}");
        i += 1;
    }
    used.push(candidate.clone());
    candidate
}

/// A detected robot-pose file.
struct PoseFile {
    rel: PathBuf,
    format: RobotPoseFormat,
    count: usize,
}

/// Find and classify a robot-pose file among the non-image files. A file is
/// a pose candidate only if its name contains `pose` (case-insensitive) —
/// this deliberately excludes `*.json` exports, `dataset.toml` manifests,
/// and other numeric sidecars. Candidates are tried in lexicographic order;
/// any that is not readable UTF-8 text (e.g. a binary `pose_cache.npy`) is
/// skipped in favor of the next, since the name match is only heuristic.
fn detect_pose_file(root: &Path, others: &[PathBuf]) -> Option<PoseFile> {
    let mut candidates: Vec<&PathBuf> = others
        .iter()
        .filter(|rel| {
            rel.file_name()
                .map(|n| n.to_string_lossy().to_ascii_lowercase().contains("pose"))
                .unwrap_or(false)
        })
        .collect();
    candidates.sort();

    for rel in candidates {
        // Skip binary / unreadable sidecars rather than aborting the sniff.
        let Ok(contents) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        let nonempty: Vec<&str> = contents.lines().filter(|l| !l.trim().is_empty()).collect();
        let ext = rel
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        let format = match ext.as_str() {
            "csv" => RobotPoseFormat::Csv,
            "json" => RobotPoseFormat::Json,
            "jsonl" => RobotPoseFormat::Jsonl,
            // Headerless text: a 16-float first line is a row-major 4x4 export.
            _ if is_rowmajor4x4_line(nonempty.first().copied().unwrap_or("")) => {
                RobotPoseFormat::Rowmajor4x4
            }
            _ => RobotPoseFormat::Csv,
        };

        // The pose count drives by-index pairing. For rowmajor4x4 each
        // non-empty line is one pose; for CSV the header row is not a pose.
        let count = match format {
            RobotPoseFormat::Rowmajor4x4 => nonempty.len(),
            RobotPoseFormat::Csv => nonempty.len().saturating_sub(1),
            RobotPoseFormat::Json | RobotPoseFormat::Jsonl => nonempty.len(),
        };

        return Some(PoseFile {
            rel: rel.clone(),
            format,
            count,
        });
    }

    None
}

/// Is this line 16 whitespace-separated floats (a flattened row-major 4x4)?
fn is_rowmajor4x4_line(line: &str) -> bool {
    let fields: Vec<&str> = line.split_whitespace().collect();
    fields.len() == 16 && fields.iter().all(|f| f.parse::<f64>().is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Minimal self-cleaning temp directory (avoids a `tempfile` dep).
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("vc_sniff_{}_{}_{}", std::process::id(), tag, n));
            std::fs::create_dir_all(&path).unwrap();
            TempDir { path }
        }

        fn touch(&self, rel: &str) {
            let p = self.path.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"").unwrap();
        }

        fn write(&self, rel: &str, contents: &str) {
            let p = self.path.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, contents).unwrap();
        }

        fn write_bytes(&self, rel: &str, bytes: &[u8]) {
            let p = self.path.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, bytes).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn cam_glob<'a>(spec: &'a DatasetSpec, id: &str) -> &'a str {
        let cam = spec.cameras.iter().find(|c| c.id == id).unwrap();
        match &cam.images {
            ImagePattern::Glob { pattern } => pattern.as_str(),
            ImagePattern::List { .. } => panic!("expected glob"),
        }
    }

    #[test]
    fn single_camera_with_rowmajor_poses() {
        let dir = TempDir::new("kuka");
        for i in 1..=3 {
            dir.touch(&format!("{i:02}.png"));
        }
        let row = "1 0 0 1.2\t0 1 0 -0.8\t0 0 1 0.5\t0 0 0 1\n";
        dir.write("RobotPosesVec.txt", &row.repeat(3));

        let spec = sniff_folder(&dir.path).unwrap();

        assert_eq!(spec.cameras.len(), 1);
        assert_eq!(spec.cameras[0].id, "cam0");
        assert_eq!(cam_glob(&spec, "cam0"), "*.png");
        assert_eq!(spec.topology, Topology::SingleCamHandeye);
        assert!(matches!(spec.pose_pairing, Some(PosePairing::ByIndex)));

        let poses = spec.robot_poses.as_ref().expect("poses detected");
        assert_eq!(poses.format, RobotPoseFormat::Rowmajor4x4);
        assert_eq!(poses.path, PathBuf::from("RobotPosesVec.txt"));

        let conv = spec.pose_convention.as_ref().unwrap();
        assert_eq!(conv.rotation_format, RotationFormat::Matrix4x4RowMajor);

        // Topology is unambiguous (1 cam + poses); geometry + convention
        // direction/units are not.
        assert!(!spec.unresolved.contains(&"topology".to_string()));
        assert!(spec.unresolved.contains(&"target".to_string()));
        assert!(
            spec.unresolved
                .contains(&"pose_convention.transform".to_string())
        );
        assert!(
            spec.unresolved
                .contains(&"pose_convention.translation_units".to_string())
        );
    }

    #[test]
    fn two_cameras_no_poses_flags_topology() {
        let dir = TempDir::new("stereo");
        for i in 1..=4 {
            dir.touch(&format!("imgs/leftcamera/Im_L_{i}.png"));
            dir.touch(&format!("imgs/rightcamera/Im_R_{i}.png"));
        }
        // Decoy files that must NOT be read as poses.
        dir.write("viewer_export.json", "{}");
        dir.write("dataset_rig.toml", "version = 1\n");

        let spec = sniff_folder(&dir.path).unwrap();

        assert_eq!(spec.cameras.len(), 2);
        let ids: Vec<&str> = spec.cameras.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, vec!["leftcamera", "rightcamera"]);
        assert_eq!(cam_glob(&spec, "leftcamera"), "imgs/leftcamera/*.png");
        assert_eq!(cam_glob(&spec, "rightcamera"), "imgs/rightcamera/*.png");

        assert_eq!(spec.topology, Topology::RigExtrinsics);
        assert!(spec.robot_poses.is_none());
        assert!(matches!(spec.pose_pairing, Some(PosePairing::ByIndex)));
        assert!(spec.unresolved.contains(&"topology".to_string()));
        assert!(spec.unresolved.contains(&"target".to_string()));
    }

    #[test]
    fn single_camera_no_poses_is_planar_and_ambiguous() {
        let dir = TempDir::new("planar");
        for i in 0..5 {
            dir.touch(&format!("frame_{i}.jpg"));
        }
        let spec = sniff_folder(&dir.path).unwrap();

        assert_eq!(spec.cameras.len(), 1);
        assert_eq!(cam_glob(&spec, "cam0"), "*.jpg");
        assert_eq!(spec.topology, Topology::PlanarIntrinsics);
        // Pinhole vs Scheimpflug is undecidable from images.
        assert!(spec.unresolved.contains(&"topology".to_string()));
        // No pose pairing is relevant for single-cam intrinsics.
        assert!(spec.pose_pairing.is_none());
    }

    #[test]
    fn mixed_extension_counts_by_dominant_glob() {
        // camA mixes 3 PNG + 1 stray JPG; the glob is `*.png` so the pairing
        // count must be 3 (PNG only), matching camB's 3 — otherwise the count
        // would be 4 and pairing would (wrongly) be flagged inconsistent.
        let dir = TempDir::new("mixedext");
        for i in 0..3 {
            dir.touch(&format!("a/{i}.png"));
        }
        dir.touch("a/stray.jpg");
        for i in 0..3 {
            dir.touch(&format!("b/{i}.png"));
        }

        let spec = sniff_folder(&dir.path).unwrap();
        assert_eq!(cam_glob(&spec, "a"), "a/*.png");
        assert!(matches!(spec.pose_pairing, Some(PosePairing::ByIndex)));
        assert!(!spec.unresolved.contains(&"pose_pairing".to_string()));
    }

    #[test]
    fn binary_pose_sidecar_is_skipped() {
        // A non-UTF-8 sidecar whose name contains "pose" sorts before the real
        // pose file; it must be skipped, not abort the whole sniff.
        let dir = TempDir::new("binpose");
        dir.touch("img.png");
        dir.write_bytes("camera_pose.npy", &[0x93, b'N', 0xff, 0xfe, 0x00, 0x01]);
        let row = "1 0 0 0.5\t0 1 0 0\t0 0 1 0\t0 0 0 1\n";
        dir.write("robot_poses.txt", row);

        let spec = sniff_folder(&dir.path).unwrap();
        let poses = spec.robot_poses.as_ref().expect("real pose file used");
        assert_eq!(poses.path, PathBuf::from("robot_poses.txt"));
        assert_eq!(poses.format, RobotPoseFormat::Rowmajor4x4);
    }

    #[test]
    fn empty_folder_errors() {
        let dir = TempDir::new("empty");
        assert!(matches!(
            sniff_folder(&dir.path),
            Err(SniffError::NoCameras(_))
        ));
    }

    #[test]
    fn missing_folder_errors() {
        let missing = std::env::temp_dir().join("vc_sniff_does_not_exist_xyz");
        assert!(matches!(
            sniff_folder(&missing),
            Err(SniffError::NotADirectory(_))
        ));
    }

    #[test]
    fn sniffed_spec_serializes_and_round_trips() {
        let dir = TempDir::new("roundtrip");
        dir.touch("a.png");
        let spec = sniff_folder(&dir.path).unwrap();
        let json = serde_json::to_value(&spec).unwrap();
        // `_unresolved` is the serialized name of `unresolved`.
        assert!(json.get("_unresolved").is_some());
        let back: DatasetSpec = serde_json::from_value(json).unwrap();
        assert_eq!(back.cameras.len(), 1);
    }

    // ── Acceptance against committed fixtures ───────────────────────────────
    //
    // Skipped when the fixture's images aren't present: `data/kuka_1` ships
    // its 30 PNGs, but `data/stereo` only commits its `.toml` sidecars (the
    // image tree is gitignored), so its sniff yields `NoCameras` in CI.

    /// The fixture's `DatasetSpec`, or `None` when its directory or images
    /// aren't present in this checkout.
    fn sniff_repo_data(rel: &str) -> Option<DatasetSpec> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(rel);
        if !root.is_dir() {
            return None;
        }
        match sniff_folder(&root) {
            Ok(spec) => Some(spec),
            Err(SniffError::NoCameras(_)) => None, // images not committed (CI)
            Err(e) => panic!("unexpected sniff error for {rel}: {e}"),
        }
    }

    #[test]
    fn acceptance_kuka_1() {
        let Some(spec) = sniff_repo_data("kuka_1") else {
            return;
        };
        assert_eq!(spec.cameras.len(), 1);
        assert_eq!(cam_glob(&spec, "cam0"), "*.png");
        assert_eq!(spec.topology, Topology::SingleCamHandeye);
        let poses = spec.robot_poses.as_ref().unwrap();
        assert_eq!(poses.format, RobotPoseFormat::Rowmajor4x4);
        assert_eq!(poses.path, PathBuf::from("RobotPosesVec.txt"));
        assert!(spec.unresolved.contains(&"target".to_string()));
    }

    #[test]
    fn acceptance_stereo() {
        let Some(spec) = sniff_repo_data("stereo") else {
            return;
        };
        assert_eq!(spec.cameras.len(), 2);
        assert_eq!(cam_glob(&spec, "leftcamera"), "imgs/leftcamera/*.png");
        assert_eq!(cam_glob(&spec, "rightcamera"), "imgs/rightcamera/*.png");
        assert_eq!(spec.topology, Topology::RigExtrinsics);
        assert!(spec.robot_poses.is_none());
        assert!(spec.unresolved.contains(&"topology".to_string()));
    }
}
