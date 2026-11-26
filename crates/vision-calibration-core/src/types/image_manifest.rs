//! Per-export image manifest for the diagnose UI.
//!
//! `ImageManifest` is an optional addition to selected `*Export` types that
//! lets a downstream viewer (the Tauri/React diagnose UI; ad-hoc
//! viewers; CI screenshot tools) locate the source image for each
//! `(view, camera)` slot referenced by [`PerFeatureResiduals`].
//!
//! The manifest is a viewer-side contract: the calibration pipeline does not
//! consume it. Existing exports remain valid by absence (the field is
//! `Option<…>` on the host export and `skip_serializing_if = "Option::is_none"`
//! so JSON byte-stability is preserved).
//!
//! # Indexing
//!
//! Frames are view-major and align with the `(view, camera)` indexing used by
//! [`TargetFeatureResidual`]/[`LaserFeatureResidual`]. Lookup by linear scan
//! is fine at the dataset sizes the diagnose UI targets (≤ a few hundred
//! frames); no map index is materialised.
//!
//! # Tiled multi-camera frames
//!
//! Industrial rigs sometimes emit a single physical image that horizontally
//! tiles N cameras (for example six 720×540 tiles in one 4320×540 strip).
//! [`FrameRef::roi`] expresses this without inventing a
//! tiled-image format: multiple `FrameRef`s point at the same `path` with
//! disjoint ROIs.
//!
//! # Coordinate convention
//!
//! `observed_px` and `projected_px` in any [`PerFeatureResiduals`] entry
//! that refers to a frame are in **the camera's own pixel frame** — i.e.
//! the frame the camera intrinsics were calibrated against. With
//! `roi = None` that is the full source image; with `roi = Some(_)` it is
//! the ROI-local frame, with origin at the ROI's top-left corner.
//!
//! ROI is therefore a **render-time crop hint**, never a coordinate
//! transform: a viewer that renders the per-camera tile by blitting the
//! ROI rectangle onto a fresh canvas at `(0, 0)` can plot residual
//! coordinates directly, without subtracting `roi.x` / `roi.y`. Doing so
//! would double-correct and push residuals off-canvas.
//!
//! [`PerFeatureResiduals`]: crate::PerFeatureResiduals
//! [`TargetFeatureResidual`]: crate::TargetFeatureResidual
//! [`LaserFeatureResidual`]: crate::LaserFeatureResidual

use serde::{Deserialize, Serialize};
use std::ffi::{OsStr, OsString};
use std::path::{Component, Path, PathBuf};

/// Image manifest carried alongside a calibration export.
///
/// `root` is interpreted relative to the directory of the `export.json` that
/// embeds the manifest. Each [`FrameRef::path`] is interpreted relative to
/// `root`. Both layers of indirection make exports portable across machines
/// as long as the image directory is co-located with the export.
///
/// The dataset runner fills the manifest before the export has a file: its
/// `root` is `.`, the dataset folder its manifest's paths resolve against,
/// and each frame's path is relative to that folder (absolute for an image
/// outside it). A writer that puts the export in another folder restates
/// the root with [`rebased`](Self::rebased).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ImageManifest {
    /// Image root directory, relative to the export file's directory.
    pub root: PathBuf,
    /// One entry per `(view, camera)` slot the viewer can render.
    /// View-major; aligns with the indexing of `PerFeatureResiduals.target`.
    pub frames: Vec<FrameRef>,
}

/// What a frame depicts, distinguishing target images from laser-on images.
///
/// Laser problem types capture two images per `(view, camera)` slot: one of
/// the calibration target and one with the laser line on. Both can appear in
/// the same [`ImageManifest`], discriminated by [`FrameRef::kind`].
///
/// Serialized in `snake_case`; absent in JSON means [`FrameKind::Target`],
/// which keeps exports without laser frames byte-stable and forward-readable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum FrameKind {
    /// Image of the calibration target (chessboard / ChArUco / …).
    #[default]
    Target,
    /// Image with the laser line on; residuals of flavor
    /// `PerFeatureResiduals.laser` plot onto frames of this kind.
    Laser,
}

impl FrameKind {
    /// `true` for [`FrameKind::Target`]. Used as `skip_serializing_if` so the
    /// default kind never appears in JSON.
    pub fn is_target(&self) -> bool {
        *self == FrameKind::Target
    }
}

/// Reference to a single image (or sub-image) for one `(view, camera)`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct FrameRef {
    /// View index in the input dataset.
    pub view: usize,
    /// Camera index. `0` for single-camera problem types.
    pub camera: usize,
    /// Image path relative to the enclosing [`ImageManifest::root`].
    pub path: PathBuf,
    /// Sub-image rectangle in pixel coordinates. `None` means the entire
    /// image is this `(view, camera)`. Used for tiled multi-camera frames.
    ///
    /// **Coordinate convention:** see the module-level `# Coordinate
    /// convention` block. ROI is a render-time crop hint only —
    /// per-feature residuals are already in the ROI-local pixel frame
    /// (i.e. the camera's own image frame, since intrinsics were
    /// calibrated against the cropped tile), so a viewer must not
    /// subtract `(x, y)` from `observed_px` / `projected_px` when
    /// drawing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roi: Option<PixelRect>,
    /// What the frame depicts. Absent in JSON means [`FrameKind::Target`],
    /// so target-only exports deserialize unchanged.
    #[serde(default, skip_serializing_if = "FrameKind::is_target")]
    pub kind: FrameKind,
}

/// Inclusive-exclusive pixel rectangle: `[x, x+w) × [y, y+h)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct PixelRect {
    /// Left edge in pixels.
    pub x: u32,
    /// Top edge in pixels.
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl ImageManifest {
    /// Find the **target** frame for a given `(view, camera)` pair, if any.
    ///
    /// Equivalent to [`frame_of_kind`](Self::frame_of_kind) with
    /// [`FrameKind::Target`] — the historical behavior, kept target-only so
    /// manifests that also carry laser frames stay unambiguous.
    pub fn frame(&self, view: usize, camera: usize) -> Option<&FrameRef> {
        self.frame_of_kind(view, camera, FrameKind::Target)
    }

    /// Find the frame of a given kind for a `(view, camera)` pair, if any.
    ///
    /// Linear scan; first match wins. Returns `None` if the manifest has
    /// no entry of that kind for that slot.
    pub fn frame_of_kind(&self, view: usize, camera: usize, kind: FrameKind) -> Option<&FrameRef> {
        self.frames
            .iter()
            .find(|f| f.view == view && f.camera == camera && f.kind == kind)
    }

    /// Iterate over frames of kind [`FrameKind::Target`].
    pub fn target_frames(&self) -> impl Iterator<Item = &FrameRef> {
        self.frames.iter().filter(|f| f.kind == FrameKind::Target)
    }

    /// Iterate over frames of kind [`FrameKind::Laser`].
    pub fn laser_frames(&self) -> impl Iterator<Item = &FrameRef> {
        self.frames.iter().filter(|f| f.kind == FrameKind::Laser)
    }
}

impl ImageManifest {
    /// Construct a manifest rooted at `root` with `frames`.
    pub fn new(root: impl Into<PathBuf>, frames: Vec<FrameRef>) -> Self {
        Self {
            root: root.into(),
            frames,
        }
    }

    /// This manifest for an export that moves from folder `from` to folder
    /// `to`: the same images, with [`root`](Self::root) restated relative
    /// to `to`. The frames are unchanged.
    ///
    /// `root` is read relative to `from` (unless it is absolute). The new
    /// root is a relative path (`/`-separated, `..` where it climbs) when
    /// both the images and `to` lie inside `within`, the folder that moves
    /// as one, such as a project folder, or the dataset folder for output
    /// written inside it; otherwise it is the images' absolute folder, so
    /// the export still finds them when it is moved alone.
    ///
    /// The paths are compared lexically, after making each absolute against
    /// the current directory and folding its `.` and `..` components;
    /// symbolic links are not followed, so pass canonical paths where they
    /// matter.
    ///
    /// ```
    /// use std::path::Path;
    /// use vision_calibration_core::ImageManifest;
    ///
    /// // Frames relative to the dataset folder, the export written to a run
    /// // folder of a project that holds the dataset.
    /// let manifest = ImageManifest::new(".", Vec::new());
    /// let moved = manifest.rebased(
    ///     Path::new("/work/project/images"),
    ///     Path::new("/work/project/runs/run-001"),
    ///     Path::new("/work/project"),
    /// );
    /// assert_eq!(moved.root, Path::new("../../images"));
    /// ```
    pub fn rebased(&self, from: &Path, to: &Path, within: &Path) -> Self {
        let images = lexical_absolute(&from.join(&self.root));
        let to = lexical_absolute(to);
        let within = lexical_absolute(within);
        let root = if images.starts_with(&within) && to.starts_with(&within) {
            relative_path(&to, &images)
        } else {
            images
        };
        Self {
            root,
            frames: self.frames.clone(),
        }
    }
}

/// `path` as a frame path under `data_root`: relative to it and
/// `/`-separated when the image lies inside it, the image's absolute path
/// otherwise. Both are compared as [`lexical_absolute`] states them.
///
/// The dataset runner's one rule for the paths it records, shared by the
/// workspace crates; not part of the public API.
#[doc(hidden)]
pub fn data_relative_path(data_root: &Path, path: &Path) -> PathBuf {
    let root = lexical_absolute(data_root);
    let path = lexical_absolute(path);
    if path.starts_with(&root) {
        relative_path(&root, &path)
    } else {
        path
    }
}

/// `path` made absolute against the current directory (when it can be) and
/// with its `.` and `..` components folded lexically. Symbolic links are not
/// followed.
///
/// Shared by the workspace crates; not part of the public API.
#[doc(hidden)]
pub fn lexical_absolute(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let after_parent =
                    matches!(out.components().next_back(), Some(Component::ParentDir));
                if after_parent || out.as_os_str().is_empty() {
                    // A relative path that climbs out of its start keeps it.
                    out.push(component);
                } else if out.parent().is_some() {
                    out.pop();
                }
                // `..` at the root stays at the root.
            }
            other => out.push(other),
        }
    }
    out
}

/// The `/`-separated relative path from folder `from` to `to`, both
/// absolute and folded; `.` when they are the same folder.
fn relative_path(from: &Path, to: &Path) -> PathBuf {
    let from: Vec<Component<'_>> = from.components().collect();
    let to: Vec<Component<'_>> = to.components().collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let parts = std::iter::repeat_n(OsStr::new(".."), from.len() - common)
        .chain(to[common..].iter().map(|c| c.as_os_str()));
    let mut out = OsString::new();
    for part in parts {
        if !out.is_empty() {
            out.push("/");
        }
        out.push(part);
    }
    if out.is_empty() {
        PathBuf::from(".")
    } else {
        PathBuf::from(out)
    }
}

impl FrameRef {
    /// Construct a target-frame entry for `(view, camera)` pointing at `path`.
    pub fn new(view: usize, camera: usize, path: impl Into<PathBuf>) -> Self {
        Self {
            view,
            camera,
            path: path.into(),
            roi: None,
            kind: FrameKind::Target,
        }
    }
}

impl PixelRect {
    /// Construct a crop rectangle at `(x, y)` with extent `w × h` pixels.
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ImageManifest {
        ImageManifest {
            root: PathBuf::from("images"),
            frames: vec![
                FrameRef {
                    view: 0,
                    camera: 0,
                    path: PathBuf::from("pose_0_cam_0.png"),
                    roi: None,
                    kind: FrameKind::Target,
                },
                FrameRef {
                    view: 1,
                    camera: 0,
                    path: PathBuf::from("strip_pose_1.png"),
                    roi: Some(PixelRect {
                        x: 0,
                        y: 0,
                        w: 720,
                        h: 540,
                    }),
                    kind: FrameKind::Target,
                },
                FrameRef {
                    view: 1,
                    camera: 1,
                    path: PathBuf::from("strip_pose_1.png"),
                    roi: Some(PixelRect {
                        x: 720,
                        y: 0,
                        w: 720,
                        h: 540,
                    }),
                    kind: FrameKind::Target,
                },
                FrameRef {
                    view: 0,
                    camera: 0,
                    path: PathBuf::from("laser_pose_0_cam_0.png"),
                    roi: None,
                    kind: FrameKind::Laser,
                },
            ],
        }
    }

    #[test]
    fn frame_lookup_returns_match() {
        let m = sample();
        let f = m.frame(1, 1).unwrap();
        assert_eq!(f.path, PathBuf::from("strip_pose_1.png"));
        assert_eq!(f.roi.unwrap().x, 720);
    }

    #[test]
    fn frame_lookup_misses_cleanly() {
        let m = sample();
        assert!(m.frame(2, 0).is_none());
        assert!(m.frame(0, 5).is_none());
    }

    #[test]
    fn frame_lookup_is_target_only() {
        let m = sample();
        // (0, 0) has both a target and a laser frame; `frame()` must return
        // the target one.
        assert_eq!(
            m.frame(0, 0).unwrap().path,
            PathBuf::from("pose_0_cam_0.png")
        );
    }

    #[test]
    fn frame_of_kind_finds_laser_frame() {
        let m = sample();
        let f = m.frame_of_kind(0, 0, FrameKind::Laser).unwrap();
        assert_eq!(f.path, PathBuf::from("laser_pose_0_cam_0.png"));
        // No laser frame for view 1.
        assert!(m.frame_of_kind(1, 0, FrameKind::Laser).is_none());
    }

    #[test]
    fn kind_iterators_partition_frames() {
        let m = sample();
        assert_eq!(m.target_frames().count(), 3);
        assert_eq!(m.laser_frames().count(), 1);
        assert_eq!(
            m.target_frames().count() + m.laser_frames().count(),
            m.frames.len()
        );
    }

    #[test]
    fn kind_serde_back_compat() {
        // Target kind never appears in JSON.
        let target = FrameRef {
            view: 0,
            camera: 0,
            path: PathBuf::from("a.png"),
            roi: None,
            kind: FrameKind::Target,
        };
        let json = serde_json::to_string(&target).unwrap();
        assert!(
            !json.contains("kind"),
            "default kind must be omitted: {json}"
        );

        // Laser kind serializes in snake_case and round-trips.
        let laser = FrameRef {
            kind: FrameKind::Laser,
            ..target.clone()
        };
        let json = serde_json::to_string(&laser).unwrap();
        assert!(json.contains("\"kind\":\"laser\""), "got: {json}");
        let restored: FrameRef = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, laser);

        // JSON without a `kind` field deserializes as Target (ADR 0016 §5).
        let no_kind: FrameRef =
            serde_json::from_str(r#"{"view":3,"camera":1,"path":"b.png"}"#).unwrap();
        assert_eq!(no_kind.kind, FrameKind::Target);
    }

    #[test]
    fn json_roundtrip_with_and_without_roi() {
        let m = sample();
        let json = serde_json::to_string_pretty(&m).unwrap();
        let restored: ImageManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, m);

        // ROI absence must not appear in JSON.
        assert!(json.contains("\"pose_0_cam_0.png\""));
        let pose_0_block = json
            .lines()
            .skip_while(|l| !l.contains("pose_0_cam_0.png"))
            .take(2)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !pose_0_block.contains("roi"),
            "skip_serializing_if should omit absent ROI; got:\n{pose_0_block}"
        );
    }

    /// Absolute folders under one temporary-looking root, so the tests
    /// read the same on every platform.
    fn abs(path: &str) -> PathBuf {
        std::path::absolute(Path::new("/vc_rebase_test").join(path)).unwrap()
    }

    #[test]
    fn rebased_is_relative_inside_the_folder_that_moves_as_one() {
        let m = sample();
        // The dataset is the project folder; the export goes to a run folder.
        let moved = m.rebased(
            &abs("project"),
            &abs("project/runs/run-001"),
            &abs("project"),
        );
        assert_eq!(moved.root, PathBuf::from("../../images"));
        assert_eq!(moved.frames, m.frames);

        // A dataset sub-folder of the project.
        let flat = ImageManifest::new(".", Vec::new());
        let moved = flat.rebased(
            &abs("project/cam_left"),
            &abs("project/runs/a"),
            &abs("project"),
        );
        assert_eq!(moved.root, PathBuf::from("../../cam_left"));

        // The export written where the images are.
        let same = flat.rebased(&abs("data"), &abs("data"), &abs("data"));
        assert_eq!(same.root, PathBuf::from("."));

        // Output inside the dataset folder.
        let out = flat.rebased(&abs("data"), &abs("data/out"), &abs("data"));
        assert_eq!(out.root, PathBuf::from(".."));
    }

    #[test]
    fn rebased_is_absolute_outside_the_folder_that_moves_as_one() {
        let flat = ImageManifest::new(".", Vec::new());
        let moved = flat.rebased(&abs("data"), &abs("project/runs/a"), &abs("project"));
        assert_eq!(moved.root, abs("data"));

        // An absolute root stays where it points.
        let pinned = ImageManifest::new(abs("elsewhere/images"), Vec::new());
        let moved = pinned.rebased(&abs("data"), &abs("out"), &abs("out"));
        assert_eq!(moved.root, abs("elsewhere/images"));
    }

    #[test]
    fn rebased_folds_dot_components_and_round_trips() {
        let m = ImageManifest::new("./sub/../images", Vec::new());
        let moved = m.rebased(&abs("data/./x/.."), &abs("data/out"), &abs("data"));
        assert_eq!(moved.root, PathBuf::from("../images"));
        // Back again: the images resolve to the same folder.
        let back = moved.rebased(&abs("data/out"), &abs("data"), &abs("data"));
        assert_eq!(back.root, PathBuf::from("images"));
    }

    #[test]
    fn lexical_absolute_folds_parents_and_stops_at_the_root() {
        assert_eq!(lexical_absolute(&abs("a/b/../..")), abs(""));
        let root = std::path::absolute(Path::new("/")).unwrap();
        assert_eq!(lexical_absolute(&root.join("..")), root);
    }

    #[test]
    fn data_relative_path_is_relative_inside_the_root_and_absolute_outside() {
        assert_eq!(
            data_relative_path(&abs("data"), &abs("data/./cam0/../cam1/a.png")),
            PathBuf::from("cam1/a.png")
        );
        assert_eq!(
            data_relative_path(&abs("data"), &abs("other/a.png")),
            abs("other/a.png")
        );
    }

    #[test]
    fn empty_manifest_roundtrips() {
        let m = ImageManifest::default();
        let json = serde_json::to_string(&m).unwrap();
        let restored: ImageManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, m);
    }
}
