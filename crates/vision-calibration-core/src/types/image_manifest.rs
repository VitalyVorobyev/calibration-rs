//! Per-export image manifest for the diagnose UI (Track B / ADR 0014).
//!
//! `ImageManifest` is an optional addition to selected `*Export` types that
//! lets a downstream viewer (the planned Tauri/React diagnose UI; ad-hoc
//! viewers; CI screenshot tools) locate the source image for each
//! `(pose, camera)` slot referenced by [`PerFeatureResiduals`].
//!
//! The manifest is a viewer-side contract: the calibration pipeline does not
//! consume it. Existing exports remain valid by absence (the field is
//! `Option<…>` on the host export and `skip_serializing_if = "Option::is_none"`
//! so JSON byte-stability is preserved).
//!
//! # Indexing
//!
//! Frames are pose-major and align with the `(pose, camera)` indexing used by
//! [`TargetFeatureResidual`]/[`LaserFeatureResidual`]. Lookup by linear scan
//! is fine at the dataset sizes the diagnose UI targets (≤ a few hundred
//! frames); no map index is materialised.
//!
//! # Tiled multi-camera frames
//!
//! Industrial rigs sometimes emit a single physical image that horizontally
//! tiles N cameras (the puzzle 130×130 dataset uses 6× 720×540 tiles in a
//! 4320×540 strip). [`FrameRef::roi`] expresses this without inventing a
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
//! would double-correct and push residuals off-canvas — see the bug fix
//! that introduced this convention.
//!
//! [`PerFeatureResiduals`]: crate::PerFeatureResiduals
//! [`TargetFeatureResidual`]: crate::TargetFeatureResidual
//! [`LaserFeatureResidual`]: crate::LaserFeatureResidual

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Image manifest carried alongside a calibration export.
///
/// `root` is interpreted relative to the directory of the `export.json` that
/// embeds the manifest. Each [`FrameRef::path`] is interpreted relative to
/// `root`. Both layers of indirection make exports portable across machines
/// as long as the image directory is co-located with the export.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageManifest {
    /// Image root directory, relative to the export file's directory.
    pub root: PathBuf,
    /// One entry per `(pose, camera)` slot the viewer can render.
    /// Pose-major; aligns with the indexing of `PerFeatureResiduals.target`.
    pub frames: Vec<FrameRef>,
}

/// Reference to a single image (or sub-image) for one `(pose, camera)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRef {
    /// Pose / view index in the input dataset.
    pub pose: usize,
    /// Camera index. `0` for single-camera problem types.
    pub camera: usize,
    /// Image path relative to the enclosing [`ImageManifest::root`].
    pub path: PathBuf,
    /// Sub-image rectangle in pixel coordinates. `None` means the entire
    /// image is this `(pose, camera)`. Used for tiled multi-camera frames.
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
}

/// Inclusive-exclusive pixel rectangle: `[x, x+w) × [y, y+h)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Find the frame for a given `(pose, camera)` pair, if any.
    ///
    /// Linear scan; first match wins. Returns `None` if the manifest has
    /// no entry for that slot.
    pub fn frame(&self, pose: usize, camera: usize) -> Option<&FrameRef> {
        self.frames
            .iter()
            .find(|f| f.pose == pose && f.camera == camera)
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
                    pose: 0,
                    camera: 0,
                    path: PathBuf::from("pose_0_cam_0.png"),
                    roi: None,
                },
                FrameRef {
                    pose: 1,
                    camera: 0,
                    path: PathBuf::from("strip_pose_1.png"),
                    roi: Some(PixelRect {
                        x: 0,
                        y: 0,
                        w: 720,
                        h: 540,
                    }),
                },
                FrameRef {
                    pose: 1,
                    camera: 1,
                    path: PathBuf::from("strip_pose_1.png"),
                    roi: Some(PixelRect {
                        x: 720,
                        y: 0,
                        w: 720,
                        h: 540,
                    }),
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

    #[test]
    fn empty_manifest_roundtrips() {
        let m = ImageManifest::default();
        let json = serde_json::to_string(&m).unwrap();
        let restored: ImageManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, m);
    }
}
