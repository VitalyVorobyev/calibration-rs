//! Shared helpers for private calibration examples.
//!
//! This crate is `publish = false`; it carries path dependencies to the
//! sibling `calib-targets-rs` and `vision-metrology` workspaces so the
//! end-to-end example can run on physical sensor data.

use anyhow::{Context, Result, anyhow};
use image::{GrayImage, ImageReader};
use serde::Deserialize;
use std::path::Path;
use vision_calibration_core::{CorrespondenceView, Iso3, Pt2, Pt3, Real};

use calib_targets::{
    detect::detect_puzzleboard,
    puzzleboard::{PuzzleBoardParams, PuzzleBoardSpec},
};

use vision_metrology::{LaserExtractConfig, LaserExtractor, ScanAxis};
use vm_primitives::core::ImageView;

/// Pose entry as serialized in `poses.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct PoseEntry {
    /// File name (relative to the data directory) of the laser image.
    pub laser_image: String,
    /// File name (relative to the data directory) of the target image.
    pub target_image: String,
    /// 4x4 row-major homogeneous transform from tool center point (TCP) to base.
    pub tcp2base: [[f64; 4]; 4],
    /// Snap type: `"double_snap"` or `"target_snap"`. Laser pixels are only
    /// consumed for `"double_snap"` entries.
    #[serde(rename = "type")]
    pub snap_type: String,
}

impl PoseEntry {
    /// Convert `tcp2base` to an [`Iso3`] (robot base_se3_gripper).
    ///
    /// Translation is read from the manifest in millimeters and converted to
    /// meters here so every `Iso3` returned by this crate is SI-consistent
    /// (matching the target-frame 3D points emitted by [`detect_target`]).
    pub fn base_se3_gripper(&self) -> Iso3 {
        use nalgebra::{Matrix3, Matrix4, Translation3, UnitQuaternion};
        let m = Matrix4::<f64>::from_iterator(self.tcp2base.iter().flatten().copied()).transpose();
        let rot = Matrix3::<f64>::new(
            m[(0, 0)],
            m[(0, 1)],
            m[(0, 2)],
            m[(1, 0)],
            m[(1, 1)],
            m[(1, 2)],
            m[(2, 0)],
            m[(2, 1)],
            m[(2, 2)],
        );
        // poses.json encodes tcp2base translation in millimeters; convert to
        // meters to stay consistent with detect_target's mm→m conversion.
        const MM_TO_M: f64 = 1.0e-3;
        let trans = Translation3::new(
            m[(0, 3)] * MM_TO_M,
            m[(1, 3)] * MM_TO_M,
            m[(2, 3)] * MM_TO_M,
        );
        let quat = UnitQuaternion::from_matrix(&rot);
        Iso3::from_parts(trans, quat)
    }

    /// Whether this snap provides usable laser observations.
    pub fn has_laser(&self) -> bool {
        self.snap_type == "double_snap"
    }
}

/// Load the `poses.json` manifest.
pub fn load_poses(path: &Path) -> Result<Vec<PoseEntry>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {:?}", path.display()))?;
    let poses: Vec<PoseEntry> =
        serde_json::from_str(&s).with_context(|| format!("parse {:?}", path.display()))?;
    Ok(poses)
}

/// Load a PNG as grayscale.
pub fn load_gray(path: &Path) -> Result<GrayImage> {
    let img = ImageReader::open(path)
        .with_context(|| format!("open {:?}", path.display()))?
        .decode()
        .with_context(|| format!("decode {:?}", path.display()))?
        .to_luma8();
    Ok(img)
}

/// Slice a horizontally-concatenated image into `num_cameras` equal-width tiles.
pub fn split_horizontal(img: &GrayImage, num_cameras: usize) -> Vec<GrayImage> {
    let total_w = img.width() as usize;
    let tile_w = total_w / num_cameras;
    let h = img.height() as usize;
    (0..num_cameras)
        .map(|i| {
            let x0 = (i * tile_w) as u32;
            let tile = image::imageops::crop_imm(img, x0, 0, tile_w as u32, h as u32);
            tile.to_image()
        })
        .collect()
}

/// Detect a 130x130 puzzleboard in a tile and build a [`CorrespondenceView`]
/// with 3D points in millimeters (Z=0).
pub fn detect_target(
    tile: &GrayImage,
    rows: u32,
    cols: u32,
    cell_size_mm: f64,
) -> Result<CorrespondenceView> {
    let spec = PuzzleBoardSpec::new(rows, cols, cell_size_mm as f32)
        .map_err(|e| anyhow!("puzzleboard spec: {e}"))?;
    let params = PuzzleBoardParams::for_board(&spec);
    let result = detect_puzzleboard(tile, &params).map_err(|e| anyhow!("detect: {e}"))?;

    let mut pts_3d: Vec<Pt3> = Vec::new();
    let mut pts_2d: Vec<Pt2> = Vec::new();
    // Find target_position range so we can recenter to a board-local frame.
    // The puzzleboard detector returns absolute positions on the 501×501 master
    // pattern; for each view only a small subregion is visible. Without
    // recentering, Zhang sees a narrow slab offset from the origin, which
    // biases the recovered principal point.
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for corner in &result.detection.corners {
        if let (Some(_id), Some(t)) = (corner.id, corner.target_position) {
            min_x = min_x.min(t.x);
            max_x = max_x.max(t.x);
            min_y = min_y.min(t.y);
            max_y = max_y.max(t.y);
        }
    }
    let mid_x = 0.5 * (min_x + max_x);
    let mid_y = 0.5 * (min_y + max_y);
    for corner in &result.detection.corners {
        if let (Some(_id), Some(t)) = (corner.id, corner.target_position) {
            pts_3d.push(Pt3::new(
                (t.x - mid_x) as Real / 1000.0,
                (t.y - mid_y) as Real / 1000.0,
                0.0,
            ));
            pts_2d.push(Pt2::new(
                corner.position.x as Real,
                corner.position.y as Real,
            ));
        }
    }

    if pts_3d.len() < 8 {
        return Err(anyhow!(
            "puzzleboard detection too sparse: {} corners",
            pts_3d.len()
        ));
    }
    CorrespondenceView::new(pts_3d, pts_2d).map_err(|e| anyhow!("correspondence: {e}"))
}

/// Extract a laser line in a tile as subpixel 2D points.
pub fn detect_laser(tile: &GrayImage) -> Vec<Pt2> {
    let width = tile.width() as usize;
    let height = tile.height() as usize;
    let data = tile.as_raw();
    let view = ImageView::<u8>::from_slice(width, height, width, data)
        .expect("image slice length matches width*height");

    let cfg = LaserExtractConfig {
        axis: ScanAxis::Rows,
        ..LaserExtractConfig::default()
    };
    let mut extractor = LaserExtractor::new(cfg.edge_cfg.sigma);
    let line = extractor.extract_line_u8(&view, 0..height, &cfg, None);

    line.points
        .into_iter()
        .map(|p| Pt2::new(p.x as Real, p.y as Real))
        .collect()
}
