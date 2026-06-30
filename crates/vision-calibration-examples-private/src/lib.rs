//! Shared helpers for private calibration examples.
//!
//! This crate is `publish = false`; it carries path dependencies to the
//! sibling `calib-targets-rs` and `vision-metrology` workspaces so the
//! end-to-end example can run on physical sensor data.

use anyhow::{Context, Result, anyhow};
use image::{GrayImage, ImageReader, imageops::FilterType};
use serde::Deserialize;
use std::path::Path;
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, HomographySensor, Iso3, Pinhole, Pt2,
    Pt3, Real, ScheimpflugParams,
};

use calib_targets::{
    aruco::builtins,
    charuco::{CharucoBoardSpec, CharucoDetector, CharucoParams, MarkerLayout},
    detect::{self, detect_puzzleboard},
    puzzleboard::{PuzzleBoardParams, PuzzleBoardSearchMode, PuzzleBoardSpec},
};

use vision_calibration_detect::{Detector, RinggridDetector};
use vision_metrology::{ColAccess, Edge1DConfig, LaserExtractConfig, LaserExtractor, ScanAxis};
use vm_primitives::core::ImageView;

/// Default coded-ring band width in millimetres, mirroring `ringgrid`'s own
/// `DEFAULT_RING_WIDTH_MM`. The `ringgrid.target.v3` board manifest omits the
/// band width (it became mandatory only in the v4 schema), so a board loaded
/// from v3 falls back to this value. It governs decode sampling, not the
/// marker→board-coordinate map, so it can be swept without changing geometry.
pub const DEFAULT_RING_WIDTH_MM: f64 = 1.152;

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
        use nalgebra::{Matrix3, Matrix4, Rotation3, Translation3, UnitQuaternion};
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
        // `UnitQuaternion::from_matrix` is iterative and mis-converges on exact
        // 180° rotations (the rtv3d_ringgrid robot poses are 180° about y),
        // silently returning the wrong axis and breaking the hand-eye stage.
        // The robot matrices are orthonormal, so convert exactly via `Rotation3`.
        let quat = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot));
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

/// Full Scheimpflug camera (pinhole projection, Brown-Conrady distortion,
/// tilted-sensor homography, fx/fy/cx/cy/skew intrinsics). Unlike the
/// workspace's `PinholeCamera` alias (which uses an identity sensor and carries
/// the Scheimpflug tilt in a side channel), this folds the tilt into the
/// projection chain so `project_point` reprojects with the tilt applied.
pub type ScheimpflugCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, HomographySensor<f64>, FxFyCxCySkew<Real>>;

/// One per-camera intrinsics block from a reference `artifacts.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RefIntrinsic {
    /// OpenCV-layout distortion `[k1, k2, p1, p2, k3, tau_x, tau_y]` (7 coeffs).
    pub distortion: Vec<f64>,
    /// Image width in pixels.
    pub frame_cols: u32,
    /// Image height in pixels.
    pub frame_rows: u32,
    /// Row-major 3x3 camera matrix K.
    pub matrix: [[f64; 3]; 3],
    /// Reference per-camera reprojection error in pixels.
    pub reprojection_error_pix: f64,
}

/// One per-camera extrinsics block (camera-to-sensor/rig pose) from a reference
/// `artifacts.json`. Translation is in millimeters.
#[derive(Debug, Clone, Deserialize)]
pub struct RefExtrinsic {
    /// 4x4 row-major homogeneous transform `camera_se3_sensor` (mm).
    pub camera_se3_sensor: [[f64; 4]; 4],
    /// Reference per-camera reprojection error in pixels.
    #[serde(default)]
    pub reprojection_error_pix: f64,
}

/// Hand-eye block from a reference `artifacts.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct RefHandeye {
    /// Reference hand-eye quality score.
    #[serde(default)]
    pub quality_score: f64,
    /// 4x4 row-major homogeneous transform `tcp_se3_sensor` (mm).
    pub tcp_se3_sensor: [[f64; 4]; 4],
}

/// One laser plane from a reference `artifacts.json`, in the legacy
/// origin/axes representation (columns are 3x1). Parsed for completeness; the
/// laser stage is out of scope for the reprojection-parity harness.
#[derive(Debug, Clone, Deserialize)]
pub struct RefLaserPlane {
    /// Plane origin (3x1, mm).
    pub origin: [[f64; 1]; 3],
    /// In-plane x axis (3x1, unit).
    pub xaxis: [[f64; 1]; 3],
    /// In-plane y axis (3x1, unit).
    pub yaxis: [[f64; 1]; 3],
    /// Point-to-plane fit standard deviation in millimeters.
    pub standard_deviation_mm: f64,
}

/// A parsed reference `artifacts.json` (the "QUICK"-system oracle).
#[derive(Debug, Clone, Deserialize)]
pub struct RefArtifacts {
    /// Per-camera intrinsics.
    pub intrinsic: Vec<RefIntrinsic>,
    /// Per-camera extrinsics (`camera_se3_sensor`).
    pub extrinsic: Vec<RefExtrinsic>,
    /// Hand-eye calibration.
    pub handeye: RefHandeye,
    /// Per-camera laser planes.
    #[serde(default)]
    pub laserplanes: Vec<RefLaserPlane>,
    /// Free-form metadata (date, device, ...).
    #[serde(default)]
    pub meta: serde_json::Value,
    /// Number of cameras in the rig.
    pub num_cameras: usize,
}

/// Load and parse a reference `artifacts.json`.
pub fn load_ref_artifacts(path: &Path) -> Result<RefArtifacts> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {:?}", path.display()))?;
    let art: RefArtifacts =
        serde_json::from_str(&s).with_context(|| format!("parse {:?}", path.display()))?;
    Ok(art)
}

/// Split an OpenCV-layout 7-coefficient distortion vector
/// `[k1, k2, p1, p2, k3, tau_x, tau_y]` into the workspace's separate
/// Brown-Conrady and Scheimpflug parameter blocks.
///
/// This is the single choke point that reconciles the two coefficient orders:
/// OpenCV places `k3` *after* the tangential terms `p1, p2`, whereas our
/// [`BrownConrady5`] stores `k3` *before* them. The reorder happens here and
/// nowhere else; the roundtrip test guards it.
pub fn split_opencv_distortion(dist: &[f64]) -> Result<(BrownConrady5<Real>, ScheimpflugParams)> {
    if dist.len() != 7 {
        return Err(anyhow!(
            "expected 7 distortion coeffs [k1,k2,p1,p2,k3,tau_x,tau_y], got {}",
            dist.len()
        ));
    }
    let bc = BrownConrady5 {
        k1: dist[0],
        k2: dist[1],
        k3: dist[4],
        p1: dist[2],
        p2: dist[3],
        iters: 10,
    };
    let tilt = ScheimpflugParams {
        tilt_x: dist[5],
        tilt_y: dist[6],
    };
    Ok((bc, tilt))
}

/// Build a frozen [`ScheimpflugCamera`] from a reference intrinsics block.
pub fn intrinsic_to_camera(intr: &RefIntrinsic) -> Result<ScheimpflugCamera> {
    let m = &intr.matrix;
    let k = FxFyCxCySkew {
        fx: m[0][0],
        fy: m[1][1],
        cx: m[0][2],
        cy: m[1][2],
        skew: m[0][1],
    };
    let (dist, tilt) = split_opencv_distortion(&intr.distortion)?;
    Ok(Camera::new(Pinhole, dist, tilt.compile(), k))
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
/// with 3D points in meters (Z=0) in the canonical master-board frame.
pub fn detect_target(
    tile: &GrayImage,
    rows: u32,
    cols: u32,
    cell_size_mm: f64,
) -> Result<CorrespondenceView> {
    const TARGET_DETECT_UPSCALE: u32 = 2;
    let detection_image = image::imageops::resize(
        tile,
        tile.width() * TARGET_DETECT_UPSCALE,
        tile.height() * TARGET_DETECT_UPSCALE,
        FilterType::Triangle,
    );
    let spec = PuzzleBoardSpec::new(rows, cols, cell_size_mm as f32)
        .map_err(|e| anyhow!("puzzleboard spec: {e}"))?;
    let mut params = PuzzleBoardParams::for_board(&spec);
    params.decode.search_all_components = false;
    params.decode.search_mode = PuzzleBoardSearchMode::FixedBoard;
    let result =
        detect_puzzleboard(&detection_image, &params).map_err(|e| anyhow!("detect: {e}"))?;

    let mut pts_3d: Vec<Pt3> = Vec::new();
    let mut pts_2d: Vec<Pt2> = Vec::new();
    let origin_x_mm = 0.5 * (cols.saturating_sub(1) as f64) * cell_size_mm;
    let origin_y_mm = 0.5 * (rows.saturating_sub(1) as f64) * cell_size_mm;
    let max_x_mm = cols.saturating_sub(1) as f64 * cell_size_mm;
    let max_y_mm = rows.saturating_sub(1) as f64 * cell_size_mm;
    for corner in &result.corners {
        let t = corner.target_position;
        let x_mm = t.x as f64;
        let y_mm = t.y as f64;
        if !(0.0..=max_x_mm).contains(&x_mm) || !(0.0..=max_y_mm).contains(&y_mm) {
            continue;
        }
        pts_3d.push(Pt3::new(
            ((x_mm - origin_x_mm) / 1000.0) as Real,
            ((y_mm - origin_y_mm) / 1000.0) as Real,
            0.0,
        ));
        pts_2d.push(Pt2::new(
            (corner.position.x / TARGET_DETECT_UPSCALE as f32) as Real,
            (corner.position.y / TARGET_DETECT_UPSCALE as f32) as Real,
        ));
    }

    if pts_3d.len() < 8 {
        return Err(anyhow!(
            "puzzleboard detection too sparse: {} corners",
            pts_3d.len()
        ));
    }
    CorrespondenceView::new(pts_3d, pts_2d).map_err(|e| anyhow!("correspondence: {e}"))
}

/// Detect a ChArUco board in a tile and build a [`CorrespondenceView`] with
/// 3D points in meters (Z=0) in the board frame.
///
/// Thin port of `vision-calibration-bench/src/detect.rs::detect_charuco_view`
/// (same dictionary handling, `OpenCvCharuco` marker layout, ChESS corner
/// pre-detection) so rtv3d numbers reproduce against the bench harness.
/// ChArUco detections are sparse — only decoded cells contribute corners.
pub fn detect_charuco(
    tile: &GrayImage,
    rows: u32,
    cols: u32,
    cell_size_mm: f64,
    marker_size_rel: f32,
    dictionary: &str,
) -> Result<CorrespondenceView> {
    let dict = match dictionary {
        "DICT_4X4_1000" => builtins::DICT_4X4_1000,
        other => return Err(anyhow!("unsupported ChArUco dictionary '{other}'")),
    };
    let board = CharucoBoardSpec {
        rows,
        cols,
        cell_size: (cell_size_mm / 1000.0) as f32,
        marker_size_rel,
        dictionary: dict,
        marker_layout: MarkerLayout::OpenCvCharuco,
    };
    let params = CharucoParams::for_board(&board);
    let chess_config = detect::default_chess_config();
    let corners = detect::detect_corners(tile, &chess_config);
    let detector = CharucoDetector::new(params).map_err(|e| anyhow!("charuco detector: {e}"))?;
    let detection = detector
        .detect(&detect::gray_view(tile), &corners)
        .map_err(|e| anyhow!("charuco detect: {e}"))?;

    let mut points_3d = Vec::with_capacity(detection.corners.len());
    let mut points_2d = Vec::with_capacity(detection.corners.len());
    for corner in detection.corners {
        let target = corner.target_position;
        points_3d.push(Pt3::new(target.x as f64, target.y as f64, 0.0));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    if points_3d.len() < 8 {
        return Err(anyhow!(
            "charuco detection too sparse: {} corners",
            points_3d.len()
        ));
    }
    CorrespondenceView::new(points_3d, points_2d).map_err(|e| anyhow!("correspondence: {e}"))
}

/// A coded ring-grid board manifest (`ringgrid.target.v3`, as shipped with the
/// `rtv3d_ringgrid` dataset). Geometry is in millimetres. `marker_ring_width_mm`
/// is optional because the v3 schema omits it; [`detect_ringgrid`] falls back to
/// [`DEFAULT_RING_WIDTH_MM`].
#[derive(Debug, Clone, Deserialize)]
pub struct BoardRinggridSpec {
    /// Center-to-center spacing between adjacent markers (mm).
    pub pitch_mm: f64,
    /// Number of marker rows.
    pub rows: u32,
    /// Number of columns in the longest (even-indexed) row.
    pub long_row_cols: u32,
    /// Outer ring radius (mm).
    pub marker_outer_radius_mm: f64,
    /// Inner ring radius (mm).
    pub marker_inner_radius_mm: f64,
    /// Coded-band width (mm). Present in the v4 schema; absent in v3.
    #[serde(default)]
    pub marker_ring_width_mm: Option<f64>,
}

impl BoardRinggridSpec {
    /// The band width to use for decoding: the manifest value when present,
    /// else [`DEFAULT_RING_WIDTH_MM`].
    pub fn ring_width_mm(&self) -> f64 {
        self.marker_ring_width_mm.unwrap_or(DEFAULT_RING_WIDTH_MM)
    }
}

/// Load a `board_ringgrid.json` manifest.
pub fn load_ringgrid_board(path: &Path) -> Result<BoardRinggridSpec> {
    let s = std::fs::read_to_string(path).with_context(|| format!("read {:?}", path.display()))?;
    let spec: BoardRinggridSpec =
        serde_json::from_str(&s).with_context(|| format!("parse {:?}", path.display()))?;
    Ok(spec)
}

/// Detect a coded ring-grid in a tile and build a [`CorrespondenceView`] with
/// 3D points in metres (Z=0) in the board's own frame.
///
/// Wraps [`RinggridDetector`] (the `vision-calibration-detect` adaptive-scale
/// wrapper around the `ringgrid` crate). Ring markers are self-identifying, so
/// each decoded marker maps to a globally-consistent board position — the
/// returned coordinates are in the board's **native** frame and are *not*
/// re-centered, because the marker set differs per view and a per-view centroid
/// shift would break cross-view frame consistency. `ring_width_mm` overrides the
/// board's band width (a decode-tuning knob; it does not affect the geometry).
pub fn detect_ringgrid(
    tile: &GrayImage,
    spec: &BoardRinggridSpec,
    ring_width_mm: f64,
) -> Result<CorrespondenceView> {
    let config = serde_json::json!({
        "pitch_m": spec.pitch_mm / 1000.0,
        "rows": spec.rows,
        "long_row_cols": spec.long_row_cols,
        "marker_outer_radius_m": spec.marker_outer_radius_mm / 1000.0,
        "marker_inner_radius_m": spec.marker_inner_radius_mm / 1000.0,
        "marker_ring_width_m": ring_width_mm / 1000.0,
    });
    let dynimg = image::DynamicImage::ImageLuma8(tile.clone());
    let features = RinggridDetector
        .detect_json(&dynimg, &config)
        .map_err(|e| anyhow!("ringgrid detect: {e}"))?;

    let mut pts_3d: Vec<Pt3> = Vec::with_capacity(features.len());
    let mut pts_2d: Vec<Pt2> = Vec::with_capacity(features.len());
    for f in &features {
        pts_3d.push(Pt3::new(
            f.world_xyz[0] as Real,
            f.world_xyz[1] as Real,
            f.world_xyz[2] as Real,
        ));
        pts_2d.push(Pt2::new(f.image_xy[0] as Real, f.image_xy[1] as Real));
    }

    if pts_3d.len() < 8 {
        return Err(anyhow!(
            "ringgrid detection too sparse: {} markers",
            pts_3d.len()
        ));
    }
    CorrespondenceView::new(pts_3d, pts_2d).map_err(|e| anyhow!("correspondence: {e}"))
}

/// Detect the ring-grid in every camera tile of every pose, returning per-pose,
/// per-camera detections as `[pose][cam] -> Option<view>` (inner `None` where a
/// camera failed to decode enough markers in that pose).
///
/// Ring-grid detection is the slow part of the pipeline (adaptive-scale decode
/// of every tile), so an optional `cache_path` memoizes the result as JSON:
/// when the file exists it is loaded verbatim; otherwise detection runs and the
/// result is written there. The caller owns cache invalidation — vary the path
/// (e.g. by `ring_width_mm`) when the inputs change.
pub fn detect_ringgrid_all(
    data_dir: &Path,
    poses: &[PoseEntry],
    board: &BoardRinggridSpec,
    ring_width_mm: f64,
    cache_path: Option<&Path>,
) -> Result<Vec<Vec<Option<CorrespondenceView>>>> {
    if let Some(p) = cache_path
        && p.exists()
    {
        let s =
            std::fs::read_to_string(p).with_context(|| format!("read cache {:?}", p.display()))?;
        let cached: Vec<Vec<Option<CorrespondenceView>>> =
            serde_json::from_str(&s).with_context(|| format!("parse cache {:?}", p.display()))?;
        return Ok(cached);
    }

    let mut out: Vec<Vec<Option<CorrespondenceView>>> = Vec::with_capacity(poses.len());
    let num_cameras = 6; // rtv3d horizontal strip = six 720×540 tiles
    for (i, pose) in poses.iter().enumerate() {
        let img = load_gray(&data_dir.join(&pose.target_image))
            .with_context(|| format!("pose {i} target"))?;
        let tiles = split_horizontal(&img, num_cameras);
        let mut cams: Vec<Option<CorrespondenceView>> = Vec::with_capacity(num_cameras);
        for tile in &tiles {
            cams.push(detect_ringgrid(tile, board, ring_width_mm).ok());
        }
        out.push(cams);
    }

    if let Some(p) = cache_path {
        let s = serde_json::to_string(&out).context("serialize detection cache")?;
        std::fs::write(p, s).with_context(|| format!("write cache {:?}", p.display()))?;
    }
    Ok(out)
}

/// Probe whether recovered per-view rig poses trace the robot motion — the
/// precondition the hand-eye stage needs.
///
/// For eye-in-hand `A·X = X·B`, where `A` is the robot's relative gripper motion
/// and `B` the rig's relative motion, and conjugation preserves the rotation
/// angle, so `∠A` must match `∠B` for every pose pair. Reports the distribution
/// of `|∠robot − ∠rig|` over pairs with meaningful (>5°) robot motion: near-0°
/// means the poses are consistent (hand-eye solvable); large values mean the
/// recovered rig rotations do not track the robot (hand-eye cannot fit one
/// transform). Independent of hand-eye topology — relative angles are invariant.
pub fn handeye_consistency_probe(poses: &[PoseEntry], rig_se3_target: &[Iso3]) {
    let n = poses.len().min(rig_se3_target.len());
    let mut diffs: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let a_ang = (poses[i].base_se3_gripper().rotation.inverse()
                * poses[j].base_se3_gripper().rotation)
                .angle()
                .to_degrees();
            let b_ang = (rig_se3_target[i].rotation.inverse() * rig_se3_target[j].rotation)
                .angle()
                .to_degrees();
            if a_ang > 5.0 {
                diffs.push((a_ang - b_ang).abs());
            }
        }
    }
    if diffs.is_empty() {
        println!("  hand-eye consistency: no pose pairs with >5° robot motion");
        return;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let median = diffs[diffs.len() / 2];
    let max = *diffs.last().unwrap();
    println!(
        "  hand-eye consistency (|∠robot − ∠rig| over {} pairs): mean {mean:.2}° median {median:.2}° max {max:.2}°",
        diffs.len(),
    );
    println!("    (≈0° ⇒ rig poses trace the robot, hand-eye solvable; large ⇒ inconsistent)");

    if std::env::var("RTV3D_RINGGRID_DIAG").as_deref() == Ok("1") {
        println!("    per-view rel-to-view0 rotation angle (robot vs rig):");
        for i in 0..n {
            let r = (poses[0].base_se3_gripper().rotation.inverse()
                * poses[i].base_se3_gripper().rotation)
                .angle()
                .to_degrees();
            let g = (rig_se3_target[0].rotation.inverse() * rig_se3_target[i].rotation)
                .angle()
                .to_degrees();
            println!("      v{i:2}: robot {r:6.1}°   rig {g:6.1}°");
        }
    }
}

/// Extract a laser line in a tile as subpixel 2D points.
pub fn detect_laser(tile: &GrayImage) -> Vec<Pt2> {
    let width = tile.width() as usize;
    let height = tile.height() as usize;
    let data = tile.as_raw();
    let view = ImageView::<u8>::from_slice(width, height, width, data)
        .expect("image slice length matches width*height");

    let cfg = LaserExtractConfig {
        axis: ScanAxis::Cols {
            access: ColAccess::Gather,
        },
        edge_cfg: Edge1DConfig {
            sigma: 1.2,
            pos_thresh: 4.0,
            neg_thresh: 4.0,
            ..Edge1DConfig::default()
        },
        ..LaserExtractConfig::default()
    };
    let mut extractor = LaserExtractor::new(cfg.edge_cfg.sigma);
    let line = extractor.extract_line_u8(&view, 0..width, &cfg, None);

    line.points
        .into_iter()
        .map(|p| Pt2::new(p.x as Real, p.y as Real))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_matrix_iterative_mangles_180deg_about_y() {
        use nalgebra::{Matrix3, Rotation3, UnitQuaternion};
        // Exact 180° rotation about the y-axis (a real rtv3d_ringgrid robot pose).
        let r = Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let q_iter = UnitQuaternion::from_matrix(&r);
        let q_exact = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
        let err_iter = (q_iter.to_rotation_matrix().into_inner() - r).norm();
        let err_exact = (q_exact.to_rotation_matrix().into_inner() - r).norm();
        println!("from_matrix err_iter={err_iter:.6}  err_exact={err_exact:.6}");
        // The exact path round-trips; document whatever the iterative path does.
        assert!(err_exact < 1e-9, "exact conversion must round-trip");
    }

    #[test]
    fn base_se3_gripper_roundtrips_180deg_pose() {
        // tcp2base = 180° about y, translation (320,0,155) mm — rtv3d_ringgrid pose 0.
        let pose = PoseEntry {
            laser_image: String::new(),
            target_image: String::new(),
            tcp2base: [
                [-1.0, 0.0, 0.0, 320.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 155.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            snap_type: "double_snap".to_string(),
        };
        let iso = pose.base_se3_gripper();
        let r = iso.rotation.to_rotation_matrix().into_inner();
        let expect = nalgebra::Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let err = (r - expect).norm();
        assert!(
            err < 1e-9,
            "base_se3_gripper must round-trip a 180° pose (err={err:.6})"
        );
    }

    #[test]
    fn opencv_distortion_split_reorders_k3_past_tangentials() {
        // Distinct sentinels so a wrong slot is unmistakable.
        // OpenCV layout: [k1, k2, p1, p2, k3, tau_x, tau_y].
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (bc, tilt) = split_opencv_distortion(&d).unwrap();
        assert_eq!(bc.k1, 1.0);
        assert_eq!(bc.k2, 2.0);
        assert_eq!(bc.p1, 3.0, "p1 comes from OpenCV index 2");
        assert_eq!(bc.p2, 4.0, "p2 comes from OpenCV index 3");
        assert_eq!(bc.k3, 5.0, "k3 comes from OpenCV index 4, NOT index 2");
        assert_eq!(tilt.tilt_x, 6.0);
        assert_eq!(tilt.tilt_y, 7.0);
    }

    #[test]
    fn opencv_distortion_split_rejects_wrong_length() {
        assert!(split_opencv_distortion(&[0.0; 5]).is_err());
        assert!(split_opencv_distortion(&[0.0; 8]).is_err());
    }

    #[test]
    fn detect_laser_returns_at_most_one_point_per_column() {
        let width = 80;
        let height = 48;
        let mut img = GrayImage::new(width, height);
        for x in 5..75 {
            let y = 20 + (x as i32 - 40).abs() / 16;
            for dy in -1..=1 {
                img.put_pixel(x, (y + dy) as u32, image::Luma([220]));
            }
        }

        let points = detect_laser(&img);
        assert!(!points.is_empty());
        assert!(points.len() <= width as usize);

        let mut seen = std::collections::BTreeSet::new();
        for p in points {
            assert!(seen.insert(p.x.round() as i32));
            assert!((18.0..=24.0).contains(&p.y));
        }
    }
}
