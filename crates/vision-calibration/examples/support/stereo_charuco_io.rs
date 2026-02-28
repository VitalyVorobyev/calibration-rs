//! Shared I/O helpers for the stereo ChArUco rig calibration example.

use anyhow::{Context, Result, ensure};
use calib_targets::aruco::builtins;
use calib_targets::charuco::{CharucoBoardSpec, CharucoDetectorParams, MarkerLayout};
use calib_targets::detect;
use chess_corners::ChessConfig;
use std::collections::BTreeSet;
use std::path::Path;
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::RigExtrinsicsInput;

pub const BOARD_ROWS: u32 = 22;
pub const BOARD_COLS: u32 = 22;
pub const BOARD_CELL_SIZE_MM: f64 = 0.00135;
pub const BOARD_MARKER_SIZE_REL: f32 = 0.75;
pub const BOARD_DICTIONARY_NAME: &str = "DICT_4X4_1000";

const GRAPH_MAX_SPACING_PX: f32 = 120.0;

/// Summary of ChArUco stereo dataset loading.
#[derive(Debug, Clone)]
pub struct StereoCharucoDatasetSummary {
    pub total_pairs: usize,
    pub used_views: usize,
    pub skipped_views: usize,
    pub usable_left: usize,
    pub usable_right: usize,
}

/// Build ChArUco detector params used by the stereo example.
pub fn make_charuco_detector_params() -> CharucoDetectorParams {
    let board = CharucoBoardSpec {
        rows: BOARD_ROWS,
        cols: BOARD_COLS,
        cell_size: BOARD_CELL_SIZE_MM as f32,
        marker_size_rel: BOARD_MARKER_SIZE_REL,
        dictionary: builtins::DICT_4X4_1000,
        marker_layout: MarkerLayout::OpenCvCharuco,
    };

    let mut params = CharucoDetectorParams::for_board(&board);
    params.graph.max_spacing_pix = GRAPH_MAX_SPACING_PX;
    params
}

/// Load stereo ChArUco dataset from `base_dir/cam1` and `base_dir/cam2`.
///
/// Pairs are formed by matching filename suffixes:
/// - left: `Cam1_<suffix>.png`
/// - right: `Cam2_<suffix>.png`
pub fn load_stereo_charuco_input_with_progress<F>(
    base_dir: &Path,
    chess_config: &ChessConfig,
    charuco_params: &CharucoDetectorParams,
    max_views: Option<usize>,
    mut progress: F,
) -> Result<(RigExtrinsicsInput, StereoCharucoDatasetSummary)>
where
    F: FnMut(usize, usize, &str),
{
    let left_dir = base_dir.join("cam1");
    let right_dir = base_dir.join("cam2");
    ensure!(
        left_dir.exists(),
        "left camera folder not found: {}",
        left_dir.display()
    );
    ensure!(
        right_dir.exists(),
        "right camera folder not found: {}",
        right_dir.display()
    );

    let mut suffixes = list_stereo_pair_suffixes(&left_dir, &right_dir)?;
    if let Some(max) = max_views {
        ensure!(max > 0, "--max-views must be > 0");
        suffixes.truncate(max);
    }
    ensure!(
        !suffixes.is_empty(),
        "no usable stereo pairs found in {}",
        base_dir.display()
    );

    let total_pairs = suffixes.len();
    let mut views = Vec::new();
    let mut usable_left = 0usize;
    let mut usable_right = 0usize;
    let mut skipped_views = 0usize;

    for (pair_idx, suffix) in suffixes.iter().enumerate() {
        progress(pair_idx + 1, total_pairs, suffix);

        let left_path = left_dir.join(format!("Cam1_{suffix}"));
        let right_path = right_dir.join(format!("Cam2_{suffix}"));
        ensure!(
            left_path.exists() && right_path.exists(),
            "missing stereo pair: left={}, right={}",
            left_path.display(),
            right_path.display()
        );

        let left = detect_view(&left_path, chess_config, charuco_params)
            .with_context(|| format!("left detection failed for {}", left_path.display()))?;
        let right = detect_view(&right_path, chess_config, charuco_params)
            .with_context(|| format!("right detection failed for {}", right_path.display()))?;

        if left.is_none() && right.is_none() {
            skipped_views += 1;
            continue;
        }
        if left.is_some() {
            usable_left += 1;
        }
        if right.is_some() {
            usable_right += 1;
        }

        use vision_calibration::core::{NoMeta, RigView, RigViewObs};
        views.push(RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![left, right],
            },
        });
    }

    ensure!(
        usable_left >= 3 && usable_right >= 3,
        "need >=3 usable views per camera (left={}, right={})",
        usable_left,
        usable_right
    );

    let summary = StereoCharucoDatasetSummary {
        total_pairs,
        used_views: views.len(),
        skipped_views,
        usable_left,
        usable_right,
    };

    use vision_calibration::core::RigDataset;
    let input = RigDataset::new(views, 2).context("failed to create rig dataset")?;

    Ok((input, summary))
}

fn list_stereo_pair_suffixes(left_dir: &Path, right_dir: &Path) -> Result<Vec<String>> {
    fn collect_suffixes(dir: &Path, prefix: &str) -> Result<BTreeSet<String>> {
        let mut out = BTreeSet::new();
        for entry in
            std::fs::read_dir(dir).with_context(|| format!("failed to list {}", dir.display()))?
        {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.ends_with(".png") {
                continue;
            }
            let Some(suffix) = name.strip_prefix(prefix) else {
                continue;
            };
            out.insert(suffix.to_owned());
        }
        Ok(out)
    }

    let left = collect_suffixes(left_dir, "Cam1_")?;
    let right = collect_suffixes(right_dir, "Cam2_")?;

    Ok(left.intersection(&right).cloned().collect())
}

fn detect_view(
    path: &Path,
    chess_config: &ChessConfig,
    charuco_params: &CharucoDetectorParams,
) -> Result<Option<CorrespondenceView>> {
    let img = image::ImageReader::open(path)
        .with_context(|| format!("failed to read image {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_luma8();

    let detection = match detect::detect_charuco(&img, chess_config, charuco_params.clone()) {
        Ok(detection) => detection,
        Err(_) => return Ok(None),
    };

    detection_to_view_data(detection)
}

fn detection_to_view_data(
    detection: calib_targets::charuco::CharucoDetectionResult,
) -> Result<Option<CorrespondenceView>> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();
    for corner in detection.detection.corners {
        let Some(target) = corner.target_position else {
            continue;
        };
        points_3d.push(Pt3::new(target.x as f64, target.y as f64, 0.0));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    if points_3d.len() < 4 {
        return Ok(None);
    }

    Ok(Some(CorrespondenceView::new(points_3d, points_2d)?))
}
