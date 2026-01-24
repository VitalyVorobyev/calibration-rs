//! Shared I/O helpers for the stereo rig calibration examples.

use anyhow::{ensure, Context, Result};
use calib::prelude::*;
use calib::rig_extrinsics::RigExtrinsicsInput;
use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;
use std::collections::BTreeSet;
use std::path::Path;

/// Summary of dataset loading.
#[derive(Debug, Clone)]
pub struct StereoDatasetSummary {
    pub total_pairs: usize,
    pub used_views: usize,
    pub skipped_views: usize,
    pub usable_left: usize,
    pub usable_right: usize,
}

/// Load stereo dataset from images directory.
///
/// Expects `imgs_dir/leftcamera/Im_L_N.png` and `imgs_dir/rightcamera/Im_R_N.png`
/// pairs.
pub fn load_stereo_input_with_progress<F>(
    imgs_dir: &Path,
    chess_config: &ChessConfig,
    board_params: &ChessboardParams,
    square_size_m: f64,
    max_views: Option<usize>,
    mut progress: F,
) -> Result<(RigExtrinsicsInput, StereoDatasetSummary)>
where
    F: FnMut(usize, usize, usize),
{
    let left_dir = imgs_dir.join("leftcamera");
    let right_dir = imgs_dir.join("rightcamera");
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

    let mut indices = list_stereo_pair_indices(&left_dir, &right_dir)?;
    if let Some(max) = max_views {
        ensure!(max > 0, "--max-views must be > 0");
        indices.truncate(max);
    }
    ensure!(
        !indices.is_empty(),
        "no usable stereo pairs found in {}",
        imgs_dir.display()
    );

    let total_pairs = indices.len();
    let mut views = Vec::new();
    let mut usable_left = 0usize;
    let mut usable_right = 0usize;
    let mut skipped_views = 0usize;

    for (pair_idx, image_index) in indices.iter().enumerate() {
        progress(pair_idx + 1, total_pairs, *image_index);

        let left_path = left_dir.join(format!("Im_L_{image_index}.png"));
        let right_path = right_dir.join(format!("Im_R_{image_index}.png"));
        ensure!(
            left_path.exists() && right_path.exists(),
            "missing stereo pair: left={}, right={}",
            left_path.display(),
            right_path.display()
        );

        let left = detect_view(
            &left_path,
            chess_config,
            board_params.clone(),
            square_size_m,
        )
        .with_context(|| format!("left detection failed for {}", left_path.display()))?;
        let right = detect_view(
            &right_path,
            chess_config,
            board_params.clone(),
            square_size_m,
        )
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

        // Create RigViewObs with cameras = [left, right]
        use calib::core::{NoMeta, RigView, RigViewObs};
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

    let summary = StereoDatasetSummary {
        total_pairs,
        used_views: views.len(),
        skipped_views,
        usable_left,
        usable_right,
    };

    use calib::core::RigDataset;
    let input = RigDataset::new(views, 2).context("failed to create rig dataset")?;

    Ok((input, summary))
}

fn list_stereo_pair_indices(left_dir: &Path, right_dir: &Path) -> Result<Vec<usize>> {
    fn parse_idx(file_name: &str, prefix: &str) -> Option<usize> {
        let name = file_name.strip_suffix(".png")?;
        let raw = name.strip_prefix(prefix)?;
        raw.parse().ok()
    }

    let mut left = BTreeSet::new();
    for entry in std::fs::read_dir(left_dir)
        .with_context(|| format!("failed to list {}", left_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if let Some(idx) = parse_idx(name, "Im_L_") {
            left.insert(idx);
        }
    }

    let mut right = BTreeSet::new();
    for entry in std::fs::read_dir(right_dir)
        .with_context(|| format!("failed to list {}", right_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if let Some(idx) = parse_idx(name, "Im_R_") {
            right.insert(idx);
        }
    }

    let indices: Vec<usize> = left.intersection(&right).copied().collect();
    Ok(indices)
}

fn detect_view(
    path: &Path,
    chess_config: &ChessConfig,
    board_params: ChessboardParams,
    square_size_m: f64,
) -> Result<Option<CorrespondenceView>> {
    let img = image::ImageReader::open(path)
        .with_context(|| format!("failed to read image {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))?
        .to_luma8();

    let detection = detect::detect_chessboard(&img, chess_config, board_params);
    let Some(detection) = detection else {
        return Ok(None);
    };
    Ok(Some(detection_to_view_data(detection, square_size_m)?))
}

fn detection_to_view_data(
    detection: ChessboardDetectionResult,
    square_size_m: f64,
) -> Result<CorrespondenceView> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();
    for corner in detection.detection.corners {
        let Some(grid) = corner.grid else {
            continue;
        };
        points_3d.push(Pt3::new(
            grid.i as f64 * square_size_m,
            grid.j as f64 * square_size_m,
            0.0,
        ));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    ensure!(
        points_3d.len() >= 4,
        "insufficient corners after grid filtering"
    );

    CorrespondenceView::new(points_3d, points_2d)
}
