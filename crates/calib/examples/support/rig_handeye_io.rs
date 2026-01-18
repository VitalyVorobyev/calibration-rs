//! Shared I/O helpers for the multi-camera rig hand-eye calibration example.
//!
//! This module loads real calibration images from a 3-camera rig setup with
//! robot poses for hand-eye calibration scenarios.

use anyhow::{ensure, Context, Result};
use calib::core::{CorrespondenceView, Pt3, Real, Vec2};
use calib::rig_handeye::{RigHandEyeInput, RigHandEyeViewData};
use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;
use image::ImageReader;
use std::path::Path;

const NUM_CAMERAS: usize = 3;

#[derive(Debug, Clone)]
pub struct RigHandEyeDatasetSummary {
    pub total_views: usize,
    pub used_views: usize,
    pub skipped_views: usize,
    pub usable_per_camera: Vec<usize>,
}

fn image_filename(index: usize) -> String {
    format!("image{index}.png")
}

fn image2_filename(index: usize) -> String {
    format!("colorFrame_0_{index:05}.png")
}

pub fn load_rig_handeye_input_with_progress<F>(
    imgs_dir: &Path,
    chess_config: &ChessConfig,
    board_params: &ChessboardParams,
    square_size_m: Real,
    max_views: Option<usize>,
    mut progress: F,
) -> Result<(RigHandEyeInput, RigHandEyeDatasetSummary)>
where
    F: FnMut(usize, usize, usize),
{
    let cam0_dir = imgs_dir.join("camera0");
    let cam1_dir = imgs_dir.join("camera1");
    let cam2_dir = imgs_dir.join("camera2");
    ensure!(
        cam0_dir.exists(),
        "camera 0 folder not found: {}",
        cam0_dir.display()
    );
    ensure!(
        cam1_dir.exists(),
        "camera 1 folder not found: {}",
        cam1_dir.display()
    );
    ensure!(
        cam2_dir.exists(),
        "camera 2 folder not found: {}",
        cam2_dir.display()
    );

    let total_images = 42;
    let mut views = Vec::new();
    let mut skipped_views = 0usize;
    let mut usable_per_camera = vec![0usize; NUM_CAMERAS];

    for img_index in 0..total_images {
        if let Some(max) = max_views {
            if views.len() >= max {
                break;
            }
        }

        progress(img_index + 1, total_images, img_index);
        let img0_path = cam0_dir.join(image_filename(img_index));
        let img1_path = cam1_dir.join(image_filename(img_index));
        let img2_path = cam2_dir.join(image2_filename(img_index));

        ensure!(
            img0_path.exists(),
            "image not found: {}",
            img0_path.display()
        );
        ensure!(
            img1_path.exists(),
            "image not found: {}",
            img1_path.display()
        );
        ensure!(
            img2_path.exists(),
            "image not found: {}",
            img2_path.display()
        );

        let view0 = detect_view(&img0_path, chess_config, *board_params, square_size_m)
            .with_context(|| format!("failed to detect view for {}", img0_path.display()))?;
        let view1 = detect_view(&img1_path, chess_config, *board_params, square_size_m)
            .with_context(|| format!("failed to detect view for {}", img1_path.display()))?;
        let view2 = detect_view(&img2_path, chess_config, *board_params, square_size_m)
            .with_context(|| format!("failed to detect view for {}", img2_path.display()))?;

        // Skip if all cameras failed to detect
        if view0.is_none() && view1.is_none() && view2.is_none() {
            skipped_views += 1;
            continue;
        }

        // Count usable detections per camera
        if view0.is_some() {
            usable_per_camera[0] += 1;
        }
        if view1.is_some() {
            usable_per_camera[1] += 1;
        }
        if view2.is_some() {
            usable_per_camera[2] += 1;
        }

        // For hand-eye we need robot poses - placeholder identity for now
        // In real usage, these would come from robot controller or a poses file
        let base_from_gripper = calib::core::Iso3::identity();

        views.push(RigHandEyeViewData {
            cameras: vec![view0, view1, view2],
            base_from_gripper,
        });
    }

    ensure!(
        usable_per_camera.iter().all(|&c| c >= 3),
        "need >=3 usable views per camera (got {:?})",
        usable_per_camera
    );

    let summary = RigHandEyeDatasetSummary {
        total_views: total_images,
        used_views: views.len(),
        skipped_views,
        usable_per_camera,
    };

    let input = RigHandEyeInput {
        views,
        num_cameras: NUM_CAMERAS,
        mode: calib::optim::ir::HandEyeMode::EyeToHand,
    };

    Ok((input, summary))
}

fn detect_view(
    path: &Path,
    chess_config: &ChessConfig,
    board_params: ChessboardParams,
    square_size_m: Real,
) -> Result<Option<CorrespondenceView>> {
    let img = ImageReader::open(path)
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
    square_size_m: Real,
) -> Result<CorrespondenceView> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();
    for corner in detection.detection.corners {
        let Some(grid) = corner.grid else {
            continue;
        };
        points_3d.push(Pt3::new(
            grid.i as Real * square_size_m,
            grid.j as Real * square_size_m,
            0.0,
        ));
        points_2d.push(Vec2::new(
            corner.position.x as Real,
            corner.position.y as Real,
        ));
    }

    ensure!(
        points_3d.len() >= 4,
        "insufficient corners after grid filtering"
    );

    Ok(CorrespondenceView {
        points_3d,
        points_2d,
        weights: None,
    })
}
