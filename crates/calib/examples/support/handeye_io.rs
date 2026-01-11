//! Shared I/O helpers for the KUKA hand-eye examples.

use anyhow::{ensure, Context, Result};
use calib::core::{Iso3, Pt3, Vec2};
use calib::pipeline::PlanarViewData;
use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;
use image::ImageReader;
use nalgebra::{Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ViewSample {
    pub view: PlanarViewData,
    pub robot_pose: Iso3,
    #[allow(dead_code)]
    pub image_index: usize,
    #[allow(dead_code)]
    pub num_corners: usize,
}

#[derive(Debug, Clone)]
pub struct DatasetSummary {
    pub total_images: usize,
    pub used_views: usize,
    pub skipped_views: usize,
    pub total_corners: usize,
    pub square_size_m: f64,
}

pub fn kuka_chessboard_params() -> ChessboardParams {
    ChessboardParams {
        expected_rows: Some(17),
        expected_cols: Some(28),
        ..ChessboardParams::default()
    }
}

pub fn load_kuka_dataset(
    base_path: &Path,
    chess_config: &ChessConfig,
    board_params: &ChessboardParams,
) -> Result<(Vec<ViewSample>, DatasetSummary)> {
    load_kuka_dataset_with_progress(base_path, chess_config, board_params, |_, _| {})
}

pub fn load_kuka_dataset_with_progress<F>(
    base_path: &Path,
    chess_config: &ChessConfig,
    board_params: &ChessboardParams,
    mut progress: F,
) -> Result<(Vec<ViewSample>, DatasetSummary)>
where
    F: FnMut(usize, usize),
{
    ensure!(
        base_path.exists(),
        "dataset not found at {}",
        base_path.display()
    );

    let square_size_m = load_square_size_m(&base_path.join("squaresize.txt"))?;
    let robot_poses = load_robot_poses(&base_path.join("RobotPosesVec.txt"))?;

    let mut samples = Vec::new();
    let mut total_corners = 0usize;

    for (idx, robot_pose) in robot_poses.iter().enumerate() {
        let image_index = idx + 1;
        progress(image_index, robot_poses.len());
        let img_path = base_path.join(format!("{:02}.png", image_index));
        let img = ImageReader::open(&img_path)
            .with_context(|| format!("failed to read image {}", img_path.display()))?
            .decode()
            .with_context(|| format!("failed to decode {}", img_path.display()))?
            .to_luma8();

        let detection = match detect::detect_chessboard(&img, chess_config, board_params.clone()) {
            Some(result) => result,
            None => {
                eprintln!("Skipping view {:02}: chessboard not detected", image_index);
                continue;
            }
        };

        let view = detection_to_view_data(detection, square_size_m)?;
        let num_corners = view.points_2d.len();
        total_corners += num_corners;

        samples.push(ViewSample {
            view,
            robot_pose: *robot_pose,
            image_index,
            num_corners,
        });
    }

    let summary = DatasetSummary {
        total_images: robot_poses.len(),
        used_views: samples.len(),
        skipped_views: robot_poses.len().saturating_sub(samples.len()),
        total_corners,
        square_size_m,
    };

    Ok((samples, summary))
}

fn load_square_size_m(path: &Path) -> Result<f64> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read square size from {}", path.display()))?;
    let raw = text.trim().to_lowercase();
    ensure!(!raw.is_empty(), "square size file is empty");

    if let Some(mm) = raw.strip_suffix("mm") {
        let value: f64 = mm.trim().parse()?;
        return Ok(value / 1000.0);
    }

    if let Some(m) = raw.strip_suffix('m') {
        let value: f64 = m.trim().parse()?;
        return Ok(value);
    }

    Ok(raw.parse()?)
}

fn parse_pose_line(line: &str, idx: usize) -> Result<Iso3> {
    let values: Vec<f64> = line
        .split_whitespace()
        .map(|v| v.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("invalid float in robot pose line {}", idx + 1))?;
    ensure!(
        values.len() == 16,
        "robot pose line {} expected 16 values, got {}",
        idx + 1,
        values.len()
    );

    let r = Matrix3::new(
        values[0], values[1], values[2], values[4], values[5], values[6], values[8], values[9],
        values[10],
    );
    let t = Vector3::new(values[3], values[7], values[11]);
    let rot = Rotation3::from_matrix_unchecked(r);
    Ok(Iso3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_rotation_matrix(&rot),
    ))
}

fn load_robot_poses(path: &Path) -> Result<Vec<Iso3>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read robot poses from {}", path.display()))?;
    let mut poses = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        poses.push(parse_pose_line(line, idx)?);
    }
    ensure!(!poses.is_empty(), "robot pose file is empty");
    Ok(poses)
}

fn detection_to_view_data(
    detection: ChessboardDetectionResult,
    square_size_m: f64,
) -> Result<PlanarViewData> {
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
        points_2d.push(Vec2::new(
            corner.position.x as f64,
            corner.position.y as f64,
        ));
    }

    ensure!(
        points_3d.len() >= 4,
        "insufficient corners after grid filtering"
    );

    Ok(PlanarViewData {
        points_3d,
        points_2d,
        weights: None,
    })
}
