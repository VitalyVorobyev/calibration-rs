//! Tier-B detection adapters: wrap `calib-targets` for use in the bench harness.
//!
//! This is a thin port of the detection logic in
//! `crates/vision-calibration/examples/planar_real.rs` and its `support`
//! modules. It deliberately reuses the same `calib_targets::detect::detect_chessboard`
//! call and the same 2D↔3D correspondence construction so that bench numbers
//! reproduce the example's reprojection error. Do not introduce a new detector
//! here.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use calib_targets::aruco::builtins;
use calib_targets::charuco::{CharucoBoardSpec, CharucoParams, MarkerLayout};
use calib_targets::chessboard::{ChessboardDetection, DetectorParams, GraphBuildAlgorithm};
use calib_targets::detect::{self, default_chess_config};
use calib_targets::puzzleboard::{PuzzleBoardParams, PuzzleBoardSearchMode, PuzzleBoardSpec};
use image::imageops::FilterType;
use vision_calibration_core::{CorrespondenceView, Pt2, Pt3};

/// Which detector a rig camera uses to find target features in an image.
///
/// Chosen per registry entry from the [`crate::registry::BoardGeometry::layout`]
/// string so the multi-camera rig runner ([`crate::run::run_rig_extrinsics`])
/// can dispatch chessboard vs ChArUco without duplicating the per-view loop.
/// Both arms return the same [`CorrespondenceView`] shape; the ChArUco arm is a
/// thin port of `stereo_charuco_session`'s `support/stereo_charuco_io.rs`.
#[derive(Clone)]
pub enum DetectorKind {
    /// `calib-targets` chessboard detector. `square_size_m` is the metric cell
    /// pitch; the detector auto-discovers the interior-corner grid, and
    /// `require_known_grid` can reject detections that do not match the
    /// registry dimensions exactly.
    Chessboard {
        /// Expected interior-corner rows.
        rows: usize,
        /// Expected interior-corner columns.
        cols: usize,
        /// Reject detections that do not match the known grid exactly.
        require_known_grid: bool,
        /// Metric square (cell) size in metres.
        square_size_m: f64,
    },
    /// `calib-targets` ChArUco detector, parameterised by a board spec. Mirrors
    /// `stereo_charuco_session`'s detector exactly (same dict, marker scale,
    /// `OpenCvCharuco` layout) so bench numbers reproduce the example.
    Charuco(Box<CharucoParams>),
    /// `calib-targets` PuzzleBoard detector in known-board mode. `rows`/`cols`
    /// are square counts; output 3D points are centred in the printed board
    /// frame and expressed in metres.
    Puzzleboard(Box<PuzzleBoardParams>),
}

impl DetectorKind {
    /// Detect target features in one decoded image, returning `Ok(None)` when
    /// no usable board is found (so the caller counts a skipped image).
    pub fn detect(&self, img: &image::DynamicImage) -> Result<Option<CorrespondenceView>> {
        match self {
            DetectorKind::Chessboard {
                rows,
                cols,
                require_known_grid,
                square_size_m,
            } => detect_chessboard_view(img, *rows, *cols, *square_size_m, *require_known_grid),
            DetectorKind::Charuco(params) => detect_charuco_view(img, params),
            DetectorKind::Puzzleboard(params) => detect_puzzleboard_view(img, params),
        }
    }
}

/// Build the ChArUco detector parameters used by `stereo_charuco_session`.
///
/// `rows`/`cols` are the board square counts, `cell_size_m` the metric cell
/// pitch (metres), `marker_size_rel` the marker-to-cell scale, and `dictionary`
/// the ArUco dictionary name (only `DICT_4X4_1000` is wired, matching the
/// example). Returns an error for an unknown dictionary so a registry typo
/// surfaces loudly rather than silently mis-detecting.
pub fn charuco_params_for(
    rows: u32,
    cols: u32,
    cell_size_m: f64,
    marker_size_rel: f32,
    dictionary: &str,
) -> Result<CharucoParams> {
    let dict = match dictionary {
        "DICT_4X4_1000" => builtins::DICT_4X4_1000,
        other => anyhow::bail!("unsupported ChArUco dictionary '{other}' (only DICT_4X4_1000)"),
    };
    let board = CharucoBoardSpec {
        rows,
        cols,
        cell_size: cell_size_m as f32,
        marker_size_rel,
        dictionary: dict,
        marker_layout: MarkerLayout::OpenCvCharuco,
    };
    Ok(CharucoParams::for_board(&board))
}

/// Build PuzzleBoard detector parameters for a known printed board.
///
/// `rows`/`cols` are square counts, and `cell_size_m` is converted to the
/// millimetre unit expected by `calib-targets-puzzleboard`.
pub fn puzzleboard_params_for(rows: u32, cols: u32, cell_size_m: f64) -> Result<PuzzleBoardParams> {
    let cell_size_mm = (cell_size_m * 1000.0) as f32;
    let spec = PuzzleBoardSpec::with_origin(rows, cols, cell_size_mm, 0, 0)
        .map_err(|e| anyhow::anyhow!("puzzleboard spec: {e}"))?;
    let mut params = PuzzleBoardParams::for_board(&spec);
    params.decode.search_all_components = false;
    params.decode.search_mode = PuzzleBoardSearchMode::FixedBoard;
    Ok(params)
}

/// Detect ChArUco corners in one decoded image and build a [`CorrespondenceView`].
///
/// Thin port of `stereo_charuco_io.rs::detect_view` + `detection_to_view_data`:
/// converts to luma8, runs `detect_charuco`, then maps each corner's metric
/// `target_position` (`x`, `y`, `z=0`) to its pixel position. The ChArUco
/// detector returns *sparse* correspondences (not every cell is seen), so the
/// resulting view has a variable length; `< 4` corners is treated as a skip
/// (`Ok(None)`), exactly like the example.
pub fn detect_charuco_view(
    img: &image::DynamicImage,
    params: &CharucoParams,
) -> Result<Option<CorrespondenceView>> {
    let luma = img.to_luma8();
    let detection = match detect::detect_charuco(&luma, params) {
        Ok(detection) => detection,
        Err(_) => return Ok(None),
    };

    let mut points_3d = Vec::with_capacity(detection.corners.len());
    let mut points_2d = Vec::with_capacity(detection.corners.len());
    for corner in detection.corners {
        let target = corner.target_position;
        points_3d.push(Pt3::new(target.x as f64, target.y as f64, 0.0));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    if points_3d.len() < 4 {
        return Ok(None);
    }

    Ok(Some(
        CorrespondenceView::new(points_3d, points_2d).map_err(|e| anyhow::anyhow!("{e}"))?,
    ))
}

/// Detect a PuzzleBoard in one decoded image and build a [`CorrespondenceView`].
///
/// Mirrors the private 130x130 example's known-board path: resize 2x before
/// detection, force `FixedBoard` via [`puzzleboard_params_for`], drop decoded
/// corners outside the declared printed board, and convert target coordinates
/// from millimetres to centred metres.
pub fn detect_puzzleboard_view(
    img: &image::DynamicImage,
    params: &PuzzleBoardParams,
) -> Result<Option<CorrespondenceView>> {
    const UPSCALE: u32 = 2;
    let luma = img.to_luma8();
    let detection_image = image::imageops::resize(
        &luma,
        luma.width() * UPSCALE,
        luma.height() * UPSCALE,
        FilterType::Triangle,
    );
    let result = match detect::detect_puzzleboard(&detection_image, params) {
        Ok(result) => result,
        Err(_) => return Ok(None),
    };

    let rows = params.board.rows;
    let cols = params.board.cols;
    let cell_size_mm = params.board.cell_size as f64;
    let origin_x_mm = 0.5 * (cols.saturating_sub(1) as f64) * cell_size_mm;
    let origin_y_mm = 0.5 * (rows.saturating_sub(1) as f64) * cell_size_mm;
    let max_x_mm = cols.saturating_sub(1) as f64 * cell_size_mm;
    let max_y_mm = rows.saturating_sub(1) as f64 * cell_size_mm;

    let mut points_3d = Vec::with_capacity(result.corners.len());
    let mut points_2d = Vec::with_capacity(result.corners.len());
    for corner in &result.corners {
        let x_mm = corner.target_position.x as f64;
        let y_mm = corner.target_position.y as f64;
        if !(0.0..=max_x_mm).contains(&x_mm) || !(0.0..=max_y_mm).contains(&y_mm) {
            continue;
        }
        points_3d.push(Pt3::new(
            (x_mm - origin_x_mm) * 1.0e-3,
            (y_mm - origin_y_mm) * 1.0e-3,
            0.0,
        ));
        points_2d.push(Pt2::new(
            corner.position.x as f64 / UPSCALE as f64,
            corner.position.y as f64 / UPSCALE as f64,
        ));
    }

    if points_3d.len() < 8 {
        return Ok(None);
    }

    Ok(Some(
        CorrespondenceView::new(points_3d, points_2d).map_err(|e| anyhow::anyhow!("{e}"))?,
    ))
}

/// Detect a chessboard in one decoded image and build a [`CorrespondenceView`].
///
/// Mirrors `planar_real.rs::detect_chessboard` + `detection_to_view`:
/// converts to luma8, runs `detect_chessboard` with the default chess config and
/// the supplied [`DetectorParams`], then maps each detected corner's grid index
/// to a metric `(i*square, j*square, 0)` target point and its pixel position.
///
/// Returns `Ok(None)` when no board is found (so the caller can count it as a
/// skipped image), `Ok(Some(view))` on success. When `require_known_grid` is
/// true, detections must contain exactly `rows * cols` corners spanning the
/// declared grid. This prevents partial, locally indexed chessboard detections
/// from moving the target frame between hand-eye views.
pub fn detect_chessboard_view(
    img: &image::DynamicImage,
    rows: usize,
    cols: usize,
    square_size_m: f64,
    require_known_grid: bool,
) -> Result<Option<CorrespondenceView>> {
    let luma = img.to_luma8();
    let params = topological_chessboard_params();
    let Some(detection) = detect::detect_chessboard(&luma, &default_chess_config(), &params) else {
        return Ok(None);
    };
    if require_known_grid && !detection_matches_known_grid(&detection, rows, cols) {
        return Ok(None);
    }
    Ok(Some(detection_to_view(detection, square_size_m)?))
}

fn topological_chessboard_params() -> DetectorParams {
    let mut params = DetectorParams::default();
    params.graph_build_algorithm = GraphBuildAlgorithm::Topological;
    params
}

fn detection_matches_known_grid(detection: &ChessboardDetection, rows: usize, cols: usize) -> bool {
    if detection.corners.len() != rows * cols {
        return false;
    }
    let max_i = detection.corners.iter().map(|c| c.grid.i).max();
    let max_j = detection.corners.iter().map(|c| c.grid.j).max();
    matches!(
        (max_i, max_j),
        (Some(i), Some(j)) if i >= 0
            && j >= 0
            && (i as usize) + 1 == cols
            && (j as usize) + 1 == rows
    )
}

/// Map a [`ChessboardDetection`] to a [`CorrespondenceView`] using the board's
/// metric square size. Identical convention to the example
/// (`grid.i * square`, `grid.j * square`, `z = 0`).
fn detection_to_view(
    detection: ChessboardDetection,
    square_size_m: f64,
) -> Result<CorrespondenceView> {
    let mut points_3d = Vec::with_capacity(detection.corners.len());
    let mut points_2d = Vec::with_capacity(detection.corners.len());
    for corner in detection.corners {
        let grid = corner.grid;
        points_3d.push(Pt3::new(
            grid.i as f64 * square_size_m,
            grid.j as f64 * square_size_m,
            0.0,
        ));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    anyhow::ensure!(
        points_3d.len() >= 4,
        "insufficient corners after grid filtering"
    );

    CorrespondenceView::new(points_3d, points_2d).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Glob a folder for image files matching `pattern` and return absolute paths
/// sorted in natural (numeric-aware) order.
///
/// `folder` is the directory to search; `pattern` is a filename glob such as
/// `*.png` (joined onto `folder`). Sorting is numeric-aware so `Im_L_2.png`
/// precedes `Im_L_10.png`, matching the index-sorted order the example uses.
pub fn glob_sorted_images(folder: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let glob_pat = folder.join(pattern);
    let glob_pat = glob_pat
        .to_str()
        .with_context(|| format!("non-UTF-8 glob path: {}", glob_pat.display()))?;
    let mut paths: Vec<PathBuf> = glob::glob(glob_pat)
        .with_context(|| format!("invalid glob pattern: {glob_pat}"))?
        .filter_map(|e| e.ok())
        .filter(|p| p.is_file())
        .collect();
    paths.sort_by(|a, b| natural_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
    Ok(paths)
}

/// Load and decode one image from disk.
pub fn load_image(path: &Path) -> Result<image::DynamicImage> {
    image::ImageReader::open(path)
        .with_context(|| format!("failed to open {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode {}", path.display()))
}

/// Natural (numeric-aware) string comparison: splits each string into runs of
/// digits and non-digits, comparing digit runs by numeric value. Keeps
/// `Im_L_2` < `Im_L_10` so globbed image order matches the example's
/// index-sorted order.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();
    loop {
        match (ai.peek().copied(), bi.peek().copied()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(ca), Some(cb)) => {
                if ca.is_ascii_digit() && cb.is_ascii_digit() {
                    let na: String = take_digits(&mut ai);
                    let nb: String = take_digits(&mut bi);
                    // Compare by numeric value; fall back to string length then
                    // lexically for runs too long to fit in u128.
                    let cmp = match (na.parse::<u128>(), nb.parse::<u128>()) {
                        (Ok(x), Ok(y)) => x.cmp(&y),
                        _ => na.len().cmp(&nb.len()).then_with(|| na.cmp(&nb)),
                    };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                } else {
                    let cmp = ca.cmp(&cb);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                    ai.next();
                    bi.next();
                }
            }
        }
    }
}

fn take_digits(it: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut s = String::new();
    while let Some(&c) = it.peek() {
        if c.is_ascii_digit() {
            s.push(c);
            it.next();
        } else {
            break;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_cmp_orders_numeric_suffixes() {
        let mut v = vec![
            "Im_L_10.png".to_string(),
            "Im_L_2.png".to_string(),
            "Im_L_1.png".to_string(),
        ];
        v.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(v, vec!["Im_L_1.png", "Im_L_2.png", "Im_L_10.png"]);
    }

    #[test]
    fn puzzleboard_params_use_fixed_known_board_mode() {
        let params = puzzleboard_params_for(130, 130, 0.001014).expect("params");
        assert_eq!(params.board.rows, 130);
        assert_eq!(params.board.cols, 130);
        assert!((params.board.cell_size - 1.014).abs() < 1.0e-6);
        assert!(!params.decode.search_all_components);
        assert!(matches!(
            params.decode.search_mode,
            PuzzleBoardSearchMode::FixedBoard
        ));
    }
}
