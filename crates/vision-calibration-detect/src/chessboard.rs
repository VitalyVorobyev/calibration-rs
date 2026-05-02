//! Chessboard detector wrapping `chess-corners` + `calib-targets`.
//!
//! The detector returns one [`Feature`] per detected interior corner
//! whose grid-coordinate disambiguation succeeded. Corners without a
//! grid index are filtered out — calibration consumes 2D-3D
//! correspondences, so an unindexed corner is unusable.

use anyhow::{Result, anyhow};
use calib_targets::chessboard::DetectorParams as ChessboardDetectorParams;
use calib_targets::detect;
use chess_corners::ChessConfig;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::{Detector, Feature};

/// Chessboard detector configuration. Mirrors the shape of the
/// chessboard variant in
/// [`vision_calibration_dataset::TargetSpec`] so the dispatcher can
/// translate one to the other directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ChessboardConfig {
    /// Number of interior corners along the rows axis.
    pub rows: u32,
    /// Number of interior corners along the cols axis.
    pub cols: u32,
    /// Edge length of one square in metres. Used to lift each detected
    /// corner from grid index `(i, j)` to its 3D point
    /// `(i * square_size_m, j * square_size_m, 0)`.
    pub square_size_m: f64,
}

/// Stateless chessboard detector instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct ChessboardDetector;

impl Detector for ChessboardDetector {
    fn name(&self) -> &'static str {
        "chessboard"
    }

    fn detect_json(&self, image: &image::DynamicImage, config: &Value) -> Result<Vec<Feature>> {
        let cfg: ChessboardConfig = serde_json::from_value(config.clone())
            .map_err(|e| anyhow!("invalid chessboard config: {e}"))?;
        let luma = image.to_luma8();

        // The underlying detector auto-labels corners from
        // intersection clustering — `rows`/`cols` from our config are
        // used only for output validation, not as input parameters.
        // `chess-corners` provides defaults tuned for typical
        // 5–13 mm calibration boards.
        let _chess_config = ChessConfig::default();
        let board_params = ChessboardDetectorParams::default();

        let detection = detect::detect_chessboard(&luma, &board_params);
        let Some(detection) = detection else {
            return Ok(Vec::new());
        };

        // Filter to corners whose grid index falls inside the expected
        // `rows × cols` range, then lift to 3D using the supplied
        // square size.
        let max_i = cfg.rows as i32;
        let max_j = cfg.cols as i32;
        let mut features = Vec::new();
        for corner in detection.target.corners {
            let Some(grid) = corner.grid else { continue };
            if grid.i < 0 || grid.j < 0 || grid.i >= max_i || grid.j >= max_j {
                continue;
            }
            features.push(Feature {
                image_xy: [corner.position.x as f64, corner.position.y as f64],
                world_xyz: [
                    grid.i as f64 * cfg.square_size_m,
                    grid.j as f64 * cfg.square_size_m,
                    0.0,
                ],
            });
        }
        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn invalid_config_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let det = ChessboardDetector;
        let err = det
            .detect_json(&img, &json!({"rows": 9})) // missing cols, square_size_m
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid chessboard config"), "got: {msg}");
    }

    #[test]
    fn empty_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let det = ChessboardDetector;
        let cfg = json!({ "rows": 9, "cols": 6, "square_size_m": 0.025 });
        let features = det.detect_json(&img, &cfg).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }
}
