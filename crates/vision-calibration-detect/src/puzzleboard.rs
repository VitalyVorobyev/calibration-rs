//! PuzzleBoard detector wrapping `calib-targets` (Stelldinger 2024).
//!
//! A PuzzleBoard is a self-identifying chessboard: each saddle is decoded
//! to a unique grid index from a local code window, so a single partial
//! view yields globally-consistent correspondences. The detector returns
//! one [`Feature`] per decoded saddle that falls inside the printed board,
//! with metric target coordinates in the board's own (top-left-origin)
//! frame — the same origin convention as the chessboard/charuco detectors,
//! not the centred frame used by the private bench examples.
//!
//! Two fixed parts of the recipe mirror the proven bench/example path
//! (`bench/src/detect.rs::detect_puzzleboard_view`): a 2× upscale before
//! detection (PuzzleBoard decoding is sensitive to saddle sharpness on
//! small tiles) and `FixedBoard` search (the board dimensions are known
//! from the manifest, so we never search for an unknown sub-board).

use calib_targets::detect;
use calib_targets::puzzleboard::{PuzzleBoardParams, PuzzleBoardSearchMode, PuzzleBoardSpec};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::{DetectError, Detector, Feature};

/// Upscale factor applied before detection. PuzzleBoard saddle decoding
/// degrades on small/blurred tiles; the bench and private examples both
/// resize 2× first, then divide detected pixel coordinates back down.
const DETECT_UPSCALE: u32 = 2;

/// PuzzleBoard detector configuration.
///
/// Unlike chessboard/charuco, the manifest's
/// `vision_calibration_dataset::TargetSpec::Puzzleboard` carries a named
/// `layout` string (e.g. `"puzzle_130x130"`) rather than explicit
/// dimensions; the pipeline dispatcher resolves that name to the
/// `rows`/`cols` this detector consumes. Keeping the detector parametric
/// (an R×C board with a cell size) decouples it from the manifest's
/// layout-naming convention.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PuzzleboardConfig {
    /// Number of cells along the rows axis.
    pub rows: u32,
    /// Number of cells along the cols axis.
    pub cols: u32,
    /// Edge length of one cell in metres. Lifts each decoded saddle to a
    /// metric target point and bounds the in-board filter.
    pub cell_size_m: f64,
}

/// Stateless PuzzleBoard detector instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct PuzzleboardDetector;

impl crate::sealed::Sealed for PuzzleboardDetector {}

impl Detector for PuzzleboardDetector {
    fn name(&self) -> &'static str {
        "puzzleboard"
    }

    fn detect_json(
        &self,
        image: &image::DynamicImage,
        config: &Value,
    ) -> Result<Vec<Feature>, DetectError> {
        let cfg: PuzzleboardConfig =
            serde_json::from_value(config.clone()).map_err(|e| DetectError::Config {
                detector: "puzzleboard",
                source: e,
            })?;

        // The detector decodes saddles in millimetres; build the spec in
        // mm to match the bench/example tolerances, and convert metric
        // outputs back to metres below.
        let cell_size_mm = (cfg.cell_size_m * 1000.0) as f32;
        let spec = PuzzleBoardSpec::new(cfg.rows, cfg.cols, cell_size_mm)
            .map_err(|e| DetectError::InvalidConfig(format!("invalid puzzleboard spec: {e}")))?;
        let mut params = PuzzleBoardParams::for_board(&spec);
        params.decode.search_all_components = false;
        params.decode.search_mode = PuzzleBoardSearchMode::FixedBoard;

        let luma = image.to_luma8();
        let detection_image = image::imageops::resize(
            &luma,
            luma.width().saturating_mul(DETECT_UPSCALE),
            luma.height().saturating_mul(DETECT_UPSCALE),
            FilterType::Triangle,
        );

        // No board in frame is "no features", not an error — same
        // semantics as the chessboard/charuco detectors on a blank image.
        let detection = match detect::detect_puzzleboard(&detection_image, &params) {
            Ok(detection) => detection,
            Err(_) => return Ok(Vec::new()),
        };

        // Decoded saddles outside the printed board are spurious; bound
        // them to `[0, (n-1) * cell]` mm before lifting to metres.
        let cell_size_mm = cfg.cell_size_m * 1000.0;
        let max_x_mm = cfg.cols.saturating_sub(1) as f64 * cell_size_mm;
        let max_y_mm = cfg.rows.saturating_sub(1) as f64 * cell_size_mm;

        let mut features = Vec::with_capacity(detection.corners.len());
        for corner in &detection.corners {
            let x_mm = corner.target_position.x as f64;
            let y_mm = corner.target_position.y as f64;
            if !(0.0..=max_x_mm).contains(&x_mm) || !(0.0..=max_y_mm).contains(&y_mm) {
                continue;
            }
            features.push(Feature {
                image_xy: [
                    corner.position.x as f64 / DETECT_UPSCALE as f64,
                    corner.position.y as f64 / DETECT_UPSCALE as f64,
                ],
                world_xyz: [x_mm * 1.0e-3, y_mm * 1.0e-3, 0.0],
            });
        }
        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const CELL_MM: f64 = 12.0;
    const ROWS: u32 = 12;
    const COLS: u32 = 12;

    fn board_config() -> Value {
        json!({
            "rows": ROWS,
            "cols": COLS,
            "cell_size_m": CELL_MM / 1000.0,
        })
    }

    /// Render a synthetic frontal PuzzleBoard via calib-targets' printable
    /// pipeline and decode it back into a `DynamicImage`.
    fn synthetic_board_image() -> image::DynamicImage {
        let document = calib_targets::generate::puzzleboard_document(ROWS, COLS, CELL_MM);
        let bundle =
            calib_targets::printable::render_target_bundle(&document).expect("render bundle");
        image::load_from_memory(&bundle.png_bytes).expect("decode rendered PNG")
    }

    #[test]
    fn detects_synthetic_board() {
        let img = synthetic_board_image();
        let features = PuzzleboardDetector
            .detect_json(&img, &board_config())
            .unwrap();
        assert!(
            features.len() >= 8,
            "expected >= 8 saddles on a clean frontal board, got {}",
            features.len()
        );
        let cell_m = CELL_MM / 1000.0;
        let max_x = (COLS - 1) as f64 * cell_m;
        let max_y = (ROWS - 1) as f64 * cell_m;
        for f in &features {
            assert_eq!(f.world_xyz[2], 0.0, "target is planar");
            assert!(
                (0.0..=max_x + 1e-9).contains(&f.world_xyz[0])
                    && (0.0..=max_y + 1e-9).contains(&f.world_xyz[1]),
                "decoded saddle {:?} outside the printed board",
                f.world_xyz
            );
        }
    }

    #[test]
    fn invalid_config_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let err = PuzzleboardDetector
            .detect_json(&img, &json!({"rows": 12})) // missing cols, cell_size_m
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid puzzleboard config"), "got: {msg}");
    }

    #[test]
    fn blank_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let features = PuzzleboardDetector
            .detect_json(&img, &board_config())
            .unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg: PuzzleboardConfig = serde_json::from_value(board_config()).unwrap();
        let back: PuzzleboardConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(cfg, back);
    }
}
