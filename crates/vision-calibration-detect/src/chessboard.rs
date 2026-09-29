//! Chessboard detector wrapping `chess-corners` + `calib-targets`.
//!
//! The detector returns one [`Feature`] per detected interior corner
//! whose grid-coordinate disambiguation succeeded. Corners without a
//! grid index are filtered out — calibration consumes 2D-3D
//! correspondences, so an unindexed corner is unusable.

use calib_targets::chessboard::ChessboardParams;
use calib_targets::detect;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::chess_options::{ChessCornersConfig, apply_board_override, chess_config_for_override};
use crate::{DetectError, Detector, Feature};

/// Chessboard detector configuration. Mirrors the shape of the
/// chessboard variant in
/// `vision_calibration_dataset::TargetSpec` so the dispatcher can
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
    /// Optional ChESS corner-stage overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chess_corners: Option<ChessCornersConfig>,
}

/// Stateless chessboard detector instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct ChessboardDetector;

impl crate::sealed::Sealed for ChessboardDetector {}

impl Detector for ChessboardDetector {
    fn name(&self) -> &'static str {
        "chessboard"
    }

    fn detect_json(
        &self,
        image: &image::DynamicImage,
        config: &Value,
    ) -> Result<Vec<Feature>, DetectError> {
        let cfg: ChessboardConfig =
            serde_json::from_value(config.clone()).map_err(|e| DetectError::Config {
                detector: "chessboard",
                source: e,
            })?;
        let luma = image.to_luma8();

        // The underlying detector auto-labels corners from
        // intersection clustering — `rows`/`cols` from our config are
        // used only for output validation, not as input parameters.
        let mut board_params = ChessboardParams::default();
        apply_board_override(cfg.chess_corners, &mut board_params);
        let chess_cfg = chess_config_for_override(cfg.chess_corners);

        // No board in frame is "no features", not an error. Every other
        // `DetectError` variant is a genuine backend failure and propagates.
        let detection = match detect::detect_chessboard(&luma, &chess_cfg, &board_params) {
            Ok(detection) => detection,
            Err(detect::DetectError::NoDetection { .. }) => return Ok(Vec::new()),
            Err(err) => {
                return Err(DetectError::Backend {
                    detector: "chessboard",
                    message: err.to_string(),
                });
            }
        };

        // Filter to corners whose grid index falls inside the board, then
        // lift to 3D using the supplied square size.
        let (max_u, max_v) = label_bounds(
            detection.corners.iter().map(|c| (c.grid.u, c.grid.v)),
            cfg.rows,
            cfg.cols,
        );
        let mut features = Vec::new();
        for corner in detection.corners {
            let grid = corner.grid;
            if grid.u < 0 || grid.v < 0 || grid.u >= max_u || grid.v >= max_v {
                continue;
            }
            features.push(Feature {
                image_xy: [corner.position.x as f64, corner.position.y as f64],
                world_xyz: [
                    grid.u as f64 * cfg.square_size_m,
                    grid.v as f64 * cfg.square_size_m,
                    0.0,
                ],
            });
        }
        Ok(crate::reject_ambiguous_detection(features))
    }
}

/// The exclusive `(u, v)` label bounds of a `rows × cols` board, for one
/// detection with these labels.
///
/// `calib-targets` labels each view on its own: `(0, 0)` is the detection's
/// top-left corner in the image, `u` runs along +x and `v` along +y. A board
/// seen as the manifest declares it (`cols` corners across, `rows` down)
/// spans `u < cols` and `v < rows`.
///
/// A board seen turned by 90° spans `u < rows` and `v < cols`. Its labels
/// are a rotation of the declared ones, as every view's labels are, so it is
/// kept whole. A detection that fits neither orientation is cut to the
/// declared bounds.
fn label_bounds(labels: impl Iterator<Item = (i32, i32)>, rows: u32, cols: u32) -> (i32, i32) {
    let (rows, cols) = (rows as i32, cols as i32);
    let (max_u, max_v) = labels.fold((0, 0), |(mu, mv), (u, v)| (mu.max(u), mv.max(v)));
    let declared = max_u < cols && max_v < rows;
    let turned = max_u < rows && max_v < cols;
    if turned && !declared {
        (rows, cols)
    } else {
        (cols, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn grid(w: i32, h: i32) -> impl Iterator<Item = (i32, i32)> {
        (0..h).flat_map(move |v| (0..w).map(move |u| (u, v)))
    }

    #[test]
    fn a_board_seen_as_declared_spans_cols_across() {
        // 17 rows × 28 cols (kuka_1): u runs over the 28 columns.
        assert_eq!(label_bounds(grid(28, 17), 17, 28), (28, 17));
        // A partial view of it is bounded the same way.
        assert_eq!(label_bounds(grid(20, 10), 17, 28), (28, 17));
    }

    #[test]
    fn a_board_seen_turned_is_kept_whole() {
        assert_eq!(label_bounds(grid(17, 28), 17, 28), (17, 28));
    }

    #[test]
    fn a_detection_larger_than_the_board_is_cut_to_the_declared_bounds() {
        assert_eq!(label_bounds(grid(30, 18), 17, 28), (28, 17));
    }

    /// `data/kuka_1`: 17 × 28 interior corners, 20 mm squares, the board
    /// seen landscape in every view.
    fn kuka_view() -> image::DynamicImage {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/kuka_1/01.png");
        image::open(path).expect("data/kuka_1/01.png")
    }

    fn extent(features: &[Feature]) -> [f64; 2] {
        features.iter().fold([0.0f64; 2], |[x, y], f| {
            [x.max(f.world_xyz[0]), y.max(f.world_xyz[1])]
        })
    }

    #[test]
    fn keeps_every_corner_of_a_landscape_board() {
        let cfg = json!({ "rows": 17, "cols": 28, "square_size_m": 0.02 });
        let features = ChessboardDetector.detect_json(&kuka_view(), &cfg).unwrap();
        assert_eq!(features.len(), 17 * 28);
        let [x, y] = extent(&features);
        assert!((x - 27.0 * 0.02).abs() < 1e-12 && (y - 16.0 * 0.02).abs() < 1e-12);
    }

    #[test]
    fn radon_strategy_detects_the_board() {
        let cfg = json!({
            "rows": 17,
            "cols": 28,
            "square_size_m": 0.02,
            "chess_corners": { "strategy": "radon" },
        });
        let features = ChessboardDetector.detect_json(&kuka_view(), &cfg).unwrap();
        // Radon misses a few of the 476 corners at the edge of this view.
        assert!(features.len() >= 470, "{} features", features.len());
    }

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
