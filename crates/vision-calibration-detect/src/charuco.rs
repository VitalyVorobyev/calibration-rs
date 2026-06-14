//! ChArUco detector wrapping `chess-corners` + `calib-targets`.
//!
//! The detector returns one [`Feature`] per ChArUco corner whose marker-based
//! identification succeeded. ChArUco correspondences are *sparse* — occluded
//! or unidentified cells simply produce no feature — so the per-view feature
//! count varies; the runner decides whether a view is usable.
//!
//! Canonical impl note: `vision-calibration-bench/src/detect.rs::detect_charuco_view`
//! and the `examples-private` helper carry near-identical calib-targets glue
//! with different output shapes. This is the canonical app-facing version;
//! dedup of the other two is tracked as B3c follow-up.

use anyhow::{Result, anyhow};
use calib_targets::aruco::builtins;
use calib_targets::charuco::{
    CharucoBoard, CharucoBoardSpec, CharucoDetector as CtCharucoDetector, CharucoParams,
    MarkerLayout,
};
use calib_targets::detect;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::chess_options::{ChessCornersConfig, chess_config_for_override};
use crate::{Detector, Feature};

/// ChArUco detector configuration. Mirrors the shape of the charuco
/// variant in `vision_calibration_dataset::TargetSpec` so the
/// dispatcher can translate one to the other directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CharucoConfig {
    /// Number of squares along the rows axis (full grid, not interior
    /// corners — ChArUco boards are described by their square counts).
    pub rows: u32,
    /// Number of squares along the cols axis.
    pub cols: u32,
    /// Edge length of one square in metres. The detector's board spec
    /// is built in metres, so detected corners carry metric target
    /// coordinates directly.
    pub square_size_m: f64,
    /// Edge length of an embedded ArUco marker in metres. Must be
    /// smaller than `square_size_m`.
    pub marker_size_m: f64,
    /// ArUco dictionary identifier (e.g. `"DICT_4X4_50"`). Validated
    /// against the embedded builtin set; see [`validate_dictionary`].
    pub dictionary: String,
    /// Optional ChESS corner-stage overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chess_corners: Option<ChessCornersConfig>,
}

/// Check `name` against the ArUco dictionaries embedded in
/// `calib-targets`. Lets the dispatcher fail fast on a manifest typo
/// before touching any image files. The error lists the supported names.
pub fn validate_dictionary(name: &str) -> Result<()> {
    if builtins::builtin_dictionary(name).is_some() {
        Ok(())
    } else {
        Err(anyhow!(
            "unknown ArUco dictionary {name:?}; supported: {}",
            builtins::BUILTIN_DICTIONARY_NAMES.join(", ")
        ))
    }
}

/// Validate a full ChArUco board layout before any image I/O: the
/// dictionary name must be known *and* hold enough marker codes for the
/// requested `rows × cols` OpenCV layout. A larger board needs more
/// markers (e.g. a 12×12 board needs 72, which `DICT_4X4_50` cannot
/// supply), so validating the name alone is not enough — that lets an
/// impossible board reach detection with a guaranteed-empty result. This
/// subsumes [`validate_dictionary`]: an unknown name fails here too.
/// Geometry (square/marker size) is checked separately by the detector.
pub fn validate_charuco_layout(rows: u32, cols: u32, dictionary: &str) -> Result<()> {
    let dictionary = builtins::builtin_dictionary(dictionary).ok_or_else(|| {
        anyhow!(
            "unknown ArUco dictionary {dictionary:?}; supported: {}",
            builtins::BUILTIN_DICTIONARY_NAMES.join(", ")
        )
    })?;
    // Unit geometry: `CharucoBoard::new` only inspects rows/cols and the
    // dictionary capacity here; cell/marker size are placeholders.
    let spec = CharucoBoardSpec {
        rows,
        cols,
        cell_size: 1.0,
        marker_size_rel: 0.5,
        dictionary,
        marker_layout: MarkerLayout::OpenCvCharuco,
    };
    CharucoBoard::new(spec)
        .map(|_| ())
        .map_err(|e| anyhow!("invalid charuco board: {e}"))
}

fn params_for(cfg: &CharucoConfig) -> Result<CharucoParams> {
    validate_charuco_layout(cfg.rows, cfg.cols, &cfg.dictionary)?;
    let dictionary = builtins::builtin_dictionary(&cfg.dictionary)
        .expect("validated above; builtin lookup is deterministic");
    if !(cfg.marker_size_m > 0.0 && cfg.marker_size_m <= cfg.square_size_m) {
        return Err(anyhow!(
            "marker_size_m ({}) must be in (0, square_size_m = {}]",
            cfg.marker_size_m,
            cfg.square_size_m
        ));
    }
    let board = CharucoBoardSpec {
        rows: cfg.rows,
        cols: cfg.cols,
        cell_size: cfg.square_size_m as f32,
        marker_size_rel: (cfg.marker_size_m / cfg.square_size_m) as f32,
        dictionary,
        marker_layout: MarkerLayout::OpenCvCharuco,
    };
    Ok(CharucoParams::for_board(&board))
}

/// Stateless ChArUco detector instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct CharucoDetector;

impl crate::sealed::Sealed for CharucoDetector {}

impl Detector for CharucoDetector {
    fn name(&self) -> &'static str {
        "charuco"
    }

    fn detect_json(&self, image: &image::DynamicImage, config: &Value) -> Result<Vec<Feature>> {
        let cfg: CharucoConfig = serde_json::from_value(config.clone())
            .map_err(|e| anyhow!("invalid charuco config: {e}"))?;
        let params = params_for(&cfg)?;
        let luma = image.to_luma8();

        // ChESS corner pre-detection feeds the ChArUco identifier.
        let chess_cfg = chess_config_for_override(cfg.chess_corners);
        let corners = detect::detect_corners(&luma, &chess_cfg);
        let detector = CtCharucoDetector::new(params)?;
        // A failed board identification (too few markers, no board in
        // frame) is "no features", not an error — same semantics as
        // the chessboard detector on a blank image.
        let detection = match detector.detect(&detect::gray_view(&luma), &corners) {
            Ok(detection) => detection,
            Err(_) => return Ok(Vec::new()),
        };

        // `target_position` is metric because the board spec's
        // `cell_size` was given in metres.
        Ok(detection
            .corners
            .into_iter()
            .map(|corner| Feature {
                image_xy: [corner.position.x as f64, corner.position.y as f64],
                world_xyz: [
                    corner.target_position.x as f64,
                    corner.target_position.y as f64,
                    0.0,
                ],
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // 8x8 squares => 32 markers in the OpenCV layout, which fits the
    // 50-code DICT_4X4_50 dictionary.
    fn board_config() -> Value {
        json!({
            "rows": 8,
            "cols": 8,
            "square_size_m": 0.020,
            "marker_size_m": 0.015,
            "dictionary": "DICT_4X4_50",
        })
    }

    /// Render a synthetic frontal board via calib-targets' printable
    /// pipeline and decode it back into a `DynamicImage`.
    fn synthetic_board_image() -> image::DynamicImage {
        let dictionary = builtins::builtin_dictionary("DICT_4X4_50").expect("builtin dict");
        let document = calib_targets::generate::charuco_document(8, 8, 20.0, 0.75, dictionary);
        let bundle =
            calib_targets::printable::render_target_bundle(&document).expect("render bundle");
        image::load_from_memory(&bundle.png_bytes).expect("decode rendered PNG")
    }

    #[test]
    fn detects_synthetic_board() {
        let img = synthetic_board_image();
        let features = CharucoDetector.detect_json(&img, &board_config()).unwrap();
        assert!(
            features.len() >= 4,
            "expected >= 4 corners on a clean frontal board, got {}",
            features.len()
        );
        for f in &features {
            assert_eq!(f.world_xyz[2], 0.0, "target is planar");
            // Corners sit on the square lattice; allow float slack from
            // the f32 target coordinates.
            for &coord in &f.world_xyz[..2] {
                let cells = coord / 0.020;
                assert!(
                    (cells - cells.round()).abs() < 1e-4,
                    "world coord {coord} not on the 20 mm lattice"
                );
            }
        }
    }

    #[test]
    fn invalid_config_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let err = CharucoDetector
            .detect_json(&img, &json!({"rows": 12})) // missing the rest
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid charuco config"), "got: {msg}");
    }

    #[test]
    fn unknown_dictionary_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let mut cfg = board_config();
        cfg["dictionary"] = json!("DICT_9X9_1");
        let err = CharucoDetector.detect_json(&img, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown ArUco dictionary") && msg.contains("DICT_4X4_50"),
            "error should name the bad dictionary and list supported ones, got: {msg}"
        );
    }

    #[test]
    fn dictionary_too_small_for_board_rejected() {
        // 12×12 needs 72 markers; DICT_4X4_50 only has 50.
        assert!(validate_charuco_layout(8, 8, "DICT_4X4_50").is_ok());
        let err = validate_charuco_layout(12, 12, "DICT_4X4_50").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("72") && msg.contains("50"),
            "expected a needs-72/has-50 capacity error, got: {msg}"
        );

        // The detector itself must also reject the impossible board
        // rather than silently returning no features.
        let img = image::DynamicImage::new_luma8(64, 64);
        let mut cfg = board_config();
        cfg["rows"] = json!(12);
        cfg["cols"] = json!(12);
        let err = CharucoDetector.detect_json(&img, &cfg).unwrap_err();
        assert!(format!("{err}").contains("charuco board"));
    }

    #[test]
    fn oversized_marker_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let mut cfg = board_config();
        cfg["marker_size_m"] = json!(0.021);
        let err = CharucoDetector.detect_json(&img, &cfg).unwrap_err();
        assert!(format!("{err}").contains("marker_size_m"));
    }

    #[test]
    fn blank_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let features = CharucoDetector.detect_json(&img, &board_config()).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }
}
