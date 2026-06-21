//! Coded ring-grid detector wrapping the `ringgrid` crate.
//!
//! Ring markers are self-identifying: each decoded marker carries a unique
//! codebook id whose board position is known from the layout, so a single
//! partial view yields globally-consistent correspondences. The detector
//! returns one [`Feature`] per decoded marker that maps to a board
//! position, with metric target coordinates in the board's own frame.
//!
//! Detection runs in adaptive-scale mode (`Detector::detect_adaptive`): the
//! marker pixel size is unknown for an arbitrary dataset image, and adaptive
//! mode auto-selects scale tiers rather than relying on a hand-tuned prior.

use ringgrid::{BoardLayout, Detector as RinggridDetectorImpl};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::{DetectError, Detector, Feature};

/// Coded ring-grid detector configuration. Mirrors the geometry of the
/// ringgrid variant in `vision_calibration_dataset::TargetSpec` (and, in
/// turn, `ringgrid::BoardLayout`) so the dispatcher can translate one to
/// the other directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct RinggridConfig {
    /// Center-to-center spacing between adjacent markers in metres.
    pub pitch_m: f64,
    /// Number of marker rows.
    pub rows: u32,
    /// Number of columns in the longest (even-indexed) row.
    pub long_row_cols: u32,
    /// Outer ring radius in metres.
    pub marker_outer_radius_m: f64,
    /// Inner ring radius in metres.
    pub marker_inner_radius_m: f64,
    /// Width of each ring band in metres.
    pub marker_ring_width_m: f64,
}

impl RinggridConfig {
    /// Build the `ringgrid` board layout this config describes. The
    /// `ringgrid` API works in millimetres, so metric fields are scaled
    /// here; geometry validation (radii ordering, marker-vs-pitch fit) is
    /// delegated to `BoardLayout::new`.
    fn board_layout(&self) -> Result<BoardLayout, DetectError> {
        let mm = |m: f64| (m * 1000.0) as f32;
        BoardLayout::new(
            mm(self.pitch_m),
            self.rows as usize,
            self.long_row_cols as usize,
            mm(self.marker_outer_radius_m),
            mm(self.marker_inner_radius_m),
            mm(self.marker_ring_width_m),
        )
        .map_err(|e| DetectError::InvalidConfig(format!("invalid ringgrid board: {e}")))
    }
}

/// Stateless coded ring-grid detector instance.
#[derive(Debug, Default, Clone, Copy)]
pub struct RinggridDetector;

impl crate::sealed::Sealed for RinggridDetector {}

impl Detector for RinggridDetector {
    fn name(&self) -> &'static str {
        "ringgrid"
    }

    fn detect_json(
        &self,
        image: &image::DynamicImage,
        config: &Value,
    ) -> Result<Vec<Feature>, DetectError> {
        let cfg: RinggridConfig =
            serde_json::from_value(config.clone()).map_err(|e| DetectError::Config {
                detector: "ringgrid",
                source: e,
            })?;
        let board = cfg.board_layout()?;
        let detector = RinggridDetectorImpl::new(board);

        let luma = image.to_luma8();
        let result = detector.detect_adaptive(&luma);

        // A decoded marker contributes a correspondence only when its id
        // maps to a known board position (`board_xy_mm`); undecoded or
        // off-board detections are dropped. `center` is always raw image
        // pixels. `board_xy_mm` is in millimetres → convert to metres.
        Ok(result
            .detected_markers
            .into_iter()
            .filter_map(|marker| {
                marker.board_xy_mm.map(|xy_mm| Feature {
                    image_xy: marker.center,
                    world_xyz: [xy_mm[0] * 1.0e-3, xy_mm[1] * 1.0e-3, 0.0],
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ringgrid::PngTargetOptions;
    use serde_json::json;

    const PITCH_M: f64 = 0.020;
    const ROWS: u32 = 5;
    const LONG_ROW_COLS: u32 = 5;
    const OUTER_M: f64 = 0.006;
    const INNER_M: f64 = 0.003;
    const RING_WIDTH_M: f64 = 0.001;

    fn board_config() -> Value {
        json!({
            "pitch_m": PITCH_M,
            "rows": ROWS,
            "long_row_cols": LONG_ROW_COLS,
            "marker_outer_radius_m": OUTER_M,
            "marker_inner_radius_m": INNER_M,
            "marker_ring_width_m": RING_WIDTH_M,
        })
    }

    /// Render a synthetic frontal ring-grid via the `ringgrid` printable
    /// pipeline and wrap it as a `DynamicImage`.
    fn synthetic_board_image() -> image::DynamicImage {
        let cfg: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        let board = cfg.board_layout().unwrap();
        let gray = board
            .render_target_png(&PngTargetOptions::default())
            .expect("render ring-grid PNG");
        image::DynamicImage::ImageLuma8(gray)
    }

    #[test]
    fn detects_synthetic_board() {
        let img = synthetic_board_image();
        let features = RinggridDetector.detect_json(&img, &board_config()).unwrap();
        // A conservative floor: a 5×5 hex board carries ~23 markers, but
        // adaptive-scale detection on a small synthetic render is not
        // guaranteed to decode every one; ≥4 is enough to confirm the
        // wrapping decodes + maps markers to metric board coordinates.
        assert!(
            features.len() >= 4,
            "expected >= 4 decoded markers on a clean frontal board, got {}",
            features.len()
        );
        // Board coordinates are metric (board span ~0.1 m for this config);
        // a sub-metre bound guards the mm→m conversion — an unconverted
        // result would land in the tens-of-metres range.
        for f in &features {
            assert_eq!(f.world_xyz[2], 0.0, "target is planar");
            assert!(
                f.world_xyz[0].abs() < 1.0 && f.world_xyz[1].abs() < 1.0,
                "mapped marker {:?} is not in plausible metric range (mm→m bug?)",
                f.world_xyz
            );
        }
    }

    #[test]
    fn invalid_config_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let err = RinggridDetector
            .detect_json(&img, &json!({"rows": 5})) // missing the rest
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("invalid ringgrid config"), "got: {msg}");
    }

    #[test]
    fn invalid_board_geometry_rejected() {
        // Inner radius >= outer radius is an impossible board; the wrapped
        // `BoardLayout::new` must reject it before any detection.
        let mut cfg = board_config();
        cfg["marker_inner_radius_m"] = json!(OUTER_M);
        let img = image::DynamicImage::new_luma8(64, 64);
        let err = RinggridDetector.detect_json(&img, &cfg).unwrap_err();
        assert!(
            format!("{err}").contains("invalid ringgrid board"),
            "got: {err}"
        );
    }

    #[test]
    fn blank_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let features = RinggridDetector.detect_json(&img, &board_config()).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        let back: RinggridConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(cfg, back);
    }
}
