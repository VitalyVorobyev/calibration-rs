//! PuzzleBoard detector wrapping `calib-targets` (Stelldinger 2024).
//!
//! A PuzzleBoard is a self-identifying chessboard: each saddle is decoded
//! to a unique grid index from a local code window, so a single partial
//! view yields globally-consistent correspondences. The detector returns
//! one [`Feature`] per decoded saddle that falls inside the printed board,
//! with metric target coordinates in the board's own (top-left-origin)
//! frame — the same origin convention as the chessboard/charuco detectors.
//!
//! Two parts of the recipe are fixed: a 2× upscale before detection
//! (PuzzleBoard decoding is sensitive to saddle sharpness on small tiles)
//! and `FixedBoard` search (the board dimensions are known from the
//! manifest, so we never search for an unknown sub-board).

use calib_targets::blur::puzzleboard_blur;
use calib_targets::detect;
use calib_targets::puzzleboard::{PuzzleBoardParams, PuzzleBoardSearchMode, PuzzleBoardSpec};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::corner_blur::feature_blurs;
use crate::{DetectError, Feature};

/// Upscale factor applied before detection. PuzzleBoard saddle decoding
/// degrades on small/blurred tiles, so the image is resized 2× first and
/// detected pixel coordinates are divided back down.
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
#[non_exhaustive]
pub struct PuzzleboardConfig {
    /// Number of cells along the rows axis.
    pub rows: u32,
    /// Number of cells along the cols axis.
    pub cols: u32,
    /// Edge length of one cell in metres. Lifts each decoded saddle to a
    /// metric target point and bounds the in-board filter.
    pub cell_size_m: f64,
    /// Measure the blur at each saddle ([`Feature::blur`]): the covariance
    /// `Σ` (px², image axes, including the pixel aperture) of the Gaussian
    /// kernel that best blurs the printed squares and dots around the saddle
    /// into the image, with the covariance of that estimate. It is measured
    /// on the image as given, not on the upscaled copy detection runs on,
    /// at the reported `image_xy`. A saddle the estimate declines (blur too
    /// wide for the cells, sharper than a pixel, too few labelled
    /// neighbours) has none. It costs a few milliseconds per saddle,
    /// single-threaded (476 saddles in view add about 1–2 s per
    /// image), hence the default `false`. Saddle positions are the same
    /// with or without it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub measure_blur: bool,
}

impl PuzzleboardConfig {
    /// A PuzzleBoard of `rows × cols` cells of `cell_size_m`.
    pub fn new(rows: u32, cols: u32, cell_size_m: f64) -> Self {
        Self {
            rows,
            cols,
            cell_size_m,
            measure_blur: false,
        }
    }

    /// The same config with [`measure_blur`](Self::measure_blur) set.
    pub fn with_measure_blur(mut self, measure_blur: bool) -> Self {
        self.measure_blur = measure_blur;
        self
    }
}

/// A corner coordinate in the upscaled detection image, in the original
/// image's pixels. Division by a power of two is exact, so the features and
/// the blur measurement see the same positions.
fn to_original(coordinate: f32) -> f32 {
    coordinate / DETECT_UPSCALE as f32
}

/// Detect the PuzzleBoard `cfg` describes in `luma`.
pub(crate) fn detect(
    cfg: &PuzzleboardConfig,
    luma: &image::GrayImage,
) -> Result<Vec<Feature>, DetectError> {
    // The detector decodes saddles in millimetres; build the spec in
    // mm, the unit its tolerances are tuned in, and convert metric
    // outputs back to metres below.
    let cell_size_mm = (cfg.cell_size_m * 1000.0) as f32;
    let spec = PuzzleBoardSpec::new(cfg.rows, cfg.cols, cell_size_mm)
        .map_err(|e| DetectError::InvalidConfig(format!("invalid puzzleboard spec: {e}")))?;
    let mut params = PuzzleBoardParams::for_board(spec);
    params.decode.search_all_components = false;
    params.decode.search_mode = PuzzleBoardSearchMode::FixedBoard;

    let detection_image = image::imageops::resize(
        luma,
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

    // The blur is measured on the original image (resampling blurs it), so
    // the corners are mapped back first, by the map the features use.
    let mut blurs = if cfg.measure_blur {
        let mut original = detection.clone();
        for corner in &mut original.corners {
            corner.position.x = to_original(corner.position.x);
            corner.position.y = to_original(corner.position.y);
        }
        feature_blurs(puzzleboard_blur(luma, &original))
    } else {
        Vec::new()
    };
    blurs.resize(detection.corners.len(), None);

    let mut features = Vec::with_capacity(detection.corners.len());
    for (corner, blur) in detection.corners.iter().zip(blurs) {
        let x_mm = corner.target_position.x as f64;
        let y_mm = corner.target_position.y as f64;
        if !(0.0..=max_x_mm).contains(&x_mm) || !(0.0..=max_y_mm).contains(&y_mm) {
            continue;
        }
        let image_xy = [
            f64::from(to_original(corner.position.x)),
            f64::from(to_original(corner.position.y)),
        ];
        features.push(Feature::new(image_xy, [x_mm * 1.0e-3, y_mm * 1.0e-3, 0.0]).with_blur(blur));
    }
    Ok(crate::reject_ambiguous_detection(features))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Detect with a JSON config, as a manifest would spell it.
    fn run(image: &image::DynamicImage, config: &Value) -> Result<Vec<Feature>, DetectError> {
        let cfg: PuzzleboardConfig = serde_json::from_value(config.clone())
            .map_err(|e| DetectError::InvalidConfig(e.to_string()))?;
        detect(&cfg, &image.to_luma8())
    }
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
        let features = run(&img, &board_config()).unwrap();
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

    /// A board seen through a lens of Gaussian blur σ reports the rendered
    /// `tr Σ / 2` (the lens's `σ²` plus the pixel aperture's term) within
    /// its stated standard errors. The blur is measured on the image as
    /// given, at the saddles mapped back from the upscaled detection, and
    /// measuring it moves no saddle.
    #[test]
    fn measured_blur_matches_the_rendered_blur() {
        use crate::corner_blur::known_blur::{assert_matches_render, camera_image};
        // 10 × 10 cells of 24 px: σ = 2 px is 0.08 of a cell, well inside
        // the estimator's range (it declines beyond 0.3).
        let document = calib_targets::generate::puzzleboard_document(10, 10, 25.4);
        let cfg = PuzzleboardConfig::new(10, 10, 0.0254);
        for sigma in [1.0, 2.0] {
            let img = camera_image(document.clone(), 24, 275.0, sigma, 1.0, 7);
            let plain = detect(&cfg, &img).unwrap();
            assert_eq!(plain.len(), 81, "every saddle of the board");
            assert!(plain.iter().all(|f| f.blur.is_none()));

            let with = detect(&cfg.with_measure_blur(true), &img).unwrap();
            assert_eq!(with.len(), plain.len());
            for (a, b) in with.iter().zip(&plain) {
                assert_eq!(a.image_xy, b.image_xy);
                assert_eq!(a.world_xyz, b.world_xyz);
            }
            assert_matches_render(&with, sigma);
        }
    }

    #[test]
    fn blank_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let features = run(&img, &board_config()).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg: PuzzleboardConfig = serde_json::from_value(board_config()).unwrap();
        let back: PuzzleboardConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(cfg, back);
        // `measure_blur` reads as off when left out, is not written when
        // off, and round-trips when on.
        assert!(!cfg.measure_blur);
        assert!(
            !serde_json::to_string(&cfg)
                .unwrap()
                .contains("measure_blur")
        );
        let with = cfg.with_measure_blur(true);
        let json = serde_json::to_string(&with).unwrap();
        assert!(json.contains(r#""measure_blur":true"#), "{json}");
        assert_eq!(
            serde_json::from_str::<PuzzleboardConfig>(&json).unwrap(),
            with
        );
    }
}
