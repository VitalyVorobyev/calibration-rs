//! PuzzleBoard detector wrapping `calib-targets` (Stelldinger 2024).
//!
//! A PuzzleBoard is a self-identifying chessboard: each saddle is decoded
//! to a unique grid index from a local code window, so a single partial
//! view yields globally-consistent correspondences. The detector returns
//! one [`Feature`] per decoded saddle that falls inside the printed board,
//! with metric target coordinates in the board's own (top-left-origin)
//! frame — the same origin convention as the chessboard/charuco detectors.
//!
//! Two parts of the recipe are fixed: the corner detector's own 2× upscale
//! stage (saddle detection degrades on small or blurred tiles; the stage
//! reports corners in the image's own pixels) and `FixedBoard` search (the
//! board dimensions are known from the manifest, so we never search for an
//! unknown sub-board).

use calib_targets::blur::puzzleboard_blur;
use calib_targets::detect;
use calib_targets::puzzleboard::{PuzzleBoardParams, PuzzleBoardSearchMode, PuzzleBoardSpec};
use chess_corners::UpscaleConfig;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::corner_blur::feature_blurs;
use crate::{DetectError, Feature};

/// Factor of the corner detector's upscale stage. Saddle detection degrades
/// on small or blurred tiles at native resolution; the stage maps the
/// corners back to the image's own pixels.
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
    /// on the image as given, not on the upscaled copy corner detection
    /// runs on, at the reported `image_xy`. A saddle the estimate declines
    /// (blur too wide for the cells, sharper than a pixel, too few labelled
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
    params.chess = params
        .chess
        .with_upscale(UpscaleConfig::fixed(DETECT_UPSCALE));

    // No board in frame is "no features", not an error — same
    // semantics as the chessboard/charuco detectors on a blank image.
    let detection = match detect::detect_puzzleboard(luma, &params) {
        Ok(detection) => detection,
        Err(_) => return Ok(Vec::new()),
    };

    // Decoded saddles outside the printed board are spurious; bound
    // them to `[0, (n-1) * cell]` mm before lifting to metres.
    let cell_size_mm = cfg.cell_size_m * 1000.0;
    let max_x_mm = cfg.cols.saturating_sub(1) as f64 * cell_size_mm;
    let max_y_mm = cfg.rows.saturating_sub(1) as f64 * cell_size_mm;

    // The corners are in `luma`'s pixels, and the blur is measured there
    // (resampling would blur it).
    let mut blurs = if cfg.measure_blur {
        feature_blurs(puzzleboard_blur(luma, &detection))
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
        let image_xy = [f64::from(corner.position.x), f64::from(corner.position.y)];
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
    /// given, at the reported saddles, and measuring it moves no saddle.
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

    /// Sub-pixels per pixel and axis that [`shifted_board`] rasterizes the
    /// print at.
    const SUB: u32 = 8;

    /// A frontal `cells × cells` PuzzleBoard of 25.4 mm cells with `cell_px`
    /// pixels per cell, as a camera with box-shaped pixels sees it, the print
    /// shifted by `shift` sub-pixels (`1 / SUB` px) right and down.
    ///
    /// The print is rasterized at `cell_px · SUB` dots per inch on a page one
    /// cell wider than the board on every side, so the board's top-left edge
    /// sits at sub-pixel `cell_px · SUB` and every square edge on a sub-pixel
    /// boundary. Pixel `(x, y)` averages the sub-pixels
    /// `[x · SUB + shift, (x + 1) · SUB + shift)` per axis. Returns the image
    /// and the map from a saddle's board position (mm, the frame the decoder
    /// reports `target_position` in) to its true pixel position, integer
    /// coordinates at pixel centres.
    fn shifted_board(
        cells: u32,
        cell_px: u32,
        shift: [u32; 2],
    ) -> (image::GrayImage, impl Fn([f64; 2]) -> [f64; 2]) {
        use calib_targets::printable::{PageSize, PageSpec};
        let mut document = calib_targets::generate::puzzleboard_document(cells, cells, 25.4);
        let page_mm = f64::from(cells + 2) * 25.4;
        document.page = PageSpec::default().with_size(PageSize::Custom {
            width_mm: page_mm,
            height_mm: page_mm,
        });
        document.render.png_dpi = cell_px * SUB;
        let bundle =
            calib_targets::printable::render_target_bundle(&document).expect("render bundle");
        let fine = image::load_from_memory(&bundle.png_bytes)
            .expect("decode rendered PNG")
            .to_luma8();
        assert_eq!(fine.width(), (cells + 2) * cell_px * SUB, "page width");
        let size = fine.width() / SUB - 1;
        let image = image::GrayImage::from_fn(size, size, |x, y| {
            let mut sum = 0u32;
            for j in 0..SUB {
                for i in 0..SUB {
                    let fx = x * SUB + shift[0] + i;
                    let fy = y * SUB + shift[1] + j;
                    sum += u32::from(fine.get_pixel(fx, fy).0[0]);
                }
            }
            let reflectance = f64::from(sum) / f64::from(SUB * SUB) / 255.0;
            image::Luma([(30.0 + 190.0 * reflectance).round() as u8])
        });
        let sub = f64::from(SUB);
        let origin = f64::from(cell_px * SUB);
        let truth = move |board_mm: [f64; 2]| {
            let at = |mm: f64, shift: u32| {
                let edge = origin + mm / 25.4 * f64::from(cell_px) * sub;
                (edge - f64::from(shift)) / sub - 0.5
            };
            [at(board_mm[0], shift[0]), at(board_mm[1], shift[1])]
        };
        (image, truth)
    }

    /// Saddles land where the board puts them, at several cell sizes and
    /// sub-pixel placements: the mean error over a board is within 0.03 px
    /// on each axis. An error in the map between the upscaled copy corner
    /// detection runs on and the image given shows up here as a shift of
    /// every saddle: a map that only divides by the factor 2 puts each one a
    /// quarter pixel right and down.
    #[test]
    fn saddles_land_at_the_rendered_corners() {
        for (cell_px, shift) in [(12, [3, 6]), (17, [1, 5]), (24, [6, 2])] {
            let (image, truth) = shifted_board(10, cell_px, shift);
            let cfg = PuzzleboardConfig::new(10, 10, 0.0254);
            let features = detect(&cfg, &image).unwrap();
            assert!(
                features.len() >= 49,
                "cell {cell_px} px: {} saddles",
                features.len()
            );
            let n = features.len() as f64;
            let (mut mean, mut sq) = ([0.0; 2], 0.0);
            for f in &features {
                let t = truth([f.world_xyz[0] * 1.0e3, f.world_xyz[1] * 1.0e3]);
                let e = [f.image_xy[0] - t[0], f.image_xy[1] - t[1]];
                mean[0] += e[0] / n;
                mean[1] += e[1] / n;
                sq += (e[0] * e[0] + e[1] * e[1]) / n;
            }
            eprintln!(
                "cell {cell_px} px, shift {shift:?}: {} saddles, mean error ({:+.4}, {:+.4}) px, \
                 rms {:.4} px",
                features.len(),
                mean[0],
                mean[1],
                sq.sqrt()
            );
            assert!(
                mean[0].abs() < 0.03 && mean[1].abs() < 0.03,
                "cell {cell_px} px, shift {shift:?}: mean error ({:+.4}, {:+.4}) px",
                mean[0],
                mean[1]
            );
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
