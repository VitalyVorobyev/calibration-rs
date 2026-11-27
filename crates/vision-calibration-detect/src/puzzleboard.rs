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

use std::collections::HashMap;

use calib_targets::blur::puzzleboard_blur;
use calib_targets::core::{Coord, GridAlignment};
use calib_targets::detect;
use calib_targets::puzzleboard::code_maps::{
    EdgeOrientation, horizontal_edge_bit, vertical_edge_bit,
};
use calib_targets::puzzleboard::diagnostics::PuzzleBoardObservedEdge;
use calib_targets::puzzleboard::{
    MASTER_COLS, MASTER_ROWS, PuzzleBoardDetector, PuzzleBoardParams, PuzzleBoardSearchMode,
    PuzzleBoardSpec,
};
use chess_corners::UpscaleConfig;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::corner_blur::feature_blurs;
use crate::detail::{
    CornerDetail, CornerQuality, DetectionDetail, EdgeBitDetail, PointRejection, PuzzleboardDecode,
    PuzzleboardDetail, RejectedPoint, TargetDetail,
};
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
    detect_detailed(cfg, luma).map(|detail| detail.features)
}

/// [`detect`], with the detection's detail.
pub(crate) fn detect_detailed(
    cfg: &PuzzleboardConfig,
    luma: &image::GrayImage,
) -> Result<DetectionDetail, DetectError> {
    if cfg.rows < 2 || cfg.cols < 2 || cfg.rows > i32::MAX as u32 || cfg.cols > i32::MAX as u32 {
        return Err(DetectError::InvalidConfig(format!(
            "puzzleboard rows={} cols={} must be in 2..={}",
            cfg.rows,
            cfg.cols,
            i32::MAX
        )));
    }
    if !(cfg.cell_size_m.is_finite() && cfg.cell_size_m > 0.0) {
        return Err(DetectError::InvalidConfig(format!(
            "puzzleboard cell_size_m ({}) must be finite and > 0",
            cfg.cell_size_m
        )));
    }
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

    // `detect::detect_puzzleboard` through the detector's diagnosing path:
    // the same detection (both run one decode, which records its evidence
    // either way), with the edge bits it read.
    let backend = |message: String| DetectError::Backend {
        detector: "puzzleboard",
        message,
    };
    let detector = PuzzleBoardDetector::new(params).map_err(|e| backend(e.to_string()))?;
    let (result, diagnostics) = detector.diagnose(&detect::gray_view(luma));
    let target = |corners, edge_bits, decode, quality| {
        TargetDetail::Puzzleboard(PuzzleboardDetail {
            corners,
            edge_bits,
            decode,
            quality,
        })
    };
    // No board in frame is "no features", not an error — same
    // semantics as the chessboard/charuco detectors on a blank image.
    let detection = match result {
        Ok(detection) => detection,
        Err(err) if err.is_not_found() => {
            return Ok(DetectionDetail::finish(
                Vec::new(),
                Vec::<CornerDetail>::new(),
                Vec::new(),
                Some(err.to_string()),
                |corners| target(corners, Vec::new(), None, CornerQuality::default()),
            ));
        }
        Err(err) => return Err(backend(err.to_string())),
    };

    // Decoded saddles outside the printed board are spurious; bound them to
    // `[0, (n-1) * cell]` mm before lifting to metres, allowing f32 rounding slack
    // from `cell_size_mm as f32` so valid border saddles are never clipped.
    let cell_size_mm = cfg.cell_size_m * 1000.0;
    let max_x_mm = cfg.cols.saturating_sub(1) as f64 * cell_size_mm;
    let max_y_mm = cfg.rows.saturating_sub(1) as f64 * cell_size_mm;
    let tol_mm = 1e-4 * cell_size_mm;

    // The corners are in `luma`'s pixels, and the blur is measured there
    // (resampling would blur it).
    let mut blurs = if cfg.measure_blur {
        feature_blurs(puzzleboard_blur(luma, &detection))
    } else {
        Vec::new()
    };
    blurs.resize(detection.corners.len(), None);

    let mut features = Vec::with_capacity(detection.corners.len());
    let mut corners = Vec::with_capacity(detection.corners.len());
    let mut rejected = Vec::new();
    // Master corner → image position of the kept corners, for the edge bits.
    let mut kept: HashMap<[i32; 2], [f64; 2]> = HashMap::new();
    for (corner, blur) in detection.corners.iter().zip(blurs) {
        let x_mm = corner.target_position.x as f64;
        let y_mm = corner.target_position.y as f64;
        let image_xy = [f64::from(corner.position.x), f64::from(corner.position.y)];
        let world_xyz = [x_mm * 1.0e-3, y_mm * 1.0e-3, 0.0];
        let grid = [corner.grid.u, corner.grid.v];
        if !(-tol_mm..=max_x_mm + tol_mm).contains(&x_mm)
            || !(-tol_mm..=max_y_mm + tol_mm).contains(&y_mm)
        {
            rejected.push(RejectedPoint {
                image_xy,
                reason: PointRejection::OutsideBoard,
                world_xyz: Some(world_xyz),
                grid: Some(grid),
                id: Some(corner.id),
            });
            continue;
        }
        features.push(Feature::new(image_xy, world_xyz).with_blur(blur));
        corners.push(CornerDetail {
            grid,
            id: Some(corner.id),
            score: f64::from(corner.score),
            axes_rad: None,
        });
        kept.insert(master_key(grid), image_xy);
    }
    let edge_bits = edge_bits(&diagnostics.observed_edges, detection.alignment, &kept);
    let d = &detection.decode;
    let decode = PuzzleboardDecode {
        edges_observed: d.edges_observed,
        edges_matched: d.edges_matched,
        bit_error_rate: f64::from(d.bit_error_rate),
        logical_bit_error_rate: f64::from(d.logical_bit_error_rate),
        dot_dissent_rate: f64::from(d.dot_dissent_rate),
        master_origin: [d.master_origin_row, d.master_origin_col],
    };
    let quality = CornerQuality::from_calib_targets(&detection.quality);
    Ok(DetectionDetail::finish(
        features,
        corners,
        rejected,
        None,
        |corners| target(corners, edge_bits, Some(decode), quality),
    ))
}

/// A master corner `[u, v]` reduced to the master pattern's period, so that
/// labels equal modulo the period meet.
fn master_key([u, v]: [i32; 2]) -> [i32; 2] {
    [
        u.rem_euclid(MASTER_COLS as i32),
        v.rem_euclid(MASTER_ROWS as i32),
    ]
}

/// The edge bits the decoder read, on the edges whose two corners were kept.
///
/// An observation is recorded in the view's local grid at corner `(c, r)`:
/// a horizontal edge joins local corners `(c, r)` and `(c + 1, r)`, a
/// vertical one `(c, r)` and `(c, r + 1)`. `alignment` carries local corners
/// to the master pattern, where the expected bit of the edge between master
/// corners `(C, R)` and `(C + 1, R)` is `horizontal_edge_bit(R - 1, C)`, and
/// between `(C, R)` and `(C, R + 1)` is `vertical_edge_bit(R, C - 1)`
/// (`calib-targets`' edge convention, the decoder's own).
fn edge_bits(
    observed: &[PuzzleBoardObservedEdge],
    alignment: GridAlignment,
    kept: &HashMap<[i32; 2], [f64; 2]>,
) -> Vec<EdgeBitDetail> {
    let master = |u: i32, v: i32| {
        let c = alignment.apply(Coord::new(u, v));
        [c.u, c.v]
    };
    observed
        .iter()
        .filter_map(|edge| {
            let (c, r) = (edge.col, edge.row);
            let a = master(c, r);
            let b = match edge.orientation {
                EdgeOrientation::Horizontal => master(c + 1, r),
                EdgeOrientation::Vertical => master(c, r + 1),
                // The two orientations of a square grid.
                _ => return None,
            };
            let endpoints_px = [*kept.get(&master_key(a))?, *kept.get(&master_key(b))?];
            let expected = if a[1] == b[1] {
                horizontal_edge_bit(a[1] - 1, a[0].min(b[0]))
            } else {
                vertical_edge_bit(a[1].min(b[1]), a[0] - 1)
            };
            Some(EdgeBitDetail {
                endpoints_px,
                bit: edge.bit,
                confidence: f64::from(edge.confidence),
                matches_master: edge.bit == expected,
            })
        })
        .collect()
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

    #[test]
    fn f32_rounding_up_cell_size_keeps_border_saddles() {
        // `5.2_f32 as f64 == 5.199999809265137` whereas `8.3_f32 as f64 == 8.300000190734863`.
        // Using `cell_size_m = 0.0083` renders with `8.3_f32` mm where the far border saddle
        // `(11 * 8.3_f32) as f64 > 11.0 * 8.3`, which previously dropped the entire border row/col.
        let document = calib_targets::generate::puzzleboard_document(12, 12, 8.3);
        let bundle =
            calib_targets::printable::render_target_bundle(&document).expect("render bundle");
        let img = image::load_from_memory(&bundle.png_bytes).expect("decode rendered PNG");
        let cfg = PuzzleboardConfig::new(12, 12, 0.0083);
        let features = detect(&cfg, &img.to_luma8()).unwrap();
        assert_eq!(features.len(), 11 * 11, "all 121 interior saddles kept");
    }

    // ── Detail ───────────────────────────────────────────────────────────────

    fn puzzle_detail(detail: &DetectionDetail) -> &PuzzleboardDetail {
        match &detail.target {
            TargetDetail::Puzzleboard(board) => board,
            other => panic!("a PuzzleBoard detail: {other:?}"),
        }
    }

    fn rendered() -> image::GrayImage {
        synthetic_board_image().to_luma8()
    }

    fn turned(img: &image::GrayImage, quarter_turns: u32) -> image::GrayImage {
        match quarter_turns % 4 {
            0 => img.clone(),
            1 => image::imageops::rotate90(img),
            2 => image::imageops::rotate180(img),
            _ => image::imageops::rotate270(img),
        }
    }

    /// However the board is turned, the detail's features are `detect`'s,
    /// and on a clean render every edge bit read between two kept corners
    /// matches the master pattern: the edges are mapped onto the master
    /// pattern as the decoder maps them.
    #[test]
    fn detail_matches_detect_and_every_clean_edge_matches_the_master() {
        let cfg: PuzzleboardConfig = serde_json::from_value(board_config()).unwrap();
        let upright = rendered();
        for quarter_turns in 0..4 {
            let img = turned(&upright, quarter_turns);
            let detail = detect_detailed(&cfg, &img).unwrap();
            assert_eq!(detail.features, detect(&cfg, &img).unwrap());
            assert_eq!(detail.outcome, crate::DetectionOutcome::Detected);
            let board = puzzle_detail(&detail);
            assert_eq!(board.corners.len(), detail.features.len());
            assert!(board.corners.iter().all(|c| c.id.is_some()));
            let decode = board.decode.as_ref().expect("a decoded board");
            assert!(decode.edges_observed > 0);
            // Each interior edge joins two kept corners.
            assert!(
                board.edge_bits.len() >= 100,
                "turn {quarter_turns}: {} edge bits",
                board.edge_bits.len()
            );
            let wrong = board.edge_bits.iter().filter(|e| !e.matches_master).count();
            assert_eq!(wrong, 0, "turn {quarter_turns}: {wrong} edges mismatch");
            // Every endpoint is a feature's pixel.
            for edge in &board.edge_bits {
                for p in edge.endpoints_px {
                    assert!(detail.features.iter().any(|f| f.image_xy == p));
                }
            }
            let json = serde_json::to_string(&detail).unwrap();
            assert_eq!(
                serde_json::from_str::<DetectionDetail>(&json).unwrap(),
                detail
            );
        }
    }

    /// One edge's dot painted over in the other colour: the board still
    /// decodes, and that edge's bit no longer matches the master pattern.
    #[test]
    fn a_corrupted_edge_does_not_match_the_master() {
        let cfg: PuzzleboardConfig = serde_json::from_value(board_config()).unwrap();
        let mut img = rendered();
        let clean = detect_detailed(&cfg, &img).unwrap();
        let edges = &puzzle_detail(&clean).edge_bits;
        // An edge near the middle of the view, read with confidence.
        let centre = {
            let n = clean.features.len() as f64;
            let sum = clean.features.iter().fold([0.0, 0.0], |s, f| {
                [s[0] + f.image_xy[0], s[1] + f.image_xy[1]]
            });
            [sum[0] / n, sum[1] / n]
        };
        let mid = |e: &EdgeBitDetail| {
            let [a, b] = e.endpoints_px;
            [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
        };
        let edge = edges
            .iter()
            .filter(|e| e.confidence > 0.8)
            .min_by(|a, b| {
                let d = |e: &EdgeBitDetail| {
                    let m = mid(e);
                    (m[0] - centre[0]).hypot(m[1] - centre[1])
                };
                d(a).total_cmp(&d(b))
            })
            .expect("a confident edge")
            .clone();
        let [a, b] = edge.endpoints_px;
        let length = (b[0] - a[0]).hypot(b[1] - a[1]);
        let m = mid(&edge);
        // The dot is black for bit 0 and white for bit 1: paint the other.
        let paint = if edge.bit == 0 { 255 } else { 0 };
        let radius = 0.2 * length;
        for y in (m[1] - radius) as u32..=(m[1] + radius) as u32 {
            for x in (m[0] - radius) as u32..=(m[0] + radius) as u32 {
                if (f64::from(x) - m[0]).hypot(f64::from(y) - m[1]) <= radius {
                    img.put_pixel(x, y, image::Luma([paint]));
                }
            }
        }

        let detail = detect_detailed(&cfg, &img).unwrap();
        assert_eq!(detail.features, clean.features, "one dot moves no saddle");
        let corrupted = puzzle_detail(&detail)
            .edge_bits
            .iter()
            .find(|e| e.endpoints_px == edge.endpoints_px)
            .expect("the edge is still read");
        assert_ne!(corrupted.bit, edge.bit);
        assert!(!corrupted.matches_master);
        let wrong = puzzle_detail(&detail)
            .edge_bits
            .iter()
            .filter(|e| !e.matches_master)
            .count();
        assert_eq!(wrong, 1);
    }

    #[test]
    fn a_blank_image_is_not_found() {
        let cfg: PuzzleboardConfig = serde_json::from_value(board_config()).unwrap();
        let detail = detect_detailed(&cfg, &image::GrayImage::new(64, 64)).unwrap();
        assert!(
            matches!(&detail.outcome, crate::DetectionOutcome::NotFound { reason } if !reason.is_empty()),
            "{:?}",
            detail.outcome
        );
        assert!(puzzle_detail(&detail).decode.is_none());
    }

    #[test]
    fn invalid_config_is_rejected_before_detection() {
        let img = image::GrayImage::new(16, 16);
        for cfg in [
            PuzzleboardConfig::new(1, 10, 0.01),
            PuzzleboardConfig::new(10, 10, 0.0),
            PuzzleboardConfig::new(10, 10, -0.01),
            PuzzleboardConfig::new(10, 10, f64::NAN),
            PuzzleboardConfig::new(10, 10, f64::INFINITY),
        ] {
            assert!(matches!(
                detect(&cfg, &img),
                Err(DetectError::InvalidConfig(_))
            ));
        }
    }
}
