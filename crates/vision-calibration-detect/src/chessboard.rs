//! Chessboard detector wrapping `chess-corners` + `calib-targets`.
//!
//! The detector returns one [`Feature`] per detected interior corner
//! whose grid-coordinate disambiguation succeeded. Corners without a
//! grid index are filtered out — calibration consumes 2D-3D
//! correspondences, so an unindexed corner is unusable.

use calib_targets::blur::chessboard_blur;
use calib_targets::chessboard::{self, ChessboardParams};
use calib_targets::detect;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::chess_options::{
    ChessCornersConfig, apply_board_override, chess_config_for_override, cov_px,
};
use crate::corner_blur::feature_blurs;
use crate::detail::{
    ChessboardDetail, CornerDetail, CornerQuality, DetectionDetail, PointRejection, RejectedPoint,
    TargetDetail,
};
use crate::{DetectError, Feature};

/// Chessboard detector configuration. Mirrors the shape of the
/// chessboard variant in
/// `vision_calibration_dataset::TargetSpec` so the dispatcher can
/// translate one to the other directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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
    /// Measure the blur at each corner ([`Feature::blur`]): the covariance
    /// `Σ` (px², image axes, including the pixel aperture) of the Gaussian
    /// kernel that best blurs the printed squares around the corner into the
    /// image, with the covariance of that estimate. A corner the estimate
    /// declines (blur too wide for the squares, sharper than a pixel, too
    /// few labelled neighbours) has none. It costs a few milliseconds per
    /// corner, single-threaded (a 476-corner board adds about 1–2 s per image),
    /// hence the default `false`. Corner positions and covariances are the
    /// same with or without it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub measure_blur: bool,
}

impl ChessboardConfig {
    /// A chessboard of `rows × cols` interior corners and `square_size_m`
    /// squares, with the default ChESS corner stage.
    pub fn new(rows: u32, cols: u32, square_size_m: f64) -> Self {
        Self {
            rows,
            cols,
            square_size_m,
            chess_corners: None,
            measure_blur: false,
        }
    }

    /// The same config with [`measure_blur`](Self::measure_blur) set.
    pub fn with_measure_blur(mut self, measure_blur: bool) -> Self {
        self.measure_blur = measure_blur;
        self
    }
}

/// Detect the chessboard `cfg` describes in `luma`: one [`Feature`] per
/// interior corner whose grid index falls inside the board.
pub(crate) fn detect(
    cfg: &ChessboardConfig,
    luma: &image::GrayImage,
) -> Result<Vec<Feature>, DetectError> {
    detect_detailed(cfg, luma).map(|detail| detail.features)
}

/// Unlabelled corner candidates are listed within the labelled corners'
/// bounding box widened by this many cells.
const UNLABELLED_MARGIN_CELLS: f64 = 2.0;

/// With nothing labelled, the strongest this many corner candidates are
/// listed.
const UNLABELLED_STRONGEST: usize = 1000;

/// [`detect`], with the detection's detail.
pub(crate) fn detect_detailed(
    cfg: &ChessboardConfig,
    luma: &image::GrayImage,
) -> Result<DetectionDetail, DetectError> {
    if cfg.rows < 2 || cfg.cols < 2 || cfg.rows > i32::MAX as u32 || cfg.cols > i32::MAX as u32 {
        return Err(DetectError::InvalidConfig(format!(
            "chessboard rows={} cols={} must be in 2..={}",
            cfg.rows,
            cfg.cols,
            i32::MAX
        )));
    }
    if !(cfg.square_size_m.is_finite() && cfg.square_size_m > 0.0) {
        return Err(DetectError::InvalidConfig(format!(
            "chessboard square_size_m ({}) must be finite and > 0",
            cfg.square_size_m
        )));
    }

    // The underlying detector auto-labels corners from
    // intersection clustering — `rows`/`cols` from our config are
    // used only for output validation, not as input parameters.
    let mut board_params = ChessboardParams::default();
    apply_board_override(cfg.chess_corners, &mut board_params)?;
    let chess_cfg = chess_config_for_override(cfg.chess_corners)?;

    // `detect::detect_chessboard`, in its two stages, so the corner cloud the
    // board was labelled from is at hand: the corner pass, then the board
    // detector on its corners, with the frame's noise level recorded on the
    // detection's quality.
    let frame = chessboard::detect_corner_frame(&detect::gray_view(luma), &chess_cfg);
    let detector =
        chessboard::ChessboardDetector::new(board_params).map_err(|err| DetectError::Backend {
            detector: "chessboard",
            message: err.to_string(),
        })?;
    let target = |corners, cell_size_px, quality| {
        TargetDetail::Chessboard(ChessboardDetail {
            corners,
            cell_size_px,
            quality,
        })
    };
    // No board in frame is "no features", not an error.
    let Some(mut detection) = detector.detect(&frame.corners) else {
        let rejected = unlabelled(&frame.corners, &[], None);
        return Ok(DetectionDetail::finish(
            Vec::new(),
            Vec::<CornerDetail>::new(),
            rejected,
            Some("no chessboard was detected in the image".to_string()),
            |corners| target(corners, None, CornerQuality::default()),
        ));
    };
    detection.quality = detection.quality.with_noise_sigma(frame.noise_sigma);

    // Filter to corners whose grid index falls inside the board, then
    // lift to 3D using the supplied square size.
    let (max_u, max_v) = label_bounds(
        detection.corners.iter().map(|c| (c.grid.u, c.grid.v)),
        cfg.rows,
        cfg.cols,
    );
    // The blur at every corner, in `detection.corners` order, measured on
    // the image the corners were found in.
    let mut blurs = if cfg.measure_blur {
        feature_blurs(chessboard_blur(luma, &detection))
    } else {
        Vec::new()
    };
    blurs.resize(detection.corners.len(), None);
    let mut rejected = unlabelled(
        &frame.corners,
        &detection.corners,
        detection.cell_size.map(f64::from),
    );
    let mut features = Vec::new();
    let mut corners = Vec::new();
    for (corner, blur) in detection.corners.iter().zip(blurs) {
        let grid = corner.grid;
        let image_xy = [corner.position.x as f64, corner.position.y as f64];
        let world_xyz = [
            grid.u as f64 * cfg.square_size_m,
            grid.v as f64 * cfg.square_size_m,
            0.0,
        ];
        if grid.u < 0 || grid.v < 0 || grid.u >= max_u || grid.v >= max_v {
            rejected.push(RejectedPoint {
                image_xy,
                reason: PointRejection::OutsideBoard,
                world_xyz: Some(world_xyz),
                grid: Some([grid.u, grid.v]),
                id: None,
            });
            continue;
        }
        features.push(
            Feature::new(image_xy, world_xyz)
                .with_cov_px(corner.position_covariance.map(cov_px))
                .with_blur(blur),
        );
        corners.push(CornerDetail {
            grid: [grid.u, grid.v],
            id: None,
            score: f64::from(corner.score),
            axes_rad: frame.corners.get(corner.input_index).and_then(axes_rad),
        });
    }
    let cell_size_px = detection.cell_size.map(f64::from);
    let quality = CornerQuality::from_calib_targets(&detection.quality);
    Ok(DetectionDetail::finish(
        features,
        corners,
        rejected,
        None,
        |corners| target(corners, cell_size_px, quality),
    ))
}

/// The two grid-axis directions the corner stage measured at `corner`,
/// radians; `None` when it fitted neither (an axis without information has
/// a standard error of π).
fn axes_rad(corner: &chessboard::ChessCorner) -> Option<[f64; 2]> {
    let informative = |a: &calib_targets::core::AxisEstimate| {
        a.fit_std_error < std::f32::consts::PI && a.angle.is_finite()
    };
    corner
        .axes
        .iter()
        .any(informative)
        .then(|| corner.axes.map(|a| f64::from(a.angle)))
}

/// The corner candidates of `cloud` that no corner of `labelled` came from,
/// as rejected points: those within the labelled corners' bounding box
/// widened by [`UNLABELLED_MARGIN_CELLS`] cells of `cell_size_px`, or, with
/// nothing labelled, the [`UNLABELLED_STRONGEST`] strongest. In the cloud's
/// order, or by decreasing strength for the strongest.
fn unlabelled(
    cloud: &[chessboard::ChessCorner],
    labelled: &[chessboard::ChessboardCorner],
    cell_size_px: Option<f64>,
) -> Vec<RejectedPoint> {
    let point = |c: &chessboard::ChessCorner| RejectedPoint {
        image_xy: [f64::from(c.position.x), f64::from(c.position.y)],
        reason: PointRejection::Unlabelled {
            strength: f64::from(c.strength),
        },
        world_xyz: None,
        grid: None,
        id: None,
    };
    if labelled.is_empty() {
        let mut order: Vec<usize> = (0..cloud.len()).collect();
        // Strongest first; the index breaks ties, so the list is the same on
        // every run.
        order.sort_by(|&a, &b| {
            cloud[b]
                .strength
                .total_cmp(&cloud[a].strength)
                .then(a.cmp(&b))
        });
        return order
            .into_iter()
            .take(UNLABELLED_STRONGEST)
            .map(|i| point(&cloud[i]))
            .collect();
    }
    let taken: std::collections::BTreeSet<usize> = labelled.iter().map(|c| c.input_index).collect();
    let margin = UNLABELLED_MARGIN_CELLS * cell_size_px.unwrap_or(0.0);
    let (lo, hi) = labelled.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(lo, hi), c| {
            let (x, y) = (f64::from(c.position.x), f64::from(c.position.y));
            ([lo[0].min(x), lo[1].min(y)], [hi[0].max(x), hi[1].max(y)])
        },
    );
    cloud
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            let (x, y) = (f64::from(c.position.x), f64::from(c.position.y));
            !taken.contains(i)
                && (lo[0] - margin..=hi[0] + margin).contains(&x)
                && (lo[1] - margin..=hi[1] + margin).contains(&y)
        })
        .map(|(_, c)| point(c))
        .collect()
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
    use serde_json::Value;

    /// Detect with a JSON config, as a manifest would spell it.
    fn run(image: &image::DynamicImage, config: &Value) -> Result<Vec<Feature>, DetectError> {
        let cfg: ChessboardConfig = serde_json::from_value(config.clone())
            .map_err(|e| DetectError::InvalidConfig(e.to_string()))?;
        detect(&cfg, &image.to_luma8())
    }
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
    #[ignore = "needs the actual data/kuka_1/01.png image; run with --ignored"]
    fn keeps_every_corner_of_a_landscape_board() {
        let cfg = json!({ "rows": 17, "cols": 28, "square_size_m": 0.02 });
        let features = run(&kuka_view(), &cfg).unwrap();
        assert_eq!(features.len(), 17 * 28);
        let [x, y] = extent(&features);
        assert!((x - 27.0 * 0.02).abs() < 1e-12 && (y - 16.0 * 0.02).abs() < 1e-12);
    }

    #[test]
    #[ignore = "needs the actual data/kuka_1/01.png image; run with --ignored"]
    fn radon_strategy_detects_the_board() {
        let cfg = json!({
            "rows": 17,
            "cols": 28,
            "square_size_m": 0.02,
            "chess_corners": { "strategy": "radon" },
        });
        let features = run(&kuka_view(), &cfg).unwrap();
        // Radon misses a few of the 476 corners at the edge of this view.
        assert!(features.len() >= 470, "{} features", features.len());
    }

    #[test]
    fn empty_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let cfg = json!({ "rows": 9, "cols": 6, "square_size_m": 0.025 });
        let features = run(&img, &cfg).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }

    /// A frontal rendered board of 6 rows x 8 cols of inner corners.
    fn synthetic_board_image() -> image::DynamicImage {
        let document = calib_targets::generate::chessboard_document(6, 8, 20.0);
        let bundle =
            calib_targets::printable::render_target_bundle(&document).expect("render bundle");
        image::load_from_memory(&bundle.png_bytes).expect("decode rendered PNG")
    }

    fn board_config() -> Value {
        json!({ "rows": 6, "cols": 8, "square_size_m": 0.020 })
    }

    #[test]
    fn covariance_is_opt_in_and_leaves_positions_unchanged() {
        let img = synthetic_board_image();
        let plain = run(&img, &board_config()).unwrap();
        assert!(plain.len() >= 4, "got {} corners", plain.len());
        assert!(plain.iter().all(|f| f.cov_px.is_none()));

        let mut cfg = board_config();
        cfg["chess_corners"] = json!({ "covariance": { "image_noise": { "stated": 2.0 } } });
        let with = run(&img, &cfg).unwrap();
        assert_eq!(with.len(), plain.len());
        for (a, b) in with.iter().zip(&plain) {
            assert_eq!(a.image_xy, b.image_xy);
            assert_eq!(a.world_xyz, b.world_xyz);
            let [xx, xy, yy] = a.cov_px.expect("covariance requested");
            assert!(
                xx > 0.0 && yy > 0.0 && xx * yy - xy * xy > 0.0,
                "{:?}",
                a.cov_px
            );
        }
    }

    /// A board seen through a lens of Gaussian blur σ reports the rendered
    /// `tr Σ / 2` (the lens's `σ²` plus the pixel aperture's term) within
    /// its stated standard errors, and measuring it moves no corner.
    #[test]
    fn measured_blur_matches_the_rendered_blur() {
        use crate::corner_blur::known_blur::{assert_matches_render, camera_image};
        // 7 × 9 squares of 24 px: σ = 2 px is 0.08 of a cell, well inside
        // the estimator's range (it declines beyond 0.3).
        let document = calib_targets::generate::chessboard_document(6, 8, 25.4);
        let cfg = ChessboardConfig::new(6, 8, 0.0254);
        for sigma in [1.0, 2.0] {
            let img = camera_image(document.clone(), 24, 250.0, sigma, 1.0, 7);
            let plain = detect(&cfg, &img).unwrap();
            assert_eq!(plain.len(), 48, "every corner of the board");
            assert!(plain.iter().all(|f| f.blur.is_none()));

            let with = detect(&cfg.with_measure_blur(true), &img).unwrap();
            assert_eq!(with.len(), plain.len());
            for (a, b) in with.iter().zip(&plain) {
                assert_eq!(a.image_xy, b.image_xy);
                assert_eq!(a.world_xyz, b.world_xyz);
                assert_eq!(a.cov_px, b.cov_px);
            }
            assert_matches_render(&with, sigma);
        }
    }

    #[test]
    fn measure_blur_round_trips_and_defaults_to_off() {
        let cfg: ChessboardConfig = serde_json::from_value(board_config()).unwrap();
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
            serde_json::from_str::<ChessboardConfig>(&json).unwrap(),
            with
        );
    }

    // ── Detail ───────────────────────────────────────────────────────────────

    /// The detail's features are `detect`'s, its corners align with them,
    /// and the corners are those `detect::detect_chessboard` (the one-call
    /// form of the two stages the detail runs) labels.
    fn assert_detail_matches_detect(img: &image::GrayImage, cfg: &ChessboardConfig) {
        let detail = detect_detailed(cfg, img).unwrap();
        assert_eq!(detail.features, detect(cfg, img).unwrap());
        assert_eq!(detail.outcome, crate::DetectionOutcome::Detected);
        let TargetDetail::Chessboard(board) = &detail.target else {
            panic!("a chessboard detail: {:?}", detail.target);
        };
        assert_eq!(board.corners.len(), detail.features.len());
        assert!(board.cell_size_px.is_some_and(|s| s > 0.0));

        let mut params = ChessboardParams::default();
        apply_board_override(cfg.chess_corners, &mut params).unwrap();
        let chess = chess_config_for_override(cfg.chess_corners).unwrap();
        let reference = detect::detect_chessboard(img, &chess, &params).unwrap();
        for (corner, feature) in board.corners.iter().zip(&detail.features) {
            let c = reference
                .corners
                .iter()
                .find(|c| [c.grid.u, c.grid.v] == corner.grid)
                .expect("every kept corner is labelled by detect_chessboard");
            assert_eq!(
                feature.image_xy,
                [f64::from(c.position.x), f64::from(c.position.y)]
            );
            assert_eq!(corner.score, f64::from(c.score));
            assert_eq!(corner.id, None);
            assert_eq!(
                feature.world_xyz,
                [
                    f64::from(corner.grid[0]) * cfg.square_size_m,
                    f64::from(corner.grid[1]) * cfg.square_size_m,
                    0.0
                ]
            );
        }
        assert!(
            board.corners.iter().all(|c| c.axes_rad.is_some()),
            "the ChESS stage fits both axes at a board corner"
        );
        // An unlabelled candidate is never a kept corner.
        for r in &detail.rejected {
            if matches!(r.reason, PointRejection::Unlabelled { .. }) {
                assert!(!detail.features.iter().any(|f| f.image_xy == r.image_xy));
            }
        }
        let json = serde_json::to_string(&detail).unwrap();
        assert_eq!(
            serde_json::from_str::<DetectionDetail>(&json).unwrap(),
            detail
        );
    }

    #[test]
    fn detail_matches_detect_on_a_rendered_board() {
        let img = synthetic_board_image().to_luma8();
        let cfg: ChessboardConfig = serde_json::from_value(board_config()).unwrap();
        assert_detail_matches_detect(&img, &cfg);
        let mut with_cov = cfg;
        with_cov.chess_corners = Some(ChessCornersConfig {
            covariance: Some(crate::CornerCovariance::new(crate::ImageNoise::Stated(2.0))),
            ..ChessCornersConfig::default()
        });
        assert_detail_matches_detect(&img, &with_cov);
        let detail = detect_detailed(&with_cov, &img).unwrap();
        let TargetDetail::Chessboard(board) = detail.target else {
            unreachable!()
        };
        assert_eq!(board.quality.noise_sigma, Some(2.0));
        assert!(board.quality.median_sigma_px.is_some());
    }

    #[test]
    #[ignore = "needs the actual data/kuka_1/01.png image; run with --ignored"]
    fn detail_matches_detect_on_a_kuka_view() {
        let img = kuka_view().to_luma8();
        let cfg = ChessboardConfig::new(17, 28, 0.02);
        assert_detail_matches_detect(&img, &cfg);
        let detail = detect_detailed(&cfg, &img).unwrap();
        assert_eq!(detail.features.len(), 17 * 28);
        let TargetDetail::Chessboard(board) = &detail.target else {
            unreachable!()
        };
        assert!(board.quality.local_residual_count > 300);
    }

    /// A board declared narrower than the one in view: the labels past the
    /// declared columns are rejected as outside the board, with the target
    /// points they would have had.
    #[test]
    fn corners_past_the_declared_board_are_rejected_with_their_labels() {
        let img = synthetic_board_image().to_luma8();
        let cfg = ChessboardConfig::new(6, 7, 0.02);
        let detail = detect_detailed(&cfg, &img).unwrap();
        assert_eq!(detail.features.len(), 6 * 7);
        let outside: Vec<_> = detail
            .rejected
            .iter()
            .filter(|r| r.reason == PointRejection::OutsideBoard)
            .collect();
        assert_eq!(outside.len(), 6, "the eighth column");
        for r in outside {
            let [u, v] = r.grid.expect("a labelled corner");
            assert_eq!(u, 7);
            assert_eq!(r.world_xyz, Some([7.0 * 0.02, f64::from(v) * 0.02, 0.0]));
        }
    }

    #[test]
    fn a_blank_image_is_not_found_with_the_strongest_candidates() {
        let img = image::GrayImage::new(64, 64);
        let cfg: ChessboardConfig = serde_json::from_value(board_config()).unwrap();
        let detail = detect_detailed(&cfg, &img).unwrap();
        assert!(detail.features.is_empty());
        assert!(
            matches!(&detail.outcome, crate::DetectionOutcome::NotFound { reason } if reason.contains("no chessboard")),
            "{:?}",
            detail.outcome
        );
        assert!(detail.rejected.len() <= UNLABELLED_STRONGEST);
    }

    #[test]
    fn invalid_stated_noise_is_rejected_before_detection() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let mut cfg = board_config();
        cfg["chess_corners"] = json!({ "covariance": { "image_noise": { "stated": -1.0 } } });
        let err = run(&img, &cfg).unwrap_err();
        assert!(matches!(err, DetectError::InvalidConfig(_)), "{err}");
    }
}
