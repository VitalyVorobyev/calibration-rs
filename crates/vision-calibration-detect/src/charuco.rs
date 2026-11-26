//! ChArUco detector wrapping `chess-corners` + `calib-targets`.
//!
//! The detector returns one [`Feature`] per ChArUco corner whose marker-based
//! identification succeeded. ChArUco correspondences are *sparse* — occluded
//! or unidentified cells simply produce no feature — so the per-view feature
//! count varies; the runner decides whether a view is usable.

use calib_targets::aruco::builtins;
use calib_targets::charuco::diagnostics::{CharucoDetectDiagnostics, ComponentOutcome};
use calib_targets::charuco::{
    CharucoBoard, CharucoBoardSpec, CharucoDetection, CharucoDetector as CtCharucoDetector,
    CharucoParams, ChessCorner, MarkerLayout,
};
use calib_targets::core::GRID_TRANSFORMS_D4;
use calib_targets::detect;
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::chess_options::{
    ChessCornersConfig, apply_board_override, chess_config_for_override, cov_px,
};
use crate::detail::{
    CharucoDetail, CornerDetail, CornerQuality, DetectionDetail, MarkerDetail, RejectedMarker,
    TargetDetail,
};
use crate::{DetectError, Feature};

/// ChArUco detector configuration. Mirrors the shape of the charuco
/// variant in `vision_calibration_dataset::TargetSpec` so the
/// dispatcher can translate one to the other directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
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

impl CharucoConfig {
    /// A `rows × cols`-square ChArUco board with `square_size_m` squares and
    /// `marker_size_m` markers from `dictionary`, with the default ChESS
    /// corner stage.
    pub fn new(
        rows: u32,
        cols: u32,
        square_size_m: f64,
        marker_size_m: f64,
        dictionary: impl Into<String>,
    ) -> Self {
        Self {
            rows,
            cols,
            square_size_m,
            marker_size_m,
            dictionary: dictionary.into(),
            chess_corners: None,
        }
    }
}

/// Check `name` against the ArUco dictionaries embedded in
/// `calib-targets`. Lets the dispatcher fail fast on a manifest typo
/// before touching any image files. The error lists the supported names.
pub fn validate_dictionary(name: &str) -> Result<(), DetectError> {
    if builtins::builtin_dictionary(name).is_some() {
        Ok(())
    } else {
        Err(DetectError::InvalidConfig(format!(
            "unknown ArUco dictionary {name:?}; supported: {}",
            builtins::BUILTIN_DICTIONARY_NAMES.join(", ")
        )))
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
pub fn validate_charuco_layout(rows: u32, cols: u32, dictionary: &str) -> Result<(), DetectError> {
    let dictionary = builtins::builtin_dictionary(dictionary).ok_or_else(|| {
        DetectError::InvalidConfig(format!(
            "unknown ArUco dictionary {dictionary:?}; supported: {}",
            builtins::BUILTIN_DICTIONARY_NAMES.join(", ")
        ))
    })?;
    // Unit geometry: `CharucoBoard::new` only inspects rows/cols and the
    // dictionary capacity here; cell/marker size are placeholders.
    let spec = CharucoBoardSpec::new(rows, cols, 1.0, 0.5, dictionary)
        .with_marker_layout(MarkerLayout::OpenCvCharuco);
    CharucoBoard::new(spec)
        .map(|_| ())
        .map_err(|e| DetectError::InvalidConfig(format!("invalid charuco board: {e}")))
}

fn params_for(cfg: &CharucoConfig) -> Result<CharucoParams, DetectError> {
    validate_charuco_layout(cfg.rows, cfg.cols, &cfg.dictionary)?;
    let dictionary = builtins::builtin_dictionary(&cfg.dictionary)
        .expect("validated above; builtin lookup is deterministic");
    if !(cfg.square_size_m.is_finite() && cfg.square_size_m > 0.0) {
        return Err(DetectError::InvalidConfig(format!(
            "square_size_m ({}) must be finite and > 0",
            cfg.square_size_m
        )));
    }
    if !(cfg.marker_size_m.is_finite()
        && cfg.marker_size_m > 0.0
        && cfg.marker_size_m <= cfg.square_size_m)
    {
        return Err(DetectError::InvalidConfig(format!(
            "marker_size_m ({}) must be finite and in (0, square_size_m = {}]",
            cfg.marker_size_m, cfg.square_size_m
        )));
    }
    // The marker layout is spelled out rather than left to `MarkerLayout`'s
    // default: it fixes the marker numbering that our world coordinates are
    // indexed against, so an upstream default change must not move it
    // silently.
    let board = CharucoBoardSpec::new(
        cfg.rows,
        cfg.cols,
        cfg.square_size_m as f32,
        (cfg.marker_size_m / cfg.square_size_m) as f32,
        dictionary,
    )
    .with_marker_layout(MarkerLayout::OpenCvCharuco);
    let mut params = CharucoParams::for_board(board);
    // One corner config for the whole detection: the front-end pass below
    // and the detector's own corner re-detection both read `params.chess`.
    // Re-detection left at the default would drop the override (a threshold,
    // the Radon strategy, the covariance) for the corners it replaces.
    params.chess = chess_config_for_override(cfg.chess_corners)?;
    apply_board_override(cfg.chess_corners, &mut params.chessboard)?;
    Ok(params)
}

/// Detect the ChArUco board `cfg` describes in `luma`.
pub(crate) fn detect(
    cfg: &CharucoConfig,
    luma: &image::GrayImage,
) -> Result<Vec<Feature>, DetectError> {
    let (detector, corners) = prepare(cfg, luma)?;
    // A failed board identification (too few markers, no board in
    // frame) is "no features", not an error — same semantics as
    // the chessboard detector on a blank image.
    let detection = match detector.detect_with_corners(&detect::gray_view(luma), &corners) {
        Ok(detection) => detection,
        Err(err) if err.is_not_found() => return Ok(Vec::new()),
        Err(err) => return Err(backend_error(err)),
    };
    Ok(crate::reject_ambiguous_detection(features_of(&detection)))
}

/// [`detect`], with the detection's detail. The detector runs its
/// diagnosing path, which returns the detection `detect` returns and the
/// marker matcher's per-cell evidence besides.
pub(crate) fn detect_detailed(
    cfg: &CharucoConfig,
    luma: &image::GrayImage,
) -> Result<DetectionDetail, DetectError> {
    let (detector, corners) = prepare(cfg, luma)?;
    let (result, diagnostics) = detector.diagnose_with_corners(&detect::gray_view(luma), &corners);
    let target = |corners, markers, rejected_markers, quality| {
        TargetDetail::Charuco(CharucoDetail {
            corners,
            markers,
            rejected_markers,
            quality,
        })
    };
    let detection = match result {
        Ok(detection) => detection,
        Err(err) if err.is_not_found() => {
            return Ok(DetectionDetail::finish(
                Vec::new(),
                Vec::<CornerDetail>::new(),
                Vec::new(),
                Some(not_found_reason(&err, &diagnostics)),
                |corners| target(corners, Vec::new(), Vec::new(), CornerQuality::default()),
            ));
        }
        Err(err) => return Err(backend_error(err)),
    };

    // Detected cells map to board cells through the alignment, whose linear
    // part orders each cell's quad on the board.
    let linear = detection.alignment.matrix();
    let board = detector.board();
    let markers: Vec<MarkerDetail> = detection
        .markers
        .iter()
        .filter_map(|m| {
            // Every marker the matcher emits carries its image quad and is
            // one of the board's.
            let quad = m.corners_img?;
            let cell = board.marker_position(m.id)?;
            Some(MarkerDetail {
                id: m.id,
                cell: [cell.u, cell.v],
                quad_px: board_ordered(quad.map(|p| [p.x, p.y]), linear),
                rotation: m.rotation,
                hamming: m.hamming,
                score: f64::from(m.score),
            })
        })
        .collect();
    let rejected_markers = rejected_markers(&diagnostics, linear);
    let corners = detection
        .corners
        .iter()
        .map(|c| CornerDetail {
            grid: [c.grid.u, c.grid.v],
            id: Some(c.id),
            score: f64::from(c.score),
            axes_rad: None,
        })
        .collect();
    let quality = CornerQuality::from_calib_targets(&detection.quality);
    Ok(DetectionDetail::finish(
        features_of(&detection),
        corners,
        Vec::new(),
        None,
        |corners| target(corners, markers, rejected_markers, quality),
    ))
}

/// The detector `cfg` describes, and the corner cloud of `luma` it
/// identifies the board from.
fn prepare(
    cfg: &CharucoConfig,
    luma: &image::GrayImage,
) -> Result<(CtCharucoDetector, Vec<ChessCorner>), DetectError> {
    let params = params_for(cfg)?;
    // ChESS corner pre-detection feeds the ChArUco identifier, with the
    // same config the detector re-detects corners with.
    let corners = detect::detect_corners(luma, &params.chess);
    let detector = CtCharucoDetector::new(params)
        .map_err(|e| DetectError::InvalidConfig(format!("invalid charuco board: {e}")))?;
    Ok((detector, corners))
}

fn backend_error(err: impl std::fmt::Display) -> DetectError {
    DetectError::Backend {
        detector: "charuco",
        message: err.to_string(),
    }
}

/// One feature per ChArUco corner, before the ambiguity check.
fn features_of(detection: &CharucoDetection) -> Vec<Feature> {
    // `target_position` is metric because the board spec's
    // `cell_size` was given in metres.
    detection
        .corners
        .iter()
        .map(|corner| Feature {
            image_xy: [corner.position.x as f64, corner.position.y as f64],
            world_xyz: [
                corner.target_position.x as f64,
                corner.target_position.y as f64,
                0.0,
            ],
            cov_px: corner.position_covariance.map(cov_px),
            blur: None,
        })
        .collect()
}

/// The detector's reason for finding no board: its error, then why the
/// marker match of each grid component failed.
fn not_found_reason(
    err: &impl std::fmt::Display,
    diagnostics: &CharucoDetectDiagnostics,
) -> String {
    let mut reason = err.to_string();
    for component in &diagnostics.components {
        let rejection = component.board.as_ref().and_then(|b| b.rejection);
        let why = match (rejection, &component.outcome) {
            (Some(rejection), _) => format!("marker match rejected: {rejection:?}"),
            (None, ComponentOutcome::Failed { reason }) => reason.clone(),
            _ => continue,
        };
        reason.push_str(&format!("; component {}: {why}", component.index));
    }
    reason
}

/// The marker cells, of the grid components the detection was built from,
/// whose best decoding is another board marker than the one the chosen
/// alignment places there.
fn rejected_markers(
    diagnostics: &CharucoDetectDiagnostics,
    linear: [[i32; 2]; 2],
) -> Vec<RejectedMarker> {
    let mut out = Vec::new();
    for component in &diagnostics.components {
        let Some(board) = &component.board else {
            continue;
        };
        // Components are merged when their alignments share the linear part;
        // a component under another one did not contribute.
        let contributed = matches!(component.outcome, ComponentOutcome::Ok { .. })
            && board.chosen.is_some_and(|h| {
                GRID_TRANSFORMS_D4
                    .get(usize::from(h.rotation))
                    .is_some_and(|t| t.matrix() == linear)
            });
        if !contributed {
            continue;
        }
        for cell in &board.cells {
            let (Some(expected_id), Some(best)) = (cell.expected_id, cell.best) else {
                continue;
            };
            // Strictly better than the expected marker: a tie (a cell whose
            // bits favour no marker) names no other one.
            let other = best.marker_id != expected_id && best.score > cell.expected_score;
            if !cell.sampled || !other {
                continue;
            }
            out.push(RejectedMarker {
                quad_px: board_ordered(cell.corners_img, linear),
                decoded_id: best.marker_id,
                expected_id,
                score: f64::from(best.score),
            });
        }
    }
    out
}

/// A detected cell's image quad, given in the detected grid's `[TL, TR, BR,
/// BL]` order, in the board's order instead: first the corner the
/// alignment's linear part `linear` carries to the board cell's top-left,
/// then clockwise on the board.
fn board_ordered(quad: [[f32; 2]; 4], linear: [[i32; 2]; 2]) -> [[f64; 2]; 4] {
    const UNIT: [[i32; 2]; 4] = [[0, 0], [1, 0], [1, 1], [0, 1]];
    let mapped = UNIT.map(|[u, v]| {
        [
            linear[0][0] * u + linear[0][1] * v,
            linear[1][0] * u + linear[1][1] * v,
        ]
    });
    let min = mapped
        .iter()
        .fold([i32::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
    let mut out = [[0.0; 2]; 4];
    for (k, p) in mapped.iter().enumerate() {
        // A unimodular linear part permutes the unit square's corners, so
        // the offset is always one of them.
        let offset = [p[0] - min[0], p[1] - min[1]];
        let slot = UNIT.iter().position(|u| *u == offset).unwrap_or(k);
        out[slot] = [f64::from(quad[k][0]), f64::from(quad[k][1])];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Detect with a JSON config, as a manifest would spell it.
    fn run(image: &image::DynamicImage, config: &Value) -> Result<Vec<Feature>, DetectError> {
        let cfg: CharucoConfig = serde_json::from_value(config.clone())
            .map_err(|e| DetectError::InvalidConfig(e.to_string()))?;
        detect(&cfg, &image.to_luma8())
    }
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
        let features = run(&img, &board_config()).unwrap();
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
    fn unknown_dictionary_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let mut cfg = board_config();
        cfg["dictionary"] = json!("DICT_9X9_1");
        let err = run(&img, &cfg).unwrap_err();
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
        let err = run(&img, &cfg).unwrap_err();
        assert!(format!("{err}").contains("charuco board"));
    }

    #[test]
    fn oversized_marker_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let mut cfg = board_config();
        cfg["marker_size_m"] = json!(0.021);
        let err = run(&img, &cfg).unwrap_err();
        assert!(format!("{err}").contains("marker_size_m"));
    }

    #[test]
    fn blank_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let features = run(&img, &board_config()).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }

    #[test]
    fn covariance_is_opt_in_and_leaves_positions_unchanged() {
        let img = synthetic_board_image();
        let plain = run(&img, &board_config()).unwrap();
        assert!(plain.len() >= 4);
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

    #[test]
    fn re_detection_runs_with_the_corner_override() {
        let mut cfg: CharucoConfig = serde_json::from_value(board_config()).unwrap();
        assert_eq!(
            params_for(&cfg).unwrap().chess,
            calib_targets::detect::default_chess_config(),
            "no override keeps the detector's default"
        );
        cfg.chess_corners = Some(ChessCornersConfig {
            threshold_value: Some(30.0),
            covariance: Some(crate::CornerCovariance {
                image_noise: crate::ImageNoise::Stated(2.0),
            }),
            ..ChessCornersConfig::default()
        });
        assert_eq!(
            params_for(&cfg).unwrap().chess,
            chess_config_for_override(cfg.chess_corners).unwrap()
        );
    }

    // ── Detail ───────────────────────────────────────────────────────────────

    fn charuco_detail(detail: &DetectionDetail) -> &CharucoDetail {
        match &detail.target {
            TargetDetail::Charuco(board) => board,
            other => panic!("a ChArUco detail: {other:?}"),
        }
    }

    /// The board seen turned by `quarter_turns` quarter turns clockwise.
    fn turned(img: &image::GrayImage, quarter_turns: u32) -> image::GrayImage {
        match quarter_turns % 4 {
            0 => img.clone(),
            1 => image::imageops::rotate90(img),
            2 => image::imageops::rotate180(img),
            _ => image::imageops::rotate270(img),
        }
    }

    /// However the board is turned in the image, the detail's features are
    /// `detect`'s, its corners carry their ids, and every marker's quad
    /// starts at its square's top-left on the board: where the feature of
    /// that board corner is, for a square whose top-left is an inner corner.
    #[test]
    fn detail_matches_detect_and_orders_marker_quads_on_the_board() {
        let cfg: CharucoConfig = serde_json::from_value(board_config()).unwrap();
        let upright = synthetic_board_image().to_luma8();
        for quarter_turns in 0..4 {
            let img = turned(&upright, quarter_turns);
            let detail = detect_detailed(&cfg, &img).unwrap();
            assert_eq!(detail.features, detect(&cfg, &img).unwrap());
            assert_eq!(detail.outcome, crate::DetectionOutcome::Detected);
            let board = charuco_detail(&detail);
            assert_eq!(board.corners.len(), detail.features.len());
            assert!(board.corners.iter().all(|c| c.id.is_some()));
            assert!(board.markers.len() >= 12, "{} markers", board.markers.len());
            assert!(
                board.rejected_markers.is_empty(),
                "a clean render: {:?}",
                board.rejected_markers
            );
            let mut checked = 0;
            for marker in &board.markers {
                let [col, row] = marker.cell;
                let top_left = [f64::from(col) * 0.02, f64::from(row) * 0.02];
                let Some(feature) = detail.features.iter().find(|f| {
                    (f.world_xyz[0] - top_left[0]).abs() < 1e-6
                        && (f.world_xyz[1] - top_left[1]).abs() < 1e-6
                }) else {
                    continue; // the square's top-left is on the board's edge
                };
                let [x, y] = marker.quad_px[0];
                let err = (x - feature.image_xy[0]).hypot(y - feature.image_xy[1]);
                assert!(
                    err < 2.0,
                    "turn {quarter_turns}: marker {} quad starts {err} px from its top-left",
                    marker.id
                );
                checked += 1;
            }
            assert!(
                checked >= 6,
                "turn {quarter_turns}: {checked} markers checked"
            );
        }
    }

    /// A marker square overwritten with another marker of the board: the
    /// cell decodes as that marker, and the detail lists it with the id the
    /// board expects there.
    #[test]
    fn a_marker_with_a_wrong_id_is_a_rejected_marker() {
        let cfg: CharucoConfig = serde_json::from_value(board_config()).unwrap();
        let mut img = synthetic_board_image().to_luma8();
        let detail = detect_detailed(&cfg, &img).unwrap();
        let markers = &charuco_detail(&detail).markers;
        // Two interior markers: copy the first's square over the second's.
        let bbox = |m: &MarkerDetail| {
            let xs = m.quad_px.map(|p| p[0]);
            let ys = m.quad_px.map(|p| p[1]);
            let lo = [
                xs.iter().cloned().fold(f64::MAX, f64::min),
                ys.iter().cloned().fold(f64::MAX, f64::min),
            ];
            let hi = [
                xs.iter().cloned().fold(f64::MIN, f64::max),
                ys.iter().cloned().fold(f64::MIN, f64::max),
            ];
            (lo, hi)
        };
        let interior =
            |m: &&MarkerDetail| (2..=5).contains(&m.cell[0]) && (2..=5).contains(&m.cell[1]);
        let mut inner = markers.iter().filter(interior);
        let (source, target) = (inner.next().unwrap(), inner.next().unwrap());
        let ((s_lo, s_hi), (t_lo, _)) = (bbox(source), bbox(target));
        // Copy the square's interior, a pixel inside its corners.
        let (w, h) = (
            (s_hi[0] - s_lo[0]) as u32 - 2,
            (s_hi[1] - s_lo[1]) as u32 - 2,
        );
        let patch = image::imageops::crop_imm(&img, s_lo[0] as u32 + 1, s_lo[1] as u32 + 1, w, h)
            .to_image();
        image::imageops::replace(&mut img, &patch, t_lo[0] as i64 + 1, t_lo[1] as i64 + 1);

        let detail = detect_detailed(&cfg, &img).unwrap();
        let board = charuco_detail(&detail);
        assert!(
            board
                .rejected_markers
                .iter()
                .any(|r| r.expected_id == target.id && r.decoded_id == source.id),
            "{:?}",
            board.rejected_markers
        );
        assert!(
            !board
                .markers
                .iter()
                .any(|m| m.cell == target.cell && m.id == source.id)
        );
    }

    #[test]
    fn a_blank_image_is_not_found_with_the_detector_s_reason() {
        let cfg: CharucoConfig = serde_json::from_value(board_config()).unwrap();
        let detail = detect_detailed(&cfg, &image::GrayImage::new(64, 64)).unwrap();
        match &detail.outcome {
            crate::DetectionOutcome::NotFound { reason } => assert!(!reason.is_empty()),
            other => panic!("{other:?}"),
        }
        let json = serde_json::to_string(&detail).unwrap();
        assert_eq!(
            serde_json::from_str::<DetectionDetail>(&json).unwrap(),
            detail
        );
    }

    #[test]
    fn board_order_follows_the_alignment() {
        let quad = [[0.0_f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let f = |q: [[f64; 2]; 4]| q;
        assert_eq!(
            board_ordered(quad, [[1, 0], [0, 1]]),
            f([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        );
        // (u, v) -> (v, -u): detected +x is the board's up, so the detected
        // top-right is the board's top-left.
        assert_eq!(
            board_ordered(quad, [[0, 1], [-1, 0]]),
            f([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        );
        // A half turn: the detected bottom-right is the board's top-left.
        assert_eq!(
            board_ordered(quad, [[-1, 0], [0, -1]]),
            f([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
        );
    }

    #[test]
    fn invalid_stated_noise_is_rejected() {
        let img = image::DynamicImage::new_luma8(8, 8);
        let mut cfg = board_config();
        cfg["chess_corners"] = json!({ "covariance": { "image_noise": { "stated": null } } });
        assert!(run(&img, &cfg).is_err());
        cfg["chess_corners"] = json!({ "covariance": { "image_noise": { "stated": -1.0 } } });
        let err = run(&img, &cfg).unwrap_err();
        assert!(matches!(err, DetectError::InvalidConfig(_)), "{err}");
    }
}
