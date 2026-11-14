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

use ringgrid::{
    CenterSource, CircleRefinementMethod, Detector as RinggridDetectorImpl, TargetLayout,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::{DetectError, Feature};

/// How a ring marker's centre is estimated; see [`RinggridConfig::center_method`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RinggridCenterMethod {
    /// The projective centre recovered from the marker's fitted inner and
    /// outer ellipses. Fast, and accurate on sharp images; the default.
    #[default]
    Ellipse,
    /// A per-marker photometric fit: the marker's pixels are explained by the
    /// printed ring and code pattern, seen through the board homography and
    /// blurred by a fitted Gaussian PSF, over an affine background. It removes
    /// the blur- and code-dependent bias of edge-based centres, so use it on
    /// blurred or low-contrast images. It costs roughly 2.5x the detection
    /// time. Only markers the fit refined are returned (a marker that fell
    /// back to its ellipse centre is less accurate than its neighbours and is
    /// dropped). The default stays `Ellipse` until the benefit on real
    /// calibration data is established.
    Photometric,
}

impl RinggridCenterMethod {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Coded ring-grid detector configuration. Mirrors the geometry of the
/// ringgrid variant in `vision_calibration_dataset::TargetSpec` so the
/// dispatcher can translate one to the other directly.
///
/// The flat shape below describes exactly one point in `ringgrid`'s
/// compositional target space: a **hexagonal lattice of 16-sector coded
/// rings, with no origin fiducials**. `ringgrid` 0.9+ can also express
/// rectangular lattices and plain (uncoded) annuli; those are deliberately
/// not surfaced here — no dataset in this workspace uses them, and a config
/// knob without a consumer is a knob that rots.
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
    /// Centre estimator. Defaults to [`RinggridCenterMethod::Ellipse`]; an
    /// omitted key keeps it, and the default is left out when serializing so
    /// existing detection-cache entries stay valid.
    #[serde(default, skip_serializing_if = "RinggridCenterMethod::is_default")]
    pub center_method: RinggridCenterMethod,
}

impl RinggridConfig {
    /// Build the `ringgrid` target layout this config describes. The
    /// `ringgrid` API works in millimetres, so metric fields are scaled
    /// here; geometry validation (radii ordering, marker-vs-pitch fit) is
    /// delegated to `TargetLayout::coded_hex`.
    ///
    /// `coded_hex` takes the six board-geometry arguments and derives a
    /// geometry-based board name.
    fn board_layout(&self) -> Result<TargetLayout, DetectError> {
        let mm = |m: f64| (m * 1000.0) as f32;
        TargetLayout::coded_hex(
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

/// Detect the ring grid `cfg` describes in `luma`.
pub(crate) fn detect(
    cfg: &RinggridConfig,
    luma: &image::GrayImage,
) -> Result<Vec<Feature>, DetectError> {
    let board = cfg.board_layout()?;
    let mut detector = RinggridDetectorImpl::new(board);
    if cfg.center_method == RinggridCenterMethod::Photometric {
        detector.config_mut().circle_refinement = CircleRefinementMethod::Photometric;
    }

    // Neither `ringgrid::DetectError` variant means "no board in this
    // frame" — an empty frame yields `Ok` with zero markers.
    // `UnsupportedTarget` is a configuration bug and `IncompleteBoard`
    // can only fire under `require_complete_board`, which this wrapper
    // never sets. Both are real failures and must not be flattened into
    // an empty feature list, which would surface much later as an
    // unexplained initialisation failure.
    let result = detector
        .detect_adaptive(luma)
        .map_err(|e| DetectError::Backend {
            detector: "ringgrid",
            message: e.to_string(),
        })?;

    // A decoded marker contributes a correspondence only when its id
    // maps to a known board position (`board_xy_mm`); undecoded or
    // off-board detections are dropped. `center` is always raw image
    // pixels. `board_xy_mm` is in millimetres → convert to metres.
    Ok(crate::reject_ambiguous_detection(
        result
            .detected_markers
            .into_iter()
            .filter(|marker| {
                cfg.center_method != RinggridCenterMethod::Photometric
                    || marker.center_source == CenterSource::Photometric
            })
            .filter_map(|marker| {
                marker.board_xy_mm.map(|xy_mm| Feature {
                    image_xy: marker.center,
                    world_xyz: [xy_mm[0] * 1.0e-3, xy_mm[1] * 1.0e-3, 0.0],
                    cov_px: None,
                })
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Detect with a JSON config, as a manifest would spell it.
    fn run(image: &image::DynamicImage, config: &Value) -> Result<Vec<Feature>, DetectError> {
        let cfg: RinggridConfig = serde_json::from_value(config.clone())
            .map_err(|e| DetectError::InvalidConfig(e.to_string()))?;
        detect(&cfg, &image.to_luma8())
    }
    use ringgrid::TargetRenderOptions;
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
            .render_target_png(&TargetRenderOptions::default())
            .expect("render ring-grid PNG");
        image::DynamicImage::ImageLuma8(gray)
    }

    /// Pin the geometry the config lowers to, independently of detection.
    ///
    /// The render→detect round-trip below uses one layout for both halves, so
    /// it cannot catch a layout that is wrong but self-consistent — a swapped
    /// `rows`/`long_row_cols`, or a metres value passed where millimetres were
    /// expected, would survive it. These assertions are against values derived
    /// from the config by hand.
    #[test]
    fn config_lowers_to_the_declared_board_geometry() {
        let cfg: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        let board = cfg.board_layout().unwrap();

        // A 5-row hex board whose long rows hold 5 markers alternates
        // 5/4/5/4/5 = 23 cells. Swapping rows and long_row_cols would still
        // give 23 here, so also check the extent below.
        assert_eq!(board.n_cells(), 23);

        assert!(
            (board.pitch_mm() - 20.0).abs() < 1e-4,
            "pitch is millimetres"
        );
        assert!((board.ring().outer_radius_mm - 6.0).abs() < 1e-4);
        assert!((board.ring().inner_radius_mm - 3.0).abs() < 1e-4);
        assert!(board.is_coded(), "ring markers must be self-identifying");
        assert!(board.fiducials().is_none(), "coded boards carry no dots");

        // Extent distinguishes a 5-row/5-long-col board from its transpose.
        // In this axial layout the nearest-neighbour distance along a row is
        // `pitch * sqrt(3)` (`min_center_spacing_mm`) and the row step is
        // `1.5 * pitch`; five rows and five long-row columns give four gaps
        // on each axis.
        let (xs, ys): (Vec<f32>, Vec<f32>) = board
            .cells()
            .iter()
            .map(|c| (c.xy_mm[0], c.xy_mm[1]))
            .unzip();
        let span = |v: &[f32]| {
            v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
        };
        let expected_x = 4.0 * board.min_center_spacing_mm();
        assert!(
            (span(&xs) - expected_x).abs() < 1e-3,
            "x span should be 4 nearest-neighbour steps ({expected_x}), got {}",
            span(&xs)
        );
        let expected_y = 4.0 * 1.5 * 20.0;
        assert!(
            (span(&ys) - expected_y).abs() < 1e-3,
            "y span should be 4 row steps ({expected_y}), got {}",
            span(&ys)
        );
    }

    #[test]
    fn detects_synthetic_board() {
        let img = synthetic_board_image();
        let features = run(&img, &board_config()).unwrap();
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
    fn invalid_board_geometry_rejected() {
        // Inner radius >= outer radius is an impossible board; the wrapped
        // `TargetLayout::new` must reject it before any detection.
        let mut cfg = board_config();
        cfg["marker_inner_radius_m"] = json!(OUTER_M);
        let img = image::DynamicImage::new_luma8(64, 64);
        let err = run(&img, &cfg).unwrap_err();
        assert!(
            format!("{err}").contains("invalid ringgrid board"),
            "got: {err}"
        );
    }

    #[test]
    fn blank_image_returns_no_features() {
        let img = image::DynamicImage::new_luma8(64, 64);
        let features = run(&img, &board_config()).unwrap();
        assert!(features.is_empty(), "blank image should yield no features");
    }

    #[test]
    fn config_json_roundtrip() {
        let cfg: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        assert_eq!(cfg.center_method, RinggridCenterMethod::Ellipse);
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(
            !json.contains("center_method"),
            "default is omitted: {json}"
        );
        let back: RinggridConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);

        let mut photometric = board_config();
        photometric["center_method"] = json!("photometric");
        let cfg: RinggridConfig = serde_json::from_value(photometric).unwrap();
        assert_eq!(cfg.center_method, RinggridCenterMethod::Photometric);
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains(r#""center_method":"photometric""#), "{json}");
        let back: RinggridConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    /// The photometric centre method keeps only photometric markers and is
    /// at least as accurate as the default on a blurred render.
    #[test]
    fn photometric_centres_match_or_beat_the_default() {
        // The photometric fit models a blurred marker and declines an
        // unblurred render, so blur the render the way a defocused lens would.
        let img = image::DynamicImage::ImageLuma8(image::imageops::blur(
            &synthetic_board_image().to_luma8(),
            2.0,
        ));
        let base = run(&img, &board_config()).unwrap();
        let mut cfg = board_config();
        cfg["center_method"] = json!("photometric");
        let photo = run(&img, &cfg).unwrap();
        assert!(photo.len() >= 4, "photometric found {}", photo.len());
        assert!(photo.iter().all(|f| f.cov_px.is_none()));

        // The render is frontal at a uniform scale: every marker centre is an
        // affine image of its board position, so the fit residual of that
        // affine map measures localisation error without ground truth.
        let rms = |fs: &[Feature]| -> f64 {
            // Least squares of u = c . [X, Y, 1] by the normal equations,
            // solved with Cramer's rule.
            let det = |m: [[f64; 3]; 3]| {
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            };
            let mut ata = [[0.0; 3]; 3];
            let mut atb = [[0.0; 3]; 2];
            for f in fs {
                let a = [f.world_xyz[0], f.world_xyz[1], 1.0];
                for i in 0..3 {
                    for j in 0..3 {
                        ata[i][j] += a[i] * a[j];
                    }
                    atb[0][i] += a[i] * f.image_xy[0];
                    atb[1][i] += a[i] * f.image_xy[1];
                }
            }
            let d = det(ata);
            let solve = |b: [f64; 3]| -> [f64; 3] {
                let mut c = [0.0; 3];
                for k in 0..3 {
                    let mut m = ata;
                    for r in 0..3 {
                        m[r][k] = b[r];
                    }
                    c[k] = det(m) / d;
                }
                c
            };
            let c = [solve(atb[0]), solve(atb[1])];
            let mut sum = 0.0;
            for f in fs {
                let a = [f.world_xyz[0], f.world_xyz[1], 1.0];
                for (axis, ck) in c.iter().enumerate() {
                    let pred: f64 = (0..3).map(|i| ck[i] * a[i]).sum();
                    sum += (pred - f.image_xy[axis]).powi(2);
                }
            }
            (sum / fs.len() as f64).sqrt()
        };
        let (rb, rp) = (rms(&base), rms(&photo));
        // On this render the ellipse centres scatter ~0.5 px and the
        // photometric ones ~0.015 px; the bound is loose on purpose.
        assert!(rp < 0.5 * rb, "photometric {rp} px vs default {rb} px");
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn center_method_changes_the_cache_key() {
        let base = RinggridConfig {
            pitch_m: 0.02,
            rows: 5,
            long_row_cols: 5,
            marker_outer_radius_m: 0.006,
            marker_inner_radius_m: 0.003,
            marker_ring_width_m: 0.001,
            center_method: RinggridCenterMethod::Ellipse,
        };
        let photometric = RinggridConfig {
            center_method: RinggridCenterMethod::Photometric,
            ..base
        };
        let key =
            |c| crate::CacheKey::for_detection(b"img", &crate::DetectorConfig::Ringgrid(c), None);
        assert_ne!(key(base), key(photometric));
    }
}
