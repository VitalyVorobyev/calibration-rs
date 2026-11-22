//! Ring-grid detector wrapping the `ringgrid` crate.
//!
//! A ring grid is a hexagonal or rectangular lattice of ring markers
//! ([`RinggridLattice`]). Coded markers are self-identifying: each decoded
//! marker carries a unique codebook id whose board position is known from the
//! layout, so a single partial view yields globally consistent
//! correspondences. Plain markers carry no identity; a plain board is anchored
//! by a triad of filled origin dots printed near its centre
//! ([`RinggridCoding`]). The detector returns one [`Feature`] per marker that
//! maps to a board position, with metric target coordinates in the board's
//! own frame.
//!
//! Detection runs in adaptive-scale mode (`Detector::detect_adaptive`): the
//! marker pixel size is unknown for an arbitrary dataset image, and adaptive
//! mode auto-selects scale tiers rather than relying on a hand-tuned prior.

use ringgrid::{
    CenterSource, CircleRefinementMethod, Detector as RinggridDetectorImpl, OriginDots,
    TargetLayout,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::{DetectError, Feature, FeatureBlur};

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
    /// dropped). A returned marker also carries the blur the fit measured
    /// ([`Feature::blur`]) unless its fitted width exceeds a quarter of the
    /// marker's outer radius. The default stays `Ellipse` until the benefit
    /// on real calibration data is established.
    Photometric,
}

impl RinggridCenterMethod {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// The lattice the ring markers sit on; see [`RinggridConfig::lattice`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RinggridLattice {
    /// A hexagonal lattice: rows alternate between `long_row_cols` markers
    /// and one fewer, offset by half a step. The default.
    #[default]
    Hex,
    /// A rectangular lattice of `rows × long_row_cols` markers, `pitch_m`
    /// apart along both axes.
    Rect,
}

impl RinggridLattice {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// How the ring markers are told apart; see [`RinggridConfig::coding`].
///
/// In JSON: `"coded"`, or `{"plain": {"origin_dots": true}}`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RinggridCoding {
    /// Every marker carries a 16-sector code between its two rings, so each
    /// detected marker identifies its own board position and any partial
    /// view is labelled in the board frame. The default.
    #[default]
    Coded,
    /// Plain rings without a code. The markers do not identify themselves:
    /// the board frame comes from the origin dots.
    Plain {
        /// The board carries a triad of filled origin dots, placed by the
        /// target generator in the lattice gaps near the board centre.
        /// Defaults to `true`.
        ///
        /// The detector places a view on the board through the dots: it tries
        /// each placement of the visible markers that puts the dot triad
        /// among them, and keeps the one that puts the dots on dark ink when
        /// it clearly beats every other. A view is therefore labelled in the
        /// board frame whenever all three dots are in it among detected
        /// markers, however small a part of the board it shows. A view
        /// without the dots, or with only one or two of them, yields no
        /// features rather than a guessed position.
        ///
        /// `false` (a plain board without dots) is not supported: such a
        /// board is labelled only up to the lattice symmetry, so its views
        /// carry no board coordinates. A config that sets it is rejected
        /// before detection.
        #[serde(default = "origin_dots_default")]
        origin_dots: bool,
    },
}

fn origin_dots_default() -> bool {
    true
}

impl RinggridCoding {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// Ring-grid detector configuration. Mirrors the geometry of the ringgrid
/// variant in `vision_calibration_dataset::TargetSpec` so the dispatcher can
/// translate one to the other directly.
///
/// The board is a [`lattice`](Self::lattice) of ring markers, either coded or
/// plain with origin dots ([`coding`](Self::coding)). The defaults describe a
/// hexagonal lattice of coded rings, so a config that leaves both out keeps
/// its meaning and its detection-cache key.
///
/// Detection reports a feature for a marker only when the view's placement
/// on the board is corroborated (by agreeing decoded codes, or by the origin
/// dots); a view without one yields no features rather than guessed
/// positions. A marker is reported only with a perspective-corrected centre
/// (or, with [`RinggridCenterMethod::Photometric`], a photometric one) and a
/// confidence of at least 0.03, the `ringgrid` detector's default floor on
/// its centre-accuracy score, which this wrapper keeps.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct RinggridConfig {
    /// Center-to-center spacing between adjacent markers in metres. On a
    /// rectangular lattice it is both the row and the column step.
    pub pitch_m: f64,
    /// Number of marker rows.
    pub rows: u32,
    /// Number of columns: on a hexagonal lattice, of the longest
    /// (even-indexed) rows; on a rectangular lattice, of every row.
    pub long_row_cols: u32,
    /// Outer ring radius in metres.
    pub marker_outer_radius_m: f64,
    /// Inner ring radius in metres.
    pub marker_inner_radius_m: f64,
    /// Width of each ring band in metres. Coded markers need it, greater
    /// than zero; plain markers ignore it, so it may be left out (`0`) for
    /// them.
    #[serde(default)]
    pub marker_ring_width_m: f64,
    /// Centre estimator. Defaults to [`RinggridCenterMethod::Ellipse`]; an
    /// omitted key keeps it, and the default is left out when serializing so
    /// existing detection-cache entries stay valid.
    #[serde(default, skip_serializing_if = "RinggridCenterMethod::is_default")]
    pub center_method: RinggridCenterMethod,
    /// Marker lattice. Defaults to [`RinggridLattice::Hex`]; the default is
    /// left out when serializing, like `center_method`'s.
    #[serde(default, skip_serializing_if = "RinggridLattice::is_default")]
    pub lattice: RinggridLattice,
    /// Marker coding. Defaults to [`RinggridCoding::Coded`]; the default is
    /// left out when serializing, like `center_method`'s.
    #[serde(default, skip_serializing_if = "RinggridCoding::is_default")]
    pub coding: RinggridCoding,
}

impl RinggridConfig {
    /// A coded ring grid on a hexagonal lattice of `rows` rows,
    /// `long_row_cols` markers in the long rows and `pitch_m` spacing, with
    /// markers of the given outer and inner radii and ring width (metres),
    /// and the default centre estimator. Set [`lattice`](Self::lattice) and
    /// [`coding`](Self::coding) for the other boards.
    pub fn new(
        pitch_m: f64,
        rows: u32,
        long_row_cols: u32,
        marker_outer_radius_m: f64,
        marker_inner_radius_m: f64,
        marker_ring_width_m: f64,
    ) -> Self {
        Self {
            pitch_m,
            rows,
            long_row_cols,
            marker_outer_radius_m,
            marker_inner_radius_m,
            marker_ring_width_m,
            center_method: RinggridCenterMethod::default(),
            lattice: RinggridLattice::default(),
            coding: RinggridCoding::default(),
        }
    }

    /// Build the `ringgrid` target layout this config describes. The
    /// `ringgrid` API works in millimetres, so metric fields are scaled
    /// here; geometry validation (radii ordering, marker-vs-pitch fit, room
    /// for the origin dots) is delegated to the `TargetLayout` constructors,
    /// which derive a geometry-based board name. A plain board without
    /// origin dots is rejected here, before any detection.
    fn board_layout(&self) -> Result<TargetLayout, DetectError> {
        let mm = |m: f64| (m * 1000.0) as f32;
        let pitch = mm(self.pitch_m);
        let (rows, cols) = (self.rows as usize, self.long_row_cols as usize);
        let outer = mm(self.marker_outer_radius_m);
        let inner = mm(self.marker_inner_radius_m);
        let ring_width = mm(self.marker_ring_width_m);
        let layout = match (self.lattice, self.coding) {
            (RinggridLattice::Hex, RinggridCoding::Coded) => {
                TargetLayout::coded_hex(pitch, rows, cols, outer, inner, ring_width)
            }
            (RinggridLattice::Rect, RinggridCoding::Coded) => {
                TargetLayout::coded_rect(pitch, rows, cols, outer, inner, ring_width)
            }
            (RinggridLattice::Hex, RinggridCoding::Plain { origin_dots: true }) => {
                TargetLayout::plain_hex(pitch, rows, cols, outer, inner, OriginDots::Auto)
            }
            (RinggridLattice::Rect, RinggridCoding::Plain { origin_dots: true }) => {
                TargetLayout::plain_rect(pitch, rows, cols, outer, inner, OriginDots::Auto)
            }
            (_, RinggridCoding::Plain { origin_dots: false }) => {
                return Err(DetectError::InvalidConfig(
                    "a plain ring grid without origin dots is not supported: its views carry \
                     no board coordinates"
                        .to_string(),
                ));
            }
        };
        layout.map_err(|e| DetectError::InvalidConfig(format!("invalid ringgrid board: {e}")))
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

    // No `ringgrid::DetectError` means "no board in this frame" — an empty
    // frame yields `Ok` with zero markers. `IncompleteBoard` can only fire
    // under `require_complete_board`, which this wrapper never sets. An
    // error is a real failure and must not be flattened into an empty
    // feature list, which would surface much later as an unexplained
    // initialisation failure.
    let result = detector
        .detect_adaptive(luma)
        .map_err(|e| DetectError::Backend {
            detector: "ringgrid",
            message: e.to_string(),
        })?;

    // A marker contributes a correspondence only when it maps to a known
    // board position (`board_xy_mm`), which `ringgrid` sets only for a view
    // whose board placement is corroborated: a coded view through its
    // decoded ids, a plain one through its origin dots. Markers of an
    // unanchored view, and off-board ones, carry none and are dropped. With
    // photometric centres, only markers whose centre the fit produced are
    // kept. `center` is always raw image pixels. `board_xy_mm` is in
    // millimetres → convert to metres.
    Ok(crate::reject_ambiguous_detection(
        result
            .detected_markers
            .into_iter()
            .filter(|marker| {
                cfg.center_method != RinggridCenterMethod::Photometric
                    || marker.center_source == CenterSource::Photometric
            })
            .filter_map(|marker| {
                // `ringgrid` reports a blur only with a photometric centre.
                let blur = marker.blur.as_ref().and_then(feature_blur);
                marker.board_xy_mm.map(|xy_mm| {
                    Feature::new(marker.center, [xy_mm[0] * 1.0e-3, xy_mm[1] * 1.0e-3, 0.0])
                        .with_blur(blur)
                })
            })
            .collect(),
    ))
}

/// A marker's blur as a [`FeatureBlur`]: `ringgrid`'s `(σ, a, b)` mapped to
/// the PSF covariance and its covariance by [`psf_from_sigma_anisotropy`].
/// `None` (the marker keeps its centre, without a blur) when the estimate is
/// not finite, the PSF covariance is not positive definite or its covariance
/// has a negative variance, none of which a converged fit produces (`ringgrid`
/// bounds `|[a, b]| ≤ 0.9`, so `Σ` has eigenvalues `σ²(1 ± |[a, b]|) > 0`).
fn feature_blur(blur: &ringgrid::BlurEstimate) -> Option<FeatureBlur> {
    let (psf_px, cov) = psf_from_sigma_anisotropy(blur.sigma_px, blur.anisotropy, blur.cov);
    let [xx, xy, yy] = psf_px;
    let finite = psf_px
        .iter()
        .chain(cov.iter().flatten())
        .all(|v| v.is_finite());
    let positive = xx > 0.0 && yy > 0.0 && xx * yy - xy * xy > 0.0;
    let variances = (0..3).all(|i| cov[i][i] >= 0.0);
    (finite && positive && variances).then(|| FeatureBlur::new(psf_px, cov))
}

/// The PSF covariance `[xx, xy, yy]` (px²) of `ringgrid`'s blur parameters,
/// and its covariance (px⁴).
///
/// `ringgrid` parameterises the PSF covariance as `σ² [[1 + a, b], [b, 1 −
/// a]]` with `σ` the edge width (px, including the pixel's area
/// integration) and `[a, b]` the anisotropy, so
///
/// ```text
/// Σ = [xx, xy, yy] = [σ²(1 + a), σ² b, σ²(1 − a)]
/// ```
///
/// and `tr Σ / 2 = σ²`. Its Jacobian in `(σ, a, b)` is
///
/// ```text
/// J = [[2σ(1 + a),  σ²,  0 ],
///      [2σ b,       0,   σ²],
///      [2σ(1 − a), −σ²,  0 ]]
/// ```
///
/// (`det J = 4σ⁵`, so the map is one to one for `σ > 0`), and the covariance
/// of `Σ` is `J C Jᵀ` with `C` the covariance of `(σ, a, b)`, symmetrized
/// against rounding.
fn psf_from_sigma_anisotropy(
    sigma: f64,
    [a, b]: [f64; 2],
    c: [[f64; 3]; 3],
) -> ([f64; 3], [[f64; 3]; 3]) {
    let s2 = sigma * sigma;
    let psf = [s2 * (1.0 + a), s2 * b, s2 * (1.0 - a)];
    let j = [
        [2.0 * sigma * (1.0 + a), s2, 0.0],
        [2.0 * sigma * b, 0.0, s2],
        [2.0 * sigma * (1.0 - a), -s2, 0.0],
    ];
    // J C Jᵀ, then the mean of it and its transpose.
    let jc: [[f64; 3]; 3] =
        std::array::from_fn(|r| std::array::from_fn(|k| (0..3).map(|m| j[r][m] * c[m][k]).sum()));
    let full: [[f64; 3]; 3] =
        std::array::from_fn(|r| std::array::from_fn(|k| (0..3).map(|m| jc[r][m] * j[k][m]).sum()));
    let cov = std::array::from_fn(|r| std::array::from_fn(|k| 0.5 * (full[r][k] + full[k][r])));
    (psf, cov)
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
        let (rb, rp) = (affine_fit(&base).1, affine_fit(&photo).1);
        // On this render the ellipse centres scatter ~0.5 px and the
        // photometric ones ~0.015 px; the bound is loose on purpose.
        assert!(rp < 0.5 * rb, "photometric {rp} px vs default {rb} px");
    }

    /// The least-squares affine map `image = A · [X, Y, 1]` from the
    /// features' board positions (metres) to their image centres, and its RMS
    /// residual in pixels.
    fn affine_fit(fs: &[Feature]) -> ([[f64; 3]; 2], f64) {
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
        (c, (sum / fs.len() as f64).sqrt())
    }

    // ── The measured blur ──────────────────────────────────────────────────

    /// The inverse of [`psf_from_sigma_anisotropy`]'s value map: `σ² = tr Σ /
    /// 2`, `a = (xx − yy) / tr Σ`, `b = 2 xy / tr Σ`.
    fn sigma_anisotropy_from_psf([xx, xy, yy]: [f64; 3]) -> (f64, [f64; 2]) {
        let tr = xx + yy;
        ((0.5 * tr).sqrt(), [(xx - yy) / tr, 2.0 * xy / tr])
    }

    fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        std::array::from_fn(|r| std::array::from_fn(|k| (0..3).map(|m| a[r][m] * b[m][k]).sum()))
    }

    fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        std::array::from_fn(|r| std::array::from_fn(|k| a[k][r]))
    }

    /// The `(σ, a, b)` → `Σ` map is inverted exactly, its covariance is `J C
    /// Jᵀ` with `J` the Jacobian of the value map (checked against central
    /// differences), and the inverse map's Jacobian takes the covariance
    /// back to `C`.
    #[test]
    fn blur_map_round_trips_and_propagates_its_covariance() {
        let c = [[4e-4, 3e-5, -2e-5], [3e-5, 9e-4, 1e-5], [-2e-5, 1e-5, 6e-4]];
        for (sigma, ab) in [
            (0.5, [0.0, 0.0]),
            (1.53, [0.12, -0.07]),
            (3.2, [-0.6, 0.5]),
            (2.0, [0.0, 0.89]),
        ] {
            let (psf, cov) = psf_from_sigma_anisotropy(sigma, ab, c);
            let (s, [a, b]) = sigma_anisotropy_from_psf(psf);
            assert!((s - sigma).abs() < 1e-14 && (a - ab[0]).abs() < 1e-14);
            assert!((b - ab[1]).abs() < 1e-14, "{b} vs {}", ab[1]);
            assert!(((psf[0] + psf[2]) / 2.0 - sigma * sigma).abs() < 1e-12);

            // Numerical Jacobian of the value map, column by parameter.
            let p = [sigma, ab[0], ab[1]];
            let value = |p: [f64; 3]| psf_from_sigma_anisotropy(p[0], [p[1], p[2]], c).0;
            let mut j = [[0.0; 3]; 3];
            for col in 0..3 {
                let h = 1e-6;
                let (mut lo, mut hi) = (p, p);
                lo[col] -= h;
                hi[col] += h;
                let (vl, vh) = (value(lo), value(hi));
                for row in 0..3 {
                    j[row][col] = (vh[row] - vl[row]) / (2.0 * h);
                }
            }
            let want = mat_mul(mat_mul(j, c), transpose(j));
            for r in 0..3 {
                for k in 0..3 {
                    assert!(
                        (cov[r][k] - want[r][k]).abs() < 1e-8 * (1.0 + want[r][k].abs()),
                        "cov[{r}][{k}] = {} vs J C Jᵀ {}",
                        cov[r][k],
                        want[r][k]
                    );
                    assert_eq!(cov[r][k], cov[k][r], "symmetric");
                }
            }

            // The inverse map's Jacobian (analytic in Σ) brings it back to C.
            let [xx, xy, yy] = psf;
            let tr = xx + yy;
            let ji = [
                [0.25 / s, 0.0, 0.25 / s],
                [2.0 * yy / (tr * tr), 0.0, -2.0 * xx / (tr * tr)],
                [-2.0 * xy / (tr * tr), 2.0 / tr, -2.0 * xy / (tr * tr)],
            ];
            let back = mat_mul(mat_mul(ji, cov), transpose(ji));
            for r in 0..3 {
                for k in 0..3 {
                    assert!(
                        (back[r][k] - c[r][k]).abs() < 1e-12,
                        "C[{r}][{k}]: {} vs {}",
                        back[r][k],
                        c[r][k]
                    );
                }
            }
        }
    }

    /// The frontal render of `cfg` at `dpi` as a camera with a Gaussian PSF
    /// of `sigma_px` camera pixels and square pixels sees it: rendered at
    /// twice the density, blurred there by an exact (truncated at 5σ,
    /// normalized) separable Gaussian, averaged over 2 × 2 blocks (the
    /// pixel's area integration) and padded with paper.
    fn camera_render(cfg: &RinggridConfig, dpi: f32, sigma_px: f64, pad: u32) -> image::GrayImage {
        const N: u32 = 2;
        let options = TargetRenderOptions {
            png_dpi: dpi * N as f32,
            ..TargetRenderOptions::default()
        };
        let fine = cfg
            .board_layout()
            .unwrap()
            .render_target_png(&options)
            .unwrap();
        let (w, h) = (fine.width() as usize, fine.height() as usize);
        let s = sigma_px * f64::from(N);
        let r = (5.0 * s).ceil() as isize;
        let mut kernel: Vec<f64> = (-r..=r)
            .map(|i| (-0.5 * (i as f64 / s).powi(2)).exp())
            .collect();
        let total: f64 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= total);
        // Separable convolution, the board extended by paper (255).
        let src: Vec<f64> = fine.pixels().map(|p| f64::from(p.0[0])).collect();
        let at = |v: &[f64], x: isize, y: isize, w: usize, h: usize| {
            if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                255.0
            } else {
                v[y as usize * w + x as usize]
            }
        };
        let mut rows = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                rows[y * w + x] = kernel
                    .iter()
                    .enumerate()
                    .map(|(i, k)| k * at(&src, x as isize + i as isize - r, y as isize, w, h))
                    .sum();
            }
        }
        let mut both = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                both[y * w + x] = kernel
                    .iter()
                    .enumerate()
                    .map(|(i, k)| k * at(&rows, x as isize, y as isize + i as isize - r, w, h))
                    .sum();
            }
        }
        let (cw, ch) = (w as u32 / N, h as u32 / N);
        image::GrayImage::from_fn(cw + 2 * pad, ch + 2 * pad, |x, y| {
            if x < pad || y < pad || x >= cw + pad || y >= ch + pad {
                return image::Luma([255]);
            }
            let (x0, y0) = (((x - pad) * N) as usize, ((y - pad) * N) as usize);
            let sum: f64 = (0..N as usize)
                .flat_map(|dy| (0..N as usize).map(move |dx| (dx, dy)))
                .map(|(dx, dy)| both[(y0 + dy) * w + x0 + dx])
                .sum();
            image::Luma([(sum / f64::from(N * N)).round().clamp(0.0, 255.0) as u8])
        })
    }

    /// A camera with a Gaussian PSF of 1.5 px sees edges of width
    /// `√(1.5² + 1/12)` px (the pixel's area integration adds `1/12`): the
    /// photometric centres report that width as `√(tr Σ / 2)`, isotropic.
    ///
    /// `ringgrid` states its fitted width within ±0.5 % of `√(σ_psf² +
    /// 1/12)` for a Gaussian PSF of 0.7–3 px on markers of outer radius
    /// ≥ 17 px with the print radii right; these markers are 23.6 px.
    #[test]
    fn photometric_centres_report_the_rendered_blur() {
        let sigma_psf = 1.5;
        let truth = (sigma_psf * sigma_psf + 1.0 / 12.0_f64).sqrt();
        let mut cfg: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        let img = camera_render(&cfg, 100.0, sigma_psf, 40);

        // Edge centres carry no blur.
        assert!(detect(&cfg, &img).unwrap().iter().all(|f| f.blur.is_none()));

        cfg.center_method = RinggridCenterMethod::Photometric;
        let features = detect(&cfg, &img).unwrap();
        let blurs: Vec<FeatureBlur> = features.iter().filter_map(|f| f.blur).collect();
        // Of the board's 23 markers, the fit reports a blur for 20 on this
        // render.
        assert!(
            blurs.len() >= 18,
            "{} of {} markers",
            blurs.len(),
            features.len()
        );

        let mut rel: Vec<f64> = Vec::new();
        for b in &blurs {
            let [xx, _, yy] = b.psf_px;
            let sigma = (0.5 * (xx + yy)).sqrt();
            // Its standard deviation from the reported covariance:
            // ∂σ/∂(xx, xy, yy) = (1, 0, 1) / (4σ).
            let sd = (b.cov[0][0] + 2.0 * b.cov[0][2] + b.cov[2][2]).sqrt() / (4.0 * sigma);
            assert!(
                (sigma - truth).abs() < 0.01 * truth + 4.0 * sd,
                "σ {sigma} px (sd {sd}) vs {truth} px"
            );
            let (_, [a, bb]) = sigma_anisotropy_from_psf(b.psf_px);
            // Measured: |[a, b]| ≤ 0.01 on this render.
            assert!(a.hypot(bb) < 0.02, "an isotropic PSF fitted as [{a}, {bb}]");
            rel.push(sigma / truth - 1.0);
        }
        rel.sort_by(f64::total_cmp);
        // Measured: −0.36 % (markers −1.4 … +0.35 %, each within 1.8
        // reported standard deviations).
        let median = rel[rel.len() / 2];
        assert!(median.abs() < 0.005, "median relative error {median}");
    }

    // ── The other lattice × coding combinations ─────────────────────────────

    /// A frontal render of the board `cfg` describes at `dpi`, blurred by a
    /// Gaussian of `blur_px` (0 = sharp), with its scale in pixels per metre.
    fn render(cfg: &RinggridConfig, dpi: f32, blur_px: f32) -> (image::GrayImage, f64) {
        let options = TargetRenderOptions {
            png_dpi: dpi,
            ..TargetRenderOptions::default()
        };
        let gray = cfg
            .board_layout()
            .unwrap()
            .render_target_png(&options)
            .expect("render ring-grid PNG");
        let gray = if blur_px > 0.0 {
            image::imageops::blur(&gray, blur_px)
        } else {
            gray
        };
        (gray, f64::from(dpi) / 0.0254)
    }

    /// Where the render of `cfg` (`width_px` wide, `px_per_m`) puts the board
    /// point `xy_m`, in the pixels of a window cropped at `crop`. The render
    /// centres the board's marker extent on a square page and samples pixel
    /// `i` at `i` (not `i + 0.5`), so this is exact up to the page's rounding
    /// to whole pixels.
    fn rendered_px(
        cfg: &RinggridConfig,
        width_px: u32,
        px_per_m: f64,
        crop: [u32; 2],
        xy_m: [f64; 2],
    ) -> [f64; 2] {
        let (min, max) = cfg.board_layout().unwrap().marker_bounds_mm().unwrap();
        std::array::from_fn(|axis| {
            let centre_m = 0.5e-3 * f64::from(min[axis] + max[axis]);
            0.5 * f64::from(width_px) + (xy_m[axis] - centre_m) * px_per_m - f64::from(crop[axis])
        })
    }

    /// Check `features`, detected on the render of `cfg` (`width_px` wide,
    /// `px_per_m`, cropped at `crop`), against the board: there are at least
    /// `min_features`, every target point is a distinct cell of the layout,
    /// every centre is within a quarter pitch of where the render drew that
    /// cell (a view labelled up to a lattice symmetry, or shifted by a cell,
    /// is off by a pitch), and the centres scatter about the drawn ones by
    /// less than `rms_px`.
    fn assert_board_frame(
        cfg: &RinggridConfig,
        features: &[Feature],
        (width_px, px_per_m, crop): (u32, f64, [u32; 2]),
        min_features: usize,
        rms_px: f64,
    ) {
        assert!(
            features.len() >= min_features,
            "expected >= {min_features} markers, got {}",
            features.len()
        );
        let layout = cfg.board_layout().unwrap();
        let mut cells = std::collections::BTreeSet::new();
        let (mut worst, mut sum_sq): (f64, f64) = (0.0, 0.0);
        for f in features {
            assert_eq!(f.world_xyz[2], 0.0, "target is planar");
            let cell = layout
                .cells()
                .iter()
                .position(|c| {
                    (f64::from(c.xy_mm[0]) - f.world_xyz[0] * 1e3).abs() < 1e-3
                        && (f64::from(c.xy_mm[1]) - f.world_xyz[1] * 1e3).abs() < 1e-3
                })
                .unwrap_or_else(|| panic!("{:?} is not a cell of the board", f.world_xyz));
            assert!(cells.insert(cell), "cell {cell} reported twice");
            let want = rendered_px(
                cfg,
                width_px,
                px_per_m,
                crop,
                [f.world_xyz[0], f.world_xyz[1]],
            );
            let err = (f.image_xy[0] - want[0]).hypot(f.image_xy[1] - want[1]);
            worst = worst.max(err);
            sum_sq += err * err;
        }
        let quarter_pitch_px = 0.25 * cfg.pitch_m * px_per_m;
        assert!(
            worst < quarter_pitch_px,
            "a centre is {worst} px from its rendered position: mislabelled"
        );
        let rms = (sum_sq / features.len() as f64).sqrt();
        assert!(
            rms < rms_px,
            "centres scatter {rms} px about the drawn ones"
        );
    }

    fn config_with(
        lattice: RinggridLattice,
        coding: RinggridCoding,
        (pitch_m, rows, cols): (f64, u32, u32),
        (outer_m, inner_m): (f64, f64),
    ) -> RinggridConfig {
        let mut cfg = RinggridConfig::new(pitch_m, rows, cols, outer_m, inner_m, RING_WIDTH_M);
        cfg.lattice = lattice;
        cfg.coding = coding;
        cfg
    }

    const DOTS: RinggridCoding = RinggridCoding::Plain { origin_dots: true };
    const MARKER: (f64, f64) = (OUTER_M, INNER_M);
    /// The renders below are at 100 dpi with a 1 px blur: markers ~47 px
    /// across, the size range adaptive detection is tuned for, and edges as
    /// soft as a focused lens leaves them.
    const DPI: f32 = 100.0;
    /// Bound on the RMS distance of the centres from the drawn ones. Ellipse
    /// centres on these renders scatter ~0.5 px (see
    /// `photometric_centres_match_or_beat_the_default`), with the odd marker
    /// ~2 px off; the distance also includes the ~0.2 px every centre is
    /// shifted by the page's rounding to whole pixels.
    const RMS_PX: f64 = 0.75;

    /// Detect on the standard render of `cfg` and check it against the board.
    fn detect_and_check(cfg: &RinggridConfig, min_features: usize) {
        let (img, px_per_m) = render(cfg, DPI, 1.0);
        let features = detect(cfg, &img).unwrap();
        let frame = (img.width(), px_per_m, [0, 0]);
        assert_board_frame(cfg, &features, frame, min_features, RMS_PX);
    }

    #[test]
    fn coded_hex_centres_match_the_board() {
        let cfg: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        detect_and_check(&cfg, 21);
    }

    #[test]
    fn coded_rect_centres_match_the_board() {
        let geometry = (PITCH_M, 4, 5);
        let cfg = config_with(
            RinggridLattice::Rect,
            RinggridCoding::Coded,
            geometry,
            MARKER,
        );
        let layout = cfg.board_layout().unwrap();
        assert_eq!(layout.n_cells(), 20, "4 rows of 5 columns");
        assert!(layout.is_coded());
        detect_and_check(&cfg, 18);
    }

    #[test]
    fn plain_rect_with_origin_dots_centres_match_the_board() {
        let cfg = config_with(RinggridLattice::Rect, DOTS, (PITCH_M, 6, 7), MARKER);
        let layout = cfg.board_layout().unwrap();
        assert_eq!(layout.n_cells(), 42, "6 rows of 7 columns");
        assert!(!layout.is_coded());
        assert_eq!(layout.fiducial_dots_mm().len(), 3, "an origin-dot triad");
        detect_and_check(&cfg, 40);
    }

    #[test]
    fn plain_hex_with_origin_dots_centres_match_the_board() {
        let cfg = config_with(RinggridLattice::Hex, DOTS, (PITCH_M, 5, 5), MARKER);
        let layout = cfg.board_layout().unwrap();
        assert_eq!(layout.n_cells(), 23);
        assert!(!layout.is_coded());
        assert_eq!(layout.fiducial_dots_mm().len(), 3, "an origin-dot triad");
        detect_and_check(&cfg, 21);
    }

    /// A frontal 300 dpi render of a `rows × cols` plain rect board at 2 mm
    /// pitch with 0.8 / 0.4 mm rings (~19 px per marker), and the window of
    /// `w × h` pitches whose `(0.4, 0.6)` point is the origin dots' centroid:
    /// the dots in view, off its centre. Returns the config, the window, and
    /// its frame for [`assert_board_frame`].
    fn plain_rect_window(
        (rows, cols): (u32, u32),
        (w, h): (f64, f64),
    ) -> (RinggridConfig, image::GrayImage, (u32, f64, [u32; 2])) {
        let cfg = config_with(
            RinggridLattice::Rect,
            DOTS,
            (0.002, rows, cols),
            (0.0008, 0.0004),
        );
        let (full, px_per_m) = render(&cfg, 300.0, 0.0);
        let layout = cfg.board_layout().unwrap();
        let dots = layout.fiducial_dots_mm();
        let dots_m: [f64; 2] = std::array::from_fn(|axis| {
            dots.iter().map(|d| f64::from(d[axis])).sum::<f64>() / 3.0 * 1e-3
        });
        let dots_px = rendered_px(&cfg, full.width(), px_per_m, [0, 0], dots_m);
        let (w, h) = (w * cfg.pitch_m * px_per_m, h * cfg.pitch_m * px_per_m);
        let crop = [(dots_px[0] - 0.4 * w) as u32, (dots_px[1] - 0.6 * h) as u32];
        let window = image::imageops::crop_imm(&full, crop[0], crop[1], w as u32, h as u32);
        (cfg, window.to_image(), (full.width(), px_per_m, crop))
    }

    /// Only a window around the origin dots is in view: the view is still
    /// labelled in the board frame.
    #[test]
    fn plain_rect_partial_view_keeps_absolute_board_coordinates() {
        let (cfg, window, frame) = plain_rect_window((18, 16), (13.0, 11.0));
        let features = detect(&cfg, &window).unwrap();
        // 13 × 11 pitches hold 12 × 10 whole markers or more, of 288.
        assert_board_frame(&cfg, &features, frame, 100, RMS_PX);
        assert!(features.len() < 160, "a partial view: {}", features.len());
    }

    /// A view is anchored by the origin dots, not by how much of the board
    /// it shows: a 14 × 11 window around the dots of a 58 × 58 board (3364
    /// markers) is labelled in the board frame.
    #[test]
    fn plain_rect_small_window_of_a_large_board_gets_absolute_coordinates() {
        let (cfg, window, frame) = plain_rect_window((58, 58), (14.0, 11.0));
        let features = detect(&cfg, &window).unwrap();
        // 14 × 11 pitches hold 13 × 10 whole markers or more.
        assert_board_frame(&cfg, &features, frame, 100, RMS_PX);
        assert!(features.len() < 180, "a partial view: {}", features.len());
    }

    /// Without the origin dots in view a plain board has no board frame: the
    /// view yields no features rather than mislabelled ones.
    #[test]
    fn plain_rect_view_without_the_origin_dots_yields_nothing() {
        let cfg = config_with(
            RinggridLattice::Rect,
            DOTS,
            (0.002, 18, 16),
            (0.0008, 0.0004),
        );
        let (full, px_per_m) = render(&cfg, 300.0, 0.0);
        let side = (6.0 * cfg.pitch_m * px_per_m) as u32;
        let corner = image::imageops::crop_imm(&full, 0, 0, side, side);
        assert!(detect(&cfg, &corner.to_image()).unwrap().is_empty());
    }

    /// Photometric centres work on plain markers: every returned marker was
    /// refined, labelled in the board frame, and closer to its drawn centre
    /// than the ellipse centres are.
    #[test]
    fn plain_board_with_photometric_centres() {
        let mut cfg = config_with(RinggridLattice::Rect, DOTS, (PITCH_M, 6, 7), MARKER);
        // The photometric fit models a blurred marker; see
        // `photometric_centres_match_or_beat_the_default`.
        let (img, px_per_m) = render(&cfg, DPI, 2.0);
        let base = detect(&cfg, &img).unwrap();
        cfg.center_method = RinggridCenterMethod::Photometric;
        let photo = detect(&cfg, &img).unwrap();
        // Of the 42 markers, the render's left and right columns touch the
        // image border, where most fall below the detector's confidence
        // floor, and the fit declines a few more.
        let frame = (img.width(), px_per_m, [0, 0]);
        assert_board_frame(&cfg, &photo, frame, 27, RMS_PX);
        let (rb, rp) = (affine_fit(&base).1, affine_fit(&photo).1);
        assert!(rp < 0.5 * rb, "photometric {rp} px vs default {rb} px");
    }

    #[test]
    fn plain_board_without_origin_dots_is_rejected_before_detection() {
        let no_dots = RinggridCoding::Plain { origin_dots: false };
        let cfg = config_with(RinggridLattice::Rect, no_dots, (PITCH_M, 6, 7), MARKER);
        let err = detect(&cfg, &image::GrayImage::new(64, 64)).unwrap_err();
        assert!(matches!(err, DetectError::InvalidConfig(_)), "{err}");
        assert!(format!("{err}").contains("without origin dots"), "{err}");
    }

    /// Plain markers have no code band, so their ring width is neither
    /// needed nor checked.
    #[test]
    fn plain_board_ignores_the_ring_width() {
        let mut cfg = config_with(RinggridLattice::Hex, DOTS, (PITCH_M, 5, 5), MARKER);
        cfg.marker_ring_width_m = 0.0;
        assert!(cfg.board_layout().is_ok());
        cfg.coding = RinggridCoding::Coded;
        assert!(cfg.board_layout().is_err(), "a coded board needs it");
    }

    #[test]
    fn lattice_and_coding_json() {
        // A config without the keys is the coded hex board, and writes none.
        let old: RinggridConfig = serde_json::from_value(board_config()).unwrap();
        assert_eq!(old.lattice, RinggridLattice::Hex);
        assert_eq!(old.coding, RinggridCoding::Coded);
        assert_eq!(serde_json::to_value(old).unwrap(), board_config());

        let plain = json!({
            "pitch_m": 0.002,
            "rows": 58,
            "long_row_cols": 58,
            "marker_outer_radius_m": 0.0008,
            "marker_inner_radius_m": 0.0004,
            "lattice": "rect",
            "coding": {"plain": {"origin_dots": true}},
        });
        let cfg: RinggridConfig = serde_json::from_value(plain).unwrap();
        assert_eq!(cfg.lattice, RinggridLattice::Rect);
        assert_eq!(cfg.coding, DOTS);
        assert_eq!(cfg.marker_ring_width_m, 0.0, "plain boards may omit it");
        let back: RinggridConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(cfg, back);

        // The dots default to on; the defaults are spelled "coded" and "hex".
        let coding: RinggridCoding = serde_json::from_value(json!({"plain": {}})).unwrap();
        assert_eq!(coding, DOTS);
        let coding: RinggridCoding = serde_json::from_value(json!("coded")).unwrap();
        assert_eq!(coding, RinggridCoding::Coded);
        let lattice: RinggridLattice = serde_json::from_value(json!("hex")).unwrap();
        assert_eq!(lattice, RinggridLattice::Hex);
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
            lattice: RinggridLattice::Hex,
            coding: RinggridCoding::Coded,
        };
        let photometric = RinggridConfig {
            center_method: RinggridCenterMethod::Photometric,
            ..base
        };
        assert_ne!(key(base), key(photometric));
    }

    fn base() -> RinggridConfig {
        RinggridConfig::new(0.02, 5, 5, 0.006, 0.003, 0.001)
    }

    fn key(c: RinggridConfig) -> crate::CacheKey {
        crate::CacheKey::for_detection(b"img", &crate::DetectorConfig::Ringgrid(c), None)
    }

    /// The key hashes the config's JSON. A coded hex board writes exactly
    /// the keys it always had, so its cached detections stay valid.
    #[test]
    fn coded_hex_cache_key_is_unchanged() {
        assert_eq!(
            serde_json::to_string(&crate::DetectorConfig::Ringgrid(base())).unwrap(),
            r#"{"detector":"ringgrid","pitch_m":0.02,"rows":5,"long_row_cols":5,"marker_outer_radius_m":0.006,"marker_inner_radius_m":0.003,"marker_ring_width_m":0.001}"#
        );
    }

    #[test]
    fn lattice_and_coding_change_the_cache_key() {
        let rect = RinggridConfig {
            lattice: RinggridLattice::Rect,
            ..base()
        };
        let plain = RinggridConfig {
            coding: RinggridCoding::Plain { origin_dots: true },
            ..base()
        };
        let plain_rect = RinggridConfig {
            coding: plain.coding,
            ..rect
        };
        let keys = [key(base()), key(rect), key(plain), key(plain_rect)];
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
