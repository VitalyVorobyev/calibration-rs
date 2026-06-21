//! C5 dense-matching benchmark harness (ADR 0015, amended 2026-06-21).
//!
//! This module defines the **infrastructure** for benchmarking dense stereo
//! matchers: image buffers, a disparity map type, the [`DenseMatcher`] trait,
//! synthetic rectified ground-truth fixtures, and error metrics.  It does
//! **not** contain any real matching algorithm.
//!
//! # Architecture
//!
//! Two concrete implementations will eventually plug in here:
//!
//! * **OpenCV SGBM baseline** — lives outside this crate, in a separate binary
//!   or feature-gated crate, because it requires a local OpenCV install.  It
//!   implements [`DenseMatcher`] and is scored by [`evaluate`].
//! * **Pure-Rust matcher** — the ultimate goal of Track C5; will live inside
//!   this workspace and implement the same trait.
//!
//! Both are scored against [`synthetic_rectified_pair`] (analytic ground
//! truth) and, later, against real stereo pairs whose ground-truth depth comes
//! from the calibrated target plane produced by Track C4 Scheimpflug
//! rectification.
//!
//! # Reproducibility
//!
//! All synthetic fixtures are fully deterministic: no RNG, no wall-clock state.
//! The right image is constructed as a bilinear warp of the left image by the
//! GT disparity, so an oracle recovers the GT exactly and a real matcher can be
//! scored without ambiguity.
//!
//! # OpenCV note
//!
//! OpenCV lives **only** in an OpenCV-equipped environment.  It is never a
//! dependency of this crate or any published workspace crate.

// ---------------------------------------------------------------------------
// GrayBuffer
// ---------------------------------------------------------------------------

/// 8-bit grayscale image buffer, row-major (x is the fast axis).
#[derive(Debug, Clone)]
pub struct GrayBuffer {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Raw pixel data, `data[y * width + x]`.
    pub data: Vec<u8>,
}

impl GrayBuffer {
    /// Construct a new buffer.  Panics if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "GrayBuffer: data length mismatch (expected {}, got {})",
            width * height,
            data.len()
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Read one pixel.  Panics in debug if `(x, y)` is out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        debug_assert!(
            x < self.width && y < self.height,
            "GrayBuffer::get out of bounds"
        );
        self.data[y * self.width + x]
    }

    /// Bilinear interpolation at sub-pixel position `(xf, yf)`.
    ///
    /// Coordinates are clamped to the border (i.e. the last valid pixel is
    /// replicated rather than extrapolated).  This matches the behaviour
    /// expected when warping by a disparity that is close to the image edge.
    pub fn bilinear(&self, xf: f64, yf: f64) -> f64 {
        let w = self.width as f64;
        let h = self.height as f64;

        // Clamp to valid coordinate range.
        let xf = xf.clamp(0.0, w - 1.0);
        let yf = yf.clamp(0.0, h - 1.0);

        let x0 = xf.floor() as usize;
        let y0 = yf.floor() as usize;
        // x1/y1 are clamped so we never read past the last pixel.
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let dx = xf - x0 as f64;
        let dy = yf - y0 as f64;

        let p00 = self.get(x0, y0) as f64;
        let p10 = self.get(x1, y0) as f64;
        let p01 = self.get(x0, y1) as f64;
        let p11 = self.get(x1, y1) as f64;

        p00 * (1.0 - dx) * (1.0 - dy)
            + p10 * dx * (1.0 - dy)
            + p01 * (1.0 - dx) * dy
            + p11 * dx * dy
    }
}

// ---------------------------------------------------------------------------
// DisparityMap
// ---------------------------------------------------------------------------

/// Per-pixel horizontal disparity (left x − right x), in pixels.
///
/// `f32::NAN` marks an invalid/unknown pixel.  Use [`DisparityMap::is_valid`]
/// to test validity; never compare disparity values with `!=` or `<` directly
/// because those comparisons return `false` for `NaN`.
#[derive(Debug, Clone)]
pub struct DisparityMap {
    /// Map width in pixels.
    pub width: usize,
    /// Map height in pixels.
    pub height: usize,
    /// Raw data, `data[y * width + x]`, `NaN` = invalid.
    pub data: Vec<f32>,
}

impl DisparityMap {
    /// Construct a new map.  Panics if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "DisparityMap: data length mismatch"
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Construct a map filled with a single value (use `f32::NAN` for all-invalid).
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    /// Read one disparity value.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Write one disparity value.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data[y * self.width + x] = value;
    }

    /// Returns `true` if the pixel is valid (finite, not NaN).
    #[inline]
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        self.get(x, y).is_finite()
    }
}

// ---------------------------------------------------------------------------
// DenseMatcher trait + options
// ---------------------------------------------------------------------------

/// Options controlling dense stereo matching.
///
/// The field names and semantics mirror OpenCV's `StereoSGBM` for easy
/// interoperability with the future SGBM baseline.
#[derive(Debug, Clone)]
pub struct DenseMatchOptions {
    /// Minimum disparity, in pixels.
    pub min_disparity: i32,
    /// Number of disparity levels (must be a positive multiple of 16,
    /// OpenCV-style).
    pub num_disparities: i32,
    /// Block (window) size for matching, in pixels (should be odd).
    pub block_size: usize,
}

impl Default for DenseMatchOptions {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 5,
        }
    }
}

/// Interface for dense stereo matchers.
///
/// Both the OpenCV SGBM baseline (external) and a future pure-Rust matcher
/// implement this trait.  Callers score results with [`evaluate`].
///
/// # Contract
///
/// * Both images **must** be rectified: corresponding points share the same
///   row (`y` coordinate).
/// * The returned [`DisparityMap`] has the same dimensions as `left`.
/// * Invalid pixels (occluded, out-of-range, border effects) are marked with
///   `f32::NAN`.
pub trait DenseMatcher {
    /// Human-readable matcher name, used in benchmark logs and reports.
    fn name(&self) -> &str;

    /// Compute a dense disparity map from a rectified stereo pair.
    fn match_disparity(
        &self,
        left: &GrayBuffer,
        right: &GrayBuffer,
        opts: &DenseMatchOptions,
    ) -> DisparityMap;
}

// ---------------------------------------------------------------------------
// Synthetic fixture
// ---------------------------------------------------------------------------

/// A slanted fronto-parallel plane parameterised by its ground-truth
/// disparity field `d(x, y) = d0 + dx * x + dy * y`.
///
/// Reasonable defaults: `d0 = 8.0`, `dx = 0.02`, `dy = 0.01`.
pub struct SlantedPlane {
    /// Disparity at the origin.
    pub d0: f32,
    /// Horizontal disparity gradient (pixels / pixel).
    pub dx: f32,
    /// Vertical disparity gradient (pixels / pixel).
    pub dy: f32,
}

/// Compute the ground-truth disparity at pixel `(x, y)`.
fn gt_disparity(plane: &SlantedPlane, x: usize, y: usize) -> f32 {
    plane.d0 + plane.dx * x as f32 + plane.dy * y as f32
}

/// Deterministic high-entropy texture value at `(x, y)`, range `[0, 255]`.
///
/// Combines two sine gratings at irrational-ish frequencies (for broad
/// spectral content) with a hash term for high-frequency detail.  No RNG
/// is used — the same `(x, y)` always produces the same value.
fn texture_u8(x: usize, y: usize) -> u8 {
    let sine_part = (x as f64 * 0.21).sin() * 0.5 + 0.5 + (y as f64 * 0.13).sin() * 0.5 + 0.5;
    // Map sine_part [0, 2] → [0, 127].
    let base = (sine_part * 63.5) as u8;

    // High-frequency hash term — wrapping_mul keeps it deterministic.
    let hash = (x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x6c62_272e)) as u8;

    base.wrapping_add(hash >> 1) // blend 50 % hash, 50 % sine
}

/// Generate a synthetic rectified stereo pair with known ground-truth disparity.
///
/// Returns `(left, right, gt_disparity)` where:
///
/// * `left` has a deterministic high-entropy texture (two sine gratings plus a
///   hash term — no RNG).
/// * `gt_disparity` is the left-frame slanted-plane disparity
///   `d(x, y) = d0 + dx*x + dy*y`, following the standard convention
///   `d = x_left − x_right` (so a left pixel `(x, y)` matches right pixel
///   `(x − d, y)`).
/// * `right` is the left image resampled at `(x + d(x,y), y)` by bilinear
///   interpolation — the inverse warp that makes `right[x − d] == left[x]`,
///   i.e. the "ideal" right image for the given (positive) disparity field.
/// * A left pixel whose right correspondence `x − d` falls outside `[0, width)`
///   is occluded — it is marked invalid (`NaN`) in the ground-truth map so a
///   matcher is not penalised for it.
pub fn synthetic_rectified_pair(
    width: usize,
    height: usize,
    plane: &SlantedPlane,
) -> (GrayBuffer, GrayBuffer, DisparityMap) {
    // --- left image: pure texture ---
    let mut left_data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            left_data[y * width + x] = texture_u8(x, y);
        }
    }
    let left = GrayBuffer::new(width, height, left_data);

    // --- GT disparity map ---
    let mut gt_data = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let d = gt_disparity(plane, x, y);
            // The right correspondence of left pixel x is x - d (convention
            // d = x_left - x_right). If it leaves the image the pixel is
            // occluded -> no ground truth to score against.
            let match_x = x as f32 - d;
            gt_data[y * width + x] = if match_x >= 0.0 && match_x < width as f32 {
                d
            } else {
                f32::NAN
            };
        }
    }
    let gt = DisparityMap::new(width, height, gt_data);

    // --- right image: bilinear warp of left by the plane disparity ---
    // Synthesize EVERY right pixel from the (raw) plane disparity, not from
    // `gt` — gt's NaNs mark left-frame occlusion for scoring, not right-image
    // coverage, and a right pixel can be the match for a valid left pixel even
    // where its own gt is invalid. `bilinear` border-clamps an out-of-range
    // source.
    let mut right_data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            // Inverse warp: right[x] = left[x + d], so the matching pixel
            // right[x - d] == left[x] (convention d = x_left - x_right).
            let src_xf = x as f64 + gt_disparity(plane, x, y) as f64;
            right_data[y * width + x] = left.bilinear(src_xf, y as f64).round() as u8;
        }
    }
    let right = GrayBuffer::new(width, height, right_data);

    (left, right, gt)
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

/// Aggregate dense-disparity error metrics.
///
/// All rates / means are `0.0` (not `NaN`) when no pixels are evaluated.
#[derive(Debug, Clone)]
pub struct DenseMetrics {
    /// Pixels valid in BOTH `gt` and `estimate`.
    pub evaluated_px: usize,
    /// Pixels valid in `gt` (the denominator for density).
    pub gt_valid_px: usize,
    /// `evaluated_px / gt_valid_px`, or `0.0` when `gt_valid_px == 0`.
    pub density: f64,
    /// Root-mean-square disparity error over evaluated pixels.
    pub rms_px: f64,
    /// Mean absolute error over evaluated pixels.
    pub mae_px: f64,
    /// Fraction of evaluated pixels with `|error| > bad_threshold_px`.
    pub bad_pixel_rate: f64,
}

/// Evaluate an estimated disparity map against ground truth.
///
/// Only pixels that are finite in **both** `estimate` and `gt` contribute to
/// the error metrics.  If no such pixel exists, all metrics are zero and no
/// `NaN` or panic is produced.
///
/// # Parameters
///
/// * `estimate` — output of a [`DenseMatcher`].
/// * `gt` — ground-truth disparity, e.g. from [`synthetic_rectified_pair`].
/// * `bad_threshold_px` — threshold in pixels for the bad-pixel metric.
pub fn evaluate(estimate: &DisparityMap, gt: &DisparityMap, bad_threshold_px: f32) -> DenseMetrics {
    assert_eq!(
        (estimate.width, estimate.height),
        (gt.width, gt.height),
        "evaluate: estimate and gt must have the same dimensions"
    );

    let mut gt_valid_px: usize = 0;
    let mut evaluated_px: usize = 0;
    let mut sum_sq_err: f64 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut bad_px: usize = 0;

    for y in 0..gt.height {
        for x in 0..gt.width {
            let g = gt.get(x, y);
            if !g.is_finite() {
                continue;
            }
            gt_valid_px += 1;

            let e = estimate.get(x, y);
            if !e.is_finite() {
                continue;
            }
            evaluated_px += 1;

            let err = (e - g).abs();
            sum_sq_err += (err as f64) * (err as f64);
            sum_abs_err += err as f64;
            if err > bad_threshold_px {
                bad_px += 1;
            }
        }
    }

    if evaluated_px == 0 {
        return DenseMetrics {
            evaluated_px: 0,
            gt_valid_px,
            density: 0.0,
            rms_px: 0.0,
            mae_px: 0.0,
            bad_pixel_rate: 0.0,
        };
    }

    let n = evaluated_px as f64;
    let gt_valid = gt_valid_px as f64;

    DenseMetrics {
        evaluated_px,
        gt_valid_px,
        density: n / gt_valid,
        rms_px: (sum_sq_err / n).sqrt(),
        mae_px: sum_abs_err / n,
        bad_pixel_rate: bad_px as f64 / n,
    }
}

// ---------------------------------------------------------------------------
// Oracle matcher
// ---------------------------------------------------------------------------

/// A perfect matcher that returns the supplied ground-truth disparity verbatim.
///
/// This is used exclusively to validate harness plumbing (fixture construction,
/// `evaluate` logic).  It is **not** a real algorithm.
pub struct OracleMatcher {
    /// The ground-truth disparity that will be returned as-is.
    pub gt: DisparityMap,
}

impl DenseMatcher for OracleMatcher {
    fn name(&self) -> &str {
        "oracle"
    }

    fn match_disparity(
        &self,
        _left: &GrayBuffer,
        _right: &GrayBuffer,
        _opts: &DenseMatchOptions,
    ) -> DisparityMap {
        self.gt.clone()
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust block-matcher adapter (Track C5 deliverable)
// ---------------------------------------------------------------------------

use vision_calibration::mvg::dense as mvgd;

/// Adapter exposing the pure-Rust [`vision_calibration::mvg::dense::match_block`]
/// matcher through the harness [`DenseMatcher`] trait.
///
/// The harness-level [`DenseMatchOptions`] (`min_disparity` / `num_disparities`
/// / `block_size`) are forwarded verbatim; the matcher's additional controls
/// (ZNCC min-correlation, uniqueness, left-right consistency, sub-pixel) keep
/// their library defaults. This is the in-workspace counterpart to the external
/// OpenCV SGBM baseline.
#[derive(Debug, Clone, Default)]
pub struct BlockMatcher;

impl DenseMatcher for BlockMatcher {
    fn name(&self) -> &str {
        "block-zncc"
    }

    fn match_disparity(
        &self,
        left: &GrayBuffer,
        right: &GrayBuffer,
        opts: &DenseMatchOptions,
    ) -> DisparityMap {
        let l = mvgd::GrayImage::new(left.width, left.height, left.data.clone());
        let r = mvgd::GrayImage::new(right.width, right.height, right.data.clone());
        let mopts = mvgd::BlockMatchOptions {
            min_disparity: opts.min_disparity,
            num_disparities: opts.num_disparities,
            block_size: opts.block_size,
            ..Default::default()
        };
        match mvgd::match_block(&l, &r, &mopts) {
            Ok(d) => DisparityMap::new(d.width, d.height, d.data),
            // The matcher only errors on invalid configuration; surface that as
            // an all-invalid map rather than panicking inside the harness.
            Err(_) => DisparityMap::filled(left.width, left.height, f32::NAN),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample plane used across tests: modest disparity gradient so interior
    /// pixels are always valid.
    fn sample_plane() -> SlantedPlane {
        SlantedPlane {
            d0: 8.0,
            dx: 0.02,
            dy: 0.01,
        }
    }

    const W: usize = 80;
    const H: usize = 60;

    // -----------------------------------------------------------------------
    // Fixture consistency
    // -----------------------------------------------------------------------

    /// Verify that `right.get(x, y) ≈ left.bilinear(x − gt(x,y), y)` for
    /// several interior pixels.
    ///
    /// The fixture builds the right image by sampling the left at
    /// `(x − d(x,y), y)` and rounding to u8.  So the correct assertion is
    /// that `right.get(x, y)` equals `round(left.bilinear(x−d, y))`, which we
    /// check with a 1 grey-level tolerance for the rounding step.
    #[test]
    fn synthetic_pair_follows_disparity_sign_convention() {
        // Constant (fronto-parallel) disparity makes the inverse warp exact, so
        // we can assert the standard stereo convention `d = x_left - x_right`,
        // i.e. left pixel x matches right pixel (x - d), to rounding precision.
        // This is the test that catches a flipped warp sign: with the wrong
        // `right[x] = left[x - d]`, `right[x - d]` would equal `left[x - 2d]`,
        // not `left[x]`.
        let plane = SlantedPlane {
            d0: 8.0,
            dx: 0.0,
            dy: 0.0,
        };
        let (left, right, gt) = synthetic_rectified_pair(W, H, &plane);

        for y in [10usize, 20, 30, 40] {
            for x in [20usize, 30, 40, 50, 60] {
                assert!(gt.is_valid(x, y), "GT should be valid at ({x},{y})");
                let d = gt.get(x, y) as f64;
                assert!((d - 8.0).abs() < 1e-6, "constant disparity expected");
                // Convention: left[x] corresponds to right[x - d].
                let left_val = left.get(x, y) as f64;
                let right_match = right.bilinear(x as f64 - d, y as f64);
                assert!(
                    (left_val - right_match).abs() < 1.0,
                    "sign convention violated at ({x},{y}): left = {left_val} vs \
                     right[x - d] = {right_match:.3}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Oracle matcher end-to-end
    // -----------------------------------------------------------------------

    /// Running the oracle through the harness and scoring it against GT must
    /// yield perfect metrics.
    #[test]
    fn oracle_recovers_ground_truth() {
        let plane = sample_plane();
        let (_left, _right, gt) = synthetic_rectified_pair(W, H, &plane);

        let matcher = OracleMatcher { gt: gt.clone() };
        let opts = DenseMatchOptions::default();
        // We pass dummy images — the oracle ignores them.
        let dummy = GrayBuffer::new(W, H, vec![0u8; W * H]);
        let estimate = matcher.match_disparity(&dummy, &dummy, &opts);

        let metrics = evaluate(&estimate, &gt, 1.0);

        assert_eq!(
            metrics.gt_valid_px, metrics.evaluated_px,
            "oracle covers all valid GT pixels"
        );
        assert!(
            (metrics.density - 1.0).abs() < 1e-9,
            "density should be 1.0, got {}",
            metrics.density
        );
        assert!(
            metrics.rms_px < 1e-6,
            "oracle RMS should be ~0, got {}",
            metrics.rms_px
        );
        assert_eq!(
            metrics.bad_pixel_rate, 0.0,
            "oracle bad-pixel rate must be zero"
        );
    }

    // -----------------------------------------------------------------------
    // evaluate — constant offset
    // -----------------------------------------------------------------------

    /// Feeding `estimate = gt + 0.5` must produce RMS ≈ 0.5 and, with a 0.25 px
    /// threshold, bad_pixel_rate == 1.0.
    #[test]
    fn evaluate_constant_offset() {
        let plane = sample_plane();
        let (_left, _right, gt) = synthetic_rectified_pair(W, H, &plane);

        // Build an estimate that is GT + 0.5 wherever GT is valid.
        let mut est_data = vec![f32::NAN; W * H];
        for y in 0..H {
            for x in 0..W {
                let g = gt.get(x, y);
                if g.is_finite() {
                    est_data[y * W + x] = g + 0.5;
                }
            }
        }
        let estimate = DisparityMap::new(W, H, est_data);

        let metrics_strict = evaluate(&estimate, &gt, 0.25);
        assert!(
            (metrics_strict.rms_px - 0.5).abs() < 1e-5,
            "RMS should be 0.5, got {}",
            metrics_strict.rms_px
        );
        assert!(
            (metrics_strict.bad_pixel_rate - 1.0).abs() < 1e-9,
            "bad_pixel_rate with threshold 0.25 px should be 1.0, got {}",
            metrics_strict.bad_pixel_rate
        );

        // With a generous threshold, bad_pixel_rate should be 0.
        let metrics_loose = evaluate(&estimate, &gt, 1.0);
        assert_eq!(
            metrics_loose.bad_pixel_rate, 0.0,
            "bad_pixel_rate with threshold 1.0 px should be 0.0"
        );
    }

    // -----------------------------------------------------------------------
    // evaluate — no overlap (all-NaN estimate)
    // -----------------------------------------------------------------------

    /// An all-NaN estimate must return `evaluated_px == 0` without panicking
    /// or producing NaN in any metric field.
    #[test]
    fn evaluate_handles_no_overlap() {
        let plane = sample_plane();
        let (_left, _right, gt) = synthetic_rectified_pair(W, H, &plane);

        let all_nan = DisparityMap::filled(W, H, f32::NAN);
        let metrics = evaluate(&all_nan, &gt, 1.0);

        assert_eq!(metrics.evaluated_px, 0);
        assert!(metrics.gt_valid_px > 0, "GT should have valid pixels");
        assert_eq!(metrics.density, 0.0);
        assert_eq!(metrics.rms_px, 0.0);
        assert_eq!(metrics.mae_px, 0.0);
        assert_eq!(metrics.bad_pixel_rate, 0.0);

        // Paranoia: ensure no NaN leaked into the struct.
        assert!(!metrics.density.is_nan());
        assert!(!metrics.rms_px.is_nan());
        assert!(!metrics.mae_px.is_nan());
        assert!(!metrics.bad_pixel_rate.is_nan());
    }

    // -----------------------------------------------------------------------
    // Additional unit-level checks
    // -----------------------------------------------------------------------

    #[test]
    fn gray_buffer_bilinear_at_integer_coordinate_equals_get() {
        let buf = GrayBuffer::new(4, 4, (0u8..16).collect());
        for y in 0..4usize {
            for x in 0..4usize {
                let expected = buf.get(x, y) as f64;
                let actual = buf.bilinear(x as f64, y as f64);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "bilinear at integer ({x},{y}) should equal get: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn disparity_map_filled_and_is_valid() {
        let map = DisparityMap::filled(3, 3, f32::NAN);
        for y in 0..3 {
            for x in 0..3 {
                assert!(!map.is_valid(x, y), "all-NaN map must be all-invalid");
            }
        }
        let mut map2 = DisparityMap::filled(3, 3, 5.0);
        assert!(map2.is_valid(1, 1));
        map2.set(1, 1, f32::NAN);
        assert!(!map2.is_valid(1, 1));
    }

    #[test]
    fn oracle_matcher_name() {
        let gt = DisparityMap::filled(2, 2, 0.0);
        let m = OracleMatcher { gt };
        assert_eq!(m.name(), "oracle");
    }

    #[test]
    fn dense_match_options_default() {
        let opts = DenseMatchOptions::default();
        assert_eq!(opts.min_disparity, 0);
        assert_eq!(opts.num_disparities, 64);
        assert_eq!(opts.block_size, 5);
    }

    // -----------------------------------------------------------------------
    // Pure-Rust block matcher, scored through the harness
    // -----------------------------------------------------------------------

    /// The C5 matcher run through the full harness (synthetic GT + `evaluate`)
    /// must recover the slanted plane densely and accurately. This is the
    /// quantitative gate that mirrors what the visual `dense_synth` example
    /// renders.
    #[test]
    fn block_matcher_recovers_synthetic_plane() {
        let plane = sample_plane();
        let (left, right, gt) = synthetic_rectified_pair(W, H, &plane);

        let matcher = BlockMatcher;
        assert_eq!(matcher.name(), "block-zncc");
        let opts = DenseMatchOptions {
            min_disparity: 0,
            num_disparities: 16,
            block_size: 7,
        };
        let est = matcher.match_disparity(&left, &right, &opts);
        let metrics = evaluate(&est, &gt, 1.0);

        assert!(
            metrics.density > 0.7,
            "block matcher density too low: {} ({} / {})",
            metrics.density,
            metrics.evaluated_px,
            metrics.gt_valid_px
        );
        assert!(
            metrics.rms_px < 0.5,
            "block matcher RMS too high: {} px",
            metrics.rms_px
        );
    }
}
