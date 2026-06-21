//! Pure-Rust dense stereo matching (Track C5, ADR 0015 amended 2026-06-21).
//!
//! [`match_block`] is a block-matching stereo correspondence search: for every
//! pixel of the (rectified) left image it finds the horizontal shift that best
//! aligns a local window with the right image, producing a per-pixel
//! [`DisparityMap`].
//!
//! # Algorithm (MVP)
//!
//! * **Cost** — zero-mean normalized cross-correlation (ZNCC) over a square
//!   window, so the matcher is invariant to per-camera gain/bias (important on
//!   real two-camera pairs). Window statistics are computed in `O(1)` per pixel
//!   from summed-area tables, so the whole search is `O(W·H·D)` regardless of
//!   the block size.
//! * **Selection** — winner-take-all over the disparity range, with optional
//!   parabolic **sub-pixel** refinement of the correlation peak.
//! * **Invalidation** — three independent filters mark a pixel `NaN`:
//!   a minimum correlation gate (rejects textureless regions), a uniqueness
//!   margin (rejects ambiguous matches), and a left-right consistency check
//!   (rejects occlusions / mismatches).
//!
//! Semi-global aggregation (SGM) on top of this cost is the documented
//! follow-up; this module is the block-matching MVP.
//!
//! # Conventions
//!
//! Both images must be **rectified** — corresponding points share the same row.
//! Disparity follows the standard sign `d = x_left − x_right ≥ 0`, so a left
//! pixel `(x, y)` matches right pixel `(x − d, y)`. The returned map has the
//! same dimensions as the left image; invalid pixels are `f32::NAN`.
//!
//! # Memory
//!
//! The MVP materializes a full `W·H·num_disparities` cost volume (`f32`). This
//! keeps the selection/sub-pixel/uniqueness logic simple and is fine for the
//! demo and modest images; SGM (the follow-up) replaces it with bounded-memory
//! path aggregation.

use crate::{MvgError, Result};

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// 8-bit grayscale image buffer, row-major (`x` is the fast axis).
#[derive(Debug, Clone)]
pub struct GrayImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Raw pixel data, `data[y * width + x]`.
    pub data: Vec<u8>,
}

impl GrayImage {
    /// Construct a new buffer.
    ///
    /// # Panics
    /// Panics if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "GrayImage: data length mismatch (expected {}, got {})",
            width * height,
            data.len()
        );
        Self {
            width,
            height,
            data,
        }
    }

    /// Read one pixel.
    ///
    /// # Panics
    /// Panics in debug builds if `(x, y)` is out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        debug_assert!(
            x < self.width && y < self.height,
            "GrayImage::get out of bounds"
        );
        self.data[y * self.width + x]
    }

    /// Bilinear interpolation at sub-pixel position `(xf, yf)`.
    ///
    /// Coordinates are clamped to the border (the last valid pixel is
    /// replicated rather than extrapolated).
    pub fn bilinear(&self, xf: f64, yf: f64) -> f64 {
        let w = self.width as f64;
        let h = self.height as f64;

        let xf = xf.clamp(0.0, w - 1.0);
        let yf = yf.clamp(0.0, h - 1.0);

        let x0 = xf.floor() as usize;
        let y0 = yf.floor() as usize;
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

/// Per-pixel horizontal disparity (`x_left − x_right`), in pixels.
///
/// `f32::NAN` marks an invalid/unknown pixel. Use [`DisparityMap::is_valid`] to
/// test validity; never compare disparity values with `!=` or `<` directly
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
    /// Construct a new map.
    ///
    /// # Panics
    /// Panics if `data.len() != width * height`.
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

    /// Returns `true` if the pixel is valid (finite, not `NaN`).
    #[inline]
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        self.get(x, y).is_finite()
    }

    /// Number of valid (finite) pixels.
    pub fn valid_count(&self) -> usize {
        self.data.iter().filter(|v| v.is_finite()).count()
    }

    /// Min and max over the valid pixels, or `None` if there are none.
    pub fn valid_range(&self) -> Option<(f32, f32)> {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        let mut any = false;
        for &v in &self.data {
            if v.is_finite() {
                any = true;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        any.then_some((lo, hi))
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Options controlling [`match_block`].
#[derive(Debug, Clone)]
pub struct BlockMatchOptions {
    /// Minimum disparity searched, in pixels (usually `0`).
    pub min_disparity: i32,
    /// Number of disparity levels searched, `[min_disparity, min_disparity + num_disparities)`.
    pub num_disparities: i32,
    /// Square matching-window side length, in pixels (must be odd and ≥ 1).
    pub block_size: usize,
    /// Reject a pixel whose best ZNCC is below this (textureless / poor match), in `[-1, 1]`.
    pub min_correlation: f32,
    /// Reject a pixel unless its best ZNCC exceeds the best non-adjacent
    /// alternative by at least this margin (ambiguity rejection).
    pub uniqueness_ratio: f32,
    /// Enable the left-right consistency cross-check (occlusion / mismatch rejection).
    pub lr_consistency: bool,
    /// Maximum `|d_left − d_right|`, in pixels, tolerated by the LR check.
    pub lr_max_diff: f32,
    /// Enable parabolic sub-pixel refinement of the correlation peak.
    pub subpixel: bool,
}

impl Default for BlockMatchOptions {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 7,
            min_correlation: 0.5,
            uniqueness_ratio: 0.03,
            lr_consistency: true,
            lr_max_diff: 1.0,
            subpixel: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Summed-area table
// ---------------------------------------------------------------------------

/// A summed-area table (integral image) for `O(1)` windowed sums.
struct SummedArea {
    /// Image width (the table has `width + 1` columns).
    width: usize,
    /// `(width + 1) * (height + 1)` prefix sums; `sat[(y+1)*(w+1) + (x+1)]`
    /// holds the sum of the source over `[0..=x] × [0..=y]`.
    sat: Vec<f64>,
}

impl SummedArea {
    /// Build the table from a per-pixel value function.
    fn build(width: usize, height: usize, value: impl Fn(usize, usize) -> f64) -> Self {
        let stride = width + 1;
        let mut sat = vec![0.0f64; stride * (height + 1)];
        for y in 0..height {
            for x in 0..width {
                sat[(y + 1) * stride + (x + 1)] =
                    value(x, y) + sat[y * stride + (x + 1)] + sat[(y + 1) * stride + x]
                        - sat[y * stride + x];
            }
        }
        Self { width, sat }
    }

    /// Sum over the `(2r+1)×(2r+1)` window centred at `(cx, cy)`.
    ///
    /// The caller must guarantee the window lies fully inside the image
    /// (`cx >= r`, `cx + r < width`, `cy >= r`, `cy + r < height`).
    #[inline]
    fn window_sum(&self, cx: usize, cy: usize, r: usize) -> f64 {
        let stride = self.width + 1;
        let x0 = cx - r;
        let x1 = cx + r + 1;
        let y0 = cy - r;
        let y1 = cy + r + 1;
        self.sat[y1 * stride + x1] - self.sat[y0 * stride + x1] - self.sat[y1 * stride + x0]
            + self.sat[y0 * stride + x0]
    }
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// Sentinel correlation for a window that cannot be evaluated (out of range or
/// zero-variance). Stored in the cost volume; never selected as a peak.
const INVALID_CORR: f32 = f32::NEG_INFINITY;

/// Compute a dense disparity map for a rectified stereo pair by block matching.
///
/// See the [module documentation](self) for the algorithm and conventions.
///
/// # Errors
///
/// Returns [`MvgError::InvalidInput`] if the two images differ in size, if
/// `block_size` is even or zero, if `num_disparities <= 0`, or if either image
/// is smaller than the matching window.
pub fn match_block(
    left: &GrayImage,
    right: &GrayImage,
    opts: &BlockMatchOptions,
) -> Result<DisparityMap> {
    let (w, h) = (left.width, left.height);
    if right.width != w || right.height != h {
        return Err(MvgError::invalid_input(format!(
            "left ({w}x{h}) and right ({}x{}) images must have equal dimensions",
            right.width, right.height
        )));
    }
    if opts.block_size == 0 || opts.block_size.is_multiple_of(2) {
        return Err(MvgError::invalid_input(format!(
            "block_size must be odd and >= 1, got {}",
            opts.block_size
        )));
    }
    if opts.num_disparities <= 0 {
        return Err(MvgError::invalid_input(format!(
            "num_disparities must be positive, got {}",
            opts.num_disparities
        )));
    }
    let r = opts.block_size / 2;
    if w < opts.block_size || h < opts.block_size {
        return Err(MvgError::invalid_input(format!(
            "image {w}x{h} is smaller than the {0}x{0} matching window",
            opts.block_size
        )));
    }

    // Summed-area tables of the intensities and their squares, reused across
    // disparities and across both matching directions.
    let il = SummedArea::build(w, h, |x, y| left.get(x, y) as f64);
    let ill = SummedArea::build(w, h, |x, y| {
        let v = left.get(x, y) as f64;
        v * v
    });
    let ir = SummedArea::build(w, h, |x, y| right.get(x, y) as f64);
    let irr = SummedArea::build(w, h, |x, y| {
        let v = right.get(x, y) as f64;
        v * v
    });

    // Left-referenced disparity: a left pixel matches the right image at x - d.
    let left_disp = solve_direction(left, right, &il, &ill, &ir, &irr, -1, r, opts);

    if !opts.lr_consistency {
        return Ok(DisparityMap::new(w, h, left_disp));
    }

    // Right-referenced disparity: a right pixel matches the left image at xr + d.
    let right_disp = solve_direction(right, left, &ir, &irr, &il, &ill, 1, r, opts);

    // Cross-check: keep a left disparity only if the right map agrees.
    let mut out = left_disp;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let d = out[idx];
            if !d.is_finite() {
                continue;
            }
            let xr = (x as f32 - d).round();
            if xr < 0.0 || xr >= w as f32 {
                out[idx] = f32::NAN;
                continue;
            }
            let dr = right_disp[(y * w) + xr as usize];
            if !dr.is_finite() || (d - dr).abs() > opts.lr_max_diff {
                out[idx] = f32::NAN;
            }
        }
    }
    Ok(DisparityMap::new(w, h, out))
}

/// Compute a one-directional disparity map.
///
/// `a` is the reference image, `b` the target; for each `a` pixel the matcher
/// searches `b` at `x + sign * d` for `d` in the configured range. `sign` is
/// `-1` when `a` is the left image (match leftward) and `+1` when `a` is the
/// right image (match rightward). Returns disparity magnitudes (`>= 0`) with
/// `NaN` for rejected pixels.
#[allow(clippy::too_many_arguments)]
fn solve_direction(
    a: &GrayImage,
    b: &GrayImage,
    ia: &SummedArea,
    iaa: &SummedArea,
    ib: &SummedArea,
    ibb: &SummedArea,
    sign: i32,
    r: usize,
    opts: &BlockMatchOptions,
) -> Vec<f32> {
    let (w, h) = (a.width, a.height);
    let num_d = opts.num_disparities as usize;
    let n = ((2 * r + 1) * (2 * r + 1)) as f64;
    // Cost volume: cost[k * (w*h) + y*w + x] = ZNCC at disparity min+k.
    let mut cost = vec![INVALID_CORR; num_d * w * h];

    for k in 0..num_d {
        let disp = opts.min_disparity + k as i32;
        // Cross-correlation term a(x,y) * b(x + sign*disp, y), summed-area table
        // rebuilt per disparity (it is the only disparity-dependent term).
        let iab = SummedArea::build(w, h, |x, y| {
            let tx = x as i32 + sign * disp;
            if tx >= 0 && (tx as usize) < w {
                a.get(x, y) as f64 * b.get(tx as usize, y) as f64
            } else {
                0.0
            }
        });

        let plane = k * w * h;
        for y in r..h - r {
            for x in r..w - r {
                let tx = x as i32 + sign * disp;
                // The b-window must lie fully inside the image.
                if tx - (r as i32) < 0 || tx + (r as i32) >= w as i32 {
                    continue;
                }
                let cxb = tx as usize;

                let s_a = ia.window_sum(x, y, r);
                let s_aa = iaa.window_sum(x, y, r);
                let s_b = ib.window_sum(cxb, y, r);
                let s_bb = ibb.window_sum(cxb, y, r);
                let s_ab = iab.window_sum(x, y, r);

                let var_a = n * s_aa - s_a * s_a;
                let var_b = n * s_bb - s_b * s_b;
                if var_a <= 1e-6 || var_b <= 1e-6 {
                    continue; // flat window: correlation undefined
                }
                let cov = n * s_ab - s_a * s_b;
                let zncc = cov / (var_a.sqrt() * var_b.sqrt());
                cost[plane + y * w + x] = zncc as f32;
            }
        }
    }

    // Per-pixel selection: winner-take-all + uniqueness + min-correlation +
    // optional parabolic sub-pixel refinement.
    let mut disp_map = vec![f32::NAN; w * h];
    let area = w * h;
    for y in r..h - r {
        for x in r..w - r {
            let base = y * w + x;
            let mut best_k = usize::MAX;
            let mut best = INVALID_CORR;
            for k in 0..num_d {
                let c = cost[k * area + base];
                if c > best {
                    best = c;
                    best_k = k;
                }
            }
            if best_k == usize::MAX || best < opts.min_correlation {
                continue;
            }
            // Best non-adjacent competitor for the uniqueness test.
            let mut second = INVALID_CORR;
            for k in 0..num_d {
                if k.abs_diff(best_k) <= 1 {
                    continue;
                }
                let c = cost[k * area + base];
                if c > second {
                    second = c;
                }
            }
            if second.is_finite() && best - second < opts.uniqueness_ratio {
                continue; // ambiguous
            }

            let mut d = (opts.min_disparity + best_k as i32) as f32;
            if opts.subpixel && best_k > 0 && best_k + 1 < num_d {
                let cm = cost[(best_k - 1) * area + base];
                let cp = cost[(best_k + 1) * area + base];
                if cm.is_finite() && cp.is_finite() {
                    let denom = cm + cp - 2.0 * best;
                    if denom.abs() > 1e-6 {
                        let delta = 0.5 * (cm - cp) / denom;
                        if delta.abs() < 1.0 {
                            d += delta;
                        }
                    }
                }
            }
            disp_map[base] = d;
        }
    }
    disp_map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic high-entropy texture, range `[0, 255]` (mirrors the bench
    /// fixture so the matcher is exercised on the same kind of signal).
    fn texture_u8(x: usize, y: usize) -> u8 {
        let sine = (x as f64 * 0.21).sin() * 0.5 + 0.5 + (y as f64 * 0.13).sin() * 0.5 + 0.5;
        let base = (sine * 63.5) as u8;
        let hash = (x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x6c62_272e)) as u8;
        base.wrapping_add(hash >> 1)
    }

    /// Build a synthetic rectified pair for disparity field `d(x,y)`.
    /// `right[x] = left[x + d]` so that `right[x - d] == left[x]`.
    fn synth(
        w: usize,
        h: usize,
        disp: impl Fn(usize, usize) -> f32,
    ) -> (GrayImage, GrayImage, Vec<f32>) {
        let mut ld = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                ld[y * w + x] = texture_u8(x, y);
            }
        }
        let left = GrayImage::new(w, h, ld);
        let mut rd = vec![0u8; w * h];
        let mut gt = vec![f32::NAN; w * h];
        for y in 0..h {
            for x in 0..w {
                let d = disp(x, y);
                rd[y * w + x] = left.bilinear(x as f64 + d as f64, y as f64).round() as u8;
                let mx = x as f32 - d;
                if mx >= 0.0 && mx < w as f32 {
                    gt[y * w + x] = d;
                }
            }
        }
        (left, GrayImage::new(w, h, rd), gt)
    }

    fn rms_over_valid(est: &DisparityMap, gt: &[f32], r: usize) -> (f64, usize) {
        let (w, h) = (est.width, est.height);
        let (mut sum_sq, mut n) = (0.0f64, 0usize);
        for y in r..h - r {
            for x in r..w - r {
                let g = gt[y * w + x];
                let e = est.get(x, y);
                if g.is_finite() && e.is_finite() {
                    sum_sq += ((e - g) as f64).powi(2);
                    n += 1;
                }
            }
        }
        if n == 0 {
            (f64::INFINITY, 0)
        } else {
            ((sum_sq / n as f64).sqrt(), n)
        }
    }

    #[test]
    fn recovers_constant_disparity() {
        let (left, right, gt) = synth(80, 60, |_, _| 8.0);
        let opts = BlockMatchOptions {
            min_disparity: 0,
            num_disparities: 16,
            block_size: 7,
            ..Default::default()
        };
        let est = match_block(&left, &right, &opts).unwrap();
        let (rms, n) = rms_over_valid(&est, &gt, 4);
        assert!(n > 1500, "should recover most interior pixels, got {n}");
        assert!(rms < 0.3, "constant-disparity RMS too high: {rms}");
    }

    #[test]
    fn recovers_fractional_disparity_subpixel() {
        // A constant fractional disparity: integer WTA alone cannot reach it,
        // so this exercises the parabolic sub-pixel refinement.
        let (left, right, gt) = synth(80, 60, |_, _| 6.4);
        let opts = BlockMatchOptions {
            min_disparity: 0,
            num_disparities: 16,
            block_size: 9,
            ..Default::default()
        };
        let est = match_block(&left, &right, &opts).unwrap();
        let (rms, n) = rms_over_valid(&est, &gt, 5);
        assert!(n > 1000, "expected dense recovery, got {n}");
        assert!(rms < 0.25, "sub-pixel RMS too high: {rms}");
    }

    #[test]
    fn recovers_slanted_plane() {
        let (left, right, gt) = synth(100, 70, |x, y| 8.0 + 0.02 * x as f32 + 0.01 * y as f32);
        let opts = BlockMatchOptions {
            min_disparity: 0,
            num_disparities: 16,
            block_size: 7,
            ..Default::default()
        };
        let est = match_block(&left, &right, &opts).unwrap();
        let (rms, _) = rms_over_valid(&est, &gt, 4);
        assert!(rms < 0.4, "slanted-plane RMS too high: {rms}");
    }

    #[test]
    fn lr_consistency_invalidates_some_border_pixels() {
        // Without LR consistency the matcher fills more pixels; enabling it must
        // not increase the valid count and should drop ambiguous border matches.
        let (left, right, _) = synth(80, 60, |_, _| 8.0);
        let base = BlockMatchOptions {
            num_disparities: 16,
            block_size: 7,
            ..Default::default()
        };
        let with_lr = match_block(&left, &right, &base).unwrap();
        let without_lr = match_block(
            &left,
            &right,
            &BlockMatchOptions {
                lr_consistency: false,
                ..base
            },
        )
        .unwrap();
        assert!(
            with_lr.valid_count() <= without_lr.valid_count(),
            "LR consistency should only ever remove matches"
        );
    }

    #[test]
    fn rejects_mismatched_dimensions() {
        let a = GrayImage::new(10, 10, vec![0; 100]);
        let b = GrayImage::new(12, 10, vec![0; 120]);
        let err = match_block(&a, &b, &BlockMatchOptions::default()).unwrap_err();
        assert!(matches!(err, MvgError::InvalidInput { .. }));
    }

    #[test]
    fn rejects_even_block_size() {
        let a = GrayImage::new(20, 20, vec![0; 400]);
        let b = GrayImage::new(20, 20, vec![0; 400]);
        let err = match_block(
            &a,
            &b,
            &BlockMatchOptions {
                block_size: 8,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, MvgError::InvalidInput { .. }));
    }

    #[test]
    fn rejects_window_larger_than_image() {
        let a = GrayImage::new(5, 5, vec![0; 25]);
        let b = GrayImage::new(5, 5, vec![0; 25]);
        let err = match_block(
            &a,
            &b,
            &BlockMatchOptions {
                block_size: 7,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, MvgError::InvalidInput { .. }));
    }

    #[test]
    fn textureless_region_is_rejected() {
        // A flat (zero-variance) image yields undefined correlation everywhere,
        // so every pixel must be invalid rather than spuriously matched.
        let flat = GrayImage::new(40, 30, vec![128; 40 * 30]);
        let est = match_block(
            &flat,
            &flat.clone(),
            &BlockMatchOptions {
                num_disparities: 16,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(est.valid_count(), 0, "flat image must produce no matches");
    }
}
