//! Pure-Rust dense stereo matching.
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
//! * **Semi-global (SGM, opt-in)** — with [`BlockMatchOptions::semi_global`] the
//!   raw ZNCC cost is aggregated along 8 paths with `P1`/`P2` smoothness
//!   penalties (Hirschmüller) before selection, propagating disparity into
//!   low-texture regions that block matching leaves blank. The raw cost volume
//!   is unchanged, so SGM is strictly an add-on.
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
//! Both modes materialize a full `W·H·num_disparities` cost volume (`f32`); SGM
//! adds two more volumes of that size (the accumulated and per-direction path
//! costs). This keeps the logic simple and is fine for the demos and modest
//! images. For large inputs, downscale first (as the real-data example does) or
//! tile the search.

use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// 8-bit grayscale image buffer, row-major (`x` is the fast axis), at least
/// one pixel in each direction.
#[derive(Debug, Clone)]
pub struct GrayImage {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl GrayImage {
    /// A `width × height` image from its pixels, `data[y * width + x]`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `width` or `height` is zero or
    /// `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_input(format!(
                "GrayImage: an image needs at least one pixel, got {width}×{height}"
            )));
        }
        if Some(data.len()) != width.checked_mul(height) {
            return Err(Error::invalid_input(format!(
                "GrayImage: {width}×{height} needs {} pixels, got {}",
                width.saturating_mul(height),
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The pixels, `data[y * width + x]`.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The pixels, `data[y * width + x]`, taken out of the image.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Read one pixel.
    ///
    /// # Panics
    /// Panics in debug builds if `(x, y)` is out of bounds.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        debug_assert!(
            x < self.width && y < self.height,
            "GrayImage::get out of bounds: ({x}, {y}) for {}×{}",
            self.width,
            self.height
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
#[non_exhaustive]
pub struct DisparityMap {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl DisparityMap {
    /// A `width × height` map from its values, `data[y * width + x]`, `NaN`
    /// for an invalid pixel.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::invalid_input(format!(
                "DisparityMap: a map needs at least one pixel, got {width}×{height}"
            )));
        }
        if Some(data.len()) != width.checked_mul(height) {
            return Err(Error::invalid_input(format!(
                "DisparityMap: {width}×{height} needs {} values, got {}",
                width.saturating_mul(height),
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Map width in pixels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Map height in pixels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// The values, `data[y * width + x]`, `NaN` = invalid.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The values, `data[y * width + x]`, taken out of the map.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Construct a map filled with a single value (use `f32::NAN` for all-invalid).
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        assert!(
            width > 0 && height > 0,
            "DisparityMap: a map needs at least one pixel, got {width}×{height}"
        );
        let len = width
            .checked_mul(height)
            .expect("DisparityMap dimensions overflow usize");
        Self {
            width,
            height,
            data: vec![value; len],
        }
    }

    /// Read one disparity value.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        assert!(
            x < self.width && y < self.height,
            "DisparityMap::get out of bounds: ({x}, {y}) for {}×{}",
            self.width,
            self.height
        );
        self.data[y * self.width + x]
    }

    /// Write one disparity value.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        assert!(
            x < self.width && y < self.height,
            "DisparityMap::set out of bounds: ({x}, {y}) for {}×{}",
            self.width,
            self.height
        );
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
#[non_exhaustive]
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
    /// Enable **semi-global** (SGM) cost aggregation.
    ///
    /// When set, the ZNCC matching cost is aggregated along 8 paths with the
    /// `sgm_p1`/`sgm_p2` smoothness penalties before disparity selection. This
    /// propagates disparity into low-texture regions (e.g. the flat interiors of
    /// chessboard squares) that pure block matching leaves blank, at the cost of
    /// extra time and memory. `min_correlation` / `uniqueness_ratio` (which gate
    /// on the raw per-pixel correlation) are bypassed in this mode — left-right
    /// consistency and the disparity-search window are the filters. The raw cost
    /// volume is the same as block mode, so SGM is strictly an add-on.
    pub semi_global: bool,
    /// SGM penalty for a ±1 disparity step (small — allows slanted surfaces).
    pub sgm_p1: f32,
    /// SGM penalty for a disparity step larger than 1 (large — preserves
    /// depth discontinuities). Must be ≥ `sgm_p1`.
    pub sgm_p2: f32,
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
            semi_global: false,
            sgm_p1: 0.1,
            sgm_p2: 1.5,
        }
    }
}

impl BlockMatchOptions {
    /// Construct default ZNCC stereo block-matching options.
    pub fn new() -> Self {
        Self::default()
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
    /// Allocate an empty integral buffer sized for `width × height`.
    fn zeroed(width: usize, height: usize) -> Self {
        let stride = width + 1;
        Self {
            width,
            sat: vec![0.0f64; stride * (height + 1)],
        }
    }

    /// Build the table from a per-pixel value function.
    fn build(width: usize, height: usize, value: impl Fn(usize, usize) -> f64) -> Self {
        let mut table = Self::zeroed(width, height);
        table.reset_from(width, height, value);
        table
    }

    /// Refill this integral buffer in place for `width × height` without reallocating.
    fn reset_from(&mut self, width: usize, height: usize, value: impl Fn(usize, usize) -> f64) {
        let stride = width + 1;
        let needed = stride * (height + 1);
        self.width = width;
        if self.sat.len() != needed {
            self.sat.resize(needed, 0.0);
        }
        // Top border row is zero; left column of each row is zero.
        self.sat[..stride].fill(0.0);
        for y in 0..height {
            let prev = y * stride;
            let curr = (y + 1) * stride;
            self.sat[curr] = 0.0;
            for x in 0..width {
                self.sat[curr + x + 1] =
                    value(x, y) + self.sat[prev + x + 1] + self.sat[curr + x] - self.sat[prev + x];
            }
        }
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
/// Returns [`Error::InvalidInput`] if the two images differ in size, if
/// `block_size` is even or zero, if `num_disparities <= 0`, if the disparity
/// range lies so far from zero that pixel indices shifted by it (plus or minus
/// twice the image width) leave the `i32` range, if a threshold or penalty is
/// out of its range, or if either image is smaller than the matching window.
pub fn match_block(
    left: &GrayImage,
    right: &GrayImage,
    opts: &BlockMatchOptions,
) -> Result<DisparityMap> {
    let (w, h) = (left.width, left.height);
    if right.width != w || right.height != h {
        return Err(Error::invalid_input(format!(
            "left ({w}x{h}) and right ({}x{}) images must have equal dimensions",
            right.width, right.height
        )));
    }
    if opts.block_size == 0 || opts.block_size.is_multiple_of(2) {
        return Err(Error::invalid_input(format!(
            "block_size must be odd and >= 1, got {}",
            opts.block_size
        )));
    }
    if opts.num_disparities <= 0 {
        return Err(Error::invalid_input(format!(
            "num_disparities must be positive, got {}",
            opts.num_disparities
        )));
    }
    // The search evaluates `x ∓ d` and `x ∓ d ± r` for every pixel column `x`,
    // window radius `r < w` and disparity `d` in the range; keeping the range
    // two image widths inside `i32` keeps every such index representable.
    let width_margin = i64::try_from(w).unwrap_or(i64::MAX).saturating_mul(2);
    let d_lo = i64::from(opts.min_disparity);
    let d_hi = d_lo + i64::from(opts.num_disparities);
    if d_lo.saturating_sub(width_margin) < -i64::from(i32::MAX)
        || d_hi.saturating_add(width_margin) > i64::from(i32::MAX)
    {
        return Err(Error::invalid_input(format!(
            "disparity range [{}, {d_hi}) is too far from zero for a {w}-pixel-wide image",
            opts.min_disparity
        )));
    }
    if !opts.min_correlation.is_finite() {
        return Err(Error::invalid_input(format!(
            "min_correlation must be finite, got {}",
            opts.min_correlation
        )));
    }
    if !opts.uniqueness_ratio.is_finite() || opts.uniqueness_ratio < 0.0 {
        return Err(Error::invalid_input(format!(
            "uniqueness_ratio must be finite and non-negative, got {}",
            opts.uniqueness_ratio
        )));
    }
    if !opts.lr_max_diff.is_finite() || opts.lr_max_diff < 0.0 {
        return Err(Error::invalid_input(format!(
            "lr_max_diff must be finite and non-negative, got {}",
            opts.lr_max_diff
        )));
    }
    if !opts.sgm_p1.is_finite() || opts.sgm_p1 < 0.0 {
        return Err(Error::invalid_input(format!(
            "sgm_p1 must be finite and non-negative, got {}",
            opts.sgm_p1
        )));
    }
    if !opts.sgm_p2.is_finite() || opts.sgm_p2 < opts.sgm_p1 {
        return Err(Error::invalid_input(format!(
            "sgm_p2 ({}) must be finite and >= sgm_p1 ({})",
            opts.sgm_p2, opts.sgm_p1
        )));
    }
    let r = opts.block_size / 2;
    if w < opts.block_size || h < opts.block_size {
        return Err(Error::invalid_input(format!(
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

    // Pixel-major L-to-R ZNCC volume computed once and reused for both passes.
    let zncc = zncc_cost_volume_l2r(left, right, &il, &ill, &ir, &irr, r, opts);
    let num_d = opts.num_disparities as usize;
    let left_disp = if opts.semi_global {
        let agg = aggregate_sgm(&zncc, w, h, num_d, opts.sgm_p1, opts.sgm_p2);
        select_sgm(&zncc, &agg, w, h, num_d, r, opts)
    } else {
        select_block(&zncc, w, h, num_d, r, opts)
    };

    if !opts.lr_consistency {
        return DisparityMap::new(w, h, left_disp);
    }

    // Derive right-referenced disparities from `zncc_l2r` via the identity
    // ZNCC(right(xr), left(xr + d)) == ZNCC(left(xr + d), right(xr)),
    // skipping a second integral-table sweep and `sqrt` evaluation.
    let right_disp = if opts.semi_global {
        let zncc_r2l = remap_zncc_r2l(&zncc, w, h, num_d, r, opts.min_disparity);
        let agg_r = aggregate_sgm(&zncc_r2l, w, h, num_d, opts.sgm_p1, opts.sgm_p2);
        select_sgm(&zncc_r2l, &agg_r, w, h, num_d, r, opts)
    } else {
        select_block_r2l(&zncc, w, h, num_d, r, opts)
    };

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
    DisparityMap::new(w, h, out)
}

/// Build the per-pixel pixel-major ZNCC cost volume `cost[(y*w + x)*num_d + k]`
/// (higher is better; [`INVALID_CORR`] where the window cannot be evaluated).
#[allow(clippy::too_many_arguments)]
fn zncc_cost_volume_l2r(
    left: &GrayImage,
    right: &GrayImage,
    il: &SummedArea,
    ill: &SummedArea,
    ir: &SummedArea,
    irr: &SummedArea,
    r: usize,
    opts: &BlockMatchOptions,
) -> Vec<f32> {
    let (w, h) = (left.width, left.height);
    let num_d = opts.num_disparities as usize;
    let n = ((2 * r + 1) * (2 * r + 1)) as f64;
    let mut cost = vec![INVALID_CORR; w * h * num_d];
    let mut iab = SummedArea::zeroed(w, h);

    for k in 0..num_d {
        let disp = opts.min_disparity + k as i32;
        // Cross-correlation term left(x, y) * right(x - disp, y); single SAT buffer reused.
        iab.reset_from(w, h, |x, y| {
            let tx = x as i32 - disp;
            if tx >= 0 && (tx as usize) < w {
                left.get(x, y) as f64 * right.get(tx as usize, y) as f64
            } else {
                0.0
            }
        });

        for y in r..h - r {
            let row_base = y * w;
            for x in r..w - r {
                let tx = x as i32 - disp;
                // The right window must lie fully inside the image.
                if tx - (r as i32) < 0 || tx + (r as i32) >= w as i32 {
                    continue;
                }
                let cxb = tx as usize;

                let s_a = il.window_sum(x, y, r);
                let s_aa = ill.window_sum(x, y, r);
                let s_b = ir.window_sum(cxb, y, r);
                let s_bb = irr.window_sum(cxb, y, r);
                let s_ab = iab.window_sum(x, y, r);

                let var_a = n * s_aa - s_a * s_a;
                let var_b = n * s_bb - s_b * s_b;
                if var_a <= 1e-6 || var_b <= 1e-6 {
                    continue; // flat window: correlation undefined
                }
                let cov = n * s_ab - s_a * s_b;
                let zncc = cov / (var_a.sqrt() * var_b.sqrt());
                cost[(row_base + x) * num_d + k] = zncc as f32;
            }
        }
    }
    cost
}

/// Remap L-to-R ZNCC into R-to-L pixel-major order via `R(xr) <-> L(xr + disp)`.
fn remap_zncc_r2l(
    zncc_l2r: &[f32],
    w: usize,
    h: usize,
    num_d: usize,
    r: usize,
    min_disparity: i32,
) -> Vec<f32> {
    let mut r2l = vec![INVALID_CORR; w * h * num_d];
    let x_lo = r as i32;
    let x_hi = (w - r) as i32;
    for y in r..h - r {
        let row_base = y * w;
        for xr in r..w - r {
            let dst = &mut r2l[(row_base + xr) * num_d..][..num_d];
            for (k, slot) in dst.iter_mut().enumerate() {
                let x = xr as i32 + (min_disparity + k as i32);
                if x >= x_lo && x < x_hi {
                    *slot = zncc_l2r[(row_base + x as usize) * num_d + k];
                }
            }
        }
    }
    r2l
}

#[inline]
fn select_block_pixel(costs: impl Fn(usize) -> f32, num_d: usize, opts: &BlockMatchOptions) -> f32 {
    let mut best_k = usize::MAX;
    let mut best = INVALID_CORR;
    for k in 0..num_d {
        let c = costs(k);
        if c > best {
            best = c;
            best_k = k;
        }
    }
    if best_k == usize::MAX || best < opts.min_correlation {
        return f32::NAN;
    }
    let mut second = INVALID_CORR;
    for k in 0..num_d {
        if k.abs_diff(best_k) <= 1 {
            continue;
        }
        let c = costs(k);
        if c > second {
            second = c;
        }
    }
    if second.is_finite() && best - second < opts.uniqueness_ratio {
        return f32::NAN;
    }
    let mut d = (opts.min_disparity + best_k as i32) as f32;
    if opts.subpixel && best_k > 0 && best_k + 1 < num_d {
        let cm = costs(best_k - 1);
        let cp = costs(best_k + 1);
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
    d
}

/// Block-matching selection: winner-take-all on raw ZNCC + min-correlation +
/// uniqueness + optional parabolic sub-pixel refinement.
fn select_block(
    cost: &[f32],
    w: usize,
    h: usize,
    num_d: usize,
    r: usize,
    opts: &BlockMatchOptions,
) -> Vec<f32> {
    let mut disp_map = vec![f32::NAN; w * h];
    for y in r..h - r {
        let row_base = y * w;
        for x in r..w - r {
            let slice = &cost[(row_base + x) * num_d..][..num_d];
            disp_map[row_base + x] = select_block_pixel(|k| slice[k], num_d, opts);
        }
    }
    disp_map
}

/// Zero-copy R-to-L winner-take-all directly from the L-to-R ZNCC cost volume.
fn select_block_r2l(
    cost_l2r: &[f32],
    w: usize,
    h: usize,
    num_d: usize,
    r: usize,
    opts: &BlockMatchOptions,
) -> Vec<f32> {
    let mut disp_map = vec![f32::NAN; w * h];
    let x_lo = r as i32;
    let x_hi = (w - r) as i32;
    for y in r..h - r {
        let row_base = y * w;
        for xr in r..w - r {
            disp_map[row_base + xr] = select_block_pixel(
                |k| {
                    let x = xr as i32 + (opts.min_disparity + k as i32);
                    if x >= x_lo && x < x_hi {
                        cost_l2r[(row_base + x as usize) * num_d + k]
                    } else {
                        INVALID_CORR
                    }
                },
                num_d,
                opts,
            );
        }
    }
    disp_map
}

/// Matching cost from a ZNCC value: `1 - zncc` (lower is better), with
/// out-of-range entries assigned the worst cost so SGM can still propagate
/// through them.
#[inline]
fn matching_cost(zncc: f32) -> f32 {
    if zncc.is_finite() {
        1.0 - zncc
    } else {
        COST_INVALID
    }
}

/// Worst matching cost (`1 - zncc` with `zncc = -1`), used for un-evaluable
/// entries during SGM aggregation.
const COST_INVALID: f32 = 2.0;

/// Accumulate one pixel step along a scan direction from predecessor `prev` into
/// `curr`, adding the resulting path costs into `agg_px`.
#[inline]
fn sgm_step_pixel(
    zncc_px: &[f32],
    prev: &[f32],
    curr: &mut [f32],
    agg_px: &mut [f32],
    p1: f32,
    p2: f32,
) {
    let num_d = zncc_px.len();
    let mut min_prev = f32::INFINITY;
    for &v in prev {
        min_prev = min_prev.min(v);
    }
    let fallback = min_prev + p2;
    for k in 0..num_d {
        let c = matching_cost(zncc_px[k]);
        let l_same = prev[k];
        let l_lo = if k > 0 {
            prev[k - 1] + p1
        } else {
            f32::INFINITY
        };
        let l_hi = if k + 1 < num_d {
            prev[k + 1] + p1
        } else {
            f32::INFINITY
        };
        let best = l_same.min(l_lo).min(l_hi).min(fallback);
        let l = c + best - min_prev;
        curr[k] = l;
        agg_px[k] += l;
    }
}

/// Start a path at this pixel: its path cost is the raw matching cost.
#[inline]
fn sgm_start_pixel(zncc_px: &[f32], curr: &mut [f32], agg_px: &mut [f32]) {
    for k in 0..zncc_px.len() {
        let c = matching_cost(zncc_px[k]);
        curr[k] = c;
        agg_px[k] += c;
    }
}

/// Semi-global aggregation of the pixel-major ZNCC cost volume along 8 paths.
///
/// Returns an aggregated cost volume (lower is better) where each entry sums the
/// Hirschmüller path costs `L_r` over the 8 cardinal/diagonal directions. Low-
/// texture pixels inherit a disparity from their textured neighbours through the
/// `p1`/`p2` smoothness term. Two reusable rolling row buffers keep the scratch
/// memory at `O(w * num_d)`.
fn aggregate_sgm(zncc: &[f32], w: usize, h: usize, num_d: usize, p1: f32, p2: f32) -> Vec<f32> {
    let mut agg = vec![0.0f32; w * h * num_d];
    let mut prev_row = vec![0.0f32; w * num_d];
    let mut curr_row = vec![0.0f32; w * num_d];

    // 1. Horizontal passes: (1, 0) and (-1, 0), rolling two pixel buffers.
    for dx in [1i32, -1i32] {
        for y in 0..h {
            let row_base = y * w;
            // One pixel of each row buffer (both hold at least `num_d` values,
            // also on a 1-pixel-wide image, where the path is a single start).
            let mut prev_buf = &mut prev_row[..num_d];
            let mut curr_buf = &mut curr_row[..num_d];
            for step in 0..w {
                let x = if dx > 0 { step } else { w - 1 - step };
                let base = (row_base + x) * num_d;
                if step == 0 {
                    sgm_start_pixel(
                        &zncc[base..base + num_d],
                        curr_buf,
                        &mut agg[base..base + num_d],
                    );
                } else {
                    sgm_step_pixel(
                        &zncc[base..base + num_d],
                        prev_buf,
                        curr_buf,
                        &mut agg[base..base + num_d],
                        p1,
                        p2,
                    );
                }
                std::mem::swap(&mut prev_buf, &mut curr_buf);
            }
        }
    }

    // 2. Vertical and diagonal passes using rolling row buffers.
    const DY_DIRS: [(i32, i32); 6] = [(0, 1), (1, 1), (-1, 1), (0, -1), (1, -1), (-1, -1)];
    for &(dx, dy) in &DY_DIRS {
        for y_step in 0..h {
            let y = if dy > 0 { y_step } else { h - 1 - y_step };
            let row_base = y * w;
            let has_prev_row = y_step > 0;
            for x in 0..w {
                let base = (row_base + x) * num_d;
                let px = x as i32 - dx;
                let curr_slice = &mut curr_row[x * num_d..(x + 1) * num_d];
                if !has_prev_row || px < 0 || px >= w as i32 {
                    sgm_start_pixel(
                        &zncc[base..base + num_d],
                        curr_slice,
                        &mut agg[base..base + num_d],
                    );
                } else {
                    let p_off = px as usize * num_d;
                    sgm_step_pixel(
                        &zncc[base..base + num_d],
                        &prev_row[p_off..p_off + num_d],
                        curr_slice,
                        &mut agg[base..base + num_d],
                        p1,
                        p2,
                    );
                }
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }
    }
    agg
}

/// SGM selection: winner-take-all (argmin) on the pixel-major aggregated cost
/// + optional parabolic sub-pixel refinement.
fn select_sgm(
    zncc: &[f32],
    agg: &[f32],
    w: usize,
    h: usize,
    num_d: usize,
    r: usize,
    opts: &BlockMatchOptions,
) -> Vec<f32> {
    let mut disp_map = vec![f32::NAN; w * h];
    for y in r..h - r {
        let row_base = y * w;
        for x in r..w - r {
            let base = (row_base + x) * num_d;
            let z_slice = &zncc[base..base + num_d];
            if !z_slice.iter().any(|v| v.is_finite()) {
                continue;
            }
            let a_slice = &agg[base..base + num_d];
            let mut best_k = 0usize;
            let mut best = f32::INFINITY;
            for (k, &c) in a_slice.iter().enumerate() {
                if c < best {
                    best = c;
                    best_k = k;
                }
            }
            let mut d = (opts.min_disparity + best_k as i32) as f32;
            if opts.subpixel && best_k > 0 && best_k + 1 < num_d {
                let cm = a_slice[best_k - 1];
                let cp = a_slice[best_k + 1];
                let denom = cm + cp - 2.0 * best;
                if denom.abs() > 1e-6 {
                    let delta = 0.5 * (cm - cp) / denom;
                    if delta.abs() < 1.0 {
                        d += delta;
                    }
                }
            }
            disp_map[row_base + x] = d;
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
        let left = GrayImage::new(w, h, ld).unwrap();
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
        (left, GrayImage::new(w, h, rd).unwrap(), gt)
    }

    fn rms_over_valid(est: &DisparityMap, gt: &[f32], r: usize) -> (f64, usize) {
        let (w, h) = (est.width(), est.height());
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
    fn semi_global_stays_accurate_and_densifies() {
        // SGM must remain accurate on textured data and never recover fewer
        // pixels than block matching (it fills, it does not drop).
        let (left, right, gt) = synth(100, 70, |x, _| 8.0 + 0.012 * x as f32);
        let block = BlockMatchOptions {
            min_disparity: 0,
            num_disparities: 16,
            block_size: 7,
            ..Default::default()
        };
        let sgm = BlockMatchOptions {
            semi_global: true,
            ..block.clone()
        };
        let est_block = match_block(&left, &right, &block).unwrap();
        let est_sgm = match_block(&left, &right, &sgm).unwrap();

        let (rms_sgm, n_sgm) = rms_over_valid(&est_sgm, &gt, 4);
        assert!(n_sgm > 1000, "SGM should recover densely, got {n_sgm}");
        assert!(rms_sgm < 0.5, "SGM RMS too high: {rms_sgm}");
        assert!(
            est_sgm.valid_count() >= est_block.valid_count(),
            "SGM ({}) should not recover fewer pixels than block ({})",
            est_sgm.valid_count(),
            est_block.valid_count()
        );
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
        let a = GrayImage::new(10, 10, vec![0; 100]).unwrap();
        let b = GrayImage::new(12, 10, vec![0; 120]).unwrap();
        let err = match_block(&a, &b, &BlockMatchOptions::default()).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
    }

    #[test]
    fn rejects_even_block_size() {
        let a = GrayImage::new(20, 20, vec![0; 400]).unwrap();
        let b = GrayImage::new(20, 20, vec![0; 400]).unwrap();
        let err = match_block(
            &a,
            &b,
            &BlockMatchOptions {
                block_size: 8,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
    }

    #[test]
    fn rejects_window_larger_than_image() {
        let a = GrayImage::new(5, 5, vec![0; 25]).unwrap();
        let b = GrayImage::new(5, 5, vec![0; 25]).unwrap();
        let err = match_block(
            &a,
            &b,
            &BlockMatchOptions {
                block_size: 7,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
    }

    #[test]
    fn textureless_region_is_rejected() {
        // A flat (zero-variance) image yields undefined correlation everywhere,
        // so every pixel must be invalid rather than spuriously matched.
        let flat = GrayImage::new(40, 30, vec![128; 40 * 30]).unwrap();
        for semi_global in [false, true] {
            let est = match_block(
                &flat,
                &flat.clone(),
                &BlockMatchOptions {
                    num_disparities: 16,
                    semi_global,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                est.valid_count(),
                0,
                "flat image (semi_global={semi_global}) must produce no matches"
            );
        }
    }

    #[test]
    fn images_need_a_pixel_and_a_matching_buffer() {
        assert!(GrayImage::new(0, 0, Vec::new()).is_err());
        assert!(GrayImage::new(3, 2, vec![0; 5]).is_err());
        assert!(GrayImage::new(usize::MAX, 2, vec![0; 5]).is_err());
        assert!(DisparityMap::new(0, 0, Vec::new()).is_err());
        assert!(DisparityMap::new(0, 3, Vec::new()).is_err());
        assert!(DisparityMap::new(3, 2, vec![0.0; 7]).is_err());
        let a = GrayImage::new(20, 20, vec![0; 400]).unwrap();
        for bad in [
            BlockMatchOptions {
                sgm_p1: -0.1,
                ..Default::default()
            },
            BlockMatchOptions {
                sgm_p1: 0.5,
                sgm_p2: 0.2,
                ..Default::default()
            },
            BlockMatchOptions {
                min_correlation: f32::NAN,
                ..Default::default()
            },
            BlockMatchOptions {
                min_disparity: i32::MAX,
                num_disparities: 8,
                ..Default::default()
            },
        ] {
            assert!(match_block(&a, &a, &bad).is_err());
        }
        let one = GrayImage::new(1, 1, vec![7]).unwrap();
        assert_eq!(one.bilinear(5.0, -3.0), 7.0);
    }

    #[test]
    fn rejects_disparity_ranges_that_would_overflow_pixel_indices() {
        let (left, right, _) = synth(32, 24, |_, _| 3.0);
        // `min + num` fits in `i32`, but `x + d` for the last column does not.
        for (min_disparity, num_disparities) in [
            (i32::MAX - 16, 8),
            (i32::MIN, 8),
            (i32::MIN + 40, 8),
            (i32::MAX - 70, 8),
        ] {
            for semi_global in [false, true] {
                let opts = BlockMatchOptions {
                    min_disparity,
                    num_disparities,
                    block_size: 5,
                    semi_global,
                    ..Default::default()
                };
                assert!(
                    match_block(&left, &right, &opts).is_err(),
                    "range [{min_disparity}, +{num_disparities}) must be rejected"
                );
            }
        }
        // Far but representable ranges are searched (and simply find nothing).
        for min_disparity in [i32::MAX - 200, i32::MIN + 200] {
            let opts = BlockMatchOptions {
                min_disparity,
                num_disparities: 8,
                block_size: 5,
                semi_global: true,
                ..Default::default()
            };
            let est = match_block(&left, &right, &opts).unwrap();
            assert_eq!(est.valid_count(), 0);
        }
    }

    #[test]
    fn one_pixel_wide_images_match_without_panicking() {
        let data: Vec<u8> = (0..9).map(|y| texture_u8(0, y)).collect();
        let left = GrayImage::new(1, 9, data.clone()).unwrap();
        let right = GrayImage::new(1, 9, data).unwrap();
        for semi_global in [false, true] {
            for lr_consistency in [false, true] {
                let opts = BlockMatchOptions {
                    min_disparity: 0,
                    num_disparities: 4,
                    block_size: 1,
                    semi_global,
                    lr_consistency,
                    ..Default::default()
                };
                let est = match_block(&left, &right, &opts).unwrap();
                assert_eq!((est.width(), est.height()), (1, 9));
            }
        }
    }
}
