//! What limits reprojection error: a calibration-free smooth-fit floor and a
//! decomposition of the calibrated residuals.
//!
//! A camera model is a smooth map from board to image, so it cannot absorb
//! per-detection scatter. This diagnostic measures that scatter without any
//! calibration (a per-view polynomial fit of the image coordinates as a
//! function of board position), splits what *repeats* from what does not
//! (per-marker offsets in the board frame, camera-fixed and board-fixed
//! fields, per-view offsets), and places the calibrated residual next to it.
//!
//! The analysis ([`analyze`]) is pure: it takes [`FloorCamera`]s, one per
//! camera, each a list of detections with the calibrated residual attached.
//! With `--features tier-b`, [`measure_entry`] builds a [`FloorCamera`] from a
//! registry entry by running the entry's calibration route (detection through
//! the detection cache). `calib-bench diagnose floor` drives both.
//!
//! # Sections
//!
//! 1. **Smooth-fit scatter** — per view, a least-squares polynomial (quartic,
//!    15 terms; cubic, 10 terms, for the convergence check) of each image
//!    coordinate in board `(x, y)` normalized to `[-1, 1]`, three rounds of
//!    outlier rejection at `max(4·median|r|, 0.3 px)`. Inlier residuals are
//!    scaled by `sqrt(n / (n − p))` for the degrees of freedom the fit used.
//! 2. **Per-marker repeatable part** — each quartic residual mapped into the
//!    board frame (mm) by the inverse Jacobian of its view's fit, grouped by
//!    board point across views and cameras; the between-marker variance of
//!    the group means, corrected for the noise they carry.
//! 3. **Neighbour correlation and isotropy** of the quartic residuals.
//! 4. **Decomposition of the calibrated residuals** — the camera-fixed field
//!    (image cells), the board-fixed field (board points), the per-view mean,
//!    each as a noise-corrected fraction of the residual variance; the
//!    calibrated and smooth-fit magnitudes side by side. Sections 3 and 4 use
//!    the points that are quartic-fit inliers, so all numbers describe one
//!    point set. The three fractions are independent views of the same
//!    residuals, not an orthogonal split.
//!
//! Everything is deterministic: no wall-clock values, ordered maps only.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use anyhow::{Result, ensure};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A view needs at least this many points for the smooth fit (and at least
/// this many inliers after rejection).
pub const MIN_VIEW_POINTS: usize = 27;
/// Rounds of outlier rejection in the smooth fit.
const REJECTION_ROUNDS: usize = 3;
/// Outlier threshold: `max(REJECTION_MEDIANS · median|r|, REJECTION_FLOOR_PX)`.
const REJECTION_MEDIANS: f64 = 4.0;
const REJECTION_FLOOR_PX: f64 = 0.3;
/// A group (marker, cell, view) needs this many observations to enter a
/// variance decomposition.
pub const MIN_GROUP_OBS: usize = 8;
/// Each half of the cameras needs this many observations of a marker for the
/// split correlation.
const MIN_SPLIT_OBS: usize = 4;
/// Board points are the same marker when their coordinates agree to this many
/// millimetres (10 µm).
const KEY_RESOLUTION_MM: f64 = 0.01;
/// Image cells of the camera-fixed field: columns × rows.
const CELL_GRID: (usize, usize) = (6, 4);

/// One detected feature of one view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorPoint {
    /// View index (unique within the camera).
    pub view: usize,
    /// Board position of the feature, millimetres.
    pub board_mm: [f64; 2],
    /// Detected image position, pixels.
    pub observed_px: [f64; 2],
    /// Calibrated residual `observed − projected`, pixels; `None` when the
    /// calibration could not project the feature.
    pub residual_px: Option<[f64; 2]>,
}

/// The detections of one camera.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorCamera {
    /// Camera label for the report.
    pub id: String,
    /// Image `[width, height]` in pixels; `None` takes the bounding box of the
    /// detections as the image frame.
    pub image_size: Option<[f64; 2]>,
    /// Every detected feature of every view.
    pub points: Vec<FloorPoint>,
}

// ───────────────────────── report types ─────────────────────────

/// Smooth-fit scatter of one degree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmoothFitStats {
    /// Polynomial degree.
    pub degree: usize,
    /// Number of polynomial terms per coordinate.
    pub terms: usize,
    /// Views that were fitted.
    pub views_used: usize,
    /// Views skipped (too few points or a singular fit).
    pub views_skipped: usize,
    /// Inlier points the statistics cover.
    pub points: usize,
    /// Points rejected as outliers in the fitted views.
    pub outliers: usize,
    /// RMS of the scaled inlier residual magnitude, pixels.
    pub rms_px: f64,
    /// Median of the scaled inlier residual magnitude, pixels.
    pub median_px: f64,
    /// Median local scale `sqrt|det J|` of the fit, px/mm.
    pub median_scale_px_per_mm: f64,
}

/// Variance components of grouped 2-vectors (see [`variance_components`]).
/// Variances are in the squared unit of the vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarianceComponents {
    /// Groups with at least [`MIN_GROUP_OBS`] observations.
    pub groups: usize,
    /// Observations in those groups.
    pub observations: usize,
    /// Mean squared vector norm.
    pub total_var: f64,
    /// Within-group variance (trace).
    pub within_var: f64,
    /// Between-group variance of the means, corrected for their noise (can be
    /// slightly negative when the true value is zero).
    pub between_var: f64,
    /// `sqrt(max(between_var, 0))`.
    pub between_rms: f64,
    /// `between_var / total_var`.
    pub fraction: f64,
}

/// Neighbour correlation and isotropy of the quartic residuals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Isotropy {
    /// Residuals with a neighbour.
    pub pairs: usize,
    /// Correlation of a residual vector with its nearest board neighbour's.
    pub neighbour_correlation: f64,
    /// RMS of the u component, pixels.
    pub sigma_u_px: f64,
    /// RMS of the v component, pixels.
    pub sigma_v_px: f64,
    /// RMS of the component along the direction from the image centre, pixels.
    pub radial_px: f64,
    /// RMS of the component across that direction, pixels.
    pub tangential_px: f64,
}

/// The calibrated residual next to the smooth fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibratedStats {
    /// Points with a calibrated residual that are quartic inliers.
    pub points: usize,
    /// RMS of the calibrated residual magnitude on those points, pixels.
    pub rms_px: f64,
    /// Median of the same, pixels.
    pub median_px: f64,
    /// RMS over every finite calibrated residual (outliers included), pixels.
    pub rms_all_px: f64,
    /// Median over every finite calibrated residual, pixels.
    pub median_all_px: f64,
    /// Quartic smooth-fit RMS, pixels.
    pub smooth_rms_px: f64,
    /// Quartic smooth-fit median, pixels.
    pub smooth_median_px: f64,
    /// `sqrt(max(rms² − smooth_rms², 0))`: the smooth systematic gap, pixels.
    pub smooth_gap_rms_px: f64,
}

/// Mean calibrated residual of one image cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellStat {
    /// Cell column (0 = left).
    pub col: usize,
    /// Cell row (0 = top).
    pub row: usize,
    /// Observations in the cell.
    pub count: usize,
    /// Mean residual vector, pixels.
    pub mean_px: [f64; 2],
    /// Standard error of the mean vector's norm, pixels.
    pub std_err_px: f64,
}

/// One camera's report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraFloor {
    /// Camera label.
    pub id: String,
    /// Detections in the input.
    pub points: usize,
    /// Quartic smooth fit (section 1).
    pub quartic: SmoothFitStats,
    /// Cubic smooth fit (section 1).
    pub cubic: SmoothFitStats,
    /// Neighbour correlation and isotropy (section 3).
    pub isotropy: Isotropy,
    /// Calibrated residual vs smooth fit (section 4d).
    pub calibrated: CalibratedStats,
    /// Mean residual per image cell, row-major (section 4a).
    pub cells: Vec<CellStat>,
    /// Camera-fixed field: variance explained by image cells (section 4a).
    pub camera_field: VarianceComponents,
    /// Board-fixed field of this camera (section 4b).
    pub board_field: VarianceComponents,
    /// Per-view mean vector (section 4c).
    pub view_field: VarianceComponents,
}

/// Per-marker repeatable part of the smooth-fit residuals (section 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MarkerReport {
    /// Variance components of the board-frame residuals, mm².
    pub components: VarianceComponents,
    /// Markers entering the split correlation.
    pub split_markers: usize,
    /// Correlation of per-marker mean offsets between two halves of the
    /// cameras (even- and odd-indexed); `None` with fewer than two cameras or
    /// markers.
    pub split_correlation: Option<f64>,
}

/// Correlation of per-board-point mean calibrated residuals of two cameras.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPairCorrelation {
    /// First camera label.
    pub a: String,
    /// Second camera label.
    pub b: String,
    /// Board points in common.
    pub markers: usize,
    /// Pearson correlation of the mean vectors' components.
    pub correlation: f64,
}

/// Quantities pooled over all given cameras.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PooledFloor {
    /// Quartic smooth fit.
    pub quartic: SmoothFitStats,
    /// Cubic smooth fit.
    pub cubic: SmoothFitStats,
    /// Per-marker repeatable part (section 2).
    pub markers: MarkerReport,
    /// Neighbour correlation and isotropy.
    pub isotropy: Isotropy,
    /// Calibrated residual vs smooth fit.
    pub calibrated: CalibratedStats,
    /// Camera-fixed field, groups keyed by (camera, cell).
    pub camera_field: VarianceComponents,
    /// Board-fixed field, groups keyed by board point across cameras.
    pub board_field: VarianceComponents,
    /// Per-view mean vector, groups keyed by (camera, view).
    pub view_field: VarianceComponents,
    /// Between-camera correlation of the per-point means, per camera pair.
    pub board_field_pairs: Vec<CameraPairCorrelation>,
    /// Mean of the pair correlations, `None` without a pair.
    pub board_field_mean_correlation: Option<f64>,
}

/// The `diagnose floor` result (a bench-internal format, no schema).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorReport {
    /// One entry per camera, in input order.
    pub cameras: Vec<CameraFloor>,
    /// Pooled over the cameras.
    pub pooled: PooledFloor,
}

// ───────────────────────── numerics ─────────────────────────

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

/// Pearson correlation of two equal-length samples, `None` when degenerate.
fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() < 3 || a.len() != b.len() {
        return None;
    }
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ab += (x - ma) * (y - mb);
        aa += (x - ma) * (x - ma);
        bb += (y - mb) * (y - mb);
    }
    let denom = (aa * bb).sqrt();
    (denom > 0.0).then(|| ab / denom)
}

/// The `(i, j)` exponents of the monomials `x^i y^j`, `i + j ≤ degree`.
fn monomials(degree: usize) -> Vec<(i32, i32)> {
    let d = degree as i32;
    (0..=d)
        .flat_map(|total| (0..=total).map(move |i| (i, total - i)))
        .collect()
}

fn monomial_row(terms: &[(i32, i32)], x: f64, y: f64) -> Vec<f64> {
    terms.iter().map(|&(i, j)| x.powi(i) * y.powi(j)).collect()
}

/// `(∂/∂x, ∂/∂y)` of each monomial.
fn monomial_gradient(terms: &[(i32, i32)], x: f64, y: f64) -> (Vec<f64>, Vec<f64>) {
    let dx = terms
        .iter()
        .map(|&(i, j)| {
            if i == 0 {
                0.0
            } else {
                f64::from(i) * x.powi(i - 1) * y.powi(j)
            }
        })
        .collect();
    let dy = terms
        .iter()
        .map(|&(i, j)| {
            if j == 0 {
                0.0
            } else {
                f64::from(j) * x.powi(i) * y.powi(j - 1)
            }
        })
        .collect();
    (dx, dy)
}

/// Variance components of grouped 2-vectors.
///
/// Groups with fewer than `min_obs` observations are dropped. With group
/// means `m̄ₖ`, `N` observations in `K` groups: `within = Σ‖x − m̄ₖ‖² / (N−K)`,
/// `total = Σ‖x‖² / N`, and `between = Σ nₖ‖m̄ₖ‖² / N − K·within / N`, the
/// variance of the true group means once the noise `within / nₖ` each sample
/// mean carries is removed.
pub fn variance_components<'a>(
    groups: impl IntoIterator<Item = &'a Vec<[f64; 2]>>,
    min_obs: usize,
) -> VarianceComponents {
    let (mut k, mut n) = (0usize, 0usize);
    let (mut sum_sq, mut ss_within, mut weighted_mean_sq) = (0.0, 0.0, 0.0);
    for g in groups.into_iter().filter(|g| g.len() >= min_obs.max(1)) {
        let len = g.len() as f64;
        let mx = g.iter().map(|v| v[0]).sum::<f64>() / len;
        let my = g.iter().map(|v| v[1]).sum::<f64>() / len;
        for v in g {
            sum_sq += v[0] * v[0] + v[1] * v[1];
            ss_within += (v[0] - mx).powi(2) + (v[1] - my).powi(2);
        }
        weighted_mean_sq += len * (mx * mx + my * my);
        k += 1;
        n += g.len();
    }
    if n == 0 {
        return VarianceComponents {
            groups: 0,
            observations: 0,
            total_var: 0.0,
            within_var: 0.0,
            between_var: 0.0,
            between_rms: 0.0,
            fraction: 0.0,
        };
    }
    let nf = n as f64;
    let total_var = sum_sq / nf;
    let within_var = if n > k {
        ss_within / (n - k) as f64
    } else {
        0.0
    };
    let between_var = weighted_mean_sq / nf - k as f64 * within_var / nf;
    VarianceComponents {
        groups: k,
        observations: n,
        total_var,
        within_var,
        between_var,
        between_rms: between_var.max(0.0).sqrt(),
        fraction: if total_var > 0.0 {
            between_var / total_var
        } else {
            0.0
        },
    }
}

fn mean2(g: &[[f64; 2]]) -> [f64; 2] {
    let n = g.len() as f64;
    [
        g.iter().map(|v| v[0]).sum::<f64>() / n,
        g.iter().map(|v| v[1]).sum::<f64>() / n,
    ]
}

type MarkerKey = (i64, i64);

fn marker_key(board_mm: [f64; 2]) -> MarkerKey {
    (
        (board_mm[0] / KEY_RESOLUTION_MM).round() as i64,
        (board_mm[1] / KEY_RESOLUTION_MM).round() as i64,
    )
}

// ───────────────────────── smooth fit ─────────────────────────

/// One view's polynomial fit: the inliers with their scaled residuals and the
/// local Jacobian of the fitted map.
struct ViewFit {
    /// Inlier indices into the camera's `points`.
    inliers: Vec<usize>,
    /// Scaled residual `observed − fit`, pixels, per inlier.
    residual: Vec<[f64; 2]>,
    /// `[du/dX, du/dY, dv/dX, dv/dY]` per inlier, px/mm.
    jacobian: Vec<[f64; 4]>,
    /// Points rejected as outliers.
    outliers: usize,
}

struct CameraFit {
    views: Vec<ViewFit>,
    views_skipped: usize,
}

/// Normal-equation least squares of two right-hand sides over the rows where
/// `keep` holds; `None` when the system is singular.
fn solve_normal(rows: &[Vec<f64>], rhs: &[[f64; 2]], keep: &[bool]) -> Option<DMatrix<f64>> {
    let p = rows[0].len();
    let mut ata = DMatrix::<f64>::zeros(p, p);
    let mut atb = DMatrix::<f64>::zeros(p, 2);
    for (row, b) in rows
        .iter()
        .zip(rhs)
        .zip(keep)
        .filter_map(|(rb, &k)| k.then_some(rb))
    {
        for i in 0..p {
            for j in 0..p {
                ata[(i, j)] += row[i] * row[j];
            }
            atb[(i, 0)] += row[i] * b[0];
            atb[(i, 1)] += row[i] * b[1];
        }
    }
    let solution = ata.cholesky()?.solve(&atb);
    solution.iter().all(|v| v.is_finite()).then_some(solution)
}

fn fit_view(points: &[FloorPoint], idx: &[usize], degree: usize) -> Option<ViewFit> {
    let n = idx.len();
    if n < MIN_VIEW_POINTS {
        return None;
    }
    let terms = monomials(degree);
    let p = terms.len();
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for &i in idx {
        for a in 0..2 {
            lo[a] = lo[a].min(points[i].board_mm[a]);
            hi[a] = hi[a].max(points[i].board_mm[a]);
        }
    }
    let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
    let half = 0.5 * (hi[0] - lo[0]).max(hi[1] - lo[1]);
    if !(half > 0.0 && half.is_finite()) {
        return None;
    }
    let xy: Vec<[f64; 2]> = idx
        .iter()
        .map(|&i| {
            let b = points[i].board_mm;
            [(b[0] - centre[0]) / half, (b[1] - centre[1]) / half]
        })
        .collect();
    let rows: Vec<Vec<f64>> = xy
        .iter()
        .map(|q| monomial_row(&terms, q[0], q[1]))
        .collect();
    let obs: Vec<[f64; 2]> = idx.iter().map(|&i| points[i].observed_px).collect();

    let predict = |coef: &DMatrix<f64>, row: &[f64]| -> [f64; 2] {
        let mut out = [0.0; 2];
        for (t, &r) in row.iter().enumerate() {
            out[0] += r * coef[(t, 0)];
            out[1] += r * coef[(t, 1)];
        }
        out
    };
    let residuals = |coef: &DMatrix<f64>| -> Vec<[f64; 2]> {
        rows.iter()
            .zip(&obs)
            .map(|(row, o)| {
                let f = predict(coef, row);
                [o[0] - f[0], o[1] - f[1]]
            })
            .collect()
    };

    let mut keep = vec![true; n];
    for _ in 0..REJECTION_ROUNDS {
        let coef = solve_normal(&rows, &obs, &keep)?;
        let mags: Vec<f64> = residuals(&coef).iter().map(|r| r[0].hypot(r[1])).collect();
        let threshold = (REJECTION_MEDIANS * median(&mut mags.clone())).max(REJECTION_FLOOR_PX);
        keep = mags.iter().map(|&m| m <= threshold).collect();
        if keep.iter().filter(|&&k| k).count() < MIN_VIEW_POINTS {
            return None;
        }
    }
    let coef = solve_normal(&rows, &obs, &keep)?;
    let n_in = keep.iter().filter(|&&k| k).count();
    let scale = (n_in as f64 / (n_in as f64 - p as f64)).sqrt();
    let all = residuals(&coef);

    let mut fit = ViewFit {
        inliers: Vec::with_capacity(n_in),
        residual: Vec::with_capacity(n_in),
        jacobian: Vec::with_capacity(n_in),
        outliers: n - n_in,
    };
    for (k, &keep_k) in keep.iter().enumerate() {
        if !keep_k {
            continue;
        }
        let (gx, gy) = monomial_gradient(&terms, xy[k][0], xy[k][1]);
        let (mut du_dx, mut du_dy, mut dv_dx, mut dv_dy) = (0.0, 0.0, 0.0, 0.0);
        for t in 0..p {
            du_dx += gx[t] * coef[(t, 0)];
            du_dy += gy[t] * coef[(t, 0)];
            dv_dx += gx[t] * coef[(t, 1)];
            dv_dy += gy[t] * coef[(t, 1)];
        }
        fit.inliers.push(idx[k]);
        fit.residual.push([all[k][0] * scale, all[k][1] * scale]);
        fit.jacobian
            .push([du_dx / half, du_dy / half, dv_dx / half, dv_dy / half]);
    }
    Some(fit)
}

fn fit_camera(camera: &FloorCamera, degree: usize) -> CameraFit {
    let mut by_view: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, p) in camera.points.iter().enumerate() {
        if p.observed_px
            .iter()
            .chain(&p.board_mm)
            .all(|v| v.is_finite())
        {
            by_view.entry(p.view).or_default().push(i);
        }
    }
    let mut views = Vec::new();
    let mut views_skipped = 0;
    for idx in by_view.values() {
        match fit_view(&camera.points, idx, degree) {
            Some(v) => views.push(v),
            None => views_skipped += 1,
        }
    }
    CameraFit {
        views,
        views_skipped,
    }
}

fn smooth_stats(fits: &[&CameraFit], degree: usize) -> SmoothFitStats {
    let mut mags = Vec::new();
    let mut scales = Vec::new();
    let (mut views_used, mut views_skipped, mut outliers) = (0, 0, 0);
    for fit in fits {
        views_skipped += fit.views_skipped;
        for v in &fit.views {
            views_used += 1;
            outliers += v.outliers;
            mags.extend(v.residual.iter().map(|r| r[0].hypot(r[1])));
            scales.extend(
                v.jacobian
                    .iter()
                    .map(|j| (j[0] * j[3] - j[1] * j[2]).abs().sqrt()),
            );
        }
    }
    SmoothFitStats {
        degree,
        terms: monomials(degree).len(),
        views_used,
        views_skipped,
        points: mags.len(),
        outliers,
        rms_px: rms(&mags),
        median_px: median(&mut mags),
        median_scale_px_per_mm: median(&mut scales),
    }
}

// ───────────────────────── analysis ─────────────────────────

/// Image frame `[x0, y0, width, height]`: the given size, else the bounding
/// box of the detections.
fn frame(camera: &FloorCamera) -> [f64; 4] {
    if let Some([w, h]) = camera.image_size {
        return [0.0, 0.0, w, h];
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in &camera.points {
        for a in 0..2 {
            lo[a] = lo[a].min(p.observed_px[a]);
            hi[a] = hi[a].max(p.observed_px[a]);
        }
    }
    [
        lo[0],
        lo[1],
        (hi[0] - lo[0]).max(1.0),
        (hi[1] - lo[1]).max(1.0),
    ]
}

fn cell_of(frame: [f64; 4], px: [f64; 2]) -> (usize, usize) {
    let cell = |v: f64, origin: f64, extent: f64, n: usize| {
        (((v - origin) / extent * n as f64).floor().max(0.0) as usize).min(n - 1)
    };
    (
        cell(px[0], frame[0], frame[2], CELL_GRID.0),
        cell(px[1], frame[1], frame[3], CELL_GRID.1),
    )
}

/// Sums for the neighbour correlation and isotropy.
#[derive(Default, Clone, Copy)]
struct IsotropySums {
    pairs: usize,
    dot: f64,
    aa: f64,
    bb: f64,
    uu: f64,
    vv: f64,
    radial: f64,
    tangential: f64,
    n: usize,
}

impl IsotropySums {
    fn add(&mut self, other: &Self) {
        self.pairs += other.pairs;
        self.dot += other.dot;
        self.aa += other.aa;
        self.bb += other.bb;
        self.uu += other.uu;
        self.vv += other.vv;
        self.radial += other.radial;
        self.tangential += other.tangential;
        self.n += other.n;
    }

    fn finish(&self) -> Isotropy {
        let n = self.n.max(1) as f64;
        let denom = (self.aa * self.bb).sqrt();
        Isotropy {
            pairs: self.pairs,
            neighbour_correlation: if denom > 0.0 { self.dot / denom } else { 0.0 },
            sigma_u_px: (self.uu / n).sqrt(),
            sigma_v_px: (self.vv / n).sqrt(),
            radial_px: (self.radial / n).sqrt(),
            tangential_px: (self.tangential / n).sqrt(),
        }
    }
}

fn isotropy_sums(camera: &FloorCamera, fit: &CameraFit) -> IsotropySums {
    let f = frame(camera);
    let centre = [f[0] + 0.5 * f[2], f[1] + 0.5 * f[3]];
    let mut s = IsotropySums::default();
    for v in &fit.views {
        for (a, ra) in v.inliers.iter().zip(&v.residual) {
            let p = &camera.points[*a];
            s.n += 1;
            s.uu += ra[0] * ra[0];
            s.vv += ra[1] * ra[1];
            let d = [p.observed_px[0] - centre[0], p.observed_px[1] - centre[1]];
            let len = d[0].hypot(d[1]);
            if len > 0.0 {
                let er = [d[0] / len, d[1] / len];
                let radial = ra[0] * er[0] + ra[1] * er[1];
                let tangential = -ra[0] * er[1] + ra[1] * er[0];
                s.radial += radial * radial;
                s.tangential += tangential * tangential;
            }
            // Nearest board neighbour in the same view (ties: lowest index).
            let mut best: Option<(f64, usize)> = None;
            for (k, b) in v.inliers.iter().enumerate() {
                if b == a {
                    continue;
                }
                let q = &camera.points[*b];
                let d2 = (q.board_mm[0] - p.board_mm[0]).powi(2)
                    + (q.board_mm[1] - p.board_mm[1]).powi(2);
                if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, k));
                }
            }
            if let Some((_, k)) = best {
                let rb = &v.residual[k];
                s.pairs += 1;
                s.dot += ra[0] * rb[0] + ra[1] * rb[1];
                s.aa += ra[0] * ra[0] + ra[1] * ra[1];
                s.bb += rb[0] * rb[0] + rb[1] * rb[1];
            }
        }
    }
    s
}

/// One calibrated residual of a quartic inlier.
struct CalObs {
    view: usize,
    key: MarkerKey,
    cell: (usize, usize),
    d: [f64; 2],
}

fn calibrated_observations(camera: &FloorCamera, fit: &CameraFit) -> Vec<CalObs> {
    let f = frame(camera);
    fit.views
        .iter()
        .flat_map(|v| v.inliers.iter())
        .filter_map(|&i| {
            let p = &camera.points[i];
            let d = p.residual_px.filter(|d| d.iter().all(|x| x.is_finite()))?;
            Some(CalObs {
                view: p.view,
                key: marker_key(p.board_mm),
                cell: cell_of(f, p.observed_px),
                d,
            })
        })
        .collect()
}

fn calibrated_stats(
    camera_points: &[&FloorCamera],
    obs: &[&[CalObs]],
    smooth: &SmoothFitStats,
) -> CalibratedStats {
    let inlier: Vec<f64> = obs
        .iter()
        .flat_map(|o| o.iter())
        .map(|o| o.d[0].hypot(o.d[1]))
        .collect();
    let all: Vec<f64> = camera_points
        .iter()
        .flat_map(|c| &c.points)
        .filter_map(|p| p.residual_px)
        .filter(|d| d.iter().all(|x| x.is_finite()))
        .map(|d| d[0].hypot(d[1]))
        .collect();
    let rms_px = rms(&inlier);
    CalibratedStats {
        points: inlier.len(),
        rms_px,
        median_px: median(&mut inlier.clone()),
        rms_all_px: rms(&all),
        median_all_px: median(&mut all.clone()),
        smooth_rms_px: smooth.rms_px,
        smooth_median_px: smooth.median_px,
        smooth_gap_rms_px: (rms_px * rms_px - smooth.rms_px * smooth.rms_px)
            .max(0.0)
            .sqrt(),
    }
}

fn group<K: Ord>(items: impl Iterator<Item = (K, [f64; 2])>) -> BTreeMap<K, Vec<[f64; 2]>> {
    let mut groups: BTreeMap<K, Vec<[f64; 2]>> = BTreeMap::new();
    for (k, v) in items {
        groups.entry(k).or_default().push(v);
    }
    groups
}

fn cell_stats(obs: &[CalObs]) -> Vec<CellStat> {
    let groups = group(obs.iter().map(|o| ((o.cell.1, o.cell.0), o.d)));
    groups
        .into_iter()
        .map(|((row, col), g)| {
            let n = g.len();
            let m = mean2(&g);
            let ss: f64 = g
                .iter()
                .map(|v| (v[0] - m[0]).powi(2) + (v[1] - m[1]).powi(2))
                .sum();
            let var = if n > 1 { ss / (n - 1) as f64 } else { 0.0 };
            CellStat {
                col,
                row,
                count: n,
                mean_px: m,
                std_err_px: (var / n as f64).sqrt(),
            }
        })
        .collect()
}

/// Board-frame residuals (mm) of a camera's quartic inliers, by marker.
fn board_frame_residuals(camera: &FloorCamera, fit: &CameraFit) -> Vec<(MarkerKey, [f64; 2])> {
    let mut out = Vec::new();
    for v in &fit.views {
        for ((&i, r), j) in v.inliers.iter().zip(&v.residual).zip(&v.jacobian) {
            let det = j[0] * j[3] - j[1] * j[2];
            if det.abs() < 1e-9 {
                continue;
            }
            // m = J⁻¹ r.
            let m = [
                (j[3] * r[0] - j[1] * r[1]) / det,
                (-j[2] * r[0] + j[0] * r[1]) / det,
            ];
            out.push((marker_key(camera.points[i].board_mm), m));
        }
    }
    out
}

fn marker_report(per_camera: &[Vec<(MarkerKey, [f64; 2])>]) -> MarkerReport {
    let groups = group(
        per_camera
            .iter()
            .flat_map(|c| c.iter().map(|(k, m)| (*k, *m))),
    );
    let components = variance_components(groups.values(), MIN_GROUP_OBS);

    // Split correlation: even- vs odd-indexed cameras.
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut split_markers = 0;
    if per_camera.len() >= 2 {
        let half = |parity: usize| {
            group(
                per_camera
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 2 == parity)
                    .flat_map(|(_, c)| c.iter().map(|(k, m)| (*k, *m))),
            )
        };
        let (ga, gb) = (half(0), half(1));
        for (key, total) in &groups {
            if total.len() < MIN_GROUP_OBS {
                continue;
            }
            if let (Some(x), Some(y)) = (ga.get(key), gb.get(key))
                && x.len() >= MIN_SPLIT_OBS
                && y.len() >= MIN_SPLIT_OBS
            {
                let (mx, my) = (mean2(x), mean2(y));
                a.extend_from_slice(&mx);
                b.extend_from_slice(&my);
                split_markers += 1;
            }
        }
    }
    MarkerReport {
        components,
        split_markers,
        split_correlation: pearson(&a, &b),
    }
}

/// Run the whole analysis.
///
/// # Errors
///
/// Fails when no camera is given or no view has enough points for the fit.
pub fn analyze(cameras: &[FloorCamera]) -> Result<FloorReport> {
    ensure!(!cameras.is_empty(), "the floor analysis needs a camera");
    let quartic: Vec<CameraFit> = cameras.iter().map(|c| fit_camera(c, 4)).collect();
    let cubic: Vec<CameraFit> = cameras.iter().map(|c| fit_camera(c, 3)).collect();
    ensure!(
        quartic.iter().any(|f| !f.views.is_empty()),
        "no view has {MIN_VIEW_POINTS} usable points for the smooth fit"
    );
    let cal: Vec<Vec<CalObs>> = cameras
        .iter()
        .zip(&quartic)
        .map(|(c, f)| calibrated_observations(c, f))
        .collect();

    let mut per_camera = Vec::with_capacity(cameras.len());
    let mut iso_total = IsotropySums::default();
    for (i, camera) in cameras.iter().enumerate() {
        let iso = isotropy_sums(camera, &quartic[i]);
        iso_total.add(&iso);
        let q = smooth_stats(&[&quartic[i]], 4);
        let obs = &cal[i];
        per_camera.push(CameraFloor {
            id: camera.id.clone(),
            points: camera.points.len(),
            cubic: smooth_stats(&[&cubic[i]], 3),
            isotropy: iso.finish(),
            calibrated: calibrated_stats(&[camera], &[obs], &q),
            cells: cell_stats(obs),
            camera_field: variance_components(
                group(obs.iter().map(|o| (o.cell, o.d))).values(),
                MIN_GROUP_OBS,
            ),
            board_field: variance_components(
                group(obs.iter().map(|o| (o.key, o.d))).values(),
                MIN_GROUP_OBS,
            ),
            view_field: variance_components(
                group(obs.iter().map(|o| (o.view, o.d))).values(),
                MIN_GROUP_OBS,
            ),
            quartic: q,
        });
    }

    let q_refs: Vec<&CameraFit> = quartic.iter().collect();
    let c_refs: Vec<&CameraFit> = cubic.iter().collect();
    let pooled_q = smooth_stats(&q_refs, 4);
    let camera_refs: Vec<&FloorCamera> = cameras.iter().collect();
    let obs_refs: Vec<&[CalObs]> = cal.iter().map(Vec::as_slice).collect();

    let board_frame: Vec<_> = cameras
        .iter()
        .zip(&quartic)
        .map(|(c, f)| board_frame_residuals(c, f))
        .collect();

    // Between-camera correlation of the per-point means.
    let means: Vec<BTreeMap<MarkerKey, [f64; 2]>> = cal
        .iter()
        .map(|obs| {
            group(obs.iter().map(|o| (o.key, o.d)))
                .into_iter()
                .filter(|(_, g)| g.len() >= MIN_GROUP_OBS)
                .map(|(k, g)| (k, mean2(&g)))
                .collect()
        })
        .collect();
    let mut pairs = Vec::new();
    for i in 0..cameras.len() {
        for j in i + 1..cameras.len() {
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let mut markers = 0;
            for (key, ma) in &means[i] {
                if let Some(mb) = means[j].get(key) {
                    a.extend_from_slice(ma);
                    b.extend_from_slice(mb);
                    markers += 1;
                }
            }
            if let Some(correlation) = pearson(&a, &b) {
                pairs.push(CameraPairCorrelation {
                    a: cameras[i].id.clone(),
                    b: cameras[j].id.clone(),
                    markers,
                    correlation,
                });
            }
        }
    }
    let mean_correlation = (!pairs.is_empty())
        .then(|| pairs.iter().map(|p| p.correlation).sum::<f64>() / pairs.len() as f64);

    let pooled = PooledFloor {
        cubic: smooth_stats(&c_refs, 3),
        markers: marker_report(&board_frame),
        isotropy: iso_total.finish(),
        calibrated: calibrated_stats(&camera_refs, &obs_refs, &pooled_q),
        camera_field: variance_components(
            group(
                cal.iter()
                    .enumerate()
                    .flat_map(|(i, o)| o.iter().map(move |o| ((i, o.cell), o.d))),
            )
            .values(),
            MIN_GROUP_OBS,
        ),
        board_field: variance_components(
            group(cal.iter().flatten().map(|o| (o.key, o.d))).values(),
            MIN_GROUP_OBS,
        ),
        view_field: variance_components(
            group(
                cal.iter()
                    .enumerate()
                    .flat_map(|(i, o)| o.iter().map(move |o| ((i, o.view), o.d))),
            )
            .values(),
            MIN_GROUP_OBS,
        ),
        board_field_pairs: pairs,
        board_field_mean_correlation: mean_correlation,
        quartic: pooled_q,
    };
    Ok(FloorReport {
        cameras: per_camera,
        pooled,
    })
}

// ───────────────────────── rendering ─────────────────────────

fn opt(v: Option<f64>) -> String {
    v.map_or_else(|| "-".into(), |v| format!("{v:.3}"))
}

fn um(mm: f64) -> f64 {
    mm * 1000.0
}

/// The human-readable report.
pub fn render_table(report: &FloorReport) -> String {
    let mut s = String::new();
    // `write!` to a `String` cannot fail.
    let mut line = |text: String| {
        s.push_str(&text);
        s.push('\n');
    };
    line("# Reprojection floor".into());
    line(String::new());
    line("## 1. Smooth-fit scatter (calibration-free, scaled for dof)".into());
    line("| camera | views | points | outl | quartic rms | median | cubic rms | median | scale px/mm |".into());
    line("|---|---:|---:|---:|---:|---:|---:|---:|---:|".into());
    let row = |id: &str, q: &SmoothFitStats, c: &SmoothFitStats| {
        format!(
            "| {id} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} |",
            q.views_used,
            q.points,
            q.outliers,
            q.rms_px,
            q.median_px,
            c.rms_px,
            c.median_px,
            q.median_scale_px_per_mm
        )
    };
    for c in &report.cameras {
        line(row(&c.id, &c.quartic, &c.cubic));
    }
    line(row("pooled", &report.pooled.quartic, &report.pooled.cubic));

    let m = &report.pooled.markers;
    line(String::new());
    line("## 2. Per-marker repeatable part (board frame, pooled over cameras)".into());
    line(format!(
        "markers {} ({} obs): total {:.1} um, within {:.1} um, repeatable {:.1} um = {:.1} % of variance",
        m.components.groups,
        m.components.observations,
        um(m.components.total_var.sqrt()),
        um(m.components.within_var.sqrt()),
        um(m.components.between_rms),
        100.0 * m.components.fraction
    ));
    line(format!(
        "split-half correlation of marker mean offsets: {} ({} markers)",
        opt(m.split_correlation),
        m.split_markers
    ));

    line(String::new());
    line("## 3. Neighbour correlation and isotropy (quartic residuals)".into());
    line("| camera | nn corr | sigma_u | sigma_v | radial | tangential |".into());
    line("|---|---:|---:|---:|---:|---:|".into());
    let iso_row = |id: &str, i: &Isotropy| {
        format!(
            "| {id} | {:+.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            i.neighbour_correlation, i.sigma_u_px, i.sigma_v_px, i.radial_px, i.tangential_px
        )
    };
    for c in &report.cameras {
        line(iso_row(&c.id, &c.isotropy));
    }
    line(iso_row("pooled", &report.pooled.isotropy));

    line(String::new());
    line("## 4. Decomposition of the calibrated residuals".into());
    line("(d) calibrated vs smooth fit, px (quartic inliers; `all` includes outliers)".into());
    line(
        "| camera | calib rms | median | all rms | all median | smooth rms | median | gap rms |"
            .into(),
    );
    line("|---|---:|---:|---:|---:|---:|---:|---:|".into());
    let cal_row = |id: &str, c: &CalibratedStats| {
        format!(
            "| {id} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            c.rms_px,
            c.median_px,
            c.rms_all_px,
            c.median_all_px,
            c.smooth_rms_px,
            c.smooth_median_px,
            c.smooth_gap_rms_px
        )
    };
    for c in &report.cameras {
        line(cal_row(&c.id, &c.calibrated));
    }
    line(cal_row("pooled", &report.pooled.calibrated));

    line(String::new());
    line("(a)-(c) noise-corrected fraction of residual variance explained".into());
    line("| camera | (a) camera-fixed cells | rms px | (b) board-fixed points | rms px | (c) per-view mean | rms px |".into());
    line("|---|---:|---:|---:|---:|---:|---:|".into());
    let frac_row =
        |id: &str, a: &VarianceComponents, b: &VarianceComponents, c: &VarianceComponents| {
            format!(
                "| {id} | {:.1} % ({}) | {:.3} | {:.1} % ({}) | {:.3} | {:.1} % ({}) | {:.3} |",
                100.0 * a.fraction,
                a.groups,
                a.between_rms,
                100.0 * b.fraction,
                b.groups,
                b.between_rms,
                100.0 * c.fraction,
                c.groups,
                c.between_rms
            )
        };
    for c in &report.cameras {
        line(frac_row(
            &c.id,
            &c.camera_field,
            &c.board_field,
            &c.view_field,
        ));
    }
    let p = &report.pooled;
    line(frac_row(
        "pooled",
        &p.camera_field,
        &p.board_field,
        &p.view_field,
    ));
    line(format!(
        "between-camera correlation of per-point means (b): mean {} over {} pairs",
        opt(p.board_field_mean_correlation),
        p.board_field_pairs.len()
    ));
    for pair in &p.board_field_pairs {
        line(format!(
            "  {} vs {}: {:+.3} ({} points)",
            pair.a, pair.b, pair.correlation, pair.markers
        ));
    }

    line(String::new());
    line("(a) mean residual per image cell, 6x4 grid: count, mean (u, v) px, std err".into());
    for c in &report.cameras {
        line(format!("{}:", c.id));
        for r in 0..CELL_GRID.1 {
            let mut cells = String::new();
            for col in 0..CELL_GRID.0 {
                match c.cells.iter().find(|x| x.row == r && x.col == col) {
                    Some(x) => {
                        let _ = write!(
                            cells,
                            " [{:>4} {:+.2} {:+.2} ±{:.2}]",
                            x.count, x.mean_px[0], x.mean_px[1], x.std_err_px
                        );
                    }
                    None => cells.push_str(" [   -                 ]"),
                }
            }
            line(cells);
        }
    }
    s
}

// ───────────────────────── registry glue ─────────────────────────

/// Run `entry`'s calibration (detection through the cache) and collect its
/// detections with the calibrated residuals.
///
/// # Errors
///
/// Fails when the entry is not a planar-intrinsics dataset, detection or the
/// calibration fails, or the run produced no residual sidecar.
#[cfg(feature = "tier-b")]
pub fn measure_entry(entry: &crate::registry::BenchEntry) -> Result<FloorCamera> {
    use anyhow::Context;
    use vision_calibration_pipeline::analysis::ReprojLevel;

    use crate::registry::ProblemKind;
    use crate::run::{run_planar_intrinsics, run_scheimpflug_intrinsics};

    ensure!(
        entry.problem == ProblemKind::PlanarIntrinsics,
        "diagnose floor supports planar-intrinsics datasets, `{}` is {:?}",
        entry.id,
        entry.problem
    );
    let record = if entry.is_scheimpflug_planar() {
        run_scheimpflug_intrinsics(entry)?
    } else {
        run_planar_intrinsics(entry)?
    };
    let sidecar = record
        .residual_sidecar
        .context("the run produced no residual sidecar")?;
    let level = sidecar
        .levels
        .iter()
        .find(|l| l.level == ReprojLevel::Intrinsic)
        .context("the residual sidecar has no intrinsic level")?;
    let spec = crate::manifest::dataset_spec(entry)?;
    let image_size = spec
        .cameras
        .first()
        .and_then(|c| c.roi_xywh)
        .map(|[_, _, w, h]| [f64::from(w), f64::from(h)]);
    let points = level
        .residuals
        .iter()
        .map(|r| FloorPoint {
            view: r.view,
            board_mm: [r.target_xyz_m[0] * 1000.0, r.target_xyz_m[1] * 1000.0],
            observed_px: r.observed_px,
            residual_px: r
                .projected_px
                .map(|p| [r.observed_px[0] - p[0], r.observed_px[1] - p[1]]),
        })
        .collect();
    Ok(FloorCamera {
        id: entry.id.clone(),
        image_size,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    const W: f64 = 1280.0;
    const H: f64 = 960.0;
    const F: f64 = 2800.0;
    const DEPTH: f64 = 800.0;

    fn gauss(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.random_range(f64::EPSILON..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    struct Scene {
        noise_px: f64,
        marker_offset_rms_mm: f64,
        field_amp_px: f64,
        cameras: usize,
        views: usize,
    }

    /// Project a board point `(x, y)` mm through a tilted board pose placed
    /// at image position `pos`, with mild radial distortion.
    fn project(
        pose: &(nalgebra::Matrix3<f64>, nalgebra::Vector3<f64>),
        x: f64,
        y: f64,
    ) -> [f64; 2] {
        let p = pose.0 * nalgebra::Vector3::new(x, y, 0.0) + pose.1;
        let (nx, ny) = (p.x / p.z, p.y / p.z);
        let k = 1.0 + (-0.04) * (nx * nx + ny * ny);
        [F * nx * k + W / 2.0, F * ny * k + H / 2.0]
    }

    fn make(scene: &Scene, seed: u64) -> Vec<FloorCamera> {
        let mut rng = StdRng::seed_from_u64(seed);
        let (cols, rows, pitch) = (14, 10, 6.0);
        let board: Vec<[f64; 2]> = (0..rows)
            .flat_map(|r| {
                (0..cols).map(move |c| {
                    [
                        (c as f64 - (cols - 1) as f64 / 2.0) * pitch,
                        (r as f64 - (rows - 1) as f64 / 2.0) * pitch,
                    ]
                })
            })
            .collect();
        // One physical board: the same offsets for every camera.
        let sigma_axis = scene.marker_offset_rms_mm / 2f64.sqrt();
        let offsets: Vec<[f64; 2]> = board
            .iter()
            .map(|_| [sigma_axis * gauss(&mut rng), sigma_axis * gauss(&mut rng)])
            .collect();
        let field = |u: f64, v: f64| -> [f64; 2] {
            let a = scene.field_amp_px;
            [
                a * (std::f64::consts::TAU * u / W + 0.3).sin()
                    * (std::f64::consts::TAU * v / H).cos(),
                a * (std::f64::consts::TAU * u / W).cos()
                    * (std::f64::consts::TAU * v / H + 1.0).sin(),
            ]
        };
        (0..scene.cameras)
            .map(|cam| {
                let mut points = Vec::new();
                for view in 0..scene.views {
                    let angle = |rng: &mut StdRng, a: f64| rng.random_range(-a..a);
                    let (ax, ay, az) = (
                        angle(&mut rng, 0.3),
                        angle(&mut rng, 0.3),
                        angle(&mut rng, 0.5),
                    );
                    let r = nalgebra::Rotation3::from_euler_angles(ax, ay, az).into_inner();
                    let pos = [
                        rng.random_range(0.15 * W..0.85 * W),
                        rng.random_range(0.15 * H..0.85 * H),
                    ];
                    let t = nalgebra::Vector3::new(
                        (pos[0] - W / 2.0) / F * DEPTH,
                        (pos[1] - H / 2.0) / F * DEPTH,
                        DEPTH,
                    );
                    let pose = (r, t);
                    for (b, o) in board.iter().zip(&offsets) {
                        let ideal = project(&pose, b[0], b[1]);
                        let real = project(&pose, b[0] + o[0], b[1] + o[1]);
                        let fld = field(ideal[0], ideal[1]);
                        let observed = [
                            real[0] + fld[0] + scene.noise_px * gauss(&mut rng),
                            real[1] + fld[1] + scene.noise_px * gauss(&mut rng),
                        ];
                        points.push(FloorPoint {
                            view,
                            board_mm: *b,
                            observed_px: observed,
                            residual_px: Some([observed[0] - ideal[0], observed[1] - ideal[1]]),
                        });
                    }
                }
                FloorCamera {
                    id: format!("cam{cam}"),
                    image_size: Some([W, H]),
                    points,
                }
            })
            .collect()
    }

    fn assert_within(value: f64, truth: f64, rel: f64, what: &str) {
        assert!(
            (value - truth).abs() <= rel * truth,
            "{what}: {value} vs {truth} (±{:.0} %)",
            100.0 * rel
        );
    }

    #[test]
    fn white_noise_is_recovered_and_nothing_repeats() {
        let sigma = 0.15;
        let report = analyze(&make(
            &Scene {
                noise_px: sigma,
                marker_offset_rms_mm: 0.0,
                field_amp_px: 0.0,
                cameras: 2,
                views: 12,
            },
            7,
        ))
        .unwrap();
        let p = &report.pooled;
        assert_within(p.isotropy.sigma_u_px, sigma, 0.1, "sigma_u");
        assert_within(p.isotropy.sigma_v_px, sigma, 0.1, "sigma_v");
        assert!(
            p.cubic.rms_px >= p.quartic.rms_px,
            "a lower degree cannot fit better"
        );
        // Least-squares residuals of white noise are slightly anti-correlated
        // between neighbours (the hat matrix couples them): about -H_ij.
        let nn = p.isotropy.neighbour_correlation;
        assert!((-0.25..0.05).contains(&nn), "neighbour correlation {nn}");
        assert!(p.markers.components.fraction.abs() < 0.03);
        assert!(p.camera_field.fraction.abs() < 0.05);
        assert!(p.view_field.fraction.abs() < 0.05);
        let scale = p.quartic.median_scale_px_per_mm;
        assert!((scale - F / DEPTH).abs() < 0.2, "scale {scale}");
    }

    #[test]
    fn marker_offsets_are_recovered_per_marker() {
        let rho = 0.03;
        let report = analyze(&make(
            &Scene {
                noise_px: 0.15,
                marker_offset_rms_mm: rho,
                field_amp_px: 0.0,
                cameras: 2,
                views: 12,
            },
            11,
        ))
        .unwrap();
        let p = &report.pooled;
        assert_within(p.markers.components.between_rms, rho, 0.2, "marker rms");
        assert!(p.markers.split_correlation.unwrap() > 0.5);
        // The offsets are fixed on the board, so they appear as a board-fixed
        // field of the calibrated residuals and not as a camera-fixed one.
        assert!(p.board_field.fraction > 0.1, "{}", p.board_field.fraction);
        assert!(p.board_field_mean_correlation.unwrap() > 0.3);
    }

    #[test]
    fn camera_fixed_field_is_not_a_marker_effect() {
        let report = analyze(&make(
            &Scene {
                noise_px: 0.05,
                marker_offset_rms_mm: 0.0,
                field_amp_px: 0.4,
                cameras: 1,
                views: 30,
            },
            13,
        ))
        .unwrap();
        let c = &report.cameras[0];
        assert!(c.camera_field.fraction > 0.4, "{}", c.camera_field.fraction);
        assert!(
            c.board_field.fraction < 0.5 * c.camera_field.fraction,
            "board {} camera {}",
            c.board_field.fraction,
            c.camera_field.fraction
        );
        // The per-view polynomial absorbs the smooth field: it must not show
        // up as per-marker scatter, and the smooth-fit floor stays at noise.
        assert!(report.pooled.markers.components.fraction.abs() < 0.03);
        assert!(c.isotropy.sigma_u_px < 0.06, "{}", c.isotropy.sigma_u_px);
        assert!(c.calibrated.smooth_gap_rms_px > 0.2);
    }

    #[test]
    fn report_round_trips_through_json() {
        let report = analyze(&make(
            &Scene {
                noise_px: 0.1,
                marker_offset_rms_mm: 0.0,
                field_amp_px: 0.0,
                cameras: 1,
                views: 3,
            },
            3,
        ))
        .unwrap();
        let text = serde_json::to_string(&report).unwrap();
        let back: FloorReport = serde_json::from_str(&text).unwrap();
        assert_eq!(report, back);
        assert!(render_table(&report).contains("## 4."));
    }

    #[test]
    fn views_with_too_few_points_are_skipped() {
        let mut cameras = make(
            &Scene {
                noise_px: 0.1,
                marker_offset_rms_mm: 0.0,
                field_amp_px: 0.0,
                cameras: 1,
                views: 2,
            },
            5,
        );
        cameras[0]
            .points
            .retain(|p| p.view == 0 || p.board_mm[0] < -30.0);
        let report = analyze(&cameras).unwrap();
        assert_eq!(report.cameras[0].quartic.views_used, 1);
        assert_eq!(report.cameras[0].quartic.views_skipped, 1);
        assert!(analyze(&[]).is_err());
    }
}
