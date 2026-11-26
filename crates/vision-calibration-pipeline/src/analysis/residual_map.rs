//! The systematic part of a camera's reprojection residuals over its image:
//! [`residual_map`].

use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use serde::{Deserialize, Serialize};

use vision_calibration_core::{ImageGrid, TargetFeatureResidual};

use crate::Error;

/// The Rayleigh median of an isotropic 2-D Gaussian, `√(2 ln 2)`: the
/// median of `|e|` over `σ`.
const RAYLEIGH_MEDIAN: f64 = 1.177_410_022_515_474_7;

/// The kernel is truncated at this many bandwidths.
const TRUNCATION: f64 = 4.0;

/// The largest condition number of the local design `Σ w z zᵀ` with a
/// value.
const MAX_CONDITION: f64 = 1e8;

/// The cross-validation ladder, fractions of the image diagonal.
const LADDER: [f64; 6] = [0.03, 0.045, 0.07, 0.10, 0.15, 0.22];

/// The bandwidth, as a fraction of the diagonal, with too few views to
/// cross-validate.
const FALLBACK_FRACTION: f64 = 0.10;

/// The fewest views cross-validation takes.
const MIN_CV_VIEWS: usize = 4;

/// The most folds cross-validation splits the views into.
const MAX_FOLDS: usize = 5;

/// How [`residual_map`] chooses the kernel's bandwidth.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Bandwidth {
    /// K-fold cross-validation over views on a ladder of fractions of the
    /// image diagonal (see [`residual_map`]).
    #[default]
    Auto,
    /// The Gaussian kernel's σ, pixels (finite, > 0).
    Pixels(f64),
}

/// Options of [`residual_map`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ResidualMapOptions {
    /// The kernel's bandwidth. Default [`Bandwidth::Auto`]: a field's scale
    /// differs from capture to capture, and cross-validation over views
    /// finds it without fitting each view's pose error as a field.
    pub bandwidth: Bandwidth,
    /// The views to map, by their index in the input; `None` (the default)
    /// maps all. With one view (or fewer effective views than
    /// [`min_effective_views`](Self::min_effective_views)) the mean includes
    /// that view's pose error and has no significance.
    pub views: Option<Vec<usize>>,
    /// Records with `|e|` above this many robust σ (`median |e| / 1.1774`,
    /// the Rayleigh median) are left out as outliers; finite, > 0. Default
    /// 8, a tail of `exp(−32)` under Gaussian noise: a genuine systematic
    /// field at the corners can reach several σ, and a global threshold must
    /// not clip the signal the map is meant to show.
    pub outlier_factor: f64,
    /// The fewest effective samples `1/Σ l_k²` a node needs for a value;
    /// finite, ≥ 1. Default 8: below it the local fit is mostly one or two
    /// records' noise.
    pub min_effective_samples: f64,
    /// The fewest effective views a node needs for a standard error and a
    /// significance; finite, ≥ 2. Default 5: the view-clustered standard
    /// error is estimated from that many independent captures, and with
    /// fewer it understates the scatter.
    pub min_effective_views: f64,
}

impl Default for ResidualMapOptions {
    fn default() -> Self {
        Self {
            bandwidth: Bandwidth::Auto,
            views: None,
            outlier_factor: 8.0,
            min_effective_samples: 8.0,
            min_effective_views: 5.0,
        }
    }
}

impl ResidualMapOptions {
    /// The default options.
    pub fn new() -> Self {
        Self::default()
    }

    fn validate(&self) -> Result<(), Error> {
        let bad = |name: &str, value: f64, range: &str| {
            Err(Error::invalid_input(format!(
                "residual map {name} must be {range}, got {value}"
            )))
        };
        if let Bandwidth::Pixels(h) = self.bandwidth
            && !(h.is_finite() && h > 0.0)
        {
            return bad("bandwidth", h, "finite and positive");
        }
        if !(self.outlier_factor.is_finite() && self.outlier_factor > 0.0) {
            return bad("outlier_factor", self.outlier_factor, "finite and positive");
        }
        if !(self.min_effective_samples.is_finite() && self.min_effective_samples >= 1.0) {
            return bad(
                "min_effective_samples",
                self.min_effective_samples,
                "finite and at least 1",
            );
        }
        if !(self.min_effective_views.is_finite() && self.min_effective_views >= 2.0) {
            return bad(
                "min_effective_views",
                self.min_effective_views,
                "finite and at least 2",
            );
        }
        Ok(())
    }
}

/// How [`residual_map`] chose its bandwidth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum BandwidthSelection {
    /// The options gave it ([`Bandwidth::Pixels`]).
    Given,
    /// Cross-validated over views ([`Bandwidth::Auto`]).
    CrossValidated {
        /// The number of folds.
        folds: usize,
        /// The scored ladder entries, pixels, ascending. Empty when no
        /// entry predicted any held-out record; the bandwidth is then
        /// `0.1·diagonal`.
        ladder_px: Vec<f64>,
        /// The mean squared held-out error at each scored entry, px².
        scores_px2: Vec<f64>,
        /// The standard error of each score, from its per-fold scores, px².
        score_std_errors_px2: Vec<f64>,
    },
    /// Fewer than four views to cross-validate: `0.1·diagonal`.
    FewViews,
}

/// The systematic part of one camera's reprojection residuals over its
/// image (see [`residual_map`] for the estimator and how to read it).
///
/// Every plane is row-major over [`grid`](Self::grid), one value per node,
/// `NaN` where the node has none.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ResidualMap {
    /// The nodes the planes are sampled at.
    pub grid: ImageGrid,
    /// The camera mapped.
    pub camera: usize,
    /// The Gaussian kernel's σ, pixels.
    pub bandwidth_px: f64,
    /// How the bandwidth was chosen.
    pub bandwidth_selection: BandwidthSelection,
    /// The local-linear mean of `observed − projected`, `x`, pixels; `NaN`
    /// where the support is too thin.
    pub mean_dx: Vec<f64>,
    /// The local-linear mean of `observed − projected`, `y`, pixels.
    pub mean_dy: Vec<f64>,
    /// The kernel-weighted RMS of `|observed − projected|`, pixels.
    pub rms: Vec<f64>,
    /// The major-axis standard error of the mean, view-clustered, pixels;
    /// `NaN` with fewer effective views than the options ask.
    pub std_error: Vec<f64>,
    /// `χ = √(μ̂ᵀ V̂⁻¹ μ̂)`, pointwise; `NaN` as `std_error`, and where the
    /// scatter is zero.
    pub significance: Vec<f64>,
    /// The effective number of samples `1/Σ l_k²`; 0 where the node has no
    /// weight or a collinear support. Always filled, for a support contour.
    pub effective_samples: Vec<f64>,
    /// The effective number of views; 0 where `effective_samples` is.
    pub effective_views: Vec<f64>,
    /// The records mapped.
    pub used: usize,
    /// The selected records without a projection, left out.
    pub unprojected: usize,
    /// The records left out as outliers, as indices into the input slice,
    /// ascending.
    pub outliers: Vec<usize>,
    /// The number of views among the records mapped.
    pub views: usize,
}

/// The systematic part of `camera`'s reprojection residuals at the nodes of
/// `grid`.
///
/// `residuals` are the per-feature records of a result
/// (`per_feature_residuals.target`); the records of other cameras are
/// ignored. `grid` spans the camera's image in its own pixels (a region of
/// interest's size for a tiled camera). The map is deterministic: the same
/// input gives bit-identical planes.
///
/// # Estimand
///
/// At image position `x`, `μ(x) = E[observed − projected | projected = x]`
/// over the views of the capture: the misfit that stays fixed in the image
/// while the target moves, in pixels, with the sign of
/// [`TargetFeatureResidual::residual_px`]. A lens distortion the model lacks
/// shows in it with the sign and units it has in a
/// [`distortion_map`](vision_calibration_core::distortion_map). The map also
/// carries the local RMS `ρ(x) = √E[|e|² | x]`.
///
/// Records are placed at their **projection**, never at their observation:
/// binning by the observation correlates position with the noise being
/// averaged.
///
/// # Estimator
///
/// At each grid node `x₀`, a Gaussian-kernel local-linear fit: weights
/// `w_k = exp(−|p_k − x₀|²/2h²)` (zero beyond `4h`), the design
/// `z_k = [1, (p_k − x₀)/h]`, and `μ̂(x₀)` the intercept of the weighted
/// least-squares plane through the residuals. It is linear in the
/// residuals, `μ̂ = Σ l_k e_k` with `Σ l_k = 1`. Unlike a kernel or binned
/// average, its bias at the edge of the coverage does not grow with the
/// field's slope (`O(h²)` rather than `O(h)`), and the edges are where a
/// distortion misfit is largest.
///
/// A node has a mean when its support is wide enough: the effective number
/// of samples `n_eff = 1/Σ l_k²` is at least
/// [`min_effective_samples`](ResidualMapOptions::min_effective_samples) and
/// the local design is not collinear (condition number of `Σ w z zᵀ` at most
/// `1e8`, as along a single board edge). Extrapolating past the coverage
/// makes `Σ l_k²` large, so the same rule ends the map there.
///
/// # Standard error and significance
///
/// Residuals of one view share its pose error, so they are correlated;
/// views are the independent unit. The covariance of `μ̂` is the
/// view-clustered sandwich `V̂ = G/(G − 1)·Σ_v s_v s_vᵀ`, `s_v` the sum of
/// `l_k ê_k` over the view's records (`ê` the local fit's residuals) and `G`
/// the effective number of views `(Σ_v a_v)²/Σ_v a_v²`, `a_v = Σ_{k∈v}|l_k|`.
/// It measures the scatter, so it needs no noise model and is not affected
/// by a stated feature covariance that is too small. `std_error` is the
/// major-axis σ, `significance` is `χ = √(μ̂ᵀ V̂⁻¹ μ̂)`, pointwise
/// `p = exp(−χ²/2)` under a zero field (`χ > 3.03` is `p < 0.01`). Both need
/// `G ≥` [`min_effective_views`](ResidualMapOptions::min_effective_views).
///
/// # Bandwidth
///
/// [`Bandwidth::Auto`] chooses the kernel's `h` by K-fold cross-validation
/// over views on a ladder of fractions of the image diagonal (3, 4.5, 7, 10,
/// 15 and 22 %): views sorted by index go to fold `rank mod K`,
/// `K = min(5, views)`. For each fold and `h`, the map of the other folds is
/// fitted on a lattice of spacing `h/2` and interpolated bilinearly at the
/// held-out records; the score is the mean `|e − μ̂|²` over the held-out
/// records predicted at every scored `h` (one evaluation set for all). A
/// ladder entry is scored when it predicts at least half as many held-out
/// records as the entry that predicts most (a small `h` covers nothing in a
/// sparse capture). The choice is the largest `h` within one standard error
/// (of the fold scores) of the best: with no field the scores are flat and
/// the smoothest map wins. Fewer than four views take `h = 0.1·diagonal`.
///
/// # Reading the map
///
/// - **A significant field is not a reason to add a model term.** It is the
///   average over this capture's depths, defocus and view geometry, and much
///   of such a field is commonly depth- or defocus-coupled rather than fixed
///   to the camera.
/// - **A flat map does not prove the model.** At the solution the residuals
///   are orthogonal to every free parameter's direction (camera and poses):
///   a missing term the parameters can partly mimic is absorbed into biased
///   parameters and does not show.
/// - **It is not a correction.** Subtracting it from the observations would
///   evaluate a term at the observed pixel, fitted in-sample.
/// - **Chain-level residuals** (hand-eye and rig results, whose board poses
///   come through the robot or rig chain) contain pose-chain error with a
///   smooth per-view pattern. For a camera-model diagnosis, map residuals of
///   a fit with a free pose per view.
/// - Significance is pointwise, and nodes within about `2h` are strongly
///   correlated: count regions (about image area / `πh²`), not nodes. It
///   assumes independent views, which repeated poses or a drifting session
///   are not.
///
/// # Errors
///
/// [`Error::InvalidInput`] for an option out of range, no record of
/// `camera`, or a `views` entry that no record of `camera` has.
///
/// # Example
///
/// ```
/// use vision_calibration_core::{ImageGrid, TargetFeatureResidual};
/// use vision_calibration_pipeline::analysis::{Bandwidth, ResidualMapOptions, residual_map};
///
/// // Records whose observations sit half a pixel right of their projections.
/// let mut records = Vec::new();
/// for view in 0..6 {
///     for i in 0..400 {
///         let mut r = TargetFeatureResidual::default();
///         r.view = view;
///         let p = [(i % 20) as f64 * 32.0 + 16.0, (i / 20) as f64 * 24.0 + 12.0];
///         r.projected_px = Some(p);
///         r.observed_px = [p[0] + 0.5, p[1]];
///         records.push(r);
///     }
/// }
/// let grid = ImageGrid::with_columns([640, 480], 16)?;
/// let mut options = ResidualMapOptions::new();
/// options.bandwidth = Bandwidth::Pixels(40.0);
/// let map = residual_map(&records, 0, &grid, &options)?;
/// assert!((map.mean_dx[grid.len() / 2] - 0.5).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn residual_map(
    residuals: &[TargetFeatureResidual],
    camera: usize,
    grid: &ImageGrid,
    options: &ResidualMapOptions,
) -> Result<ResidualMap, Error> {
    options.validate()?;
    let selected = select(residuals, camera, options.views.as_deref())?;
    let Selected {
        mut samples,
        unprojected,
    } = selected;
    let outliers = remove_outliers(&mut samples, options.outlier_factor);
    let view_ids = renumber_views(&mut samples);

    let diagonal = f64::from(grid.image_size()[0]).hypot(f64::from(grid.image_size()[1]));
    let (bandwidth_px, bandwidth_selection) = match options.bandwidth {
        Bandwidth::Pixels(h) => (h, BandwidthSelection::Given),
        Bandwidth::Auto if view_ids.len() < MIN_CV_VIEWS => {
            (FALLBACK_FRACTION * diagonal, BandwidthSelection::FewViews)
        }
        Bandwidth::Auto => cross_validate(&samples, view_ids.len(), grid, diagonal, options),
    };

    let smoother = Smoother::new(&samples, bandwidth_px);
    let mut scratch = Scratch::new(view_ids.len());
    let len = grid.len();
    let mut map = ResidualMap {
        grid: *grid,
        camera,
        bandwidth_px,
        bandwidth_selection,
        mean_dx: vec![f64::NAN; len],
        mean_dy: vec![f64::NAN; len],
        rms: vec![f64::NAN; len],
        std_error: vec![f64::NAN; len],
        significance: vec![f64::NAN; len],
        effective_samples: vec![0.0; len],
        effective_views: vec![0.0; len],
        used: samples.len(),
        unprojected,
        outliers,
        views: view_ids.len(),
    };
    for (i, node) in grid.nodes().enumerate() {
        let fit = smoother.fit(node, &mut scratch, Clusters::Views);
        map.effective_samples[i] = fit.effective_samples;
        map.effective_views[i] = fit.effective_views;
        if !fit.supported(options.min_effective_samples) {
            continue;
        }
        map.mean_dx[i] = fit.mean[0];
        map.mean_dy[i] = fit.mean[1];
        map.rms[i] = fit.rms;
        if fit.effective_views >= options.min_effective_views {
            map.std_error[i] = fit.std_error;
            map.significance[i] = fit.significance;
        }
    }
    Ok(map)
}

/// One record as the smoother sees it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    /// Index of the record in the input slice.
    index: usize,
    /// The projection, pixels.
    p: [f64; 2],
    /// `observed − projected`, pixels.
    e: [f64; 2],
    /// The view: its input index until [`renumber_views`], then its rank.
    view: usize,
}

struct Selected {
    samples: Vec<Sample>,
    unprojected: usize,
}

/// The records of `camera` (and of `views`, when given) with a finite
/// projection and residual.
fn select(
    residuals: &[TargetFeatureResidual],
    camera: usize,
    views: Option<&[usize]>,
) -> Result<Selected, Error> {
    let mut camera_views: Vec<usize> = residuals
        .iter()
        .filter(|r| r.camera == camera)
        .map(|r| r.view)
        .collect();
    if camera_views.is_empty() {
        return Err(Error::invalid_input(format!(
            "residual map: no record of camera {camera}"
        )));
    }
    camera_views.sort_unstable();
    camera_views.dedup();
    let mut wanted = views.map(<[usize]>::to_vec);
    if let Some(wanted) = &mut wanted {
        if wanted.is_empty() {
            return Err(Error::invalid_input(
                "residual map: `views` selects no view",
            ));
        }
        wanted.sort_unstable();
        wanted.dedup();
        if let Some(missing) = wanted
            .iter()
            .find(|v| camera_views.binary_search(v).is_err())
        {
            return Err(Error::invalid_input(format!(
                "residual map: no record of camera {camera} in view {missing}"
            )));
        }
    }
    let mut samples = Vec::new();
    let mut unprojected = 0;
    for (index, r) in residuals.iter().enumerate() {
        if r.camera != camera
            || wanted
                .as_ref()
                .is_some_and(|w| w.binary_search(&r.view).is_err())
        {
            continue;
        }
        match (r.projected_px, r.residual_px()) {
            (Some(p), Some(e)) if p.iter().chain(&e).all(|v| v.is_finite()) => {
                samples.push(Sample {
                    index,
                    p,
                    e,
                    view: r.view,
                });
            }
            _ => unprojected += 1,
        }
    }
    Ok(Selected {
        samples,
        unprojected,
    })
}

/// Remove the records with `|e| > factor · median|e| / 1.1774` and return
/// their input indices, ascending. Nothing is removed when the median is
/// zero.
fn remove_outliers(samples: &mut Vec<Sample>, factor: f64) -> Vec<usize> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut norms: Vec<f64> = samples.iter().map(|s| s.e[0].hypot(s.e[1])).collect();
    let mid = norms.len() / 2;
    let (_, &mut upper, _) = norms.select_nth_unstable_by(mid, f64::total_cmp);
    let median = if norms.len() % 2 == 1 {
        upper
    } else {
        let lower = norms[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lower + upper)
    };
    let sigma = median / RAYLEIGH_MEDIAN;
    if sigma <= 0.0 {
        return Vec::new();
    }
    let limit = factor * sigma;
    let mut outliers = Vec::new();
    samples.retain(|s| {
        let keep = s.e[0].hypot(s.e[1]) <= limit;
        if !keep {
            outliers.push(s.index);
        }
        keep
    });
    outliers
}

/// Replace each sample's view by its rank among the distinct views, and
/// return the distinct views ascending.
fn renumber_views(samples: &mut [Sample]) -> Vec<usize> {
    let mut ids: Vec<usize> = samples.iter().map(|s| s.view).collect();
    ids.sort_unstable();
    ids.dedup();
    for s in samples.iter_mut() {
        // The id is in `ids`, which was built from the samples.
        s.view = ids.binary_search(&s.view).unwrap_or(0);
    }
    ids
}

/// Choose the bandwidth by K-fold cross-validation over views (see
/// [`residual_map`]).
fn cross_validate(
    samples: &[Sample],
    views: usize,
    grid: &ImageGrid,
    diagonal: f64,
    options: &ResidualMapOptions,
) -> (f64, BandwidthSelection) {
    let folds = views.min(MAX_FOLDS);
    let ladder: Vec<f64> = LADDER.iter().map(|f| f * diagonal).collect();
    // predictions[j][k]: the prediction at held-out sample k for ladder entry j.
    let mut predictions = vec![vec![None; samples.len()]; ladder.len()];
    for fold in 0..folds {
        let training: Vec<Sample> = samples
            .iter()
            .filter(|s| s.view % folds != fold)
            .copied()
            .collect();
        let held: Vec<usize> = (0..samples.len())
            .filter(|&k| samples[k].view % folds == fold)
            .collect();
        for (j, &h) in ladder.iter().enumerate() {
            let lattice = Lattice::fit(&training, h, grid, options.min_effective_samples);
            for &k in &held {
                predictions[j][k] = lattice.interpolate(samples[k].p);
            }
        }
    }
    let coverage: Vec<usize> = predictions
        .iter()
        .map(|p| p.iter().filter(|v| v.is_some()).count())
        .collect();
    let most = coverage.iter().copied().max().unwrap_or(0);
    let scored: Vec<usize> = (0..ladder.len())
        .filter(|&j| most > 0 && 2 * coverage[j] >= most)
        .collect();
    let evaluated: Vec<usize> = (0..samples.len())
        .filter(|&k| !scored.is_empty() && scored.iter().all(|&j| predictions[j][k].is_some()))
        .collect();
    if evaluated.is_empty() {
        return (
            FALLBACK_FRACTION * diagonal,
            BandwidthSelection::CrossValidated {
                folds,
                ladder_px: Vec::new(),
                scores_px2: Vec::new(),
                score_std_errors_px2: Vec::new(),
            },
        );
    }
    let mut scores = Vec::with_capacity(scored.len());
    let mut std_errors = Vec::with_capacity(scored.len());
    for &j in &scored {
        let mut fold_sum = vec![0.0; folds];
        let mut fold_count = vec![0usize; folds];
        let mut sum = 0.0;
        for &k in &evaluated {
            let [mx, my] = predictions[j][k].unwrap_or([f64::NAN; 2]);
            let s = &samples[k];
            let err = (s.e[0] - mx).powi(2) + (s.e[1] - my).powi(2);
            sum += err;
            fold_sum[s.view % folds] += err;
            fold_count[s.view % folds] += 1;
        }
        scores.push(sum / evaluated.len() as f64);
        let fold_scores: Vec<f64> = fold_sum
            .iter()
            .zip(&fold_count)
            .filter(|(_, n)| **n > 0)
            .map(|(s, n)| s / *n as f64)
            .collect();
        std_errors.push(standard_error(&fold_scores));
    }
    // The best score (the larger h on a tie), then the largest h within one
    // standard error of it.
    let best = (0..scores.len())
        .rev()
        .min_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap_or(0);
    let limit = scores[best] + std_errors[best];
    let chosen = (0..scores.len())
        .rev()
        .find(|&j| scores[j] <= limit)
        .unwrap_or(best);
    let ladder_px: Vec<f64> = scored.iter().map(|&j| ladder[j]).collect();
    (
        ladder_px[chosen],
        BandwidthSelection::CrossValidated {
            folds,
            ladder_px,
            scores_px2: scores,
            score_std_errors_px2: std_errors,
        },
    )
}

/// The standard error of the mean of `values`; 0 with fewer than two.
fn standard_error(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (var / n as f64).sqrt()
}

/// A local-linear mean fitted on a lattice of spacing `h/2` over the image,
/// for cross-validation.
struct Lattice {
    spacing: f64,
    columns: usize,
    rows: usize,
    /// Row-major means; `None` where the support is too thin.
    means: Vec<Option<[f64; 2]>>,
}

impl Lattice {
    /// Nodes at `−0.5 + i·spacing` along each axis, spanning the image's
    /// pixel area `[−0.5, size − 0.5]`.
    fn fit(samples: &[Sample], h: f64, grid: &ImageGrid, min_samples: f64) -> Self {
        let spacing = 0.5 * h;
        let [w, hgt] = grid.image_size();
        let columns = (f64::from(w) / spacing).ceil() as usize + 1;
        let rows = (f64::from(hgt) / spacing).ceil() as usize + 1;
        let smoother = Smoother::new(samples, h);
        let mut scratch = Scratch::new(0);
        let mut means = Vec::with_capacity(columns * rows);
        for r in 0..rows {
            for c in 0..columns {
                let node = [-0.5 + c as f64 * spacing, -0.5 + r as f64 * spacing];
                let fit = smoother.fit(node, &mut scratch, Clusters::None);
                means.push(fit.supported(min_samples).then_some(fit.mean));
            }
        }
        Self {
            spacing,
            columns,
            rows,
            means,
        }
    }

    /// The bilinear interpolation of the means at `p`; `None` off the
    /// lattice or next to a node without a mean.
    fn interpolate(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let fx = (p[0] + 0.5) / self.spacing;
        let fy = (p[1] + 0.5) / self.spacing;
        if !(fx >= 0.0 && fy >= 0.0) {
            return None;
        }
        let (c, r) = (fx.floor() as usize, fy.floor() as usize);
        let (c, r) = (c.min(self.columns - 2), r.min(self.rows - 2));
        let (tx, ty) = (fx - c as f64, fy - r as f64);
        if tx > 1.0 || ty > 1.0 {
            return None;
        }
        let at = |c: usize, r: usize| self.means[r * self.columns + c];
        let (a, b, d, e) = (at(c, r)?, at(c + 1, r)?, at(c, r + 1)?, at(c + 1, r + 1)?);
        let lerp = |i: usize| {
            (1.0 - ty) * ((1.0 - tx) * a[i] + tx * b[i]) + ty * ((1.0 - tx) * d[i] + tx * e[i])
        };
        Some([lerp(0), lerp(1)])
    }
}

/// What the standard error clusters the records by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Clusters {
    /// No standard error (cross-validation needs the mean only).
    None,
    /// One cluster per view: the estimator's.
    Views,
    /// One cluster per record, as if the records of a view were
    /// independent: the tests show it understates the scatter.
    #[cfg(test)]
    Records,
}

/// The local fit at one node.
#[derive(Debug, Clone, Copy)]
pub(super) struct NodeFit {
    pub(super) mean: [f64; 2],
    pub(super) rms: f64,
    pub(super) std_error: f64,
    pub(super) significance: f64,
    pub(super) effective_samples: f64,
    pub(super) effective_views: f64,
    /// Whether the local design is well conditioned.
    conditioned: bool,
}

impl NodeFit {
    fn empty() -> Self {
        Self {
            mean: [f64::NAN; 2],
            rms: f64::NAN,
            std_error: f64::NAN,
            significance: f64::NAN,
            effective_samples: 0.0,
            effective_views: 0.0,
            conditioned: false,
        }
    }

    /// Whether the node has a mean: a conditioned design and enough
    /// effective samples.
    pub(super) fn supported(&self, min_samples: f64) -> bool {
        self.conditioned && self.effective_samples >= min_samples
    }
}

/// Per-call buffers of [`Smoother::fit`], reused across nodes.
pub(super) struct Scratch {
    /// The neighbours of the node: sample index, weight, scaled offsets.
    neighbours: Vec<(usize, f64, f64, f64)>,
    /// Per view: `Σ|l_k|` and `Σ l_k ê_k`.
    view_abs: Vec<f64>,
    view_sum: Vec<[f64; 2]>,
    /// Whether the view has a record among the neighbours, and those views.
    seen: Vec<bool>,
    touched: Vec<usize>,
}

impl Scratch {
    pub(super) fn new(views: usize) -> Self {
        Self {
            neighbours: Vec::new(),
            view_abs: vec![0.0; views],
            view_sum: vec![[0.0; 2]; views],
            seen: vec![false; views],
            touched: Vec::new(),
        }
    }
}

/// The Gaussian local-linear smoother over a set of samples, with a uniform
/// bin index of cell `4h` for the neighbour search.
pub(super) struct Smoother<'a> {
    samples: &'a [Sample],
    h: f64,
    origin: [f64; 2],
    cell: f64,
    columns: usize,
    rows: usize,
    /// `starts[b]..starts[b + 1]` indexes `order` for bin `b`.
    starts: Vec<usize>,
    /// Sample indices, by bin, in input order within a bin.
    order: Vec<usize>,
}

impl<'a> Smoother<'a> {
    /// The most bins of the index; a wider spread of projections gets
    /// coarser bins.
    const MAX_BINS: f64 = 1e6;

    pub(super) fn new(samples: &'a [Sample], h: f64) -> Self {
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for s in samples {
            for a in 0..2 {
                lo[a] = lo[a].min(s.p[a]);
                hi[a] = hi[a].max(s.p[a]);
            }
        }
        if samples.is_empty() {
            lo = [0.0; 2];
            hi = [0.0; 2];
        }
        let span = [hi[0] - lo[0], hi[1] - lo[1]];
        let mut cell = TRUNCATION * h;
        let bins = |cell: f64| (span[0] / cell + 1.0) * (span[1] / cell + 1.0);
        if bins(cell) > Self::MAX_BINS {
            cell *= (bins(cell) / Self::MAX_BINS).sqrt();
        }
        let columns = (span[0] / cell).floor() as usize + 1;
        let rows = (span[1] / cell).floor() as usize + 1;
        let bin_of = |p: [f64; 2]| {
            let c = (((p[0] - lo[0]) / cell).floor() as usize).min(columns - 1);
            let r = (((p[1] - lo[1]) / cell).floor() as usize).min(rows - 1);
            r * columns + c
        };
        // Counting sort: stable, so samples stay in input order within a bin.
        let mut starts = vec![0usize; columns * rows + 1];
        for s in samples {
            starts[bin_of(s.p) + 1] += 1;
        }
        for b in 0..columns * rows {
            starts[b + 1] += starts[b];
        }
        let mut next = starts.clone();
        let mut order = vec![0usize; samples.len()];
        for (k, s) in samples.iter().enumerate() {
            let b = bin_of(s.p);
            order[next[b]] = k;
            next[b] += 1;
        }
        Self {
            samples,
            h,
            origin: lo,
            cell,
            columns,
            rows,
            starts,
            order,
        }
    }

    /// The local-linear fit at `x0` (see [`residual_map`]).
    pub(super) fn fit(&self, x0: [f64; 2], scratch: &mut Scratch, clusters: Clusters) -> NodeFit {
        let h = self.h;
        let radius = TRUNCATION * h;
        let radius2 = radius * radius;
        let inv_2h2 = 0.5 / (h * h);
        scratch.neighbours.clear();
        let bin = |v: f64, a: usize, n: usize| {
            ((v - self.origin[a]) / self.cell)
                .floor()
                .clamp(0.0, (n - 1) as f64) as usize
        };
        if x0[0] + radius < self.origin[0]
            || x0[1] + radius < self.origin[1]
            || self.samples.is_empty()
        {
            return NodeFit::empty();
        }
        let (c0, c1) = (
            bin(x0[0] - radius, 0, self.columns),
            bin(x0[0] + radius, 0, self.columns),
        );
        let (r0, r1) = (
            bin(x0[1] - radius, 1, self.rows),
            bin(x0[1] + radius, 1, self.rows),
        );
        // Σ w z zᵀ (symmetric), Σ w z eᵀ, Σ w, Σ w |e|².
        let mut m = Matrix3::<f64>::zeros();
        let mut b = [[0.0f64; 2]; 3];
        let mut sum_w = 0.0;
        let mut sum_we2 = 0.0;
        for r in r0..=r1 {
            for c in c0..=c1 {
                let bin = r * self.columns + c;
                for &k in &self.order[self.starts[bin]..self.starts[bin + 1]] {
                    let s = &self.samples[k];
                    let (dx, dy) = (s.p[0] - x0[0], s.p[1] - x0[1]);
                    let d2 = dx * dx + dy * dy;
                    if d2 > radius2 {
                        continue;
                    }
                    let w = (-d2 * inv_2h2).exp();
                    let z = [1.0, dx / h, dy / h];
                    for i in 0..3 {
                        for j in i..3 {
                            m[(i, j)] += w * z[i] * z[j];
                        }
                        b[i][0] += w * z[i] * s.e[0];
                        b[i][1] += w * z[i] * s.e[1];
                    }
                    sum_w += w;
                    sum_we2 += w * (s.e[0] * s.e[0] + s.e[1] * s.e[1]);
                    scratch.neighbours.push((k, w, z[1], z[2]));
                }
            }
        }
        if scratch.neighbours.is_empty() || sum_w <= 0.0 {
            return NodeFit::empty();
        }
        for i in 0..3 {
            for j in 0..i {
                m[(i, j)] = m[(j, i)];
            }
        }
        let eigen = SymmetricEigen::new(m);
        let (lmin, lmax) = eigen
            .eigenvalues
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), &l| {
                (lo.min(l), hi.max(l))
            });
        let Some(m_inv) = (lmin > 0.0 && lmax <= MAX_CONDITION * lmin)
            .then(|| m.try_inverse())
            .flatten()
        else {
            return NodeFit::empty();
        };
        // β = M⁻¹ B (3×2); the equivalent kernel's row a = e₁ᵀ M⁻¹.
        let beta: [[f64; 2]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|col| (0..3).map(|j| m_inv[(i, j)] * b[j][col]).sum())
        });
        let a = Vector3::new(m_inv[(0, 0)], m_inv[(0, 1)], m_inv[(0, 2)]);
        let mean = beta[0];

        let mut sum_l2 = 0.0;
        // Clustered sums: per view, or per record (accumulated directly).
        let mut v = [[0.0f64; 2]; 2];
        let mut abs_sum = 0.0;
        let mut abs_sq = 0.0;
        for &(k, w, zx, zy) in &scratch.neighbours {
            let l = w * (a[0] + a[1] * zx + a[2] * zy);
            sum_l2 += l * l;
            if clusters == Clusters::None {
                continue;
            }
            let s = &self.samples[k];
            let fitted = [
                beta[0][0] + beta[1][0] * zx + beta[2][0] * zy,
                beta[0][1] + beta[1][1] * zx + beta[2][1] * zy,
            ];
            let le = [l * (s.e[0] - fitted[0]), l * (s.e[1] - fitted[1])];
            match clusters {
                Clusters::Views => {
                    if !scratch.seen[s.view] {
                        scratch.seen[s.view] = true;
                        scratch.touched.push(s.view);
                    }
                    scratch.view_abs[s.view] += l.abs();
                    scratch.view_sum[s.view][0] += le[0];
                    scratch.view_sum[s.view][1] += le[1];
                }
                #[cfg(test)]
                Clusters::Records => {
                    abs_sum += l.abs();
                    abs_sq += l * l;
                    accumulate_outer(&mut v, le);
                }
                Clusters::None => {}
            }
        }
        if clusters == Clusters::Views {
            scratch.touched.sort_unstable();
            for &view in &scratch.touched {
                let av = scratch.view_abs[view];
                abs_sum += av;
                abs_sq += av * av;
                accumulate_outer(&mut v, scratch.view_sum[view]);
                scratch.view_abs[view] = 0.0;
                scratch.view_sum[view] = [0.0; 2];
                scratch.seen[view] = false;
            }
            scratch.touched.clear();
        }
        let effective_samples = 1.0 / sum_l2;
        let effective_views = if abs_sq > 0.0 {
            abs_sum * abs_sum / abs_sq
        } else {
            0.0
        };
        let (std_error, significance) = if clusters != Clusters::None && effective_views > 1.0 {
            let scale = effective_views / (effective_views - 1.0);
            let (vxx, vxy, vyy) = (scale * v[0][0], scale * v[0][1], scale * v[1][1]);
            let half_trace = 0.5 * (vxx + vyy);
            let major = half_trace + (0.5 * (vxx - vyy)).hypot(vxy);
            let det = vxx * vyy - vxy * vxy;
            let chi = if det > 0.0 {
                let q = (vyy * mean[0] * mean[0] - 2.0 * vxy * mean[0] * mean[1]
                    + vxx * mean[1] * mean[1])
                    / det;
                q.max(0.0).sqrt()
            } else {
                f64::NAN
            };
            (major.max(0.0).sqrt(), chi)
        } else {
            (f64::NAN, f64::NAN)
        };
        NodeFit {
            mean,
            rms: (sum_we2 / sum_w).sqrt(),
            std_error,
            significance,
            effective_samples,
            effective_views: if clusters == Clusters::None {
                0.0
            } else {
                effective_views
            },
            conditioned: true,
        }
    }
}

/// `v += x xᵀ` for a 2-vector (upper triangle and diagonal).
fn accumulate_outer(v: &mut [[f64; 2]; 2], x: [f64; 2]) {
    v[0][0] += x[0] * x[0];
    v[0][1] += x[0] * x[1];
    v[1][1] += x[1] * x[1];
}

#[cfg(test)]
mod tests;
