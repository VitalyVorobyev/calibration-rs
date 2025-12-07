//! Generic, model-agnostic RANSAC implementation.
//!
//! To use this module, implement the [`Estimator`] trait for your model and
//! call [`ransac`] with a slice of input data and some [`RansacOptions`].
//!
//! This implementation is deliberately minimal and does not panic on failure:
//! when consensus is not found, [`ransac`] returns a [`RansacResult`] with
//! `success == false` and `model == None`.

use rand::prelude::IndexedRandom;
use rand::{rngs::StdRng, SeedableRng};

/// Configuration parameters for the generic RANSAC engine.
#[derive(Debug, Clone)]
pub struct RansacOptions {
    /// Maximum number of RANSAC iterations.
    pub max_iters: usize,
    /// Inlier residual threshold.
    pub thresh: f64,
    /// Minimum number of inliers required to accept a model.
    pub min_inliers: usize,
    /// Desired confidence level in `[0, 1]` for finding a good model.
    pub confidence: f64,
    /// Random-number generator seed (for reproducibility).
    pub seed: u64,
    /// If `true`, refit the model on all inliers before scoring.
    pub refit_on_inliers: bool,
}

impl Default for RansacOptions {
    fn default() -> Self {
        Self {
            max_iters: 1000,
            thresh: 2.0,
            min_inliers: 12,
            confidence: 0.99,
            seed: 1_234_567,
            refit_on_inliers: true,
        }
    }
}

/// Output of a RANSAC run.
///
/// Check the [`success`] flag before using the model; if it is `false`, then
/// [`model`] will be `None` and the other fields are unspecified.
#[derive(Debug, Clone)]
pub struct RansacResult<M> {
    /// Whether a consensus set satisfying the options was found.
    pub success: bool,
    /// Best model found (if any).
    pub model: Option<M>,
    /// Indices of inlier data points.
    pub inliers: Vec<usize>,
    /// Root-mean-square residual over inliers.
    pub inlier_rms: f64,
    /// Number of iterations actually performed.
    pub iters: usize,
}

impl<M> Default for RansacResult<M> {
    fn default() -> Self {
        Self {
            success: false,
            model: None,
            inliers: Vec::new(),
            inlier_rms: f64::INFINITY,
            iters: 0,
        }
    }
}

/// Generic estimator for RANSAC-like methods.
///
/// Implement this for your geometric models: lines, planes, homographies, etc.
pub trait Estimator {
    type Datum;
    type Model;

    /// Minimal number of samples needed to estimate a model.
    const MIN_SAMPLES: usize;

    /// Fit a model from a subset of data indices.
    ///
    /// Return `None` if the subset is degenerate or fitting fails.
    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model>;

    /// Residual/error for one datum (e.g. reprojection error, distance).
    ///
    /// This should be a **non-negative scalar** in the same units as `opts.thresh`.
    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64;

    /// Optional degeneracy check on the sample subset.
    ///
    /// Default: assume non-degenerate.
    fn is_degenerate(_data: &[Self::Datum], _sample_indices: &[usize]) -> bool {
        false
    }

    /// Optional refit on full inlier set.
    ///
    /// Default: no refit; use the original model.
    fn refit(_data: &[Self::Datum], _inliers: &[usize]) -> Option<Self::Model> {
        None
    }
}

fn rms(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::INFINITY;
    }
    let ss: f64 = vals.iter().map(|&v| v * v).sum();
    (ss / (vals.len() as f64)).sqrt()
}

/// Dynamic iteration bound from current inlier ratio.
/// Same formula as in your C++ code.
fn calculate_iterations(
    confidence: f64,
    inlier_ratio: f64,
    min_samples: usize,
    iters_so_far: usize,
    max_iters: usize,
) -> usize {
    if confidence <= 0.0 || inlier_ratio <= 0.0 {
        return max_iters;
    }

    let p = confidence;
    let w = inlier_ratio;
    let m = min_samples as f64;

    let denom = (1.0 - w.powf(m)).max(1e-12).ln();
    if denom >= 0.0 {
        return max_iters;
    }

    let n_iter = ((1.0 - p).ln() / denom).ceil() as usize;
    n_iter.clamp(iters_so_far, max_iters)
}

fn is_better_model(
    has_current_best: bool,
    new_inlier_count: usize,
    new_inlier_rms: f64,
    best_inlier_count: usize,
    best_inlier_rms: f64,
) -> bool {
    !has_current_best
        || (new_inlier_count > best_inlier_count)
        || (new_inlier_count == best_inlier_count && new_inlier_rms < best_inlier_rms)
}

/// Run a generic RANSAC loop for a given [`Estimator`] implementation.
///
/// This function never panics under normal circumstances. If there is
/// insufficient data or no consensus model can be found within the iteration
/// budget, it returns a [`RansacResult`] with `success == false` and
/// `model == None`.
pub fn ransac<E: Estimator>(data: &[E::Datum], opts: &RansacOptions) -> RansacResult<E::Model> {
    let mut best: RansacResult<E::Model> = RansacResult::default();

    if data.len() < E::MIN_SAMPLES {
        return best;
    }

    let all_indices: Vec<usize> = (0..data.len()).collect();
    let mut sample_idxs = vec![0usize; E::MIN_SAMPLES];

    let mut rng = StdRng::seed_from_u64(opts.seed);

    let mut dynamic_max_iters = opts.max_iters;

    let mut inliers = Vec::<usize>::new();
    let mut inlier_residuals = Vec::<f64>::new();

    let mut refined_inliers = Vec::<usize>::new();
    let mut refined_residuals = Vec::<f64>::new();

    let mut num_iters = 0;
    while num_iters < dynamic_max_iters {
        num_iters += 1;
        // Draw a random sample of MIN_SAMPLES indices
        all_indices
            .as_slice()
            .choose_multiple(&mut rng, E::MIN_SAMPLES)
            .enumerate()
            .for_each(|(k, &idx)| sample_idxs[k] = idx);

        if E::is_degenerate(data, &sample_idxs) {
            continue;
        }

        let Some(model) = E::fit(data, &sample_idxs) else {
            continue;
        };

        // Find inliers
        inliers.clear();
        inlier_residuals.clear();
        inliers.reserve(data.len());
        inlier_residuals.reserve(data.len());

        for (i, datum) in data.iter().enumerate() {
            let r = E::residual(&model, datum);
            if r <= opts.thresh {
                inliers.push(i);
                inlier_residuals.push(r);
            }
        }

        if inliers.len() < opts.min_inliers {
            continue;
        }

        let mut model_refit = model;
        let (final_inliers, final_residuals) = if opts.refit_on_inliers {
            refined_inliers.clear();
            refined_inliers.extend_from_slice(&inliers);
            refined_residuals.clear();
            refined_residuals.extend_from_slice(&inlier_residuals);

            if let Some(m2) = E::refit(data, &refined_inliers) {
                model_refit = m2;

                // Recompute inliers for the refined model
                refined_inliers.clear();
                refined_residuals.clear();
                for (i, datum) in data.iter().enumerate() {
                    let r = E::residual(&model_refit, datum);
                    if r <= opts.thresh {
                        refined_inliers.push(i);
                        refined_residuals.push(r);
                    }
                }
            }

            (&refined_inliers, &refined_residuals)
        } else {
            (&inliers, &inlier_residuals)
        };

        let final_rms = rms(final_residuals);

        if is_better_model(
            best.success,
            final_inliers.len(),
            final_rms,
            best.inliers.len(),
            best.inlier_rms,
        ) {
            best.success = true;
            best.model = Some(model_refit);
            best.inliers = final_inliers.clone();
            best.inlier_rms = final_rms;
            best.iters = num_iters;
        }

        let inlier_ratio = final_inliers.len() as f64 / data.len() as f64;
        dynamic_max_iters = calculate_iterations(
            opts.confidence,
            inlier_ratio,
            E::MIN_SAMPLES,
            num_iters,
            opts.max_iters,
        );
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct LineModel {
        slope: f64,
        intercept: f64,
    }

    struct LineEstimator;

    impl Estimator for LineEstimator {
        type Datum = (f64, f64); // (x, y)
        type Model = LineModel;

        const MIN_SAMPLES: usize = 2;

        fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
            let p0 = data[sample_indices[0]];
            let p1 = data[sample_indices[1]];
            let dx = p1.0 - p0.0;
            let dy = p1.1 - p0.1;
            if dx.abs() < 1e-9 {
                return None;
            }
            let slope = dy / dx;
            let intercept = p0.1 - slope * p0.0;
            Some(LineModel { slope, intercept })
        }

        fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
            // Perpendicular distance to the line y = m x + b
            let (x, y) = *datum;
            let numer = (model.slope * x - y + model.intercept).abs();
            let denom = (model.slope * model.slope + 1.0).sqrt();
            numer / denom
        }

        fn is_degenerate(_data: &[Self::Datum], sample_indices: &[usize]) -> bool {
            sample_indices.len() >= 2 && sample_indices[0] == sample_indices[1]
        }

        fn refit(data: &[Self::Datum], inliers: &[usize]) -> Option<Self::Model> {
            if inliers.len() < 2 {
                return None;
            }
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xx = 0.0;
            let mut sum_xy = 0.0;
            for &idx in inliers {
                let (x, y) = data[idx];
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
            }
            let n = inliers.len() as f64;
            let denom = n * sum_xx - sum_x * sum_x;
            if denom.abs() < 1e-12 {
                return None;
            }
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n;
            Some(LineModel { slope, intercept })
        }
    }

    fn default_opts() -> RansacOptions {
        RansacOptions {
            max_iters: 500,
            thresh: 0.05,
            min_inliers: 6,
            confidence: 0.99,
            seed: 42,
            refit_on_inliers: true,
        }
    }

    #[test]
    fn ransac_handles_insufficient_data() {
        let data = vec![(0.0, 0.0)];
        let res = ransac::<LineEstimator>(&data, &default_opts());
        assert!(!res.success);
        assert!(res.model.is_none());
        assert!(res.inliers.is_empty());
    }

    #[test]
    fn ransac_recovers_line_with_outliers() {
        let mut data = Vec::new();
        for i in 0..10 {
            let x = i as f64 * 0.5;
            let y = 2.0 * x + 1.0 + (if i % 2 == 0 { 0.01 } else { -0.01 });
            data.push((x, y));
        }
        // Gross outliers
        data.push((5.0, -3.0));
        data.push((6.0, 10.0));
        data.push((7.0, -8.0));

        let opts = default_opts();
        let res = ransac::<LineEstimator>(&data, &opts);

        assert!(res.success);
        let model = res.model.expect("model should be present");
        assert!((model.slope - 2.0).abs() < 0.05);
        assert!((model.intercept - 1.0).abs() < 0.05);
        assert!(res.inliers.len() >= opts.min_inliers);
    }
}
