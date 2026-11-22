//! Held-out comparison of camera models for a planar calibration.
//!
//! A richer camera model always fits the views it was fitted to at least as
//! well as a simpler one it contains, so the training reprojection error
//! cannot tell a model that describes the lens better from one that bends to
//! the noise. [`compare_camera_models`] scores each candidate on views its
//! fit did not see.
//!
//! # Procedure
//!
//! The views are split into `K` folds by a seeded shuffle
//! ([`CrossValidationOptions`]); fold sizes differ by at most one. For each
//! candidate camera configuration and each fold, the planar calibration
//! (`run_calibration`, with the candidate in place of the config's camera)
//! is fitted to the other folds' views. Each held-out view is then scored
//! against that fold's camera alone: its target pose is seeded from the
//! camera's back-projected bearings and solved by a pose-only least-squares
//! fit with every camera parameter held fixed (the config's robust loss and
//! solver backend), and every point's error is the Euclidean pixel distance
//! between the observation and its projection under the solved pose.
//! Nothing is trimmed: the robust loss shapes the pose solve, not the score.
//! Every view is held out exactly once, so the pooled held-out RMS covers
//! the same points for every candidate whose points all project.
//!
//! # Reading the result
//!
//! [`ModelComparison::ranking`] orders the candidates by pooled held-out
//! RMS. The paired standard error [`HeldOutScore::delta_se_px`] says how
//! far apart two candidates must be for the difference to mean anything: a
//! rule that works well is to prefer the candidate with the fewest
//! [`free_parameters`](ModelScore::free_parameters) among those within
//! about two standard errors of the best.
//!
//! The held-out pose is re-solved per view, so a camera term a pose can
//! absorb (part of a principal-point shift, for one) is partly forgiven:
//! the comparison is sharpest for distortion terms, which no rigid pose of a
//! plane can imitate.

use serde::{Deserialize, Serialize};

use vision_calibration_core::{
    CameraFix, CameraParams, CameraProject, INTRINSICS_PARAM_NAMES, NoMeta, PlanarDataset,
    ReprojectionStats, RunControl, View, compute_planar_target_residuals,
};
use vision_calibration_optim::{
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, SolveTermination,
    optimize_planar_intrinsics,
};

use super::seed_view_pose;
use crate::Error;
use crate::common::config::CameraConfig;
use crate::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarIntrinsicsProblem, run_calibration,
};
use crate::session::{CalibrationSession, ProblemType};

/// The fewest views a planar fit takes (Zhang's method).
const MIN_TRAINING_VIEWS: usize = 3;

/// How [`compare_camera_models`] splits the views into folds.
///
/// The views are shuffled by a seeded generator whose sequence is fixed by
/// this library, not by a dependency, so a seed gives the same folds on
/// every platform and release; the `i`-th shuffled view goes to fold
/// `i mod folds`, so fold sizes differ by at most one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct CrossValidationOptions {
    /// Number of folds `K`, at least 2 and at most the number of views
    /// (`K` equal to the number of views is leave-one-view-out). Each fit
    /// sees about `(K − 1)/K` of the views and must keep at least three.
    /// Default 5: each fit keeps 80 % of the views, and five fits per
    /// candidate keep the cost moderate.
    pub folds: usize,
    /// Seed of the shuffle that assigns views to folds, any value. Default
    /// 0. Another seed gives another partition; a ranking that changes with
    /// the seed is not a reliable one.
    pub seed: u64,
}

impl Default for CrossValidationOptions {
    fn default() -> Self {
        Self { folds: 5, seed: 0 }
    }
}

impl CrossValidationOptions {
    /// `folds` folds, shuffled with `seed`.
    pub fn new(folds: usize, seed: u64) -> Self {
        Self { folds, seed }
    }
}

/// The candidates of [`compare_camera_models`], each scored on held-out
/// views, and their ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ModelComparison {
    /// The held-out view indices of each fold, ascending within a fold.
    /// Every view is in exactly one fold.
    pub folds: Vec<Vec<usize>>,
    /// One score per candidate, in input order: the config's camera first,
    /// then the alternatives.
    pub candidates: Vec<ModelScore>,
    /// Indices into `candidates` of the comparable ones (scored, with every
    /// held-out point projected), by pooled held-out RMS, lowest first;
    /// ties go to fewer free parameters, then to the earlier candidate.
    /// Empty when no candidate is comparable.
    pub ranking: Vec<usize>,
}

/// One candidate camera configuration and how it scored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ModelScore {
    /// The candidate: the camera model and its fixed parameters.
    pub camera: CameraConfig,
    /// The camera parameters the candidate's fit estimates: its intrinsics,
    /// projection, distortion and sensor parameters minus those its fix
    /// lists (or the model's default fix list) hold fixed. Not the model's
    /// slot count: Brown-Conrady with `k3` fixed has 8.
    pub free_parameters: usize,
    /// The held-out score, or why the candidate could not be scored.
    pub outcome: ModelOutcome,
}

/// Whether a candidate was scored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelOutcome {
    /// Every fit and every held-out pose solve succeeded.
    Scored(HeldOutScore),
    /// A fit or a held-out pose solve failed; the other candidates are still
    /// scored.
    Failed {
        /// The fold whose fit or held-out view failed; `None` for the fit to
        /// all views.
        fold: Option<usize>,
        /// The error, as text.
        reason: String,
    },
}

/// A candidate's reprojection error on the views it was fitted to and on
/// the views it was not. All errors are Euclidean pixel distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct HeldOutScore {
    /// The fit to all views, on those views (in-sample), pixels. It only
    /// falls as the model grows; it does not rank models.
    pub training: ReprojectionStats,
    /// Pooled over the folds: each view scored once, by the fit of the fold
    /// that held it out, pixels. `count` excludes points that do not
    /// project.
    pub held_out: ReprojectionStats,
    /// One entry per fold, in fold order.
    pub folds: Vec<FoldScore>,
    /// The held-out RMS of each view, pixels, by view index; 0 for a view
    /// none of whose points project.
    pub per_view_rms_px: Vec<f64>,
    /// The sample standard deviation of the folds' held-out RMS, pixels
    /// (≥ 0): how much the score moves with the views a fit sees.
    pub fold_scatter_px: f64,
    /// This candidate's pooled held-out RMS minus that of the best ranked
    /// candidate (`ranking[0]`), pixels; 0 for the best itself and when no
    /// candidate is ranked.
    pub delta_vs_best_px: f64,
    /// The standard error of `delta_vs_best_px`, paired by fold: the sample
    /// standard deviation over the folds of this candidate's held-out RMS
    /// minus the best's, divided by `√K`, pixels (≥ 0). The candidates see
    /// the same folds, so the shared view-to-view variation cancels.
    pub delta_se_px: f64,
    /// Held-out points that do not project through the fold's camera at
    /// the solved pose. A candidate with any is listed but not ranked: its
    /// score covers fewer points than the others'.
    pub unprojected_points: usize,
}

/// One fold of a candidate's score.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct FoldScore {
    /// The RMS of the fold's fit on its own training views, pixels.
    pub training_rms_px: f64,
    /// The RMS of the fold's held-out views, pixels.
    pub held_out_rms_px: f64,
    /// Held-out points that project, the count behind `held_out_rms_px`.
    pub held_out_points: usize,
    /// Whether every solve of the fold (the fit and each held-out pose)
    /// stopped by a convergence rule rather than at its iteration budget.
    pub converged: bool,
}

/// Rank candidate camera configurations of a planar calibration by their
/// reprojection error on held-out views (see the module documentation).
///
/// `config` is the estimator: its initialization, refinement and solver
/// settings serve every fit, and its `camera` is candidate 0. Each of
/// `alternatives` replaces `config.camera` in turn. `control` reports the
/// steps and solver iterations of every fit and is polled for
/// cancellation; nothing is reported before the inputs are checked.
///
/// The result carries scores, not fitted cameras: calibrate the chosen
/// configuration on all views as usual.
///
/// # Errors
///
/// Raised before any fit:
///
/// - [`Error::InvalidInput`] when `opts.folds` is below 2, a candidate
///   fails the planar configuration checks (an unsupported model, a fix
///   list naming a parameter the model does not have), two candidates are
///   equal, or `config.optimize.fix_poses` is not empty (its indices name
///   views of the whole input, which a fold does not have);
/// - [`Error::InsufficientData`] when there are fewer views than folds or a
///   fold's fit would keep fewer than three views;
/// - the input checks of a planar calibration (too few points in a view,
///   mixed point covariances).
///
/// Then [`Error::Cancelled`] once `control`'s token is cancelled. A fit or
/// held-out solve that fails makes that candidate
/// [`ModelOutcome::Failed`]; it is not an error.
///
/// # Example
///
/// ```no_run
/// use vision_calibration_core::{CameraModelSpec, DistortionKind, RunControl, SensorKind};
/// use vision_calibration_pipeline::analysis::{CrossValidationOptions, compare_camera_models};
/// use vision_calibration_pipeline::common::config::CameraConfig;
/// use vision_calibration_pipeline::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsInput};
/// # fn main() -> Result<(), vision_calibration_pipeline::Error> {
/// # let input: PlanarIntrinsicsInput = unimplemented!("the observed views");
/// let config = PlanarIntrinsicsConfig::default(); // Brown-Conrady, k3 fixed
/// let rational = CameraConfig::new(CameraModelSpec::pinhole(
///     DistortionKind::Rational8,
///     SensorKind::Identity,
/// ));
/// let comparison = compare_camera_models(
///     &input,
///     &config,
///     &[rational],
///     &CrossValidationOptions::default(),
///     &RunControl::default(),
/// )?;
/// let best = &comparison.candidates[comparison.ranking[0]];
/// println!("{:?} with {} free parameters", best.camera.model, best.free_parameters);
/// # Ok(())
/// # }
/// ```
pub fn compare_camera_models(
    input: &PlanarIntrinsicsInput,
    config: &PlanarIntrinsicsConfig,
    alternatives: &[CameraConfig],
    opts: &CrossValidationOptions,
    control: &RunControl,
) -> Result<ModelComparison, Error> {
    let candidates: Vec<CameraConfig> = std::iter::once(config.camera.clone())
        .chain(alternatives.iter().cloned())
        .collect();
    let free_parameters = check_inputs(input, config, &candidates, opts)?;
    control.check()?;

    let folds = partition(input.num_views(), opts.folds, opts.seed);
    let mut scores = Vec::with_capacity(candidates.len());
    for (camera, free_parameters) in candidates.into_iter().zip(free_parameters) {
        control.check()?;
        let mut candidate_config = config.clone();
        candidate_config.camera = camera.clone();
        let outcome = match score_candidate(input, &candidate_config, &folds, control) {
            Ok(score) => ModelOutcome::Scored(score),
            Err(Failure::Cancelled) => return Err(Error::Cancelled),
            Err(Failure::Failed { fold, reason }) => ModelOutcome::Failed { fold, reason },
        };
        scores.push(ModelScore {
            camera,
            free_parameters,
            outcome,
        });
    }

    let ranking = rank(&scores);
    set_deltas(&mut scores, ranking.first().copied());
    Ok(ModelComparison {
        folds,
        candidates: scores,
        ranking,
    })
}

/// Check everything [`compare_camera_models`] can reject before it fits,
/// and return each candidate's free parameter count.
fn check_inputs(
    input: &PlanarIntrinsicsInput,
    config: &PlanarIntrinsicsConfig,
    candidates: &[CameraConfig],
    opts: &CrossValidationOptions,
) -> Result<Vec<usize>, Error> {
    if opts.folds < 2 {
        return Err(Error::invalid_input(format!(
            "cross-validation needs at least 2 folds, got {}",
            opts.folds
        )));
    }
    PlanarIntrinsicsProblem::validate_input(input)?;
    let views = input.num_views();
    let need = min_views(opts.folds);
    if views < need {
        return Err(Error::InsufficientData { need, got: views });
    }
    if !config.optimize.fix_poses.is_empty() {
        return Err(Error::invalid_input(
            "optimize.fix_poses names views of the whole input, which a fold does not \
             have; leave it empty to compare camera models",
        ));
    }
    let mut free = Vec::with_capacity(candidates.len());
    for (i, camera) in candidates.iter().enumerate() {
        let mut candidate_config = config.clone();
        candidate_config.camera = camera.clone();
        PlanarIntrinsicsProblem::validate_config(&candidate_config).map_err(|err| match err {
            Error::InvalidInput { reason } => {
                Error::invalid_input(format!("candidate {i}: {reason}"))
            }
            other => other,
        })?;
        free.push(
            free_parameters(camera)
                .map_err(|err| Error::invalid_input(format!("candidate {i}: {err}")))?,
        );
        if let Some(first) = candidates[..i].iter().position(|c| c == camera) {
            return Err(Error::invalid_input(format!(
                "candidate {i} repeats candidate {first}"
            )));
        }
    }
    Ok(free)
}

/// The fewest views `folds` folds can split while every fit keeps
/// [`MIN_TRAINING_VIEWS`]: at least `folds`, and `v − ⌈v/K⌉ ≥ 3`.
fn min_views(folds: usize) -> usize {
    (folds.max(MIN_TRAINING_VIEWS)..=usize::MAX)
        .find(|&v| v - v.div_ceil(folds) >= MIN_TRAINING_VIEWS)
        // v − ⌈v/K⌉ ≥ v − ⌈v/2⌉ grows without bound, and at v = usize::MAX
        // with K ≥ 2 it is far above 3.
        .unwrap_or(usize::MAX)
}

/// The camera parameters `camera`'s fit estimates.
fn free_parameters(camera: &CameraConfig) -> Result<usize, vision_calibration_core::Error> {
    let spec = camera.model;
    let fixed = camera.fix.resolve(spec)?;
    Ok(
        INTRINSICS_PARAM_NAMES.len() - fixed.intrinsics.len() + spec.projection.param_names().len()
            - fixed.projection.len()
            + spec.distortion.param_names().len()
            - fixed.distortion.len()
            + spec.sensor.param_names().len()
            - fixed.sensor.len(),
    )
}

/// The views split into `folds` folds: a Fisher–Yates shuffle driven by
/// SplitMix64 from `seed`, the `i`-th shuffled view into fold `i mod folds`,
/// each fold sorted.
fn partition(num_views: usize, folds: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..num_views).collect();
    let mut rng = SplitMix64(seed);
    for i in (1..num_views).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut out = vec![Vec::new(); folds];
    for (position, view) in order.into_iter().enumerate() {
        out[position % folds].push(view);
    }
    for fold in &mut out {
        fold.sort_unstable();
    }
    out
}

/// SplitMix64: a fixed, value-stable sequence for a seed, independent of
/// any dependency's generator.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (`n ≥ 1`), by the multiply-shift reduction.
    fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next()) * n as u128) >> 64) as usize
    }
}

/// Why a candidate was not scored.
enum Failure {
    /// The run was cancelled: the whole comparison stops.
    Cancelled,
    /// A fit or held-out solve failed: only this candidate is affected.
    Failed { fold: Option<usize>, reason: String },
}

impl Failure {
    fn at(fold: Option<usize>) -> impl FnOnce(Error) -> Self {
        move |err| match err {
            Error::Cancelled => Self::Cancelled,
            err => Self::Failed {
                fold,
                reason: err.to_string(),
            },
        }
    }
}

/// A planar fit: its camera, its training errors and whether it converged.
struct Fit {
    camera: CameraParams,
    training_errors: Vec<f64>,
    converged: bool,
}

/// Fit `config` to `dataset` with `run_calibration`.
fn fit(
    dataset: PlanarDataset,
    config: &PlanarIntrinsicsConfig,
    control: &RunControl,
) -> Result<Fit, Error> {
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_control(control.clone());
    session.set_config(config.clone())?;
    session.set_input(dataset)?;
    run_calibration(&mut session)?;
    let output = session.require_output()?;
    let camera = output.params.camera.clone();
    let residuals = compute_planar_target_residuals(
        &camera.build()?,
        session.require_input()?,
        &output.params.cam_se3_target,
    )?;
    Ok(Fit {
        camera,
        training_errors: finite_errors(residuals.iter().map(|r| r.error_px)),
        converged: converged(output.report.termination),
    })
}

/// Whether a solve stopped by a convergence rule (an unknown reason counts
/// as one).
fn converged(termination: Option<SolveTermination>) -> bool {
    termination.is_none_or(SolveTermination::is_converged)
}

fn finite_errors(errors: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    errors.flatten().filter(|e| e.is_finite()).collect()
}

/// One held-out view scored against a fold's camera.
struct HeldOutView {
    errors: Vec<f64>,
    unprojected: usize,
    converged: bool,
}

/// Seed `view`'s pose from `camera` alone, solve it with every camera
/// parameter fixed, and measure each point's pixel error.
fn score_held_out_view(
    view: &View<NoMeta>,
    camera: &CameraParams,
    config: &PlanarIntrinsicsConfig,
    control: &RunControl,
) -> Result<HeldOutView, Error> {
    let model = camera.build()?;
    let obs = &view.obs;
    let seed = seed_view_pose(&model, &camera.intrinsics, &obs.points_3d, &obs.points_2d)
        .ok_or_else(|| Error::numerical("the held-out view's pose could not be seeded"))?;
    let dataset = PlanarDataset::new(vec![view.clone()])?;
    let initial = PlanarIntrinsicsParams::new(camera.clone(), vec![seed])?;
    let opts = PlanarIntrinsicsSolveOptions {
        robust_loss: config.optimize.reprojection_loss,
        fix: CameraFix::all(config.camera.model),
        fix_poses: Vec::new(),
        bounds: None,
    };
    let solved = optimize_planar_intrinsics(
        &dataset,
        &initial,
        opts,
        config.solver.backend_options(control),
    )?;
    let pose = solved.params.cam_se3_target[0];
    let mut errors = Vec::with_capacity(obs.len());
    let mut unprojected = 0;
    for (p3d, p2d) in obs.points_3d.iter().zip(&obs.points_2d) {
        let error = model
            .project_camera_point(&(pose * p3d).coords)
            .map(|projected| (projected.coords - p2d.coords).norm())
            .filter(|e| e.is_finite());
        match error {
            Some(e) => errors.push(e),
            None => unprojected += 1,
        }
    }
    Ok(HeldOutView {
        errors,
        unprojected,
        converged: converged(solved.report.termination),
    })
}

/// Score one candidate (already substituted into `config`) over `folds`;
/// the deltas against the best are filled in later.
fn score_candidate(
    input: &PlanarIntrinsicsInput,
    config: &PlanarIntrinsicsConfig,
    folds: &[Vec<usize>],
    control: &RunControl,
) -> Result<HeldOutScore, Failure> {
    let full = fit(input.clone(), config, control).map_err(Failure::at(None))?;

    let num_views = input.num_views();
    let mut in_fold = vec![usize::MAX; num_views];
    for (f, fold) in folds.iter().enumerate() {
        for &v in fold {
            in_fold[v] = f;
        }
    }

    let mut per_view_rms_px = vec![0.0; num_views];
    let mut pooled = Vec::new();
    let mut unprojected_points = 0;
    let mut fold_scores = Vec::with_capacity(folds.len());
    for (f, held_out) in folds.iter().enumerate() {
        let training: Vec<View<NoMeta>> = (0..num_views)
            .filter(|&v| in_fold[v] != f)
            .map(|v| input.views[v].clone())
            .collect();
        let dataset =
            PlanarDataset::new(training).map_err(|err| Failure::at(Some(f))(Error::from(err)))?;
        let fold_fit = fit(dataset, config, control).map_err(Failure::at(Some(f)))?;

        let mut fold_errors = Vec::new();
        let mut converged = fold_fit.converged;
        for &v in held_out {
            let scored = score_held_out_view(&input.views[v], &fold_fit.camera, config, control)
                .map_err(Failure::at(Some(f)))?;
            per_view_rms_px[v] = rms(&scored.errors);
            unprojected_points += scored.unprojected;
            converged &= scored.converged;
            fold_errors.extend_from_slice(&scored.errors);
        }
        fold_scores.push(FoldScore {
            training_rms_px: rms(&fold_fit.training_errors),
            held_out_rms_px: rms(&fold_errors),
            held_out_points: fold_errors.len(),
            converged,
        });
        pooled.extend(fold_errors);
    }

    let fold_rms: Vec<f64> = fold_scores.iter().map(|s| s.held_out_rms_px).collect();
    Ok(HeldOutScore {
        training: ReprojectionStats::from_errors(&full.training_errors),
        held_out: ReprojectionStats::from_errors(&pooled),
        fold_scatter_px: sample_sd(&fold_rms),
        folds: fold_scores,
        per_view_rms_px,
        delta_vs_best_px: 0.0,
        delta_se_px: 0.0,
        unprojected_points,
    })
}

/// The comparable candidates by pooled held-out RMS, then free parameters,
/// then index.
fn rank(scores: &[ModelScore]) -> Vec<usize> {
    let mut ranking: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match &s.outcome {
            ModelOutcome::Scored(h) if h.unprojected_points == 0 => Some(i),
            _ => None,
        })
        .collect();
    let rms_of = |i: usize| match &scores[i].outcome {
        ModelOutcome::Scored(h) => h.held_out.rms,
        ModelOutcome::Failed { .. } => f64::INFINITY,
    };
    ranking.sort_by(|&a, &b| {
        rms_of(a)
            .total_cmp(&rms_of(b))
            .then(scores[a].free_parameters.cmp(&scores[b].free_parameters))
            .then(a.cmp(&b))
    });
    ranking
}

/// Fill every scored candidate's difference from `best` and its paired
/// standard error.
fn set_deltas(scores: &mut [ModelScore], best: Option<usize>) {
    let Some(ModelOutcome::Scored(best)) = best.map(|b| scores[b].outcome.clone()) else {
        return;
    };
    for score in scores {
        let ModelOutcome::Scored(h) = &mut score.outcome else {
            continue;
        };
        h.delta_vs_best_px = h.held_out.rms - best.held_out.rms;
        let paired: Vec<f64> = h
            .folds
            .iter()
            .zip(&best.folds)
            .map(|(c, b)| c.held_out_rms_px - b.held_out_rms_px)
            .collect();
        h.delta_se_px = sample_sd(&paired) / (paired.len() as f64).sqrt();
    }
}

/// `sqrt(mean(e²))`; 0 for no errors.
fn rms(errors: &[f64]) -> f64 {
    ReprojectionStats::from_errors(errors).rms
}

/// The sample standard deviation (`n − 1` in the denominator); 0 for fewer
/// than two values.
fn sample_sd(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let ss: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    (ss / (n - 1) as f64).sqrt()
}

#[cfg(test)]
mod tests;
