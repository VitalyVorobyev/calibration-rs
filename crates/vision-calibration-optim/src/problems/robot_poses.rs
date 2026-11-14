//! Robot-pose corrections in the hand-eye problems, and how their priors are
//! weighed against the reprojection residuals.
//!
//! Each view's measured base-to-gripper pose is corrected by a tangent
//! `δᵢ ∈ se(3)` in the gripper frame, `T_B_G,i = T_B_G,i^meas · exp(δᵢ)`,
//! with the zero-mean prior
//! `δᵢ ~ N(0, diag(σ_rot², σ_rot², σ_rot², σ_t², σ_t², σ_t²))`: the tool's
//! orientation and position error. Every view's correction is free; the priors remove the gauge
//! the corrections would otherwise share with the hand-eye and target poses.
//!
//! The reprojection residuals are in pixels and their noise `σ_px` is not
//! known in advance, so the objective weighs each prior in pixels:
//! `‖r_px‖² + Σᵢ ‖σ_px · L δᵢ‖²` with `L = diag(1/σ_rot, …, 1/σ_t)`, which is
//! the maximum-likelihood objective scaled by `σ_px²`. [`solve_weighted`] solves at
//! `σ_px = 1`, estimates `σ_px` from the reprojection residuals and solves
//! again from that solution with the priors weighed by it; the covariance
//! counts each prior's residual as having standard deviation `σ_px`.
//!
//! The configured `σ_rot`, `σ_t` may instead be starting values: with
//! `estimate`, [`solve_weighted`] estimates the robot's error from the
//! corrections by variance-component estimation. Each prior group's
//! variance factor is `ŝ² = Σ (δ̂ₖ/σ)² / Σ rₖ`, with `rₖ = 1 − Var(δ̂ₖ)/σ²`
//! the redundancy of the prior row (the share of it the data do not
//! determine); `σ ← ŝ σ` and the solve repeats until both factors are 1.
//!
//! A correction `δ̂ₖ` is the robot's error, shrunk, plus the pose-like part
//! of its view's pixel error, whose variance is `(Σ − Σ P Σ)ₖₖ` for white
//! residuals (`Σ` the covariance without its view-scatter scaling, `P` the
//! priors' information); that is the `Var(δ̂ₖ) = σ² − Σₖₖ` the factor
//! expects. When the residuals are correlated within views, the pixel part
//! is `k_px` times larger, and the plain estimate counts the excess as the
//! robot's. With `discount_view_scatter` (opt-in), `(k_px − 1) (Σ − Σ P Σ)ₖₖ`
//! is taken out of `δ̂ₖ²`. `k_px` comes from the covariance's view scatter `k`, which
//! measures the shared coordinates' inflation over pixel and prior rows
//! alike; with the priors' share honest, `k_px = 1 + (k − 1) s / t` (see
//! [`pixel_scatter`]). The first solve, at `σ_px = 1`, weighs the priors
//! against the wrong pixel noise, so its corrections are not the ones `Σ`
//! describes and it takes nothing out.
//!
//! `k_px` assumes the pose directions inflated as much as the shared
//! coordinates are, on average. Nothing in the solve can check that
//! directly: a view's pose-like pixel error and the robot's error at that
//! view are the same six-dimensional perturbation. Where the assumption
//! fails (a per-view error that the distortion scatters with far more than
//! the poses), the subtraction exceeds what the corrections hold and would
//! put a group's `σ` on its floor. So the corrected numerator
//! `c = Σ (δ̂ₖ² − excessₖ) / σ²` is used only as far as the corrections
//! resolve it, judged by its standard error `se` over the views (the
//! scatter of each view's share of `c`):
//!
//! - **Rejected:** when one group's `c` falls more than [`RESOLUTION_Z`]
//!   standard errors below zero, the subtraction claims more pixel error
//!   than the corrections contain, `k_px` overstates the pose directions,
//!   and nothing is taken out of either group: `σ` is the uncorrected
//!   estimate, an upper bound.
//! - **Resolved:** `c ≥ z·se`, the group's factor uses `c`.
//! - **Unresolved:** `c < z·se`, the group's factor uses `min(N, z·se)`
//!   (`N` the uncorrected numerator): the robot error the corrections cannot
//!   tell from none, again an upper bound. Continuous in `c`, so the
//!   iteration can pass through it to a resolved estimate.
//!
//! [`RobotRefinement::unresolved`] names the groups whose `σ` is such an
//! upper bound. A correlated error that is itself a per-view pose (a target
//! that moves between views, a per-view focus change) is indistinguishable
//! from the robot's and stays in the estimate.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use vision_calibration_core::ResidualKind;

use crate::Error;
use crate::backend::covariance::{Coordinate, ParamUncertainty};
use crate::backend::{self, BackendSolution, BackendSolveOptions};
use crate::ir::{
    FactorKind, FixedMask, ManifoldKind, ParamId, ProblemIR, ResidualBlock, RobustLoss,
};

/// The lowest pixel noise the priors are weighed with. Corner noise below
/// 0.01 px is not physical; the floor keeps the priors from vanishing on
/// noise-free data, where they alone fix the corrections' gauge.
const MIN_PIXEL_SIGMA: f64 = 0.01;
/// The lowest estimated rotation error, radians (about 0.0006°): far below
/// any arm's, it keeps an estimate on exact poses from collapsing the prior.
const MIN_ROT_SIGMA: f64 = 1e-5;
/// The lowest estimated translation error, metres (1 µm).
const MIN_TRANS_SIGMA: f64 = 1e-6;
/// A prior group's variance factor is estimated only above this total
/// redundancy (rows' worth of evidence).
const MIN_REDUNDANCY: f64 = 2.0;
/// The estimate has converged when every weight changes by less than this
/// fraction from one solve to the next: well inside the estimate's own
/// statistical spread (about 7 % from 30 views' three rotation rows).
const WEIGHT_TOLERANCE: f64 = 1e-2;
/// Solves of the variance-component iteration after the first.
const MAX_REWEIGHTS: usize = 10;
/// A group's corrected numerator is resolved above this many standard
/// errors, and the subtraction is rejected below minus this many: two
/// standard errors, about the 2.5 % one-sided level of the view scatter's
/// own significance test.
const RESOLUTION_Z: f64 = 2.0;

/// A robot's pose error: the standard deviations of the tool's orientation
/// error about each gripper axis and of its position error along each.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RobotPoseSigma {
    /// Orientation error, radians.
    pub rot_sigma: f64,
    /// Position error, metres.
    pub trans_sigma: f64,
}
impl RobotPoseSigma {
    /// A [`RobotPoseSigma`] with the given fields.
    pub fn new(rot_sigma: f64, trans_sigma: f64) -> Self {
        Self {
            rot_sigma,
            trans_sigma,
        }
    }
}

impl RobotPoseSigma {
    /// Both standard deviations are positive and finite.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !(self.rot_sigma.is_finite() && self.rot_sigma > 0.0) {
            return Err(Error::invalid_input("robot_rot_sigma must be positive"));
        }
        if !(self.trans_sigma.is_finite() && self.trans_sigma > 0.0) {
            return Err(Error::invalid_input("robot_trans_sigma must be positive"));
        }
        Ok(())
    }
}

/// The robot-pose correction a hand-eye solve estimated with
/// `refine_robot_poses`.
///
/// Absent from a result when the robot poses were taken as exact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RobotRefinement {
    /// Per-view robot-pose corrections, one se(3) tangent
    /// `[rx, ry, rz, tx, ty, tz]` per view, in the gripper frame: the view's
    /// corrected pose is `base_se3_gripper · exp(δ)`.
    pub deltas: Vec<[f64; 6]>,
    /// The robot's pose error the corrections were weighed with: the
    /// configured one, or the estimate when it was estimated.
    pub sigma: RobotPoseSigma,
    /// The parts of an estimated `sigma` that are upper bounds rather than
    /// estimates: the corrections could not separate the robot's error
    /// there from corner noise correlated within views, so that part of
    /// `sigma` counts some of the corner noise as the robot's. Reported only
    /// when the estimate leaves view-correlated corner noise out
    /// (`discount_view_scatter`); `None` when every estimated part was
    /// resolved, without that option, and when `sigma` was configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unresolved: Option<RobotSigmaParts>,
}
impl RobotRefinement {
    /// A [`RobotRefinement`] with the given fields, every part resolved.
    pub fn new(deltas: Vec<[f64; 6]>, sigma: RobotPoseSigma) -> Self {
        Self {
            deltas,
            sigma,
            unresolved: None,
        }
    }
}

/// Parts of a [`RobotPoseSigma`]: its rotation, its translation or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RobotSigmaParts {
    /// `rot_sigma`.
    Rotation,
    /// `trans_sigma`.
    Translation,
    /// `rot_sigma` and `trans_sigma`.
    Both,
}

impl RobotSigmaParts {
    /// The parts flagged in `[rotation, translation]`; `None` for neither.
    fn from_flags([rotation, translation]: [bool; 2]) -> Option<Self> {
        match (rotation, translation) {
            (true, true) => Some(Self::Both),
            (true, false) => Some(Self::Rotation),
            (false, true) => Some(Self::Translation),
            (false, false) => None,
        }
    }
}

/// How the robot-pose priors are weighed in one solve.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Weighting {
    /// Noise of the reprojection residuals, in their unit.
    pub pixel_sigma: f64,
    /// The robot's pose error.
    pub priors: RobotPoseSigma,
}

impl Weighting {
    /// The first solve's weighting: unit pixel noise.
    pub(crate) fn initial(priors: RobotPoseSigma) -> Self {
        Self {
            pixel_sigma: 1.0,
            priors,
        }
    }

    /// Adds view `view`'s correction block `robot_delta/<view>`, starting at
    /// `initial` (zero when `None`), and its prior.
    pub(crate) fn add_view(
        &self,
        ir: &mut ProblemIR,
        initial_map: &mut HashMap<String, DVector<f64>>,
        view: usize,
        initial: Option<&[f64; 6]>,
    ) -> ParamId {
        let key = format!("robot_delta/{view}");
        let id = ir.add_param_block(
            &key,
            6,
            ManifoldKind::Euclidean,
            FixedMask::all_free(),
            None,
        );
        initial_map.insert(
            key,
            initial.map_or_else(|| DVector::zeros(6), |d| DVector::from_row_slice(d)),
        );
        let rot = self.pixel_sigma / self.priors.rot_sigma;
        let trans = self.pixel_sigma / self.priors.trans_sigma;
        ir.add_residual_block(ResidualBlock {
            params: vec![id],
            loss: RobustLoss::None,
            factor: FactorKind::Se3TangentPrior {
                sqrt_info: [rot, rot, rot, trans, trans, trans],
                residual_sigma: self.pixel_sigma,
            },
            residual_dim: 6,
        });
        id
    }
}

/// Solves the IR `build(weighting)` returns (with its initial values), with
/// the robot-pose priors `priors` of `num_views` views: once at
/// `σ_px = 1`, then from that solution at the `σ̂` of its reprojection
/// residuals (floored at [`MIN_PIXEL_SIGMA`]). With `estimate`, the solve
/// repeats, re-estimating `σ_px` and the priors' `σ` (see the module docs),
/// until no weight changes by more than [`WEIGHT_TOLERANCE`], with
/// `discount_view_scatter` leaving the view-correlated corner noise out of
/// the estimate. Returns the IR
/// and solution of the last solve, the priors it used and, for estimated
/// priors, the parts of them that are upper bounds.
///
/// With reprojection residuals whitened by stated covariances, that `σ̂` is
/// the dimensionless `σ̂₀`, the whitened residuals' standard deviation, so the
/// priors are weighed against them in the same way.
pub(crate) fn solve_weighted<F>(
    build: F,
    priors: RobotPoseSigma,
    estimate: bool,
    discount_view_scatter: bool,
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<
    (
        ProblemIR,
        BackendSolution,
        RobotPoseSigma,
        Option<RobotSigmaParts>,
    ),
    Error,
>
where
    F: Fn(Weighting) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error>,
{
    priors.validate()?;
    let mut weighting = Weighting::initial(priors);
    let (mut ir, initial) = build(weighting)?;
    // Every solve's covariance steers the next weights; only the last one's
    // is reported, so only it gets the second-order prediction.
    let mut solution = backend::solve_first_order(&ir, &initial, backend_opts)?;
    let mut unresolved = None;
    for pass in 0..=MAX_REWEIGHTS {
        let pixel_sigma = solution
            .noise
            .iter()
            .find(|n| n.family == ResidualKind::Reprojection)
            .map(|n| n.sigma)
            .ok_or_else(|| Error::numerical("no reprojection residuals to weigh the robot poses"))?
            .max(MIN_PIXEL_SIGMA);
        let priors = if estimate {
            let (priors, parts) = estimated_priors(
                &ir,
                &solution,
                weighting.priors,
                num_views,
                discount_view_scatter && pass > 0,
            );
            unresolved = parts;
            priors
        } else {
            weighting.priors
        };
        let next = Weighting {
            pixel_sigma,
            priors,
        };
        let changed = relative_change(weighting.pixel_sigma, next.pixel_sigma)
            .max(relative_change(
                weighting.priors.rot_sigma,
                next.priors.rot_sigma,
            ))
            .max(relative_change(
                weighting.priors.trans_sigma,
                next.priors.trans_sigma,
            ));
        let done_after = !estimate;
        weighting = next;
        let (next_ir, _) = build(weighting)?;
        ir = next_ir;
        solution = backend::solve_first_order(&ir, &solution.params, backend_opts)?;
        if done_after || changed < WEIGHT_TOLERANCE {
            break;
        }
    }
    backend::add_second_order(&ir, &mut solution)?;
    Ok((ir, solution, weighting.priors, unresolved))
}

fn relative_change(a: f64, b: f64) -> f64 {
    (b / a - 1.0).abs()
}

/// The priors with each group's `σ` scaled by its estimated variance
/// factor, or the group's `σ` unchanged when the covariance is unavailable
/// or the group's redundancy is below [`MIN_REDUNDANCY`]; and the groups
/// whose `σ` is an upper bound (see the module docs).
/// With `discount` (the option, on a solve after the first), the
/// view-correlated excess is taken out; without it the arithmetic is the
/// plain estimate's.
fn estimated_priors(
    ir: &ProblemIR,
    solution: &BackendSolution,
    priors: RobotPoseSigma,
    num_views: usize,
    discount: bool,
) -> (RobotPoseSigma, Option<RobotSigmaParts>) {
    let ParamUncertainty::Available(cov) = &solution.uncertainty else {
        return (priors, None);
    };
    let model_scale = cov.view_scatter.unwrap_or(1.0);
    let row: HashMap<Coordinate, usize> = cov
        .coordinates
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    let block: HashMap<&str, usize> = ir
        .params
        .iter()
        .map(|p| (p.name.as_str(), p.id.0))
        .collect();
    // Every correction coordinate: its view, row, group, prior σ and value.
    let mut coords = Vec::with_capacity(6 * num_views);
    for view in 0..num_views {
        let key = format!("robot_delta/{view}");
        let (Some(&id), Some(delta)) = (block.get(key.as_str()), solution.params.get(&key)) else {
            return (priors, None);
        };
        for k in 0..6 {
            let Some(&i) = row.get(&Coordinate {
                block: id,
                index: k,
            }) else {
                return (priors, None);
            };
            let (group, sigma) = if k < 3 {
                (0, priors.rot_sigma)
            } else {
                (1, priors.trans_sigma)
            };
            coords.push((view, i, group, sigma, delta[k]));
        }
    }
    // The white covariance `Σ` (the view-scatter scaling undone), from which
    // the redundancy is the noise model's.
    let white = |a: usize, b: usize| cov.matrix[(a, b)] / model_scale;
    // `(Σ P Σ)_ab`, the part of `Σ_ab` the priors supply (`P` their
    // information, on the corrections' coordinates only).
    let from_priors = |a: usize, b: usize| {
        coords
            .iter()
            .map(|&(_, j, _, s, _)| white(a, j) * white(j, b) / (s * s))
            .sum::<f64>()
    };
    let inflation = if discount && model_scale > 1.0 {
        // The shared coordinates, as the view scatter takes them.
        let shared: Vec<usize> = cov
            .coordinates
            .iter()
            .enumerate()
            .filter(|(_, c)| ir.params.get(c.block).is_some_and(|p| p.view().is_none()))
            .map(|(i, _)| i)
            .collect();
        let n = shared.len();
        pixel_scatter(
            model_scale,
            DMatrix::from_fn(n, n, |a, b| white(shared[a], shared[b])),
            &DMatrix::from_fn(n, n, |a, b| from_priors(shared[a], shared[b])),
        )
    } else {
        1.0
    };
    let mut groups = [GroupSums::new(num_views), GroupSums::new(num_views)];
    for &(view, i, group, sigma, delta) in &coords {
        // The pixel noise in the correction beyond what `Σ` gives it.
        let excess = if inflation > 1.0 {
            (inflation - 1.0) * (white(i, i) - from_priors(i, i)).max(0.0) / (sigma * sigma)
        } else {
            0.0
        };
        let g = &mut groups[group];
        g.corrections += (delta / sigma).powi(2);
        g.redundancy += 1.0 - white(i, i) / (sigma * sigma);
        g.excess += excess;
        g.per_view[view] += (delta / sigma).powi(2) - excess;
    }
    let estimated = groups.each_ref().map(|g| g.redundancy >= MIN_REDUNDANCY);
    // Each group's factor numerator, and whether it is an upper bound.
    let numerators: [(f64, bool); 2] = if inflation > 1.0 {
        let rejected = (0..2).any(|g| estimated[g] && groups[g].rejected());
        std::array::from_fn(|g| {
            let group = &groups[g];
            if rejected {
                (group.corrections, true)
            } else {
                group.bounded()
            }
        })
    } else {
        groups.each_ref().map(|g| (g.corrections, false))
    };
    let scaled = |g: usize, sigma: f64, floor: f64| {
        if estimated[g] {
            (sigma * (numerators[g].0.max(0.0) / groups[g].redundancy).sqrt()).max(floor)
        } else {
            sigma
        }
    };
    let estimate = RobotPoseSigma {
        rot_sigma: scaled(0, priors.rot_sigma, MIN_ROT_SIGMA),
        trans_sigma: scaled(1, priors.trans_sigma, MIN_TRANS_SIGMA),
    };
    let unresolved = RobotSigmaParts::from_flags([0, 1].map(|g| estimated[g] && numerators[g].1));
    (estimate, unresolved)
}

/// One prior group's sums over its correction coordinates, each scaled by
/// the group's prior variance `σ²`.
struct GroupSums {
    /// `N = Σ δ̂ₖ²`.
    corrections: f64,
    /// `Σ rₖ`.
    redundancy: f64,
    /// `E = Σ excessₖ`, the view-correlated pixel noise beyond `Σ`.
    excess: f64,
    /// Each view's share of `N − E`.
    per_view: Vec<f64>,
}

impl GroupSums {
    fn new(num_views: usize) -> Self {
        Self {
            corrections: 0.0,
            redundancy: 0.0,
            excess: 0.0,
            per_view: vec![0.0; num_views],
        }
    }

    /// `c = N − E` and its standard error over the views,
    /// `√(V/(V−1) Σ (cᵥ − c̄)²)` (infinite with fewer than two views).
    fn corrected(&self) -> (f64, f64) {
        let v = self.per_view.len() as f64;
        let se = if self.per_view.len() < 2 {
            f64::INFINITY
        } else {
            let mean = self.per_view.iter().sum::<f64>() / v;
            (v / (v - 1.0)
                * self
                    .per_view
                    .iter()
                    .map(|c| (c - mean).powi(2))
                    .sum::<f64>())
            .sqrt()
        };
        (self.corrections - self.excess, se)
    }

    /// The subtraction claims more pixel noise than the corrections hold:
    /// `c` more than [`RESOLUTION_Z`] standard errors below zero.
    fn rejected(&self) -> bool {
        let (c, se) = self.corrected();
        c < -RESOLUTION_Z * se
    }

    /// The numerator the factor uses, `c` where resolved and
    /// `min(N, z·se)` (an upper bound, `true`) where not.
    fn bounded(&self) -> (f64, bool) {
        let (c, se) = self.corrected();
        let limit = RESOLUTION_Z * se;
        if c >= limit {
            (c, false)
        } else {
            (self.corrections.min(limit), true)
        }
    }
}

/// The view scatter of the pixel noise alone, `k_px`: how much more the
/// reprojection residuals' share of the estimate scatters across views than
/// the white covariance `Σ` says, from the covariance's view scatter `k`,
/// the white covariance `shared` of the `s` shared coordinates and the part
/// of it the priors supply, `from_priors = (Σ P Σ)_ss`.
///
/// `k = 1 + tr(Σ_ss⁻¹ (Σ_J − E[Σ_J])) / s` is the shared coordinates'
/// inflation over pixel and prior rows alike. With the priors' `σ` the
/// robot's, only the pixel part `(Σ − Σ P Σ)_ss` is inflated, so
/// `k s = k_px t + (s − t)` with `t = tr(Σ_ss⁻¹ (Σ − Σ P Σ)_ss)`, and
/// `k_px = 1 + (k − 1) s / t`. Where the priors supply none of the shared
/// coordinates (`t = s`) it is `k`. `k` itself when `Σ_ss` is singular or
/// the pixels supply none of it.
fn pixel_scatter(k: f64, shared: DMatrix<f64>, from_priors: &DMatrix<f64>) -> f64 {
    let s = shared.nrows() as f64;
    let Some(chol) = shared.cholesky() else {
        return k;
    };
    let pixel_share = s - chol.solve(from_priors).trace();
    if pixel_share <= f64::EPSILON * s {
        return k;
    }
    1.0 + (k - 1.0) * s / pixel_share
}

/// The solved correction of each of `num_views` views.
pub(crate) fn solved_deltas(
    params: &HashMap<String, DVector<f64>>,
    num_views: usize,
) -> Result<Vec<[f64; 6]>, Error> {
    (0..num_views)
        .map(|view| {
            let key = format!("robot_delta/{view}");
            let d = params
                .get(&key)
                .ok_or_else(|| Error::numerical(format!("missing {key} solution block")))?;
            Ok([d[0], d[1], d[2], d[3], d[4], d[5]])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_scatter_counts_only_the_pixel_share() {
        let shared = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 1.0]));
        // No prior share: the view scatter is the pixels'.
        let none = DMatrix::zeros(2, 2);
        assert!((pixel_scatter(3.0, shared.clone(), &none) - 3.0).abs() < 1e-12);
        // The priors supply half of each coordinate: the pixels carry
        // `t = 1` of `s = 2`, so `k_px − 1 = 2 (k − 1)`.
        let half = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 0.5]));
        assert!((pixel_scatter(3.0, shared.clone(), &half) - 5.0).abs() < 1e-12);
        // All of it from the priors: nothing to scale, `k` stands.
        assert!((pixel_scatter(3.0, shared.clone(), &shared) - 3.0).abs() < 1e-12);
    }

    fn sums(corrections: f64, excess: f64, per_view: Vec<f64>) -> GroupSums {
        GroupSums {
            corrections,
            redundancy: 30.0,
            excess,
            per_view,
        }
    }

    #[test]
    fn group_sums_bound_what_the_views_do_not_resolve() {
        // Four views' shares of `c = 40`, standard error `√(4/3 · 16) ≈ 4.62`.
        let views = vec![8.0, 12.0, 8.0, 12.0];
        let resolved = sums(100.0, 60.0, views.clone());
        assert_eq!(resolved.bounded(), (40.0, false));
        assert!(!resolved.rejected());
        // `c = 4`, under two standard errors: the bound `2 se ≈ 9.24`.
        let unresolved = sums(64.0, 60.0, views.iter().map(|c| c - 9.0).collect());
        let (numerator, bounded) = unresolved.bounded();
        assert!(bounded && (numerator - 2.0 * (64.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        // Never above the uncorrected numerator.
        assert_eq!(sums(5.0, 1.0, views.clone()).bounded(), (5.0, true));
        // `c = −20`, four standard errors below zero: the subtraction claims
        // more than the corrections hold.
        let over = sums(40.0, 60.0, views.iter().map(|c| c - 15.0).collect());
        assert!(over.rejected());
        // One view: no standard error, nothing resolved or rejected.
        let one = sums(10.0, 5.0, vec![5.0]);
        assert_eq!(one.bounded(), (10.0, true));
        assert!(!one.rejected());
    }

    #[test]
    fn robot_sigma_parts_from_flags() {
        assert_eq!(RobotSigmaParts::from_flags([false, false]), None);
        assert_eq!(
            RobotSigmaParts::from_flags([true, false]),
            Some(RobotSigmaParts::Rotation)
        );
        assert_eq!(
            RobotSigmaParts::from_flags([false, true]),
            Some(RobotSigmaParts::Translation)
        );
        assert_eq!(
            RobotSigmaParts::from_flags([true, true]),
            Some(RobotSigmaParts::Both)
        );
    }

    #[test]
    fn robot_refinement_json_roundtrip() {
        let mut refinement = RobotRefinement::new(
            vec![[0.0; 6], [1e-3, 0.0, 0.0, 0.0, 2e-4, 0.0]],
            RobotPoseSigma::new(1e-3, 5e-4),
        );
        let json = serde_json::to_string(&refinement).unwrap();
        assert!(json.contains("\"deltas\"") && json.contains("\"sigma\""));
        // Every part resolved: no key, and a payload without one reads back.
        assert!(!json.contains("unresolved"));
        let back: RobotRefinement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, refinement);
        refinement.unresolved = Some(RobotSigmaParts::Translation);
        let json = serde_json::to_string(&refinement).unwrap();
        assert!(json.contains("\"unresolved\":\"translation\""));
        let back: RobotRefinement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, refinement);
    }
}
