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

use std::collections::HashMap;

use nalgebra::DVector;
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

/// A robot's pose error: the standard deviations of the tool's orientation
/// error about each gripper axis and of its position error along each.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RobotPoseSigma {
    /// Orientation error, radians.
    pub rot_sigma: f64,
    /// Position error, metres.
    pub trans_sigma: f64,
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
/// until no weight changes by more than [`WEIGHT_TOLERANCE`]. Returns the IR
/// and solution of the last solve and the priors it used.
///
/// With reprojection residuals whitened by stated covariances, that `σ̂` is
/// the dimensionless `σ̂₀`, the whitened residuals' standard deviation, so the
/// priors are weighed against them in the same way.
pub(crate) fn solve_weighted<F>(
    build: F,
    priors: RobotPoseSigma,
    estimate: bool,
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<(ProblemIR, BackendSolution, RobotPoseSigma), Error>
where
    F: Fn(Weighting) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error>,
{
    priors.validate()?;
    let mut weighting = Weighting::initial(priors);
    let (mut ir, initial) = build(weighting)?;
    let mut solution = backend::solve(&ir, &initial, backend_opts)?;
    for _ in 0..=MAX_REWEIGHTS {
        let pixel_sigma = solution
            .noise
            .iter()
            .find(|n| n.family == ResidualKind::Reprojection)
            .map(|n| n.sigma)
            .ok_or_else(|| Error::numerical("no reprojection residuals to weigh the robot poses"))?
            .max(MIN_PIXEL_SIGMA);
        let priors = if estimate {
            estimated_priors(&ir, &solution, weighting.priors, num_views)
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
        solution = backend::solve(&ir, &solution.params, backend_opts)?;
        if done_after || changed < WEIGHT_TOLERANCE {
            break;
        }
    }
    Ok((ir, solution, weighting.priors))
}

fn relative_change(a: f64, b: f64) -> f64 {
    (b / a - 1.0).abs()
}

/// The priors with each group's `σ` scaled by its estimated variance
/// factor, or the group's `σ` unchanged when the covariance is unavailable
/// or the group's redundancy is below [`MIN_REDUNDANCY`].
fn estimated_priors(
    ir: &ProblemIR,
    solution: &BackendSolution,
    priors: RobotPoseSigma,
    num_views: usize,
) -> RobotPoseSigma {
    let ParamUncertainty::Available(cov) = &solution.uncertainty else {
        return priors;
    };
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
    // [rotation, translation] × [Σ (δ/σ)², Σ redundancy]
    let mut sums = [[0.0; 2]; 2];
    for view in 0..num_views {
        let key = format!("robot_delta/{view}");
        let (Some(&id), Some(delta)) = (block.get(key.as_str()), solution.params.get(&key)) else {
            return priors;
        };
        for k in 0..6 {
            let Some(&i) = row.get(&Coordinate {
                block: id,
                index: k,
            }) else {
                return priors;
            };
            let (group, sigma) = if k < 3 {
                (0, priors.rot_sigma)
            } else {
                (1, priors.trans_sigma)
            };
            sums[group][0] += (delta[k] / sigma).powi(2);
            sums[group][1] += 1.0 - cov.matrix[(i, i)] / (sigma * sigma);
        }
    }
    let scaled = |sigma: f64, [num, redundancy]: [f64; 2], floor: f64| {
        if redundancy < MIN_REDUNDANCY {
            sigma
        } else {
            (sigma * (num / redundancy).sqrt()).max(floor)
        }
    };
    RobotPoseSigma {
        rot_sigma: scaled(priors.rot_sigma, sums[0], MIN_ROT_SIGMA),
        trans_sigma: scaled(priors.trans_sigma, sums[1], MIN_TRANS_SIGMA),
    }
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
