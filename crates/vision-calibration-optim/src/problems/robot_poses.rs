//! Robot-pose corrections in the hand-eye problems, and how their priors are
//! weighed against the reprojection residuals.
//!
//! Each view's measured base-to-gripper pose is corrected by a tangent
//! `δᵢ ∈ se(3)`, `T_B_G,i = exp(δᵢ) · T_B_G,i^meas`, with the zero-mean prior
//! `δᵢ ~ N(0, diag(σ_rot², σ_rot², σ_rot², σ_t², σ_t², σ_t²))`: the robot's
//! pose error. Every view's correction is free; the priors remove the gauge
//! the corrections would otherwise share with the hand-eye and target poses.
//!
//! The reprojection residuals are in pixels and their noise `σ_px` is not
//! known in advance, so the objective weighs each prior in pixels:
//! `‖r_px‖² + Σᵢ ‖σ_px · L δᵢ‖²` with `L = diag(1/σ_rot, …, 1/σ_t)`, which is
//! the maximum-likelihood objective scaled by `σ_px²`. [`solve_weighted`] solves at
//! `σ_px = 1`, estimates `σ_px` from the reprojection residuals and solves
//! again from that solution with the priors weighed by it; the covariance
//! counts each prior's residual as having standard deviation `σ_px`.

use std::collections::HashMap;

use nalgebra::DVector;
use vision_calibration_core::ResidualKind;

use crate::Error;
use crate::backend::{self, BackendSolution, BackendSolveOptions};
use crate::ir::{
    FactorKind, FixedMask, ManifoldKind, ParamId, ProblemIR, ResidualBlock, RobustLoss,
};

/// The lowest pixel noise the priors are weighed with. Corner noise below
/// 0.01 px is not physical; the floor keeps the priors from vanishing on
/// noise-free data, where they alone fix the corrections' gauge.
const MIN_PIXEL_SIGMA: f64 = 0.01;

/// The robot's pose error: per-axis standard deviations of the correction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RobotPosePriors {
    /// Rotation, radians.
    pub rot_sigma: f64,
    /// Translation, metres.
    pub trans_sigma: f64,
}

impl RobotPosePriors {
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

    /// Adds view `view`'s correction block `robot_delta/<view>`, starting at
    /// `initial` (zero when `None`), and its prior weighed against pixel
    /// residuals of noise `pixel_sigma`.
    pub(crate) fn add_view(
        &self,
        ir: &mut ProblemIR,
        initial_map: &mut HashMap<String, DVector<f64>>,
        view: usize,
        initial: Option<&[f64; 6]>,
        pixel_sigma: f64,
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
        let rot = pixel_sigma / self.rot_sigma;
        let trans = pixel_sigma / self.trans_sigma;
        ir.add_residual_block(ResidualBlock {
            params: vec![id],
            loss: RobustLoss::None,
            factor: FactorKind::Se3TangentPrior {
                sqrt_info: [rot, rot, rot, trans, trans, trans],
                residual_sigma: pixel_sigma,
            },
            residual_dim: 6,
        });
        id
    }
}

/// Solves the IR `build(σ_px)` returns (with its initial values), whose
/// robot-pose priors are weighed against pixel noise `σ_px`: once at
/// `σ_px = 1`, then from that solution at the `σ̂` of its reprojection
/// residuals (floored at [`MIN_PIXEL_SIGMA`]). Returns the IR and solution of
/// the second solve.
///
/// With reprojection residuals whitened by stated covariances, that `σ̂` is
/// the dimensionless `σ̂₀`, the whitened residuals' standard deviation, so the
/// priors are weighed against them in the same way.
pub(crate) fn solve_weighted<F>(
    build: F,
    backend_opts: &BackendSolveOptions,
) -> Result<(ProblemIR, BackendSolution), Error>
where
    F: Fn(f64) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error>,
{
    let (ir, initial) = build(1.0)?;
    let first = backend::solve(&ir, &initial, backend_opts)?;
    let pixel_sigma = first
        .noise
        .iter()
        .find(|n| n.family == ResidualKind::Reprojection)
        .map(|n| n.sigma)
        .ok_or_else(|| Error::numerical("no reprojection residuals to weigh the robot poses"))?
        .max(MIN_PIXEL_SIGMA);
    let (ir, _) = build(pixel_sigma)?;
    let second = backend::solve(&ir, &first.params, backend_opts)?;
    Ok((ir, second))
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
