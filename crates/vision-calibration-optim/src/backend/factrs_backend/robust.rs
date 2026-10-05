//! The IR robust losses as a factrs [`RobustCost`].

use factrs::dtype;
use factrs::robust::RobustCost;

use crate::ir::RobustLoss;

/// An IR loss in factrs' convention: `loss = ½ ρ(s)` and `weight = ρ′(s)`,
/// with `ρ` the IR's Ceres-convention loss ([`RobustLoss::rho`]). factrs
/// linearizes it by iterative reweighting: each block's residual and
/// Jacobian are scaled by `√ρ′`.
#[derive(Clone, Debug)]
pub(super) struct IrRobustCost(pub(super) RobustLoss);

impl RobustCost for IrRobustCost {
    fn loss(&self, d2: dtype) -> dtype {
        0.5 * self.0.rho(d2)
    }

    fn weight(&self, d2: dtype) -> dtype {
        self.0.rho_prime(d2)
    }
}
