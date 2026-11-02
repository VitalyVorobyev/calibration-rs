//! A residual defined outside this crate, as the IR carries it.
//!
//! The public [`Residual`](crate::problem::Residual) trait has a generic
//! `eval`, so it is not object-safe. [`CustomFactor`] is its object-safe
//! form, evaluated at the two scalar types the backend linearizes with;
//! the problem builder wraps every user residual in it once.

use std::fmt;
use std::sync::Arc;

use nalgebra::DVector;
use num_dual::DualDVec64;
use vision_calibration_core::ResidualKind;

/// A residual over blocks, evaluated at `f64` and at dual numbers.
pub(crate) trait CustomFactor: Send + Sync {
    /// Residual length.
    fn dim(&self) -> usize;
    /// Ambient dimension of each block it reads, in order.
    fn block_dims(&self) -> &[usize];
    /// The residual at `f64` blocks.
    fn eval_f64(&self, blocks: &[DVector<f64>]) -> DVector<f64>;
    /// The residual at dual-number blocks, for the Jacobian.
    fn eval_dual(&self, blocks: &[DVector<DualDVec64>]) -> DVector<DualDVec64>;
}

/// How a custom residual's noise enters the covariance.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CustomNoise {
    /// Pooled into a family whose noise level is estimated from its
    /// residuals; always a [`ResidualKind::Custom`].
    Family(ResidualKind),
    /// Known information, like a prior: the residual is whitened (divided
    /// by its standard deviation), so each scalar residual has unit
    /// variance.
    Whitened,
}

/// A custom factor with its noise model.
#[derive(Clone)]
pub(crate) struct CustomResidual {
    pub factor: Arc<dyn CustomFactor>,
    pub noise: CustomNoise,
}

impl fmt::Debug for CustomResidual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomResidual")
            .field("dim", &self.factor.dim())
            .field("block_dims", &self.factor.block_dims())
            .field("noise", &self.noise)
            .finish()
    }
}

/// Two custom residuals are equal when they share the factor (by pointer)
/// and the noise model: a factor's behaviour is opaque.
impl PartialEq for CustomResidual {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.factor, &other.factor) && self.noise == other.noise
    }
}
