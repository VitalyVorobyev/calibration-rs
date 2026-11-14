//! A camera's residual field, held fixed, as factor data.
//!
//! A residual field is a camera block like the others (`[intrinsics,
//! projection?, distortion?, sensor?, field?]`), but it has hundreds of
//! coordinates and no solve estimates it yet. Differentiating through it
//! would widen every camera factor's dual numbers by its size (tiny-solver)
//! or overflow the fused camera variable (factrs), so both backends take a
//! fully fixed field block out of the factor's variables and splice its
//! values back into the kernel's block list as constants. The kernels see
//! the same blocks either way.

use std::sync::Arc;

use nalgebra::{DVector, RealField};

use crate::Error;
use crate::ir::{CameraSpecLayout, FactorKind, ProblemIR, ResidualBlock};

/// A factor's held residual field: where its block sits among the factor's
/// blocks and its values.
#[derive(Debug, Clone)]
pub(super) struct HeldField {
    /// Index of the field block in the factor's block list (right after the
    /// other camera blocks).
    pub(super) at: usize,
    /// IR id of the field block.
    pub(super) id: usize,
    values: Arc<[f64]>,
}

impl HeldField {
    /// The held field of `residual`'s camera, `None` when it has no field.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when the field block is not fully fixed:
    /// estimating a residual field is not supported, or when `values` has no
    /// entry for it.
    pub(super) fn of(
        ir: &ProblemIR,
        residual: &ResidualBlock,
        values: &std::collections::HashMap<String, DVector<f64>>,
    ) -> Result<Option<Self>, Error> {
        let model = match &residual.factor {
            FactorKind::ReprojPoint { model, .. }
            | FactorKind::ReprojFreePoint { model, .. }
            | FactorKind::LaserPointToPlane { model, .. }
            | FactorKind::LaserLineDistance { model, .. } => *model,
            FactorKind::Se3TangentPrior { .. } | FactorKind::Custom(_) => return Ok(None),
        };
        if model.field.dim() == 0 {
            return Ok(None);
        }
        let at = model.num_cam_blocks() - 1;
        let id = residual.params[at].0;
        let block = &ir.params[id];
        if !block.fixed.is_all_fixed(block.dim) {
            return Err(Error::invalid_input(format!(
                "the residual field block {} must be fixed: estimating a residual field \
                 is not supported",
                block.name
            )));
        }
        let v = values.get(&block.name).ok_or_else(|| {
            Error::invalid_input(format!("initial values missing parameter {}", block.name))
        })?;
        Ok(Some(Self {
            at,
            id,
            values: v.as_slice().into(),
        }))
    }

    /// The field's values as a block of `T` constants.
    pub(super) fn block<T: RealField>(&self) -> DVector<T> {
        DVector::from_iterator(
            self.values.len(),
            self.values
                .iter()
                .map(|&v| T::from_f64(v).expect("a RealField holds every f64 constant")),
        )
    }

    /// `params` (the factor's blocks without the field) with the field
    /// spliced in at its position.
    pub(super) fn splice<T: RealField>(&self, params: &[DVector<T>]) -> Vec<DVector<T>> {
        let mut out = Vec::with_capacity(params.len() + 1);
        out.extend_from_slice(&params[..self.at]);
        out.push(self.block());
        out.extend_from_slice(&params[self.at..]);
        out
    }
}
