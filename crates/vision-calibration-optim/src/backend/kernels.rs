//! The residual of each built-in IR factor, generic over the scalar type:
//! the one kernel set both backends evaluate.
//!
//! A backend adapts [`Kernel`] to its own factor trait: tiny-solver
//! evaluates it at `f64` and at its dual number, factrs at its static-size
//! dual after unpacking its fused variables into the IR's parameter blocks.
//! A new built-in factor adds its arm to [`Kernel::eval`] here and its
//! variable layout to the factrs adapter.

use nalgebra::{DVector, RealField};

use crate::factors::laserline::{laser_line_distance_residual, laser_point_to_plane_residual};
use crate::factors::reprojection_model::{reproj_free_point_residual, reproj_residual};
use crate::ir::{CameraModelSpec, FactorKind};

/// A built-in IR factor, ready to evaluate.
///
/// It is never [`FactorKind::Custom`]: a residual defined outside the crate
/// is evaluated through its own object-safe form, not a generic kernel.
#[derive(Debug, Clone)]
pub(super) struct Kernel(FactorKind);

impl Kernel {
    /// The kernel of `factor`; `None` for a custom residual.
    pub(super) fn new(factor: &FactorKind) -> Option<Self> {
        (!matches!(factor, FactorKind::Custom(_))).then(|| Self(factor.clone()))
    }

    /// The factor this kernel evaluates.
    pub(super) fn factor(&self) -> &FactorKind {
        &self.0
    }

    /// The camera at the head of the parameter blocks; `None` for a factor
    /// without one.
    pub(super) fn camera(&self) -> Option<CameraModelSpec> {
        match &self.0 {
            FactorKind::ReprojPoint { model, .. }
            | FactorKind::ReprojFreePoint { model, .. }
            | FactorKind::LaserPointToPlane { model, .. }
            | FactorKind::LaserLineDistance { model, .. } => Some(*model),
            FactorKind::Se3TangentPrior { .. } => None,
            FactorKind::Custom(_) => unreachable!("Kernel::new rejects custom residuals"),
        }
    }

    /// The residual at the IR parameter blocks `params`, in the factor's
    /// slot order ([`FactorKind::param_layout`]).
    pub(super) fn eval<T: RealField>(&self, params: &[DVector<T>]) -> DVector<T> {
        match &self.0 {
            FactorKind::ReprojPoint {
                model,
                chain,
                pw,
                uv,
                sqrt_info,
            } => {
                let r = reproj_residual(*model, chain, params, *pw, *uv, sqrt_info);
                DVector::from_column_slice(r.as_slice())
            }
            FactorKind::ReprojFreePoint {
                model,
                chain,
                uv,
                sqrt_info,
            } => {
                let r = reproj_free_point_residual(*model, chain, params, *uv, sqrt_info);
                DVector::from_column_slice(r.as_slice())
            }
            FactorKind::LaserPointToPlane {
                model,
                chain,
                laser_pixel,
                w,
            } => {
                let r = laser_point_to_plane_residual(*model, chain, params, *laser_pixel, *w);
                DVector::from_column_slice(r.as_slice())
            }
            FactorKind::LaserLineDistance {
                model,
                chain,
                laser_pixel,
                w,
            } => {
                let r = laser_line_distance_residual(*model, chain, params, *laser_pixel, *w);
                DVector::from_column_slice(r.as_slice())
            }
            FactorKind::Se3TangentPrior { sqrt_info, .. } => {
                debug_assert_eq!(params.len(), 1, "expected [robot_delta] parameter blocks");
                DVector::from_fn(6, |i, _| {
                    params[0][i].clone() * T::from_f64(sqrt_info[i]).unwrap()
                })
            }
            FactorKind::Custom(_) => unreachable!("Kernel::new rejects custom residuals"),
        }
    }
}
