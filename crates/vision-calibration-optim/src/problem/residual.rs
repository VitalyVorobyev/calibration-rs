//! The user residual: the [`Residual`] trait, its [`Noise`] model, and the
//! bridge to the backend's object-safe factor.

use nalgebra::{DVector, Isometry3, Quaternion, RealField, Translation3, UnitQuaternion};
use num_dual::DualDVec64;

use crate::factors::reprojection_model::se3_from_block;
use crate::ir::CustomFactor;

/// A residual over parameter blocks, written once for every scalar type.
///
/// `eval` is generic over [`RealField`]: the solver evaluates it at `f64`
/// for the cost and at dual numbers for an exact Jacobian, so a residual
/// needs no derivative of its own. Write it with the operations of
/// `RealField` (and nalgebra types over `T`), lifting constants with
/// `T::from_f64(c).unwrap()`.
///
/// `blocks` holds each block's stored values, in the order the residual was
/// added with: a Euclidean block's components, an SE3 block's `[qx, qy, qz,
/// qw, tx, ty, tz]` (read it with [`se3`]), a unit vector's `[x, y, z]`, a
/// rotation's `[qx, qy, qz, qw]` (read it with [`so3`]).
///
/// `dim` and `block_dims` must not change, and `eval` must return `dim`
/// values and must not panic. A non-finite value is a step the solver
/// rejects, as for any residual; a result of the wrong length counts as one.
///
/// # Example
///
/// A measured distance between two points, each a 3-vector block:
///
/// ```
/// use nalgebra::{DVector, RealField, Vector3};
/// use vision_calibration_optim::problem::Residual;
///
/// struct Distance {
///     measured: f64,
/// }
///
/// impl Residual for Distance {
///     fn dim(&self) -> usize {
///         1
///     }
///     fn block_dims(&self) -> Vec<usize> {
///         vec![3, 3]
///     }
///     fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
///         let a = Vector3::new(blocks[0][0].clone(), blocks[0][1].clone(), blocks[0][2].clone());
///         let b = Vector3::new(blocks[1][0].clone(), blocks[1][1].clone(), blocks[1][2].clone());
///         let measured = T::from_f64(self.measured).unwrap();
///         DVector::from_element(1, (a - b).norm() - measured)
///     }
/// }
/// ```
pub trait Residual: Send + Sync + 'static {
    /// The residual's length.
    fn dim(&self) -> usize;

    /// The ambient dimension of each block the residual reads, in order: 7
    /// for an SE3 block, 3 for a unit vector, 4 for a rotation.
    fn block_dims(&self) -> Vec<usize>;

    /// The residual at the blocks' values.
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T>;
}

/// How a residual's noise enters the parameter covariance.
///
/// The covariance needs the standard deviation of every residual. A
/// residual either has one known in advance (a measurement with a stated
/// σ, a prior) or shares an unknown one with residuals of its kind, which
/// the solve estimates from their scatter at the solution.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Noise {
    /// The residuals of the family with this name share one noise level,
    /// estimated from their scatter at the solution. A family is pooled on
    /// its own, never with the library's residuals (reprojection, laser),
    /// so it needs enough residuals to estimate a spread.
    Estimated(String),
    /// Each scalar residual has this standard deviation, in the residual's
    /// unit. The solve divides the residual by it, so it weighs as much as
    /// its precision says: a residual already divided by its σ takes `1`.
    Stated(f64),
}

/// The SE3 pose a block stores (`[qx, qy, qz, qw, tx, ty, tz]`), at any
/// scalar type: the transform from the frame the pose is of into the frame
/// it is in, such as `camera_se3_target`.
///
/// # Panics
///
/// When `block` has fewer than 7 entries; a residual that declares the
/// block as 7-dimensional is only evaluated with 7.
pub fn se3<T: RealField>(block: &DVector<T>) -> Isometry3<T> {
    let (rotation, translation) = se3_from_block(block);
    Isometry3::from_parts(Translation3::from(translation), rotation)
}

/// The rotation a rotation block stores (`[qx, qy, qz, qw]`), at any scalar
/// type.
///
/// # Panics
///
/// When `block` has fewer than 4 entries; a residual that declares the
/// block as 4-dimensional is only evaluated with 4.
pub fn so3<T: RealField>(block: &DVector<T>) -> UnitQuaternion<T> {
    UnitQuaternion::from_quaternion(Quaternion::new(
        block[3].clone(),
        block[0].clone(),
        block[1].clone(),
        block[2].clone(),
    ))
}

/// A [`Residual`] in the object-safe form the backend stores, with its
/// dimensions read once and the factor it is scaled by.
pub(super) struct Bridge<R> {
    residual: R,
    dim: usize,
    block_dims: Vec<usize>,
    /// `1 / σ` for a residual of known noise, so the solve sees it
    /// whitened; 1 otherwise.
    scale: f64,
}

impl<R: Residual> Bridge<R> {
    pub(super) fn new(residual: R, scale: f64) -> Self {
        Self {
            dim: residual.dim(),
            block_dims: residual.block_dims(),
            residual,
            scale,
        }
    }

    /// The scaled residual, or `NaN`s (a rejected step) when it has the
    /// wrong length: the backend sizes its rows by `dim`.
    fn checked<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let r = self.residual.eval(blocks);
        if r.len() != self.dim {
            DVector::from_element(self.dim, nan())
        } else if self.scale == 1.0 {
            r
        } else {
            r * T::from_f64(self.scale).unwrap()
        }
    }
}

fn nan<T: RealField>() -> T {
    T::from_f64(f64::NAN).unwrap()
}

impl<R: Residual> CustomFactor for Bridge<R> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn block_dims(&self) -> &[usize] {
        &self.block_dims
    }

    fn eval_f64(&self, blocks: &[DVector<f64>]) -> DVector<f64> {
        self.checked(blocks)
    }

    fn eval_dual(&self, blocks: &[DVector<DualDVec64>]) -> DVector<DualDVec64> {
        self.checked(blocks)
    }
}
