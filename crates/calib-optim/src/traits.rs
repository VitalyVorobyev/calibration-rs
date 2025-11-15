use calib-core::Real;
use nalgebra::DMatrix;

/// Generic non-linear least squares problem.
pub trait NonlinearLeastSquares {
    type Param;
    type Residual;

    fn residuals(&self, param: &Self::Param) -> Self::Residual;
    fn jacobian(&self, param: &Self::Param) -> DMatrix<Real>;
}
