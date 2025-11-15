use crate::{NllsProblem, NllsSolverBackend, SolveOptions};
use calib_core::{Pt3, Vec2, Real};
use nalgebra::{DMatrix, DVector};

pub struct PlanarIntrinsicsProblem {
    pub points_3d: Vec<Pt3>,
    pub points_2d: Vec<Vec2>,
    // maybe initial intrinsics / distortion, board poses, etc.
}

impl NllsProblem for PlanarIntrinsicsProblem {
    fn residuals(&self, _x: &DVector<Real>) -> DVector<Real> {
        // decode x → intrinsics/distortion, maybe board pose(s)
        // compute projected points & residuals
        // placeholder:
        DVector::zeros(self.points_3d.len() * 2)
    }

    fn jacobian(&self, x: &DVector<Real>) -> DMatrix<Real> {
        // analytic jacobian; start with finite differences if needed
        DMatrix::zeros(self.points_3d.len() * 2, x.len())
    }
}

pub fn refine_intrinsics<B: NllsSolverBackend>(
    backend: &B,
    initial: DVector<Real>,
    problem: &PlanarIntrinsicsProblem,
) -> (DVector<Real>, Real) {
    let opts = SolveOptions::default();
    let (x_opt, report) = backend.solve(problem, initial, &opts);
    (x_opt, report.final_cost)
}
