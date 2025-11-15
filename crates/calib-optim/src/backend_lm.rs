use crate::{NllsProblem, NllsSolverBackend, SolveOptions, SolveReport};
use calib_core::Real;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector};

struct LmWrapper<'a, P: NllsProblem> {
    problem: &'a P,
}

impl<'a, P: NllsProblem<Param = DVector<Real>, Residual = DVector<Real>>> LeastSquaresProblem
    for LmWrapper<'a, P>
{
    type Parameter = DVector<Real>;
    type Residual = DVector<Real>;
    type Jacobian = DMatrix<Real>;

    fn set_param(&mut self, _p: &Self::Parameter) {
        // LM crate uses supplied parameter; we don't need to store it here
    }

    fn residual(&self, p: &Self::Parameter) -> Self::Residual {
        self.problem.residuals(p)
    }

    fn jacobian(&self, p: &Self::Parameter) -> Self::Jacobian {
        self.problem.jacobian(p)
    }
}

#[derive(Debug, Default, Clone)]
pub struct LmBackend;

impl NllsSolverBackend for LmBackend {
    fn solve<P: NllsProblem<Param = DVector<Real>, Residual = DVector<Real>>>(
        &self,
        problem: &P,
        x0: P::Param,
        opts: &SolveOptions,
    ) -> (P::Param, SolveReport) {
        let mut lm = LevenbergMarquardt::new();
        lm.set_max_iterations(opts.max_iters);
        // lm has more knobs – you can expose them via SolveOptions if needed.

        let wrapper = LmWrapper { problem };
        let (x_opt, report) = lm.minimize(wrapper, x0);

        let final_res = problem.residuals(&x_opt);
        let final_cost = 0.5 * final_res.dot(&final_res);

        (
            x_opt,
            SolveReport {
                iterations: report.iterations(),
                final_cost,
                converged: report.terminated(),
            },
        )
    }
}
