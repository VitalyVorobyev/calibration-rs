use crate::{NllsProblem, NllsSolverBackend, SolveOptions, SolveReport};
use calib_core::Real;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{storage::Owned, DMatrix, DVector, Dyn};

struct LmWrapper<'a, P: NllsProblem> {
    problem: &'a P,
    params: DVector<Real>,
}

impl<'a, P: NllsProblem> LeastSquaresProblem<Real, Dyn, Dyn> for LmWrapper<'a, P> {
    type ResidualStorage = Owned<Real, Dyn>;
    type JacobianStorage = Owned<Real, Dyn, Dyn>;
    type ParameterStorage = Owned<Real, Dyn>;

    fn set_params(&mut self, x: &DVector<Real>) {
        self.params.clone_from(x);
    }

    fn params(&self) -> DVector<Real> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<Real>> {
        Some(self.problem.residuals(&self.params))
    }

    fn jacobian(&self) -> Option<DMatrix<Real>> {
        Some(self.problem.jacobian(&self.params))
    }
}

#[derive(Debug, Default, Clone)]
pub struct LmBackend;

impl NllsSolverBackend for LmBackend {
    fn solve<P: NllsProblem>(
        &self,
        problem: &P,
        x0: DVector<Real>,
        opts: &SolveOptions,
    ) -> (DVector<Real>, SolveReport) {
        let lm = LevenbergMarquardt::new()
            .with_ftol(opts.ftol)
            .with_xtol(opts.ftol)
            .with_gtol(opts.gtol)
            .with_patience(opts.max_iters.max(1));

        let wrapper = LmWrapper {
            problem,
            params: x0,
        };

        let (wrapper, report) = lm.minimize(wrapper);
        let x_opt = wrapper.params();

        (
            x_opt,
            SolveReport {
                iterations: report.number_of_evaluations,
                final_cost: report.objective_function,
                converged: report.termination.was_successful(),
            },
        )
    }
}
