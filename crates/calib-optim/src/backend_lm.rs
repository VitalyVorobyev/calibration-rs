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

#[cfg(test)]
mod tests {
    use super::LmBackend;
    use crate::{NllsProblem, NllsSolverBackend, SolveOptions};
    use calib_core::Real;
    use nalgebra::{DMatrix, DVector};

    #[derive(Debug)]
    struct OneDimProblem;

    impl NllsProblem for OneDimProblem {
        fn residuals(&self, x: &DVector<Real>) -> DVector<Real> {
            DVector::from_element(1, x[0] - 3.0)
        }

        fn jacobian(&self, _x: &DVector<Real>) -> DMatrix<Real> {
            DMatrix::from_element(1, 1, 1.0)
        }
    }

    #[test]
    fn lm_backend_solves_trivial_problem() {
        let backend = LmBackend;
        let problem = OneDimProblem;
        let x0 = DVector::from_element(1, 10.0);
        let opts = SolveOptions::default();

        let (x_opt, report) = backend.solve(&problem, x0, &opts);
        let x_final = x_opt[0];

        assert!(
            (x_final - 3.0).abs() < 1e-6,
            "expected optimizer to reach 3.0, got {}",
            x_final
        );
        assert!(
            report.final_cost.abs() < 1e-12,
            "final cost too high: {}",
            report.final_cost
        );
        assert!(
            report.converged,
            "LM backend did not report convergence: {:?}",
            report
        );
        assert!(
            report.iterations > 0,
            "expected positive iterations, got {}",
            report.iterations
        );
    }
}
