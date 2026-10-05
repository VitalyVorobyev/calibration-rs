//! factrs backend.
//!
//! factrs supplies the variables (with their retractions), forward-mode
//! autodiff with static-size duals, the robust reweighting and the sparse
//! Jacobian assembly. The shared Levenberg–Marquardt loop
//! ([`lm`](super::lm)) drives it, so the two backends take the same kind of
//! step and differ only in how they linearize.
//!
//! factrs' own optimizers are not used: the shared loop also handles rank
//! deficiency, bounds and the iteration count, and keeps both backends'
//! stopping rules identical.

mod compile;
mod residuals;
mod robust;
mod variables;

use std::collections::HashMap;

use factrs::containers::{GraphOrder, Values, ValuesOrder};
use factrs::linalg::DiffResult;
use factrs::linear::LinearValues;
use faer::sparse::SparseColMat;
use nalgebra::DVector;

use crate::Error;
use crate::backend::lm::{self, LinearizationEngine, LmSolution};
use crate::backend::{BackendSolution, BackendSolveOptions, OptimBackend, SolveReport};
use crate::ir::ProblemIR;
use compile::Compiled;

/// factrs backend adapter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FactrsBackend;

impl OptimBackend for FactrsBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error> {
        let compiled = compile::compile(ir, initial)?;
        let engine = FactrsEngine::new(&compiled);
        let LmSolution {
            state,
            cost,
            num_iters,
        } = lm::levenberg_marquardt(&engine, compiled.values.clone(), opts)
            .ok_or_else(|| Error::numerical("factrs failed to converge"))?;

        Ok(BackendSolution {
            params: compiled.read_back(ir, &state),
            solve_report: SolveReport {
                final_cost: 0.5 * cost,
                num_iters,
            },
        })
    }
}

/// Residual, dense Jacobian and robust cost `Σρ` of `ir` at `initial`, as
/// the shared LM sees them.
#[cfg(test)]
pub(super) fn linearize_at(
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
) -> (DVector<f64>, nalgebra::DMatrix<f64>, f64) {
    use faer_ext::IntoNalgebra;
    let compiled = compile::compile(ir, initial).expect("compile IR");
    let engine = FactrsEngine::new(&compiled);
    let (r, j) = engine.linearize(&compiled.values);
    (
        r.as_ref().into_nalgebra().column(0).clone_owned(),
        j.to_dense().as_ref().into_nalgebra().clone_owned(),
        engine.cost(&compiled.values),
    )
}

/// [`LinearizationEngine`] over a compiled factrs graph.
struct FactrsEngine<'c> {
    compiled: &'c Compiled,
    order: ValuesOrder,
    graph_order: GraphOrder,
    /// Jacobian columns of fixed components, zeroed after each
    /// linearization: they decouple from the system, so the step leaves
    /// them exactly unchanged.
    fixed_columns: Vec<usize>,
}

impl<'c> FactrsEngine<'c> {
    fn new(compiled: &'c Compiled) -> Self {
        let order = ValuesOrder::from_values(&compiled.values);
        let graph_order = compiled.graph.sparsity_pattern(order.clone());
        let fixed_columns = compiled.fixed_columns(&order);
        Self {
            compiled,
            order,
            graph_order,
            fixed_columns,
        }
    }
}

impl LinearizationEngine for FactrsEngine<'_> {
    type State = Values;

    fn dim(&self) -> usize {
        self.order.dim()
    }

    fn cost(&self, x: &Values) -> f64 {
        // factrs' error is ½ Σ ρ.
        2.0 * self.compiled.graph.error(x)
    }

    fn linearize(&self, x: &Values) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
        let DiffResult {
            value: b,
            diff: mut jac,
        } = self
            .compiled
            .graph
            .linearize(x)
            .residual_jacobian(&self.graph_order);
        for &c in &self.fixed_columns {
            jac.val_of_col_mut(c).fill(0.0);
        }
        // factrs stores the right-hand side `b = -√w r`.
        let r = faer::Mat::from_fn(b.nrows(), 1, |i, _| -b[(i, 0)]);
        (r, jac)
    }

    fn retract(&self, x: &Values, dx: &DVector<f64>) -> Values {
        let mut out = x.clone();
        out.oplus_mut(&LinearValues::from_order_and_vector(
            self.order.clone(),
            dx.clone(),
        ));
        self.compiled.clamp_bounds(&mut out);
        out
    }

    fn norm(&self, x: &Values) -> f64 {
        self.compiled.ambient_norm(x)
    }
}
