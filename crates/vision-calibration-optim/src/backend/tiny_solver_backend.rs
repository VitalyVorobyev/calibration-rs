use crate::Error;
use crate::backend::covariance::{self, RawLinearization};
use crate::backend::kernels::Kernel;
use crate::backend::lm::LinearizationEngine;
use crate::backend::tiny_solver_manifolds::UnitVector3Manifold;
use crate::backend::{BackendSolution, BackendSolveOptions, OptimBackend, SolveEngine, solve_on};
use crate::ir::{CustomFactor, FactorKind, ManifoldKind, ParamBlock, ProblemIR, RobustLoss};
use faer::sparse::SparseColMat;
use nalgebra::DVector;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tiny_solver::factors::{DualStride, Factor, FactorImpl, STRIDE};
use tiny_solver::loss_functions::{ArctanLoss, CauchyLoss, HuberLoss, Loss};
use tiny_solver::manifold::Manifold;
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::manifold::so3::QuaternionManifold;
use tiny_solver::parameter_block::ParameterBlock;
use tiny_solver::problem::{Problem, SymbolicStructure};
use tiny_solver::{GaussNewtonOptimizer, Optimizer};

/// tiny-solver backend adapter.
#[derive(Debug, Clone, Copy)]
pub struct TinySolverBackend;

impl TinySolverBackend {
    fn compile(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
    ) -> Result<Compiled, Error> {
        ir.validate()?;
        ir.check_initial(initial)?;

        let mut problem = Problem::new();
        let non_finite = Arc::new(AtomicBool::new(false));

        for param in &ir.params {
            let manifold: Option<Arc<dyn Manifold + Sync + Send>> = match param.manifold {
                ManifoldKind::Euclidean => None,
                ManifoldKind::SE3 => Some(Arc::new(SE3Manifold)),
                ManifoldKind::SO3 => Some(Arc::new(QuaternionManifold)),
                ManifoldKind::S2 => Some(Arc::new(UnitVector3Manifold)),
            };
            // A fully fixed manifold block keeps its value without one.
            if let Some(manifold) = manifold {
                if param.fixed.is_empty() {
                    problem.set_variable_manifold(&param.name, manifold);
                } else if !param.fixed.is_all_fixed(param.dim) {
                    return Err(Error::invalid_input(format!(
                        "tiny-solver cannot partially fix {:?} manifold {}",
                        param.manifold, param.name
                    )));
                }
            }

            for idx in param.fixed.iter() {
                problem.fix_variable(&param.name, idx);
            }

            if let Some(bounds) = &param.bounds {
                for bound in bounds {
                    problem.set_variable_bounds(&param.name, bound.idx, bound.lower, bound.upper);
                }
            }
        }

        for residual in &ir.residuals {
            let (factor, loss) = compile_factor(residual);
            let factor = Box::new(Watched {
                factor,
                non_finite: Arc::clone(&non_finite),
            });
            let param_names: Vec<&str> = residual
                .params
                .iter()
                .map(|id| ir.params[id.0].name.as_str())
                .collect();
            problem.add_residual_block(residual.residual_dim, &param_names, factor, loss);
        }

        Ok(Compiled {
            problem,
            non_finite,
        })
    }
}

/// A compiled problem and the flag its factors raise on a non-finite
/// derivative ([`Watched`]).
struct Compiled {
    problem: Problem,
    non_finite: Arc<AtomicBool>,
}

/// A factor that raises `non_finite` when a derivative of its residual is
/// not finite. tiny-solver replaces such a Jacobian entry with 0 (and logs a
/// warning), so a solve would stop where a residual is not differentiable
/// and report a minimum; [`TinySolverEngine::linearize`] reports it instead.
///
/// Every lane is checked, a fixed coordinate's too: forward-mode duals scale
/// all lanes of an operand by the same derivative, so a non-finite one
/// reaches the free coordinates' lanes as well (`0 · ∞` is NaN), as it does
/// on factrs.
struct Watched {
    factor: Box<dyn FactorImpl + Send>,
    non_finite: Arc<AtomicBool>,
}

impl Factor<f64> for Watched {
    fn residual_func(&self, params: &[DVector<f64>]) -> DVector<f64> {
        self.factor.residual_func_f64(params)
    }
}

impl Factor<DualStride> for Watched {
    fn residual_func(&self, params: &[DVector<DualStride>]) -> DVector<DualStride> {
        let residual = self.factor.residual_func_dual(params);
        let finite = residual.iter().all(|value| {
            value
                .eps
                .unwrap_generic(nalgebra::Const::<STRIDE>, nalgebra::Const::<1>)
                .iter()
                .all(|d| d.is_finite())
        });
        if !finite {
            // Read after the linearization's evaluations have joined.
            self.non_finite.store(true, Ordering::Relaxed);
        }
        residual
    }
}

impl OptimBackend for TinySolverBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error> {
        self.solve_with(ir, initial, opts, true)
    }
}

impl TinySolverBackend {
    /// [`OptimBackend::solve`], with the covariance's second-order
    /// prediction only when `second_order` is set.
    pub(super) fn solve_with(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
        second_order: bool,
    ) -> Result<BackendSolution, Error> {
        let compiled = self.compile(ir, initial)?;
        let blocks = compiled.problem.initialize_parameter_blocks(initial);
        let engine = TinySolverEngine::new(&compiled.problem, &compiled.non_finite, &blocks);
        solve_on(ir, &engine, blocks, opts, second_order)
    }
}

impl TinySolverBackend {
    /// The loss-free residual and Jacobian of `ir` at `params`, with each
    /// IR block's columns; `None` when every coordinate is fixed.
    pub(super) fn raw_linearization(
        &self,
        ir: &ProblemIR,
        params: &HashMap<String, DVector<f64>>,
    ) -> Result<Option<RawLinearization>, Error> {
        let compiled = self.compile(&ir.without_losses(), params)?;
        let blocks = compiled.problem.initialize_parameter_blocks(params);
        let engine = TinySolverEngine::new(&compiled.problem, &compiled.non_finite, &blocks);
        Ok(raw_from_engine(ir, &engine, &blocks))
    }
}

/// Residual blocks per chunk of [`ResidualProbe::project`]: the unit of
/// work a thread takes, fixed so that the sums do not depend on the number
/// of threads.
const PROBE_CHUNK: usize = 256;

/// The loss-free residuals of an IR at its solution moved by tangent steps,
/// each reduced by a linear map: what [`covariance`] evaluates the
/// second-order error with.
///
/// It evaluates the IR's factors directly, in fixed chunks of residual
/// blocks spread over threads, and sums each chunk's share in chunk order,
/// so a result does not depend on the thread count. Losses and bounds do not
/// apply: the residuals are raw, and a step is not clamped.
///
/// tiny-solver's `Problem::compute_residuals` gives the same residuals, but
/// per call it looks each block up by name and clones its values for every
/// residual block: one call per step made the prediction 5–6× slower on the
/// regression suite's rigs (2–2.5× with the steps spread over threads), so
/// the factors are evaluated here, over buffers reused across the steps.
pub(crate) struct ResidualProbe<'ir> {
    ir: &'ir ProblemIR,
    /// Per IR block, the tiny-solver block at the solution (with its
    /// manifold).
    blocks: Vec<ParameterBlock>,
    /// Per IR residual block, its first row.
    rows: Vec<usize>,
    num_rows: usize,
}

impl<'ir> ResidualProbe<'ir> {
    /// The probe of `ir` at `params`.
    pub(crate) fn new(
        ir: &'ir ProblemIR,
        params: &HashMap<String, DVector<f64>>,
    ) -> Result<Self, Error> {
        let problem = TinySolverBackend
            .compile(&ir.without_losses(), params)?
            .problem;
        let mut by_name = problem.initialize_parameter_blocks(params);
        let blocks = ir
            .params
            .iter()
            .map(|block| {
                by_name.remove(&block.name).ok_or_else(|| {
                    Error::invalid_input(format!("no parameter block {}", block.name))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut rows = Vec::with_capacity(ir.residuals.len());
        let mut num_rows = 0;
        for residual in &ir.residuals {
            rows.push(num_rows);
            num_rows += residual.residual_dim;
        }
        Ok(Self {
            ir,
            blocks,
            rows,
            num_rows,
        })
    }

    /// For each step (column `k` of `steps`, one tangent step per coordinate
    /// of `coordinates`), `projection · r(θ̂ ⊕ step)`: the residuals at the
    /// solution moved by the step, in IR residual order, reduced by the
    /// `q × m` map `projection`. Column `k` of the result is step `k`'s.
    ///
    /// `None` when a residual is not finite at a step.
    pub(crate) fn project(
        &self,
        coordinates: &[covariance::Coordinate],
        steps: &nalgebra::DMatrix<f64>,
        projection: &nalgebra::DMatrix<f64>,
    ) -> Option<nalgebra::DMatrix<f64>> {
        debug_assert_eq!(steps.nrows(), coordinates.len());
        debug_assert_eq!(projection.ncols(), self.num_rows);
        // Every block's value at every step.
        let moved: Vec<Vec<DVector<f64>>> = (0..steps.ncols())
            .map(|k| {
                let mut tangents: Vec<Option<DVector<f64>>> = vec![None; self.blocks.len()];
                for (c, &value) in coordinates.iter().zip(steps.column(k).iter()) {
                    let block = &self.blocks[c.block];
                    tangents[c.block].get_or_insert_with(|| DVector::zeros(block.tangent_size()))
                        [c.index] = value;
                }
                self.blocks
                    .iter()
                    .zip(tangents)
                    .map(|(block, dx)| match dx {
                        Some(dx) => block.plus_f64(dx.rows(0, dx.len())),
                        None => block.params.clone(),
                    })
                    .collect()
            })
            .collect();
        let chunks: Vec<std::ops::Range<usize>> = (0..self.ir.residuals.len())
            .step_by(PROBE_CHUNK)
            .map(|start| start..(start + PROBE_CHUNK).min(self.ir.residuals.len()))
            .collect();
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(chunks.len())
            .max(1);
        let q = projection.nrows();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut partial: Vec<Option<nalgebra::DMatrix<f64>>> = vec![None; chunks.len()];
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(range) = chunks.get(c) else {
                                break;
                            };
                            done.push((c, self.chunk(range.clone(), &moved, projection, q)));
                        }
                        done
                    })
                })
                .collect();
            for worker in workers {
                for (c, sum) in worker.join().expect("a probe thread does not panic") {
                    partial[c] = sum;
                }
            }
        });
        let mut total = nalgebra::DMatrix::zeros(q, steps.ncols());
        for sum in partial {
            total += sum?;
        }
        Some(total)
    }

    /// One chunk's share of [`Self::project`]; `None` when a residual is not
    /// finite or a factor does not compile.
    fn chunk(
        &self,
        range: std::ops::Range<usize>,
        moved: &[Vec<DVector<f64>>],
        projection: &nalgebra::DMatrix<f64>,
        q: usize,
    ) -> Option<nalgebra::DMatrix<f64>> {
        let factors: Vec<_> = self.ir.residuals[range.clone()]
            .iter()
            .map(|residual| compile_factor(residual).0)
            .collect();
        let mut sum = nalgebra::DMatrix::zeros(q, moved.len());
        let mut scratch: Vec<DVector<f64>> = Vec::new();
        for (k, values) in moved.iter().enumerate() {
            for (factor, b) in factors.iter().zip(range.clone()) {
                let residual = &self.ir.residuals[b];
                // Reuse the buffers: neighbouring blocks read blocks of the
                // same sizes.
                scratch.truncate(residual.params.len());
                for (slot, id) in residual.params.iter().enumerate() {
                    let value = &values[id.0];
                    match scratch.get_mut(slot) {
                        Some(buffer) if buffer.len() == value.len() => buffer.copy_from(value),
                        Some(buffer) => *buffer = value.clone(),
                        None => scratch.push(value.clone()),
                    }
                }
                let r = factor.residual_func_f64(&scratch);
                let row = self.rows[b];
                // Column-major: the projection's column `row + i` and the
                // sum's column `k` are contiguous.
                let out = &mut sum.as_mut_slice()[k * q..(k + 1) * q];
                for (i, &v) in r.iter().enumerate() {
                    if !v.is_finite() {
                        return None;
                    }
                    let column = &projection.as_slice()[(row + i) * q..(row + i + 1) * q];
                    for (o, &p) in out.iter_mut().zip(column) {
                        *o += p * v;
                    }
                }
            }
        }
        Some(sum)
    }
}

/// `engine`'s linearization at `state` with each IR block's columns, for an
/// engine whose problem has no robust loss; `None` when every coordinate is
/// fixed.
fn raw_from_engine(
    ir: &ProblemIR,
    engine: &TinySolverEngine<'_>,
    state: &HashMap<String, ParameterBlock>,
) -> Option<RawLinearization> {
    if engine.dim == 0 {
        return None;
    }
    let (residual, jacobian) = engine.linearize(state);
    let columns = ir
        .params
        .iter()
        .map(|block| block_columns(block, engine.col_idx.get(&block.name).copied()))
        .collect();
    Some(RawLinearization {
        residual,
        jacobian,
        columns,
    })
}

/// The Jacobian column of each tangent coordinate of `block`, whose columns
/// start at `start`: a manifold block's tangent in order (none when fully
/// fixed), a Euclidean block's free components in order.
fn block_columns(block: &ParamBlock, start: Option<usize>) -> Vec<Option<usize>> {
    let tangent = covariance::tangent_dim(block.manifold, block.dim);
    let Some(start) = start else {
        return vec![None; tangent];
    };
    match block.manifold {
        ManifoldKind::Euclidean => {
            let mut next = start;
            (0..block.dim)
                .map(|i| {
                    (!block.fixed.contains(i)).then(|| {
                        next += 1;
                        next - 1
                    })
                })
                .collect()
        }
        _ if block.fixed.is_all_fixed(block.dim) => vec![None; tangent],
        _ => (start..start + tangent).map(Some).collect(),
    }
}

/// [`LinearizationEngine`] over a compiled tiny-solver [`Problem`]: its
/// autodiff residual blocks with the Triggs loss correction, manifolds, fixed
/// indices and bounds.
struct TinySolverEngine<'p> {
    problem: &'p Problem,
    /// Raised by the problem's factors on a non-finite derivative.
    non_finite: &'p AtomicBool,
    col_idx: HashMap<String, usize>,
    /// `None` when every parameter is fixed.
    symbolic: Option<SymbolicStructure>,
    dim: usize,
    /// Block names in a fixed order, so float sums over the blocks do not
    /// depend on hash order.
    block_names: Vec<String>,
}

impl<'p> TinySolverEngine<'p> {
    fn new(
        problem: &'p Problem,
        non_finite: &'p AtomicBool,
        blocks: &HashMap<String, ParameterBlock>,
    ) -> Self {
        let col_idx = problem.get_variable_name_to_col_idx_dict(blocks);
        let dim = total_variable_dimension(blocks, &col_idx);
        let symbolic = (dim > 0).then(|| problem.build_symbolic_structure(blocks, dim, &col_idx));
        let mut block_names: Vec<String> = blocks.keys().cloned().collect();
        block_names.sort_unstable();
        Self {
            problem,
            non_finite,
            col_idx,
            symbolic,
            dim,
            block_names,
        }
    }
}

impl LinearizationEngine for TinySolverEngine<'_> {
    type State = HashMap<String, ParameterBlock>;

    fn free_dim(&self) -> usize {
        self.dim
    }

    fn cost(&self, x: &Self::State) -> f64 {
        self.problem.compute_cost(x)
    }

    fn linearize(&self, x: &Self::State) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
        let symbolic = self
            .symbolic
            .as_ref()
            .expect("linearize is only called with free parameters");
        self.non_finite.store(false, Ordering::Relaxed);
        let (mut residual, jacobian) =
            self.problem
                .compute_residual_and_jacobian(x, &self.col_idx, symbolic);
        if self.non_finite.load(Ordering::Relaxed) {
            // tiny-solver zeroed the entry; the residual carries it.
            residual[(0, 0)] = f64::NAN;
        }
        (residual, jacobian)
    }

    /// tiny-solver's own step, the inverse of the column layout of
    /// `get_variable_name_to_col_idx_dict`: each block's tangent step through
    /// `ParameterBlock::plus_f64` (a Euclidean block's fixed components
    /// skipped), then `ParameterBlock::update_params`, which clamps the bounds
    /// and restores the fixed values. It is a provided method of tiny-solver's
    /// `Optimizer` trait, called on its stateless Gauss-Newton optimizer.
    fn retract(&self, x: &Self::State, dx: &DVector<f64>) -> Self::State {
        let mut out = x.clone();
        GaussNewtonOptimizer::new().apply_dx2(dx, &mut out, &self.col_idx);
        out
    }

    fn norm(&self, x: &Self::State) -> f64 {
        self.block_names
            .iter()
            .map(|name| x[name].params.norm_squared())
            .sum::<f64>()
            .sqrt()
    }
}

impl SolveEngine for TinySolverEngine<'_> {
    const NAME: &'static str = "tiny-solver";

    fn read_back(&self, _ir: &ProblemIR, x: &Self::State) -> HashMap<String, DVector<f64>> {
        params_from_blocks(x)
    }

    fn raw_at(&self, ir: &ProblemIR, x: &Self::State) -> Option<RawLinearization> {
        raw_from_engine(ir, self, x)
    }

    fn raw_linearization(
        ir: &ProblemIR,
        params: &HashMap<String, DVector<f64>>,
    ) -> Result<Option<RawLinearization>, Error> {
        TinySolverBackend.raw_linearization(ir, params)
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
    let compiled = TinySolverBackend.compile(ir, initial).expect("compile IR");
    let blocks = compiled.problem.initialize_parameter_blocks(initial);
    let engine = TinySolverEngine::new(&compiled.problem, &compiled.non_finite, &blocks);
    let (r, j) = engine.linearize(&blocks);
    (
        r.as_ref().into_nalgebra().column(0).clone_owned(),
        j.to_dense().as_ref().into_nalgebra().clone_owned(),
        engine.cost(&blocks),
    )
}

/// The number of Jacobian columns: the free tangent coordinates of the
/// blocks that have columns. A block no residual reads gets none (it stays
/// unchanged), so it is not counted.
fn total_variable_dimension(
    parameter_blocks: &HashMap<String, ParameterBlock>,
    col_idx: &HashMap<String, usize>,
) -> usize {
    parameter_blocks
        .iter()
        .filter(|(name, _)| col_idx.contains_key(*name))
        .map(|(_, p)| {
            if p.manifold.is_some() {
                p.tangent_size()
            } else {
                p.tangent_size() - p.fixed_variables.len()
            }
        })
        .sum()
}

fn params_from_blocks(
    parameter_blocks: &HashMap<String, ParameterBlock>,
) -> HashMap<String, DVector<f64>> {
    parameter_blocks
        .iter()
        .map(|(k, v)| (k.to_owned(), v.params.clone()))
        .collect()
}

/// The tiny-solver loss for an IR loss; scales are checked by
/// [`ProblemIR::validate`].
fn compile_loss(loss: RobustLoss) -> Option<Box<dyn Loss + Send>> {
    match loss {
        RobustLoss::None => None,
        RobustLoss::Huber { scale } => Some(Box::new(HuberLoss::new(scale))),
        RobustLoss::Cauchy { scale } => Some(Box::new(CauchyLoss::new(scale))),
        RobustLoss::Arctan { scale } => Some(Box::new(ArctanLoss::new(scale))),
    }
}

type CompiledFactor = (
    Box<dyn tiny_solver::factors::FactorImpl + Send>,
    Option<Box<dyn Loss + Send>>,
);

fn compile_factor(residual: &crate::ir::ResidualBlock) -> CompiledFactor {
    let loss = compile_loss(residual.loss);
    let factor: Box<dyn tiny_solver::factors::FactorImpl + Send> =
        match Kernel::new(&residual.factor) {
            Some(kernel) => Box::new(kernel),
            None => {
                let FactorKind::Custom(custom) = &residual.factor else {
                    unreachable!("only a custom residual has no kernel");
                };
                Box::new(TinyCustomFactor(Arc::clone(&custom.factor)))
            }
        };
    (factor, loss)
}

/// A built-in factor at the two scalar types tiny-solver linearizes with.
impl<T: nalgebra::RealField> Factor<T> for Kernel {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        self.eval(params)
    }
}

/// A residual defined outside this crate, evaluated through its
/// object-safe form at the two scalar types tiny-solver linearizes with.
struct TinyCustomFactor(Arc<dyn CustomFactor>);

impl Factor<f64> for TinyCustomFactor {
    fn residual_func(&self, params: &[DVector<f64>]) -> DVector<f64> {
        self.0.eval_f64(params)
    }
}

impl Factor<DualStride> for TinyCustomFactor {
    fn residual_func(&self, params: &[DVector<DualStride>]) -> DVector<DualStride> {
        self.0.eval_dual(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::lm;
    use crate::ir::{DistortionKind, SensorKind};
    use crate::ir::{HandeyeMode, LaserChain, ReprojChain, ReprojSqrtInfo};
    use faer_ext::IntoNalgebra;
    use nalgebra::RealField;

    /// Run the shared LM on a hand-built problem: the solved parameters and
    /// the iteration count.
    fn solve_problem(
        problem: &Problem,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Option<(HashMap<String, DVector<f64>>, usize)> {
        let blocks = problem.initialize_parameter_blocks(initial);
        let non_finite = AtomicBool::new(false);
        let engine = TinySolverEngine::new(problem, &non_finite, &blocks);
        lm::levenberg_marquardt(&engine, blocks, opts)
            .ok()
            .map(|s| (params_from_blocks(&s.state), s.num_iters))
    }

    #[derive(Debug, Clone)]
    struct SquareMinusOneFactor;

    impl<T: RealField> Factor<T> for SquareMinusOneFactor {
        fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
            let x = params[0][0].clone();
            DVector::from_element(1, x.clone() * x - T::from_f64(1.0).unwrap())
        }
    }

    #[test]
    fn lm_retries_rejected_step_instead_of_stopping_at_initial_point() {
        let mut problem = Problem::new();
        problem.add_residual_block(1, &["x"], Box::new(SquareMinusOneFactor), None);

        let mut initial = HashMap::new();
        initial.insert("x".to_owned(), DVector::from_element(1, 0.1));
        let initial_blocks = problem.initialize_parameter_blocks(&initial);
        let initial_error = problem.compute_cost(&initial_blocks);

        let opts = BackendSolveOptions {
            max_iters: 25,
            verbosity: 0,
            min_abs_decrease: Some(0.0),
            min_rel_decrease: Some(0.0),
            min_error: Some(1e-24),
            ..BackendSolveOptions::default()
        };
        let (params, num_iters) = solve_problem(&problem, &initial, &opts)
            .expect("LM should recover after increasing damping");
        let solved_blocks = problem.initialize_parameter_blocks(&params);
        let solved_error = problem.compute_cost(&solved_blocks);
        let x = params["x"][0];
        assert!(
            num_iters > 0 && num_iters <= opts.max_iters,
            "iteration count should be within bounds, got {num_iters}"
        );

        assert!(
            solved_error < initial_error * 1e-8,
            "expected retrying LM to reduce error, initial={initial_error}, solved={solved_error}"
        );
        assert!(
            (x - 1.0).abs() < 1e-6,
            "positive initial point should converge to the positive root, got {x}"
        );
    }

    /// `r = x - y` for one sample `y`.
    #[derive(Debug, Clone)]
    struct OffsetFactor(f64);

    impl<T: RealField> Factor<T> for OffsetFactor {
        fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
            DVector::from_element(1, params[0][0].clone() - T::from_f64(self.0).unwrap())
        }
    }

    /// LM must reach the minimum of the robust objective Σρ(‖r‖²), not stop
    /// where the squared norm of the loss-corrected residuals stalls.
    #[test]
    fn lm_minimizes_the_robust_cost() {
        let samples = [0.0, 0.1, -0.1, 0.05, 0.2, 4.0, 6.0];
        for loss in [
            RobustLoss::Huber { scale: 0.5 },
            RobustLoss::Cauchy { scale: 0.5 },
            RobustLoss::Arctan { scale: 0.5 },
        ] {
            let mut problem = Problem::new();
            for &y in &samples {
                problem.add_residual_block(
                    1,
                    &["x"],
                    Box::new(OffsetFactor(y)),
                    compile_loss(loss),
                );
            }
            let cost_at = |x: f64| {
                let mut values = HashMap::new();
                values.insert("x".to_owned(), DVector::from_element(1, x));
                problem.compute_cost(&problem.initialize_parameter_blocks(&values))
            };
            // Brute-force reference: a fine grid, then golden-section.
            let mut best = (f64::INFINITY, 0.0);
            for i in 0..=8000 {
                let x = -2.0 + 8.0 * i as f64 / 8000.0;
                let c = cost_at(x);
                if c < best.0 {
                    best = (c, x);
                }
            }
            let (mut lo, mut hi) = (best.1 - 1e-3, best.1 + 1e-3);
            let g = (5.0_f64.sqrt() - 1.0) / 2.0;
            for _ in 0..100 {
                let (a, b) = (hi - g * (hi - lo), lo + g * (hi - lo));
                if cost_at(a) < cost_at(b) {
                    hi = b
                } else {
                    lo = a
                }
            }
            let x_ref = 0.5 * (lo + hi);

            let mut initial = HashMap::new();
            initial.insert("x".to_owned(), DVector::from_element(1, 1.5));
            let opts = BackendSolveOptions {
                max_iters: 200,
                min_abs_decrease: Some(0.0),
                min_rel_decrease: Some(0.0),
                min_error: Some(0.0),
                ..BackendSolveOptions::default()
            };
            let (params, _) =
                solve_problem(&problem, &initial, &opts).expect("robust LM converges");
            let x = params["x"][0];
            assert!(
                (x - x_ref).abs() < 1e-6,
                "{loss:?}: LM stopped at {x}, robust minimum is at {x_ref}"
            );
        }
    }

    use crate::ir::{CameraModelSpec, FixedMask, ParamSlotSpec, ResidualBlock};

    /// Compile an IR and evaluate its stacked residual vector at the initial
    /// parameter values.
    fn eval_residuals(ir: &ProblemIR, initial: &HashMap<String, DVector<f64>>) -> DVector<f64> {
        let backend = TinySolverBackend;
        let problem = backend.compile(ir, initial).expect("compile IR").problem;
        let blocks = problem.initialize_parameter_blocks(initial);
        let residuals = problem.compute_residuals(&blocks, true);
        residuals.as_ref().into_nalgebra().column(0).clone_owned()
    }

    fn values_for_role(role: &str) -> DVector<f64> {
        match role {
            "intrinsics" => DVector::from_row_slice(&[812.3, 798.7, 645.2, 357.9]),
            "distortion" => DVector::from_row_slice(&[-0.11, 0.07, 0.012, 0.0015, -0.0023]),
            "sensor" => DVector::from_row_slice(&[0.021, -0.013]),
            "cam_se3_target" | "pose" => {
                DVector::from_row_slice(&[0.051, -0.022, 0.041, 0.997_55, 0.41, 0.21, 0.92])
            }
            "extrinsics" | "cam_se3_rig" | "rig_se3_cam" => {
                DVector::from_row_slice(&[0.021, 0.034, -0.012, 0.999_03, 0.12, -0.05, 0.83])
            }
            "handeye" => {
                DVector::from_row_slice(&[-0.031, 0.018, 0.009, 0.999_24, 0.08, -0.04, 1.12])
            }
            "target" | "target_ref" => {
                DVector::from_row_slice(&[0.04, 0.05, -0.02, 0.997_7, 0.3, 0.4, 0.7])
            }
            "robot_delta" => {
                DVector::from_row_slice(&[0.0012, -0.0021, 0.0033, 0.0006, -0.0011, 0.0024])
            }
            "plane_normal" => {
                let n = nalgebra::Vector3::new(0.09, 0.17, 1.0).normalize();
                DVector::from_row_slice(&[n.x, n.y, n.z])
            }
            "plane_distance" => DVector::from_row_slice(&[-0.33]),
            other => panic!("no fixture value for role {other}"),
        }
    }

    /// Build a 1-residual IR for `factor` whose blocks follow `layout`, plus
    /// the matching initial-value map.
    fn one_residual_ir(
        factor: FactorKind,
        layout: &[ParamSlotSpec],
    ) -> (ProblemIR, HashMap<String, DVector<f64>>) {
        let mut ir = ProblemIR::new();
        let mut initial = HashMap::new();
        let params: Vec<_> = layout
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                let name = format!("{}_{i}", slot.role);
                initial.insert(name.clone(), values_for_role(slot.role));
                ir.add_param_block(name, slot.dim, slot.manifold, FixedMask::all_free(), None)
            })
            .collect();
        ir.add_residual_block(ResidualBlock {
            params,
            loss: RobustLoss::None,
            residual_dim: factor.residual_dim(),
            factor,
        });
        (ir, initial)
    }

    /// Golden-value pins for every camera-model combo and chain through the
    /// backend. The expected values were captured from the enumerated factor
    /// kernels before their removal (bit-identical to the descriptor kernels
    /// on every production path; the no-distortion case differs from the old
    /// z+1e-9 guard by <1e-5 px and was re-captured from the unified kernel).
    #[test]
    fn descriptor_factors_match_golden_values() {
        let pw = [0.113, -0.072, 0.004];
        let uv = [684.2, 341.7];
        let w = 1.7;
        let laser_pixel = [702.0, 391.0];
        let robot_se3 = [0.024, 0.011, 0.032, 0.999_15, 0.51, -0.22, 0.78];
        let mode = HandeyeMode::EyeToHand;

        let p4 = CameraModelSpec::pinhole(DistortionKind::None, SensorKind::Identity);
        let d5 = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
        let d5s2 = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);

        let reproj = |model, chain| FactorKind::ReprojPoint {
            model,
            chain,
            pw,
            uv,
            sqrt_info: ReprojSqrtInfo::IDENTITY.weighted(w),
        };
        let he = ReprojChain::Handeye {
            base_se3_gripper: robot_se3,
            mode,
        };
        let hed = ReprojChain::HandeyeRobotDelta {
            base_se3_gripper: robot_se3,
            mode,
        };
        let laser_he = LaserChain::RigHandeye {
            base_se3_gripper: robot_se3,
            mode,
        };
        let laser_hed = LaserChain::RigHandeyeRobotDelta {
            base_se3_gripper: robot_se3,
            mode,
        };
        let p2p = |chain| FactorKind::LaserPointToPlane {
            model: d5s2,
            chain,
            laser_pixel,
            w,
        };
        let line = |chain| FactorKind::LaserLineDistance {
            model: d5s2,
            chain,
            laser_pixel,
            w,
        };

        let cases: Vec<(FactorKind, Vec<f64>)> = vec![
            (
                reproj(p4, ReprojChain::SinglePose),
                vec![-555.9931118393514, -187.33549025211838],
            ),
            (
                reproj(d5, ReprojChain::SinglePose),
                vec![-535.8459867248937, -182.60588526685396],
            ),
            (
                reproj(d5s2, ReprojChain::SinglePose),
                vec![-542.0979548239526, -184.50807626154352],
            ),
            (
                reproj(d5, ReprojChain::TwoSe3),
                vec![-267220.3026198549, -139976.51715243037],
            ),
            (
                reproj(d5s2, ReprojChain::TwoSe3),
                vec![-2.6721222127527805e17, -1.4001530498032622e17],
            ),
            (
                reproj(d5, he),
                vec![-416.00573485171145, -198.5803973626507],
            ),
            (
                reproj(d5s2, he),
                vec![-420.4333338843965, -200.4142770137691],
            ),
            (
                reproj(d5, hed),
                vec![-414.0226723312016, -197.9186695815422],
            ),
            (
                reproj(d5s2, hed),
                vec![-418.41360965271605, -199.73884917208755],
            ),
            (p2p(LaserChain::SinglePose), vec![0.721897348979456]),
            (line(LaserChain::SinglePose), vec![2841.74611718797]),
            (p2p(laser_he), vec![1.9389626687488233]),
            (line(laser_he), vec![4657.542541258955]),
            (p2p(laser_hed), vec![1.9416558566988162]),
            (line(laser_hed), vec![4651.171929495054]),
        ];

        for (factor, expected) in cases {
            let layout = factor.param_layout();
            let (ir, init) = one_residual_ir(factor.clone(), &layout);
            let r = eval_residuals(&ir, &init);
            assert_eq!(
                r.as_slice(),
                expected.as_slice(),
                "golden residual mismatch for {factor:?}"
            );
        }
    }
}
