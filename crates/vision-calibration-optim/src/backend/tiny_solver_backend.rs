use crate::Error;
use crate::backend::tiny_solver_manifolds::UnitVector3Manifold;
use crate::backend::{
    BackendSolution, BackendSolveOptions, LinearSolverKind, OptimBackend, SolveReport,
};
use crate::factors::camera_kernels::{
    BrownConrady5Kernel, DistortionKernel, DivisionKernel, IdentitySensorKernel,
    NoDistortionKernel, PinholeKernel, ProjectionKernel, RationalKernel, Scheimpflug2Kernel,
    SensorKernel, ThinPrismKernel,
};
use crate::factors::laserline::{
    laser_line_distance_model_generic, laser_point_to_plane_model_generic,
};
use crate::factors::reprojection_model::reproj_residual_model_generic;
use crate::ir::{
    DistortionKind, FactorKind, LaserChain, ManifoldKind, ProblemIR, ProjectionKind, ReprojChain,
    RobustLoss, SensorKind,
};
use faer::sparse::Triplet;
use faer_ext::IntoNalgebra;
use nalgebra::DVector;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Mul;
use std::sync::Arc;
use tiny_solver::factors::Factor;
use tiny_solver::linear::sparse::LinearSolverType;
use tiny_solver::linear::sparse::SparseLinearSolver;
use tiny_solver::linear::{SparseCholeskySolver, SparseQRSolver};
use tiny_solver::loss_functions::{ArctanLoss, CauchyLoss, HuberLoss, Loss};
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::manifold::so3::QuaternionManifold;
use tiny_solver::optimizer::OptimizerOptions;
use tiny_solver::parameter_block::ParameterBlock;
use tiny_solver::problem::Problem;

const LM_MIN_DIAGONAL: f64 = 1e-6;
const LM_MAX_DIAGONAL: f64 = 1e32;
const LM_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;
const LM_MAX_STEP_ATTEMPTS: usize = 32;

/// tiny-solver backend adapter.
#[derive(Debug, Clone, Copy)]
pub struct TinySolverBackend;

impl TinySolverBackend {
    fn compile(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
    ) -> Result<(Problem, HashMap<String, DVector<f64>>), Error> {
        ir.validate()?;

        let mut problem = Problem::new();

        for param in &ir.params {
            let init = initial.get(&param.name).ok_or_else(|| {
                Error::invalid_input(format!(
                    "initial values missing parameter {} (id {:?})",
                    param.name, param.id
                ))
            })?;
            if init.len() != param.dim {
                return Err(Error::invalid_input(format!(
                    "initial dimension mismatch for {}: expected {}, got {}",
                    param.name,
                    param.dim,
                    init.len()
                )));
            }

            let mut set_manifold = true;

            match param.manifold {
                ManifoldKind::Euclidean => {}
                ManifoldKind::SE3 => {
                    if !param.fixed.is_empty() {
                        if param.fixed.is_all_fixed(param.dim) {
                            set_manifold = false;
                        } else {
                            return Err(Error::invalid_input(format!(
                                "tiny-solver cannot partially fix SE3 manifold {}",
                                param.name
                            )));
                        }
                    }
                    if set_manifold {
                        problem.set_variable_manifold(&param.name, Arc::new(SE3Manifold));
                    }
                }
                ManifoldKind::SO3 => {
                    if !param.fixed.is_empty() {
                        if param.fixed.is_all_fixed(param.dim) {
                            set_manifold = false;
                        } else {
                            return Err(Error::invalid_input(format!(
                                "tiny-solver cannot partially fix SO3 manifold {}",
                                param.name
                            )));
                        }
                    }
                    if set_manifold {
                        problem.set_variable_manifold(&param.name, Arc::new(QuaternionManifold));
                    }
                }
                ManifoldKind::S2 => {
                    if !param.fixed.is_empty() {
                        if param.fixed.is_all_fixed(param.dim) {
                            set_manifold = false;
                        } else {
                            return Err(Error::invalid_input(format!(
                                "tiny-solver cannot partially fix S2 manifold {}",
                                param.name
                            )));
                        }
                    }
                    if set_manifold {
                        problem.set_variable_manifold(&param.name, Arc::new(UnitVector3Manifold));
                    }
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
            let (factor, loss) = compile_factor(residual)?;
            let param_names: Vec<String> = residual
                .params
                .iter()
                .map(|id| ir.params[id.0].name.clone())
                .collect();
            let param_refs: Vec<&str> = param_names.iter().map(|s| s.as_str()).collect();
            problem.add_residual_block(residual.residual_dim, &param_refs, factor, loss);
        }

        Ok((problem, initial.clone()))
    }
}

impl OptimBackend for TinySolverBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error> {
        let (problem, initial_map) = self.compile(ir, initial)?;
        let LmSolution { params, num_iters } =
            solve_levenberg_marquardt(&problem, &initial_map, opts)
                .ok_or_else(|| Error::numerical("tiny-solver failed to converge"))?;

        let param_blocks = problem.initialize_parameter_blocks(&params);
        let residuals = problem.compute_residuals(&param_blocks, true);
        let final_cost = 0.5 * residuals.as_ref().squared_norm_l2();

        Ok(BackendSolution {
            params,
            solve_report: SolveReport {
                final_cost,
                num_iters,
            },
        })
    }
}

fn to_optimizer_options(opts: &BackendSolveOptions) -> OptimizerOptions {
    let mut options = OptimizerOptions {
        max_iteration: opts.max_iters,
        verbosity_level: opts.verbosity,
        ..OptimizerOptions::default()
    };
    if let Some(solver) = opts.linear_solver {
        options.linear_solver_type = match solver {
            LinearSolverKind::SparseCholesky => LinearSolverType::SparseCholesky,
            LinearSolverKind::SparseQR => LinearSolverType::SparseQR,
        };
    }
    if let Some(v) = opts.min_abs_decrease {
        options.min_abs_error_decrease_threshold = v;
    }
    if let Some(v) = opts.min_rel_decrease {
        options.min_rel_error_decrease_threshold = v;
    }
    if let Some(v) = opts.min_error {
        options.min_error_threshold = v;
    }
    options
}

/// Outcome of a Levenberg-Marquardt solve: optimized parameters plus the
/// number of outer iterations executed.
struct LmSolution {
    params: HashMap<String, DVector<f64>>,
    num_iters: usize,
}

fn solve_levenberg_marquardt(
    problem: &Problem,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Option<LmSolution> {
    let opt_options = to_optimizer_options(opts);
    let mut parameter_blocks = problem.initialize_parameter_blocks(initial);
    let variable_name_to_col_idx_dict =
        problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
    let total_variable_dimension = total_variable_dimension(&parameter_blocks);
    if total_variable_dimension == 0 {
        return Some(LmSolution {
            params: params_from_blocks(&parameter_blocks),
            num_iters: 0,
        });
    }

    let symbolic_structure = problem.build_symbolic_structure(
        &parameter_blocks,
        total_variable_dimension,
        &variable_name_to_col_idx_dict,
    );
    let mut linear_solver = make_linear_solver(opt_options.linear_solver_type);
    let mut jacobi_scaling_diagonal = None;
    let mut damping = 1.0 / LM_INITIAL_TRUST_REGION_RADIUS;
    let mut current_error = compute_error(problem, &parameter_blocks);
    if !current_error.is_finite() {
        return None;
    }

    let mut num_iters = 0usize;
    for outer_iter in 0..opt_options.max_iteration {
        num_iters = outer_iter + 1;
        let last_error = current_error;
        let (residuals, mut jac) = problem.compute_residual_and_jacobian(
            &parameter_blocks,
            &variable_name_to_col_idx_dict,
            &symbolic_structure,
        );

        if jacobi_scaling_diagonal.is_none() {
            jacobi_scaling_diagonal = Some(build_jacobi_scaling(&jac));
        }
        let scaling = jacobi_scaling_diagonal
            .as_ref()
            .expect("scaling initialized");
        jac = jac * scaling;

        let jtj = jac
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jac.as_ref());
        let jtr = jac.as_ref().transpose().mul(-&residuals);

        let residual_norm2 = residuals.as_ref().squared_norm_l2();
        let mut accepted = false;
        for step_attempt in 0..LM_MAX_STEP_ATTEMPTS {
            let mut jtj_regularized = jtj.clone();
            for i in 0..total_variable_dimension {
                let diag = jtj[(i, i)].clamp(LM_MIN_DIAGONAL, LM_MAX_DIAGONAL);
                jtj_regularized[(i, i)] += damping * diag;
            }

            let Some(lm_step) = linear_solver.solve_jtj(&jtr, &jtj_regularized) else {
                damping *= 2.0;
                continue;
            };
            let dx = scaling * &lm_step;
            let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
            if !dx_na.iter().all(|v| v.is_finite()) {
                damping *= 2.0;
                continue;
            }

            let mut new_param_blocks = parameter_blocks.clone();
            apply_dx(
                &dx_na,
                &mut new_param_blocks,
                &variable_name_to_col_idx_dict,
            );

            let new_residuals = problem.compute_residuals(&new_param_blocks, true);
            let new_residual_norm2 = new_residuals.as_ref().squared_norm_l2();
            let actual_residual_change = residual_norm2 - new_residual_norm2;
            let linear_residual_change: faer::Mat<f64> =
                lm_step.transpose().mul(2.0 * &jtr - &jtj * &lm_step);
            let predicted_residual_change = linear_residual_change[(0, 0)];
            let rho = actual_residual_change / predicted_residual_change;

            if rho.is_finite()
                && rho > 0.0
                && predicted_residual_change > 0.0
                && new_residual_norm2.is_finite()
            {
                parameter_blocks = new_param_blocks;
                current_error = new_residual_norm2;
                let tmp = 2.0 * rho - 1.0;
                damping *= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
                accepted = true;
                if opt_options.verbosity_level > 1 {
                    println!(
                        "tiny-solver lm iter={outer_iter} attempt={step_attempt} error={current_error:.6e} rho={rho:.3e} damping={damping:.3e}"
                    );
                }
                break;
            }

            damping *= 2.0;
        }

        if !accepted {
            if opt_options.verbosity_level > 0 {
                println!(
                    "tiny-solver lm stopped: no accepted step after {LM_MAX_STEP_ATTEMPTS} damping retries"
                );
            }
            break;
        }

        if current_error < opt_options.min_error_threshold {
            break;
        }
        let abs_decrease = (last_error - current_error).abs();
        if abs_decrease < opt_options.min_abs_error_decrease_threshold {
            break;
        }
        if last_error > 0.0
            && abs_decrease / last_error < opt_options.min_rel_error_decrease_threshold
        {
            break;
        }
    }

    Some(LmSolution {
        params: params_from_blocks(&parameter_blocks),
        num_iters,
    })
}

fn make_linear_solver(linear_solver_type: LinearSolverType) -> Box<dyn SparseLinearSolver> {
    match linear_solver_type {
        LinearSolverType::SparseCholesky => Box::new(SparseCholeskySolver::new()),
        LinearSolverType::SparseQR => Box::new(SparseQRSolver::new()),
    }
}

fn total_variable_dimension(parameter_blocks: &HashMap<String, ParameterBlock>) -> usize {
    parameter_blocks
        .values()
        .map(|p| {
            if p.manifold.is_some() {
                p.tangent_size()
            } else {
                p.tangent_size() - p.fixed_variables.len()
            }
        })
        .sum()
}

fn build_jacobi_scaling(
    jac: &faer::sparse::SparseColMat<usize, f64>,
) -> faer::sparse::SparseColMat<usize, f64> {
    let cols = jac.shape().1;
    let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
        .map(|c| {
            let v = jac.val_of_col(c).iter().map(|&i| i * i).sum::<f64>().sqrt();
            Triplet::new(c, c, 1.0 / (1.0 + v))
        })
        .collect();

    faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(cols, cols, &jacobi_scaling_vec)
        .unwrap()
}

fn apply_dx(
    dx: &DVector<f64>,
    params: &mut HashMap<String, ParameterBlock>,
    variable_name_to_col_idx_dict: &HashMap<String, usize>,
) {
    params.iter_mut().for_each(|(key, param)| {
        if let Some(col_idx) = variable_name_to_col_idx_dict.get(key) {
            let tangent_size = param.tangent_size();
            let effective_size = if param.manifold.is_some() {
                tangent_size
            } else {
                tangent_size - param.fixed_variables.len()
            };

            let dx_reduced = dx.rows(*col_idx, effective_size);
            let mut dx_full = DVector::zeros(tangent_size);
            if param.manifold.is_some() {
                dx_full.copy_from(&dx_reduced);
            } else {
                let mut reduced_idx = 0;
                for i in 0..tangent_size {
                    if !param.fixed_variables.contains(&i) {
                        dx_full[i] = dx_reduced[reduced_idx];
                        reduced_idx += 1;
                    }
                }
            }
            param.update_params(param.plus_f64(dx_full.rows(0, tangent_size)));
        }
    });
}

fn compute_error(problem: &Problem, params: &HashMap<String, ParameterBlock>) -> f64 {
    problem
        .compute_residuals(params, true)
        .as_ref()
        .squared_norm_l2()
}

fn params_from_blocks(
    parameter_blocks: &HashMap<String, ParameterBlock>,
) -> HashMap<String, DVector<f64>> {
    parameter_blocks
        .iter()
        .map(|(k, v)| (k.to_owned(), v.params.clone()))
        .collect()
}

fn compile_loss(loss: RobustLoss) -> Result<Option<Box<dyn Loss + Send>>, Error> {
    match loss {
        RobustLoss::None => Ok(None),
        RobustLoss::Huber { scale } => {
            if scale <= 0.0 {
                return Err(Error::invalid_input("Huber scale must be positive"));
            }
            Ok(Some(Box::new(HuberLoss::new(scale))))
        }
        RobustLoss::Cauchy { scale } => {
            if scale <= 0.0 {
                return Err(Error::invalid_input("Cauchy scale must be positive"));
            }
            Ok(Some(Box::new(CauchyLoss::new(scale))))
        }
        RobustLoss::Arctan { scale } => {
            if scale <= 0.0 {
                return Err(Error::invalid_input("Arctan scale must be positive"));
            }
            Ok(Some(Box::new(ArctanLoss::new(scale))))
        }
    }
}

type CompiledFactor = (
    Box<dyn tiny_solver::factors::FactorImpl + Send>,
    Option<Box<dyn Loss + Send>>,
);

/// Camera-model dispatch table: maps a [`CameraModelDesc`](crate::ir::CameraModelDesc)
/// to concrete kernel types and expands `$mk!(P, D, S)` for the matched row.
///
/// Adding a camera model = one descriptor enum variant + one kernel type +
/// one row here. Chains are factor data and do not multiply rows.
macro_rules! dispatch_camera_model {
    ($model:expr, $mk:ident) => {
        match ($model.projection, $model.distortion, $model.sensor) {
            (ProjectionKind::Pinhole, DistortionKind::None, SensorKind::None) => {
                $mk!(PinholeKernel, NoDistortionKernel, IdentitySensorKernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::BrownConrady5, SensorKind::None) => {
                $mk!(PinholeKernel, BrownConrady5Kernel, IdentitySensorKernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::None, SensorKind::Scheimpflug2) => {
                $mk!(PinholeKernel, NoDistortionKernel, Scheimpflug2Kernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::BrownConrady5, SensorKind::Scheimpflug2) => {
                $mk!(PinholeKernel, BrownConrady5Kernel, Scheimpflug2Kernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::Rational8, SensorKind::None) => {
                $mk!(PinholeKernel, RationalKernel, IdentitySensorKernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::Rational8, SensorKind::Scheimpflug2) => {
                $mk!(PinholeKernel, RationalKernel, Scheimpflug2Kernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::ThinPrism9, SensorKind::None) => {
                $mk!(PinholeKernel, ThinPrismKernel, IdentitySensorKernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::ThinPrism9, SensorKind::Scheimpflug2) => {
                $mk!(PinholeKernel, ThinPrismKernel, Scheimpflug2Kernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::Division1, SensorKind::None) => {
                $mk!(PinholeKernel, DivisionKernel, IdentitySensorKernel)
            }
            (ProjectionKind::Pinhole, DistortionKind::Division1, SensorKind::Scheimpflug2) => {
                $mk!(PinholeKernel, DivisionKernel, Scheimpflug2Kernel)
            }
        }
    };
}

fn compile_factor(residual: &crate::ir::ResidualBlock) -> Result<CompiledFactor, Error> {
    let loss = compile_loss(residual.loss)?;
    match &residual.factor {
        FactorKind::Se3TangentPrior { sqrt_info } => {
            let factor = TinySe3TangentPriorFactor {
                sqrt_info: *sqrt_info,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPoint {
            model,
            chain,
            pw,
            uv,
            w,
        } => {
            macro_rules! mk {
                ($P:ty, $D:ty, $S:ty) => {
                    Box::new(TinyReprojFactor::<$P, $D, $S> {
                        chain: *chain,
                        pw: *pw,
                        uv: *uv,
                        w: *w,
                        _kernels: PhantomData,
                    }) as Box<dyn tiny_solver::factors::FactorImpl + Send>
                };
            }
            let factor = dispatch_camera_model!(model, mk);
            Ok((factor, loss))
        }
        FactorKind::LaserPointToPlane {
            model,
            chain,
            laser_pixel,
            w,
        } => {
            macro_rules! mk {
                ($P:ty, $D:ty, $S:ty) => {
                    Box::new(TinyLaserPlaneFactor::<$D, $S> {
                        chain: *chain,
                        laser_pixel: *laser_pixel,
                        w: *w,
                        _kernels: PhantomData,
                    }) as Box<dyn tiny_solver::factors::FactorImpl + Send>
                };
            }
            let factor = dispatch_camera_model!(model, mk);
            Ok((factor, loss))
        }
        FactorKind::LaserLineDistance {
            model,
            chain,
            laser_pixel,
            w,
        } => {
            macro_rules! mk {
                ($P:ty, $D:ty, $S:ty) => {
                    Box::new(TinyLaserLineFactor::<$D, $S> {
                        chain: *chain,
                        laser_pixel: *laser_pixel,
                        w: *w,
                        _kernels: PhantomData,
                    }) as Box<dyn tiny_solver::factors::FactorImpl + Send>
                };
            }
            let factor = dispatch_camera_model!(model, mk);
            Ok((factor, loss))
        }
    }
}

/// Marker tying a factor struct to its camera-model kernel types without
/// storing them (the `fn() -> K` form keeps the struct `Send + Sync`).
type KernelMarker<K> = PhantomData<fn() -> K>;

#[derive(Debug, Clone)]
struct TinyReprojFactor<P, D, S> {
    chain: ReprojChain,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
    _kernels: KernelMarker<(P, D, S)>,
}

impl<P, D, S, T> Factor<T> for TinyReprojFactor<P, D, S>
where
    P: ProjectionKernel,
    D: DistortionKernel,
    S: SensorKernel,
    T: nalgebra::RealField,
{
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let r = reproj_residual_model_generic::<P, D, S, T>(
            &self.chain,
            params,
            self.pw,
            self.uv,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserPlaneFactor<D, S> {
    chain: LaserChain,
    laser_pixel: [f64; 2],
    w: f64,
    _kernels: KernelMarker<(D, S)>,
}

impl<D, S, T> Factor<T> for TinyLaserPlaneFactor<D, S>
where
    D: DistortionKernel,
    S: SensorKernel,
    T: nalgebra::RealField,
{
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let r = laser_point_to_plane_model_generic::<D, S, T>(
            &self.chain,
            params,
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserLineFactor<D, S> {
    chain: LaserChain,
    laser_pixel: [f64; 2],
    w: f64,
    _kernels: KernelMarker<(D, S)>,
}

impl<D, S, T> Factor<T> for TinyLaserLineFactor<D, S>
where
    D: DistortionKernel,
    S: SensorKernel,
    T: nalgebra::RealField,
{
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        let r = laser_line_distance_model_generic::<D, S, T>(
            &self.chain,
            params,
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinySe3TangentPriorFactor {
    sqrt_info: [f64; 6],
}

impl<T: nalgebra::RealField> Factor<T> for TinySe3TangentPriorFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(params.len(), 1, "expected [robot_delta] parameter blocks");
        DVector::from_fn(6, |idx, _| {
            let w = T::from_f64(self.sqrt_info[idx]).unwrap();
            params[0][idx].clone() * w
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::HandEyeMode;
    use nalgebra::RealField;

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
        let initial_error = compute_error(&problem, &initial_blocks);

        let opts = BackendSolveOptions {
            max_iters: 25,
            verbosity: 0,
            linear_solver: Some(LinearSolverKind::SparseCholesky),
            min_abs_decrease: Some(0.0),
            min_rel_decrease: Some(0.0),
            min_error: Some(1e-24),
        };
        let solution = solve_levenberg_marquardt(&problem, &initial, &opts)
            .expect("LM should recover after increasing damping");
        let solved_blocks = problem.initialize_parameter_blocks(&solution.params);
        let solved_error = compute_error(&problem, &solved_blocks);
        let x = solution.params["x"][0];
        assert!(
            solution.num_iters > 0 && solution.num_iters <= opts.max_iters,
            "iteration count should be within bounds, got {}",
            solution.num_iters
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

    use crate::ir::{CameraModelDesc, FixedMask, ParamSlotSpec, ResidualBlock};

    /// Compile an IR and evaluate its stacked residual vector at the initial
    /// parameter values.
    fn eval_residuals(ir: &ProblemIR, initial: &HashMap<String, DVector<f64>>) -> DVector<f64> {
        let backend = TinySolverBackend;
        let (problem, init) = backend.compile(ir, initial).expect("compile IR");
        let blocks = problem.initialize_parameter_blocks(&init);
        let residuals = problem.compute_residuals(&blocks, true);
        residuals.as_ref().into_nalgebra().column(0).clone_owned()
    }

    fn values_for_role(role: &str) -> DVector<f64> {
        match role {
            "intrinsics" => DVector::from_row_slice(&[812.3, 798.7, 645.2, 357.9]),
            "distortion" => DVector::from_row_slice(&[-0.11, 0.07, 0.012, 0.0015, -0.0023]),
            "sensor" => DVector::from_row_slice(&[0.021, -0.013]),
            "camera_se3_target" | "pose" => {
                DVector::from_row_slice(&[0.051, -0.022, 0.041, 0.997_55, 0.41, 0.21, 0.92])
            }
            "extrinsics" | "cam_se3_rig" => {
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
        let mode = HandEyeMode::EyeToHand;

        let p4 = CameraModelDesc::PINHOLE4;
        let d5 = CameraModelDesc::PINHOLE4_DIST5;
        let d5s2 = CameraModelDesc::PINHOLE4_DIST5_SCHEIMPFLUG2;

        let reproj = |model, chain| FactorKind::ReprojPoint {
            model,
            chain,
            pw,
            uv,
            w,
        };
        let he = ReprojChain::HandEye {
            base_se3_gripper: robot_se3,
            mode,
        };
        let hed = ReprojChain::HandEyeRobotDelta {
            base_se3_gripper: robot_se3,
            mode,
        };
        let laser_he = LaserChain::RigHandEye {
            base_se3_gripper: robot_se3,
            mode,
        };
        let laser_hed = LaserChain::RigHandEyeRobotDelta {
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
                vec![-413.2385726371392, -198.46310627517104],
            ),
            (
                reproj(d5s2, hed),
                vec![-417.62274375554296, -200.28853535202757],
            ),
            (p2p(LaserChain::SinglePose), vec![0.721897348979456]),
            (line(LaserChain::SinglePose), vec![2841.74611718797]),
            (p2p(laser_he), vec![1.9389626687488233]),
            (line(laser_he), vec![4657.542541258955]),
            (p2p(laser_hed), vec![1.9424589859523902]),
            (line(laser_hed), vec![4652.567259031746]),
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
