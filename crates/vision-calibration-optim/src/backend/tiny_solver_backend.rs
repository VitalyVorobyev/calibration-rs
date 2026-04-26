use crate::backend::tiny_solver_manifolds::UnitVector3Manifold;
use crate::backend::{
    BackendSolution, BackendSolveOptions, LinearSolverKind, OptimBackend, SolveReport,
};
use crate::factors::laserline::{
    laser_line_dist_normalized_generic, laser_line_dist_normalized_rig_handeye_residual_generic,
    laser_line_dist_normalized_rig_handeye_robot_delta_residual_generic,
    laser_plane_pixel_residual_generic, laser_plane_pixel_rig_handeye_residual_generic,
    laser_plane_pixel_rig_handeye_robot_delta_residual_generic,
};
use crate::factors::reprojection_model::{
    RobotPoseData, reproj_residual_pinhole4_dist5_handeye_generic,
    reproj_residual_pinhole4_dist5_handeye_robot_delta_generic,
    reproj_residual_pinhole4_dist5_scheimpflug2_handeye_generic,
    reproj_residual_pinhole4_dist5_scheimpflug2_handeye_robot_delta_generic,
    reproj_residual_pinhole4_dist5_scheimpflug2_se3_generic,
    reproj_residual_pinhole4_dist5_scheimpflug2_two_se3_generic,
    reproj_residual_pinhole4_dist5_se3_generic, reproj_residual_pinhole4_dist5_two_se3_generic,
    reproj_residual_pinhole4_se3_generic,
};
use crate::ir::{FactorKind, HandEyeMode, ManifoldKind, ProblemIR, RobustLoss};
use anyhow::{Result, anyhow, ensure};
use faer::sparse::Triplet;
use faer_ext::IntoNalgebra;
use nalgebra::DVector;
use std::collections::HashMap;
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
    ) -> Result<(Problem, HashMap<String, DVector<f64>>)> {
        ir.validate()?;

        let mut problem = Problem::new();

        for param in &ir.params {
            let init = initial.get(&param.name).ok_or_else(|| {
                anyhow!(
                    "initial values missing parameter {} (id {:?})",
                    param.name,
                    param.id
                )
            })?;
            ensure!(
                init.len() == param.dim,
                "initial dimension mismatch for {}: expected {}, got {}",
                param.name,
                param.dim,
                init.len()
            );

            let mut set_manifold = true;

            match param.manifold {
                ManifoldKind::Euclidean => {}
                ManifoldKind::SE3 => {
                    if !param.fixed.is_empty() {
                        if param.fixed.is_all_fixed(param.dim) {
                            set_manifold = false;
                        } else {
                            return Err(anyhow!(
                                "tiny-solver cannot partially fix SE3 manifold {}",
                                param.name
                            ));
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
                            return Err(anyhow!(
                                "tiny-solver cannot partially fix SO3 manifold {}",
                                param.name
                            ));
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
                            return Err(anyhow!(
                                "tiny-solver cannot partially fix S2 manifold {}",
                                param.name
                            ));
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
    ) -> Result<BackendSolution> {
        let (problem, initial_map) = self.compile(ir, initial)?;
        let solution = solve_levenberg_marquardt(&problem, &initial_map, opts)
            .ok_or_else(|| anyhow!("tiny-solver failed to converge"))?;

        let param_blocks = problem.initialize_parameter_blocks(&solution);
        let residuals = problem.compute_residuals(&param_blocks, true);
        let final_cost = 0.5 * residuals.as_ref().squared_norm_l2();

        Ok(BackendSolution {
            params: solution,
            solve_report: SolveReport { final_cost },
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

fn solve_levenberg_marquardt(
    problem: &Problem,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Option<HashMap<String, DVector<f64>>> {
    let opt_options = to_optimizer_options(opts);
    let mut parameter_blocks = problem.initialize_parameter_blocks(initial);
    let variable_name_to_col_idx_dict =
        problem.get_variable_name_to_col_idx_dict(&parameter_blocks);
    let total_variable_dimension = total_variable_dimension(&parameter_blocks);
    if total_variable_dimension == 0 {
        return Some(params_from_blocks(&parameter_blocks));
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

    for outer_iter in 0..opt_options.max_iteration {
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

    Some(params_from_blocks(&parameter_blocks))
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

fn compile_loss(loss: RobustLoss) -> Result<Option<Box<dyn Loss + Send>>> {
    match loss {
        RobustLoss::None => Ok(None),
        RobustLoss::Huber { scale } => {
            ensure!(scale > 0.0, "Huber scale must be positive");
            Ok(Some(Box::new(HuberLoss::new(scale))))
        }
        RobustLoss::Cauchy { scale } => {
            ensure!(scale > 0.0, "Cauchy scale must be positive");
            Ok(Some(Box::new(CauchyLoss::new(scale))))
        }
        RobustLoss::Arctan { scale } => {
            ensure!(scale > 0.0, "Arctan scale must be positive");
            Ok(Some(Box::new(ArctanLoss::new(scale))))
        }
    }
}

type CompiledFactor = (
    Box<dyn tiny_solver::factors::FactorImpl + Send>,
    Option<Box<dyn Loss + Send>>,
);

fn compile_factor(residual: &crate::ir::ResidualBlock) -> Result<CompiledFactor> {
    let loss = compile_loss(residual.loss)?;
    match &residual.factor {
        FactorKind::ReprojPointPinhole4 { pw, uv, w } => {
            let factor = TinyReprojPointFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5 { pw, uv, w } => {
            let factor = TinyReprojPointDistFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 { pw, uv, w } => {
            let factor = TinyReprojPointDistScheimpflugFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5TwoSE3 { pw, uv, w } => {
            let factor = TinyReprojPointDistTwoSE3Factor {
                pw: *pw,
                uv: *uv,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5HandEye {
            pw,
            uv,
            w,
            base_to_gripper_se3,
            mode,
        } => {
            let factor = TinyReprojPointDistHandEyeFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
                robot_se3: *base_to_gripper_se3,
                mode: *mode,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5HandEyeRobotDelta {
            pw,
            uv,
            w,
            base_to_gripper_se3,
            mode,
        } => {
            let factor = TinyReprojPointDistHandEyeDeltaFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
                robot_se3: *base_to_gripper_se3,
                mode: *mode,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5Scheimpflug2TwoSE3 { pw, uv, w } => {
            let factor = TinyReprojPointDistScheimpflugTwoSE3Factor {
                pw: *pw,
                uv: *uv,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEye {
            pw,
            uv,
            w,
            base_to_gripper_se3,
            mode,
        } => {
            let factor = TinyReprojPointDistScheimpflugHandEyeFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
                robot_se3: *base_to_gripper_se3,
                mode: *mode,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta {
            pw,
            uv,
            w,
            base_to_gripper_se3,
            mode,
        } => {
            let factor = TinyReprojPointDistScheimpflugHandEyeDeltaFactor {
                pw: *pw,
                uv: *uv,
                w: *w,
                robot_se3: *base_to_gripper_se3,
                mode: *mode,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::Se3TangentPrior { sqrt_info } => {
            let factor = TinySe3TangentPriorFactor {
                sqrt_info: *sqrt_info,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::LaserPlanePixel { laser_pixel, w } => {
            let factor = TinyLaserPlanePixelFactor {
                laser_pixel: *laser_pixel,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::LaserLineDist2D { laser_pixel, w } => {
            let factor = TinyLaserLineDist2DFactor {
                laser_pixel: *laser_pixel,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::LaserPlanePixelRigHandEye {
            laser_pixel,
            robot_se3,
            mode,
            w,
        } => {
            let factor = TinyLaserPlanePixelRigHandEyeFactor {
                laser_pixel: *laser_pixel,
                robot_se3: *robot_se3,
                mode: *mode,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::LaserPlanePixelRigHandEyeRobotDelta {
            laser_pixel,
            robot_se3,
            mode,
            w,
        } => {
            let factor = TinyLaserPlanePixelRigHandEyeDeltaFactor {
                laser_pixel: *laser_pixel,
                robot_se3: *robot_se3,
                mode: *mode,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::LaserLineDist2DRigHandEye {
            laser_pixel,
            robot_se3,
            mode,
            w,
        } => {
            let factor = TinyLaserLineDist2DRigHandEyeFactor {
                laser_pixel: *laser_pixel,
                robot_se3: *robot_se3,
                mode: *mode,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        FactorKind::LaserLineDist2DRigHandEyeRobotDelta {
            laser_pixel,
            robot_se3,
            mode,
            w,
        } => {
            let factor = TinyLaserLineDist2DRigHandEyeDeltaFactor {
                laser_pixel: *laser_pixel,
                robot_se3: *robot_se3,
                mode: *mode,
                w: *w,
            };
            Ok((Box::new(factor), loss))
        }
        other => Err(anyhow!("factor kind {:?} not supported", other)),
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(params.len(), 2, "expected [cam, pose] parameter blocks");
        let r = reproj_residual_pinhole4_se3_generic(
            params[0].as_view(),
            params[1].as_view(),
            self.pw,
            self.uv,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            3,
            "expected [cam, dist, pose] parameter blocks"
        );
        let r = reproj_residual_pinhole4_dist5_se3_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // pose
            self.pw,
            self.uv,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistScheimpflugFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistScheimpflugFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            4,
            "expected [cam, dist, sensor, pose] parameter blocks"
        );
        let r = reproj_residual_pinhole4_dist5_scheimpflug2_se3_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor (Scheimpflug)
            params[3].as_view(), // pose
            self.pw,
            self.uv,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistTwoSE3Factor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistTwoSE3Factor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            4,
            "expected [cam, dist, extr, pose] parameter blocks"
        );
        let obs = crate::factors::reprojection_model::ObservationData {
            pw: self.pw,
            uv: self.uv,
            w: self.w,
        };
        let r = reproj_residual_pinhole4_dist5_two_se3_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // extr (camera-to-rig)
            params[3].as_view(), // pose (target-to-rig)
            &obs,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistHandEyeFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
    robot_se3: [f64; 7],
    mode: crate::ir::HandEyeMode,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistHandEyeFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            5,
            "expected [cam, dist, extr, handeye, target] parameter blocks"
        );
        let obs = crate::factors::reprojection_model::ObservationData {
            pw: self.pw,
            uv: self.uv,
            w: self.w,
        };
        let robot_data = crate::factors::reprojection_model::RobotPoseData {
            robot_se3: self.robot_se3,
            mode: self.mode,
        };
        let r = reproj_residual_pinhole4_dist5_handeye_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // extr (camera-to-rig)
            params[3].as_view(), // handeye
            params[4].as_view(), // target
            &robot_data,
            &obs,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistHandEyeDeltaFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
    robot_se3: [f64; 7],
    mode: crate::ir::HandEyeMode,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistHandEyeDeltaFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            6,
            "expected [cam, dist, extr, handeye, target, robot_delta] parameter blocks"
        );
        let data = crate::factors::reprojection_model::HandEyeRobotDeltaData {
            robot: crate::factors::reprojection_model::RobotPoseData {
                robot_se3: self.robot_se3,
                mode: self.mode,
            },
            obs: crate::factors::reprojection_model::ObservationData {
                pw: self.pw,
                uv: self.uv,
                w: self.w,
            },
        };
        let r = reproj_residual_pinhole4_dist5_handeye_robot_delta_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // extr (camera-to-rig)
            params[3].as_view(), // handeye
            params[4].as_view(), // target
            params[5].as_view(), // robot delta (se3 tangent)
            &data,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistScheimpflugTwoSE3Factor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistScheimpflugTwoSE3Factor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            5,
            "expected [cam, dist, sensor, extr, pose] parameter blocks"
        );
        let obs = crate::factors::reprojection_model::ObservationData {
            pw: self.pw,
            uv: self.uv,
            w: self.w,
        };
        let r = reproj_residual_pinhole4_dist5_scheimpflug2_two_se3_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor (Scheimpflug)
            params[3].as_view(), // extr (camera-to-rig)
            params[4].as_view(), // pose (target-to-rig)
            &obs,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistScheimpflugHandEyeFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
    robot_se3: [f64; 7],
    mode: crate::ir::HandEyeMode,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistScheimpflugHandEyeFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            6,
            "expected [cam, dist, sensor, extr, handeye, target] parameter blocks"
        );
        let obs = crate::factors::reprojection_model::ObservationData {
            pw: self.pw,
            uv: self.uv,
            w: self.w,
        };
        let robot_data = crate::factors::reprojection_model::RobotPoseData {
            robot_se3: self.robot_se3,
            mode: self.mode,
        };
        let r = reproj_residual_pinhole4_dist5_scheimpflug2_handeye_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor (Scheimpflug)
            params[3].as_view(), // extr
            params[4].as_view(), // handeye
            params[5].as_view(), // target
            &robot_data,
            &obs,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyReprojPointDistScheimpflugHandEyeDeltaFactor {
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
    robot_se3: [f64; 7],
    mode: crate::ir::HandEyeMode,
}

impl<T: nalgebra::RealField> Factor<T> for TinyReprojPointDistScheimpflugHandEyeDeltaFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            7,
            "expected [cam, dist, sensor, extr, handeye, target, robot_delta] parameter blocks"
        );
        let data = crate::factors::reprojection_model::HandEyeRobotDeltaData {
            robot: crate::factors::reprojection_model::RobotPoseData {
                robot_se3: self.robot_se3,
                mode: self.mode,
            },
            obs: crate::factors::reprojection_model::ObservationData {
                pw: self.pw,
                uv: self.uv,
                w: self.w,
            },
        };
        let r = reproj_residual_pinhole4_dist5_scheimpflug2_handeye_robot_delta_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor (Scheimpflug)
            params[3].as_view(), // extr
            params[4].as_view(), // handeye
            params[5].as_view(), // target
            params[6].as_view(), // robot delta (se3 tangent)
            &data,
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

#[derive(Debug, Clone)]
struct TinyLaserPlanePixelFactor {
    laser_pixel: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyLaserPlanePixelFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            6,
            "expected [cam, dist, sensor, pose, plane_normal, plane_distance] parameter blocks"
        );
        let r = laser_plane_pixel_residual_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor (Scheimpflug)
            params[3].as_view(), // pose (camera-to-target)
            params[4].as_view(), // plane normal
            params[5].as_view(), // plane distance
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserLineDist2DFactor {
    laser_pixel: [f64; 2],
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyLaserLineDist2DFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            6,
            "expected [cam, dist, sensor, pose, plane_normal, plane_distance] parameter blocks"
        );
        let r = laser_line_dist_normalized_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor (Scheimpflug)
            params[3].as_view(), // pose (camera-to-target)
            params[4].as_view(), // plane normal
            params[5].as_view(), // plane distance
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserPlanePixelRigHandEyeFactor {
    laser_pixel: [f64; 2],
    robot_se3: [f64; 7],
    mode: HandEyeMode,
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyLaserPlanePixelRigHandEyeFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            8,
            "expected [cam, dist, sensor, cam_to_rig, handeye, target_ref, plane_normal, plane_distance] parameter blocks"
        );
        let robot_data = RobotPoseData {
            robot_se3: self.robot_se3,
            mode: self.mode,
        };
        let r = laser_plane_pixel_rig_handeye_residual_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor
            params[3].as_view(), // cam_to_rig
            params[4].as_view(), // handeye
            params[5].as_view(), // target_ref
            params[6].as_view(), // plane normal
            params[7].as_view(), // plane distance
            robot_data,
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserPlanePixelRigHandEyeDeltaFactor {
    laser_pixel: [f64; 2],
    robot_se3: [f64; 7],
    mode: HandEyeMode,
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyLaserPlanePixelRigHandEyeDeltaFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            9,
            "expected [cam, dist, sensor, cam_to_rig, handeye, target_ref, plane_normal, plane_distance, robot_delta] parameter blocks"
        );
        let robot_data = RobotPoseData {
            robot_se3: self.robot_se3,
            mode: self.mode,
        };
        let r = laser_plane_pixel_rig_handeye_robot_delta_residual_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor
            params[3].as_view(), // cam_to_rig
            params[4].as_view(), // handeye
            params[5].as_view(), // target_ref
            params[6].as_view(), // plane normal
            params[7].as_view(), // plane distance
            params[8].as_view(), // robot delta
            robot_data,
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserLineDist2DRigHandEyeFactor {
    laser_pixel: [f64; 2],
    robot_se3: [f64; 7],
    mode: HandEyeMode,
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyLaserLineDist2DRigHandEyeFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            8,
            "expected [cam, dist, sensor, cam_to_rig, handeye, target_ref, plane_normal, plane_distance] parameter blocks"
        );
        let robot_data = RobotPoseData {
            robot_se3: self.robot_se3,
            mode: self.mode,
        };
        let r = laser_line_dist_normalized_rig_handeye_residual_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor
            params[3].as_view(), // cam_to_rig
            params[4].as_view(), // handeye
            params[5].as_view(), // target_ref
            params[6].as_view(), // plane normal
            params[7].as_view(), // plane distance
            robot_data,
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[derive(Debug, Clone)]
struct TinyLaserLineDist2DRigHandEyeDeltaFactor {
    laser_pixel: [f64; 2],
    robot_se3: [f64; 7],
    mode: HandEyeMode,
    w: f64,
}

impl<T: nalgebra::RealField> Factor<T> for TinyLaserLineDist2DRigHandEyeDeltaFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(
            params.len(),
            9,
            "expected [cam, dist, sensor, cam_to_rig, handeye, target_ref, plane_normal, plane_distance, robot_delta] parameter blocks"
        );
        let robot_data = RobotPoseData {
            robot_se3: self.robot_se3,
            mode: self.mode,
        };
        let r = laser_line_dist_normalized_rig_handeye_robot_delta_residual_generic(
            params[0].as_view(), // intrinsics
            params[1].as_view(), // distortion
            params[2].as_view(), // sensor
            params[3].as_view(), // cam_to_rig
            params[4].as_view(), // handeye
            params[5].as_view(), // target_ref
            params[6].as_view(), // plane normal
            params[7].as_view(), // plane distance
            params[8].as_view(), // robot delta
            robot_data,
            self.laser_pixel,
            self.w,
        );
        DVector::from_row_slice(r.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let solved_blocks = problem.initialize_parameter_blocks(&solution);
        let solved_error = compute_error(&problem, &solved_blocks);
        let x = solution["x"][0];

        assert!(
            solved_error < initial_error * 1e-8,
            "expected retrying LM to reduce error, initial={initial_error}, solved={solved_error}"
        );
        assert!(
            (x - 1.0).abs() < 1e-6,
            "positive initial point should converge to the positive root, got {x}"
        );
    }
}
