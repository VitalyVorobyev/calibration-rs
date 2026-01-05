use crate::backend::{BackendSolution, BackendSolveOptions, LinearSolverKind, OptimBackend};
use crate::factors::reprojection_model::{
    reproj_residual_pinhole4_dist5_handeye_generic,
    reproj_residual_pinhole4_dist5_scheimpflug2_se3_generic,
    reproj_residual_pinhole4_dist5_se3_generic, reproj_residual_pinhole4_dist5_two_se3_generic,
    reproj_residual_pinhole4_se3_generic,
};
use crate::ir::{FactorKind, ManifoldKind, ProblemIR, RobustLoss};
use anyhow::{anyhow, ensure, Result};
use nalgebra::DVector;
use std::collections::HashMap;
use std::sync::Arc;
use tiny_solver::factors::Factor;
use tiny_solver::loss_functions::{ArctanLoss, CauchyLoss, HuberLoss, Loss};
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::manifold::so3::QuaternionManifold;
use tiny_solver::optimizer::{Optimizer, OptimizerOptions};
use tiny_solver::problem::Problem;
use tiny_solver::{linear::sparse::LinearSolverType, LevenbergMarquardtOptimizer};

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
                    return Err(anyhow!("tiny-solver backend does not support S2 manifolds"));
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
        let optimizer = LevenbergMarquardtOptimizer::default();
        let options = to_optimizer_options(opts);
        let solution = optimizer
            .optimize(&problem, &initial_map, Some(options))
            .ok_or_else(|| anyhow!("tiny-solver failed to converge"))?;

        let param_blocks = problem.initialize_parameter_blocks(&solution);
        let residuals = problem.compute_residuals(&param_blocks, true);
        let final_cost = 0.5 * residuals.as_ref().squared_norm_l2();

        Ok(BackendSolution {
            params: solution,
            final_cost,
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
            params[3].as_view(), // pose (rig-to-target)
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
