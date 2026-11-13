//! The camera → rig → hand-eye → target chain shared by the hand-eye and the
//! joint rig hand-eye laser problems: the blocks it adds to the IR, its
//! target-corner reprojection residuals, the solve (with or without robot-pose
//! refinement), and reading the chain back from the solution.
//!
//! Each problem adds its own target block(s) and any extra blocks between
//! [`ChainInit::add_blocks`] and the per-view residuals, so the block and
//! residual order of either IR is the order its problem chooses.

use crate::Error;
use crate::backend::{self, BackendSolveOptions, SolveReport};
use crate::ir::{
    CameraModelSpec, FactorKind, FixedMask, HandeyeMode, ManifoldKind, ParamId, ProblemIR,
    ReprojChain, ReprojSqrtInfo, ResidualBlock, RobustLoss,
};
use crate::params::camera::{CameraBlockIds, CameraBlockInput, add_camera_blocks, unpack_camera};
use crate::params::pose_se3::iso3_to_se3_dvec;
use crate::problems::rig_extrinsics::solved_pose;
use crate::problems::robot_poses::{self, RobotPoseSigma, RobotRefinement, Weighting};
use crate::uncertainty::{self, RigLabelling};
use nalgebra::DVector;
use std::collections::HashMap;
use vision_calibration_core::{CameraParams, CorrespondenceView, FixedParams, Iso3, Uncertainty};

/// The initial chain values and which of them are held fixed.
pub(crate) struct ChainInit<'a> {
    /// Per-camera parameters.
    pub cameras: &'a [CameraParams],
    /// Per-camera fixed parameters, resolved against each camera's model.
    pub fixed: &'a [FixedParams],
    /// Per-camera `rig_se3_cam`, one per camera.
    pub rig_se3_cam: &'a [Iso3],
    /// Per-camera extrinsics fix flags; a missing entry is free.
    pub fix_extrinsics: &'a [bool],
    /// The mode's hand-eye pose.
    pub handeye: &'a Iso3,
    /// Hold the hand-eye pose fixed.
    pub fix_handeye: bool,
}

/// The IR blocks of the chain that every view shares.
pub(crate) struct ChainBlocks {
    /// Per-camera camera blocks.
    pub cameras: Vec<CameraBlockIds>,
    /// Per-camera `extr/<i>` blocks.
    pub extr: Vec<ParamId>,
    /// The `handeye` block.
    pub handeye: ParamId,
}

impl ChainInit<'_> {
    /// Adds the camera blocks, then `extr/<i>` per camera, then `handeye`.
    pub(crate) fn add_blocks(
        &self,
        ir: &mut ProblemIR,
        initial_map: &mut HashMap<String, DVector<f64>>,
    ) -> Result<ChainBlocks, Error> {
        let inputs: Vec<CameraBlockInput<'_>> = self
            .cameras
            .iter()
            .zip(self.fixed)
            .map(|(params, fixed)| CameraBlockInput {
                params,
                fixed,
                bounds: None,
            })
            .collect();
        let cameras = add_camera_blocks(ir, initial_map, &inputs, true)?;

        let mut extr = Vec::with_capacity(self.rig_se3_cam.len());
        for (cam_idx, rig_se3_cam) in self.rig_se3_cam.iter().enumerate() {
            let fixed = if self.fix_extrinsics.get(cam_idx).copied().unwrap_or(false) {
                FixedMask::all_fixed(7)
            } else {
                FixedMask::all_free()
            };
            let key = format!("extr/{cam_idx}");
            extr.push(ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None));
            initial_map.insert(key, iso3_to_se3_dvec(rig_se3_cam));
        }

        let handeye_fixed = if self.fix_handeye {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let handeye = ir.add_param_block("handeye", 7, ManifoldKind::SE3, handeye_fixed, None);
        initial_map.insert("handeye".to_string(), iso3_to_se3_dvec(self.handeye));

        Ok(ChainBlocks {
            cameras,
            extr,
            handeye,
        })
    }
}

/// One view's end of the chain: its target block, robot pose and optional
/// robot-pose correction block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewChain {
    /// The view's target block.
    pub target: ParamId,
    /// The view's `robot_delta/<view>` block, with robot-pose refinement.
    pub robot_delta: Option<ParamId>,
    /// The measured `base_se3_gripper`, as `[tx, ty, tz, qx, qy, qz, qw]`.
    pub base_se3_gripper: [f64; 7],
    /// The hand-eye mode.
    pub mode: HandeyeMode,
}

impl ViewChain {
    /// The chain's blocks after the camera's own: `[extr/<cam>, handeye,
    /// target, robot_delta?]`, then `extra` before the robot-pose correction.
    pub(crate) fn params(
        &self,
        blocks: &ChainBlocks,
        cam_idx: usize,
        extra: &[ParamId],
    ) -> Vec<ParamId> {
        let mut chain = vec![blocks.extr[cam_idx], blocks.handeye, self.target];
        chain.extend_from_slice(extra);
        chain.extend(self.robot_delta);
        blocks.cameras[cam_idx].with(&chain)
    }

    /// Adds a reprojection residual for every target corner of every camera
    /// that sees it, in camera then point order, each weighted by `weight`.
    pub(crate) fn add_reprojections(
        &self,
        ir: &mut ProblemIR,
        blocks: &ChainBlocks,
        cameras: &[Option<CorrespondenceView>],
        specs: &[CameraModelSpec],
        loss: RobustLoss,
        weight: f64,
    ) {
        let chain = match self.robot_delta {
            Some(_) => ReprojChain::HandeyeRobotDelta {
                base_se3_gripper: self.base_se3_gripper,
                mode: self.mode,
            },
            None => ReprojChain::Handeye {
                base_se3_gripper: self.base_se3_gripper,
                mode: self.mode,
            },
        };
        for (cam_idx, cam_obs) in cameras.iter().enumerate() {
            let Some(obs) = cam_obs else { continue };
            let params = self.params(blocks, cam_idx, &[]);
            for (pt_idx, (pw, uv)) in obs.points_3d.iter().zip(&obs.points_2d).enumerate() {
                ir.add_residual_block(ResidualBlock {
                    params: params.clone(),
                    loss,
                    factor: FactorKind::ReprojPoint {
                        model: specs[cam_idx],
                        chain,
                        pw: [pw.x, pw.y, pw.z],
                        uv: [uv.x, uv.y],
                        sqrt_info: ReprojSqrtInfo::for_point(obs, pt_idx).weighted(weight),
                    },
                    residual_dim: 2,
                });
            }
        }
    }
}

/// A robot pose as the `[tx, ty, tz, qx, qy, qz, qw]` array a chain factor
/// carries.
pub(crate) fn se3_array(pose: &Iso3) -> [f64; 7] {
    let v = iso3_to_se3_dvec(pose);
    [v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
}

/// How the robot poses enter a chain solve.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RobotSolve {
    /// Refine per-view robot-pose corrections.
    pub refine: bool,
    /// The robot's pose error the corrections are weighed with.
    pub priors: RobotPoseSigma,
    /// Estimate the robot's pose error from the corrections.
    pub estimate: bool,
}

/// A solved chain problem.
pub(crate) struct ChainSolution {
    /// The solved blocks by name.
    pub params: HashMap<String, DVector<f64>>,
    /// The backend's solve report.
    pub report: SolveReport,
    /// The labelled covariance of the solved blocks.
    pub uncertainty: Uncertainty,
    /// The robot pose error the corrections were weighed with, with
    /// robot-pose refinement.
    robot_pose_sigma: Option<RobotPoseSigma>,
}

/// Solves the IR `build` makes: reweighed against the robot-pose priors with
/// robot-pose refinement, a single solve without. The covariance is labelled
/// as a rig hand-eye result's, one camera per entry of `specs`.
pub(crate) fn solve<F>(
    build: F,
    robot: RobotSolve,
    specs: &[CameraModelSpec],
    mode: HandeyeMode,
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<ChainSolution, Error>
where
    F: Fn(Weighting) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error>,
{
    let (ir, solution, robot_pose_sigma) = if robot.refine {
        let (ir, solution, priors) = robot_poses::solve_weighted(
            build,
            robot.priors,
            robot.estimate,
            num_views,
            backend_opts,
        )?;
        (ir, solution, Some(priors))
    } else {
        let (ir, initial_map) = build(Weighting::initial(robot.priors))?;
        let solution = backend::solve(&ir, &initial_map, backend_opts)?;
        (ir, solution, None)
    };
    let labels = RigLabelling {
        cameras: specs,
        multi_camera: specs.len() > 1,
        handeye_mode: Some(mode),
    }
    .labels(&ir, &solution);
    Ok(ChainSolution {
        uncertainty: uncertainty::labelled(&ir, &solution.uncertainty, &solution.noise, &labels),
        params: solution.params,
        report: solution.solve_report,
        robot_pose_sigma,
    })
}

impl ChainSolution {
    /// The per-view robot-pose corrections and the error they were weighed
    /// with; `None` without robot-pose refinement.
    pub(crate) fn robot_refinement(
        &self,
        num_views: usize,
    ) -> Result<Option<RobotRefinement>, Error> {
        self.robot_pose_sigma
            .map(|sigma| {
                Ok(RobotRefinement {
                    deltas: robot_poses::solved_deltas(&self.params, num_views)?,
                    sigma,
                })
            })
            .transpose()
    }
}

/// The solved chain shared by every view.
pub(crate) struct SolvedChain {
    /// Per-camera parameters.
    pub cameras: Vec<CameraParams>,
    /// Per-camera `rig_se3_cam`.
    pub rig_se3_cam: Vec<Iso3>,
    /// The mode's hand-eye pose.
    pub handeye: Iso3,
}

/// Reads the cameras, extrinsics and hand-eye pose back from the solved
/// blocks, one camera per entry of `specs`.
///
/// # Errors
///
/// [`Error::Numerical`] when a block is missing from `solution`.
pub(crate) fn unpack(
    solution: &HashMap<String, DVector<f64>>,
    specs: &[CameraModelSpec],
) -> Result<SolvedChain, Error> {
    let cameras = specs
        .iter()
        .enumerate()
        .map(|(i, spec)| unpack_camera(solution, *spec, Some(i)))
        .collect::<Result<Vec<_>, Error>>()?;
    let rig_se3_cam = (0..specs.len())
        .map(|i| solved_pose(solution, &format!("extr/{i}")))
        .collect::<Result<Vec<_>, Error>>()?;
    let handeye = solved_pose(solution, "handeye")?;
    Ok(SolvedChain {
        cameras,
        rig_se3_cam,
        handeye,
    })
}
