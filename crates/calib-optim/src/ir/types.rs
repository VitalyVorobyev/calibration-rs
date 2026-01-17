use anyhow::{anyhow, ensure, Result};
use std::collections::HashSet;

/// Identifier for a parameter block in the IR.
///
/// This is stable within a `ProblemIR` instance and is used by residual blocks
/// to reference their parameter dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub usize);

/// Supported manifold types for parameter blocks.
///
/// Each variant implies an expected ambient parameter dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldKind {
    /// Standard Euclidean vector space.
    Euclidean,
    /// SE(3) pose stored as `[qx, qy, qz, qw, tx, ty, tz]`.
    SE3,
    /// SO(3) rotation stored as quaternion `[qx, qy, qz, qw]`.
    SO3,
    /// S2 unit sphere stored as `[x, y, z]`.
    S2,
}

impl ManifoldKind {
    /// Returns `true` if the given ambient dimension matches the manifold storage.
    pub fn compatible_dim(self, dim: usize) -> bool {
        match self {
            ManifoldKind::Euclidean => true,
            ManifoldKind::SE3 => dim == 7,
            ManifoldKind::SO3 => dim == 4,
            ManifoldKind::S2 => dim == 3,
        }
    }
}

/// Bounds for a single parameter index.
///
/// Bounds are applied after each update in backends that support them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    pub idx: usize,
    pub lower: f64,
    pub upper: f64,
}

/// Fixed parameter mask for a block.
///
/// Backends interpret this as per-index fixing for Euclidean blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedMask {
    fixed_indices: HashSet<usize>,
}

impl FixedMask {
    /// Creates a mask with no fixed indices.
    pub fn all_free() -> Self {
        Self {
            fixed_indices: HashSet::new(),
        }
    }

    /// Creates a mask with all indices fixed.
    pub fn all_fixed(dim: usize) -> Self {
        Self {
            fixed_indices: (0..dim).collect(),
        }
    }

    /// Creates a mask from an explicit list of indices.
    pub fn fix_indices(indices: &[usize]) -> Self {
        Self {
            fixed_indices: indices.iter().copied().collect(),
        }
    }

    /// Returns `true` if the index is fixed.
    pub fn is_fixed(&self, idx: usize) -> bool {
        self.fixed_indices.contains(&idx)
    }

    /// Returns `true` if all indices `[0, dim)` are fixed.
    pub fn is_all_fixed(&self, dim: usize) -> bool {
        self.fixed_indices.len() == dim
    }

    /// Iterates over fixed indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.fixed_indices.iter().copied()
    }

    /// Returns `true` if no indices are fixed.
    pub fn is_empty(&self) -> bool {
        self.fixed_indices.is_empty()
    }
}

/// Robust loss applied to a residual block.
///
/// Each residual block has its own loss; per-point robustification is achieved
/// by using one residual block per observation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustLoss {
    #[default]
    None,
    Huber {
        scale: f64,
    },
    Cauchy {
        scale: f64,
    },
    Arctan {
        scale: f64,
    },
}

/// Hand-eye calibration mode.
///
/// Specifies the transform chain used for hand-eye calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandEyeMode {
    /// Camera mounted on robot end-effector (gripper).
    ///
    /// Transform chain: P_camera = extr^-1 * handeye^-1 * robot^-1 * target * P_world
    EyeInHand,
    /// Camera fixed in workspace, observes robot end-effector.
    ///
    /// Transform chain: P_camera = extr^-1 * handeye * robot * target * P_world
    EyeToHand,
}

/// Backend-agnostic factor kinds.
///
/// Each factor kind implies its parameter layout and residual dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum FactorKind {
    /// Reprojection residual for a pinhole camera with 4 intrinsics and an SE3 pose.
    ReprojPointPinhole4 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    /// Reprojection residual with pinhole intrinsics, Brown-Conrady distortion, and SE3 pose.
    ReprojPointPinhole4Dist5 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    /// Reprojection with pinhole, distortion, Scheimpflug sensor, and SE3 pose.
    ReprojPointPinhole4Dist5Scheimpflug2 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    /// Reprojection with two composed SE3 transforms for rig extrinsics.
    ///
    /// Parameters: [intrinsics, distortion, extr_camera_to_rig, pose_rig_to_target]
    /// Transform chain: P_camera = extr^-1 * pose * P_world
    ReprojPointPinhole4Dist5TwoSE3 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    /// Reprojection for hand-eye calibration with robot pose as measurement.
    ///
    /// Parameters: [intrinsics, distortion, extr, handeye, target]
    /// Robot pose (base-to-gripper) is stored in the factor as known data.
    ReprojPointPinhole4Dist5HandEye {
        pw: [f64; 3],
        uv: [f64; 2],
        w: f64,
        base_to_gripper_se3: [f64; 7],
        mode: HandEyeMode,
    },
    /// Reprojection for hand-eye calibration with per-view robot pose correction.
    ///
    /// Parameters: [intrinsics, distortion, extr, handeye, target, robot_delta]
    /// Robot pose (base-to-gripper) is stored in the factor as known data.
    /// `robot_delta` is a 6D se(3) tangent correction applied via exp(delta) * T_B_E.
    ReprojPointPinhole4Dist5HandEyeRobotDelta {
        pw: [f64; 3],
        uv: [f64; 2],
        w: f64,
        base_to_gripper_se3: [f64; 7],
        mode: HandEyeMode,
    },
    /// Laser line pixel constrained to lie on laser plane.
    ///
    /// Parameters: [intrinsics, distortion, pose_cam_to_target, plane_normal, plane_distance]
    /// Residual: point-to-plane distance for ray-target intersection point.
    /// Note: Target is always planar (Z=0), so 3D point is computed as ray intersection.
    LaserPlanePixel { laser_pixel: [f64; 2], w: f64 },
    /// Laser line pixel constrained by line-distance in normalized plane.
    ///
    /// Parameters: [intrinsics, distortion, pose_cam_to_target, plane_normal, plane_distance]
    /// Residual: perpendicular distance from normalized pixel to projected
    ///           laser-target intersection line, scaled by sqrt(fx*fy).
    /// Note: Alternative to LaserPlanePixel using normalized plane geometry.
    LaserLineDist2D { laser_pixel: [f64; 2], w: f64 },
    /// Placeholder for future prior factors.
    Prior,
    /// Zero-mean prior on a 6D se(3) tangent vector.
    ///
    /// Parameters: \[se3_delta\] (6D Euclidean).
    Se3TangentPrior { sqrt_info: [f64; 6] },
    /// Placeholder for future distortion-aware reprojection.
    ReprojPointWithDistortion,
}

impl FactorKind {
    /// Residual dimension implied by the factor.
    pub fn residual_dim(&self) -> usize {
        match self {
            FactorKind::ReprojPointPinhole4 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5TwoSE3 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5HandEye { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5HandEyeRobotDelta { .. } => 2,
            FactorKind::LaserPlanePixel { .. } => 1,
            FactorKind::LaserLineDist2D { .. } => 1,
            FactorKind::Prior => 0,
            FactorKind::Se3TangentPrior { .. } => 6,
            FactorKind::ReprojPointWithDistortion => 2,
        }
    }
}

/// Parameter block definition in the IR.
///
/// This describes the storage layout and constraints for a single variable.
#[derive(Debug, Clone)]
pub struct ParamBlock {
    pub id: ParamId,
    pub name: String,
    pub dim: usize,
    pub manifold: ManifoldKind,
    pub fixed: FixedMask,
    pub bounds: Option<Vec<Bound>>,
}

/// Residual block definition in the IR.
///
/// The order of `params` must match the factor's expected parameter order.
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    pub params: Vec<ParamId>,
    pub loss: RobustLoss,
    pub factor: FactorKind,
    pub residual_dim: usize,
}

/// Backend-agnostic optimization problem representation.
///
/// Backends compile this IR into solver-specific problems.
#[derive(Debug, Default, Clone)]
pub struct ProblemIR {
    pub params: Vec<ParamBlock>,
    pub residuals: Vec<ResidualBlock>,
}

impl ProblemIR {
    /// Creates an empty IR.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a parameter block and returns its `ParamId`.
    pub fn add_param_block(
        &mut self,
        name: impl Into<String>,
        dim: usize,
        manifold: ManifoldKind,
        fixed: FixedMask,
        bounds: Option<Vec<Bound>>,
    ) -> ParamId {
        let id = ParamId(self.params.len());
        self.params.push(ParamBlock {
            id,
            name: name.into(),
            dim,
            manifold,
            fixed,
            bounds,
        });
        id
    }

    /// Adds a residual block to the IR.
    pub fn add_residual_block(&mut self, residual: ResidualBlock) {
        self.residuals.push(residual);
    }

    /// Finds a parameter by name.
    pub fn param_by_name(&self, name: &str) -> Option<ParamId> {
        self.params.iter().find(|p| p.name == name).map(|p| p.id)
    }

    /// Validates internal consistency and factor expectations.
    pub fn validate(&self) -> Result<()> {
        for (idx, param) in self.params.iter().enumerate() {
            ensure!(
                param.id.0 == idx,
                "param id mismatch: expected {}, got {:?}",
                idx,
                param.id
            );
            ensure!(
                param.manifold.compatible_dim(param.dim),
                "param {} manifold {:?} incompatible with dim {}",
                param.name,
                param.manifold,
                param.dim
            );
            for fixed_idx in param.fixed.iter() {
                ensure!(
                    fixed_idx < param.dim,
                    "param {} fixed index {} out of range",
                    param.name,
                    fixed_idx
                );
            }
            if let Some(bounds) = &param.bounds {
                for bound in bounds {
                    ensure!(
                        bound.idx < param.dim,
                        "param {} bound index {} out of range",
                        param.name,
                        bound.idx
                    );
                    ensure!(
                        bound.lower <= bound.upper,
                        "param {} bound lower {} > upper {}",
                        param.name,
                        bound.lower,
                        bound.upper
                    );
                }
            }
        }

        for (r_idx, residual) in self.residuals.iter().enumerate() {
            ensure!(
                residual.residual_dim == residual.factor.residual_dim(),
                "residual {} dim {} does not match factor expectation {}",
                r_idx,
                residual.residual_dim,
                residual.factor.residual_dim()
            );
            for param in &residual.params {
                ensure!(
                    param.0 < self.params.len(),
                    "residual {} references missing param {:?}",
                    r_idx,
                    param
                );
            }

            match &residual.factor {
                FactorKind::ReprojPointPinhole4 { .. } => {
                    ensure!(
                        residual.params.len() == 2,
                        "reprojection factor requires 2 params"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let pose = &self.params[residual.params[1].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "reprojection factor expects 4D Euclidean intrinsics"
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "reprojection factor expects 7D SE3 pose"
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5 { .. } => {
                    ensure!(
                        residual.params.len() == 3,
                        "distortion reprojection factor requires 3 params [cam, dist, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let pose = &self.params[residual.params[2].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "distortion reprojection expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "distortion reprojection expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "distortion reprojection expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 { .. } => {
                    ensure!(
                        residual.params.len() == 4,
                        "Scheimpflug distortion reprojection factor requires 4 params [cam, dist, sensor, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let pose = &self.params[residual.params[3].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug reprojection expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug reprojection expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug reprojection expects 2D Euclidean sensor, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "Scheimpflug reprojection expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5TwoSE3 { .. } => {
                    ensure!(
                        residual.params.len() == 4,
                        "TwoSE3 factor requires 4 params [cam, dist, extr, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let extr = &self.params[residual.params[2].0];
                    let pose = &self.params[residual.params[3].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "TwoSE3 factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "TwoSE3 factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "TwoSE3 factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "TwoSE3 factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5HandEye { .. } => {
                    ensure!(
                        residual.params.len() == 5,
                        "HandEye factor requires 5 params [cam, dist, extr, handeye, target]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let extr = &self.params[residual.params[2].0];
                    let handeye = &self.params[residual.params[3].0];
                    let target = &self.params[residual.params[4].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 hand-eye transform, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target.dim == 7 && target.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 target pose, got dim={} manifold={:?}",
                        target.dim,
                        target.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5HandEyeRobotDelta { .. } => {
                    ensure!(
                        residual.params.len() == 6,
                        "HandEye factor requires 6 params [cam, dist, extr, handeye, target, robot_delta]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let extr = &self.params[residual.params[2].0];
                    let handeye = &self.params[residual.params[3].0];
                    let target = &self.params[residual.params[4].0];
                    let robot_delta = &self.params[residual.params[5].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 hand-eye transform, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target.dim == 7 && target.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 target pose, got dim={} manifold={:?}",
                        target.dim,
                        target.manifold
                    );
                    ensure!(
                        robot_delta.dim == 6 && robot_delta.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 6D Euclidean robot delta, got dim={} manifold={:?}",
                        robot_delta.dim,
                        robot_delta.manifold
                    );
                }
                FactorKind::LaserPlanePixel { .. } => {
                    ensure!(
                        residual.params.len() == 5,
                        "LaserPlanePixel factor requires 5 params [cam, dist, pose, plane_normal, plane_distance]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let pose = &self.params[residual.params[2].0];
                    let plane_normal = &self.params[residual.params[3].0];
                    let plane_distance = &self.params[residual.params[4].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "LaserPlanePixel factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                    ensure!(
                        plane_normal.dim == 3 && plane_normal.manifold == ManifoldKind::S2,
                        "LaserPlanePixel factor expects 3D S2 plane normal, got dim={} manifold={:?}",
                        plane_normal.dim,
                        plane_normal.manifold
                    );
                    ensure!(
                        plane_distance.dim == 1 && plane_distance.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 1D Euclidean plane distance, got dim={} manifold={:?}",
                        plane_distance.dim,
                        plane_distance.manifold
                    );
                }
                FactorKind::LaserLineDist2D { .. } => {
                    ensure!(
                        residual.params.len() == 5,
                        "LaserLineDist2D factor requires 5 params [cam, dist, pose, plane_normal, plane_distance]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let pose = &self.params[residual.params[2].0];
                    let plane_normal = &self.params[residual.params[3].0];
                    let plane_distance = &self.params[residual.params[4].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "LaserLineDist2D factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                    ensure!(
                        plane_normal.dim == 3 && plane_normal.manifold == ManifoldKind::S2,
                        "LaserLineDist2D factor expects 3D S2 plane normal, got dim={} manifold={:?}",
                        plane_normal.dim,
                        plane_normal.manifold
                    );
                    ensure!(
                        plane_distance.dim == 1 && plane_distance.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 1D Euclidean plane distance, got dim={} manifold={:?}",
                        plane_distance.dim,
                        plane_distance.manifold
                    );
                }
                FactorKind::Se3TangentPrior { .. } => {
                    ensure!(
                        residual.params.len() == 1,
                        "Se3TangentPrior requires 1 param [robot_delta]"
                    );
                    let delta = &self.params[residual.params[0].0];
                    ensure!(
                        delta.dim == 6 && delta.manifold == ManifoldKind::Euclidean,
                        "Se3TangentPrior expects 6D Euclidean delta, got dim={} manifold={:?}",
                        delta.dim,
                        delta.manifold
                    );
                }
                FactorKind::Prior | FactorKind::ReprojPointWithDistortion => {
                    return Err(anyhow!(
                        "factor kind {:?} not implemented in validation",
                        residual.factor
                    ));
                }
            }
        }

        Ok(())
    }
}
