use anyhow::{anyhow, ensure, Result};
use std::collections::HashSet;

/// Identifier for a parameter block in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub usize);

/// Supported manifold types for parameter blocks.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    pub idx: usize,
    pub lower: f64,
    pub upper: f64,
}

/// Fixed parameter mask for a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedMask {
    fixed_indices: HashSet<usize>,
}

impl FixedMask {
    pub fn all_free() -> Self {
        Self {
            fixed_indices: HashSet::new(),
        }
    }

    pub fn all_fixed(dim: usize) -> Self {
        Self {
            fixed_indices: (0..dim).collect(),
        }
    }

    pub fn fix_indices(indices: &[usize]) -> Self {
        Self {
            fixed_indices: indices.iter().copied().collect(),
        }
    }

    pub fn is_fixed(&self, idx: usize) -> bool {
        self.fixed_indices.contains(&idx)
    }

    pub fn is_all_fixed(&self, dim: usize) -> bool {
        self.fixed_indices.len() == dim
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.fixed_indices.iter().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.fixed_indices.is_empty()
    }
}

/// Robust loss applied to a residual block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobustLoss {
    None,
    Huber { scale: f64 },
    Cauchy { scale: f64 },
    Arctan { scale: f64 },
}

impl Default for RobustLoss {
    fn default() -> Self {
        Self::None
    }
}

/// Backend-agnostic factor kinds.
#[derive(Debug, Clone, PartialEq)]
pub enum FactorKind {
    /// Reprojection residual for a pinhole camera with 4 intrinsics and an SE3 pose.
    ReprojPointPinhole4 { pw: [f64; 3], uv: [f64; 2], w: f64 },
    /// Placeholder for future prior factors.
    Prior,
    /// Placeholder for future distortion-aware reprojection.
    ReprojPointWithDistortion,
}

impl FactorKind {
    pub fn residual_dim(&self) -> usize {
        match self {
            FactorKind::ReprojPointPinhole4 { .. } => 2,
            FactorKind::Prior => 0,
            FactorKind::ReprojPointWithDistortion => 2,
        }
    }
}

/// Parameter block definition in the IR.
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
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    pub params: Vec<ParamId>,
    pub loss: RobustLoss,
    pub factor: FactorKind,
    pub residual_dim: usize,
}

/// Backend-agnostic optimization problem representation.
#[derive(Debug, Default, Clone)]
pub struct ProblemIR {
    pub params: Vec<ParamBlock>,
    pub residuals: Vec<ResidualBlock>,
}

impl ProblemIR {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn add_residual_block(&mut self, residual: ResidualBlock) {
        self.residuals.push(residual);
    }

    pub fn param_by_name(&self, name: &str) -> Option<ParamId> {
        self.params.iter().find(|p| p.name == name).map(|p| p.id)
    }

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
