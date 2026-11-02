//! Your own least-squares problem, solved by the library's solver with its
//! covariance.
//!
//! A [`Problem`] holds parameter blocks and the residuals that read them:
//! residuals you write (a [`Residual`]) and the library's own reprojection
//! residual through a camera you add, of a known point or one being
//! estimated, through one pose or a rig chain. A calibration enters with
//! its uncertainty through [`Problem::add_calibrated_rig`]: held fixed with
//! its covariance carried into the solution's ([`Use::Consider`]), or
//! refined under a prior ([`Use::Prior`]). [`solve`] runs the Levenberg–Marquardt
//! loop the calibration workflows use and returns a [`Solution`]: each
//! block's value, the [`SolveReport`], the noise level of each residual
//! family and the covariance of the free parameters, labelled as the
//! workflows' results are, so [`projection_uncertainty`](crate::projection_uncertainty)
//! reads a camera solved here.
//!
//! Every input is checked as it is added: a residual's blocks against the
//! dimensions it declares, and its value at the initial blocks (its length,
//! and that it is finite). A problem that was built is one [`solve`] runs.
//!
//! # Example
//!
//! Fit a line `y = a·x + b` to points, with the noise of `y` estimated:
//!
//! ```
//! use nalgebra::{DVector, RealField};
//! use vision_calibration_optim::RobustLoss;
//! use vision_calibration_optim::problem::{
//!     Manifold, Noise, Problem, Residual, SolveOptions, solve,
//! };
//!
//! struct Point {
//!     x: f64,
//!     y: f64,
//! }
//!
//! impl Residual for Point {
//!     fn dim(&self) -> usize {
//!         1
//!     }
//!     fn block_dims(&self) -> Vec<usize> {
//!         vec![2]
//!     }
//!     fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
//!         let (a, b) = (blocks[0][0].clone(), blocks[0][1].clone());
//!         let x = T::from_f64(self.x).unwrap();
//!         let y = T::from_f64(self.y).unwrap();
//!         DVector::from_element(1, a * x + b - y)
//!     }
//! }
//!
//! let mut problem = Problem::new();
//! let line = problem.add_block("line", &[0.0, 0.0], Manifold::Euclidean)?;
//! problem.set_labels(line, &["a", "b"])?;
//! for (i, noise) in [0.01, -0.02, 0.015, -0.005, 0.0, 0.01].iter().enumerate() {
//!     let x = i as f64;
//!     let point = Point { x, y: 2.0 * x + 1.0 + noise };
//!     problem.add_residual(point, &[line], RobustLoss::None, Noise::Estimated("y".into()))?;
//! }
//! let solution = solve(&problem, &SolveOptions::default())?;
//! let fit = solution.value(line).unwrap();
//! assert!((fit[0] - 2.0).abs() < 0.01 && (fit[1] - 1.0).abs() < 0.03);
//! let covariance = solution.uncertainty.covariance().unwrap();
//! assert_eq!(covariance.labels, ["line.a", "line.b"]);
//! # Ok::<(), vision_calibration_optim::Error>(())
//! ```

mod calibration;
mod residual;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_measurement;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::DVector;
use vision_calibration_core::{
    CameraFix, CameraModelSpec, CameraParams, DistortionParams, FxFyCxCy, Iso3, PointCovariance,
    ProjectionParams, Pt2, Pt3, ResidualKind, ResidualNoise, SensorParams, Uncertainty,
    UncertaintyUnavailable,
};

pub use calibration::{RigBlocks, Use};
pub use residual::{Noise, Residual, se3};

use crate::Error;
use crate::backend::covariance::{self, Coordinate, ParamUncertainty};
use crate::backend::{self, BackendSolveOptions, SolveReport, SolverBackend};
use crate::factors::reprojection_model::{reproj_free_point_residual, reproj_residual};
use crate::ir::{
    Bound, CustomFactor, CustomNoise, CustomResidual, FactorKind, FixedMask, ManifoldKind, ParamId,
    ProblemIR, ReprojChain, ReprojSqrtInfo, ResidualBlock, RobustLoss,
};
use crate::params::camera::{camera_block_labels, camera_spec};
use crate::params::pose_se3::iso3_to_se3_dvec;
use crate::uncertainty::{self, BlockLabels};
use calibration::Calibration;
use residual::Bridge;

/// The slot names of a camera's blocks, in the order of
/// [`CameraParams::blocks`].
const CAMERA_SLOTS: [&str; 4] = ["cam", "proj", "dist", "sensor"];

/// Unit-norm tolerance of an SE3 block's quaternion and a unit vector.
const UNIT_TOLERANCE: f64 = 1e-9;

/// Distinguishes problems, so an id is only used with the problem that
/// issued it.
static NEXT_PROBLEM: AtomicU64 = AtomicU64::new(0);

/// The manifold a parameter block lives on, which sets how it is stored and
/// how a step updates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Manifold {
    /// A vector of any length, updated by addition.
    Euclidean,
    /// A rigid transform, stored `[qx, qy, qz, qw, tx, ty, tz]` with a unit
    /// quaternion and updated on SE(3); its covariance is over the
    /// right-perturbation tangent `[rx, ry, rz, tx, ty, tz]`.
    Se3,
    /// A unit 3-vector `[x, y, z]`, updated on the sphere: a direction such
    /// as a plane normal.
    UnitVector,
}

impl Manifold {
    fn ir(self) -> ManifoldKind {
        match self {
            Manifold::Euclidean => ManifoldKind::Euclidean,
            Manifold::Se3 => ManifoldKind::SE3,
            Manifold::UnitVector => ManifoldKind::S2,
        }
    }
}

/// A parameter block of a [`Problem`], as [`Problem::add_block`] returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId {
    problem: u64,
    index: usize,
}

/// A camera of a [`Problem`], as [`Problem::add_camera`] returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraId {
    problem: u64,
    index: usize,
}

/// What a block is: one of yours, or a slot of a camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    User,
    /// Camera `camera`'s block of slot `slot` (`cam`, `proj`, `dist` or
    /// `sensor`).
    Camera {
        camera: usize,
        slot: &'static str,
    },
}

#[derive(Debug, Clone)]
struct Block {
    name: String,
    values: DVector<f64>,
    manifold: Manifold,
    fixed: FixedMask,
    bounds: Vec<Bound>,
    /// Coordinate names (without the block name); `None` for the default.
    labels: Option<Vec<String>>,
    owner: Owner,
}

impl Block {
    /// Coordinates the covariance reports for the block: a Euclidean
    /// block's components, an SE3 block's tangent, a unit vector's
    /// components.
    fn num_coordinates(&self) -> usize {
        match self.manifold {
            Manifold::Euclidean => self.values.len(),
            Manifold::Se3 => 6,
            Manifold::UnitVector => 3,
        }
    }

    /// The block's coordinate labels in the covariance (see
    /// [`Problem::set_labels`]).
    fn block_labels(&self) -> BlockLabels {
        let names: Vec<String> = match (&self.labels, self.manifold) {
            (Some(names), _) => names.clone(),
            (None, Manifold::Euclidean) => (0..self.values.len()).map(|i| i.to_string()).collect(),
            (None, Manifold::Se3) => ["rx", "ry", "rz", "tx", "ty", "tz"]
                .map(String::from)
                .to_vec(),
            (None, Manifold::UnitVector) => ["nx", "ny", "nz"].map(String::from).to_vec(),
        };
        let coordinates = names.iter().map(|c| format!("{}.{c}", self.name)).collect();
        BlockLabels::shared_owned(self.name.clone(), coordinates)
    }
}

#[derive(Debug, Clone)]
struct Camera {
    spec: CameraModelSpec,
    /// Its blocks in factor order: intrinsics, then projection, distortion
    /// and sensor where the model has them.
    blocks: Vec<BlockId>,
}

/// How a reprojected point reaches the camera frame, as
/// [`Problem::add_reprojection`] takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PoseChain {
    /// One SE3 block, `camera_se3_target`.
    Pose(BlockId),
    /// A camera of a rig: `cam_se3_rig · rig_se3_target`, two SE3 blocks.
    /// [`RigBlocks::chain`] builds it for a calibrated rig.
    Rig {
        /// The camera's pose in the rig frame, `cam_se3_rig`.
        cam_se3_rig: BlockId,
        /// The target's pose in the rig frame, `rig_se3_target`.
        rig_se3_target: BlockId,
    },
}

/// The point a reprojection residual projects, in the target frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum TargetPoint {
    /// A known point, such as a board corner.
    Known(Pt3),
    /// A point being estimated: a Euclidean block of 3 values `[x, y, z]`.
    Block(BlockId),
}

/// The noise of an observed pixel, as [`Problem::add_reprojection`] takes
/// it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PixelNoise {
    /// Shared by the problem's reprojections of estimated noise, and
    /// estimated from their scatter at the solution, in pixels.
    Estimated,
    /// A stated covariance, taken as exact: the residual is whitened by it,
    /// and the covariance counts it as known information. A robust loss
    /// changes the estimate; the covariance takes the residual at its
    /// stated variance.
    Stated(PointCovariance),
}

#[derive(Debug, Clone)]
struct Term {
    factor: FactorKind,
    blocks: Vec<usize>,
    loss: RobustLoss,
}

/// A least-squares problem: parameter blocks and the residuals that read
/// them. See the [module docs](self).
#[derive(Debug)]
pub struct Problem {
    id: u64,
    blocks: Vec<Block>,
    cameras: Vec<Camera>,
    residuals: Vec<Term>,
    calibration: Option<Calibration>,
}

impl Default for Problem {
    fn default() -> Self {
        Self::new()
    }
}

impl Problem {
    /// An empty problem.
    pub fn new() -> Self {
        Self {
            id: NEXT_PROBLEM.fetch_add(1, Ordering::Relaxed),
            blocks: Vec::new(),
            cameras: Vec::new(),
            residuals: Vec::new(),
            calibration: None,
        }
    }

    /// Add a parameter block named `name` with initial values `initial` on
    /// `manifold`.
    ///
    /// The name is unique in the problem and labels the block's coordinates
    /// in the covariance (`name.0`, `name.1`, … until
    /// [`set_labels`](Self::set_labels) names them); `cam` followed by
    /// digits is kept for cameras.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for an empty, repeated or reserved name, a
    /// non-finite value, an empty Euclidean block, an SE3 block that is not
    /// 7 values with a unit quaternion, or a unit vector that is not 3
    /// values of unit norm.
    pub fn add_block(
        &mut self,
        name: impl Into<String>,
        initial: &[f64],
        manifold: Manifold,
    ) -> Result<BlockId, Error> {
        let name = name.into();
        self.check_name(&name)?;
        if initial.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid_input(format!(
                "block {name}: initial values must be finite"
            )));
        }
        let unit = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>().sqrt() - 1.0).abs();
        match manifold {
            Manifold::Euclidean if initial.is_empty() => {
                return Err(Error::invalid_input(format!(
                    "block {name}: a Euclidean block needs at least one value"
                )));
            }
            Manifold::Se3 if initial.len() != 7 || unit(&initial[..4]) > UNIT_TOLERANCE => {
                return Err(Error::invalid_input(format!(
                    "block {name}: an SE3 block is [qx, qy, qz, qw, tx, ty, tz] with a unit \
                     quaternion"
                )));
            }
            Manifold::UnitVector if initial.len() != 3 || unit(initial) > UNIT_TOLERANCE => {
                return Err(Error::invalid_input(format!(
                    "block {name}: a unit vector is [x, y, z] of unit norm"
                )));
            }
            _ => {}
        }
        Ok(self.push_block(Block {
            name,
            values: DVector::from_column_slice(initial),
            manifold,
            fixed: FixedMask::all_free(),
            bounds: Vec::new(),
            labels: None,
            owner: Owner::User,
        }))
    }

    /// Add an SE3 block holding `pose`; see [`add_block`](Self::add_block).
    ///
    /// # Errors
    ///
    /// As [`add_block`](Self::add_block).
    pub fn add_pose(&mut self, name: impl Into<String>, pose: &Iso3) -> Result<BlockId, Error> {
        self.add_block(name, iso3_to_se3_dvec(pose).as_slice(), Manifold::Se3)
    }

    /// Hold the whole block at its initial value.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a block of another problem.
    pub fn fix(&mut self, block: BlockId) -> Result<(), Error> {
        let b = self.block_mut(block)?;
        b.fixed = FixedMask::all_fixed(b.values.len());
        Ok(())
    }

    /// Hold the components `indices` of a Euclidean block at their initial
    /// values.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a block of another problem, a block that
    /// is not Euclidean, or an index out of range.
    pub fn fix_components(&mut self, block: BlockId, indices: &[usize]) -> Result<(), Error> {
        let b = self.euclidean_mut(block)?;
        if let Some(&i) = indices.iter().find(|&&i| i >= b.values.len()) {
            return Err(Error::invalid_input(format!(
                "block {}: component {i} is out of range for {} values",
                b.name,
                b.values.len()
            )));
        }
        let fixed: Vec<usize> = b.fixed.iter().chain(indices.iter().copied()).collect();
        b.fixed = FixedMask::fix_indices(&fixed);
        Ok(())
    }

    /// Keep component `index` of a Euclidean block within `[lower,
    /// upper]`.
    ///
    /// A solution on a bound has no Gaussian covariance: the solution's
    /// uncertainty then names the parameter instead.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a block of another problem, a block that
    /// is not Euclidean, an index out of range, `lower > upper` or a NaN
    /// bound, or an initial value outside the bounds.
    pub fn set_bounds(
        &mut self,
        block: BlockId,
        index: usize,
        lower: f64,
        upper: f64,
    ) -> Result<(), Error> {
        let b = self.euclidean_mut(block)?;
        let name = &b.name;
        let Some(&value) = b.values.as_slice().get(index) else {
            return Err(Error::invalid_input(format!(
                "block {name}: component {index} is out of range"
            )));
        };
        if lower.is_nan() || upper.is_nan() || lower > upper {
            return Err(Error::invalid_input(format!(
                "block {name}: bounds [{lower}, {upper}] are not an interval"
            )));
        }
        if !(lower..=upper).contains(&value) {
            return Err(Error::invalid_input(format!(
                "block {name}: initial value {value} of component {index} is outside \
                 [{lower}, {upper}]"
            )));
        }
        b.bounds.retain(|bound| bound.idx != index);
        b.bounds.push(Bound {
            idx: index,
            lower,
            upper,
        });
        Ok(())
    }

    /// Name the block's coordinates in the covariance: `<block>.<label>`.
    /// A Euclidean block has one per value, an SE3 block six (its tangent;
    /// `rx … tz` by default), a unit vector three (`nx`, `ny`, `nz` by
    /// default).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a block of another problem, the wrong
    /// number of labels, or a label that is empty, repeated or contains a
    /// `.`.
    pub fn set_labels(&mut self, block: BlockId, labels: &[&str]) -> Result<(), Error> {
        let b = self.block_mut(block)?;
        if labels.len() != b.num_coordinates() {
            return Err(Error::invalid_input(format!(
                "block {}: {} labels for {} coordinates",
                b.name,
                labels.len(),
                b.num_coordinates()
            )));
        }
        for (i, label) in labels.iter().enumerate() {
            if label.is_empty() || label.contains('.') || labels[..i].contains(label) {
                return Err(Error::invalid_input(format!(
                    "block {}: label {label:?} is empty, repeated or contains a '.'",
                    b.name
                )));
            }
        }
        b.labels = Some(labels.iter().map(|l| l.to_string()).collect());
        Ok(())
    }

    /// Add a residual you wrote, reading `blocks` in order, with a robust
    /// loss and its noise model.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`], naming the residual by its index, for a
    /// block of another problem, blocks that do not match
    /// [`Residual::block_dims`], a block given twice, an empty residual, a loss with a scale that
    /// is not positive, an empty family name or a known σ that is not
    /// positive and finite, or a value at the initial blocks of the wrong
    /// length or not finite.
    pub fn add_residual<R: Residual>(
        &mut self,
        residual: R,
        blocks: &[BlockId],
        loss: RobustLoss,
        noise: Noise,
    ) -> Result<(), Error> {
        let index = self.residuals.len();
        let fail = |what: String| Error::invalid_input(format!("residual {index}: {what}"));
        let (noise, scale) = match noise {
            Noise::Estimated(family) if family.is_empty() => {
                return Err(fail("a noise family needs a name".into()));
            }
            Noise::Estimated(family) => (CustomNoise::Family(ResidualKind::Custom(family)), 1.0),
            Noise::Known(sigma) if !(sigma.is_finite() && sigma > 0.0) => {
                return Err(fail(format!("known σ {sigma} must be positive and finite")));
            }
            Noise::Known(sigma) => (CustomNoise::Whitened, 1.0 / sigma),
        };
        let bridge = Bridge::new(residual, scale);
        let dim = bridge.dim();
        let dims = bridge.block_dims().to_vec();
        if dim == 0 {
            return Err(fail("a residual needs at least one value".into()));
        }
        let indices = blocks
            .iter()
            .map(|&b| self.index(b))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| fail(e.to_string()))?;
        if dims.len() != indices.len() {
            return Err(fail(format!(
                "it declares {} blocks and was given {}",
                dims.len(),
                indices.len()
            )));
        }
        if let Some((k, &i)) = indices
            .iter()
            .enumerate()
            .find(|(k, i)| indices[..*k].contains(i))
        {
            return Err(fail(format!(
                "block {} is given twice (at {k}); a residual reads each block once",
                self.blocks[i].name
            )));
        }
        for (&i, &want) in indices.iter().zip(&dims) {
            let block = &self.blocks[i];
            if block.values.len() != want {
                return Err(fail(format!(
                    "block {} has {} values where it declares {want}",
                    block.name,
                    block.values.len()
                )));
            }
        }
        loss.validate().map_err(|e| fail(e.to_string()))?;
        let values: Vec<DVector<f64>> = indices
            .iter()
            .map(|&i| self.blocks[i].values.clone())
            .collect();
        let at_initial = bridge.eval_f64(&values);
        if at_initial.iter().any(|v| !v.is_finite()) {
            return Err(fail(format!(
                "its value at the initial blocks is not finite, or not {dim} long"
            )));
        }
        self.residuals.push(Term {
            factor: FactorKind::Custom(CustomResidual {
                factor: Arc::new(bridge),
                noise,
            }),
            blocks: indices,
            loss,
        });
        Ok(())
    }

    /// Add a camera with initial values `camera`, holding the parameters
    /// `fix` names (its unset fields take the model's defaults, as in the
    /// workflows' configs).
    ///
    /// Its parameters are labelled as in the workflows' results: `fx`,
    /// `k1`, … with one camera in the problem, `cam<i>.fx` with several
    /// (`i` in the order they were added), which
    /// [`projection_uncertainty`](crate::projection_uncertainty) reads.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a camera with a general homography
    /// sensor or non-finite parameters, or a `fix` naming a parameter the
    /// model does not have.
    pub fn add_camera(
        &mut self,
        camera: &CameraParams,
        fix: &CameraFix,
    ) -> Result<CameraId, Error> {
        let spec = camera_spec(camera)?;
        let fixed = fix
            .resolve(spec)
            .map_err(|e| Error::invalid_input(format!("camera fix: {e}")))?;
        let index = self.cameras.len();
        let values = camera.blocks();
        if values.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::invalid_input(format!(
                "camera {index}: parameters must be finite"
            )));
        }
        let fixed = [
            &fixed.intrinsics,
            &fixed.projection,
            &fixed.distortion,
            &fixed.sensor,
        ];
        let mut blocks = Vec::new();
        for ((slot, values), fixed) in CAMERA_SLOTS.into_iter().zip(values).zip(fixed) {
            if values.is_empty() {
                continue;
            }
            let id = self.push_block(Block {
                name: format!("camera{index}.{slot}"),
                values: DVector::from_vec(values),
                manifold: Manifold::Euclidean,
                fixed: FixedMask::fix_indices(fixed),
                bounds: Vec::new(),
                labels: None,
                owner: Owner::Camera {
                    camera: index,
                    slot,
                },
            });
            blocks.push(id);
        }
        self.cameras.push(Camera { spec, blocks });
        Ok(CameraId {
            problem: self.id,
            index,
        })
    }

    /// Add the library's reprojection residual: `point`, moved into the
    /// camera frame through `chain` and projected by `camera`, against the
    /// observed pixel `pixel`, with its noise and a robust loss.
    ///
    /// A [`TargetPoint::Known`] point is data, such as a board corner; a
    /// [`TargetPoint::Block`] is a point being estimated. With
    /// [`PixelNoise::Estimated`] the residual is in pixels and joins the
    /// reprojection family, whose noise level is estimated from the scatter
    /// of its residuals; with [`PixelNoise::Stated`] it is whitened by the
    /// pixel's covariance. Within one problem the reprojections of
    /// estimated noise are all whitened by a view's stated covariances or
    /// none are, so stated noise is a [`PixelNoise::Stated`] here.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a camera or block of another problem, a
    /// chain block that is not SE3 or a rig chain through one block twice,
    /// a point block that is not a Euclidean block of 3 values, a
    /// non-finite point or pixel, a loss with a scale that is not positive,
    /// or a point that does not project at the initial values.
    pub fn add_reprojection(
        &mut self,
        camera: CameraId,
        chain: PoseChain,
        point: TargetPoint,
        pixel: Pt2,
        noise: PixelNoise,
        loss: RobustLoss,
    ) -> Result<(), Error> {
        let index = self.residuals.len();
        let fail = |what: String| Error::invalid_input(format!("residual {index}: {what}"));
        let cam = self.camera(camera).map_err(|e| fail(e.to_string()))?;
        let (ir_chain, poses) = match chain {
            PoseChain::Pose(pose) => (ReprojChain::SinglePose, vec![pose]),
            PoseChain::Rig {
                cam_se3_rig,
                rig_se3_target,
            } => (ReprojChain::Rig, vec![cam_se3_rig, rig_se3_target]),
        };
        let poses = poses
            .into_iter()
            .map(|b| self.index(b))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| fail(e.to_string()))?;
        if let Some(&p) = poses
            .iter()
            .find(|&&p| self.blocks[p].manifold != Manifold::Se3)
        {
            return Err(fail(format!(
                "the pose block {} is not SE3",
                self.blocks[p].name
            )));
        }
        if poses.len() == 2 && poses[0] == poses[1] {
            return Err(fail(format!(
                "the rig chain reads block {} twice",
                self.blocks[poses[0]].name
            )));
        }
        let point_block = match point {
            TargetPoint::Known(p) => {
                if ![p.x, p.y, p.z].iter().all(|v| v.is_finite()) {
                    return Err(fail("the point must be finite".into()));
                }
                None
            }
            TargetPoint::Block(b) => {
                let b = self.index(b).map_err(|e| fail(e.to_string()))?;
                let block = &self.blocks[b];
                if block.manifold != Manifold::Euclidean || block.values.len() != 3 {
                    return Err(fail(format!(
                        "the point block {} is not a Euclidean block of 3 values",
                        block.name
                    )));
                }
                Some(b)
            }
        };
        let uv = [pixel.x, pixel.y];
        if !uv.iter().all(|v| v.is_finite()) {
            return Err(fail("the pixel must be finite".into()));
        }
        loss.validate().map_err(|e| fail(e.to_string()))?;
        let sqrt_info = match noise {
            PixelNoise::Estimated => ReprojSqrtInfo::IDENTITY,
            PixelNoise::Stated(covariance) => ReprojSqrtInfo::Stated {
                upper: covariance.sqrt_information(),
            },
        };

        let spec = self.cameras[cam].spec;
        let mut blocks: Vec<usize> = self.cameras[cam].blocks.iter().map(|b| b.index).collect();
        blocks.extend(&poses);
        blocks.extend(point_block);
        let values: Vec<DVector<f64>> = blocks
            .iter()
            .map(|&i| self.blocks[i].values.clone())
            .collect();
        let target = match point {
            TargetPoint::Known(p) => p,
            TargetPoint::Block(_) => {
                let v = &values[values.len() - 1];
                Pt3::new(v[0], v[1], v[2])
            }
        };
        // The projection is total (a pinhole clamps the depth), so a point
        // outside its domain has a finite residual that means nothing.
        let pose = |i: usize| se3(&self.blocks[poses[i]].values);
        let camera_se3_target = match ir_chain {
            ReprojChain::Rig => pose(0) * pose(1),
            _ => pose(0),
        };
        let p_camera = camera_se3_target.transform_point(&target);
        // The projection's block follows the intrinsics when it has one.
        let projection = if spec.projection.dim() > 0 {
            values[1].as_slice()
        } else {
            &[]
        };
        let pw = [target.x, target.y, target.z];
        let factor = match point {
            TargetPoint::Known(_) => FactorKind::ReprojPoint {
                model: spec,
                chain: ir_chain,
                pw,
                uv,
                sqrt_info,
            },
            TargetPoint::Block(_) => FactorKind::ReprojFreePoint {
                model: spec,
                chain: ir_chain,
                uv,
                sqrt_info,
            },
        };
        let at_initial = match point {
            TargetPoint::Known(_) => reproj_residual(spec, &ir_chain, &values, pw, uv, &sqrt_info),
            TargetPoint::Block(_) => {
                reproj_free_point_residual(spec, &ir_chain, &values, uv, &sqrt_info)
            }
        };
        if !spec.projection.in_domain(projection, &p_camera.coords)
            || at_initial.iter().any(|v| !v.is_finite())
        {
            return Err(fail(format!(
                "point {pw:?} does not project through camera {cam} at the initial values"
            )));
        }
        self.residuals.push(Term {
            factor,
            blocks,
            loss,
        });
        Ok(())
    }

    /// Whether a residual's noise is estimated from its family's scatter: a
    /// residual the solve weighs as if that noise were 1 in its unit.
    fn has_estimated_noise(&self) -> bool {
        self.residuals.iter().any(|term| match &term.factor {
            FactorKind::ReprojPoint { sqrt_info, .. }
            | FactorKind::ReprojFreePoint { sqrt_info, .. } => {
                !matches!(sqrt_info, ReprojSqrtInfo::Stated { .. })
            }
            FactorKind::Custom(custom) => matches!(custom.noise, CustomNoise::Family(_)),
            _ => false,
        })
    }

    fn check_name(&self, name: &str) -> Result<(), Error> {
        let reserved = name
            .strip_prefix("cam")
            .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()));
        if name.is_empty() || reserved {
            return Err(Error::invalid_input(format!(
                "block name {name:?} is empty or kept for cameras"
            )));
        }
        if self.blocks.iter().any(|b| b.name == name) {
            return Err(Error::invalid_input(format!(
                "block name {name:?} is already used"
            )));
        }
        Ok(())
    }

    fn push_block(&mut self, block: Block) -> BlockId {
        self.blocks.push(block);
        BlockId {
            problem: self.id,
            index: self.blocks.len() - 1,
        }
    }

    /// The block's index, for a block of this problem.
    fn index(&self, block: BlockId) -> Result<usize, Error> {
        if block.problem == self.id && block.index < self.blocks.len() {
            Ok(block.index)
        } else {
            Err(Error::invalid_input("the block belongs to another problem"))
        }
    }

    fn camera(&self, camera: CameraId) -> Result<usize, Error> {
        if camera.problem == self.id && camera.index < self.cameras.len() {
            Ok(camera.index)
        } else {
            Err(Error::invalid_input(
                "the camera belongs to another problem",
            ))
        }
    }

    fn block_mut(&mut self, block: BlockId) -> Result<&mut Block, Error> {
        let index = self.index(block)?;
        Ok(&mut self.blocks[index])
    }

    fn euclidean_mut(&mut self, block: BlockId) -> Result<&mut Block, Error> {
        let b = self.block_mut(block)?;
        if b.manifold != Manifold::Euclidean {
            return Err(Error::invalid_input(format!(
                "block {}: components can be fixed or bounded on a Euclidean block only",
                b.name
            )));
        }
        Ok(b)
    }

    /// The IR name of block `index`: positional, so no name the IR reads
    /// meaning into (a per-view pose) reaches it.
    fn ir_name(index: usize) -> String {
        format!("#{index}")
    }

    /// The IR and its initial values; with `free_consider`, the
    /// coordinates of a calibration added with [`Use::Consider`] are free,
    /// so its linearization has their columns.
    fn lower(&self, free_consider: bool) -> (ProblemIR, HashMap<String, DVector<f64>>) {
        let mut ir = ProblemIR::new();
        let mut initial = HashMap::with_capacity(self.blocks.len());
        let consider: &[(usize, usize)] = match &self.calibration {
            Some(c) if free_consider && c.usage == Use::Consider => &c.coordinates,
            _ => &[],
        };
        for (index, block) in self.blocks.iter().enumerate() {
            let name = Self::ir_name(index);
            let free: Vec<usize> = consider
                .iter()
                .filter(|&&(b, _)| b == index)
                .map(|&(_, i)| i)
                .collect();
            let fixed = if free.is_empty() {
                block.fixed.clone()
            } else if block.manifold == Manifold::Euclidean {
                let held: Vec<usize> = (0..block.values.len())
                    .filter(|i| !free.contains(i))
                    .collect();
                FixedMask::fix_indices(&held)
            } else {
                FixedMask::all_free()
            };
            ir.add_param_block(
                &name,
                block.values.len(),
                block.manifold.ir(),
                fixed,
                (!block.bounds.is_empty()).then(|| block.bounds.clone()),
            );
            initial.insert(name, block.values.clone());
        }
        for term in &self.residuals {
            ir.add_residual_block(ResidualBlock {
                params: term.blocks.iter().map(|&i| ParamId(i)).collect(),
                loss: term.loss,
                residual_dim: term.factor.residual_dim(),
                factor: term.factor.clone(),
            });
        }
        (ir, initial)
    }

    /// Labels of the free coordinates of blocks no residual reads: nothing
    /// observes them, and the backend leaves them out of the solve.
    fn unread_coordinates(&self, labels: &[BlockLabels]) -> Vec<String> {
        let mut read = vec![false; self.blocks.len()];
        for term in &self.residuals {
            for &b in &term.blocks {
                read[b] = true;
            }
        }
        let mut out = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            if read[index] {
                continue;
            }
            let labels = &labels[index].coordinates;
            match block.manifold {
                Manifold::Euclidean => out.extend(
                    (0..block.values.len())
                        .filter(|&i| !block.fixed.contains(i))
                        .map(|i| labels[i].clone()),
                ),
                _ if block.fixed.is_all_fixed(block.values.len()) => {}
                _ => out.extend(labels.iter().cloned()),
            }
        }
        out
    }

    /// Each block's coordinate labels, in block order.
    fn labels(&self) -> Vec<BlockLabels> {
        let multi_camera = self.cameras.len() > 1;
        self.blocks
            .iter()
            .map(|block| match block.owner {
                Owner::User => block.block_labels(),
                Owner::Camera { camera, slot } => {
                    let prefix = |name: &str| {
                        if multi_camera {
                            format!("cam{camera}.{name}")
                        } else {
                            name.to_string()
                        }
                    };
                    camera_block_labels(slot, self.cameras[camera].spec, prefix)
                        .expect("camera blocks have a camera slot name")
                }
            })
            .collect()
    }
}

/// Options of [`solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SolveOptions {
    /// The engine that linearizes the problem. A problem with a residual
    /// you wrote runs on [`SolverBackend::TinySolver`] only.
    pub backend: SolverBackend,
    /// Most Levenberg–Marquardt iterations; a solve that stops on this
    /// budget has no covariance.
    pub max_iters: usize,
}

impl Default for SolveOptions {
    /// tiny-solver, 100 iterations.
    fn default() -> Self {
        Self {
            backend: SolverBackend::default(),
            max_iters: BackendSolveOptions::default().max_iters,
        }
    }
}

/// The result of [`solve`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Solution {
    /// Final cost and iterations.
    pub report: SolveReport,
    /// The noise level of each residual family at the solution (with
    /// [`Noise::Estimated`] families as [`ResidualKind::Custom`]),
    /// whether or not the covariance is available.
    pub noise: Vec<ResidualNoise>,
    /// The covariance of the free parameters, labelled by coordinate (see
    /// [`Problem::set_labels`] and [`Problem::add_camera`]), or why it is
    /// unavailable.
    pub uncertainty: Uncertainty,
    problem: u64,
    values: Vec<DVector<f64>>,
    manifolds: Vec<Manifold>,
    cameras: Vec<Camera>,
}

impl Solution {
    /// The block's values at the solution; `None` for a block of another
    /// problem.
    pub fn value(&self, block: BlockId) -> Option<&[f64]> {
        (block.problem == self.problem)
            .then(|| self.values.get(block.index))
            .flatten()
            .map(DVector::as_slice)
    }

    /// An SE3 block's pose at the solution; `None` for a block of another
    /// problem or one that is not SE3.
    pub fn pose(&self, block: BlockId) -> Option<Iso3> {
        let values = self.value(block)?;
        (self.manifolds[block.index] == Manifold::Se3).then(|| {
            let v = DVector::from_column_slice(values);
            se3(&v)
        })
    }

    /// The camera at the solution; `None` for a camera of another problem.
    pub fn camera(&self, camera: CameraId) -> Option<CameraParams> {
        if camera.problem != self.problem {
            return None;
        }
        let cam = self.cameras.get(camera.index)?;
        let mut blocks = cam.blocks.iter().map(|&i| self.values[i.index].as_slice());
        let intrinsics = blocks.next()?;
        // A slot without parameters has no block.
        let mut next = |dim: usize| {
            if dim == 0 {
                Some(&[][..])
            } else {
                blocks.next()
            }
        };
        let spec = cam.spec;
        let projection = next(spec.projection.dim())?;
        let distortion = next(spec.distortion.dim())?;
        let sensor = next(spec.sensor.dim())?;
        Some(CameraParams {
            intrinsics: FxFyCxCy {
                fx: intrinsics[0],
                fy: intrinsics[1],
                cx: intrinsics[2],
                cy: intrinsics[3],
            },
            projection: ProjectionParams::from_values(spec.projection, projection).ok()?,
            distortion: DistortionParams::from_values(spec.distortion, distortion).ok()?,
            sensor: SensorParams::from_values(spec.sensor, sensor).ok()?,
        })
    }
}

/// Add the share of a [`Use::Consider`] calibration to the covariance
/// `estimate` of a solve at `params` (see [`covariance::consider_share`]).
fn add_consider_share(
    problem: &Problem,
    calibration: &Calibration,
    sigma: &nalgebra::DMatrix<f64>,
    params: &HashMap<String, DVector<f64>>,
    estimate: &mut covariance::CovarianceEstimate,
) -> Result<(), Error> {
    if calibration.coordinates.is_empty() || estimate.coordinates.is_empty() {
        return Ok(());
    }
    let (ir, _) = problem.lower(true);
    let raw = backend::raw_linearization(&ir, params)?
        .ok_or_else(|| Error::numerical("the consider linearization has no columns"))?;
    let consider: Vec<Coordinate> = calibration
        .coordinates
        .iter()
        .map(|&(block, index)| Coordinate { block, index })
        .collect();
    let (coordinates, share) = covariance::consider_share(&ir, params, &raw, &consider, sigma)?;
    let at: HashMap<Coordinate, usize> = estimate
        .coordinates
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let rows = coordinates
        .iter()
        .map(|c| {
            at.get(c).copied().ok_or_else(|| {
                Error::numerical(format!("coordinate {c:?} is missing from the covariance"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (a, &ra) in rows.iter().enumerate() {
        for (b, &rb) in rows.iter().enumerate() {
            estimate.matrix[(ra, rb)] += share[(a, b)];
        }
    }
    Ok(())
}

/// Solve `problem` from its initial values.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a problem with a residual you wrote and the
/// factrs backend; [`Error::Numerical`] when the solve fails.
pub fn solve(problem: &Problem, options: &SolveOptions) -> Result<Solution, Error> {
    let calibration = problem.calibration.as_ref();
    if calibration.is_some_and(|c| c.usage == Use::Prior) {
        if options.backend == SolverBackend::Factrs {
            return Err(Error::invalid_input(
                "a calibration added with Use::Prior runs on the tiny-solver backend only",
            ));
        }
        if problem.has_estimated_noise() {
            return Err(Error::invalid_input(
                "a calibration added with Use::Prior is weighed by its covariance, so every \
                 residual's noise is stated too (PixelNoise::Stated, Noise::Known)",
            ));
        }
    }
    let (ir, initial) = problem.lower(false);
    let opts = BackendSolveOptions {
        backend: options.backend,
        max_iters: options.max_iters,
        ..BackendSolveOptions::default()
    };
    let mut solution = backend::solve(&ir, &initial, &opts)?;
    let labels = problem.labels();
    let unread = problem.unread_coordinates(&labels);
    let uncertainty = if !unread.is_empty() {
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::RankDeficient {
                unobservable: unread,
            },
        }
    } else {
        let consider = calibration.filter(|c| c.usage == Use::Consider);
        match (consider, &mut solution.uncertainty) {
            (Some(c), ParamUncertainty::Available(estimate)) => match &c.sigma {
                Some(sigma) => {
                    add_consider_share(problem, c, sigma, &solution.params, estimate)?;
                    uncertainty::labelled(&ir, &solution.uncertainty, &solution.noise, &labels)
                }
                None => Uncertainty::Unavailable {
                    reason: UncertaintyUnavailable::CalibrationWithoutCovariance,
                },
            },
            _ => uncertainty::labelled(&ir, &solution.uncertainty, &solution.noise, &labels),
        }
    };
    let values = (0..problem.blocks.len())
        .map(|i| {
            solution
                .params
                .get(&Problem::ir_name(i))
                .cloned()
                .ok_or_else(|| Error::numerical(format!("block {i} missing from the solution")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Solution {
        report: solution.solve_report,
        noise: uncertainty::residual_noise(&solution.noise),
        uncertainty,
        problem: problem.id,
        values,
        manifolds: problem.blocks.iter().map(|b| b.manifold).collect(),
        cameras: problem.cameras.clone(),
    })
}
