//! Non-linear optimization for camera calibration with automatic differentiation.
//!
//! This crate provides a backend-agnostic optimization framework for camera calibration
//! problems. The core design separates problem definition from solver implementation using
//! an intermediate representation (IR) that can be compiled to different optimization backends.
//!
//! # Architecture
//!
//! The optimization pipeline has three stages:
//!
//! 1. **Problem Definition** - Build a [`ir::ProblemIR`] describing parameters, factors, and constraints
//! 2. **Backend Compilation** - Translate IR into solver-specific problem (e.g., [`backend::TinySolverBackend`])
//! 3. **Optimization** - Run solver and extract solution as domain types
//!
//! ```text
//! Problem Builder → ProblemIR → Backend.compile() → Backend.solve() → Domain Result
//! ```
//!
//! ## Key Components
//!
//! - **[`ir`]** - Backend-agnostic intermediate representation for optimization problems
//! - **[`params`]** - Parameter block definitions (intrinsics, distortion, poses)
//! - **[`factors`]** - Residual functions with automatic differentiation support
//! - **[`backend`]** - Solver implementations (currently tiny-solver with Levenberg-Marquardt)
//! - **[`problems`]** - High-level calibration problem builders (planar intrinsics, etc.)
//!
//! # Examples
//!
//! ## Basic Planar Intrinsics Calibration
//!
//! ```rust,no_run
//! use calib_optim::problems::planar_intrinsics::*;
//! use calib_core::{BrownConrady5, DistortionFixMask, FxFyCxCySkew};
//! use calib_optim::{BackendSolveOptions, ir::RobustLoss};
//! use nalgebra::{Isometry3, Vector3};
//!
//! # fn example() -> anyhow::Result<()> {
//! // 1. Prepare observations (world points + image detections)
//! let views = vec![/* CorrespondenceView */];
//! let dataset = PlanarDataset::new(views)?;
//!
//! // 2. Initialize with linear method or prior calibration
//! let init = PlanarIntrinsicsInit {
//!     intrinsics: FxFyCxCySkew {
//!         fx: 800.0,
//!         fy: 800.0,
//!         cx: 640.0,
//!         cy: 360.0,
//!         skew: 0.0,
//!     },
//!     distortion: BrownConrady5 {
//!         k1: 0.0,
//!         k2: 0.0,
//!         k3: 0.0,
//!         p1: 0.0,
//!         p2: 0.0,
//!         iters: 8,
//!     },
//!     poses: vec![/* initial poses */],
//! };
//!
//! // 3. Configure optimization
//! let opts = PlanarIntrinsicsSolveOptions {
//!     robust_loss: RobustLoss::Huber { scale: 2.0 },
//!     fix_distortion: DistortionFixMask { k3: true, ..Default::default() }, // Fix k3 to prevent overfitting
//!     ..Default::default()
//! };
//!
//! // 4. Run optimization
//! let result = optimize_planar_intrinsics(dataset, init, opts, BackendSolveOptions::default())?;
//!
//! println!("Calibrated camera: {:?}", result.camera);
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Problem with IR
//!
//! ```rust,no_run
//! use calib_optim::ir::*;
//! use calib_optim::backend::{TinySolverBackend, OptimBackend};
//! use nalgebra::DVector;
//! use std::collections::HashMap;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Build problem IR
//! let mut ir = ProblemIR::new();
//!
//! // Add parameter block (4D intrinsics)
//! let cam_id = ir.add_param_block(
//!     "camera",
//!     4,
//!     ManifoldKind::Euclidean,
//!     FixedMask::all_free(),
//!     None,
//! );
//!
//! // Add residual blocks (factors) referencing parameters
//! // ... (see ir::ResidualBlock documentation)
//!
//! // Compile and solve
//! let mut initial = HashMap::new();
//! initial.insert("camera".to_string(), DVector::from_row_slice(&[800.0, 800.0, 640.0, 360.0]));
//!
//! let backend = TinySolverBackend;
//! let solution = backend.solve(&ir, &initial, &Default::default())?;
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Highlights
//!
//! ## Automatic Differentiation
//!
//! All residual functions are generic over [`nalgebra::RealField`], enabling automatic
//! differentiation via dual numbers. The [`factors::reprojection_model`] module provides
//! autodiff-compatible implementations of:
//!
//! - Pinhole projection with SE3 poses
//! - Brown-Conrady distortion (k1, k2, k3, p1, p2)
//! - Weighted reprojection residuals
//!
//! ## Flexible Parameter Fixing
//!
//! Use [`ir::FixedMask`] to selectively fix optimization variables:
//!
//! ```rust
//! # use calib_optim::problems::planar_intrinsics::PlanarIntrinsicsSolveOptions;
//! # use calib_core::{DistortionFixMask, IntrinsicsFixMask};
//! let opts = PlanarIntrinsicsSolveOptions {
//!     fix_intrinsics: IntrinsicsFixMask { fx: true, ..Default::default() }, // Fix focal length
//!     fix_distortion: DistortionFixMask { p1: true, p2: true, ..Default::default() }, // Fix tangential distortion
//!     ..Default::default()
//! };
//! ```
//!
//! ## Robust Loss Functions
//!
//! Handle outliers with M-estimators ([`ir::RobustLoss`]):
//!
//! - `Huber` - L2 near zero, L1 for outliers
//! - `Cauchy` - Gradual outlier suppression
//! - `Arctan` - Bounded influence
//!
//! # Performance Considerations
//!
//! - **Initialization**: Always initialize with linear methods (the `calib-linear` crate) for faster convergence
//! - **Robust Loss**: Use Huber with `scale ≈ 2.0` for real data with corner detection noise
//! - **Distortion**: Fix `k3` by default unless calibrating wide-angle lenses
//! - **Manifolds**: SE3/SO3 parameters use proper Lie group updates for stability
//!
//! # Numerical Stability
//!
//! The implementation uses several techniques for numerical robustness:
//!
//! - Safe division with epsilon thresholds in projection
//! - Hartley normalization in linear initialization (via calib-linear)
//! - Manifold-aware parameter updates for rotations
//! - Sparse linear solvers for large problems

pub mod backend;
pub mod factors;
pub mod ir;
pub mod math;
pub mod params;
pub mod problems;

pub use crate::backend::{BackendKind, BackendSolution, BackendSolveOptions};
pub use crate::problems::handeye;
pub use crate::problems::planar_intrinsics;
