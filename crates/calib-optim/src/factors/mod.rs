//! Residual factor implementations with automatic differentiation support.
//!
//! Factors (also called cost functions or residual blocks) compute the difference between
//! observed measurements and predictions from estimated parameters. All factor implementations
//! are generic over [`nalgebra::RealField`] to support both f64 evaluation and automatic
//! differentiation via dual numbers.
//!
//! # Design Pattern
//!
//! Factor functions follow this pattern:
//!
//! ```rust,ignore
//! pub(crate) fn my_factor_generic<T: RealField>(
//!     param1: DVectorView<'_, T>,
//!     param2: DVectorView<'_, T>,
//!     measurement: [f64; N],
//! ) -> SVector<T, M> {
//!     // Extract parameters with .clone() (cheap for dual numbers)
//!     let p1 = param1[0].clone();
//!
//!     // Perform computations using generic operations
//!     let prediction = compute_prediction(p1, ...);
//!
//!     // Return residual as SVector
//!     SVector::<T, M>::new(...)
//! }
//! ```
//!
//! ## Key Guidelines
//!
//! - Use `.clone()` liberally on `T: RealField` values (dual numbers are `Copy`-like)
//! - Avoid in-place mutations
//! - Convert constants with `T::from_f64().unwrap()`
//! - Use `T::one()` and `T::zero()` for identity elements
//! - Include `debug_assert!` for parameter dimension checks
//!
//! # Available Factors
//!
//! - [`reprojection_model`] - Pinhole camera reprojection with optional Brown-Conrady distortion
//! - [`linescan`] - Laser plane residuals for linescan calibration

pub mod linescan;
pub mod reprojection_model;
