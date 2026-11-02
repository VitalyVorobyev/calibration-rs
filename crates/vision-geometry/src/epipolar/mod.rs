//! Epipolar geometry solvers for fundamental and essential matrices.
//!
//! Includes normalized 8-point, 7-point, and 5-point minimal solvers, plus
//! decomposition of the essential matrix into candidate poses.
//!
//! - Fundamental matrix `F` expects **pixel coordinates** in both images.
//! - Essential matrix `E` expects **bearings**: rays in each camera's frame
//!   from its own unprojection (any nonzero length), with `b2ᵀ E b1 = 0`. A
//!   pinhole's `(x, y, 1)` of normalized coordinates is one.

mod decomposition;
mod essential;
mod fundamental;
pub(crate) mod polynomial;

pub use decomposition::decompose_essential;
pub use essential::{essential_5point, essential_linear};
pub use fundamental::{fundamental_7point, fundamental_8point, fundamental_8point_ransac};
