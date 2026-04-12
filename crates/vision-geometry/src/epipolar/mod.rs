//! Epipolar geometry solvers for fundamental and essential matrices.
//!
//! Includes normalized 8-point, 7-point, and 5-point minimal solvers, plus
//! decomposition of the essential matrix into candidate poses.
//!
//! - Fundamental matrix `F` expects **pixel coordinates** in both images.
//! - Essential matrix `E` expects **normalized coordinates** (after applying
//!   `K⁻¹`), or equivalently calibrated rays on the normalized image plane.

mod decomposition;
mod essential;
mod fundamental;
pub(crate) mod polynomial;

pub use decomposition::decompose_essential;
pub use essential::essential_5point;
pub use fundamental::{fundamental_7point, fundamental_8point, fundamental_8point_ransac};
