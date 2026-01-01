//! High-level entry crate for the `calibration-rs` toolbox.
//!
//! This crate re-exports the lower level building blocks (`calib-core`,
//! `calib-linear`, `calib-optim`, `calib-pipeline`) under a single,
//! documented surface that is convenient to depend on from applications.
//!
//! The intent is:
//! - **`core`**: basic math types and camera / RANSAC primitives,
//! - **`linear`**: closed-form/linear initialisation algorithms,
//! - **`optim`**: non-linear least squares abstractions and solvers,
//! - **`pipeline`**: end-to-end calibration pipelines.
//!
//! A minimal example using the planar intrinsics pipeline:
//!
//! ```no_run
//! use calib::pipeline::{PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarViewData};
//! use calib::core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, IntrinsicsConfig, Pinhole, Pt3, Vec2};
//!
//! # fn main() {
//! // Build some synthetic observations (normally these come from a detector).
//! let board_points: Vec<Pt3> = (0..4)
//!     .flat_map(|y| {
//!         (0..6).map(move |x| Pt3::new(x as f64 * 0.03, y as f64 * 0.03, 0.0))
//!     })
//!     .collect();
//! let mut views = Vec::new();
//!
//! // Ground-truth camera just for the example.
//! let k_gt = FxFyCxCySkew {
//!     fx: 800.0,
//!     fy: 780.0,
//!     cx: 640.0,
//!     cy: 360.0,
//!     skew: 0.0,
//! };
//! let dist_gt = BrownConrady5 {
//!     k1: -0.1,
//!     k2: 0.01,
//!     k3: 0.0,
//!     p1: 0.0,
//!     p2: 0.0,
//!     iters: 8,
//! };
//! let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
//!
//! // For simplicity we just create one fronto-parallel view here.
//! let points_2d: Vec<Vec2> = board_points
//!     .iter()
//!     .map(|pw| {
//!         // In a real setup you would transform by the camera pose first.
//!         let pc = Pt3::new(pw.x, pw.y, 1.0);
//!         cam_gt.project_point(&pc).unwrap()
//!     })
//!     .collect();
//!
//! views.push(PlanarViewData {
//!     points_3d: board_points,
//!     points_2d,
//! });
//!
//! let input = PlanarIntrinsicsInput { views };
//! let config = PlanarIntrinsicsConfig::default();
//!
//! let report = calib::pipeline::run_planar_intrinsics(&input, &config)
//!     .expect("planar intrinsics failed");
//! if let IntrinsicsConfig::FxFyCxCySkew { fx, fy, cx, cy, skew } = &report.camera.intrinsics {
//!     println!("estimated intrinsics: fx={fx} fy={fy} cx={cx} cy={cy} skew={skew}");
//! }
//! # }
//! ```

/// Core math and primitive camera / RANSAC utilities.
pub mod core {
    pub use calib_core::*;
}

/// Linear / closed-form initialisation algorithms.
pub mod linear {
    pub use calib_linear::*;
}

/// Non-linear optimisation traits, backends and problems.
pub mod optim {
    pub use calib_optim::*;
}

/// High-level calibration pipelines suitable for CLI / applications.
pub mod pipeline {
    pub use calib_pipeline::*;
}
