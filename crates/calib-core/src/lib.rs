//! Core math and geometry primitives for `calibration-rs`.
//!
//! This crate provides the foundational building blocks used by all other
//! crates in the workspace:
//!
//! - linear algebra type aliases (`Real`, `Vec2`, `Pt3`, and friends),
//! - composable camera models (projection + distortion + sensor + intrinsics),
//! - a deterministic, model-agnostic RANSAC engine.
//!
//! Camera pipeline (conceptually):
//! `pixel = intrinsics(sensor(distortion(projection(dir))))`
//!
//! The sensor stage supports a Scheimpflug/tilted sensor homography aligned
//! with OpenCV's `computeTiltProjectionMatrix`.
//!
//! # Modules
//!
//! - \[`math`\]: basic type aliases and homogeneous helpers.
//! - \[`models`\]: camera model traits and configuration wrappers.
//! - \[`ransac`\]: generic robust estimation helpers.
//! - \[`synthetic`\]: deterministic synthetic data helpers (tests/examples/benchmarks).
//!
//! # Example
//!
//! ```no_run
//! use calib_core::{
//!     CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, ProjectionParams,
//!     SensorParams,
//! };
//!
//! let params = CameraParams {
//!     projection: ProjectionParams::Pinhole,
//!     distortion: DistortionParams::None,
//!     sensor: SensorParams::Identity,
//!     intrinsics: IntrinsicsParams::FxFyCxCySkew {
//!         params: FxFyCxCySkew {
//!             fx: 800.0,
//!             fy: 800.0,
//!             cx: 640.0,
//!             cy: 360.0,
//!             skew: 0.0,
//!         },
//!     },
//! };
//! let cam = params.build();
//! let px = cam.project_point_c(&nalgebra::Vector3::new(0.1, 0.2, 1.0));
//! assert!(px.is_some());
//! ```

/// Linear algebra type aliases and helpers.
mod math;
/// Camera models and distortion utilities.
mod models;
/// Generic RANSAC engine and traits.
mod ransac;
/// Deterministic synthetic data generation helpers.
///
/// This module provides small, reusable building blocks for constructing
/// synthetic calibration problems (planar grids, poses, projections, noise).
/// It is used in workspace tests/examples and can be useful for benchmarking
/// and regression testing.
pub mod synthetic;
/// Test utilities for cross-crate calibration testing.
///
/// This module is public to allow usage in integration tests across
/// the workspace, but is not intended for production use.
pub mod test_utils;
/// Common types for observations, results, and options.
mod types;
mod view;

pub use math::*;
pub use models::*;
pub use ransac::*;
pub use types::*;
pub use view::*;

use anyhow::Result;

pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

pub fn make_pinhole_camera(k: FxFyCxCySkew<Real>, dist: BrownConrady5<Real>) -> PinholeCamera {
    Camera::new(Pinhole, dist, IdentitySensor, k)
}

pub fn pinhole_camera_params(camera: &PinholeCamera) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: camera.dist.k1,
                k2: camera.dist.k2,
                k3: camera.dist.k3,
                p1: camera.dist.p1,
                p2: camera.dist.p2,
                iters: camera.dist.iters,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: IntrinsicsParams::FxFyCxCySkew {
            params: FxFyCxCySkew {
                fx: camera.k.fx,
                fy: camera.k.fy,
                cx: camera.k.cx,
                cy: camera.k.cy,
                skew: camera.k.skew,
            },
        },
    }
}

pub struct TargetPose {
    pub camera_se3_target: Iso3,
}

pub fn compute_mean_reproj_error(
    camera: &PinholeCamera,
    views: &[View<TargetPose>],
) -> Result<Real> {
    let mut total_error = 0.0;
    let mut total_points = 0;

    for view in views {
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = view.meta.camera_se3_target * p3d;
            if let Some(projected) = camera.project_point_c(&p_cam.coords) {
                let error = (projected - *p2d).norm();
                total_error += error;
                total_points += 1;
            }
        }
    }

    if total_points == 0 {
        anyhow::bail!("No valid projections for error computation");
    }

    Ok(total_error / total_points as Real)
}
