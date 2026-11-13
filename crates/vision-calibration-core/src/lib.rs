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
//! - `synthetic` (feature `synthetic`): deterministic synthetic data helpers
//!   for tests, examples and benchmarks.
//!
//! # Example
//!
//! ```
//! use vision_calibration_core::{
//!     CameraParams, DistortionParams, FxFyCxCy, ProjectionParams, SensorParams,
//! };
//!
//! let params = CameraParams::new(
//!     ProjectionParams::Pinhole,
//!     DistortionParams::None,
//!     SensorParams::Identity,
//!     FxFyCxCy {
//!         fx: 800.0,
//!         fy: 800.0,
//!         cx: 640.0,
//!         cy: 360.0,
//!     },
//! );
//! // `build` is fallible only for a `SensorParams::Homography` that is
//! // not invertible; an identity sensor always succeeds.
//! let cam = params.build()?;
//! let px = cam.project_point_c(&nalgebra::Vector3::new(0.1, 0.2, 1.0));
//! assert!(px.is_some());
//! # Ok::<(), vision_calibration_core::Error>(())
//! ```

/// Progress reports and cooperative cancellation of long-running calls.
mod control;
/// Typed error enum for this crate.
pub mod error;
/// Low-level linear-algebra primitives shared by the calibration and MVG
/// solver crates (Hartley normalization, polynomial roots, null space, SVD
/// reshaping).
pub mod linalg;
/// Linear algebra type aliases and helpers.
mod math;
/// Camera models and distortion utilities.
mod models;
/// Generic RANSAC engine and traits.
mod ransac;
/// Reprojection statistics and per-feature target residuals.
mod residuals;
/// Deterministic synthetic data generation helpers.
///
/// This module provides small, reusable building blocks for constructing
/// synthetic calibration problems (planar grids, poses, projections, noise).
/// It is used in workspace tests/examples and can be useful for benchmarking
/// and regression testing. Enabled by the `synthetic` feature.
#[cfg(any(test, feature = "synthetic"))]
pub mod synthetic;
/// Test utilities for cross-crate calibration testing.
///
/// Gated behind the unstable, internal `test-utils` feature so it is not part
/// of the published API surface. Enabled only by workspace test suites; do not
/// rely on it from production code.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod test_utils;
/// Common types for observations and results.
mod types;
mod view;

pub use control::{CancelToken, Cancelled, Progress, RunControl};
pub use error::Error;

// Crate-root public surface, re-exported explicitly from the (private) modules.
// Keep these lists curated: a new `pub` item in a private module is not public
// until it is listed here.
#[cfg(feature = "schemars")]
pub use math::Iso3Schema;
pub use math::{
    Iso3, Mat3, Mat4, Pt2, Pt3, Real, Vec2, Vec3, distort_to_pixel, from_homogeneous,
    normalized_to_pixel, pixel_to_normalized, to_homogeneous, undistort_pixel,
};
pub use models::{
    AnyDistortion, AnyProjection, AnySensor, BrownConrady5, Camera, CameraFix, CameraModel,
    CameraModelSpec, CameraParams, CameraProject, DistortionKind, DistortionModel,
    DistortionParams, Division, Equidistant, FxFyCxCy, HomographySensor, IdentitySensor,
    IntrinsicsModel, KannalaBrandt4, NoDistortion, Pinhole, ProjectionKind, ProjectionModel,
    ProjectionParams, RationalPolynomial, Ray, ScheimpflugParams, SensorKind, SensorModel,
    SensorParams, ThinPrism,
};
// Workspace-internal: the parameter-fixing mask and intrinsics labels the
// optimizer packs a camera with. Hidden from the docs and outside the semver
// promise; name fixed parameters through `CameraFix`.
#[doc(hidden)]
pub use models::{FixedParams, INTRINSICS_PARAM_NAMES};
pub use ransac::{Estimator, RansacOptions, RansacResult, ransac_fit};
pub use residuals::{
    TargetPose, build_feature_histogram, compute_mean_reproj_error,
    compute_planar_target_residuals, compute_planar_target_residuals_views,
    compute_rig_reprojection_stats, compute_rig_reprojection_stats_per_camera,
    compute_rig_target_residuals,
};
pub use types::{
    CalibratedRig, CombinationTerm, CorrespondenceView, FeatureResidualHistogram, FrameKind,
    FrameRef, Handeye, ImageManifest, LaserFeatureResidual, LaserLineTrace, LaserPlane, NoiseLevel,
    ParameterCovariance, PerFeatureResiduals, PixelRect, PointCovariance,
    REPROJECTION_HISTOGRAM_EDGES_PX, ReprojectionStats, ResidualKind, ResidualNoise, RobotFrame,
    RobotPose, SecondOrderError, TargetFeatureResidual, Uncertainty, UncertaintyUnavailable,
    ViewCovariance, stated_covariance,
};
// The frame Jacobians the measurement crates share; outside the semver
// promise.
#[doc(hidden)]
pub use types::{RigInFrame, se3_adjoint};
pub use view::{NoMeta, PlanarDataset, RigDataset, RigView, RigViewObs, View};

/// Concrete pinhole camera with Brown-Conrady distortion and an untilted
/// sensor: the camera Zhang's linear initialization returns and the
/// synthetic generators project with. Calibration results carry
/// [`CameraParams`], which states any model.
///
/// Composition:
/// - projection: [`Pinhole`]
/// - distortion: [`BrownConrady5`]
/// - sensor: [`IdentitySensor`]
/// - intrinsics: [`FxFyCxCy`]
pub type PinholeCamera = Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCy<Real>>;

/// Build a [`PinholeCamera`] from intrinsics and Brown-Conrady distortion.
pub fn make_pinhole_camera(k: FxFyCxCy<Real>, dist: BrownConrady5<Real>) -> PinholeCamera {
    Camera::new(Pinhole, dist, IdentitySensor, k)
}

/// Convert a concrete [`PinholeCamera`] into serializable [`CameraParams`].
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
        intrinsics: FxFyCxCy {
            fx: camera.k.fx,
            fy: camera.k.fy,
            cx: camera.k.cx,
            cy: camera.k.cy,
        },
    }
}
