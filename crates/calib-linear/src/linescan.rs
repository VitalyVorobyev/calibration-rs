//! Linescan laser plane estimation from planar target observations.
//!
//! This module provides linear closed-form estimation of laser plane parameters
//! from pixel observations of a laser line projected onto a planar calibration target.
//!
//! # Algorithm
//!
//! Given:
//! - Laser line pixel observations
//! - Camera pose (T_C_T) from calibration features (via PnP/homography)
//! - Known planar target geometry (Z=0 in target frame)
//!
//! The algorithm:
//! 1. Back-projects each laser pixel to a ray in camera frame
//! 2. Intersects each ray with the target plane to get 3D points
//! 3. Fits a plane to the 3D points via SVD (DLT-style)
//!
//! The resulting laser plane is parameterized as (n̂, d) where n̂ is the unit
//! normal vector and d is the signed distance from the camera origin.

use calib_core::{
    BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Pt2, Pt3, Real, Vec2,
};
use nalgebra::{Point3, UnitVector3, Vector3};
use thiserror::Error;

/// Errors that can occur during linescan plane estimation.
#[derive(Debug, Error)]
pub enum LinescanError {
    #[error("insufficient points: got {got}, need at least {min}")]
    InsufficientPoints { got: usize, min: usize },

    #[error("numerical failure: {0}")]
    NumericalFailure(String),

    #[error("ray parallel to target plane (degenerate geometry)")]
    RayParallelToTargetPlane,
}

/// Result type for linescan operations.
pub type LinescanResult<T> = std::result::Result<T, LinescanError>;

/// Laser line observations for a single view.
///
/// The camera pose must be known (typically from PnP solution using
/// calibration features in the same view).
#[derive(Debug, Clone)]
pub struct LinescanView {
    /// 2D pixel observations along the laser line
    pub laser_pixels: Vec<Pt2>,
    /// Camera-to-target transform (T_C_T) from calibration features
    pub camera_pose: Iso3,
}

/// Result of linear laser plane estimation.
#[derive(Debug, Clone)]
pub struct LinearPlaneEstimate {
    /// Unit normal vector in camera frame
    pub normal: UnitVector3<f64>,
    /// Signed distance from camera origin
    pub distance: f64,
    /// Indices of inlier observations (all if no outlier rejection)
    pub inliers: Vec<usize>,
    /// Root mean square point-to-plane distance
    pub rmse: f64,
}

/// Solver for laser plane estimation via closed-form methods.
pub struct LinescanPlaneSolver;

impl LinescanPlaneSolver {
    /// Estimate laser plane from 3D points in camera frame via SVD.
    ///
    /// Fits a plane ax + by + cz + d = 0 to the given 3D points using
    /// homogeneous DLT (Direct Linear Transform). Returns the plane
    /// parameterized as unit normal (n̂ = [a, b, c] / ||(a,b,c)||) and
    /// signed distance (d / ||(a,b,c)||).
    ///
    /// # Algorithm
    ///
    /// 1. Build constraint matrix A where each row is [x, y, z, 1]
    /// 2. Solve A * [a, b, c, d]^T = 0 via SVD (smallest singular value)
    /// 3. Normalize to unit normal
    ///
    /// Requires at least 3 non-colinear points.
    /// TODO: check non-colinearity
    pub fn from_points_3d(points_camera: &[Pt3]) -> LinescanResult<LinearPlaneEstimate> {
        if points_camera.len() < 3 {
            return Err(LinescanError::InsufficientPoints {
                got: points_camera.len(),
                min: 3,
            });
        }

        // Use covariance-based approach for numerical stability
        let n = points_camera.len();
        // Compute centroid
        let mut centroid = Vector3::zeros();
        for pt in points_camera {
            centroid += pt.coords;
        }
        centroid /= n as f64;

        // Build 3x3 covariance matrix of centered points
        let mut cov = nalgebra::Matrix3::zeros();
        for pt in points_camera {
            let centered = pt.coords - centroid;
            cov += centered * centered.transpose();
        }

        // Smallest eigenvector of covariance matrix is plane normal
        let eigen = cov.symmetric_eigen();
        let min_idx = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .ok_or(LinescanError::NumericalFailure(
                "failed to find minimum eigenvalue".into(),
            ))?;

        let normal_vec = eigen.eigenvectors.column(min_idx);
        let normal = UnitVector3::new_normalize(normal_vec.into());

        // Compute signed distance: d = -n · centroid
        let distance = -normal.dot(&centroid);

        // Compute RMSE
        let mut sum_sq_dist = 0.0;
        for pt in points_camera {
            let dist = normal.dot(&pt.coords) + distance;
            sum_sq_dist += dist * dist;
        }
        let rmse = (sum_sq_dist / n as f64).sqrt();

        Ok(LinearPlaneEstimate {
            normal,
            distance,
            inliers: (0..n).collect(),
            rmse,
        })
    }

    /// Estimate laser plane from pixel observations and known camera pose.
    ///
    /// This is the complete pipeline:
    /// 1. Undistorts each laser pixel using camera intrinsics and distortion
    /// 2. Back-projects to ray in camera frame
    /// 3. Intersects ray with planar target (Z=0 in target frame)
    /// 4. Transforms intersection point to camera frame
    /// 5. Fits plane to 3D points via [`Self::from_points_3d`]
    ///
    /// # Arguments
    ///
    /// - `view`: Laser pixel observations and camera pose
    /// - `camera`: Calibrated camera model (with intrinsics and distortion)
    ///
    /// TODO: single view only provides collinear points. That is a degenerate case. Need
    /// multiple views to fit a plane. Review this API, probably change to `from_views`.
    pub fn from_view(
        view: &LinescanView,
        camera: &Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>,
    ) -> LinescanResult<LinearPlaneEstimate> {
        if view.laser_pixels.is_empty() {
            return Err(LinescanError::InsufficientPoints { got: 0, min: 1 });
        }

        // Compute 3D points in camera frame
        let points_camera = Self::compute_3d_points(&view.laser_pixels, camera, &view.camera_pose)?;

        // Fit plane to 3D points
        Self::from_points_3d(&points_camera)
    }

    /// Compute 3D laser points in camera frame from pixel observations.
    ///
    /// For each pixel:
    /// 1. Backproject pixel to ray using camera model (handles undistortion)
    /// 2. Transform ray to target frame (inverse of camera pose)
    /// 3. Intersect ray with target plane (Z=0)
    /// 4. Transform intersection point back to camera frame
    fn compute_3d_points(
        laser_pixels: &[Pt2],
        camera: &Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>,
        camera_pose: &Iso3,
    ) -> LinescanResult<Vec<Pt3>> {
        let mut points_camera = Vec::with_capacity(laser_pixels.len());

        for pixel in laser_pixels {
            // 1. Backproject pixel to ray on z=1 plane in camera frame
            // This handles undistortion and normalization internally
            let ray = camera.backproject_pixel(&Vec2::new(pixel.x, pixel.y));
            let ray_dir_camera = ray.point.normalize();

            // 2. Transform ray to target frame
            // T_T_C = T_C_T^-1
            let ray_origin_camera = Point3::origin();
            let ray_origin_target = camera_pose.inverse_transform_point(&ray_origin_camera);
            let ray_dir_target = camera_pose.inverse_transform_vector(&ray_dir_camera);

            // 3. Intersect ray with target plane (Z=0)
            // Ray: p(t) = ray_origin_target + t * ray_dir_target
            // Plane: Z = 0
            // Solve: ray_origin_target.z + t * ray_dir_target.z = 0
            if ray_dir_target.z.abs() < 1e-12 {
                return Err(LinescanError::RayParallelToTargetPlane);
            }

            let t = -ray_origin_target.z / ray_dir_target.z;
            let pt_target = ray_origin_target + ray_dir_target * t;

            // 4. Transform back to camera frame
            let pt_camera = camera_pose.transform_point(&pt_target);
            points_camera.push(pt_camera);
        }

        Ok(points_camera)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_from_perfect_points() {
        // Ground truth plane: z = 0.5 (parallel to XY plane)
        // Normal: [0, 0, 1], distance: -0.5
        let points = vec![
            Pt3::new(0.0, 0.0, 0.5),
            Pt3::new(1.0, 0.0, 0.5),
            Pt3::new(0.0, 1.0, 0.5),
            Pt3::new(1.0, 1.0, 0.5),
            Pt3::new(0.5, 0.5, 0.5),
        ];

        let result = LinescanPlaneSolver::from_points_3d(&points).unwrap();

        // Check normal (should be close to [0, 0, 1])
        assert!((result.normal.z.abs() - 1.0).abs() < 1e-6);
        assert!(result.normal.x.abs() < 1e-6);
        assert!(result.normal.y.abs() < 1e-6);

        // Check distance (should be close to -0.5)
        assert!((result.distance + 0.5).abs() < 1e-6);

        // Check RMSE (should be near zero)
        assert!(result.rmse < 1e-10);
    }

    #[test]
    fn plane_from_tilted_points() {
        // Tilted plane: z = 0.5 * x + 0.3
        // Rewrite as: -0.5*x + 0*y + 1*z - 0.3 = 0
        // Normal: [-0.5, 0, 1] / sqrt(1.25) ≈ [-0.447, 0, 0.894]
        let points = vec![
            Pt3::new(0.0, 0.0, 0.3),
            Pt3::new(1.0, 0.0, 0.8),
            Pt3::new(0.0, 1.0, 0.3),
            Pt3::new(1.0, 1.0, 0.8),
            Pt3::new(0.5, 0.5, 0.55),
        ];

        let result = LinescanPlaneSolver::from_points_3d(&points).unwrap();

        // Normal should be close to [-0.447, 0, 0.894]
        let expected_normal = Vector3::new(-0.5, 0.0, 1.0).normalize();
        let dot = result.normal.dot(&expected_normal);
        assert!(dot.abs() > 0.99, "normal mismatch: dot={}", dot);

        // RMSE should be small
        assert!(result.rmse < 1e-6);
    }

    #[test]
    fn plane_insufficient_points() {
        let points = vec![Pt3::new(0.0, 0.0, 0.5), Pt3::new(1.0, 0.0, 0.5)];
        assert!(LinescanPlaneSolver::from_points_3d(&points).is_err());
    }
}
