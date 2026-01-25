//! Laserline plane estimation from planar target observations.
//!
//! This module provides linear closed-form estimation of laser plane parameters
//! from pixel observations of a laser line projected onto a planar calibration target.
//!
//! # Algorithm
//!
//! Given:
//! - Laser line pixel observations from multiple views
//! - Camera poses (T_C_T) from calibration features (via PnP/homography)
//! - Known planar target geometry (Z=0 in target frame)
//!
//! The algorithm:
//! 1. Back-projects each laser pixel to a ray in camera frame
//! 2. Intersects each ray with the target plane to get 3D points
//! 3. Fits a plane to the 3D points via SVD (covariance eigenvector)
//!
//! The resulting laser plane is parameterized as (n̂, d) where n̂ is the unit
//! normal vector and d is the signed distance from the camera origin.
//!
//! # Note on View Requirements
//!
//! A single view with a laser line produces **collinear** 3D points (all on one line),
//! which is degenerate for plane fitting. Use [`LaserlinePlaneSolver::from_views`] with
//! at least 2 views at different poses to obtain non-collinear points.

use anyhow::Result;
use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, Iso3, Pinhole, Pt2, Pt3, Real, SensorModel};
use nalgebra::{Point3, UnitVector3, Vector3};

/// Laser line observations for a single view.
///
/// The camera pose must be known (typically from PnP solution using
/// calibration features in the same view).
#[derive(Debug, Clone)]
pub struct LaserlineView {
    /// 2D pixel observations along the laser line
    pub laser_pixels: Vec<Pt2>,
    /// Camera-to-target transform (T_C_T) from calibration features
    pub camera_se3_target: Iso3,
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
pub struct LaserlinePlaneSolver;

impl LaserlinePlaneSolver {
    /// Estimate laser plane from 3D points in camera frame via SVD.
    ///
    /// Fits a plane ax + by + cz + d = 0 to the given 3D points using
    /// covariance eigendecomposition. Returns the plane parameterized as
    /// unit normal (n̂) and signed distance from the camera origin.
    ///
    /// # Algorithm
    ///
    /// 1. Compute centroid of points
    /// 2. Build 3x3 covariance matrix of centered points
    /// 3. Find smallest eigenvector (plane normal)
    /// 4. Compute distance as -n̂ · centroid
    ///
    /// Requires at least 3 non-collinear points. Collinearity is detected
    /// by checking if the smallest eigenvalue is negligibly small compared
    /// to the second-smallest.
    pub fn from_points_3d(points_camera: &[Pt3]) -> Result<LinearPlaneEstimate> {
        if points_camera.len() < 3 {
            anyhow::bail!(
                "insufficient points: got {}, need at least {}",
                points_camera.len(),
                3
            );
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

        // Sort eigenvalues to identify smallest and second-smallest
        let mut indexed_eigenvalues: Vec<(usize, f64)> =
            eigen.eigenvalues.iter().copied().enumerate().collect();
        indexed_eigenvalues.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (min_idx, min_eigenvalue) = indexed_eigenvalues[0];
        let (_second_idx, second_eigenvalue) = indexed_eigenvalues[1];
        let (_max_idx, max_eigenvalue) = indexed_eigenvalues[2];

        // Check for collinearity: points are collinear if the smallest eigenvalue
        // is negligible AND the second-smallest is significant (i.e., rank 1).
        // For a proper plane, we expect rank 2: two significant eigenvalues.
        // For collinear points (rank 1): only one significant eigenvalue.
        // The ratio of second-smallest to largest should be significant for planes.
        const RANK_THRESHOLD: f64 = 1e-8;
        if max_eigenvalue > RANK_THRESHOLD {
            // Relative threshold: check if points span 2D (plane) or 1D (line)
            let rank1_ratio = second_eigenvalue / max_eigenvalue;
            if rank1_ratio < RANK_THRESHOLD {
                anyhow::bail!("collinear points: cannot fit plane to points along a single line");
            }
        } else if min_eigenvalue.abs() < RANK_THRESHOLD && second_eigenvalue < RANK_THRESHOLD {
            // All eigenvalues near zero: degenerate (all points at same location)
            anyhow::bail!("collinear points: cannot fit plane to points along a single line");
        }

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

    /// Estimate laser plane from multiple views with known camera poses.
    ///
    /// This is the recommended method for laser plane estimation. Each view
    /// contributes laser pixels that are back-projected to 3D points using
    /// the known camera pose. Multiple views at different poses provide
    /// non-collinear points, enabling robust plane fitting.
    ///
    /// # Algorithm
    ///
    /// For each view:
    /// 1. Undistort laser pixels using camera intrinsics and distortion
    /// 2. Back-project to rays in camera frame
    /// 3. Intersect rays with planar target (Z=0 in target frame)
    /// 4. Transform intersection points to camera frame
    ///
    /// Then fit plane to all 3D points via [`Self::from_points_3d`].
    ///
    /// # Requirements
    ///
    /// - At least 2 views with different poses
    /// - Total points across views >= 3
    /// - Views must be at sufficiently different angles to break collinearity
    pub fn from_views<Sm>(
        views: &[LaserlineView],
        camera: &Camera<Real, Pinhole, BrownConrady5<Real>, Sm, FxFyCxCySkew<Real>>,
    ) -> Result<LinearPlaneEstimate>
    where
        Sm: SensorModel<Real>,
    {
        if views.len() < 2 {
            anyhow::bail!(
                "insufficient views: got {}, need at least {}",
                views.len(),
                2
            );
        }

        // Gather all 3D points from all views
        let mut all_points = Vec::new();
        for view in views {
            let points =
                Self::compute_3d_points(&view.laser_pixels, camera, &view.camera_se3_target)?;
            all_points.extend(points);
        }

        if all_points.len() < 3 {
            anyhow::bail!(
                "insufficient points: got {}, need at least {}",
                all_points.len(),
                3
            );
        }

        // Fit plane to 3D points
        Self::from_points_3d(&all_points)
    }

    /// Estimate laser plane from a single view (DEPRECATED).
    ///
    /// **Warning**: A single view produces collinear 3D points, which is
    /// degenerate for plane fitting. This method will likely fail with
    /// an error because a single view produces collinear points.
    ///
    /// Use [`Self::from_views`] with multiple views instead.
    #[deprecated(
        since = "0.2.0",
        note = "Single view produces collinear points. Use from_views() with multiple views."
    )]
    pub fn from_view<Sm>(
        view: &LaserlineView,
        camera: &Camera<Real, Pinhole, BrownConrady5<Real>, Sm, FxFyCxCySkew<Real>>,
    ) -> Result<LinearPlaneEstimate>
    where
        Sm: SensorModel<Real>,
    {
        if view.laser_pixels.is_empty() {
            anyhow::bail!("insufficient points: got 0, need at least 1");
        }

        // Compute 3D points in camera frame
        let points_camera =
            Self::compute_3d_points(&view.laser_pixels, camera, &view.camera_se3_target)?;

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
    fn compute_3d_points<Sm>(
        laser_pixels: &[Pt2],
        camera: &Camera<Real, Pinhole, BrownConrady5<Real>, Sm, FxFyCxCySkew<Real>>,
        camera_se3_target: &Iso3,
    ) -> Result<Vec<Pt3>>
    where
        Sm: SensorModel<Real>,
    {
        let mut points_camera = Vec::with_capacity(laser_pixels.len());

        for pixel in laser_pixels {
            // 1. Backproject pixel to ray on z=1 plane in camera frame
            // This handles undistortion and normalization internally
            let ray = camera.backproject_pixel(&Pt2::new(pixel.x, pixel.y));
            let ray_dir_camera = ray.point.normalize();

            // 2. Transform ray to target frame
            // T_T_C = T_C_T^-1
            let ray_origin_camera = Point3::origin();
            let ray_origin_target = camera_se3_target.inverse_transform_point(&ray_origin_camera);
            let ray_dir_target = camera_se3_target.inverse_transform_vector(&ray_dir_camera);

            // 3. Intersect ray with target plane (Z=0)
            // Ray: p(t) = ray_origin_target + t * ray_dir_target
            // Plane: Z = 0
            // Solve: ray_origin_target.z + t * ray_dir_target.z = 0
            if ray_dir_target.z.abs() < 1e-12 {
                anyhow::bail!("ray parallel to target plane (degenerate geometry)");
            }

            let t = -ray_origin_target.z / ray_dir_target.z;
            let pt_target = ray_origin_target + ray_dir_target * t;

            // 4. Transform back to camera frame
            let pt_camera = camera_se3_target.transform_point(&pt_target);
            points_camera.push(pt_camera);
        }

        Ok(points_camera)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{make_pinhole_camera, PinholeCamera};
    use nalgebra::UnitQuaternion;

    fn make_test_camera() -> PinholeCamera {
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
            skew: 0.0,
        };
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        make_pinhole_camera(intrinsics, distortion)
    }

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

        let result = LaserlinePlaneSolver::from_points_3d(&points).unwrap();

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

        let result = LaserlinePlaneSolver::from_points_3d(&points).unwrap();

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
        assert!(LaserlinePlaneSolver::from_points_3d(&points).is_err());
    }

    #[test]
    fn plane_detects_collinear_points() {
        // Collinear points (all on a line along X axis)
        let points = vec![
            Pt3::new(0.0, 0.0, 0.5),
            Pt3::new(1.0, 0.0, 0.5),
            Pt3::new(2.0, 0.0, 0.5),
            Pt3::new(3.0, 0.0, 0.5),
        ];

        let result = LaserlinePlaneSolver::from_points_3d(&points);
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("collinear points"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn from_views_insufficient_views() {
        let camera = make_test_camera();
        let view = LaserlineView {
            laser_pixels: vec![Pt2::new(100.0, 100.0), Pt2::new(200.0, 100.0)],
            camera_se3_target: Iso3::identity(),
        };

        let result = LaserlinePlaneSolver::from_views(&[view], &camera);
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("insufficient views"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn from_views_with_two_poses() {
        let camera = make_test_camera();

        // Ground truth laser plane: z = -0.5 in camera frame (facing towards target)
        // We simulate laser hitting target at different Y positions

        // Pose 1: camera at Z=1.0 looking at target at Z=0
        let pose1 = Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            UnitQuaternion::identity(),
        );

        // Pose 2: camera at Z=1.0, shifted in Y, with slight rotation
        let pose2 = Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.1, 1.0),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.1),
        );

        // Generate laser pixels that project to the laser plane
        // For a horizontal laser line at z=-0.5 in camera frame,
        // we simulate pixels along the laser line
        let view1 = LaserlineView {
            laser_pixels: vec![
                Pt2::new(400.0, 480.0),
                Pt2::new(640.0, 480.0),
                Pt2::new(880.0, 480.0),
            ],
            camera_se3_target: pose1,
        };

        let view2 = LaserlineView {
            laser_pixels: vec![
                Pt2::new(400.0, 520.0),
                Pt2::new(640.0, 520.0),
                Pt2::new(880.0, 520.0),
            ],
            camera_se3_target: pose2,
        };

        let result = LaserlinePlaneSolver::from_views(&[view1, view2], &camera);
        // This should succeed (may not match ground truth due to simplified setup)
        assert!(result.is_ok(), "from_views should succeed: {:?}", result);
    }
}
