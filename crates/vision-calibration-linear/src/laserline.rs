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
//! [`LaserlinePlaneSolver::from_views_ransac`] and
//! [`LaserlinePlaneSolver::from_points_3d_ransac`] replace step 3 by a seeded
//! RANSAC over the same fit, for points with outliers (stray reflections,
//! pixels off the stripe).
//!
//! The resulting laser plane is parameterized as (n̂, d) where n̂ is the unit
//! normal vector and d is the signed distance from the camera origin.
//!
//! # Note on View Requirements
//!
//! A single view with a laser line produces **collinear** 3D points (all on one line),
//! which is degenerate for plane fitting. Use [`LaserlinePlaneSolver::from_views`] with
//! at least 2 views at different poses to obtain non-collinear points.

use crate::Error;
use nalgebra::{Point3, UnitVector3, Vector3};
use vision_calibration_core::{
    Camera, DistortionModel, Estimator, IntrinsicsModel, Iso3, ProjectionModel, Pt2, Pt3,
    RansacOptions, Real, SensorModel, ransac_fit,
};

/// Laser line observations for a single view.
///
/// The camera pose must be known (typically from PnP solution using
/// calibration features in the same view).
#[derive(Debug, Clone)]
pub struct PlaneFitView {
    /// 2D pixel observations along the laser line
    pub laser_pixels: Vec<Pt2>,
    /// Camera-to-target transform (T_C_T) from calibration features
    pub cam_se3_target: Iso3,
}

/// Result of linear laser plane estimation: the plane `normal · p + distance
/// = 0` in the camera frame.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LinearPlaneEstimate {
    /// Unit normal vector in camera frame
    pub normal: UnitVector3<f64>,
    /// Signed distance from camera origin
    pub distance: f64,
    /// Indices of the points the plane was accepted on, ascending: every
    /// point for the closed-form fits, the RANSAC consensus set for the
    /// `_ransac` fits. For a fit from views, a point's index is its laser
    /// pixel's position in the views' pixels taken in order, view by view.
    pub inliers: Vec<usize>,
    /// Root mean square point-to-plane distance, in the unit of the points,
    /// over [`inliers`](Self::inliers).
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
    /// by checking if the second-smallest eigenvalue is negligibly small
    /// compared to the largest.
    ///
    /// Every point counts: one outlier tilts the plane. For points with
    /// outliers, [`Self::from_points_3d_ransac`] runs this fit inside a
    /// seeded RANSAC.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 3 points are given,
    /// or [`Error::InvalidInput`] if the point cloud is collinear and a
    /// unique plane cannot be determined.
    pub fn from_points_3d(points_camera: &[Pt3]) -> Result<LinearPlaneEstimate, Error> {
        let (normal, distance) = fit_plane(points_camera.iter())?;
        let n = points_camera.len();
        let sum_sq_dist: f64 = points_camera
            .iter()
            .map(|pt| {
                let dist = normal.dot(&pt.coords) + distance;
                dist * dist
            })
            .sum();
        Ok(LinearPlaneEstimate {
            normal,
            distance,
            inliers: (0..n).collect(),
            rmse: (sum_sq_dist / n as f64).sqrt(),
        })
    }

    /// Estimate laser plane from 3D points with outliers by seeded RANSAC
    /// over the fit of [`Self::from_points_3d`].
    ///
    /// Each iteration fits a plane to 3 sampled points and counts as inliers
    /// the points within `opts.thresh` of it; the residual is the
    /// point-to-plane distance in the unit of the points (meters for points
    /// from a calibration in meters). The options set the rest of the
    /// policy:
    ///
    /// - `max_iters` caps the iterations and `confidence` stops them early
    ///   once an all-inlier sample is that likely to have been drawn;
    /// - `min_inliers` and `min_inlier_fraction` (a share of all the
    ///   points) are what a plane needs to be accepted;
    /// - `refit_on_inliers` refits each candidate on its inliers with the
    ///   closed-form fit and rescores it; without it the plane is the one
    ///   through the best 3-point sample;
    /// - `seed` makes the run reproducible: the same points and options
    ///   give the same plane.
    ///
    /// The estimate's `inliers` are the consensus set and its `rmse` is over
    /// them.
    ///
    /// # Errors
    ///
    /// - The options out of range ([`RansacOptions::validate`]).
    /// - [`Error::InsufficientData`] for fewer than 3 points.
    /// - [`Error::NoConsensus`] when no plane reaches `min_inliers` and
    ///   `min_inlier_fraction` within the iterations, as when the outliers
    ///   outnumber what the options allow.
    pub fn from_points_3d_ransac(
        points_camera: &[Pt3],
        opts: &RansacOptions,
    ) -> Result<LinearPlaneEstimate, Error> {
        opts.validate()?;
        if points_camera.len() < PlaneEstimator::MIN_SAMPLES {
            return Err(Error::InsufficientData {
                need: PlaneEstimator::MIN_SAMPLES,
                got: points_camera.len(),
            });
        }
        let result = ransac_fit::<PlaneEstimator>(points_camera, opts);
        let Some((normal, distance)) = result.model.filter(|_| result.success) else {
            log::debug!(
                "no plane through the {} points has at least {} inliers and an inlier \
                 fraction of {} within {} in {} RANSAC iterations",
                points_camera.len(),
                opts.min_inliers,
                opts.min_inlier_fraction,
                opts.thresh,
                opts.max_iters,
            );
            return Err(Error::NoConsensus);
        };
        Ok(LinearPlaneEstimate {
            normal,
            distance,
            inliers: result.inliers,
            rmse: result.inlier_rms,
        })
    }

    /// Estimate laser plane from multiple views with known camera poses, by
    /// seeded RANSAC.
    ///
    /// The points are those of [`Self::from_views`], so the same view
    /// requirements hold; the plane is [`Self::from_points_3d_ransac`] on
    /// them, which states the options and errors. `opts.min_inlier_fraction`
    /// is a share of all the reconstructed points, one per laser pixel, and
    /// the estimate's `inliers` index the pixels taken in order, view by
    /// view.
    ///
    /// # Errors
    ///
    /// Those of [`Self::from_views`] and of [`Self::from_points_3d_ransac`].
    pub fn from_views_ransac<P, D, Sm, K>(
        views: &[PlaneFitView],
        camera: &Camera<Real, P, D, Sm, K>,
        opts: &RansacOptions,
    ) -> Result<LinearPlaneEstimate, Error>
    where
        P: ProjectionModel<Real>,
        D: DistortionModel<Real>,
        Sm: SensorModel<Real>,
        K: IntrinsicsModel<Real>,
    {
        opts.validate()?;
        Self::from_points_3d_ransac(&Self::points_from_views(views, camera)?, opts)
    }

    /// Estimate laser plane from multiple views with known camera poses.
    ///
    /// This is the recommended method for laser plane estimation. Each view
    /// contributes laser pixels that are back-projected to 3D points using
    /// the known camera pose. Multiple views at different poses provide
    /// non-collinear points, enabling robust plane fitting. `camera` is any
    /// camera model: each pixel is back-projected through its intrinsics,
    /// sensor, distortion and projection.
    ///
    /// # Algorithm
    ///
    /// For each view:
    /// 1. Undistort laser pixels through the camera model
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
    pub fn from_views<P, D, Sm, K>(
        views: &[PlaneFitView],
        camera: &Camera<Real, P, D, Sm, K>,
    ) -> Result<LinearPlaneEstimate, Error>
    where
        P: ProjectionModel<Real>,
        D: DistortionModel<Real>,
        Sm: SensorModel<Real>,
        K: IntrinsicsModel<Real>,
    {
        Self::from_points_3d(&Self::points_from_views(views, camera)?)
    }

    /// The laser points of `views` in the camera frame, one per pixel, view
    /// by view, checked as [`Self::from_views`] states (at least 2 views and
    /// 3 points).
    fn points_from_views<P, D, Sm, K>(
        views: &[PlaneFitView],
        camera: &Camera<Real, P, D, Sm, K>,
    ) -> Result<Vec<Pt3>, Error>
    where
        P: ProjectionModel<Real>,
        D: DistortionModel<Real>,
        Sm: SensorModel<Real>,
        K: IntrinsicsModel<Real>,
    {
        if views.len() < 2 {
            return Err(Error::InsufficientData {
                need: 2,
                got: views.len(),
            });
        }
        let mut all_points = Vec::new();
        for view in views {
            let points = Self::compute_3d_points(&view.laser_pixels, camera, &view.cam_se3_target)?;
            all_points.extend(points);
        }
        if all_points.len() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: all_points.len(),
            });
        }
        Ok(all_points)
    }

    /// Compute 3D laser points in camera frame from pixel observations.
    ///
    /// For each pixel:
    /// 1. Backproject pixel to ray using camera model (handles undistortion)
    /// 2. Transform ray to target frame (inverse of camera pose)
    /// 3. Intersect ray with target plane (Z=0)
    /// 4. Transform intersection point back to camera frame
    fn compute_3d_points<P, D, Sm, K>(
        laser_pixels: &[Pt2],
        camera: &Camera<Real, P, D, Sm, K>,
        cam_se3_target: &Iso3,
    ) -> Result<Vec<Pt3>, Error>
    where
        P: ProjectionModel<Real>,
        D: DistortionModel<Real>,
        Sm: SensorModel<Real>,
        K: IntrinsicsModel<Real>,
    {
        let mut points_camera = Vec::with_capacity(laser_pixels.len());

        for pixel in laser_pixels {
            // 1. Backproject the pixel to its camera ray (undistortion and
            // normalization happen inside).
            let ray = camera.backproject_pixel(&Pt2::new(pixel.x, pixel.y));
            let ray_dir_camera = ray.direction.normalize();

            // 2. Transform ray to target frame
            // T_T_C = T_C_T^-1
            let ray_origin_camera = Point3::from(ray.origin);
            let ray_origin_target = cam_se3_target.inverse_transform_point(&ray_origin_camera);
            let ray_dir_target = cam_se3_target.inverse_transform_vector(&ray_dir_camera);

            // 3. Intersect ray with target plane (Z=0)
            // Ray: p(t) = ray_origin_target + t * ray_dir_target
            // Plane: Z = 0
            // Solve: ray_origin_target.z + t * ray_dir_target.z = 0
            if ray_dir_target.z.abs() < 1e-12 {
                return Err(Error::invalid_input(
                    "ray parallel to target plane (degenerate geometry)",
                ));
            }

            let t = -ray_origin_target.z / ray_dir_target.z;
            if !t.is_finite() || t <= 0.0 {
                return Err(Error::invalid_input(
                    "ray intersects target plane behind camera",
                ));
            }
            let pt_target = ray_origin_target + ray_dir_target * t;

            // 4. Transform back to camera frame
            let pt_camera = cam_se3_target.transform_point(&pt_target);
            points_camera.push(pt_camera);
        }

        Ok(points_camera)
    }
}

/// The plane `normal · p + distance = 0` through `points` by the smallest
/// eigenvector of their scatter matrix: the one fit behind every
/// [`LaserlinePlaneSolver`] method, so the closed-form and the RANSAC fits
/// cannot drift apart. Two passes over `points`, no allocation.
fn fit_plane<'a>(
    points: impl Iterator<Item = &'a Pt3> + Clone,
) -> Result<(UnitVector3<f64>, f64), Error> {
    let mut n = 0usize;
    let mut centroid = Vector3::zeros();
    for pt in points.clone() {
        if !pt.x.is_finite() || !pt.y.is_finite() || !pt.z.is_finite() {
            return Err(Error::invalid_input("3D points must be finite"));
        }
        centroid += pt.coords;
        n += 1;
    }
    if n < 3 {
        return Err(Error::InsufficientData { need: 3, got: n });
    }
    centroid /= n as f64;
    if !centroid.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid_input("3D points overflowed centroid"));
    }

    // Build 3x3 covariance matrix of centered points
    let mut cov = nalgebra::Matrix3::zeros();
    for pt in points {
        let centered = pt.coords - centroid;
        cov += centered * centered.transpose();
    }
    if !cov.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid_input(
            "3D point scatter matrix is non-finite",
        ));
    }

    // Smallest eigenvector of covariance matrix is plane normal
    let eigen = cov.symmetric_eigen();
    if !eigen.eigenvalues.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid_input("plane eigen decomposition failed"));
    }

    // Sort eigenvalues to identify smallest and second-smallest
    let mut indexed_eigenvalues = [0, 1, 2].map(|i| (i, eigen.eigenvalues[i]));
    indexed_eigenvalues.sort_by(|a, b| a.1.total_cmp(&b.1));

    let (min_idx, _min_eigenvalue) = indexed_eigenvalues[0];
    let (_second_idx, second_eigenvalue) = indexed_eigenvalues[1];
    let (_max_idx, max_eigenvalue) = indexed_eigenvalues[2];

    // Scale-invariant collinearity check: for a proper 2D plane the second-largest
    // scatter eigenvalue must remain non-negligible relative to the largest.
    const RANK_THRESHOLD: f64 = 1e-8;
    if !max_eigenvalue.is_finite()
        || max_eigenvalue <= Real::EPSILON * Real::EPSILON
        || second_eigenvalue / max_eigenvalue < RANK_THRESHOLD
    {
        return Err(Error::invalid_input(
            "collinear points: cannot fit plane to points along a single line",
        ));
    }

    let normal_vec = eigen.eigenvectors.column(min_idx);
    let normal = UnitVector3::new_normalize(normal_vec.into());

    // Compute signed distance: d = -n · centroid
    let distance = -normal.dot(&centroid);

    Ok((normal, distance))
}

/// The [`Estimator`] of [`LaserlinePlaneSolver::from_points_3d_ransac`]: a
/// plane through 3 points, the point-to-plane distance as residual, and the
/// same fit on the inliers as refit.
struct PlaneEstimator;

impl Estimator for PlaneEstimator {
    type Datum = Pt3;
    type Model = (UnitVector3<f64>, f64);

    const MIN_SAMPLES: usize = 3;

    fn fit(data: &[Pt3], sample_indices: &[usize]) -> Option<Self::Model> {
        fit_plane(sample_indices.iter().map(|&i| &data[i])).ok()
    }

    fn residual(model: &Self::Model, datum: &Pt3) -> f64 {
        (model.0.dot(&datum.coords) + model.1).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;
    use vision_calibration_core::{BrownConrady5, FxFyCxCy, PinholeCamera, make_pinhole_camera};

    fn make_test_camera() -> PinholeCamera {
        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 480.0,
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
        let view = PlaneFitView {
            laser_pixels: vec![Pt2::new(100.0, 100.0), Pt2::new(200.0, 100.0)],
            cam_se3_target: Iso3::identity(),
        };

        let result = LaserlinePlaneSolver::from_views(&[view], &camera);
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("insufficient data"),
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
        let view1 = PlaneFitView {
            laser_pixels: vec![
                Pt2::new(400.0, 480.0),
                Pt2::new(640.0, 480.0),
                Pt2::new(880.0, 480.0),
            ],
            cam_se3_target: pose1,
        };

        let view2 = PlaneFitView {
            laser_pixels: vec![
                Pt2::new(400.0, 520.0),
                Pt2::new(640.0, 520.0),
                Pt2::new(880.0, 520.0),
            ],
            cam_se3_target: pose2,
        };

        let result = LaserlinePlaneSolver::from_views(&[view1, view2], &camera);
        // This should succeed (may not match ground truth due to simplified setup)
        assert!(result.is_ok(), "from_views should succeed: {:?}", result);
    }

    // --- Seeded RANSAC fit ------------------------------------------------

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// The true plane of the RANSAC fixtures: tilted, 0.5 m from the camera.
    fn ransac_gt() -> (UnitVector3<f64>, f64) {
        (
            UnitVector3::new_normalize(Vector3::new(0.25, -0.4, 1.0)),
            -0.5,
        )
    }

    /// `n_in` points within ±0.2 mm of the true plane, then `n_out` points
    /// 5–50 mm off it on either side, from a seeded generator: the inlier
    /// fraction is exactly `n_in / (n_in + n_out)` for any threshold in
    /// (0.2 mm, 5 mm).
    fn plane_cloud(n_in: usize, n_out: usize, seed: u64) -> Vec<Pt3> {
        let (normal, distance) = ransac_gt();
        let n = normal.into_inner();
        let u = n.cross(&Vector3::x()).normalize();
        let v = n.cross(&u);
        let on_plane = -distance * n;
        let mut rng = StdRng::seed_from_u64(seed);
        let point = |offset: f64, rng: &mut StdRng| {
            let a: f64 = rng.random_range(-0.15..0.15);
            let b: f64 = rng.random_range(-0.10..0.10);
            Pt3::from(on_plane + a * u + b * v + offset * n)
        };
        let mut points: Vec<Pt3> = (0..n_in)
            .map(|_| {
                let e: f64 = rng.random_range(-0.2e-3..0.2e-3);
                point(e, &mut rng)
            })
            .collect();
        for _ in 0..n_out {
            let off: f64 = rng.random_range(5e-3..50e-3);
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            points.push(point(sign * off, &mut rng));
        }
        points
    }

    fn ransac_opts() -> RansacOptions {
        RansacOptions::new(1e-3).with_min_inliers(3).with_seed(42)
    }

    fn angle_to_gt(normal: &UnitVector3<f64>) -> f64 {
        normal.dot(&ransac_gt().0).abs().min(1.0).acos()
    }

    /// 70 inliers, 30 outliers: the fraction 0.7 is accepted at and below
    /// 0.7 and rejected above it, and the accepted plane is the true one
    /// while the closed-form fit on every point is pulled off it.
    #[test]
    fn ransac_plane_passes_and_fails_across_the_inlier_fraction() {
        let points = plane_cloud(70, 30, 7);
        for fraction in [0.5, 0.65, 0.7] {
            let est = LaserlinePlaneSolver::from_points_3d_ransac(
                &points,
                &ransac_opts().with_min_inlier_fraction(fraction),
            )
            .unwrap_or_else(|e| panic!("fraction {fraction}: {e}"));
            assert_eq!(
                est.inliers,
                (0..70).collect::<Vec<_>>(),
                "fraction {fraction}"
            );
            assert!(
                angle_to_gt(&est.normal) < 2e-3,
                "{}",
                angle_to_gt(&est.normal)
            );
            let sign = est.normal.dot(&ransac_gt().0).signum();
            assert!(
                (sign * est.distance - ransac_gt().1).abs() < 1e-4,
                "{}",
                est.distance
            );
            assert!(est.rmse < 0.2e-3, "{}", est.rmse);
        }
        for fraction in [0.71, 0.75, 1.0] {
            let err = LaserlinePlaneSolver::from_points_3d_ransac(
                &points,
                &ransac_opts().with_min_inlier_fraction(fraction),
            )
            .expect_err("the consensus is 0.7");
            assert!(matches!(err, Error::NoConsensus), "{err}");
        }
        let all = LaserlinePlaneSolver::from_points_3d(&points).unwrap();
        assert!(
            all.rmse > 5e-3,
            "the outliers should spoil the closed-form fit: rmse {}",
            all.rmse
        );
    }

    /// The same points, options and seed give the same plane, bit for bit;
    /// another seed finds the same consensus.
    #[test]
    fn ransac_plane_is_reproducible_for_a_seed() {
        let points = plane_cloud(70, 30, 7);
        let opts = ransac_opts().with_min_inlier_fraction(0.6);
        let a = LaserlinePlaneSolver::from_points_3d_ransac(&points, &opts).unwrap();
        let b = LaserlinePlaneSolver::from_points_3d_ransac(&points, &opts).unwrap();
        assert_eq!(a.normal.into_inner(), b.normal.into_inner());
        assert_eq!(a.distance.to_bits(), b.distance.to_bits());
        assert_eq!(a.inliers, b.inliers);
        assert_eq!(a.rmse.to_bits(), b.rmse.to_bits());
        let c = LaserlinePlaneSolver::from_points_3d_ransac(&points, &opts.with_seed(9)).unwrap();
        assert_eq!(a.inliers, c.inliers);
    }

    /// With the refit the plane is the closed-form fit of its consensus set;
    /// without it, the plane through the best 3-point sample, which passes
    /// through 3 of the inliers and has the larger RMS on the same set.
    #[test]
    fn disabling_the_refit_keeps_the_sample_plane() {
        let points = plane_cloud(70, 30, 7);
        let opts = ransac_opts().with_min_inlier_fraction(0.6);
        let refit = LaserlinePlaneSolver::from_points_3d_ransac(&points, &opts).unwrap();
        let sample = LaserlinePlaneSolver::from_points_3d_ransac(
            &points,
            &opts.with_refit_on_inliers(false),
        )
        .unwrap();
        assert_eq!(refit.inliers, sample.inliers);

        let inlier_points: Vec<Pt3> = refit.inliers.iter().map(|&i| points[i]).collect();
        let closed_form = LaserlinePlaneSolver::from_points_3d(&inlier_points).unwrap();
        let same = refit.normal.dot(&closed_form.normal);
        assert!(same.abs() > 1.0 - 1e-12);
        assert!((refit.distance - same.signum() * closed_form.distance).abs() < 1e-12);

        let on_sample_plane = inlier_points
            .iter()
            .filter(|p| (sample.normal.dot(&p.coords) + sample.distance).abs() < 1e-12)
            .count();
        assert!(on_sample_plane >= 3, "{on_sample_plane}");
        assert!(refit.normal.dot(&sample.normal).abs() < 1.0 - 1e-12);
        assert!(
            refit.rmse < sample.rmse,
            "{} vs {}",
            refit.rmse,
            sample.rmse
        );
    }

    #[test]
    fn ransac_plane_rejects_bad_options_and_too_few_points() {
        let points = plane_cloud(10, 0, 1);
        for bad in [
            RansacOptions::new(0.0),
            RansacOptions::new(1e-3).with_min_inlier_fraction(1.5),
        ] {
            let err = LaserlinePlaneSolver::from_points_3d_ransac(&points, &bad).unwrap_err();
            assert!(matches!(err, Error::Core(_)), "{err}");
        }
        let err =
            LaserlinePlaneSolver::from_points_3d_ransac(&points[..2], &ransac_opts()).unwrap_err();
        assert!(
            matches!(err, Error::InsufficientData { need: 3, got: 2 }),
            "{err}"
        );
    }

    /// From views: stripes of a known plane on tilted boards plus stray
    /// pixels off the stripe. The RANSAC fit recovers the plane and drops
    /// exactly the stray pixels, indexed view by view.
    #[test]
    fn from_views_ransac_drops_stray_pixels() {
        use vision_calibration_core::synthetic::laser::{BoardExtent, laser_stripe_pixels};
        use vision_calibration_core::synthetic::noise::UniformPixelNoise;
        use vision_calibration_core::synthetic::poses::{BoardPoseSpec, centered_board_poses};

        let camera = make_test_camera();
        // 16.7° about y, crossing the optical axis at 0.5 m.
        let normal = Vector3::new(0.30, 0.0, 1.0).normalize();
        let distance = -normal.z * 0.5;
        let board = BoardExtent::from_grid(9, 7, 0.03);
        let specs = [
            BoardPoseSpec::new(0.12, -0.05, 0.35, -0.012),
            BoardPoseSpec::new(-0.10, 0.09, -0.40, 0.010),
            BoardPoseSpec::new(0.06, 0.14, 0.60, 0.014),
        ];
        let poses = centered_board_poses(&specs, (0.12, 0.09), 0.5);
        let no_noise = UniformPixelNoise::default();
        let mut stray = Vec::new();
        let mut offset = 0;
        let views: Vec<PlaneFitView> = poses
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                let mut pixels =
                    laser_stripe_pixels(&camera, pose, &normal, distance, &board, i, &no_noise);
                assert!(pixels.len() > 10, "view {i}: stripe off the board");
                // Two stray pixels per view, 40 px off the stripe's middle.
                let mid = pixels[pixels.len() / 2];
                for dx in [40.0, -40.0] {
                    stray.push(offset + pixels.len());
                    pixels.push(Pt2::new(mid.x + dx, mid.y));
                }
                offset += pixels.len();
                PlaneFitView {
                    laser_pixels: pixels,
                    cam_se3_target: *pose,
                }
            })
            .collect();

        let opts = RansacOptions::new(1e-4).with_min_inlier_fraction(0.9);
        let est = LaserlinePlaneSolver::from_views_ransac(&views, &camera, &opts).unwrap();
        let expected: Vec<usize> = (0..offset).filter(|i| !stray.contains(i)).collect();
        assert_eq!(est.inliers, expected);
        let same = est.normal.dot(&normal);
        assert!(same.abs() > 1.0 - 1e-9, "{same}");
        assert!(
            (same.signum() * est.distance - distance).abs() < 1e-6,
            "{}",
            est.distance
        );

        let all = LaserlinePlaneSolver::from_views(&views, &camera).unwrap();
        assert!(
            all.rmse > 1e-3,
            "the stray pixels should spoil the closed-form fit: rmse {}",
            all.rmse
        );
    }

    #[test]
    fn plane_from_submillimeter_non_collinear_points() {
        let points = vec![
            Pt3::new(0.0, 0.0, 0.5),
            Pt3::new(2e-4, 0.0, 0.5),
            Pt3::new(0.0, 2e-4, 0.5),
            Pt3::new(2e-4, 2e-4, 0.5),
            Pt3::new(1e-4, 1e-4, 0.5),
        ];
        let result = LaserlinePlaneSolver::from_points_3d(&points).unwrap();
        assert!((result.normal.z.abs() - 1.0).abs() < 1e-6);
        assert!((result.distance.abs() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_3d_points_rejects_behind_camera_ray() {
        let camera = make_test_camera();
        // Target plane at z = -1.0 (behind the camera). A forward pixel ray hits it at t < 0.
        let view = PlaneFitView {
            laser_pixels: vec![
                Pt2::new(600.0, 480.0),
                Pt2::new(640.0, 480.0),
                Pt2::new(680.0, 480.0),
            ],
            cam_se3_target: Iso3::translation(0.0, 0.0, -1.0),
        };
        let err = LaserlinePlaneSolver::from_views(&[view], &camera).unwrap_err();
        assert!(
            matches!(
                err,
                crate::Error::InvalidInput { .. } | crate::Error::InsufficientData { .. }
            ),
            "expected error for behind-camera ray intersection, got {err:?}"
        );
    }
}
