//! Laserline bundle adjustment: joint optimization of a camera (any model),
//! target poses, and laser plane parameters.
//!
//! This problem builder combines standard planar calibration observations (corners)
//! with laser line observations to estimate laser plane parameters in camera frame.
//! The camera's model ([`CameraParams::spec`]) selects its parameter blocks: any
//! distortion kind, and an identity or Scheimpflug sensor (an untilted camera has
//! no sensor block).

use crate::Error;
#[cfg(test)]
use crate::backend::SolverBackend;
use crate::backend::{self, BackendSolveOptions, SolveReport};
use crate::factors::camera_blocks::CameraBlocks;
use crate::factors::laserline::{
    laser_line_dist_core, laser_line_distance, laser_point_to_plane_core,
};
use crate::ir::ReprojSqrtInfo;
use crate::ir::{
    FactorKind, FixedMask, LaserChain, ManifoldKind, ProblemIR, ReprojChain, ResidualBlock,
    RobustLoss,
};
use crate::params::camera::{
    CameraBlockInput, add_camera_blocks, camera_block_values, camera_spec, unpack_camera,
};
use crate::params::laser_plane::LaserPlane;
use crate::params::pose_se3::{iso3_to_se3_dvec, se3_dvec_to_iso3};
use crate::uncertainty;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vision_calibration_core::{
    CameraFix, CameraModel, CameraModelSpec, CameraParams, Iso3, Pt2, Uncertainty, View,
};

/// Metadata for a laserline view (laser pixels + optional weights).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct LaserlineMeta {
    /// Laser line pixel observations.
    #[cfg_attr(feature = "schemars", schemars(with = "Vec<[f64; 2]>"))]
    pub laser_pixels: Vec<Pt2>,
    /// Per-pixel weights (empty = unweighted, all default to 1.0).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub laser_weights: Vec<f64>,
}

impl LaserlineMeta {
    /// Validate metadata consistency (non-empty pixels and optional weights).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `laser_pixels` is empty.
    /// Returns [`Error::InvalidInput`] if weight count mismatches pixel count, or if any weight is negative.
    pub fn validate(&self) -> Result<(), Error> {
        if self.laser_pixels.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        if !self.laser_weights.is_empty() {
            if self.laser_weights.len() != self.laser_pixels.len() {
                return Err(Error::invalid_input(format!(
                    "laser weight count {} must match pixel count {}",
                    self.laser_weights.len(),
                    self.laser_pixels.len()
                )));
            }
            if !self.laser_weights.iter().all(|w| *w >= 0.0) {
                return Err(Error::invalid_input("laser weights must be non-negative"));
            }
        }
        Ok(())
    }

    /// Return per-pixel weight, defaulting to `1.0` when absent.
    pub fn laser_weight(&self, idx: usize) -> f64 {
        if self.laser_weights.is_empty() {
            1.0
        } else {
            self.laser_weights[idx]
        }
    }
}

/// A single view with calibration correspondences and laserline pixels.
pub type LaserlineView = View<LaserlineMeta>;

/// Dataset for laserline bundle adjustment.
pub type LaserlineDataset = Vec<LaserlineView>;

/// Initial values for laserline bundle adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct LaserlineParams {
    /// The camera, in its model: any distortion kind, and an identity or
    /// Scheimpflug sensor.
    pub camera: CameraParams,
    /// Per-view target poses (`camera_se3_target`).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub poses: Vec<Iso3>,
    /// Laser plane parameters in camera frame.
    pub plane: LaserPlane,
}

impl LaserlineParams {
    /// Construct parameter pack with basic cardinality validation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `poses` is empty.
    pub fn new(camera: CameraParams, poses: Vec<Iso3>, plane: LaserPlane) -> Result<Self, Error> {
        if poses.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        Ok(Self {
            camera,
            poses,
            plane,
        })
    }
}

/// Result of laserline bundle adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct LaserlineEstimate {
    /// Refined bundle parameters.
    pub params: LaserlineParams,
    /// Backend solve report.
    pub report: SolveReport,
    /// Covariance of the refined parameters: the camera's free parameters
    /// (intrinsics, distortion and, for a tilted sensor, the tilt) and the
    /// laser plane (`plane.nx … plane.d`, in the camera frame), with each
    /// view's target pose as a per-view block.
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

/// Summary statistics for a laserline calibration result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct LaserlineStats {
    /// Mean reprojection error for calibration points (pixels).
    pub mean_reproj_error: f64,
    /// Mean laser residual (units depend on residual type).
    pub mean_laser_error: f64,
    /// Per-view mean reprojection error (pixels).
    pub per_view_reproj_errors: Vec<f64>,
    /// Per-view mean laser residual (same units as `mean_laser_error`).
    pub per_view_laser_errors: Vec<f64>,
}

/// Type of laser plane residual to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum LaserlineResidualType {
    /// Point-to-plane distance (original approach).
    ///
    /// Undistorts pixel, back-projects to 3D ray, intersects with target plane,
    /// computes 3D point in camera frame, measures signed distance to laser plane.
    /// Residual: 1D distance in meters.
    PointToPlane,
    /// Line-distance in normalized plane (alternative approach).
    ///
    /// Computes 3D intersection line of laser plane and target plane, projects
    /// line onto z=1 normalized camera plane, undistorts laser pixels to normalized
    /// coordinates, measures perpendicular distance from pixel to projected line,
    /// scales by sqrt(fx*fy) for pixel-comparable residual.
    /// Residual: 1D distance in pixels.
    #[default]
    LineDistNormalized,
}

/// Solve options for laserline bundle adjustment.
#[derive(Debug, Clone)]
pub struct LaserlineSolveOptions {
    /// Robust loss applied to calibration reprojection residuals
    pub calib_loss: RobustLoss,
    /// Global weight multiplier for calibration reprojection residuals.
    pub calib_weight: f64,
    /// Robust loss applied to laser residuals. Its scale is in the residual's
    /// unit: pixels for the line distance, meters for point-to-plane.
    pub laser_loss: RobustLoss,
    /// Global weight multiplier for laser plane residuals.
    pub laser_weight: f64,
    /// Camera parameters held fixed, named per slot and resolved against the
    /// initial camera's model. The default holds the model's defaults (`k3`
    /// for Brown-Conrady; everything but the low-order radial terms under a
    /// Scheimpflug sensor).
    pub fix: CameraFix,
    /// Indices of target poses to hold fixed. None by default: each view's
    /// pose is measured by its own corners, so there is no gauge to fix. A
    /// held pose keeps its initial value, and the linear init is not robust
    /// to outlying corners: holding one keeps the solve off the minimum when
    /// some corners are outliers.
    pub fix_poses: Vec<usize>,
    /// Hold the laser plane fixed.
    pub fix_plane: bool,
    /// Laser residual type: point-to-plane distance or line-distance in normalized plane
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for LaserlineSolveOptions {
    fn default() -> Self {
        Self {
            calib_loss: RobustLoss::Huber { scale: 1.0 },
            calib_weight: 1.0,
            // 1 px, as for the corners: the default residual is the line
            // distance in pixels.
            laser_loss: RobustLoss::Huber { scale: 1.0 },
            laser_weight: 1.0,
            fix: CameraFix::default(),
            fix_poses: Vec::new(),
            fix_plane: false,
            laser_residual_type: LaserlineResidualType::LineDistNormalized, // New default
        }
    }
}

/// Compute reprojection and laser residual statistics for a solution.
///
/// - Reprojection errors are reported in pixels (mean Euclidean error).
/// - Laser residual units depend on `residual_type`:
///   - `PointToPlane`: meters (signed distance to plane)
///   - `LineDistNormalized`: pixels (normalized line distance scaled by sqrt(fx*fy))
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if the dataset and params have different view counts, or if
/// the camera has a general homography sensor.
pub fn compute_laserline_stats(
    dataset: &LaserlineDataset,
    params: &LaserlineParams,
    residual_type: LaserlineResidualType,
) -> Result<LaserlineStats, Error> {
    if dataset.len() != params.poses.len() {
        return Err(Error::invalid_input(format!(
            "dataset has {} views but {} poses",
            dataset.len(),
            params.poses.len()
        )));
    }

    let camera = LaserCameraModel::new(&params.camera)?;
    let plane_normal = params.plane.normal_to_dvec();
    let plane_distance = params.plane.distance_to_dvec();

    let mut per_view_reproj_errors = Vec::with_capacity(dataset.len());
    let mut per_view_laser_errors = Vec::with_capacity(dataset.len());

    let mut total_reproj_sum = 0.0;
    let mut total_reproj_count = 0usize;
    let mut total_laser_sum = 0.0;
    let mut total_laser_count = 0usize;

    for (view_idx, view) in dataset.iter().enumerate() {
        let pose = &params.poses[view_idx];
        let pose_vec = iso3_to_se3_dvec(pose);

        let mut view_reproj_sum = 0.0;
        let mut view_reproj_count = 0usize;
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            if let Some(projected) = camera.camera.project_point(&p_cam) {
                let err = (projected - *p2d).norm();
                view_reproj_sum += err;
                view_reproj_count += 1;
            }
        }

        let view_reproj_mean = if view_reproj_count > 0 {
            view_reproj_sum / view_reproj_count as f64
        } else {
            0.0
        };
        per_view_reproj_errors.push(view_reproj_mean);
        total_reproj_sum += view_reproj_sum;
        total_reproj_count += view_reproj_count;

        let mut view_laser_sum = 0.0;
        let mut view_laser_count = 0usize;
        for laser_pixel in &view.meta.laser_pixels {
            let residual = match residual_type {
                LaserlineResidualType::PointToPlane => laser_point_to_plane_core(
                    &camera.blocks(),
                    pose_vec.as_view(),
                    plane_normal.as_view(),
                    plane_distance.as_view(),
                    [laser_pixel.x, laser_pixel.y],
                    1.0,
                )[0],
                LaserlineResidualType::LineDistNormalized => laser_line_dist_core(
                    &camera.blocks(),
                    pose_vec.as_view(),
                    plane_normal.as_view(),
                    plane_distance.as_view(),
                    [laser_pixel.x, laser_pixel.y],
                    1.0,
                )[0],
            };
            view_laser_sum += residual.abs();
            view_laser_count += 1;
        }

        let view_laser_mean = if view_laser_count > 0 {
            view_laser_sum / view_laser_count as f64
        } else {
            0.0
        };
        per_view_laser_errors.push(view_laser_mean);
        total_laser_sum += view_laser_sum;
        total_laser_count += view_laser_count;
    }

    if total_reproj_count == 0 {
        return Err(Error::invalid_input(
            "no valid reprojection errors for stats",
        ));
    }
    if total_laser_count == 0 {
        return Err(Error::invalid_input("no laser pixels for stats"));
    }

    Ok(LaserlineStats {
        mean_reproj_error: total_reproj_sum / total_reproj_count as f64,
        mean_laser_error: total_laser_sum / total_laser_count as f64,
        per_view_reproj_errors,
        per_view_laser_errors,
    })
}

/// Per-pixel laser residual records and per-view projected laser lines.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LaserResidualRecords {
    /// One record per laser pixel, pose-major.
    pub features: Vec<vision_calibration_core::LaserFeatureResidual>,
    /// The projected laser line of each view with laser pixels.
    pub lines: Vec<vision_calibration_core::LaserLineTrace>,
}

/// Compute per-pixel laser residual records for every laser observation in the
/// dataset, indexed pose-major, and each view's projected laser line.
///
/// For each `(view, laser_pixel)` in the dataset the record carries:
/// - `residual_m`: signed point-to-plane distance in meters between the
///   back-projected ray's target-plane point and the laser plane (`None`
///   when the ray does not meet the target plane).
/// - `residual_px`: the line-distance residual the laser solve minimizes
///   (`None` when the laser and target planes have no line in common).
///
/// Records carry `camera = 0` (single-camera problem). Iteration order: outer
/// = view, inner = pixel index in `view.meta.laser_pixels`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if `dataset.len() != params.poses.len()`
/// or the camera has a general homography sensor.
pub fn compute_laserline_feature_residuals(
    dataset: &LaserlineDataset,
    params: &LaserlineParams,
) -> Result<LaserResidualRecords, Error> {
    if dataset.len() != params.poses.len() {
        return Err(Error::invalid_input(format!(
            "dataset has {} views but {} poses",
            dataset.len(),
            params.poses.len()
        )));
    }

    let camera = LaserCameraModel::new(&params.camera)?;
    let mut out = LaserResidualRecords::default();
    for (view_idx, view) in dataset.iter().enumerate() {
        camera.view_residuals(
            &params.poses[view_idx],
            &params.plane,
            &view.meta.laser_pixels,
            (view_idx, 0),
            &mut out,
        );
    }
    Ok(out)
}

/// One camera of a laser problem, both as a projection model and as the
/// packed parameter blocks the laser residual kernels take.
pub(crate) struct LaserCameraModel {
    camera: CameraModel,
    spec: CameraModelSpec,
    blocks: Vec<DVector<f64>>,
}

/// Points sampled along a view's projected laser line.
const LASER_LINE_SAMPLES: usize = 33;

impl LaserCameraModel {
    /// The laser model of `camera`, in its own model.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a general homography sensor (no camera
    /// model describes it).
    pub(crate) fn new(camera: &CameraParams) -> Result<Self, Error> {
        let spec = camera_spec(camera)?;
        let model = camera
            .build()
            .map_err(|e| Error::invalid_input(format!("camera model: {e}")))?;
        Ok(Self {
            camera: model,
            spec,
            blocks: camera_block_values(camera),
        })
    }

    /// The camera's blocks, as the residual kernels take them.
    fn blocks(&self) -> CameraBlocks<'_, f64> {
        CameraBlocks::split(self.spec, &self.blocks).0
    }

    /// Append the records of one view's laser pixels and its projected laser
    /// line; `slot` is `(pose, camera)`.
    pub(crate) fn view_residuals(
        &self,
        cam_se3_target: &Iso3,
        plane: &LaserPlane,
        pixels: &[Pt2],
        slot: (usize, usize),
        out: &mut LaserResidualRecords,
    ) {
        let (pose, camera) = slot;
        let pose_vec = iso3_to_se3_dvec(cam_se3_target);
        let normal = plane.normal_to_dvec();
        let distance = plane.distance_to_dvec();
        let mut points = Vec::with_capacity(pixels.len());
        for (feature, px) in pixels.iter().enumerate() {
            let point = target_plane_point(&self.camera, cam_se3_target, px);
            let mut residual = vision_calibration_core::LaserFeatureResidual::default();
            residual.pose = pose;
            residual.camera = camera;
            residual.feature = feature;
            residual.observed_px = [px.x, px.y];
            residual.residual_m = point.map(|p| plane.point_distance(&p));
            residual.residual_px = laser_line_distance(
                &self.blocks(),
                pose_vec.as_view(),
                normal.as_view(),
                distance.as_view(),
                [px.x, px.y],
            );
            out.features.push(residual);
            points.extend(point);
        }
        if let Some(points_px) = self.laser_line_px(cam_se3_target, plane, &points) {
            let mut line = vision_calibration_core::LaserLineTrace::default();
            line.pose = pose;
            line.camera = camera;
            line.points_px = points_px;
            out.lines.push(line);
        }
    }

    /// The laser plane's line on the target plane, projected into the image
    /// and sampled over the extent of `points` (the laser pixels' points on
    /// the target plane, camera frame). `None` if the planes are parallel or
    /// fewer than two samples project.
    fn laser_line_px(
        &self,
        cam_se3_target: &Iso3,
        plane: &LaserPlane,
        points: &[nalgebra::Point3<f64>],
    ) -> Option<Vec<[f64; 2]>> {
        use nalgebra::Vector3;

        let target_normal = cam_se3_target.rotation * Vector3::z_axis().into_inner();
        let target_d = -target_normal.dot(&cam_se3_target.translation.vector);
        let laser_normal = plane.normal.into_inner();
        let direction = laser_normal.cross(&target_normal);
        if direction.norm() < 1.0e-12 {
            return None;
        }
        let direction = direction.normalize();
        let origin = line_origin(
            &laser_normal,
            plane.distance,
            &target_normal,
            target_d,
            &direction,
        )?;
        let (lo, hi) = points
            .iter()
            .map(|p| (p.coords - origin).dot(&direction))
            .fold(None, |range: Option<(f64, f64)>, s| {
                Some(range.map_or((s, s), |(lo, hi)| (lo.min(s), hi.max(s))))
            })?;
        let step = (hi - lo) / (LASER_LINE_SAMPLES - 1) as f64;
        let points_px: Vec<[f64; 2]> = (0..LASER_LINE_SAMPLES)
            .filter_map(|i| {
                let p = nalgebra::Point3::from(origin + direction * (lo + step * i as f64));
                self.camera.project_point(&p).map(|px| [px.x, px.y])
            })
            .collect();
        (points_px.len() >= 2).then_some(points_px)
    }
}

/// Back-project a laser pixel and intersect its ray with the target plane
/// (z = 0 in the target frame); the point in the camera frame. `None` if the
/// ray is parallel to the plane or meets it behind the camera.
fn target_plane_point(
    camera: &CameraModel,
    cam_se3_target: &Iso3,
    px: &Pt2,
) -> Option<nalgebra::Point3<f64>> {
    use nalgebra::Point3;

    let ray = camera.backproject_pixel(px);
    let ray_dir_camera = ray.direction.normalize();
    let target_se3_cam = cam_se3_target.inverse();
    let ray_origin_target = target_se3_cam.transform_point(&Point3::from(ray.origin));
    let ray_dir_target = target_se3_cam.rotation.transform_vector(&ray_dir_camera);
    if ray_dir_target.z.abs() < 1.0e-12 {
        return None;
    }
    let t = -ray_origin_target.z / ray_dir_target.z;
    if t < 0.0 {
        return None;
    }
    Some(cam_se3_target.transform_point(&(ray_origin_target + ray_dir_target * t)))
}

/// Find a particular point on the line of intersection of two planes
/// `n1 · p = -d1` and `n2 · p = -d2`, given the line direction `v`.
fn line_origin(
    n1: &nalgebra::Vector3<f64>,
    d1: f64,
    n2: &nalgebra::Vector3<f64>,
    d2: f64,
    v: &nalgebra::Vector3<f64>,
) -> Option<nalgebra::Vector3<f64>> {
    use nalgebra::Vector3;
    let abs_v = [v.x.abs(), v.y.abs(), v.z.abs()];
    let axis = abs_v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)?;
    match axis {
        2 => solve_2x2(n1.x, n1.y, -d1, n2.x, n2.y, -d2).map(|(x, y)| Vector3::new(x, y, 0.0)),
        1 => solve_2x2(n1.x, n1.z, -d1, n2.x, n2.z, -d2).map(|(x, z)| Vector3::new(x, 0.0, z)),
        _ => solve_2x2(n1.y, n1.z, -d1, n2.y, n2.z, -d2).map(|(y, z)| Vector3::new(0.0, y, z)),
    }
}

fn solve_2x2(a00: f64, a01: f64, b0: f64, a10: f64, a11: f64, b1: f64) -> Option<(f64, f64)> {
    let det = a00 * a11 - a01 * a10;
    if det.abs() < 1.0e-12 {
        return None;
    }
    Some(((b0 * a11 - a01 * b1) / det, (a00 * b1 - b0 * a10) / det))
}

/// Extract solution from backend result.
fn extract_solution(
    ir: &ProblemIR,
    solution: crate::backend::BackendSolution,
    spec: CameraModelSpec,
    num_poses: usize,
) -> Result<LaserlineEstimate, Error> {
    let camera = unpack_camera(&solution.params, spec, None)?;

    let mut poses = Vec::new();
    for i in 0..num_poses {
        let name = format!("pose/{i}");
        let pose_vec = solution
            .params
            .get(&name)
            .ok_or_else(|| Error::numerical(format!("missing {} in solution", name)))?;
        poses.push(se3_dvec_to_iso3(pose_vec.as_view())?);
    }

    let normal_name = "plane_n".to_owned();
    let distance_name = "plane_d".to_owned();
    let plane_normal = solution
        .params
        .get(&normal_name)
        .ok_or_else(|| Error::numerical(format!("missing {} in solution", normal_name)))?;
    let plane_distance = solution
        .params
        .get(&distance_name)
        .ok_or_else(|| Error::numerical(format!("missing {} in solution", distance_name)))?;

    let plane = LaserPlane::from_split_dvec(plane_normal.as_view(), plane_distance.as_view())?;

    let labels: Vec<_> = ir
        .params
        .iter()
        .map(|block| uncertainty::planar_block_labels(block, spec))
        .collect();
    Ok(LaserlineEstimate {
        params: LaserlineParams::new(camera, poses, plane)?,
        uncertainty: uncertainty::labelled(ir, &solution.uncertainty, &solution.noise, &labels),
        report: solution.solve_report,
    })
}

/// Optimize laserline calibration with joint bundle adjustment.
///
/// This function jointly optimizes the camera (its parameters not held by
/// `opts.fix`), camera-to-target poses, and laser plane parameters using both
/// calibration feature observations and laser line observations. The camera
/// model is the one `initial.camera` holds: any distortion kind, and an
/// identity or Scheimpflug sensor.
///
/// # Errors
///
/// Returns [`Error`] if the camera has a general homography sensor, a fix
/// list names a parameter the model does not have, the line-distance
/// residual is asked of a projection that bends lines, or IR construction or
/// the solver backend fails.
///
/// # Example
///
/// ```ignore
/// use vision_calibration_optim::*;
///
/// let initial = LaserlineParams::new(camera, poses, plane)?;
/// let opts = LaserlineSolveOptions::default();
/// let backend_opts = BackendSolveOptions::default();
///
/// let result = optimize_laserline(&dataset, &initial, &opts, &backend_opts)?;
/// println!(
///     "Laser plane: normal={:?}, distance={}",
///     result.params.plane.normal,
///     result.params.plane.distance
/// );
/// ```
pub fn optimize_laserline(
    dataset: &LaserlineDataset,
    initial: &LaserlineParams,
    opts: &LaserlineSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<LaserlineEstimate, Error> {
    let spec = camera_spec(&initial.camera)?;
    let (ir, initial_map) = build_laserline_ir(dataset, initial, opts)?;
    let solution = backend::solve(&ir, &initial_map, backend_opts)?;
    extract_solution(&ir, solution, spec, dataset.len())
}

/// Build IR for laserline bundle adjustment.
fn build_laserline_ir(
    dataset: &LaserlineDataset,
    initial: &LaserlineParams,
    opts: &LaserlineSolveOptions,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>), Error> {
    if dataset.is_empty() {
        return Err(Error::invalid_input("need at least one view"));
    }
    if dataset.len() != initial.poses.len() {
        return Err(Error::invalid_input(format!(
            "dataset has {} views but {} initial poses",
            dataset.len(),
            initial.poses.len()
        )));
    }
    for (idx, view) in dataset.iter().enumerate() {
        view.meta
            .validate()
            .map_err(|e| Error::invalid_input(format!("view {}: {}", idx, e)))?;
    }

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

    let spec = camera_spec(&initial.camera)?;
    let fixed = opts.fix.resolve(spec)?;
    let camera_blocks = add_camera_blocks(
        &mut ir,
        &mut initial_map,
        &[CameraBlockInput {
            params: &initial.camera,
            fixed: &fixed,
            bounds: None,
        }],
        false,
    )?
    .remove(0);

    // Add pose parameter blocks (one per view)
    let mut pose_ids = Vec::new();
    for (i, pose) in initial.poses.iter().enumerate() {
        let fixed = if opts.fix_poses.contains(&i) {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let name = format!("pose/{i}");
        let id = ir.add_param_block(&name, 7, ManifoldKind::SE3, fixed, None);
        pose_ids.push(id);
        initial_map.insert(name, iso3_to_se3_dvec(pose));
    }

    // Add laser plane parameter blocks (unit normal on S2 + scalar distance)
    let normal_fixed = if opts.fix_plane {
        FixedMask::all_fixed(3)
    } else {
        FixedMask::all_free()
    };
    let distance_fixed = if opts.fix_plane {
        FixedMask::all_fixed(1)
    } else {
        FixedMask::all_free()
    };

    let normal_name = "plane_n";
    let distance_name = "plane_d";
    let normal_id = ir.add_param_block(normal_name, 3, ManifoldKind::S2, normal_fixed, None);
    let distance_id = ir.add_param_block(
        distance_name,
        1,
        ManifoldKind::Euclidean,
        distance_fixed,
        None,
    );
    initial_map.insert(normal_name.to_owned(), initial.plane.normal_to_dvec());
    initial_map.insert(distance_name.to_owned(), initial.plane.distance_to_dvec());

    // Add residual blocks
    for (view_idx, view) in dataset.iter().enumerate() {
        let pose_id = pose_ids[view_idx];

        // Calibration reprojection residuals
        for (pt_idx, (pt_3d, pt_2d)) in view
            .obs
            .points_3d
            .iter()
            .zip(&view.obs.points_2d)
            .enumerate()
        {
            let sqrt_info =
                ReprojSqrtInfo::for_point(&view.obs, pt_idx).weighted(opts.calib_weight);
            ir.add_residual_block(ResidualBlock {
                params: camera_blocks.with(&[pose_id]),
                loss: opts.calib_loss,
                factor: FactorKind::ReprojPoint {
                    model: spec,
                    chain: ReprojChain::SinglePose,
                    pw: [pt_3d.x, pt_3d.y, pt_3d.z],
                    uv: [pt_2d.x, pt_2d.y],
                    sqrt_info,
                },
                residual_dim: 2,
            });
        }

        // Laser plane residuals (one per plane, typically just one)
        for (laser_idx, laser_pixel) in view.meta.laser_pixels.iter().enumerate() {
            let w = view.meta.laser_weight(laser_idx) * opts.laser_weight;

            // Select factor type based on options
            let factor = match opts.laser_residual_type {
                LaserlineResidualType::PointToPlane => FactorKind::LaserPointToPlane {
                    model: spec,
                    chain: LaserChain::SinglePose,
                    laser_pixel: [laser_pixel.x, laser_pixel.y],
                    w,
                },
                LaserlineResidualType::LineDistNormalized => FactorKind::LaserLineDistance {
                    model: spec,
                    chain: LaserChain::SinglePose,
                    laser_pixel: [laser_pixel.x, laser_pixel.y],
                    w,
                },
            };

            ir.add_residual_block(ResidualBlock {
                params: camera_blocks.with(&[pose_id, normal_id, distance_id]),
                loss: opts.laser_loss,
                factor,
                residual_dim: 1,
            });
        }
    }

    ir.validate()?;
    Ok((ir, initial_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, Camera, CorrespondenceView, DistortionParams, FxFyCxCy, Pinhole,
        ProjectionParams, Pt3, Real, ScheimpflugParams, SensorParams,
    };

    /// A pinhole camera with Brown-Conrady distortion and `sensor`.
    fn bc_camera(
        intrinsics: FxFyCxCy<Real>,
        distortion: BrownConrady5<Real>,
        sensor: SensorParams,
    ) -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 { params: distortion },
            sensor,
            intrinsics,
        }
    }

    fn tilt(sensor: ScheimpflugParams) -> SensorParams {
        SensorParams::Scheimpflug { params: sensor }
    }

    #[test]
    fn ir_validation_catches_missing_param() {
        let dataset = vec![View::new(
            CorrespondenceView::new(
                vec![Pt3::new(0.0, 0.0, 0.0); 4],
                vec![Pt2::new(100.0, 100.0); 4],
            )
            .unwrap(),
            LaserlineMeta {
                laser_pixels: vec![Pt2::new(200.0, 200.0); 5],
                laser_weights: Vec::new(),
            },
        )];

        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 512.0,
            cy: 384.0,
        };
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let poses = vec![Iso3::identity()];
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);
        let initial = LaserlineParams::new(
            bc_camera(intrinsics, distortion, SensorParams::Identity),
            poses,
            plane,
        )
        .unwrap();

        let opts = LaserlineSolveOptions::default();
        let (ir, mut initial_map) = build_laserline_ir(&dataset, &initial, &opts).unwrap();

        // Remove intrinsics from initial values
        initial_map.remove("cam");

        // Every backend must refuse to compile with a missing parameter.
        for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
            let backend_opts = BackendSolveOptions {
                backend,
                ..BackendSolveOptions::default()
            };
            let result = crate::backend::solve(&ir, &initial_map, &backend_opts);
            assert!(result.is_err(), "{backend:?}");
        }
    }

    /// The default fix holds the model's defaults: intrinsics free and `k3`
    /// for Brown-Conrady; an untilted camera has no sensor block, a tilted
    /// one a free tilt and its higher-order distortion terms held.
    #[test]
    fn default_fix_follows_the_camera_model() {
        let dataset = vec![View::new(
            CorrespondenceView::new(
                vec![Pt3::new(0.0, 0.0, 0.0); 4],
                vec![Pt2::new(100.0, 100.0); 4],
            )
            .unwrap(),
            LaserlineMeta {
                laser_pixels: vec![Pt2::new(200.0, 200.0); 5],
                laser_weights: Vec::new(),
            },
        )];
        let k = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 512.0,
            cy: 384.0,
        };
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);
        let opts = LaserlineSolveOptions::default();

        let untilted = LaserlineParams::new(
            bc_camera(k, BrownConrady5::default(), SensorParams::Identity),
            vec![Iso3::identity()],
            plane.clone(),
        )
        .unwrap();
        let (ir, _) = build_laserline_ir(&dataset, &untilted, &opts).unwrap();
        let fixed =
            |ir: &ProblemIR, name: &str| ir.params[ir.param_by_name(name).unwrap().0].fixed.clone();
        assert_eq!(fixed(&ir, "cam"), FixedMask::all_free());
        assert_eq!(fixed(&ir, "dist"), FixedMask::fix_indices(&[2]));
        assert!(ir.param_by_name("sensor").is_none());

        let tilted = LaserlineParams::new(
            bc_camera(
                k,
                BrownConrady5::default(),
                tilt(ScheimpflugParams {
                    tilt_x: 0.02,
                    tilt_y: 0.0,
                }),
            ),
            vec![Iso3::identity()],
            plane,
        )
        .unwrap();
        let (ir, _) = build_laserline_ir(&dataset, &tilted, &opts).unwrap();
        assert_eq!(fixed(&ir, "dist"), FixedMask::fix_indices(&[2, 3, 4]));
        assert_eq!(fixed(&ir, "sensor"), FixedMask::all_free());
    }

    #[test]
    fn compute_laserline_feature_residuals_zero_on_synthetic_perfect_data() {
        // Synthetic scenario where every laser pixel lies exactly on the
        // projected laser line, so both residual_m and residual_px should be
        // ~0 (within float roundoff).
        //
        // Setup:
        // - Camera at origin, fx=fy=800, cx=320, cy=240, no distortion, no
        //   Scheimpflug.
        // - Target frame is translated by +1m along z (cam_se3_target =
        //   translation(0, 0, 1)). Target plane (z=0 in target frame) lives
        //   at z=1 in camera frame.
        // - Laser plane in camera frame: y = 0.1 (normal=(0,1,0), d=-0.1).
        // - Plane intersection line in camera frame: (x, 0.1, 1) for any x.
        // - Laser pixels generated by projecting points along that line, so
        //   they sit exactly on the projected line in image space.
        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };
        let distortion = BrownConrady5::default();
        let sensor = ScheimpflugParams::default();
        let cam_se3_target = Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 1.0, 0.0), -0.1);
        let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);

        // Generate laser pixels by projecting line points to image space.
        let line_points_camera = [
            (-0.10, 0.1, 1.0),
            (-0.05, 0.1, 1.0),
            (0.0, 0.1, 1.0),
            (0.05, 0.1, 1.0),
            (0.10, 0.1, 1.0),
        ];
        let laser_pixels: Vec<Pt2> = line_points_camera
            .iter()
            .map(|&(x, y, z)| {
                camera
                    .project_point_c(&nalgebra::Vector3::new(x, y, z))
                    .expect("projection in front of camera")
            })
            .collect();

        // Need at least one target observation for the dataset; placeholder.
        let target_view = CorrespondenceView::new(
            vec![Pt3::new(0.0, 0.0, 0.0); 4],
            vec![Pt2::new(100.0, 100.0); 4],
        )
        .unwrap();
        let dataset = vec![View::new(
            target_view,
            LaserlineMeta {
                laser_pixels: laser_pixels.clone(),
                laser_weights: Vec::new(),
            },
        )];

        let params = LaserlineParams::new(
            bc_camera(intrinsics, distortion, SensorParams::Identity),
            vec![cam_se3_target],
            plane,
        )
        .unwrap();

        let records = compute_laserline_feature_residuals(&dataset, &params).unwrap();
        let residuals = &records.features;
        assert_eq!(residuals.len(), laser_pixels.len());
        assert_eq!(records.lines.len(), 1);
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.pose, 0);
            assert_eq!(r.camera, 0);
            assert_eq!(r.feature, i);
            let res_m = r.residual_m.expect("ray intersects target plane");
            assert!(
                res_m.abs() < 1.0e-6,
                "feature {i} residual_m {res_m} not within 1e-6"
            );
            let res_px = r.residual_px.expect("the planes meet in a line");
            assert!(
                res_px < 1.0e-6,
                "feature {i} residual_px {res_px} not within 1e-6"
            );
        }
    }

    /// Distance from `p` to the polyline `line`, pixels.
    fn polyline_distance(p: [f64; 2], line: &[[f64; 2]]) -> f64 {
        line.windows(2)
            .map(|seg| {
                let (a, b) = (seg[0], seg[1]);
                let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
                let t = (((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / (vx * vx + vy * vy))
                    .clamp(0.0, 1.0);
                ((p[0] - a[0] - t * vx).powi(2) + (p[1] - a[1] - t * vy).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn laser_residuals_and_line_follow_a_strongly_distorted_camera() {
        // A long-focal camera with strong radial distortion and a tilted
        // sensor, and a horizontal laser line far from the image centre, where
        // distortion bends it. The pixels are exact projections, so the solver
        // residual is zero and the projected line passes through every pixel.
        let intrinsics = FxFyCxCy {
            fx: 2100.0,
            fy: 2120.0,
            cx: 360.0,
            cy: 270.0,
        };
        let distortion = BrownConrady5 {
            k1: -0.4,
            k2: 6.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let sensor = ScheimpflugParams {
            tilt_x: -0.06,
            tilt_y: 0.005,
        };
        let cam_se3_target = Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 0.18),
            nalgebra::UnitQuaternion::from_euler_angles(0.4, 0.05, 0.0),
        );
        // A laser plane cutting the target along a line near the image's top.
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.94, -0.34), 0.05);
        let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
        let target_normal = cam_se3_target.rotation * nalgebra::Vector3::z();
        let target_d = -target_normal.dot(&cam_se3_target.translation.vector);
        let direction = plane.normal.cross(&target_normal).normalize();
        let origin = line_origin(
            &plane.normal,
            plane.distance,
            &target_normal,
            target_d,
            &direction,
        )
        .unwrap();
        let laser_pixels: Vec<Pt2> = (-20..=20)
            .filter_map(|i| {
                let p = nalgebra::Point3::from(origin + direction * (i as f64 * 1.5e-3));
                camera.project_point(&p)
            })
            .filter(|px| (0.0..720.0).contains(&px.x) && (0.0..540.0).contains(&px.y))
            .collect();
        assert!(
            laser_pixels.len() > 20,
            "{} pixels in view",
            laser_pixels.len()
        );
        let rows: Vec<f64> = laser_pixels.iter().map(|p| p.y).collect();
        let bend = rows.iter().cloned().fold(f64::MIN, f64::max)
            - rows.iter().cloned().fold(f64::MAX, f64::min);
        assert!(bend > 1.0, "the line bends {bend} px across the image");

        let target_view = CorrespondenceView::new(
            vec![Pt3::new(0.0, 0.0, 0.0); 4],
            vec![Pt2::new(100.0, 100.0); 4],
        )
        .unwrap();
        let dataset = vec![View::new(
            target_view,
            LaserlineMeta {
                laser_pixels: laser_pixels.clone(),
                laser_weights: Vec::new(),
            },
        )];
        let params = LaserlineParams::new(
            bc_camera(intrinsics, distortion, tilt(sensor)),
            vec![cam_se3_target],
            plane,
        )
        .unwrap();
        let records = compute_laserline_feature_residuals(&dataset, &params).unwrap();

        let line = &records.lines[0].points_px;
        assert_eq!(line.len(), LASER_LINE_SAMPLES);
        for r in &records.features {
            let px = r.residual_px.unwrap();
            assert!(px < 1e-6, "feature {}: residual {px} px", r.feature);
            let m = r.residual_m.unwrap();
            assert!(m.abs() < 1e-9, "feature {}: residual {m} m", r.feature);
            let off = polyline_distance(r.observed_px, line);
            assert!(off < 0.01, "feature {}: {off} px off the line", r.feature);
        }
    }

    #[test]
    fn compute_laserline_feature_residuals_rejects_pose_count_mismatch() {
        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };
        let distortion = BrownConrady5::default();
        let sensor = ScheimpflugParams::default();
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);

        let target_view = CorrespondenceView::new(
            vec![Pt3::new(0.0, 0.0, 0.0); 4],
            vec![Pt2::new(100.0, 100.0); 4],
        )
        .unwrap();
        // 2 dataset views, but only 1 pose in params → mismatch.
        let dataset = vec![
            View::new(
                target_view.clone(),
                LaserlineMeta {
                    laser_pixels: vec![Pt2::new(200.0, 200.0); 3],
                    laser_weights: Vec::new(),
                },
            ),
            View::new(
                target_view,
                LaserlineMeta {
                    laser_pixels: vec![Pt2::new(200.0, 200.0); 3],
                    laser_weights: Vec::new(),
                },
            ),
        ];
        let params = LaserlineParams::new(
            bc_camera(intrinsics, distortion, tilt(sensor)),
            vec![Iso3::identity()],
            plane,
        )
        .unwrap();

        let result = compute_laserline_feature_residuals(&dataset, &params);
        assert!(result.is_err());
    }

    #[test]
    fn ir_uses_s2_for_plane_normal() {
        let dataset = vec![View::new(
            CorrespondenceView::new(
                vec![Pt3::new(0.0, 0.0, 0.0); 4],
                vec![Pt2::new(100.0, 100.0); 4],
            )
            .unwrap(),
            LaserlineMeta {
                laser_pixels: vec![Pt2::new(200.0, 200.0); 5],
                laser_weights: Vec::new(),
            },
        )];

        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 512.0,
            cy: 384.0,
        };
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let poses = vec![Iso3::identity()];
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);
        let initial = LaserlineParams::new(
            bc_camera(intrinsics, distortion, SensorParams::Identity),
            poses,
            plane,
        )
        .unwrap();

        let opts = LaserlineSolveOptions::default();
        let (ir, _) = build_laserline_ir(&dataset, &initial, &opts).unwrap();

        let normal_id = ir.param_by_name("plane_n").unwrap();
        let distance_id = ir.param_by_name("plane_d").unwrap();

        let normal_param = &ir.params[normal_id.0];
        let distance_param = &ir.params[distance_id.0];

        assert_eq!(normal_param.dim, 3);
        assert_eq!(normal_param.manifold, ManifoldKind::S2);
        assert_eq!(distance_param.dim, 1);
        assert_eq!(distance_param.manifold, ManifoldKind::Euclidean);
    }
}
