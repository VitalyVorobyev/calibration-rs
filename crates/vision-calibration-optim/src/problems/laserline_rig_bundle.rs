//! Rig-level laserline calibration.
//!
//! Thin orchestrator around [`super::laserline_bundle::optimize_laserline`]
//! that runs per-camera single-device calibration with intrinsics, distortion,
//! sensor, and per-view target poses all frozen (they are expected to come
//! from an upstream rig + hand-eye calibration). The resulting per-camera
//! planes (in camera frame) are transformed into the rig frame via the
//! provided `cam_to_rig` (`T_R_C`) transforms.
//!
//! Joint rig+plane bundle adjustment with rig-frame plane parameterization is
//! a v1.1 follow-up; see the workspace plan.
//!
//! # Input shape
//!
//! The dataset is organized as a 2-D grid of `(view_idx, cam_idx)`:
//!
//! ```text
//! views[v].cameras[c]       -- Option<CorrespondenceView>  (3D-2D target pairs)
//! views[v].laser_pixels[c]  -- Option<Vec<Pt2>>            (laser-line pixel coords)
//! ```
//!
//! `None` in either slot means the camera produced no usable data for that
//! view (occluded, out-of-FOV, etc.).  An empty `Some(vec![])` is never
//! returned by the solvers but is accepted as input and treated the same as
//! `None` for residual purposes.  The outer `Vec` length must equal
//! `num_cameras` for every view; the constructor enforces this at creation
//! time.

use crate::Error;
use crate::backend::BackendSolveOptions;
use crate::params::laser_plane::LaserPlane;
use crate::problems::laserline_bundle::{
    LaserlineMeta, LaserlineParams, LaserlineResidualType, LaserlineSolveOptions, LaserlineStats,
    LaserlineView, compute_laserline_stats, laser_line_endpoints_px,
    laser_point_to_plane_residual_m, optimize_laserline, point_line_distance,
};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Iso3, LaserFeatureResidual, Pinhole,
    Pt2, Real, ScheimpflugParams, View,
};

/// Per-view observations for a rig-level laserline calibration.
///
/// Both `cameras` and `laser_pixels` are indexed by `cam_idx ∈ [0, num_cameras)`.
/// A slot is `None` when that camera had no usable data in this view (occlusion,
/// out-of-FOV, etc.).  An empty inner `Vec` is accepted but treated the same as
/// `None` by the solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigLaserlineView {
    /// Target correspondences per camera (`cam_idx` → `Option<CorrespondenceView>`).
    ///
    /// `None` means the camera did not see the calibration target in this view.
    /// The slice length must equal `RigLaserlineDataset::num_cameras`.
    pub cameras: Vec<Option<CorrespondenceView>>,
    /// Laser pixel observations per camera (`cam_idx` → `Option<Vec<Pt2>>`).
    ///
    /// `None` means the camera has no usable laser data for this view.
    /// The slice length must equal `RigLaserlineDataset::num_cameras`.
    pub laser_pixels: Vec<Option<Vec<Pt2>>>,
}

/// Dataset for rig-level laserline calibration.
///
/// Each view contains one row of the `(view, camera)` observation grid.
/// Use [`RigLaserlineDataset::new`] to construct — it validates that every
/// view has exactly `num_cameras` slots in both `cameras` and `laser_pixels`.
///
/// # Missing-observation convention
///
/// * `Some(obs)` — camera contributed data; `obs` may still be empty (no
///   correspondences / pixels), in which case it contributes zero residuals.
/// * `None` — camera is absent for this view and is skipped entirely.
///
/// # Example
///
/// ```no_run
/// # use vision_calibration_optim::{RigLaserlineDataset, RigLaserlineView};
/// // Two views, two cameras; camera 1 missed the target in view 0.
/// let views = vec![
///     RigLaserlineView {
///         cameras:      vec![Some(/* cam0 obs */ todo!()), None],
///         laser_pixels: vec![Some(vec![]),                None],
///     },
///     RigLaserlineView {
///         cameras:      vec![Some(/* cam0 obs */ todo!()), Some(/* cam1 obs */ todo!())],
///         laser_pixels: vec![Some(vec![]),                 Some(vec![])],
///     },
/// ];
/// let dataset = RigLaserlineDataset::new(views, 2).unwrap();
/// assert_eq!(dataset.num_cameras, 2);
/// assert_eq!(dataset.num_views(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigLaserlineDataset {
    /// Per-view observation rows; length = number of views.
    ///
    /// Each `RigLaserlineView` contains two `Vec`s of length `num_cameras`,
    /// one for target correspondences and one for laser pixels.
    pub views: Vec<RigLaserlineView>,
    /// Number of cameras in the rig; governs the expected slot count per view.
    pub num_cameras: usize,
}

impl RigLaserlineDataset {
    /// Construct with consistency checks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if any view's per-camera slice length
    /// does not match `num_cameras`, or if `views` is empty.
    pub fn new(views: Vec<RigLaserlineView>, num_cameras: usize) -> Result<Self, Error> {
        if views.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        for (i, v) in views.iter().enumerate() {
            if v.cameras.len() != num_cameras || v.laser_pixels.len() != num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {i} has {}/{} camera slots, expected {num_cameras}",
                    v.cameras.len(),
                    v.laser_pixels.len()
                )));
            }
        }
        Ok(Self { views, num_cameras })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}

/// Upstream calibration consumed by the rig laserline solver.
///
/// All of these parameters are held fixed during optimization (by design).
/// They typically come from a Scheimpflug hand-eye calibration.
#[derive(Debug, Clone)]
pub struct RigLaserlineUpstream {
    /// Per-camera intrinsics.
    pub intrinsics: Vec<FxFyCxCySkew<Real>>,
    /// Per-camera distortion.
    pub distortion: Vec<BrownConrady5<Real>>,
    /// Per-camera Scheimpflug sensor parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera extrinsics (camera-to-rig, `T_R_C`).
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view rig poses (`rig_se3_target`, `T_R_T`).
    pub rig_se3_target: Vec<Iso3>,
}

/// Solve options for rig laserline calibration.
#[derive(Debug, Clone)]
pub struct RigLaserlineSolveOptions {
    /// Per-camera laserline solver knobs (shared by all cameras).
    ///
    /// The fix-flags for intrinsics / distortion / sensor / poses are
    /// overridden by this orchestrator so that upstream calibration stays
    /// frozen.
    pub laserline: LaserlineSolveOptions,
    /// Laser residual type (propagated to the per-camera solver).
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for RigLaserlineSolveOptions {
    fn default() -> Self {
        let laserline = LaserlineSolveOptions {
            fix_intrinsics: true,
            fix_distortion: true,
            fix_sensor: true,
            fix_poses: Vec::new(),
            ..LaserlineSolveOptions::default()
        };
        Self {
            laserline,
            laser_residual_type: LaserlineResidualType::default(),
        }
    }
}

/// Compute per-pixel laser residual records for every laser observation in a
/// rig dataset, indexed pose-major then by camera then by pixel.
///
/// Like [`super::laserline_bundle::compute_laserline_feature_residuals`] but
/// fans out across all cameras of a rig: each (view, cam_idx, pixel_idx)
/// triple becomes one [`LaserFeatureResidual`] tagged with its triple. None
/// slots in `dataset.views[v].laser_pixels[c]` produce no records.
///
/// `cam_se3_target_per_view_cam[v][c]` is the camera-frame target pose for
/// view `v`, camera `c`, derived by composing `cam_se3_rig[c] *
/// rig_se3_target[v]`. Cameras are reconstructed from `intrinsics`,
/// `distortion`, and `sensors`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if the per-camera or per-view slice
/// lengths are inconsistent with the dataset shape.
pub fn compute_rig_laserline_feature_residuals(
    dataset: &RigLaserlineDataset,
    intrinsics: &[FxFyCxCySkew<Real>],
    distortion: &[BrownConrady5<Real>],
    sensors: &[ScheimpflugParams],
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
    laser_planes_cam: &[LaserPlane],
) -> Result<Vec<LaserFeatureResidual>, Error> {
    let n = dataset.num_cameras;
    if intrinsics.len() != n
        || distortion.len() != n
        || sensors.len() != n
        || cam_se3_rig.len() != n
        || laser_planes_cam.len() != n
    {
        return Err(Error::invalid_input(format!(
            "per-camera slice length mismatch (expected {n})"
        )));
    }
    if rig_se3_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "rig_se3_target has {} entries, expected {}",
            rig_se3_target.len(),
            dataset.num_views()
        )));
    }

    let cameras: Vec<_> = (0..n)
        .map(|c| Camera::new(Pinhole, distortion[c], sensors[c].compile(), intrinsics[c]))
        .collect();

    let mut out = Vec::new();
    for (view_idx, view) in dataset.views.iter().enumerate() {
        for cam_idx in 0..n {
            let Some(pixels) = view.laser_pixels.get(cam_idx).and_then(|p| p.as_ref()) else {
                continue;
            };
            let cam_se3_target = cam_se3_rig[cam_idx] * rig_se3_target[view_idx];
            let camera = &cameras[cam_idx];
            let plane = &laser_planes_cam[cam_idx];
            let line_endpoints = laser_line_endpoints_px(camera, &cam_se3_target, plane);
            for (feature_idx, px) in pixels.iter().enumerate() {
                let residual_m =
                    laser_point_to_plane_residual_m(camera, &cam_se3_target, plane, px);
                let residual_px =
                    line_endpoints.map(|line| point_line_distance([px.x, px.y], line));
                out.push(LaserFeatureResidual {
                    pose: view_idx,
                    camera: cam_idx,
                    feature: feature_idx,
                    observed_px: [px.x, px.y],
                    residual_m,
                    residual_px,
                    projected_line_px: line_endpoints,
                });
            }
        }
    }
    Ok(out)
}

/// Result of rig-level laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigLaserlineEstimate {
    /// Per-camera laser plane expressed in rig frame.
    pub laser_planes_rig: Vec<LaserPlane>,
    /// Per-camera laser plane expressed in the camera's own frame.
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Per-camera stats (mean reprojection + laser residual).
    pub per_camera_stats: Vec<LaserlineStats>,
}

/// Run per-camera laserline calibration and assemble rig-frame planes.
///
/// Intrinsics / distortion / sensor / per-view poses are always frozen:
/// the wrapper mutates `opts.laserline` to force this invariant so callers
/// cannot accidentally refine upstream parameters here. Use
/// [`super::laserline_bundle::optimize_laserline`] directly for single-
/// camera cases that need joint refinement.
///
/// `initial_planes_cam` supplies the starting normal/distance for each camera's
/// plane in that camera's own frame. Typical sources: a short per-camera
/// linear plane-fit, or a constant (e.g. `LaserPlane::new([0,0,1], -0.1)`).
///
/// # Errors
///
/// Returns [`Error`] if any per-camera optimization fails, if camera
/// observation or laser-pixel slices have inconsistent shapes with
/// `upstream`, or if intermediate stats computation fails.
pub fn optimize_rig_laserline(
    dataset: &RigLaserlineDataset,
    upstream: &RigLaserlineUpstream,
    initial_planes_cam: &[LaserPlane],
    opts: &RigLaserlineSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<RigLaserlineEstimate, Error> {
    let n = dataset.num_cameras;
    if upstream.intrinsics.len() != n
        || upstream.distortion.len() != n
        || upstream.sensors.len() != n
        || upstream.cam_to_rig.len() != n
        || initial_planes_cam.len() != n
    {
        return Err(Error::invalid_input(format!(
            "upstream or initial_planes_cam length mismatch: cameras={n}, \
            intrinsics={}, distortion={}, sensors={}, cam_to_rig={}, planes={}",
            upstream.intrinsics.len(),
            upstream.distortion.len(),
            upstream.sensors.len(),
            upstream.cam_to_rig.len(),
            initial_planes_cam.len(),
        )));
    }
    if upstream.rig_se3_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "rig_se3_target has {} entries, expected {} (one per view)",
            upstream.rig_se3_target.len(),
            dataset.num_views()
        )));
    }

    let mut laser_planes_cam = Vec::with_capacity(n);
    let mut laser_planes_rig = Vec::with_capacity(n);
    let mut per_camera_stats = Vec::with_capacity(n);

    // Force upstream-freezing flags; propagate residual type.
    let mut laser_opts = opts.laserline.clone();
    laser_opts.fix_intrinsics = true;
    laser_opts.fix_distortion = true;
    laser_opts.fix_sensor = true;
    laser_opts.laser_residual_type = opts.laser_residual_type;

    for (cam_idx, initial_plane_cam) in initial_planes_cam.iter().enumerate() {
        let cam_to_rig = upstream.cam_to_rig[cam_idx];
        let rig_to_cam = cam_to_rig.inverse();

        // Build per-camera LaserlineView sequence + camera_se3_target poses.
        // Skip views that lack either target correspondences or laser pixels
        // for this camera.
        let mut views: Vec<LaserlineView> = Vec::new();
        let mut poses_cam: Vec<Iso3> = Vec::new();
        for (view_idx, view) in dataset.views.iter().enumerate() {
            let Some(corr) = view.cameras.get(cam_idx).and_then(|o| o.clone()) else {
                continue;
            };
            let Some(laser) = view
                .laser_pixels
                .get(cam_idx)
                .and_then(|o| o.clone())
                .filter(|v| !v.is_empty())
            else {
                continue;
            };
            let cam_se3_target = rig_to_cam * upstream.rig_se3_target[view_idx];
            views.push(View::new(
                corr,
                LaserlineMeta {
                    laser_pixels: laser,
                    laser_weights: Vec::new(),
                },
            ));
            poses_cam.push(cam_se3_target);
        }

        if views.is_empty() {
            return Err(Error::invalid_input(format!(
                "camera {cam_idx} has no views with both target correspondences and laser pixels"
            )));
        }

        // Fix all poses for this camera (upstream already calibrated them).
        let mut per_cam_opts = laser_opts.clone();
        per_cam_opts.fix_poses = (0..views.len()).collect();

        let initial = LaserlineParams::new(
            upstream.intrinsics[cam_idx],
            upstream.distortion[cam_idx],
            upstream.sensors[cam_idx],
            poses_cam,
            initial_plane_cam.clone(),
        )?;

        let est = optimize_laserline(&views, &initial, &per_cam_opts, backend_opts)?;
        let stats = compute_laserline_stats(&views, &est.params, opts.laser_residual_type)?;

        let plane_cam = est.params.plane.clone();
        let plane_rig = plane_cam.transform_by(&cam_to_rig);
        laser_planes_cam.push(plane_cam);
        laser_planes_rig.push(plane_rig);
        per_camera_stats.push(stats);
    }

    Ok(RigLaserlineEstimate {
        laser_planes_rig,
        laser_planes_cam,
        per_camera_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3, Unit, Vector3};
    use vision_calibration_core::{BrownConrady5, Camera, FxFyCxCySkew, Pinhole, Pt3};

    fn make_target_points() -> Vec<Pt3> {
        let mut pts = Vec::new();
        for y in -2..=2 {
            for x in -3..=3 {
                pts.push(Pt3::new(x as Real * 0.02, y as Real * 0.02, 0.0));
            }
        }
        pts
    }

    #[test]
    fn rig_laserline_two_cameras_zero_tilt_recovers_planes() {
        let intr = FxFyCxCySkew {
            fx: 600.0,
            fy: 600.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        let dist = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 5,
        };
        let sensor = ScheimpflugParams::default();

        // Two cameras: cam0 at origin, cam1 displaced.
        let cam_to_rig = [
            Isometry3::identity(),
            Isometry3::from_parts(
                Translation3::new(0.15, 0.0, 0.0),
                Rotation3::identity().into(),
            ),
        ];

        // Rig poses put target at varying depths in front of the cameras.
        let rig_se3_target = vec![
            Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.30),
                Rotation3::identity().into(),
            ),
            Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.32),
                Rotation3::identity().into(),
            ),
            Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.34),
                Rotation3::identity().into(),
            ),
            Isometry3::from_parts(
                Translation3::new(0.0, 0.0, 0.36),
                Rotation3::identity().into(),
            ),
        ];

        // Ground-truth laser planes in rig frame: intersect target plane (z=d in cam frame)
        // but are distinct per camera. Planes pass through y=0 with slight tilt.
        let planes_rig_gt = [
            LaserPlane::new(Vector3::new(0.0, 1.0, 0.1), -0.0),
            LaserPlane::new(Vector3::new(0.0, 1.0, -0.1), -0.0),
        ];
        // Convert to each camera's frame for synthetic pixel generation.
        let planes_cam_gt: Vec<LaserPlane> = planes_rig_gt
            .iter()
            .zip(cam_to_rig.iter())
            .map(|(p_rig, t)| p_rig.transform_by(&t.inverse()))
            .collect();

        // Generate synthetic views: target corners + laser pixels per camera per view.
        let mut views: Vec<RigLaserlineView> = Vec::new();
        let cam_gt: Vec<Camera<Real, Pinhole, BrownConrady5<Real>, _, FxFyCxCySkew<Real>>> = (0..2)
            .map(|_| Camera::new(Pinhole, dist, sensor.compile(), intr))
            .collect();
        let target_pts = make_target_points();

        for rig_pose in &rig_se3_target {
            let mut cam_obs: Vec<Option<CorrespondenceView>> = Vec::new();
            let mut laser_obs: Vec<Option<Vec<Pt2>>> = Vec::new();
            for cam_idx in 0..2 {
                let cam_se3_target = cam_to_rig[cam_idx].inverse() * rig_pose;

                // Target correspondences
                let mut pts3 = Vec::new();
                let mut pts2 = Vec::new();
                for p in &target_pts {
                    let p_cam = cam_se3_target.transform_point(p);
                    if let Some(uv) = cam_gt[cam_idx].project_point(&p_cam) {
                        pts3.push(*p);
                        pts2.push(uv);
                    }
                }
                cam_obs.push(Some(CorrespondenceView::new(pts3, pts2).unwrap()));

                // Synthetic laser pixels: intersect the GT plane with the target plane (z=0 in
                // target frame) along x ∈ [-0.04, 0.04], solve y, project.
                let plane = &planes_cam_gt[cam_idx];
                let n = plane.normal.into_inner();
                let d = plane.distance;
                let rot = cam_se3_target.rotation.to_rotation_matrix();
                let trans = cam_se3_target.translation.vector;
                // Point in camera frame = R * (x, y, 0) + t. The laser plane
                // constraint yields a linear equation in (x, y) for each x.
                // n . (R*[x, y, 0]^T + t) + d = 0
                // => (n . R_col1) * x + (n . R_col2) * y + (n . t + d) = 0
                let r_col0: Vector3<f64> = rot.matrix().column(0).into_owned();
                let r_col1: Vector3<f64> = rot.matrix().column(1).into_owned();
                let c1 = n.dot(&r_col0);
                let c2 = n.dot(&r_col1);
                let c0 = n.dot(&trans) + d;
                if c2.abs() < 1e-9 {
                    laser_obs.push(Some(Vec::new()));
                    continue;
                }
                let mut pixels = Vec::new();
                for k in -10..=10 {
                    let x = k as f64 * 0.004;
                    let y = -(c1 * x + c0) / c2;
                    let pt_target = Pt3::new(x, y, 0.0);
                    let pt_cam = cam_se3_target.transform_point(&pt_target);
                    if let Some(uv) = cam_gt[cam_idx].project_point(&pt_cam) {
                        pixels.push(uv);
                    }
                }
                laser_obs.push(Some(pixels));
            }
            views.push(RigLaserlineView {
                cameras: cam_obs,
                laser_pixels: laser_obs,
            });
        }

        let dataset = RigLaserlineDataset::new(views, 2).unwrap();
        let upstream = RigLaserlineUpstream {
            intrinsics: vec![intr, intr],
            distortion: vec![dist, dist],
            sensors: vec![sensor, sensor],
            cam_to_rig: cam_to_rig.to_vec(),
            rig_se3_target,
        };

        // Start planes slightly perturbed (in camera frame) from GT.
        let initial_planes_cam = planes_cam_gt
            .iter()
            .map(|p| {
                let n = p.normal.into_inner();
                let perturbed = Unit::new_normalize(Vector3::new(n.x + 0.05, n.y, n.z));
                LaserPlane {
                    normal: perturbed,
                    distance: p.distance + 0.01,
                }
            })
            .collect::<Vec<_>>();

        let opts = RigLaserlineSolveOptions::default();
        let backend_opts = BackendSolveOptions {
            max_iters: 100,
            verbosity: 0,
            min_abs_decrease: Some(1e-12),
            min_rel_decrease: Some(1e-12),
            min_error: Some(1e-14),
            ..Default::default()
        };

        let est = optimize_rig_laserline(
            &dataset,
            &upstream,
            &initial_planes_cam,
            &opts,
            &backend_opts,
        )
        .unwrap();

        // Compare recovered rig-frame planes with ground truth.
        for (cam_idx, (gt, got)) in planes_rig_gt.iter().zip(&est.laser_planes_rig).enumerate() {
            let n_dot = gt.normal.dot(&got.normal);
            let ang = n_dot.abs().min(1.0).acos();
            let d_err = (gt.distance - got.distance).abs();
            println!("camera {cam_idx}: plane ang={ang:.3e} d_err={d_err:.3e}");
            assert!(ang < 5e-3, "cam{cam_idx} normal error too large: {ang}");
            assert!(
                d_err < 5e-3,
                "cam{cam_idx} distance error too large: {d_err}"
            );
        }
    }
}
