//! Bundle adjustment from a calibration, with the covariance of what it refines.
//!
//! Jointly refines the capture poses of a rig and the 3D points on the
//! reprojection error of the stated pixel noise, through the public problem
//! builder of `vision-calibration-optim`. The cameras come from a
//! [`CalibratedRig`] and enter in one of two ways ([`Use`]): as **consider**
//! parameters (they stay at their values and their covariance is carried into
//! the result's) or under a **prior** (they are refined too, held by their own
//! covariance). Every reprojection can carry a robust loss.
//!
//! # The gauge
//!
//! Reprojection error does not fix the world frame or the scale of a
//! reconstruction, so both are stated:
//!
//! - **Frame.** Capture 0 is fixed: the world frame is capture 0's rig frame
//!   as given. Every other capture pose and every point is refined relative
//!   to it.
//! - **Scale.** [`Scale`] has no default. [`Scale::Rig`] takes it from the
//!   rig's extrinsics (the baseline between two cameras that see the same
//!   capture); [`Scale::Distance`] from a stated distance between two points.
//!   A call without a workable scale is an [`MvgError::InvalidInput`] before
//!   any work, never a silent choice.
//!
//! The covariance is in that gauge: capture 0 has none, and the scale's
//! uncertainty (the rig's under [`Scale::Rig`], the stated σ under
//! [`Scale::Distance`]) is part of it.

use crate::triangulation::{RigObservation, sub_covariance, track_views, with_context};
use crate::{MvgError, Result};
use nalgebra::{DVector, Matrix6, RealField};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CalibratedRig, Iso3, Mat3, ParameterCovariance, Pt3, Real, Uncertainty,
};
use vision_calibration_optim::problem::{
    BlockId, Manifold, Noise, PixelNoise, Problem, Residual, SolveOptions, TargetPoint, Use, solve,
};
use vision_calibration_optim::{RobustLoss, SolveReport};

/// How the reconstruction's scale is fixed. There is no default: reprojection
/// error alone cannot tell a large scene far away from a small one near.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Scale {
    /// The rig's extrinsics fix it: the baseline between two cameras with
    /// distinct centres that observe points in the same capture. Needs a rig
    /// of two or more cameras.
    Rig,
    /// A stated distance between two of the points.
    Distance {
        /// Indices of the two points in `points` (distinct).
        points: [usize; 2],
        /// Their distance, metres (positive).
        distance: Real,
        /// The σ of that distance, metres (positive): a distance weighs in
        /// the solution as much as its precision says.
        sigma: Real,
    },
}

/// Options of [`bundle_adjust`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct BundleAdjustOptions {
    /// How the calibration enters: [`Use::Consider`] (the cameras stay
    /// fixed, their covariance is carried into the result's) or
    /// [`Use::Prior`] (the cameras are refined, held by their covariance;
    /// needs a rig with one).
    pub calibration: Use,
    /// Robust loss on every reprojection, its scale in σ units (the
    /// residuals are whitened by their stated noise). The covariance takes
    /// the residuals at their stated variance.
    pub loss: RobustLoss,
    /// The scale gauge.
    pub scale: Scale,
    /// Most Levenberg-Marquardt iterations; a solve that stops on this
    /// budget has no covariance.
    pub max_iters: usize,
}

impl BundleAdjustOptions {
    /// Consider the calibration, no robust loss, 100 iterations, and the
    /// given scale.
    pub fn new(scale: Scale) -> Self {
        Self {
            calibration: Use::Consider,
            loss: RobustLoss::None,
            scale,
            max_iters: 100,
        }
    }
}

/// The result of [`bundle_adjust`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BundleAdjustment {
    /// The refined capture poses (world → rig), in the order given.
    /// `rig_se3_world[0]` is the given one (the gauge); a capture without
    /// observations is returned as given.
    pub rig_se3_world: Vec<Iso3>,
    /// The refined points, in the world frame, in the order given.
    pub points: Vec<Pt3>,
    /// The rig: the given one under [`Use::Consider`], the refined one under
    /// [`Use::Prior`], with the covariance of its cameras and extrinsics after
    /// the adjustment (it can be fed to [`triangulate`](crate::triangulation::triangulate)
    /// or to another adjustment).
    pub rig: CalibratedRig,
    /// The joint covariance of the free parameters: capture poses
    /// (`capture{k}.rx` … `tz`, the right-perturbation tangent), points
    /// (`p{i}.x`, `.y`, `.z`) and, under [`Use::Prior`], the cameras. In the
    /// gauge above, the calibration's share included; or why it is
    /// unavailable.
    pub uncertainty: Uncertainty,
    /// The solve's final cost and iteration count.
    pub report: SolveReport,
    /// RMS reprojection error over every observation, pixels, at the
    /// solution: the root of the mean squared distance between a pixel and
    /// its point's projection (`√2 σ` for isotropic noise of `σ` per axis).
    pub reprojection_rms_px: Real,
}

impl BundleAdjustment {
    /// Point `i`'s 3×3 covariance, m²; `None` when the uncertainty is
    /// unavailable or `i` is out of range.
    pub fn point_covariance(&self, i: usize) -> Option<Mat3> {
        let axes = |p: usize| ["x", "y", "z"].map(|a| format!("p{p}.{a}"));
        sub_covariance(&self.uncertainty, axes(i), axes(i))
    }

    /// Capture `k`'s 6×6 pose covariance over its right-perturbation tangent
    /// `[rx ry rz tx ty tz]` (`T·Exp(δ)`), in rad and m. `None` for capture 0
    /// (fixed), for a capture without observations, when `k` is out of range
    /// or when the uncertainty is unavailable.
    pub fn pose_covariance(&self, k: usize) -> Option<Matrix6<Real>> {
        let axes =
            |k: usize| ["rx", "ry", "rz", "tx", "ty", "tz"].map(|a| format!("capture{k}.{a}"));
        sub_covariance(&self.uncertainty, axes(k), axes(k))
    }
}

/// `|a − b| − d` over two point blocks: the stated distance.
struct PointDistance {
    distance: Real,
}

impl Residual for PointDistance {
    fn dim(&self) -> usize {
        1
    }

    fn block_dims(&self) -> Vec<usize> {
        vec![3, 3]
    }

    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let d = (0..3)
            .map(|i| blocks[0][i].clone() - blocks[1][i].clone())
            .fold(T::zero(), |s, v| s + v.clone() * v)
            .sqrt();
        DVector::from_element(1, d - T::from_f64(self.distance).unwrap())
    }
}

/// Refine the capture poses and the points of a reconstruction on the
/// reprojection error, each pixel weighted by its stated noise, and give the
/// joint covariance.
///
/// `tracks[i]` holds the observations of `points[i]` (the initial point, in
/// the world frame); `rig_se3_world[k]` is the initial pose of the rig at
/// capture `k` (world → rig frame), as for
/// [`triangulate`](crate::triangulation::triangulate). Capture 0 is fixed and
/// defines the world frame; `opts.scale` fixes the scale (see the
/// [module docs](self)). The initial values come from pose recovery and
/// triangulation, or from any other reconstruction.
///
/// The covariance has the observations' stated noise and the calibration's
/// uncertainty as sources. Under [`Use::Consider`] the cameras and extrinsics
/// stay at their values and the rig's covariance is propagated into the
/// poses' and points'; under [`Use::Prior`] they are refined, held by that
/// covariance, so the observations sharpen them (`BundleAdjustment::rig`).
/// The covariance is dense over every free parameter, so the cost grows as
/// the cube of the parameter count: for thousands of points, adjust a
/// window of captures. A capture or point that no connected observation
/// constrains makes the covariance unavailable, with the unobservable
/// parameters named.
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, PointCovariance, Pt3, Uncertainty, Vec3,
///     make_pinhole_camera, pinhole_camera_params,
/// };
/// use vision_mvg::bundle_adjust::{BundleAdjustOptions, Scale, bundle_adjust};
/// use vision_mvg::triangulation::RigObservation;
///
/// // A stereo rig: camera 1 sits 10 cm to the right of camera 0.
/// let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 320.0, cy: 240.0 };
/// let camera = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
/// let rig = CalibratedRig::new(
///     vec![camera.clone(), camera],
///     vec![Iso3::identity(), Iso3::translation(-0.1, 0.0, 0.0)],
///     Uncertainty::default(),
/// )?;
///
/// // Two captures (the rig moves 0.3 m sideways) of six points, pixels stated to 0.3 px.
/// let truth_poses = [Iso3::identity(), Iso3::translation(-0.3, 0.0, 0.0)];
/// let truth: Vec<Pt3> = (0..6)
///     .map(|i| Pt3::new(0.1 * f64::from(i) - 0.25, 0.05 * f64::from(i % 3), 2.0 + 0.2 * f64::from(i)))
///     .collect();
/// let noise = PointCovariance::isotropic(0.3)?;
/// let tracks: Vec<Vec<RigObservation>> = truth
///     .iter()
///     .map(|p| {
///         let mut track = Vec::new();
///         for (capture, pose) in truth_poses.iter().enumerate() {
///             for camera in 0..2 {
///                 let in_camera = rig.cam_se3_rig()[camera] * pose * p;
///                 let pixel = rig.model(camera).unwrap().project_point(&in_camera).unwrap();
///                 track.push(RigObservation::new(capture, camera, pixel, noise));
///             }
///         }
///         track
///     })
///     .collect();
///
/// // Start 2 cm off and adjust; the rig's baseline fixes the scale.
/// let start: Vec<Pt3> = truth.iter().map(|p| p + Vec3::new(0.02, -0.02, 0.02)).collect();
/// let initial = [Iso3::identity(), Iso3::translation(-0.31, 0.01, 0.0)];
/// let opts = BundleAdjustOptions::new(Scale::Rig);
/// let adjusted = bundle_adjust(&rig, &initial, &start, &tracks, &opts)?;
/// assert!((adjusted.points[0] - truth[0]).norm() < 1e-6);
/// assert!(adjusted.reprojection_rms_px < 1e-3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// [`MvgError::InvalidInput`], naming the item and before any solve, for
/// different numbers of points and tracks; a track with fewer than 2
/// observations; a camera or capture index out of range; a non-finite
/// initial pose or point; a pixel without a finite viewing ray; capture 0
/// without observations; fewer than two observed captures (or one seen by a
/// single camera), where there is nothing to adjust; a [`Scale`] the data
/// cannot fix (see its variants) or with invalid numbers;
/// [`Use::Prior`] with a rig without a covariance; and a problem the
/// builder rejects, such as an initial point that does not project.
/// [`MvgError::Numerical`] when the solve fails to run. A solve that stops
/// short of convergence is not an error: the result stands and its
/// uncertainty says [`UncertaintyUnavailable::NotConverged`](vision_calibration_core::UncertaintyUnavailable::NotConverged).
pub fn bundle_adjust(
    rig: &CalibratedRig,
    rig_se3_world: &[Iso3],
    points: &[Pt3],
    tracks: &[Vec<RigObservation>],
    opts: &BundleAdjustOptions,
) -> Result<BundleAdjustment> {
    let observed = validate(rig, rig_se3_world, points, tracks, opts)?;

    let mut problem = Problem::new();
    let blocks = problem
        .add_calibrated_rig(rig, opts.calibration)
        .map_err(builder_error)?;

    // One pose block per observed capture; capture 0 is the gauge.
    let mut captures: Vec<Option<BlockId>> = vec![None; rig_se3_world.len()];
    for (k, pose) in rig_se3_world.iter().enumerate() {
        if !observed[k] {
            continue;
        }
        let id = problem
            .add_pose(format!("capture{k}"), pose)
            .map_err(builder_error)?;
        if k == 0 {
            problem.fix(id).map_err(builder_error)?;
        }
        captures[k] = Some(id);
    }

    let mut point_blocks = Vec::with_capacity(points.len());
    for (i, (point, track)) in points.iter().zip(tracks).enumerate() {
        let block = problem
            .add_block(
                format!("p{i}"),
                &[point.x, point.y, point.z],
                Manifold::Euclidean,
            )
            .map_err(builder_error)?;
        problem
            .set_labels(block, &["x", "y", "z"])
            .map_err(builder_error)?;
        for obs in track {
            // Indices were validated.
            let (Some(camera), Some(capture)) = (blocks.camera(obs.camera), captures[obs.capture])
            else {
                return Err(MvgError::numerical(format!(
                    "track {i}: camera {} or capture {} missing from the problem",
                    obs.camera, obs.capture
                )));
            };
            let Some(chain) = blocks.chain(obs.camera, capture) else {
                return Err(MvgError::numerical(format!(
                    "track {i}: no chain for camera {}",
                    obs.camera
                )));
            };
            problem
                .add_reprojection(
                    camera,
                    chain,
                    TargetPoint::Block(block),
                    obs.pixel,
                    PixelNoise::Stated(obs.noise),
                    opts.loss,
                )
                .map_err(|e| with_context(&format!("track {i}"), builder_error(e)))?;
        }
        point_blocks.push(block);
    }
    if let Scale::Distance {
        points: [a, b],
        distance,
        sigma,
    } = opts.scale
    {
        problem
            .add_residual(
                PointDistance { distance },
                &[point_blocks[a], point_blocks[b]],
                RobustLoss::None,
                Noise::Known(sigma),
            )
            .map_err(builder_error)?;
    }

    // `SolveOptions` is non-exhaustive: start from its default.
    let mut solve_options = SolveOptions::default();
    solve_options.max_iters = opts.max_iters;
    let solution = solve(&problem, &solve_options).map_err(builder_error)?;

    let refined_poses: Vec<Iso3> = rig_se3_world
        .iter()
        .zip(&captures)
        .map(|(initial, block)| block.and_then(|b| solution.pose(b)).unwrap_or(*initial))
        .collect();
    let refined_points = point_blocks
        .iter()
        .zip(points)
        .map(|(&b, initial)| {
            solution
                .value(b)
                .map_or(*initial, |v| Pt3::new(v[0], v[1], v[2]))
        })
        .collect::<Vec<_>>();

    let (out_rig, uncertainty) = match opts.calibration {
        Use::Prior => {
            let cameras = (0..rig.num_cameras())
                .map(|k| {
                    blocks
                        .camera(k)
                        .and_then(|id| solution.camera(id))
                        .ok_or_else(|| {
                            MvgError::numerical(format!("camera {k} missing from the solution"))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let extrinsics = rig
                .cam_se3_rig()
                .iter()
                .enumerate()
                .map(|(k, given)| {
                    blocks
                        .cam_se3_rig(k)
                        .and_then(|b| solution.pose(b))
                        .unwrap_or(*given)
                })
                .collect();
            let refined = CalibratedRig::new(
                cameras,
                extrinsics,
                calibration_share(&solution.uncertainty),
            )?;
            (refined, solution.uncertainty)
        }
        _ => (rig.clone(), solution.uncertainty),
    };

    let reprojection_rms_px = reprojection_rms(&out_rig, &refined_poses, &refined_points, tracks)?;
    Ok(BundleAdjustment {
        rig_se3_world: refined_poses,
        points: refined_points,
        rig: out_rig,
        uncertainty,
        report: solution.report,
        reprojection_rms_px,
    })
}

/// Everything checked before the problem is built. Returns which captures
/// have observations.
fn validate(
    rig: &CalibratedRig,
    rig_se3_world: &[Iso3],
    points: &[Pt3],
    tracks: &[Vec<RigObservation>],
    opts: &BundleAdjustOptions,
) -> Result<Vec<bool>> {
    if points.len() != tracks.len() {
        return Err(MvgError::invalid_input(format!(
            "{} points but {} tracks: tracks[i] observes points[i]",
            points.len(),
            tracks.len()
        )));
    }
    for (k, pose) in rig_se3_world.iter().enumerate() {
        let finite = pose.translation.vector.iter().all(|v| v.is_finite())
            && pose.rotation.coords.iter().all(|v| v.is_finite());
        if !finite {
            return Err(MvgError::invalid_input(format!(
                "capture {k}: the initial pose is not finite"
            )));
        }
    }
    for (i, p) in points.iter().enumerate() {
        if !p.coords.iter().all(|v| v.is_finite()) {
            return Err(MvgError::invalid_input(format!(
                "point {i}: the initial point is not finite"
            )));
        }
    }
    let mut observed = vec![false; rig_se3_world.len()];
    for (i, track) in tracks.iter().enumerate() {
        // Track length, index ranges and a finite viewing ray per pixel.
        track_views(rig, rig_se3_world, track)
            .map_err(|e| with_context(&format!("track {i}"), e))?;
        for obs in track {
            observed[obs.capture] = true;
        }
    }
    if !observed.first().copied().unwrap_or(false) {
        return Err(MvgError::invalid_input(
            "capture 0 has no observations, but it defines the world frame",
        ));
    }
    let captures = observed.iter().filter(|&&o| o).count();
    if captures < 2 {
        let mut cameras: Vec<usize> = tracks.iter().flatten().map(|o| o.camera).collect();
        cameras.sort_unstable();
        cameras.dedup();
        if cameras.len() < 2 {
            return Err(MvgError::invalid_input(
                "one capture seen by one camera: there is nothing to adjust, which needs at \
                 least two captures or two cameras",
            ));
        }
    }
    if opts.calibration == Use::Prior && rig.uncertainty().covariance().is_none() {
        return Err(MvgError::invalid_input(
            "Use::Prior needs a rig with a covariance; use Use::Consider for one without",
        ));
    }
    match opts.scale {
        Scale::Rig => check_rig_scale(rig, tracks)?,
        Scale::Distance {
            points: [a, b],
            distance,
            sigma,
        } => {
            if a >= points.len() || b >= points.len() {
                return Err(MvgError::invalid_input(format!(
                    "Scale::Distance: points {a} and {b} must be in range: {} points given",
                    points.len()
                )));
            }
            if a == b {
                return Err(MvgError::invalid_input(format!(
                    "Scale::Distance: a distance needs two distinct points, got {a} twice"
                )));
            }
            if !(distance.is_finite() && distance > 0.0 && sigma.is_finite() && sigma > 0.0) {
                return Err(MvgError::invalid_input(format!(
                    "Scale::Distance: distance {distance} and sigma {sigma} must be finite and \
                     positive"
                )));
            }
            if (points[a] - points[b]).norm() == 0.0 {
                return Err(MvgError::invalid_input(format!(
                    "Scale::Distance: the initial points {a} and {b} coincide"
                )));
            }
        }
    }
    Ok(observed)
}

/// `Scale::Rig` needs a capture in which two cameras with distinct centres
/// both observe points.
fn check_rig_scale(rig: &CalibratedRig, tracks: &[Vec<RigObservation>]) -> Result<()> {
    let hint = "state a Scale::Distance instead";
    if rig.num_cameras() < 2 {
        return Err(MvgError::invalid_input(format!(
            "Scale::Rig needs a rig of two or more cameras; {hint}"
        )));
    }
    let centre = |c: usize| rig.cam_se3_rig()[c].inverse().translation.vector;
    let mut seen: Vec<(usize, usize)> = tracks
        .iter()
        .flatten()
        .map(|o| (o.capture, o.camera))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    let fixes_scale = seen.iter().enumerate().any(|(n, &(k, a))| {
        seen[n + 1..]
            .iter()
            .take_while(|&&(k2, _)| k2 == k)
            .any(|&(_, b)| (centre(a) - centre(b)).norm() > 1e-9)
    });
    if fixes_scale {
        Ok(())
    } else {
        Err(MvgError::invalid_input(format!(
            "Scale::Rig needs a capture observed by two cameras with distinct centres; {hint}"
        )))
    }
}

/// The RMS distance, in pixels, between each observation and its point
/// projected at the solution.
fn reprojection_rms(
    rig: &CalibratedRig,
    rig_se3_world: &[Iso3],
    points: &[Pt3],
    tracks: &[Vec<RigObservation>],
) -> Result<Real> {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (i, (point, track)) in points.iter().zip(tracks).enumerate() {
        for obs in track {
            let in_camera = rig.cam_se3_rig()[obs.camera] * rig_se3_world[obs.capture] * point;
            let model = rig.model(obs.camera).expect("camera index was validated");
            let pixel = model.project_point(&in_camera).ok_or_else(|| {
                MvgError::numerical(format!(
                    "track {i}: the adjusted point does not project in camera {}",
                    obs.camera
                ))
            })?;
            sum += (pixel - obs.pixel).norm_squared();
            count += 1;
        }
    }
    Ok((sum / count as Real).sqrt())
}

/// The part of an adjustment's uncertainty that belongs to the calibration:
/// the cameras' and extrinsics' labels, without capture poses and points.
fn calibration_share(uncertainty: &Uncertainty) -> Uncertainty {
    let Some(c) = uncertainty.covariance() else {
        return uncertainty.clone();
    };
    let is_calibration = |label: &str| {
        let head = label.split('.').next().unwrap_or("");
        let numbered = |prefix: &str| {
            head.strip_prefix(prefix)
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        };
        !(numbered("capture") || numbered("p"))
    };
    let keep: Vec<usize> = (0..c.labels.len())
        .filter(|&i| is_calibration(&c.labels[i]))
        .collect();
    Uncertainty::Available(ParameterCovariance {
        labels: keep.iter().map(|&i| c.labels[i].clone()).collect(),
        covariance: keep
            .iter()
            .map(|&i| keep.iter().map(|&j| c.covariance[i][j]).collect())
            .collect(),
        views: Vec::new(),
        fixed: c
            .fixed
            .iter()
            .filter(|l| is_calibration(l))
            .cloned()
            .collect(),
        noise: c.noise.clone(),
        num_residuals: c.num_residuals,
        dof: c.dof,
        view_scatter: c.view_scatter,
    })
}

/// An input the builder or the solve rejects is invalid input; its message says why.
fn builder_error(e: vision_calibration_optim::Error) -> MvgError {
    match e {
        vision_calibration_optim::Error::InvalidInput { reason } => MvgError::invalid_input(reason),
        other => MvgError::numerical(format!("bundle adjustment: {other}")),
    }
}
