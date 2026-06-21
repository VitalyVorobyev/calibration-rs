//! Multi-view bundle adjustment with **frozen intrinsics**.
//!
//! Feature-gated behind `refine`. Jointly refines camera **poses** and 3D
//! **structure** by minimizing reprojection error across all views, using
//! tiny-solver's Levenberg-Marquardt optimizer with analytic residuals.
//!
//! Per-camera intrinsics `K` are held constant (baked into each reprojection
//! factor, not exposed as parameter blocks); only the extrinsic poses and the
//! world points move. This respects the ADR 0015 MVG ceiling for this crate: no
//! structure-from-motion, no pose-graph optimization, no loop closure — just a
//! final joint refinement over an already-initialized reconstruction.
//!
//! # Conventions
//!
//! - Poses are `T_C_W` ([`Iso3`]): they map a world point into the camera
//!   frame (`p_cam = R · p_world + t`). See ADR 0009.
//! - Observations are **pixel** coordinates; the frozen `K` projects the
//!   camera-frame ray to pixels.
//! - The reconstruction carries a 7-DOF similarity gauge freedom. Fixing the
//!   first camera ([`BundleAdjustmentOptions::fix_first_camera`], on by default)
//!   removes the 6-DOF rigid part (world-frame rotation + translation). The
//!   remaining 1-DOF **global scale** cannot be observed from reprojection error
//!   alone when structure is free — uniformly scaling all point depths and all
//!   free-camera translations leaves every projection unchanged. Anchor scale
//!   externally (e.g. a known baseline) for metric output; in practice LM
//!   damping keeps it near the initial reconstruction.

use crate::{MvgError, Result};
use nalgebra::{DVector, Quaternion, Translation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::sync::Arc;
use tiny_solver::LevenbergMarquardtOptimizer;
use tiny_solver::factors::Factor;
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::optimizer::{Optimizer, OptimizerOptions};
use tiny_solver::problem::Problem;
use vision_calibration_core::{Iso3, Mat3, Pt2, Pt3, Real};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A single image observation: camera `cam` saw world point `point` at `pixel`.
///
/// Indices reference the `intrinsics`/`init_poses` slices (`cam`) and the
/// `init_points` slice (`point`) passed to [`bundle_adjust`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BundleObservation {
    /// Index of the observing camera.
    pub cam: usize,
    /// Index of the observed 3D point.
    pub point: usize,
    /// Observed pixel coordinate.
    pub pixel: Pt2,
}

impl BundleObservation {
    /// Create a new observation.
    pub fn new(cam: usize, point: usize, pixel: Pt2) -> Self {
        Self { cam, point, pixel }
    }
}

/// Tuning knobs for [`bundle_adjust`].
#[derive(Debug, Clone, Copy)]
pub struct BundleAdjustmentOptions {
    /// Levenberg-Marquardt iteration ceiling.
    pub max_iterations: usize,
    /// Hold camera 0's pose fixed to remove the rigid gauge freedom.
    ///
    /// On by default. Fixing camera 0 removes the 6-DOF world-frame rotation
    /// and translation, but **not** global scale (a 1-DOF gauge that
    /// reprojection cannot constrain while structure is free). Disable only when
    /// an external prior already pins the world frame; otherwise the
    /// reconstruction drifts along the 6 rigid gauge directions, which LM
    /// damping merely regularizes rather than removes.
    pub fix_first_camera: bool,
}

impl Default for BundleAdjustmentOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            fix_first_camera: true,
        }
    }
}

/// Outcome of [`bundle_adjust`].
#[derive(Debug, Clone)]
pub struct BundleAdjustmentResult {
    /// Refined camera poses (`T_C_W`), one per input camera, in input order.
    ///
    /// Cameras with no observations are returned unchanged from `init_poses`.
    pub poses: Vec<Iso3>,
    /// Refined world points, one per input point, in input order.
    ///
    /// Points with no observations are returned unchanged from `init_points`.
    pub points: Vec<Pt3>,
    /// Root-mean-square reprojection error (pixels) before optimization.
    pub initial_rms: Real,
    /// Root-mean-square reprojection error (pixels) after optimization.
    pub final_rms: Real,
    /// Per-camera RMS reprojection error (pixels) after optimization.
    ///
    /// One entry per input camera; cameras with no observations report `0.0`.
    pub per_camera_rms: Vec<Real>,
}

/// Jointly refine camera poses and 3D structure by minimizing reprojection
/// error, holding per-camera intrinsics frozen.
///
/// # Arguments
///
/// - `intrinsics`: frozen `K` matrix per camera (`intrinsics.len()` == number
///   of cameras == `init_poses.len()`).
/// - `observations`: flat list of `(cam, point, pixel)` measurements.
/// - `init_poses`: initial `T_C_W` per camera.
/// - `init_points`: initial world-point positions.
/// - `opts`: tuning knobs (see [`BundleAdjustmentOptions`]).
///
/// Points should be observed in at least two views for their depth to be
/// well-constrained; under-observed points are still admitted (LM damping
/// leaves them near their initial value) so partially-observed datasets do not
/// fail outright.
///
/// # Errors
///
/// - [`MvgError::CountMismatch`] if `intrinsics.len() != init_poses.len()`.
/// - [`MvgError::InsufficientData`] if there are no observations.
/// - [`MvgError::InvalidInput`] if any observation indexes a camera or point
///   out of range.
/// - [`MvgError::NotConverged`] if the optimizer fails to return a solution.
pub fn bundle_adjust(
    intrinsics: &[Mat3],
    observations: &[BundleObservation],
    init_poses: &[Iso3],
    init_points: &[Pt3],
    opts: &BundleAdjustmentOptions,
) -> Result<BundleAdjustmentResult> {
    let n_cams = init_poses.len();
    let n_points = init_points.len();

    if intrinsics.len() != n_cams {
        return Err(MvgError::CountMismatch {
            expected: n_cams,
            got: intrinsics.len(),
        });
    }
    if observations.is_empty() {
        return Err(MvgError::InsufficientData { need: 1, got: 0 });
    }
    for obs in observations {
        if obs.cam >= n_cams || obs.point >= n_points {
            return Err(MvgError::invalid_input(format!(
                "observation references camera {} / point {} but there are {n_cams} cameras and {n_points} points",
                obs.cam, obs.point
            )));
        }
    }

    // Pre-flatten the frozen intrinsics (row-major) once.
    let k_flat: Vec<[Real; 9]> = intrinsics.iter().map(flatten_k).collect();

    // --- Build the problem ---------------------------------------------------
    let mut problem = Problem::new();
    let mut cam_used = vec![false; n_cams];
    let mut point_used = vec![false; n_points];

    for obs in observations {
        let factor = ReprojectionFactor {
            k: k_flat[obs.cam],
            obs: obs.pixel,
        };
        problem.add_residual_block(
            2,
            &[&pose_key(obs.cam), &point_key(obs.point)],
            Box::new(factor),
            None,
        );
        cam_used[obs.cam] = true;
        point_used[obs.point] = true;
    }

    // Seed initial values and attach manifolds to the free pose blocks.
    let mut initial: HashMap<String, DVector<Real>> = HashMap::new();
    for (i, pose) in init_poses.iter().enumerate() {
        if !cam_used[i] {
            continue;
        }
        initial.insert(pose_key(i), DVector::from_vec(iso3_to_block(pose)));

        let fixed = opts.fix_first_camera && i == 0;
        if fixed {
            // Gauge fix: do *not* attach the SE3 manifold; fix all seven raw
            // components so the block contributes zero columns (the proven
            // recipe from the optim tiny-solver backend).
            for idx in 0..7 {
                problem.fix_variable(&pose_key(i), idx);
            }
        } else {
            problem.set_variable_manifold(&pose_key(i), Arc::new(SE3Manifold));
        }
    }
    for (j, pt) in init_points.iter().enumerate() {
        if !point_used[j] {
            continue;
        }
        initial.insert(point_key(j), DVector::from_vec(vec![pt.x, pt.y, pt.z]));
    }

    let initial_rms = reprojection_rms(&k_flat, observations, init_poses, init_points).0;

    let optimizer = LevenbergMarquardtOptimizer::default();
    let result = optimizer
        .optimize(
            &problem,
            &initial,
            Some(OptimizerOptions {
                max_iteration: opts.max_iterations,
                ..Default::default()
            }),
        )
        .ok_or_else(|| MvgError::not_converged("bundle adjustment failed to converge"))?;

    // --- Read back the refined state ----------------------------------------
    let mut poses = init_poses.to_vec();
    for (i, pose) in poses.iter_mut().enumerate() {
        if let Some(block) = result.get(&pose_key(i)) {
            *pose = block_to_iso3(block);
        }
    }
    let mut points = init_points.to_vec();
    for (j, pt) in points.iter_mut().enumerate() {
        if let Some(block) = result.get(&point_key(j)) {
            *pt = Pt3::new(block[0], block[1], block[2]);
        }
    }

    let (final_rms, per_camera_rms) =
        reprojection_rms_per_camera(&k_flat, observations, &poses, &points, n_cams);

    Ok(BundleAdjustmentResult {
        poses,
        points,
        initial_rms,
        final_rms,
        per_camera_rms,
    })
}

// ---------------------------------------------------------------------------
// Reprojection factor
// ---------------------------------------------------------------------------

/// Reprojection error of one world point in one view under a frozen `K`.
///
/// Parameter blocks: `[pose (SE3, 7), point (3)]`.
struct ReprojectionFactor {
    /// Frozen intrinsics, row-major.
    k: [Real; 9],
    /// Observed pixel.
    obs: Pt2,
}

impl<T: nalgebra::RealField> Factor<T> for ReprojectionFactor {
    fn residual_func(&self, params: &[DVector<T>]) -> DVector<T> {
        debug_assert_eq!(params.len(), 2);
        let pose = &params[0]; // [qx, qy, qz, qw, tx, ty, tz]
        let point = &params[1]; // [x, y, z]

        let (rot, t) = se3_from_block(pose);
        let pw = Vector3::new(point[0].clone(), point[1].clone(), point[2].clone());
        let (u, v) = project_pinhole(&self.k, &rot, &t, &pw);

        let u_obs = T::from_f64(self.obs.x).unwrap();
        let v_obs = T::from_f64(self.obs.y).unwrap();
        DVector::from_vec(vec![u - u_obs, v - v_obs])
    }
}

/// Project a world point through `T_C_W` and a frozen pinhole `K`.
///
/// Generic over the scalar so the same path serves both the autodiff factor
/// and the `f64` RMS bookkeeping. The full 3×3 `K` is applied to the
/// normalized ray, so a non-standard bottom row degrades gracefully (for a
/// canonical pinhole `K` the homogeneous `w` is exactly 1).
fn project_pinhole<T: nalgebra::RealField>(
    k: &[Real; 9],
    rot: &UnitQuaternion<T>,
    t: &Vector3<T>,
    pw: &Vector3<T>,
) -> (T, T) {
    let pc = rot.transform_vector(pw) + t;
    let xn = pc.x.clone() / pc.z.clone();
    let yn = pc.y.clone() / pc.z.clone();
    let kf = |i: usize| T::from_f64(k[i]).unwrap();
    let u = kf(0) * xn.clone() + kf(1) * yn.clone() + kf(2);
    let v = kf(3) * xn.clone() + kf(4) * yn.clone() + kf(5);
    let w = kf(6) * xn + kf(7) * yn + kf(8);
    (u / w.clone(), v / w)
}

/// Decode an SE3 parameter block `[qx, qy, qz, qw, tx, ty, tz]`.
///
/// Mirrors the optim crate's `se3_from_block` (kept private there).
fn se3_from_block<T: nalgebra::RealField>(v: &DVector<T>) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert_eq!(v.len(), 7, "SE3 block must have 7 params");
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        v[3].clone(),
        v[0].clone(),
        v[1].clone(),
        v[2].clone(),
    ));
    let t = Vector3::new(v[4].clone(), v[5].clone(), v[6].clone());
    (q, t)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pose_key(i: usize) -> String {
    format!("pose_{i}")
}

fn point_key(j: usize) -> String {
    format!("point_{j}")
}

fn flatten_k(k: &Mat3) -> [Real; 9] {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = k[(r, c)];
        }
    }
    out
}

/// Encode an [`Iso3`] (`T_C_W`) as `[qx, qy, qz, qw, tx, ty, tz]`.
fn iso3_to_block(pose: &Iso3) -> Vec<Real> {
    let q = pose.rotation.coords; // (i, j, k, w)
    let t = pose.translation.vector;
    vec![q[0], q[1], q[2], q[3], t.x, t.y, t.z]
}

/// Decode `[qx, qy, qz, qw, tx, ty, tz]` back into an [`Iso3`].
fn block_to_iso3(v: &DVector<Real>) -> Iso3 {
    let rot = UnitQuaternion::from_quaternion(Quaternion::new(v[3], v[0], v[1], v[2]));
    let trans = Translation3::new(v[4], v[5], v[6]);
    Iso3::from_parts(trans, rot)
}

/// Total RMS reprojection error (pixels) and the squared-error sum / count.
fn reprojection_rms(
    k_flat: &[[Real; 9]],
    observations: &[BundleObservation],
    poses: &[Iso3],
    points: &[Pt3],
) -> (Real, usize) {
    let mut sum_sq = 0.0;
    for obs in observations {
        let (du, dv) = residual_xy(
            &k_flat[obs.cam],
            &poses[obs.cam],
            &points[obs.point],
            obs.pixel,
        );
        sum_sq += du * du + dv * dv;
    }
    let n = observations.len();
    let rms = if n == 0 {
        0.0
    } else {
        (sum_sq / n as Real).sqrt()
    };
    (rms, n)
}

/// Total RMS plus per-camera RMS (pixels).
fn reprojection_rms_per_camera(
    k_flat: &[[Real; 9]],
    observations: &[BundleObservation],
    poses: &[Iso3],
    points: &[Pt3],
    n_cams: usize,
) -> (Real, Vec<Real>) {
    let mut sum_sq = 0.0;
    let mut per_cam_sq = vec![0.0; n_cams];
    let mut per_cam_n = vec![0usize; n_cams];
    for obs in observations {
        let (du, dv) = residual_xy(
            &k_flat[obs.cam],
            &poses[obs.cam],
            &points[obs.point],
            obs.pixel,
        );
        let e = du * du + dv * dv;
        sum_sq += e;
        per_cam_sq[obs.cam] += e;
        per_cam_n[obs.cam] += 1;
    }
    let total = if observations.is_empty() {
        0.0
    } else {
        (sum_sq / observations.len() as Real).sqrt()
    };
    let per_camera_rms = per_cam_sq
        .iter()
        .zip(per_cam_n.iter())
        .map(|(&s, &n)| if n == 0 { 0.0 } else { (s / n as Real).sqrt() })
        .collect();
    (total, per_camera_rms)
}

/// Single-observation reprojection residual `(Δu, Δv)` in pixels.
fn residual_xy(k: &[Real; 9], pose: &Iso3, point: &Pt3, pixel: Pt2) -> (Real, Real) {
    let (u, v) = project_pinhole(k, &pose.rotation, &pose.translation.vector, &point.coords);
    (u - pixel.x, v - pixel.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3 as V3};

    /// Pinhole intrinsics used across the synthetic scene.
    fn intrinsics() -> Mat3 {
        Mat3::new(600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0)
    }

    /// Deterministic pseudo-random value in `[-1, 1]` (avoids a `rand` dep).
    fn jitter(seed: usize) -> Real {
        ((seed.wrapping_mul(2_654_435_761) % 1000) as Real) / 500.0 - 1.0
    }

    fn pose(euler: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(euler.0, euler.1, euler.2);
        Iso3::from_parts(Translation3::new(t.0, t.1, t.2), UnitQuaternion::from(rot))
    }

    /// Three ground-truth `T_C_W` poses; camera 0 is the identity (world frame
    /// anchor) so a `fix_first_camera` run is pinned to ground truth.
    fn gt_poses() -> Vec<Iso3> {
        vec![
            Iso3::identity(),
            pose((0.03, 0.05, -0.02), (-1.0, 0.1, 0.2)),
            pose((-0.04, -0.06, 0.03), (0.9, -0.15, -0.1)),
        ]
    }

    /// A 4×4 lattice of world points with depth variation (so pose and depth
    /// are both observable). All points sit well in front of every camera.
    fn gt_points() -> Vec<Pt3> {
        let mut pts = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = -0.6 + 0.4 * i as Real;
                let y = -0.6 + 0.4 * j as Real;
                let z = 4.5 + 0.3 * ((i + j) % 3) as Real;
                pts.push(Pt3::new(x, y, z));
            }
        }
        pts
    }

    fn project_gt(k: &Mat3, p: &Iso3, pt: &Pt3) -> Pt2 {
        let (u, v) = project_pinhole(
            &flatten_k(k),
            &p.rotation,
            &p.translation.vector,
            &pt.coords,
        );
        Pt2::new(u, v)
    }

    /// Every point seen by every camera (noiseless unless `noise_px > 0`).
    fn observe(k: &Mat3, poses: &[Iso3], points: &[Pt3], noise_px: Real) -> Vec<BundleObservation> {
        let mut obs = Vec::new();
        let mut s = 0usize;
        for (c, pose) in poses.iter().enumerate() {
            for (p, pt) in points.iter().enumerate() {
                let mut pix = project_gt(k, pose, pt);
                if noise_px > 0.0 {
                    pix.x += noise_px * jitter(s);
                    pix.y += noise_px * jitter(s + 1);
                }
                s += 2;
                obs.push(BundleObservation::new(c, p, pix));
            }
        }
        obs
    }

    fn perturb_pose(p: &Iso3, seed: usize) -> Iso3 {
        let delta = pose(
            (
                0.02 * jitter(seed),
                -0.03 * jitter(seed + 1),
                0.015 * jitter(seed + 2),
            ),
            (
                0.08 * jitter(seed + 3),
                -0.05 * jitter(seed + 4),
                0.06 * jitter(seed + 5),
            ),
        );
        p * delta
    }

    fn perturb_points(points: &[Pt3]) -> Vec<Pt3> {
        points
            .iter()
            .enumerate()
            .map(|(j, pt)| pt + 0.12 * V3::new(jitter(3 * j), jitter(3 * j + 1), jitter(3 * j + 2)))
            .collect()
    }

    /// Scale factor that maps a fixed-camera-0 estimate onto ground truth.
    ///
    /// With camera 0 at the origin, the only residual gauge is global scale;
    /// camera 1's baseline length fixes it.
    fn gt_scale(res: &BundleAdjustmentResult, gt: &[Iso3]) -> Real {
        gt[1].translation.vector.norm() / res.poses[1].translation.vector.norm()
    }

    #[test]
    fn converges_to_ground_truth_noiseless() {
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let obs = observe(&k, &poses, &points, 0.0);
        let intr = vec![k; poses.len()];

        // Camera 0 stays at GT (fixed); cameras 1..N and all points perturbed.
        let mut init_poses = poses.clone();
        init_poses[1] = perturb_pose(&poses[1], 11);
        init_poses[2] = perturb_pose(&poses[2], 23);
        let init_points = perturb_points(&points);

        let res = bundle_adjust(
            &intr,
            &obs,
            &init_poses,
            &init_points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap();

        assert!(
            res.final_rms < res.initial_rms,
            "rms did not drop: init={}, final={}",
            res.initial_rms,
            res.final_rms
        );
        // LM stops at its default error-decrease floor (~1e-5 px residual);
        // that is convergence for any practical purpose.
        assert!(
            res.final_rms < 1e-3,
            "final rms too large: {}",
            res.final_rms
        );

        // Recovery is exact only up to the unobservable global scale; remove it
        // via camera 1's baseline, then GT must be recovered to machine slop.
        let f = gt_scale(&res, &poses);
        assert!(
            (f - 1.0).abs() < 0.05,
            "scale drifted unexpectedly far: {f}"
        );
        for (i, (got, want)) in res.poses.iter().zip(poses.iter()).enumerate() {
            let dt = (f * got.translation.vector - want.translation.vector).norm();
            let dr = got.rotation.angle_to(&want.rotation); // rotation is scale-invariant
            assert!(
                dt < 1e-3 && dr < 1e-4,
                "camera {i} not recovered: dt={dt}, dr={dr}"
            );
        }
        for (j, (got, want)) in res.points.iter().zip(points.iter()).enumerate() {
            let d = (f * got.coords - want.coords).norm();
            assert!(d < 1e-3, "point {j} not recovered: d={d}");
        }
        assert!(res.per_camera_rms.iter().all(|&r| r < 1e-3));
    }

    #[test]
    fn perfect_init_does_not_diverge() {
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let obs = observe(&k, &poses, &points, 0.0);
        let intr = vec![k; poses.len()];

        let res = bundle_adjust(
            &intr,
            &obs,
            &poses,
            &points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap();

        assert!(
            res.final_rms < 1e-9,
            "perfect init drifted: {}",
            res.final_rms
        );
        for (got, want) in res.poses.iter().zip(poses.iter()) {
            assert!((got.translation.vector - want.translation.vector).norm() < 1e-6);
            assert!(got.rotation.angle_to(&want.rotation) < 1e-6);
        }
    }

    #[test]
    fn handles_pixel_noise() {
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let obs = observe(&k, &poses, &points, 0.3); // ~0.3 px deterministic noise
        let intr = vec![k; poses.len()];

        let mut init_poses = poses.clone();
        init_poses[1] = perturb_pose(&poses[1], 5);
        init_poses[2] = perturb_pose(&poses[2], 17);
        let init_points = perturb_points(&points);

        let res = bundle_adjust(
            &intr,
            &obs,
            &init_poses,
            &init_points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap();

        assert!(
            res.final_rms < res.initial_rms,
            "rms did not drop under noise: init={}, final={}",
            res.initial_rms,
            res.final_rms
        );
        assert!(
            res.final_rms < 0.5,
            "final rms above noise band: {}",
            res.final_rms
        );
        // Up to the scale gauge, poses land near GT (noise perturbs the optimum
        // only slightly).
        let f = gt_scale(&res, &poses);
        for (g, w) in res.poses.iter().skip(1).zip(poses.iter().skip(1)) {
            assert!((f * g.translation.vector - w.translation.vector).norm() < 0.02);
            assert!(g.rotation.angle_to(&w.rotation) < 0.01);
        }
    }

    #[test]
    fn without_gauge_fix_still_reduces_error() {
        // Exercises the SE3-manifold path on camera 0 (no fixing).
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let obs = observe(&k, &poses, &points, 0.0);
        let intr = vec![k; poses.len()];

        let mut init_poses = poses.clone();
        init_poses[1] = perturb_pose(&poses[1], 31);
        init_poses[2] = perturb_pose(&poses[2], 41);
        let init_points = perturb_points(&points);

        let res = bundle_adjust(
            &intr,
            &obs,
            &init_poses,
            &init_points,
            &BundleAdjustmentOptions {
                max_iterations: 100,
                fix_first_camera: false,
            },
        )
        .unwrap();

        // Without a gauge anchor the frame can drift, but the reprojection
        // residual must still collapse.
        assert!(res.final_rms < res.initial_rms);
        assert!(
            res.final_rms < 1e-4,
            "final rms too large: {}",
            res.final_rms
        );
    }

    #[test]
    fn unobserved_entities_pass_through() {
        let k = intrinsics();
        let poses = gt_poses();
        let mut points = gt_points();
        // Add an extra point that nobody observes.
        let orphan = Pt3::new(99.0, 99.0, 99.0);
        points.push(orphan);
        let obs = observe(&k, &poses, &points[..points.len() - 1], 0.0);
        let intr = vec![k; poses.len()];

        let init_points = points.clone();
        let res = bundle_adjust(
            &intr,
            &obs,
            &poses,
            &init_points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap();

        // The orphan point is returned unchanged.
        assert_eq!(*res.points.last().unwrap(), orphan);
    }

    #[test]
    fn rejects_count_mismatch() {
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let obs = observe(&k, &poses, &points, 0.0);
        let intr = vec![k; poses.len() - 1]; // one too few

        let err = bundle_adjust(
            &intr,
            &obs,
            &poses,
            &points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, MvgError::CountMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_empty_observations() {
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let intr = vec![k; poses.len()];

        let err = bundle_adjust(
            &intr,
            &[],
            &poses,
            &points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap_err();
        assert!(
            matches!(err, MvgError::InsufficientData { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn rejects_out_of_range_index() {
        let k = intrinsics();
        let poses = gt_poses();
        let points = gt_points();
        let intr = vec![k; poses.len()];
        let bad = vec![BundleObservation::new(0, points.len(), Pt2::new(1.0, 1.0))];

        let err = bundle_adjust(
            &intr,
            &bad,
            &poses,
            &points,
            &BundleAdjustmentOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(err, MvgError::InvalidInput { .. }), "got {err:?}");
    }
}
