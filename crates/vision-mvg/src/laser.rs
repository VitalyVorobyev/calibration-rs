//! Laser points from a calibration: a laser-line pixel on a known plane.
//!
//! A pixel `u` of the laser line seen by rig camera `k` lies on the ray
//! through it and on the laser plane `nᵀp + d = 0` of that camera, so it
//! fixes one 3D point: the ray's intersection with the plane. The point is
//! found through the camera's own model (distortion, fisheye and tilted
//! sensors need no separate undistortion) and polished on the forward
//! projection, `π(p; θ) = u` and `nᵀp + d = 0`, so it satisfies both
//! equations to within a nanopixel.
//!
//! The covariance is first-order. The point is the implicit function
//! `F(p; u, θ, n, d) = [π(p; θ) − u; nᵀp + d] = 0` of the pixel, the
//! camera's parameters `θ`, the plane `(n, d)` and, in the rig frame, the
//! camera's pose, so with `A = [∂π/∂p; nᵀ]` (3×3), in the camera frame,
//!
//! ```text
//! ∂p/∂u = A⁻¹ [I₂; 0]        ∂p/∂θ = −A⁻¹ [∂π/∂θ; 0]
//! ∂p/∂n = −A⁻¹ [0; pᵀ]       ∂p/∂d = −A⁻¹ [0; 1]
//! ```
//!
//! and `q = R p + t` carries the point into the rig frame with the camera's
//! `rig_se3_cam = (R, t)`. A perturbation `cam_se3_rig · Exp(ω, v)` of the
//! camera's extrinsic moves `q` by `[q]ₓ ω − v`.
//!
//! The capture's pose `rig_se3_world` carries `q` into the frame the points
//! are stated in, `x = rig_se3_world⁻¹ q`, rotating every derivative above;
//! a perturbation `rig_se3_world · Exp(ξ)` moves `x` by `[x]ₓ ω − v`. On a
//! robot `ξ = A ε_h + B ε_r` with `ε_h` the hand-eye's perturbation and
//! `ε_r` the robot pose's (`B = 0` in the frame the rig is fixed in), so the
//! hand-eye enters as one more calibration parameter, correlated with the
//! cameras and planes in the calibration's covariance, and each robot pose
//! as a source of its own. The calibration's covariance `Σ_c` (cameras,
//! planes, extrinsics and hand-eye, with their correlations, from the rig),
//! the robot poses' `Σ_r` and the pixels' `Σ_u` give
//!
//! ```text
//! Σ_x = J_c Σ_c J_cᵀ + Σ_k J_rk Σ_rk J_rkᵀ + blockdiag(J_u Σ_u J_uᵀ)
//! ```
//!
//! over all points. Every point shares the calibration, so the points are
//! correlated through the first term, and the points of one capture
//! through the second.

use crate::bearing::{bearing, camera_model};
use crate::sensitivity::{POSE_AXES, camera_coordinates};
use crate::triangulation::{sub_covariance, with_context};
use crate::types::{CapturePose, Captures};
use crate::{Error, Result};
use nalgebra::{DMatrix, Matrix2, Matrix3, Matrix3x6, Matrix6, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CalibratedRig, Mat3, ParameterCovariance, PointCovariance, Pt2, Pt3, Real, Uncertainty,
    UncertaintyUnavailable,
};
use vision_calibration_optim::CameraJacobian;

/// The pixel residual, in pixels, below which the intersection is polished.
const PIXEL_TOLERANCE: Real = 1e-9;
/// The pixel residual, in pixels, above which a point is not found.
const PIXEL_FAILURE: Real = 1e-6;
/// Newton steps of the polish.
const MAX_STEPS: usize = 5;
/// The smallest `|n·b|` of a ray `b` and a plane normal `n` that intersect.
const MIN_INCIDENCE_SINE: Real = 1e-9;

/// One pixel of the laser line, seen by rig camera `camera` at capture
/// `capture`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LaserObservation {
    /// Index of the capture that saw the line, into the call's
    /// [`Captures`]: 0 for a single capture.
    pub capture: usize,
    /// Index of the rig camera that saw the line; it needs a laser plane
    /// ([`CalibratedRig::with_laser_planes`]).
    pub camera: usize,
    /// The observed pixel on the laser line.
    pub pixel: Pt2,
    /// The pixel's covariance, px²: the laser-peak detector's, stated, not
    /// estimated ([`PointCovariance::new`], [`PointCovariance::isotropic`]).
    pub noise: PointCovariance,
}

impl LaserObservation {
    /// A laser pixel `pixel` of camera `camera` at capture `capture`, with
    /// the covariance `noise`.
    pub fn new(capture: usize, camera: usize, pixel: Pt2, noise: PointCovariance) -> Self {
        Self {
            capture,
            camera,
            pixel,
            noise,
        }
    }
}

/// A 3D point of the laser line.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LaserPoint {
    /// The point in the frame of the call's [`Captures`], metres.
    pub point: Pt3,
    /// The camera that saw it.
    pub camera: usize,
    /// The angle, in degrees, between the viewing ray and the laser plane:
    /// near 0 the ray grazes the plane and the depth is ill-determined, 90
    /// is head-on.
    pub incidence_deg: Real,
}

/// The points of one [`triangulate_laser`] call and their joint uncertainty.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct LaserTriangulation {
    /// One point per observation, in the frame of the call's [`Captures`].
    pub points: Vec<LaserPoint>,
    /// The joint covariance of the points' coordinates, m², with the labels
    /// `p{i}.x`, `p{i}.y`, `p{i}.z` for observation `i`: from the pixels'
    /// stated noise plus the calibration's share (cameras, laser planes,
    /// extrinsics and, on a robot, the hand-eye) and the robot poses'. Its
    /// `noise` is empty, `num_residuals` is the number of
    /// pixel coordinates and `dof` is 0: the points are determined, not
    /// fitted. Or why it is unavailable.
    pub uncertainty: Uncertainty,
}

impl LaserTriangulation {
    /// Point `i`'s 3×3 covariance, m²; `None` when the uncertainty is
    /// unavailable or `i` is out of range.
    pub fn point_covariance(&self, i: usize) -> Option<Mat3> {
        self.cross_covariance(i, i)
    }

    /// The 3×3 cross-covariance of points `i` and `j` (rows `i`, columns
    /// `j`), m²: the calibration is shared by every point, so it correlates
    /// them. `None` as for [`point_covariance`](Self::point_covariance).
    pub fn cross_covariance(&self, i: usize, j: usize) -> Option<Mat3> {
        let axes = |p: usize| ["x", "y", "z"].map(|a| format!("p{p}.{a}"));
        sub_covariance(&self.uncertainty, axes(i), axes(j))
    }
}

/// The 3D points of the laser line seen at `observations`, in the frame of
/// `captures`, with their joint covariance.
///
/// Each pixel is intersected with its camera's laser plane (the rig's
/// [`laser_plane`](CalibratedRig::laser_plane)) through the camera's own
/// model. `captures` states where the rig was at each capture and so the
/// frame the points are stated in: [`Captures::Rig`] for the rig frame,
/// [`Captures::World`] with the rig's exact pose at each capture, or, for a
/// rig calibrated on a robot, its gripper or base frame through the
/// calibration's hand-eye ([`Captures::Handeye`], [`Captures::Robot`] with
/// the robot's pose at each capture of a scan).
///
/// The covariance propagates, to first order, the pixels' stated noise, the
/// calibration's covariance (camera parameters, plane, the camera's
/// extrinsic and the hand-eye, with their correlations) and each robot
/// pose's stated covariance (independent of the calibration's and of the
/// other poses'). Parameters the covariance does not list were held fixed by
/// the calibration and count as exact, so the σ is conditional on what the
/// calibration held fixed.
///
/// The cost is `O(N² m)` for `N` points and `m` calibration parameters, and
/// the covariance is dense over the 3N coordinates; call per group of points
/// that need a joint covariance.
///
/// The uncertainty is
/// [`UncertaintyUnavailable::CalibrationWithoutCovariance`] when the rig has
/// no covariance, or its covariance lists none of the intrinsic, distortion
/// or sensor parameters of an observing camera (a laser-only calibration
/// that held the cameras fixed): a σ from the pixels alone would leave out
/// the calibration's share.
///
/// # Errors
///
/// [`Error::InvalidInput`] for a camera or capture out of range, a camera
/// without a laser plane, a pixel without a finite viewing ray or a camera
/// model with a general sensor homography, a robot frame the rig cannot
/// serve (no hand-eye, or a frame the variant of [`Captures`] does not
/// state), and for a covariance that lists part of a camera's extrinsic (a
/// pose is estimated whole or held); [`Error::Degenerate`] for a ray
/// parallel to its plane or meeting it behind the camera;
/// [`Error::Numerical`] when the projection has no usable derivative at
/// the point. The message names the observation.
pub fn triangulate_laser(
    rig: &CalibratedRig,
    captures: Captures<'_>,
    observations: &[LaserObservation],
) -> Result<LaserTriangulation> {
    let poses = captures.resolve(rig)?;
    let Located { cameras, found } = locate(rig, &poses, observations)?;
    let uncertainty = if observations.is_empty() {
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::NotComputed,
        }
    } else {
        uncertainty(rig, &poses, observations, &cameras, &found)?
    };
    Ok(LaserTriangulation {
        points: found.into_iter().map(|(point, _)| point).collect(),
        uncertainty,
    })
}

/// The points of `observations` with what their covariance needs, and the
/// derivatives of each observing camera (`None` for the others).
fn locate(
    rig: &CalibratedRig,
    poses: &[CapturePose],
    observations: &[LaserObservation],
) -> Result<Located> {
    let mut cameras: Vec<Option<CameraJacobian>> = (0..rig.num_cameras()).map(|_| None).collect();
    let mut found = Vec::with_capacity(observations.len());
    for (i, obs) in observations.iter().enumerate() {
        let context = format!("observation {i}");
        let capture = poses.get(obs.capture).ok_or_else(|| {
            Error::invalid_input(format!(
                "{context}: capture {} out of range: {} captures given",
                obs.capture,
                poses.len()
            ))
        })?;
        let model = camera_model(rig, obs.camera).map_err(|e| with_context(&context, e))?;
        let plane = rig.laser_plane(obs.camera).ok_or_else(|| {
            Error::invalid_input(format!(
                "{context}: camera {} has no laser plane",
                obs.camera
            ))
        })?;
        if cameras[obs.camera].is_none() {
            let jacobian = CameraJacobian::new(&rig.cameras()[obs.camera])
                .map_err(|e| Error::invalid_input(format!("{context}: {e}")))?;
            cameras[obs.camera] = Some(jacobian);
        }
        let jacobian = cameras[obs.camera].as_ref().expect("built above");
        let n = plane.normal().into_inner();
        let d = plane.distance();

        let b = bearing(model, &obs.pixel, i).map_err(|e| with_context(&context, e))?;
        let along = n.dot(&b);
        if along.abs() <= MIN_INCIDENCE_SINE {
            return Err(Error::degenerate(format!(
                "{context}: the viewing ray is parallel to the laser plane"
            )));
        }
        let t = -d / along;
        if !(t.is_finite() && t > 0.0) {
            return Err(Error::degenerate(format!(
                "{context}: the viewing ray meets the laser plane behind the camera"
            )));
        }

        // Polish p on [π(p) − u; nᵀp + d] = 0 with Newton steps.
        let u = Vector2::new(obs.pixel.x, obs.pixel.y);
        let no_derivative = || {
            Error::numerical(format!(
                "{context}: the camera's projection has no usable derivative at the point"
            ))
        };
        let mut p = t * b;
        let mut step = 0;
        let (projected, a_inv) = loop {
            let projected = jacobian.project(&p).ok_or_else(no_derivative)?;
            let miss = Vector2::new(projected.pixel.x, projected.pixel.y) - u;
            let dp = &projected.d_point;
            let a = Matrix3::new(
                dp[(0, 0)],
                dp[(0, 1)],
                dp[(0, 2)],
                dp[(1, 0)],
                dp[(1, 1)],
                dp[(1, 2)],
                n.x,
                n.y,
                n.z,
            );
            let a_inv = a
                .try_inverse()
                .filter(|m| m.iter().all(|v| v.is_finite()))
                .ok_or_else(|| {
                    Error::degenerate(format!(
                        "{context}: the pixel and the plane do not determine the point \
                         (singular system)"
                    ))
                })?;
            if miss.norm() < PIXEL_TOLERANCE || step == MAX_STEPS {
                if miss.norm() > PIXEL_FAILURE {
                    return Err(Error::numerical(format!(
                        "{context}: the point does not converge ({} px off the pixel)",
                        miss.norm()
                    )));
                }
                break (projected, a_inv);
            }
            p -= a_inv * Vector3::new(miss.x, miss.y, n.dot(&p) + d);
            step += 1;
        };

        let world_se3_rig = capture.rig_se3_world.inverse();
        let q = rig.cam_se3_rig()[obs.camera].inverse() * Pt3::from(p);
        let world_se3_cam = world_se3_rig * rig.cam_se3_rig()[obs.camera].inverse();
        found.push((
            LaserPoint {
                point: world_se3_rig * q,
                camera: obs.camera,
                incidence_deg: along.abs().min(1.0).asin().to_degrees(),
            },
            Sensitivity {
                p,
                q: q.coords,
                a_inv,
                d_params: projected.d_params,
                rotation: world_se3_cam.rotation.to_rotation_matrix().into_inner(),
                world_rotation: world_se3_rig.rotation.to_rotation_matrix().into_inner(),
            },
        ));
    }

    Ok(Located { cameras, found })
}

/// The result of [`locate`].
struct Located {
    /// The derivatives of each observing camera; `None` for the others.
    cameras: Vec<Option<CameraJacobian>>,
    /// Each observation's point and sensitivity.
    found: Vec<(LaserPoint, Sensitivity)>,
}

/// What one point's covariance needs, in its camera's frame.
struct Sensitivity {
    /// The point in the camera frame.
    p: Vector3<Real>,
    /// The point in the rig frame.
    q: Vector3<Real>,
    /// `A⁻¹`.
    a_inv: Matrix3<Real>,
    /// `∂π/∂θ`, 2 × the camera's parameters.
    d_params: nalgebra::OMatrix<Real, nalgebra::U2, nalgebra::Dyn>,
    /// The rotation of `world_se3_cam`, camera → the call's frame.
    rotation: Matrix3<Real>,
    /// The rotation of `world_se3_rig`, rig → the call's frame.
    world_rotation: Matrix3<Real>,
}

/// Where one camera's calibration parameters are among the calibration
/// coordinates `c` (columns of `J_c`) and the rig covariance.
struct CameraColumns {
    /// `(column of ∂π/∂θ, column in c)` of each parameter the covariance lists.
    theta: Vec<(usize, usize)>,
    /// The first of the four plane columns `nx ny nz d`, when the
    /// calibration estimated the plane.
    plane: Option<usize>,
    /// The first of the six extrinsic columns `rx ry rz tx ty tz`, when the
    /// calibration estimated the camera's pose.
    extrinsic: Option<usize>,
}

/// The first-order map from the calibration coordinates, the robot poses
/// and the pixels to the points, for the observations of one call.
struct Linearization {
    /// The labels of the columns of `j_c`: the calibration's covariance
    /// labels, then `capture<k>.base_se3_gripper.rx … tz` for each robot
    /// pose with a covariance.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the tests name columns by label")
    )]
    labels: Vec<String>,
    /// The positions in the calibration's covariance of its columns, the
    /// first `index.len()` of `j_c`.
    index: Vec<usize>,
    /// Each robot pose's first column and its covariance.
    robot: Vec<(usize, Matrix6<Real>)>,
    /// `∂x/∂c`, 3N × m.
    j_c: DMatrix<Real>,
    /// `∂x_i/∂u_i` (3×2) of each point.
    j_u: Vec<nalgebra::Matrix3x2<Real>>,
}

/// The linearization at the points `found`; `None` when the rig has no
/// covariance or it lists none of the parameters of an observing camera.
fn linearize(
    rig: &CalibratedRig,
    covariance: &ParameterCovariance,
    poses: &[CapturePose],
    observations: &[LaserObservation],
    cameras: &[Option<CameraJacobian>],
    found: &[(LaserPoint, Sensitivity)],
) -> Result<Option<Linearization>> {
    let multi = rig.num_cameras() > 1;

    // The calibration coordinates c: per observing camera, ascending, its
    // parameters (the order of CameraJacobian::labels), its plane and its
    // extrinsic, each only if the covariance lists it.
    let mut labels: Vec<String> = Vec::new();
    let mut index: Vec<usize> = Vec::new();
    let mut columns: Vec<Option<CameraColumns>> = (0..rig.num_cameras()).map(|_| None).collect();
    let mut push = |label: String, at: usize| {
        labels.push(label);
        index.push(at);
        index.len() - 1
    };
    for (k, jacobian) in cameras.iter().enumerate() {
        let Some(jacobian) = jacobian else {
            continue;
        };
        let coordinates = camera_coordinates(rig, covariance, k, jacobian)?;
        if coordinates.theta.is_empty() {
            return Ok(None);
        }
        let theta = coordinates
            .theta
            .into_iter()
            .map(|(j, label, at)| (j, push(label, at)))
            .collect();
        // The plane, whole or held (`CalibratedRig::with_laser_planes` checked).
        let plane_labels = CalibratedRig::plane_labels(k, multi);
        let plane = if plane_labels.iter().all(|l| covariance.index(l).is_some()) {
            let mut first = 0;
            for (i, label) in plane_labels.into_iter().enumerate() {
                let at = covariance.index(&label).expect("checked");
                let column = push(label, at);
                if i == 0 {
                    first = column;
                }
            }
            Some(first)
        } else {
            None
        };
        let extrinsic = coordinates.extrinsic.map(|pose| {
            let mut first = 0;
            for (i, (label, at)) in pose.into_iter().enumerate() {
                let column = push(label, at);
                if i == 0 {
                    first = column;
                }
            }
            first
        });
        columns[k] = Some(CameraColumns {
            theta,
            plane,
            extrinsic,
        });
    }

    // The hand-eye, when a capture is on the robot and the calibration
    // estimated it (whole: `CalibratedRig::with_handeye` checked).
    let on_robot = observations
        .iter()
        .any(|o| poses[o.capture].robot.is_some());
    let handeye = match rig.handeye_labels() {
        Some(names) if on_robot && names.iter().all(|l| covariance.index(l).is_some()) => {
            let mut first = 0;
            for (i, label) in names.into_iter().enumerate() {
                let at = covariance.index(&label).expect("checked");
                let column = push(label, at);
                if i == 0 {
                    first = column;
                }
            }
            Some(first)
        }
        _ => None,
    };
    // Each robot pose with a covariance: six columns of its own.
    let mut robot = Vec::new();
    let mut robot_column: Vec<Option<usize>> = vec![None; poses.len()];
    for obs in observations {
        let k = obs.capture;
        let stated = poses[k]
            .robot
            .as_ref()
            .and_then(|link| link.robot.as_ref())
            .and_then(|(_, pose)| pose.covariance());
        if let (None, Some(sigma)) = (robot_column[k], stated) {
            robot_column[k] = Some(labels.len());
            robot.push((labels.len(), sigma));
            labels.extend(POSE_AXES.map(|c| format!("capture{k}.base_se3_gripper.{c}")));
        }
    }

    let m = labels.len();
    let n = observations.len();
    let mut j_c = DMatrix::<Real>::zeros(3 * n, m);
    let mut j_u = Vec::with_capacity(n);
    for (i, (obs, (point, s))) in observations.iter().zip(found).enumerate() {
        let cols = columns[obs.camera].as_ref().expect("camera was observed");
        let (r, g) = (&s.rotation, &s.a_inv);
        // ∂p/∂u = A⁻¹ [I₂; 0]: the first two columns; `r` carries every
        // camera-frame derivative into the call's frame.
        let g2 = g.fixed_columns::<2>(0).into_owned();
        j_u.push(r * g2);

        // ∂p/∂θ = −A⁻¹ [∂π/∂θ; 0].
        for &(jcol, ccol) in &cols.theta {
            let column = -(r * (g2 * s.d_params.column(jcol)));
            j_c.fixed_view_mut::<3, 1>(3 * i, ccol).copy_from(&column);
        }
        // ∂p/∂n = −A⁻¹ [0; pᵀ], ∂p/∂d = −A⁻¹ [0; 1].
        if let Some(first) = cols.plane {
            let g3 = g.column(2).into_owned();
            let d_normal = -(r * (g3 * s.p.transpose()));
            j_c.fixed_view_mut::<3, 3>(3 * i, first)
                .copy_from(&d_normal);
            j_c.fixed_view_mut::<3, 1>(3 * i, first + 3)
                .copy_from(&(-(r * g3)));
        }
        // The extrinsic: q moves by [q]ₓ ω − v, x by its rotation.
        if let Some(first) = cols.extrinsic {
            let w = &s.world_rotation;
            j_c.fixed_view_mut::<3, 3>(3 * i, first)
                .copy_from(&(w * s.q.cross_matrix()));
            j_c.fixed_view_mut::<3, 3>(3 * i, first + 3)
                .copy_from(&(-w));
        }
        // The capture's pose: rig_se3_world · Exp(ξ) moves x by [x]ₓ ω − v,
        // with ξ = A ε_h + B ε_r on a robot.
        if let Some(link) = &poses[obs.capture].robot {
            let mut d_xi = Matrix3x6::<Real>::zeros();
            d_xi.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&point.point.coords.cross_matrix());
            d_xi.fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-Matrix3::identity()));
            if let Some(first) = handeye {
                j_c.fixed_view_mut::<3, 6>(3 * i, first)
                    .copy_from(&(d_xi * link.d_handeye));
            }
            if let (Some(first), Some((b, _))) = (robot_column[obs.capture], &link.robot) {
                j_c.fixed_view_mut::<3, 6>(3 * i, first)
                    .copy_from(&(d_xi * b));
            }
        }
    }
    Ok(Some(Linearization {
        labels,
        index,
        robot,
        j_c,
        j_u,
    }))
}

fn uncertainty(
    rig: &CalibratedRig,
    poses: &[CapturePose],
    observations: &[LaserObservation],
    cameras: &[Option<CameraJacobian>],
    found: &[(LaserPoint, Sensitivity)],
) -> Result<Uncertainty> {
    let without = Uncertainty::Unavailable {
        reason: UncertaintyUnavailable::CalibrationWithoutCovariance,
    };
    let Some(covariance) = rig.uncertainty().covariance() else {
        return Ok(without);
    };
    let Some(lin) = linearize(rig, covariance, poses, observations, cameras, found)? else {
        return Ok(without);
    };
    let n = observations.len();
    let m = lin.index.len();
    // The calibration's covariance, then each robot pose's, independent.
    let mut sigma_c = DMatrix::zeros(lin.j_c.ncols(), lin.j_c.ncols());
    sigma_c
        .view_mut((0, 0), (m, m))
        .copy_from(&DMatrix::from_fn(m, m, |a, b| {
            covariance.covariance[lin.index[a]][lin.index[b]]
        }));
    for (first, sigma_r) in &lin.robot {
        sigma_c
            .fixed_view_mut::<6, 6>(*first, *first)
            .copy_from(sigma_r);
    }
    let mut sigma = &lin.j_c * &sigma_c * lin.j_c.transpose();
    for (i, (obs, j_u)) in observations.iter().zip(&lin.j_u).enumerate() {
        let sigma_u = Matrix2::new(
            obs.noise.xx(),
            obs.noise.xy(),
            obs.noise.xy(),
            obs.noise.yy(),
        );
        let mut view = sigma.fixed_view_mut::<3, 3>(3 * i, 3 * i);
        view += j_u * sigma_u * j_u.transpose();
    }
    // Symmetrise away the rounding of the products.
    let sigma = (&sigma + sigma.transpose()) * 0.5;

    Ok(Uncertainty::Available(ParameterCovariance {
        labels: (0..n)
            .flat_map(|i| ["x", "y", "z"].map(|a| format!("p{i}.{a}")))
            .collect(),
        covariance: (0..3 * n)
            .map(|r| (0..3 * n).map(|c| sigma[(r, c)]).collect())
            .collect(),
        views: Vec::new(),
        fixed: Vec::new(),
        noise: Vec::new(),
        num_residuals: 2 * n,
        dof: 0,
        view_scatter: None,
        second_order: None,
    }))
}

#[cfg(test)]
mod tests;
