//! The information one robot pose's view carries about a rig hand-eye
//! calibration: every camera's parameters, the rig, the hand-eye and the
//! target.
//!
//! A view at the robot pose `G = base_se3_gripper` observes target points
//! `X` through camera `i` at the pixel `q(θᵢ, p)`, with the camera-frame
//! point
//!
//! ```text
//! eye-in-hand:  p = Cᵢ · H⁻¹ · G⁻¹ · B · X    (H = gripper_se3_rig, B = base_se3_target)
//! eye-to-hand:  p = Cᵢ · H · G · T · X        (H = rig_se3_base,    T = gripper_se3_target)
//! ```
//!
//! and `Cᵢ = cam_se3_rig`. Per unit pixel variance the view's information is
//! `Σ JᵀJ` over its points, `J` the derivative of the pixel with respect to
//! the parameters in the coordinates a rig hand-eye covariance reports: a
//! camera's parameters by name, and each pose `P` by the right-perturbation
//! tangent `[ω, v]` of `P · Exp([ω, v])`, labelled `<pose>.rx … <pose>.tz`.
//!
//! A robot pose the solve corrects (`G · Exp(δ)`, under a zero-mean prior
//! of the robot's pose error) has an unknown of its own: `δ` is
//! marginalized, its prior entering as the information
//! `σ_px² · diag(1/σ_rot², 1/σ_trans²)` per unit pixel variance, as the
//! solve weighs it against the reprojection residuals.
//!
//! `J` is exact: the forward-mode derivative of the camera kernels with
//! respect to `θᵢ` and the camera-frame point, chained analytically through
//! the poses.

use nalgebra::{DMatrix, Matrix3, Vector3};
use num_dual::{DualDVec64, jacobian};
use vision_calibration_core::{CameraParams, Iso3, Pt3};

use super::{invert_spd, project};
use crate::Error;
use crate::ir::HandeyeMode;
use crate::problems::RobotPoseSigma;
use crate::projection::CameraVector;

/// A rig hand-eye calibration at its estimate: the chain a view's target
/// points are projected through.
#[derive(Debug, Clone, Copy)]
pub struct RigHandeyeGeometry<'a> {
    /// Each camera's parameters, in its own model.
    pub cameras: &'a [CameraParams],
    /// Each camera's `cam_se3_rig`, one per camera.
    pub cam_se3_rig: &'a [Iso3],
    /// Where the rig is mounted.
    pub mode: HandeyeMode,
    /// `gripper_se3_rig` eye-in-hand, `rig_se3_base` eye-to-hand.
    pub handeye: Iso3,
    /// `base_se3_target` eye-in-hand, `gripper_se3_target` eye-to-hand.
    pub target: Iso3,
}

/// How a view's robot pose enters the calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum RobotPose {
    /// The robot pose is taken as exact.
    Exact,
    /// The solve corrects the robot pose under a zero-mean prior of
    /// `sigma`, weighed against reprojection residuals of noise
    /// `pixel_sigma` (pixels); the correction is marginalized.
    Corrected {
        /// The robot's pose error.
        sigma: RobotPoseSigma,
        /// The reprojection noise the prior is weighed against, pixels.
        pixel_sigma: f64,
    },
}

/// The information a view carries about labelled parameters, per unit
/// pixel variance.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RigViewInformation {
    /// The parameters, labelled as a rig hand-eye covariance labels them:
    /// each camera's parameters (prefixed `cam<i>.` with more than one
    /// camera), each camera's `cam<i>.cam_se3_rig` (more than one camera),
    /// the hand-eye and the target pose.
    pub labels: Vec<String>,
    /// The information over those parameters, in px⁻² per unit of each:
    /// symmetric, one row per label.
    pub matrix: DMatrix<f64>,
}

/// The information the view at the robot pose `base_se3_gripper` carries
/// about the rig hand-eye calibration `geometry`, per unit pixel variance
/// (see the module documentation). `points[i]` are the target points (its
/// own frame, metres) camera `i` observes; every one counts.
///
/// # Errors
///
/// [`Error::InvalidInput`] when the camera counts disagree, a camera has a
/// general sensor homography, a point is outside its camera's projection
/// domain, or a robot pose prior or its pixel noise is not positive and
/// finite; [`Error::Numerical`] when a projection is not finite.
pub fn rig_handeye_view_information(
    geometry: &RigHandeyeGeometry<'_>,
    base_se3_gripper: &Iso3,
    points: &[Vec<Pt3>],
    robot: RobotPose,
) -> Result<RigViewInformation, Error> {
    let cameras = geometry.cameras;
    if geometry.cam_se3_rig.len() != cameras.len() || points.len() != cameras.len() {
        return Err(Error::invalid_input(format!(
            "{} cameras, {} cam_se3_rig poses and {} point sets",
            cameras.len(),
            geometry.cam_se3_rig.len(),
            points.len()
        )));
    }
    let prior = match robot {
        RobotPose::Exact => None,
        RobotPose::Corrected { sigma, pixel_sigma } => {
            sigma.validate()?;
            if !(pixel_sigma.is_finite() && pixel_sigma > 0.0) {
                return Err(Error::invalid_input(format!(
                    "the reprojection noise {pixel_sigma} is not positive and finite"
                )));
            }
            let rot = (pixel_sigma / sigma.rot_sigma).powi(2);
            let trans = (pixel_sigma / sigma.trans_sigma).powi(2);
            Some([rot, rot, rot, trans, trans, trans])
        }
    };

    let layout = Layout::new(cameras, geometry.mode)?;
    let n = layout.labels.len();
    // The robot correction's six columns follow the parameters.
    let mut a = DMatrix::<f64>::zeros(n + 6, n + 6);
    let mut row = DMatrix::<f64>::zeros(2, n + 6);
    for (i, (camera, model)) in cameras.iter().zip(&layout.vectors).enumerate() {
        let chain = Chain::new(geometry, i, base_se3_gripper);
        let m = model.theta.len();
        let mut x = model.theta.clone().resize_vertically(m + 3, 0.0);
        let projection = camera.projection.values();
        for point in &points[i] {
            let p = chain.camera_point(point);
            if !p.iter().all(|v| v.is_finite())
                || !model.model.projection.in_domain(&projection, &p)
            {
                return Err(Error::invalid_input(format!(
                    "target point {point} is outside camera {i}'s field of view"
                )));
            }
            x.rows_mut(m, 3).copy_from(&p);
            let (q, j) = jacobian(
                |v: nalgebra::DVector<DualDVec64>| project(model.model, &v, m),
                &x,
            );
            if !q.iter().chain(j.iter()).all(|v| v.is_finite()) {
                return Err(Error::numerical(format!(
                    "the projection of target point {point} by camera {i} is not finite"
                )));
            }
            let j_p = j.fixed_columns::<3>(m).into_owned();
            row.fill(0.0);
            let start = layout.camera[i];
            row.view_mut((0, start), (2, m)).copy_from(&j.columns(0, m));
            for (column, d_p) in chain.pose_derivatives(point, &layout, i) {
                // d_p: ∂p/∂[ω, v], 3 × 6.
                row.view_mut((0, column), (2, 6)).copy_from(&(j_p * d_p));
            }
            a.gemm_tr(1.0, &row, &row, 1.0);
        }
    }

    let a_tt = a.view((0, 0), (n, n)).into_owned();
    let matrix = match prior {
        None => a_tt,
        Some(prior) => {
            let a_td = a.view((0, n), (n, 6)).into_owned();
            let mut a_dd = a.view((n, n), (6, 6)).into_owned();
            for (k, p) in prior.iter().enumerate() {
                a_dd[(k, k)] += p;
            }
            // The prior makes `a_dd` positive definite.
            let a_dd_inv = invert_spd(&a_dd).ok_or_else(|| {
                Error::numerical("the robot pose correction's information is not positive definite")
            })?;
            let s = a_tt - &a_td * a_dd_inv * a_td.transpose();
            (&s + s.transpose()) * 0.5
        }
    };
    Ok(RigViewInformation {
        labels: layout.labels,
        matrix,
    })
}

/// Where each parameter group's columns start, and their labels.
struct Layout {
    vectors: Vec<CameraVector>,
    labels: Vec<String>,
    /// First column of each camera's parameters.
    camera: Vec<usize>,
    /// First column of each camera's `cam_se3_rig`; `None` for a single
    /// camera, whose camera is the rig.
    extrinsics: Vec<Option<usize>>,
    handeye: usize,
    target: usize,
}

impl Layout {
    fn new(cameras: &[CameraParams], mode: HandeyeMode) -> Result<Self, Error> {
        let multi = cameras.len() > 1;
        let prefix = |i: usize, s: &str| {
            if multi {
                format!("cam{i}.{s}")
            } else {
                s.to_string()
            }
        };
        let vectors = cameras
            .iter()
            .map(CameraVector::new)
            .collect::<Result<Vec<_>, _>>()?;
        let mut labels = Vec::new();
        let mut camera = Vec::new();
        for (i, vector) in vectors.iter().enumerate() {
            camera.push(labels.len());
            labels.extend(vector.labels.iter().map(|l| prefix(i, l)));
        }
        let mut extrinsics = Vec::new();
        for i in 0..cameras.len() {
            extrinsics.push(multi.then_some(labels.len()));
            if multi {
                labels.extend(pose_labels(&prefix(i, "cam_se3_rig")));
            }
        }
        let rig = if multi { "rig" } else { "cam" };
        let (handeye, target) = match mode {
            HandeyeMode::EyeInHand => (format!("gripper_se3_{rig}"), "base_se3_target"),
            HandeyeMode::EyeToHand => (format!("{rig}_se3_base"), "gripper_se3_target"),
        };
        let handeye_column = labels.len();
        labels.extend(pose_labels(&handeye));
        let target_column = labels.len();
        labels.extend(pose_labels(target));
        Ok(Self {
            vectors,
            labels,
            camera,
            extrinsics,
            handeye: handeye_column,
            target: target_column,
        })
    }
}

fn pose_labels(name: &str) -> Vec<String> {
    ["rx", "ry", "rz", "tx", "ty", "tz"]
        .iter()
        .map(|c| format!("{name}.{c}"))
        .collect()
}

/// One camera's chain from the target to the camera at one robot pose.
struct Chain {
    mode: HandeyeMode,
    c: Iso3,
    h: Iso3,
    g: Iso3,
    t: Iso3,
}

impl Chain {
    fn new(geometry: &RigHandeyeGeometry<'_>, camera: usize, base_se3_gripper: &Iso3) -> Self {
        Self {
            mode: geometry.mode,
            c: geometry.cam_se3_rig[camera],
            h: geometry.handeye,
            g: *base_se3_gripper,
            t: geometry.target,
        }
    }

    fn camera_point(&self, x: &Pt3) -> Vector3<f64> {
        match self.mode {
            HandeyeMode::EyeInHand => {
                (self.c * self.h.inverse() * self.g.inverse() * self.t).transform_point(x)
            }
            HandeyeMode::EyeToHand => (self.c * self.h * self.g * self.t).transform_point(x),
        }
        .coords
    }

    /// `∂p/∂[ω, v]` (3 × 6) of each pose in the chain, with its first
    /// column: the camera's `cam_se3_rig` (more than one camera), the
    /// hand-eye, the target, and the robot correction (the last six
    /// columns).
    fn pose_derivatives(
        &self,
        x: &Pt3,
        layout: &Layout,
        camera: usize,
    ) -> Vec<(usize, nalgebra::Matrix3x6<f64>)> {
        let robot = layout.labels.len();
        let rot = |t: &Iso3| t.rotation.to_rotation_matrix().into_inner();
        let mut out = Vec::with_capacity(4);
        match self.mode {
            HandeyeMode::EyeInHand => {
                // p = C · H⁻¹ · G⁻¹ · B · X.
                let p_b = self.t.transform_point(x).coords;
                let p_g = self
                    .g
                    .inverse_transform_vector(&(p_b - self.g.translation.vector));
                let p_r = self
                    .h
                    .inverse_transform_vector(&(p_g - self.h.translation.vector));
                let r_c = rot(&self.c);
                let r_ch = r_c * rot(&self.h).transpose();
                let r_chg = r_ch * rot(&self.g).transpose();
                if let Some(column) = layout.extrinsics[camera] {
                    out.push((column, direct(&r_c, &p_r)));
                }
                out.push((layout.handeye, inverted(&r_c, &p_r)));
                out.push((layout.target, direct(&(r_chg * rot(&self.t)), &x.coords)));
                out.push((robot, inverted(&r_ch, &p_g)));
            }
            HandeyeMode::EyeToHand => {
                // p = C · H · G · T · X.
                let p_g = self.t.transform_point(x).coords;
                let p_b = self.g.transform_point(&Pt3::from(p_g)).coords;
                let p_r = self.h.transform_point(&Pt3::from(p_b)).coords;
                let r_c = rot(&self.c);
                let r_ch = r_c * rot(&self.h);
                let r_chg = r_ch * rot(&self.g);
                if let Some(column) = layout.extrinsics[camera] {
                    out.push((column, direct(&r_c, &p_r)));
                }
                out.push((layout.handeye, direct(&r_ch, &p_b)));
                out.push((layout.target, direct(&(r_chg * rot(&self.t)), &x.coords)));
                out.push((robot, direct(&r_chg, &p_g)));
            }
        }
        out
    }
}

/// `∂p/∂[ω, v]` for `p = A · P · Exp([ω, v]) · y`, `R` the rotation of
/// `A · P`: `R · [−[y]×, I]`.
fn direct(r: &Matrix3<f64>, y: &Vector3<f64>) -> nalgebra::Matrix3x6<f64> {
    let mut d = nalgebra::Matrix3x6::zeros();
    d.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-r * y.cross_matrix()));
    d.fixed_view_mut::<3, 3>(0, 3).copy_from(r);
    d
}

/// `∂p/∂[ω, v]` for `p = A · (P · Exp([ω, v]))⁻¹ · y = A · Exp(−[ω, v]) · z`,
/// `z = P⁻¹ · y` and `R` the rotation of `A`: `R · [[z]×, −I]`.
fn inverted(r: &Matrix3<f64>, z: &Vector3<f64>) -> nalgebra::Matrix3x6<f64> {
    let mut d = nalgebra::Matrix3x6::zeros();
    d.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(r * z.cross_matrix()));
    d.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-r));
    d
}

#[cfg(test)]
mod tests;
