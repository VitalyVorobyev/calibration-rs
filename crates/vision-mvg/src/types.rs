//! Core types for multiple-view geometry.
//!
//! The input types of MVG estimation: [`Correspondence2D`] for the uncalibrated estimators,
//! [`PixelMatch`] for the calibrated two-view functions and [`Captures`] for
//! the frame a measurement is stated in.

use crate::{Error, Result};
use nalgebra::Matrix6;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CalibratedRig, Iso3, PointCovariance, Pt2, Pt3, Real, RobotFrame, RobotPose,
};

/// A pair of corresponding points observed in two views, as plain image
/// coordinates, for the uncalibrated estimators (fundamental matrix,
/// homography). The calibrated functions take [`PixelMatch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correspondence2D {
    /// Point observed in the first view.
    pub pt1: Pt2,
    /// Point observed in the second view.
    pub pt2: Pt2,
}

impl Correspondence2D {
    /// Create a new correspondence from two 2D points.
    pub fn new(pt1: Pt2, pt2: Pt2) -> Self {
        Self { pt1, pt2 }
    }

    /// Split a slice of correspondences into two separate point vectors.
    ///
    /// This is useful for calling low-level solvers that accept `(&[Pt2], &[Pt2])`.
    pub fn split(corrs: &[Self]) -> (Vec<Pt2>, Vec<Pt2>) {
        let mut pts1 = Vec::with_capacity(corrs.len());
        let mut pts2 = Vec::with_capacity(corrs.len());
        for c in corrs {
            pts1.push(c.pt1);
            pts2.push(c.pt2);
        }
        (pts1, pts2)
    }
}

/// A pair of matched pixels with their stated noise, for the calibrated
/// two-view functions.
///
/// `pt1` is a pixel of the first camera and `pt2` of the second, the two
/// cameras the call names. Each pixel's covariance states its measurement
/// noise, and the functions weigh the epipolar constraint by it. Build a
/// covariance with [`PointCovariance::new`] (per observation),
/// [`PointCovariance::isotropic`] (one σ) or
/// [`CalibratedRig::reprojection_noise`](vision_calibration_core::CalibratedRig::reprojection_noise)
/// (the calibration's level).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelMatch {
    /// The pixel in the first camera.
    pub pt1: Pt2,
    /// Its covariance, px².
    pub noise1: PointCovariance,
    /// The pixel in the second camera.
    pub pt2: Pt2,
    /// Its covariance, px².
    pub noise2: PointCovariance,
}

impl PixelMatch {
    /// A match of `pt1` (with covariance `noise1`) to `pt2` (with `noise2`).
    pub fn new(pt1: Pt2, noise1: PointCovariance, pt2: Pt2, noise2: PointCovariance) -> Self {
        Self {
            pt1,
            noise1,
            pt2,
            noise2,
        }
    }
}

/// A triangulated 3D point with quality diagnostics.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct TriangulatedPoint {
    /// Reconstructed 3D position, in the frame the triangulation's poses
    /// define (the world frame, or the first camera's frame for a relative
    /// pose).
    pub point: Pt3,
    /// Root-mean-square distance, in pixels, between each observation and the
    /// point projected through that observation's camera model. `None` when
    /// the point does not project in one of the cameras.
    pub reprojection_error: Option<Real>,
    /// Largest angle, in degrees, between two observing rays (the directions
    /// from each camera centre to the point).
    ///
    /// Small angles indicate poor triangulation geometry.
    pub parallax_deg: Real,
    /// Whether the point lies ahead along every observing bearing
    /// (`bearing · p_camera > 0`). For a pinhole this is positive depth; a
    /// wide-angle camera can see a point at negative `z` in front of it.
    pub in_front: bool,
}

/// Where the rig was at each capture of a measurement, and so the frame its
/// points are stated in. Each observation names its capture by index.
///
/// On a robot the rig's pose follows from its calibration's hand-eye
/// ([`CalibratedRig::handeye`](vision_calibration_core::CalibratedRig::handeye))
/// and, in the frame that moves with the robot, the robot's pose at each
/// capture; their uncertainty then enters the points' covariance. In the
/// frame the rig is fixed in (the gripper eye-in-hand, the base eye-to-hand)
/// the robot does not enter, and every capture is the same one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Captures<'a> {
    /// One capture: the points are stated in the rig frame (the reference
    /// camera's).
    Rig,
    /// The rig's pose at each capture in a world frame, `rig_se3_world[k]`
    /// (world → rig frame), exact: the points are stated in the world frame.
    World(&'a [Iso3]),
    /// One capture: the points are stated in the robot frame the rig is
    /// fixed in, through the calibration's hand-eye: the gripper for an
    /// eye-in-hand rig, the base for an eye-to-hand one. Naming the other
    /// frame is an error: it needs the robot's pose ([`Captures::Robot`]).
    Handeye(RobotFrame),
    /// The robot's pose at each capture, exact or with its covariance: the
    /// points are stated in the robot frame the rig moves in, the base for
    /// an eye-in-hand rig and the gripper for an eye-to-hand one. Naming the
    /// frame the rig is fixed in is an error ([`Captures::Handeye`]).
    Robot {
        /// The frame the points are stated in.
        frame: RobotFrame,
        /// The robot's pose `base_se3_gripper` at each capture.
        robot: &'a [RobotPose],
    },
}

/// One capture's rig pose and, on a robot, how it moves with the hand-eye
/// and the robot's pose.
#[derive(Debug, Clone)]
pub(crate) struct CapturePose {
    /// `rig_se3_world` of the capture (world = the measurement's frame).
    pub rig_se3_world: Iso3,
    /// `None` off a robot.
    pub robot: Option<RobotLink>,
}

/// How a capture's rig pose on a robot moves with the hand-eye and the
/// robot's pose (core's `Handeye::rig_in_frame`).
#[derive(Debug, Clone)]
pub(crate) struct RobotLink {
    /// `A`: the tangent of `rig_se3_world` per unit of the hand-eye's.
    pub d_handeye: Matrix6<Real>,
    /// `B` and the robot pose's covariance, when it is in the chain (the
    /// frame that moves with the robot); the robot pose itself.
    pub robot: Option<(Matrix6<Real>, RobotPose)>,
}

impl Captures<'_> {
    /// Each capture's rig pose, in capture order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] for a robot frame and a rig without a
    /// hand-eye, and for a frame the variant does not serve (see the
    /// variants).
    pub(crate) fn resolve(&self, rig: &CalibratedRig) -> Result<Vec<CapturePose>> {
        let on_robot = |frame: RobotFrame, robot: Option<&RobotPose>| -> Result<CapturePose> {
            let handeye = rig.handeye().ok_or_else(|| {
                Error::invalid_input(format!(
                    "points in the {} frame need the calibration's hand-eye; the rig has none",
                    frame.name()
                ))
            })?;
            let fixed_in = handeye.mount_frame();
            let base_se3_gripper = match (frame == fixed_in, robot) {
                (true, None) => Iso3::identity(),
                (false, Some(robot)) => *robot.base_se3_gripper(),
                (true, Some(_)) => {
                    return Err(Error::invalid_input(format!(
                        "the rig is fixed in the {} frame: state the points there with \
                         Captures::Handeye, the robot's pose does not enter",
                        frame.name()
                    )));
                }
                (false, None) => {
                    return Err(Error::invalid_input(format!(
                        "the rig moves in the {} frame with the robot: state the robot's \
                         pose at each capture with Captures::Robot",
                        frame.name()
                    )));
                }
            };
            let at = handeye.rig_in_frame(frame, &base_se3_gripper);
            Ok(CapturePose {
                rig_se3_world: at.rig_se3_frame,
                robot: Some(RobotLink {
                    d_handeye: at.d_handeye,
                    robot: at.d_robot.zip(robot.copied()),
                }),
            })
        };
        let exact = |rig_se3_world: Iso3| CapturePose {
            rig_se3_world,
            robot: None,
        };
        match *self {
            Captures::Rig => Ok(vec![exact(Iso3::identity())]),
            Captures::World(poses) => Ok(poses.iter().copied().map(exact).collect()),
            Captures::Handeye(frame) => Ok(vec![on_robot(frame, None)?]),
            Captures::Robot { frame, robot } => {
                robot.iter().map(|r| on_robot(frame, Some(r))).collect()
            }
        }
    }
}
