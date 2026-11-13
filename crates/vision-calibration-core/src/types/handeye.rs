//! A calibrated rig on a robot: its hand-eye, the robot's pose at a capture
//! and the robot frame a measurement is stated in.

use nalgebra::Matrix6;
use serde::{Deserialize, Serialize};

use crate::{Error, Iso3, Real};

/// Where a calibrated rig sits on a robot: the hand-eye of its calibration.
///
/// The rig rides on the gripper (`EyeInHand`) or stands in the robot's base
/// frame (`EyeToHand`), and each mode states the one pose that fixes it,
/// the pose its calibration estimates and exports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Handeye {
    /// The rig rides on the gripper.
    EyeInHand {
        /// The rig's pose on the gripper (rig frame → gripper frame).
        gripper_se3_rig: Iso3,
    },
    /// The rig stands in the robot's base frame.
    EyeToHand {
        /// The base's pose in the rig (base frame → rig frame).
        rig_se3_base: Iso3,
    },
}

/// A frame of the robot a measurement's points are stated in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RobotFrame {
    /// The gripper (flange) frame.
    Gripper,
    /// The robot's base frame.
    Base,
}

impl RobotFrame {
    /// `gripper` or `base`, as it names a pose (`rig_se3_base`).
    pub fn name(self) -> &'static str {
        match self {
            RobotFrame::Gripper => "gripper",
            RobotFrame::Base => "base",
        }
    }
}

impl Handeye {
    /// The robot frame the rig is fixed in: the gripper eye-in-hand, the
    /// base eye-to-hand. In it the rig's pose is the hand-eye at every
    /// capture; in the other frame it moves with the robot.
    pub fn mount_frame(&self) -> RobotFrame {
        match self {
            Handeye::EyeInHand { .. } => RobotFrame::Gripper,
            Handeye::EyeToHand { .. } => RobotFrame::Base,
        }
    }

    /// The pose the calibration estimates: `gripper_se3_rig` eye-in-hand,
    /// `rig_se3_base` eye-to-hand.
    pub fn pose(&self) -> &Iso3 {
        match self {
            Handeye::EyeInHand { gripper_se3_rig } => gripper_se3_rig,
            Handeye::EyeToHand { rig_se3_base } => rig_se3_base,
        }
    }

    /// The rig's pose in the robot frame `frame`, `rig_se3_<frame>`, and its
    /// first-order dependence on the hand-eye and on the robot's pose.
    ///
    /// Every perturbation is a right perturbation over `[rx, ry, rz, tx,
    /// ty, tz]`, the tangent the calibration's covariance is over: with
    /// `ε_h` the hand-eye's ([`pose`](Self::pose)`·Exp(ε_h)`) and `ε_r` the
    /// robot's (`base_se3_gripper·Exp(ε_r)`), `rig_se3_<frame>` moves to
    /// `rig_se3_<frame>·Exp(A ε_h + B ε_r)`. `B` is `None` in the mount
    /// frame, where the robot's pose does not enter.
    #[doc(hidden)]
    pub fn rig_in_frame(&self, frame: RobotFrame, base_se3_gripper: &Iso3) -> RigInFrame {
        let p = base_se3_gripper;
        match (self, frame) {
            // rig_se3_gripper = H⁻¹.
            (Handeye::EyeInHand { gripper_se3_rig: h }, RobotFrame::Gripper) => RigInFrame {
                rig_se3_frame: h.inverse(),
                d_handeye: -se3_adjoint(h),
                d_robot: None,
            },
            // rig_se3_base = (P H)⁻¹.
            (Handeye::EyeInHand { gripper_se3_rig: h }, RobotFrame::Base) => {
                let base_se3_rig = p * h;
                RigInFrame {
                    rig_se3_frame: base_se3_rig.inverse(),
                    d_handeye: -se3_adjoint(&base_se3_rig),
                    d_robot: Some(-se3_adjoint(p)),
                }
            }
            // rig_se3_base = H.
            (Handeye::EyeToHand { rig_se3_base: h }, RobotFrame::Base) => RigInFrame {
                rig_se3_frame: *h,
                d_handeye: Matrix6::identity(),
                d_robot: None,
            },
            // rig_se3_gripper = H P.
            (Handeye::EyeToHand { rig_se3_base: h }, RobotFrame::Gripper) => RigInFrame {
                rig_se3_frame: h * p,
                d_handeye: se3_adjoint(&p.inverse()),
                d_robot: Some(Matrix6::identity()),
            },
        }
    }
}

/// The rig's pose in a robot frame and its first-order dependence on the
/// hand-eye and the robot's pose ([`Handeye::rig_in_frame`]).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigInFrame {
    /// `rig_se3_<frame>`.
    pub rig_se3_frame: Iso3,
    /// `A`: the tangent of `rig_se3_<frame>` per unit of the hand-eye's.
    pub d_handeye: Matrix6<Real>,
    /// `B`: the tangent of `rig_se3_<frame>` per unit of the robot pose's;
    /// `None` in the frame the rig is fixed in.
    pub d_robot: Option<Matrix6<Real>>,
}

/// `Ad(T) = [[R, 0], [[t]ₓ R, R]]` for a tangent `δ = [ω, v]`:
/// `T·Exp(δ) = Exp(Ad(T) δ)·T`.
#[doc(hidden)]
pub fn se3_adjoint(pose: &Iso3) -> Matrix6<Real> {
    let r = pose.rotation.to_rotation_matrix().into_inner();
    let t = pose.translation.vector;
    let tr = t.cross_matrix() * r;
    let mut ad = Matrix6::zeros();
    ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    ad.fixed_view_mut::<3, 3>(3, 0).copy_from(&tr);
    ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
    ad
}

/// The robot's pose at a capture, `base_se3_gripper` (gripper frame → base
/// frame), exact or with the covariance of its error.
///
/// The covariance is over the right perturbation `base_se3_gripper·Exp(δ)`,
/// `δ = [rx, ry, rz, tx, ty, tz]`: rotations in rad about the gripper's
/// axes and translations in metres along them, the robot's error as its
/// repeatability or accuracy states it. It may be singular (a component
/// stated exact) but not indefinite.
///
/// ```
/// use vision_calibration_core::{Iso3, RobotPose};
///
/// let pose = Iso3::translation(0.4, 0.0, 0.3);
/// let exact = RobotPose::exact(pose);
/// assert!(exact.covariance().is_none());
/// // 0.01° and 20 µm on each axis.
/// let robot = RobotPose::isotropic(pose, 0.01_f64.to_radians(), 20e-6)?;
/// assert!((robot.covariance().unwrap()[(3, 3)] - 4e-10).abs() < 1e-20);
/// # Ok::<(), vision_calibration_core::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(try_from = "RobotPoseRaw")]
pub struct RobotPose {
    #[cfg_attr(feature = "schemars", schemars(with = "crate::Iso3Schema"))]
    base_se3_gripper: Iso3,
    covariance: Option<[[Real; 6]; 6]>,
}

/// The wire form of [`RobotPose`], validated on conversion; it also
/// describes its JSON Schema.
#[derive(Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars", schemars(rename = "RobotPose"))]
struct RobotPoseRaw {
    /// The robot's pose, gripper frame → base frame.
    #[cfg_attr(feature = "schemars", schemars(with = "crate::Iso3Schema"))]
    base_se3_gripper: Iso3,
    /// The covariance of its right perturbation `[rx, ry, rz, tx, ty, tz]`
    /// (rad², m²), row-major; `null` for an exact pose.
    #[serde(default)]
    covariance: Option<[[Real; 6]; 6]>,
}

impl TryFrom<RobotPoseRaw> for RobotPose {
    type Error = Error;

    fn try_from(raw: RobotPoseRaw) -> Result<Self, Self::Error> {
        match raw.covariance {
            None => Ok(Self::exact(raw.base_se3_gripper)),
            Some(c) => Self::new(raw.base_se3_gripper, Matrix6::from_fn(|r, s| c[r][s])),
        }
    }
}

/// Relative tolerance of the covariance's symmetry and of its smallest
/// eigenvalue below zero.
const COVARIANCE_TOLERANCE: Real = 1e-9;

impl RobotPose {
    /// The robot's pose `base_se3_gripper`, taken as exact.
    pub fn exact(base_se3_gripper: Iso3) -> Self {
        Self {
            base_se3_gripper,
            covariance: None,
        }
    }

    /// The robot's pose `base_se3_gripper` with the covariance `covariance`
    /// of its right perturbation (see the type).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] unless the pose and every entry are finite and
    /// the covariance is symmetric and positive semi-definite (both to a
    /// relative `1e-9`).
    pub fn new(base_se3_gripper: Iso3, covariance: Matrix6<Real>) -> Result<Self, Error> {
        let finite = covariance.iter().all(|v| v.is_finite())
            && base_se3_gripper
                .translation
                .vector
                .iter()
                .all(|v| v.is_finite())
            && base_se3_gripper
                .rotation
                .coords
                .iter()
                .all(|v| v.is_finite());
        if !finite {
            return Err(Error::invalid_input(
                "a robot pose and its covariance must be finite",
            ));
        }
        let scale = covariance.abs().max().max(Real::MIN_POSITIVE);
        if (covariance - covariance.transpose()).abs().max() > COVARIANCE_TOLERANCE * scale {
            return Err(Error::invalid_input(
                "a robot pose's covariance must be symmetric",
            ));
        }
        let symmetric = (covariance + covariance.transpose()) * 0.5;
        let smallest = symmetric.symmetric_eigenvalues().min();
        if smallest < -COVARIANCE_TOLERANCE * scale {
            return Err(Error::invalid_input(format!(
                "a robot pose's covariance must be positive semi-definite \
                 (smallest eigenvalue {smallest})"
            )));
        }
        Ok(Self {
            base_se3_gripper,
            covariance: Some(std::array::from_fn(|r| {
                std::array::from_fn(|s| symmetric[(r, s)])
            })),
        })
    }

    /// The robot's pose with an uncorrelated error of `rot_sigma` rad about
    /// each gripper axis and `trans_sigma` m along each.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] unless both σ are finite and non-negative.
    pub fn isotropic(
        base_se3_gripper: Iso3,
        rot_sigma: Real,
        trans_sigma: Real,
    ) -> Result<Self, Error> {
        let valid = |s: Real| s.is_finite() && s >= 0.0;
        if !(valid(rot_sigma) && valid(trans_sigma)) {
            return Err(Error::invalid_input(format!(
                "a robot pose's σ must be finite and non-negative, got {rot_sigma} rad and \
                 {trans_sigma} m"
            )));
        }
        let (r, t) = (rot_sigma * rot_sigma, trans_sigma * trans_sigma);
        Self::new(
            base_se3_gripper,
            Matrix6::from_diagonal(&nalgebra::Vector6::new(r, r, r, t, t, t)),
        )
    }

    /// The robot's pose, gripper frame → base frame.
    pub fn base_se3_gripper(&self) -> &Iso3 {
        &self.base_se3_gripper
    }

    /// The covariance of the pose's right perturbation `[rx, ry, rz, tx,
    /// ty, tz]` (rad², m²); `None` for an exact pose.
    pub fn covariance(&self) -> Option<Matrix6<Real>> {
        self.covariance.map(|c| Matrix6::from_fn(|r, s| c[r][s]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Translation3, UnitQuaternion, Vector3, Vector6};

    fn iso(r: [f64; 3], t: [f64; 3]) -> Iso3 {
        Iso3::from_parts(
            Translation3::new(t[0], t[1], t[2]),
            UnitQuaternion::from_scaled_axis(Vector3::new(r[0], r[1], r[2])),
        )
    }

    /// `T·Exp(δ)` for one tangent coordinate at a time (exact: a pure
    /// rotation or a pure translation).
    fn perturbed(pose: &Iso3, delta: &Vector6<f64>) -> Iso3 {
        pose * iso(
            [delta[0], delta[1], delta[2]],
            [delta[3], delta[4], delta[5]],
        )
    }

    /// The right tangent `Log(base⁻¹ · moved)`, to first order.
    fn tangent(base: &Iso3, moved: &Iso3) -> Vector6<f64> {
        let d = base.inverse() * moved;
        let w = d.rotation.scaled_axis();
        let v = d.translation.vector;
        Vector6::new(w.x, w.y, w.z, v.x, v.y, v.z)
    }

    /// Central differences of `rig_se3_<frame>` against `A` and `B`, in
    /// both modes and both frames.
    #[test]
    fn frame_jacobians_match_finite_differences() {
        let robot = iso([0.3, -0.5, 0.9], [0.4, -0.2, 0.6]);
        let modes = [
            Handeye::EyeInHand {
                gripper_se3_rig: iso([0.1, 0.2, -0.3], [0.05, -0.02, 0.12]),
            },
            Handeye::EyeToHand {
                rig_se3_base: iso([-0.4, 0.1, 0.25], [0.3, 0.1, 1.2]),
            },
        ];
        let h = 1e-6;
        let mut worst: f64 = 0.0;
        for handeye in modes {
            for frame in [RobotFrame::Gripper, RobotFrame::Base] {
                let at = handeye.rig_in_frame(frame, &robot);
                assert_eq!(at.d_robot.is_none(), frame == handeye.mount_frame());
                for j in 0..6 {
                    let mut e = Vector6::zeros();
                    e[j] = h;
                    let shift = |handeye: &Handeye, robot: &Iso3| {
                        handeye.rig_in_frame(frame, robot).rig_se3_frame
                    };
                    let with = |sign: f64| -> Handeye {
                        match handeye {
                            Handeye::EyeInHand { gripper_se3_rig } => Handeye::EyeInHand {
                                gripper_se3_rig: perturbed(&gripper_se3_rig, &(sign * e)),
                            },
                            Handeye::EyeToHand { rig_se3_base } => Handeye::EyeToHand {
                                rig_se3_base: perturbed(&rig_se3_base, &(sign * e)),
                            },
                        }
                    };
                    let fd = (tangent(&at.rig_se3_frame, &shift(&with(1.0), &robot))
                        - tangent(&at.rig_se3_frame, &shift(&with(-1.0), &robot)))
                        / (2.0 * h);
                    worst = worst.max((fd - at.d_handeye.column(j)).abs().max());

                    let fd = (tangent(&at.rig_se3_frame, &shift(&handeye, &perturbed(&robot, &e)))
                        - tangent(
                            &at.rig_se3_frame,
                            &shift(&handeye, &perturbed(&robot, &(-e))),
                        ))
                        / (2.0 * h);
                    match at.d_robot {
                        Some(b) => worst = worst.max((fd - b.column(j)).abs().max()),
                        None => assert!(fd.abs().max() < 1e-12),
                    }
                }
            }
        }
        assert!(worst < 1e-7, "worst |FD − analytic| {worst}");
    }

    #[test]
    fn the_frames_compose_the_chain() {
        let robot = iso([0.3, -0.5, 0.9], [0.4, -0.2, 0.6]);
        let h = iso([0.1, 0.2, -0.3], [0.05, -0.02, 0.12]);
        let p = nalgebra::Point3::new(0.1, -0.2, 0.5);
        let eih = Handeye::EyeInHand { gripper_se3_rig: h };
        let in_base = eih.rig_in_frame(RobotFrame::Base, &robot).rig_se3_frame;
        assert!((in_base.inverse() * p - robot * (h * p)).norm() < 1e-12);
        let eth = Handeye::EyeToHand { rig_se3_base: h };
        let in_gripper = eth.rig_in_frame(RobotFrame::Gripper, &robot).rig_se3_frame;
        assert!((in_gripper.inverse() * p - robot.inverse() * (h.inverse() * p)).norm() < 1e-12);
        assert_eq!(eth.mount_frame(), RobotFrame::Base);
        assert_eq!(eih.pose(), &h);
    }

    #[test]
    fn robot_pose_validates_its_covariance() {
        let pose = iso([0.1, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(RobotPose::isotropic(pose, 1e-4, 1e-5).is_ok());
        assert!(RobotPose::isotropic(pose, 0.0, 0.0).is_ok());
        assert!(RobotPose::isotropic(pose, -1.0, 0.0).is_err());
        assert!(RobotPose::isotropic(pose, 0.0, Real::NAN).is_err());
        let mut asymmetric = Matrix6::identity();
        asymmetric[(0, 1)] = 0.5;
        assert!(RobotPose::new(pose, asymmetric).is_err());
        let mut indefinite = Matrix6::identity();
        indefinite[(0, 0)] = -1.0;
        assert!(RobotPose::new(pose, indefinite).is_err());

        let robot = RobotPose::isotropic(pose, 1e-3, 2e-3).unwrap();
        let json = serde_json::to_string(&robot).unwrap();
        let back: RobotPose = serde_json::from_str(&json).unwrap();
        assert_eq!(back.covariance(), robot.covariance());
        // The quaternion is renormalised on the way in.
        assert!(
            (back.base_se3_gripper().to_homogeneous() - pose.to_homogeneous())
                .abs()
                .max()
                < 1e-15
        );
        let exact: RobotPose =
            serde_json::from_str(&serde_json::to_string(&RobotPose::exact(pose)).unwrap()).unwrap();
        assert!(exact.covariance().is_none());
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["covariance"][0][0] = serde_json::json!(-1.0);
        assert!(serde_json::from_value::<RobotPose>(value).is_err());
        assert_eq!(
            serde_json::to_string(&RobotFrame::Gripper).unwrap(),
            "\"gripper\""
        );
    }
}
