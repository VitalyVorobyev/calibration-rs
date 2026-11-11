//! The hand-eye a rig export carries, with the target pose of its mode.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{Handeye, Iso3};
use vision_calibration_optim::HandeyeMode;

/// The solution of a rig hand-eye calibration: the hand-eye of its mode and
/// the pose of the static target in the frame that mode fixes it in.
///
/// Each mode holds exactly the two poses it estimates, so an export with the
/// wrong pose for its mode cannot be written or read. In JSON it is
/// internally tagged by `mode`:
///
/// ```json
/// {"mode": "eye_in_hand", "gripper_se3_rig": {…}, "base_se3_target": {…}}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum HandeyeSolution {
    /// The rig rides on the gripper; the target stands in the base frame.
    EyeInHand {
        /// The rig's pose on the gripper (`gripper_se3_rig`, rig frame →
        /// gripper frame).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        gripper_se3_rig: Iso3,
        /// The target's pose in the robot's base frame (`base_se3_target`).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        base_se3_target: Iso3,
    },
    /// The rig stands in the base frame; the target rides on the gripper.
    EyeToHand {
        /// The base's pose in the rig (`rig_se3_base`, base frame → rig
        /// frame).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        rig_se3_base: Iso3,
        /// The target's pose on the gripper (`gripper_se3_target`).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        gripper_se3_target: Iso3,
    },
}

impl HandeyeSolution {
    /// The solution of `mode` from its hand-eye pose (`gripper_se3_rig`
    /// eye-in-hand, `rig_se3_base` eye-to-hand) and its target pose
    /// (`base_se3_target`, `gripper_se3_target`).
    pub fn new(mode: HandeyeMode, handeye: Iso3, target: Iso3) -> Self {
        match mode {
            HandeyeMode::EyeInHand => Self::EyeInHand {
                gripper_se3_rig: handeye,
                base_se3_target: target,
            },
            HandeyeMode::EyeToHand => Self::EyeToHand {
                rig_se3_base: handeye,
                gripper_se3_target: target,
            },
        }
    }

    /// The mode the solution is in.
    pub fn mode(&self) -> HandeyeMode {
        match self {
            Self::EyeInHand { .. } => HandeyeMode::EyeInHand,
            Self::EyeToHand { .. } => HandeyeMode::EyeToHand,
        }
    }

    /// The rig's hand-eye.
    pub fn handeye(&self) -> Handeye {
        match *self {
            Self::EyeInHand {
                gripper_se3_rig, ..
            } => Handeye::EyeInHand { gripper_se3_rig },
            Self::EyeToHand { rig_se3_base, .. } => Handeye::EyeToHand { rig_se3_base },
        }
    }

    /// The target's pose of the mode: `base_se3_target` eye-in-hand,
    /// `gripper_se3_target` eye-to-hand.
    pub fn target_pose(&self) -> &Iso3 {
        match self {
            Self::EyeInHand {
                base_se3_target, ..
            } => base_se3_target,
            Self::EyeToHand {
                gripper_se3_target, ..
            } => gripper_se3_target,
        }
    }
}
