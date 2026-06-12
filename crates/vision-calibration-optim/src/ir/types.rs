use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Identifier for a parameter block in the IR.
///
/// This is stable within a `ProblemIR` instance and is used by residual blocks
/// to reference their parameter dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamId(pub usize);

/// Supported manifold types for parameter blocks.
///
/// Each variant implies an expected ambient parameter dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldKind {
    /// Standard Euclidean vector space.
    Euclidean,
    /// SE(3) pose stored as `[qx, qy, qz, qw, tx, ty, tz]`.
    SE3,
    /// SO(3) rotation stored as quaternion `[qx, qy, qz, qw]`.
    SO3,
    /// S2 unit sphere stored as `[x, y, z]`.
    S2,
}

impl ManifoldKind {
    /// Returns `true` if the given ambient dimension matches the manifold storage.
    pub fn compatible_dim(self, dim: usize) -> bool {
        match self {
            ManifoldKind::Euclidean => true,
            ManifoldKind::SE3 => dim == 7,
            ManifoldKind::SO3 => dim == 4,
            ManifoldKind::S2 => dim == 3,
        }
    }
}

/// Bounds for a single parameter index.
///
/// Bounds are applied after each update in backends that support them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    pub idx: usize,
    pub lower: f64,
    pub upper: f64,
}

/// Fixed parameter mask for a block.
///
/// Backends interpret this as per-index fixing for Euclidean blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedMask {
    fixed_indices: HashSet<usize>,
}

impl FixedMask {
    /// Creates a mask with no fixed indices.
    pub fn all_free() -> Self {
        Self {
            fixed_indices: HashSet::new(),
        }
    }

    /// Creates a mask with all indices fixed.
    pub fn all_fixed(dim: usize) -> Self {
        Self {
            fixed_indices: (0..dim).collect(),
        }
    }

    /// Creates a mask from an explicit list of indices.
    pub fn fix_indices(indices: &[usize]) -> Self {
        Self {
            fixed_indices: indices.iter().copied().collect(),
        }
    }

    /// Returns `true` if the index is fixed.
    pub fn is_fixed(&self, idx: usize) -> bool {
        self.fixed_indices.contains(&idx)
    }

    /// Returns `true` if all indices `[0, dim)` are fixed.
    pub fn is_all_fixed(&self, dim: usize) -> bool {
        self.fixed_indices.len() == dim
    }

    /// Iterates over fixed indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.fixed_indices.iter().copied()
    }

    /// Returns `true` if no indices are fixed.
    pub fn is_empty(&self) -> bool {
        self.fixed_indices.is_empty()
    }
}

/// Robust loss applied to a residual block.
///
/// Each residual block has its own loss; per-point robustification is achieved
/// by using one residual block per observation.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum RobustLoss {
    /// No robustification (plain squared residual).
    #[default]
    None,
    /// Huber loss with transition scale.
    Huber {
        /// Scale parameter controlling quadratic-to-linear transition.
        scale: f64,
    },
    /// Cauchy loss with scale parameter.
    Cauchy {
        /// Scale parameter controlling outlier down-weighting.
        scale: f64,
    },
    /// Arctangent loss with bounded influence.
    Arctan {
        /// Scale parameter controlling curvature.
        scale: f64,
    },
}

/// Hand-eye calibration mode.
///
/// Specifies the transform chain used for hand-eye calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub enum HandEyeMode {
    /// Camera mounted on robot end-effector (gripper).
    ///
    /// Transform chain: P_camera = extr^-1 * handeye^-1 * robot^-1 * target * P_world
    EyeInHand,
    /// Camera fixed in workspace, observes robot end-effector.
    ///
    /// Transform chain: P_camera = extr^-1 * handeye * robot * target * P_world
    EyeToHand,
}

/// Projection slot of a [`CameraModelDesc`].
///
/// Determines the dimension of the intrinsics parameter block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    /// Pinhole projection; intrinsics `[fx, fy, cx, cy]`.
    Pinhole,
}

impl ProjectionKind {
    /// Dimension of the intrinsics parameter block for this projection.
    pub fn intrinsics_dim(self) -> usize {
        match self {
            ProjectionKind::Pinhole => 4,
        }
    }
}

/// Distortion slot of a [`CameraModelDesc`].
///
/// `dim() == 0` means the factor takes no distortion parameter block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionKind {
    /// No distortion; no parameter block.
    None,
    /// Brown-Conrady `[k1, k2, k3, p1, p2]`.
    BrownConrady5,
}

impl DistortionKind {
    /// Dimension of the distortion parameter block (0 = no block).
    pub fn dim(self) -> usize {
        match self {
            DistortionKind::None => 0,
            DistortionKind::BrownConrady5 => 5,
        }
    }
}

/// Sensor slot of a [`CameraModelDesc`].
///
/// `dim() == 0` means the factor takes no sensor parameter block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorKind {
    /// Identity sensor; no parameter block.
    None,
    /// Scheimpflug tilt `[tau_x, tau_y]`.
    Scheimpflug2,
}

impl SensorKind {
    /// Dimension of the sensor parameter block (0 = no block).
    pub fn dim(self) -> usize {
        match self {
            SensorKind::None => 0,
            SensorKind::Scheimpflug2 => 2,
        }
    }
}

/// Camera model carried as data by reprojection and laser factors.
///
/// The descriptor selects which residual kernel a backend compiles and which
/// leading parameter blocks the factor expects: intrinsics always, then a
/// distortion block if `distortion.dim() > 0`, then a sensor block if
/// `sensor.dim() > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraModelDesc {
    /// Projection slot (determines intrinsics dimension).
    pub projection: ProjectionKind,
    /// Distortion slot.
    pub distortion: DistortionKind,
    /// Sensor slot.
    pub sensor: SensorKind,
}

impl CameraModelDesc {
    /// Pinhole, no distortion, identity sensor.
    pub const PINHOLE4: Self = Self {
        projection: ProjectionKind::Pinhole,
        distortion: DistortionKind::None,
        sensor: SensorKind::None,
    };
    /// Pinhole with Brown-Conrady distortion.
    pub const PINHOLE4_DIST5: Self = Self {
        projection: ProjectionKind::Pinhole,
        distortion: DistortionKind::BrownConrady5,
        sensor: SensorKind::None,
    };
    /// Pinhole with Brown-Conrady distortion and a Scheimpflug sensor.
    pub const PINHOLE4_DIST5_SCHEIMPFLUG2: Self = Self {
        projection: ProjectionKind::Pinhole,
        distortion: DistortionKind::BrownConrady5,
        sensor: SensorKind::Scheimpflug2,
    };

    /// Number of leading camera parameter blocks implied by the descriptor.
    pub fn num_cam_blocks(self) -> usize {
        1 + usize::from(self.distortion.dim() > 0) + usize::from(self.sensor.dim() > 0)
    }

    /// Expected camera parameter slots, in IR order.
    fn param_slots(self) -> Vec<ParamSlotSpec> {
        let mut slots = vec![ParamSlotSpec {
            dim: self.projection.intrinsics_dim(),
            manifold: ManifoldKind::Euclidean,
            role: "intrinsics",
        }];
        if self.distortion.dim() > 0 {
            slots.push(ParamSlotSpec {
                dim: self.distortion.dim(),
                manifold: ManifoldKind::Euclidean,
                role: "distortion",
            });
        }
        if self.sensor.dim() > 0 {
            slots.push(ParamSlotSpec {
                dim: self.sensor.dim(),
                manifold: ManifoldKind::Euclidean,
                role: "sensor",
            });
        }
        slots
    }
}

/// Pose chain of a [`FactorKind::ReprojPoint`] factor, carried as data.
///
/// The chain selects how the target-frame point reaches the camera frame and
/// which parameter blocks follow the camera blocks. Per-view measured robot
/// poses are chain data, not parameter blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReprojChain {
    /// Blocks: `[camera_se3_target]`. `P_camera = pose * P_target`.
    SinglePose,
    /// Blocks: `[extrinsics, pose]` (rig extrinsics).
    /// `P_camera = extr^-1 * pose * P_target`.
    TwoSe3,
    /// Blocks: `[extrinsics, handeye, target]` (hand-eye).
    ///
    /// The measured robot pose enters as data; the transform chain depends on
    /// [`HandEyeMode`].
    HandEye {
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_se3_gripper: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
    },
    /// Blocks: `[extrinsics, handeye, target, robot_delta]` (hand-eye with a
    /// per-view robot pose correction).
    ///
    /// `robot_delta` is a 6D se(3) tangent correction applied as
    /// `exp(delta) * T_B_G`.
    HandEyeRobotDelta {
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_se3_gripper: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
    },
}

impl ReprojChain {
    /// Expected chain parameter slots, in IR order (after the camera blocks).
    fn param_slots(self) -> Vec<ParamSlotSpec> {
        let se3 = |role| ParamSlotSpec {
            dim: 7,
            manifold: ManifoldKind::SE3,
            role,
        };
        match self {
            ReprojChain::SinglePose => vec![se3("camera_se3_target")],
            ReprojChain::TwoSe3 => vec![se3("extrinsics"), se3("pose")],
            ReprojChain::HandEye { .. } => {
                vec![se3("extrinsics"), se3("handeye"), se3("target")]
            }
            ReprojChain::HandEyeRobotDelta { .. } => vec![
                se3("extrinsics"),
                se3("handeye"),
                se3("target"),
                ParamSlotSpec {
                    dim: 6,
                    manifold: ManifoldKind::Euclidean,
                    role: "robot_delta",
                },
            ],
        }
    }
}

/// Pose chain of a laser factor ([`FactorKind::LaserPointToPlane`] /
/// [`FactorKind::LaserLineDistance`]), carried as data.
///
/// Every laser chain ends with the laser-plane blocks
/// `[plane_normal (S2), plane_distance (1D)]`; the robot-delta correction, if
/// present, comes last.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaserChain {
    /// Blocks: `[camera_se3_target, plane_normal, plane_distance]`.
    SinglePose,
    /// Blocks: `[cam_se3_rig, handeye, target_ref, plane_normal,
    /// plane_distance]` (rig + hand-eye).
    ///
    /// The target pose in the camera frame is composed from `cam_se3_rig`,
    /// `handeye`, `target_ref`, and the measured robot pose according to
    /// [`HandEyeMode`].
    RigHandEye {
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_se3_gripper: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
    },
    /// Blocks: `[cam_se3_rig, handeye, target_ref, plane_normal,
    /// plane_distance, robot_delta]`.
    ///
    /// `robot_delta` is a 6D se(3) tangent correction applied as
    /// `exp(delta) * T_B_G`.
    RigHandEyeRobotDelta {
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_se3_gripper: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
    },
}

impl LaserChain {
    /// Expected chain parameter slots, in IR order (after the camera blocks).
    fn param_slots(self) -> Vec<ParamSlotSpec> {
        let se3 = |role| ParamSlotSpec {
            dim: 7,
            manifold: ManifoldKind::SE3,
            role,
        };
        let plane = [
            ParamSlotSpec {
                dim: 3,
                manifold: ManifoldKind::S2,
                role: "plane_normal",
            },
            ParamSlotSpec {
                dim: 1,
                manifold: ManifoldKind::Euclidean,
                role: "plane_distance",
            },
        ];
        let robot_delta = ParamSlotSpec {
            dim: 6,
            manifold: ManifoldKind::Euclidean,
            role: "robot_delta",
        };
        match self {
            LaserChain::SinglePose => {
                let mut v = vec![se3("camera_se3_target")];
                v.extend(plane);
                v
            }
            LaserChain::RigHandEye { .. } => {
                let mut v = vec![se3("cam_se3_rig"), se3("handeye"), se3("target_ref")];
                v.extend(plane);
                v
            }
            LaserChain::RigHandEyeRobotDelta { .. } => {
                let mut v = vec![se3("cam_se3_rig"), se3("handeye"), se3("target_ref")];
                v.extend(plane);
                v.push(robot_delta);
                v
            }
        }
    }
}

/// One expected parameter slot of a factor: dimension, manifold, and a
/// diagnostic role name used in validation error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamSlotSpec {
    /// Exact ambient dimension expected for the block.
    pub dim: usize,
    /// Manifold expected for the block.
    pub manifold: ManifoldKind,
    /// Diagnostic role name (e.g. `"intrinsics"`, `"plane_normal"`).
    pub role: &'static str,
}

/// Backend-agnostic factor kinds.
///
/// Each factor kind implies its parameter layout and residual dimension.
#[derive(Debug, Clone, PartialEq)]
pub enum FactorKind {
    /// Reprojection residual for a pinhole camera with 4 intrinsics and an SE3 pose.
    ReprojPointPinhole4 {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Reprojection residual with pinhole intrinsics, Brown-Conrady distortion, and SE3 pose.
    ReprojPointPinhole4Dist5 {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Reprojection with pinhole, distortion, Scheimpflug sensor, and SE3 pose.
    ReprojPointPinhole4Dist5Scheimpflug2 {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Reprojection with two composed SE3 transforms for rig extrinsics.
    ///
    /// Parameters: [intrinsics, distortion, extr_camera_to_rig, pose_target_to_rig]
    /// Transform chain: P_camera = extr^-1 * pose * P_world
    ReprojPointPinhole4Dist5TwoSE3 {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Reprojection for hand-eye calibration with robot pose as measurement.
    ///
    /// Parameters: [intrinsics, distortion, extr, handeye, target]
    /// Robot pose (`base_se3_gripper`, gripper in base frame) is stored in the factor as known data.
    ReprojPointPinhole4Dist5HandEye {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_to_gripper_se3: [f64; 7],
        /// Hand-eye mode defining transform chain.
        mode: HandEyeMode,
    },
    /// Reprojection for hand-eye calibration with per-view robot pose correction.
    ///
    /// Parameters: [intrinsics, distortion, extr, handeye, target, robot_delta]
    /// Robot pose (`base_se3_gripper`) is stored in the factor as known data.
    /// `robot_delta` is a 6D se(3) tangent correction applied via exp(delta) * T_B_E.
    ReprojPointPinhole4Dist5HandEyeRobotDelta {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_to_gripper_se3: [f64; 7],
        /// Hand-eye mode defining transform chain.
        mode: HandEyeMode,
    },
    /// Reprojection with two composed SE3 transforms and a Scheimpflug sensor (rig extrinsics).
    ///
    /// Parameters: [intrinsics, distortion, sensor, extr_camera_to_rig, pose_target_to_rig]
    /// Transform chain: P_camera = extr^-1 * pose * P_world, then Scheimpflug projection.
    ReprojPointPinhole4Dist5Scheimpflug2TwoSE3 {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Reprojection for hand-eye calibration with a Scheimpflug sensor.
    ///
    /// Parameters: [intrinsics, distortion, sensor, extr, handeye, target]
    /// Robot pose (`base_se3_gripper`) is stored in the factor as known data.
    ReprojPointPinhole4Dist5Scheimpflug2HandEye {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_to_gripper_se3: [f64; 7],
        /// Hand-eye mode defining transform chain.
        mode: HandEyeMode,
    },
    /// Reprojection for hand-eye calibration with per-view robot pose correction and a Scheimpflug sensor.
    ///
    /// Parameters: [intrinsics, distortion, sensor, extr, handeye, target, robot_delta]
    /// Robot pose (`base_se3_gripper`) is stored in the factor as known data.
    /// `robot_delta` is a 6D se(3) tangent correction applied via exp(delta) * T_B_E.
    ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta {
        /// World/target point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
        /// Measured robot pose `T_B_G` packed as `[qx,qy,qz,qw,tx,ty,tz]`.
        base_to_gripper_se3: [f64; 7],
        /// Hand-eye mode defining transform chain.
        mode: HandEyeMode,
    },
    /// Laser line pixel constrained to lie on laser plane.
    ///
    /// Parameters: [intrinsics, distortion, sensor, pose_cam_to_target, plane_normal, plane_distance]
    /// Residual: point-to-plane distance for ray-target intersection point.
    /// Note: Target is always planar (Z=0), so 3D point is computed as ray intersection.
    LaserPlanePixel {
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Laser line pixel constrained by line-distance in normalized plane.
    ///
    /// Parameters: [intrinsics, distortion, sensor, pose_cam_to_target, plane_normal, plane_distance]
    /// Residual: perpendicular distance from normalized pixel to projected
    ///           laser-target intersection line, scaled by sqrt(fx*fy).
    /// Note: Alternative to LaserPlanePixel using normalized plane geometry.
    LaserLineDist2D {
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Rig + hand-eye version of [`FactorKind::LaserPlanePixel`].
    ///
    /// Parameters: [intrinsics, distortion, sensor, cam_to_rig, handeye,
    /// target_ref, plane_normal, plane_distance].
    ///
    /// The target pose in camera frame is composed from `cam_to_rig`,
    /// `handeye`, `target_ref`, and the per-view robot pose according to
    /// [`HandEyeMode`] — so intrinsics, rig extrinsics, hand-eye transform,
    /// target reference pose and the laser plane all move simultaneously
    /// under a single point-to-plane cost.
    ///
    /// Residual: 1D signed distance in meters (same units as
    /// [`FactorKind::LaserPlanePixel`]).
    LaserPlanePixelRigHandEye {
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Known robot pose (base-to-gripper) as SE3 `[qx, qy, qz, qw, tx, ty, tz]`.
        robot_se3: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
        /// Residual weight.
        w: f64,
    },
    /// Rig + hand-eye point-to-plane laser residual with a per-view robot pose correction.
    ///
    /// Parameters: [intrinsics, distortion, sensor, cam_to_rig, handeye,
    /// target_ref, plane_normal, plane_distance, robot_delta].
    ///
    /// `robot_delta` is a 6D se(3) tangent correction applied as
    /// `exp(delta) * T_B_G`, matching
    /// [`FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta`].
    LaserPlanePixelRigHandEyeRobotDelta {
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Known robot pose (base-to-gripper) as SE3 `[qx, qy, qz, qw, tx, ty, tz]`.
        robot_se3: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
        /// Residual weight.
        w: f64,
    },
    /// Rig + hand-eye version of [`FactorKind::LaserLineDist2D`].
    ///
    /// Parameters: [intrinsics, distortion, sensor, cam_to_rig, handeye,
    /// target_ref, plane_normal, plane_distance].
    ///
    /// Same chain composition as [`FactorKind::LaserPlanePixelRigHandEye`],
    /// but measures perpendicular pixel distance between the undistorted
    /// laser pixel and the projected laser-target intersection line. This is
    /// the undistorted-image-space formulation described in the design plan.
    ///
    /// Residual: 1D distance in pixels (same units as
    /// [`FactorKind::LaserLineDist2D`]).
    LaserLineDist2DRigHandEye {
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Known robot pose (base-to-gripper) as SE3 `[qx, qy, qz, qw, tx, ty, tz]`.
        robot_se3: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
        /// Residual weight.
        w: f64,
    },
    /// Rig + hand-eye line-distance laser residual with a per-view robot pose correction.
    ///
    /// Parameters: [intrinsics, distortion, sensor, cam_to_rig, handeye,
    /// target_ref, plane_normal, plane_distance, robot_delta].
    ///
    /// `robot_delta` is a 6D se(3) tangent correction applied as
    /// `exp(delta) * T_B_G`, matching
    /// [`FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta`].
    LaserLineDist2DRigHandEyeRobotDelta {
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Known robot pose (base-to-gripper) as SE3 `[qx, qy, qz, qw, tx, ty, tz]`.
        robot_se3: [f64; 7],
        /// Hand-eye mode defining the transform chain.
        mode: HandEyeMode,
        /// Residual weight.
        w: f64,
    },
    /// Reprojection residual generic over camera model and pose chain.
    ///
    /// Parameters: the camera blocks implied by `model` (intrinsics,
    /// then distortion / sensor blocks if their dimensions are non-zero),
    /// followed by the chain blocks implied by `chain`.
    ///
    /// Residual: 2D pixel error, scaled by `w`.
    ReprojPoint {
        /// Camera model descriptor (selects the projection kernel and the
        /// leading parameter blocks).
        model: CameraModelDesc,
        /// Pose chain (selects the trailing parameter blocks and how the
        /// point reaches the camera frame).
        chain: ReprojChain,
        /// Target-frame point coordinates.
        pw: [f64; 3],
        /// Observed pixel coordinates.
        uv: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Point-to-plane laser residual generic over camera model and chain.
    ///
    /// The observed laser pixel is back-projected through the camera model,
    /// intersected with the planar target (Z=0 in the target frame), and the
    /// resulting 3D point is constrained to the laser plane.
    ///
    /// Parameters: the camera blocks implied by `model`, then the chain
    /// blocks implied by `chain` (which include the laser-plane blocks).
    ///
    /// Residual: 1D signed distance in meters, scaled by `w`.
    LaserPointToPlane {
        /// Camera model descriptor.
        model: CameraModelDesc,
        /// Pose chain including the laser-plane blocks.
        chain: LaserChain,
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Line-distance laser residual generic over camera model and chain.
    ///
    /// Measures the perpendicular distance in the normalized image plane
    /// between the undistorted laser pixel and the projected laser-target
    /// intersection line, scaled by `sqrt(fx*fy)` to pixel units.
    ///
    /// Parameters: identical layout to [`FactorKind::LaserPointToPlane`].
    ///
    /// Residual: 1D distance in pixels, scaled by `w`.
    LaserLineDistance {
        /// Camera model descriptor.
        model: CameraModelDesc,
        /// Pose chain including the laser-plane blocks.
        chain: LaserChain,
        /// Observed laser pixel coordinates.
        laser_pixel: [f64; 2],
        /// Residual weight.
        w: f64,
    },
    /// Placeholder for future prior factors.
    Prior,
    /// Zero-mean prior on a 6D se(3) tangent vector.
    ///
    /// Parameters: \[se3_delta\] (6D Euclidean).
    Se3TangentPrior {
        /// Diagonal square-root information for `[rx, ry, rz, tx, ty, tz]`.
        sqrt_info: [f64; 6],
    },
    /// Placeholder for future distortion-aware reprojection.
    ReprojPointWithDistortion,
}

impl FactorKind {
    /// Residual dimension implied by the factor.
    pub fn residual_dim(&self) -> usize {
        match self {
            FactorKind::ReprojPointPinhole4 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5TwoSE3 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5HandEye { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5HandEyeRobotDelta { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5Scheimpflug2TwoSE3 { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEye { .. } => 2,
            FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta { .. } => 2,
            FactorKind::LaserPlanePixel { .. } => 1,
            FactorKind::LaserLineDist2D { .. } => 1,
            FactorKind::LaserPlanePixelRigHandEye { .. } => 1,
            FactorKind::LaserPlanePixelRigHandEyeRobotDelta { .. } => 1,
            FactorKind::LaserLineDist2DRigHandEye { .. } => 1,
            FactorKind::LaserLineDist2DRigHandEyeRobotDelta { .. } => 1,
            FactorKind::ReprojPoint { .. } => 2,
            FactorKind::LaserPointToPlane { .. } | FactorKind::LaserLineDistance { .. } => 1,
            FactorKind::Prior => 0,
            FactorKind::Se3TangentPrior { .. } => 6,
            FactorKind::ReprojPointWithDistortion => 2,
        }
    }

    /// Short factor name used in validation error messages.
    pub fn name(&self) -> &'static str {
        match self {
            FactorKind::ReprojPoint { .. } => "ReprojPoint",
            FactorKind::LaserPointToPlane { .. } => "LaserPointToPlane",
            FactorKind::LaserLineDistance { .. } => "LaserLineDistance",
            FactorKind::Se3TangentPrior { .. } => "Se3TangentPrior",
            _ => "legacy",
        }
    }

    /// Full expected parameter layout for descriptor-based factors, in IR
    /// order: camera blocks first, then chain blocks.
    ///
    /// Returns `None` for factor kinds whose layout is not descriptor-driven.
    pub fn param_layout(&self) -> Option<Vec<ParamSlotSpec>> {
        match self {
            FactorKind::ReprojPoint { model, chain, .. } => {
                let mut slots = model.param_slots();
                slots.extend(chain.param_slots());
                Some(slots)
            }
            FactorKind::LaserPointToPlane { model, chain, .. }
            | FactorKind::LaserLineDistance { model, chain, .. } => {
                let mut slots = model.param_slots();
                slots.extend(chain.param_slots());
                Some(slots)
            }
            FactorKind::Se3TangentPrior { .. } => Some(vec![ParamSlotSpec {
                dim: 6,
                manifold: ManifoldKind::Euclidean,
                role: "robot_delta",
            }]),
            _ => None,
        }
    }
}

/// Parameter block definition in the IR.
///
/// This describes the storage layout and constraints for a single variable.
#[derive(Debug, Clone)]
pub struct ParamBlock {
    /// Stable parameter block ID within this IR.
    pub id: ParamId,
    /// Human-readable parameter block name.
    pub name: String,
    /// Ambient parameter dimension.
    pub dim: usize,
    /// Manifold type for updates/projection.
    pub manifold: ManifoldKind,
    /// Per-index fixed mask.
    pub fixed: FixedMask,
    /// Optional per-index bounds.
    pub bounds: Option<Vec<Bound>>,
}

/// Residual block definition in the IR.
///
/// The order of `params` must match the factor's expected parameter order.
#[derive(Debug, Clone)]
pub struct ResidualBlock {
    /// Parameter block IDs used by this residual (factor-dependent order).
    pub params: Vec<ParamId>,
    /// Robust loss applied to this residual block.
    pub loss: RobustLoss,
    /// Residual factor model.
    pub factor: FactorKind,
    /// Residual vector dimension.
    pub residual_dim: usize,
}

/// Backend-agnostic optimization problem representation.
///
/// Backends compile this IR into solver-specific problems.
#[derive(Debug, Default, Clone)]
pub struct ProblemIR {
    /// Parameter blocks in this optimization problem.
    pub params: Vec<ParamBlock>,
    /// Residual blocks in this optimization problem.
    pub residuals: Vec<ResidualBlock>,
}

impl ProblemIR {
    /// Creates an empty IR.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a parameter block and returns its `ParamId`.
    pub fn add_param_block(
        &mut self,
        name: impl Into<String>,
        dim: usize,
        manifold: ManifoldKind,
        fixed: FixedMask,
        bounds: Option<Vec<Bound>>,
    ) -> ParamId {
        let id = ParamId(self.params.len());
        self.params.push(ParamBlock {
            id,
            name: name.into(),
            dim,
            manifold,
            fixed,
            bounds,
        });
        id
    }

    /// Adds a residual block to the IR.
    pub fn add_residual_block(&mut self, residual: ResidualBlock) {
        self.residuals.push(residual);
    }

    /// Finds a parameter by name.
    pub fn param_by_name(&self, name: &str) -> Option<ParamId> {
        self.params.iter().find(|p| p.name == name).map(|p| p.id)
    }

    /// Validates internal consistency and factor expectations.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Numerical`] (converted from the internal
    /// structural error) if the IR has inconsistent parameter indices,
    /// malformed factor blocks, or residual-dimension mismatches.
    pub fn validate(&self) -> std::result::Result<(), crate::Error> {
        self.validate_inner()?;
        Ok(())
    }

    fn validate_inner(&self) -> Result<()> {
        for (idx, param) in self.params.iter().enumerate() {
            ensure!(
                param.id.0 == idx,
                "param id mismatch: expected {}, got {:?}",
                idx,
                param.id
            );
            ensure!(
                param.manifold.compatible_dim(param.dim),
                "param {} manifold {:?} incompatible with dim {}",
                param.name,
                param.manifold,
                param.dim
            );
            for fixed_idx in param.fixed.iter() {
                ensure!(
                    fixed_idx < param.dim,
                    "param {} fixed index {} out of range",
                    param.name,
                    fixed_idx
                );
            }
            if let Some(bounds) = &param.bounds {
                for bound in bounds {
                    ensure!(
                        bound.idx < param.dim,
                        "param {} bound index {} out of range",
                        param.name,
                        bound.idx
                    );
                    ensure!(
                        bound.lower <= bound.upper,
                        "param {} bound lower {} > upper {}",
                        param.name,
                        bound.lower,
                        bound.upper
                    );
                }
            }
        }

        for (r_idx, residual) in self.residuals.iter().enumerate() {
            ensure!(
                residual.residual_dim == residual.factor.residual_dim(),
                "residual {} dim {} does not match factor expectation {}",
                r_idx,
                residual.residual_dim,
                residual.factor.residual_dim()
            );
            for param in &residual.params {
                ensure!(
                    param.0 < self.params.len(),
                    "residual {} references missing param {:?}",
                    r_idx,
                    param
                );
            }

            match &residual.factor {
                factor @ (FactorKind::ReprojPoint { .. }
                | FactorKind::LaserPointToPlane { .. }
                | FactorKind::LaserLineDistance { .. }
                | FactorKind::Se3TangentPrior { .. }) => {
                    let layout = factor
                        .param_layout()
                        .expect("descriptor-based factor has a parameter layout");
                    ensure!(
                        residual.params.len() == layout.len(),
                        "{} factor requires {} params [{}], got {}",
                        factor.name(),
                        layout.len(),
                        layout.iter().map(|s| s.role).collect::<Vec<_>>().join(", "),
                        residual.params.len()
                    );
                    for (slot, pid) in layout.iter().zip(&residual.params) {
                        let block = &self.params[pid.0];
                        ensure!(
                            block.dim == slot.dim && block.manifold == slot.manifold,
                            "{} factor expects {}D {:?} {}, got dim={} manifold={:?}",
                            factor.name(),
                            slot.dim,
                            slot.manifold,
                            slot.role,
                            block.dim,
                            block.manifold
                        );
                    }
                }
                FactorKind::ReprojPointPinhole4 { .. } => {
                    ensure!(
                        residual.params.len() == 2,
                        "reprojection factor requires 2 params"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let pose = &self.params[residual.params[1].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "reprojection factor expects 4D Euclidean intrinsics"
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "reprojection factor expects 7D SE3 pose"
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5 { .. } => {
                    ensure!(
                        residual.params.len() == 3,
                        "distortion reprojection factor requires 3 params [cam, dist, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let pose = &self.params[residual.params[2].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "distortion reprojection expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "distortion reprojection expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "distortion reprojection expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 { .. } => {
                    ensure!(
                        residual.params.len() == 4,
                        "Scheimpflug distortion reprojection factor requires 4 params [cam, dist, sensor, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let pose = &self.params[residual.params[3].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug reprojection expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug reprojection expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug reprojection expects 2D Euclidean sensor, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "Scheimpflug reprojection expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5TwoSE3 { .. } => {
                    ensure!(
                        residual.params.len() == 4,
                        "TwoSE3 factor requires 4 params [cam, dist, extr, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let extr = &self.params[residual.params[2].0];
                    let pose = &self.params[residual.params[3].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "TwoSE3 factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "TwoSE3 factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "TwoSE3 factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "TwoSE3 factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5HandEye { .. } => {
                    ensure!(
                        residual.params.len() == 5,
                        "HandEye factor requires 5 params [cam, dist, extr, handeye, target]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let extr = &self.params[residual.params[2].0];
                    let handeye = &self.params[residual.params[3].0];
                    let target = &self.params[residual.params[4].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 hand-eye transform, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target.dim == 7 && target.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 target pose, got dim={} manifold={:?}",
                        target.dim,
                        target.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5HandEyeRobotDelta { .. } => {
                    ensure!(
                        residual.params.len() == 6,
                        "HandEye factor requires 6 params [cam, dist, extr, handeye, target, robot_delta]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let extr = &self.params[residual.params[2].0];
                    let handeye = &self.params[residual.params[3].0];
                    let target = &self.params[residual.params[4].0];
                    let robot_delta = &self.params[residual.params[5].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 hand-eye transform, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target.dim == 7 && target.manifold == ManifoldKind::SE3,
                        "HandEye factor expects 7D SE3 target pose, got dim={} manifold={:?}",
                        target.dim,
                        target.manifold
                    );
                    ensure!(
                        robot_delta.dim == 6 && robot_delta.manifold == ManifoldKind::Euclidean,
                        "HandEye factor expects 6D Euclidean robot delta, got dim={} manifold={:?}",
                        robot_delta.dim,
                        robot_delta.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5Scheimpflug2TwoSE3 { .. } => {
                    ensure!(
                        residual.params.len() == 5,
                        "Scheimpflug TwoSE3 factor requires 5 params [cam, dist, sensor, extr, pose]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let extr = &self.params[residual.params[3].0];
                    let pose = &self.params[residual.params[4].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug TwoSE3 factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug TwoSE3 factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug TwoSE3 factor expects 2D Euclidean sensor, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "Scheimpflug TwoSE3 factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "Scheimpflug TwoSE3 factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEye { .. } => {
                    ensure!(
                        residual.params.len() == 6,
                        "Scheimpflug HandEye factor requires 6 params [cam, dist, sensor, extr, handeye, target]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let extr = &self.params[residual.params[3].0];
                    let handeye = &self.params[residual.params[4].0];
                    let target = &self.params[residual.params[5].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEye factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEye factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEye factor expects 2D Euclidean sensor, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "Scheimpflug HandEye factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "Scheimpflug HandEye factor expects 7D SE3 hand-eye transform, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target.dim == 7 && target.manifold == ManifoldKind::SE3,
                        "Scheimpflug HandEye factor expects 7D SE3 target pose, got dim={} manifold={:?}",
                        target.dim,
                        target.manifold
                    );
                }
                FactorKind::ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta { .. } => {
                    ensure!(
                        residual.params.len() == 7,
                        "Scheimpflug HandEyeRobotDelta factor requires 7 params [cam, dist, sensor, extr, handeye, target, robot_delta]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let extr = &self.params[residual.params[3].0];
                    let handeye = &self.params[residual.params[4].0];
                    let target = &self.params[residual.params[5].0];
                    let robot_delta = &self.params[residual.params[6].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEyeRobotDelta factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEyeRobotDelta factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEyeRobotDelta factor expects 2D Euclidean sensor, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        extr.dim == 7 && extr.manifold == ManifoldKind::SE3,
                        "Scheimpflug HandEyeRobotDelta factor expects 7D SE3 extrinsics, got dim={} manifold={:?}",
                        extr.dim,
                        extr.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "Scheimpflug HandEyeRobotDelta factor expects 7D SE3 hand-eye transform, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target.dim == 7 && target.manifold == ManifoldKind::SE3,
                        "Scheimpflug HandEyeRobotDelta factor expects 7D SE3 target pose, got dim={} manifold={:?}",
                        target.dim,
                        target.manifold
                    );
                    ensure!(
                        robot_delta.dim == 6 && robot_delta.manifold == ManifoldKind::Euclidean,
                        "Scheimpflug HandEyeRobotDelta factor expects 6D Euclidean robot delta, got dim={} manifold={:?}",
                        robot_delta.dim,
                        robot_delta.manifold
                    );
                }
                FactorKind::LaserPlanePixel { .. } => {
                    ensure!(
                        residual.params.len() == 6,
                        "LaserPlanePixel factor requires 6 params [cam, dist, sensor, pose, plane_normal, plane_distance]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let pose = &self.params[residual.params[3].0];
                    let plane_normal = &self.params[residual.params[4].0];
                    let plane_distance = &self.params[residual.params[5].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 2D Euclidean sensor params, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "LaserPlanePixel factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                    ensure!(
                        plane_normal.dim == 3 && plane_normal.manifold == ManifoldKind::S2,
                        "LaserPlanePixel factor expects 3D S2 plane normal, got dim={} manifold={:?}",
                        plane_normal.dim,
                        plane_normal.manifold
                    );
                    ensure!(
                        plane_distance.dim == 1
                            && plane_distance.manifold == ManifoldKind::Euclidean,
                        "LaserPlanePixel factor expects 1D Euclidean plane distance, got dim={} manifold={:?}",
                        plane_distance.dim,
                        plane_distance.manifold
                    );
                }
                FactorKind::LaserLineDist2D { .. } => {
                    ensure!(
                        residual.params.len() == 6,
                        "LaserLineDist2D factor requires 6 params [cam, dist, sensor, pose, plane_normal, plane_distance]"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let pose = &self.params[residual.params[3].0];
                    let plane_normal = &self.params[residual.params[4].0];
                    let plane_distance = &self.params[residual.params[5].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 2D Euclidean sensor params, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        pose.dim == 7 && pose.manifold == ManifoldKind::SE3,
                        "LaserLineDist2D factor expects 7D SE3 pose, got dim={} manifold={:?}",
                        pose.dim,
                        pose.manifold
                    );
                    ensure!(
                        plane_normal.dim == 3 && plane_normal.manifold == ManifoldKind::S2,
                        "LaserLineDist2D factor expects 3D S2 plane normal, got dim={} manifold={:?}",
                        plane_normal.dim,
                        plane_normal.manifold
                    );
                    ensure!(
                        plane_distance.dim == 1
                            && plane_distance.manifold == ManifoldKind::Euclidean,
                        "LaserLineDist2D factor expects 1D Euclidean plane distance, got dim={} manifold={:?}",
                        plane_distance.dim,
                        plane_distance.manifold
                    );
                }
                FactorKind::LaserPlanePixelRigHandEye { .. }
                | FactorKind::LaserLineDist2DRigHandEye { .. }
                | FactorKind::LaserPlanePixelRigHandEyeRobotDelta { .. }
                | FactorKind::LaserLineDist2DRigHandEyeRobotDelta { .. } => {
                    let label = match residual.factor {
                        FactorKind::LaserPlanePixelRigHandEye { .. } => "LaserPlanePixelRigHandEye",
                        FactorKind::LaserLineDist2DRigHandEye { .. } => "LaserLineDist2DRigHandEye",
                        FactorKind::LaserPlanePixelRigHandEyeRobotDelta { .. } => {
                            "LaserPlanePixelRigHandEyeRobotDelta"
                        }
                        _ => "LaserLineDist2DRigHandEyeRobotDelta",
                    };
                    let has_robot_delta = matches!(
                        residual.factor,
                        FactorKind::LaserPlanePixelRigHandEyeRobotDelta { .. }
                            | FactorKind::LaserLineDist2DRigHandEyeRobotDelta { .. }
                    );
                    let expected_len = if has_robot_delta { 9 } else { 8 };
                    let expected_desc = if has_robot_delta {
                        "[cam, dist, sensor, cam_to_rig, handeye, target_ref, plane_normal, plane_distance, robot_delta]"
                    } else {
                        "[cam, dist, sensor, cam_to_rig, handeye, target_ref, plane_normal, plane_distance]"
                    };
                    ensure!(
                        residual.params.len() == expected_len,
                        "{label} factor requires {expected_len} params {expected_desc}"
                    );
                    let cam = &self.params[residual.params[0].0];
                    let dist = &self.params[residual.params[1].0];
                    let sensor = &self.params[residual.params[2].0];
                    let cam_to_rig = &self.params[residual.params[3].0];
                    let handeye = &self.params[residual.params[4].0];
                    let target_ref = &self.params[residual.params[5].0];
                    let plane_normal = &self.params[residual.params[6].0];
                    let plane_distance = &self.params[residual.params[7].0];
                    ensure!(
                        cam.dim == 4 && cam.manifold == ManifoldKind::Euclidean,
                        "{label} factor expects 4D Euclidean intrinsics, got dim={} manifold={:?}",
                        cam.dim,
                        cam.manifold
                    );
                    ensure!(
                        dist.dim == 5 && dist.manifold == ManifoldKind::Euclidean,
                        "{label} factor expects 5D Euclidean distortion, got dim={} manifold={:?}",
                        dist.dim,
                        dist.manifold
                    );
                    ensure!(
                        sensor.dim == 2 && sensor.manifold == ManifoldKind::Euclidean,
                        "{label} factor expects 2D Euclidean sensor params, got dim={} manifold={:?}",
                        sensor.dim,
                        sensor.manifold
                    );
                    ensure!(
                        cam_to_rig.dim == 7 && cam_to_rig.manifold == ManifoldKind::SE3,
                        "{label} factor expects 7D SE3 cam_to_rig, got dim={} manifold={:?}",
                        cam_to_rig.dim,
                        cam_to_rig.manifold
                    );
                    ensure!(
                        handeye.dim == 7 && handeye.manifold == ManifoldKind::SE3,
                        "{label} factor expects 7D SE3 handeye, got dim={} manifold={:?}",
                        handeye.dim,
                        handeye.manifold
                    );
                    ensure!(
                        target_ref.dim == 7 && target_ref.manifold == ManifoldKind::SE3,
                        "{label} factor expects 7D SE3 target_ref, got dim={} manifold={:?}",
                        target_ref.dim,
                        target_ref.manifold
                    );
                    ensure!(
                        plane_normal.dim == 3 && plane_normal.manifold == ManifoldKind::S2,
                        "{label} factor expects 3D S2 plane normal, got dim={} manifold={:?}",
                        plane_normal.dim,
                        plane_normal.manifold
                    );
                    ensure!(
                        plane_distance.dim == 1
                            && plane_distance.manifold == ManifoldKind::Euclidean,
                        "{label} factor expects 1D Euclidean plane distance, got dim={} manifold={:?}",
                        plane_distance.dim,
                        plane_distance.manifold
                    );
                    if has_robot_delta {
                        let robot_delta = &self.params[residual.params[8].0];
                        ensure!(
                            robot_delta.dim == 6 && robot_delta.manifold == ManifoldKind::Euclidean,
                            "{label} factor expects 6D Euclidean robot delta, got dim={} manifold={:?}",
                            robot_delta.dim,
                            robot_delta.manifold
                        );
                    }
                }
                FactorKind::Prior | FactorKind::ReprojPointWithDistortion => {
                    return Err(anyhow!(
                        "factor kind {:?} not implemented in validation",
                        residual.factor
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODELS: [CameraModelDesc; 3] = [
        CameraModelDesc::PINHOLE4,
        CameraModelDesc::PINHOLE4_DIST5,
        CameraModelDesc::PINHOLE4_DIST5_SCHEIMPFLUG2,
    ];

    fn reproj_chains() -> Vec<ReprojChain> {
        let robot = [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3];
        vec![
            ReprojChain::SinglePose,
            ReprojChain::TwoSe3,
            ReprojChain::HandEye {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeInHand,
            },
            ReprojChain::HandEyeRobotDelta {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeToHand,
            },
        ]
    }

    fn laser_chains() -> Vec<LaserChain> {
        let robot = [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3];
        vec![
            LaserChain::SinglePose,
            LaserChain::RigHandEye {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeToHand,
            },
            LaserChain::RigHandEyeRobotDelta {
                base_se3_gripper: robot,
                mode: HandEyeMode::EyeToHand,
            },
        ]
    }

    /// Adds the param blocks for `layout` to `ir` and returns their ids.
    fn add_blocks_for_layout(ir: &mut ProblemIR, layout: &[ParamSlotSpec]) -> Vec<ParamId> {
        layout
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                ir.add_param_block(
                    format!("{}_{i}", slot.role),
                    slot.dim,
                    slot.manifold,
                    FixedMask::all_free(),
                    None,
                )
            })
            .collect()
    }

    #[test]
    fn camera_model_desc_block_counts() {
        assert_eq!(CameraModelDesc::PINHOLE4.num_cam_blocks(), 1);
        assert_eq!(CameraModelDesc::PINHOLE4_DIST5.num_cam_blocks(), 2);
        assert_eq!(
            CameraModelDesc::PINHOLE4_DIST5_SCHEIMPFLUG2.num_cam_blocks(),
            3
        );
    }

    #[test]
    fn reproj_layout_dims_cover_all_models_and_chains() {
        for model in MODELS {
            for chain in reproj_chains() {
                let factor = FactorKind::ReprojPoint {
                    model,
                    chain,
                    pw: [0.0; 3],
                    uv: [0.0; 2],
                    w: 1.0,
                };
                let layout = factor.param_layout().unwrap();
                let chain_blocks = match chain {
                    ReprojChain::SinglePose => 1,
                    ReprojChain::TwoSe3 => 2,
                    ReprojChain::HandEye { .. } => 3,
                    ReprojChain::HandEyeRobotDelta { .. } => 4,
                };
                assert_eq!(layout.len(), model.num_cam_blocks() + chain_blocks);
                // Camera slots lead with the intrinsics block.
                assert_eq!(layout[0].role, "intrinsics");
                assert_eq!(layout[0].dim, 4);
                assert_eq!(factor.residual_dim(), 2);
            }
        }
    }

    #[test]
    fn laser_layout_ends_with_plane_then_optional_delta() {
        for model in MODELS {
            for chain in laser_chains() {
                let factor = FactorKind::LaserPointToPlane {
                    model,
                    chain,
                    laser_pixel: [0.0; 2],
                    w: 1.0,
                };
                let layout = factor.param_layout().unwrap();
                let roles: Vec<_> = layout.iter().map(|s| s.role).collect();
                match chain {
                    LaserChain::RigHandEyeRobotDelta { .. } => {
                        assert_eq!(roles[roles.len() - 1], "robot_delta");
                        assert_eq!(roles[roles.len() - 2], "plane_distance");
                        assert_eq!(roles[roles.len() - 3], "plane_normal");
                    }
                    _ => {
                        assert_eq!(roles[roles.len() - 1], "plane_distance");
                        assert_eq!(roles[roles.len() - 2], "plane_normal");
                    }
                }
                assert_eq!(factor.residual_dim(), 1);
            }
        }
    }

    #[test]
    fn validate_accepts_descriptor_factors_for_every_model_chain_combo() {
        for model in MODELS {
            for chain in reproj_chains() {
                let factor = FactorKind::ReprojPoint {
                    model,
                    chain,
                    pw: [0.1, 0.2, 0.0],
                    uv: [320.0, 240.0],
                    w: 1.0,
                };
                let mut ir = ProblemIR::new();
                let params = add_blocks_for_layout(&mut ir, &factor.param_layout().unwrap());
                ir.add_residual_block(ResidualBlock {
                    params,
                    loss: RobustLoss::None,
                    residual_dim: factor.residual_dim(),
                    factor,
                });
                ir.validate().expect("well-formed descriptor IR validates");
            }
            for chain in laser_chains() {
                for laser_factor in [
                    FactorKind::LaserPointToPlane {
                        model,
                        chain,
                        laser_pixel: [100.0, 200.0],
                        w: 1.0,
                    },
                    FactorKind::LaserLineDistance {
                        model,
                        chain,
                        laser_pixel: [100.0, 200.0],
                        w: 1.0,
                    },
                ] {
                    let mut ir = ProblemIR::new();
                    let params =
                        add_blocks_for_layout(&mut ir, &laser_factor.param_layout().unwrap());
                    ir.add_residual_block(ResidualBlock {
                        params,
                        loss: RobustLoss::None,
                        residual_dim: laser_factor.residual_dim(),
                        factor: laser_factor,
                    });
                    ir.validate().expect("well-formed laser IR validates");
                }
            }
        }
    }

    #[test]
    fn validate_rejects_wrong_block_count() {
        let factor = FactorKind::ReprojPoint {
            model: CameraModelDesc::PINHOLE4_DIST5,
            chain: ReprojChain::SinglePose,
            pw: [0.0; 3],
            uv: [0.0; 2],
            w: 1.0,
        };
        let mut ir = ProblemIR::new();
        let mut params = add_blocks_for_layout(&mut ir, &factor.param_layout().unwrap());
        params.pop();
        ir.add_residual_block(ResidualBlock {
            params,
            loss: RobustLoss::None,
            residual_dim: factor.residual_dim(),
            factor,
        });
        let err = ir.validate().unwrap_err().to_string();
        assert!(err.contains("requires 3 params"), "unexpected error: {err}");
        assert!(
            err.contains("intrinsics, distortion"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_rejects_wrong_dim_and_manifold() {
        // Wrong distortion dim (4 instead of 5).
        let factor = FactorKind::ReprojPoint {
            model: CameraModelDesc::PINHOLE4_DIST5,
            chain: ReprojChain::SinglePose,
            pw: [0.0; 3],
            uv: [0.0; 2],
            w: 1.0,
        };
        let mut ir = ProblemIR::new();
        let cam = ir.add_param_block("k", 4, ManifoldKind::Euclidean, FixedMask::all_free(), None);
        let dist = ir.add_param_block("d", 4, ManifoldKind::Euclidean, FixedMask::all_free(), None);
        let pose = ir.add_param_block("p", 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        ir.add_residual_block(ResidualBlock {
            params: vec![cam, dist, pose],
            loss: RobustLoss::None,
            residual_dim: factor.residual_dim(),
            factor,
        });
        let err = ir.validate().unwrap_err().to_string();
        assert!(err.contains("distortion"), "unexpected error: {err}");

        // Wrong plane-normal manifold (Euclidean instead of S2).
        let factor = FactorKind::LaserPointToPlane {
            model: CameraModelDesc::PINHOLE4_DIST5_SCHEIMPFLUG2,
            chain: LaserChain::SinglePose,
            laser_pixel: [0.0; 2],
            w: 1.0,
        };
        let mut ir = ProblemIR::new();
        let layout = factor.param_layout().unwrap();
        let params: Vec<_> = layout
            .iter()
            .map(|slot| {
                let manifold = if slot.role == "plane_normal" {
                    ManifoldKind::Euclidean
                } else {
                    slot.manifold
                };
                ir.add_param_block(slot.role, slot.dim, manifold, FixedMask::all_free(), None)
            })
            .collect();
        ir.add_residual_block(ResidualBlock {
            params,
            loss: RobustLoss::None,
            residual_dim: factor.residual_dim(),
            factor,
        });
        let err = ir.validate().unwrap_err().to_string();
        assert!(err.contains("plane_normal"), "unexpected error: {err}");
    }
}
