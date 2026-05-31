//! Tier-A frozen fixtures: the image-free `*Input` IR.
//!
//! A [`FrozenFixture`] captures everything a Tier-A run needs to reproduce a
//! calibration *without* touching images or the detector: the already-extracted
//! correspondences ([`CorrespondenceView`]), any robot poses, and laser pixels.
//! This lets the math/serde tier run deterministically in CI on committed data.
//!
//! # Pose convention
//!
//! Poses are stored as [`Pose7`] = `[f64; 7]` in the workspace SE3 order
//! `[qx, qy, qz, qw, tx, ty, tz]` (quaternion `xyzw` then translation `xyz`;
//! see ADR 0009). The fixed-array form is used instead of
//! `nalgebra::Isometry3` to keep the on-disk JSON stable and explicit. Convert
//! to/from `Isometry3<f64>` at the boundary: build a `UnitQuaternion` from the
//! first four components and a `Translation3` from the last three.
//!
//! # No `PartialEq`
//!
//! [`CorrespondenceView`] does not implement `PartialEq`, so neither does
//! [`FrozenFixture`]. The roundtrip tests compare re-serialized JSON.

use serde::{Deserialize, Serialize};
use vision_calibration_core::CorrespondenceView;

use crate::registry::BoardGeometry;

/// SE3 pose in `[qx, qy, qz, qw, tx, ty, tz]` order (ADR 0009).
pub type Pose7 = [f64; 7];

/// A frozen, image-free fixture for a single dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenFixture {
    /// Fixture schema version.
    pub schema_version: u32,
    /// Dataset identifier this fixture was frozen from.
    pub dataset_id: String,
    /// Problem kind (string form, matching the payload variant).
    pub problem: String,
    /// Git SHA the fixture was frozen at.
    pub source_git_sha: String,
    /// Board geometry used for the dataset.
    pub board: BoardGeometry,
    /// The problem-specific frozen payload.
    pub payload: FrozenPayload,
}

/// Problem-specific frozen input IR.
///
/// Externally tagged on a `"kind"` field with snake_case variant names that
/// mirror [`crate::registry::ProblemKind`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrozenPayload {
    /// Single-camera planar intrinsics: one view list.
    PlanarIntrinsics {
        /// Correspondence views, one per image.
        views: Vec<CorrespondenceView>,
    },
    /// Scheimpflug intrinsics: one view list.
    ScheimpflugIntrinsics {
        /// Correspondence views, one per image.
        views: Vec<CorrespondenceView>,
    },
    /// Multi-camera rig extrinsics.
    RigExtrinsics {
        /// Number of cameras in the rig.
        num_cameras: usize,
        /// Outer = pose, inner = per-camera view (`None` if not seen).
        views: Vec<Vec<Option<CorrespondenceView>>>,
    },
    /// Single-camera hand-eye.
    SingleCamHandeye {
        /// Hand-eye mode (e.g. `"eye_in_hand"` / `"eye_to_hand"`).
        mode: String,
        /// Per-pose views with the robot pose attached.
        views: Vec<HandeyeViewFx>,
    },
    /// Multi-camera rig hand-eye.
    RigHandeye {
        /// Number of cameras in the rig.
        num_cameras: usize,
        /// Hand-eye mode.
        mode: String,
        /// Per-pose multi-camera views with the robot pose attached.
        views: Vec<RigHandeyeViewFx>,
    },
    /// Single laserline device.
    LaserlineDevice {
        /// Per-view target observation + laser pixels.
        views: Vec<LaserViewFx>,
    },
    /// Multi-camera rig laserline device.
    RigLaserlineDevice {
        /// Number of cameras in the rig.
        num_cameras: usize,
        /// Frozen upstream hand-eye export (free-form for now).
        upstream: serde_json::Value,
        /// Per-view multi-camera observations + laser pixels.
        views: Vec<RigLaserViewFx>,
    },
}

/// One hand-eye view: board observation plus the robot pose at capture time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandeyeViewFx {
    /// Target correspondences for this view.
    pub obs: CorrespondenceView,
    /// Robot pose `base_se3_gripper` ([`Pose7`]).
    pub base_se3_gripper: Pose7,
}

/// One rig hand-eye view: per-camera observations plus the robot pose.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeViewFx {
    /// Per-camera target observations (`None` if a camera did not see it).
    pub cameras: Vec<Option<CorrespondenceView>>,
    /// Robot pose `base_se3_gripper` ([`Pose7`]).
    pub base_se3_gripper: Pose7,
}

/// One laserline view: a board observation plus extracted laser pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserViewFx {
    /// Target correspondences for this view.
    pub obs: CorrespondenceView,
    /// Extracted laser pixels `[u, v]`.
    pub laser_pixels: Vec<[f64; 2]>,
}

/// One rig laserline view: per-camera observations plus per-camera laser pixels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigLaserViewFx {
    /// Per-camera target observations (`None` if a camera did not see it).
    pub cameras: Vec<Option<CorrespondenceView>>,
    /// Per-camera extracted laser pixels (`laser_pixels_per_cam[cam][i] = [u, v]`).
    pub laser_pixels_per_cam: Vec<Vec<[f64; 2]>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{Pt2, Pt3};

    fn sample_view() -> CorrespondenceView {
        CorrespondenceView::new(
            vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.01, 0.0, 0.0)],
            vec![Pt2::new(100.0, 100.0), Pt2::new(110.0, 100.0)],
        )
        .expect("valid view")
    }

    fn sample_board() -> BoardGeometry {
        BoardGeometry {
            rows: 13,
            cols: 13,
            cell_size_m: 0.01,
            dictionary: None,
            layout: Some("checkerboard".into()),
            marker_size_rel: None,
        }
    }

    fn fixture(problem: &str, payload: FrozenPayload) -> FrozenFixture {
        FrozenFixture {
            schema_version: 1,
            dataset_id: "ds".into(),
            problem: problem.into(),
            source_git_sha: "cafef00d".into(),
            board: sample_board(),
            payload,
        }
    }

    /// `FrozenFixture` cannot derive `PartialEq` (`CorrespondenceView` lacks
    /// it), so the roundtrip is asserted by comparing re-serialized JSON.
    fn roundtrip(fx: &FrozenFixture) {
        let json = serde_json::to_string(fx).expect("serialize");
        let back: FrozenFixture = serde_json::from_str(&json).expect("deserialize");
        let json2 = serde_json::to_string(&back).expect("re-serialize");
        assert_eq!(json, json2);
    }

    #[test]
    fn planar_intrinsics_roundtrips() {
        roundtrip(&fixture(
            "planar_intrinsics",
            FrozenPayload::PlanarIntrinsics {
                views: vec![sample_view()],
            },
        ));
    }

    #[test]
    fn rig_extrinsics_roundtrips() {
        roundtrip(&fixture(
            "rig_extrinsics",
            FrozenPayload::RigExtrinsics {
                num_cameras: 2,
                views: vec![vec![Some(sample_view()), None]],
            },
        ));
    }

    #[test]
    fn handeye_roundtrips() {
        roundtrip(&fixture(
            "single_cam_handeye",
            FrozenPayload::SingleCamHandeye {
                mode: "eye_in_hand".into(),
                views: vec![HandeyeViewFx {
                    obs: sample_view(),
                    base_se3_gripper: [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3],
                }],
            },
        ));
    }

    #[test]
    fn rig_handeye_roundtrips() {
        roundtrip(&fixture(
            "rig_handeye",
            FrozenPayload::RigHandeye {
                num_cameras: 2,
                mode: "eye_to_hand".into(),
                views: vec![RigHandeyeViewFx {
                    cameras: vec![Some(sample_view()), None],
                    base_se3_gripper: [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3],
                }],
            },
        ));
    }

    #[test]
    fn laserline_roundtrips() {
        roundtrip(&fixture(
            "laserline_device",
            FrozenPayload::LaserlineDevice {
                views: vec![LaserViewFx {
                    obs: sample_view(),
                    laser_pixels: vec![[10.0, 20.0], [11.0, 21.0]],
                }],
            },
        ));
    }

    #[test]
    fn rig_laserline_roundtrips() {
        roundtrip(&fixture(
            "rig_laserline_device",
            FrozenPayload::RigLaserlineDevice {
                num_cameras: 2,
                upstream: serde_json::json!({"frozen": "handeye"}),
                views: vec![RigLaserViewFx {
                    cameras: vec![Some(sample_view()), None],
                    laser_pixels_per_cam: vec![vec![[10.0, 20.0]], vec![]],
                }],
            },
        ));
    }

    #[test]
    fn scheimpflug_intrinsics_roundtrips() {
        roundtrip(&fixture(
            "scheimpflug_intrinsics",
            FrozenPayload::ScheimpflugIntrinsics {
                views: vec![sample_view()],
            },
        ));
    }
}
