//! Which target detections enter a calibration input, and the record of
//! the ones that do not.
//!
//! Every converter applies the same [`ViewFilter`]: a per-view feature
//! floor, plus — for a plain chessboard in a topology whose labels must
//! agree beyond one image — the rule that the detection spans the whole
//! board. A rig's whole-board detections are then matched across the
//! cameras of each view (`board_turn`).

use std::path::PathBuf;

use vision_calibration_dataset::{DatasetSpec, TargetSpec, Topology};
use vision_calibration_detect::Feature;

use super::RunError;

/// Why the runner left a target detection out of the calibration input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RejectReason {
    /// Fewer features than the per-view floor: 4 (the homography minimum),
    /// or the manifest's `detector.min_features_per_view`.
    TooFewFeatures,
    /// A plain chessboard whose detected corners do not reach every edge of
    /// the board. The detector labels a view from its own top-left corner,
    /// so a partial board would move the shared target frame.
    PartialChessboard,
    /// A plain chessboard seen turned by 90° where the target frame is fixed
    /// through the views (hand-eye, and the laser rigs). Its corners span
    /// `rows` across and `cols` down instead of the declared `cols` across
    /// and `rows` down, as camera 0 sees the board, so its labels are a
    /// rotation of the fixed frame's.
    TurnedChessboard,
    /// A plain chessboard in a rig view whose turn relative to the view's
    /// other cameras cannot be told: the board's direction in this image
    /// is not within the tolerance of any of its candidate labellings, or
    /// the camera's turn relative to the other cameras is not consistent
    /// across the views they share.
    AmbiguousChessboardTurn,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TooFewFeatures => "too few features detected",
            Self::PartialChessboard => {
                "the chessboard's corners do not reach every edge of the board"
            }
            Self::TurnedChessboard => "the chessboard is seen turned by 90°",
            Self::AmbiguousChessboardTurn => {
                "the chessboard's turn relative to the view's other cameras is ambiguous"
            }
        })
    }
}

/// A target detection the runner left out of the calibration input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RejectedDetection {
    /// Index of the camera in the manifest's `cameras`.
    pub camera: usize,
    /// The image the detection came from.
    pub path: PathBuf,
    /// Why it was left out.
    pub reason: RejectReason,
}

/// What a plain chessboard's labels must agree with in a topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LabelScope {
    /// Nothing beyond the image: planar intrinsics and the single-camera
    /// laserline device estimate one target pose per view from that view's
    /// corners alone, and any consistent relabelling of a view only
    /// changes that view's pose.
    Image,
    /// The other cameras of the same view: rig extrinsics estimates one
    /// target pose per view, seen by every camera of the view.
    View,
    /// One target frame through every view: hand-eye fixes the target
    /// relative to the robot base or gripper, and the rig laserline
    /// device takes its per-view target poses from an upstream
    /// calibration.
    Fixed,
}

impl LabelScope {
    fn of(topology: Topology) -> Self {
        match topology {
            Topology::PlanarIntrinsics | Topology::LaserlineDevice => Self::Image,
            Topology::RigExtrinsics => Self::View,
            Topology::SingleCamHandeye
            | Topology::RigHandeye
            | Topology::RigLaserlineDevice
            | Topology::RigHandeyeLaserline => Self::Fixed,
        }
    }
}

/// Whether `topology` has more than one camera per view, so that a
/// chessboard's labels are matched across the cameras of a view.
fn is_rig(topology: Topology) -> bool {
    match topology {
        Topology::PlanarIntrinsics | Topology::LaserlineDevice | Topology::SingleCamHandeye => {
            false
        }
        Topology::RigExtrinsics
        | Topology::RigHandeye
        | Topology::RigLaserlineDevice
        | Topology::RigHandeyeLaserline => true,
    }
}

/// The declared interior-corner grid of a plain chessboard.
#[derive(Debug, Clone, Copy)]
pub(super) struct ChessboardGrid {
    /// Interior corners down the board.
    pub(super) rows: u32,
    /// Interior corners across the board.
    pub(super) cols: u32,
    /// Edge length of one square, metres.
    pub(super) square_size_m: f64,
}

/// How a whole-board detection's labels lie on the declared board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoardExtent {
    /// `cols` corners across the image and `rows` down, as declared.
    Declared,
    /// `rows` across and `cols` down: the board seen turned by 90°.
    Turned,
}

impl ChessboardGrid {
    /// How a detection's labels place the board, or why they do not.
    ///
    /// The detector labels each view on its own: `(0, 0)` is the
    /// detection's top-left corner, `u` runs across the image and `v`
    /// down, and a corner at `(u, v)` sits at `(u, v) · square_size_m` in
    /// the target frame. Those labels are the board's own, up to a turn of
    /// the board in its plane, exactly when the detection reaches every
    /// edge of the board: `u` spans `0..cols` and `v` spans `0..rows` as
    /// declared, or `0..rows` and `0..cols` when the board is seen turned
    /// by 90°. Interior corners the detector missed do not move the frame.
    pub(super) fn extent(&self, features: &[Feature]) -> Result<BoardExtent, RejectReason> {
        let Some(extent) = self.label_extent(features) else {
            return Err(RejectReason::PartialChessboard);
        };
        let (cols, rows) = (i64::from(self.cols), i64::from(self.rows));
        if extent == (0, cols - 1, 0, rows - 1) {
            Ok(BoardExtent::Declared)
        } else if extent == (0, rows - 1, 0, cols - 1) {
            Ok(BoardExtent::Turned)
        } else {
            Err(RejectReason::PartialChessboard)
        }
    }

    /// The integer label `(u, v)` of a feature, recovered from its
    /// target-frame position.
    pub(super) fn label(&self, feature: &Feature) -> (i64, i64) {
        let label = |metres: f64| (metres / self.square_size_m).round() as i64;
        (label(feature.world_xyz[0]), label(feature.world_xyz[1]))
    }

    /// `(min u, max u, min v, max v)` of the labels; `None` for an empty
    /// detection.
    fn label_extent(&self, features: &[Feature]) -> Option<(i64, i64, i64, i64)> {
        features.iter().fold(None, |extent, f| {
            let (u, v) = self.label(f);
            Some(match extent {
                None => (u, u, v, v),
                Some((u0, u1, v0, v1)) => (u0.min(u), u1.max(u), v0.min(v), v1.max(v)),
            })
        })
    }
}

/// What a target detection must satisfy to enter the calibration input.
#[derive(Debug, Clone, Copy)]
pub(super) struct ViewFilter {
    min_features: usize,
    /// Set for a plain chessboard in a topology whose labels must agree
    /// beyond one image.
    whole_board: Option<(ChessboardGrid, LabelScope)>,
    /// Whether the topology is a rig, whose turned detections are matched
    /// across the cameras of each view rather than rejected here.
    rig: bool,
}

impl ViewFilter {
    pub(super) fn for_spec(spec: &DatasetSpec) -> Self {
        let min_features = spec
            .detector
            .as_ref()
            .and_then(|d| d.min_features_per_view)
            .unwrap_or(4)
            .max(4);
        let scope = LabelScope::of(spec.topology);
        let whole_board = match spec.target {
            TargetSpec::Chessboard {
                rows,
                cols,
                square_size_m,
            } if scope != LabelScope::Image => Some((
                ChessboardGrid {
                    rows,
                    cols,
                    square_size_m,
                },
                scope,
            )),
            _ => None,
        };
        Self {
            min_features,
            whole_board,
            rig: is_rig(spec.topology),
        }
    }

    /// `Ok` when the detection is kept, otherwise why it is not.
    ///
    /// A rig's chessboard detection that passes is whole but may be seen
    /// turned; [`Self::rig_chessboard`] says how to match it with the
    /// view's other cameras.
    pub(super) fn check(&self, features: &[Feature]) -> Result<(), RejectReason> {
        if features.len() < self.min_features {
            return Err(RejectReason::TooFewFeatures);
        }
        let Some((grid, _)) = self.whole_board else {
            return Ok(());
        };
        match grid.extent(features)? {
            BoardExtent::Turned if !self.rig => Err(RejectReason::TurnedChessboard),
            _ => Ok(()),
        }
    }

    /// For a plain chessboard in a rig topology, the board and what its
    /// labels must agree with: the input to `board_turn::match_rig_labels`.
    pub(super) fn rig_chessboard(&self) -> Option<(ChessboardGrid, LabelScope)> {
        self.whole_board.filter(|_| self.rig)
    }
}

/// The error for a camera left with no usable view: why every one of its
/// `total` views was rejected, as far as `rejected` records it.
pub(super) fn no_usable_views(
    spec: &DatasetSpec,
    camera: usize,
    rejected: &[RejectedDetection],
    total: usize,
) -> RunError {
    let camera_id = spec.cameras[camera].id.clone();
    let count = |reason| {
        rejected
            .iter()
            .filter(|r| r.camera == camera && r.reason == reason)
            .count()
    };
    let (partial, turned, ambiguous) = (
        count(RejectReason::PartialChessboard),
        count(RejectReason::TurnedChessboard),
        count(RejectReason::AmbiguousChessboardTurn),
    );
    if partial + turned + ambiguous > 0 {
        RunError::NoWholeChessboardViews {
            camera_id,
            partial,
            turned,
            ambiguous,
            total,
        }
    } else {
        RunError::InsufficientUsableViews {
            camera_id,
            usable: 0,
            total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: f64 = 0.025;

    /// The corners `(u, v)` with `u < width`, `v < height`, skipping any
    /// in `missing`.
    fn labels(width: i64, height: i64, missing: &[(i64, i64)]) -> Vec<Feature> {
        (0..height)
            .flat_map(|v| (0..width).map(move |u| (u, v)))
            .filter(|uv| !missing.contains(uv))
            .map(|(u, v)| Feature {
                cov_px: None,
                image_xy: [10.0 * u as f64, 10.0 * v as f64],
                world_xyz: [u as f64 * SQUARE, v as f64 * SQUARE, 0.0],
            })
            .collect()
    }

    /// A 10-row × 14-column board (DS8's).
    fn grid() -> ChessboardGrid {
        ChessboardGrid {
            rows: 10,
            cols: 14,
            square_size_m: SQUARE,
        }
    }

    #[test]
    fn a_complete_upright_board_is_declared() {
        assert_eq!(
            grid().extent(&labels(14, 10, &[])),
            Ok(BoardExtent::Declared)
        );
    }

    #[test]
    fn missing_corners_that_leave_every_edge_keep_the_frame() {
        // An interior corner and the top-left corner itself: the rest of
        // row 0 and column 0 still pin the labels.
        let features = labels(14, 10, &[(5, 5), (0, 0)]);
        assert_eq!(grid().extent(&features), Ok(BoardExtent::Declared));
    }

    #[test]
    fn a_partial_board_is_rejected() {
        // The right-most two columns are out of view.
        assert_eq!(
            grid().extent(&labels(12, 10, &[])),
            Err(RejectReason::PartialChessboard)
        );
        // The bottom row is out of view.
        assert_eq!(
            grid().extent(&labels(14, 9, &[])),
            Err(RejectReason::PartialChessboard)
        );
        // Partly, and overflowing the declared 10 rows.
        assert_eq!(
            grid().extent(&labels(8, 12, &[])),
            Err(RejectReason::PartialChessboard)
        );
        assert_eq!(grid().extent(&[]), Err(RejectReason::PartialChessboard));
    }

    #[test]
    fn a_whole_turned_board_is_told_by_its_extent() {
        // Seen turned by 90°, the 10 × 14 board spans 10 across, 14 down.
        assert_eq!(grid().extent(&labels(10, 14, &[])), Ok(BoardExtent::Turned));
    }

    #[test]
    fn a_turned_square_board_is_indistinguishable() {
        let square = ChessboardGrid {
            rows: 9,
            cols: 9,
            square_size_m: SQUARE,
        };
        assert_eq!(square.extent(&labels(9, 9, &[])), Ok(BoardExtent::Declared));
    }

    fn spec(topology: Topology, target: TargetSpec) -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![],
            target,
            detector: None,
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
            topology,
            pose_pairing: None,
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
    }

    fn chessboard() -> TargetSpec {
        TargetSpec::Chessboard {
            rows: 10,
            cols: 14,
            square_size_m: SQUARE,
        }
    }

    #[test]
    fn only_topologies_that_share_labels_require_the_whole_board() {
        let partial = labels(12, 10, &[]);
        for topology in [Topology::PlanarIntrinsics, Topology::LaserlineDevice] {
            let filter = ViewFilter::for_spec(&spec(topology, chessboard()));
            assert_eq!(filter.check(&partial), Ok(()), "{topology:?}");
        }
        for topology in [
            Topology::SingleCamHandeye,
            Topology::RigExtrinsics,
            Topology::RigHandeye,
            Topology::RigLaserlineDevice,
            Topology::RigHandeyeLaserline,
        ] {
            let filter = ViewFilter::for_spec(&spec(topology, chessboard()));
            assert_eq!(
                filter.check(&partial),
                Err(RejectReason::PartialChessboard),
                "{topology:?}"
            );
        }
    }

    #[test]
    fn self_identifying_targets_are_not_screened_for_the_whole_board() {
        let target = TargetSpec::Puzzleboard {
            layout: "puzzle_10x14".into(),
            cell_size_m: SQUARE,
        };
        let filter = ViewFilter::for_spec(&spec(Topology::RigHandeye, target));
        assert_eq!(filter.check(&labels(4, 3, &[])), Ok(()));
    }

    #[test]
    fn a_turned_board_is_rejected_by_single_camera_hand_eye_only() {
        let turned = labels(10, 14, &[]);
        let filter = ViewFilter::for_spec(&spec(Topology::SingleCamHandeye, chessboard()));
        assert_eq!(filter.check(&turned), Err(RejectReason::TurnedChessboard));
        assert!(filter.rig_chessboard().is_none());
        // A rig matches it with the view's other cameras instead.
        for (topology, scope) in [
            (Topology::RigExtrinsics, LabelScope::View),
            (Topology::RigHandeye, LabelScope::Fixed),
            (Topology::RigLaserlineDevice, LabelScope::Fixed),
            (Topology::RigHandeyeLaserline, LabelScope::Fixed),
        ] {
            let filter = ViewFilter::for_spec(&spec(topology, chessboard()));
            assert_eq!(filter.check(&turned), Ok(()), "{topology:?}");
            assert_eq!(filter.rig_chessboard().map(|(_, s)| s), Some(scope));
        }
    }

    #[test]
    fn the_feature_floor_is_checked_first() {
        let filter = ViewFilter::for_spec(&spec(Topology::RigHandeye, chessboard()));
        assert_eq!(
            filter.check(&labels(3, 1, &[])),
            Err(RejectReason::TooFewFeatures)
        );
    }
}
