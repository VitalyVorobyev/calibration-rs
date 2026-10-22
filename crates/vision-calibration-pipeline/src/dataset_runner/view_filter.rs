//! Which target detections enter a calibration input, and the record of
//! the ones that do not.
//!
//! Every converter applies the same [`ViewFilter`]: a per-view feature
//! floor, plus — for a plain chessboard in a topology whose views share one
//! target frame — the rule that the detection spans the whole board in its
//! declared orientation.

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
    /// A plain chessboard seen turned by 90°: its corners span `rows`
    /// across and `cols` down instead of the declared `cols` across and
    /// `rows` down, so its labels are a rotation of the board's.
    TurnedChessboard,
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

/// Whether the views of `topology` share one target frame — across views
/// (a target fixed relative to the robot base or gripper) or across the
/// cameras of one view (a rig).
///
/// Planar intrinsics and the single-camera laserline device estimate one
/// target pose per view from that view's corners alone; any consistent
/// in-plane relabelling of a view only changes that view's pose.
fn shares_target_frame(topology: Topology) -> bool {
    match topology {
        Topology::PlanarIntrinsics | Topology::LaserlineDevice => false,
        Topology::SingleCamHandeye
        | Topology::RigExtrinsics
        | Topology::RigHandeye
        | Topology::RigLaserlineDevice
        | Topology::RigHandeyeLaserline => true,
    }
}

/// The declared interior-corner grid of a plain chessboard.
#[derive(Debug, Clone, Copy)]
struct ChessboardGrid {
    rows: u32,
    cols: u32,
    square_size_m: f64,
}

impl ChessboardGrid {
    /// Whether a detection's labels place the board's own frame.
    ///
    /// The detector labels each view on its own: `(0, 0)` is the
    /// detection's top-left corner, `u` runs across the image and `v` down,
    /// and a corner at `(u, v)` sits at `(u, v) · square_size_m` in the
    /// target frame. Those labels are the board's own exactly when the
    /// detection reaches every edge of the board as declared: `u` spans
    /// `0..cols` and `v` spans `0..rows`. Interior corners the detector
    /// missed do not move the frame.
    ///
    /// Labels beyond the declared extent come from a board seen turned by
    /// 90°, the only overflow the detector keeps. A square board seen
    /// turned, or any board seen turned by 180°, spans the declared extent
    /// too; nothing in the labels tells it apart, so those views must be
    /// avoided at capture.
    fn check(&self, features: &[Feature]) -> Result<(), RejectReason> {
        let Some((min_u, max_u, min_v, max_v)) = self.label_extent(features) else {
            return Err(RejectReason::PartialChessboard);
        };
        let (last_u, last_v) = (i64::from(self.cols) - 1, i64::from(self.rows) - 1);
        if max_u > last_u || max_v > last_v {
            Err(RejectReason::TurnedChessboard)
        } else if (min_u, max_u, min_v, max_v) == (0, last_u, 0, last_v) {
            Ok(())
        } else {
            Err(RejectReason::PartialChessboard)
        }
    }

    /// `(min u, max u, min v, max v)` of the labels, recovered from the
    /// target-frame positions; `None` for an empty detection.
    fn label_extent(&self, features: &[Feature]) -> Option<(i64, i64, i64, i64)> {
        let label = |metres: f64| (metres / self.square_size_m).round() as i64;
        features.iter().fold(None, |extent, f| {
            let (u, v) = (label(f.world_xyz[0]), label(f.world_xyz[1]));
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
    /// Set for a plain chessboard in a topology whose views share one
    /// target frame.
    whole_board: Option<ChessboardGrid>,
}

impl ViewFilter {
    pub(super) fn for_spec(spec: &DatasetSpec) -> Self {
        let min_features = spec
            .detector
            .as_ref()
            .and_then(|d| d.min_features_per_view)
            .unwrap_or(4)
            .max(4);
        let whole_board = match spec.target {
            TargetSpec::Chessboard {
                rows,
                cols,
                square_size_m,
            } if shares_target_frame(spec.topology) => Some(ChessboardGrid {
                rows,
                cols,
                square_size_m,
            }),
            _ => None,
        };
        Self {
            min_features,
            whole_board,
        }
    }

    /// `Ok` when the detection is kept, otherwise why it is not.
    pub(super) fn check(&self, features: &[Feature]) -> Result<(), RejectReason> {
        if features.len() < self.min_features {
            return Err(RejectReason::TooFewFeatures);
        }
        self.whole_board.map_or(Ok(()), |grid| grid.check(features))
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
    let (partial, turned) = (
        count(RejectReason::PartialChessboard),
        count(RejectReason::TurnedChessboard),
    );
    if partial + turned > 0 {
        RunError::NoWholeChessboardViews {
            camera_id,
            partial,
            turned,
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
    fn a_complete_upright_board_is_kept() {
        assert_eq!(grid().check(&labels(14, 10, &[])), Ok(()));
    }

    #[test]
    fn missing_corners_that_leave_every_edge_keep_the_frame() {
        // An interior corner and the top-left corner itself: the rest of
        // row 0 and column 0 still pin the labels.
        let features = labels(14, 10, &[(5, 5), (0, 0)]);
        assert_eq!(grid().check(&features), Ok(()));
    }

    #[test]
    fn a_partial_board_is_rejected() {
        // The right-most two columns are out of view.
        assert_eq!(
            grid().check(&labels(12, 10, &[])),
            Err(RejectReason::PartialChessboard)
        );
        // The bottom row is out of view.
        assert_eq!(
            grid().check(&labels(14, 9, &[])),
            Err(RejectReason::PartialChessboard)
        );
        assert_eq!(grid().check(&[]), Err(RejectReason::PartialChessboard));
    }

    #[test]
    fn a_turned_board_is_rejected() {
        // Seen turned by 90°, the 10 × 14 board spans 10 across, 14 down.
        assert_eq!(
            grid().check(&labels(10, 14, &[])),
            Err(RejectReason::TurnedChessboard)
        );
        // Partly, but still overflowing the declared 10 rows.
        assert_eq!(
            grid().check(&labels(8, 12, &[])),
            Err(RejectReason::TurnedChessboard)
        );
    }

    #[test]
    fn a_turned_square_board_is_indistinguishable() {
        let square = ChessboardGrid {
            rows: 9,
            cols: 9,
            square_size_m: SQUARE,
        };
        assert_eq!(square.check(&labels(9, 9, &[])), Ok(()));
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
    fn only_shared_frame_topologies_require_the_whole_board() {
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
    fn the_feature_floor_is_checked_first() {
        let filter = ViewFilter::for_spec(&spec(Topology::RigHandeye, chessboard()));
        assert_eq!(
            filter.check(&labels(3, 1, &[])),
            Err(RejectReason::TooFewFeatures)
        );
    }
}
