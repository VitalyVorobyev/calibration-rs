//! What a detection looked like: the features [`DetectorConfig::detect`]
//! returns, with what an overlay needs to judge them.
//!
//! [`DetectorConfig::detect_detailed`] returns a [`DetectionDetail`]: the
//! features, bit for bit those of `detect`, the image's
//! [`DetectionOutcome`], per-target overlay data ([`TargetDetail`]) whose
//! per-feature vectors are aligned one to one with the features, and the
//! point candidates left out ([`RejectedPoint`]), each with its reason.
//!
//! Every position is in the pixels of the image the detector ran on, with
//! the centre of pixel `i` at `i` (the convention of [`Feature::image_xy`]).
//! Angles are in radians, measured from the image's +x axis towards its +y
//! axis: clockwise on screen, where y runs down.
//!
//! What a detector does not report is left empty rather than estimated: each
//! field says when it is empty.
//!
//! [`DetectorConfig::detect`]: crate::DetectorConfig::detect
//! [`DetectorConfig::detect_detailed`]: crate::DetectorConfig::detect_detailed

use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

use crate::Feature;

/// What a detector found in one image: the correspondences
/// [`DetectorConfig::detect`](crate::DetectorConfig::detect) returns, and
/// what an overlay needs to judge them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct DetectionDetail {
    /// The features, equal bit for bit to what
    /// [`DetectorConfig::detect`](crate::DetectorConfig::detect) returns for
    /// the same image and config.
    pub features: Vec<Feature>,
    /// The image-level outcome.
    pub outcome: DetectionOutcome,
    /// Target-specific overlay data. Its per-feature vector (`corners` or
    /// `rings`) holds one entry per feature, in the order of
    /// [`features`](Self::features).
    pub target: TargetDetail,
    /// Point candidates the detector or this crate saw and did not keep,
    /// each with its reason. Empty when there were none, and on an
    /// [`DetectionOutcome::Ambiguous`] image (its dropped points are not
    /// listed).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected: Vec<RejectedPoint>,
}

/// The outcome of one image's detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DetectionOutcome {
    /// The target was found and [`DetectionDetail::features`] is not empty.
    Detected,
    /// The detector found no target; `reason` is its own account (for a
    /// ChArUco board, with each grid component's rejection by the marker
    /// matcher).
    NotFound {
        /// The detector's reason, for a reader.
        reason: String,
    },
    /// The detection repeated an image point or a target point, so its
    /// labels are not trustworthy and it was dropped whole
    /// ([`reject_ambiguous_detection`](crate::reject_ambiguous_detection)).
    Ambiguous {
        /// Features whose pixel repeats an earlier feature's.
        repeated_pixels: usize,
        /// Features whose target point repeats an earlier feature's.
        repeated_points: usize,
    },
    /// A target was detected, but this crate's filters removed every
    /// feature: see [`DetectionDetail::rejected`].
    Filtered,
}

/// Overlay data of one target kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TargetDetail {
    /// A plain chessboard.
    Chessboard(ChessboardDetail),
    /// A ChArUco board.
    Charuco(CharucoDetail),
    /// A PuzzleBoard.
    Puzzleboard(PuzzleboardDetail),
    /// A ring grid, coded or plain, on a hex or rect lattice.
    Ringgrid(RinggridDetail),
}

/// One labelled board corner, aligned with a [`Feature`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct CornerDetail {
    /// The detector's lattice label `[u, v]`: `u` along the board's +x, `v`
    /// along its +y. A plain chessboard labels each view from its own
    /// top-left corner; a ChArUco board and a PuzzleBoard label it on the
    /// board (the PuzzleBoard on its master pattern).
    pub grid: [i32; 2],
    /// The ChArUco corner id, or the PuzzleBoard master corner id; `None`
    /// on a chessboard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    /// The detector's corner score: higher is better, not comparable across
    /// detectors.
    pub score: f64,
    /// Directions of the two grid lines through the corner, radians in
    /// `[0, 2π)`, as the corner stage measured them. Present on a chessboard
    /// corner whose corner stage fitted at least one axis; `None` on ChArUco
    /// and PuzzleBoard corners, whose detectors do not report them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axes_rad: Option<[f64; 2]>,
}

/// How well a detection's corners are localized, as the detector measured
/// it on the corners it returned.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct CornerQuality {
    /// The image noise level, gray levels (1σ), the corner covariances were
    /// computed with; `None` when no covariance was computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_sigma: Option<f64>,
    /// Median over the corners of their 1σ position uncertainty along the
    /// least certain direction, px; `None` when no corner carries a
    /// covariance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_sigma_px: Option<f64>,
    /// Number of corners with enough labelled neighbours (six of the eight)
    /// for the leave-one-out check: each such corner's position is
    /// predicted from its neighbours.
    pub local_residual_count: usize,
    /// RMS length of the leave-one-out residuals (observed minus predicted
    /// position) over those corners, px; `None` when no corner qualifies.
    /// The detectors report this aggregate only, not each corner's residual.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_residual_rms_px: Option<f64>,
}

/// A chessboard's overlay data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct ChessboardDetail {
    /// One per feature, in feature order.
    pub corners: Vec<CornerDetail>,
    /// The board's cell size in pixels (median labelled edge length);
    /// `None` when no board was detected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_size_px: Option<f64>,
    /// How well the corners are localized.
    pub quality: CornerQuality,
}

/// A ChArUco board's overlay data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct CharucoDetail {
    /// One per feature, in feature order.
    pub corners: Vec<CornerDetail>,
    /// The markers the board alignment kept.
    pub markers: Vec<MarkerDetail>,
    /// Marker cells, of the board the detection was built from, whose bits
    /// match another marker of the board better than the one the chosen
    /// alignment places there: a misread or wrongly placed marker.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected_markers: Vec<RejectedMarker>,
    /// How well the corners are localized.
    pub quality: CornerQuality,
}

/// One decoded ArUco marker of a ChArUco board.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct MarkerDetail {
    /// The marker's dictionary id.
    pub id: u32,
    /// The board square it sits in, `[column, row]`.
    pub cell: [i32; 2],
    /// The square's corners in image pixels, in the board's order: the
    /// square's top-left on the board (the marker's own top-left) first,
    /// then clockwise on the board.
    pub quad_px: [[f64; 2]; 4],
    /// The rotation, in quarter turns, that aligns the bits read in the
    /// image with the dictionary's code.
    pub rotation: u8,
    /// Bits read in the image that differ from the marker's code.
    pub hamming: u8,
    /// The decoding's confidence, in `[0, 1]`.
    pub score: f64,
}

/// A ChArUco marker cell whose bits match another marker of the board
/// better than the one the board places there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct RejectedMarker {
    /// The cell's corners in image pixels, in the board's order (as
    /// [`MarkerDetail::quad_px`]).
    pub quad_px: [[f64; 2]; 4],
    /// The board marker the cell's bits match best, over every marker of the
    /// board and every rotation.
    pub decoded_id: u32,
    /// The marker the board places in this cell under the chosen alignment.
    pub expected_id: u32,
    /// The matcher's log-likelihood of the cell's bits under the decoded
    /// marker: higher is better, comparable within one image.
    pub score: f64,
}

/// A PuzzleBoard's overlay data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct PuzzleboardDetail {
    /// One per feature, in feature order.
    pub corners: Vec<CornerDetail>,
    /// The code bits read on the edges between two kept corners.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub edge_bits: Vec<EdgeBitDetail>,
    /// The decode summary; `None` when no board was decoded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<PuzzleboardDecode>,
    /// How well the corners are localized.
    pub quality: CornerQuality,
}

/// One code bit read on a PuzzleBoard edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct EdgeBitDetail {
    /// The two corners the edge joins, image pixels. An edge is listed only
    /// when both are features.
    pub endpoints_px: [[f64; 2]; 2],
    /// The bit read: the dot on the edge is black (`0`) or white (`1`).
    pub bit: u8,
    /// The read's confidence, in `[0, 1]`.
    pub confidence: f64,
    /// Whether the bit equals the master pattern's bit at this edge under
    /// the decoded alignment.
    pub matches_master: bool,
}

/// How a PuzzleBoard decoded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct PuzzleboardDecode {
    /// Edges whose bit was read.
    pub edges_observed: usize,
    /// Edges whose bit matched the master pattern after alignment.
    pub edges_matched: usize,
    /// Fraction of the read bits that disagree with the master pattern.
    pub bit_error_rate: f64,
    /// The same, over the distinct master bits the view reads, after the
    /// repeated reads of each were majority-voted.
    pub logical_bit_error_rate: f64,
    /// Fraction of the read confidence that lost its majority vote; high
    /// means noisy dots or a mislabelled grid.
    pub dot_dissent_rate: f64,
    /// Where the view's local corner `(0, 0)` lies on the master pattern,
    /// `[row, column]`.
    pub master_origin: [i32; 2],
}

/// A ring grid's overlay data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct RinggridDetail {
    /// One per feature, in feature order.
    pub rings: Vec<RingDetail>,
    /// The origin dots of a plain board, predicted through the board-to-image
    /// homography (no lens model), image pixels. The detector does not
    /// report the dots it found, so these are where the board places them.
    /// Empty for a coded board and for a view not placed on the board.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origin_dots_px: Vec<[f64; 2]>,
    /// Whether the labels are absolute board positions: always for a coded
    /// board that was labelled, for a plain one when its origin dots placed
    /// the view on the board.
    pub anchored: bool,
    /// Whether every marker of the board was detected; `None` when nothing
    /// was labelled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board_complete: Option<bool>,
}

/// One ring marker, aligned with a [`Feature`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct RingDetail {
    /// The decoded codebook id (coded boards); `None` on a plain board.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
    /// The lattice cell `[u, v]` (axial coordinates on a hex lattice).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<[i32; 2]>,
    /// The detector's confidence in the centre, in `[0, 1]`.
    pub confidence: f64,
    /// The fitted outer ellipse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outer: Option<EllipseDetail>,
    /// The fitted inner ellipse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner: Option<EllipseDetail>,
    /// Which estimator produced the feature's position.
    pub center: RingCenter,
    /// The covariance of the centre as the detector stated it, `[xx, xy,
    /// yy]` px²: photometric centres only. The calibration does not read it
    /// ([`Feature::cov_px`] stays `None` for ring markers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_cov_px: Option<[f64; 3]>,
}

/// An ellipse in image pixels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct EllipseDetail {
    /// The centre, px.
    pub center_px: [f64; 2],
    /// The semi-axes `[a, b]`, `a ≥ b`, px.
    pub semi_axes_px: [f64; 2],
    /// The direction of the major axis, radians.
    pub angle_rad: f64,
}

/// Which estimator produced a ring marker's centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RingCenter {
    /// The centre of the fitted outer ellipse.
    OuterEllipse,
    /// The projective centre recovered from the inner and outer ellipses.
    Projective,
    /// A photometric fit of the blurred marker pattern.
    Photometric,
}

/// A point candidate the detector or this crate saw and did not keep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct RejectedPoint {
    /// Its position, image pixels.
    pub image_xy: [f64; 2],
    /// Why it was left out.
    pub reason: PointRejection,
    /// The target point it would have had, metres, when it had a label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_xyz: Option<[f64; 3]>,
    /// Its lattice label, when it had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<[i32; 2]>,
    /// Its id (a ring's codebook id, a PuzzleBoard master corner id), when
    /// it had one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<u32>,
}

/// Why a point candidate was left out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum PointRejection {
    /// A chessboard corner candidate no labelled grid took. Listed within
    /// the labelled corners' bounding box widened by two cells, or, when
    /// nothing was labelled, the strongest 1 000. ChArUco and PuzzleBoard
    /// details list none: their detectors do not say which candidates they
    /// took.
    Unlabelled {
        /// The corner detector's response.
        strength: f64,
    },
    /// Labelled outside the declared board.
    OutsideBoard,
    /// A ring marker with no board position: a view the detector could not
    /// place on the board, or a marker off it.
    Unanchored {
        /// The detector's confidence in the centre, in `[0, 1]`.
        confidence: f64,
    },
    /// A ring marker whose photometric fit fell back to an ellipse centre,
    /// with photometric centres requested.
    UncorrectedCenter {
        /// The detector's confidence in the centre, in `[0, 1]`.
        confidence: f64,
    },
}

impl DetectionDetail {
    /// The detail of `features` (before the ambiguity check) with `target`'s
    /// per-feature entries `per_feature`, finished: a detection that repeats
    /// a pixel or a target point is dropped whole, as
    /// [`reject_ambiguous_detection`](crate::reject_ambiguous_detection)
    /// drops it, and the outcome follows from what is left.
    ///
    /// `not_found` is the detector's reason when it found no target.
    pub(crate) fn finish<T>(
        features: Vec<Feature>,
        per_feature: Vec<T>,
        rejected: Vec<RejectedPoint>,
        not_found: Option<String>,
        target: impl FnOnce(Vec<T>) -> TargetDetail,
    ) -> Self {
        debug_assert_eq!(features.len(), per_feature.len());
        if let Some(reason) = not_found {
            return Self {
                features: Vec::new(),
                outcome: DetectionOutcome::NotFound { reason },
                target: target(Vec::new()),
                rejected,
            };
        }
        if let Some((repeated_pixels, repeated_points)) = crate::feature::repetitions(&features) {
            return Self {
                features: Vec::new(),
                outcome: DetectionOutcome::Ambiguous {
                    repeated_pixels,
                    repeated_points,
                },
                target: target(Vec::new()),
                rejected: Vec::new(),
            };
        }
        let outcome = if !features.is_empty() {
            DetectionOutcome::Detected
        } else if !rejected.is_empty() {
            DetectionOutcome::Filtered
        } else {
            DetectionOutcome::NotFound {
                reason: "the detector labelled no point".to_string(),
            }
        };
        Self {
            features,
            outcome,
            target: target(per_feature),
            rejected,
        }
    }
}

impl CornerQuality {
    /// The quality the calib-targets detectors measured.
    #[cfg(any(feature = "chessboard", feature = "charuco", feature = "puzzleboard"))]
    pub(crate) fn from_calib_targets(q: &calib_targets::core::DetectionQuality) -> Self {
        Self {
            noise_sigma: q.noise_sigma.map(f64::from),
            median_sigma_px: q.median_sigma_px.map(f64::from),
            local_residual_count: q.local_residual_count,
            local_residual_rms_px: q.local_residual_rms_px.map(f64::from),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(detail: &DetectionDetail) {
        let json = serde_json::to_string(detail).unwrap();
        assert_eq!(
            &serde_json::from_str::<DetectionDetail>(&json).unwrap(),
            detail
        );
    }

    fn corner(u: i32, v: i32) -> CornerDetail {
        CornerDetail {
            grid: [u, v],
            id: Some(7),
            score: 0.75,
            axes_rad: Some([0.1, 1.6707963]),
        }
    }

    fn quality() -> CornerQuality {
        CornerQuality {
            noise_sigma: Some(1.5),
            median_sigma_px: Some(0.031),
            local_residual_count: 12,
            local_residual_rms_px: Some(0.044),
        }
    }

    #[test]
    fn every_target_kind_round_trips() {
        let features = vec![
            Feature::new([10.25, 20.5], [0.0, 0.0, 0.0]).with_cov_px(Some([0.01, 0.0, 0.02])),
            Feature::new([30.125, 20.5], [0.02, 0.0, 0.0]),
        ];
        let rejected = vec![
            RejectedPoint {
                image_xy: [1.0, 2.0],
                reason: PointRejection::Unlabelled { strength: 31.5 },
                world_xyz: None,
                grid: None,
                id: None,
            },
            RejectedPoint {
                image_xy: [3.0, 4.0],
                reason: PointRejection::OutsideBoard,
                world_xyz: Some([0.1, 0.2, 0.0]),
                grid: Some([5, 10]),
                id: Some(3),
            },
            RejectedPoint {
                image_xy: [5.0, 6.0],
                reason: PointRejection::Unanchored { confidence: 0.4 },
                world_xyz: None,
                grid: Some([1, 1]),
                id: None,
            },
            RejectedPoint {
                image_xy: [7.0, 8.0],
                reason: PointRejection::UncorrectedCenter { confidence: 0.2 },
                world_xyz: Some([0.004, 0.0, 0.0]),
                grid: None,
                id: Some(17),
            },
        ];
        let quad = [[0.5, 0.5], [10.5, 0.5], [10.5, 10.5], [0.5, 10.5]];
        let ellipse = EllipseDetail {
            center_px: [5.0, 6.0],
            semi_axes_px: [4.0, 3.5],
            angle_rad: -0.3,
        };
        let targets = vec![
            TargetDetail::Chessboard(ChessboardDetail {
                corners: vec![corner(0, 0), corner(1, 0)],
                cell_size_px: Some(20.0),
                quality: quality(),
            }),
            TargetDetail::Charuco(CharucoDetail {
                corners: vec![corner(0, 0), corner(1, 0)],
                markers: vec![MarkerDetail {
                    id: 4,
                    cell: [1, 2],
                    quad_px: quad,
                    rotation: 1,
                    hamming: 0,
                    score: 0.99,
                }],
                rejected_markers: vec![RejectedMarker {
                    quad_px: quad,
                    decoded_id: 9,
                    expected_id: 5,
                    score: -12.5,
                }],
                quality: CornerQuality::default(),
            }),
            TargetDetail::Puzzleboard(PuzzleboardDetail {
                corners: vec![corner(0, 0), corner(1, 0)],
                edge_bits: vec![EdgeBitDetail {
                    endpoints_px: [[10.25, 20.5], [30.125, 20.5]],
                    bit: 1,
                    confidence: 0.9,
                    matches_master: false,
                }],
                decode: Some(PuzzleboardDecode {
                    edges_observed: 40,
                    edges_matched: 39,
                    bit_error_rate: 0.025,
                    logical_bit_error_rate: 0.0,
                    dot_dissent_rate: 0.01,
                    master_origin: [100, 200],
                }),
                quality: quality(),
            }),
            TargetDetail::Ringgrid(RinggridDetail {
                rings: vec![
                    RingDetail {
                        id: Some(3),
                        grid: Some([0, 0]),
                        confidence: 0.8,
                        outer: Some(ellipse.clone()),
                        inner: Some(ellipse.clone()),
                        center: RingCenter::Photometric,
                        center_cov_px: Some([1e-4, 0.0, 2e-4]),
                    },
                    RingDetail {
                        id: None,
                        grid: None,
                        confidence: 0.5,
                        outer: None,
                        inner: None,
                        center: RingCenter::Projective,
                        center_cov_px: None,
                    },
                ],
                origin_dots_px: vec![[1.0, 1.0]],
                anchored: true,
                board_complete: Some(false),
            }),
        ];
        for target in targets {
            round_trip(&DetectionDetail {
                features: features.clone(),
                outcome: DetectionOutcome::Detected,
                target,
                rejected: rejected.clone(),
            });
        }
        for outcome in [
            DetectionOutcome::NotFound {
                reason: "no board".to_string(),
            },
            DetectionOutcome::Ambiguous {
                repeated_pixels: 2,
                repeated_points: 0,
            },
            DetectionOutcome::Filtered,
        ] {
            round_trip(&DetectionDetail {
                features: Vec::new(),
                outcome,
                target: TargetDetail::Puzzleboard(PuzzleboardDetail {
                    corners: Vec::new(),
                    edge_bits: Vec::new(),
                    decode: None,
                    quality: CornerQuality::default(),
                }),
                rejected: Vec::new(),
            });
        }
    }

    #[test]
    fn the_wire_shape_is_tagged_and_sparse() {
        let detail = DetectionDetail {
            features: Vec::new(),
            outcome: DetectionOutcome::NotFound {
                reason: "no board".to_string(),
            },
            target: TargetDetail::Chessboard(ChessboardDetail {
                corners: Vec::new(),
                cell_size_px: None,
                quality: CornerQuality::default(),
            }),
            rejected: vec![RejectedPoint {
                image_xy: [1.0, 2.0],
                reason: PointRejection::Unlabelled { strength: 3.0 },
                world_xyz: None,
                grid: None,
                id: None,
            }],
        };
        assert_eq!(
            serde_json::to_value(&detail).unwrap(),
            serde_json::json!({
                "features": [],
                "outcome": { "status": "not_found", "reason": "no board" },
                "target": {
                    "kind": "chessboard",
                    "corners": [],
                    "quality": { "local_residual_count": 0 },
                },
                "rejected": [{
                    "image_xy": [1.0, 2.0],
                    "reason": { "kind": "unlabelled", "strength": 3.0 },
                }],
            })
        );
    }
}
