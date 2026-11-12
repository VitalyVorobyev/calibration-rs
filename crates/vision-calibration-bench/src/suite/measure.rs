//! A measurement check: the board of an entry's held-out views, measured
//! with a stereo calibration of its other views.
//!
//! This is the `measure_board` route of the facade's example of that name
//! (the book's "Measuring a Board with a Stereo Calibration"), run on the
//! entry's frozen observations instead of fresh detections, so it needs no
//! images. The example's measurement code is compiled in here unchanged, so
//! the suite gates exactly the route the book walks through.
//!
//! The views of the observation file split by index: the even ones
//! calibrate a two-camera rig with the entry's config, the odd ones are held
//! out. Each held-out view that both cameras saw is triangulated in one call,
//! every pixel stated at the reprojection noise level the calibration reports
//! for its camera, and every pair of corners on one board row or column, 1 to
//! [`MAX_SQUARES`] squares apart, is measured with `metrology::distance`
//! against the board's known size. The result is the measuring calibration's
//! summary and, per held-out view and over all of them, per length in
//! squares: the number of pairs, the mean and RMS error, the mean stated σ
//! and the RMS of the error over the σ (about 1 when the σ is honest).
//!
//! A drift check compares every number with the reference
//! ([`super::Gate::Exact`]), or the mean error of each row in units of the
//! row's mean stated σ ([`super::Gate::Sigma`], see [`measurement_changes`]).

#[path = "../../../vision-calibration/examples/support/board_measure.rs"]
mod board_measure;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use vision_calibration_core::RigDataset;
use vision_calibration_optim::SolverBackend;
use vision_calibration_pipeline::common::CalibratedCameras;
use vision_calibration_pipeline::rig_extrinsics::{self, RigExtrinsicsProblem};

pub use board_measure::MAX_SQUARES;
use board_measure::{Stats, measure_view, summarize};

use super::observations::calibrate_typed;
use super::significance::ParameterChange;
use super::{ObservationSet, ProblemInput};

/// `measure_board` of a suite entry: what the observation file does not say
/// about the board.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasureBoard {
    /// The board's square size, in the unit of the observations' target
    /// points (metres; 1 for a board declared with squares of 1). It sorts
    /// the corner pairs into rows, columns and lengths in squares; the true
    /// length of a pair is the distance of its target points.
    pub square_size_m: f64,
}

impl MeasureBoard {
    /// Checks the square size is finite and positive.
    ///
    /// # Errors
    ///
    /// When it is not.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.square_size_m.is_finite() && self.square_size_m > 0.0,
            "square_size_m must be finite and positive, got {}",
            self.square_size_m
        );
        Ok(())
    }
}

/// The measurement of an entry's held-out boards.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoardMeasurement {
    /// The calibration the boards were measured with.
    pub calibration: MeasuringCalibration,
    /// Each held-out view both cameras saw, in view order.
    pub views: Vec<MeasuredView>,
    /// Held-out views one camera did not see, so not measured.
    pub unmeasured_views: Vec<usize>,
    /// Every measured view's distances together, per length.
    pub lengths: Vec<LengthStats>,
}

/// The stereo calibration of the even views.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuringCalibration {
    /// The views it was calibrated from (indices into the observation file).
    pub views: Vec<usize>,
    /// Its mean reprojection error, pixels.
    pub mean_reproj_error: f64,
    /// Its view scatter, the factor its covariance is scaled by, when the
    /// views scatter significantly more than the corner noise predicts.
    pub view_scatter: Option<f64>,
    /// Per camera, the reprojection noise level σ̂ it reports, pixels: the
    /// stated noise of every measured pixel of that camera.
    pub pixel_sigma: Vec<f64>,
}

/// One measured held-out view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredView {
    /// The view's index in the observation file.
    pub view: usize,
    /// Corners both cameras saw, triangulated together.
    pub corners: usize,
    /// RMS distance of the triangulated corners to their best-fit plane, in
    /// the target's unit: a planarity check.
    pub plane_rms: f64,
    /// The view's distances, per length.
    pub lengths: Vec<LengthStats>,
}

/// The distances of one length, in the target's unit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LengthStats {
    /// Squares between the two corners.
    pub squares: usize,
    /// Corner pairs measured.
    pub pairs: usize,
    /// Mean of measured minus true length.
    pub mean_error: f64,
    /// RMS of measured minus true length.
    pub rms_error: f64,
    /// Mean stated σ.
    pub mean_sigma: f64,
    /// RMS of the error over its stated σ: about 1 when the σ is honest.
    pub z_rms: f64,
}

/// The lengths of `stats` that have pairs, shortest first.
fn lengths(stats: &[Stats; MAX_SQUARES]) -> Vec<LengthStats> {
    stats
        .iter()
        .enumerate()
        .filter(|(_, s)| s.n > 0)
        .map(|(k, s)| LengthStats {
            squares: k + 1,
            pairs: s.n,
            mean_error: s.mean_error(),
            rms_error: s.rms_error(),
            mean_sigma: s.mean_sigma(),
            z_rms: s.z_rms(),
        })
        .collect()
}

/// Calibrate the even views of `set` with the config `overrides` (on
/// `backend`, when given) and measure the board in the odd ones.
///
/// # Errors
///
/// Fails for an observation file that is not a two-camera `rig_extrinsics`
/// input, when the calibration fails or has no covariance, when a
/// measurement fails, and when no held-out view can be measured.
pub fn measure_board(
    set: &ObservationSet,
    overrides: Option<&Value>,
    backend: Option<SolverBackend>,
    spec: &MeasureBoard,
) -> Result<BoardMeasurement> {
    let ProblemInput::RigExtrinsics(input) = &set.input else {
        bail!(
            "measure_board measures a stereo rig_extrinsics entry, not a {} one",
            set.input.problem_name()
        );
    };
    ensure!(
        input.num_cameras == 2,
        "measure_board measures a stereo pair; the entry has {} cameras",
        input.num_cameras
    );

    // Even views calibrate, odd ones are held out; each keeps both cameras'
    // observations, so the pairing stays intact.
    let (calibration, held_out): (Vec<_>, Vec<_>) = input
        .views
        .iter()
        .enumerate()
        .partition(|(i, _)| i % 2 == 0);
    let dataset = RigDataset::new(
        calibration.iter().map(|(_, v)| (*v).clone()).collect(),
        input.num_cameras,
    )?;
    let export = calibrate_typed::<RigExtrinsicsProblem>(
        "rig_extrinsics",
        &dataset,
        overrides,
        backend,
        rig_extrinsics::run_calibration,
    )
    .context("calibrating the even views")?;
    let rig = export.calibrated_rig()?;
    let covariance = rig
        .uncertainty()
        .covariance()
        .context("the measuring calibration has no covariance")?;
    let pixel_sigma = (0..rig.num_cameras())
        .map(|c| Ok(rig.reprojection_noise(c)?.xx().sqrt()))
        .collect::<Result<Vec<_>>>()?;
    let summary = MeasuringCalibration {
        views: calibration.iter().map(|(i, _)| *i).collect(),
        mean_reproj_error: export.mean_reproj_error,
        view_scatter: covariance.view_scatter,
        pixel_sigma,
    };

    let mut measured = Vec::new();
    let mut views = Vec::new();
    let mut unmeasured_views = Vec::new();
    for (index, view) in held_out {
        let Some(m) = measure_view(&rig, &view.obs, spec.square_size_m)
            .with_context(|| format!("measuring view {index}"))?
        else {
            unmeasured_views.push(index);
            continue;
        };
        views.push(MeasuredView {
            view: index,
            corners: m.corners,
            plane_rms: m.plane_rms,
            lengths: lengths(&summarize(std::slice::from_ref(&m))?),
        });
        measured.push(m);
    }
    ensure!(!measured.is_empty(), "no held-out view could be measured");
    Ok(BoardMeasurement {
        calibration: summary,
        lengths: lengths(&summarize(&measured)?),
        views,
        unmeasured_views,
    })
}

/// The views, corners and pairs a measurement has: what must not change
/// for its numbers to be comparable.
fn shape(m: &BoardMeasurement) -> Vec<(Option<usize>, usize, usize)> {
    let rows = |view: Option<usize>, lengths: &[LengthStats]| {
        lengths
            .iter()
            .map(move |l| (view, l.squares, l.pairs))
            .collect::<Vec<_>>()
    };
    m.views
        .iter()
        .flat_map(|v| rows(Some(v.view), &v.lengths))
        .chain(rows(None, &m.lengths))
        .collect()
}

/// The change of every row's mean error from the reference, in units of
/// the reference row's mean stated σ: per held-out view and length
/// (`measure_board.view<v>.<k>sq`), and over all views (`measure_board.<k>sq`).
///
/// The mean stated σ bounds the σ of the mean of a row's lengths, however
/// correlated they are, so a change within `k` of it is within `k` σ of
/// that mean. Only the lengths are compared; the stated σ themselves are
/// checked by the exact gate.
///
/// # Errors
///
/// When the measured views, corners or pairs differ from the reference's:
/// the rows are then not comparable.
pub fn measurement_changes(
    reference: &BoardMeasurement,
    actual: &BoardMeasurement,
) -> Result<Vec<ParameterChange>> {
    ensure!(
        shape(reference) == shape(actual),
        "the measured views or corner pairs differ from the reference's"
    );
    let rows = |view: Option<usize>, r: &[LengthStats], a: &[LengthStats]| {
        r.iter()
            .zip(a)
            .map(move |(r, a)| ParameterChange {
                label: match view {
                    Some(v) => format!("measure_board.view{v}.{}sq", r.squares),
                    None => format!("measure_board.{}sq", r.squares),
                },
                delta: a.mean_error - r.mean_error,
                sigma: r.mean_sigma,
            })
            .collect::<Vec<_>>()
    };
    Ok(reference
        .views
        .iter()
        .zip(&actual.views)
        .flat_map(|(r, a)| rows(Some(r.view), &r.lengths, &a.lengths))
        .chain(rows(None, &reference.lengths, &actual.lengths))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(squares: usize, mean_error: f64) -> LengthStats {
        LengthStats {
            squares,
            pairs: 10,
            mean_error,
            rms_error: 0.02,
            mean_sigma: 0.01,
            z_rms: 1.0,
        }
    }

    fn measurement(error: f64) -> BoardMeasurement {
        BoardMeasurement {
            calibration: MeasuringCalibration {
                views: vec![0, 2],
                mean_reproj_error: 0.2,
                view_scatter: None,
                pixel_sigma: vec![0.15, 0.15],
            },
            views: vec![MeasuredView {
                view: 1,
                corners: 54,
                plane_rms: 0.01,
                lengths: vec![row(1, error), row(2, 0.0)],
            }],
            unmeasured_views: vec![3],
            lengths: vec![row(1, error), row(2, 0.0)],
        }
    }

    #[test]
    fn square_size_must_be_positive() {
        assert!(
            MeasureBoard {
                square_size_m: 0.02
            }
            .validate()
            .is_ok()
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(MeasureBoard { square_size_m: bad }.validate().is_err());
        }
    }

    #[test]
    fn changes_are_in_units_of_the_rows_sigma() {
        let changes = measurement_changes(&measurement(0.0), &measurement(0.005)).unwrap();
        let labels: Vec<_> = changes.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "measure_board.view1.1sq",
                "measure_board.view1.2sq",
                "measure_board.1sq",
                "measure_board.2sq"
            ]
        );
        assert!((changes[0].z() - 0.5).abs() < 1e-12);
        assert_eq!(changes[1].z(), 0.0);
    }

    #[test]
    fn a_changed_pairing_is_not_comparable() {
        let mut other = measurement(0.0);
        other.views[0].lengths[1].pairs = 9;
        assert!(measurement_changes(&measurement(0.0), &other).is_err());
        let mut other = measurement(0.0);
        other.views[0].view = 5;
        assert!(measurement_changes(&measurement(0.0), &other).is_err());
    }

    #[test]
    fn a_measurement_roundtrips_through_json() {
        let m = measurement(0.001);
        let back: BoardMeasurement =
            serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
    }
}
