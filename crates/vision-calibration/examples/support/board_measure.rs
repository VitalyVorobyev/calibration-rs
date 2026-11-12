//! Measuring a board of known geometry with a calibrated stereo rig.
//!
//! The logic of the `measure_board` example, kept apart so a test can run it
//! on synthetic data without the network, and the regression suite on the
//! committed observations of `opencv_stereo`.

// The example, its test and the regression suite each use a part of this
// module.
#![allow(dead_code)]

use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use vision_calibration::core::{CalibratedRig, PointCovariance, Pt3, RigViewObs, Uncertainty};
use vision_calibration::mvg::metrology::{PointSet, distance, fit_plane};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::mvg::types::Captures;

/// Longest distance measured, in squares.
pub const MAX_SQUARES: usize = 7;

/// One measured distance between two corners of a board row or column.
#[derive(Debug, Clone, Copy)]
pub struct Measured {
    /// Squares between the corners.
    pub squares: usize,
    /// The triangulated distance, metres.
    pub value: f64,
    /// Its standard deviation, metres.
    pub sigma: f64,
    /// The distance on the board, metres.
    pub truth: f64,
}

impl Measured {
    pub fn error(&self) -> f64 {
        self.value - self.truth
    }
}

/// What one held-out view yields.
#[derive(Debug, Clone)]
pub struct ViewMeasurement {
    /// Corners seen by both cameras.
    pub corners: usize,
    /// Every pair of corners on one board row or column, up to
    /// [`MAX_SQUARES`] squares apart.
    pub distances: Vec<Measured>,
    /// RMS distance of the triangulated corners to their best-fit plane,
    /// metres: a planarity check of the measurement.
    pub plane_rms: f64,
}

/// The corners both cameras saw in `view`, as tracks of two observations and
/// the corners' positions on the board. `None` when fewer than 3 are shared.
fn shared_corners(
    view: &RigViewObs,
    noise: &[PointCovariance],
) -> Option<(Vec<Vec<RigObservation>>, Vec<Pt3>)> {
    let (Some(Some(left)), Some(Some(right))) = (view.cameras.first(), view.cameras.get(1)) else {
        return None;
    };
    // A corner is identified by its position on the board, the same
    // coordinates in both cameras.
    let key = |p: &Pt3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let in_right: HashMap<_, _> = right
        .points_3d
        .iter()
        .zip(&right.points_2d)
        .map(|(p, px)| (key(p), *px))
        .collect();
    let mut tracks = Vec::new();
    let mut board = Vec::new();
    for (p, px) in left.points_3d.iter().zip(&left.points_2d) {
        if let Some(px_right) = in_right.get(&key(p)) {
            tracks.push(vec![
                RigObservation::new(0, 0, *px, noise[0]),
                RigObservation::new(0, 1, *px_right, noise[1]),
            ]);
            board.push(*p);
        }
    }
    (board.len() >= 3).then_some((tracks, board))
}

/// Triangulate the corners of one held-out view and measure the distances
/// between corners of a row or a column. The view is taken as one capture with
/// the rig at the origin, so the points come out in the rig frame.
///
/// `square` is the board's square size, metres. The pixel noise is the calibration's own reprojection level of each camera.
/// `Ok(None)` when the cameras share too few corners.
pub fn measure_view(
    rig: &CalibratedRig,
    view: &RigViewObs,
    square: f64,
) -> Result<Option<ViewMeasurement>> {
    let noise = [rig.reprojection_noise(0)?, rig.reprojection_noise(1)?];
    let Some((tracks, board)) = shared_corners(view, &noise) else {
        return Ok(None);
    };
    let triangulation = triangulate(
        rig,
        Captures::Rig,
        &tracks,
        &vision_calibration::control::RunControl::default(),
    )?;
    let set = PointSet::from_triangulation(&triangulation)?;

    let mut distances = Vec::new();
    for a in 0..board.len() {
        for b in (a + 1)..board.len() {
            let d = board[b] - board[a];
            let on_row = d.y.abs() < 1e-9 * square;
            let on_column = d.x.abs() < 1e-9 * square;
            let squares = (d.norm() / square).round() as usize;
            if on_row == on_column || squares == 0 || squares > MAX_SQUARES {
                continue;
            }
            let length = distance(&set, a, b)?;
            distances.push(Measured {
                squares,
                value: length.value,
                sigma: length.sigma,
                truth: d.norm(),
            });
        }
    }
    let all: Vec<usize> = (0..board.len()).collect();
    let plane = fit_plane(&set, &all).context("fitting the board plane")?;
    Ok(Some(ViewMeasurement {
        corners: board.len(),
        distances,
        plane_rms: plane.rms,
    }))
}

/// The same calibration with its covariance shrunk to nothing: the σ of the
/// observations alone, to show what the calibration's share adds.
pub fn without_calibration_share(rig: &CalibratedRig) -> Result<CalibratedRig> {
    let mut covariance = rig
        .uncertainty()
        .covariance()
        .context("the calibration has no covariance")?
        .clone();
    let n = covariance.labels.len();
    covariance.covariance = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1e-20 } else { 0.0 }).collect())
        .collect();
    Ok(CalibratedRig::new(
        rig.cameras().to_vec(),
        rig.cam_se3_rig().to_vec(),
        Uncertainty::Available(covariance),
    )?)
}

/// Running sums of the errors and stated σ of one distance class.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub n: usize,
    sum_error: f64,
    sum_error2: f64,
    sum_sigma: f64,
    sum_z2: f64,
}

impl Stats {
    pub fn add(&mut self, m: &Measured) {
        let z = m.error() / m.sigma;
        self.n += 1;
        self.sum_error += m.error();
        self.sum_error2 += m.error() * m.error();
        self.sum_sigma += m.sigma;
        self.sum_z2 += z * z;
    }

    fn mean(&self, sum: f64) -> f64 {
        sum / self.n as f64
    }

    /// Mean signed error, metres.
    pub fn mean_error(&self) -> f64 {
        self.mean(self.sum_error)
    }

    /// RMS error, metres.
    pub fn rms_error(&self) -> f64 {
        self.mean(self.sum_error2).sqrt()
    }

    /// Mean stated σ, metres.
    pub fn mean_sigma(&self) -> f64 {
        self.mean(self.sum_sigma)
    }

    /// RMS of error over stated σ: about 1 when the σ is honest.
    pub fn z_rms(&self) -> f64 {
        self.mean(self.sum_z2).sqrt()
    }
}

/// The statistics of `views`' distances per number of squares
/// (`result[k - 1]` for k squares).
pub fn summarize(views: &[ViewMeasurement]) -> Result<[Stats; MAX_SQUARES]> {
    let mut stats = [Stats::default(); MAX_SQUARES];
    for m in views.iter().flat_map(|v| &v.distances) {
        stats[m.squares - 1].add(m);
    }
    ensure!(
        stats.iter().any(|s| s.n > 0),
        "no distance could be measured"
    );
    Ok(stats)
}
