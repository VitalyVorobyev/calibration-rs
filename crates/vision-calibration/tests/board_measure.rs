//! The `measure_board` example's measurement on a synthetic stereo board.
//!
//! A calibrated synthetic rig measures a held-out board of known square size;
//! the distances must come out near the truth with a σ that covers the error,
//! and the calibration's share must make the σ larger.

#[path = "../examples/support/board_measure.rs"]
mod board_measure;
mod common;

use board_measure::{MAX_SQUARES, measure_view, summarize, without_calibration_share};
use common::{calibrated_stereo_rig, stereo_cameras, stereo_noisy_pixel};
use vision_calibration::core::{CorrespondenceView, Iso3, Pt3, RigViewObs};
use vision_calibration::synthetic::planar;
use vision_calibration::synthetic::poses::make_iso;

const COLS: usize = 8;
const ROWS: usize = 6;
const SQUARE: f64 = 0.025;

/// A held-out view: the board 0.7 m ahead and tilted, noisy pixels in both cameras.
fn held_out_view(seed: u64) -> RigViewObs {
    let cameras = stereo_cameras();
    let board = planar::grid_points(COLS, ROWS, SQUARE);
    let rig_se3_board: Iso3 = make_iso((0.15, -0.12, 0.05), (-0.09, -0.06, 0.70));
    RigViewObs {
        cameras: (0..2)
            .map(|c| {
                let pixels = board
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let key = (seed << 32) + 30_000_000 + (c * 1000 + i) as u64;
                        let at: Pt3 = rig_se3_board * p;
                        stereo_noisy_pixel(&cameras, c, &at, key)
                    })
                    .collect();
                Some(CorrespondenceView::new(board.clone(), pixels).unwrap())
            })
            .collect(),
    }
}

#[test]
fn board_distances_agree_with_the_square_size() {
    let rig = calibrated_stereo_rig(&stereo_cameras(), 11);
    let view = held_out_view(11);

    let measured = measure_view(&rig, &view, SQUARE)
        .unwrap()
        .expect("both cameras see the board");
    assert_eq!(measured.corners, COLS * ROWS);
    // Row pairs: 6 rows x (7+6+...+1); column pairs: 8 columns x (5+4+...+1).
    assert_eq!(measured.distances.len(), 6 * 28 + 8 * 15);
    for m in &measured.distances {
        assert!((m.truth - m.squares as f64 * SQUARE).abs() < 1e-12);
        assert!(m.error().abs() < 5.0 * m.sigma, "{m:?}");
    }
    assert!(
        measured.plane_rms < 5e-3,
        "plane rms {}",
        measured.plane_rms
    );

    let full = summarize(std::slice::from_ref(&measured)).unwrap();
    let bare = without_calibration_share(&rig).unwrap();
    let only = summarize(&[measure_view(&bare, &view, SQUARE).unwrap().unwrap()]).unwrap();
    eprintln!(
        "z RMS by length (1..{MAX_SQUARES} squares): with the calibration {:.2?}, observations \
         only {:.2?}",
        full.iter().map(|s| s.z_rms()).collect::<Vec<_>>(),
        only.iter().map(|s| s.z_rms()).collect::<Vec<_>>()
    );
    for k in 0..MAX_SQUARES {
        assert!(full[k].n > 0 && full[k].n == only[k].n);
        // The calibration's share can only add to the observations' σ.
        assert!(full[k].mean_sigma() >= only[k].mean_sigma());
        assert!(full[k].z_rms() < 3.0, "k={}: z {}", k + 1, full[k].z_rms());
    }
}

#[test]
fn a_view_with_one_camera_is_skipped() {
    let rig = calibrated_stereo_rig(&stereo_cameras(), 11);
    let mut view = held_out_view(11);
    view.cameras[1] = None;
    assert!(measure_view(&rig, &view, SQUARE).unwrap().is_none());
}
