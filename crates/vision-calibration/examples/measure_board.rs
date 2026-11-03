//! Measure a board of known size with a stereo calibration, error bars included.
//!
//! Calibrates the fisheye stereo pair of OpenCV's own test set on half of its
//! views, triangulates the chessboard corners of the other half, and measures
//! the distances between corners of a row or column. The board's squares are
//! 24.4 mm, so every distance has a known truth, and the example compares the
//! error with the σ the library states, with and without the calibration's
//! share in it.
//!
//! The sample is downloaded from its publisher on the first run (set
//! `VISION_CALIBRATION_DATA` to choose the cache directory).
//!
//! ```bash
//! cargo run --release -p vision-calibration --features fetch --example measure_board
//! ```

#[path = "support/board_measure.rs"]
mod board_measure;

use anyhow::{Context, Result, ensure};
use board_measure::{
    MAX_SQUARES, ViewMeasurement, measure_view, summarize, without_calibration_share,
};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{
    CameraModelSpec, DistortionKind, ProjectionKind, RigDataset, SensorKind,
};
use vision_calibration::dataset::{DatasetSpec, TargetSpec, samples};
use vision_calibration::dataset_runner::{Detection, build_rig_extrinsics_input};
use vision_calibration::detect::FsDetectionCache;
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsProblem, run_calibration,
};

fn main() -> Result<()> {
    // 1. The sample: 34 stereo views of a 6×8-corner chessboard, 24.4 mm squares.
    let manifest = samples::fetch("opencv_fisheye_stereo", &samples::default_cache_dir()?)?;
    let dir = manifest
        .parent()
        .context("a fetched manifest sits in its sample's directory")?;
    let spec = DatasetSpec::from_path(&manifest)?;
    let TargetSpec::Chessboard { square_size_m, .. } = spec.target else {
        anyhow::bail!("the sample's target is a chessboard");
    };
    let cache = FsDetectionCache::new(dir.join(".detections"));
    let run = build_rig_extrinsics_input(&spec, dir, Detection::Cached(&cache))?;
    println!("{} of {} views usable", run.usable_views, run.total_views);

    // 2. Split the views: even ones calibrate, odd ones are held out. Each view
    //    keeps both cameras' detections, so the pairing stays intact.
    let (calibration, held_out): (Vec<_>, Vec<_>) = run
        .input
        .views
        .into_iter()
        .enumerate()
        .partition(|(i, _)| i % 2 == 0);
    let calibration = RigDataset::new(calibration.into_iter().map(|(_, v)| v).collect(), 2)?;
    let held_out: Vec<_> = held_out.into_iter().map(|(_, v)| v).collect();
    println!(
        "{} calibration views, {} held out",
        calibration.num_views(),
        held_out.len()
    );

    // 3. Calibrate the rig with the fisheye model (equidistant projection,
    //    Kannala-Brandt distortion).
    let mut config = RigExtrinsicsConfig::default();
    config.camera.model = CameraModelSpec::new(
        ProjectionKind::Equidistant,
        DistortionKind::KannalaBrandt4,
        SensorKind::Identity,
    );
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_config(config)?;
    session.set_input(calibration)?;
    run_calibration(&mut session)?;
    let export = session.export()?;
    let rig = export.calibrated_rig()?;
    let covariance = rig
        .uncertainty()
        .covariance()
        .context("the calibration has no covariance")?;
    println!(
        "calibration: mean reprojection error {:.3} px, view scatter {}, σ̂ {:.3} / {:.3} px",
        export.mean_reproj_error,
        covariance
            .view_scatter
            .map_or("none".to_string(), |s| format!("{s:.2}")),
        rig.reprojection_noise(0)?.xx().sqrt(),
        rig.reprojection_noise(1)?.xx().sqrt(),
    );

    // 4. Measure every held-out view with the full σ and with the observations'
    //    share alone.
    let bare = without_calibration_share(&rig)?;
    let mut with: Vec<ViewMeasurement> = Vec::new();
    let mut without: Vec<ViewMeasurement> = Vec::new();
    for view in &held_out {
        if let (Some(a), Some(b)) = (
            measure_view(&rig, &view.obs, square_size_m)?,
            measure_view(&bare, &view.obs, square_size_m)?,
        ) {
            with.push(a);
            without.push(b);
        }
    }
    ensure!(!with.is_empty(), "no held-out view could be measured");
    let plane_rms: f64 = with.iter().map(|v| v.plane_rms).sum::<f64>() / with.len() as f64;
    println!(
        "{} of {} held-out views measured (both cameras need the board); corners shared per view: {}..{}; \
         board planarity rms {:.3} mm",
        with.len(),
        held_out.len(),
        with.iter().map(|v| v.corners).min().unwrap_or(0),
        with.iter().map(|v| v.corners).max().unwrap_or(0),
        plane_rms * 1e3
    );

    // 5. Error against the σ the library states, per length.
    let full = summarize(&with)?;
    let only = summarize(&without)?;
    println!(
        "\nsquares  pairs  mean err   rms err  mean σ   z rms | σ without calibration: mean σ   z rms   (mm)"
    );
    for k in 1..=MAX_SQUARES {
        let (f, o) = (&full[k - 1], &only[k - 1]);
        if f.n == 0 {
            continue;
        }
        println!(
            "{k:7}  {:5}  {:+8.3}  {:8.3}  {:6.3}  {:6.2} | {:35.3}  {:6.2}",
            f.n,
            f.mean_error() * 1e3,
            f.rms_error() * 1e3,
            f.mean_sigma() * 1e3,
            f.z_rms(),
            o.mean_sigma() * 1e3,
            o.z_rms(),
        );
    }
    Ok(())
}
