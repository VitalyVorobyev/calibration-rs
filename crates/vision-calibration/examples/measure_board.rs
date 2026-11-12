//! Measure a board of known size with a stereo calibration, error bars included.
//!
//! Calibrates a stereo sample on half of its views, triangulates the
//! chessboard corners of the other half, and measures the distances between
//! corners of a row or column. The board's squares have a known size, so
//! every distance has a known truth, and the example compares the error with
//! the σ the library states, with and without the calibration's share in it.
//!
//! Two samples can be measured: OpenCV's fisheye stereo test set (the
//! default; 24.4 mm squares, calibrated with a fisheye model) and OpenCV's
//! pinhole stereo sample (`opencv_stereo`, whose square size is not
//! published, so its lengths are in squares). The sample is downloaded from
//! its publisher on the first run (set `VISION_CALIBRATION_DATA` to choose
//! the cache directory).
//!
//! ```bash
//! cargo run --release -p vision-calibration --features fetch --example measure_board
//! cargo run --release -p vision-calibration --features fetch --example measure_board -- opencv_stereo
//! ```

#[path = "support/board_measure.rs"]
mod board_measure;

use anyhow::{Context, Result, ensure};
use board_measure::{
    MAX_SQUARES, ViewMeasurement, measure_view, summarize, without_calibration_share,
};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::control::RunControl;
use vision_calibration::core::{
    CameraModelSpec, DistortionKind, ProjectionKind, RigDataset, SensorKind,
};
use vision_calibration::dataset::{DatasetSpec, TargetSpec, samples};
use vision_calibration::dataset_runner::{Detection, build_rig_extrinsics_input};
use vision_calibration::detect::FsDetectionCache;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsProblem, run_calibration,
};
use vision_calibration::session::CalibrationSession;

/// A stereo sample this example measures: the camera model it is calibrated
/// with, and the unit its lengths are printed in.
struct Sample {
    id: &'static str,
    model: CameraModelSpec,
    unit: &'static str,
    /// Metres per printed unit.
    unit_m: f64,
    /// Decimals the lengths are printed with.
    decimals: usize,
}

fn sample(id: &str) -> Result<Sample> {
    match id {
        // Fisheye lenses: the equidistant projection with Kannala-Brandt
        // distortion. 24.4 mm squares.
        "opencv_fisheye_stereo" => Ok(Sample {
            id: "opencv_fisheye_stereo",
            model: CameraModelSpec::new(
                ProjectionKind::Equidistant,
                DistortionKind::KannalaBrandt4,
                SensorKind::Identity,
            ),
            unit: "mm",
            unit_m: 1e-3,
            decimals: 3,
        }),
        // Pinhole lenses with Brown-Conrady distortion, the default model.
        // The board is declared with squares of 1.
        "opencv_stereo" => Ok(Sample {
            id: "opencv_stereo",
            model: CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
            unit: "squares",
            unit_m: 1.0,
            decimals: 4,
        }),
        other => anyhow::bail!(
            "measure_board measures opencv_fisheye_stereo or opencv_stereo, not {other:?}"
        ),
    }
}

fn main() -> Result<()> {
    // 1. The sample: synchronised stereo views of a chessboard of known size.
    let id = std::env::args().nth(1);
    let sample = sample(id.as_deref().unwrap_or("opencv_fisheye_stereo"))?;
    let manifest = samples::fetch(sample.id, &samples::default_cache_dir()?)?;
    let dir = manifest
        .parent()
        .context("a fetched manifest sits in its sample's directory")?;
    let spec = DatasetSpec::from_path(&manifest)?;
    let TargetSpec::Chessboard { square_size_m, .. } = spec.target else {
        anyhow::bail!("the sample's target is a chessboard");
    };
    let cache = FsDetectionCache::new(dir.join(".detections"));
    let run = build_rig_extrinsics_input(
        &spec,
        dir,
        Detection::Cached(&cache),
        &RunControl::default(),
    )?;
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

    // 3. Calibrate the rig with the sample's camera model.
    let mut config = RigExtrinsicsConfig::default();
    config.camera.model = sample.model;
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
         board planarity rms {:.p$} {}",
        with.len(),
        held_out.len(),
        with.iter().map(|v| v.corners).min().unwrap_or(0),
        with.iter().map(|v| v.corners).max().unwrap_or(0),
        plane_rms / sample.unit_m,
        sample.unit,
        p = sample.decimals,
    );

    // 5. Error against the σ the library states, per length.
    let full = summarize(&with)?;
    let only = summarize(&without)?;
    println!(
        "\nsquares  pairs  mean err   rms err  mean σ   z rms | σ without calibration: mean σ   z rms   ({})",
        sample.unit
    );
    for k in 1..=MAX_SQUARES {
        let (f, o) = (&full[k - 1], &only[k - 1]);
        if f.n == 0 {
            continue;
        }
        println!(
            "{k:7}  {:5}  {:+8.p$}  {:8.p$}  {:6.p$}  {:6.2} | {:35.p$}  {:6.2}",
            f.n,
            f.mean_error() / sample.unit_m,
            f.rms_error() / sample.unit_m,
            f.mean_sigma() / sample.unit_m,
            f.z_rms(),
            o.mean_sigma() / sample.unit_m,
            o.z_rms(),
            p = sample.decimals,
        );
    }
    Ok(())
}
