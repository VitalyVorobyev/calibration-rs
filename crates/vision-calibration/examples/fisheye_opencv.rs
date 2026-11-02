//! Fisheye calibration of OpenCV's own fisheye test set.
//!
//! 1. Load the corners of the left camera (34 views of an 8×6 chessboard,
//!    1280×800 fisheye images)
//! 2. Initialize a Kannala-Brandt camera from bearings, then refine it
//! 3. Calibrate the stereo pair as a rig, intrinsics refined jointly
//!
//! Run with: `cargo run -p vision-calibration --example fisheye_opencv`
//!
//! Dataset: `regression/observations/opencv_fisheye_left.json` and
//! `opencv_fisheye_stereo.json`, the corners OpenCV publishes with the
//! images of its fisheye calibration tests (`opencv_extra`,
//! `calib-3_stereo_from_JY`).

#[path = "support/observations.rs"]
mod observations;

use anyhow::Result;
use vision_calibration::core::{
    CameraModelSpec, CameraParams, DistortionKind, ProjectionKind, SensorKind, Uncertainty,
};
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsConfig, step_init, step_optimize};
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsProblem, run_calibration,
};

/// The Kannala-Brandt fisheye model of OpenCV's `cv::fisheye`.
fn fisheye() -> CameraModelSpec {
    CameraModelSpec::new(
        ProjectionKind::Equidistant,
        DistortionKind::KannalaBrandt4,
        SensorKind::Identity,
    )
}

fn print_camera(label: &str, camera: &CameraParams) {
    let k = camera.intrinsics;
    let d = camera.distortion.values();
    println!(
        "  {label}: fx={:.3} fy={:.3} cx={:.3} cy={:.3}  k1..k4={:+.6} {:+.6} {:+.6} {:+.6}",
        k.fx, k.fy, k.cx, k.cy, d[0], d[1], d[2], d[3]
    );
}

fn main() -> Result<()> {
    println!("=== Left camera ===");
    let input: PlanarDataset = observations::load_input("opencv_fisheye_left")?;
    println!("Loaded {} views", input.num_views());

    let mut config = PlanarIntrinsicsConfig::default();
    config.camera.model = fisheye();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_config(config)?;
    session.set_input(input)?;

    // The bearing-based initializer: no focal length or image size needed.
    let init = step_init(&mut session, None)?;
    print_camera("init     ", &init.camera);
    step_optimize(&mut session, None)?;
    let export = session.export()?;
    print_camera("optimized", &export.params.camera);
    println!(
        "  mean reprojection error {:.4} px",
        export.mean_reproj_error
    );
    if let Uncertainty::Available(cov) = &export.uncertainty {
        let sigma = |label: &str| cov.std_dev(label).unwrap_or(f64::NAN);
        println!(
            "  σ: fx {:.3} fy {:.3} cx {:.3} cy {:.3} px",
            sigma("fx"),
            sigma("fy"),
            sigma("cx"),
            sigma("cy")
        );
    }

    println!("\n=== Stereo pair ===");
    let mut config = RigExtrinsicsConfig::default();
    config.camera.model = fisheye();
    let mut rig = CalibrationSession::<RigExtrinsicsProblem>::new();
    rig.set_config(config)?;
    rig.set_input(observations::load_input("opencv_fisheye_stereo")?)?;
    run_calibration(&mut rig)?;
    let export = rig.export()?;
    for (i, camera) in export.cameras.iter().enumerate() {
        print_camera(&format!("camera {i} "), camera);
    }
    // cam_se3_rig[1] maps camera-0 coordinates into camera 1, the R, T of
    // OpenCV's stereoCalibrate.
    let t = export.cam_se3_rig[1].translation.vector;
    println!(
        "  T = [{:.6}, {:.6}, {:.6}] m, baseline {:.2} mm",
        t.x,
        t.y,
        t.z,
        t.norm() * 1e3
    );
    println!(
        "  mean reprojection error {:.4} px",
        export.mean_reproj_error
    );
    Ok(())
}
