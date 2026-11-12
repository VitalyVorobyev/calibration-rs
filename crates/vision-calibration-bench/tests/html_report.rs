//! The HTML report renders the export of every problem type: one page per
//! calibration, with the quality checks and one section per camera.

use vision_calibration::Error;
use vision_calibration::analysis::{ReportOptions, html_report};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::laserline_device::{
    LaserlineDeviceProblem, run_calibration as run_laserline_device,
};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsProblem, run_calibration as run_planar_intrinsics,
};
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, run_calibration as run_rig_extrinsics,
};
use vision_calibration::rig_handeye::{RigHandeyeProblem, run_calibration as run_rig_handeye};
use vision_calibration::rig_handeye_laserline::{
    RigHandeyeLaserlineProblem, run_calibration as run_rig_handeye_laserline,
};
use vision_calibration::rig_laserline_device::{
    RigLaserlineDeviceProblem, run_calibration as run_rig_laserline_device,
};
use vision_calibration::session::{CalibrationSession, ProblemType};
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeProblem, run_calibration as run_single_cam_handeye,
};
use vision_calibration_bench::solver::scenes::{
    CameraKind, Preset, Problem, Scene, SceneData, scene_specs,
};

/// The quick preset's pinhole scene of `problem`.
fn scene(problem: Problem) -> Scene {
    let spec = scene_specs(Preset::Quick)
        .into_iter()
        .find(|s| s.problem == problem && s.camera == CameraKind::Pinhole)
        .expect("the quick preset has a pinhole scene of every problem");
    Scene::build(&spec).expect("scene builds")
}

/// Calibrates `input`, renders the report and checks what every workflow's
/// page holds: the title, the quality checks, and a section with a parameter
/// table and figures per camera.
fn check_report<P: ProblemType>(
    name: &str,
    image_sizes: &[[u32; 2]],
    config: P::Config,
    input: P::Input,
    run: fn(&mut CalibrationSession<P>) -> Result<(), Error>,
) where
    P::Export: CalibratedCameras,
{
    let cameras = image_sizes.len();
    let mut session = CalibrationSession::<P>::new();
    session.set_config(config).unwrap();
    session.set_input(input).unwrap();
    run(&mut session).unwrap_or_else(|e| panic!("{name}: calibration failed: {e}"));
    let export = session.export().unwrap();
    assert_eq!(export.camera_params().len(), cameras, "{name}: cameras");

    let mut options = ReportOptions::default();
    options.title = format!("{name} report");
    options.camera_names = (0..cameras).map(|i| format!("sensor{i}")).collect();
    let page = html_report(&export, image_sizes, &options)
        .unwrap_or_else(|e| panic!("{name}: report failed: {e}"));

    assert!(page.starts_with("<!doctype html>"), "{name}");
    assert!(
        page.contains(&format!("<title>{name} report</title>")),
        "{name}"
    );
    assert!(page.contains("Quality checks"), "{name}: checks");
    assert_eq!(
        page.matches("<section class=\"camera\">").count(),
        cameras,
        "{name}: one section per camera"
    );
    for i in 0..cameras {
        assert!(
            page.contains(&format!("Camera {i} (sensor{i})")),
            "{name}: camera {i}"
        );
    }
    for label in ["fx", "fy", "cx", "cy"] {
        assert_eq!(
            page.matches(&format!("<tr><td>{label}</td>")).count(),
            cameras,
            "{name}: {label} per camera"
        );
    }
    assert_eq!(
        page.matches("<h3>Parameters</h3>").count(),
        cameras,
        "{name}: parameters"
    );
    assert!(page.contains("<svg"), "{name}: figures");
    // One self-contained page: nothing loads from elsewhere.
    for external in ["<script", "src=", "href=", "@import", "url("] {
        assert!(!page.contains(external), "{name}: {external}");
    }
}

#[test]
fn the_report_renders_every_problem_type() {
    let s = scene(Problem::PlanarIntrinsics);
    let SceneData::Planar { input, config } = s.data else {
        panic!("planar scene");
    };
    check_report::<PlanarIntrinsicsProblem>(
        "planar_intrinsics",
        &s.image_sizes,
        config,
        input,
        run_planar_intrinsics,
    );

    let s = scene(Problem::SingleCamHandeye);
    let SceneData::SingleCamHandeye { input, config } = s.data else {
        panic!("single-camera hand-eye scene");
    };
    check_report::<SingleCamHandeyeProblem>(
        "single_cam_handeye",
        &s.image_sizes,
        config,
        input,
        run_single_cam_handeye,
    );

    let s = scene(Problem::LaserlineDevice);
    let SceneData::Laserline { input, config } = s.data else {
        panic!("laserline scene");
    };
    check_report::<LaserlineDeviceProblem>(
        "laserline_device",
        &s.image_sizes,
        config,
        input,
        run_laserline_device,
    );

    let s = scene(Problem::RigExtrinsics);
    let SceneData::Rig { input, config, .. } = s.data else {
        panic!("rig scene");
    };
    assert!(s.image_sizes.len() >= 2);
    check_report::<RigExtrinsicsProblem>(
        "rig_extrinsics",
        &s.image_sizes,
        config,
        input,
        run_rig_extrinsics,
    );

    let s = scene(Problem::RigHandeye);
    let SceneData::RigHandeye { input, config } = s.data else {
        panic!("rig hand-eye scene");
    };
    check_report::<RigHandeyeProblem>(
        "rig_handeye",
        &s.image_sizes,
        config,
        input,
        run_rig_handeye,
    );

    let s = scene(Problem::RigLaserlineDevice);
    let SceneData::RigLaserline { input, config } = s.data else {
        panic!("rig laserline scene");
    };
    check_report::<RigLaserlineDeviceProblem>(
        "rig_laserline_device",
        &s.image_sizes,
        config,
        input,
        run_rig_laserline_device,
    );

    let s = scene(Problem::RigHandeyeLaserline);
    let SceneData::RigHandeyeLaserline { input, config } = s.data else {
        panic!("rig hand-eye laserline scene");
    };
    check_report::<RigHandeyeLaserlineProblem>(
        "rig_handeye_laserline",
        &s.image_sizes,
        config,
        input,
        run_rig_handeye_laserline,
    );
}
