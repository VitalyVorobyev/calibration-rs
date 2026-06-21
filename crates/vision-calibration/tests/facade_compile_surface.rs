#![allow(missing_docs)]

use vision_calibration::*;

#[test]
fn facade_modules_compile_with_glob_import() {
    // Session type wiring for all public problem modules.
    let _planar_session: Option<
        session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
    > = None;
    let _scheimpflug_session: Option<
        session::CalibrationSession<scheimpflug_intrinsics::ScheimpflugIntrinsicsProblem>,
    > = None;
    let _single_handeye_session: Option<
        session::CalibrationSession<single_cam_handeye::SingleCamHandeyeProblem>,
    > = None;
    let _rig_extrinsics_session: Option<
        session::CalibrationSession<rig_extrinsics::RigExtrinsicsProblem>,
    > = None;
    let _rig_handeye_session: Option<session::CalibrationSession<rig_handeye::RigHandeyeProblem>> =
        None;
    let _laserline_session: Option<
        session::CalibrationSession<laserline_device::LaserlineDeviceProblem>,
    > = None;

    // Planar intrinsics API.
    let _planar_step_init: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
        Option<planar_intrinsics::IntrinsicsInitOptions>,
    ) -> Result<planar_intrinsics::PlanarInitResult, Error> = planar_intrinsics::step_init;
    let _planar_step_optimize: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
        Option<planar_intrinsics::IntrinsicsOptimizeOptions>,
    ) -> Result<planar_intrinsics::PlanarOptimizeResult, Error> = planar_intrinsics::step_optimize;
    let _planar_step_filter: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
        planar_intrinsics::FilterOptions,
    ) -> Result<(), Error> = planar_intrinsics::step_filter;
    let _planar_run: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
    ) -> Result<(), Error> = planar_intrinsics::run_calibration;

    // Scheimpflug intrinsics API.
    let _scheimpflug_run: fn(
        &mut session::CalibrationSession<scheimpflug_intrinsics::ScheimpflugIntrinsicsProblem>,
        Option<scheimpflug_intrinsics::ScheimpflugIntrinsicsConfig>,
    ) -> Result<(), Error> = scheimpflug_intrinsics::run_calibration;

    // Single-camera hand-eye API.
    let _single_handeye_run: fn(
        &mut session::CalibrationSession<single_cam_handeye::SingleCamHandeyeProblem>,
    ) -> Result<(), Error> = single_cam_handeye::run_calibration;

    // Rig extrinsics API.
    let _rig_extrinsics_run: fn(
        &mut session::CalibrationSession<rig_extrinsics::RigExtrinsicsProblem>,
    ) -> Result<(), Error> = rig_extrinsics::run_calibration;

    // Rig hand-eye API.
    let _rig_handeye_run: fn(
        &mut session::CalibrationSession<rig_handeye::RigHandeyeProblem>,
    ) -> Result<(), Error> = rig_handeye::run_calibration;

    // Laserline device API.
    let _laserline_run: fn(
        &mut session::CalibrationSession<laserline_device::LaserlineDeviceProblem>,
        Option<laserline_device::LaserlineDeviceConfig>,
    ) -> Result<(), Error> = laserline_device::run_calibration;
}

#[test]
fn prelude_compiles_for_hello_world_surface() {
    use vision_calibration::prelude::*;

    let _session: Option<CalibrationSession<PlanarIntrinsicsProblem>> = None;
    let _runner: fn(
        &mut CalibrationSession<PlanarIntrinsicsProblem>,
    ) -> Result<(), vision_calibration::Error> = run_planar_intrinsics;
}

#[test]
fn mvg_module_surface_compiles() {
    use vision_calibration::mvg;

    // Lock the key re-exported paths (multiple-view geometry pipelines).
    let _ = mvg::pose_recovery::recover_relative_pose;
    let _ = mvg::robust::recover_relative_pose_robust;
    let _ = mvg::triangulation::triangulate_nview;
    let _ = mvg::rectification::rectify_stereo_pair;
    let _ = mvg::homography::decompose_homography;
    let _ = mvg::dense::match_block;
    let _: Option<mvg::rectification::StereoRectification> = None;
    let _: Option<mvg::dense::DisparityMap> = None;
    let _: Option<mvg::dense::BlockMatchOptions> = None;
    let _: Option<mvg::types::Correspondence2D> = None;
    let _: Option<mvg::MvgError> = None;
    let _: Option<mvg::error::Result<()>> = None;
}

#[test]
fn mvg_rectification_runs_through_facade() {
    use nalgebra::{Matrix3, Translation3, UnitQuaternion};
    use vision_calibration::core::Iso3;
    use vision_calibration::mvg::rectification::{
        RectifyCamera, RectifyOptions, rectify_stereo_pair,
    };

    let k = Matrix3::new(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0);
    let pose = Iso3::from_parts(
        Translation3::new(-1.0, 0.0, 0.0),
        UnitQuaternion::identity(),
    );
    let rect = rectify_stereo_pair(
        &RectifyCamera::pinhole(k),
        &RectifyCamera::pinhole(k),
        &pose,
        &RectifyOptions::default(),
    )
    .expect("rectify a trivial pinhole pair");
    assert!((rect.baseline - 1.0).abs() < 1e-12);
}

#[cfg(feature = "refine")]
#[test]
fn mvg_bundle_adjust_reachable_under_refine() {
    // The `refine` facade feature surfaces frozen-intrinsics bundle adjustment.
    let _ = vision_calibration::mvg::bundle_adjust::bundle_adjust;
}
