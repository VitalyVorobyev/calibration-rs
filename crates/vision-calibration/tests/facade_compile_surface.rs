#![allow(missing_docs)]

use vision_calibration::*;

#[test]
fn facade_modules_compile_with_glob_import() {
    // Session type wiring for all public problem modules.
    let _planar_session: Option<
        session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
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
    let _rig_laserline_session: Option<
        session::CalibrationSession<rig_laserline_device::RigLaserlineDeviceProblem>,
    > = None;
    let _rig_handeye_laserline_session: Option<
        session::CalibrationSession<rig_handeye_laserline::RigHandeyeLaserlineProblem>,
    > = None;

    // Every input type is constructible from its workflow module alone.
    let _rig_laserline_view = rig_laserline_device::RigLaserlineView {
        cameras: vec![None],
        laser_pixels: vec![None],
    };
    let _rig_laserline_dataset: Option<rig_laserline_device::RigLaserlineDataset> = None;
    let _rig_handeye_laserline_input = rig_handeye_laserline::RigHandeyeLaserlineInput {
        views: vec![rig_handeye_laserline::RigHandeyeLaserlineView {
            obs: rig_handeye_laserline::RigLaserlineView {
                cameras: vec![None],
                laser_pixels: vec![None],
            },
            meta: rig_handeye_laserline::RobotPoseMeta {
                base_se3_gripper: core::Iso3::identity(),
            },
        }],
        num_cameras: 1,
    };
}

/// Each problem names its `<Name>Input`, `<Name>Config`, output estimate and
/// `<Name>Export` in its own module.
fn assert_problem_types<P, I, C, O, E>()
where
    P: session::ProblemType<Input = I, Config = C, Output = O, Export = E>,
{
}

/// Every problem's `run_calibration` has the same shape.
fn assert_run_calibration<P: session::ProblemType>(
    _run: fn(&mut session::CalibrationSession<P>) -> Result<(), Error>,
) {
}

#[test]
fn every_problem_follows_the_vocabulary() {
    use laserline_device as ld;
    use planar_intrinsics as pi;
    use rig_extrinsics as re;
    use rig_handeye as rh;
    use rig_handeye_laserline as rhl;
    use rig_laserline_device as rld;
    use single_cam_handeye as sch;

    assert_problem_types::<
        pi::PlanarIntrinsicsProblem,
        pi::PlanarIntrinsicsInput,
        pi::PlanarIntrinsicsConfig,
        pi::PlanarIntrinsicsEstimate,
        pi::PlanarIntrinsicsExport,
    >();
    assert_problem_types::<
        sch::SingleCamHandeyeProblem,
        sch::SingleCamHandeyeInput,
        sch::SingleCamHandeyeConfig,
        sch::HandeyeEstimate,
        sch::SingleCamHandeyeExport,
    >();
    assert_problem_types::<
        ld::LaserlineDeviceProblem,
        ld::LaserlineDeviceInput,
        ld::LaserlineDeviceConfig,
        ld::LaserlineDeviceEstimate,
        ld::LaserlineDeviceExport,
    >();
    assert_problem_types::<
        re::RigExtrinsicsProblem,
        re::RigExtrinsicsInput,
        re::RigExtrinsicsConfig,
        re::RigExtrinsicsEstimate,
        re::RigExtrinsicsExport,
    >();
    assert_problem_types::<
        rh::RigHandeyeProblem,
        rh::RigHandeyeInput,
        rh::RigHandeyeConfig,
        rh::HandeyeEstimate,
        rh::RigHandeyeExport,
    >();
    assert_problem_types::<
        rld::RigLaserlineDeviceProblem,
        rld::RigLaserlineDeviceInput,
        rld::RigLaserlineDeviceConfig,
        rld::RigLaserlineEstimate,
        rld::RigLaserlineDeviceExport,
    >();
    assert_problem_types::<
        rhl::RigHandeyeLaserlineProblem,
        rhl::RigHandeyeLaserlineInput,
        rhl::RigHandeyeLaserlineConfig,
        rhl::RigHandeyeLaserlineEstimate,
        rhl::RigHandeyeLaserlineExport,
    >();

    assert_run_calibration(pi::run_calibration);
    assert_run_calibration(sch::run_calibration);
    assert_run_calibration(ld::run_calibration);
    assert_run_calibration(re::run_calibration);
    assert_run_calibration(rh::run_calibration);
    assert_run_calibration(rld::run_calibration);
    assert_run_calibration(rhl::run_calibration);

    // Steps take `Option<<Stage><Action>Options>` and return
    // `<Stage><Action>Result`; `*_with_seed` takes `<Stage>ManualInit` first.
    type Pi = session::CalibrationSession<pi::PlanarIntrinsicsProblem>;
    let _: fn(
        &mut Pi,
        Option<pi::IntrinsicsInitOptions>,
    ) -> Result<pi::IntrinsicsInitResult, Error> = pi::step_init;
    let _: fn(
        &mut Pi,
        pi::IntrinsicsManualInit,
        Option<pi::IntrinsicsInitOptions>,
    ) -> Result<pi::IntrinsicsInitResult, Error> = pi::step_init_with_seed;
    let _: fn(
        &mut Pi,
        Option<pi::IntrinsicsOptimizeOptions>,
    ) -> Result<pi::IntrinsicsOptimizeResult, Error> = pi::step_optimize;
    let _: fn(&mut Pi, Option<pi::IntrinsicsFilterOptions>) -> Result<(), Error> = pi::step_filter;
    let _: fn(&mut Pi, Option<pi::IntrinsicsFilterOptions>) -> Result<(), Error> =
        pi::run_calibration_with_filtering;

    type Sch = session::CalibrationSession<sch::SingleCamHandeyeProblem>;
    let _: fn(
        &mut Sch,
        Option<sch::IntrinsicsInitOptions>,
    ) -> Result<sch::IntrinsicsInitResult, Error> = sch::step_intrinsics_init;
    let _: fn(
        &mut Sch,
        sch::IntrinsicsManualInit,
        Option<sch::IntrinsicsInitOptions>,
    ) -> Result<sch::IntrinsicsInitResult, Error> = sch::step_intrinsics_init_with_seed;
    let _: fn(
        &mut Sch,
        Option<sch::IntrinsicsOptimizeOptions>,
    ) -> Result<sch::IntrinsicsOptimizeResult, Error> = sch::step_intrinsics_optimize;
    let _: fn(&mut Sch, Option<sch::HandeyeInitOptions>) -> Result<sch::HandeyeInitResult, Error> =
        sch::step_handeye_init;
    let _: fn(
        &mut Sch,
        sch::HandeyeManualInit,
        Option<sch::HandeyeInitOptions>,
    ) -> Result<sch::HandeyeInitResult, Error> = sch::step_handeye_init_with_seed;
    let _: fn(
        &mut Sch,
        Option<sch::HandeyeOptimizeOptions>,
    ) -> Result<sch::HandeyeOptimizeResult, Error> = sch::step_handeye_optimize;

    type Ld = session::CalibrationSession<ld::LaserlineDeviceProblem>;
    let _: fn(&mut Ld, Option<ld::DeviceInitOptions>) -> Result<ld::DeviceInitResult, Error> =
        ld::step_init;
    let _: fn(
        &mut Ld,
        ld::DeviceManualInit,
        Option<ld::DeviceInitOptions>,
    ) -> Result<ld::DeviceInitResult, Error> = ld::step_init_with_seed;
    let _: fn(
        &mut Ld,
        Option<ld::DeviceOptimizeOptions>,
    ) -> Result<ld::DeviceOptimizeResult, Error> = ld::step_optimize;

    type Re = session::CalibrationSession<re::RigExtrinsicsProblem>;
    let _: fn(
        &mut Re,
        Option<re::IntrinsicsInitOptions>,
    ) -> Result<re::IntrinsicsInitResult, Error> = re::step_intrinsics_init_all;
    let _: fn(
        &mut Re,
        re::IntrinsicsManualInit,
        Option<re::IntrinsicsInitOptions>,
    ) -> Result<re::IntrinsicsInitResult, Error> = re::step_intrinsics_init_all_with_seed;
    let _: fn(
        &mut Re,
        Option<re::IntrinsicsOptimizeOptions>,
    ) -> Result<re::IntrinsicsOptimizeResult, Error> = re::step_intrinsics_optimize_all;
    let _: fn(&mut Re, Option<re::RigInitOptions>) -> Result<re::RigInitResult, Error> =
        re::step_rig_init;
    let _: fn(
        &mut Re,
        re::RigManualInit,
        Option<re::RigInitOptions>,
    ) -> Result<re::RigInitResult, Error> = re::step_rig_init_with_seed;
    let _: fn(&mut Re, Option<re::RigOptimizeOptions>) -> Result<re::RigOptimizeResult, Error> =
        re::step_rig_optimize;

    type Rh = session::CalibrationSession<rh::RigHandeyeProblem>;
    let _: fn(
        &mut Rh,
        Option<rh::IntrinsicsInitOptions>,
    ) -> Result<rh::IntrinsicsInitResult, Error> = rh::step_intrinsics_init_all;
    let _: fn(
        &mut Rh,
        rh::IntrinsicsManualInit,
        Option<rh::IntrinsicsInitOptions>,
    ) -> Result<rh::IntrinsicsInitResult, Error> = rh::step_intrinsics_init_all_with_seed;
    let _: fn(
        &mut Rh,
        Option<rh::IntrinsicsOptimizeOptions>,
    ) -> Result<rh::IntrinsicsOptimizeResult, Error> = rh::step_intrinsics_optimize_all;
    let _: fn(&mut Rh, Option<rh::RigInitOptions>) -> Result<rh::RigInitResult, Error> =
        rh::step_rig_init;
    let _: fn(
        &mut Rh,
        rh::RigManualInit,
        Option<rh::RigInitOptions>,
    ) -> Result<rh::RigInitResult, Error> = rh::step_rig_init_with_seed;
    let _: fn(&mut Rh, Option<rh::RigOptimizeOptions>) -> Result<rh::RigOptimizeResult, Error> =
        rh::step_rig_optimize;
    let _: fn(&mut Rh, Option<rh::HandeyeInitOptions>) -> Result<rh::HandeyeInitResult, Error> =
        rh::step_handeye_init;
    let _: fn(
        &mut Rh,
        rh::HandeyeManualInit,
        Option<rh::HandeyeInitOptions>,
    ) -> Result<rh::HandeyeInitResult, Error> = rh::step_handeye_init_with_seed;
    let _: fn(
        &mut Rh,
        Option<rh::HandeyeOptimizeOptions>,
    ) -> Result<rh::HandeyeOptimizeResult, Error> = rh::step_handeye_optimize;

    type Rld = session::CalibrationSession<rld::RigLaserlineDeviceProblem>;
    let _: fn(&mut Rld, Option<rld::PlaneInitOptions>) -> Result<(), Error> = rld::step_init;
    let _: fn(&mut Rld, rld::PlaneManualInit, Option<rld::PlaneInitOptions>) -> Result<(), Error> =
        rld::step_init_with_seed;
    let _: fn(&mut Rld, Option<rld::PlaneOptimizeOptions>) -> Result<(), Error> =
        rld::step_optimize;

    // Shared step options are one type, reachable from `common` and from
    // every module that uses them.
    let _: common::IntrinsicsInitOptions = pi::IntrinsicsInitOptions::default();
    let _: common::RigInitOptions = rh::RigInitOptions::default();
    let _: common::RigOptimizeOptions = re::RigOptimizeOptions::default();
    let _: common::HandeyeOptimizeOptions = sch::HandeyeOptimizeOptions::default();

    // The planar camera model: a tilted sensor is a config value.
    let _tilted_camera = common::config::CameraConfig::new(core::CameraModelSpec::pinhole(
        core::DistortionKind::BrownConrady5,
        core::SensorKind::Scheimpflug,
    ));
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
    let _ = mvg::RansacOptions::new(1.0e-3);
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
