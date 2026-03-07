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
    ) -> anyhow::Result<()> = planar_intrinsics::step_init;
    let _planar_step_optimize: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
        Option<planar_intrinsics::IntrinsicsOptimizeOptions>,
    ) -> anyhow::Result<()> = planar_intrinsics::step_optimize;
    let _planar_step_filter: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
        planar_intrinsics::FilterOptions,
    ) -> anyhow::Result<()> = planar_intrinsics::step_filter;
    let _planar_run: fn(
        &mut session::CalibrationSession<planar_intrinsics::PlanarIntrinsicsProblem>,
    ) -> anyhow::Result<()> = planar_intrinsics::run_calibration;

    // Scheimpflug intrinsics API.
    let _scheimpflug_run: fn(
        &mut session::CalibrationSession<scheimpflug_intrinsics::ScheimpflugIntrinsicsProblem>,
        Option<scheimpflug_intrinsics::ScheimpflugIntrinsicsConfig>,
    ) -> anyhow::Result<()> = scheimpflug_intrinsics::run_calibration;

    // Single-camera hand-eye API.
    let _single_handeye_run: fn(
        &mut session::CalibrationSession<single_cam_handeye::SingleCamHandeyeProblem>,
    ) -> anyhow::Result<()> = single_cam_handeye::run_calibration;

    // Rig extrinsics API.
    let _rig_extrinsics_run: fn(
        &mut session::CalibrationSession<rig_extrinsics::RigExtrinsicsProblem>,
    ) -> anyhow::Result<()> = rig_extrinsics::run_calibration;

    // Rig hand-eye API.
    let _rig_handeye_run: fn(
        &mut session::CalibrationSession<rig_handeye::RigHandeyeProblem>,
    ) -> anyhow::Result<()> = rig_handeye::run_calibration;

    // Laserline device API.
    let _laserline_run: fn(
        &mut session::CalibrationSession<laserline_device::LaserlineDeviceProblem>,
        Option<laserline_device::LaserlineDeviceConfig>,
    ) -> anyhow::Result<()> = laserline_device::run_calibration;
}

#[test]
fn prelude_compiles_for_hello_world_surface() {
    use vision_calibration::prelude::*;

    let _session: Option<CalibrationSession<PlanarIntrinsicsProblem>> = None;
    let _runner: fn(&mut CalibrationSession<PlanarIntrinsicsProblem>) -> anyhow::Result<()> =
        run_planar_intrinsics;
}
