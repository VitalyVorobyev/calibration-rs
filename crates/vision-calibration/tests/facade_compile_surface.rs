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

    // Every input type is constructible from its workflow module and `common`.
    let _rig_laserline_view = common::RigLaserlineView::new(vec![None], vec![None]);
    let _rig_laserline_dataset: Option<rig_laserline_device::RigLaserlineDataset> = None;
    let _rig_handeye_laserline_input = rig_handeye_laserline::RigHandeyeLaserlineInput::new(
        vec![rig_handeye_laserline::RigHandeyeLaserlineView::new(
            common::RigLaserlineView::new(vec![None], vec![None]),
            common::RobotPoseMeta::new(core::Iso3::identity()),
        )],
        1,
    );
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
        common::HandeyeEstimate,
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
        common::HandeyeEstimate,
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
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<pi::IntrinsicsInitResult, Error> = pi::step_init;
    let _: fn(
        &mut Pi,
        common::IntrinsicsManualInit,
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<pi::IntrinsicsInitResult, Error> = pi::step_init_with_seed;
    let _: fn(
        &mut Pi,
        Option<common::IntrinsicsOptimizeOptions>,
    ) -> Result<pi::IntrinsicsOptimizeResult, Error> = pi::step_optimize;
    let _: fn(&mut Pi, Option<pi::IntrinsicsFilterOptions>) -> Result<(), Error> = pi::step_filter;
    let _: fn(&mut Pi, Option<pi::IntrinsicsFilterOptions>) -> Result<(), Error> =
        pi::run_calibration_with_filtering;

    type Sch = session::CalibrationSession<sch::SingleCamHandeyeProblem>;
    let _: fn(
        &mut Sch,
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<sch::IntrinsicsInitResult, Error> = sch::step_intrinsics_init;
    let _: fn(
        &mut Sch,
        common::IntrinsicsManualInit,
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<sch::IntrinsicsInitResult, Error> = sch::step_intrinsics_init_with_seed;
    let _: fn(
        &mut Sch,
        Option<common::IntrinsicsOptimizeOptions>,
    ) -> Result<sch::IntrinsicsOptimizeResult, Error> = sch::step_intrinsics_optimize;
    let _: fn(
        &mut Sch,
        Option<common::HandeyeInitOptions>,
    ) -> Result<sch::HandeyeInitResult, Error> = sch::step_handeye_init;
    let _: fn(
        &mut Sch,
        sch::HandeyeManualInit,
        Option<common::HandeyeInitOptions>,
    ) -> Result<sch::HandeyeInitResult, Error> = sch::step_handeye_init_with_seed;
    let _: fn(
        &mut Sch,
        Option<common::HandeyeOptimizeOptions>,
    ) -> Result<common::HandeyeOptimizeResult, Error> = sch::step_handeye_optimize;

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
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<common::RigIntrinsicsInitResult, Error> = re::step_intrinsics_init_all;
    let _: fn(
        &mut Re,
        common::RigIntrinsicsManualInit,
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<common::RigIntrinsicsInitResult, Error> = re::step_intrinsics_init_all_with_seed;
    let _: fn(
        &mut Re,
        Option<common::IntrinsicsOptimizeOptions>,
    ) -> Result<re::IntrinsicsOptimizeResult, Error> = re::step_intrinsics_optimize_all;
    let _: fn(&mut Re, Option<common::RigInitOptions>) -> Result<common::RigInitResult, Error> =
        re::step_rig_init;
    let _: fn(
        &mut Re,
        common::RigManualInit,
        Option<common::RigInitOptions>,
    ) -> Result<common::RigInitResult, Error> = re::step_rig_init_with_seed;
    let _: fn(&mut Re, Option<common::RigOptimizeOptions>) -> Result<re::RigOptimizeResult, Error> =
        re::step_rig_optimize;

    type Rh = session::CalibrationSession<rh::RigHandeyeProblem>;
    let _: fn(
        &mut Rh,
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<common::RigIntrinsicsInitResult, Error> = rh::step_intrinsics_init_all;
    let _: fn(
        &mut Rh,
        common::RigIntrinsicsManualInit,
        Option<common::IntrinsicsInitOptions>,
    ) -> Result<common::RigIntrinsicsInitResult, Error> = rh::step_intrinsics_init_all_with_seed;
    let _: fn(
        &mut Rh,
        Option<common::IntrinsicsOptimizeOptions>,
    ) -> Result<rh::IntrinsicsOptimizeResult, Error> = rh::step_intrinsics_optimize_all;
    let _: fn(&mut Rh, Option<common::RigInitOptions>) -> Result<common::RigInitResult, Error> =
        rh::step_rig_init;
    let _: fn(
        &mut Rh,
        common::RigManualInit,
        Option<common::RigInitOptions>,
    ) -> Result<common::RigInitResult, Error> = rh::step_rig_init_with_seed;
    let _: fn(&mut Rh, Option<common::RigOptimizeOptions>) -> Result<rh::RigOptimizeResult, Error> =
        rh::step_rig_optimize;
    let _: fn(&mut Rh, Option<common::HandeyeInitOptions>) -> Result<rh::HandeyeInitResult, Error> =
        rh::step_handeye_init;
    let _: fn(
        &mut Rh,
        rh::HandeyeManualInit,
        Option<common::HandeyeInitOptions>,
    ) -> Result<rh::HandeyeInitResult, Error> = rh::step_handeye_init_with_seed;
    let _: fn(
        &mut Rh,
        Option<common::HandeyeOptimizeOptions>,
    ) -> Result<common::HandeyeOptimizeResult, Error> = rh::step_handeye_optimize;

    type Rld = session::CalibrationSession<rld::RigLaserlineDeviceProblem>;
    let _: fn(&mut Rld, Option<rld::PlaneInitOptions>) -> Result<rld::PlaneInitResult, Error> =
        rld::step_init;
    let _: fn(
        &mut Rld,
        rld::PlaneManualInit,
        Option<rld::PlaneInitOptions>,
    ) -> Result<rld::PlaneInitResult, Error> = rld::step_init_with_seed;
    let _: fn(
        &mut Rld,
        Option<rld::PlaneOptimizeOptions>,
    ) -> Result<rld::PlaneOptimizeResult, Error> = rld::step_optimize;

    // The joint rig laser problem has no steps; its seeded runner takes the
    // warm start's `JointManualInit`.
    type Rhl = session::CalibrationSession<rhl::RigHandeyeLaserlineProblem>;
    let _: fn(&mut Rhl, rhl::JointManualInit) -> Result<(), Error> = rhl::run_calibration_with_seed;

    // Shared step options are one type, reachable only from `common`.
    let _: common::IntrinsicsInitOptions = common::IntrinsicsInitOptions::default();
    let _: common::RigInitOptions = common::RigInitOptions::default();
    let _: common::RigOptimizeOptions = common::RigOptimizeOptions::default();
    let _: common::HandeyeOptimizeOptions = common::HandeyeOptimizeOptions::default();

    // The planar camera model: a tilted sensor is a config value.
    let _tilted_camera = common::config::CameraConfig::new(core::CameraModelSpec::pinhole(
        core::DistortionKind::BrownConrady5,
        core::SensorKind::Scheimpflug,
    ));
}

#[test]
fn types_in_facade_signatures_are_nameable() {
    // A type a facade function returns, or a public field holds, can be named.
    let _: Option<core::Ray<core::Real>> = None;
    let _: Option<core::FixedParams> = None;
    let _: Option<core::HomographySensor<core::Real>> = None;
    let _: Option<mvg::solvers::GeometryError> = None;
    let _: Option<dataset::ValidationError> = None;
    let _: Option<dataset::RobotPoseSource> = None;
    let _: Option<dataset::PoseConvention> = None;
    let _: Option<dataset::PoseColumnMap> = None;
    let _: Option<dataset::TransformConvention> = None;
    let _: Option<dataset::TranslationUnits> = None;
    let _: Option<dataset::RotationFormat> = None;
    let _: Option<dataset::RobotPoseFormat> = None;
    let _: Option<dataset::ChessCornersDetectorSpec> = None;
    let _: Option<dataset::CornerStrategySpec> = None;
    let _: Option<dataset::CornerCovarianceSpec> = None;
    let _: Option<dataset::ImageNoiseSpec> = None;
    let _: Option<laserline_device::LaserlineParams> = None;
    let _: Option<laserline_device::LaserlineStats> = None;
    let _: Option<laserline_device::LaserlineEstimate> = None;
    let _: Option<rig_handeye_laserline::RigHandeyeLaserlineParams> = None;
    let _: Option<rig_handeye_laserline::RigHandeyeLaserlinePerCamStats> = None;
    let _: Option<init::PlaneFitView> = None;
    let _: Option<init::pnp::PoseEstimate> = None;
}

#[test]
fn model_comparison_surface_compiles() {
    use analysis::{
        CrossValidationOptions, FoldScore, HeldOutScore, ModelComparison, ModelOutcome, ModelScore,
    };

    type Compare = fn(
        &planar_intrinsics::PlanarIntrinsicsInput,
        &planar_intrinsics::PlanarIntrinsicsConfig,
        &[common::config::CameraConfig],
        &CrossValidationOptions,
        &control::RunControl,
    ) -> Result<ModelComparison, Error>;
    let _compare: Compare = analysis::compare_camera_models;
    let opts = CrossValidationOptions::new(5, 0);
    assert_eq!(opts, CrossValidationOptions::default());
    let _: Option<ModelScore> = None;
    let _: Option<HeldOutScore> = None;
    let _: Option<FoldScore> = None;
    let outcome: Option<ModelOutcome> = None;
    if let Some(ModelOutcome::Failed { fold, reason }) = outcome {
        let _: (Option<usize>, String) = (fold, reason);
    }
}

#[test]
fn diagnostic_maps_surface_compiles() {
    use analysis::{
        Bandwidth, BandwidthSelection, DisplacementPlane, DistortionMap, ImageGrid, ResidualMap,
        ResidualMapOptions, TargetFeatureResidual,
    };

    type Distortion = fn(&core::CameraParams, &ImageGrid) -> Result<DistortionMap, core::Error>;
    let _distortion: Distortion = analysis::distortion_map;
    type Residual = fn(
        &[TargetFeatureResidual],
        usize,
        &ImageGrid,
        &ResidualMapOptions,
    ) -> Result<ResidualMap, Error>;
    let _residual: Residual = analysis::residual_map;
    let grid = ImageGrid::with_columns([1280, 960], 64).expect("grid");
    assert_eq!((grid.columns(), grid.rows()), (64, 48));
    let mut options = ResidualMapOptions::new();
    options.bandwidth = Bandwidth::Pixels(40.0);
    assert_eq!(ResidualMapOptions::new().bandwidth, Bandwidth::Auto);
    let _: Option<DisplacementPlane> = None;
    let selection: Option<BandwidthSelection> = None;
    if let Some(BandwidthSelection::CrossValidated { folds, .. }) = selection {
        let _: usize = folds;
    }
    let residual = TargetFeatureResidual::default();
    let _: Option<[f64; 2]> = residual.residual_px();
}

#[test]
fn device_history_and_project_seed_surface_compiles() {
    use device_seed::{DeviceSeed, DeviceSeedError, SeedAggregate, SeedSpread};
    use project::{DeviceHistory, DeviceRun, ProjectError, RunSeed, SeedSource};

    let _aggregate: fn(&[DeviceSeed]) -> Result<SeedAggregate, DeviceSeedError> =
        DeviceSeed::aggregate;
    let _from_run =
        |run: &std::path::Path| -> Result<DeviceSeed, ProjectError> { DeviceSeed::from_run(run) };
    let _representative: fn(&DeviceHistory) -> Option<&DeviceRun> =
        DeviceHistory::representative_run;
    let _spread = |s: &SeedSpread| (s.median, s.mad_sigma, s.min, s.max, s.values.len());
    let _kept = |s: &RunSeed| {
        (
            s.source.clone(),
            s.value.clone(),
            s.applied.len(),
            s.ignored.len(),
        )
    };
    let sources = [
        SeedSource::Run("runs/run-001".into()),
        SeedSource::DeviceSpec("spec.json".into()),
        SeedSource::File("seed.json".into()),
    ];
    let mut options = project::ProjectRunOptions::new("tool");
    options.seed = Some(sources[2].clone());
    let mut spec = project::ProjectSpec::for_manifest("dataset.toml");
    spec.seed = Some(sources[0].clone());
}

#[test]
fn projection_difference_surface_compiles() {
    use analysis::{
        Compensation, FitRegion, ImageGrid, ProjectionDifference, ProjectionDifferenceOptions,
    };

    type Difference = fn(
        &core::CameraParams,
        &core::CameraParams,
        &ImageGrid,
        &ProjectionDifferenceOptions,
    ) -> Result<ProjectionDifference, core::Error>;
    let _difference: Difference = analysis::projection_difference;
    let defaults = ProjectionDifferenceOptions::default();
    assert_eq!(defaults.compensation, Compensation::Rotation);
    assert_eq!(defaults.fit_region, FitRegion::WholeImage);
    let options = ProjectionDifferenceOptions::new(
        Compensation::None,
        FitRegion::Covered(vec![[100.0, 200.0]]),
    );
    assert_eq!(options.compensation, Compensation::None);
    let result: Option<ProjectionDifference> = None;
    if let Some(diff) = result {
        let _: (nalgebra::UnitQuaternion<f64>, f64) = (diff.camb_so3_cama, diff.rotation_angle());
        let _: (usize, usize, &[bool]) = (diff.valid_nodes, diff.fit_nodes, &diff.fitted);
        let _: [f64; 4] = [
            diff.rms_fit_px,
            diff.max_fit_px,
            diff.rms_valid_px,
            diff.max_valid_px,
        ];
    }
}

#[test]
fn compare_runs_surface_compiles() {
    use analysis::{
        CameraDifference, CameraEvidence, CompareRunsOptions, ComparedRun, ConfigChange,
        CovarianceStatus, DataRelation, DifferenceTest, NoTest, ParameterComparison,
        ParameterDifference, PlaneDifference, PoseDifference, PoseTest, ProjectionComparison,
        RunComparison, RunDifference, RunEvidence, RunSide,
    };

    type Compare = fn(
        &[ComparedRun<'_>],
        &CompareRunsOptions,
    ) -> Result<RunComparison, vision_calibration::Error>;
    let _compare: Compare = analysis::compare_runs;
    let options = CompareRunsOptions::new().with_relations(vec![
        DataRelation::Independent,
        DataRelation::Nested {
            smaller: RunSide::Other,
        },
    ]);
    assert_eq!(options.grid_columns, 64);
    assert_eq!(options.compensation, analysis::Compensation::Rotation);
    let export: Option<planar_intrinsics::PlanarIntrinsicsExport> = None;
    if let Some(export) = &export {
        let run = ComparedRun::new("run-001", export, vec!["cam".into()], vec![[1280, 960]]);
        let _: &str = &run.name;
    }
    let result: Option<RunComparison> = None;
    if let Some(result) = result {
        let evidence: &RunEvidence = &result.runs[0];
        let _: (&CovarianceStatus, &[CameraEvidence]) = (&evidence.covariance, &evidence.cameras);
        let difference: &RunDifference = &result.differences[0];
        let _: (&[CameraDifference], &[PoseDifference], &[PlaneDifference]) = (
            &difference.cameras,
            &difference.rig,
            &difference.laser_planes,
        );
        let _: Option<&Vec<ConfigChange>> = difference.config_changes.as_ref();
        if let ParameterComparison::SameModel { parameters } = &difference.cameras[0].parameters {
            let p: &ParameterDifference = &parameters[0];
            if let DifferenceTest::NotAvailable { reason } = &p.test {
                let _: NoTest = *reason;
            }
        }
        let _: &ProjectionComparison = &difference.cameras[0].projection;
        let _: &PoseTest = &difference.rig[0].test;
    }
}

#[test]
fn focus_surface_compiles() {
    use analysis::{
        DepthOfField, FocusDiagnostic, FocusFit, FocusImageLine, FocusJackknife,
        FocusJackknifeCovariance, FocusOptions, FocusViewStats,
    };

    type Fit = fn(
        &planar_intrinsics::PlanarIntrinsicsInput,
        &core::CameraParams,
        &[core::Iso3],
        &FocusOptions,
    ) -> Result<FocusDiagnostic, Error>;
    let _fit: Fit = analysis::fit_focus;
    assert_eq!(FocusOptions::new(None, false), FocusOptions::default());
    let _: Option<FocusFit> = None;
    let _: Option<FocusImageLine> = None;
    let _: Option<DepthOfField> = None;
    let _: Option<FocusViewStats> = None;
    let _: Option<FocusJackknifeCovariance> = None;
    let diagnostic: Option<FocusDiagnostic> = None;
    if let Some(FocusDiagnostic::TooFewPoints { need, got }) = diagnostic {
        let _: (usize, usize) = (need, got);
    }
    if let Some(FocusDiagnostic::NotCrossed(fit)) = diagnostic
        && let Some(FocusJackknife::TooFewViews { views }) = fit.view_jackknife
    {
        let _: usize = views;
    }
}

#[test]
fn mvg_module_surface_compiles() {
    use vision_calibration::mvg;

    // Lock the key re-exported paths (multiple-view geometry pipelines).
    let _ = mvg::pose_recovery::recover_relative_pose;
    let _ = mvg::robust::recover_relative_pose_robust;
    let _ = mvg::RansacOptions::new(3.0);
    let _ = mvg::triangulation::triangulate;
    let _: Option<mvg::triangulation::RigObservation> = None;
    let _: Option<mvg::triangulation::Triangulation> = None;
    let _: Option<vision_calibration::core::CalibratedRig> = None;
    let _ = mvg::rectification::rectify;
    let _: Option<mvg::rectification::RectifyOptions> = None;
    let _: Option<mvg::rectification::RectifiedMatch> = None;
    let _ = mvg::homography::decompose_homography;
    let _ = mvg::dense::match_block;
    let _: Option<mvg::rectification::StereoRectification> = None;
    let _: Option<mvg::dense::DisparityMap> = None;
    let _: Option<mvg::dense::BlockMatchOptions> = None;
    let _: Option<mvg::types::Correspondence2D> = None;
    let _: Option<mvg::types::PixelMatch> = None;
    let _: Option<mvg::Error> = None;
    let _: Option<Result<(), mvg::Error>> = None;
    let _ = mvg::solvers::dlt_homography;
    let _: Option<mvg::solvers::FundamentalEstimate> = None;
    let _: Option<mvg::solvers::PoseCandidate> = None;
    let _: Option<mvg::residuals::EpipolarLineDistance> = None;
    let _ = mvg::solvers::fundamental_8point_ransac;
    let _ = mvg::solvers::rq_decompose;
    let _ = mvg::solvers::dlt_camera_matrix;
    let _ = mvg::solvers::decompose_camera_matrix;
    let _: Option<mvg::solvers::CameraMatrixDecomposition> = None;
    let _: Option<mvg::solvers::Mat34> = None;
    let _ = mvg::solvers::essential_5point;
    let _ = mvg::solvers::fundamental_7point;
    let _ = mvg::solvers::fundamental_8point;
    let _ = mvg::solvers::decompose_essential;
    let _ = mvg::solvers::triangulate_point_linear;
}

#[test]
fn mvg_rectification_runs_through_facade() {
    use vision_calibration::core::{
        BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, Uncertainty, make_pinhole_camera,
        pinhole_camera_params,
    };
    use vision_calibration::mvg::rectification::{RectifyOptions, rectify};

    let k = FxFyCxCy {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
    };
    let camera = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
    let rig = CalibratedRig::new(
        vec![camera.clone(), camera],
        vec![Iso3::identity(), Iso3::translation(-1.0, 0.0, 0.0)],
        Uncertainty::default(),
    )
    .expect("a two-camera rig");
    let rect = rectify(&rig, [0, 1], &RectifyOptions::default()).expect("rectify a pinhole pair");
    assert!((rect.baseline() - 1.0).abs() < 1e-12);
}

#[test]
fn mvg_bundle_adjust_reachable() {
    let _ = vision_calibration::mvg::bundle_adjust::bundle_adjust;
    let _ = vision_calibration::mvg::bundle_adjust::BundleAdjustOptions::new(
        vision_calibration::mvg::bundle_adjust::Scale::Rig,
    );
}

#[test]
fn facade_camera_traits_refine_essential_linear_and_validate_reachable() {
    fn assert_traits<
        P: core::ProjectionModel<core::Real>,
        D: core::DistortionModel<core::Real>,
        S: core::SensorModel<core::Real>,
        K: core::IntrinsicsModel<core::Real>,
    >(
        cam: &core::Camera<core::Real, P, D, S, K>,
    ) where
        core::Camera<core::Real, P, D, S, K>: core::CameraProject,
    {
        let _ = cam.project_point(&core::Pt3::new(0.0, 0.0, 1.0));
    }

    let k = core::FxFyCxCy {
        fx: 800.0,
        fy: 800.0,
        cx: 400.0,
        cy: 300.0,
    };
    let cam = core::Camera::new(core::Pinhole, core::NoDistortion, core::IdentitySensor, k);
    assert_traits(&cam);

    let _ = mvg::refine::refine_homography;
    let _ = mvg::solvers::essential_linear;
    let _ = dataset::validate;
    let _ = core::ImageManifest::new("images", vec![core::FrameRef::new(0, 0, "00.png")]);
    let _ = core::PixelRect::new(0, 0, 640, 480);
}
