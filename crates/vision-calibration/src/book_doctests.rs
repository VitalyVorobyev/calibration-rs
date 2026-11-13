//! Every Rust block of the book, compiled as a doctest of the facade.
//!
//! One item per chapter that has a `rust` block; `cargo xtask check-docs`
//! fails when such a chapter is missing here.

#[doc = include_str!("../../../book/src/architecture.md")]
struct Architecture;

#[doc = include_str!("../../../book/src/autodiff.md")]
struct Autodiff;

#[doc = include_str!("../../../book/src/camera_matrix.md")]
struct CameraMatrix;

#[doc = include_str!("../../../book/src/camera_pipeline.md")]
struct CameraPipeline;

#[doc = include_str!("../../../book/src/capture.md")]
struct Capture;

#[doc = include_str!("../../../book/src/compose_pipeline.md")]
struct ComposePipeline;

#[doc = include_str!("../../../book/src/custom_residual.md")]
struct CustomResidual;

#[doc = include_str!("../../../book/src/detection.md")]
struct Detection;

#[doc = include_str!("../../../book/src/distortion.md")]
struct Distortion;

#[doc = include_str!("../../../book/src/distortion_fit.md")]
struct DistortionFit;

#[doc = include_str!("../../../book/src/distortion_selection.md")]
struct DistortionSelection;

#[doc = include_str!("../../../book/src/epipolar.md")]
struct Epipolar;

#[doc = include_str!("../../../book/src/factor_catalog.md")]
struct FactorCatalog;

#[doc = include_str!("../../../book/src/fisheye.md")]
struct Fisheye;

#[doc = include_str!("../../../book/src/fisheye_init.md")]
struct FisheyeInit;

#[doc = include_str!("../../../book/src/fisheye_walkthrough.md")]
struct FisheyeWalkthrough;

#[doc = include_str!("../../../book/src/getting_started.md")]
struct GettingStarted;

#[doc = include_str!("../../../book/src/handeye_linear.md")]
struct HandeyeLinear;

#[doc = include_str!("../../../book/src/handeye_real_data.md")]
struct HandeyeRealData;

#[doc = include_str!("../../../book/src/handeye_workflow.md")]
struct HandeyeWorkflow;

#[doc = include_str!("../../../book/src/homography.md")]
struct Homography;

#[doc = include_str!("../../../book/src/intrinsics.md")]
struct Intrinsics;

#[doc = include_str!("../../../book/src/ir_architecture.md")]
struct IrArchitecture;

#[doc = include_str!("../../../book/src/iterative_intrinsics.md")]
struct IterativeIntrinsics;

#[doc = include_str!("../../../book/src/laser_init.md")]
struct LaserInit;

#[doc = include_str!("../../../book/src/laserline.md")]
struct Laserline;

#[doc = include_str!("../../../book/src/manifest.md")]
struct Manifest;

#[doc = include_str!("../../../book/src/manifest_laser.md")]
struct ManifestLaser;

#[doc = include_str!("../../../book/src/manifolds.md")]
struct Manifolds;

#[doc = include_str!("../../../book/src/manual_init.md")]
struct ManualInit;

#[doc = include_str!("../../../book/src/mvg.md")]
struct Mvg;

#[doc = include_str!("../../../book/src/mvg_bundle_adjust.md")]
struct MvgBundleAdjust;

#[doc = include_str!("../../../book/src/mvg_dense.md")]
struct MvgDense;

#[doc = include_str!("../../../book/src/mvg_measure_board.md")]
struct MvgMeasureBoard;

#[doc = include_str!("../../../book/src/mvg_metrology.md")]
struct MvgMetrology;

#[doc = include_str!("../../../book/src/mvg_object_pose.md")]
struct MvgObjectPose;

#[doc = include_str!("../../../book/src/mvg_pose.md")]
struct MvgPose;

#[doc = include_str!("../../../book/src/mvg_rectification.md")]
struct MvgRectification;

#[doc = include_str!("../../../book/src/new_pipeline.md")]
struct NewPipeline;

#[doc = include_str!("../../../book/src/new_problem.md")]
struct NewProblem;

#[doc = include_str!("../../../book/src/params.md")]
struct Params;

#[doc = include_str!("../../../book/src/per_feature_residuals.md")]
struct PerFeatureResiduals;

#[doc = include_str!("../../../book/src/pinhole.md")]
struct Pinhole;

#[doc = include_str!("../../../book/src/planar_intrinsics.md")]
struct PlanarIntrinsics;

#[doc = include_str!("../../../book/src/planar_pose.md")]
struct PlanarPose;

#[doc = include_str!("../../../book/src/planar_real_data.md")]
struct PlanarRealData;

#[doc = include_str!("../../../book/src/pnp.md")]
struct Pnp;

#[doc = include_str!("../../../book/src/problem_type.md")]
struct ProblemType;

#[doc = include_str!("../../../book/src/quality.md")]
struct Quality;

#[doc = include_str!("../../../book/src/ransac.md")]
struct Ransac;

#[doc = include_str!("../../../book/src/report.md")]
struct Report;

#[doc = include_str!("../../../book/src/rig_extrinsics.md")]
struct RigExtrinsics;

#[doc = include_str!("../../../book/src/rig_handeye.md")]
struct RigHandeye;

#[doc = include_str!("../../../book/src/rig_init.md")]
struct RigInit;

#[doc = include_str!("../../../book/src/rigid_transforms.md")]
struct RigidTransforms;

#[doc = include_str!("../../../book/src/robust_loss.md")]
struct RobustLoss;

#[doc = include_str!("../../../book/src/scheimpflug_laser_rigs.md")]
struct ScheimpflugLaserRigs;

#[doc = include_str!("../../../book/src/sensor.md")]
struct Sensor;

#[doc = include_str!("../../../book/src/session.md")]
struct Session;

#[doc = include_str!("../../../book/src/solver_backends.md")]
struct SolverBackends;

#[doc = include_str!("../../../book/src/step_functions.md")]
struct StepFunctions;

#[doc = include_str!("../../../book/src/stereo_real_data.md")]
struct StereoRealData;

#[doc = include_str!("../../../book/src/synthetic.md")]
struct Synthetic;

#[doc = include_str!("../../../book/src/triangulation.md")]
struct Triangulation;

#[doc = include_str!("../../../book/src/uncertainty_next_view.md")]
struct UncertaintyNextView;

#[doc = include_str!("../../../book/src/uncertainty_point_covariance.md")]
struct UncertaintyPointCovariance;

#[doc = include_str!("../../../book/src/uncertainty_projection.md")]
struct UncertaintyProjection;

#[doc = include_str!("../../../book/src/uncertainty_reading.md")]
struct UncertaintyReading;

#[doc = include_str!("../../../book/src/zhang.md")]
struct Zhang;
