//! High-level entry crate for the `calibration-rs` camera calibration library.
//!
//! This crate provides a unified API for camera calibration workflows:
//! - Single-camera intrinsics calibration (Zhang's method with distortion)
//! - Single-camera intrinsics calibration with Scheimpflug tilt
//! - Single-camera hand-eye calibration (camera on robot arm)
//! - Multi-camera rig extrinsics calibration
//! - Multi-camera rig + hand-eye calibration
//!
//! # Quick Start
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//! use vision_calibration::prelude::*;
//! use vision_calibration::planar_intrinsics::{step_init, step_optimize};
//!
//! // Create calibration session
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step (recommended for inspection)
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function (convenience)
//! // run_planar_intrinsics(&mut session)?;
//!
//! // Export results
//! let result = session.export()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Module Organization
//!
//! ## High-Level Calibration Workflows
//!
//! - [`session`] - Session framework (`CalibrationSession`, `ProblemType`)
//! - [`planar_intrinsics`] - Single-camera intrinsics (Zhang's method)
//! - [`single_cam_handeye`] - Single camera + hand-eye calibration
//! - [`rig_extrinsics`] - Multi-camera rig extrinsics
//! - [`rig_handeye`] - Multi-camera rig + hand-eye
//! - [`laserline_device`] - Single camera + laser plane device
//! - [`scheimpflug_intrinsics`] - Single-camera planar intrinsics with Scheimpflug tilt
//!
//! ## Foundation Crates (Advanced Users)
//!
//! - [`core`] - Math types, camera models, RANSAC primitives
//! - [`linear`] - Closed-form initialization algorithms
//! - [`optim`] - Non-linear least-squares optimization
//! - [`synthetic`] - Synthetic data generation for testing
//!
//! # Session API
//!
//! All calibration workflows use the [`session::CalibrationSession`] state container.
//! Each problem type has its own set of step functions — see the table below
//! and the per-module documentation for details.
//!
//! The common pattern is:
//! 1. Create a session for the problem type
//! 2. Set input data with `set_input`
//! 3. Optionally configure with `update_config`
//! 4. Run the problem-specific step functions (or a convenience `run_calibration`)
//! 5. Export results with `export`
//!
//! # Available Problem Types
//!
//! | Problem Type | Input | Steps |
//! |--------------|-------|-------|
//! | [`PlanarIntrinsicsProblem`](planar_intrinsics) | `PlanarDataset` | `step_init` → `step_optimize` |
//! | [`SingleCamHandeyeProblem`](single_cam_handeye) | `SingleCamHandeyeInput` | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
//! | [`RigExtrinsicsProblem`](rig_extrinsics) | `RigExtrinsicsInput` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
//! | [`RigHandeyeProblem`](rig_handeye) | `RigHandeyeInput` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` → `step_handeye_init` → `step_handeye_optimize` |
//! | [`LaserlineDeviceProblem`](laserline_device) | `LaserlineDeviceInput` | `step_init` → `step_optimize` |
//! | [`ScheimpflugIntrinsicsProblem`](scheimpflug_intrinsics) | `PlanarDataset` | `step_init` → `step_optimize` |

#![deny(missing_docs)]

/// Typed error returned by all public calibration step functions.
pub use vision_calibration_pipeline::Error;

/// Single-camera planar intrinsics with Scheimpflug/tilted sensor refinement.
///
/// This high-level helper mirrors planar intrinsics calibration, but optimizes a
/// Brown-Conrady camera together with two Scheimpflug tilt parameters.
pub mod scheimpflug_intrinsics {
    pub use vision_calibration_pipeline::scheimpflug_intrinsics::{
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        ScheimpflugFixMask,
        ScheimpflugIntrinsicsConfig,
        ScheimpflugIntrinsicsExport,
        // Typed step results (Phase 1c of 0.5.0 API revision)
        ScheimpflugIntrinsicsInitResult,
        ScheimpflugIntrinsicsInput,
        ScheimpflugIntrinsicsOptimizeResult,
        ScheimpflugIntrinsicsParams,
        ScheimpflugIntrinsicsProblem,
        ScheimpflugIntrinsicsResult,
        ScheimpflugManualInit,
        run_calibration,
        step_init,
        step_init_with_seed,
        step_optimize,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session Framework
// ═══════════════════════════════════════════════════════════════════════════════

/// Session framework for structured calibration workflows.
///
/// Provides mutable state containers, step functions, and JSON checkpointing.
pub mod session {
    pub use vision_calibration_pipeline::session::{
        CalibrationSession, ExportRecord, InvalidationPolicy, LogEntry, ProblemType,
        SessionMetadata,
    };
}

/// Step-option types shared across problem modules.
///
/// The intrinsics and hand-eye step-option structs are identical across every
/// problem that exposes the corresponding step. They are defined once here; the
/// per-problem modules re-export the ones they use, so both
/// `vision_calibration::common::IntrinsicsInitOptions` and
/// `vision_calibration::planar_intrinsics::IntrinsicsInitOptions` resolve to the
/// same type.
///
/// The nested [`common::config`] module holds the sibling set of shared
/// **config** sub-structs (ADR 0024) — the grouped building blocks embedded
/// by top-level `*Config` types (`init: IntrinsicsInitConfig`, `solver:
/// SolverConfig`, ...).
pub mod common {
    pub use vision_calibration_pipeline::common::{
        HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
    };

    /// Shared config sub-structs embedded by grouped top-level `*Config`
    /// types (ADR 0024): per-camera linear init, non-linear solve, robot-pose
    /// refinement, hand-eye linear init, and rig frame options.
    pub mod config {
        pub use vision_calibration_pipeline::common::config::{
            HandeyeInitConfig, IntrinsicsInitConfig, RigConfig, RobotPoseConfig, SolverConfig,
        };
    }
}

/// Device-spec → manual-init seed derivation (ADR 0023).
///
/// [`DeviceSpec`](device_seed::DeviceSpec) is the sidecar `spec.json` schema
/// describing the capture hardware (lens focal, pixel pitch, Scheimpflug
/// mount tilt, nominal rig layout); the functions here derive the ADR 0011
/// manual-init seeds that make spec-seeded initialization (ADR 0022) the
/// structured default.
pub mod device_seed {
    pub use vision_calibration_pipeline::device_seed::{
        // Schema types (from `vision-calibration-dataset`)
        CameraDeviceSpec,
        CameraMountSpec,
        DEVICE_SPEC_FILENAME,
        DEVICE_SPEC_VERSION,
        // Derivation
        DeviceSeedError,
        DeviceSpec,
        DeviceSpecError,
        HandeyeMountSpec,
        LaserPlaneSpec,
        NominalPoseSpec,
        RigLayoutSpec,
        ScheimpflugMountSpec,
        handeye_seed,
        nominal_cam_se3_rig,
        rig_intrinsics_seed,
        rig_layout_seed,
        scheimpflug_seed,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Problem-Specific Modules
// ═══════════════════════════════════════════════════════════════════════════════

/// Planar intrinsics calibration (Zhang's method with Brown-Conrady distortion).
///
/// # Steps
/// 1. `step_init` - Zhang's method with iterative distortion estimation
/// 2. `step_optimize` - Non-linear refinement
/// 3. `step_filter` (optional) - Remove outliers by reprojection error
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let dataset = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::planar_intrinsics::{step_init, step_optimize, run_calibration};
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod planar_intrinsics {
    pub use vision_calibration_pipeline::planar_intrinsics::{
        // Step options
        FilterOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        // Typed step results (Phase 1a of 0.5.0 API revision)
        PlanarInitResult,
        // Problem type and config
        PlanarIntrinsicsConfig,
        // Re-exports from vision-calibration-optim
        PlanarIntrinsicsEstimate,
        PlanarIntrinsicsExport,
        PlanarIntrinsicsParams,
        PlanarIntrinsicsProblem,
        PlanarIntrinsicsSolveOptions,
        // Manual init seed (ADR 0011)
        PlanarManualInit,
        PlanarOptimizeResult,
        // Step functions
        run_calibration,
        run_calibration_with_filtering,
        step_filter,
        step_init,
        step_init_with_seed,
        step_optimize,
    };
}

/// Single-camera hand-eye calibration (intrinsics + hand-eye transform).
///
/// For calibrating a camera mounted on a robot arm.
///
/// # Steps
/// 1. `step_intrinsics_init` - Zhang's method
/// 2. `step_intrinsics_optimize` - Non-linear intrinsics refinement
/// 3. `step_handeye_init` - Tsai-Lenz linear estimation
/// 4. `step_handeye_optimize` - Bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::single_cam_handeye::{
///     run_calibration, SingleCamHandeyeInput, SingleCamHandeyeProblem
/// };
///
/// let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod single_cam_handeye {
    pub use vision_calibration_pipeline::single_cam_handeye::{
        // Step options
        HandeyeInitOptions,
        // Problem type and config
        HandeyeMeta,
        HandeyeOptimizeOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        SingleCamHandeyeConfig,
        SingleCamHandeyeExport,
        // Typed step results (Phase 1a of 0.5.0 API revision)
        SingleCamHandeyeInitResult,
        SingleCamHandeyeInput,
        // Manual init seeds (ADR 0011)
        SingleCamHandeyeManualInit,
        SingleCamHandeyeOptimizeResult,
        SingleCamHandeyeProblem,
        SingleCamHandeyeView,
        SingleCamIntrinsicsInitResult,
        SingleCamIntrinsicsManualInit,
        SingleCamIntrinsicsOptimizeResult,
        // Step functions
        run_calibration,
        step_handeye_init,
        step_handeye_init_with_seed,
        step_handeye_optimize,
        step_intrinsics_init,
        step_intrinsics_init_with_seed,
        step_intrinsics_optimize,
    };
}

/// Single laserline device calibration (camera + laser plane).
///
/// # Steps
/// 1. `step_init` - Iterative intrinsics + linear laser plane init
/// 2. `step_optimize` - Joint bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::laserline_device::{
///     run_calibration, LaserlineDeviceInput, LaserlineDeviceProblem
/// };
///
/// let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session, None)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod laserline_device {
    pub use vision_calibration_pipeline::laserline_device::{
        DeviceInitOptions,
        DeviceOptimizeOptions,
        LaserlineDeviceConfig,
        LaserlineDeviceExport,
        // Typed step results (Phase 1c of 0.5.0 API revision)
        LaserlineDeviceInitResult,
        LaserlineDeviceInput,
        LaserlineDeviceManualInit,
        LaserlineDeviceOptimizeConfig,
        LaserlineDeviceOptimizeResult,
        LaserlineDeviceOutput,
        LaserlineDeviceProblem,
        run_calibration,
        step_init,
        step_init_with_seed,
        step_optimize,
    };
}

/// Multi-camera rig extrinsics calibration.
///
/// For calibrating a multi-camera rig, estimating per-camera intrinsics
/// and camera-to-rig transforms.
///
/// # Steps
/// 1. `step_intrinsics_init_all` - Per-camera Zhang's method
/// 2. `step_intrinsics_optimize_all` - Per-camera non-linear refinement
/// 3. `step_rig_init` - Linear estimation of camera-to-rig transforms
/// 4. `step_rig_optimize` - Joint bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::rig_extrinsics::{
///     run_calibration, RigExtrinsicsInput, RigExtrinsicsProblem
/// };
///
/// let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod rig_extrinsics {
    pub use vision_calibration_pipeline::rig_extrinsics::{
        // Step options
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        // Problem type and config
        RigExtrinsicsConfig,
        RigExtrinsicsExport,
        RigExtrinsicsInput,
        // Manual init seeds (ADR 0011)
        RigExtrinsicsManualInit,
        // Output (pinhole or Scheimpflug variant; A6 unified rig family)
        RigExtrinsicsOutput,
        RigExtrinsicsProblem,
        // Typed step results (Phase 1a of 0.5.0 API revision)
        RigInitResult,
        RigIntrinsicsInitAllResult,
        RigIntrinsicsManualInit,
        RigIntrinsicsOptimizeAllResult,
        RigOptimizeOptions,
        RigOptimizeResult,
        // Sensor flavour selector (pinhole vs Scheimpflug)
        SensorMode,
        // Step functions
        run_calibration,
        step_intrinsics_init_all,
        step_intrinsics_init_all_with_seed,
        step_intrinsics_optimize_all,
        step_rig_init,
        step_rig_init_with_seed,
        step_rig_optimize,
    };
}

/// Multi-camera rig hand-eye calibration.
///
/// For calibrating a multi-camera rig mounted on a robot arm, including
/// per-camera intrinsics, rig extrinsics, and hand-eye transform.
///
/// # Steps (6 total)
/// 1. `step_intrinsics_init_all` - Per-camera Zhang's method
/// 2. `step_intrinsics_optimize_all` - Per-camera non-linear refinement
/// 3. `step_rig_init` - Linear estimation of camera-to-rig transforms
/// 4. `step_rig_optimize` - Rig bundle adjustment
/// 5. `step_handeye_init` - Tsai-Lenz linear estimation
/// 6. `step_handeye_optimize` - Hand-eye bundle adjustment
///
/// # Example
/// ```no_run
/// # fn main() -> anyhow::Result<()> {
/// # let input = unimplemented!();
/// use vision_calibration::prelude::*;
/// use vision_calibration::rig_handeye::{run_calibration, RigHandeyeInput, RigHandeyeProblem};
///
/// let mut session = CalibrationSession::<RigHandeyeProblem>::new();
/// session.set_input(input)?;
/// run_calibration(&mut session)?;
/// let result = session.export()?;
/// # Ok(())
/// # }
/// ```
pub mod rig_handeye {
    pub use vision_calibration_pipeline::rig_handeye::{
        // Problem type and config
        HandeyeBaConfig,
        // Step options
        HandeyeInitOptions,
        HandeyeOptimizeOptions,
        IntrinsicsInitOptions,
        IntrinsicsOptimizeOptions,
        RigHandeyeConfig,
        RigHandeyeExport,
        // Typed step results (Phase 1a of 0.5.0 API revision)
        RigHandeyeHandeyeInitResult,
        // Manual init seeds (ADR 0011)
        RigHandeyeHandeyeManualInit,
        RigHandeyeHandeyeOptimizeResult,
        RigHandeyeInput,
        RigHandeyeIntrinsicsInitAllResult,
        RigHandeyeIntrinsicsManualInit,
        RigHandeyeIntrinsicsOptimizeAllResult,
        // Output (pinhole or Scheimpflug variant; A6 unified rig family)
        RigHandeyeOutput,
        RigHandeyeProblem,
        RigHandeyeRigInitResult,
        RigHandeyeRigManualInit,
        RigHandeyeRigOptimizeResult,
        RigOptimizeOptions,
        // Sensor flavour selector (pinhole vs Scheimpflug)
        SensorMode,
        // Step functions
        run_calibration,
        step_handeye_init,
        step_handeye_init_with_seed,
        step_handeye_optimize,
        step_intrinsics_init_all,
        step_intrinsics_init_all_with_seed,
        step_intrinsics_optimize_all,
        step_rig_init,
        step_rig_init_with_seed,
        step_rig_optimize,
    };
}

/// Rig-level laserline calibration.
///
/// Given an upstream Scheimpflug rig hand-eye calibration ([`rig_handeye`] with
/// `SensorMode::Scheimpflug`), fits one laser plane per camera and reports each
/// plane in the rig frame.
pub mod rig_laserline_device {
    pub use vision_calibration_pipeline::rig_laserline_device::{
        RigLaserlineDeviceConfig, RigLaserlineDeviceExport, RigLaserlineDeviceInput,
        RigLaserlineDeviceManualInit, RigLaserlineDeviceProblem, RigUpstreamCalibration,
        StepOptions, pixel_to_gripper_point, run_calibration, step_init, step_init_with_seed,
        step_optimize,
    };
}

/// Joint rig hand-eye + laserline calibration.
pub mod rig_handeye_laserline {
    pub use vision_calibration_pipeline::rig_handeye_laserline::{
        RigHandeyeLaserlineBaConfig, RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport,
        RigHandeyeLaserlineInput, RigHandeyeLaserlineOutput, RigHandeyeLaserlineProblem,
        run_calibration,
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Foundation Crates (Advanced Users)
// ═══════════════════════════════════════════════════════════════════════════════

/// Core math types, camera models, and RANSAC primitives.
///
/// Re-exports selected foundational types from `vision_calibration_core`. The
/// per-feature residual/diagnostics types and helpers (`ReprojectionStats`,
/// `PerFeatureResiduals`, `compute_*_target_residuals`, …) live in
/// [`analysis`], not here — see that module's doc comment.
pub mod core {
    pub use vision_calibration_core::{
        BrownConrady5, Camera, CameraModel, CameraParams, CorrespondenceView, DistortionFixMask,
        DistortionParams, Division, FrameKind, FrameRef, FxFyCxCySkew, IdentitySensor,
        ImageManifest, IntrinsicsFixMask, IntrinsicsParams, Iso3, Mat3, NoMeta, Pinhole,
        PinholeCamera, PixelRect, PlanarDataset, ProjectionParams, Pt2, Pt3, RationalPolynomial,
        Real, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorModel, SensorParams,
        ThinPrism, Vec2, Vec3, View, distort_to_pixel, make_pinhole_camera, pinhole_camera_params,
        pixel_to_normalized,
    };
}

/// Canonical input-data manifest (`DatasetSpec` and friends).
///
/// Re-exported from `vision-calibration-dataset`. [`DatasetSpec`](dataset::DatasetSpec)
/// is the on-disk manifest describing where images, robot poses, and target
/// metadata live for a calibration run; see [`dataset_runner`] for the
/// converters that turn a manifest into the per-problem `*Input` types.
/// [`sniff_folder`](dataset::sniff_folder) heuristically infers a
/// `DatasetSpec` from a dataset folder (used by the `generate-manifest` CLI
/// and the diagnose app's "Sniff folder" command).
pub mod dataset {
    pub use vision_calibration_dataset::{
        CameraSource, DatasetSpec, ImagePattern, LaserExtractionSpec, LaserScanAxis, PosePairing,
        SniffError, TargetSpec, Topology, sniff_folder,
    };
}

/// Dataset-driven runner: turns a [`dataset::DatasetSpec`] manifest plus a
/// per-problem config into the existing `*Input` IR by running (cached)
/// detection on the per-camera images.
///
/// Re-exported from the pipeline crate's `dataset_runner` module — see that
/// module's doc comment for the full list of `build_*_input` converters and
/// the ADR 0019 ask-user-on-ambiguity contract that
/// [`RunError::AskUser`](dataset_runner::RunError::AskUser) implements.
pub mod dataset_runner {
    pub use vision_calibration_pipeline::dataset_runner::{
        HandeyeRunResult, LaserPixelExtractor, LaserlineRunResult, PairedViews, PlanarRunResult,
        RigHandeyeLaserlineRunResult, RigLaserlineRunResult, RigRunResult, RunError,
        build_laserline_device_input, build_planar_input, build_rig_extrinsics_input,
        build_rig_handeye_input, build_rig_handeye_laserline_input,
        build_rig_laserline_device_input, build_single_cam_handeye_input,
    };
}

/// Feature-detection cache primitives.
///
/// Re-exported from `vision-calibration-detect`. [`dataset_runner`] uses
/// these to avoid re-detecting corners/markers on repeated runs over the
/// same images; [`FsDetectionCache`](detect::FsDetectionCache) is the
/// filesystem-backed implementation of the [`DetectionCache`](detect::DetectionCache)
/// trait.
pub mod detect {
    pub use vision_calibration_detect::{
        CacheError, CacheKey, CachedFeatures, DetectionCache, FsDetectionCache,
    };
}

/// Closed-form initialization algorithms.
///
/// Includes Zhang's method, PnP solvers, hand-eye solvers, and more.
/// Two-view geometric solvers (homography, fundamental/essential matrix,
/// camera matrix, triangulation) are in the `geometry` module below.
///
/// Re-exports the algorithm modules of `vision_calibration_linear` by name —
/// items are reached via their owning module, e.g.
/// `vision_calibration::linear::zhang_intrinsics::estimate_intrinsics_from_homographies`.
/// The curated [`prelude`](linear::prelude) gathers the most-used items.
pub mod linear {
    pub use vision_calibration_linear::{
        distortion_fit, extrinsics, handeye, iterative_intrinsics, laserline, math, planar_pose,
        pnp, scheimpflug_init, zhang_intrinsics,
    };

    pub mod prelude {
        //! Curated, most-used items from `vision-calibration-linear`.
        pub use vision_calibration_linear::prelude::*;
    }
}

/// Deterministic two-view geometric solvers: epipolar geometry, homography,
/// triangulation, and camera-matrix decomposition. Re-exported from
/// `vision_geometry`, e.g. `vision_calibration::geometry::homography::dlt_homography`.
pub mod geometry {
    pub use vision_geometry::{camera_matrix, epipolar, homography, triangulation};
}

/// Multiple-view geometry pipelines over the [`geometry`] solvers, re-exported
/// from `vision_mvg`: calibrated relative-pose recovery, robust (RANSAC)
/// estimation, cheirality/degeneracy checks, N-view triangulation, homography
/// decomposition, Scheimpflug-aware stereo rectification, and pure-Rust dense
/// stereo matching — reached via the owning module, e.g.
/// `vision_calibration::mvg::rectification::rectify_stereo_pair` or
/// `vision_calibration::mvg::dense::match_block`.
///
/// With the `refine` feature, `vision_calibration::mvg::bundle_adjust` adds
/// frozen-intrinsics bundle adjustment (tiny-solver).
pub mod mvg {
    pub use vision_mvg::{
        MvgError, cheirality, degeneracy, dense, error, homography, pose_recovery, rectification,
        residuals, robust, triangulation, types,
    };

    #[cfg(feature = "refine")]
    pub use vision_mvg::bundle_adjust;
}

/// Non-linear optimization vocabulary.
///
/// `vision-calibration-optim` is the optimization-backend implementation
/// crate; the typical consumer never touches it directly — they go through
/// the per-problem `step_*` functions, which wrap it. This module re-exports
/// only the small value/enum vocabulary a facade consumer legitimately names:
/// the robust-loss selector, the laser-plane parameter type, the hand-eye
/// mode enum, and the per-problem input-construction `*Meta`/`*View` types.
///
/// The `compute_*_feature_residuals` / `handeye_observer_se3_target` helpers
/// live in [`analysis`], not here.
pub mod optim {
    /// Hand-eye configuration mode (eye-in-hand vs eye-to-hand).
    pub use vision_calibration_optim::HandEyeMode;
    /// Laser-plane parameter type.
    pub use vision_calibration_optim::LaserPlane;
    /// Per-view metadata for laserline device input.
    pub use vision_calibration_optim::LaserlineMeta;
    /// Per-view observation type for laserline device input.
    pub use vision_calibration_optim::LaserlineView;
    /// Per-view robot-pose metadata for rig hand-eye input.
    pub use vision_calibration_optim::RobotPoseMeta;
    /// Robust loss (M-estimator) selector for optimization.
    pub use vision_calibration_optim::RobustLoss;
}

/// Deterministic synthetic data generation for testing.
///
/// Provides builders for creating synthetic calibration datasets. The surface
/// is a hand-picked subset of `vision_calibration_core::synthetic`: only the
/// generator submodules genuinely useful to consumers are re-exported, not the
/// whole module (which is primarily a test/example helper).
pub mod synthetic {
    /// Deterministic noise helpers (e.g. [`noise::UniformPixelNoise`]).
    pub use vision_calibration_core::synthetic::noise;
    /// Planar target generators: point grids, pose ramps, and projection helpers.
    pub use vision_calibration_core::synthetic::planar;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prelude (Quick Start)
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimal re-exports for planar "hello world" calibration.
///
/// ```no_run
/// use vision_calibration::prelude::*;
/// ```
pub mod prelude {
    /// Session framework for calibration workflows.
    pub use vision_calibration_pipeline::session::CalibrationSession;

    /// Planar intrinsics problem type for hello-world calibration.
    pub use vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsProblem;

    /// Convenience planar calibration runner.
    pub use vision_calibration_pipeline::planar_intrinsics::run_calibration as run_planar_intrinsics;

    /// Core geometry and dataset types used in minimal planar workflows.
    pub use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCySkew, PlanarDataset, Pt2, Pt3, View,
        make_pinhole_camera,
    };
}

/// Reprojection-error diagnostics: per-feature residuals and multi-level
/// analysis reports.
///
/// Two families of items live here:
/// - Multi-level reprojection reports (re-exported from the pipeline):
///   [`ReprojLevel`](vision_calibration_pipeline::analysis::ReprojLevel),
///   [`ReprojReport`](vision_calibration_pipeline::analysis::ReprojReport),
///   and the `*_report` builders — see the pipeline crate's `analysis` module
///   doc comment for the constraint-level model.
/// - Per-feature residual types and computation helpers (from
///   `vision-calibration-core` and `vision-calibration-optim`):
///   [`ReprojectionStats`](analysis::ReprojectionStats),
///   [`PerFeatureResiduals`](analysis::PerFeatureResiduals),
///   [`TargetFeatureResidual`](analysis::TargetFeatureResidual),
///   [`LaserFeatureResidual`](analysis::LaserFeatureResidual),
///   [`FeatureResidualHistogram`](analysis::FeatureResidualHistogram),
///   `compute_planar_target_residuals{,_views}`, `compute_rig_target_residuals`,
///   `build_feature_histogram`, `compute_laserline_feature_residuals`,
///   `compute_rig_laserline_feature_residuals`, and
///   `handeye_observer_se3_target` — used across the per-problem `*Export`
///   types' `per_feature_residuals` field.
pub mod analysis {
    pub use vision_calibration_pipeline::analysis::*;

    // Per-feature residual/histogram types and their computation helpers.
    pub use vision_calibration_core::{
        FeatureResidualHistogram, LaserFeatureResidual, PerFeatureResiduals,
        REPROJECTION_HISTOGRAM_EDGES_PX, ReprojectionStats, TargetFeatureResidual,
        build_feature_histogram, compute_planar_target_residuals,
        compute_planar_target_residuals_views, compute_rig_target_residuals,
    };

    // Per-feature residual helpers implemented against the optimization
    // backend: `handeye_observer_se3_target` (used by hand-eye exports) and
    // `compute_*_feature_residuals` (used by laser exports).
    pub use vision_calibration_optim::{
        compute_laserline_feature_residuals, compute_rig_laserline_feature_residuals,
        handeye_observer_se3_target,
    };
}
