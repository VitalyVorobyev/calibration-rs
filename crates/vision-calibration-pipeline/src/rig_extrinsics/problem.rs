//! [`ProblemType`] implementation for multi-camera rig extrinsics calibration.
//!
//! This module provides the `RigExtrinsicsProblem` type that implements
//! the session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, FeatureResidualHistogram, ImageManifest, Iso3, NoMeta, PerFeatureResiduals,
    RigDataset, Uncertainty, build_feature_histogram, compute_rig_target_residuals,
};
use vision_calibration_optim::RigExtrinsicsEstimate;

use crate::common::ExportKind;
use crate::common::config::{
    CameraConfig, CameraOverride, IntrinsicsInitConfig, RigConfig, SolverConfig, rig_camera_for,
    validate_rig_camera_indices, validate_rig_cameras,
};
use crate::rig_family::rig_camera_models;
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::RigExtrinsicsState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Type
// ─────────────────────────────────────────────────────────────────────────────

/// Input for rig extrinsics calibration.
///
/// Reuses `RigDataset<NoMeta>` from vision_calibration_core.
pub type RigExtrinsicsInput = RigDataset<NoMeta>;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-camera rig extrinsics calibration.
///
/// Grouped by stage. Every camera has the model of `camera` unless
/// `camera_overrides` names it, so one rig can mix lenses and sensors.
///
/// The reference-camera gauge fix (`rig.reference_camera_idx`) alone removes
/// the full 6-DOF rig gauge; an extra per-view pose constraint would be
/// redundant and mildly pessimizing. View-0's rig-from-target pose is always free in the rig BA.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigExtrinsicsConfig {
    /// Per-camera linear-initialization stage settings. An unset
    /// `fix_tangential` keeps the tangential terms free for every camera.
    pub intrinsics: IntrinsicsInitConfig,

    /// The camera model of every rig camera and the parameters its solves
    /// hold fixed. A tilted (Scheimpflug) sensor's per-camera solve runs a
    /// multi-start tilt sweep.
    #[serde(default)]
    pub camera: CameraConfig,

    /// Cameras whose model differs from `camera`, at most one entry per
    /// camera index.
    #[serde(default)]
    pub camera_overrides: Vec<CameraOverride>,

    /// Rig frame options (reference camera, rig-BA scope).
    pub rig: RigConfig,

    /// Non-linear solve stage settings.
    pub solver: SolverConfig,
}

impl RigExtrinsicsConfig {
    /// The configuration of camera `index`: its override, or `camera`.
    pub fn camera_for(&self, index: usize) -> CameraConfig {
        rig_camera_for(&self.camera, &self.camera_overrides, index)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Export format for rig extrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RigExtrinsicsExport {
    /// Export-type discriminator — always [`ExportKind::RigExtrinsics`].
    pub kind: ExportKind,

    /// Per-camera calibrated parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,

    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    /// Transform from rig frame to camera frame.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_rig: Vec<Iso3>,

    /// Per-view rig poses: `rig_se3_target` (T_R_T).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub rig_se3_target: Vec<Iso3>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,

    /// Per-camera reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,

    /// Per-feature reprojection residuals. For rig extrinsics
    /// `target` is populated and `laser` is empty. `target_hist_per_camera`
    /// is `Some(vec)` with one entry per camera.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,

    /// Optional image manifest (viewer-side contract). When
    /// populated, downstream viewers (the diagnose UI) can locate the source
    /// image for each `(pose, camera)` slot. Tiled multi-camera frames
    /// (e.g. six 720×540 tiles in one horizontal strip) point
    /// multiple `FrameRef`s at the same `path` with disjoint ROIs. `None`
    /// means "no images shipped"; the calibration pipeline never reads
    /// this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
    /// Covariance of the calibrated parameters: `fx ± σ`, correlations, and
    /// the noise level the residuals imply.
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

// ─────────────────────────────────────────────────────────────────────────────
// ProblemType Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-camera rig extrinsics calibration problem.
///
/// Calibrates a multi-camera rig, including:
/// - Per-camera intrinsics, distortion and sensor tilt, each camera in its
///   own model
/// - Per-camera extrinsics (camera-to-rig transforms)
/// - Per-view rig poses (rig-to-target)
///
/// # Conventions
///
/// - `cam_se3_rig` = T_C_R (transform from rig to camera frame)
/// - Reference camera has identity extrinsics (defines rig frame)
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::rig_extrinsics::{
///     RigExtrinsicsProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
///     step_rig_init, step_rig_optimize,
/// };
/// # fn main() -> anyhow::Result<()> {
/// # let rig_dataset = unimplemented!();
///
/// let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
/// session.set_input(rig_dataset)?;
///
/// step_intrinsics_init_all(&mut session, None)?;
/// step_intrinsics_optimize_all(&mut session, None)?;
/// step_rig_init(&mut session, None)?;
/// step_rig_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RigExtrinsicsProblem;

impl ProblemState for RigExtrinsicsProblem {
    type State = RigExtrinsicsState;
}

impl ProblemType for RigExtrinsicsProblem {
    type Config = RigExtrinsicsConfig;
    type Input = RigExtrinsicsInput;
    type Output = RigExtrinsicsEstimate;
    type Export = RigExtrinsicsExport;

    fn name() -> &'static str {
        "rig_extrinsics_v2"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        if input.num_views() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: input.num_views(),
            });
        }

        if input.num_cameras < 2 {
            return Err(Error::InsufficientData {
                need: 2,
                got: input.num_cameras,
            });
        }

        // Check each view has correct number of cameras
        for (i, view) in input.views.iter().enumerate() {
            if view.obs.cameras.len() != input.num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {} has {} cameras, expected {}",
                    i,
                    view.obs.cameras.len(),
                    input.num_cameras
                )));
            }

            // Check at least one camera has observations in this view
            let has_obs = view.obs.cameras.iter().any(|c| c.is_some());
            if !has_obs {
                return Err(Error::invalid_input(format!(
                    "view {} has no observations from any camera",
                    i
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.solver.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.intrinsics.iterations == 0 {
            return Err(Error::invalid_input(
                "intrinsics.iterations must be positive",
            ));
        }
        validate_rig_cameras(&config.camera, &config.camera_overrides)
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        if config.rig.reference_camera_idx >= input.num_cameras {
            return Err(Error::invalid_input(format!(
                "reference_camera_idx {} is out of range (num_cameras = {})",
                config.rig.reference_camera_idx, input.num_cameras
            )));
        }
        validate_rig_camera_indices(&config.camera_overrides, input.num_cameras)
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        let params = &output.params;
        let cam_se3_rig: Vec<Iso3> = params.cam_to_rig.iter().map(|t| t.inverse()).collect();
        let rig_se3_target = params.rig_from_target.clone();
        let num_cameras = input.num_cameras;

        let target = compute_rig_target_residuals(
            &rig_camera_models(&params.cameras)?,
            input,
            &cam_se3_rig,
            &rig_se3_target,
        )?;

        let target_hist_per_camera: Vec<FeatureResidualHistogram> = (0..num_cameras)
            .map(|cam_idx| {
                build_feature_histogram(
                    target
                        .iter()
                        .filter(|r| r.camera == cam_idx)
                        .filter_map(|r| r.error_px),
                )
            })
            .collect();

        let mut per_feature_residuals = PerFeatureResiduals::default();
        per_feature_residuals.target = target;
        per_feature_residuals.target_hist_per_camera = Some(target_hist_per_camera);
        Ok(RigExtrinsicsExport {
            kind: ExportKind::RigExtrinsics,
            cameras: params.cameras.clone(),
            cam_se3_rig,
            rig_se3_target,
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
            per_feature_residuals,
            // Manifest is populated by callers that also wrote images for
            // the dataset; the pipeline itself never has image paths to
            // fill in. Mirrors the rig_handeye precedent.
            image_manifest: None,
            uncertainty: output.uncertainty.clone(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{CorrespondenceView, Pt2, Pt3, RigView, RigViewObs};

    fn make_minimal_obs() -> CorrespondenceView {
        CorrespondenceView::new(
            vec![
                Pt3::new(0.0, 0.0, 0.0),
                Pt3::new(0.05, 0.0, 0.0),
                Pt3::new(0.05, 0.05, 0.0),
                Pt3::new(0.0, 0.05, 0.0),
            ],
            vec![
                Pt2::new(100.0, 100.0),
                Pt2::new(200.0, 100.0),
                Pt2::new(200.0, 200.0),
                Pt2::new(100.0, 200.0),
            ],
        )
        .unwrap()
    }

    fn make_minimal_input() -> RigExtrinsicsInput {
        let views = (0..3)
            .map(|_| RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs()), Some(make_minimal_obs())],
                },
            })
            .collect();

        RigDataset::new(views, 2).unwrap()
    }

    #[test]
    fn validate_input_requires_3_views() {
        let input = make_minimal_input();
        let result = RigExtrinsicsProblem::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_requires_2_cameras() {
        // Create single-camera input (should fail)
        let views = (0..3)
            .map(|_| RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs())],
                },
            })
            .collect();

        let input = RigDataset::new(views, 1).unwrap();
        let result = RigExtrinsicsProblem::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("need 2"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = RigExtrinsicsConfig::default();
        let result = RigExtrinsicsProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_config_checks_reference_camera() {
        let input = make_minimal_input();
        let config = RigExtrinsicsConfig {
            rig: RigConfig {
                reference_camera_idx: 5, // Out of range
                ..RigConfig::default()
            },
            ..RigExtrinsicsConfig::default()
        };

        let result = RigExtrinsicsProblem::validate_input_config(&input, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reference_camera_idx")
        );
    }

    #[test]
    fn config_json_roundtrip() {
        let config = RigExtrinsicsConfig {
            solver: SolverConfig {
                max_iters: 100,
                robust_loss: vision_calibration_optim::RobustLoss::Huber { scale: 2.5 },
                ..SolverConfig::default()
            },
            rig: RigConfig {
                reference_camera_idx: 1,
                refine_intrinsics: true,
                refine_sensor: true,
            },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigExtrinsicsConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.solver.max_iters, 100);
        assert_eq!(restored.rig.reference_camera_idx, 1);
        assert!(restored.rig.refine_intrinsics);
        assert!(restored.rig.refine_sensor);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigExtrinsicsProblem::name(), "rig_extrinsics_v2");
        assert_eq!(RigExtrinsicsProblem::schema_version(), 1);
    }

    #[test]
    fn export_target_residuals_zero_for_perfect_rig_data() {
        // Synthesize a 2-camera 3-view rig where every observed pixel was
        // generated by projecting through the calibrated camera + extrinsics
        // + target poses. The exported per_feature_residuals.target must
        // therefore have error_px ~ 0 for every record, and per-camera
        // histograms must place all counts in the <=1 px bucket.
        use vision_calibration_core::pinhole_camera_params;
        use vision_calibration_core::{BrownConrady5, FxFyCxCy, View, make_pinhole_camera};
        use vision_calibration_optim::{RigExtrinsicsEstimate, RigExtrinsicsParams, SolveReport};

        let make_camera = || {
            make_pinhole_camera(
                FxFyCxCy {
                    fx: 800.0,
                    fy: 800.0,
                    cx: 320.0,
                    cy: 240.0,
                },
                BrownConrady5::default(),
            )
        };
        let cam0 = make_camera();
        let cam1 = make_camera();
        // T_R_C convention: cam_to_rig (line 30 of rig_extrinsics.rs).
        let cam_to_rig = vec![
            Iso3::identity(),
            Iso3::from_parts(
                nalgebra::Translation3::new(-0.2, 0.0, 0.0),
                nalgebra::UnitQuaternion::identity(),
            ),
        ];
        let cam_se3_rig: Vec<Iso3> = cam_to_rig.iter().map(|t| t.inverse()).collect();
        let rig_from_target = vec![
            Iso3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, 1.0),
                nalgebra::UnitQuaternion::identity(),
            ),
            Iso3::from_parts(
                nalgebra::Translation3::new(0.05, 0.0, 1.1),
                nalgebra::UnitQuaternion::identity(),
            ),
            Iso3::from_parts(
                nalgebra::Translation3::new(0.0, 0.05, 1.05),
                nalgebra::UnitQuaternion::identity(),
            ),
        ];
        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];

        let mut views = Vec::new();
        for rig_se3_target in &rig_from_target {
            let mut cameras = Vec::new();
            for (cam_idx, cam) in [&cam0, &cam1].iter().enumerate() {
                let cam_se3_target = cam_se3_rig[cam_idx] * rig_se3_target;
                let pixels: Vec<Pt2> = board
                    .iter()
                    .map(|p| {
                        let p_cam = cam_se3_target * p;
                        cam.project_point_c(&p_cam.coords).unwrap()
                    })
                    .collect();
                cameras.push(Some(
                    CorrespondenceView::new(board.clone(), pixels).unwrap(),
                ));
            }
            views.push(RigView {
                meta: NoMeta,
                obs: RigViewObs { cameras },
            });
        }
        let _ = View::<NoMeta>::without_meta; // silence unused import warning if any
        let dataset = RigDataset::new(views, 2).unwrap();

        let output = RigExtrinsicsEstimate {
            params: RigExtrinsicsParams {
                cameras: vec![pinhole_camera_params(&cam0), pinhole_camera_params(&cam1)],
                cam_to_rig,
                rig_from_target,
            },
            uncertainty: Default::default(),
            report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0, 0.0],
        };

        let export =
            RigExtrinsicsProblem::export(&dataset, &output, &RigExtrinsicsConfig::default())
                .expect("export");

        // 2 cameras × 3 views × 4 features = 24 records.
        assert_eq!(export.per_feature_residuals.target.len(), 24);
        for r in &export.per_feature_residuals.target {
            let err = r.error_px.unwrap_or_else(|| panic!("residual missing"));
            assert!(err < 1e-9, "error_px {err} not near zero");
        }
        let hist = export
            .per_feature_residuals
            .target_hist_per_camera
            .as_ref()
            .expect("rig export emits per-camera histograms");
        assert_eq!(hist.len(), 2);
        for (i, h) in hist.iter().enumerate() {
            assert_eq!(h.count, 12, "cam {i} count mismatch");
            assert_eq!(h.counts, [12, 0, 0, 0, 0], "cam {i} buckets");
        }
        assert!(export.per_feature_residuals.laser.is_empty());
        assert!(export.per_feature_residuals.laser_hist_per_camera.is_none());
    }

    #[test]
    fn export_per_feature_residuals_json_roundtrip() {
        // Round-trip the rig export through JSON.
        use vision_calibration_core::pinhole_camera_params;
        use vision_calibration_core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};
        use vision_calibration_optim::{RigExtrinsicsEstimate, RigExtrinsicsParams, SolveReport};

        let camera = make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            },
            BrownConrady5::default(),
        );
        // One pose, one view, two cameras with the same model — produces
        // 8 records.
        let cam_to_rig = vec![Iso3::identity(), Iso3::identity()];
        let rig_from_target = vec![Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        )];
        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let cam_se3_target = rig_from_target[0];
        let pixels: Vec<Pt2> = board
            .iter()
            .map(|p| {
                let p_cam = cam_se3_target * p;
                camera.project_point_c(&p_cam.coords).unwrap()
            })
            .collect();
        let view = RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![
                    Some(CorrespondenceView::new(board.clone(), pixels.clone()).unwrap()),
                    Some(CorrespondenceView::new(board.clone(), pixels).unwrap()),
                ],
            },
        };
        let dataset = RigDataset::new(vec![view], 2).unwrap();

        let output = RigExtrinsicsEstimate {
            params: RigExtrinsicsParams {
                cameras: vec![pinhole_camera_params(&camera); 2],
                cam_to_rig,
                rig_from_target,
            },
            uncertainty: Default::default(),
            report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0, 0.0],
        };
        let export =
            RigExtrinsicsProblem::export(&dataset, &output, &RigExtrinsicsConfig::default())
                .expect("export");

        let json = serde_json::to_string(&export).unwrap();
        let restored: RigExtrinsicsExport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.per_feature_residuals.target.len(),
            export.per_feature_residuals.target.len()
        );
        assert_eq!(
            restored.per_feature_residuals.target_hist_per_camera,
            export.per_feature_residuals.target_hist_per_camera
        );
    }

    fn make_minimal_export() -> RigExtrinsicsExport {
        // Smallest export the trait can produce — one pose, two cameras
        // with identical pinhole models and identity extrinsics. Used
        // by the manifest tests below; never round-tripped through the
        // calibration pipeline.
        use vision_calibration_core::pinhole_camera_params;
        use vision_calibration_core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};
        use vision_calibration_optim::{RigExtrinsicsEstimate, RigExtrinsicsParams, SolveReport};

        let camera = make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            },
            BrownConrady5::default(),
        );
        let cam_to_rig = vec![Iso3::identity(), Iso3::identity()];
        let rig_from_target = vec![Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        )];
        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let cam_se3_target = rig_from_target[0];
        let pixels: Vec<Pt2> = board
            .iter()
            .map(|p| {
                let p_cam = cam_se3_target * p;
                camera.project_point_c(&p_cam.coords).unwrap()
            })
            .collect();
        let view = RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![
                    Some(CorrespondenceView::new(board.clone(), pixels.clone()).unwrap()),
                    Some(CorrespondenceView::new(board.clone(), pixels).unwrap()),
                ],
            },
        };
        let dataset = RigDataset::new(vec![view], 2).unwrap();
        let output = RigExtrinsicsEstimate {
            params: RigExtrinsicsParams {
                cameras: vec![pinhole_camera_params(&camera); 2],
                cam_to_rig,
                rig_from_target,
            },
            uncertainty: Default::default(),
            report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0, 0.0],
        };
        RigExtrinsicsProblem::export(&dataset, &output, &RigExtrinsicsConfig::default())
            .expect("export")
    }

    #[test]
    fn export_image_manifest_defaults_absent_from_wire() {
        // ADR 0014: when the pipeline emits an export the manifest is
        // None and `skip_serializing_if` keeps existing serialized JSON byte-stable.
        let export = make_minimal_export();
        assert!(export.image_manifest.is_none());

        let json = serde_json::to_string(&export).unwrap();
        assert!(
            !json.contains("image_manifest"),
            "absent manifest must not appear on the wire"
        );
        let restored: RigExtrinsicsExport = serde_json::from_str(&json).unwrap();
        assert!(restored.image_manifest.is_none());
    }

    #[test]
    fn export_image_manifest_roundtrip_with_tiled_roi() {
        // A tiled six-camera rig writes a single 4320×540 PNG per pose
        // and references each of its six 720×540 camera tiles via ROI;
        // mirror that shape here so any future change to PixelRect or
        // FrameRef serialization is caught at PR time.
        use vision_calibration_core::{FrameRef, ImageManifest, PixelRect};

        let mut export = make_minimal_export();
        let frames: Vec<FrameRef> = (0..6)
            .map(|cam| {
                let mut roi = PixelRect::default();
                roi.x = (cam as u32) * 720;
                roi.w = 720;
                roi.h = 540;
                let mut frame = FrameRef::default();
                frame.pose = 0;
                frame.camera = cam;
                frame.path = std::path::PathBuf::from("target_0.png");
                frame.roi = Some(roi);
                frame
            })
            .collect();
        let mut manifest = ImageManifest::default();
        manifest.root = std::path::PathBuf::from(".");
        manifest.frames = frames;
        export.image_manifest = Some(manifest);

        let json = serde_json::to_string(&export).unwrap();
        let restored: RigExtrinsicsExport = serde_json::from_str(&json).unwrap();
        let manifest = restored
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.frames.len(), 6);
        assert_eq!(manifest.frames[0].camera, 0);
        assert_eq!(manifest.frames[5].roi.unwrap().x, 5 * 720);
    }
}
