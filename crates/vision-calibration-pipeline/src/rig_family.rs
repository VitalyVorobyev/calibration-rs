//! Shared helpers for multi-camera rig calibration workflows.
//!
//! Internal to the pipeline crate. Consolidates **sensor-axis-aware** helpers
//! used by `rig_extrinsics`, `rig_handeye`, `rig_laserline_device`, and their
//! Scheimpflug siblings.
//!
//! Per the Track A `rig_family` refactor, the shared abstraction is
//! sensor-axis-only: each workflow module keeps its own `state.rs`,
//! `problem.rs`, and `steps.rs` and consumes these helpers for sensor-aware
//! bootstrap. The workflow axis (extrinsics-only / hand-eye / laser-plane)
//! stays per-module because each has different state fields, step sequences,
//! residual blocks, and export shapes.
//!
//! This module is registered as `mod rig_family;` (private) in
//! `lib.rs` and not re-exported.
//!
//! Mirrors the [`crate::planar_family`] precedent (single-camera planar
//! bootstrap helpers).
//!
//! ## Roadmap
//!
//! - **A6.1** — Introduce types + `bootstrap_rig_intrinsics` (this file). No
//!   migration of existing rig modules.
//! - **A6.2** — Migrate `rig_extrinsics` + collapse `rig_scheimpflug_extrinsics`
//!   into it.
//! - **A6.3** — Same for `rig_handeye` family.
//! - **A6.4** — `rig_laserline_device` (if non-trivial) + ADR + roadmap hygiene.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Mat3, NoMeta, PinholeCamera,
    PlanarDataset, Pt2, Real, ScheimpflugParams, View, make_pinhole_camera,
};
use vision_calibration_linear::{
    IterativeIntrinsicsOptions, dlt_homography, estimate_intrinsics_iterative,
    estimate_planar_pose_from_h,
};

/// Per-camera intrinsics bundle covering both pinhole and Scheimpflug rigs.
///
/// `scheimpflug` is `None` for pinhole-only rigs and `Some(_)` when each
/// camera also carries a Scheimpflug tilt model.
///
/// **Invariant:** when `scheimpflug` is `Some`, its length equals
/// `cameras.len()`. Use [`Self::pinhole`] or [`Self::scheimpflug`] to construct
/// to enforce this; direct field access is allowed inside the pipeline crate
/// for migration ergonomics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RigSensorBundle {
    pub cameras: Vec<PinholeCamera>,
    pub scheimpflug: Option<Vec<ScheimpflugParams>>,
}

impl RigSensorBundle {
    pub fn pinhole(cameras: Vec<PinholeCamera>) -> Self {
        Self {
            cameras,
            scheimpflug: None,
        }
    }

    pub fn scheimpflug(cameras: Vec<PinholeCamera>, sensors: Vec<ScheimpflugParams>) -> Self {
        debug_assert_eq!(
            cameras.len(),
            sensors.len(),
            "RigSensorBundle::scheimpflug: cameras vs sensors length mismatch"
        );
        Self {
            cameras,
            scheimpflug: Some(sensors),
        }
    }

    /// Number of cameras in the bundle.
    #[allow(dead_code)] // Public accessor; consumed by upcoming A6.2b/A6.3 collapse.
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// `true` when the bundle carries Scheimpflug sensor params.
    #[allow(dead_code)] // Public accessor; consumed by upcoming A6.2b/A6.3 collapse.
    pub fn is_scheimpflug(&self) -> bool {
        self.scheimpflug.is_some()
    }
}

/// Manual seeds for the per-camera rig intrinsics bootstrap.
///
/// All fields are `Option<T>` per ADR 0011. A `None` field runs the
/// corresponding auto step. `per_cam_sensors` is consulted only when
/// [`bootstrap_rig_intrinsics`] is called with [`SensorFlavour::Scheimpflug`];
/// supplying it for a pinhole bootstrap is silently ignored.
///
/// Per-camera vectors must equal `num_cameras` (validated by
/// [`bootstrap_rig_intrinsics`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct RigIntrinsicsSeeds {
    pub per_cam_intrinsics: Option<Vec<FxFyCxCySkew<Real>>>,
    pub per_cam_distortion: Option<Vec<BrownConrady5<Real>>>,
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
}

/// Sensor flavour selector for [`bootstrap_rig_intrinsics`].
///
/// `Scheimpflug { default_tilt_x, default_tilt_y }` supplies the per-camera
/// fallback tilt when `seeds.per_cam_sensors` is `None`. The defaults are
/// typically read from the workflow's `*Config::init_tilt_x/y`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SensorFlavour {
    Pinhole,
    Scheimpflug {
        default_tilt_x: Real,
        default_tilt_y: Real,
    },
}

/// Bootstrap result for the per-camera intrinsics stage.
#[derive(Debug, Clone)]
pub(crate) struct RigIntrinsicsBootstrap {
    /// Per-camera intrinsics + (optionally) per-camera Scheimpflug params.
    pub bundle: RigSensorBundle,
    /// `[view][cam] -> Option<Iso3>`, `cam_se3_target` (T_C_T).
    pub per_cam_target_poses: Vec<Vec<Option<Iso3>>>,
    /// Manually-seeded fields, for the session log.
    pub manual_fields: Vec<&'static str>,
    /// Auto-fitted fields, for the session log.
    pub auto_fields: Vec<&'static str>,
}

/// Per-camera rig intrinsics bootstrap shared by all rig workflows.
///
/// Loops over cameras, extracts per-camera planar views via the supplied
/// closure, and either uses seeded intrinsics or runs the iterative auto-fit.
/// Per-camera target poses are recovered from per-camera homographies using
/// whichever intrinsics were chosen.
///
/// For [`SensorFlavour::Scheimpflug`], per-camera sensor params are seeded
/// from `seeds.per_cam_sensors` if present, else from the flavour's defaults.
///
/// # Errors
///
/// - Any per-camera Vec seed length disagrees with `num_cameras`.
/// - A camera has fewer than 3 valid views.
/// - DLT homography or pose recovery fails on any view.
/// - Iterative intrinsics estimation fails (auto-fit only).
pub(crate) fn bootstrap_rig_intrinsics<F>(
    num_cameras: usize,
    num_views: usize,
    extract_views: F,
    seeds: RigIntrinsicsSeeds,
    init_opts: IterativeIntrinsicsOptions,
    flavour: SensorFlavour,
) -> Result<RigIntrinsicsBootstrap, Error>
where
    F: Fn(usize) -> Vec<Option<View<NoMeta>>>,
{
    validate_seed_lengths(&seeds, num_cameras, &flavour)?;
    let (manual_fields, auto_fields) = classify_seed_provenance(&seeds, &flavour);

    let mut cameras = Vec::with_capacity(num_cameras);
    let mut sensors = match flavour {
        SensorFlavour::Pinhole => None,
        SensorFlavour::Scheimpflug { .. } => Some(Vec::with_capacity(num_cameras)),
    };
    let mut per_cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];

    #[allow(clippy::needless_range_loop)]
    for cam_idx in 0..num_cameras {
        let cam_views = extract_views(cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let camera = build_camera_for_index(cam_idx, &seeds, &planar_dataset, init_opts)?;
        let k_matrix = intrinsics_k_matrix(&camera.k);

        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            let view = &planar_dataset.views[local_idx];
            let pose = estimate_target_pose(&k_matrix, &view.obs).map_err(|e| {
                Error::numerical(format!(
                    "pose estimation failed for cam {cam_idx} view {global_idx}: {e}"
                ))
            })?;
            per_cam_target_poses[global_idx][cam_idx] = Some(pose);
        }

        if let Some(s) = sensors.as_mut() {
            s.push(sensor_for_index(cam_idx, &seeds, &flavour));
        }

        cameras.push(camera);
    }

    let bundle = match sensors {
        Some(s) => RigSensorBundle::scheimpflug(cameras, s),
        None => RigSensorBundle::pinhole(cameras),
    };

    Ok(RigIntrinsicsBootstrap {
        bundle,
        per_cam_target_poses,
        manual_fields,
        auto_fields,
    })
}

fn validate_seed_lengths(
    seeds: &RigIntrinsicsSeeds,
    num_cameras: usize,
    flavour: &SensorFlavour,
) -> Result<(), Error> {
    if let Some(s) = &seeds.per_cam_intrinsics
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_intrinsics length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }
    if let Some(s) = &seeds.per_cam_distortion
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_distortion length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }
    if matches!(flavour, SensorFlavour::Scheimpflug { .. })
        && let Some(s) = &seeds.per_cam_sensors
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_sensors length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }
    Ok(())
}

fn classify_seed_provenance(
    seeds: &RigIntrinsicsSeeds,
    flavour: &SensorFlavour,
) -> (Vec<&'static str>, Vec<&'static str>) {
    let mut manual: Vec<&'static str> = Vec::new();
    let mut auto: Vec<&'static str> = Vec::new();
    if seeds.per_cam_intrinsics.is_some() {
        manual.push("per_cam_intrinsics");
    } else {
        auto.push("per_cam_intrinsics");
    }
    if seeds.per_cam_distortion.is_some() {
        manual.push("per_cam_distortion");
    } else {
        auto.push("per_cam_distortion");
    }
    if matches!(flavour, SensorFlavour::Scheimpflug { .. }) {
        if seeds.per_cam_sensors.is_some() {
            manual.push("per_cam_sensors");
        } else {
            auto.push("per_cam_sensors");
        }
    }
    (manual, auto)
}

fn build_camera_for_index(
    cam_idx: usize,
    seeds: &RigIntrinsicsSeeds,
    planar_dataset: &PlanarDataset,
    init_opts: IterativeIntrinsicsOptions,
) -> Result<PinholeCamera, Error> {
    if let Some(intrinsics_seeds) = seeds.per_cam_intrinsics.as_ref() {
        let k = intrinsics_seeds[cam_idx];
        let dist = seeds
            .per_cam_distortion
            .as_ref()
            .map(|d| d[cam_idx])
            .unwrap_or_default();
        return Ok(make_pinhole_camera(k, dist));
    }
    let bootstrap = estimate_intrinsics_iterative(planar_dataset, init_opts).map_err(|e| {
        Error::numerical(format!(
            "intrinsics estimation failed for camera {cam_idx}: {e}"
        ))
    })?;
    let dist = seeds
        .per_cam_distortion
        .as_ref()
        .map(|d| d[cam_idx])
        .unwrap_or(bootstrap.dist);
    Ok(make_pinhole_camera(bootstrap.k, dist))
}

fn sensor_for_index(
    cam_idx: usize,
    seeds: &RigIntrinsicsSeeds,
    flavour: &SensorFlavour,
) -> ScheimpflugParams {
    if let Some(per_cam_sensors) = seeds.per_cam_sensors.as_ref() {
        return per_cam_sensors[cam_idx];
    }
    match *flavour {
        SensorFlavour::Pinhole => ScheimpflugParams::default(),
        SensorFlavour::Scheimpflug {
            default_tilt_x,
            default_tilt_y,
        } => ScheimpflugParams {
            tilt_x: default_tilt_x,
            tilt_y: default_tilt_y,
        },
    }
}

/// Build a planar dataset from per-view per-camera observations, returning
/// the dataset and the original (global) view indices it covers.
///
/// `views[i] = None` means the camera does not observe the target in view `i`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if fewer than 3 views have observations.
pub(crate) fn views_to_planar_dataset(
    views: &[Option<View<NoMeta>>],
) -> Result<(PlanarDataset, Vec<usize>), Error> {
    let (valid_views, indices): (Vec<_>, Vec<_>) = views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|view| (view.clone(), i)))
        .unzip();

    if valid_views.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: valid_views.len(),
        });
    }
    let dataset = PlanarDataset::new(valid_views)?;
    Ok((dataset, indices))
}

/// Estimate the target pose from camera intrinsics and view observations
/// using DLT homography + planar pose recovery.
pub(crate) fn estimate_target_pose(
    k_matrix: &Mat3,
    obs: &CorrespondenceView,
) -> Result<Iso3, Error> {
    let board_2d: Vec<Pt2> = obs.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = obs.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();

    let h = dlt_homography(&board_2d, &pixel_2d)
        .map_err(|e| Error::numerical(format!("failed to compute homography: {e}")))?;
    estimate_planar_pose_from_h(k_matrix, &h)
        .map_err(|e| Error::numerical(format!("failed to recover pose from homography: {e}")))
}

/// 3x3 K matrix from intrinsics parameters.
pub(crate) fn intrinsics_k_matrix(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

/// Format a "(manual: …; auto: …)" provenance string for session logs.
pub(crate) fn format_init_source(manual: &[&str], auto: &[&str]) -> String {
    match (manual.is_empty(), auto.is_empty()) {
        (false, false) => format!("(manual: {}; auto: {})", manual.join(", "), auto.join(", ")),
        (false, true) => format!("(manual: {})", manual.join(", ")),
        (true, false) => format!("(auto: {})", auto.join(", ")),
        (true, true) => "(empty)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, Camera, FxFyCxCySkew, Pinhole, ScheimpflugParams, View, make_pinhole_camera,
        synthetic::planar,
    };
    use vision_calibration_linear::DistortionFitOptions;

    /// `[cam][view] -> Option<View<NoMeta>>`.
    type PerCamViews = Vec<Vec<Option<View<NoMeta>>>>;

    fn default_init_opts() -> IterativeIntrinsicsOptions {
        IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
            zero_skew: true,
        }
    }

    fn pinhole_two_camera_views() -> (PerCamViews, FxFyCxCySkew<Real>, usize) {
        let cam_gt = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 900.0,
                fy: 880.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        let board = planar::grid_points(6, 5, 0.04);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.1, 0.6, 0.05);
        let views_a = planar::project_views_all(&cam_gt, &board, &poses).expect("views a");
        let views_b = planar::project_views_all(&cam_gt, &board, &poses).expect("views b");
        let num_views = views_a.len();
        let per_cam_views: PerCamViews = vec![
            views_a
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
            views_b
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
        ];
        (per_cam_views, cam_gt.k, num_views)
    }

    fn scheimpflug_two_camera_views(
        sensor: ScheimpflugParams,
    ) -> (PerCamViews, FxFyCxCySkew<Real>, usize) {
        let base = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        let camera = Camera::new(Pinhole, base.dist, sensor.compile(), base.k);
        let board = planar::grid_points(6, 5, 0.03);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.08, 0.55, 0.03);
        let views_a = planar::project_views_all(&camera, &board, &poses).expect("views a");
        let views_b = planar::project_views_all(&camera, &board, &poses).expect("views b");
        let num_views = views_a.len();
        let per_cam_views: PerCamViews = vec![
            views_a
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
            views_b
                .into_iter()
                .map(|cv| Some(View::without_meta(cv)))
                .collect(),
        ];
        (per_cam_views, base.k, num_views)
    }

    #[test]
    fn bundle_constructors_enforce_invariants() {
        let cams = vec![make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        )];
        let pinhole = RigSensorBundle::pinhole(cams.clone());
        assert_eq!(pinhole.num_cameras(), 1);
        assert!(!pinhole.is_scheimpflug());

        let scheim = RigSensorBundle::scheimpflug(cams, vec![ScheimpflugParams::default()]);
        assert!(scheim.is_scheimpflug());
        assert_eq!(scheim.scheimpflug.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn bundle_serde_roundtrip() {
        let cams = vec![make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        )];
        let sensor = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.01,
        };
        let bundle = RigSensorBundle::scheimpflug(cams, vec![sensor]);

        let json = serde_json::to_string_pretty(&bundle).expect("serialize");
        let restored: RigSensorBundle = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.num_cameras(), 1);
        assert!(restored.is_scheimpflug());
        assert_eq!(restored.cameras[0].k.fx, 800.0);
        let sensors = restored.scheimpflug.as_ref().unwrap();
        assert_eq!(sensors.len(), 1);
        assert_eq!(sensors[0].tilt_x, sensor.tilt_x);
        assert_eq!(sensors[0].tilt_y, sensor.tilt_y);
    }

    #[test]
    fn bootstrap_pinhole_recovers_synthetic_intrinsics() {
        let (per_cam_views, k_gt, num_views) = pinhole_two_camera_views();
        let result = bootstrap_rig_intrinsics(
            2,
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            RigIntrinsicsSeeds::default(),
            default_init_opts(),
            SensorFlavour::Pinhole,
        )
        .expect("bootstrap");

        assert_eq!(result.bundle.num_cameras(), 2);
        assert!(!result.bundle.is_scheimpflug());
        assert_eq!(result.per_cam_target_poses.len(), num_views);
        for cam in &result.bundle.cameras {
            assert!((cam.k.fx - k_gt.fx).abs() / k_gt.fx < 0.05, "fx off");
            assert!((cam.k.fy - k_gt.fy).abs() / k_gt.fy < 0.05, "fy off");
        }
        assert!(result.manual_fields.is_empty());
        assert_eq!(
            result.auto_fields,
            vec!["per_cam_intrinsics", "per_cam_distortion"]
        );
    }

    #[test]
    fn bootstrap_scheimpflug_recovers_synthetic_intrinsics() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let (per_cam_views, k_gt, num_views) = scheimpflug_two_camera_views(sensor_gt);
        let result = bootstrap_rig_intrinsics(
            2,
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            RigIntrinsicsSeeds::default(),
            default_init_opts(),
            SensorFlavour::Scheimpflug {
                default_tilt_x: 0.0,
                default_tilt_y: 0.0,
            },
        )
        .expect("bootstrap");

        assert_eq!(result.bundle.num_cameras(), 2);
        assert!(result.bundle.is_scheimpflug());
        let sensors = result.bundle.scheimpflug.as_ref().unwrap();
        assert_eq!(sensors.len(), 2);
        // Auto-fitted sensors take the default since per_cam_sensors was None.
        for s in sensors {
            assert_eq!(s.tilt_x, 0.0);
            assert_eq!(s.tilt_y, 0.0);
        }
        for cam in &result.bundle.cameras {
            // Scheimpflug-projected views are still close to pinhole at small tilts.
            assert!(cam.k.fx.is_finite());
            assert!(cam.k.fy.is_finite());
            assert!((cam.k.cx - k_gt.cx).abs() < k_gt.cx * 0.2);
        }
        assert_eq!(
            result.auto_fields,
            vec![
                "per_cam_intrinsics",
                "per_cam_distortion",
                "per_cam_sensors"
            ]
        );
    }

    #[test]
    fn bootstrap_with_intrinsics_seed_skips_autofit() {
        let (per_cam_views, k_gt, num_views) = pinhole_two_camera_views();
        let seeds = RigIntrinsicsSeeds {
            per_cam_intrinsics: Some(vec![k_gt, k_gt]),
            ..Default::default()
        };
        let result = bootstrap_rig_intrinsics(
            2,
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            seeds,
            default_init_opts(),
            SensorFlavour::Pinhole,
        )
        .expect("bootstrap");

        for cam in &result.bundle.cameras {
            assert_eq!(cam.k.fx, k_gt.fx);
            assert_eq!(cam.k.fy, k_gt.fy);
        }
        assert_eq!(result.manual_fields, vec!["per_cam_intrinsics"]);
        assert_eq!(result.auto_fields, vec!["per_cam_distortion"]);
    }

    #[test]
    fn bootstrap_scheimpflug_seeded_sensors_used_verbatim() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let (per_cam_views, _, num_views) = scheimpflug_two_camera_views(sensor_gt);
        let seeds = RigIntrinsicsSeeds {
            per_cam_sensors: Some(vec![sensor_gt, sensor_gt]),
            ..Default::default()
        };
        let result = bootstrap_rig_intrinsics(
            2,
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            seeds,
            default_init_opts(),
            SensorFlavour::Scheimpflug {
                default_tilt_x: 0.0,
                default_tilt_y: 0.0,
            },
        )
        .expect("bootstrap");

        for s in result.bundle.scheimpflug.as_ref().unwrap() {
            assert_eq!(s.tilt_x, sensor_gt.tilt_x);
            assert_eq!(s.tilt_y, sensor_gt.tilt_y);
        }
        assert!(result.manual_fields.contains(&"per_cam_sensors"));
    }

    #[test]
    fn bootstrap_rejects_seed_length_mismatch() {
        let (per_cam_views, k_gt, num_views) = pinhole_two_camera_views();
        let seeds = RigIntrinsicsSeeds {
            per_cam_intrinsics: Some(vec![k_gt]), // wrong length: 1 vs 2 cameras
            ..Default::default()
        };
        let err = bootstrap_rig_intrinsics(
            2,
            num_views,
            |cam_idx| per_cam_views[cam_idx].clone(),
            seeds,
            default_init_opts(),
            SensorFlavour::Pinhole,
        )
        .expect_err("seed length mismatch should error");
        let msg = err.to_string();
        assert!(msg.contains("per_cam_intrinsics"), "got: {msg}");
    }

    #[test]
    fn bootstrap_rejects_insufficient_views() {
        let (per_cam_views, _, num_views) = pinhole_two_camera_views();
        // Only present the first view; cam_idx 0 will have <3 valid views.
        let truncated: PerCamViews = vec![
            vec![per_cam_views[0][0].clone(), None, None, None, None],
            per_cam_views[1].clone(),
        ];
        let err = bootstrap_rig_intrinsics(
            2,
            num_views,
            |cam_idx| truncated[cam_idx].clone(),
            RigIntrinsicsSeeds::default(),
            default_init_opts(),
            SensorFlavour::Pinhole,
        )
        .expect_err("insufficient views should error");
        let msg = err.to_string();
        assert!(msg.contains("camera 0"), "got: {msg}");
    }

    #[test]
    fn format_init_source_cases() {
        assert_eq!(format_init_source(&[], &[]), "(empty)");
        assert_eq!(format_init_source(&["a"], &[]), "(manual: a)");
        assert_eq!(format_init_source(&[], &["b", "c"]), "(auto: b, c)");
        assert_eq!(format_init_source(&["a"], &["b"]), "(manual: a; auto: b)");
    }

    #[test]
    fn intrinsics_k_matrix_layout() {
        let k = intrinsics_k_matrix(&FxFyCxCySkew {
            fx: 100.0,
            fy: 200.0,
            cx: 50.0,
            cy: 60.0,
            skew: 1.0,
        });
        assert_eq!(k[(0, 0)], 100.0);
        assert_eq!(k[(0, 1)], 1.0);
        assert_eq!(k[(0, 2)], 50.0);
        assert_eq!(k[(1, 1)], 200.0);
        assert_eq!(k[(1, 2)], 60.0);
        assert_eq!(k[(2, 2)], 1.0);
        assert_eq!(k[(1, 0)], 0.0);
        assert_eq!(k[(2, 0)], 0.0);
        assert_eq!(k[(2, 1)], 0.0);
    }
}
