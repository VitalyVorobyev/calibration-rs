//! Shared helpers for calibrating one camera from planar target views.
//!
//! This module is internal to the pipeline crate. It holds the planar
//! bootstrap (homographies, Zhang's method, per-view poses) and the
//! per-camera linear bootstrap and refinement every workflow runs for each
//! of its cameras — planar intrinsics, the rig family, single-camera
//! hand-eye and the laserline device — each camera in its own model
//! ([`CameraModelSpec`]).

use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    DistortionParams, FxFyCxCy, Iso3, Mat3, PinholeCamera, PlanarDataset, ProjectionKind,
    ProjectionParams, Pt2, Real, ScheimpflugParams, SensorKind, SensorParams, make_pinhole_camera,
};

use crate::Error;
use crate::common::config::{CameraConfig, CameraInitOptions, SolverConfig};
use vision_calibration_linear::prelude::{
    EquidistantInitOptions, IterativeIntrinsicsOptions, dlt_homography,
    estimate_equidistant_intrinsics, estimate_intrinsics_iterative,
    estimate_planar_pose_from_bearings, estimate_planar_pose_from_h,
};
use vision_calibration_linear::scheimpflug_init::{
    ScheimpflugIntrinsicsInitOptions as LinearScheimpflugIntrinsicsInitOptions,
    estimate_scheimpflug_intrinsics_iterative,
};
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, ScheimpflugStagedInitOptions, optimize_planar_intrinsics,
    optimize_scheimpflug_intrinsics_staged,
};

/// Result of shared planar bootstrap (homographies + initial camera + poses).
#[derive(Debug, Clone)]
pub(crate) struct PlanarBootstrap {
    pub homographies: Vec<Mat3>,
    pub camera: PinholeCamera,
    pub poses: Vec<Iso3>,
}

/// Estimate homographies, intrinsics/distortion, and per-view poses.
pub(crate) fn bootstrap_planar_intrinsics(
    dataset: &PlanarDataset,
    init_opts: IterativeIntrinsicsOptions,
) -> Result<PlanarBootstrap, Error> {
    let homographies = estimate_view_homographies(dataset)?;
    let camera = estimate_intrinsics_iterative(dataset, init_opts).map_err(|e| {
        Error::numerical(format!(
            "iterative planar intrinsics estimation failed: {e}"
        ))
    })?;
    let poses = recover_planar_poses_from_homographies(&homographies, &camera.k)?;

    Ok(PlanarBootstrap {
        homographies,
        camera,
        poses,
    })
}

pub(crate) fn estimate_view_homographies(dataset: &PlanarDataset) -> Result<Vec<Mat3>, Error> {
    let mut homographies = Vec::with_capacity(dataset.num_views());
    for (idx, view) in dataset.views.iter().enumerate() {
        let (board_2d, pixel_2d) = board_and_pixel_points(&view.obs);
        let h = dlt_homography(&board_2d, &pixel_2d).map_err(|e| {
            Error::numerical(format!(
                "failed to compute homography for view {} (need >=4 well-conditioned points): {e}",
                idx
            ))
        })?;
        homographies.push(h);
    }
    Ok(homographies)
}

pub(crate) fn recover_planar_poses_from_homographies(
    homographies: &[Mat3],
    intrinsics: &FxFyCxCy<Real>,
) -> Result<Vec<Iso3>, Error> {
    let k = intrinsics_k_matrix(intrinsics);
    let mut poses = Vec::with_capacity(homographies.len());
    for (idx, h) in homographies.iter().enumerate() {
        let pose = estimate_planar_pose_from_h(&k, h).map_err(|e| {
            Error::numerical(format!("failed to recover pose for view {}: {e}", idx))
        })?;
        poses.push(pose);
    }
    Ok(poses)
}

fn board_and_pixel_points(view: &CorrespondenceView) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
    (board_2d, pixel_2d)
}

pub(crate) fn intrinsics_k_matrix(k: &FxFyCxCy<Real>) -> Mat3 {
    Mat3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

/// Estimate the target pose from camera intrinsics and view observations
/// using DLT homography + planar pose recovery.
pub(crate) fn estimate_target_pose(
    k_matrix: &Mat3,
    obs: &CorrespondenceView,
) -> Result<Iso3, Error> {
    let (board_2d, pixel_2d) = board_and_pixel_points(obs);
    let h = dlt_homography(&board_2d, &pixel_2d)
        .map_err(|e| Error::numerical(format!("failed to compute homography: {e}")))?;
    estimate_planar_pose_from_h(k_matrix, &h)
        .map_err(|e| Error::numerical(format!("failed to recover pose from homography: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-camera bootstrap and refinement
// ─────────────────────────────────────────────────────────────────────────────

/// Manual seeds of one camera's linear bootstrap; a `None` field is
/// auto-fitted.
#[derive(Debug, Clone, Default)]
pub(crate) struct CameraSeed {
    /// Intrinsics; seeding them skips Zhang's method.
    pub intrinsics: Option<FxFyCxCy<Real>>,
    /// Distortion in the camera's own model ([`check_distortion_seed`]),
    /// used as given; zero when the intrinsics are seeded and this is not.
    pub distortion: Option<DistortionParams>,
    /// Sensor tilt; a non-zero tilt needs a Scheimpflug camera
    /// ([`check_sensor_seed`]).
    pub sensor: Option<ScheimpflugParams>,
}

/// One camera's linear bootstrap.
#[derive(Debug, Clone)]
pub(crate) struct CameraBootstrap {
    /// The camera's initial parameters, in its model.
    pub camera: CameraParams,
    /// Per-view target poses (`cam_se3_target`), one per dataset view.
    pub poses: Vec<Iso3>,
}

/// Check a distortion seed is a distortion of `model`'s kind: a seed is a
/// value in the camera's own model, never converted.
///
/// # Errors
///
/// [`Error::InvalidInput`] naming `field` when the seed's kind differs from
/// the camera's configured distortion.
pub(crate) fn check_distortion_seed(
    seed: &DistortionParams,
    model: CameraModelSpec,
    field: &str,
) -> Result<(), Error> {
    if seed.kind() != model.distortion {
        return Err(Error::invalid_input(format!(
            "{field}: the seed is a {:?} distortion; the camera's distortion model is {:?}",
            seed.kind(),
            model.distortion
        )));
    }
    Ok(())
}

/// Check a tilt seed fits `model`: a non-zero tilt needs a Scheimpflug
/// sensor. A zero tilt is accepted for any camera — it is the identity a
/// device spec gives a frontal camera.
///
/// # Errors
///
/// [`Error::InvalidInput`] naming `field` for a non-zero tilt of an
/// untilted camera.
pub(crate) fn check_sensor_seed(
    model: CameraModelSpec,
    seed: &ScheimpflugParams,
    field: &str,
) -> Result<(), Error> {
    if model.sensor != SensorKind::Scheimpflug && (seed.tilt_x != 0.0 || seed.tilt_y != 0.0) {
        return Err(Error::invalid_input(format!(
            "{field}: a non-zero tilt seed needs a Scheimpflug camera; the camera's sensor is \
             {:?}",
            model.sensor
        )));
    }
    Ok(())
}

/// Linear bootstrap of camera `cam_idx` (named in errors) of `model` from its
/// planar views, by the initializer of the model's projection:
/// [`bootstrap_pinhole`] or [`bootstrap_fisheye`].
///
/// The caller checks the tilt and distortion seeds against the model
/// ([`check_sensor_seed`], [`check_distortion_seed`]).
///
/// # Errors
///
/// [`Error::InvalidInput`] for a model without an initializer;
/// otherwise those of the initializer.
pub(crate) fn bootstrap_camera(
    cam_idx: usize,
    model: CameraModelSpec,
    dataset: &PlanarDataset,
    seed: &CameraSeed,
    init_opts: CameraInitOptions,
) -> Result<CameraBootstrap, Error> {
    match model.projection {
        ProjectionKind::Pinhole => {
            bootstrap_pinhole(cam_idx, model, dataset, seed, init_opts.pinhole)
        }
        ProjectionKind::Equidistant => {
            bootstrap_fisheye(cam_idx, model, dataset, seed, init_opts.fisheye)
        }
        other => Err(unsupported_projection(other)),
    }
}

/// Linear bootstrap of a pinhole camera: seeded or auto-fitted intrinsics —
/// Zhang's method, or for a Scheimpflug sensor without seeded intrinsics the
/// tilted variant that also sweeps the tilt (a seeded tilt replaces the
/// sweep) — then per-view poses. The linear stage fits Brown-Conrady
/// coefficients, embedded into the model's distortion (extra coefficients
/// start at zero); a distortion seed replaces the result as given and feeds
/// no linear step.
///
/// # Errors
///
/// [`Error::Numerical`] when Zhang's method, the tilted linear init, a
/// homography or a pose recovery fails.
fn bootstrap_pinhole(
    cam_idx: usize,
    model: CameraModelSpec,
    dataset: &PlanarDataset,
    seed: &CameraSeed,
    init_opts: IterativeIntrinsicsOptions,
) -> Result<CameraBootstrap, Error> {
    let (camera, sensor, poses) = match model.sensor {
        SensorKind::Identity => {
            let camera = seeded_or_fitted_camera(cam_idx, seed, dataset, init_opts)?;
            let poses = estimate_camera_poses(cam_idx, dataset, &camera.k)?;
            (camera, None, poses)
        }
        SensorKind::Scheimpflug if seed.intrinsics.is_some() => {
            let camera = seeded_or_fitted_camera(cam_idx, seed, dataset, init_opts)?;
            let poses = estimate_camera_poses(cam_idx, dataset, &camera.k)?;
            (camera, Some(seed.sensor.unwrap_or_default()), poses)
        }
        SensorKind::Scheimpflug => {
            let mut scheimpflug_opts = LinearScheimpflugIntrinsicsInitOptions {
                iterations: init_opts.iterations,
                distortion_opts: init_opts.distortion_opts,
                ..Default::default()
            };
            if let Some(manual) = seed.sensor {
                scheimpflug_opts.tilt_x_seeds = vec![manual.tilt_x];
                scheimpflug_opts.tilt_y_seeds = vec![manual.tilt_y];
                scheimpflug_opts.refine_rounds = 0;
            }
            let mut init = estimate_scheimpflug_intrinsics_iterative(dataset, scheimpflug_opts)
                .map_err(|e| {
                    Error::numerical(format!(
                        "Scheimpflug intrinsics initialization failed for camera {cam_idx}: {e}"
                    ))
                })?;
            if let Some(sensor) = seed.sensor {
                init.sensor = sensor;
            }
            (init.camera, Some(init.sensor), init.poses)
        }
        other => return Err(unsupported_sensor(other)),
    };
    let mut camera = camera_params(&camera, sensor, model)?;
    if let Some(distortion) = &seed.distortion {
        camera.distortion = distortion.clone();
    }
    Ok(CameraBootstrap { camera, poses })
}

/// Linear bootstrap of an untilted equidistant (fisheye) camera: seeded
/// intrinsics with the seeded distortion (zero without one), or the
/// bearing-based initializer ([`estimate_equidistant_intrinsics`]), whose
/// Kannala-Brandt `k1`, `k2` are embedded into the model's distortion by
/// name and replaced by a distortion seed; per-view poses from the bearings
/// of the pixels back-projected through the full camera.
///
/// # Errors
///
/// - [`Error::InvalidInput`] for a tilted sensor.
/// - [`Error::Numerical`] when the initializer or a pose recovery fails.
fn bootstrap_fisheye(
    cam_idx: usize,
    model: CameraModelSpec,
    dataset: &PlanarDataset,
    seed: &CameraSeed,
    opts: EquidistantInitOptions,
) -> Result<CameraBootstrap, Error> {
    if model.sensor != SensorKind::Identity {
        return Err(unsupported_sensor(model.sensor));
    }
    let fisheye_camera = |intrinsics, k1, k2| CameraParams {
        projection: ProjectionParams::Equidistant,
        distortion: embed_coefficients(&[("k1", k1), ("k2", k2)], model.distortion),
        sensor: SensorParams::Identity,
        intrinsics,
    };
    if let Some(intrinsics) = seed.intrinsics {
        let mut camera = fisheye_camera(intrinsics, 0.0, 0.0);
        if let Some(distortion) = &seed.distortion {
            camera.distortion = distortion.clone();
        }
        let poses = bearing_poses(cam_idx, dataset, &camera)?;
        return Ok(CameraBootstrap { camera, poses });
    }
    let init = estimate_equidistant_intrinsics(dataset, opts).map_err(|e| {
        Error::numerical(format!(
            "fisheye intrinsics initialization failed for camera {cam_idx}: {e}"
        ))
    })?;
    let mut camera = fisheye_camera(init.intrinsics, init.distortion.k1, init.distortion.k2);
    if let Some(distortion) = &seed.distortion {
        camera.distortion = distortion.clone();
    }
    Ok(CameraBootstrap {
        camera,
        poses: init.poses,
    })
}

/// Per-view target poses of a central camera from the bearings of its
/// back-projected pixels.
///
/// # Errors
///
/// [`Error::InvalidInput`] if the camera does not build;
/// [`Error::Numerical`] naming the view whose pose recovery fails.
pub(crate) fn bearing_poses(
    cam_idx: usize,
    dataset: &PlanarDataset,
    camera: &CameraParams,
) -> Result<Vec<Iso3>, Error> {
    let model = camera
        .build()
        .map_err(|e| Error::invalid_input(format!("camera {cam_idx}: {e}")))?;
    dataset
        .views
        .iter()
        .enumerate()
        .map(|(local_idx, view)| {
            let (board, pixels) = board_and_pixel_points(&view.obs);
            let bearings: Vec<_> = pixels
                .iter()
                .map(|p| model.backproject_pixel(p).direction)
                .collect();
            estimate_planar_pose_from_bearings(&board, &bearings).map_err(|e| {
                Error::numerical(format!(
                    "pose estimation failed for cam {cam_idx} view {local_idx}: {e}"
                ))
            })
        })
        .collect()
}

/// The parameters of a camera of `model` from its linear-init values:
/// Brown-Conrady coefficients embedded into the model's distortion, and the
/// tilt for a Scheimpflug sensor.
fn camera_params(
    camera: &PinholeCamera,
    tilt: Option<ScheimpflugParams>,
    model: CameraModelSpec,
) -> Result<CameraParams, Error> {
    Ok(CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: embed_distortion(&camera.dist, model.distortion),
        sensor: planar_sensor(model.sensor, tilt)?,
        intrinsics: camera.k,
    })
}

/// The seeded intrinsics (zero distortion), or Zhang's method; the caller
/// applies a distortion seed to the result.
fn seeded_or_fitted_camera(
    cam_idx: usize,
    seed: &CameraSeed,
    dataset: &PlanarDataset,
    init_opts: IterativeIntrinsicsOptions,
) -> Result<PinholeCamera, Error> {
    if let Some(k) = seed.intrinsics {
        return Ok(make_pinhole_camera(k, BrownConrady5::default()));
    }
    let bootstrap = estimate_intrinsics_iterative(dataset, init_opts).map_err(|e| {
        Error::numerical(format!(
            "intrinsics estimation failed for camera {cam_idx}: {e}"
        ))
    })?;
    Ok(make_pinhole_camera(bootstrap.k, bootstrap.dist))
}

fn estimate_camera_poses(
    cam_idx: usize,
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCy<Real>,
) -> Result<Vec<Iso3>, Error> {
    let k_matrix = intrinsics_k_matrix(intrinsics);
    let mut poses = Vec::with_capacity(dataset.num_views());
    for (local_idx, view) in dataset.views.iter().enumerate() {
        let pose = estimate_target_pose(&k_matrix, &view.obs).map_err(|e| {
            Error::numerical(format!(
                "pose estimation failed for cam {cam_idx} view {local_idx}: {e}"
            ))
        })?;
        poses.push(pose);
    }
    Ok(poses)
}

/// The planar solve options of a camera: its fix lists, and every pose
/// free. A camera calibrated from its own planar views has no pose gauge to
/// remove, and a held pose would keep the error of its linear-init value
/// (fitted to distorted points) in the solve.
pub(crate) fn camera_solve_opts(
    camera: &CameraConfig,
    solver: &SolverConfig,
) -> PlanarIntrinsicsSolveOptions {
    PlanarIntrinsicsSolveOptions {
        robust_loss: solver.robust_loss,
        fix: camera.fix.clone(),
        fix_poses: Vec::new(),
        bounds: None,
    }
}

/// Refine camera `cam_idx` (named in errors) on its own views: a plain
/// planar solve for an untilted camera, the staged multi-start for a tilted
/// one (it breaks the tilt / principal-point / distortion degeneracy a cold
/// `tilt = 0` solve falls into).
///
/// # Errors
///
/// [`Error::Numerical`] naming the camera when the solve fails.
pub(crate) fn optimize_camera(
    cam_idx: usize,
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    camera: &CameraConfig,
    solver: &SolverConfig,
    backend_opts: BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    let solve_opts = camera_solve_opts(camera, solver);
    match camera.model.sensor {
        SensorKind::Identity => {
            optimize_planar_intrinsics(dataset, initial, solve_opts, backend_opts).map_err(|e| {
                Error::numerical(format!("optimization failed for camera {cam_idx}: {e}"))
            })
        }
        SensorKind::Scheimpflug => optimize_scheimpflug_intrinsics_staged(
            dataset,
            initial,
            solve_opts,
            &ScheimpflugStagedInitOptions::default(),
            backend_opts,
        )
        .map_err(|e| {
            Error::numerical(format!(
                "Scheimpflug intrinsics optimization failed for camera {cam_idx}: {e}"
            ))
        }),
        other => Err(unsupported_sensor(other)),
    }
}

/// Embed Brown-Conrady linear-init coefficients into `kind`, zeroing any
/// extra degrees of freedom (the division model's `lambda` starts at zero).
pub(crate) fn embed_distortion(
    bc5: &BrownConrady5<Real>,
    kind: DistortionKind,
) -> DistortionParams {
    convert_distortion(&DistortionParams::BrownConrady5 { params: *bc5 }, kind)
}

/// `distortion` as a distortion of `kind`: itself when it is one, else each
/// coefficient of `kind` taken from the one of the same name (`k1`, `p1`,
/// …), and zero where `distortion` has none.
pub(crate) fn convert_distortion(
    distortion: &DistortionParams,
    kind: DistortionKind,
) -> DistortionParams {
    if distortion.kind() == kind {
        return distortion.clone();
    }
    let names = distortion.kind().param_names();
    let values = distortion.values();
    let named: Vec<(&str, Real)> = names.iter().copied().zip(values).collect();
    embed_coefficients(&named, kind)
}

/// The distortion of `kind` with each coefficient named in `values` set,
/// and every other one zero.
pub(crate) fn embed_coefficients(
    values: &[(&str, Real)],
    kind: DistortionKind,
) -> DistortionParams {
    let values: Vec<Real> = kind
        .param_names()
        .iter()
        .map(|name| {
            values
                .iter()
                .find(|(n, _)| n == name)
                .map_or(0.0, |(_, v)| *v)
        })
        .collect();
    DistortionParams::from_values(kind, &values)
        .expect("one value per parameter name, so the dimension matches")
}

/// The error for a projection kind the planar workflows have no
/// initializer for.
pub(crate) fn unsupported_projection(kind: ProjectionKind) -> Error {
    Error::invalid_input(format!(
        "planar calibration has no initializer for the {kind:?} projection"
    ))
}

/// The sensor of a planar camera of `kind`, tilted by `tilt` when it is a
/// Scheimpflug sensor (untilted without one).
///
/// # Errors
///
/// [`Error::InvalidInput`] for a sensor kind planar calibration does not
/// support.
pub(crate) fn planar_sensor(
    kind: SensorKind,
    tilt: Option<ScheimpflugParams>,
) -> Result<SensorParams, Error> {
    match kind {
        SensorKind::Identity => Ok(SensorParams::Identity),
        SensorKind::Scheimpflug => Ok(SensorParams::Scheimpflug {
            params: tilt.unwrap_or_default(),
        }),
        other => Err(unsupported_sensor(other)),
    }
}

/// The error for a sensor kind the planar workflows have no initializer or
/// solver route for.
pub(crate) fn unsupported_sensor(kind: SensorKind) -> Error {
    Error::invalid_input(format!(
        "planar calibration does not support the {kind:?} sensor"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCy, View, make_pinhole_camera, synthetic::planar,
    };
    use vision_calibration_linear::distortion_fit::DistortionFitOptions;

    fn make_test_dataset() -> PlanarDataset {
        let cam_gt = make_pinhole_camera(
            FxFyCxCy {
                fx: 900.0,
                fy: 880.0,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5::default(),
        );
        let board_points = planar::grid_points(6, 5, 0.04);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).expect("views");
        PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
    }

    #[test]
    fn camera_solve_opts_frees_every_pose() {
        for sensor in [SensorKind::Identity, SensorKind::Scheimpflug] {
            let camera = CameraConfig::new(CameraModelSpec::pinhole(
                DistortionKind::BrownConrady5,
                sensor,
            ));
            let opts = camera_solve_opts(&camera, &SolverConfig::default());
            assert!(opts.fix_poses.is_empty(), "{sensor:?}");
        }
    }

    #[test]
    fn intrinsics_k_matrix_layout() {
        let k = intrinsics_k_matrix(&FxFyCxCy {
            fx: 100.0,
            fy: 200.0,
            cx: 50.0,
            cy: 60.0,
        });
        assert_eq!(k[(0, 0)], 100.0);
        assert_eq!(k[(0, 1)], 0.0);
        assert_eq!(k[(0, 2)], 50.0);
        assert_eq!(k[(1, 1)], 200.0);
        assert_eq!(k[(1, 2)], 60.0);
        assert_eq!(k[(2, 2)], 1.0);
        assert_eq!(k[(1, 0)], 0.0);
        assert_eq!(k[(2, 0)], 0.0);
        assert_eq!(k[(2, 1)], 0.0);
    }

    #[test]
    fn bootstrap_estimates_homographies_and_poses_per_view() {
        let dataset = make_test_dataset();
        let result = bootstrap_planar_intrinsics(
            &dataset,
            IterativeIntrinsicsOptions {
                iterations: 2,
                distortion_opts: DistortionFitOptions {
                    fix_k3: true,
                    fix_tangential: false,
                    iters: 8,
                },
            },
        )
        .expect("bootstrap");

        assert_eq!(result.homographies.len(), dataset.num_views());
        assert_eq!(result.poses.len(), dataset.num_views());
        assert!(result.camera.k.fx.is_finite());
        assert!(result.camera.k.fy.is_finite());
    }
}
