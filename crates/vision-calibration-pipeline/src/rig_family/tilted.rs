//! Recovery of the tilted cameras of a rig initialized from scratch.
//!
//! A Scheimpflug camera auto-initialized without seeds can settle in a wrong
//! tilt / principal-point / distortion basin. The rig intrinsics stage of
//! the problem types that opt in ([`super::RigStages::RECOVERS_TILTED_CAMERAS`])
//! solves such a camera through a wide multi-start over linear tilt seeds
//! and, for a Brown-Conrady camera that still fits badly, re-solves it from
//! a nominal seed built from the well-fitted cameras of the rig (and,
//! failing that, against their rig).

use crate::Error;
use crate::common::config::{CameraConfig, IntrinsicsInitConfig, SolverConfig};
use crate::planar_family::{camera_solve_opts, embed_distortion};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraModel, CameraModelSpec, CameraParams, CorrespondenceView,
    DistortionKind, DistortionModel, FxFyCxCy, IntrinsicsModel, Iso3, NoMeta, PlanarDataset, Pt2,
    RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorKind, SensorModel, SensorParams,
    Uncertainty,
};
use vision_calibration_linear::extrinsics::{
    average_isometries, estimate_extrinsics_from_cam_target_poses,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, CameraBounds, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RigExtrinsicsParams, RigExtrinsicsSolveOptions,
    ScheimpflugStagedInitOptions, SolveReport, optimize_planar_intrinsics, optimize_rig_extrinsics,
    optimize_scheimpflug_intrinsics_staged,
};

use super::{rig_distortion, rig_tilt, tilted_planar_params};

/// The model of the cameras the nominal recovery serves: Brown-Conrady
/// distortion and a Scheimpflug sensor.
const TILTED_BROWN_CONRADY: CameraModelSpec =
    CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);

#[derive(Debug, Clone)]
struct ScheimpflugCameraWork {
    cam_idx: usize,
    dataset: PlanarDataset,
    valid_indices: Vec<usize>,
    initial: PlanarIntrinsicsParams,
    solve_opts: PlanarIntrinsicsSolveOptions,
}

#[derive(Debug, Clone, Copy)]
struct NominalScheimpflugSeed {
    intrinsics: FxFyCxCy<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
}

#[derive(Debug, Clone)]
struct ScoredScheimpflugSeed {
    score: f64,
    params: PlanarIntrinsicsParams,
}

#[derive(Debug, Clone)]
struct GoodRigContext {
    cam_to_rig: Vec<Iso3>,
    rig_from_target: Vec<Iso3>,
}

/// The tilted cameras of one rig intrinsics stage: each solved as it comes
/// ([`Self::solve`]), then the badly fitted ones recovered together
/// ([`Self::recover`]).
#[derive(Debug, Default)]
pub(super) struct TiltedRecovery {
    work_items: Vec<ScheimpflugCameraWork>,
}

impl TiltedRecovery {
    /// Solve tilted camera `cam_idx` after a full auto-init, by the wide
    /// multi-start over linear tilt seeds, and keep a
    /// [`TILTED_BROWN_CONRADY`] camera for [`Self::recover`].
    ///
    /// `valid_indices` maps `dataset`'s views to the rig's.
    #[allow(clippy::too_many_arguments)] // one camera's whole solve context
    pub(super) fn solve(
        &mut self,
        cam_idx: usize,
        dataset: &PlanarDataset,
        valid_indices: &[usize],
        initial: &PlanarIntrinsicsParams,
        camera: &CameraConfig,
        solver: &SolverConfig,
        intrinsics: &IntrinsicsInitConfig,
        backend_opts: BackendSolveOptions,
    ) -> Result<PlanarIntrinsicsEstimate, Error> {
        let solve_opts = camera_solve_opts(camera, solver);
        if camera.model == TILTED_BROWN_CONRADY {
            self.work_items.push(ScheimpflugCameraWork {
                cam_idx,
                dataset: dataset.clone(),
                valid_indices: valid_indices.to_vec(),
                initial: initial.clone(),
                solve_opts: solve_opts.clone(),
            });
        }
        optimize_scheimpflug_intrinsics_auto_multistart(
            cam_idx,
            dataset,
            initial,
            solve_opts,
            backend_opts,
            intrinsics.iterations,
            intrinsics.distortion_fit_opts(camera),
        )
    }

    /// Re-solve each kept camera that fits badly from a nominal seed of the
    /// well-fitted ones, replacing its parameters, error and target poses
    /// (`[view][cam]`) when the result fits better.
    pub(super) fn recover(
        self,
        cameras: &mut [CameraParams],
        per_cam_reproj_errors: &mut [f64],
        per_cam_target_poses: &mut [Vec<Option<Iso3>>],
        backend_opts: BackendSolveOptions,
        reference_camera_idx: usize,
    ) -> Result<(), Error> {
        if self.work_items.is_empty() {
            return Ok(());
        }
        recover_bad_scheimpflug_cameras_from_nominal(
            &self.work_items,
            cameras,
            per_cam_reproj_errors,
            per_cam_target_poses,
            backend_opts,
            reference_camera_idx,
        )
    }
}

fn optimize_scheimpflug_intrinsics_auto_multistart(
    cam_idx: usize,
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    mut solve_opts: PlanarIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
    init_iterations: usize,
    distortion_opts: DistortionFitOptions,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    solve_opts.fix_poses.clear();
    let distortion_kind = initial
        .camera
        .spec()
        .map_or(DistortionKind::BrownConrady5, |s| s.distortion);
    let mut best: Option<PlanarIntrinsicsEstimate> = None;
    let mut last_err: Option<Error> = None;

    let mut seeds = Vec::with_capacity(12);
    seeds.push((initial.clone(), false));
    for tilt_x in [-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16] {
        let init = match estimate_scheimpflug_intrinsics_iterative(
            dataset,
            ScheimpflugIntrinsicsInitOptions {
                iterations: init_iterations,
                distortion_opts,
                tilt_x_seeds: vec![tilt_x],
                tilt_y_seeds: vec![0.0],
                refine_rounds: 1,
                ..Default::default()
            },
        ) {
            Ok(init) => init,
            Err(e) => {
                last_err = Some(Error::numerical(format!(
                    "Scheimpflug linear multistart failed for camera {cam_idx}, tilt_x={tilt_x}: {e}"
                )));
                continue;
            }
        };
        // Each linear seed starts from zero distortion in the camera's model.
        let params = PlanarIntrinsicsParams::new(
            CameraParams {
                distortion: embed_distortion(
                    &BrownConrady5 {
                        iters: 8,
                        ..BrownConrady5::default()
                    },
                    distortion_kind,
                ),
                sensor: SensorParams::Scheimpflug {
                    params: init.sensor,
                },
                intrinsics: init.camera.k,
                ..initial.camera.clone()
            },
            init.poses,
        )?;
        if !seeds.iter().any(|(s, _)| {
            (rig_tilt(&s.camera).tilt_x - rig_tilt(&params.camera).tilt_x).abs() < 1e-4
                && (rig_tilt(&s.camera).tilt_y - rig_tilt(&params.camera).tilt_y).abs() < 1e-4
                && (s.camera.intrinsics.fx - params.camera.intrinsics.fx).abs() < 1.0
                && (s.camera.intrinsics.fy - params.camera.intrinsics.fy).abs() < 1.0
        }) {
            seeds.push((params, false));
        }
    }

    for (params, _) in seeds {
        let staged = ScheimpflugStagedInitOptions {
            tilt_x_seeds: vec![rig_tilt(&params.camera).tilt_x],
            tilt_y_seeds: vec![rig_tilt(&params.camera).tilt_y],
            sweep_max_iters: backend_opts.max_iters.clamp(1, 20),
            ..Default::default()
        };
        match optimize_scheimpflug_intrinsics_staged(
            dataset,
            &params,
            solve_opts.clone(),
            &staged,
            backend_opts.clone(),
        ) {
            Ok(candidate) if candidate.mean_reproj_error.is_finite() => {
                let better = best
                    .as_ref()
                    .is_none_or(|current| candidate.mean_reproj_error < current.mean_reproj_error);
                if better {
                    best = Some(candidate);
                }
                if best
                    .as_ref()
                    .is_some_and(|current| current.mean_reproj_error < 0.45)
                {
                    break;
                }
            }
            Ok(_) => {}
            Err(e) => {
                last_err = Some(Error::numerical(format!(
                    "Scheimpflug intrinsics optimization failed for camera {cam_idx}: {e}"
                )));
            }
        }
    }

    // A cancelled multistart has no winner, whatever it tried before.
    backend_opts.control.check()?;
    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            Error::numerical(format!(
                "all Scheimpflug intrinsics multistart candidates failed for camera {cam_idx}"
            ))
        })
    })
}

const SCHEIMPFLUG_NOMINAL_GOOD_REPROJ_PX: f64 = 0.75;
const SCHEIMPFLUG_NOMINAL_RECOVER_REPROJ_PX: f64 = 0.50;
const SCHEIMPFLUG_NOMINAL_TOP_LOCAL_CANDIDATES: usize = 10;
const SCHEIMPFLUG_NOMINAL_RIG_EXPAND_SOURCE: usize = 80;
const SCHEIMPFLUG_NOMINAL_TOP_RIG_CANDIDATES: usize = 6;

/// Re-solve each badly fitted tilted camera from a nominal seed built from
/// the well-fitted ones (and, failing that, against their rig). The
/// recovery is a Brown-Conrady heuristic: it serves cameras of
/// [`TILTED_BROWN_CONRADY`] only.
fn recover_bad_scheimpflug_cameras_from_nominal(
    work_items: &[ScheimpflugCameraWork],
    cameras: &mut [CameraParams],
    per_cam_reproj_errors: &mut [f64],
    per_cam_target_poses: &mut [Vec<Option<Iso3>>],
    backend_opts: BackendSolveOptions,
    reference_camera_idx: usize,
) -> Result<(), Error> {
    let good_indices = good_nominal_scheimpflug_indices(cameras, per_cam_reproj_errors);
    if good_indices.len() < 2 {
        return Ok(());
    };
    let Some(nominal) = build_nominal_scheimpflug_seed(cameras, &good_indices) else {
        return Ok(());
    };
    let rig_context = build_good_scheimpflug_rig_context(
        per_cam_target_poses,
        &good_indices,
        reference_camera_idx,
    )
    .ok();

    for item in work_items {
        let cam_idx = item.cam_idx;
        if per_cam_reproj_errors
            .get(cam_idx)
            .is_some_and(|e| e.is_finite() && *e <= SCHEIMPFLUG_NOMINAL_RECOVER_REPROJ_PX)
        {
            continue;
        }

        let Some(candidate) = recover_one_scheimpflug_camera_from_nominal(
            item,
            nominal,
            rig_context.as_ref(),
            cameras,
            per_cam_target_poses.len(),
            &backend_opts,
        )?
        else {
            continue;
        };

        let current = per_cam_reproj_errors
            .get(cam_idx)
            .copied()
            .unwrap_or(f64::INFINITY);
        if candidate.mean_reproj_error.is_finite()
            && (!current.is_finite() || candidate.mean_reproj_error < current)
        {
            cameras[cam_idx] = candidate.params.camera.clone();
            per_cam_reproj_errors[cam_idx] = candidate.mean_reproj_error;
            for (local_idx, &global_idx) in item.valid_indices.iter().enumerate() {
                per_cam_target_poses[global_idx][cam_idx] =
                    Some(candidate.params.camera_se3_target[local_idx]);
            }
        }
    }

    Ok(())
}

fn build_nominal_scheimpflug_seed(
    cameras: &[CameraParams],
    good_indices: &[usize],
) -> Option<NominalScheimpflugSeed> {
    let k = |idx: usize| cameras[idx].intrinsics;
    let dist = |idx: usize| rig_distortion(&cameras[idx]);
    let tilt = |idx: usize| rig_tilt(&cameras[idx]);
    let intrinsics = FxFyCxCy {
        fx: median_of(good_indices.iter().map(|&idx| k(idx).fx))?,
        fy: median_of(good_indices.iter().map(|&idx| k(idx).fy))?,
        cx: median_of(good_indices.iter().map(|&idx| k(idx).cx))?,
        cy: median_of(good_indices.iter().map(|&idx| k(idx).cy))?,
    };
    let distortion = BrownConrady5 {
        k1: median_of(good_indices.iter().map(|&idx| dist(idx).k1))?,
        k2: median_of(good_indices.iter().map(|&idx| dist(idx).k2))?,
        k3: median_of(good_indices.iter().map(|&idx| dist(idx).k3))?,
        p1: median_of(good_indices.iter().map(|&idx| dist(idx).p1))?,
        p2: median_of(good_indices.iter().map(|&idx| dist(idx).p2))?,
        iters: dist(good_indices[0]).iters.max(8),
    };
    let sensor = ScheimpflugParams {
        tilt_x: median_of(good_indices.iter().map(|&idx| tilt(idx).tilt_x))?,
        tilt_y: median_of(good_indices.iter().map(|&idx| tilt(idx).tilt_y))?,
    };

    Some(NominalScheimpflugSeed {
        intrinsics,
        distortion,
        sensor,
    })
}

fn good_nominal_scheimpflug_indices(cameras: &[CameraParams], errors: &[f64]) -> Vec<usize> {
    cameras
        .iter()
        .zip(errors.iter())
        .enumerate()
        .filter_map(|(idx, (camera, error))| {
            (camera.spec() == Some(TILTED_BROWN_CONRADY)
                && is_good_nominal_scheimpflug_source(camera, *error))
            .then_some(idx)
        })
        .collect()
}

fn build_good_scheimpflug_rig_context(
    per_cam_target_poses: &[Vec<Option<Iso3>>],
    good_indices: &[usize],
    reference_camera_idx: usize,
) -> Result<GoodRigContext, Error> {
    if per_cam_target_poses.is_empty() || good_indices.len() < 2 {
        return Err(Error::invalid_input(
            "need at least two good cameras with target poses",
        ));
    }
    let ref_pos = good_indices
        .iter()
        .position(|&idx| idx == reference_camera_idx)
        .unwrap_or(0);
    let reduced: Vec<Vec<Option<Iso3>>> = per_cam_target_poses
        .iter()
        .map(|view| {
            good_indices
                .iter()
                .map(|&cam_idx| view.get(cam_idx).cloned().unwrap_or(None))
                .collect()
        })
        .collect();

    let reduced_extrinsics =
        estimate_extrinsics_from_cam_target_poses(&reduced, ref_pos).map_err(|e| {
            Error::numerical(format!(
                "good-camera rig context initialization failed: {e}"
            ))
        })?;
    let num_cameras = per_cam_target_poses[0].len();
    let mut cam_to_rig = vec![Iso3::identity(); num_cameras];
    for (reduced_idx, &cam_idx) in good_indices.iter().enumerate() {
        cam_to_rig[cam_idx] = reduced_extrinsics.cam_to_rig[reduced_idx];
    }

    Ok(GoodRigContext {
        cam_to_rig,
        rig_from_target: reduced_extrinsics.rig_from_target,
    })
}

fn is_good_nominal_scheimpflug_source(camera: &CameraParams, error: f64) -> bool {
    let k = camera.intrinsics;
    let dist = rig_distortion(camera);
    let sensor = rig_tilt(camera);
    error.is_finite()
        && error <= SCHEIMPFLUG_NOMINAL_GOOD_REPROJ_PX
        && k.fx.is_finite()
        && k.fy.is_finite()
        && k.cx.is_finite()
        && k.cy.is_finite()
        && (100.0..10_000.0).contains(&k.fx)
        && (100.0..10_000.0).contains(&k.fy)
        && dist.k1.is_finite()
        && dist.k2.is_finite()
        && dist.k3.is_finite()
        && dist.p1.is_finite()
        && dist.p2.is_finite()
        && dist.k1.abs() < 2.0
        && dist.k2.abs() < 2.0
        && sensor.tilt_x.is_finite()
        && sensor.tilt_y.is_finite()
        && sensor.tilt_x.abs() < 0.30
        && sensor.tilt_y.abs() < 0.30
}

fn median_of(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    })
}

fn recover_one_scheimpflug_camera_from_nominal(
    item: &ScheimpflugCameraWork,
    nominal: NominalScheimpflugSeed,
    rig_context: Option<&GoodRigContext>,
    cameras: &[CameraParams],
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<Option<PlanarIntrinsicsEstimate>, Error> {
    let solve_opts = &item.solve_opts;
    let mut scored = score_nominal_scheimpflug_seeds(&item.dataset, nominal);
    // The initial camera is a `TILTED_BROWN_CONRADY` one, so its parts
    // rebuild it exactly.
    let initial = &item.initial.camera;
    score_scheimpflug_seed_parts(
        &item.dataset,
        initial.intrinsics,
        rig_distortion(initial),
        rig_tilt(initial),
        &mut scored,
    );
    scored.sort_by(|a, b| a.score.total_cmp(&b.score));

    let mut best = scored
        .first()
        .cloned()
        .map(scheimpflug_estimate_from_scored_seed);

    for scored_seed in scored
        .iter()
        .take(SCHEIMPFLUG_NOMINAL_TOP_LOCAL_CANDIDATES)
        .cloned()
    {
        let candidate = optimize_planar_intrinsics(
            &item.dataset,
            &scored_seed.params,
            nominal_pose_only_solve_opts(solve_opts.clone()),
            BackendSolveOptions {
                max_iters: backend_opts.max_iters.max(60),
                ..backend_opts.clone()
            },
        )
        .unwrap_or_else(|_| scheimpflug_estimate_from_scored_seed(scored_seed));

        let candidate = optimize_planar_intrinsics(
            &item.dataset,
            &candidate.params,
            nominal_tight_refine_solve_opts(solve_opts.clone(), nominal),
            BackendSolveOptions {
                max_iters: (backend_opts.max_iters * 2).max(100),
                ..backend_opts.clone()
            },
        )
        .unwrap_or(candidate);

        if candidate.mean_reproj_error.is_finite()
            && best
                .as_ref()
                .is_none_or(|current| candidate.mean_reproj_error < current.mean_reproj_error)
        {
            best = Some(candidate);
        }
    }

    backend_opts.control.check()?;
    if best.as_ref().is_some_and(|candidate| {
        candidate.mean_reproj_error <= SCHEIMPFLUG_NOMINAL_RECOVER_REPROJ_PX
    }) {
        return Ok(best);
    }

    if let Some(context) = rig_context {
        let mut rig_scored = scored
            .iter()
            .filter_map(|seed| {
                score_bad_scheimpflug_seed_against_good_rig(item, &seed.params, context)
                    .filter(|score| score.is_finite())
                    .map(|score| (score, seed.clone()))
            })
            .collect::<Vec<_>>();
        rig_scored.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut rig_candidates = nominal_rig_anchor_seeds(&item.dataset, nominal);
        rig_candidates.extend(
            rig_scored
                .into_iter()
                .take(SCHEIMPFLUG_NOMINAL_TOP_RIG_CANDIDATES)
                .map(|(_, seed)| seed),
        );

        for seed in rig_candidates {
            if let Some(candidate) = optimize_bad_scheimpflug_camera_against_good_rig(
                item,
                &seed.params,
                context,
                cameras,
                num_views,
                backend_opts,
            )? && candidate.mean_reproj_error.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|current| candidate.mean_reproj_error < current.mean_reproj_error)
            {
                best = Some(candidate);
            }
        }
    }

    // The candidates swallow failed solves; a cancelled one ends the search.
    backend_opts.control.check()?;
    Ok(best)
}

fn score_bad_scheimpflug_seed_against_good_rig(
    item: &ScheimpflugCameraWork,
    params: &PlanarIntrinsicsParams,
    context: &GoodRigContext,
) -> Option<f64> {
    let cam_to_rig =
        estimate_candidate_cam_to_rig(params, &item.valid_indices, &context.rig_from_target)?;
    let cam_from_rig = cam_to_rig.inverse();
    let Ok(camera) = params.camera.build() else {
        return None;
    };
    let views = item
        .valid_indices
        .iter()
        .enumerate()
        .filter(|&(_, &global_idx)| global_idx < context.rig_from_target.len())
        .map(|(local_idx, &global_idx)| {
            (
                &item.dataset.views[local_idx].obs,
                cam_from_rig * context.rig_from_target[global_idx],
            )
        });
    mean_reprojection(&camera, views)
}

fn optimize_bad_scheimpflug_camera_against_good_rig(
    item: &ScheimpflugCameraWork,
    candidate_params: &PlanarIntrinsicsParams,
    context: &GoodRigContext,
    cameras: &[CameraParams],
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<Option<PlanarIntrinsicsEstimate>, Error> {
    let Some(initial_cam_to_rig) = estimate_candidate_cam_to_rig(
        candidate_params,
        &item.valid_indices,
        &context.rig_from_target,
    ) else {
        return Ok(None);
    };
    let dataset = bad_camera_only_scheimpflug_rig_dataset(item, cameras.len(), num_views)?;

    let mut rig_cameras = cameras.to_vec();
    rig_cameras[item.cam_idx] = candidate_params.camera.clone();
    let mut cam_to_rig = context.cam_to_rig.clone();
    if cam_to_rig.len() != cameras.len() {
        return Ok(None);
    }
    cam_to_rig[item.cam_idx] = initial_cam_to_rig;

    let mut fix_extrinsics = vec![true; cameras.len()];
    fix_extrinsics[item.cam_idx] = false;
    let fix = rig_cameras
        .iter()
        .map(|c| c.spec().map(CameraFix::all))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::invalid_input("a rig camera has a general homography sensor"))?;
    let initial = RigExtrinsicsParams {
        cameras: rig_cameras,
        cam_to_rig,
        rig_from_target: context.rig_from_target.clone(),
    };
    let solve_opts = RigExtrinsicsSolveOptions {
        robust_loss: item.solve_opts.robust_loss,
        fix,
        fix_extrinsics,
        fix_rig_poses: (0..num_views).collect(),
    };
    let Ok(result) = optimize_rig_extrinsics(
        dataset,
        initial,
        solve_opts,
        BackendSolveOptions {
            max_iters: backend_opts.max_iters.max(80),
            ..backend_opts.clone()
        },
    ) else {
        return Ok(None);
    };
    let mean = result
        .per_cam_reproj_errors
        .get(item.cam_idx)
        .copied()
        .unwrap_or(f64::INFINITY);
    if !mean.is_finite() {
        return Ok(None);
    }

    let camera = &result.params.cameras[item.cam_idx];
    let cam_from_rig = result.params.cam_to_rig[item.cam_idx].inverse();
    let mut poses = Vec::with_capacity(item.valid_indices.len());
    for &global_idx in &item.valid_indices {
        if global_idx >= result.params.rig_from_target.len() {
            return Ok(None);
        }
        poses.push(cam_from_rig * result.params.rig_from_target[global_idx]);
    }
    let params = PlanarIntrinsicsParams::new(camera.clone(), poses)?;
    Ok(Some(PlanarIntrinsicsEstimate {
        params,
        report: result.report,
        mean_reproj_error: mean,
        // Derived from the rig solve, not solved on its own.
        uncertainty: Uncertainty::default(),
    }))
}

fn bad_camera_only_scheimpflug_rig_dataset(
    item: &ScheimpflugCameraWork,
    num_cameras: usize,
    num_views: usize,
) -> Result<RigDataset<NoMeta>, Error> {
    let mut obs_by_view: Vec<Option<CorrespondenceView>> = vec![None; num_views];
    for (local_idx, &global_idx) in item.valid_indices.iter().enumerate() {
        if global_idx < num_views {
            obs_by_view[global_idx] = Some(item.dataset.views[local_idx].obs.clone());
        }
    }

    let views = obs_by_view
        .into_iter()
        .map(|obs| {
            let mut cameras = vec![None; num_cameras];
            cameras[item.cam_idx] = obs;
            RigView {
                obs: RigViewObs { cameras },
                meta: NoMeta,
            }
        })
        .collect();
    RigDataset::new(views, num_cameras).map_err(Error::from)
}

fn estimate_candidate_cam_to_rig(
    params: &PlanarIntrinsicsParams,
    valid_indices: &[usize],
    rig_from_target: &[Iso3],
) -> Option<Iso3> {
    let mut candidates = Vec::with_capacity(valid_indices.len());
    for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
        if global_idx >= rig_from_target.len() || local_idx >= params.camera_se3_target.len() {
            continue;
        }
        candidates
            .push(rig_from_target[global_idx] * params.camera_se3_target[local_idx].inverse());
    }
    average_isometries(&candidates).ok()
}

fn score_nominal_scheimpflug_seeds(
    dataset: &PlanarDataset,
    nominal: NominalScheimpflugSeed,
) -> Vec<ScoredScheimpflugSeed> {
    let mut scored = Vec::new();
    let pp_offsets = [-24.0, -12.0, 0.0, 12.0, 24.0];
    let tilt_x_offsets = [-0.020, -0.010, 0.0, 0.010, 0.020];
    let tilt_y_offsets = [-0.010, 0.0, 0.010];

    for dcx in pp_offsets {
        for dcy in pp_offsets {
            for dtx in tilt_x_offsets {
                for dty in tilt_y_offsets {
                    let mut k = nominal.intrinsics;
                    k.cx += dcx;
                    k.cy += dcy;
                    let sensor = ScheimpflugParams {
                        tilt_x: nominal.sensor.tilt_x + dtx,
                        tilt_y: nominal.sensor.tilt_y + dty,
                    };
                    score_scheimpflug_seed_parts(
                        dataset,
                        k,
                        nominal.distortion,
                        sensor,
                        &mut scored,
                    );
                }
            }
        }
    }

    scored.sort_by(|a, b| a.score.total_cmp(&b.score));
    let mut expanded = scored
        .iter()
        .take(SCHEIMPFLUG_NOMINAL_RIG_EXPAND_SOURCE)
        .cloned()
        .collect::<Vec<_>>();
    for base in scored.iter().take(SCHEIMPFLUG_NOMINAL_RIG_EXPAND_SOURCE) {
        for dk1 in [-0.04, 0.0, 0.04] {
            for dk2 in [-0.08, 0.0, 0.08] {
                if dk1 == 0.0 && dk2 == 0.0 {
                    continue;
                }
                let mut dist = rig_distortion(&base.params.camera);
                dist.k1 += dk1;
                dist.k2 += dk2;
                score_scheimpflug_seed_parts(
                    dataset,
                    base.params.camera.intrinsics,
                    dist,
                    rig_tilt(&base.params.camera),
                    &mut expanded,
                );
            }
        }
    }
    expanded
}

fn nominal_rig_anchor_seeds(
    dataset: &PlanarDataset,
    nominal: NominalScheimpflugSeed,
) -> Vec<ScoredScheimpflugSeed> {
    let mut scored = Vec::new();
    for (dcx, dcy, dk1, dk2, dty) in [
        (12.0, 24.0, -0.04, 0.08, 0.0),
        (12.0, 24.0, -0.04, 0.12, 0.0),
        (18.0, 24.0, -0.04, 0.08, 0.0),
        (12.0, 18.0, -0.04, 0.08, 0.0),
        (24.0, 24.0, -0.04, 0.08, 0.0),
        (0.0, 24.0, -0.04, 0.08, 0.0),
        (12.0, 30.0, -0.04, 0.08, 0.0),
        (12.0, 24.0, -0.06, 0.08, 0.0),
        (12.0, 24.0, -0.02, 0.08, 0.0),
        (12.0, 24.0, -0.04, 0.08, -0.01),
    ] {
        let mut k = nominal.intrinsics;
        k.cx += dcx;
        k.cy += dcy;
        let mut dist = nominal.distortion;
        dist.k1 += dk1;
        dist.k2 += dk2;
        let sensor = ScheimpflugParams {
            tilt_x: nominal.sensor.tilt_x,
            tilt_y: nominal.sensor.tilt_y + dty,
        };
        score_scheimpflug_seed_parts(dataset, k, dist, sensor, &mut scored);
    }
    scored
}

fn score_scheimpflug_seed_parts(
    dataset: &PlanarDataset,
    intrinsics: FxFyCxCy<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    scored: &mut Vec<ScoredScheimpflugSeed>,
) {
    if !intrinsics.fx.is_finite()
        || !intrinsics.fy.is_finite()
        || !intrinsics.cx.is_finite()
        || !intrinsics.cy.is_finite()
        || !distortion.k1.is_finite()
        || !distortion.k2.is_finite()
        || !sensor.tilt_x.is_finite()
        || !sensor.tilt_y.is_finite()
    {
        return;
    }
    let Ok(poses) = recover_scheimpflug_poses_for_seed(dataset, &intrinsics, &distortion, sensor)
    else {
        return;
    };
    let Ok(params) = tilted_planar_params(intrinsics, distortion, sensor, poses) else {
        return;
    };
    let score = mean_scheimpflug_reproj(dataset, &params);
    if score.is_finite() {
        scored.push(ScoredScheimpflugSeed { score, params });
    }
}

fn nominal_pose_only_solve_opts(
    mut opts: PlanarIntrinsicsSolveOptions,
) -> PlanarIntrinsicsSolveOptions {
    opts.fix = CameraFix::all(TILTED_BROWN_CONRADY);
    opts.fix_poses.clear();
    opts.bounds = None;
    opts
}

fn nominal_tight_refine_solve_opts(
    mut opts: PlanarIntrinsicsSolveOptions,
    nominal: NominalScheimpflugSeed,
) -> PlanarIntrinsicsSolveOptions {
    let k = nominal.intrinsics;
    opts.fix.intrinsics = Some(Vec::new());
    opts.fix_poses.clear();
    opts.bounds = Some(CameraBounds {
        fx: Some((0.94 * k.fx, 1.06 * k.fx)),
        fy: Some((0.94 * k.fy, 1.06 * k.fy)),
        cx: Some((k.cx - 60.0, k.cx + 60.0)),
        cy: Some((k.cy - 60.0, k.cy + 60.0)),
        tilt_x: Some((nominal.sensor.tilt_x - 0.06, nominal.sensor.tilt_x + 0.06)),
        tilt_y: Some((nominal.sensor.tilt_y - 0.04, nominal.sensor.tilt_y + 0.04)),
    });
    opts
}

fn scheimpflug_estimate_from_scored_seed(seed: ScoredScheimpflugSeed) -> PlanarIntrinsicsEstimate {
    PlanarIntrinsicsEstimate {
        params: seed.params,
        report: SolveReport {
            final_cost: seed.score * seed.score,
            num_iters: 0,
        },
        mean_reproj_error: seed.score,
        uncertainty: Uncertainty::default(),
    }
}

fn mean_scheimpflug_reproj(dataset: &PlanarDataset, params: &PlanarIntrinsicsParams) -> f64 {
    let Ok(camera) = params.camera.build() else {
        return f64::INFINITY;
    };
    let views = dataset
        .views
        .iter()
        .zip(params.camera_se3_target.iter())
        .map(|(view, pose)| (&view.obs, *pose));
    mean_reprojection(&camera, views).unwrap_or(f64::INFINITY)
}

/// The mean of the finite reprojection errors (px) of `camera` over the
/// points of each view at its pose (`camera_se3_target`); `None` without
/// one.
fn mean_reprojection<'a>(
    camera: &CameraModel,
    views: impl Iterator<Item = (&'a CorrespondenceView, Iso3)>,
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (obs, pose) in views {
        for (p3d, p2d) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2d).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn recover_scheimpflug_poses_for_seed(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCy<f64>,
    distortion: &BrownConrady5<f64>,
    sensor: ScheimpflugParams,
) -> Result<Vec<Iso3>, Error> {
    let sensor_model = sensor.compile();
    let k_matrix = intrinsics.k_matrix();
    let mut poses = Vec::with_capacity(dataset.num_views());
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let board = view.obs.planar_points();
        let ideal_pixels: Vec<Pt2> = view
            .obs
            .points_2d
            .iter()
            .map(|p| {
                let sensor_pt = intrinsics.pixel_to_sensor(p);
                let distorted = sensor_model.sensor_to_normalized(&sensor_pt);
                let undistorted = distortion.undistort(&distorted);
                intrinsics.sensor_to_pixel(&undistorted)
            })
            .collect();
        let h = dlt_homography(&board, &ideal_pixels).map_err(|e| {
            Error::numerical(format!(
                "Scheimpflug cross-camera pose homography failed for view {view_idx}: {e}"
            ))
        })?;
        let pose = estimate_planar_pose_from_h(&k_matrix, &h).map_err(|e| {
            Error::numerical(format!(
                "Scheimpflug cross-camera pose recovery failed for view {view_idx}: {e}"
            ))
        })?;
        poses.push(pose);
    }
    Ok(poses)
}
