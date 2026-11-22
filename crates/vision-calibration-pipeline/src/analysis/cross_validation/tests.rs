//! Synthetic ground truth for the held-out model comparison.
//!
//! The two model-selection tests fit four candidates on two backends and
//! are `#[ignore]`d; run them in release mode:
//!
//! ```text
//! cargo test --release -p vision-calibration-pipeline --lib analysis::cross_validation -- --ignored
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use vision_calibration_core::synthetic::{noise::UniformPixelNoise, planar, poses};
use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, CancelToken, CorrespondenceView, DistortionKind, FxFyCxCy,
    Iso3, PlanarDataset, PointCovariance, Pt2, RunControl, SensorKind, View, make_pinhole_camera,
};
use vision_calibration_optim::SolverBackend;

use super::*;

/// Pixel noise σ per axis. The synthetic noise is uniform in `[−a, a]`,
/// whose standard deviation is `a/√3`.
const SIGMA_PX: f64 = 0.3;

/// `views` views of a `nx × ny` board (25 mm pitch) seen by a 1280×960
/// camera with `distortion`, the board centred at a spread of image
/// positions and tilted by up to 0.45 rad, with uniform pixel noise of
/// standard deviation [`SIGMA_PX`].
fn scene(distortion: BrownConrady5<f64>, nx: usize, ny: usize, views: usize) -> PlanarDataset {
    let truth = make_pinhole_camera(
        FxFyCxCy {
            fx: 800.0,
            fy: 790.0,
            cx: 640.0,
            cy: 480.0,
        },
        distortion,
    );
    let pitch = 0.025;
    let board = planar::grid_points(nx, ny, pitch);
    let centre = (0.5 * pitch * (nx - 1) as f64, 0.5 * pitch * (ny - 1) as f64);
    let specs = poses::seeded_board_pose_specs(views, 0.45, 11);
    let offsets = [-0.12, 0.0, 0.12];
    let board_poses: Vec<Iso3> = poses::centered_board_poses(&specs, centre, 0.5)
        .into_iter()
        .enumerate()
        .map(|(i, mut pose)| {
            pose.translation.vector.x += offsets[i % 3];
            pose.translation.vector.y += offsets[(i / 3) % 3] * 0.8;
            pose
        })
        .collect();
    let noise = UniformPixelNoise {
        seed: 5,
        max_abs_px: SIGMA_PX * 3.0_f64.sqrt(),
    };
    let observed = planar::project_views_noisy(&truth, &board, &board_poses, &noise).unwrap();
    PlanarDataset::new(observed.into_iter().map(View::without_meta).collect()).unwrap()
}

fn brown_conrady(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> BrownConrady5<f64> {
    BrownConrady5 {
        k1,
        k2,
        k3,
        p1,
        p2,
        iters: 8,
    }
}

/// A camera of `distortion` with exactly `fixed` distortion terms held.
fn candidate(distortion: DistortionKind, fixed: &[&str]) -> CameraConfig {
    let mut camera = CameraConfig::new(CameraModelSpec::pinhole(distortion, SensorKind::Identity));
    camera.fix.distortion = Some(fixed.iter().map(|s| s.to_string()).collect());
    camera
}

/// A: Brown-Conrady with `k1`, `k2` free (`k3`, `p1`, `p2` fixed).
fn radial2() -> CameraConfig {
    candidate(DistortionKind::BrownConrady5, &["k3", "p1", "p2"])
}

/// B: Brown-Conrady with every coefficient free.
fn brown_conrady_free() -> CameraConfig {
    candidate(DistortionKind::BrownConrady5, &[])
}

/// C: Rational8 with every coefficient free.
fn rational_free() -> CameraConfig {
    candidate(DistortionKind::Rational8, &[])
}

/// D: Brown-Conrady with only `k1` free (under-modelled).
fn radial1() -> CameraConfig {
    candidate(DistortionKind::BrownConrady5, &["k2", "k3", "p1", "p2"])
}

fn config_with(camera: CameraConfig, backend: SolverBackend) -> PlanarIntrinsicsConfig {
    let mut config = PlanarIntrinsicsConfig {
        camera,
        ..PlanarIntrinsicsConfig::default()
    };
    config.solver.backend = backend;
    config
}

fn held_out(score: &ModelScore) -> &HeldOutScore {
    match &score.outcome {
        ModelOutcome::Scored(h) => h,
        ModelOutcome::Failed { fold, reason } => {
            panic!("candidate failed at fold {fold:?}: {reason}")
        }
    }
}

/// `rms(a) − rms(b)` pooled, and its fold-paired standard error.
fn paired(a: &HeldOutScore, b: &HeldOutScore) -> (f64, f64) {
    let diffs: Vec<f64> = a
        .folds
        .iter()
        .zip(&b.folds)
        .map(|(x, y)| x.held_out_rms_px - y.held_out_rms_px)
        .collect();
    (
        a.held_out.rms - b.held_out.rms,
        sample_sd(&diffs) / (diffs.len() as f64).sqrt(),
    )
}

fn print_comparison(label: &str, comparison: &ModelComparison) {
    for (i, score) in comparison.candidates.iter().enumerate() {
        let h = held_out(score);
        eprintln!(
            "{label} candidate {i} ({:?}, {} free): training {:.4} px, held-out {:.4} px, \
             delta {:+.4} ± {:.4} px, scatter {:.4} px",
            score.camera.model.distortion,
            score.free_parameters,
            h.training.rms,
            h.held_out.rms,
            h.delta_vs_best_px,
            h.delta_se_px,
            h.fold_scatter_px,
        );
    }
    eprintln!("{label} ranking {:?}", comparison.ranking);
}

/// Truth has `k1`, `k2` only. Richer models fit their training views
/// better but do not predict held-out views better; the under-modelled one
/// loses clearly.
#[test]
#[ignore = "fits 4 candidates × 6 times on two backends; run in release"]
fn a_richer_model_does_not_win_on_held_out_views() {
    let input = scene(brown_conrady(-0.25, 0.08, 0.0, 0.0, 0.0), 12, 9, 20);
    let n_v = 12.0 * 9.0;
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let started = std::time::Instant::now();
        let comparison = compare_camera_models(
            &input,
            &config_with(radial2(), backend),
            &[brown_conrady_free(), rational_free(), radial1()],
            &CrossValidationOptions::default(),
            &RunControl::default(),
        )
        .unwrap();
        eprintln!(
            "{backend:?}: compare_camera_models took {:?}",
            started.elapsed()
        );
        print_comparison(&format!("{backend:?}"), &comparison);

        let [a, b, c, d] = [0, 1, 2, 3].map(|i| held_out(&comparison.candidates[i]));
        assert_eq!(
            comparison
                .candidates
                .iter()
                .map(|s| s.free_parameters)
                .collect::<Vec<_>>(),
            vec![6, 9, 12, 5]
        );
        // Nested models: the richer fits its own views at least as well.
        let tol = 1e-4;
        assert!(c.training.rms <= b.training.rms + tol, "{backend:?}");
        assert!(b.training.rms <= a.training.rms + tol, "{backend:?}");
        // Nothing of a held-out view leaks into its fit: its mean squared
        // error is at least the noise left after a 6-parameter pose fit.
        let floor = 2.0 * SIGMA_PX * SIGMA_PX * (1.0 - 3.0 / n_v) * (1.0 - 0.1);
        for (i, h) in [a, b, c, d].iter().enumerate() {
            assert_eq!(h.unprojected_points, 0);
            assert_eq!(h.held_out.count, 20 * 12 * 9);
            let mse = h.held_out.rms * h.held_out.rms;
            assert!(mse >= floor, "{backend:?} candidate {i}: {mse} < {floor}");
        }
        // The under-modelled camera loses clearly.
        let (delta, se) = paired(d, a);
        assert!(delta > 5.0 * se, "{backend:?}: D − A = {delta} ± {se}");
        assert_ne!(comparison.ranking[0], 3);
        // A richer one does not win by more than noise.
        for (name, richer) in [("B", b), ("C", c)] {
            let (delta, se) = paired(richer, a);
            assert!(
                delta > -2.0 * se,
                "{backend:?}: {name} − A = {delta} ± {se}"
            );
        }
    }
}

/// Truth has `k3` and tangential terms: the camera that frees them wins by
/// well more than its standard error.
#[test]
#[ignore = "fits 2 candidates × 6 times on two backends; run in release"]
fn a_needed_term_wins_on_held_out_views() {
    let input = scene(brown_conrady(-0.25, 0.08, -0.03, 0.0015, -0.001), 12, 9, 20);
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let comparison = compare_camera_models(
            &input,
            &config_with(radial2(), backend),
            &[brown_conrady_free()],
            &CrossValidationOptions::default(),
            &RunControl::default(),
        )
        .unwrap();
        print_comparison(&format!("{backend:?}"), &comparison);
        let (a, b) = (
            held_out(&comparison.candidates[0]),
            held_out(&comparison.candidates[1]),
        );
        let (delta, se) = paired(a, b);
        assert!(delta > 3.0 * se, "{backend:?}: A − B = {delta} ± {se}");
        assert_eq!(comparison.ranking, vec![1, 0]);
        assert!((b.delta_vs_best_px).abs() < 1e-15);
        assert!((a.delta_vs_best_px - delta).abs() < 1e-12);
        assert!((a.delta_se_px - se).abs() < 1e-12);
    }
}

/// A small scene for the behavioural tests.
fn small_scene() -> PlanarDataset {
    scene(brown_conrady(-0.2, 0.05, 0.0, 0.0, 0.0), 8, 6, 10)
}

fn small_opts() -> CrossValidationOptions {
    CrossValidationOptions::new(3, 4)
}

fn compare_small(input: &PlanarDataset) -> ModelComparison {
    compare_camera_models(
        input,
        &config_with(radial2(), SolverBackend::TinySolver),
        &[],
        &small_opts(),
        &RunControl::default(),
    )
    .unwrap()
}

/// Corrupting a held-out view changes nothing of the fit that held it out,
/// and of that fold's scores only its own.
#[test]
fn a_held_out_view_does_not_leak_into_its_fold() {
    let input = small_scene();
    let base = compare_small(&input);
    let fold = 0;
    let corrupted_view = base.folds[fold][0];

    let mut views = input.views.clone();
    let obs = &views[corrupted_view].obs;
    let shifted: Vec<Pt2> = obs
        .points_2d
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let s = if i % 2 == 0 { 1.5 } else { -1.5 };
            Pt2::new(p.x + s, p.y - s)
        })
        .collect();
    views[corrupted_view] =
        View::without_meta(CorrespondenceView::new(obs.points_3d.clone(), shifted).unwrap());
    let corrupted = compare_small(&PlanarDataset::new(views).unwrap());

    assert_eq!(base.folds, corrupted.folds);
    let (before, after) = (
        held_out(&base.candidates[0]),
        held_out(&corrupted.candidates[0]),
    );
    assert_eq!(
        before.folds[fold].training_rms_px.to_bits(),
        after.folds[fold].training_rms_px.to_bits()
    );
    for &v in &base.folds[fold] {
        let (x, y) = (before.per_view_rms_px[v], after.per_view_rms_px[v]);
        if v == corrupted_view {
            assert!(y > x + 0.5, "view {v}: {x} -> {y}");
        } else {
            assert_eq!(x.to_bits(), y.to_bits(), "view {v}: {x} -> {y}");
        }
    }
    // The other folds' fits saw the corrupted view.
    assert_ne!(
        before.folds[1].training_rms_px,
        after.folds[1].training_rms_px
    );
}

fn assert_partition(folds: &[Vec<usize>], views: usize, k: usize) {
    assert_eq!(folds.len(), k);
    let mut seen = vec![0; views];
    for fold in folds {
        assert!(fold.windows(2).all(|w| w[0] < w[1]));
        for &v in fold {
            seen[v] += 1;
        }
    }
    assert!(seen.iter().all(|&n| n == 1), "{folds:?}");
    let sizes: Vec<usize> = folds.iter().map(Vec::len).collect();
    let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
    assert!(max - min <= 1, "{sizes:?}");
}

#[test]
fn folds_partition_the_views_evenly_and_stably() {
    for views in [3, 4, 7, 10, 20, 33] {
        for k in 2..=views.min(7) {
            for seed in [0, 1, 99] {
                assert_partition(&partition(views, k, seed), views, k);
            }
        }
    }
    // The sequence is this library's, not a dependency's: a seed gives the
    // same folds everywhere.
    assert_eq!(
        partition(10, 3, 0),
        vec![vec![4, 5, 6, 8], vec![0, 1, 9], vec![2, 3, 7]]
    );
    assert_ne!(partition(10, 3, 0), partition(10, 3, 1));
}

/// The same seed gives byte-identical JSON; another seed another partition.
#[test]
fn the_comparison_is_deterministic() {
    let input = small_scene();
    let first = serde_json::to_string(&compare_small(&input)).unwrap();
    let second = serde_json::to_string(&compare_small(&input)).unwrap();
    assert_eq!(first, second);

    let other = compare_camera_models(
        &input,
        &config_with(radial2(), SolverBackend::TinySolver),
        &[],
        &CrossValidationOptions::new(3, 5),
        &RunControl::default(),
    )
    .unwrap();
    let base: ModelComparison = serde_json::from_str(&first).unwrap();
    assert_ne!(base.folds, other.folds);
    assert_partition(&other.folds, 10, 3);
    assert_partition(&base.folds, 10, 3);
}

/// A control that counts its reports.
fn counting_control() -> (RunControl, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&count);
    let control = RunControl::new().with_observer(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
    });
    (control, count)
}

#[test]
fn bad_inputs_are_typed_errors_before_any_work() {
    let input = small_scene();
    let config = config_with(radial2(), SolverBackend::TinySolver);
    let run = |input: &PlanarDataset,
               config: &PlanarIntrinsicsConfig,
               alternatives: &[CameraConfig],
               opts: CrossValidationOptions| {
        let (control, count) = counting_control();
        let err = compare_camera_models(input, config, alternatives, &opts, &control).unwrap_err();
        assert_eq!(count.load(Ordering::SeqCst), 0, "{err}");
        err
    };

    let err = run(&input, &config, &[], CrossValidationOptions::new(1, 0));
    assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

    // Fewer views than folds.
    let err = run(&input, &config, &[], CrossValidationOptions::new(11, 0));
    assert!(
        matches!(err, Error::InsufficientData { need: 11, got: 10 }),
        "{err}"
    );
    // A fold's fit would keep fewer than three views: 4 views in 2 folds.
    let four = PlanarDataset::new(input.views[..4].to_vec()).unwrap();
    let err = run(&four, &config, &[], CrossValidationOptions::new(2, 0));
    assert!(
        matches!(err, Error::InsufficientData { need: 6, got: 4 }),
        "{err}"
    );

    let mut unknown = radial2();
    unknown.fix.distortion = Some(vec!["k9".into()]);
    let err = run(&input, &config, &[unknown], small_opts());
    assert!(
        matches!(&err, Error::InvalidInput { reason } if reason.contains("candidate 1")),
        "{err}"
    );

    let err = run(&input, &config, &[radial1(), radial2()], small_opts());
    assert!(
        matches!(&err, Error::InvalidInput { reason } if reason.contains("repeats candidate 0")),
        "{err}"
    );

    let mut fixed_pose = config.clone();
    fixed_pose.optimize.fix_poses = vec![0];
    let err = run(&input, &fixed_pose, &[], small_opts());
    assert!(
        matches!(&err, Error::InvalidInput { reason } if reason.contains("fix_poses")),
        "{err}"
    );

    // The input checks of a planar calibration: mixed point covariances.
    let mut views = input.views.clone();
    let obs = &views[0].obs;
    let cov = PointCovariance::new(0.04, 0.0, 0.04).unwrap();
    views[0] = View::without_meta(
        CorrespondenceView::new_with_covariances(
            obs.points_3d.clone(),
            obs.points_2d.clone(),
            vec![cov; obs.len()],
        )
        .unwrap(),
    );
    let err = run(
        &PlanarDataset::new(views).unwrap(),
        &config,
        &[],
        small_opts(),
    );
    assert!(
        matches!(
            err,
            Error::Core(vision_calibration_core::Error::MixedCovariance { .. })
        ),
        "{err}"
    );

    // A cancelled token stops the run before its first fit.
    let token = CancelToken::new();
    token.cancel();
    let (control, count) = counting_control();
    let err = compare_camera_models(
        &input,
        &config,
        &[],
        &small_opts(),
        &control.with_cancel(token),
    )
    .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err}");
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

/// A token cancelled during the run stops it with `Cancelled`, not a failed
/// candidate.
#[test]
fn cancelling_mid_run_is_an_error_not_a_failed_candidate() {
    let input = small_scene();
    let token = CancelToken::new();
    let observer_token = token.clone();
    let reports = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&reports);
    let control = RunControl::new()
        .with_cancel(token)
        .with_observer(move |_| {
            if seen.fetch_add(1, Ordering::SeqCst) == 40 {
                observer_token.cancel();
            }
        });
    let err = compare_camera_models(
        &input,
        &config_with(radial2(), SolverBackend::TinySolver),
        &[radial1()],
        &small_opts(),
        &control,
    )
    .unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err}");
}

#[test]
fn options_and_result_round_trip_through_json() {
    let opts = CrossValidationOptions::new(7, 42);
    let json = serde_json::to_string(&opts).unwrap();
    assert_eq!(json, r#"{"folds":7,"seed":42}"#);
    assert_eq!(
        serde_json::from_str::<CrossValidationOptions>(&json).unwrap(),
        opts
    );
    assert_eq!(
        serde_json::from_str::<CrossValidationOptions>("{}").unwrap(),
        CrossValidationOptions::default()
    );
    assert!(serde_json::from_str::<CrossValidationOptions>(r#"{"fold":3}"#).is_err());

    let mut result = compare_small(&small_scene());
    result.candidates.push(ModelScore {
        camera: radial1(),
        free_parameters: 5,
        outcome: ModelOutcome::Failed {
            fold: Some(2),
            reason: "numerical failure".into(),
        },
    });
    let json = serde_json::to_string(&result).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["candidates"][0]["outcome"]["status"], "scored");
    assert!(value["candidates"][0]["outcome"]["held_out"]["rms"].is_f64());
    assert_eq!(value["candidates"][1]["outcome"]["status"], "failed");
    assert_eq!(value["candidates"][1]["outcome"]["fold"], 2);
    let back: ModelComparison = serde_json::from_str(&json).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), json);
}
