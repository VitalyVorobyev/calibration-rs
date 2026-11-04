//! Bundle adjustment from a calibration: the ground-truth matrix, the scale
//! gauge, a robust loss against gross outliers, and the covariance against a
//! Monte-Carlo of the pixel noise.

#![allow(missing_docs)]

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use std::sync::{Arc, Mutex};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, CancelToken, DistortionParams, FxFyCxCy, Iso3,
    KannalaBrandt4, ParameterCovariance, PointCovariance, Progress, ProjectionParams, Pt3,
    RunControl, SensorParams, Uncertainty, make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::Use;
use vision_mvg::MvgError;
use vision_mvg::bundle_adjust::{BundleAdjustOptions, Scale, bundle_adjust};
use vision_mvg::triangulation::{RigObservation, triangulate};

/// A deterministic generator (SplitMix64) with normal draws.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// A random unit direction.
    fn direction(&mut self) -> Vector3<f64> {
        loop {
            let v = Vector3::new(self.normal(), self.normal(), self.normal());
            if v.norm() > 1e-3 {
                return v.normalize();
            }
        }
    }
}

fn iso(r: [f64; 3], t: [f64; 3]) -> Iso3 {
    Iso3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_euler_angles(r[0], r[1], r[2]),
    )
}

fn pinhole(fx: f64) -> CameraParams {
    let k = FxFyCxCy {
        fx,
        fy: fx - 10.0,
        cx: 640.0,
        cy: 480.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.1,
        iters: 30,
        ..Default::default()
    };
    pinhole_camera_params(&make_pinhole_camera(k, distortion))
}

fn fisheye() -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Equidistant,
        distortion: DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: 0.02,
                k2: -0.002,
                k3: 0.0,
                k4: 0.0,
                iters: 0,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: 300.0,
            fy: 300.0,
            cx: 640.0,
            cy: 480.0,
        },
    }
}

fn rig_of(a: CameraParams, b: CameraParams, uncertainty: Uncertainty) -> CalibratedRig {
    CalibratedRig::new(
        vec![a, b],
        vec![Iso3::identity(), iso([0.0, 0.05, 0.01], [-0.12, 0.0, 0.01])],
        uncertainty,
    )
    .unwrap()
}

/// The covariance of a two-camera rig's parameters at 1e-20: nothing, but
/// present.
fn negligible_covariance() -> Uncertainty {
    let mut labels = Vec::new();
    for k in 0..2 {
        for p in ["fx", "fy", "cx", "cy"] {
            labels.push(format!("cam{k}.{p}"));
        }
    }
    for c in ["rx", "ry", "rz", "tx", "ty", "tz"] {
        labels.push(format!("cam1.cam_se3_rig.{c}"));
    }
    let n = labels.len();
    Uncertainty::Available(ParameterCovariance {
        covariance: (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1e-20 } else { 0.0 }).collect())
            .collect(),
        labels,
        views: Vec::new(),
        fixed: Vec::new(),
        noise: Vec::new(),
        num_residuals: 0,
        dof: 0,
        view_scatter: None,
    })
}

/// Four captures of a rig moving around the scene (world to rig).
fn captures() -> Vec<Iso3> {
    vec![
        Iso3::identity(),
        iso([0.01, -0.05, 0.0], [-0.25, 0.02, 0.03]),
        iso([-0.02, 0.04, 0.01], [0.2, -0.03, 0.05]),
        iso([0.015, 0.08, -0.01], [-0.1, 0.05, -0.1]),
    ]
}

/// Points 2 to 4 m ahead of capture 0.
fn scene(rng: &mut Rng, n: usize) -> Vec<Pt3> {
    (0..n)
        .map(|_| {
            let z = rng.range(2.0, 4.0);
            Pt3::new(rng.range(-0.3, 0.3) * z, rng.range(-0.2, 0.2) * z, z)
        })
        .collect()
}

/// Every camera at every capture sees every point; the pixel noise is a
/// draw of `sigma` px (stated as `stated`).
fn observe(
    rig: &CalibratedRig,
    poses: &[Iso3],
    points: &[Pt3],
    sigma: f64,
    stated: f64,
    rng: &mut Rng,
) -> Vec<Vec<RigObservation>> {
    let noise = PointCovariance::isotropic(stated).unwrap();
    points
        .iter()
        .map(|p| {
            let mut track = Vec::new();
            for (k, pose) in poses.iter().enumerate() {
                for c in 0..rig.num_cameras() {
                    let q = rig
                        .model(c)
                        .unwrap()
                        .project_point(&(rig.cam_se3_rig()[c] * pose * p))
                        .expect("projects");
                    let pixel = nalgebra::Point2::new(
                        q.x + sigma * rng.normal(),
                        q.y + sigma * rng.normal(),
                    );
                    track.push(RigObservation::new(k, c, pixel, noise));
                }
            }
            track
        })
        .collect()
}

/// Capture poses 1.. moved by `degrees` and `metres`, points by `metres`.
fn perturb(
    poses: &[Iso3],
    points: &[Pt3],
    degrees: f64,
    metres: f64,
    rng: &mut Rng,
) -> (Vec<Iso3>, Vec<Pt3>) {
    let poses = poses
        .iter()
        .enumerate()
        .map(|(k, p)| {
            if k == 0 {
                return *p;
            }
            let rotation = UnitQuaternion::from_scaled_axis(rng.direction() * degrees.to_radians());
            let shift = rng.direction() * metres;
            Iso3::from_parts(Translation3::from(shift), rotation) * p
        })
        .collect();
    let points = points
        .iter()
        .map(|p| p + rng.direction() * metres)
        .collect();
    (poses, points)
}

fn rotation_error(a: &Iso3, b: &Iso3) -> f64 {
    (a.rotation.inverse() * b.rotation).angle()
}

/// Noise-free pixels give back the truth (to 10 µm: the solver stops on a
/// relative cost decrease of 1e-5), from a start 2 degrees and 2 cm away, for a pair of pinholes and for a fisheye with a pinhole.
#[test]
fn ground_truth_matrix_converges_to_the_truth() {
    for (name, rig) in [
        (
            "pinhole pair",
            rig_of(pinhole(800.0), pinhole(780.0), Uncertainty::default()),
        ),
        (
            "fisheye + pinhole",
            rig_of(fisheye(), pinhole(800.0), Uncertainty::default()),
        ),
    ] {
        let mut rng = Rng(21);
        let truth_poses = captures();
        let truth_points = scene(&mut rng, 30);
        let tracks = observe(&rig, &truth_poses, &truth_points, 0.0, 0.3, &mut rng);
        let (poses, points) = perturb(&truth_poses, &truth_points, 2.0, 0.02, &mut rng);

        let opts = BundleAdjustOptions::new(Scale::Rig);
        let found = bundle_adjust(&rig, &poses, &points, &tracks, &opts).unwrap();
        let point_err = found
            .points
            .iter()
            .zip(&truth_points)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0_f64, f64::max);
        let t_err = found
            .rig_se3_world
            .iter()
            .zip(&truth_poses)
            .map(|(a, b)| (a.translation.vector - b.translation.vector).norm())
            .fold(0.0_f64, f64::max);
        let r_err = found
            .rig_se3_world
            .iter()
            .zip(&truth_poses)
            .map(|(a, b)| rotation_error(a, b))
            .fold(0.0_f64, f64::max);
        eprintln!(
            "{name}: max point error {point_err:.2e} m, pose translation {t_err:.2e} m, \
             rotation {r_err:.2e} rad, rms {:.2e} px",
            found.reprojection_rms_px
        );
        assert!(point_err < 1e-5, "{name}: point error {point_err}");
        assert!(t_err < 1e-5, "{name}: pose translation error {t_err}");
        assert!(r_err < 1e-5, "{name}: pose rotation error {r_err}");
        assert_eq!(found.rig_se3_world[0], truth_poses[0]);
    }
}

/// With noise the reprojection RMS (a distance, so `√2 σ` per axis) is the
/// noise, up to the degrees of freedom the fit absorbs.
#[test]
fn reprojection_rms_is_the_pixel_noise() {
    let rig = rig_of(pinhole(800.0), pinhole(780.0), Uncertainty::default());
    let mut rng = Rng(5);
    let truth_poses = captures();
    let truth_points = scene(&mut rng, 30);
    let sigma = 0.4;
    let tracks = observe(&rig, &truth_poses, &truth_points, sigma, sigma, &mut rng);
    let (poses, points) = perturb(&truth_poses, &truth_points, 2.0, 0.02, &mut rng);
    let found = bundle_adjust(
        &rig,
        &poses,
        &points,
        &tracks,
        &BundleAdjustOptions::new(Scale::Rig),
    )
    .unwrap();
    // 2 coordinates per observation; 3 per point, 6 per free capture, less
    // the scale fixed by the rig (none of the 18 + 90 is lost to it).
    let residuals = (30 * 4 * 2 * 2) as f64;
    let parameters = (30 * 3 + 3 * 6) as f64;
    let per_axis = found.reprojection_rms_px / 2.0_f64.sqrt();
    let sigma_hat = per_axis * (residuals / (residuals - parameters)).sqrt();
    eprintln!(
        "rms {:.3} px, dof-corrected {:.3} px, stated {sigma}; {:?}",
        found.reprojection_rms_px, sigma_hat, found.report
    );
    assert!((0.85..=1.15).contains(&(sigma_hat / sigma)));
    // No calibration covariance: the points have none either, and say why.
    assert!(found.point_covariance(0).is_none());
}

/// One moving camera: only a stated distance fixes the scale, and it does.
#[test]
fn one_moving_camera_takes_its_scale_from_a_distance() {
    let rig = CalibratedRig::single(pinhole(800.0), Uncertainty::default()).unwrap();
    let mut rng = Rng(9);
    let truth_poses = captures();
    let truth_points = scene(&mut rng, 24);
    let tracks = observe(&rig, &truth_poses, &truth_points, 0.0, 0.3, &mut rng);

    // A start 25 % too large: the points and the translations.
    let grown =
        |p: &Iso3| Iso3::from_parts(Translation3::from(p.translation.vector * 1.25), p.rotation);
    let poses: Vec<Iso3> = truth_poses.iter().map(grown).collect();
    let points: Vec<Pt3> = truth_points
        .iter()
        .map(|p| Pt3::from(p.coords * 1.25))
        .collect();
    let distance = (truth_points[3] - truth_points[7]).norm();
    let scale = Scale::Distance {
        points: [3, 7],
        distance,
        sigma: 1e-4,
    };
    let found = bundle_adjust(
        &rig,
        &poses,
        &points,
        &tracks,
        &BundleAdjustOptions::new(scale),
    )
    .unwrap();
    let worst = found
        .points
        .iter()
        .zip(&truth_points)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    eprintln!("metric scale from a distance: worst point error {worst:.2e} m");
    assert!(worst < 1e-4, "worst point error {worst}");

    // The rig cannot fix the scale of one camera, and a distance needs two points.
    let message = |scale: Scale, rig: &CalibratedRig| -> String {
        match bundle_adjust(
            rig,
            &poses,
            &points,
            &tracks,
            &BundleAdjustOptions::new(scale),
        ) {
            Err(MvgError::InvalidInput { reason }) => reason,
            other => panic!("expected InvalidInput, got {:?}", other.map(|_| ())),
        }
    };
    assert!(message(Scale::Rig, &rig).contains("two or more cameras"));
    let same = Scale::Distance {
        points: [2, 2],
        distance,
        sigma: 1e-4,
    };
    assert!(message(same, &rig).contains("distinct"));
    let far = Scale::Distance {
        points: [2, 99],
        distance,
        sigma: 1e-4,
    };
    assert!(message(far, &rig).contains("in range"));
    let zero = Scale::Distance {
        points: [2, 3],
        distance: 0.0,
        sigma: 1e-4,
    };
    assert!(message(zero, &rig).contains("positive"));
}

/// Inputs the gauge or the data cannot support are refused before a solve.
#[test]
fn invalid_inputs_are_refused() {
    let rig = rig_of(pinhole(800.0), pinhole(780.0), Uncertainty::default());
    let mut rng = Rng(2);
    let poses = captures();
    let points = scene(&mut rng, 8);
    let tracks = observe(&rig, &poses, &points, 0.0, 0.3, &mut rng);
    let opts = BundleAdjustOptions::new(Scale::Rig);
    let message = |rig: &CalibratedRig,
                   poses: &[Iso3],
                   points: &[Pt3],
                   tracks: &[Vec<RigObservation>],
                   opts: &BundleAdjustOptions|
     -> String {
        match bundle_adjust(rig, poses, points, tracks, opts) {
            Err(MvgError::InvalidInput { reason }) => reason,
            other => panic!("expected InvalidInput, got {:?}", other.map(|_| ())),
        }
    };

    // Points and tracks must pair up.
    assert!(message(&rig, &poses, &points[..7], &tracks, &opts).contains("tracks"));
    // A short track names itself.
    let mut short = tracks.clone();
    short[2].truncate(1);
    assert!(message(&rig, &poses, &points, &short, &opts).contains("track 2"));
    // A camera out of range.
    let mut bad = tracks.clone();
    bad[1][0].camera = 4;
    assert!(message(&rig, &poses, &points, &bad, &opts).contains("camera 4"));
    // A non-finite initial point.
    let mut nan = points.clone();
    nan[3].x = f64::NAN;
    assert!(message(&rig, &poses, &nan, &tracks, &opts).contains("point 3"));
    // Capture 0 defines the world: it must be observed.
    let unseen: Vec<Vec<RigObservation>> = tracks
        .iter()
        .map(|t| t.iter().filter(|o| o.capture != 0).copied().collect())
        .collect();
    assert!(message(&rig, &poses, &points, &unseen, &opts).contains("capture 0"));
    // One capture seen by one camera: nothing to adjust.
    let lonely: Vec<Vec<RigObservation>> = tracks
        .iter()
        .map(|t| {
            t.iter()
                .filter(|o| o.capture == 0 && o.camera == 0)
                .copied()
                .collect()
        })
        .collect();
    assert!(message(&rig, &poses, &points, &lonely, &opts).contains("least 2"));
    // Prior needs a covariance.
    let mut prior = opts.clone();
    prior.calibration = Use::Prior;
    assert!(message(&rig, &poses, &points, &tracks, &prior).contains("covariance"));
    // Two cameras at one centre fix no scale.
    let twins = CalibratedRig::new(
        vec![pinhole(800.0), pinhole(780.0)],
        vec![Iso3::identity(), iso([0.0, 0.1, 0.0], [0.0, 0.0, 0.0])],
        Uncertainty::default(),
    )
    .unwrap();
    assert!(message(&twins, &poses, &points, &tracks, &opts).contains("distinct centres"));
}

fn rms_point_error(points: &[Pt3], truth: &[Pt3]) -> f64 {
    let sum: f64 = points
        .iter()
        .zip(truth)
        .map(|(a, b)| (a - b).norm_squared())
        .sum();
    (sum / truth.len() as f64).sqrt()
}

/// 5 % of the observations are off by tens of pixels. A robust loss keeps
/// the points near where clean data puts them; the plain fit does not.
#[test]
fn a_robust_loss_survives_gross_outliers() {
    let rig = rig_of(pinhole(800.0), pinhole(780.0), Uncertainty::default());
    let mut rng = Rng(77);
    let truth_poses = captures();
    let truth_points = scene(&mut rng, 30);
    let sigma = 0.3;
    let clean = observe(&rig, &truth_poses, &truth_points, sigma, sigma, &mut rng);
    let (poses, points) = perturb(&truth_poses, &truth_points, 1.0, 0.02, &mut rng);

    let mut dirty = clean.clone();
    let total: usize = dirty.iter().map(Vec::len).sum();
    for n in 0..(total / 20) {
        let flat = (n * 20 + 7) % total;
        let (mut i, mut j) = (0, flat);
        while j >= dirty[i].len() {
            j -= dirty[i].len();
            i += 1;
        }
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        dirty[i][j].pixel.x += sign * rng.range(25.0, 60.0);
        dirty[i][j].pixel.y -= sign * rng.range(10.0, 40.0);
    }

    let run = |tracks: &[Vec<RigObservation>], loss: RobustLoss| {
        let mut opts = BundleAdjustOptions::new(Scale::Rig);
        opts.loss = loss;
        let found = bundle_adjust(&rig, &poses, &points, tracks, &opts).unwrap();
        rms_point_error(&found.points, &truth_points)
    };
    let reference = run(&clean, RobustLoss::None);
    let plain = run(&dirty, RobustLoss::None);
    let huber = run(&dirty, RobustLoss::Huber { scale: 2.0 });
    let cauchy = run(&dirty, RobustLoss::Cauchy { scale: 2.0 });
    eprintln!(
        "point rms error [m]: clean {reference:.2e}, outliers plain {plain:.2e}, \
         Huber {huber:.2e}, Cauchy {cauchy:.2e}"
    );
    assert!(
        plain > 5.0 * reference,
        "plain {plain} vs clean {reference}"
    );
    for (name, robust) in [("Huber", huber), ("Cauchy", cauchy)] {
        assert!(robust < 0.5 * plain, "{name} {robust} vs plain {plain}");
        assert!(
            robust < 2.5 * reference,
            "{name} {robust} vs clean {reference}"
        );
    }
}

/// The scatter of the tangent error of a pose, against the stated σ.
fn tangent_error(truth: &Iso3, estimate: &Iso3) -> [f64; 6] {
    // `estimate = truth · Exp(δ)`: the rotation is a rotation vector, the
    // translation a step in the truth's own frame.
    let r = (truth.rotation.inverse() * estimate.rotation).scaled_axis();
    let t = truth.rotation.inverse() * (estimate.translation.vector - truth.translation.vector);
    [r.x, r.y, r.z, t.x, t.y, t.z]
}

/// Over many draws of the pixel noise, with a calibration of negligible
/// covariance, the scatter of the points and of a capture's translation is
/// the covariance `bundle_adjust` reports.
#[test]
fn consider_covariance_matches_the_scatter_of_pixel_noise() {
    let rig = rig_of(pinhole(800.0), pinhole(780.0), negligible_covariance());
    let mut rng = Rng(31);
    let truth_poses: Vec<Iso3> = captures()[..3].to_vec();
    let truth_points = scene(&mut rng, 12);
    let (poses, points) = perturb(&truth_poses, &truth_points, 0.5, 0.01, &mut rng);
    let sigma = 0.3;
    let trials = 300;

    let n = truth_points.len();
    let mut point_sq = vec![[0.0_f64; 3]; n];
    let mut point_var = vec![[0.0_f64; 3]; n];
    let mut pose_sq = [0.0_f64; 6];
    let mut pose_var = [0.0_f64; 6];
    for trial in 0..trials {
        let tracks = observe(&rig, &truth_poses, &truth_points, sigma, sigma, &mut rng);
        let found = bundle_adjust(
            &rig,
            &poses,
            &points,
            &tracks,
            &BundleAdjustOptions::new(Scale::Rig),
        )
        .unwrap();
        for (sq, (p, t)) in point_sq
            .iter_mut()
            .zip(found.points.iter().zip(&truth_points))
        {
            let e = p - t;
            for a in 0..3 {
                sq[a] += e[a] * e[a] / trials as f64;
            }
        }
        let e = tangent_error(&truth_poses[1], &found.rig_se3_world[1]);
        for a in 0..6 {
            pose_sq[a] += e[a] * e[a] / trials as f64;
        }
        if trial == 0 {
            for (i, var) in point_var.iter_mut().enumerate() {
                let c = found.point_covariance(i).expect("point covariance");
                for a in 0..3 {
                    var[a] = c[(a, a)];
                }
            }
            let c = found.pose_covariance(1).expect("pose covariance");
            for a in 0..6 {
                pose_var[a] = c[(a, a)];
            }
            assert!(found.pose_covariance(0).is_none(), "capture 0 is the gauge");
        }
    }
    let mut point_ratio = [0.0; 3];
    for a in 0..3 {
        let mean = (0..n)
            .map(|i| point_sq[i][a] / point_var[i][a])
            .sum::<f64>()
            / n as f64;
        point_ratio[a] = mean.sqrt();
    }
    let pose_ratio: Vec<f64> = (0..6).map(|a| (pose_sq[a] / pose_var[a]).sqrt()).collect();
    eprintln!("point coordinate scatter/stated (x, y, z): {point_ratio:?}");
    eprintln!("capture 1 tangent scatter/stated (rx ry rz tx ty tz): {pose_ratio:?}");
    for (a, r) in point_ratio.iter().enumerate() {
        assert!((0.9..=1.1).contains(r), "point coordinate {a}: ratio {r}");
    }
    for (a, r) in pose_ratio.iter().enumerate() {
        assert!((0.9..=1.1).contains(r), "capture 1 tangent {a}: ratio {r}");
    }
}

/// The refined rig of a Prior adjustment is a rig: it triangulates.
#[test]
fn a_prior_rig_feeds_back_into_triangulation() {
    // A prior over the cameras' intrinsics and the extrinsics, wide enough
    // for the observations to move it.
    let mut covariance = match negligible_covariance() {
        Uncertainty::Available(c) => c,
        Uncertainty::Unavailable { .. } => unreachable!(),
    };
    for (i, label) in covariance.labels.iter().enumerate() {
        covariance.covariance[i][i] = if label.contains("cam_se3_rig") {
            1e-8
        } else {
            4.0
        };
    }
    let rig = rig_of(
        pinhole(800.0),
        pinhole(780.0),
        Uncertainty::Available(covariance),
    );
    let mut rng = Rng(8);
    let truth_poses: Vec<Iso3> = captures()[..3].to_vec();
    let truth_points = scene(&mut rng, 20);
    let tracks = observe(&rig, &truth_poses, &truth_points, 0.3, 0.3, &mut rng);
    let (poses, points) = perturb(&truth_poses, &truth_points, 0.5, 0.01, &mut rng);
    let mut opts = BundleAdjustOptions::new(Scale::Rig);
    opts.calibration = Use::Prior;
    let found = bundle_adjust(&rig, &poses, &points, &tracks, &opts).unwrap();
    let again = triangulate(
        &found.rig,
        &found.rig_se3_world,
        &tracks,
        &vision_calibration_core::RunControl::default(),
    )
    .unwrap();
    assert!(again.point_covariance(0).is_some());
    for (a, b) in again.points.iter().zip(&found.points) {
        assert!((a.point - b).norm() < 1e-2);
    }
    // The refined rig carries only the cameras' and extrinsics' labels.
    let c = found.rig.uncertainty().covariance().expect("covariance");
    assert!(
        c.labels.iter().all(|l| l.starts_with("cam")),
        "{:?}",
        c.labels
    );
    assert!(
        found
            .uncertainty
            .covariance()
            .unwrap()
            .index("capture1.rx")
            .is_some()
    );
}

/// A perturbed scene with its tracks: a solve that takes several iterations.
fn progress_scene() -> (CalibratedRig, Vec<Iso3>, Vec<Pt3>, Vec<Vec<RigObservation>>) {
    let rig = rig_of(pinhole(800.0), pinhole(780.0), Uncertainty::default());
    let mut rng = Rng(11);
    let truth_poses = captures();
    let truth_points = scene(&mut rng, 20);
    let tracks = observe(&rig, &truth_poses, &truth_points, 0.3, 0.3, &mut rng);
    let (poses, points) = perturb(&truth_poses, &truth_points, 2.0, 0.02, &mut rng);
    (rig, poses, points, tracks)
}

/// An observer collecting the iteration reports.
fn recording(control: RunControl) -> (RunControl, Arc<Mutex<Vec<Progress>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let control = control.with_observer(move |p| sink.lock().unwrap().push(p));
    (control, seen)
}

/// A report stream is the solve's iterations, from 1, and an observer does
/// not change the result: the outputs agree bit for bit.
#[test]
fn bundle_adjust_reports_iterations_without_changing_the_result() {
    let (rig, poses, points, tracks) = progress_scene();
    let plain = bundle_adjust(
        &rig,
        &poses,
        &points,
        &tracks,
        &BundleAdjustOptions::new(Scale::Rig),
    )
    .unwrap();

    let (control, seen) = recording(RunControl::default());
    let mut opts = BundleAdjustOptions::new(Scale::Rig);
    opts.control = control;
    let observed = bundle_adjust(&rig, &poses, &points, &tracks, &opts).unwrap();

    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty(), "no iteration reported");
    for (i, p) in seen.iter().enumerate() {
        match p {
            Progress::Iteration { iter, .. } => assert_eq!(*iter, i + 1),
            other => panic!("unexpected report {other:?}"),
        }
    }
    assert_eq!(
        format!("{:?}", plain.rig_se3_world),
        format!("{:?}", observed.rig_se3_world)
    );
    assert_eq!(
        format!("{:?}", plain.points),
        format!("{:?}", observed.points)
    );
    assert_eq!(
        format!("{:?}", plain.uncertainty),
        format!("{:?}", observed.uncertainty)
    );
    assert_eq!(
        format!("{:?}", plain.report),
        format!("{:?}", observed.report)
    );
    assert_eq!(
        plain.reprojection_rms_px.to_bits(),
        observed.reprojection_rms_px.to_bits()
    );
}

/// A token cancelled by the first report stops the solve with a typed error,
/// and a token cancelled beforehand stops it before any iteration.
#[test]
fn bundle_adjust_cancels_with_a_typed_error() {
    let (rig, poses, points, tracks) = progress_scene();

    let token = CancelToken::new();
    let cancel = token.clone();
    let reports = Arc::new(Mutex::new(0usize));
    let counter = Arc::clone(&reports);
    let mut opts = BundleAdjustOptions::new(Scale::Rig);
    opts.control = RunControl::new()
        .with_cancel(token)
        .with_observer(move |_| {
            *counter.lock().unwrap() += 1;
            cancel.cancel();
        });
    let err = bundle_adjust(&rig, &poses, &points, &tracks, &opts).unwrap_err();
    assert!(matches!(err, MvgError::Cancelled), "{err:?}");
    assert_eq!(*reports.lock().unwrap(), 1, "stopped within one iteration");

    let token = CancelToken::new();
    token.cancel();
    let (control, seen) = recording(RunControl::new().with_cancel(token));
    opts.control = control;
    let err = bundle_adjust(&rig, &poses, &points, &tracks, &opts).unwrap_err();
    assert!(matches!(err, MvgError::Cancelled), "{err:?}");
    assert!(seen.lock().unwrap().is_empty());
}

/// Triangulation refines the points with the same control: reports, the
/// same result, and a cancel.
#[test]
fn triangulate_reports_and_cancels() {
    let (rig, poses, _, tracks) = progress_scene();
    let plain = triangulate(&rig, &poses, &tracks, &RunControl::default()).unwrap();

    let (control, seen) = recording(RunControl::default());
    let observed = triangulate(&rig, &poses, &tracks, &control).unwrap();
    assert!(!seen.lock().unwrap().is_empty());
    assert_eq!(
        format!("{:?}", plain.points),
        format!("{:?}", observed.points)
    );
    assert_eq!(
        format!("{:?}", plain.uncertainty),
        format!("{:?}", observed.uncertainty)
    );

    let token = CancelToken::new();
    token.cancel();
    let err =
        triangulate(&rig, &poses, &tracks, &RunControl::new().with_cancel(token)).unwrap_err();
    assert!(matches!(err, MvgError::Cancelled), "{err:?}");
}
