//! Object pose from a calibration: the ground-truth matrix over camera
//! models and object shapes, a rig that sees the object with two cameras,
//! RANSAC in σ against gross outliers, the covariance against a Monte-Carlo of
//! the pixel noise, and the errors.

#![allow(missing_docs)]

use nalgebra::{Point2, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    ParameterCovariance, PointCovariance, ProjectionParams, Pt3, RansacOptions, SensorParams,
    Uncertainty, UncertaintyUnavailable, make_pinhole_camera, pinhole_camera_params,
};
use vision_calibration_core::{CancelToken, Progress, RunControl};
use vision_mvg::MvgError;
use vision_mvg::pose::{ObjectPose, PoseObservation, PoseOptions, estimate_pose};

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

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// A pixel offset drawn from `cov` (Cholesky factor times normals).
    fn offset(&mut self, cov: &PointCovariance) -> Point2<f64> {
        let l00 = cov.xx().sqrt();
        let l10 = cov.xy() / l00;
        let l11 = (cov.yy() - l10 * l10).sqrt();
        let (a, b) = (self.normal(), self.normal());
        Point2::new(l00 * a, l10 * a + l11 * b)
    }
}

fn iso(r: [f64; 3], t: [f64; 3]) -> Iso3 {
    Iso3::from_parts(
        Translation3::new(t[0], t[1], t[2]),
        UnitQuaternion::from_euler_angles(r[0], r[1], r[2]),
    )
}

fn pinhole(fx: f64, k1: f64) -> CameraParams {
    let k = FxFyCxCy {
        fx,
        fy: fx - 10.0,
        cx: 640.0,
        cy: 480.0,
    };
    let distortion = BrownConrady5 {
        k1,
        k2: 0.02,
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

fn stereo(a: CameraParams, b: CameraParams, uncertainty: Uncertainty) -> CalibratedRig {
    CalibratedRig::new(
        vec![a, b],
        vec![Iso3::identity(), iso([0.0, 0.05, 0.01], [-0.12, 0.0, 0.01])],
        uncertainty,
    )
    .unwrap()
}

/// A flat 7 x 5 grid of points on a 3 cm pitch, centred on the origin.
fn planar_object() -> Vec<Pt3> {
    (0..35)
        .map(|i| {
            Pt3::new(
                0.03 * f64::from(i % 7) - 0.09,
                0.03 * f64::from(i / 7) - 0.06,
                0.0,
            )
        })
        .collect()
}

/// A 3D object: the grid with a relief of up to 4 cm.
fn spatial_object() -> Vec<Pt3> {
    planar_object()
        .into_iter()
        .enumerate()
        .map(|(i, p)| Pt3::new(p.x, p.y, 0.04 * ((i * 5 % 7) as f64 / 6.0)))
        .collect()
}

/// The object about 0.7 m ahead of the rig, tilted.
fn truth() -> Iso3 {
    iso([0.25, -0.35, 0.1], [0.02, -0.01, 0.7])
}

/// The pixels of `object` through `rig_se3_object` in the cameras `cameras`,
/// plus a draw of `cov` from `rng` (none when `rng` is `None`).
fn observe(
    rig: &CalibratedRig,
    cameras: &[usize],
    object: &[Pt3],
    rig_se3_object: &Iso3,
    noise: PointCovariance,
    mut rng: Option<&mut Rng>,
) -> Vec<PoseObservation> {
    let mut out = Vec::new();
    for &c in cameras {
        for p in object {
            let q = rig
                .model(c)
                .unwrap()
                .project_point(&(rig.cam_se3_rig()[c] * rig_se3_object * p))
                .expect("projects");
            let e = rng
                .as_deref_mut()
                .map_or(Point2::new(0.0, 0.0), |r| r.offset(&noise));
            out.push(PoseObservation::new(
                c,
                *p,
                Point2::new(q.x + e.x, q.y + e.y),
                noise,
            ));
        }
    }
    out
}

/// `estimate = truth · Exp(δ)`: the rotation is a rotation vector, the
/// translation a step in the truth's own frame.
fn tangent_error(truth: &Iso3, estimate: &Iso3) -> [f64; 6] {
    let r = (truth.rotation.inverse() * estimate.rotation).scaled_axis();
    let t = truth.rotation.inverse() * (estimate.translation.vector - truth.translation.vector);
    [r.x, r.y, r.z, t.x, t.y, t.z]
}

fn assert_exact(pose: &ObjectPose, truth: &Iso3, what: &str) {
    let e = tangent_error(truth, &pose.rig_se3_object);
    let rot = Vector3::new(e[0], e[1], e[2]).norm();
    let tr = Vector3::new(e[3], e[4], e[5]).norm();
    eprintln!("{what}: rotation error {rot:.2e} rad, translation error {tr:.2e} m");
    assert!(rot < 1e-9 && tr < 1e-9, "{what}: {rot:e} rad, {tr:e} m");
    assert!(pose.reprojection_rms_px < 1e-7, "{what}: rms");
}

/// Without noise the pose is exact, whatever the camera model, the object's
/// shape or the camera that starts it.
#[test]
fn noise_free_pose_is_exact_for_every_camera_model() {
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let rigs = [
        (
            "pinhole",
            stereo(
                pinhole(800.0, 0.0),
                pinhole(780.0, 0.0),
                Uncertainty::default(),
            ),
        ),
        (
            "brown-conrady",
            stereo(
                pinhole(800.0, -0.2),
                pinhole(780.0, -0.15),
                Uncertainty::default(),
            ),
        ),
        (
            "kb4 fisheye",
            stereo(fisheye(), fisheye(), Uncertainty::default()),
        ),
    ];
    for (name, rig) in &rigs {
        for (shape, object) in [("planar", planar_object()), ("spatial", spatial_object())] {
            for cameras in [vec![0], vec![1], vec![0, 1]] {
                let obs = observe(rig, &cameras, &object, &truth(), noise, None);
                let pose = estimate_pose(rig, &obs, &PoseOptions::default()).unwrap();
                assert_exact(
                    &pose,
                    &truth(),
                    &format!("{name} {shape} cameras {cameras:?}"),
                );
                assert_eq!(pose.inliers.len(), obs.len());
            }
        }
    }
}

/// The result is in the rig frame: a camera that is not the rig's origin
/// gives the same pose as the one that is.
#[test]
fn pose_is_in_the_rig_frame_whichever_camera_saw_it() {
    let rig = stereo(
        pinhole(800.0, -0.1),
        pinhole(780.0, -0.1),
        Uncertainty::default(),
    );
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let obs = observe(&rig, &[1], &spatial_object(), &truth(), noise, None);
    let pose = estimate_pose(&rig, &obs, &PoseOptions::default()).unwrap();
    assert_exact(&pose, &truth(), "camera 1 alone");
}

fn translation_trace(pose: &ObjectPose) -> f64 {
    let c = pose.pose_covariance().expect("covariance");
    (3..6).map(|a| c[(a, a)]).sum()
}

fn rotation_trace(pose: &ObjectPose) -> f64 {
    let c = pose.pose_covariance().expect("covariance");
    (0..3).map(|a| c[(a, a)]).sum()
}

/// Both cameras' observations enter the refinement, and the pose is more
/// certain than either camera's alone.
#[test]
fn two_cameras_are_more_certain_than_either() {
    let rig = stereo(
        pinhole(800.0, -0.1),
        pinhole(780.0, -0.1),
        negligible_covariance(),
    );
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let mut rng = Rng(11);
    let all = observe(
        &rig,
        &[0, 1],
        &spatial_object(),
        &truth(),
        noise,
        Some(&mut rng),
    );
    let n = all.len() / 2;
    let opts = PoseOptions::default();
    let both = estimate_pose(&rig, &all, &opts).unwrap();
    let first = estimate_pose(&rig, &all[..n], &opts).unwrap();
    let second = estimate_pose(&rig, &all[n..], &opts).unwrap();
    assert_eq!(both.inliers.len(), all.len());
    let (tb, t0, t1) = (
        translation_trace(&both),
        translation_trace(&first),
        translation_trace(&second),
    );
    let (rb, r0, r1) = (
        rotation_trace(&both),
        rotation_trace(&first),
        rotation_trace(&second),
    );
    eprintln!(
        "translation sigma (sqrt trace, mm): both {:.3}, camera 0 {:.3}, camera 1 {:.3}; \
         gain over the better single camera {:.2}x",
        1e3 * tb.sqrt(),
        1e3 * t0.sqrt(),
        1e3 * t1.sqrt(),
        (t0.min(t1) / tb).sqrt()
    );
    eprintln!(
        "rotation sigma (sqrt trace, mrad): both {:.3}, camera 0 {:.3}, camera 1 {:.3}; \
         gain {:.2}x",
        1e3 * rb.sqrt(),
        1e3 * r0.sqrt(),
        1e3 * r1.sqrt(),
        (r0.min(r1) / rb).sqrt()
    );
    assert!(tb < t0 && tb < t1, "translation: {tb} vs {t0}, {t1}");
    assert!(rb < r0 && rb < r1, "rotation: {rb} vs {r0}, {r1}");
}

/// With 30 % gross outliers RANSAC at 3 σ recovers the inliers; without it
/// the pose is lost.
#[test]
fn ransac_in_sigma_recovers_the_inliers() {
    let rig = stereo(
        pinhole(800.0, -0.1),
        pinhole(780.0, -0.1),
        negligible_covariance(),
    );
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let mut rng = Rng(5);
    let object = spatial_object();
    let mut obs = observe(&rig, &[0, 1], &object, &truth(), noise, Some(&mut rng));
    // 30 % of each camera's observations get a pixel 40 to 140 px off.
    let per_camera = object.len();
    let mut outliers = Vec::new();
    for c in 0..2 {
        for j in 0..per_camera {
            if j % 10 < 3 {
                let i = c * per_camera + j;
                let angle = 2.4 * i as f64;
                let size = 40.0 + 100.0 * rng.uniform();
                obs[i].pixel.x += size * angle.cos();
                obs[i].pixel.y += size * angle.sin();
                outliers.push(i);
            }
        }
    }
    let mut opts = PoseOptions::default();
    opts.ransac = Some(RansacOptions {
        min_inliers: 12,
        ..RansacOptions::new(3.0)
    });
    let pose = estimate_pose(&rig, &obs, &opts).unwrap();
    let missed: Vec<usize> = (0..obs.len())
        .filter(|i| !outliers.contains(i) && !pose.inliers.contains(i))
        .collect();
    let accepted: Vec<usize> = outliers
        .iter()
        .copied()
        .filter(|i| pose.inliers.contains(i))
        .collect();
    let e = tangent_error(&truth(), &pose.rig_se3_object);
    eprintln!(
        "ransac: {} observations, {} outliers; {} inliers, {} outliers accepted, \
         {} true inliers beyond 3 sigma; pose error {:.2e} rad, {:.2e} m; rms {:.3} px",
        obs.len(),
        outliers.len(),
        pose.inliers.len(),
        accepted.len(),
        missed.len(),
        Vector3::new(e[0], e[1], e[2]).norm(),
        Vector3::new(e[3], e[4], e[5]).norm(),
        pose.reprojection_rms_px
    );
    assert!(accepted.is_empty(), "outliers accepted: {accepted:?}");
    // About 1 % of Gaussian pixels fall beyond 3 σ.
    assert!(missed.len() <= 2, "inliers lost: {missed:?}");
    assert!(Vector3::new(e[3], e[4], e[5]).norm() < 2e-3);
    assert!(Vector3::new(e[0], e[1], e[2]).norm() < 2e-3);

    // Without RANSAC, the outliers drag the pose away.
    let plain = estimate_pose(&rig, &obs, &PoseOptions::default()).unwrap();
    let plain_error = tangent_error(&truth(), &plain.rig_se3_object);
    assert!(Vector3::new(plain_error[3], plain_error[4], plain_error[5]).norm() > 5e-3);
}

/// RANSAC works the same for a planar object seen through a fisheye.
#[test]
fn ransac_on_a_fisheye_with_a_planar_object() {
    let rig = stereo(fisheye(), fisheye(), Uncertainty::default());
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let mut obs = observe(&rig, &[0], &planar_object(), &truth(), noise, None);
    let outliers = [1, 4, 8, 13, 17, 21, 22, 26, 30, 33];
    for &i in &outliers {
        obs[i].pixel.x += 60.0;
        obs[i].pixel.y -= 35.0;
    }
    let mut opts = PoseOptions::default();
    opts.ransac = Some(RansacOptions {
        min_inliers: 12,
        ..RansacOptions::new(3.0)
    });
    let pose = estimate_pose(&rig, &obs, &opts).unwrap();
    let expected: Vec<usize> = (0..obs.len()).filter(|i| !outliers.contains(i)).collect();
    assert_eq!(pose.inliers, expected);
    assert_exact(&pose, &truth(), "fisheye ransac");
}

/// With a calibration of negligible covariance, the scatter of each tangent
/// component over draws of the pixel noise (anisotropic, correlated) is the
/// covariance `estimate_pose` reports.
///
/// A ratio from `n` trials has a relative error of `1/sqrt(2n)`: 600 trials
/// resolve it to 2.9 %, so the `[0.9, 1.1]` band is 3.4 standard errors wide.
#[test]
fn pose_covariance_matches_the_scatter_of_pixel_noise() {
    let rig = stereo(
        pinhole(800.0, -0.1),
        pinhole(780.0, -0.1),
        negligible_covariance(),
    );
    let noise = PointCovariance::new(0.16, 0.05, 0.09).unwrap();
    let object = spatial_object();
    let trials = 600;
    for cameras in [vec![0], vec![0, 1]] {
        let mut rng = Rng(77 + cameras.len() as u64);
        let mut sq = [0.0_f64; 6];
        let mut stated = [0.0_f64; 6];
        for trial in 0..trials {
            let obs = observe(&rig, &cameras, &object, &truth(), noise, Some(&mut rng));
            let pose = estimate_pose(&rig, &obs, &PoseOptions::default()).unwrap();
            let e = tangent_error(&truth(), &pose.rig_se3_object);
            let c = pose.pose_covariance().expect("pose covariance");
            for a in 0..6 {
                sq[a] += e[a] * e[a];
                stated[a] += c[(a, a)];
            }
            let _ = trial;
        }
        let ratio: Vec<f64> = (0..6).map(|a| (sq[a] / stated[a]).sqrt()).collect();
        eprintln!("cameras {cameras:?}: scatter/stated (rx ry rz tx ty tz): {ratio:.3?}");
        for (a, r) in ratio.iter().enumerate() {
            assert!(
                (0.9..=1.1).contains(r),
                "cameras {cameras:?} component {a}: {r}"
            );
        }
    }
}

/// A rig without a covariance still gives the pose and names what is missing.
#[test]
fn a_rig_without_covariance_still_gives_the_pose() {
    let rig = stereo(
        pinhole(800.0, 0.0),
        pinhole(780.0, 0.0),
        Uncertainty::default(),
    );
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let mut rng = Rng(3);
    let obs = observe(
        &rig,
        &[0, 1],
        &spatial_object(),
        &truth(),
        noise,
        Some(&mut rng),
    );
    let pose = estimate_pose(&rig, &obs, &PoseOptions::default()).unwrap();
    assert!(matches!(
        pose.uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::CalibrationWithoutCovariance
        }
    ));
    assert!(pose.pose_covariance().is_none());
    let e = tangent_error(&truth(), &pose.rig_se3_object);
    assert!(Vector3::new(e[3], e[4], e[5]).norm() < 5e-3);
}

#[test]
fn errors_are_typed_and_come_before_any_work() {
    let rig = stereo(
        pinhole(800.0, 0.0),
        pinhole(780.0, 0.0),
        Uncertainty::default(),
    );
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let good = observe(&rig, &[0], &spatial_object(), &truth(), noise, None);
    let opts = PoseOptions::default();
    let message = |obs: &[PoseObservation], opts: &PoseOptions| -> String {
        match estimate_pose(&rig, obs, opts) {
            Err(e) => {
                assert!(matches!(e, MvgError::InvalidInput { .. }), "{e}");
                e.to_string()
            }
            Ok(_) => panic!("expected an error"),
        }
    };

    // Too few observations in the best camera.
    let m = message(&good[..3], &opts);
    assert!(m.contains("at least 4"), "{m}");
    // Observations spread thinly over the cameras.
    let mut split = good[..6].to_vec();
    for o in &mut split[3..] {
        o.camera = 1;
    }
    let m = message(&split, &opts);
    assert!(m.contains("at least 4"), "{m}");

    // Camera out of range.
    let mut bad = good.clone();
    bad[2].camera = 5;
    let m = message(&bad, &opts);
    assert!(m.contains("observation 2") && m.contains("camera 5"), "{m}");

    // Not finite.
    let mut bad = good.clone();
    bad[7].pixel.x = f64::NAN;
    let m = message(&bad, &opts);
    assert!(m.contains("observation 7") && m.contains("finite"), "{m}");

    // Invalid RANSAC options.
    let mut ransac = PoseOptions::default();
    ransac.ransac = Some(RansacOptions::new(-1.0));
    assert!(matches!(
        estimate_pose(&rig, &good, &ransac),
        Err(MvgError::Core(_))
    ));

    // Collinear points have no pose.
    let line: Vec<PoseObservation> = (0..6)
        .map(|i| {
            let p = Pt3::new(0.02 * f64::from(i), 0.0, 0.0);
            let q = rig.model(0).unwrap().project_point(&(truth() * p)).unwrap();
            PoseObservation::new(0, p, q, noise)
        })
        .collect();
    assert!(matches!(
        estimate_pose(&rig, &line, &opts),
        Err(MvgError::Geometry(_))
    ));

    // RANSAC finds no consensus in pure noise.
    let mut scrambled = good.clone();
    let mut rng = Rng(9);
    for o in &mut scrambled {
        o.pixel = Point2::new(rng.uniform() * 1280.0, rng.uniform() * 960.0);
    }
    let mut ransac = PoseOptions::default();
    ransac.ransac = Some(RansacOptions {
        min_inliers: 20,
        max_iters: 200,
        ..RansacOptions::new(3.0)
    });
    assert!(matches!(
        estimate_pose(&rig, &scrambled, &ransac),
        Err(MvgError::NoConsensus)
    ));
}

/// The refinement reports its iterations without changing the pose, and a
/// cancelled token stops it with a typed error.
#[test]
fn pose_reports_iterations_and_cancels() {
    let rig = stereo(
        pinhole(800.0, -0.1),
        pinhole(780.0, -0.1),
        Uncertainty::default(),
    );
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let obs = observe(&rig, &[0, 1], &spatial_object(), &truth(), noise, None);
    let plain = estimate_pose(&rig, &obs, &PoseOptions::default()).unwrap();

    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&seen);
    let mut opts = PoseOptions::default();
    opts.control = RunControl::new().with_observer(move |p| sink.lock().unwrap().push(p));
    let observed = estimate_pose(&rig, &obs, &opts).unwrap();
    assert!(
        seen.lock()
            .unwrap()
            .iter()
            .all(|p| matches!(p, Progress::Iteration { .. }))
    );
    assert_eq!(format!("{plain:?}"), format!("{observed:?}"));

    let token = CancelToken::new();
    token.cancel();
    opts.control = RunControl::new().with_cancel(token);
    let err = estimate_pose(&rig, &obs, &opts).unwrap_err();
    assert!(matches!(err, MvgError::Cancelled), "{err:?}");
}
