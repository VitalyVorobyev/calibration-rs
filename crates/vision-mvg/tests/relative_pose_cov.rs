//! The covariance of the robust relative pose and of the essential matrix,
//! against Monte-Carlo.
//!
//! A fisheye (anisotropic, correlated stated noise) and a distorted pinhole
//! (isotropic) see a fixed scene; over many draws of the pixel noise from the
//! stated covariances, the scatter of the refined relative pose matches the
//! covariance `recover_relative_pose_robust` reports, coordinate by
//! coordinate in the gauge `|t| = 1` (three rotation-tangent coordinates and
//! two of the translation direction's tangent plane), and the scatter of `E`
//! matches the covariance `estimate_essential` reports.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    Mat3, PointCovariance, ProjectionParams, Pt2, Pt3, RansacOptions, RunControl, SensorParams,
    Uncertainty, Vec3, make_pinhole_camera, pinhole_camera_params,
};
use vision_mvg::Error;
use vision_mvg::robust::{estimate_essential, recover_relative_pose_robust};
use vision_mvg::types::PixelMatch;

/// A deterministic generator (SplitMix64) with uniform and normal draws.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1)`.
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

    /// A pixel offset drawn from `cov` (Cholesky factor times normals).
    fn offset(&mut self, cov: &PointCovariance) -> Pt2 {
        let l00 = cov.xx().sqrt();
        let l10 = cov.xy() / l00;
        let l11 = (cov.yy() - l10 * l10).sqrt();
        let (a, b) = (self.normal(), self.normal());
        Pt2::new(l00 * a, l10 * a + l11 * b)
    }
}

/// A 300 px/rad equidistant fisheye with Kannala-Brandt distortion.
fn fisheye() -> CameraParams {
    CameraParams::new(
        ProjectionParams::Equidistant,
        DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: 0.02,
                k2: -0.002,
                k3: 0.0,
                k4: 0.0,
                iters: 0,
            },
        },
        SensorParams::Identity,
        FxFyCxCy {
            fx: 300.0,
            fy: 300.0,
            cx: 640.0,
            cy: 480.0,
        },
    )
}

/// An 800 px pinhole with barrel distortion.
fn distorted_pinhole() -> CameraParams {
    let k = FxFyCxCy {
        fx: 800.0,
        fy: 800.0,
        cx: 640.0,
        cy: 480.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.2,
        k2: 0.03,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 60,
    };
    pinhole_camera_params(&make_pinhole_camera(k, distortion))
}

/// The second camera's pose in the first's frame: `second_se3_first`.
fn second_se3_first() -> Iso3 {
    Iso3::from_parts(
        Translation3::new(-0.5, 0.05, 0.1),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(0.02, -0.12, 0.03)),
    )
}

struct Scene {
    rig: CalibratedRig,
    /// The noise-free pixels of each point in the two cameras.
    pixels: Vec<(Pt2, Pt2)>,
    noise: [PointCovariance; 2],
}

fn scene() -> Scene {
    let rig = CalibratedRig::new(
        vec![fisheye(), distorted_pinhole()],
        vec![Iso3::identity(), second_se3_first()],
        Uncertainty::default(),
    )
    .unwrap();
    let mut rng = Rng(0x5EED_2F1E);
    let mut pixels = Vec::new();
    while pixels.len() < 120 {
        let z = rng.range(3.0, 8.0);
        let p = Pt3::new(z * rng.range(-0.3, 0.3), z * rng.range(-0.25, 0.25), z);
        let project = |c: usize| {
            rig.model(c)
                .unwrap()
                .project_point(&(rig.cam_se3_rig()[c] * p))
        };
        if let (Some(a), Some(b)) = (project(0), project(1)) {
            pixels.push((a, b));
        }
    }
    Scene {
        rig,
        pixels,
        noise: [
            PointCovariance::new(0.25, 0.08, 0.09).unwrap(),
            PointCovariance::isotropic(0.4).unwrap(),
        ],
    }
}

impl Scene {
    /// One draw of the matches, pixels from the stated covariances.
    fn draw(&self, rng: &mut Rng) -> Vec<PixelMatch> {
        let [n1, n2] = self.noise;
        self.pixels
            .iter()
            .map(|&(a, b)| {
                PixelMatch::new(
                    a + rng.offset(&n1).coords,
                    n1,
                    b + rng.offset(&n2).coords,
                    n2,
                )
            })
            .collect()
    }
}

fn ransac() -> RansacOptions {
    RansacOptions::new(5.0)
        .with_max_iters(200)
        .with_min_inliers(60)
        .with_seed(3)
}

/// An orthonormal basis of the plane perpendicular to the unit `t`.
fn tangent_basis(t: &Vec3) -> [Vec3; 2] {
    let a = t.cross(&Vec3::y()).normalize();
    [a, t.cross(&a)]
}

/// Per coordinate, the RMS error over the RMS stated σ: 1 when the
/// covariance describes the scatter.
fn ratios<const N: usize>(errors: &[[f64; N]], variances: &[[f64; N]]) -> [f64; N] {
    std::array::from_fn(|k| {
        let err: f64 = errors.iter().map(|e| e[k] * e[k]).sum();
        let var: f64 = variances.iter().map(|v| v[k]).sum();
        (err / var).sqrt()
    })
}

/// Over 600 draws (the standard error of a ratio is about 1 / √1200 = 0.03,
/// so [0.9, 1.1] is three of them wide), the scatter of each pose
/// coordinate matches its stated σ, as does the scatter of `E`'s entries.
#[test]
fn relative_pose_covariance_matches_the_scatter_of_pixel_noise() {
    const TRIALS: usize = 600;
    let scene = scene();
    let truth = second_se3_first();
    let t_true = truth.translation.vector.normalize();
    let basis = tangent_basis(&t_true);
    let e_true = {
        let e = t_true.cross_matrix() * truth.rotation.to_rotation_matrix().into_inner();
        e / e.norm()
    };
    let mut rng = Rng(0xD1CE);
    let control = RunControl::default();

    let (mut pose_err, mut pose_var) = (Vec::new(), Vec::new());
    let (mut e_err, mut e_var) = (Vec::new(), Vec::new());
    let mut nees = 0.0;
    for _ in 0..TRIALS {
        let matches = scene.draw(&mut rng);
        let pose = recover_relative_pose_robust(&scene.rig, [0, 1], &matches, &ransac(), &control)
            .unwrap();
        assert_eq!(pose.inliers.len(), matches.len());
        let cov = pose.pose_covariance().expect("pose covariance available");

        // R_est = R_true · Exp(δω), t_est = t_true + δt (in the tangent plane).
        let dw = (truth.rotation.inverse() * pose.second_se3_first.rotation).scaled_axis();
        let dt = pose.second_se3_first.translation.vector - t_true;
        let err = [dw.x, dw.y, dw.z, basis[0].dot(&dt), basis[1].dot(&dt)];
        // The stated covariance in the same five coordinates.
        let mut j = nalgebra::SMatrix::<f64, 5, 6>::zeros();
        for k in 0..3 {
            j[(k, k)] = 1.0;
        }
        for (row, b) in basis.iter().enumerate() {
            for k in 0..3 {
                j[(3 + row, 3 + k)] = b[k];
            }
        }
        let c5 = j * cov * j.transpose();
        let e5 = nalgebra::SVector::<f64, 5>::from(err);
        nees += (e5.transpose() * c5.try_inverse().unwrap() * e5)[0] / TRIALS as f64;
        pose_err.push(err);
        pose_var.push(std::array::from_fn::<f64, 5, _>(|k| c5[(k, k)]));

        let est = estimate_essential(&scene.rig, [0, 1], &matches, &ransac(), &control).unwrap();
        let ce = est.essential_covariance().expect("E covariance available");
        // E is scaled to ‖E‖ = √2, its sign a gauge too: align it with the truth.
        let e = est.essential / est.essential.norm();
        let sign = e.dot(&e_true).signum();
        let de: Mat3 = (e * sign - e_true) * std::f64::consts::SQRT_2;
        e_err.push(std::array::from_fn::<f64, 9, _>(|k| de[(k / 3, k % 3)]));
        e_var.push(std::array::from_fn::<f64, 9, _>(|k| ce[(k, k)]));
    }

    let pose_ratios = ratios(&pose_err, &pose_var);
    eprintln!(
        "relative pose sigma ratio (rx, ry, rz, t_a, t_b): {pose_ratios:.3?}; NEES {nees:.3} (5 expected)"
    );
    for (k, r) in pose_ratios.iter().enumerate() {
        assert!((0.9..=1.1).contains(r), "pose coordinate {k}: ratio {r}");
    }
    // The NEES has a standard error of √(10 / 600) = 0.13.
    assert!((nees - 5.0).abs() < 0.5, "NEES {nees}");

    let e_ratios = ratios(&e_err, &e_var);
    let pooled = {
        let err: f64 = e_err.iter().flatten().map(|v| v * v).sum();
        let var: f64 = e_var.iter().flatten().sum();
        (err / var).sqrt()
    };
    eprintln!("essential sigma ratio per entry: {e_ratios:.3?}; pooled {pooled:.3}");
    assert!((0.9..=1.1).contains(&pooled), "E: pooled ratio {pooled}");
}

/// The uncertainty serializes with the result, and a cancelled token stops
/// the refinement with `Cancelled`, never a partial result.
#[test]
fn the_uncertainty_is_labelled_and_a_cancelled_run_returns_none() {
    let scene = scene();
    let matches = scene.draw(&mut Rng(1));
    let control = RunControl::default();
    let pose =
        recover_relative_pose_robust(&scene.rig, [0, 1], &matches, &ransac(), &control).unwrap();
    let labels = &pose.uncertainty.covariance().unwrap().labels;
    assert_eq!(
        labels,
        &[
            "rotation.rx",
            "rotation.ry",
            "rotation.rz",
            "direction.x",
            "direction.y",
            "direction.z"
        ]
    );
    // The direction block has no variance along the translation.
    let cov = pose.pose_covariance().unwrap();
    let t = pose.second_se3_first.translation.vector;
    let along = t.transpose() * cov.fixed_view::<3, 3>(3, 3) * t;
    assert!(along[0].abs() < 1e-12 * cov[(3, 3)].max(cov[(4, 4)]));

    let est = estimate_essential(&scene.rig, [0, 1], &matches, &ransac(), &control).unwrap();
    assert!((est.essential.norm() - std::f64::consts::SQRT_2).abs() < 1e-9);
    assert_eq!(
        est.uncertainty.covariance().unwrap().labels[..3],
        ["essential.e00", "essential.e01", "essential.e02"]
    );
    // E has five degrees of freedom: its 9×9 covariance is rank 5.
    let ce = est.essential_covariance().unwrap();
    let eig = ce.symmetric_eigen().eigenvalues;
    let max = eig.max();
    assert_eq!(eig.iter().filter(|&&l| l > 1e-9 * max).count(), 5);

    let iterations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = iterations.clone();
    let observed = RunControl::new().with_observer(move |_| {
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    let again =
        recover_relative_pose_robust(&scene.rig, [0, 1], &matches, &ransac(), &observed).unwrap();
    assert!(iterations.load(std::sync::atomic::Ordering::Relaxed) > 0);
    assert_eq!(again.second_se3_first, pose.second_se3_first);
    assert_eq!(again.inliers, pose.inliers);

    let token = vision_calibration_core::CancelToken::new();
    token.cancel();
    let cancelled = RunControl::new().with_cancel(token);
    assert!(matches!(
        recover_relative_pose_robust(&scene.rig, [0, 1], &matches, &ransac(), &cancelled),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        estimate_essential(&scene.rig, [0, 1], &matches, &ransac(), &cancelled),
        Err(Error::Cancelled)
    ));
}
