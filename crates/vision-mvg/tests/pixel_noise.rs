//! The stated pixel noise sets the scale of the calibrated two-view residual.
//!
//! `estimate_essential` scores a match by the covariance-weighted Sampson
//! error: the epipolar constraint over its first-order standard deviation, in
//! σ. Two properties make that a usable unit, and are checked here against
//! synthetic ground truth:
//!
//! 1. with pixels drawn from the stated (anisotropic, correlated, different per
//!    camera) covariances through a fisheye and a distorted pinhole, the RMS
//!    over the inliers of the refined estimate is 1, and halves when the
//!    stated σ is doubled. (Its value at the true essential matrix is checked
//!    in the crate's unit tests; the covariance of the refined estimate in
//!    `relative_pose_cov`.)
//! 2. with a fixed stated noise, a threshold of 3 σ keeps the inliers and drops
//!    the gross outliers at any focal length.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    PointCovariance, ProjectionParams, Pt2, Pt3, RansacOptions, RunControl, SensorParams,
    Uncertainty, Vec3, make_pinhole_camera, pinhole_camera_params,
};
use vision_mvg::robust::estimate_essential;
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

    /// A standard normal draw (Box-Muller).
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

fn pose(rx: f64, ry: f64, rz: f64, t: Vec3) -> Iso3 {
    Iso3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(rx, ry, rz)),
    )
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

/// A pinhole of focal length `f` px with barrel distortion.
fn distorted_pinhole(f: f64) -> CameraParams {
    let k = FxFyCxCy {
        fx: f,
        fy: f,
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

/// Points in front of the rig, within a cone both cameras see.
fn scene(rng: &mut Rng, n: usize) -> Vec<Pt3> {
    (0..n)
        .map(|_| {
            let z = rng.range(3.0, 8.0);
            Pt3::new(z * rng.range(-0.3, 0.3), z * rng.range(-0.25, 0.25), z)
        })
        .collect()
}

fn project(rig: &CalibratedRig, camera: usize, p: &Pt3) -> Option<Pt2> {
    rig.model(camera)
        .unwrap()
        .project_point(&(rig.cam_se3_rig()[camera] * p))
}

/// Matches of the rig's cameras 0 and 1 with pixels drawn from the true
/// covariances `truth`, stated as `stated(truth)`.
fn noisy_matches(
    rig: &CalibratedRig,
    world: &[Pt3],
    truth: [PointCovariance; 2],
    stated_scale: f64,
    rng: &mut Rng,
) -> Vec<PixelMatch> {
    let stated = |c: &PointCovariance| {
        let s2 = stated_scale * stated_scale;
        PointCovariance::new(c.xx() * s2, c.xy() * s2, c.yy() * s2).unwrap()
    };
    world
        .iter()
        .filter_map(|p| {
            let (u1, u2) = (project(rig, 0, p)?, project(rig, 1, p)?);
            let (n1, n2) = (rng.offset(&truth[0]), rng.offset(&truth[1]));
            Some(PixelMatch::new(
                u1 + n1.coords,
                stated(&truth[0]),
                u2 + n2.coords,
                stated(&truth[1]),
            ))
        })
        .collect()
}

fn opts(thresh: f64, min_inliers: usize) -> RansacOptions {
    RansacOptions::new(thresh)
        .with_max_iters(3000)
        .with_min_inliers(min_inliers)
        .with_seed(11)
}

/// The RMS of the weighted error over the inliers of the estimated `E` is
/// 1 σ for pixels drawn from the stated covariances, and halves when every
/// stated σ is doubled: the estimate minimizes the stated residual.
#[test]
fn weighted_error_is_one_sigma_for_the_stated_noise() {
    let cam1_se3_rig = pose(0.02, -0.12, 0.03, Vec3::new(-0.5, 0.05, 0.1));
    let rig = CalibratedRig::new(
        vec![fisheye(), distorted_pinhole(800.0)],
        vec![Iso3::identity(), cam1_se3_rig],
        Uncertainty::default(),
    )
    .unwrap();
    // Anisotropic and correlated on the fisheye; isotropic on the pinhole.
    let truth = [
        PointCovariance::new(0.25, 0.08, 0.09).unwrap(),
        PointCovariance::isotropic(0.4).unwrap(),
    ];

    let mut rng = Rng(0xA11CE);
    let world = scene(&mut rng, 2200);
    let matches = noisy_matches(&rig, &world, truth, 1.0, &mut rng);
    assert!(matches.len() >= 2000, "{} matches", matches.len());
    let found = estimate_essential(
        &rig,
        [0, 1],
        &matches,
        &opts(5.0, 100),
        &RunControl::default(),
    )
    .unwrap();
    assert!(found.inliers.len() * 100 >= matches.len() * 99);
    let rms = found.inlier_rms;
    eprintln!("stated = true noise: inlier RMS {rms:.4} sigma");
    assert!((0.95..=1.05).contains(&rms), "inlier RMS {rms} sigma");

    // The same pixels with every stated σ doubled (variances x4): half the RMS.
    let doubled: Vec<_> = matches
        .iter()
        .map(|m| {
            let x4 = |c: &PointCovariance| {
                PointCovariance::new(c.xx() * 4.0, c.xy() * 4.0, c.yy() * 4.0).unwrap()
            };
            PixelMatch::new(m.pt1, x4(&m.noise1), m.pt2, x4(&m.noise2))
        })
        .collect();
    let found = estimate_essential(
        &rig,
        [0, 1],
        &doubled,
        &opts(5.0, 100),
        &RunControl::default(),
    )
    .unwrap();
    let half = found.inlier_rms;
    eprintln!("stated = 2 x true noise: inlier RMS {half:.4} sigma");
    assert!(
        (half / rms - 0.5).abs() < 0.02,
        "inlier RMS {half} sigma vs {rms} sigma"
    );
}

/// A 3 σ threshold keeps the inliers and rejects gross outliers at any focal
/// length, because the residual is in σ of the stated pixel noise.
#[test]
fn the_sigma_threshold_does_not_depend_on_the_focal_length() {
    let cam1_se3_rig = pose(0.02, -0.12, 0.03, Vec3::new(-0.5, 0.05, 0.1));
    let sigma = PointCovariance::isotropic(0.5).unwrap();
    for f in [400.0, 1600.0] {
        let rig = CalibratedRig::new(
            vec![distorted_pinhole(f), distorted_pinhole(f)],
            vec![Iso3::identity(), cam1_se3_rig],
            Uncertainty::default(),
        )
        .unwrap();
        let mut rng = Rng(0xF0C41 + f as u64);
        let world = scene(&mut rng, 600);
        let mut matches = noisy_matches(&rig, &world, [sigma, sigma], 1.0, &mut rng);
        let n_in = matches.len();
        assert!(n_in >= 450, "f = {f}: {n_in} inlier matches");
        // 25 % gross outliers: pixels anywhere in the 1280 x 960 image.
        let n_out = n_in / 3;
        for _ in 0..n_out {
            let mut px = || Pt2::new(rng.range(0.0, 1280.0), rng.range(0.0, 960.0));
            let (a, b) = (px(), px());
            matches.push(PixelMatch::new(a, sigma, b, sigma));
        }

        let found = estimate_essential(
            &rig,
            [0, 1],
            &matches,
            &opts(3.0, 100),
            &RunControl::default(),
        )
        .unwrap();
        let kept_in = found.inliers.iter().filter(|&&i| i < n_in).count();
        let kept_out = found.inliers.len() - kept_in;
        let (frac_in, frac_out) = (kept_in as f64 / n_in as f64, kept_out as f64 / n_out as f64);
        eprintln!(
            "f = {f}: kept {:.1} % of {n_in} inliers, {:.2} % of {n_out} outliers, \
             inlier RMS {:.3} sigma",
            100.0 * frac_in,
            100.0 * frac_out,
            found.inlier_rms
        );
        assert!(frac_in >= 0.95, "f = {f}: kept {frac_in} of the inliers");
        assert!(frac_out < 0.02, "f = {f}: kept {frac_out} of the outliers");
    }
}
