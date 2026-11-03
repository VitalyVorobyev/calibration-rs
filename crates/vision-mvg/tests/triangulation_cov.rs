//! The covariance of refined triangulated points, against Monte-Carlo.
//!
//! A stereo rig whose calibration has a negligible covariance isolates the
//! pixels' share: over many draws of the pixel noise (an anisotropic,
//! correlated stated covariance) the scatter of each refined point matches the
//! covariance `triangulate` reports. A rig without a covariance refines the
//! points and says why it has no covariance. On a wide-angle fisheye rig the
//! refined point is closer to the truth than the linear one.

#![allow(missing_docs)]

use nalgebra::{Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    ParameterCovariance, PointCovariance, ProjectionParams, Pt2, Pt3, SensorParams, Uncertainty,
    UncertaintyUnavailable, make_pinhole_camera, pinhole_camera_params,
};
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

    /// A pixel offset drawn from `cov` (Cholesky factor times normals).
    fn offset(&mut self, cov: &PointCovariance) -> Pt2 {
        let l00 = cov.xx().sqrt();
        let l10 = cov.xy() / l00;
        let l11 = (cov.yy() - l10 * l10).sqrt();
        let (a, b) = (self.normal(), self.normal());
        Pt2::new(l00 * a, l10 * a + l11 * b)
    }
}

fn right_camera() -> Iso3 {
    Iso3::from_parts(
        Translation3::new(-0.12, 0.0, 0.004),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.0, -0.05, 0.01)),
    )
}

fn pinhole() -> CameraParams {
    let k = FxFyCxCy {
        fx: 800.0,
        fy: 790.0,
        cx: 320.0,
        cy: 240.0,
    };
    pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()))
}

/// A calibration covariance so small it adds nothing: the labels of a
/// two-camera rig's intrinsics and extrinsics, at 1e-20.
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

fn stereo(camera: CameraParams, uncertainty: Uncertainty) -> CalibratedRig {
    CalibratedRig::new(
        vec![camera.clone(), camera],
        vec![Iso3::identity(), right_camera()],
        uncertainty,
    )
    .unwrap()
}

/// The two pixels of `p`, in the rig frame, plus the noise `draw` gives.
fn observe(
    rig: &CalibratedRig,
    p: &Pt3,
    noise: PointCovariance,
    mut draw: impl FnMut(&PointCovariance) -> Pt2,
) -> Vec<RigObservation> {
    (0..2)
        .map(|c| {
            let q = rig
                .model(c)
                .unwrap()
                .project_point(&(rig.cam_se3_rig()[c] * p))
                .unwrap();
            let e = draw(&noise);
            RigObservation::new(0, c, Pt2::new(q.x + e.x, q.y + e.y), noise)
        })
        .collect()
}

/// About twenty points 1 to 3 m ahead.
fn points(rng: &mut Rng, n: usize) -> Vec<Pt3> {
    (0..n)
        .map(|_| {
            let z = rng.range(1.0, 3.0);
            Pt3::new(rng.range(-0.3, 0.1) * z, rng.range(-0.25, 0.25) * z, z)
        })
        .collect()
}

/// Pooled over the points, the scatter of each coordinate over the predicted
/// standard deviation is 1.
#[test]
fn point_covariance_matches_the_scatter_of_pixel_noise() {
    let rig = stereo(pinhole(), negligible_covariance());
    let mut rng = Rng(7);
    let truth = points(&mut rng, 20);
    // Anisotropic, correlated, about 0.4 px.
    let noise = PointCovariance::new(0.16, 0.05, 0.09).unwrap();
    let trials = 400;

    let mut sum_sq = vec![[0.0_f64; 3]; truth.len()];
    let mut predicted = vec![[0.0_f64; 3]; truth.len()];
    let mut cross_max = 0.0_f64;
    for trial in 0..trials {
        let tracks: Vec<_> = truth
            .iter()
            .map(|p| observe(&rig, p, noise, |c| rng.offset(c)))
            .collect();
        let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
        for (i, p) in truth.iter().enumerate() {
            let e = found.points[i].point - p;
            for a in 0..3 {
                sum_sq[i][a] += e[a] * e[a] / trials as f64;
            }
        }
        if trial == 0 {
            for (i, v) in predicted.iter_mut().enumerate() {
                let c = found.point_covariance(i).expect("covariance available");
                for (a, slot) in v.iter_mut().enumerate() {
                    *slot = c[(a, a)];
                }
            }
            // The calibration adds nothing, so the points are independent.
            let (c01, c0, c1) = (
                found.cross_covariance(0, 1).unwrap(),
                found.point_covariance(0).unwrap(),
                found.point_covariance(1).unwrap(),
            );
            for a in 0..3 {
                for b in 0..3 {
                    cross_max = cross_max.max(c01[(a, b)].abs() / (c0[(a, a)] * c1[(b, b)]).sqrt());
                }
            }
        }
    }
    assert!(cross_max < 1e-6, "cross-correlation {cross_max}");
    let mut ratios = [0.0; 3];
    for a in 0..3 {
        let mean = (0..truth.len())
            .map(|i| sum_sq[i][a] / predicted[i][a])
            .sum::<f64>()
            / truth.len() as f64;
        ratios[a] = mean.sqrt();
    }
    eprintln!("pixel-only sigma ratio (x, y, z): {ratios:?}");
    for (a, r) in ratios.iter().enumerate() {
        assert!((0.9..=1.1).contains(r), "coordinate {a}: ratio {r}");
    }
}

/// A rig without a covariance refines the points and names what is missing.
#[test]
fn a_rig_without_covariance_still_refines() {
    let rig = stereo(pinhole(), Uncertainty::default());
    let mut rng = Rng(3);
    let truth = points(&mut rng, 4);
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let tracks: Vec<_> = truth
        .iter()
        .map(|p| observe(&rig, p, noise, |c| rng.offset(c)))
        .collect();
    let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
    assert!(matches!(
        found.uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::CalibrationWithoutCovariance
        }
    ));
    assert!(found.point_covariance(0).is_none());
    for (f, p) in found.points.iter().zip(&truth) {
        assert!((f.point - p).norm() < 0.1);
        assert!(f.in_front);
    }
}

/// A track behind its cameras keeps its linear point and has no covariance;
/// the other tracks are unaffected.
#[test]
fn an_unrefinable_track_has_no_covariance() {
    let rig = stereo(pinhole(), negligible_covariance());
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let good = observe(&rig, &Pt3::new(0.0, 0.0, 2.0), noise, |_| {
        Pt2::new(0.0, 0.0)
    });
    // Diverging rays: the intersection lies behind both cameras.
    let behind = vec![
        RigObservation::new(0, 0, Pt2::new(100.0, 240.0), noise),
        RigObservation::new(0, 1, Pt2::new(500.0, 240.0), noise),
    ];
    let found = triangulate(&rig, &[Iso3::identity()], &[behind, good]).unwrap();
    assert!(!found.points[0].in_front);
    assert!(found.point_covariance(0).is_none());
    assert!(found.points[1].in_front);
    assert!(found.point_covariance(1).is_some());
}

/// Refinement beats the linear estimate on a wide-angle rig: over many noise
/// draws the refined points are closer to the truth on average.
#[test]
fn refinement_improves_on_the_linear_estimate_for_a_fisheye() {
    let fisheye = CameraParams {
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
    };
    let rig = stereo(fisheye, Uncertainty::default());
    let mut rng = Rng(11);
    // Points up to ~100 degrees off the axis.
    let truth = [
        Pt3::new(1.5, 0.2, -0.3),
        Pt3::new(-1.2, 0.5, 0.2),
        Pt3::new(0.3, -0.2, 2.0),
        Pt3::new(1.0, 1.0, 1.0),
        Pt3::new(-0.8, -0.4, 1.5),
    ];
    let noise = PointCovariance::isotropic(0.8).unwrap();
    let (mut refined_err, mut linear_err) = (0.0, 0.0);
    for _ in 0..100 {
        let tracks: Vec<_> = truth
            .iter()
            .map(|p| observe(&rig, p, noise, |c| rng.offset(c)))
            .collect();
        let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
        // The linear estimate: the same views through the rig, unrefined.
        let linear = vision_mvg::triangulate_point_linear;
        for (i, p) in truth.iter().enumerate() {
            let poses: Vec<Iso3> = rig.cam_se3_rig().to_vec();
            let bearings: Vec<_> = tracks[i]
                .iter()
                .map(|o| {
                    rig.model(o.camera)
                        .unwrap()
                        .backproject_pixel(&o.pixel)
                        .direction
                        .normalize()
                })
                .collect();
            let lin = linear(&poses, &bearings).unwrap();
            refined_err += (found.points[i].point - p).norm();
            linear_err += (lin - p).norm();
        }
    }
    eprintln!(
        "fisheye mean error: refined {:.4} m, linear {:.4} m",
        refined_err / 500.0,
        linear_err / 500.0
    );
    assert!(refined_err < linear_err, "{refined_err} vs {linear_err}");
}
