use super::*;
use crate::metrology::PointSet;
use crate::test_support::{Lcg, fisheye_params, iso, pinhole_params};
use crate::types::Captures;
use nalgebra::{DVector, UnitQuaternion};
use vision_calibration_core::{
    CameraParams, DistortionParams, Iso3, LaserPlane, ProjectionParams, ScheimpflugParams,
    SensorParams, Vec3,
};

fn noise(sigma: f64) -> PointCovariance {
    PointCovariance::isotropic(sigma).unwrap()
}

/// A pinhole camera with Brown-Conrady distortion and a tilted sensor.
fn scheimpflug_params() -> CameraParams {
    CameraParams {
        sensor: SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.04,
                tilt_y: -0.03,
            },
        },
        ..pinhole_params(-0.12)
    }
}

/// The flat parameter vector of a camera, in the order of `CameraJacobian::labels`.
fn theta(camera: &CameraParams) -> Vec<f64> {
    camera.blocks().concat()
}

/// `camera` with its parameters replaced by the flat vector `theta`.
fn with_theta(camera: &CameraParams, theta: &[f64]) -> CameraParams {
    let blocks = camera.blocks();
    let (i, rest) = theta.split_at(blocks[0].len());
    let (p, rest) = rest.split_at(blocks[1].len());
    let (d, s) = rest.split_at(blocks[2].len());
    CameraParams {
        projection: ProjectionParams::from_values(camera.projection.kind(), p).unwrap(),
        distortion: DistortionParams::from_values(camera.distortion.kind(), d).unwrap(),
        sensor: SensorParams::from_values(camera.sensor.kind().unwrap(), s).unwrap(),
        intrinsics: vision_calibration_core::FxFyCxCy {
            fx: i[0],
            fy: i[1],
            cx: i[2],
            cy: i[3],
        },
    }
}

fn plane(n: [f64; 3], d: f64) -> LaserPlane {
    LaserPlane::from_coefficients(Vec3::new(n[0], n[1], n[2]), d).unwrap()
}

/// The pixel of a camera-frame point.
fn pixel(camera: &CameraParams, p: &Vec3) -> Pt2 {
    camera.build().unwrap().project_point_c(p).unwrap()
}

/// The camera-frame point of `plane` that projects along `(x, y)` at unit depth.
fn plane_point(plane: &LaserPlane, x: f64, y: f64) -> Vec3 {
    let n = plane.normal().into_inner();
    let z = (-plane.distance() - n.x * x - n.y * y) / n.z;
    Vec3::new(x, y, z)
}

/// The rig-frame point a camera-frame point is.
fn to_rig(cam_se3_rig: &Iso3, p: &Vec3) -> Pt3 {
    cam_se3_rig.inverse_transform_point(&Pt3::from(*p))
}

#[test]
fn ground_truth_points_are_recovered_through_every_model() {
    let cam1_se3_rig = iso(0.01, -0.06, 0.02, Vec3::new(-0.12, 0.01, 0.02));
    let laser = plane([0.2, -0.3, 0.9], -0.5);
    let rig = CalibratedRig::new(
        vec![pinhole_params(-0.1), scheimpflug_params()],
        vec![Iso3::identity(), cam1_se3_rig],
        Uncertainty::default(),
    )
    .unwrap()
    .with_laser_planes(vec![None, Some(laser.clone())])
    .unwrap();

    // The laser on camera 1 of a two-camera rig, tilted sensor, distortion.
    let mut observations = Vec::new();
    let mut truth = Vec::new();
    for (x, y) in [(-0.1, 0.05), (0.0, -0.08), (0.12, 0.1)] {
        let p = plane_point(&laser, x, y);
        observations.push(LaserObservation::new(
            0,
            1,
            pixel(&rig.cameras()[1], &p),
            noise(0.1),
        ));
        truth.push(to_rig(&cam1_se3_rig, &p));
    }
    let found = triangulate_laser(&rig, Captures::Rig, &observations).unwrap();
    for (f, t) in found.points.iter().zip(&truth) {
        assert!((f.point - t).norm() < 1e-9, "{:?} vs {t:?}", f.point);
        assert_eq!(f.camera, 1);
        let b = (cam1_se3_rig * f.point).coords.normalize();
        let expected = laser.normal().dot(&b).abs().asin().to_degrees();
        assert!((f.incidence_deg - expected).abs() < 1e-9);
    }
    // No covariance in the rig: no uncertainty, for the stated reason.
    assert_eq!(
        found.uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::CalibrationWithoutCovariance
        }
    );
    assert!(found.point_covariance(0).is_none());

    // A pinhole, undistorted-to-the-bearing: the plain case.
    let single = CalibratedRig::single(pinhole_params(0.0), Uncertainty::default())
        .unwrap()
        .with_laser_planes(vec![Some(laser.clone())])
        .unwrap();
    let p = plane_point(&laser, 0.07, -0.02);
    let one = triangulate_laser(
        &single,
        Captures::Rig,
        &[LaserObservation::new(
            0,
            0,
            pixel(&single.cameras()[0], &p),
            noise(0.1),
        )],
    )
    .unwrap();
    assert!((one.points[0].point.coords - p).norm() < 1e-9);

    // A fisheye ray more than 90 degrees off the axis (Kannala-Brandt).
    let theta = 105.0_f64.to_radians();
    let direction = Vec3::new(theta.sin() * 0.8, theta.sin() * 0.6, theta.cos());
    let p = 0.6 * direction;
    assert!(p.z < 0.0);
    let n = (0.8 * direction + Vec3::new(0.0, 0.5, 0.0)).normalize();
    let fisheye_plane = LaserPlane::new(n, -n.dot(&p)).unwrap();
    let fisheye = CalibratedRig::single(fisheye_params(), Uncertainty::default())
        .unwrap()
        .with_laser_planes(vec![Some(fisheye_plane)])
        .unwrap();
    let found = triangulate_laser(
        &fisheye,
        Captures::Rig,
        &[LaserObservation::new(
            0,
            0,
            pixel(&fisheye.cameras()[0], &p),
            noise(0.1),
        )],
    )
    .unwrap();
    assert!(
        (found.points[0].point.coords - p).norm() < 1e-9,
        "{:?} vs {p:?}",
        found.points[0].point
    );
}

// ---------------------------------------------------------------------------
// The synthetic calibration the Jacobian and Monte-Carlo tests share
// ---------------------------------------------------------------------------

/// Camera 1's labels in the covariance, in the order of the coordinates `z`:
/// its parameters, the plane's tangent coordinates and distance, the pose.
struct Calibration {
    cam1: CameraParams,
    cam1_se3_rig: Iso3,
    plane: LaserPlane,
    /// The labels of the covariance, in the order of its rows.
    labels: Vec<String>,
    /// `Σ_z` over `(θ, a₁, a₂, d, ω, v)`.
    sigma_z: DMatrix<f64>,
    /// `M`, `c = M z`: the plane's tangent coordinates onto `(nx, ny, nz)`.
    embed: DMatrix<f64>,
    /// The number of camera parameters.
    n_theta: usize,
}

impl Calibration {
    /// `scale_*` multiply the standard deviations; `cross` switches the
    /// camera-plane correlation (`fx`-`d`, `cx`-`n`).
    fn new(cross: bool) -> Self {
        let cam1 = scheimpflug_params();
        let cam1_se3_rig = iso(0.01, -0.06, 0.02, Vec3::new(-0.12, 0.01, 0.02));
        let plane = plane([0.5, -0.4, 0.77], -0.5);
        let labels_theta = theta_labels(&cam1);
        let n_theta = labels_theta.len();
        let n_z = n_theta + 3 + 6;

        // Standard deviations: 1.1 % of the focal lengths, 4 px on the
        // principal point, a few 1e-3 on the distortion, 1e-3 rad on the
        // tilt, 2 mrad and 1.5 mm on the plane, 1 mrad and 1 mm on the pose.
        let mut std = vec![0.0; n_z];
        for (i, label) in labels_theta.iter().enumerate() {
            std[i] = match label.as_str() {
                "fx" | "fy" => 9.0,
                "cx" | "cy" => 4.0,
                "k1" => 4e-3,
                "k2" => 8e-3,
                "k3" => 1e-9,
                "p1" | "p2" => 3e-4,
                _ => 1e-3, // tilt_x, tilt_y
            };
        }
        std[n_theta] = 2e-3; // a1
        std[n_theta + 1] = 2e-3; // a2
        std[n_theta + 2] = 1.5e-3; // d
        for i in 0..3 {
            std[n_theta + 3 + i] = 1e-3; // ω
            std[n_theta + 6 + i] = 1e-3; // v
        }
        // Correlations, as a calibration from shared views has them.
        let at = |name: &str| labels_theta.iter().position(|l| l == name).unwrap();
        let mut corr = DMatrix::<f64>::identity(n_z, n_z);
        let mut tie = |a: usize, b: usize, r: f64| {
            corr[(a, b)] = r;
            corr[(b, a)] = r;
        };
        tie(at("fx"), at("fy"), 0.5);
        tie(at("k1"), at("k2"), -0.7);
        tie(at("cx"), n_theta + 6, 0.4); // cx - v_x
        tie(n_theta, n_theta + 4, 0.3); // a1 - ω_y
        if cross {
            tie(at("fx"), n_theta + 2, 0.6); // fx - d
            tie(at("cy"), n_theta + 1, -0.6); // cy - a2
            tie(at("fy"), n_theta + 2, 0.5); // fy - d
            tie(at("cx"), n_theta, 0.6); // cx - a1
        }
        let sigma_z = DMatrix::from_fn(n_z, n_z, |a, b| std[a] * std[b] * corr[(a, b)]);
        assert!(
            sigma_z.clone().cholesky().is_some(),
            "Σ_z is positive definite"
        );

        // The plane's tangent basis: Σ_c has rank 2 on the normal.
        let n = plane.normal().into_inner();
        let helper = if n.x.abs() < 0.9 {
            Vec3::x()
        } else {
            Vec3::y()
        };
        let e1 = n.cross(&helper).normalize();
        let e2 = n.cross(&e1);
        let n_c = n_theta + 4 + 6;
        let mut embed = DMatrix::<f64>::zeros(n_c, n_z);
        for i in 0..n_theta {
            embed[(i, i)] = 1.0;
        }
        for r in 0..3 {
            embed[(n_theta + r, n_theta)] = e1[r];
            embed[(n_theta + r, n_theta + 1)] = e2[r];
        }
        embed[(n_theta + 3, n_theta + 2)] = 1.0;
        for i in 0..6 {
            embed[(n_theta + 4 + i, n_theta + 3 + i)] = 1.0;
        }
        let mut labels: Vec<String> = labels_theta.iter().map(|l| format!("cam1.{l}")).collect();
        labels.extend(CalibratedRig::plane_labels(1, true));
        labels.extend(
            ["rx", "ry", "rz", "tx", "ty", "tz"]
                .iter()
                .map(|c| format!("cam1.cam_se3_rig.{c}")),
        );
        Self {
            cam1,
            cam1_se3_rig,
            plane,
            labels,
            sigma_z,
            embed,
            n_theta,
        }
    }

    /// The covariance `M Σ_z Mᵀ` with `keep` applied to its entries.
    fn covariance(&self, keep: impl Fn(usize, usize) -> bool) -> ParameterCovariance {
        let sigma = &self.embed * &self.sigma_z * self.embed.transpose();
        let n = self.labels.len();
        ParameterCovariance {
            labels: self.labels.clone(),
            covariance: (0..n)
                .map(|r| {
                    (0..n)
                        .map(|c| if keep(r, c) { sigma[(r, c)] } else { 0.0 })
                        .collect()
                })
                .collect(),
            views: Vec::new(),
            fixed: Vec::new(),
            noise: Vec::new(),
            num_residuals: 100,
            dof: 50,
            view_scatter: None,
            second_order: None,
        }
    }

    fn rig_with(
        &self,
        cam1: CameraParams,
        cam1_se3_rig: Iso3,
        plane: LaserPlane,
        uncertainty: Uncertainty,
    ) -> CalibratedRig {
        CalibratedRig::new(
            vec![pinhole_params(-0.1), cam1],
            vec![Iso3::identity(), cam1_se3_rig],
            uncertainty,
        )
        .unwrap()
        .with_laser_planes(vec![None, Some(plane)])
        .unwrap()
    }

    fn rig(&self, covariance: ParameterCovariance) -> CalibratedRig {
        self.rig_with(
            self.cam1.clone(),
            self.cam1_se3_rig,
            self.plane.clone(),
            Uncertainty::Available(covariance),
        )
    }

    /// Pixels of laser points at 0.4-0.6 m.
    fn observations(&self, sigma: f64) -> Vec<LaserObservation> {
        [(-0.12, 0.06), (0.02, -0.09), (0.13, 0.08)]
            .iter()
            .map(|&(x, y)| {
                let p = plane_point(&self.plane, x, y);
                LaserObservation::new(0, 1, pixel(&self.cam1, &p), noise(sigma))
            })
            .collect()
    }
}

fn theta_labels(camera: &CameraParams) -> Vec<String> {
    CameraJacobian::new(camera)
        .unwrap()
        .labels()
        .iter()
        .map(|l| l.to_string())
        .collect()
}

/// The rig-frame points of a call.
fn points(rig: &CalibratedRig, observations: &[LaserObservation]) -> Vec<Vec3> {
    triangulate_laser(rig, Captures::Rig, observations)
        .unwrap()
        .points
        .iter()
        .map(|p| p.point.coords)
        .collect()
}

#[test]
fn jacobians_match_central_differences() {
    let cal = Calibration::new(true);
    let cov = cal.covariance(|_, _| true);
    let rig = cal.rig(cov.clone());
    let observations = cal.observations(0.2);

    // The analytic linearization.
    let poses = Captures::Rig.resolve(&rig).unwrap();
    let Located { cameras, found } = locate(&rig, &poses, &observations).unwrap();
    let lin = linearize(&rig, &cov, &poses, &observations, &cameras, &found)
        .unwrap()
        .expect("covariance lists the camera");
    assert_eq!(lin.labels, cal.labels);

    let theta0 = theta(&cal.cam1);
    let n = cal.plane.normal().into_inner();
    let helper = Vec3::x();
    let e1 = n.cross(&helper).normalize();
    let e2 = n.cross(&e1);

    let mut worst: f64 = 0.0;
    let mut check =
        |name: &str, column: DVector<f64>, plus: Vec<Vec3>, minus: Vec<Vec3>, h: f64| {
            let numeric: Vec<f64> = plus
                .iter()
                .zip(&minus)
                .flat_map(|(p, m)| ((p - m) / (2.0 * h)).iter().copied().collect::<Vec<_>>())
                .collect();
            let numeric = DVector::from_vec(numeric);
            let error = (&column - &numeric).norm() / column.norm().max(1e-12);
            worst = worst.max(error);
            assert!(error < 1e-6, "{name}: relative error {error:e}");
        };
    let column = |label: &str| -> DVector<f64> {
        let c = lin.labels.iter().position(|l| l == label).unwrap();
        lin.j_c.column(c).into_owned()
    };

    // Camera parameters.
    for (j, label) in theta_labels(&cal.cam1).iter().enumerate() {
        // k3 and the like move the pixel by 1e-5 px per 1e-6: a larger step
        // keeps the difference above the rounding of the pixels.
        let h = 1e-4 * theta0[j].abs().max(1.0);
        let (mut up, mut down) = (theta0.clone(), theta0.clone());
        up[j] += h;
        down[j] -= h;
        let run = |t: &[f64]| {
            let rig = cal.rig_with(
                with_theta(&cal.cam1, t),
                cal.cam1_se3_rig,
                cal.plane.clone(),
                Uncertainty::Available(cov.clone()),
            );
            points(&rig, &observations)
        };
        check(
            label,
            column(&format!("cam1.{label}")),
            run(&up),
            run(&down),
            h,
        );
    }
    // The plane: its distance, and the normal along two tangent directions
    // (renormalised, as a perturbed plane is); the radial direction is not in
    // the covariance.
    let h = 1e-6;
    let run_plane = |n: Vec3, d: f64| {
        let rig = cal.rig_with(
            cal.cam1.clone(),
            cal.cam1_se3_rig,
            LaserPlane::new(n.normalize(), d).unwrap(),
            Uncertainty::Available(cov.clone()),
        );
        points(&rig, &observations)
    };
    check(
        "plane.d",
        column("cam1.plane.d"),
        run_plane(n, cal.plane.distance() + h),
        run_plane(n, cal.plane.distance() - h),
        h,
    );
    let (cx, cy, cz) = (
        column("cam1.plane.nx"),
        column("cam1.plane.ny"),
        column("cam1.plane.nz"),
    );
    for (name, e) in [("n·e1", e1), ("n·e2", e2)] {
        let directional = &cx * e.x + &cy * e.y + &cz * e.z;
        check(
            name,
            directional,
            run_plane(n + h * e, cal.plane.distance()),
            run_plane(n - h * e, cal.plane.distance()),
            h,
        );
    }
    // The extrinsic: cam_se3_rig · Exp(ω, v), v the translation.
    for (k, axis) in ["rx", "ry", "rz", "tx", "ty", "tz"].iter().enumerate() {
        let run = |s: f64| {
            let mut xi = [0.0; 6];
            xi[k] = s;
            let delta = Iso3::from_parts(
                nalgebra::Translation3::new(xi[3], xi[4], xi[5]),
                UnitQuaternion::from_scaled_axis(Vec3::new(xi[0], xi[1], xi[2])),
            );
            let rig = cal.rig_with(
                cal.cam1.clone(),
                cal.cam1_se3_rig * delta,
                cal.plane.clone(),
                Uncertainty::Available(cov.clone()),
            );
            points(&rig, &observations)
        };
        check(
            axis,
            column(&format!("cam1.cam_se3_rig.{axis}")),
            run(1e-6),
            run(-1e-6),
            1e-6,
        );
    }
    // The pixels: each point depends on its own pixel only.
    for (i, o) in observations.iter().enumerate() {
        for axis in 0..2 {
            let h = 1e-4;
            let run = |s: f64| {
                let mut shifted = observations.clone();
                shifted[i].pixel[axis] = o.pixel[axis] + s;
                points(&rig, &shifted)
            };
            let mut col = DVector::zeros(3 * observations.len());
            col.rows_mut(3 * i, 3).copy_from(&lin.j_u[i].column(axis));
            check(&format!("pixel {i}.{axis}"), col, run(h), run(-h), h);
        }
    }
    eprintln!("laser Jacobians vs central differences: worst relative error {worst:.2e}");
}

// ---------------------------------------------------------------------------
// Monte Carlo in parameter space
// ---------------------------------------------------------------------------

/// A deterministic standard normal draw.
fn gauss(rng: &mut Lcg) -> f64 {
    let u1 = (rng.next() + 1.0) / 2.0 + 1e-12;
    let u2 = (rng.next() + 1.0) / 2.0;
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The sample covariance of the 3N-vectors in `draws`.
fn sample_covariance(draws: &[DVector<f64>]) -> DMatrix<f64> {
    let n = draws.len() as f64;
    let mean = draws
        .iter()
        .fold(DVector::zeros(draws[0].len()), |a, d| a + d)
        / n;
    let mut c = DMatrix::zeros(mean.len(), mean.len());
    for d in draws {
        let e = d - &mean;
        c += &e * e.transpose();
    }
    c / (n - 1.0)
}

fn flat(points: &[Vec3]) -> DVector<f64> {
    DVector::from_iterator(
        3 * points.len(),
        points.iter().flat_map(|p| p.iter().copied()),
    )
}

fn frobenius_error(sample: &DMatrix<f64>, stated: &DMatrix<f64>) -> f64 {
    (sample - stated).norm() / stated.norm()
}

/// The points over `trials` draws of the calibration (`perturb`: fresh
/// parameters from `Σ_z`) and of the pixels (σ `pixel_sigma`).
fn monte_carlo(
    cal: &Calibration,
    trials: usize,
    seed: u64,
    perturb_calibration: bool,
    pixel_sigma: f64,
) -> Vec<DVector<f64>> {
    let observations = cal.observations(pixel_sigma);
    let chol = cal.sigma_z.clone().cholesky().unwrap().l();
    let n = cal.plane.normal().into_inner();
    let mut rng = Lcg(seed);
    let rig = cal.rig(cal.covariance(|_, _| true));
    (0..trials)
        .map(|_| {
            let (cam1, pose, laser) = if perturb_calibration {
                let g = DVector::from_fn(chol.nrows(), |_, _| gauss(&mut rng));
                let z = &chol * g;
                let c = &cal.embed * z;
                let theta1: Vec<f64> = theta(&cal.cam1)
                    .iter()
                    .zip(c.iter())
                    .map(|(t, d)| t + d)
                    .collect();
                let k = cal.n_theta;
                let n1 = (n + Vec3::new(c[k], c[k + 1], c[k + 2])).normalize();
                let laser = LaserPlane::new(n1, cal.plane.distance() + c[k + 3]).unwrap();
                let delta = Iso3::from_parts(
                    nalgebra::Translation3::new(c[k + 7], c[k + 8], c[k + 9]),
                    UnitQuaternion::from_scaled_axis(Vec3::new(c[k + 4], c[k + 5], c[k + 6])),
                );
                (
                    with_theta(&cal.cam1, &theta1),
                    cal.cam1_se3_rig * delta,
                    laser,
                )
            } else {
                (cal.cam1.clone(), cal.cam1_se3_rig, cal.plane.clone())
            };
            let perturbed = cal.rig_with(cam1, pose, laser, rig.uncertainty().clone());
            let noisy: Vec<LaserObservation> = observations
                .iter()
                .map(|o| {
                    let mut o = *o;
                    o.pixel.x += pixel_sigma * gauss(&mut rng);
                    o.pixel.y += pixel_sigma * gauss(&mut rng);
                    o
                })
                .collect();
            flat(&points(&perturbed, &noisy))
        })
        .collect()
}

fn stated(
    rig: &CalibratedRig,
    observations: &[LaserObservation],
) -> (DMatrix<f64>, LaserTriangulation) {
    let found = triangulate_laser(rig, Captures::Rig, observations).unwrap();
    let c = found.uncertainty.covariance().expect("covariance");
    let n = c.labels.len();
    (DMatrix::from_fn(n, n, |r, s| c.covariance[r][s]), found)
}

const TRIALS: usize = 4000;
const PIXEL_SIGMA: f64 = 0.1;

#[test]
fn the_stated_covariance_matches_the_sampled_one_in_parameter_space() {
    let cal = Calibration::new(true);
    let rig = cal.rig(cal.covariance(|_, _| true));
    let observations = cal.observations(PIXEL_SIGMA);
    let (sigma, found) = stated(&rig, &observations);
    let draws = monte_carlo(&cal, TRIALS, 20_241_103, true, PIXEL_SIGMA);
    let sample = sample_covariance(&draws);

    let block = |m: &DMatrix<f64>, i: usize, j: usize| m.view((3 * i, 3 * j), (3, 3)).into_owned();
    let mut worst: f64 = 0.0;
    for i in 0..observations.len() {
        let error = frobenius_error(&block(&sample, i, i), &block(&sigma, i, i));
        eprintln!("point {i}: sample vs stated, relative Frobenius error {error:.3}");
        worst = worst.max(error);
        assert!(error <= 0.08, "point {i}: {error:.3}");
    }
    // The calibration correlates the points: one cross block.
    let cross = frobenius_error(&block(&sample, 0, 2), &block(&sigma, 0, 2));
    eprintln!("cross block (0, 2): relative Frobenius error {cross:.3}");
    assert!(cross <= 0.1, "cross block: {cross:.3}");
    // And the accessors read the same matrices.
    let direct = found.cross_covariance(0, 2).unwrap();
    assert!((direct - Mat3::from_iterator(block(&sigma, 0, 2).iter().copied())).norm() < 1e-18);
    assert!(PointSet::from_laser_triangulation(&found).is_ok());
    eprintln!("worst diagonal block {worst:.3}");

    // Dropping the camera x plane cross terms moves the stated σ by more than
    // the gate: the correlation is part of the answer.
    let n_theta = cal.n_theta;
    let is_theta = |i: usize| i < n_theta;
    let is_plane = |i: usize| (n_theta..n_theta + 4).contains(&i);
    let split =
        cal.covariance(|r, c| !((is_theta(r) && is_plane(c)) || (is_plane(r) && is_theta(c))));
    let (without, _) = stated(&cal.rig(split), &observations);
    let moved = (0..observations.len())
        .map(|i| frobenius_error(&block(&without, i, i), &block(&sigma, i, i)))
        .fold(0.0_f64, f64::max);
    eprintln!("without the camera x plane cross terms the stated σ moves by {moved:.3}");
    assert!(
        moved > 0.2,
        "cross terms move the covariance by only {moved:.3}"
    );
}

#[test]
fn pixel_noise_alone_matches_its_sample() {
    let cal = Calibration::new(true);
    // A calibration known to a nanometre: the stated σ is the pixels'.
    let tiny = cal.covariance(|r, c| r == c);
    let mut tiny = tiny;
    for (i, row) in tiny.covariance.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = if i == j { 1e-24 } else { 0.0 };
        }
    }
    let observations = cal.observations(PIXEL_SIGMA);
    let (sigma, _) = stated(&cal.rig(tiny), &observations);
    let sample = sample_covariance(&monte_carlo(&cal, TRIALS, 77_001, false, PIXEL_SIGMA));
    for i in 0..observations.len() {
        let b = |m: &DMatrix<f64>| m.view((3 * i, 3 * i), (3, 3)).into_owned();
        let error = frobenius_error(&b(&sample), &b(&sigma));
        eprintln!("pixel only, point {i}: relative Frobenius error {error:.3}");
        assert!(error <= 0.08, "point {i}: {error:.3}");
        // Points of different pixels are independent here.
        assert!(sigma.view((3 * i, 0), (3, 3 * i)).norm() < 1e-9);
    }
}

#[test]
fn the_uncertainty_says_why_it_is_unavailable() {
    let cal = Calibration::new(true);
    let observations = cal.observations(PIXEL_SIGMA);
    let unavailable = |rig: &CalibratedRig| {
        triangulate_laser(rig, Captures::Rig, &observations)
            .unwrap()
            .uncertainty
            == Uncertainty::Unavailable {
                reason: UncertaintyUnavailable::CalibrationWithoutCovariance,
            }
    };
    // No covariance at all.
    assert!(unavailable(&cal.rig_with(
        cal.cam1.clone(),
        cal.cam1_se3_rig,
        cal.plane.clone(),
        Uncertainty::default(),
    )));
    // Cameras held fixed: the covariance has the planes and extrinsics only.
    let n_theta = cal.n_theta;
    let mut covariance = cal.covariance(|_, _| true);
    covariance.labels.drain(..n_theta);
    covariance.covariance.drain(..n_theta);
    for row in &mut covariance.covariance {
        row.drain(..n_theta);
    }
    assert!(unavailable(&cal.rig(covariance)));
    // A plane the calibration held fixed is exact: the covariance has the
    // rest, and the points' σ is smaller by the plane's share.
    let mut held = cal.covariance(|_, _| true);
    let first = n_theta;
    held.labels.drain(first..first + 4);
    held.covariance.drain(first..first + 4);
    for row in &mut held.covariance {
        row.drain(first..first + 4);
    }
    let with_plane = triangulate_laser(
        &cal.rig(cal.covariance(|_, _| true)),
        Captures::Rig,
        &observations,
    )
    .unwrap();
    let without_plane = triangulate_laser(&cal.rig(held), Captures::Rig, &observations).unwrap();
    let (a, b) = (
        with_plane.point_covariance(0).unwrap().trace(),
        without_plane.point_covariance(0).unwrap().trace(),
    );
    assert!(b > 0.0 && b < a, "{b} vs {a}");
    // Part of a camera's extrinsic is an error, not a guess.
    let mut partial = cal.covariance(|_, _| true);
    let at = partial
        .labels
        .iter()
        .position(|l| l == "cam1.cam_se3_rig.tz")
        .unwrap();
    partial.labels.remove(at);
    partial.covariance.remove(at);
    for row in &mut partial.covariance {
        row.remove(at);
    }
    let rig = cal.rig(partial);
    let err = triangulate_laser(&rig, Captures::Rig, &observations).unwrap_err();
    assert!(err.to_string().contains("whole or held"), "{err}");
}

#[test]
fn errors_name_the_observation() {
    let laser = plane([0.0, 0.0, 1.0], -0.5);
    let rig = CalibratedRig::new(
        vec![pinhole_params(0.0), pinhole_params(0.0)],
        vec![
            Iso3::identity(),
            iso(0.0, 0.1, 0.0, Vec3::new(-0.1, 0.0, 0.0)),
        ],
        Uncertainty::default(),
    )
    .unwrap()
    .with_laser_planes(vec![None, Some(laser)])
    .unwrap();
    let centre = Pt2::new(640.0, 480.0);
    let message = |observations: &[LaserObservation]| {
        triangulate_laser(&rig, Captures::Rig, observations)
            .unwrap_err()
            .to_string()
    };
    let good = LaserObservation::new(0, 1, centre, noise(0.1));

    // A camera out of range, and a camera without a plane.
    let m = message(&[good, LaserObservation::new(0, 5, centre, noise(0.1))]);
    assert!(m.contains("observation 1") && m.contains("camera 5"), "{m}");
    let m = message(&[LaserObservation::new(0, 0, centre, noise(0.1))]);
    assert!(
        m.contains("observation 0") && m.contains("no laser plane"),
        "{m}"
    );

    // A pixel without a ray.
    let m = message(&[LaserObservation::new(
        0,
        1,
        Pt2::new(f64::NAN, 0.0),
        noise(0.1),
    )]);
    assert!(m.contains("observation 0"), "{m}");

    // A ray parallel to the plane: a plane through the camera's axis.
    let edge = CalibratedRig::single(pinhole_params(0.0), Uncertainty::default())
        .unwrap()
        .with_laser_planes(vec![Some(plane([1.0, 0.0, 0.0], 0.0))])
        .unwrap();
    let err = triangulate_laser(
        &edge,
        Captures::Rig,
        &[LaserObservation::new(0, 0, centre, noise(0.1))],
    )
    .unwrap_err();
    assert!(matches!(err, MvgError::Degenerate { .. }), "{err}");

    // A plane behind the camera.
    let behind = CalibratedRig::single(pinhole_params(0.0), Uncertainty::default())
        .unwrap()
        .with_laser_planes(vec![Some(plane([0.0, 0.0, 1.0], 0.5))])
        .unwrap();
    let err = triangulate_laser(
        &behind,
        Captures::Rig,
        &[LaserObservation::new(0, 0, centre, noise(0.1))],
    )
    .unwrap_err();
    assert!(err.to_string().contains("behind the camera"), "{err}");

    // No observations is no points, not an error.
    let none = triangulate_laser(&rig, Captures::Rig, &[]).unwrap();
    assert!(none.points.is_empty());
}

// ---------------------------------------------------------------------------
// On a robot: the hand-eye and the robot pose
// ---------------------------------------------------------------------------

use vision_calibration_core::{Handeye, RobotFrame, RobotPose};

/// The calibration above with a hand-eye, its covariance correlated with
/// camera 1's focal length and the plane's distance.
struct OnRobot {
    cal: Calibration,
    handeye: Handeye,
    /// The full covariance: the calibration's labels, then the hand-eye's.
    covariance: ParameterCovariance,
    /// Σ over (z, ε_h) with the cross terms, for drawing.
    sigma: DMatrix<f64>,
    /// The robot's pose at each of two captures, and its σ (rad, m).
    robot: [Iso3; 2],
    robot_sigma: (f64, f64),
}

impl OnRobot {
    fn new(eye_in_hand: bool) -> Self {
        let cal = Calibration::new(true);
        let pose = iso(0.3, -0.2, 0.5, Vec3::new(0.05, -0.03, 0.12));
        let handeye = if eye_in_hand {
            Handeye::EyeInHand {
                gripper_se3_rig: pose,
            }
        } else {
            Handeye::EyeToHand {
                rig_se3_base: iso(0.2, 0.4, -0.3, Vec3::new(0.2, -0.1, 1.1)),
            }
        };
        let n_z = cal.sigma_z.nrows();
        let n = n_z + 6;
        // 2 mrad and 1.5 mm on the hand-eye, correlated with fx and d.
        let mut sigma = DMatrix::zeros(n, n);
        sigma.view_mut((0, 0), (n_z, n_z)).copy_from(&cal.sigma_z);
        let std_h = [2e-3, 2e-3, 2e-3, 1.5e-3, 1.5e-3, 1.5e-3];
        for i in 0..6 {
            sigma[(n_z + i, n_z + i)] = std_h[i] * std_h[i];
        }
        let fx = theta_labels(&cal.cam1)
            .iter()
            .position(|l| l == "fx")
            .unwrap();
        let d = cal.n_theta + 2;
        let mut tie = |a: usize, b: usize, r: f64| {
            let v = r * (sigma[(a, a)] * sigma[(b, b)]).sqrt();
            sigma[(a, b)] = v;
            sigma[(b, a)] = v;
        };
        tie(fx, n_z + 5, 0.5); // fx - t_z of the hand-eye
        tie(d, n_z + 4, -0.4); // d - t_y
        tie(cal.n_theta + 3, n_z + 1, 0.3); // ω_x of the extrinsic - r_y
        assert!(sigma.clone().cholesky().is_some());

        // The covariance over the labels: c = blockdiag(M, I) (z, ε_h).
        let n_c = cal.embed.nrows();
        let mut embed = DMatrix::zeros(n_c + 6, n);
        embed.view_mut((0, 0), (n_c, n_z)).copy_from(&cal.embed);
        for i in 0..6 {
            embed[(n_c + i, n_z + i)] = 1.0;
        }
        let full = &embed * &sigma * embed.transpose();
        let mut labels = cal.labels.clone();
        let name = if eye_in_hand {
            "gripper_se3_rig"
        } else {
            "rig_se3_base"
        };
        labels.extend(["rx", "ry", "rz", "tx", "ty", "tz"].map(|c| format!("{name}.{c}")));
        let m = labels.len();
        let covariance = ParameterCovariance {
            labels,
            covariance: (0..m)
                .map(|r| (0..m).map(|c| full[(r, c)]).collect())
                .collect(),
            views: Vec::new(),
            fixed: Vec::new(),
            noise: Vec::new(),
            num_residuals: 100,
            dof: 50,
            view_scatter: None,
            second_order: None,
        };
        Self {
            cal,
            handeye,
            covariance,
            sigma,
            robot: [
                iso(0.1, 0.6, -0.2, Vec3::new(0.4, -0.1, 0.5)),
                iso(-0.3, 0.2, 0.9, Vec3::new(0.2, 0.3, 0.6)),
            ],
            robot_sigma: (1e-3, 0.8e-3),
        }
    }

    fn rig_with(
        &self,
        cam1: CameraParams,
        cam1_se3_rig: Iso3,
        plane: LaserPlane,
        handeye: Handeye,
    ) -> CalibratedRig {
        self.cal
            .rig_with(
                cam1,
                cam1_se3_rig,
                plane,
                Uncertainty::Available(self.covariance.clone()),
            )
            .with_handeye(handeye)
            .unwrap()
    }

    fn rig(&self) -> CalibratedRig {
        self.rig_with(
            self.cal.cam1.clone(),
            self.cal.cam1_se3_rig,
            self.cal.plane.clone(),
            self.handeye,
        )
    }

    fn robot_poses(&self, robot: &[Iso3; 2]) -> Vec<RobotPose> {
        robot
            .iter()
            .map(|p| RobotPose::isotropic(*p, self.robot_sigma.0, self.robot_sigma.1).unwrap())
            .collect()
    }

    /// The camera's three laser pixels at each of the two captures.
    fn observations(&self) -> Vec<LaserObservation> {
        let mut all = self.cal.observations(PIXEL_SIGMA);
        let mut second = all.clone();
        for o in &mut second {
            o.capture = 1;
        }
        all.extend(second);
        all
    }
}

/// `pose · Exp(ξ)` for one tangent coordinate.
fn right(pose: &Iso3, k: usize, s: f64) -> Iso3 {
    let mut xi = [0.0; 6];
    xi[k] = s;
    pose * Iso3::from_parts(
        nalgebra::Translation3::new(xi[3], xi[4], xi[5]),
        UnitQuaternion::from_scaled_axis(Vec3::new(xi[0], xi[1], xi[2])),
    )
}

fn perturbed_handeye(handeye: &Handeye, k: usize, s: f64) -> Handeye {
    match handeye {
        Handeye::EyeInHand { gripper_se3_rig } => Handeye::EyeInHand {
            gripper_se3_rig: right(gripper_se3_rig, k, s),
        },
        Handeye::EyeToHand { rig_se3_base } => Handeye::EyeToHand {
            rig_se3_base: right(rig_se3_base, k, s),
        },
    }
}

/// The hand-eye's and the robot's columns, and a camera parameter's and the
/// extrinsic's in the robot frame, against central differences: both modes,
/// both frames.
#[test]
fn robot_frame_jacobians_match_central_differences() {
    let mut worst: f64 = 0.0;
    for eye_in_hand in [true, false] {
        let robot = OnRobot::new(eye_in_hand);
        let rig = robot.rig();
        let poses = robot.robot_poses(&robot.robot);
        for frame in [RobotFrame::Gripper, RobotFrame::Base] {
            let moving = frame != robot.handeye.mount_frame();
            let all = robot.observations();
            let observations = if moving { &all[..] } else { &all[..3] };
            let captures = |poses: &[RobotPose]| -> Vec<RobotPose> { poses.to_vec() };
            let points_at = |rig: &CalibratedRig, poses: &[RobotPose]| -> Vec<Vec3> {
                let poses = captures(poses);
                let c = if moving {
                    Captures::Robot {
                        frame,
                        robot: &poses,
                    }
                } else {
                    Captures::Handeye(frame)
                };
                triangulate_laser(rig, c, observations)
                    .unwrap()
                    .points
                    .iter()
                    .map(|p| p.point.coords)
                    .collect()
            };
            let c = if moving {
                Captures::Robot {
                    frame,
                    robot: &poses,
                }
            } else {
                Captures::Handeye(frame)
            };
            let resolved = c.resolve(&rig).unwrap();
            let Located { cameras, found } = locate(&rig, &resolved, observations).unwrap();
            let lin = linearize(
                &rig,
                &robot.covariance,
                &resolved,
                observations,
                &cameras,
                &found,
            )
            .unwrap()
            .unwrap();
            let column = |label: &str| -> DVector<f64> {
                let c = lin
                    .labels
                    .iter()
                    .position(|l| l == label)
                    .unwrap_or_else(|| panic!("{label} not in {:?}", lin.labels));
                lin.j_c.column(c).into_owned()
            };
            let mut check =
                |name: &str, col: DVector<f64>, plus: Vec<Vec3>, minus: Vec<Vec3>, h: f64| {
                    let numeric = (flat(&plus) - flat(&minus)) / (2.0 * h);
                    let error = (&col - &numeric).norm() / col.norm().max(1e-12);
                    worst = worst.max(error);
                    assert!(
                        error < 1e-6,
                        "{eye_in_hand} {frame:?} {name}: relative error {error:e}"
                    );
                };

            let labels = rig.handeye_labels().unwrap();
            let h = 1e-6;
            for (k, label) in labels.iter().enumerate() {
                let run = |s: f64| {
                    let rig = robot.rig_with(
                        robot.cal.cam1.clone(),
                        robot.cal.cam1_se3_rig,
                        robot.cal.plane.clone(),
                        perturbed_handeye(&robot.handeye, k, s),
                    );
                    points_at(&rig, &poses)
                };
                check(label, column(label), run(h), run(-h), h);
            }
            if moving {
                for capture in 0..2 {
                    for (k, axis) in ["rx", "ry", "rz", "tx", "ty", "tz"].iter().enumerate() {
                        let run = |s: f64| {
                            let mut shifted = robot.robot;
                            shifted[capture] = right(&shifted[capture], k, s);
                            points_at(&rig, &robot.robot_poses(&shifted))
                        };
                        let label = format!("capture{capture}.base_se3_gripper.{axis}");
                        check(&label, column(&label), run(h), run(-h), h);
                    }
                }
            } else {
                assert!(!lin.labels.iter().any(|l| l.contains("base_se3_gripper")));
            }
            // A camera parameter and the extrinsic, carried into the frame.
            let theta0 = theta(&robot.cal.cam1);
            let j = theta_labels(&robot.cal.cam1)
                .iter()
                .position(|l| l == "fx")
                .unwrap();
            let run = |s: f64| {
                let mut t = theta0.clone();
                t[j] += s;
                let rig = robot.rig_with(
                    with_theta(&robot.cal.cam1, &t),
                    robot.cal.cam1_se3_rig,
                    robot.cal.plane.clone(),
                    robot.handeye,
                );
                points_at(&rig, &poses)
            };
            check("fx", column("cam1.fx"), run(1e-3), run(-1e-3), 1e-3);
            let run = |s: f64| {
                let rig = robot.rig_with(
                    robot.cal.cam1.clone(),
                    right(&robot.cal.cam1_se3_rig, 1, s),
                    robot.cal.plane.clone(),
                    robot.handeye,
                );
                points_at(&rig, &poses)
            };
            check("ry", column("cam1.cam_se3_rig.ry"), run(h), run(-h), h);
        }
    }
    eprintln!(
        "robot-frame laser Jacobians vs central differences: worst relative error {worst:.2e}"
    );
}

/// Points of two captures in the base frame (eye-in-hand) and the gripper
/// frame (eye-to-hand), over draws of the calibration (the hand-eye
/// correlated with the camera and the plane), the robot poses and the
/// pixels: the sample covariance is the stated one, cross terms included.
#[test]
fn the_robot_frame_covariance_matches_the_sampled_one() {
    for (eye_in_hand, seed) in [(true, 4_100_u64), (false, 4_200)] {
        let robot = OnRobot::new(eye_in_hand);
        let frame = match robot.handeye.mount_frame() {
            RobotFrame::Gripper => RobotFrame::Base,
            RobotFrame::Base => RobotFrame::Gripper,
        };
        let observations = robot.observations();
        let poses = robot.robot_poses(&robot.robot);
        let found = triangulate_laser(
            &robot.rig(),
            Captures::Robot {
                frame,
                robot: &poses,
            },
            &observations,
        )
        .unwrap();
        let c = found.uncertainty.covariance().unwrap();
        let stated = DMatrix::from_fn(c.labels.len(), c.labels.len(), |r, s| c.covariance[r][s]);

        let chol = robot.sigma.clone().cholesky().unwrap().l();
        let n_z = robot.cal.sigma_z.nrows();
        let k = robot.cal.n_theta;
        let n = robot.cal.plane.normal().into_inner();
        let mut rng = Lcg(seed);
        let draws: Vec<DVector<f64>> = (0..TRIALS)
            .map(|_| {
                let g = DVector::from_fn(chol.nrows(), |_, _| gauss(&mut rng));
                let draw = &chol * g;
                let c = &robot.cal.embed * draw.rows(0, n_z);
                let theta1: Vec<f64> = theta(&robot.cal.cam1)
                    .iter()
                    .zip(c.iter())
                    .map(|(t, d)| t + d)
                    .collect();
                let n1 = (n + Vec3::new(c[k], c[k + 1], c[k + 2])).normalize();
                let laser = LaserPlane::new(n1, robot.cal.plane.distance() + c[k + 3]).unwrap();
                let mut cam1_se3_rig = robot.cal.cam1_se3_rig;
                let mut handeye = robot.handeye;
                // One draw of each pose: Exp of the whole tangent.
                let exp = |xi: &[f64]| {
                    Iso3::from_parts(
                        nalgebra::Translation3::new(xi[3], xi[4], xi[5]),
                        UnitQuaternion::from_scaled_axis(Vec3::new(xi[0], xi[1], xi[2])),
                    )
                };
                let extr: Vec<f64> = (0..6).map(|i| c[k + 4 + i]).collect();
                cam1_se3_rig *= exp(&extr);
                let eh: Vec<f64> = (0..6).map(|i| draw[n_z + i]).collect();
                handeye = match handeye {
                    Handeye::EyeInHand { gripper_se3_rig } => Handeye::EyeInHand {
                        gripper_se3_rig: gripper_se3_rig * exp(&eh),
                    },
                    Handeye::EyeToHand { rig_se3_base } => Handeye::EyeToHand {
                        rig_se3_base: rig_se3_base * exp(&eh),
                    },
                };
                let (sr, st) = robot.robot_sigma;
                let robot_drawn: Vec<RobotPose> = robot
                    .robot
                    .iter()
                    .map(|p| {
                        let xi: Vec<f64> = (0..6)
                            .map(|i| if i < 3 { sr } else { st } * gauss(&mut rng))
                            .collect();
                        RobotPose::exact(p * exp(&xi))
                    })
                    .collect();
                let rig = robot.rig_with(
                    with_theta(&robot.cal.cam1, &theta1),
                    cam1_se3_rig,
                    laser,
                    handeye,
                );
                let noisy: Vec<LaserObservation> = observations
                    .iter()
                    .map(|o| {
                        let mut o = *o;
                        o.pixel.x += PIXEL_SIGMA * gauss(&mut rng);
                        o.pixel.y += PIXEL_SIGMA * gauss(&mut rng);
                        o
                    })
                    .collect();
                let points: Vec<Vec3> = triangulate_laser(
                    &rig,
                    Captures::Robot {
                        frame,
                        robot: &robot_drawn,
                    },
                    &noisy,
                )
                .unwrap()
                .points
                .iter()
                .map(|p| p.point.coords)
                .collect();
                flat(&points)
            })
            .collect();
        let sample = sample_covariance(&draws);
        let block =
            |m: &DMatrix<f64>, i: usize, j: usize| m.view((3 * i, 3 * j), (3, 3)).into_owned();
        for i in 0..observations.len() {
            let error = frobenius_error(&block(&sample, i, i), &block(&stated, i, i));
            eprintln!("{frame:?}: point {i}, relative Frobenius error {error:.3}");
            assert!(error <= 0.08, "{frame:?} point {i}: {error:.3}");
        }
        // Across captures (the calibration and the hand-eye) and within one
        // (its robot pose too).
        for (i, j) in [(0, 4), (0, 2)] {
            let error = frobenius_error(&block(&sample, i, j), &block(&stated, i, j));
            eprintln!("{frame:?}: cross block ({i}, {j}), relative Frobenius error {error:.3}");
            assert!(error <= 0.1, "{frame:?} cross ({i}, {j}): {error:.3}");
        }
    }
}
