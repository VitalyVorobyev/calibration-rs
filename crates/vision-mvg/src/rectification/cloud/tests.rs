use super::*;
use crate::metrology::{PointSet, distance};
use crate::rectification::{RectifyOptions, rectify};
use crate::test_support::{iso, pinhole_params};
use crate::triangulation::{RigObservation, triangulate};
use crate::types::Captures;
use nalgebra::{DVector, Matrix2, UnitQuaternion};
use vision_calibration_core::{
    CameraParams, DistortionParams, ParameterCovariance, PointCovariance, ProjectionParams,
    RunControl, ScheimpflugParams, SensorParams, Uncertainty,
};

/// A pinhole camera with Brown-Conrady distortion and a tilted sensor.
fn scheimpflug_params() -> CameraParams {
    pinhole_params(-0.12).with_sensor(SensorParams::Scheimpflug {
        params: ScheimpflugParams {
            tilt_x: 0.03,
            tilt_y: -0.02,
        },
    })
}

/// `camera` with its parameters replaced by the flat vector `theta`.
fn with_theta(camera: &CameraParams, theta: &[f64]) -> CameraParams {
    let blocks = camera.blocks();
    let (i, rest) = theta.split_at(blocks[0].len());
    let (p, rest) = rest.split_at(blocks[1].len());
    let (d, s) = rest.split_at(blocks[2].len());
    CameraParams::new(
        ProjectionParams::from_values(camera.projection.kind(), p).unwrap(),
        DistortionParams::from_values(camera.distortion.kind(), d).unwrap(),
        SensorParams::from_values(camera.sensor.kind().unwrap(), s).unwrap(),
        FxFyCxCy {
            fx: i[0],
            fy: i[1],
            cx: i[2],
            cy: i[3],
        },
    )
}

fn theta_labels(camera: &CameraParams) -> Vec<String> {
    CameraJacobian::new(camera)
        .unwrap()
        .labels()
        .iter()
        .map(|l| l.to_string())
        .collect()
}

/// A three-camera rig measured with the pair `[1, 2]`, so both cameras of
/// the pair have an estimated extrinsic and the rig frame is neither's.
struct Fixture {
    cameras: Vec<CameraParams>,
    cam_se3_rig: Vec<Iso3>,
    /// The labels of the covariance: cam1's and cam2's parameters, then
    /// their two extrinsics.
    labels: Vec<String>,
    sigma: DMatrix<f64>,
}

impl Fixture {
    fn new() -> Self {
        let cameras = vec![
            pinhole_params(0.0),
            pinhole_params(-0.1),
            scheimpflug_params(),
        ];
        let cam_se3_rig = vec![
            Iso3::identity(),
            iso(0.01, -0.03, 0.02, Vec3::new(0.05, 0.01, 0.0)),
            iso(-0.01, 0.04, 0.0, Vec3::new(-0.10, 0.0, 0.01)),
        ];
        let mut labels = Vec::new();
        let mut std = Vec::new();
        for (k, camera) in cameras.iter().enumerate().skip(1) {
            for label in theta_labels(camera) {
                std.push(match label.as_str() {
                    "fx" | "fy" => 4.0,
                    "cx" | "cy" => 2.0,
                    "k1" => 3e-3,
                    "k2" => 6e-3,
                    "p1" | "p2" => 2e-4,
                    "k3" => 1e-6,
                    _ => 5e-4, // tilt_x, tilt_y
                });
                labels.push(format!("cam{k}.{label}"));
            }
        }
        for k in 1..3 {
            for axis in crate::sensitivity::POSE_AXES {
                std.push(1e-3); // 1 mrad, 1 mm
                labels.push(format!("cam{k}.cam_se3_rig.{axis}"));
            }
        }
        // Correlations, as a joint calibration from shared views has them.
        let n = labels.len();
        let at = |name: &str| labels.iter().position(|l| l == name).unwrap();
        let mut corr = DMatrix::<f64>::identity(n, n);
        let mut tie = |a: usize, b: usize, r: f64| {
            corr[(a, b)] = r;
            corr[(b, a)] = r;
        };
        tie(at("cam1.fx"), at("cam1.fy"), 0.6);
        tie(at("cam2.fx"), at("cam2.fy"), 0.6);
        tie(at("cam1.fx"), at("cam2.fx"), 0.4);
        tie(at("cam1.k1"), at("cam1.k2"), -0.7);
        tie(at("cam2.cx"), at("cam2.cam_se3_rig.ry"), 0.5);
        tie(at("cam1.cx"), at("cam1.cam_se3_rig.ry"), 0.4);
        tie(at("cam2.cam_se3_rig.tx"), at("cam2.fx"), 0.3);
        let sigma = DMatrix::from_fn(n, n, |a, b| std[a] * std[b] * corr[(a, b)]);
        assert!(sigma.clone().cholesky().is_some(), "Σ is positive definite");
        Self {
            cameras,
            cam_se3_rig,
            labels,
            sigma,
        }
    }

    fn covariance(&self) -> ParameterCovariance {
        let n = self.labels.len();
        ParameterCovariance {
            labels: self.labels.clone(),
            covariance: (0..n)
                .map(|r| (0..n).map(|c| self.sigma[(r, c)]).collect())
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

    fn rig_with(&self, cameras: Vec<CameraParams>, cam_se3_rig: Vec<Iso3>) -> CalibratedRig {
        CalibratedRig::new(
            cameras,
            cam_se3_rig,
            Uncertainty::Available(self.covariance()),
        )
        .unwrap()
    }

    fn rig(&self) -> CalibratedRig {
        self.rig_with(self.cameras.clone(), self.cam_se3_rig.clone())
    }
}

/// Rectified pixels and disparities 0.9–1.6 m ahead of the pair.
const MATCHES: [(usize, usize, f32); 4] = [
    (640, 480, 100.0),
    (420, 300, 80.5),
    (900, 650, 120.25),
    (500, 720, 75.0),
];

fn map_of(matches: &[(usize, usize, f32)]) -> DisparityMap {
    let mut map = DisparityMap::filled(1280, 960, f32::NAN);
    for &(x, y, d) in matches {
        map.set(x, y, d);
    }
    map
}

/// The point the nominal rectification's measurement `(x, y, d)` is under a
/// perturbed rig: on the perturbed left ray through the fixed raw left pixel,
/// with its perturbed right projection on the fixed rectified column
/// `x − d`. Newton steps on the forward projections only, from `start`.
fn reference_point(
    rect: &StereoRectification,
    rig: &CalibratedRig,
    (x, y, d): (f64, f64, f64),
    start: Vec3,
) -> Vec3 {
    let [a, b] = rect.cameras();
    let u_left = rect.raw_pixel(0, &Pt2::new(x, y)).unwrap();
    let residual = |p: &Vec3| -> Vector3<f64> {
        let left = rig
            .model(a)
            .unwrap()
            .project_point(&(rig.cam_se3_rig()[a] * Pt3::from(*p)))
            .unwrap();
        let right = rig
            .model(b)
            .unwrap()
            .project_point(&(rig.cam_se3_rig()[b] * Pt3::from(*p)))
            .unwrap();
        let column = rect.rectified_pixel(1, &right).unwrap().x;
        Vector3::new(left.x - u_left.x, left.y - u_left.y, column - (x - d))
    };
    let mut p = start;
    for _ in 0..30 {
        let f = residual(&p);
        if f.norm() < 1e-11 {
            break;
        }
        let h = 1e-7;
        let mut j = Matrix3::zeros();
        for c in 0..3 {
            let mut e = Vec3::zeros();
            e[c] = h;
            j.set_column(c, &((residual(&(p + e)) - residual(&(p - e))) / (2.0 * h)));
        }
        p -= j.try_inverse().unwrap() * f;
    }
    assert!(
        residual(&p).norm() < 1e-9,
        "reference point did not converge"
    );
    p
}

#[test]
fn the_cloud_holds_the_valid_pixels_in_the_rig_frame() {
    let fix = Fixture::new();
    let rig = fix.rig();
    let rect = rectify(&rig, [1, 2], &RectifyOptions::default()).unwrap();
    let mut map = map_of(&MATCHES);
    // Invalid, zero, negative and infinite disparities are left out.
    map.set(10, 10, 0.0);
    map.set(11, 10, -3.0);
    map.set(12, 10, f32::INFINITY);
    let cloud = rect.dense_cloud(&rig, &map, 0.2).unwrap();
    assert_eq!(cloud.len(), MATCHES.len());
    assert_eq!(cloud.sigma_d(), 0.2);
    assert!(cloud.covariance_unavailable().is_none());
    // Row by row: (640, 480) comes after (420, 300) and before (900, 650).
    let pixels: Vec<[usize; 2]> = cloud.points().iter().map(|p| p.pixel).collect();
    assert_eq!(pixels, vec![[420, 300], [640, 480], [900, 650], [500, 720]]);
    for p in cloud.points() {
        let left = Pt2::new(p.pixel[0] as f64, p.pixel[1] as f64);
        let in_left = rect.point(&left, p.disparity).unwrap();
        let expected = rig.cam_se3_rig()[1].inverse() * in_left;
        assert!((p.point - expected).norm() < 1e-12, "{:?}", p.pixel);
        // Both raw pixels reproject: the point is the stereo intersection.
        let right = Pt2::new(left.x - p.disparity, left.y);
        for (side, rectified) in [(0, left), (1, right)] {
            let k = rect.cameras()[side];
            let raw = rect.raw_pixel(side, &rectified).unwrap();
            let seen = rig
                .model(k)
                .unwrap()
                .project_point(&(rig.cam_se3_rig()[k] * p.point))
                .unwrap();
            assert!(
                (seen - raw).norm() < 1e-8,
                "side {side}: {}",
                (seen - raw).norm()
            );
        }
    }
}

#[test]
fn jacobians_match_central_differences() {
    let fix = Fixture::new();
    let rig = fix.rig();
    let rect = rectify(&rig, [1, 2], &RectifyOptions::default()).unwrap();
    let cloud = rect.dense_cloud(&rig, &map_of(&MATCHES), 0.3).unwrap();
    let share = cloud.share.as_ref().unwrap();

    // The label of each column of G.
    let m = share.sigma_c.nrows();
    assert_eq!(m, fix.labels.len());
    let mut column_label = vec![String::new(); m];
    for (s, side) in share.sides.iter().enumerate() {
        let k = rect.cameras()[s];
        let names = theta_labels(&fix.cameras[k]);
        for &(j, c) in &side.theta {
            column_label[c] = format!("cam{k}.{}", names[j]);
        }
        let first = side.extrinsic.expect("both extrinsics are listed");
        for (i, axis) in crate::sensitivity::POSE_AXES.iter().enumerate() {
            column_label[first + i] = format!("cam{k}.cam_se3_rig.{axis}");
        }
    }
    let mut sorted = column_label.clone();
    sorted.sort();
    let mut expected = fix.labels.clone();
    expected.sort();
    assert_eq!(sorted, expected, "every listed coordinate has a column");

    let mut worst: f64 = 0.0;
    for p in cloud.points() {
        let (g, g_d) = share.jacobian(&p.point.coords).unwrap();
        let at = (p.pixel[0] as f64, p.pixel[1] as f64, p.disparity);
        let x0 = p.point.coords;

        // The disparity: −A⁻¹ e₃ is the closed form −(X − c_L)/d.
        let closed = -(x0 - cloud.left_centre) / p.disparity;
        let error = (g_d - closed).norm() / closed.norm();
        assert!(error < 1e-10, "disparity: relative error {error:e}");

        let mut check = |label: &str, rig_plus: CalibratedRig, rig_minus: CalibratedRig, h: f64| {
            let plus = reference_point(&rect, &rig_plus, at, x0);
            let minus = reference_point(&rect, &rig_minus, at, x0);
            let numeric = (plus - minus) / (2.0 * h);
            let c = column_label.iter().position(|l| l == label).unwrap();
            let analytic = g.column(c).into_owned();
            let miss = (analytic - numeric).norm();
            // A column near zero (k2, k3 at the principal point, 1e-8 m per
            // unit and less) is below the differences' rounding: there the
            // miss is held to 1e-9 m per unit.
            assert!(
                miss < 1e-6 * analytic.norm() + 1e-9,
                "{label} at {:?}: |analytic| {:e}, miss {miss:e}",
                p.pixel,
                analytic.norm()
            );
            if analytic.norm() > 1e-6 {
                worst = worst.max(miss / analytic.norm());
            }
        };
        for k in 1..3 {
            let theta0 = fix.cameras[k].blocks().concat();
            for (j, name) in theta_labels(&fix.cameras[k]).iter().enumerate() {
                let h = 1e-4 * theta0[j].abs().max(1.0);
                let run = |s: f64| {
                    let mut t = theta0.clone();
                    t[j] += s;
                    let mut cameras = fix.cameras.clone();
                    cameras[k] = with_theta(&fix.cameras[k], &t);
                    fix.rig_with(cameras, fix.cam_se3_rig.clone())
                };
                check(&format!("cam{k}.{name}"), run(h), run(-h), h);
            }
            for (i, axis) in crate::sensitivity::POSE_AXES.iter().enumerate() {
                let h = 1e-6;
                let run = |s: f64| {
                    let mut xi = [0.0; 6];
                    xi[i] = s;
                    let delta = Iso3::from_parts(
                        nalgebra::Translation3::new(xi[3], xi[4], xi[5]),
                        UnitQuaternion::from_scaled_axis(Vec3::new(xi[0], xi[1], xi[2])),
                    );
                    let mut poses = fix.cam_se3_rig.clone();
                    poses[k] *= delta;
                    fix.rig_with(fix.cameras.clone(), poses)
                };
                check(&format!("cam{k}.cam_se3_rig.{axis}"), run(h), run(-h), h);
            }
        }
    }
    eprintln!("dense cloud Jacobians vs central differences: worst relative error {worst:.2e}");
}

/// The dense measurement is a triangulation of the two raw pixels with the
/// left pixel exact and the right one measured only along the rectified
/// row: `triangulate` with that noise (a left σ of 1e-3 px, a right one of
/// σ_d along the rectified column and 100 px across it) states the same
/// joint covariance, cross terms included.
#[test]
fn point_covariance_is_triangulate_with_the_dense_noise_model() {
    let fix = Fixture::new();
    let rig = fix.rig();
    let rect = rectify(&rig, [1, 2], &RectifyOptions::default()).unwrap();
    let sigma_d = 0.3;
    let cloud = rect.dense_cloud(&rig, &map_of(&MATCHES), sigma_d).unwrap();
    let [a, b] = rect.cameras();

    let tracks: Vec<Vec<RigObservation>> = cloud
        .points()
        .iter()
        .map(|p| {
            let left = Pt2::new(p.pixel[0] as f64, p.pixel[1] as f64);
            let raw_left = rect.raw_pixel(0, &left).unwrap();
            let raw_right = rect
                .raw_pixel(1, &Pt2::new(left.x - p.disparity, left.y))
                .unwrap();
            // r = ∂(rectified column)/∂(raw right pixel), by differences of
            // the fixed map.
            let h = 1e-4;
            let column = |du: f64, dv: f64| {
                rect.rectified_pixel(1, &Pt2::new(raw_right.x + du, raw_right.y + dv))
                    .unwrap()
                    .x
            };
            let r = Vec2::new(
                (column(h, 0.0) - column(-h, 0.0)) / (2.0 * h),
                (column(0.0, h) - column(0.0, -h)) / (2.0 * h),
            );
            let along = r / r.norm();
            let across = Vec2::new(-along.y, along.x);
            let noise: Matrix2<f64> = along * along.transpose() * (sigma_d / r.norm()).powi(2)
                + across * across.transpose() * 100.0_f64.powi(2);
            vec![
                RigObservation::new(0, a, raw_left, PointCovariance::isotropic(1e-3).unwrap()),
                RigObservation::new(
                    0,
                    b,
                    raw_right,
                    PointCovariance::new(noise[(0, 0)], noise[(0, 1)], noise[(1, 1)]).unwrap(),
                ),
            ]
        })
        .collect();
    let found = triangulate(&rig, Captures::Rig, &tracks, &RunControl::default()).unwrap();
    let all: Vec<usize> = (0..cloud.len()).collect();
    let joint = cloud.covariance(&all).unwrap();
    let mut worst: f64 = 0.0;
    for i in 0..cloud.len() {
        // The refinement starts at the exact intersection and stays there.
        let moved = (found.points[i].point - cloud.points()[i].point).norm();
        assert!(moved < 1e-9, "point {i} moved {moved:e}");
        for j in 0..cloud.len() {
            let stated = found.cross_covariance(i, j).unwrap();
            let ours = joint.fixed_view::<3, 3>(3 * i, 3 * j).into_owned();
            let error = (ours - stated).norm() / stated.norm();
            worst = worst.max(error);
            assert!(error < 1e-4, "block ({i}, {j}): relative error {error:e}");
        }
        let single = cloud.point_covariance(i).unwrap();
        let block = joint.fixed_view::<3, 3>(3 * i, 3 * i).into_owned();
        assert!((single - block).norm() <= 1e-12 * block.norm());
    }
    eprintln!("dense cloud vs triangulate with the dense noise model: worst {worst:.2e}");

    // Against isotropic raw-pixel noise, the calibration's share differs
    // across the rows only: `triangulate` splits a vertical disparity
    // between the two cameras, the matcher puts it on the right one.
    let iso_noise = PointCovariance::isotropic(1e-4).unwrap();
    let isotropic: Vec<Vec<RigObservation>> = tracks
        .iter()
        .map(|t| {
            t.iter()
                .map(|o| RigObservation {
                    noise: iso_noise,
                    ..*o
                })
                .collect()
        })
        .collect();
    let symmetric = triangulate(&rig, Captures::Rig, &isotropic, &RunControl::default()).unwrap();
    let calibration_only = rect.dense_cloud(&rig, &map_of(&MATCHES), 0.0).unwrap();
    for i in 0..cloud.len() {
        let ours = calibration_only.point_covariance(i).unwrap();
        let theirs = symmetric.point_covariance(i).unwrap();
        // Depth along the left camera's axis, the rig's z.
        let ratio = (ours[(2, 2)] / theirs[(2, 2)]).sqrt();
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "point {i}: depth σ ratio {ratio}"
        );
        eprintln!(
            "point {i}: depth σ dense {:.3} mm, isotropic triangulate {:.3} mm (ratio {ratio:.3}); \
             y σ {:.3} vs {:.3} mm",
            ours[(2, 2)].sqrt() * 1e3,
            theirs[(2, 2)].sqrt() * 1e3,
            ours[(1, 1)].sqrt() * 1e3,
            theirs[(1, 1)].sqrt() * 1e3,
        );
    }
}

#[test]
fn a_rig_without_a_covariance_says_why() {
    let fix = Fixture::new();
    let rig = CalibratedRig::new(
        fix.cameras.clone(),
        fix.cam_se3_rig.clone(),
        Uncertainty::default(),
    )
    .unwrap();
    let rect = rectify(&rig, [1, 2], &RectifyOptions::default()).unwrap();
    let cloud = rect.dense_cloud(&rig, &map_of(&MATCHES), 0.2).unwrap();
    assert_eq!(cloud.len(), MATCHES.len());
    assert_eq!(
        cloud.covariance_unavailable(),
        Some(&UncertaintyUnavailable::CalibrationWithoutCovariance)
    );
    assert!(cloud.point_covariance(0).is_none());
    let err = cloud.covariance(&[0, 1]).unwrap_err();
    assert!(
        matches!(&err, MvgError::InvalidInput { reason } if reason.contains("CalibrationWithoutCovariance")),
        "{err:?}"
    );
    assert!(PointSet::from_dense_cloud(&cloud, &[0, 1]).is_err());
}

#[test]
fn rejects_bad_inputs() {
    let fix = Fixture::new();
    let rig = fix.rig();
    let rect = rectify(&rig, [1, 2], &RectifyOptions::default()).unwrap();
    let map = map_of(&MATCHES);
    let invalid = |e: MvgError| matches!(e, MvgError::InvalidInput { .. });
    for sigma_d in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(invalid(rect.dense_cloud(&rig, &map, sigma_d).unwrap_err()));
    }
    let mut poses = fix.cam_se3_rig.clone();
    poses[2] = Iso3::translation(-0.2, 0.0, 0.0);
    let other = fix.rig_with(fix.cameras.clone(), poses);
    assert!(invalid(rect.dense_cloud(&other, &map, 0.2).unwrap_err()));
    // A covariance with part of an extrinsic.
    let mut cov = fix.covariance();
    let keep: Vec<usize> = (0..cov.labels.len())
        .filter(|&i| cov.labels[i] != "cam2.cam_se3_rig.tz")
        .collect();
    cov.labels = keep.iter().map(|&i| cov.labels[i].clone()).collect();
    cov.covariance = keep
        .iter()
        .map(|&r| keep.iter().map(|&c| cov.covariance[r][c]).collect())
        .collect();
    let partial = CalibratedRig::new(
        fix.cameras.clone(),
        fix.cam_se3_rig.clone(),
        Uncertainty::Available(cov),
    )
    .unwrap();
    assert!(invalid(rect.dense_cloud(&partial, &map, 0.2).unwrap_err()));
    // An index out of range.
    let cloud = rect.dense_cloud(&rig, &map, 0.2).unwrap();
    assert!(invalid(cloud.covariance(&[0, 9]).unwrap_err()));
    assert!(cloud.point_covariance(9).is_none());
}

/// Metrology over cloud points sees their joint covariance, the
/// calibration's correlation included.
#[test]
fn a_point_set_of_cloud_points_measures_with_the_joint_covariance() {
    let fix = Fixture::new();
    let rig = fix.rig();
    let rect = rectify(&rig, [1, 2], &RectifyOptions::default()).unwrap();
    let cloud = rect.dense_cloud(&rig, &map_of(&MATCHES), 0.2).unwrap();
    let indices = [3, 0];
    let set = PointSet::from_dense_cloud(&cloud, &indices).unwrap();
    assert_eq!(set.point(0), Some(cloud.points()[3].point));
    let joint = cloud.covariance(&indices).unwrap();
    let d = distance(&set, 0, 1).unwrap();
    let u = (cloud.points()[3].point - cloud.points()[0].point) / d.value;
    let grad = DVector::from_iterator(6, u.iter().copied().chain(u.iter().map(|v| -v)));
    let sigma = (grad.dot(&(&joint * &grad))).sqrt();
    assert!((d.sigma - sigma).abs() < 1e-12 * sigma);
    // The points are correlated: independent errors would state another σ.
    let independent = PointSet::independent(
        set_points(&set),
        &[
            cloud.point_covariance(3).unwrap(),
            cloud.point_covariance(0).unwrap(),
        ],
    )
    .unwrap();
    let apart = distance(&independent, 0, 1).unwrap().sigma;
    assert!(
        (apart - d.sigma).abs() > 0.05 * d.sigma,
        "{apart} vs {}",
        d.sigma
    );
}

fn set_points(set: &PointSet) -> Vec<Pt3> {
    (0..set.len()).map(|i| set.point(i).unwrap()).collect()
}

type Vec2 = nalgebra::Vector2<f64>;
