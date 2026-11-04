//! Measurements with joint σ: exact values, error cases, and Monte-Carlo
//! agreement of every propagated σ against a correlated joint covariance.

#![allow(missing_docs)]

use nalgebra::{DMatrix, DVector};
use vision_calibration_core::PointCovariance;
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, FxFyCxCy, Mat3, Pt3, Real, Uncertainty, UncertaintyUnavailable,
    Vec3, make_pinhole_camera, pinhole_camera_params,
};
use vision_mvg::MvgError;
use vision_mvg::metrology::{PointSet, angle, distance, fit_plane, point_to_plane};
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

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

// ---------------------------------------------------------------------------
// Exact values
// ---------------------------------------------------------------------------

fn tiny_set(points: Vec<Pt3>) -> PointSet {
    let n = 3 * points.len();
    PointSet::new(points, DMatrix::identity(n, n) * 1e-8).unwrap()
}

#[test]
fn exact_values_on_constructed_geometry() {
    let set = tiny_set(vec![
        Pt3::new(0.0, 0.0, 1.0),
        Pt3::new(0.3, 0.4, 1.0),
        Pt3::new(0.0, 0.5, 1.0),
        Pt3::new(0.3, 0.0, 1.0),
        Pt3::new(0.1, 0.1, 1.25),
    ]);
    assert!((distance(&set, 0, 1).unwrap().value - 0.5).abs() < 1e-15);
    // Right angle at point 0 between +x (point 3) and +y (point 2).
    let right = angle(&set, 3, 0, 2).unwrap();
    assert!((right.value - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!(right.sigma > 0.0);

    // Four coplanar points in z = 1: n = +z (d >= 0), d = 1.
    let plane = fit_plane(&set, &[0, 1, 2, 3]).unwrap();
    assert!((plane.normal - Vec3::z()).norm() < 1e-14);
    assert!((plane.distance - 1.0).abs() < 1e-14);
    assert!(plane.rms < 1e-14);
    assert_eq!(plane.indices, vec![0, 1, 2, 3]);
    // The point 0.25 above the plane.
    let above = point_to_plane(&set, &[0, 1, 2, 3], 4).unwrap();
    assert!((above.value - 0.25).abs() < 1e-14);
    // A point of the plane is at distance 0 from it, with a finite σ.
    let inside = point_to_plane(&set, &[0, 1, 2, 3], 1).unwrap();
    assert!(inside.value.abs() < 1e-14 && inside.sigma > 0.0);
}

#[test]
fn independent_points_have_the_textbook_sigma() {
    // Isotropic σ per axis: the distance's σ is σ√2 (only the axial part counts).
    let sigma = 2.0e-4;
    let c = Mat3::identity() * sigma * sigma;
    let set = PointSet::independent(
        vec![Pt3::new(0.0, 0.0, 0.7), Pt3::new(0.0, 0.1, 0.7)],
        &[c, c],
    )
    .unwrap();
    let d = distance(&set, 0, 1).unwrap();
    assert!((d.sigma - 2.0_f64.sqrt() * sigma).abs() < 1e-12);
    assert!((set.covariance(1).unwrap() - c).norm() < 1e-20);
    assert!(set.covariance(2).is_none() && set.point(2).is_none());
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

fn error_text<T>(r: Result<T, MvgError>) -> String {
    r.err().expect("an error").to_string()
}

#[test]
fn degenerate_and_invalid_inputs_are_typed_errors() {
    let line = tiny_set((0..5).map(|i| Pt3::new(0.1 * i as f64, 0.0, 1.0)).collect());
    assert!(matches!(
        fit_plane(&line, &[0, 1, 2, 3]),
        Err(MvgError::Degenerate { .. })
    ));
    assert!(matches!(
        fit_plane(&line, &[0, 1]),
        Err(MvgError::InvalidInput { .. })
    ));
    assert!(error_text(fit_plane(&line, &[0, 1, 1])).contains("repeated"));
    assert!(error_text(fit_plane(&line, &[0, 1, 9])).contains("out of range"));
    assert!(error_text(distance(&line, 0, 0)).contains("repeated"));
    assert!(error_text(distance(&line, 0, 7)).contains("out of range"));
    assert!(error_text(angle(&line, 0, 1, 1)).contains("repeated"));
    assert!(error_text(point_to_plane(&line, &[0, 1, 2], 8)).contains("out of range"));

    let coincident = tiny_set(vec![Pt3::new(0.0, 0.0, 1.0), Pt3::new(0.0, 0.0, 1.0)]);
    assert!(matches!(
        distance(&coincident, 0, 1),
        Err(MvgError::Degenerate { .. })
    ));

    // Bad covariances.
    let p = vec![Pt3::new(0.0, 0.0, 1.0)];
    assert!(PointSet::new(p.clone(), DMatrix::identity(2, 2)).is_err());
    assert!(PointSet::new(p.clone(), -DMatrix::identity(3, 3)).is_err());
    let mut asym = DMatrix::identity(3, 3);
    asym[(0, 1)] = 0.5;
    assert!(PointSet::new(p.clone(), asym).is_err());
    let mut nan = DMatrix::identity(3, 3);
    nan[(2, 2)] = f64::NAN;
    assert!(PointSet::new(p, nan).is_err());
    assert!(matches!(
        PointSet::independent(vec![Pt3::origin()], &[]),
        Err(MvgError::CountMismatch { .. })
    ));
}

#[test]
fn a_triangulation_without_uncertainty_names_the_reason() {
    let k = FxFyCxCy {
        fx: 800.0,
        fy: 800.0,
        cx: 640.0,
        cy: 480.0,
    };
    let params = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
    let cam = make_pinhole_camera(k, BrownConrady5::default());
    let shift = vision_calibration_core::Iso3::translation(-0.1, 0.0, 0.0);
    // A rig of two cameras without a covariance.
    let rig = CalibratedRig::new(
        vec![params.clone(), params],
        vec![vision_calibration_core::Iso3::identity(), shift],
        Uncertainty::default(),
    )
    .unwrap();
    let p = Pt3::new(0.02, 0.01, 0.8);
    let noise = PointCovariance::isotropic(0.3).unwrap();
    let tracks = vec![vec![
        RigObservation::new(0, 0, cam.project_point(&p).unwrap(), noise),
        RigObservation::new(0, 1, cam.project_point(&(shift * p)).unwrap(), noise),
    ]];
    let t = triangulate(
        &rig,
        &[vision_calibration_core::Iso3::identity()],
        &tracks,
        &vision_calibration_core::RunControl::default(),
    )
    .unwrap();
    assert!(matches!(
        t.uncertainty,
        Uncertainty::Unavailable {
            reason: UncertaintyUnavailable::CalibrationWithoutCovariance
        }
    ));
    let message = error_text(PointSet::from_triangulation(&t));
    assert!(
        message.contains("CalibrationWithoutCovariance"),
        "{message}"
    );
    assert!(message.contains("point 0"), "{message}");
}

// ---------------------------------------------------------------------------
// Monte-Carlo with a correlated joint covariance
// ---------------------------------------------------------------------------

const DRAWS: usize = 20_000;

/// Eight points: six on a tilted plane (a 3 x 2 grid), one 5 cm above it and
/// one 3 cm below.
fn truth_points() -> Vec<Pt3> {
    let normal = Vec3::new(0.1, -0.15, 1.0).normalize();
    let u = Vec3::new(1.0, 0.0, -0.1)
        .cross(&normal)
        .cross(&normal)
        .normalize();
    let u = if u.x < 0.0 { -u } else { u };
    let v = normal.cross(&u);
    let origin = Vec3::new(0.01, -0.02, 0.7);
    let mut points: Vec<Pt3> = (0..3)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| {
            (origin + u * (0.06 * i as f64 - 0.06) + v * (0.07 * j as f64 - 0.035)).into()
        })
        .collect();
    points.push((origin + normal * 0.05 + u * 0.02).into());
    points.push((origin - normal * 0.03 - v * 0.04).into());
    points
}

/// A joint covariance `A Aᵀ` with shared components (a common translation, an
/// isotropic scale, a depth stretch and a shear: errors a calibration makes
/// in every point at once) plus independent noise. Returns the factor `A`.
fn correlated_factor(points: &[Pt3]) -> DMatrix<Real> {
    let n = points.len();
    let centroid = points.iter().fold(Vec3::zeros(), |s, p| s + p.coords) / n as f64;
    let mut cols: Vec<DVector<Real>> = Vec::new();
    let mut shared = |sigma: Real, f: &dyn Fn(&Vec3) -> Vec3| {
        cols.push(DVector::from_fn(3 * n, |r, _| {
            sigma * f(&(points[r / 3].coords - centroid))[r % 3]
        }));
    };
    for axis in 0..3 {
        shared(3.0e-4, &|_| Vec3::ith(axis, 1.0));
    }
    shared(2.0e-3, &|q| *q); // isotropic scale
    shared(4.0e-3, &|q| Vec3::new(0.0, 0.0, q.z)); // stretch along z
    shared(3.0e-3, &|q| Vec3::new(q.y, 0.0, 0.0)); // shear
    shared(1.5e-3, &|q| Vec3::new(0.0, 0.0, q.x)); // tilt-like
    for r in 0..3 * n {
        cols.push(DVector::from_fn(
            3 * n,
            |i, _| if i == r { 1.5e-4 } else { 0.0 },
        ));
    }
    DMatrix::from_columns(&cols)
}

fn scatter_ratio(errors: &[Real], stated: Real) -> Real {
    let ms = errors.iter().map(|e| e * e).sum::<Real>() / errors.len() as f64;
    ms.sqrt() / stated
}

/// Everything measured in one draw, against the truth.
struct Measures {
    distance_short: Real,
    distance_long: Real,
    angle: Real,
    outside: Real,
    inside: Real,
}

const PLANE: [usize; 6] = [0, 1, 2, 3, 4, 5];

fn measure(set: &PointSet) -> Measures {
    Measures {
        distance_short: distance(set, 0, 1).unwrap().value,
        distance_long: distance(set, 0, 5).unwrap().value,
        angle: angle(set, 0, 2, 7).unwrap().value,
        outside: point_to_plane(set, &PLANE, 6).unwrap().value,
        inside: point_to_plane(set, &PLANE, 3).unwrap().value,
    }
}

#[test]
fn joint_sigma_matches_monte_carlo_and_cross_terms_matter() {
    let points = truth_points();
    let a = correlated_factor(&points);
    let covariance = &a * a.transpose();
    let joint = PointSet::new(points.clone(), covariance.clone()).unwrap();
    let diagonal: Vec<Mat3> = (0..points.len())
        .map(|i| covariance.fixed_view::<3, 3>(3 * i, 3 * i).into_owned())
        .collect();
    let independent = PointSet::independent(points.clone(), &diagonal).unwrap();

    // Truth and stated σ.
    let truth = measure(&joint);
    let plane_truth = fit_plane(&joint, &PLANE).unwrap();
    let stated = |set: &PointSet| {
        let plane = fit_plane(set, &PLANE).unwrap();
        let sd = |k: usize| plane.covariance[(k, k)].sqrt();
        [
            distance(set, 0, 1).unwrap().sigma,
            distance(set, 0, 5).unwrap().sigma,
            angle(set, 0, 2, 7).unwrap().sigma,
            point_to_plane(set, &PLANE, 6).unwrap().sigma,
            point_to_plane(set, &PLANE, 3).unwrap().sigma,
            sd(2),
            sd(0),
            sd(1),
        ]
    };
    let (sj, si) = (stated(&joint), stated(&independent));

    // Draws: x = x0 + A z. The plane's tangents of a draw are measured in the
    // truth's basis, with the normal's orientation following the truth's.
    let mut rng = Rng(0x5EED_0001);
    let mut errors: Vec<Vec<Real>> = vec![Vec::new(); 8];
    let mut chi2 = 0.0;
    let theta_cov_inverse = plane_truth.covariance.try_inverse().unwrap();
    for _ in 0..DRAWS {
        let z = DVector::from_fn(a.ncols(), |_, _| rng.normal());
        let dx = &a * z;
        let drawn: Vec<Pt3> = points
            .iter()
            .enumerate()
            .map(|(i, p)| Pt3::new(p.x + dx[3 * i], p.y + dx[3 * i + 1], p.z + dx[3 * i + 2]))
            .collect();
        let set = PointSet::new(drawn, DMatrix::identity(24, 24)).unwrap();
        let plane = fit_plane(&set, &PLANE).unwrap();
        let m = measure(&set);
        let flip = if plane.normal.dot(&plane_truth.normal) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let theta = nalgebra::Vector3::new(
            plane_truth.tangent_basis[0].dot(&(flip * plane.normal)),
            plane_truth.tangent_basis[1].dot(&(flip * plane.normal)),
            flip * plane.distance - plane_truth.distance,
        );
        chi2 += (theta.transpose() * theta_cov_inverse * theta)[(0, 0)];
        let e = [
            m.distance_short - truth.distance_short,
            m.distance_long - truth.distance_long,
            m.angle - truth.angle,
            m.outside - truth.outside,
            m.inside - truth.inside,
            theta[2],
            theta[0],
            theta[1],
        ];
        for (store, e) in errors.iter_mut().zip(e) {
            store.push(e);
        }
    }

    let names = [
        "distance (short)",
        "distance (long)",
        "angle",
        "point-to-plane (outside)",
        "point-to-plane (in the plane)",
        "plane d",
        "plane normal u",
        "plane normal v",
    ];
    eprintln!("Monte-Carlo, {DRAWS} draws: scatter / stated sigma");
    let mut joint_ratios = Vec::new();
    let mut cross_ratios = Vec::new();
    for k in 0..names.len() {
        let (rj, ri) = (
            scatter_ratio(&errors[k], sj[k]),
            scatter_ratio(&errors[k], si[k]),
        );
        eprintln!(
            "  {:30} joint {rj:.4}   without cross terms {ri:.4}",
            names[k]
        );
        joint_ratios.push(rj);
        cross_ratios.push(ri);
    }
    let chi2_mean = chi2 / DRAWS as f64;
    eprintln!("  plane (u, v, d) Mahalanobis^2 mean {chi2_mean:.4} (3 expected)");

    for (k, r) in joint_ratios.iter().enumerate() {
        assert!(
            (0.97..=1.03).contains(r),
            "{}: joint ratio {r:.4}",
            names[k]
        );
    }
    assert!(
        (2.91..=3.09).contains(&chi2_mean),
        "chi2 mean {chi2_mean:.4}"
    );
    // Without the cross terms the stated sigma is clearly wrong. The plane's
    // distance is the exception that is only moderately wrong (the shared
    // translation along the normal is partly offset by the other shared
    // terms), so it is held to "off by far more than the 1 % resolution".
    assert!(
        !(0.97..=1.03).contains(&cross_ratios[5]),
        "plane d: ratio without cross terms {:.4}",
        cross_ratios[5]
    );
    for k in [0, 1, 2, 3, 4, 6, 7] {
        assert!(
            !(0.8..=1.25).contains(&cross_ratios[k]),
            "{}: ratio without cross terms {:.4} should be wrong",
            names[k],
            cross_ratios[k]
        );
    }
}

#[test]
fn a_straight_angle_is_degenerate() {
    let line = tiny_set((0..3).map(|i| Pt3::new(0.1 * i as f64, 0.0, 1.0)).collect());
    assert!(matches!(
        angle(&line, 0, 1, 2),
        Err(MvgError::Degenerate { .. })
    ));
    assert!(matches!(
        angle(&line, 1, 0, 2),
        Err(MvgError::Degenerate { .. })
    ));
}
