//! Stereo from a calibration, on synthetic ground truth.
//!
//! For a pinhole with Brown–Conrady distortion, a Kannala–Brandt fisheye and a
//! Scheimpflug camera: matches share a rectified row, a raw pixel survives the
//! round trip through the rectification, and `point` and `Q` recover the 3D
//! point. The disparity covariance is checked against Monte-Carlo draws of the
//! disparity noise.

#![allow(missing_docs)]

use nalgebra::{Translation3, UnitQuaternion, Vector3, Vector4};
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    ProjectionParams, Pt2, Pt3, ScheimpflugParams, SensorParams, Uncertainty, Vec3,
    make_pinhole_camera, pinhole_camera_params,
};
use vision_mvg::dense::GrayImage;
use vision_mvg::rectification::{RectifyOptions, StereoRectification, rectify};

/// SplitMix64 with normal draws.
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

fn brown_conrady(fx: f64, fy: f64, cx: f64, cy: f64) -> CameraParams {
    let distortion = BrownConrady5 {
        k1: -0.15,
        k2: 0.03,
        k3: 0.0,
        p1: 0.001,
        p2: -0.0005,
        iters: 40,
    };
    pinhole_camera_params(&make_pinhole_camera(
        FxFyCxCy { fx, fy, cx, cy },
        distortion,
    ))
}

fn fisheye(f: f64, cx: f64, cy: f64) -> CameraParams {
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
            fx: f,
            fy: f,
            cx,
            cy,
        },
    }
}

fn scheimpflug(fx: f64, cx: f64, tilt: ScheimpflugParams) -> CameraParams {
    let mut camera = brown_conrady(fx, fx, cx, 480.0);
    camera.sensor = SensorParams::Scheimpflug { params: tilt };
    camera
}

fn rig(cameras: [CameraParams; 2]) -> CalibratedRig {
    let cam1_se3_rig = Iso3::from_parts(
        Translation3::new(-0.12, 0.004, 0.006),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.04, 0.008)),
    );
    CalibratedRig::new(
        cameras.to_vec(),
        vec![Iso3::identity(), cam1_se3_rig],
        Uncertainty::default(),
    )
    .unwrap()
}

/// A grid of points 0.8 to 2 m ahead of camera 0.
fn grid() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for i in 0..7 {
        for j in 0..5 {
            for k in 0..3 {
                let z = 0.8 + 0.6 * k as f64;
                pts.push(Pt3::new(
                    (-0.3 + 0.1 * i as f64) * z,
                    (-0.2 + 0.1 * j as f64) * z,
                    z,
                ));
            }
        }
    }
    pts
}

fn raw(rig: &CalibratedRig, side: usize, p: &Pt3) -> Pt2 {
    rig.model(side)
        .unwrap()
        .project_point(&(rig.cam_se3_rig()[side] * p))
        .unwrap()
}

/// The checks every camera model passes; returns (row error, round trip,
/// point error) maxima.
fn check_model(name: &str, rig: &CalibratedRig) -> (f64, f64, f64) {
    let rect = rectify(rig, [0, 1], &RectifyOptions::default()).unwrap();
    let q = rect.disparity_to_depth();
    let (mut row, mut round_trip, mut point_err) = (0.0_f64, 0.0_f64, 0.0_f64);
    let pts = grid();
    for p in &pts {
        let (rl, rr) = (raw(rig, 0, p), raw(rig, 1, p));
        let (l, r) = (
            rect.rectified_pixel(0, &rl).expect("left rectified"),
            rect.rectified_pixel(1, &rr).expect("right rectified"),
        );
        row = row.max((l.y - r.y).abs());
        for (side, raw_px, rectified) in [(0, rl, l), (1, rr, r)] {
            let back = rect.raw_pixel(side, &rectified).expect("raw pixel");
            round_trip = round_trip.max((back - raw_px).norm());
        }

        let d = l.x - r.x;
        assert!(d > 0.0, "{name}: disparity {d}");
        let found = rect.point(&l, d).expect("point");
        let rel = (found - p).norm() / p.coords.norm();
        point_err = point_err.max(rel);
        let h = q * Vector4::new(l.x, l.y, d, 1.0);
        let via_q = rect.rect_r_cam(0).transpose() * Vec3::new(h.x / h.w, h.y / h.w, h.z / h.w);
        assert!((via_q - found.coords).norm() < 1e-9 * found.coords.norm());
    }
    eprintln!(
        "{name}: {} points, row error {row:.2e} px, raw round trip {round_trip:.2e} px, \
         point error {point_err:.2e} (relative)",
        pts.len()
    );
    assert!(row < 1e-6, "{name}: rows differ by {row}");
    assert!(round_trip < 1e-6, "{name}: round trip {round_trip}");
    assert!(point_err < 1e-9, "{name}: point error {point_err}");
    (row, round_trip, point_err)
}

#[test]
fn a_brown_conrady_pair_rectifies() {
    check_model(
        "pinhole + Brown-Conrady",
        &rig([
            brown_conrady(850.0, 845.0, 640.0, 480.0),
            brown_conrady(820.0, 815.0, 650.0, 470.0),
        ]),
    );
}

#[test]
fn a_fisheye_pair_rectifies() {
    check_model(
        "Kannala-Brandt fisheye",
        &rig([fisheye(300.0, 640.0, 480.0), fisheye(310.0, 635.0, 485.0)]),
    );
}

#[test]
fn a_scheimpflug_pair_rectifies() {
    check_model(
        "Scheimpflug",
        &rig([
            scheimpflug(
                800.0,
                640.0,
                ScheimpflugParams {
                    tilt_x: 0.09,
                    tilt_y: -0.04,
                },
            ),
            scheimpflug(
                790.0,
                635.0,
                ScheimpflugParams {
                    tilt_x: -0.06,
                    tilt_y: 0.05,
                },
            ),
        ]),
    );
}

/// Pixels with no source (outside the camera's domain) are 0; the others
/// carry the source value at the raw pixel they map to.
#[test]
fn rectify_image_samples_the_raw_pixel() {
    let r = rig([
        brown_conrady(850.0, 845.0, 640.0, 480.0),
        brown_conrady(820.0, 815.0, 650.0, 470.0),
    ]);
    let rect = rectify(&r, [0, 1], &RectifyOptions::default()).unwrap();
    // A linear ramp is reproduced exactly by bilinear sampling.
    let (w, h) = (1280, 960);
    let ramp = |x: f64, y: f64| 20.0 + 0.1 * x + 0.05 * y;
    let data = (0..w * h)
        .map(|i| ramp((i % w) as f64, (i / w) as f64).round() as u8)
        .collect();
    let src = GrayImage::new(w, h, data);
    // The reduced rectification (half size) fills its interior.
    let half = rect.rectify_image(0, &src, 640, 480);
    assert!(half.get(320, 240) > 0);
    // Full size: the value matches the ramp at the raw pixel.
    let full = rect.rectify_image(1, &src, 1280, 960);
    for (x, y) in [(640, 480), (200, 300), (900, 700)] {
        let raw = rect.raw_pixel(1, &Pt2::new(x as f64, y as f64)).unwrap();
        let want = ramp(raw.x, raw.y);
        assert!(
            (f64::from(full.get(x, y)) - want).abs() <= 1.01,
            "({x}, {y}): {} vs {want}",
            full.get(x, y)
        );
    }
}

fn rectification() -> StereoRectification {
    rectify(
        &rig([
            brown_conrady(850.0, 845.0, 640.0, 480.0),
            brown_conrady(820.0, 815.0, 650.0, 470.0),
        ]),
        [0, 1],
        &RectifyOptions::default(),
    )
    .unwrap()
}

/// The scatter of the depth that disparity noise alone causes, over the depth
/// σ `disparity_covariance` states: 1 up to the draws' sampling error, which
/// is `1/sqrt(2n)`, 0.35 % for 40 000 draws, so the gate of 2 % is about
/// six standard errors and holds for any seed.
#[test]
fn the_disparity_covariance_matches_the_scatter_of_the_depth() {
    let rect = rectification();
    let f_b = rect.k_rect().fx * rect.baseline();
    let r3 = rect.rect_r_cam(0).row(2).transpose();
    let sigma_d = 0.25;
    let draws = 40_000;
    let mut rng = Rng(11);
    for z in [0.7, 1.5, 3.0] {
        let left = Pt2::new(700.0, 520.0);
        let d = f_b / z;
        let stated_cov = rect.disparity_covariance(&left, d, sigma_d).unwrap();
        // Depth along the rectified z axis.
        let stated = (r3.transpose() * stated_cov * r3)[(0, 0)].sqrt();
        assert!(
            (stated - z * z * sigma_d / f_b).abs() < 1e-12 * stated,
            "depth sigma {stated} vs Z^2 sigma_d/(f B)"
        );

        let zs: Vec<f64> = (0..draws)
            .map(|_| {
                let p = rect.point(&left, d + sigma_d * rng.normal()).unwrap();
                r3.dot(&p.coords)
            })
            .collect();
        let mean = zs.iter().sum::<f64>() / draws as f64;
        let scatter =
            (zs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (draws - 1) as f64).sqrt();
        let ratio = scatter / stated;
        eprintln!(
            "depth {z} m: sigma {:.3} mm, scatter/stated {ratio:.3}",
            stated * 1e3
        );
        assert!((0.98..=1.02).contains(&ratio), "depth {z}: ratio {ratio}");
    }
}

#[test]
fn the_disparity_covariance_has_the_viewing_ray_as_its_axis() {
    let rect = rectification();
    let (left, d) = (Pt2::new(900.0, 300.0), 40.0);
    let cov = rect.disparity_covariance(&left, d, 0.2).unwrap();
    let p = rect.point(&left, d).unwrap().coords;
    // Rank 1 along the ray: Σ p ∝ p, and any perpendicular direction is null.
    let perp = p.cross(&Vec3::x());
    assert!((cov * perp).norm() < 1e-15);
    assert!((cov * p - p * (0.2 / d).powi(2) * p.norm_squared()).norm() < 1e-12);
    assert!(rect.disparity_covariance(&left, d, -1.0).is_none());
    assert!(rect.disparity_covariance(&left, 0.0, 0.2).is_none());
}

#[test]
fn a_scale_shrinks_the_rectified_image() {
    let r = rig([
        brown_conrady(850.0, 845.0, 640.0, 480.0),
        brown_conrady(820.0, 815.0, 650.0, 470.0),
    ]);
    let full = rectify(&r, [0, 1], &RectifyOptions::default()).unwrap();
    let mut opts = RectifyOptions::default();
    opts.scale = 0.25;
    let small = rectify(&r, [0, 1], &opts).unwrap();
    assert!((small.k_rect().fx - 0.25 * full.k_rect().fx).abs() < 1e-12);
    let p = Pt3::new(0.1, 0.05, 1.0);
    let a = full.rectified_pixel(0, &raw(&r, 0, &p)).unwrap();
    let b = small.rectified_pixel(0, &raw(&r, 0, &p)).unwrap();
    assert!((b.x - 0.25 * a.x).abs() < 1e-9 && (b.y - 0.25 * a.y).abs() < 1e-9);
}
