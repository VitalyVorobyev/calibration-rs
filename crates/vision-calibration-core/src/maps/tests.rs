//! Tests of the diagnostic maps.

use super::*;

// ─── A7: the grid ───────────────────────────────────────────────────────────

#[test]
fn grid_nodes_are_block_centres_row_major() {
    let grid = ImageGrid::new([1280, 960], 16).unwrap();
    assert_eq!((grid.columns(), grid.rows(), grid.len()), (80, 60, 4800));
    assert_eq!(grid.node(0, 0), [7.5, 7.5]);
    assert_eq!(grid.node(79, 59), [1271.5, 951.5]);
    let nodes: Vec<[f64; 2]> = grid.nodes().collect();
    assert_eq!(nodes.len(), grid.len());
    assert_eq!(nodes[1], [23.5, 7.5]);
    assert_eq!(nodes[80], [7.5, 23.5]);
    assert_eq!(nodes[4799], grid.node(79, 59));
}

#[test]
fn grid_partial_last_block_is_centred_on_its_pixels() {
    // 1000 = 62·16 + 8: the last column holds pixels 992..=999.
    let grid = ImageGrid::new([1000, 3], 16).unwrap();
    assert_eq!((grid.columns(), grid.rows()), (63, 1));
    assert_eq!(grid.node(62, 0), [995.5, 1.0]);
    // Stride 1: one node per pixel, at the pixel centre.
    let pixels = ImageGrid::new([3, 2], 1).unwrap();
    let nodes: Vec<[f64; 2]> = pixels.nodes().collect();
    assert_eq!(
        nodes,
        vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0]
        ]
    );
    // One block larger than the image: its node is the image centre.
    let one = ImageGrid::new([5, 3], 8).unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one.node(0, 0), [2.0, 1.0]);
}

#[test]
fn grid_with_columns_rounds_the_stride_up() {
    let grid = ImageGrid::with_columns([1280, 960], 64).unwrap();
    assert_eq!((grid.stride(), grid.columns(), grid.rows()), (20, 64, 48));
    let grid = ImageGrid::with_columns([1000, 750], 64).unwrap();
    assert_eq!((grid.stride(), grid.columns(), grid.rows()), (16, 63, 47));
    // More columns than pixels: stride 1, one node per pixel.
    let grid = ImageGrid::with_columns([10, 5], 64).unwrap();
    assert_eq!((grid.stride(), grid.columns(), grid.rows()), (1, 10, 5));
    assert_eq!(grid.image_size(), [10, 5]);
}

/// The layout of the viewer's dense-plane format: `columns × rows` cells of
/// `stride` pixels, the trailing cell partial.
#[test]
fn grid_matches_the_viewer_plane_layout() {
    for (size, stride) in [([1920, 1080], 7), ([640, 480], 32), ([4000, 3000], 15)] {
        let grid = ImageGrid::new(size, stride).unwrap();
        let columns = size[0].div_ceil(stride) as usize;
        let rows = size[1].div_ceil(stride) as usize;
        assert_eq!((grid.columns(), grid.rows()), (columns, rows));
        for (c, r) in [(0, 0), (columns - 1, rows - 1), (columns / 2, rows / 3)] {
            let [x, y] = grid.node(c, r);
            // Inside the image and within half a block of the cell's centre.
            assert!((0.0..=f64::from(size[0] - 1)).contains(&x));
            assert!((0.0..=f64::from(size[1] - 1)).contains(&y));
            let half = f64::from(stride) / 2.0;
            assert!((x - (c as f64 * f64::from(stride) + half - 0.5)).abs() <= half);
            assert!((y - (r as f64 * f64::from(stride) + half - 0.5)).abs() <= half);
        }
    }
}

#[test]
fn grid_rejects_empty_input() {
    assert!(ImageGrid::new([0, 10], 4).is_err());
    assert!(ImageGrid::new([10, 0], 4).is_err());
    assert!(ImageGrid::new([10, 10], 0).is_err());
    assert!(ImageGrid::with_columns([10, 10], 0).is_err());
    assert!(ImageGrid::with_columns([0, 10], 4).is_err());
}

// ─── distortion_map ─────────────────────────────────────────────────────────

use nalgebra::Point2;

use super::distortion_map::distortion_jacobian_det;
use crate::{
    BrownConrady5, CameraModel, CameraParams, DistortionParams, FxFyCxCy, IntrinsicsModel,
    KannalaBrandt4, ProjectionModel, ProjectionParams, RationalPolynomial, ScheimpflugParams,
    SensorParams, ThinPrism,
};

fn k(f: f64, cx: f64, cy: f64) -> FxFyCxCy<f64> {
    FxFyCxCy {
        fx: f,
        fy: f,
        cx,
        cy,
    }
}

fn brown(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> DistortionParams {
    DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1,
            k2,
            k3,
            p1,
            p2,
            iters: 0,
        },
    }
}

fn kannala_brandt(k1: f64, k2: f64, k3: f64, k4: f64) -> DistortionParams {
    DistortionParams::KannalaBrandt4 {
        params: KannalaBrandt4 {
            k1,
            k2,
            k3,
            k4,
            iters: 0,
        },
    }
}

/// The pixel the ray through `u` reaches in the reference camera `K∘P`,
/// through the camera model's own backprojection (an independent path to
/// the map's `u − total`).
fn ideal_pixel(model: &CameraModel, u: [f64; 2]) -> [f64; 2] {
    let ray = model.backproject_pixel(&Point2::new(u[0], u[1]));
    let n = model.proj.project(&(ray.origin + ray.direction));
    let q = model.k.sensor_to_pixel(&n);
    [q.x, q.y]
}

/// The ray the camera backprojects through `u`: its normalized point, and
/// whether it projects back to `u` within `1e-6` px.
fn backprojected(model: &CameraModel, u: [f64; 2]) -> (Point2<f64>, bool) {
    let px = Point2::new(u[0], u[1]);
    let ray = model.backproject_pixel(&px);
    let point = ray.origin + ray.direction;
    let n = model.proj.project(&point);
    let round_trips = model
        .project_point_c(&point)
        .is_some_and(|q| (q - px).norm() <= 1e-6);
    (n, round_trips)
}

/// Positive root of `g(r) = target` for `g` increasing on `[0, hi]`, by
/// bisection.
fn bisect(g: impl Fn(f64) -> f64, target: f64, hi: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, hi);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A1: no distortion and an identity sensor: every layer is zero.
#[test]
fn distortion_map_of_an_ideal_camera_is_zero() {
    for projection in [ProjectionParams::Pinhole, ProjectionParams::Equidistant] {
        let cam = CameraParams::new(
            projection,
            DistortionParams::None,
            SensorParams::Identity,
            k(500.0, 639.5, 479.5),
        );
        let grid = ImageGrid::new([1280, 960], 16).unwrap();
        let map = distortion_map(&cam, &grid).unwrap();
        assert_eq!(map.valid_nodes, grid.len());
        for plane in [&map.total, &map.distortion, &map.sensor] {
            assert!(plane.max_magnitude().unwrap() <= 1e-12, "{plane:?}");
        }
    }
}

/// A2: Brown-Conrady `k1` only on a pinhole camera, against an independent
/// inversion of `r (1 + k1 r²) = r_d` by bisection: barrel (its corners past
/// the fold) and pincushion.
#[test]
fn distortion_map_matches_the_radial_polynomial_inverse() {
    for (k1, f) in [(-0.1, 600.0), (0.15, 800.0)] {
        let cam = CameraParams::new(
            ProjectionParams::Pinhole,
            brown(k1, 0.0, 0.0, 0.0, 0.0),
            SensorParams::Identity,
            k(f, 639.5, 479.5),
        );
        let grid = ImageGrid::new([1280, 960], 8).unwrap();
        let map = distortion_map(&cam, &grid).unwrap();
        let g = |r: f64| r * (1.0 + k1 * r * r);
        // Barrel: the radius where g stops increasing, and the image radius it reaches.
        let (fold, fold_image) = if k1 < 0.0 {
            let fold = (1.0 / (-3.0 * k1)).sqrt();
            (fold, g(fold))
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
        let (mut checked, mut beyond) = (0, 0);
        for (i, [u, v]) in grid.nodes().enumerate() {
            let nd = [(u - 639.5) / f, (v - 479.5) / f];
            let rd = nd[0].hypot(nd[1]);
            if rd > fold_image * 1.001 {
                assert!(map.distortion.dx[i].is_nan(), "node {i} past the fold");
                beyond += 1;
                continue;
            }
            if rd > fold_image * 0.999 {
                continue;
            }
            assert!(map.distortion.dx[i].is_finite(), "node {i} inside the fold");
            // The root is below the fold (barrel), or below r_d (pincushion: g(r) ≥ r).
            let r = bisect(g, rd, if k1 < 0.0 { fold } else { rd });
            let scale = r / rd;
            let expected = [f * nd[0] * (1.0 - scale), f * nd[1] * (1.0 - scale)];
            let got = [map.distortion.dx[i], map.distortion.dy[i]];
            assert!(
                (got[0] - expected[0]).abs() <= 1e-9 && (got[1] - expected[1]).abs() <= 1e-9,
                "k1 = {k1}, node {i}: {got:?} vs {expected:?}"
            );
            assert_eq!([map.sensor.dx[i], map.sensor.dy[i]], [0.0, 0.0]);
            assert_eq!([map.total.dx[i], map.total.dy[i]], got);
            checked += 1;
        }
        assert!(checked > grid.len() / 2, "{checked}");
        assert_eq!(beyond > 0, k1 < 0.0, "only the barrel case folds");
    }
}

/// A3 and A6 over the camera-model matrix: the layers telescope
/// (`total = distortion + sensor`), `u − total` is the reference camera's
/// pixel of the backprojected ray, an identity sensor adds nothing, and a
/// Scheimpflug tilt adds nothing at the principal point.
#[test]
fn distortion_map_layers_telescope_for_every_camera_model() {
    let pinhole_distortions = [
        DistortionParams::None,
        brown(-0.12, 0.03, -0.002, 0.001, -0.0008),
        DistortionParams::Rational {
            params: RationalPolynomial {
                k1: -0.1,
                k2: 0.01,
                k3: 0.0,
                k4: 0.05,
                k5: 0.001,
                k6: 0.0,
                p1: 0.0005,
                p2: -0.0005,
                iters: 0,
            },
        },
        DistortionParams::ThinPrism {
            params: ThinPrism {
                k1: -0.12,
                k2: 0.03,
                k3: -0.002,
                p1: 0.001,
                p2: -0.0008,
                s1: 0.001,
                s2: -0.0005,
                s3: -0.001,
                s4: 0.0003,
                iters: 0,
            },
        },
        DistortionParams::Division { lambda: -0.05 },
        kannala_brandt(-0.05, 0.005, 0.0, 0.0),
    ];
    let fisheye_distortions = [
        DistortionParams::None,
        kannala_brandt(-0.01, 0.002, -0.0003, 0.00002),
    ];
    let sensors = [
        SensorParams::Identity,
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.02,
                tilt_y: 0.01,
            },
        },
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: -0.02,
                tilt_y: 0.01,
            },
        },
        SensorParams::Homography {
            h: [
                [1.002, 0.001, 0.0005],
                [-0.001, 0.998, -0.0004],
                [0.01, -0.008, 1.0],
            ],
        },
    ];
    // Stride 32 puts node (20, 15) on the principal point.
    let grid = ImageGrid::new([1280, 960], 32).unwrap();
    let centre = 15 * grid.columns() + 20;
    assert_eq!(grid.nodes().nth(centre), Some([655.5, 495.5]));
    let cases = pinhole_distortions
        .iter()
        .map(|d| (ProjectionParams::Pinhole, d.clone(), 800.0))
        .chain(
            fisheye_distortions
                .iter()
                .map(|d| (ProjectionParams::Equidistant, d.clone(), 400.0)),
        );
    for (projection, distortion, f) in cases {
        for sensor in &sensors {
            let cam = CameraParams::new(
                projection.clone(),
                distortion.clone(),
                sensor.clone(),
                k(f, 655.5, 495.5),
            );
            let model = cam.build().unwrap();
            let label = format!("{:?} {:?} {:?}", cam.projection, cam.distortion, cam.sensor);
            let map = distortion_map(&cam, &grid).unwrap();
            assert!(
                map.valid_nodes * 10 >= grid.len() * 9,
                "{label}: {} of {} nodes",
                map.valid_nodes,
                grid.len()
            );
            let mut valid = 0;
            for (i, u) in grid.nodes().enumerate() {
                let total = [map.total.dx[i], map.total.dy[i]];
                if total[0].is_nan() {
                    assert!(map.distortion.dx[i].is_nan() && map.sensor.dy[i].is_nan());
                    continue;
                }
                valid += 1;
                let sum = [
                    map.distortion.dx[i] + map.sensor.dx[i],
                    map.distortion.dy[i] + map.sensor.dy[i],
                ];
                assert!(
                    (total[0] - sum[0]).abs() <= 1e-9 && (total[1] - sum[1]).abs() <= 1e-9,
                    "{label}: node {i}"
                );
                // A6: u − total(u) is where the reference camera puts the ray.
                let ideal = ideal_pixel(&model, u);
                assert!(
                    (u[0] - total[0] - ideal[0]).abs() <= 1e-6
                        && (u[1] - total[1] - ideal[1]).abs() <= 1e-6,
                    "{label}: node {i} at {u:?}: u − total = {:?}, ideal {ideal:?}",
                    [u[0] - total[0], u[1] - total[1]]
                );
                if matches!(sensor, SensorParams::Identity) {
                    assert_eq!([map.sensor.dx[i], map.sensor.dy[i]], [0.0, 0.0], "{label}");
                }
            }
            assert_eq!(valid, map.valid_nodes);
            if matches!(sensor, SensorParams::Scheimpflug { .. }) {
                assert!(
                    map.sensor.dx[centre].abs() <= 1e-9 && map.sensor.dy[centre].abs() <= 1e-9,
                    "{label}: the tilt moves the principal point"
                );
                assert!(map.sensor.max_magnitude().unwrap() > 0.1, "{label}");
            }
        }
    }
}

/// A4: a fisheye (equidistant, Kannala-Brandt with `k1 < 0`): the radial
/// distortion is `f·θ·(k1θ² + …)` at the node's incidence angle, with no
/// tangential part, beyond 90° too; nodes past the polynomial's fold are
/// NaN.
#[test]
fn fisheye_distortion_is_the_polynomial_departure_from_f_theta() {
    let (f, c, k1) = (500.0, 799.5, -0.05);
    let cam = CameraParams::new(
        ProjectionParams::Equidistant,
        kannala_brandt(k1, 0.0, 0.0, 0.0),
        SensorParams::Identity,
        k(f, c, c),
    );
    let grid = ImageGrid::new([1600, 1600], 10).unwrap();
    let map = distortion_map(&cam, &grid).unwrap();
    let g = |t: f64| t * (1.0 + k1 * t * t);
    let fold = (1.0 / (-3.0 * k1)).sqrt();
    let fold_image = g(fold);
    let (mut checked, mut wide, mut beyond) = (0, 0, 0);
    for (i, [u, v]) in grid.nodes().enumerate() {
        let (x, y) = (u - c, v - c);
        let theta_d = x.hypot(y) / f;
        if theta_d > fold_image * 1.001 {
            assert!(map.distortion.dx[i].is_nan(), "node {i} past the fold");
            beyond += 1;
            continue;
        }
        if theta_d > fold_image * 0.999 {
            continue;
        }
        let theta = bisect(g, theta_d, fold);
        let radial = f * (theta_d - theta);
        // The polynomial's own form of the same departure.
        assert!((radial - f * theta * k1 * theta * theta).abs() <= 1e-9);
        let (ux, uy) = (x / (theta_d * f), y / (theta_d * f));
        let got = [map.distortion.dx[i], map.distortion.dy[i]];
        let along = got[0] * ux + got[1] * uy;
        let across = -got[0] * uy + got[1] * ux;
        assert!(
            (along - radial).abs() <= 1e-9 && across.abs() <= 1e-9,
            "node {i}, θ = {theta}: {got:?}, radial {radial}"
        );
        checked += 1;
        if theta > 95f64.to_radians() {
            wide += 1;
        }
    }
    assert!(
        checked > 0 && wide > 0 && beyond > 0,
        "{checked} {wide} {beyond}"
    );
}

/// A5: a pinhole Brown-Conrady fold inside the image. Every node with a
/// value has its ray inside the fold radius, with `det ∂D/∂n > 0` there;
/// nodes past the fold image are NaN. The determinant at the ray alone
/// would pass rays past the fold that project back to their node exactly:
/// on the branch through the axis (`k1` only) and on the branch where the
/// polynomial rises again (`k1 < 0 < k2`).
#[test]
fn distortion_map_refuses_rays_past_the_fold() {
    for (k1, k2, fold) in [(-0.3f64, 0.0f64, (1.0f64 / 0.9).sqrt()), (-0.5, 0.1, 1.0)] {
        let cam = CameraParams::new(
            ProjectionParams::Pinhole,
            brown(k1, k2, 0.0, 0.0, 0.0),
            SensorParams::Identity,
            k(500.0, 639.5, 479.5),
        );
        let model = cam.build().unwrap();
        let grid = ImageGrid::new([1280, 960], 8).unwrap();
        let map = distortion_map(&cam, &grid).unwrap();
        let g = |r: f64| r * (1.0 + k1 * r * r + k2 * r.powi(4));
        let dg = |r: f64| 1.0 + 3.0 * k1 * r * r + 5.0 * k2 * r.powi(4);
        let fold_image = g(fold);
        let mut ray_only_beyond = 0;
        for (i, u) in grid.nodes().enumerate() {
            let (n, round_trips) = backprojected(&model, u);
            let r = n.coords.norm();
            if map.total.dx[i].is_finite() {
                assert!(r < fold, "node {i}: ray at r = {r}, fold at {fold}");
                assert!(g(r) / r * dg(r) > 0.0, "node {i}: folded ray");
            } else if round_trips && distortion_jacobian_det(&model.dist, &n) > 1e-6 && r > fold {
                ray_only_beyond += 1;
            }
            let rd = ((u[0] - 639.5) / 500.0).hypot((u[1] - 479.5) / 500.0);
            if rd > fold_image * 1.001 {
                assert!(map.total.dx[i].is_nan(), "node {i} past the fold image");
            } else if rd < fold_image * 0.999 {
                assert!(
                    map.total.dx[i].is_finite(),
                    "node {i} inside the fold image"
                );
            }
        }
        assert!(
            ray_only_beyond > 1000,
            "k1 = {k1}, k2 = {k2}: the determinant at the ray alone passes \
             {ray_only_beyond} nodes past the fold"
        );
    }
}

/// A6: barrel distortion points inward at the corners, pincushion outward.
#[test]
fn distortion_map_sign_is_actual_minus_ideal() {
    for (k1, inward) in [(-0.1, true), (0.1, false)] {
        let cam = CameraParams::new(
            ProjectionParams::Pinhole,
            brown(k1, 0.0, 0.0, 0.0, 0.0),
            SensorParams::Identity,
            k(1000.0, 639.5, 479.5),
        );
        let grid = ImageGrid::new([1280, 960], 16).unwrap();
        let map = distortion_map(&cam, &grid).unwrap();
        let nodes: Vec<[f64; 2]> = grid.nodes().collect();
        let corners = [
            0,
            grid.columns() - 1,
            grid.len() - grid.columns(),
            grid.len() - 1,
        ];
        for i in corners {
            let [u, v] = nodes[i];
            let outward = map.total.dx[i] * (u - 639.5) + map.total.dy[i] * (v - 479.5);
            assert_eq!(outward < 0.0, inward, "k1 = {k1}, corner {i}");
        }
    }
}

/// A sensor homography that sends part of the image to infinity: those
/// nodes, and the ones past them, have no value; a singular one is an
/// error.
#[test]
fn distortion_map_nodes_at_infinity_are_nan() {
    // H⁻¹ has third row (−0.8, 0, 1): depth zero at sensor x = 1.25, pixel 1264.5.
    let cam = CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::None,
        SensorParams::Homography {
            h: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.0, 1.0]],
        },
        k(500.0, 639.5, 479.5),
    );
    let grid = ImageGrid::new([1600, 960], 8).unwrap();
    let map = distortion_map(&cam, &grid).unwrap();
    for (i, [u, _]) in grid.nodes().enumerate() {
        if u > 1264.0 {
            assert!(map.total.dx[i].is_nan(), "node {i} at x = {u}");
        } else if u < 1200.0 {
            assert!(map.total.dx[i].is_finite(), "node {i} at x = {u}");
        }
    }
    let singular = cam.with_sensor(SensorParams::Homography { h: [[0.0; 3]; 3] });
    assert!(matches!(
        distortion_map(&singular, &grid),
        Err(crate::Error::Singular)
    ));
}

/// A8: the same camera and grid give bit-identical planes.
#[test]
fn distortion_map_is_deterministic() {
    let cam = CameraParams::new(
        ProjectionParams::Pinhole,
        brown(-0.2, 0.05, 0.0, 0.001, 0.0),
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: 0.03,
                tilt_y: -0.01,
            },
        },
        k(700.0, 640.0, 480.0),
    );
    let grid = ImageGrid::with_columns([1280, 960], 64).unwrap();
    let a = distortion_map(&cam, &grid).unwrap();
    let b = distortion_map(&cam, &grid).unwrap();
    for (x, y) in [
        (&a.total, &b.total),
        (&a.distortion, &b.distortion),
        (&a.sensor, &b.sensor),
    ] {
        assert_eq!(bits(&x.dx), bits(&y.dx));
        assert_eq!(bits(&x.dy), bits(&y.dy));
    }
    assert_eq!(a.valid_nodes, b.valid_nodes);
    assert_eq!(a.grid, grid);
}

#[test]
fn displacement_plane_magnitude_skips_nan() {
    let plane = DisplacementPlane {
        dx: vec![3.0, f64::NAN, -1.0],
        dy: vec![4.0, 0.0, 0.0],
    };
    let m = plane.magnitude();
    assert_eq!(m[0], 5.0);
    assert!(m[1].is_nan());
    assert_eq!(plane.max_magnitude(), Some(5.0));
    let empty = DisplacementPlane {
        dx: vec![f64::NAN],
        dy: vec![f64::NAN],
    };
    assert_eq!(empty.max_magnitude(), None);
}
