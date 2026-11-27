//! Tests of [`projection_difference`] on synthetic ground truth.

use nalgebra::{Matrix2x3, Matrix3, Point2, UnitQuaternion, Vector2, Vector3};

use super::*;
use crate::{
    BrownConrady5, CameraParams, DistortionParams, Error, FxFyCxCy, KannalaBrandt4,
    ProjectionModel, ProjectionParams, RationalPolynomial, ScheimpflugParams, SensorParams,
    ThinPrism,
};

const SIZE: [u32; 2] = [1280, 960];

fn k(f: f64, cx: f64, cy: f64) -> FxFyCxCy<f64> {
    FxFyCxCy {
        fx: f,
        fy: f,
        cx,
        cy,
    }
}

fn brown(k1: f64, k2: f64, p1: f64, p2: f64) -> DistortionParams {
    DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1,
            k2,
            k3: 0.0,
            p1,
            p2,
            iters: 8,
        },
    }
}

fn pinhole(distortion: DistortionParams, intrinsics: FxFyCxCy<f64>) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        distortion,
        SensorParams::Identity,
        intrinsics,
    )
}

fn grid() -> ImageGrid {
    // Stride 20: the nodes are symmetric about the image centre (639.5, 479.5).
    ImageGrid::with_columns(SIZE, 64).unwrap()
}

fn options(compensation: Compensation) -> ProjectionDifferenceOptions {
    ProjectionDifferenceOptions::new(compensation, FitRegion::WholeImage)
}

/// The row-major homography of a rotation.
fn homography(rotation: &UnitQuaternion<f64>) -> [[f64; 3]; 3] {
    let m = rotation.to_rotation_matrix().into_inner();
    [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ]
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Every camera model the maps support.
fn model_matrix() -> Vec<CameraParams> {
    let intrinsics = k(800.0, 641.0, 478.0);
    let tilt = ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: -0.015,
    };
    vec![
        pinhole(DistortionParams::None, intrinsics),
        pinhole(brown(-0.12, 0.03, 4e-4, -3e-4), intrinsics),
        pinhole(
            DistortionParams::Rational {
                params: RationalPolynomial {
                    k1: -0.1,
                    k2: 0.02,
                    k3: 0.0,
                    k4: 0.05,
                    k5: 0.0,
                    k6: 0.0,
                    p1: 2e-4,
                    p2: 1e-4,
                    iters: 8,
                },
            },
            intrinsics,
        ),
        pinhole(
            DistortionParams::ThinPrism {
                params: ThinPrism {
                    k1: -0.1,
                    k2: 0.02,
                    k3: 0.0,
                    p1: 2e-4,
                    p2: 1e-4,
                    s1: 1e-3,
                    s2: 0.0,
                    s3: -1e-3,
                    s4: 0.0,
                    iters: 8,
                },
            },
            intrinsics,
        ),
        pinhole(DistortionParams::Division { lambda: -0.08 }, intrinsics),
        CameraParams::new(
            ProjectionParams::Equidistant,
            DistortionParams::KannalaBrandt4 {
                params: KannalaBrandt4 {
                    k1: -0.02,
                    k2: 0.002,
                    k3: 0.0,
                    k4: 0.0,
                    iters: 8,
                },
            },
            SensorParams::Identity,
            k(400.0, 641.0, 478.0),
        ),
        pinhole(brown(-0.12, 0.03, 0.0, 0.0), intrinsics)
            .with_sensor(SensorParams::Scheimpflug { params: tilt }),
    ]
}

/// a camera against itself: zero (the round trip of its own inverse),
/// no rotation, and the nodes of its distortion map, for every model.
#[test]
fn identical_cameras_differ_by_nothing() {
    for camera in model_matrix() {
        let diff = projection_difference(
            &camera,
            &camera,
            &grid(),
            &ProjectionDifferenceOptions::default(),
        )
        .unwrap();
        let map = distortion_map(&camera, &grid()).unwrap();
        assert_eq!(diff.valid_nodes, map.valid_nodes, "{camera:?}");
        assert_eq!(diff.fit_nodes, diff.valid_nodes);
        assert!(
            diff.max_valid_px <= 1e-9,
            "{camera:?}: {}",
            diff.max_valid_px
        );
        assert!(diff.rotation_angle() <= 1e-12, "{}", diff.rotation_angle());
        for i in 0..grid().len() {
            assert_eq!(
                diff.difference.dx[i].is_nan(),
                map.total.dx[i].is_nan(),
                "node {i}"
            );
        }
    }
}

/// `b` is `a` with its frame rotated by `R₀` (a pinhole camera whose
/// sensor homography is `R₀`, so `b(X) = a(R₀·X)`): the fit recovers
/// `camb_so3_cama = R₀⁻¹` and removes the field to rounding; without
/// compensation the field is the rotation's homography `K·π(R₀·K⁻¹·u) − u`.
#[test]
fn a_rotated_frame_is_compensated_exactly() {
    let intrinsics = k(800.0, 641.0, 478.0);
    let a = pinhole(DistortionParams::None, intrinsics);
    for (degrees, axis) in [
        (0.1, Vector3::new(1.0, 0.0, 0.0)),
        (1.0, Vector3::new(0.3, -1.0, 0.2)),
        (5.0, Vector3::new(-0.5, 0.4, 1.0)),
        (15.0, Vector3::new(1.0, 1.0, 0.0)),
    ] {
        let r0 = UnitQuaternion::from_scaled_axis(axis.normalize() * f64::to_radians(degrees));
        let b = a
            .clone()
            .with_sensor(SensorParams::Homography { h: homography(&r0) });
        let diff = projection_difference(&a, &b, &grid(), &ProjectionDifferenceOptions::default())
            .unwrap();
        let error = (diff.camb_so3_cama * r0).angle();
        assert!(error <= 1e-10, "{degrees}°: rotation off by {error:e} rad");
        assert!(diff.max_fit_px <= 1e-8, "{degrees}°: {}", diff.max_fit_px);
        assert_eq!(diff.valid_nodes, grid().len());

        let raw = projection_difference(&a, &b, &grid(), &options(Compensation::None)).unwrap();
        assert_eq!(raw.rotation_angle(), 0.0);
        let r = r0.to_rotation_matrix().into_inner();
        for (i, [x, y]) in grid().nodes().enumerate() {
            let ray = Vector3::new((x - 641.0) / 800.0, (y - 478.0) / 800.0, 1.0);
            let q = r * ray;
            let expected = [800.0 * q.x / q.z + 641.0 - x, 800.0 * q.y / q.z + 478.0 - y];
            assert!((raw.difference.dx[i] - expected[0]).abs() <= 1e-8);
            assert!((raw.difference.dy[i] - expected[1]).abs() <= 1e-8);
        }
        eprintln!(
            "{degrees}°: rotation error {error:.1e} rad, compensated max {:.1e} px, \
             uncompensated max {:.1} px",
            diff.max_fit_px, raw.max_valid_px
        );
    }
}

/// the sign is `b − a` at the same ray: `b` with `cx + 1` is one
/// pixel to the right everywhere, whatever the distortion.
#[test]
fn the_sign_is_other_minus_reference() {
    let a = pinhole(brown(-0.12, 0.03, 4e-4, -3e-4), k(800.0, 641.0, 478.0));
    let b = pinhole(brown(-0.12, 0.03, 4e-4, -3e-4), k(800.0, 642.0, 478.0));
    let diff = projection_difference(&a, &b, &grid(), &options(Compensation::None)).unwrap();
    for i in 0..grid().len() {
        assert!((diff.difference.dx[i] - 1.0).abs() <= 1e-9);
        assert!(diff.difference.dy[i].abs() <= 1e-9);
    }
    assert!((diff.rms_fit_px - 1.0).abs() <= 1e-9);
}

/// a focal-length change is a scale no rotation absorbs: on a grid
/// symmetric about the principal point the fitted rotation is zero and the
/// field is exactly `ε·(u − c)`.
#[test]
fn a_focal_change_leaves_the_radial_scale() {
    let c = [639.5, 479.5];
    let eps = 2e-3;
    let a = pinhole(brown(-0.12, 0.03, 0.0, 0.0), k(800.0, c[0], c[1]));
    let b = pinhole(
        brown(-0.12, 0.03, 0.0, 0.0),
        k(800.0 * (1.0 + eps), c[0], c[1]),
    );
    let diff =
        projection_difference(&a, &b, &grid(), &ProjectionDifferenceOptions::default()).unwrap();
    assert!(diff.rotation_angle() <= 1e-12, "{}", diff.rotation_angle());
    let mut worst = 0.0f64;
    for (i, [x, y]) in grid().nodes().enumerate() {
        worst = worst
            .max((diff.difference.dx[i] - eps * (x - c[0])).abs())
            .max((diff.difference.dy[i] - eps * (y - c[1])).abs());
    }
    assert!(worst <= 1e-9, "{worst:e}");
    eprintln!(
        "focal ×(1 + {eps}): field off ε(u − c) by at most {worst:.1e} px; max {:.3} px",
        diff.max_valid_px
    );
}

/// a change `κ` of `k1` on a radially symmetric lens: zero rotation
/// on a symmetric grid and the field `f·n·κ·|n|²` exactly, `n` the
/// undistorted normalized point of the reference camera's ray.
#[test]
fn a_k1_change_leaves_the_radial_term() {
    let c = [639.5, 479.5];
    let kappa = -0.005;
    let a = pinhole(brown(-0.12, 0.03, 0.0, 0.0), k(800.0, c[0], c[1]));
    let b = pinhole(brown(-0.12 + kappa, 0.03, 0.0, 0.0), k(800.0, c[0], c[1]));
    let diff =
        projection_difference(&a, &b, &grid(), &ProjectionDifferenceOptions::default()).unwrap();
    assert!(diff.rotation_angle() <= 1e-12, "{}", diff.rotation_angle());
    let model = a.build().unwrap();
    let mut worst = 0.0f64;
    for (i, [x, y]) in grid().nodes().enumerate() {
        let ray = model.backproject_pixel(&Point2::new(x, y));
        let n = model.proj.project(&(ray.origin + ray.direction));
        let r2 = n.x * n.x + n.y * n.y;
        worst = worst
            .max((diff.difference.dx[i] - 800.0 * n.x * kappa * r2).abs())
            .max((diff.difference.dy[i] - 800.0 * n.y * kappa * r2).abs());
    }
    assert!(worst <= 1e-9, "{worst:e}");
    eprintln!(
        "k1 {kappa:+}: field off f·n·κ·r² by at most {worst:.1e} px; max {:.3} px",
        diff.max_valid_px
    );
}

/// a principal-point shift `δ` against a fitted rotation. To first
/// order a rotation `ω` of the ray moves a pinhole pixel by `J(u)·ω` (the
/// rotational flow, `J = f·[[−xy, 1 + x², −y], [−(1 + y²), xy, x]]`), so the
/// compensated field is the least-squares remainder `δ − J·ω*`,
/// `ω* = (ΣJᵀJ)⁻¹ΣJᵀδ`, and the fitted rotation is `−ω*`. The departure from
/// it is second order in `δ`: halving `δ` quarters it.
#[test]
fn a_principal_point_shift_leaves_the_linear_remainder() {
    let (f, c) = (800.0, [639.5, 479.5]);
    let flow = |[u, v]: [f64; 2]| {
        let (x, y) = ((u - c[0]) / f, (v - c[1]) / f);
        Matrix2x3::new(-x * y, 1.0 + x * x, -y, -(1.0 + y * y), x * y, x) * f
    };
    // [field departure, rotation departure] from the linear prediction.
    let departure = |delta: Vector2<f64>| {
        let a = pinhole(DistortionParams::None, k(f, c[0], c[1]));
        let b = pinhole(DistortionParams::None, k(f, c[0] + delta.x, c[1] + delta.y));
        let diff = projection_difference(&a, &b, &grid(), &ProjectionDifferenceOptions::default())
            .unwrap();
        let mut normal = Matrix3::zeros();
        let mut rhs = Vector3::zeros();
        for node in grid().nodes() {
            let j = flow(node);
            normal += j.transpose() * j;
            rhs += j.transpose() * delta;
        }
        let omega = normal.cholesky().unwrap().solve(&rhs);
        let (mut worst, mut largest, mut sum) = (0.0f64, 0.0f64, 0.0);
        for (i, node) in grid().nodes().enumerate() {
            let expected = delta - flow(node) * omega;
            let got = Vector2::new(diff.difference.dx[i], diff.difference.dy[i]);
            worst = worst.max((got - expected).norm());
            largest = largest.max(expected.norm());
            sum += expected.norm_squared();
        }
        let rms = (sum / grid().len() as f64).sqrt();
        assert!((diff.rms_fit_px - rms).abs() <= 2.0 * worst);
        // The rotation absorbs most of the shift, not all of it.
        assert!(rms < 0.5 * delta.norm() && rms > 0.05 * delta.norm());
        let omega_error = (diff.camb_so3_cama.scaled_axis() + omega).norm();
        eprintln!(
            "δ = {:?} px: remainder rms {rms:.4} px, max {largest:.4} px; off the linear \
             prediction by {worst:.2e} px, rotation by {omega_error:.2e} rad (|ω*| {:.2e})",
            [delta.x, delta.y],
            omega.norm()
        );
        [worst, omega_error]
    };
    let delta = Vector2::new(0.2, -0.1);
    let full = departure(delta);
    let half = departure(delta * 0.5);
    for (at_full, at_half) in full.into_iter().zip(half) {
        let ratio = at_full / at_half;
        assert!((3.5..=4.5).contains(&ratio), "{at_full:e} / {at_half:e}");
    }
    // Second order: about |δ|²/f, times the corner factors (1 + x²)².
    assert!(full[0] <= 5.0 * delta.norm_squared() / f, "{:e}", full[0]);
}

/// `b` folds inside `a`'s image: past `b`'s fold the nodes are NaN
/// (never a value from the outer branch) and take no part in the fit.
#[test]
fn nodes_past_the_other_cameras_fold_are_nan_and_unfitted() {
    // a: an ideal wide pinhole; b: strong barrel, fold where
    // 1 + 3·k1·r² = 0, r ≈ 1.054, inside a's corners (r up to 1.6).
    let k1 = -0.3;
    let a = pinhole(DistortionParams::None, k(400.0, 500.0, 430.0));
    let b = pinhole(brown(k1, 0.0, 0.0, 0.0), k(400.0, 500.0, 430.0));
    let fold = (-1.0 / (3.0 * k1)).sqrt();
    let raw = projection_difference(&a, &b, &grid(), &options(Compensation::None)).unwrap();
    let (mut inside, mut beyond) = (0, 0);
    for (i, [x, y]) in grid().nodes().enumerate() {
        let (nx, ny) = ((x - 500.0) / 400.0, (y - 430.0) / 400.0);
        let r = nx.hypot(ny);
        if r < 0.99 * fold {
            inside += 1;
            let expected = [400.0 * nx * k1 * r * r, 400.0 * ny * k1 * r * r];
            assert!((raw.difference.dx[i] - expected[0]).abs() <= 1e-9);
            assert!((raw.difference.dy[i] - expected[1]).abs() <= 1e-9);
        } else if r > 1.01 * fold {
            beyond += 1;
            assert!(
                raw.difference.dx[i].is_nan() && !raw.fitted[i],
                "node {i}, r {r}"
            );
        }
    }
    assert!(inside > 1000 && beyond > 100, "{inside} {beyond}");

    // The rotation fit sees only the nodes with a value: fitting on exactly
    // those nodes gives the same rotation, bit for bit.
    let whole =
        projection_difference(&a, &b, &grid(), &ProjectionDifferenceOptions::default()).unwrap();
    let valid: Vec<[f64; 2]> = grid()
        .nodes()
        .enumerate()
        .filter(|(i, _)| whole.fitted[*i])
        .map(|(_, u)| u)
        .collect();
    let covered = projection_difference(
        &a,
        &b,
        &grid(),
        &ProjectionDifferenceOptions::new(Compensation::Rotation, FitRegion::Covered(valid)),
    )
    .unwrap();
    assert_eq!(whole.camb_so3_cama, covered.camb_so3_cama);
    assert_eq!(whole.fit_nodes, whole.valid_nodes);
    assert!(whole.rms_fit_px.is_finite() && whole.max_valid_px.is_finite());
}

/// a fisheye reference against a pinhole of the same lens centre:
/// the rays the fisheye sees beyond 90° have no pinhole projection (NaN),
/// and a fit on the central region recovers an `f·tanθ` vs `f·θ` field.
#[test]
fn a_fisheye_reference_and_a_pinhole_share_its_grid() {
    let (f, c) = (300.0, [639.5, 479.5]);
    let a = CameraParams::new(
        ProjectionParams::Equidistant,
        DistortionParams::None,
        SensorParams::Identity,
        k(f, c[0], c[1]),
    );
    let b = pinhole(DistortionParams::None, k(f, c[0], c[1]));
    let central: Vec<[f64; 2]> = grid()
        .nodes()
        .filter(|[x, y]| (x - c[0]).hypot(y - c[1]) < 0.5 * f)
        .collect();
    let diff = projection_difference(
        &a,
        &b,
        &grid(),
        &ProjectionDifferenceOptions::new(Compensation::Rotation, FitRegion::Covered(central)),
    )
    .unwrap();
    assert!(diff.rotation_angle() <= 1e-12);
    for (i, [x, y]) in grid().nodes().enumerate() {
        let rho = (x - c[0]).hypot(y - c[1]);
        let theta = rho / f;
        if theta > std::f64::consts::FRAC_PI_2 + 1e-9 {
            assert!(diff.difference.dx[i].is_nan(), "node {i}: θ {theta}");
        } else if theta < 1.4 {
            // b puts the ray at f·tanθ, a at f·θ: radial, outward.
            let expected = f * theta.tan() - rho;
            let got = diff.difference.dx[i].hypot(diff.difference.dy[i]);
            assert!(
                (got - expected).abs() <= 1e-6 * (1.0 + expected),
                "{got} {expected}"
            );
        }
    }
    assert!(diff.valid_nodes < grid().len() && diff.fit_nodes < diff.valid_nodes);
}

/// the covered region is the blocks the pixels fall in; off-image and
/// non-finite pixels are ignored; fewer than three fitted nodes is an error.
#[test]
fn the_covered_region_is_honoured() {
    let a = pinhole(brown(-0.12, 0.03, 0.0, 0.0), k(800.0, 641.0, 478.0));
    let b = pinhole(brown(-0.125, 0.03, 0.0, 0.0), k(801.0, 640.0, 479.0));
    let g = grid();
    // Pixel (19.4, 0) is in block 0; 19.6 rounds to pixel 20, block 1; the
    // bottom-right pixel's block is the last.
    let pixels = vec![
        [19.4, 0.0],
        [19.6, 0.0],
        [1279.4, 959.4],
        [-0.6, 10.0],
        [1279.5, 10.0],
        [f64::NAN, 5.0],
        [300.0, f64::INFINITY],
    ];
    let diff = projection_difference(
        &a,
        &b,
        &g,
        &ProjectionDifferenceOptions::new(Compensation::Rotation, FitRegion::Covered(pixels)),
    )
    .unwrap();
    let fitted: Vec<usize> = (0..g.len()).filter(|&i| diff.fitted[i]).collect();
    assert_eq!(fitted, vec![0, 1, g.len() - 1]);
    assert_eq!(diff.fit_nodes, 3);
    assert_eq!(diff.valid_nodes, g.len());

    let two = FitRegion::Covered(vec![[10.0, 10.0], [30.0, 10.0]]);
    for compensation in [Compensation::Rotation, Compensation::None] {
        let err = projection_difference(
            &a,
            &b,
            &g,
            &ProjectionDifferenceOptions::new(compensation, two.clone()),
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InsufficientData { need: 3, got: 2 }),
            "{err:?}"
        );
    }
}

/// two calls give the same result, bit for bit.
#[test]
fn projection_difference_is_deterministic() {
    for (a, b) in model_matrix().iter().zip(model_matrix().iter().rev()) {
        let Ok(first) = projection_difference(a, b, &grid(), &Default::default()) else {
            continue;
        };
        let second = projection_difference(a, b, &grid(), &Default::default()).unwrap();
        assert_eq!(bits(&first.difference.dx), bits(&second.difference.dx));
        assert_eq!(bits(&first.difference.dy), bits(&second.difference.dy));
        assert_eq!(first.camb_so3_cama, second.camb_so3_cama);
        assert_eq!(first.rms_fit_px.to_bits(), second.rms_fit_px.to_bits());
    }
}

/// The fitted rotation is the least-squares optimum: no small rotation of it
/// lowers the RMS over the fitted nodes.
#[test]
fn the_fitted_rotation_is_a_minimum() {
    let a = pinhole(brown(-0.12, 0.03, 4e-4, -3e-4), k(800.0, 641.0, 478.0));
    let b = CameraParams::new(
        ProjectionParams::Equidistant,
        DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: 0.25,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
                iters: 8,
            },
        },
        SensorParams::Identity,
        k(790.0, 644.0, 475.0),
    );
    let diff = projection_difference(&a, &b, &grid(), &Default::default()).unwrap();
    let model_a = a.build().unwrap();
    let model_b = b.build().unwrap();
    let rms = |rotation: &UnitQuaternion<f64>| {
        let mut sum = 0.0;
        for (i, [x, y]) in grid().nodes().enumerate() {
            if !diff.fitted[i] {
                continue;
            }
            let ray = model_a.backproject_pixel(&Point2::new(x, y));
            let q = model_b
                .project_point_c(&(rotation * ray.direction))
                .unwrap();
            sum += (q.x - x).powi(2) + (q.y - y).powi(2);
        }
        (sum / diff.fit_nodes as f64).sqrt()
    };
    let best = rms(&diff.camb_so3_cama);
    assert!((best - diff.rms_fit_px).abs() <= 1e-9 * (1.0 + best));
    for axis in [Vector3::x(), Vector3::y(), Vector3::z()] {
        for sign in [-1.0, 1.0] {
            let nudge = UnitQuaternion::from_scaled_axis(axis * sign * 1e-5);
            assert!(rms(&(nudge * diff.camb_so3_cama)) > best);
        }
    }
    eprintln!(
        "pinhole BC5 vs KB4: rms {best:.3} px over {} nodes, rotation {:.2e} rad",
        diff.fit_nodes,
        diff.rotation_angle()
    );
}
