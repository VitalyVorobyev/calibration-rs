//! Every public solver rejects a NaN or infinite coordinate with a typed
//! error. Without the check, the unbounded SVD and Schur iterations behind
//! `fundamental_7point` and `p3p` never return on such input.

use vision_calibration_core::{FxFyCxCy, Iso3, Mat3, Pt2, Pt3, Vec3};
use vision_geometry::GeometryError;
use vision_geometry::camera_matrix::dlt_camera_matrix;
use vision_geometry::epipolar::{
    decompose_essential, essential_5point, fundamental_7point, fundamental_8point,
};
use vision_geometry::homography::dlt_homography;
use vision_geometry::pnp;
use vision_geometry::triangulation::triangulate_point_linear;

fn image_points(n: usize) -> Vec<Pt2> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            Pt2::new(t * 0.37 + t * t * 0.05, (t * 1.3).sin())
        })
        .collect()
}

fn with_nan(mut pts: Vec<Pt2>) -> Vec<Pt2> {
    pts[2].x = f64::NAN;
    pts
}

fn world_points(n: usize) -> Vec<Pt3> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            Pt3::new(t * 0.1, t.sin() * 0.2, 1.0 + (t * 0.7).cos() * 0.3)
        })
        .collect()
}

fn k() -> FxFyCxCy<f64> {
    FxFyCxCy {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
    }
}

fn assert_invalid<T: std::fmt::Debug>(name: &str, r: Result<T, GeometryError>) {
    assert!(
        matches!(r, Err(GeometryError::InvalidInput { .. })),
        "{name}: {r:?}"
    );
}

#[test]
fn epipolar_and_homography_solvers_reject_nan() {
    let a = image_points(10);
    let b = with_nan(image_points(10));
    assert_invalid("fundamental_8point", fundamental_8point(&a, &b));
    assert_invalid("fundamental_7point", fundamental_7point(&a[..7], &b[..7]));
    assert_invalid("dlt_homography", dlt_homography(&a, &b));
    let mut e = Mat3::identity();
    e[(0, 1)] = f64::INFINITY;
    assert_invalid("decompose_essential", decompose_essential(&e));
    let bearings: Vec<Vec3> = a.iter().map(|p| Vec3::new(p.x, p.y, 1.0)).collect();
    let mut bad = bearings.clone();
    bad[1].z = f64::NAN;
    assert!(essential_5point(&bearings[..5], &bad[..5]).is_err());
}

#[test]
fn pose_solvers_reject_nan() {
    let mut world = world_points(10);
    world[2].y = f64::NAN;
    let image = image_points(10);
    assert_invalid("p3p", pnp::p3p(&world[..3], &image[..3], &k()));
    assert_invalid("epnp", pnp::epnp(&world, &image, &k()));
    assert_invalid("dlt", pnp::dlt(&world, &image, &k()));
    assert_invalid("dlt_camera_matrix", dlt_camera_matrix(&world, &image));
    let mut k_bad = k();
    k_bad.fx = f64::NAN;
    assert_invalid(
        "p3p (intrinsics)",
        pnp::p3p(&world_points(3), &image[..3], &k_bad),
    );
}

#[test]
fn triangulation_rejects_nan() {
    let poses = [Iso3::identity(), Iso3::translation(0.1, 0.0, 0.0)];
    let bearings = [Vec3::new(0.0, 0.0, 1.0), Vec3::new(f64::NAN, 0.0, 1.0)];
    assert_invalid(
        "triangulate_point_linear",
        triangulate_point_linear(&poses, &bearings),
    );
}
