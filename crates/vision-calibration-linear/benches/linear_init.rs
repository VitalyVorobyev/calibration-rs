//! Criterion benchmarks for the linear-init hot paths (Track P / P4).
//!
//! These guard the dense-SVD-hang fixes (P1): a regression that reinstates the
//! pathological `nalgebra::svd(true, true)` path on a tall design matrix would
//! blow these timings up by orders of magnitude (the homography DLT once hung
//! for >15 min on dense real data). Deterministic synthetic data keeps the
//! numbers comparable across runs.

#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Mat3, Pt2, Pt3, Real,
    make_pinhole_camera,
};
use vision_calibration_linear::distortion_fit::{
    DistortionFitOptions, DistortionView, MetaHomography, estimate_distortion_from_homographies,
};
use vision_calibration_linear::homography::dlt_homography;
use vision_calibration_linear::zhang_intrinsics::estimate_intrinsics_from_homographies;

/// A 15×15 grid (225 correspondences) mapped through a known perspective
/// homography — the dense case that exercised the original hang.
fn dense_homography_corrs() -> (Vec<Pt2>, Vec<Pt2>) {
    let h_gt = Mat3::new(1.2, 0.05, 3.0, 0.1, 0.9, -2.0, 0.0008, -0.0006, 1.0);
    let mut world = Vec::with_capacity(225);
    let mut image = Vec::with_capacity(225);
    for iy in 0..15 {
        for ix in 0..15 {
            let (x, y) = (ix as Real, iy as Real);
            let p = h_gt * Vector3::new(x, y, 1.0);
            world.push(Pt2::new(x, y));
            image.push(Pt2::new(p.x / p.z, p.y / p.z));
        }
    }
    (world, image)
}

/// A synthetic planar calibration scene: a 10×7 board (3 cm spacing) seen from
/// `n_views` poses through a pinhole camera. Returns the 3×3 intrinsics, the
/// board's 3D points, and the per-view projected pixels.
fn planar_scene(
    k: FxFyCxCySkew<Real>,
    dist: BrownConrady5<Real>,
    n_views: usize,
) -> (Mat3, Vec<Pt3>, Vec<Vec<Pt2>>) {
    let cam = make_pinhole_camera(k, dist);
    let (nx, ny, spacing) = (10usize, 7usize, 0.03_f64);
    let board_3d: Vec<Pt3> = (0..ny)
        .flat_map(|j| (0..nx).map(move |i| Pt3::new(i as Real * spacing, j as Real * spacing, 0.0)))
        .collect();

    let mut views = Vec::with_capacity(n_views);
    for v in 0..n_views {
        let angle = (v as Real * 8.0).to_radians();
        let rot = UnitQuaternion::from_euler_angles(angle, angle * 0.4, 0.0);
        let trans = Vector3::new(0.05, 0.05, 0.5 + 0.04 * v as Real);
        let pose = Iso3::from_parts(trans.into(), rot);
        let pixels: Vec<Pt2> = board_3d
            .iter()
            .map(|pw| {
                let uv = cam.project_point(&pose.transform_point(pw)).unwrap();
                Pt2::new(uv.x, uv.y)
            })
            .collect();
        views.push(pixels);
    }
    (k.k_matrix(), board_3d, views)
}

fn homography_dlt(c: &mut Criterion) {
    let (world, image) = dense_homography_corrs();
    c.bench_function("homography_dlt_225pts", |b| {
        b.iter(|| dlt_homography(black_box(&world), black_box(&image)).unwrap())
    });
}

fn zhang_intrinsics(c: &mut Criterion) {
    let k = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let (_, board_3d, views) = planar_scene(k, dist, 15);
    let board_2d: Vec<Pt2> = board_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let homographies: Vec<Mat3> = views
        .iter()
        .map(|pixels| dlt_homography(&board_2d, pixels).unwrap())
        .collect();
    c.bench_function("zhang_intrinsics_from_15h", |b| {
        b.iter(|| estimate_intrinsics_from_homographies(black_box(&homographies)).unwrap())
    });
}

fn distortion_fit(c: &mut Criterion) {
    let k = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist = BrownConrady5 {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 8,
    };
    let (k_mat, board_3d, views) = planar_scene(k, dist, 12);
    let board_2d: Vec<Pt2> = board_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let dist_views: Vec<DistortionView> = views
        .iter()
        .map(|pixels| {
            // Homography from the *distorted* pixels, as the real init does.
            let homography = dlt_homography(&board_2d, pixels).unwrap();
            let obs = CorrespondenceView::new(board_3d.clone(), pixels.clone()).unwrap();
            DistortionView::new(obs, MetaHomography { homography })
        })
        .collect();
    let opts = DistortionFitOptions::default();
    c.bench_function("distortion_fit_12views_70pts", |b| {
        b.iter(|| {
            estimate_distortion_from_homographies(black_box(&k_mat), black_box(&dist_views), opts)
                .unwrap()
        })
    });
}

criterion_group!(benches, homography_dlt, zhang_intrinsics, distortion_fit);
criterion_main!(benches);
