//! Criterion benchmark for one per-camera bundle-adjustment solve (Track P / P4).
//!
//! Exercises the `tiny-solver` LM loop end-to-end on a representative planar
//! intrinsics problem (10×7 board, 10 views → ~700 reprojection residuals). This
//! is the per-camera BA stage of the rig pipeline and the natural place to
//! measure P3 backend work (autodiff Jacobian vs JᵀJ assembly vs linear solve).
//! Deterministic synthetic data keeps the numbers comparable across runs.

#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use nalgebra::{UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, PinholeCamera, PlanarDataset, Pt2, Pt3,
    Real, View, make_pinhole_camera,
};
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
    optimize_planar_intrinsics,
};

/// Build a synthetic planar-intrinsics problem: ground-truth dataset plus a
/// noisy-but-plausible initial camera and the ground-truth poses to seed from.
fn synthetic_problem() -> (PlanarDataset, PinholeCamera, Vec<Iso3>) {
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 8,
    };
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);

    let (nx, ny, spacing) = (10usize, 7usize, 0.03_f64);
    let board_3d: Vec<Pt3> = (0..ny)
        .flat_map(|j| (0..nx).map(move |i| Pt3::new(i as Real * spacing, j as Real * spacing, 0.0)))
        .collect();

    let mut views = Vec::new();
    let mut poses_gt = Vec::new();
    for v in 0..10 {
        let angle = (v as Real * 10.0).to_radians();
        let rot = UnitQuaternion::from_euler_angles(angle, angle * 0.5, 0.0);
        let trans = Vector3::new(0.1, 0.1, 0.5 + 0.05 * v as Real);
        let pose = Iso3::from_parts(trans.into(), rot);
        poses_gt.push(pose);

        let mut p2d = Vec::new();
        let mut p3d = Vec::new();
        for pw in &board_3d {
            let pc = pose.transform_point(pw).coords;
            if pc.z > 0.1
                && let Some(uv) = cam_gt.project_point_c(&pc)
            {
                p2d.push(Pt2::new(uv.x, uv.y));
                p3d.push(*pw);
            }
        }
        views.push(View::without_meta(
            CorrespondenceView::new(p3d, p2d).unwrap(),
        ));
    }

    let dataset = PlanarDataset::new(views).unwrap();
    let cam_init = make_pinhole_camera(
        FxFyCxCySkew {
            fx: 780.0,
            fy: 760.0,
            cx: 630.0,
            cy: 350.0,
            skew: 0.0,
        },
        BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        },
    );
    (dataset, cam_init, poses_gt)
}

fn planar_intrinsics_ba(c: &mut Criterion) {
    let (dataset, cam_init, poses_gt) = synthetic_problem();
    c.bench_function("planar_intrinsics_ba_10views_70pts", |b| {
        b.iter(|| {
            let init =
                PlanarIntrinsicsParams::from_pinhole(cam_init.clone(), poses_gt.clone()).unwrap();
            optimize_planar_intrinsics(
                black_box(&dataset),
                black_box(&init),
                PlanarIntrinsicsSolveOptions::default(),
                BackendSolveOptions::default(),
            )
            .unwrap()
        })
    });
}

criterion_group!(benches, planar_intrinsics_ba);
criterion_main!(benches);
