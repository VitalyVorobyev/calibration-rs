use nalgebra::{UnitQuaternion, Vector3};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, DistortionFixMask, FxFyCxCySkew, IntrinsicsFixMask, Iso3,
    PinholeCamera, PlanarDataset, Pt2, Pt3, Real, View, make_pinhole_camera,
};
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RobustLoss,
    optimize_planar_intrinsics,
};

struct SyntheticScenario {
    dataset: PlanarDataset,
    poses_gt: Vec<Iso3>,
    cam_gt: PinholeCamera,
    cam_init: PinholeCamera,
}

fn build_synthetic_scenario(noise_amplitude: f64) -> SyntheticScenario {
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);

    let nx = 6;
    let ny = 4;
    let spacing = 0.03_f64;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let x = i as f64 * spacing;
            let y = j as f64 * spacing;
            board_points.push(Pt3::new(x, y, 0.0));
        }
    }

    let mut views = Vec::new();
    let mut poses_gt = Vec::new();

    for view_idx in 0..3 {
        let angle = 0.1 * (view_idx as f64);
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let rq = UnitQuaternion::from_scaled_axis(axis * angle);
        let rot = rq.to_rotation_matrix();
        let trans = Vector3::new(0.0, 0.0, 0.5 + 0.2 * view_idx as f64);
        let pose = Iso3::from_parts(trans.into(), rot.into());

        poses_gt.push(pose);

        let mut img_points = Vec::new();
        for (pt_idx, pw) in board_points.iter().enumerate() {
            let p_cam = pose.transform_point(pw);
            let proj = cam_gt.project_point(&p_cam).unwrap();
            let mut coords = Pt2::new(proj.x, proj.y);

            if noise_amplitude > 0.0 {
                let sign = if (view_idx + pt_idx) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let delta = noise_amplitude * sign;
                coords.x += delta;
                coords.y -= delta;
            }

            img_points.push(coords);
        }

        views.push(View::without_meta(
            CorrespondenceView::new(board_points.clone(), img_points).unwrap(),
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

    SyntheticScenario {
        dataset,
        poses_gt,
        cam_gt,
        cam_init,
    }
}

#[test]
fn synthetic_planar_intrinsics_refinement_converges() {
    let SyntheticScenario {
        dataset,
        poses_gt,
        cam_gt,
        cam_init,
    } = build_synthetic_scenario(0.0);
    let k_gt = cam_gt.k;

    let init = PlanarIntrinsicsParams::new(cam_init, poses_gt).unwrap();
    let opts = PlanarIntrinsicsSolveOptions::default();
    let solver = BackendSolveOptions::default();

    let result = optimize_planar_intrinsics(&dataset, &init, opts, solver).unwrap();

    assert!((result.params.camera.k.fx - k_gt.fx).abs() < 5.0);
    assert!((result.params.camera.k.fy - k_gt.fy).abs() < 5.0);
    assert!((result.params.camera.k.cx - k_gt.cx).abs() < 5.0);
    assert!((result.params.camera.k.cy - k_gt.cy).abs() < 5.0);
}

#[test]
fn synthetic_planar_intrinsics_with_outliers_robust_better_than_l2() {
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);

    let nx = 6;
    let ny = 4;
    let spacing = 0.03_f64;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    let mut views = Vec::new();
    let mut poses_gt = Vec::new();
    let outlier_stride = 12;
    let outlier_offset = 20.0;

    for view_idx in 0..3 {
        let angle = 0.1 * (view_idx as f64);
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let rq = UnitQuaternion::from_scaled_axis(axis * angle);
        let rot = rq.to_rotation_matrix();
        let trans = Vector3::new(0.0, 0.0, 0.5 + 0.2 * view_idx as f64);
        let pose = Iso3::from_parts(trans.into(), rot.into());
        poses_gt.push(pose);

        let mut img_points = Vec::new();
        for (pt_idx, pw) in board_points.iter().enumerate() {
            let p_cam = pose.transform_point(pw);
            let proj = cam_gt.project_point(&p_cam).unwrap();
            let mut coords = Pt2::new(proj.x, proj.y);

            if pt_idx % outlier_stride == 0 {
                coords.x += outlier_offset;
                coords.y += outlier_offset;
            }

            img_points.push(coords);
        }

        views.push(View::without_meta(
            CorrespondenceView::new(board_points.clone(), img_points).unwrap(),
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

    let init = PlanarIntrinsicsParams::new(cam_init, poses_gt).unwrap();
    let solver = BackendSolveOptions::default();

    let l2_opts = PlanarIntrinsicsSolveOptions::default();
    let robust_opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::Huber { scale: 2.0 },
        ..PlanarIntrinsicsSolveOptions::default()
    };

    let l2 = optimize_planar_intrinsics(&dataset, &init, l2_opts, solver.clone()).unwrap();
    let robust = optimize_planar_intrinsics(&dataset, &init, robust_opts, solver).unwrap();

    let err_total = |cam: &PinholeCamera| -> Real {
        (cam.k.fx - k_gt.fx).abs()
            + (cam.k.fy - k_gt.fy).abs()
            + (cam.k.cx - k_gt.cx).abs()
            + (cam.k.cy - k_gt.cy).abs()
    };

    let err_l2 = err_total(&l2.params.camera);
    let err_robust = err_total(&robust.params.camera);

    assert!(
        err_robust < err_l2,
        "robust intrinsics error {} should be smaller than L2 {}",
        err_robust,
        err_l2
    );
}

#[test]
fn intrinsics_masking_keeps_fixed_params() {
    let SyntheticScenario {
        dataset,
        poses_gt,
        cam_init,
        ..
    } = build_synthetic_scenario(0.0);

    let init = PlanarIntrinsicsParams::new(cam_init, poses_gt).unwrap();
    let opts = PlanarIntrinsicsSolveOptions {
        fix_intrinsics: IntrinsicsFixMask {
            fx: true,
            fy: true,
            ..Default::default()
        },
        ..PlanarIntrinsicsSolveOptions::default()
    };
    let solver = BackendSolveOptions::default();

    let result = optimize_planar_intrinsics(&dataset, &init, opts, solver).unwrap();

    assert!(
        (result.params.camera.k.fx - init.camera.k.fx).abs() < 1e-12,
        "fx should remain fixed"
    );
    assert!(
        (result.params.camera.k.fy - init.camera.k.fy).abs() < 1e-12,
        "fy should remain fixed"
    );
}

#[test]
fn synthetic_planar_with_distortion_converges() {
    // Test that distortion optimization works with known ground truth
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: -0.2, // Barrel distortion
        k2: 0.05,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 8,
    };
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);

    // Generate synthetic planar target (10x7 grid, 3cm spacing)
    let board_points: Vec<Pt3> = (0..10)
        .flat_map(|i| (0..7).map(move |j| Pt3::new(j as Real * 0.03, i as Real * 0.03, 0.0)))
        .collect();

    let mut views = vec![];
    let mut poses_gt = vec![];

    // Create 10 views at different poses
    for i in 0..10 {
        let angle = (i as Real * 10.0).to_radians();
        let dist_from_board = 0.5 + i as Real * 0.05;
        let rot = UnitQuaternion::from_euler_angles(angle, angle * 0.5, 0.0);
        let trans = Vector3::new(0.1, 0.1, dist_from_board);
        let pose = Iso3::from_parts(trans.into(), rot);
        poses_gt.push(pose);

        // Project points through ground truth camera
        let mut points_2d = vec![];
        let mut points_3d = vec![];

        for pw in &board_points {
            let pc = pose.transform_point(pw).coords;
            if pc.z > 0.1
                && let Some(uv) = cam_gt.project_point_c(&pc)
            {
                points_2d.push(Pt2::new(uv.x, uv.y));
                points_3d.push(*pw);
            }
        }

        views.push(View::without_meta(
            CorrespondenceView::new(points_3d, points_2d).unwrap(),
        ));
    }

    let dataset = PlanarDataset::new(views).unwrap();

    // Initialize with noisy intrinsics and zero distortion
    let cam_init = make_pinhole_camera(
        FxFyCxCySkew {
            fx: 780.0, // -20 error
            fy: 760.0, // -20 error
            cx: 630.0, // -10 error
            cy: 350.0, // -10 error
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
    let init = PlanarIntrinsicsParams::new(cam_init, poses_gt.clone()).unwrap();

    let opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::None,
        ..Default::default()
    };

    let backend_opts = BackendSolveOptions::default();
    let result = optimize_planar_intrinsics(&dataset, &init, opts, backend_opts).unwrap();

    // Verify convergence to ground truth
    println!(
        "Final camera: fx={}, fy={}, cx={}, cy={}",
        result.params.camera.k.fx,
        result.params.camera.k.fy,
        result.params.camera.k.cx,
        result.params.camera.k.cy
    );
    println!(
        "Final distortion: k1={}, k2={}, k3={}, p1={}, p2={}",
        result.params.camera.dist.k1,
        result.params.camera.dist.k2,
        result.params.camera.dist.k3,
        result.params.camera.dist.p1,
        result.params.camera.dist.p2
    );

    assert!(
        (result.params.camera.k.fx - k_gt.fx).abs() < 5.0,
        "fx off by {}",
        result.params.camera.k.fx - k_gt.fx
    );
    assert!(
        (result.params.camera.k.fy - k_gt.fy).abs() < 5.0,
        "fy off by {}",
        result.params.camera.k.fy - k_gt.fy
    );
    assert!(
        (result.params.camera.k.cx - k_gt.cx).abs() < 3.0,
        "cx off by {}",
        result.params.camera.k.cx - k_gt.cx
    );
    assert!(
        (result.params.camera.k.cy - k_gt.cy).abs() < 3.0,
        "cy off by {}",
        result.params.camera.k.cy - k_gt.cy
    );

    assert!(
        (result.params.camera.dist.k1 - dist_gt.k1).abs() < 0.01,
        "k1 off by {}",
        result.params.camera.dist.k1 - dist_gt.k1
    );
    assert!(
        (result.params.camera.dist.k2 - dist_gt.k2).abs() < 0.01,
        "k2 off by {}",
        result.params.camera.dist.k2 - dist_gt.k2
    );
    assert!(
        (result.params.camera.dist.p1 - dist_gt.p1).abs() < 0.001,
        "p1 off by {}",
        result.params.camera.dist.p1 - dist_gt.p1
    );
    assert!(
        (result.params.camera.dist.p2 - dist_gt.p2).abs() < 0.001,
        "p2 off by {}",
        result.params.camera.dist.p2 - dist_gt.p2
    );
}

#[test]
fn distortion_parameter_masking_works() {
    // Test selective fixing of distortion parameters
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: -0.15,
        k2: 0.04,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);

    // Generate smaller synthetic dataset
    let board_points: Vec<Pt3> = (0..6)
        .flat_map(|i| (0..5).map(move |j| Pt3::new(j as Real * 0.03, i as Real * 0.03, 0.0)))
        .collect();

    let mut views = vec![];
    let mut poses_gt = vec![];

    for i in 0..5 {
        let angle = (i as Real * 15.0).to_radians();
        let rot = UnitQuaternion::from_euler_angles(angle, angle * 0.3, 0.0);
        let trans = Vector3::new(0.0, 0.0, 0.6);
        let pose = Iso3::from_parts(trans.into(), rot);
        poses_gt.push(pose);

        let mut points_2d = vec![];
        let mut points_3d = vec![];

        for pw in &board_points {
            let pc = pose.transform_point(pw).coords;
            if pc.z > 0.1
                && let Some(uv) = cam_gt.project_point_c(&pc)
            {
                points_2d.push(Pt2::new(uv.x, uv.y));
                points_3d.push(*pw);
            }
        }

        views.push(View::without_meta(
            CorrespondenceView::new(points_3d, points_2d).unwrap(),
        ));
    }

    let dataset = PlanarDataset::new(views).unwrap();

    let cam_init = make_pinhole_camera(
        FxFyCxCySkew {
            fx: 790.0,
            fy: 770.0,
            cx: 635.0,
            cy: 355.0,
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
    let init = PlanarIntrinsicsParams::new(cam_init, poses_gt).unwrap();

    // Fix k3, p1, p2 (they are zero in ground truth)
    let opts = PlanarIntrinsicsSolveOptions {
        fix_distortion: DistortionFixMask {
            k3: true,
            p1: true,
            p2: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let result =
        optimize_planar_intrinsics(&dataset, &init, opts, BackendSolveOptions::default()).unwrap();

    // Fixed params should stay at initial values
    assert_eq!(
        result.params.camera.dist.k3, 0.0,
        "k3 should stay fixed at 0"
    );
    assert_eq!(
        result.params.camera.dist.p1, 0.0,
        "p1 should stay fixed at 0"
    );
    assert_eq!(
        result.params.camera.dist.p2, 0.0,
        "p2 should stay fixed at 0"
    );

    // k1, k2 should converge
    assert!(
        (result.params.camera.dist.k1 - dist_gt.k1).abs() < 0.01,
        "k1 should converge, off by {}",
        result.params.camera.dist.k1 - dist_gt.k1
    );
    assert!(
        (result.params.camera.dist.k2 - dist_gt.k2).abs() < 0.01,
        "k2 should converge, off by {}",
        result.params.camera.dist.k2 - dist_gt.k2
    );
}
