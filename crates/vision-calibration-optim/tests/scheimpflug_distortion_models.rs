//! Synthetic ground-truth round-trip tests for the Scheimpflug intrinsics path
//! across every distortion model (BrownConrady5, Rational8, ThinPrism9,
//! Division1).
//!
//! Each test builds a Scheimpflug-tilted GT camera (a modest tilt so all board
//! points stay in-frame), generates noiseless observations, seeds near GT with
//! the extended distortion coefficients zeroed, runs
//! [`optimize_scheimpflug_intrinsics`], and asserts a tight reprojection RMS
//! (~1e-2 px) plus recovery of the leading barrel coefficient. Brown-Conrady5 is
//! included as an explicit regression of the default path.

#![allow(missing_docs)]

use nalgebra::{Translation3, UnitQuaternion};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraModel, CameraParams, CameraProject, CorrespondenceView,
    DistortionParams, Division, FxFyCxCySkew, IntrinsicsParams, Iso3, Pinhole, PlanarDataset,
    ProjectionParams, Pt2, Pt3, RationalPolynomial, ScheimpflugParams, SensorParams, ThinPrism,
    View,
};
use vision_calibration_optim::{
    BackendSolveOptions, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsSolveOptions,
    optimize_scheimpflug_intrinsics,
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared GT infrastructure
// ─────────────────────────────────────────────────────────────────────────────

fn gt_intrinsics() -> FxFyCxCySkew<f64> {
    FxFyCxCySkew {
        fx: 1150.0,
        fy: 1155.0,
        cx: 372.0,
        cy: 258.0,
        skew: 0.0,
    }
}

/// Modest Scheimpflug tilt (~3° / ~1°) — observable but small enough that the
/// projected board stays inside the image at every pose.
fn gt_sensor() -> ScheimpflugParams {
    ScheimpflugParams {
        tilt_x: 0.05,
        tilt_y: -0.02,
    }
}

fn make_pose(rx: f64, ry: f64, rz: f64, tx: f64, ty: f64, tz: f64) -> Iso3 {
    Iso3::from_parts(
        Translation3::new(tx, ty, tz),
        UnitQuaternion::from_euler_angles(rx, ry, rz),
    )
}

fn gt_poses() -> Vec<Iso3> {
    vec![
        make_pose(0.0, 0.0, 0.0, -0.01, -0.01, 0.45),
        make_pose(0.20, 0.0, 0.0, -0.02, 0.0, 0.50),
        make_pose(-0.20, 0.0, 0.0, 0.01, -0.02, 0.48),
        make_pose(0.0, 0.20, 0.0, -0.01, 0.01, 0.52),
        make_pose(0.0, -0.20, 0.0, 0.02, -0.01, 0.47),
        make_pose(0.15, 0.15, 0.0, -0.02, -0.02, 0.55),
        make_pose(-0.15, 0.15, 0.05, 0.01, 0.0, 0.49),
        make_pose(0.15, -0.15, -0.05, -0.01, 0.02, 0.53),
        make_pose(-0.15, -0.15, 0.0, 0.0, -0.01, 0.46),
        make_pose(0.10, -0.10, 0.10, -0.02, 0.01, 0.51),
    ]
}

fn board_points() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for iy in 0..9 {
        for ix in 0..9 {
            pts.push(Pt3::new(
                (ix as f64 - 4.0) * 0.02,
                (iy as f64 - 4.0) * 0.02,
                0.0,
            ));
        }
    }
    pts
}

/// Build a noiseless dataset from a generic Scheimpflug camera model.
fn make_dataset<C: CameraProject>(camera: &C, poses: &[Iso3]) -> PlanarDataset {
    let object = board_points();
    let views: Vec<_> = poses
        .iter()
        .map(|pose| {
            let (p3, p2): (Vec<Pt3>, Vec<Pt2>) = object
                .iter()
                .filter_map(|o| {
                    let p_cam = pose.transform_point(o);
                    camera
                        .project_camera_point(&p_cam.coords)
                        .map(|uv| (*o, uv))
                })
                .unzip();
            View::without_meta(CorrespondenceView::new(p3, p2).expect("non-empty view"))
        })
        .collect();
    PlanarDataset::new(views).expect("valid dataset")
}

/// Build the runtime Scheimpflug camera model from a refined parameter pack.
fn built_camera(params: &ScheimpflugIntrinsicsParams) -> CameraModel {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: params.distortion.clone(),
        sensor: SensorParams::Scheimpflug {
            params: params.sensor,
        },
        intrinsics: IntrinsicsParams::FxFyCxCySkew {
            params: params.intrinsics,
        },
    }
    .build()
}

/// Mean reprojection RMS for a built camera over the dataset.
fn reproj_rms<C: CameraProject>(camera: &C, dataset: &PlanarDataset, poses: &[Iso3]) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for (view, pose) in dataset.views.iter().zip(poses.iter()) {
        for (p3, p2) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3);
            if let Some(proj) = camera.project_camera_point(&p_cam.coords) {
                sum_sq += (proj - *p2).norm_squared();
                n += 1;
            }
        }
    }
    if n == 0 {
        f64::INFINITY
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// A slightly perturbed intrinsics seed (2% focal error, a few px on the pp).
fn seed_intrinsics() -> FxFyCxCySkew<f64> {
    let k = gt_intrinsics();
    FxFyCxCySkew {
        fx: k.fx * 1.02,
        fy: k.fy * 1.02,
        cx: k.cx + 4.0,
        cy: k.cy - 4.0,
        skew: 0.0,
    }
}

/// Run the direct joint Scheimpflug solve seeded at GT poses (view 0 fixed for
/// gauge) and return the refined camera + poses as a built model.
fn run_solve(
    dataset: &PlanarDataset,
    seed_distortion: DistortionParams,
    poses: &[Iso3],
) -> (ScheimpflugIntrinsicsParams, f64) {
    let initial = ScheimpflugIntrinsicsParams::new_with_distortion(
        seed_intrinsics(),
        seed_distortion,
        // Seed the tilt near GT (mount angle is known on the supported path).
        ScheimpflugParams {
            tilt_x: 0.04,
            tilt_y: -0.015,
        },
        poses.to_vec(),
    )
    .expect("valid seed");

    let opts = ScheimpflugIntrinsicsSolveOptions {
        fix_poses: vec![0],
        ..Default::default()
    };
    let backend_opts = BackendSolveOptions {
        max_iters: 200,
        verbosity: 0,
        min_abs_decrease: Some(1e-14),
        min_rel_decrease: Some(1e-14),
        min_error: Some(1e-16),
        ..Default::default()
    };
    let est = optimize_scheimpflug_intrinsics(dataset, &initial, opts, backend_opts)
        .expect("scheimpflug solve must succeed");
    let rms = est.mean_reproj_error;
    (est.params, rms)
}

// ─────────────────────────────────────────────────────────────────────────────
// Brown-Conrady5 (default path — regression)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scheimpflug_bc5_round_trip() {
    let k = gt_intrinsics();
    let sensor = gt_sensor();
    let dist_gt = BrownConrady5 {
        k1: -0.10,
        k2: 0.03,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let camera_gt = Camera::new(Pinhole, dist_gt, sensor.compile(), k);
    let poses = gt_poses();
    let dataset = make_dataset(&camera_gt, &poses);

    let seed = DistortionParams::BrownConrady5 {
        params: BrownConrady5 {
            k1: -0.05,
            ..BrownConrady5::default()
        },
    };
    let (params, rms) = run_solve(&dataset, seed, &poses);
    println!("[scheimpflug BC5] rms={rms:.4e} px");
    assert!(rms < 1e-2, "BC5 reprojection RMS too large: {rms:.4e}");

    let d = params.distortion_bc5();
    assert!(
        (d.k1 - dist_gt.k1).abs() < 1e-2,
        "k1 not recovered: {}",
        d.k1
    );
    assert!(
        (params.sensor.tilt_x - sensor.tilt_x).abs() < 1e-2,
        "tilt_x not recovered: {}",
        params.sensor.tilt_x
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Rational-8
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scheimpflug_rational8_round_trip() {
    let k = gt_intrinsics();
    let sensor = gt_sensor();
    let dist_gt = RationalPolynomial {
        k1: -0.12,
        k2: 0.04,
        k3: 0.0,
        k4: 0.008,
        k5: 0.0,
        k6: 0.0,
        p1: 0.0006,
        p2: -0.0004,
        iters: 10,
    };
    let camera_gt = Camera::new(Pinhole, dist_gt, sensor.compile(), k);
    let poses = gt_poses();
    let dataset = make_dataset(&camera_gt, &poses);

    // Seed near GT with the extra denominator coefficients (k4..k6) zeroed.
    let seed = DistortionParams::Rational {
        params: RationalPolynomial {
            k1: -0.08,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 10,
        },
    };
    let (params, rms) = run_solve(&dataset, seed, &poses);
    println!("[scheimpflug Rational8] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "Rational8 reprojection RMS too large: {rms:.4e}"
    );

    let cam = built_camera(&params);
    let rms_check = reproj_rms(&cam, &dataset, &params.camera_se3_target);
    assert!(
        rms_check < 1e-2,
        "Rational8 direct reproj RMS: {rms_check:.4e}"
    );

    if let DistortionParams::Rational { params: p } = &params.distortion {
        assert!(
            (p.k1 - dist_gt.k1).abs() < 5e-2,
            "k1 not recovered: {}",
            p.k1
        );
    } else {
        panic!("expected Rational distortion, got {:?}", params.distortion);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ThinPrism-9
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scheimpflug_thinprism9_round_trip() {
    let k = gt_intrinsics();
    let sensor = gt_sensor();
    let dist_gt = ThinPrism {
        k1: -0.11,
        k2: 0.03,
        k3: 0.0,
        p1: 0.0006,
        p2: -0.0004,
        s1: 0.0003,
        s2: -0.0002,
        s3: 0.0002,
        s4: 0.0,
        iters: 8,
    };
    let camera_gt = Camera::new(Pinhole, dist_gt, sensor.compile(), k);
    let poses = gt_poses();
    let dataset = make_dataset(&camera_gt, &poses);

    // Seed near GT with the thin-prism terms (s1..s4) zeroed.
    let seed = DistortionParams::ThinPrism {
        params: ThinPrism {
            k1: -0.08,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            s1: 0.0,
            s2: 0.0,
            s3: 0.0,
            s4: 0.0,
            iters: 8,
        },
    };
    let (params, rms) = run_solve(&dataset, seed, &poses);
    println!("[scheimpflug ThinPrism9] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "ThinPrism9 reprojection RMS too large: {rms:.4e}"
    );

    let cam = built_camera(&params);
    let rms_check = reproj_rms(&cam, &dataset, &params.camera_se3_target);
    assert!(
        rms_check < 1e-2,
        "ThinPrism9 direct reproj RMS: {rms_check:.4e}"
    );

    if let DistortionParams::ThinPrism { params: p } = &params.distortion {
        assert!(
            (p.k1 - dist_gt.k1).abs() < 5e-2,
            "k1 not recovered: {}",
            p.k1
        );
    } else {
        panic!("expected ThinPrism distortion, got {:?}", params.distortion);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Division-1 (moderate barrel)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn scheimpflug_division1_round_trip() {
    let k = gt_intrinsics();
    let sensor = gt_sensor();
    let lambda_gt = -0.20_f64;
    let camera_gt = Camera::new(Pinhole, Division { lambda: lambda_gt }, sensor.compile(), k);
    let poses = gt_poses();
    let dataset = make_dataset(&camera_gt, &poses);

    // lambda=0 is a valid seed (rationalized formula is analytic there).
    let seed = DistortionParams::Division { lambda: 0.0 };
    let (params, rms) = run_solve(&dataset, seed, &poses);
    println!("[scheimpflug Division1] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "Division1 reprojection RMS too large: {rms:.4e}"
    );

    let cam = built_camera(&params);
    let rms_check = reproj_rms(&cam, &dataset, &params.camera_se3_target);
    assert!(
        rms_check < 1e-2,
        "Division1 direct reproj RMS: {rms_check:.4e}"
    );

    if let DistortionParams::Division { lambda } = &params.distortion {
        assert!(
            (lambda - lambda_gt).abs() < 1e-2,
            "lambda not recovered: {lambda}"
        );
    } else {
        panic!("expected Division distortion, got {:?}", params.distortion);
    }
}
