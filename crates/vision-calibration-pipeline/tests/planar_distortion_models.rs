//! End-to-end synthetic GT round-trip tests for extended distortion models
//! (Rational8, ThinPrism9, Division1) wired through the full pipeline:
//!
//!   config.distortion_model = X  →  step_init  →  step_optimize  →  export
//!
//! All scenarios use noiseless synthetic data so reprojection RMS must be
//! below 1e-2 px (tight tolerance).  Brown-Conrady5 regression is also
//! included to guard the default path.
//!
//! Division1 notes: lambda=0 is a valid seed (rationalized formula has
//! non-zero Jacobian at λ=0), but the model absorbs barrel distortion only.
//! GT lambda is chosen large enough to be observable but small enough that
//! all board points remain inside the image at the test pose distances.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3};
use vision_calibration_core::{
    Camera, CameraProject, CorrespondenceView, Division, FxFyCxCySkew, IdentitySensor, Iso3,
    Pinhole, PlanarDataset, Pt2, Pt3, RationalPolynomial, ThinPrism, View, make_pinhole_camera,
};
use vision_calibration_optim::DistortionKind;
use vision_calibration_pipeline::{
    planar_intrinsics::{
        PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init, step_optimize,
    },
    session::CalibrationSession,
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared test infrastructure
// ─────────────────────────────────────────────────────────────────────────────

const NX: usize = 7;
const NY: usize = 5;
const SPACING: f64 = 0.04;

fn board_points() -> Vec<Pt3> {
    let mut pts = Vec::new();
    for j in 0..NY {
        for i in 0..NX {
            pts.push(Pt3::new(i as f64 * SPACING, j as f64 * SPACING, 0.0));
        }
    }
    pts
}

fn gt_poses() -> Vec<Iso3> {
    (0..6)
        .map(|i| {
            let angle_y = 0.08 * (i as f64 - 2.5);
            let angle_x = 0.05 * (i % 3) as f64 - 0.05;
            let rot = Rotation3::from_euler_angles(angle_x, angle_y, 0.0);
            let tz = 0.5 + 0.07 * i as f64;
            Iso3::from_parts(Translation3::new(0.0, 0.0, tz), rot.into())
        })
        .collect()
}

fn gt_intrinsics() -> FxFyCxCySkew<f64> {
    FxFyCxCySkew {
        fx: 820.0,
        fy: 800.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    }
}

/// Synthesize a noiseless `PlanarDataset` from a generic camera model.
fn make_dataset<C: CameraProject>(camera: &C, poses: &[Iso3]) -> PlanarDataset {
    let pts3d = board_points();
    let views: Vec<_> = poses
        .iter()
        .map(|pose| {
            // Collect (3D, 2D) pairs for points that project into the image.
            let (pts3d_valid, pts2d): (Vec<Pt3>, Vec<Pt2>) = pts3d
                .iter()
                .filter_map(|pw| {
                    let pc = pose.transform_point(pw);
                    camera
                        .project_camera_point(&pc.coords)
                        .map(|proj| (*pw, proj))
                })
                .unzip();
            View::without_meta(CorrespondenceView::new(pts3d_valid, pts2d).expect("non-empty view"))
        })
        .collect();
    PlanarDataset::new(views).expect("valid dataset")
}

/// Mean reprojection RMS for a built camera over the dataset.
fn reproj_rms<C: CameraProject>(camera: &C, dataset: &PlanarDataset, poses: &[Iso3]) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for (view, pose) in dataset.views.iter().zip(poses.iter()) {
        for (p3, p2) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let pc = pose.transform_point(p3);
            if let Some(proj) = camera.project_camera_point(&pc.coords) {
                let err = (proj - *p2).norm_squared();
                sum_sq += err;
                n += 1;
            }
        }
    }
    if n == 0 {
        return f64::INFINITY;
    }
    (sum_sq / n as f64).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers to run the full pipeline and return reprojection RMS
// ─────────────────────────────────────────────────────────────────────────────

fn run_pipeline(
    dataset: PlanarDataset,
    model: DistortionKind,
) -> (
    vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsExport,
    f64,
) {
    let mut config = PlanarIntrinsicsConfig::default();
    config.distortion_model = model;
    // Allow k3 for the extended models; it doesn't affect them but keeps
    // BC5 test consistent.
    config.fix_k3_in_init = false;
    config.max_iters = 100;

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_config(config).unwrap();
    session.set_input(dataset.clone()).unwrap();

    step_init(&mut session, None).expect("step_init must succeed");
    let opt_result = step_optimize(&mut session, None).expect("step_optimize must succeed");
    let export = session.export().expect("export must succeed");

    (export, opt_result.mean_reproj_error)
}

// ─────────────────────────────────────────────────────────────────────────────
// Brown-Conrady 5  (default path — regression guard)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bc5_pipeline_noiseless_round_trip() {
    let k_gt = gt_intrinsics();
    // Use mild distortion so Zhang's linear init yields a good-enough starting
    // point for the non-linear optimizer to reach near-zero residual in a
    // fresh pipeline run (no seeded GT poses).
    let dist_gt = vision_calibration_core::BrownConrady5 {
        k1: -0.05,
        k2: 0.02,
        k3: 0.0,
        p1: 0.0003,
        p2: -0.0002,
        iters: 8,
    };
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);
    let poses = gt_poses();
    let dataset = make_dataset(&cam_gt, &poses);

    let (export, rms) = run_pipeline(dataset, DistortionKind::BrownConrady5);

    println!("[BC5 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "BC5 pipeline: reprojection RMS too large: {rms:.4e} px"
    );

    // Verify coefficients converge close to GT.
    let k_out = export.params.intrinsics();
    let d_out = export
        .params
        .distortion()
        .expect("BC5 must produce BC5 distortion");
    println!(
        "[BC5] fx_err={:.3} fy_err={:.3} k1_err={:.4e} k2_err={:.4e}",
        (k_out.fx - k_gt.fx).abs(),
        (k_out.fy - k_gt.fy).abs(),
        (d_out.k1 - dist_gt.k1).abs(),
        (d_out.k2 - dist_gt.k2).abs(),
    );
    assert!((k_out.fx - k_gt.fx).abs() < 5.0, "fx not converged");
    assert!((k_out.fy - k_gt.fy).abs() < 5.0, "fy not converged");
    // Pipeline-path coefficient accuracy: ~50% relative (loose — linear init
    // introduces mild bias that the NL optimizer absorbs into intrinsics).
    assert!((d_out.k1 - dist_gt.k1).abs() < 0.05, "k1 not converged");
    assert!((d_out.k2 - dist_gt.k2).abs() < 0.05, "k2 not converged");
}

// ─────────────────────────────────────────────────────────────────────────────
// Rational-8  (6 radial + 2 tangential)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rational8_pipeline_noiseless_round_trip() {
    let k_gt = gt_intrinsics();
    let dist_gt = RationalPolynomial {
        k1: -0.20,
        k2: 0.06,
        k3: 0.0,
        k4: 0.01,
        k5: 0.0,
        k6: 0.0,
        p1: 0.001,
        p2: -0.0005,
        iters: 10,
    };
    let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
    let poses = gt_poses();
    let dataset = make_dataset(&cam_gt, &poses);

    let (export, rms) = run_pipeline(dataset.clone(), DistortionKind::Rational8);

    println!("[Rational8 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "Rational8 pipeline: reprojection RMS too large: {rms:.4e} px"
    );

    let k_out = export.params.intrinsics();
    println!(
        "[Rational8] fx_err={:.3} fy_err={:.3}",
        (k_out.fx - k_gt.fx).abs(),
        (k_out.fy - k_gt.fy).abs(),
    );
    assert!((k_out.fx - k_gt.fx).abs() < 5.0, "fx not converged");
    assert!((k_out.fy - k_gt.fy).abs() < 5.0, "fy not converged");

    // Verify reprojection with the exported camera model.
    let cam_out = export.params.build_camera();
    let rms_check = reproj_rms(&cam_out, &dataset, &export.params.camera_se3_target);
    println!("[Rational8] direct reproj rms={rms_check:.4e}");
    assert!(
        rms_check < 1e-2,
        "Rational8 direct reproj RMS: {rms_check:.4e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ThinPrism-9  (BC5 + thin-prism s1..s4)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn thinprism9_pipeline_noiseless_round_trip() {
    let k_gt = gt_intrinsics();
    let dist_gt = ThinPrism {
        k1: -0.15,
        k2: 0.05,
        k3: 0.0,
        p1: 0.0008,
        p2: -0.0005,
        s1: 0.0002,
        s2: -0.0001,
        s3: 0.0001,
        s4: 0.0,
        iters: 8,
    };
    let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
    let poses = gt_poses();
    let dataset = make_dataset(&cam_gt, &poses);

    let (export, rms) = run_pipeline(dataset.clone(), DistortionKind::ThinPrism9);

    println!("[ThinPrism9 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "ThinPrism9 pipeline: reprojection RMS too large: {rms:.4e} px"
    );

    let k_out = export.params.intrinsics();
    println!(
        "[ThinPrism9] fx_err={:.3} fy_err={:.3}",
        (k_out.fx - k_gt.fx).abs(),
        (k_out.fy - k_gt.fy).abs(),
    );
    assert!((k_out.fx - k_gt.fx).abs() < 5.0, "fx not converged");
    assert!((k_out.fy - k_gt.fy).abs() < 5.0, "fy not converged");

    let cam_out = export.params.build_camera();
    let rms_check = reproj_rms(&cam_out, &dataset, &export.params.camera_se3_target);
    println!("[ThinPrism9] direct reproj rms={rms_check:.4e}");
    assert!(
        rms_check < 1e-2,
        "ThinPrism9 direct reproj RMS: {rms_check:.4e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Division-1  (Fitzgibbon single-parameter division model)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn division1_pipeline_noiseless_round_trip() {
    let k_gt = gt_intrinsics();
    // lambda < 0 produces barrel distortion (the typical case).
    // |lambda| is kept moderate so all board points stay in-frame.
    let lambda_gt = -0.25_f64;
    let dist_gt = Division { lambda: lambda_gt };
    let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
    let poses = gt_poses();
    let dataset = make_dataset(&cam_gt, &poses);

    let (export, rms) = run_pipeline(dataset.clone(), DistortionKind::Division1);

    println!("[Division1 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "Division1 pipeline: reprojection RMS too large: {rms:.4e} px"
    );

    let k_out = export.params.intrinsics();
    println!(
        "[Division1] fx_err={:.3} fy_err={:.3}",
        (k_out.fx - k_gt.fx).abs(),
        (k_out.fy - k_gt.fy).abs(),
    );
    assert!((k_out.fx - k_gt.fx).abs() < 5.0, "fx not converged");
    assert!((k_out.fy - k_gt.fy).abs() < 5.0, "fy not converged");

    let cam_out = export.params.build_camera();
    let rms_check = reproj_rms(&cam_out, &dataset, &export.params.camera_se3_target);
    println!("[Division1] direct reproj rms={rms_check:.4e}");
    assert!(
        rms_check < 1e-2,
        "Division1 direct reproj RMS: {rms_check:.4e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON roundtrip of PlanarIntrinsicsExport for an extended model
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rational8_export_json_roundtrip() {
    let k_gt = gt_intrinsics();
    let dist_gt = RationalPolynomial {
        k1: -0.20,
        k2: 0.06,
        k3: 0.0,
        k4: 0.01,
        k5: 0.0,
        k6: 0.0,
        p1: 0.001,
        p2: -0.0005,
        iters: 10,
    };
    let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);
    let poses = gt_poses();
    let dataset = make_dataset(&cam_gt, &poses);

    let (export, _rms) = run_pipeline(dataset, DistortionKind::Rational8);

    let json = serde_json::to_string_pretty(&export).expect("serialize export");
    let restored: vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsExport =
        serde_json::from_str(&json).expect("deserialize export");

    // Intrinsics survive the roundtrip.
    let k_orig = export.params.intrinsics();
    let k_rest = restored.params.intrinsics();
    assert!((k_orig.fx - k_rest.fx).abs() < 1e-9);
    assert!((k_orig.fy - k_rest.fy).abs() < 1e-9);
    assert!((k_orig.cx - k_rest.cx).abs() < 1e-9);
    assert!((k_orig.cy - k_rest.cy).abs() < 1e-9);

    // Metrics survive.
    assert!((export.mean_reproj_error - restored.mean_reproj_error).abs() < 1e-12);

    // Distortion kind survives (DistortionParams::Rational round-trips).
    let json_str = json.as_str();
    assert!(
        json_str.contains("rational") || json_str.contains("Rational"),
        "export JSON must contain 'rational' distortion kind"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Config JSON roundtrip with distortion_model field
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn planar_config_distortion_model_json_roundtrip() {
    for model in [
        DistortionKind::BrownConrady5,
        DistortionKind::Rational8,
        DistortionKind::ThinPrism9,
        DistortionKind::Division1,
        DistortionKind::None,
    ] {
        let mut config = PlanarIntrinsicsConfig::default();
        config.distortion_model = model;
        let json = serde_json::to_string(&config).unwrap();
        let restored: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.distortion_model, model,
            "distortion_model roundtrip failed for {model:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward compat: config without distortion_model field defaults to BC5
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn planar_config_missing_distortion_model_defaults_to_bc5() {
    // Simulate a pre-M-WIRE config JSON that has no `distortion_model` field.
    // Serialize the default config, strip the distortion_model key, and verify
    // deserialization still produces BrownConrady5 (via #[serde(default=...)]).
    let default_config = PlanarIntrinsicsConfig::default();
    let mut json_val: serde_json::Value =
        serde_json::to_value(&default_config).expect("serialize default config");
    json_val.as_object_mut().unwrap().remove("distortion_model");
    let json_without_field = serde_json::to_string(&json_val).unwrap();

    let config: PlanarIntrinsicsConfig = serde_json::from_str(&json_without_field).unwrap();
    assert_eq!(
        config.distortion_model,
        DistortionKind::BrownConrady5,
        "missing field must default to BrownConrady5"
    );
}
