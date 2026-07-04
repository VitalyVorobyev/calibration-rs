//! End-to-end synthetic GT round-trip tests for the Scheimpflug intrinsics
//! pipeline across every distortion model:
//!
//!   config.distortion_model = X  →  seeded step_init  →  step_optimize  →  export
//!
//! All scenarios use noiseless synthetic data (modest Scheimpflug tilt) so the
//! reprojection RMS through the exported camera must be below 1e-2 px.
//! Brown-Conrady5 is the default-path regression. Config JSON roundtrip (incl.
//! `None`) and missing-field back-compat guard the serde contract.

#![allow(missing_docs)]

use nalgebra::{Translation3, UnitQuaternion};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraProject, CorrespondenceView, Division, FxFyCxCySkew, Iso3,
    Pinhole, PlanarDataset, Pt2, Pt3, RationalPolynomial, ScheimpflugParams, ThinPrism, View,
};
use vision_calibration_optim::DistortionKind;
use vision_calibration_pipeline::scheimpflug_intrinsics::{
    ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsExport, ScheimpflugIntrinsicsProblem,
    ScheimpflugManualInit, step_init_with_seed, step_optimize,
};
use vision_calibration_pipeline::session::CalibrationSession;

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

fn reproj_rms(export: &ScheimpflugIntrinsicsExport, dataset: &PlanarDataset) -> f64 {
    let camera = export.params.camera.build();
    let poses = &export.params.camera_se3_target;
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

/// Seed the pipeline near GT (manual intrinsics + manual sensor tilt), which
/// takes the trusted warm-start path (exercising the leading-radial multi-start),
/// then optimize and export.
fn run_pipeline(
    dataset: &PlanarDataset,
    model: DistortionKind,
    manual: ScheimpflugManualInit,
) -> (
    ScheimpflugIntrinsicsExport,
    CalibrationSession<ScheimpflugIntrinsicsProblem>,
) {
    let mut config = ScheimpflugIntrinsicsConfig::default();
    config.distortion_model = model;
    config.max_iters = 200;
    // Free k3 so the mask does not clamp a coefficient the extended models want.
    config.fix_k3_in_init = false;

    let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
    session.set_config(config).unwrap();
    session.set_input(dataset.clone()).unwrap();

    step_init_with_seed(&mut session, manual, None).expect("seeded init");
    step_optimize(&mut session, None).expect("optimize");
    let export = session.export().expect("export");
    (export, session)
}

fn near_gt_manual() -> ScheimpflugManualInit {
    let mut manual = ScheimpflugManualInit::default();
    manual.intrinsics = Some(FxFyCxCySkew {
        fx: gt_intrinsics().fx * 1.01,
        fy: gt_intrinsics().fy * 1.01,
        cx: gt_intrinsics().cx + 3.0,
        cy: gt_intrinsics().cy - 3.0,
        skew: 0.0,
    });
    manual.sensor = Some(ScheimpflugParams {
        tilt_x: 0.04,
        tilt_y: -0.015,
    });
    manual
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-model round trips
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bc5_pipeline_round_trip() {
    let dist_gt = BrownConrady5 {
        k1: -0.10,
        k2: 0.03,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cam_gt = Camera::new(Pinhole, dist_gt, gt_sensor().compile(), gt_intrinsics());
    let dataset = make_dataset(&cam_gt, &gt_poses());

    let (export, _s) = run_pipeline(&dataset, DistortionKind::BrownConrady5, near_gt_manual());
    let rms = reproj_rms(&export, &dataset);
    println!("[BC5 pipeline] rms={rms:.4e} px");
    assert!(rms < 1e-2, "BC5 pipeline reproj RMS too large: {rms:.4e}");
}

#[test]
fn rational8_pipeline_round_trip() {
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
    let cam_gt = Camera::new(Pinhole, dist_gt, gt_sensor().compile(), gt_intrinsics());
    let dataset = make_dataset(&cam_gt, &gt_poses());

    let (export, _s) = run_pipeline(&dataset, DistortionKind::Rational8, near_gt_manual());
    let rms = reproj_rms(&export, &dataset);
    println!("[Rational8 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "Rational8 pipeline reproj RMS too large: {rms:.4e}"
    );
}

#[test]
fn thinprism9_pipeline_round_trip() {
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
    let cam_gt = Camera::new(Pinhole, dist_gt, gt_sensor().compile(), gt_intrinsics());
    let dataset = make_dataset(&cam_gt, &gt_poses());

    let (export, _s) = run_pipeline(&dataset, DistortionKind::ThinPrism9, near_gt_manual());
    let rms = reproj_rms(&export, &dataset);
    println!("[ThinPrism9 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "ThinPrism9 pipeline reproj RMS too large: {rms:.4e}"
    );
}

#[test]
fn division1_pipeline_round_trip() {
    let cam_gt = Camera::new(
        Pinhole,
        Division { lambda: -0.20 },
        gt_sensor().compile(),
        gt_intrinsics(),
    );
    let dataset = make_dataset(&cam_gt, &gt_poses());

    let (export, _s) = run_pipeline(&dataset, DistortionKind::Division1, near_gt_manual());
    let rms = reproj_rms(&export, &dataset);
    println!("[Division1 pipeline] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "Division1 pipeline reproj RMS too large: {rms:.4e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Rider: a manual Brown-Conrady seed combined with a non-BC5 model embeds
// (k1,k2,k3,p1,p2) and zeroes the extras; the fit must still converge and the
// init log must carry the warning.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn manual_bc5_seed_with_extended_model_warns_and_converges() {
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
    let cam_gt = Camera::new(Pinhole, dist_gt, gt_sensor().compile(), gt_intrinsics());
    let dataset = make_dataset(&cam_gt, &gt_poses());

    let mut manual = near_gt_manual();
    manual.distortion = Some(BrownConrady5 {
        k1: -0.08,
        ..BrownConrady5::default()
    });

    let (export, session) = run_pipeline(&dataset, DistortionKind::Rational8, manual);
    let rms = reproj_rms(&export, &dataset);
    println!("[manual-BC5 + Rational8] rms={rms:.4e} px");
    assert!(
        rms < 1e-2,
        "manual-seed + Rational8 reproj RMS too large: {rms:.4e}"
    );

    let warned = session.log().iter().any(|e| {
        e.notes
            .as_deref()
            .is_some_and(|n| n.contains("manual Brown-Conrady distortion seed"))
    });
    assert!(
        warned,
        "init log must warn about manual BC5 seed + non-BC5 model"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Config serde contract
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn config_distortion_model_json_roundtrip() {
    for model in [
        DistortionKind::BrownConrady5,
        DistortionKind::Rational8,
        DistortionKind::ThinPrism9,
        DistortionKind::Division1,
        DistortionKind::None,
    ] {
        let mut config = ScheimpflugIntrinsicsConfig::default();
        config.distortion_model = model;
        let json = serde_json::to_string(&config).unwrap();
        let restored: ScheimpflugIntrinsicsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.distortion_model, model,
            "distortion_model roundtrip failed for {model:?}"
        );
    }
}

#[test]
fn config_missing_distortion_model_defaults_to_bc5() {
    // A pre-M-WIRE config JSON has no `distortion_model` field: it must
    // deserialize to BrownConrady5 via #[serde(default = ...)].
    let default_config = ScheimpflugIntrinsicsConfig::default();
    let mut json_val: serde_json::Value = serde_json::to_value(&default_config).unwrap();
    json_val.as_object_mut().unwrap().remove("distortion_model");
    let json_without_field = serde_json::to_string(&json_val).unwrap();

    let config: ScheimpflugIntrinsicsConfig = serde_json::from_str(&json_without_field).unwrap();
    assert_eq!(
        config.distortion_model,
        DistortionKind::BrownConrady5,
        "missing field must default to BrownConrady5"
    );
}
