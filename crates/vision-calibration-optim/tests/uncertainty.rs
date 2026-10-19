//! The reported parameter uncertainty against Monte-Carlo scatter, through
//! the public solve functions.

#![allow(missing_docs)]

use nalgebra::{DMatrix, DVector, Translation3, UnitQuaternion};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::{
    BrownConrady5, CameraParams, CameraProject, CorrespondenceView, DistortionFixMask,
    DistortionParams, FxFyCxCySkew, IntrinsicsParams, Iso3, PlanarDataset, ProjectionParams, Pt2,
    Pt3, ScheimpflugParams, SensorParams, Uncertainty, Vec2, View,
};
use vision_calibration_optim::{
    BackendSolveOptions, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsSolveOptions,
    optimize_scheimpflug_intrinsics,
};

const RUNS: u64 = 200;

fn truth() -> ScheimpflugIntrinsicsParams {
    let poses = [
        (0.0, 0.0, 0.0, -0.01, -0.01, 0.45),
        (0.20, 0.0, 0.0, -0.02, 0.0, 0.50),
        (-0.20, 0.0, 0.0, 0.01, -0.02, 0.48),
        (0.0, 0.20, 0.0, -0.01, 0.01, 0.52),
        (0.0, -0.20, 0.0, 0.02, -0.01, 0.47),
        (0.15, 0.15, 0.0, -0.02, -0.02, 0.55),
        (-0.15, 0.15, 0.05, 0.01, 0.0, 0.49),
        (0.15, -0.15, -0.05, -0.01, 0.02, 0.53),
        (-0.15, -0.15, 0.0, 0.0, -0.01, 0.46),
        (0.10, -0.10, 0.10, -0.02, 0.01, 0.51),
    ]
    .map(|(rx, ry, rz, tx, ty, tz)| {
        Iso3::from_parts(
            Translation3::new(tx, ty, tz),
            UnitQuaternion::from_euler_angles(rx, ry, rz),
        )
    });
    ScheimpflugIntrinsicsParams::new_with_distortion(
        FxFyCxCySkew {
            fx: 1150.0,
            fy: 1155.0,
            cx: 372.0,
            cy: 258.0,
            skew: 0.0,
        },
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: -0.08,
                k2: 0.04,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        },
        ScheimpflugParams {
            tilt_x: 0.05,
            tilt_y: -0.02,
        },
        poses.to_vec(),
    )
    .unwrap()
}

/// Views of a 9×9 board of 20 mm cells, with seeded pixel noise.
fn dataset(params: &ScheimpflugIntrinsicsParams, noise: UniformPixelNoise) -> PlanarDataset {
    let camera = CameraParams {
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
    .unwrap();
    let board: Vec<Pt3> = (0..81)
        .map(|i| {
            Pt3::new(
                (i % 9) as f64 * 0.02 - 0.08,
                (i / 9) as f64 * 0.02 - 0.08,
                0.0,
            )
        })
        .collect();
    let views = params
        .camera_se3_target
        .iter()
        .enumerate()
        .map(|(v, pose)| {
            let (p3, p2): (Vec<Pt3>, Vec<Pt2>) = board
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    let uv = camera.project_camera_point(&pose.transform_point(p).coords)?;
                    let uv = noise.apply(v, i, Vec2::new(uv.x, uv.y));
                    Some((*p, Pt2::new(uv.x, uv.y)))
                })
                .unzip();
            View::without_meta(CorrespondenceView::new(p3, p2).unwrap())
        })
        .collect();
    PlanarDataset::new(views).unwrap()
}

/// The labelled value of a Scheimpflug parameter pack.
fn value(params: &ScheimpflugIntrinsicsParams, label: &str) -> f64 {
    let k = params.intrinsics;
    let DistortionParams::BrownConrady5 { params: d } = &params.distortion else {
        unreachable!("the scene uses Brown-Conrady");
    };
    match label {
        "fx" => k.fx,
        "fy" => k.fy,
        "cx" => k.cx,
        "cy" => k.cy,
        "k1" => d.k1,
        "k2" => d.k2,
        "tilt_x" => params.sensor.tilt_x,
        "tilt_y" => params.sensor.tilt_y,
        other => panic!("unexpected label {other}"),
    }
}

/// The Scheimpflug camera's predicted σ (intrinsics, radial distortion,
/// tilt) match their scatter over seeded noise within 10 %, and the NEES
/// matches χ².
#[test]
fn scheimpflug_covariance_matches_monte_carlo() {
    let gt = truth();
    let mut errors: Vec<DVector<f64>> = Vec::new();
    let mut predicted = None::<DVector<f64>>;
    let mut nees = 0.0;
    let mut labels = Vec::new();
    for seed in 0..RUNS {
        let data = dataset(
            &gt,
            UniformPixelNoise {
                seed: 500 + seed,
                max_abs_px: 0.5,
            },
        );
        let opts = ScheimpflugIntrinsicsSolveOptions {
            fix_distortion: DistortionFixMask::radial_only(),
            ..Default::default()
        };
        let est = optimize_scheimpflug_intrinsics(&data, &gt, opts, BackendSolveOptions::default())
            .unwrap();
        let Uncertainty::Available(cov) = &est.uncertainty else {
            panic!("no covariance: {:?}", est.uncertainty);
        };
        assert!(cov.fixed.contains(&"k3".to_string()), "{:?}", cov.fixed);
        assert!(cov.fixed.contains(&"p1".to_string()), "{:?}", cov.fixed);
        assert_eq!(cov.views.len(), gt.camera_se3_target.len());
        labels.clone_from(&cov.labels);
        let n = labels.len();
        let error = DVector::from_iterator(
            n,
            labels.iter().map(|l| value(&est.params, l) - value(&gt, l)),
        );
        let matrix = DMatrix::from_fn(n, n, |i, j| cov.covariance[i][j]);
        nees += (error.transpose() * matrix.clone().try_inverse().unwrap() * &error)[(0, 0)];
        let var = matrix.diagonal();
        predicted = Some(predicted.map_or(var.clone(), |acc| acc + var));
        errors.push(error);
    }
    assert_eq!(
        labels,
        ["fx", "fy", "cx", "cy", "k1", "k2", "tilt_x", "tilt_y"]
    );

    let runs = RUNS as f64;
    let n = labels.len();
    let predicted = (predicted.unwrap() / runs).map(f64::sqrt);
    let mean = errors.iter().fold(DVector::zeros(n), |a, e| a + e) / runs;
    let empirical = (errors
        .iter()
        .fold(DVector::zeros(n), |a, e| a + (e - &mean).map(|v| v * v))
        / (runs - 1.0))
        .map(f64::sqrt);
    for (i, label) in labels.iter().enumerate() {
        let ratio = predicted[i] / empirical[i];
        assert!(
            (ratio - 1.0).abs() < 0.10,
            "{label}: predicted σ {:.3e}, empirical {:.3e} (ratio {ratio:.3})",
            predicted[i],
            empirical[i]
        );
    }
    let k = n as f64;
    let nees = nees / runs;
    let band = 3.0 * (2.0 * k / runs).sqrt();
    assert!(
        (nees - k).abs() < band,
        "NEES mean {nees:.3}, expected {k} ± {band:.3}"
    );
}
