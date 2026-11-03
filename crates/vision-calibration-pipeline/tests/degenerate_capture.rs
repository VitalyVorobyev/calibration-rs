//! A capture whose board planes are parallel in every view (all
//! fronto-parallel, or all tilted the same way) does not determine the
//! intrinsics: the planar initialization returns the linear crate's typed
//! `DegenerateViews` error, unchanged through the pipeline, for a plain and
//! a Scheimpflug pinhole camera alike.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3, UnitQuaternion, Vector3};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::synthetic::planar::{grid_points, project_views_noisy};
use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, DistortionKind, FxFyCxCy, Iso3, PlanarDataset, SensorKind,
    View, make_pinhole_camera,
};
use vision_calibration_linear::Error as LinearError;
use vision_calibration_pipeline::Error;
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init,
};
use vision_calibration_pipeline::session::CalibrationSession;

/// Ten views of a 12 × 9 board, 0.65 m away, over the image quadrants and
/// its centre, each turned in its plane, all at the orientation `tilt`.
fn parallel_capture(tilt: Rotation3<f64>) -> PlanarDataset {
    let camera = make_pinhole_camera(
        FxFyCxCy {
            fx: 1000.0,
            fy: 1000.0,
            cx: 640.0,
            cy: 480.0,
        },
        BrownConrady5 {
            k1: -0.15,
            ..BrownConrady5::default()
        },
    );
    let board = grid_points(12, 9, 0.03);
    let centre = Vector3::new(0.165, 0.12, 0.0);
    let offsets = [
        (-0.3, -0.22),
        (0.3, -0.22),
        (-0.3, 0.22),
        (0.3, 0.22),
        (0.0, 0.0),
    ];
    let poses: Vec<Iso3> = (0..10)
        .map(|k| {
            let (x, y) = offsets[k % offsets.len()];
            let spin = Rotation3::from_euler_angles(0.0, 0.0, 0.3 * k as f64);
            let rotation = UnitQuaternion::from_rotation_matrix(&(tilt * spin));
            let at = Vector3::new(x, y, 1.0).normalize() * 0.65 - rotation * centre;
            Iso3::from_parts(Translation3::from(at), rotation)
        })
        .collect();
    let noise = UniformPixelNoise {
        seed: 3,
        max_abs_px: 0.35,
    };
    let views = project_views_noisy(&camera, &board, &poses, &noise).expect("projection");
    PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
}

#[test]
fn parallel_boards_are_a_typed_error_through_planar_init() {
    for sensor in [SensorKind::Identity, SensorKind::Scheimpflug] {
        for tilt in [
            Rotation3::identity(),
            Rotation3::from_euler_angles(0.35, 0.0, 0.0),
        ] {
            let mut config = PlanarIntrinsicsConfig::default();
            config.camera.model = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, sensor);
            let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
            session.set_config(config).expect("config");
            session.set_input(parallel_capture(tilt)).expect("input");
            let err = step_init(&mut session, None).expect_err("parallel boards");
            assert!(
                matches!(
                    err,
                    Error::Linear(LinearError::DegenerateViews { views: 10, .. })
                ),
                "{sensor:?}: {err:?}"
            );
            assert!(
                err.to_string()
                    .contains("tilt the board about both of its axes"),
                "{err}"
            );
        }
    }
}
