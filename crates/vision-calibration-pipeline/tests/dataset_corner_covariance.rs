//! End to end: a manifest asking for corner covariance reaches the
//! calibration through the dataset runner, on rendered chessboard images.
//!
//! The scene is a printed board rendered by `calib-targets`, warped into six
//! views of an ideal pinhole camera. The test checks the plumbing (the
//! observations carry covariances; the residual noise is reported as σ̂₀),
//! not the statistical quality of the covariances.

#![allow(missing_docs)]

use std::path::Path;

use nalgebra::{Matrix3, Rotation3, Vector3};
use serde_json::json;
use vision_calibration_core::{NoiseLevel, ResidualKind, Uncertainty};
use vision_calibration_dataset::DatasetSpec;
use vision_calibration_detect::FsDetectionCache;
use vision_calibration_pipeline::dataset_runner::Detection;
use vision_calibration_pipeline::{
    dataset_runner::{RunError, build_planar_input},
    planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, run_calibration},
    session::CalibrationSession,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FOCAL: f64 = 600.0;
const DPI: u32 = 50;

/// The rendered board as a gray image, with its pixel size in metres and the
/// board's center in image pixels.
struct Scene {
    image: image::GrayImage,
    metres_per_px: f64,
    center_px: [f64; 2],
}

fn render_scene() -> Scene {
    let mut doc = calib_targets::generate::chessboard_document(6, 8, 20.0);
    doc.render.png_dpi = DPI;
    let layout = doc.resolve_layout().expect("layout");
    let bundle = calib_targets::printable::render_target_bundle(&doc).expect("render bundle");
    let image = image::load_from_memory(&bundle.png_bytes)
        .expect("decode rendered PNG")
        .to_luma8();
    let px_per_mm = f64::from(DPI) / 25.4;
    // 9 x 7 squares of 20 mm.
    let center_mm = [
        layout.board_origin_mm[0] + 90.0,
        layout.board_origin_mm[1] + 70.0,
    ];
    Scene {
        image,
        metres_per_px: 1.0e-3 / px_per_mm,
        center_px: [center_mm[0] * px_per_mm, center_mm[1] * px_per_mm],
    }
}

/// The view of the board seen with the given pose, as a `WIDTH x HEIGHT` image.
fn render_view(scene: &Scene, rx: f64, ry: f64, rz: f64, distance: f64) -> image::GrayImage {
    let r = Rotation3::from_euler_angles(rx, ry, rz);
    let t = Vector3::new(0.0, 0.0, distance);
    let k = Matrix3::new(FOCAL, 0.0, 320.0, 0.0, FOCAL, 240.0, 0.0, 0.0, 1.0);
    let cols = r.matrix().fixed_columns::<2>(0);
    let h = k * Matrix3::from_columns(&[cols.column(0).into(), cols.column(1).into(), t]);
    let h_inv = h.try_inverse().expect("invertible homography");
    let (w, hgt) = scene.image.dimensions();
    image::GrayImage::from_fn(WIDTH, HEIGHT, |u, v| {
        let p = h_inv * Vector3::new(f64::from(u), f64::from(v), 1.0);
        let (x, y) = (p.x / p.z, p.y / p.z);
        let sx = x / scene.metres_per_px + scene.center_px[0];
        let sy = y / scene.metres_per_px + scene.center_px[1];
        if sx < 0.0 || sy < 0.0 || sx >= f64::from(w - 1) || sy >= f64::from(hgt - 1) {
            return image::Luma([255]);
        }
        let (x0, y0) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - x0, sy - y0);
        let at =
            |dx: u32, dy: u32| f64::from(scene.image.get_pixel(x0 as u32 + dx, y0 as u32 + dy)[0]);
        let value = at(0, 0) * (1.0 - fx) * (1.0 - fy)
            + at(1, 0) * fx * (1.0 - fy)
            + at(0, 1) * (1.0 - fx) * fy
            + at(1, 1) * fx * fy;
        image::Luma([value.round() as u8])
    })
}

fn write_views(dir: &Path) -> Vec<String> {
    let scene = render_scene();
    let poses = [
        (0.0, 0.0, 0.0, 0.40),
        (0.35, 0.0, 0.1, 0.38),
        (-0.35, 0.1, -0.1, 0.40),
        (0.1, 0.40, 0.0, 0.42),
        (-0.1, -0.40, 0.2, 0.40),
        (0.25, -0.25, -0.2, 0.37),
    ];
    poses
        .iter()
        .enumerate()
        .map(|(i, &(rx, ry, rz, d))| {
            let name = format!("view{i}.png");
            render_view(&scene, rx, ry, rz, d)
                .save(dir.join(&name))
                .expect("write view");
            name
        })
        .collect()
}

fn spec(files: &[String], detector: serde_json::Value) -> DatasetSpec {
    serde_json::from_value(json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": files } }],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
        "detector": detector,
    }))
    .expect("manifest")
}

fn reprojection_noise(uncertainty: &Uncertainty) -> NoiseLevel {
    let Uncertainty::Available(cov) = uncertainty else {
        panic!("no parameter covariance: {uncertainty:?}");
    };
    cov.noise
        .iter()
        .find(|n| n.kind == ResidualKind::Reprojection)
        .expect("a reprojection noise level")
        .level
}

#[test]
fn covariance_in_the_manifest_whitens_the_calibration() {
    let dir = tempfile::tempdir().unwrap();
    let files = write_views(dir.path());
    let cache = FsDetectionCache::new(dir.path().join("cache"));

    // Off (the default): no covariances, residual noise in pixels.
    let off = build_planar_input(
        &spec(&files, json!({})),
        dir.path(),
        Detection::Cached(&cache),
    )
    .unwrap();
    assert!(off.usable_views >= 4, "{} usable views", off.usable_views);
    assert!(
        off.input
            .views
            .iter()
            .all(|v| v.obs.covariances().is_none())
    );
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(off.input.clone()).unwrap();
    session
        .set_config(PlanarIntrinsicsConfig::default())
        .unwrap();
    run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    assert!(matches!(
        reprojection_noise(&export.uncertainty),
        NoiseLevel::Sigma(_)
    ));

    // On: every view carries covariances; the noise is σ̂₀.
    let detector =
        json!({ "chess_corners": { "covariance": { "image_noise": { "known": 2.0 } } } });
    let on = build_planar_input(
        &spec(&files, detector),
        dir.path(),
        Detection::Cached(&cache),
    )
    .unwrap();
    assert_eq!(on.usable_views, off.usable_views);
    assert!(on.input.views.iter().all(|v| v.obs.covariances().is_some()));
    // The same corners, with the same positions.
    for (a, b) in on.input.views.iter().zip(&off.input.views) {
        assert_eq!(a.obs.points_2d, b.obs.points_2d);
    }
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(on.input).unwrap();
    session
        .set_config(PlanarIntrinsicsConfig::default())
        .unwrap();
    run_calibration(&mut session).unwrap();
    let export = session.export().unwrap();
    assert!(matches!(
        reprojection_noise(&export.uncertainty),
        NoiseLevel::Sigma0(_)
    ));
}

#[test]
fn covariance_on_a_target_without_one_fails_before_detection() {
    let dir = tempfile::tempdir().unwrap();
    let cache = FsDetectionCache::new(dir.path().join("cache"));
    let spec: DatasetSpec = serde_json::from_value(json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [{ "id": "cam0", "images": { "kind": "list", "paths": ["missing.png"] } }],
        "target": { "kind": "puzzleboard", "layout": "puzzle_10x12", "cell_size_m": 0.01 },
        "detector": { "chess_corners": { "covariance": {} } },
    }))
    .unwrap();
    // The image does not exist: only a check made before any I/O gets here.
    let err = build_planar_input(&spec, dir.path(), Detection::Cached(&cache)).unwrap_err();
    assert!(
        matches!(
            err,
            RunError::CovarianceUnsupported {
                target: "puzzleboard"
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_view_without_corners_is_not_a_covariance_fault() {
    // A flat gray image has no board. A view with no corners has no corner
    // missing a covariance, so the run fails for lack of usable views.
    let dir = tempfile::tempdir().unwrap();
    image::GrayImage::from_pixel(64, 64, image::Luma([128]))
        .save(dir.path().join("blank.png"))
        .unwrap();
    let cache = FsDetectionCache::new(dir.path().join("cache"));
    let detector = json!({ "chess_corners": { "covariance": {} } });
    let err = build_planar_input(
        &spec(&["blank.png".to_string()], detector),
        dir.path(),
        Detection::Cached(&cache),
    )
    .unwrap_err();
    assert!(
        matches!(err, RunError::InsufficientUsableViews { .. }),
        "{err:?}"
    );
}
