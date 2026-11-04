//! End to end: a stereo rig calibrates from a plain chessboard turned through
//! every quarter turn, through the dataset runner and the real detector.
//!
//! The board is rendered by `calib-targets` into the images of two ideal
//! pinhole cameras. The detector labels each image from its own top-left
//! corner, so the board's labels follow its turn in each image; the runner
//! must give the two cameras of a view the same labels for the rig to come
//! out right. One rig has the cameras mounted alike; in the other, camera 1
//! is mounted upside down, so that its detector labels every view from the
//! opposite corner.

#![allow(missing_docs)]

use std::f64::consts::{FRAC_PI_2, PI};
use std::path::Path;

use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};
use serde_json::json;
use vision_calibration_core::RunControl;
use vision_calibration_dataset::DatasetSpec;
use vision_calibration_optim::SolverBackend;
use vision_calibration_pipeline::dataset_runner::{Detection, build_rig_extrinsics_input};
use vision_calibration_pipeline::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsProblem, run_calibration,
};
use vision_calibration_pipeline::session::CalibrationSession;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FOCAL: f64 = 600.0;
const DPI: u32 = 50;
const BASELINE: f64 = 0.08;

/// The rendered board as a gray image, with its pixel size in metres and the
/// board's centre in image pixels.
struct Scene {
    image: image::GrayImage,
    metres_per_px: f64,
    center_px: [f64; 2],
}

fn render_scene() -> Scene {
    // 6 × 8 interior corners: 9 × 7 squares of 20 mm.
    let mut doc = calib_targets::generate::chessboard_document(6, 8, 20.0);
    doc.render.png_dpi = DPI;
    let layout = doc.resolve_layout().expect("layout");
    let bundle = calib_targets::printable::render_target_bundle(&doc).expect("render bundle");
    let image = image::load_from_memory(&bundle.png_bytes)
        .expect("decode rendered PNG")
        .to_luma8();
    let px_per_mm = f64::from(DPI) / 25.4;
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

/// The board seen with `cam_se3_board` (board frame at its centre) by an
/// ideal pinhole camera, as a `WIDTH × HEIGHT` image.
fn render_view(scene: &Scene, cam_se3_board: &Isometry3<f64>) -> image::GrayImage {
    let r = cam_se3_board.rotation.to_rotation_matrix();
    let t = cam_se3_board.translation.vector;
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

/// Camera 1 of the rig, `cam1_se3_cam0`: `BASELINE` to the right of camera
/// 0, toed in slightly, and rolled by `roll` about its optical axis.
fn cam1_se3_cam0(roll: f64) -> Isometry3<f64> {
    let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, roll)
        * UnitQuaternion::from_euler_angles(0.01, -0.03, 0.0);
    let centre = Vector3::new(BASELINE, 0.0, 0.0);
    Isometry3::from_parts(Translation3::from(-(rotation * centre)), rotation)
}

/// Twelve board poses in camera 0, the board's in-plane turn running through
/// every quarter: `(tilt x, tilt y, turn, offset x, offset y, distance)`,
/// the offset from the point between the two cameras, so both see the
/// whole board.
fn board_poses() -> Vec<Isometry3<f64>> {
    let poses = [
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.45),
        (0.35, 0.0, FRAC_PI_2 + 0.1, 0.04, 0.02, 0.45),
        (-0.3, 0.15, PI - 0.15, -0.03, 0.03, 0.47),
        (0.1, 0.4, -FRAC_PI_2, 0.02, -0.04, 0.46),
        (-0.15, -0.4, 0.3, -0.04, -0.02, 0.44),
        (0.3, -0.25, FRAC_PI_2 - 0.3, 0.03, 0.04, 0.46),
        (-0.35, 0.0, PI + 0.25, -0.02, -0.03, 0.45),
        (0.2, 0.3, -FRAC_PI_2 + 0.2, 0.04, -0.01, 0.47),
        (0.0, -0.35, -0.2, -0.04, 0.03, 0.43),
        (0.25, 0.2, FRAC_PI_2 + 0.35, 0.01, -0.04, 0.46),
        (-0.2, -0.2, PI, 0.03, 0.01, 0.48),
        (0.15, -0.1, -FRAC_PI_2 - 0.3, -0.03, -0.03, 0.45),
    ];
    poses
        .iter()
        .map(|&(rx, ry, turn, x, y, d)| {
            let r = Rotation3::from_euler_angles(rx, ry, 0.0)
                * Rotation3::from_euler_angles(0.0, 0.0, turn);
            let t = Vector3::new(BASELINE / 2.0 + x, y, d);
            Isometry3::from_parts(
                Translation3::from(t),
                UnitQuaternion::from_rotation_matrix(&r),
            )
        })
        .collect()
}

fn write_rig(dir: &Path, cam1_se3_cam0: &Isometry3<f64>) -> [Vec<String>; 2] {
    let scene = render_scene();
    let mut files = [Vec::new(), Vec::new()];
    for (i, cam0_se3_board) in board_poses().iter().enumerate() {
        for (cam, pose) in [*cam0_se3_board, cam1_se3_cam0 * cam0_se3_board]
            .iter()
            .enumerate()
        {
            let name = format!("cam{cam}_{i:02}.png");
            render_view(&scene, pose)
                .save(dir.join(&name))
                .expect("write view");
            files[cam].push(name);
        }
    }
    files
}

fn spec(files: &[Vec<String>; 2]) -> DatasetSpec {
    serde_json::from_value(json!({
        "version": 1,
        "topology": "rig_extrinsics",
        "cameras": [
            { "id": "cam0", "images": { "kind": "list", "paths": files[0] } },
            { "id": "cam1", "images": { "kind": "list", "paths": files[1] } },
        ],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
        "pose_pairing": { "kind": "by_index" },
    }))
    .expect("manifest")
}

/// Calibrate the rig rendered with camera 1 rolled by `roll`, on both
/// solver backends, and compare camera 1's pose with the truth.
fn calibrates_through_every_turn(roll: f64) {
    let truth = cam1_se3_cam0(roll);
    let dir = tempfile::tempdir().unwrap();
    let files = write_rig(dir.path(), &truth);
    let run = build_rig_extrinsics_input(
        &spec(&files),
        dir.path(),
        Detection::Uncached,
        &RunControl::default(),
    )
    .unwrap();
    assert!(run.rejected.is_empty(), "{:?}", run.rejected);
    assert_eq!(run.usable_views, board_poses().len());
    for view in &run.input.views {
        // The two cameras label the same corners alike.
        let (a, b) = (
            view.obs.cameras[0].as_ref().unwrap(),
            view.obs.cameras[1].as_ref().unwrap(),
        );
        assert_eq!(a.points_3d.len(), 48);
        assert_eq!(b.points_3d.len(), 48);
    }
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let mut config = RigExtrinsicsConfig::default();
        config.solver.backend = backend;
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_config(config).unwrap();
        session.set_input(run.input.clone()).unwrap();
        run_calibration(&mut session).unwrap();
        let export = session.export().unwrap();
        assert!(
            export.mean_reproj_error < 0.5,
            "{backend:?}: mean reprojection error {} px",
            export.mean_reproj_error
        );
        let estimate = export.cam_se3_rig[1];
        // The rendering limits the corners to about 0.25 px; a camera whose
        // labels disagree with camera 0's is off by half a turn.
        let error = truth.inverse() * estimate;
        let (dt, dr) = (error.translation.vector.norm(), error.rotation.angle());
        assert!(
            dt < 3e-3 && dr < 0.5f64.to_radians(),
            "{backend:?}: camera 1 off by {:.3} mm and {:.4}°",
            dt * 1e3,
            dr.to_degrees()
        );
    }
}

#[test]
fn a_rig_calibrates_from_a_board_turned_through_every_quarter() {
    calibrates_through_every_turn(0.0);
}

#[test]
fn a_rig_with_a_camera_mounted_upside_down_matches_its_labels() {
    calibrates_through_every_turn(PI);
}
