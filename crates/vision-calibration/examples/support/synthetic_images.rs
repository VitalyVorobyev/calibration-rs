//! Shared fixture-generation logic for the `planar_synthetic_with_images`
//! example and its regression test.
//!
//! Lives under `examples/support` so it can be `#[path]`-included from both
//! the example binary and the integration test without forcing the logic
//! into the public library surface. The function [`write_fixture`] is the
//! single entry point — it writes `export.json` and `images/*.png` into the
//! caller-supplied directory.
//!
//! `dead_code` is suppressed at module scope because the example binary
//! and the regression test exercise different subsets of the public items
//! (only the example needs `default_fixture_dir`; only the example reads
//! `FixtureSummary::export_path`).
//!
//! Determinism: pose set, board geometry, and per-feature noise are all
//! hard-coded constants seeded with [`NOISE_SEED`]. Any change to those
//! constants is a fixture change and will move the regression-test
//! tolerances.

#![allow(dead_code)]

use anyhow::Result;
use image::{ImageBuffer, Luma};
use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};
use std::path::{Path, PathBuf};
use vision_calibration::core::{
    BrownConrady5, CorrespondenceView, FrameRef, FxFyCxCySkew, ImageManifest, Iso3, PlanarDataset,
    Pt2, Pt3, View, make_pinhole_camera,
};
use vision_calibration::planar_intrinsics::{PlanarIntrinsicsExport, step_init, step_optimize};
use vision_calibration::prelude::{CalibrationSession, PlanarIntrinsicsProblem};
use vision_calibration::synthetic::noise::UniformPixelNoise;

// ─── Fixture parameters (kept here so the regression test can mirror them) ──

/// Image width in pixels.
pub const IMAGE_W: u32 = 640;
/// Image height in pixels.
pub const IMAGE_H: u32 = 480;
/// Inner-corner counts of the checkerboard (columns × rows).
pub const BOARD_COLS: usize = 9;
/// Inner-corner counts of the checkerboard (columns × rows).
pub const BOARD_ROWS: usize = 6;
/// Square edge length in metres.
pub const SQUARE_M: f64 = 0.030;
/// Number of distinct views.
pub const NUM_VIEWS: usize = 5;
/// Per-axis pixel-noise amplitude (uniform). Tight enough that Zhang
/// recovers GT intrinsics within tolerance and the regression test can
/// pin a residual ceiling.
pub const NOISE_PX: f64 = 0.30;
/// Seed for the deterministic noise stream.
pub const NOISE_SEED: u64 = 0xCAFE_F00D;

/// Outcome summary returned by [`write_fixture`] so callers can log /
/// assert on it without re-parsing the export JSON.
pub struct FixtureSummary {
    /// Path to the written `export.json`.
    pub export_path: PathBuf,
    /// Total per-feature residual records emitted.
    pub num_residuals: usize,
    /// Mean reprojection error (pixels) reported by the export.
    pub mean_error_px: f64,
    /// Maximum per-feature reprojection error (pixels) over all records
    /// where projection did not diverge.
    pub max_error_px: f64,
}

/// Generate the fixture into `out_dir`. Creates `out_dir` and `out_dir/images/`
/// if missing.
pub fn write_fixture(out_dir: &Path) -> Result<FixtureSummary> {
    let images_dir = out_dir.join("images");
    std::fs::create_dir_all(&images_dir)?;

    let k_gt = FxFyCxCySkew {
        fx: 600.0,
        fy: 600.0,
        cx: (IMAGE_W as f64) * 0.5,
        cy: (IMAGE_H as f64) * 0.5,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5::default();
    let cam_gt = make_pinhole_camera(k_gt, dist_gt);

    let board_points = checkerboard_inner_corners();
    let poses = view_poses();
    let noise = UniformPixelNoise {
        seed: NOISE_SEED,
        max_abs_px: NOISE_PX,
    };
    let k_matrix = intrinsics_matrix(&k_gt);

    let mut views = Vec::with_capacity(NUM_VIEWS);
    let mut frames = Vec::with_capacity(NUM_VIEWS);

    for (view_idx, pose) in poses.iter().enumerate() {
        let mut points_3d = Vec::with_capacity(board_points.len());
        let mut points_2d_clean = Vec::with_capacity(board_points.len());
        for p_target in &board_points {
            let p_cam = pose.transform_point(p_target);
            let uv = cam_gt
                .project_point(&p_cam)
                .ok_or_else(|| anyhow::anyhow!("pose {view_idx}: corner not projectable"))?;
            points_3d.push(*p_target);
            points_2d_clean.push(uv);
        }

        let img = render_checkerboard_image(&k_matrix, pose);
        let rel_path = format!("pose_{view_idx}_cam_0.png");
        img.save(images_dir.join(&rel_path))?;

        let points_2d_noisy: Vec<Pt2> = points_2d_clean
            .iter()
            .enumerate()
            .map(|(point_idx, uv)| {
                let v = noise.apply(view_idx, point_idx, uv.coords);
                Pt2::new(v.x, v.y)
            })
            .collect();

        views.push(View::without_meta(
            CorrespondenceView::new(points_3d, points_2d_noisy)
                .map_err(|e| anyhow::anyhow!("view {view_idx}: {e}"))?,
        ));
        frames.push(FrameRef {
            pose: view_idx,
            camera: 0,
            path: PathBuf::from(rel_path),
            roi: None,
        });
    }

    let dataset = PlanarDataset::new(views)?;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;
    step_init(&mut session, None)?;
    step_optimize(&mut session, None)?;

    let mut export: PlanarIntrinsicsExport = session.export()?;
    export.image_manifest = Some(ImageManifest {
        root: PathBuf::from("images"),
        frames,
    });

    let export_path = out_dir.join("export.json");
    let file = std::fs::File::create(&export_path)?;
    serde_json::to_writer_pretty(file, &export)?;

    let max_error_px = export
        .per_feature_residuals
        .target
        .iter()
        .filter_map(|r| r.error_px)
        .fold(0.0_f64, f64::max);

    Ok(FixtureSummary {
        export_path,
        num_residuals: export.per_feature_residuals.target.len(),
        mean_error_px: export.mean_reproj_error,
        max_error_px,
    })
}

/// Default output directory under the workspace `target/`. Co-located with
/// cargo's build artefacts so `cargo clean` wipes it; not checked in.
pub fn default_fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `crates/vision-calibration/Cargo.toml` → workspace root is two levels up.
    p.pop();
    p.pop();
    p.push("target");
    p.push("fixtures");
    p.push("planar_synthetic_with_images");
    p
}

fn checkerboard_inner_corners() -> Vec<Pt3> {
    let mut pts = Vec::with_capacity(BOARD_COLS * BOARD_ROWS);
    let dx = (BOARD_COLS as f64 - 1.0) * 0.5 * SQUARE_M;
    let dy = (BOARD_ROWS as f64 - 1.0) * 0.5 * SQUARE_M;
    for j in 0..BOARD_ROWS {
        for i in 0..BOARD_COLS {
            pts.push(Pt3::new(
                i as f64 * SQUARE_M - dx,
                j as f64 * SQUARE_M - dy,
                0.0,
            ));
        }
    }
    pts
}

/// Hand-tuned per-view poses (cam_from_target). Each pose holds the board
/// roughly centred in the 640×480 frame at ~0.6 m, with varied yaw + pitch
/// + roll so Zhang has well-conditioned homographies.
fn view_poses() -> Vec<Iso3> {
    let make = |yaw: f64, pitch: f64, roll: f64, z: f64| -> Iso3 {
        let r = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Iso3::from_parts(Translation3::new(0.0, 0.0, z), r)
    };
    vec![
        make(0.00, 0.00, 0.00, 0.55),
        make(0.30, 0.05, 0.00, 0.60),
        make(-0.25, -0.10, 0.05, 0.65),
        make(0.10, 0.20, -0.05, 0.50),
        make(-0.15, 0.18, 0.08, 0.58),
    ]
}

fn intrinsics_matrix(k: &FxFyCxCySkew<f64>) -> Matrix3<f64> {
    Matrix3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

/// Render a grayscale checkerboard image for one `cam_from_target` pose.
///
/// With zero distortion the planar target → image map is exactly a 3×3
/// homography `H = K · [r1 r2 t]`, so we inverse-map every pixel back to
/// the target plane and sample the checker pattern.
fn render_checkerboard_image(
    k_matrix: &Matrix3<f64>,
    pose: &Iso3,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let r = pose.rotation.to_rotation_matrix();
    let r_mat = r.matrix();
    let t = pose.translation.vector;
    let r1 = r_mat.column(0);
    let r2 = r_mat.column(1);
    let h = k_matrix
        * Matrix3::from_columns(&[
            Vector3::new(r1[0], r1[1], r1[2]),
            Vector3::new(r2[0], r2[1], r2[2]),
            t,
        ]);
    let h_inv = h
        .try_inverse()
        .expect("planar→image homography must be invertible for valid poses");

    let board_half_w = (BOARD_COLS as f64 - 1.0) * 0.5 * SQUARE_M;
    let board_half_h = (BOARD_ROWS as f64 - 1.0) * 0.5 * SQUARE_M;
    // Extend the textured area one square past the inner corners so the
    // outermost corners sit on a clear black/white edge.
    let pad = SQUARE_M;
    let board_min_x = -board_half_w - pad;
    let board_max_x = board_half_w + pad;
    let board_min_y = -board_half_h - pad;
    let board_max_y = board_half_h + pad;

    let bg: Luma<u8> = Luma([180]);
    let white: Luma<u8> = Luma([240]);
    let black: Luma<u8> = Luma([20]);

    ImageBuffer::from_fn(IMAGE_W, IMAGE_H, |u, v| {
        let p_pix = Vector3::new(u as f64 + 0.5, v as f64 + 0.5, 1.0);
        let p_target_h = h_inv * p_pix;
        let w = p_target_h[2];
        if w.abs() < 1e-12 {
            return bg;
        }
        let x = p_target_h[0] / w;
        let y = p_target_h[1] / w;

        if x < board_min_x || x > board_max_x || y < board_min_y || y > board_max_y {
            return bg;
        }

        let ix = ((x - board_min_x) / SQUARE_M).floor() as i64;
        let iy = ((y - board_min_y) / SQUARE_M).floor() as i64;
        if (ix + iy).rem_euclid(2) == 0 {
            white
        } else {
            black
        }
    })
}
