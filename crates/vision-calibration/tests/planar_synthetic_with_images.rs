//! Regression test for the `planar_synthetic_with_images` example
//! (Track B / ADR 0014 fixture for the diagnose UI).
//!
//! Calls the same generation function the example does, into a tempdir,
//! and asserts:
//!
//! - Every expected file is present (`export.json`, one PNG per view).
//! - The export's image manifest matches the rendered files.
//! - Per-feature residuals stay tight enough that any drift in the
//!   rendering / noise / Zhang pipeline shows up as a test failure rather
//!   than a silent fixture change.
//!
//! The synthetic scene is noiseless apart from a deterministic σ ≈ 0.3 px
//! pixel-noise stream seeded by `synthetic_images::NOISE_SEED`. Mean error
//! is therefore O(σ) and max error well below the bucket-1 boundary at
//! 1 px, so we can pin tight ceilings without flakiness.

use std::path::PathBuf;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsExport;

#[path = "../examples/support/synthetic_images.rs"]
mod synthetic_images;

#[test]
fn fixture_writes_files_and_residuals_stay_tight() {
    let tmp = tempdir_in_target("planar_synthetic_with_images_fixture_test");
    let summary = synthetic_images::write_fixture(&tmp).expect("fixture generation must succeed");

    // Files exist.
    assert!(tmp.join("export.json").is_file(), "export.json missing");
    for view_idx in 0..synthetic_images::NUM_VIEWS {
        let img = tmp
            .join("images")
            .join(format!("pose_{view_idx}_cam_0.png"));
        assert!(
            img.is_file(),
            "expected rendered image at {}",
            img.display()
        );
    }

    // Numbers are loose enough not to flake on a tiny solver tweak, tight
    // enough that a real regression (broken homography rendering, broken
    // noise stream, broken Zhang init) trips them.
    let expected_residuals =
        synthetic_images::NUM_VIEWS * synthetic_images::BOARD_COLS * synthetic_images::BOARD_ROWS;
    assert_eq!(summary.num_residuals, expected_residuals);
    assert!(
        summary.mean_error_px < 0.5,
        "mean reproj error {:.4} px exceeds 0.5 px ceiling",
        summary.mean_error_px
    );
    assert!(
        summary.max_error_px < 1.5,
        "max per-feature reproj error {:.4} px exceeds 1.5 px ceiling",
        summary.max_error_px
    );

    // Manifest survives serde and addresses every expected (pose, cam=0)
    // slot — this is what the diagnose UI will rely on.
    let json = std::fs::read_to_string(tmp.join("export.json")).expect("read export.json");
    let export: PlanarIntrinsicsExport =
        serde_json::from_str(&json).expect("export.json round-trips");
    let manifest = export
        .image_manifest
        .as_ref()
        .expect("ImageManifest must be populated for this fixture");
    assert_eq!(manifest.root, PathBuf::from("images"));
    assert_eq!(manifest.frames.len(), synthetic_images::NUM_VIEWS);
    for view_idx in 0..synthetic_images::NUM_VIEWS {
        let frame = manifest
            .frame(view_idx, 0)
            .unwrap_or_else(|| panic!("manifest missing entry for pose {view_idx} cam 0"));
        assert_eq!(
            frame.path,
            PathBuf::from(format!("pose_{view_idx}_cam_0.png"))
        );
        assert!(frame.roi.is_none(), "single-camera frames have no ROI");
    }
}

/// Create a fresh temp directory under the workspace `target/` so the
/// test does not leak files into `/tmp` and so the build cache wipes it
/// alongside the rest of `target/`.
fn tempdir_in_target(label: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("target");
    p.push("test-tmp");
    p.push(format!("{label}-{pid}", pid = std::process::id(),));
    if p.exists() {
        std::fs::remove_dir_all(&p).expect("clean prior test tempdir");
    }
    std::fs::create_dir_all(&p).expect("create test tempdir");
    p
}
