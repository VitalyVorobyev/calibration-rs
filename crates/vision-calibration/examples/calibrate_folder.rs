//! Calibrate a folder of your own images.
//!
//! The first run drafts `dataset.json` in the folder and stops: open it, fill
//! in what it lists under `_unresolved` (at least the board), delete that
//! list and run again. The second run detects the board, calibrates and
//! prints the camera, its uncertainty and a quality verdict.
//!
//! ```bash
//! cargo run --release -p vision-calibration --example calibrate_folder -- <folder> <width> <height>
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use vision_calibration::analysis::{CheckStatus, QualityThresholds, assess};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::control::RunControl;
use vision_calibration::core::Uncertainty;
use vision_calibration::dataset::{DatasetSpec, sniff_folder};
use vision_calibration::dataset_runner::{Detection, calibrate_dataset};
use vision_calibration::detect::FsDetectionCache;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, width, height] = args.as_slice() else {
        bail!("usage: calibrate_folder <folder> <image width> <image height>");
    };
    let dir = PathBuf::from(dir);
    let image_size = [width.parse()?, height.parse()?];

    // 1. Draft the manifest. The sniffer reads file and folder names only and
    //    lists what it cannot know, such as the board, under `_unresolved`.
    let manifest = dir.join("dataset.json");
    if !manifest.exists() {
        let draft = sniff_folder(&dir)?;
        std::fs::write(&manifest, serde_json::to_string_pretty(&draft)?)?;
        println!("wrote {}", manifest.display());
        println!(
            "fill in {:?}, delete `_unresolved`, run again",
            draft.unresolved
        );
        return Ok(());
    }

    // 2. Calibrate: detect the board in every image (cached for the next
    //    run), then run the workflow the manifest's topology names.
    let spec = DatasetSpec::from_path(&manifest).context("reading the manifest")?;
    let cache = FsDetectionCache::new(dir.join(".detections"));
    let run = calibrate_dataset(
        &spec,
        &dir,
        None,
        Detection::Cached(&cache),
        None,
        &RunControl::default(),
    )?;
    println!("{} of {} views used", run.usable_views, run.total_views);
    for rejected in &run.rejected {
        println!(
            "  left out {}: {:?}",
            rejected.path.display(),
            rejected.reason
        );
    }

    // 3. Each camera and the uncertainty of its parameters. With several
    //    cameras, a camera's covariance labels carry its index (`cam1.fx`).
    let export = &run.export;
    let cameras = export.camera_params();
    let uncertainty = export.parameter_uncertainty();
    for (i, camera) in cameras.iter().enumerate() {
        let k = &camera.intrinsics;
        println!(
            "camera {i}: fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
            k.fx, k.fy, k.cx, k.cy
        );
        if let Uncertainty::Available(cov) = uncertainty {
            for name in ["fx", "fy", "cx", "cy"] {
                let label = if cameras.len() > 1 {
                    format!("cam{i}.{name}")
                } else {
                    name.to_string()
                };
                if let Some(sigma) = cov.std_dev(&label) {
                    println!("  σ({name}) = {sigma:.3} px");
                }
            }
        }
    }
    if let Uncertainty::Unavailable { reason } = uncertainty {
        println!("no covariance: {reason:?}");
    }

    // 4. The verdict: pass, warn or fail, and why.
    // Every camera here shares one image size.
    let sizes = vec![image_size; cameras.len()];
    let report = assess(export, &sizes, &QualityThresholds::default())?;
    println!("verdict: {:?}", report.verdict);
    for check in report
        .checks
        .iter()
        .filter(|c| c.status != CheckStatus::Pass)
    {
        println!("  {:?} {:?}: {}", check.kind, check.status, check.message);
    }

    // 5. The export is the result to keep: JSON with a published schema.
    let out = dir.join("calibration.json");
    std::fs::write(&out, serde_json::to_string_pretty(export)?)?;
    println!("wrote {}", out.display());
    Ok(())
}
