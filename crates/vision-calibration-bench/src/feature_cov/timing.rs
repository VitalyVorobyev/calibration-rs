//! E5: what asking the detector for the covariance costs.
//!
//! Each image is decoded and cropped once, then detected with every arm in
//! turn (the order rotates from image to image), timing only the detector
//! call. The process must run its detector single-threaded
//! (`RAYON_NUM_THREADS=1`), which `calib-bench feature-cov` arranges by
//! running this stage in a child process.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use vision_calibration::detect::GrayImageView;
use vision_calibration_dataset::DatasetSpec;
use vision_calibration_pipeline::dataset_runner::detect_target;

use super::arms::Arm;
use super::stats::median;

/// One camera's images to time.
#[derive(Debug, Clone)]
pub struct TimingInput {
    /// `<dataset>/<camera>`.
    pub label: String,
    /// The camera's single-camera manifest.
    pub spec: DatasetSpec,
    /// Its images (those with a detected board).
    pub images: Vec<PathBuf>,
}

/// Detection times of one camera, or of every camera pooled.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingSummary {
    /// Images timed.
    pub images: usize,
    /// Median detection time per arm, ms.
    pub median_ms: BTreeMap<Arm, f64>,
    /// `median(arm) / median(uniform) − 1` per covariance arm.
    pub added: BTreeMap<Arm, f64>,
    /// Median over images of the per-image `arm / uniform − 1`.
    pub added_paired: BTreeMap<Arm, f64>,
}

/// E5 over every camera.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Timing {
    /// The detector ran single-threaded (`RAYON_NUM_THREADS=1`).
    pub single_threaded: bool,
    /// The build was a release build.
    pub release_build: bool,
    /// Per camera.
    pub units: BTreeMap<String, TimingSummary>,
    /// Every image of every camera.
    pub pooled: TimingSummary,
}

fn summarize(samples: &[BTreeMap<Arm, f64>]) -> TimingSummary {
    let arms = [Arm::Uniform, Arm::Known1, Arm::Estimated];
    let median_ms: BTreeMap<Arm, f64> = arms
        .iter()
        .map(|&a| (a, median(samples.iter().filter_map(|s| s.get(&a).copied()))))
        .collect();
    let mut added = BTreeMap::new();
    let mut added_paired = BTreeMap::new();
    for arm in [Arm::Known1, Arm::Estimated] {
        added.insert(arm, median_ms[&arm] / median_ms[&Arm::Uniform] - 1.0);
        added_paired.insert(
            arm,
            median(
                samples
                    .iter()
                    .filter_map(|s| Some(s.get(&arm)? / s.get(&Arm::Uniform)? - 1.0)),
            ),
        );
    }
    TimingSummary {
        images: samples.len(),
        median_ms,
        added,
        added_paired,
    }
}

/// Time detection on every camera's images, at most `limit` per camera.
pub fn measure(inputs: &[TimingInput], limit: Option<usize>) -> Result<Timing> {
    let arms = [Arm::Uniform, Arm::Known1, Arm::Estimated];
    let mut units = BTreeMap::new();
    let mut all = Vec::new();
    for input in inputs {
        super::progress(&input.label, "E5 detection timing");
        let specs: Vec<DatasetSpec> = arms.iter().map(|a| a.spec(&input.spec)).collect();
        let roi = input.spec.cameras[0].roi_xywh;
        let mut samples = Vec::new();
        let images = input.images.iter().take(limit.unwrap_or(usize::MAX));
        for (i, path) in images.enumerate() {
            let image =
                image::open(path).with_context(|| format!("{}: decode image {i}", input.label))?;
            let image = match roi {
                Some([x, y, w, h]) => image.crop_imm(x, y, w, h),
                None => image,
            }
            .to_luma8();
            let image = GrayImageView::new(image.width(), image.height(), image.as_raw())?;
            if i == 0 {
                // Warm-up, untimed.
                detect_target(image, &specs[0].target, specs[0].detector.as_ref())?;
            }
            let mut sample = BTreeMap::new();
            for k in 0..arms.len() {
                let a = (i + k) % arms.len();
                let start = Instant::now();
                detect_target(image, &specs[a].target, specs[a].detector.as_ref())
                    .with_context(|| format!("{}: detect image {i}", input.label))?;
                sample.insert(arms[a], start.elapsed().as_secs_f64() * 1000.0);
            }
            samples.push(sample);
        }
        units.insert(input.label.clone(), summarize(&samples));
        all.extend(samples);
    }
    Ok(Timing {
        single_threaded: std::env::var("RAYON_NUM_THREADS").is_ok_and(|v| v.trim() == "1"),
        release_build: !cfg!(debug_assertions),
        units,
        pooled: summarize(&all),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_cost_is_relative_to_the_uniform_median() {
        let sample = |u: f64, k: f64, e: f64| {
            BTreeMap::from([(Arm::Uniform, u), (Arm::Known1, k), (Arm::Estimated, e)])
        };
        let s = summarize(&[
            sample(10.0, 11.0, 12.0),
            sample(20.0, 22.0, 24.0),
            sample(30.0, 33.0, 36.0),
        ]);
        assert!((s.added[&Arm::Known1] - 0.1).abs() < 1e-12);
        assert!((s.added_paired[&Arm::Estimated] - 0.2).abs() < 1e-12);
        assert_eq!(s.images, 3);
    }
}
