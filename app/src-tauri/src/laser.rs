//! `vision-metrology`-backed laser-line extraction (ADR 0021).
//!
//! The pipeline's laser converters take an injected
//! [`LaserPixelExtractor`] because `vision-metrology` is not on
//! crates.io and the published crates cannot depend on it. This app is
//! unpublished, so the reference implementation lives here.

use vision_calibration_dataset::{LaserExtractionSpec, LaserScanAxis};
use vision_calibration_pipeline::dataset_runner::LaserPixelExtractor;
use vision_metrology::{
    ColAccess, Edge1DConfig, ImageView, LaserExtractConfig, LaserExtractor, ScanAxis,
};

/// Subpixel DoG-edge laser-line extractor wrapping
/// `vision_metrology::LaserExtractor` — the implementation validated
/// against the rtv3d oracle in the V-track.
pub struct VmLaserExtractor;

impl LaserPixelExtractor for VmLaserExtractor {
    fn name(&self) -> &str {
        // Part of the detection-cache key (`laser:<name>`); bump when
        // the wrapped extractor changes behaviour.
        "vision-metrology-v0.1"
    }

    fn extract(
        &self,
        image: &image::DynamicImage,
        spec: &LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
        let luma = image.to_luma8();
        let width = luma.width() as usize;
        let height = luma.height() as usize;
        let view = ImageView::<u8>::from_slice(width, height, width, luma.as_raw())
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("image view ({width}x{height}): {e:?}").into()
            })?;

        let (axis, scan_len) = match spec.scan_axis {
            LaserScanAxis::Cols => (
                ScanAxis::Cols {
                    access: ColAccess::Gather,
                },
                width,
            ),
            LaserScanAxis::Rows => (ScanAxis::Rows, height),
        };
        let cfg = LaserExtractConfig {
            axis,
            edge_cfg: Edge1DConfig {
                sigma: spec.sigma as f32,
                pos_thresh: spec.pos_thresh as f32,
                neg_thresh: spec.neg_thresh as f32,
                ..Edge1DConfig::default()
            },
            ..LaserExtractConfig::default()
        };
        let mut extractor = LaserExtractor::new(cfg.edge_cfg.sigma);
        let line = extractor.extract_line_u8(&view, 0..scan_len, &cfg, None);

        Ok(line
            .points
            .into_iter()
            .map(|p| [p.x as f64, p.y as f64])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_a_synthetic_stripe() {
        // Bright 3-px horizontal stripe across a dark image.
        let mut img = image::GrayImage::new(80, 48);
        for x in 5..75 {
            let y = 20 + (x as i32 - 40).abs() / 16;
            for dy in -1..=1 {
                img.put_pixel(x, (y + dy) as u32, image::Luma([220]));
            }
        }
        let dynamic = image::DynamicImage::ImageLuma8(img);
        let points = VmLaserExtractor
            .extract(&dynamic, &LaserExtractionSpec::default())
            .unwrap();
        assert!(points.len() >= 20, "got {} points", points.len());
        // One point per column at most, near the stripe centre.
        for p in &points {
            assert!((p[1] - 21.0).abs() < 4.0, "outlier at {p:?}");
        }
    }

    #[test]
    fn rows_axis_scans_vertical_stripes() {
        // Bright vertical stripe → one detection per row.
        let mut img = image::GrayImage::new(48, 80);
        for y in 5..75 {
            for dx in -1i32..=1 {
                img.put_pixel((24 + dx) as u32, y, image::Luma([220]));
            }
        }
        let dynamic = image::DynamicImage::ImageLuma8(img);
        let spec = LaserExtractionSpec {
            scan_axis: LaserScanAxis::Rows,
            ..Default::default()
        };
        let points = VmLaserExtractor.extract(&dynamic, &spec).unwrap();
        assert!(points.len() >= 20, "got {} points", points.len());
        for p in &points {
            assert!((p[0] - 24.0).abs() < 4.0, "outlier at {p:?}");
        }
    }
}
