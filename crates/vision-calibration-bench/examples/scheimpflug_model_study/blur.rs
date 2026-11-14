//! Per-marker fitted blur: the ring-grid detector's photometric diagnostics
//! for every image of a dataset, keyed by the marker's image position.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use ringgrid::{CircleRefinementMethod, Detector, TargetLayout};
use vision_calibration_bench::manifest::dataset_spec;
use vision_calibration_bench::registry::BenchEntry;
use vision_calibration_dataset::{ImagePattern, TargetSpec};

/// Position key at 1e-4 px.
pub fn key(px: [f64; 2]) -> (i64, i64) {
    ((px[0] * 1e4).round() as i64, (px[1] * 1e4).round() as i64)
}

/// Fitted blur σ (px) of every photometric marker, keyed by [`key`].
pub fn blur_map(entry: &BenchEntry) -> Result<BTreeMap<(i64, i64), f64>> {
    let spec = dataset_spec(entry)?;
    let TargetSpec::Ringgrid {
        pitch_m,
        rows,
        long_row_cols,
        marker_outer_radius_m,
        marker_inner_radius_m,
        marker_ring_width_m,
        ..
    } = spec.target
    else {
        bail!("{} is not a ring-grid dataset", entry.id);
    };
    let mm = |m: f64| (m * 1000.0) as f32;
    let layout = TargetLayout::coded_hex(
        mm(pitch_m),
        rows as usize,
        long_row_cols as usize,
        mm(marker_outer_radius_m),
        mm(marker_inner_radius_m),
        mm(marker_ring_width_m),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut detector = Detector::new(layout);
    detector.config_mut().circle_refinement = CircleRefinementMethod::Photometric;
    let camera = spec.cameras.first().context("no camera")?;
    let ImagePattern::Glob { pattern } = &camera.images else {
        bail!("only glob image patterns are supported");
    };
    let mut paths: Vec<_> = glob_files(&entry.data_root, pattern)?;
    paths.sort();
    let mut map = BTreeMap::new();
    for path in paths {
        let img = image::open(&path)
            .with_context(|| format!("open {}", path.display()))?
            .to_luma8();
        let img = match camera.roi_xywh {
            Some([x, y, w, h]) => image::imageops::crop_imm(&img, x, y, w, h).to_image(),
            None => img,
        };
        let (res, diag) = detector
            .detect_adaptive_with_diagnostics(&img)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        for (m, d) in res.detected_markers.iter().zip(&diag.markers) {
            if let Some(blur) = d.photometric.as_ref().and_then(|p| p.blur_sigma_px) {
                map.insert(key(m.center), blur);
            }
        }
    }
    Ok(map)
}

/// The files in `root` matching a single-directory glob `pattern` (`*` and
/// `?` in the file name only).
fn glob_files(root: &Path, pattern: &str) -> Result<Vec<std::path::PathBuf>> {
    let (dir, name) = match pattern.rsplit_once('/') {
        Some((d, n)) => (root.join(d), n.to_string()),
        None => (root.to_path_buf(), pattern.to_string()),
    };
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir)? {
        let p = e?.path();
        if let Some(f) = p.file_name().and_then(|f| f.to_str())
            && wildcard(&name, f)
        {
            out.push(p);
        }
    }
    Ok(out)
}

fn wildcard(p: &str, s: &str) -> bool {
    let (p, s) = (p.as_bytes(), s.as_bytes());
    let (mut i, mut j, mut star, mut mark) = (0, 0, None, 0);
    while j < s.len() {
        if i < p.len() && (p[i] == b'?' || p[i] == s[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == b'*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(st) = star {
            i = st + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    while i < p.len() && p[i] == b'*' {
        i += 1;
    }
    i == p.len()
}
