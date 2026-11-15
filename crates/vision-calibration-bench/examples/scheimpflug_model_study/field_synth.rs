//! `--parts fieldsweep`: whether a lens-side term is identifiable next to
//! the camera's own parameters away from the studied operating point.
//!
//! For a grid of synthetic cameras (sensor tilt 0–20°, focal length 600,
//! 1150 and 2300 px on a 720×540 sensor, the board spread over the whole
//! image or only over its lower two thirds), 20 noise-free views are
//! linearized at the truth and the covariance of the camera with each
//! candidate term is read from the normal equations: the largest
//! correlation of a term coefficient with a camera parameter, and how much
//! freeing the camera inflates the term's σ (1 when they are independent).

use anyhow::Result;
use nalgebra::{Point2, Point3};
use vision_calibration_core::{CorrespondenceView, NoMeta, View};

use crate::field_study::{add_extras, lens_cols};
use crate::linfield::{self, CAM_NAMES, Extra, NCAM, Spec};
use crate::model::{GenCfg, project_point};
use crate::synth::{self, SIZE};

/// The candidates of the sweep (lens columns, see `field_study::LENS`).
fn candidates() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("skew (Im w̄)", lens_cols(&[("w̄ (anamorphic)", Some(false))])),
        ("w̄² (m −3)", lens_cols(&[("w̄² m-3", None)])),
        ("w̄³ (m −4)", lens_cols(&[("w̄³ m-4", None)])),
        ("w̄² + w̄³", lens_cols(&[("w̄² m-3", None), ("w̄³ m-4", None)])),
        ("NAT B5", lens_cols(&[("NAT B5", None)])),
        ("NAT A5", lens_cols(&[("NAT A5", None)])),
    ]
}

pub fn sweep() -> Result<()> {
    println!(
        "\n## Identifiability sweep: noise-free, 20 views, camera + term linearized at the truth"
    );
    println!(
        "| tilt° | f px | coverage | points | {} |",
        candidates()
            .iter()
            .map(|c| format!("{}: max|corr| (param), σ inflation", c.0))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let plain = GenCfg::base("plain");
    for tilt_deg in [0.0f64, 5.0, 10.0, 20.0] {
        for f in [600.0, 1150.0, 2300.0] {
            for lower in [false, true] {
                let mut truth = synth::typical();
                truth.intr = [f, f * 1.008, 364.0, 271.0];
                truth.tilt = [-tilt_deg.to_radians(), 0.004];
                let half = [360.0 / f, 270.0 / f];
                let poses = synth::poses_fov(40, 11, half, 0.6);
                let mut views = Vec::new();
                let mut kept = Vec::new();
                for pose in &poses {
                    let (mut p3, mut p2) = (Vec::new(), Vec::new());
                    for j in 0..18 {
                        for i in 0..24 {
                            let b = [i as f64 * 0.005, j as f64 * 0.005];
                            if (pose * Point3::new(b[0], b[1], 0.0)).z <= 0.0 {
                                continue;
                            }
                            let uv = project_point(&plain, &truth, SIZE, pose, b);
                            let top = if lower { 180.0 } else { 2.0 };
                            if uv[0] < 2.0 || uv[1] < top || uv[0] > 718.0 || uv[1] > 538.0 {
                                continue;
                            }
                            p3.push(Point3::new(b[0], b[1], 0.0));
                            p2.push(Point2::new(uv[0], uv[1]));
                        }
                    }
                    if p3.len() >= 30 && views.len() < 20 {
                        views.push(View::new(CorrespondenceView::new(p3, p2)?, NoMeta));
                        kept.push(*pose);
                    }
                }
                let ids: Vec<usize> = (0..views.len()).collect();
                let mut pts = linfield::linearize(&truth, &truth, SIZE, &ids, &views, &kept);
                add_extras(&mut pts, &truth, SIZE);
                let mut cells = Vec::new();
                for (name, idx) in candidates() {
                    let basis = Extra(idx.clone());
                    let mut spec = Spec {
                        camera_free: true,
                        local: Some(&basis),
                        penalty: None,
                        ortho: false,
                        shared: None,
                        held: None,
                    };
                    let free = linfield::fit(&[&pts], &spec).cams[0].cov.clone();
                    spec.camera_free = false;
                    let held = linfield::fit(&[&pts], &spec).cams[0].cov.clone();
                    let (mut best, mut which, mut infl) = (0.0f64, "", 0.0f64);
                    for k in 0..idx.len() {
                        let i = NCAM + k;
                        for (j, pn) in CAM_NAMES.iter().enumerate() {
                            let r = free[(i, j)] / (free[(i, i)] * free[(j, j)]).sqrt();
                            if r.abs() > best.abs() {
                                (best, which) = (r, pn);
                            }
                        }
                        infl = infl.max((free[(i, i)] / held[(k, k)]).sqrt());
                    }
                    let _ = name;
                    cells.push(format!("{best:+.2} ({which}), {infl:.1}"));
                }
                println!(
                    "| {tilt_deg:.0} | {f:.0} | {} | {} | {} |",
                    if lower { "lower 2/3" } else { "full" },
                    pts.len(),
                    cells.join(" | ")
                );
            }
        }
    }
    Ok(())
}
