//! `vcal project compare`: a run comparison as a table.

use std::fmt::Write as _;

use vision_calibration::analysis::{
    CameraEvidence, CovarianceStatus, DataRelation, DifferenceTest, ParameterComparison,
    PoseDifference, PoseTest, ProjectionComparison, RunComparison, RunDifference, RunSide,
};

use super::name;

/// The comparison as text: each run's evidence, then each run against the
/// reference, camera by camera.
pub(crate) fn render(comparison: &RunComparison) -> String {
    let mut out = String::new();
    let reference = &comparison.runs[0].name;
    for (i, run) in comparison.runs.iter().enumerate() {
        let role = if i == 0 { " (reference)" } else { "" };
        let covariance = match &run.covariance {
            CovarianceStatus::Available {
                view_scatter: Some(s),
                ..
            } => format!("covariance (scaled ×{s:.2} by the view scatter)"),
            CovarianceStatus::Available { .. } => "covariance".to_string(),
            CovarianceStatus::Unavailable { reason } => format!("no covariance ({})", tag(reason)),
            _ => "-".to_string(),
        };
        let _ = writeln!(
            out,
            "{}{role}: {}, verdict {}, {covariance}",
            run.name,
            name(&run.kind),
            name(&run.verdict)
        );
        for c in &run.cameras {
            let _ = writeln!(out, "  {}", camera_line(c));
        }
    }
    for difference in &comparison.differences {
        let other = &comparison.runs[difference.run].name;
        out.push('\n');
        render_difference(&mut out, difference, reference, other);
    }
    out.push_str(
        "\nΔ = other − reference. Δ/σ_ref is the change in units of the reference's stated σ \
         (materiality), not a test; a test is shown only where the data relation makes one \
         valid, its p two-sided (χ²: upper tail).\n",
    );
    out
}

fn camera_line(c: &CameraEvidence) -> String {
    let fit = match &c.reprojection {
        Some(s) => {
            let percentile =
                |p: Option<f64>| p.map_or_else(|| "-".to_string(), |v| format!("{v:.3}"));
            format!(
                "rms {:.3} px, median {}, p95 {}",
                s.rms,
                percentile(s.median),
                percentile(s.p95)
            )
        }
        None => "no residuals".to_string(),
    };
    let sigma = c
        .max_projection_sigma_px
        .map_or_else(String::new, |s| format!(", projection σ ≤ {s:.3} px"));
    format!(
        "{}: {} views, {} features, {fit}{sigma}",
        c.id, c.views, c.features
    )
}

fn render_difference(out: &mut String, d: &RunDifference, reference: &str, other: &str) {
    let relation = match d.relation {
        DataRelation::Nested { smaller } => format!(
            "nested ({} has the subset)",
            if smaller == RunSide::Other {
                other
            } else {
                reference
            }
        ),
        r => tag(&r),
    };
    let fit = if d.fit_comparable {
        "residuals comparable"
    } else {
        "residuals not comparable (other observations)"
    };
    let _ = writeln!(out, "{other} vs {reference}: {relation}; {fit}");
    if let Some(data) = &d.data {
        let _ = writeln!(
            out,
            "  data: {} shared images ({} only in {reference}, {} only in {other}); \
             {} shared features ({} / {})",
            data.shared_images,
            data.reference_only_images,
            data.other_only_images,
            data.shared_features,
            data.reference_only_features,
            data.other_only_features
        );
    }
    if let Some(changes) = &d.exclusion_changes {
        for (who, e) in [
            (reference, &changes.reference_only),
            (other, &changes.other_only),
        ] {
            if !e.is_empty() {
                let _ = writeln!(
                    out,
                    "  excluded only in {who}: {} views, {} images, {} feature sets",
                    e.views.len(),
                    e.images.len(),
                    e.features.len()
                );
            }
        }
    }
    match &d.config_changes {
        Some(changes) if changes.is_empty() => {
            let _ = writeln!(out, "  configuration: the same");
        }
        Some(changes) => {
            for c in changes {
                let _ = writeln!(
                    out,
                    "  configuration {}: {} -> {}",
                    c.path,
                    c.reference.as_deref().unwrap_or("(none)"),
                    c.other.as_deref().unwrap_or("(none)")
                );
            }
        }
        None => {}
    }
    for id in &d.model_changes {
        let _ = writeln!(out, "  camera {id}: another camera model");
    }
    for id in &d.unmatched_cameras {
        let _ = writeln!(out, "  camera {id}: in one run only");
    }
    for camera in &d.cameras {
        let _ = writeln!(out, "  camera {}", camera.id);
        if let ParameterComparison::SameModel { parameters } = &camera.parameters {
            let _ = writeln!(
                out,
                "    {:<10} {:>22} {:>22} {:>12} {:>9}  test",
                "parameter", reference, other, "Δ", "Δ/σ_ref"
            );
            for p in parameters {
                let _ = writeln!(
                    out,
                    "    {:<10} {:>22} {:>22} {:>12} {:>9}  {}",
                    p.label,
                    value(p.reference, p.reference_sigma),
                    value(p.other, p.other_sigma),
                    format!("{:+.6}", p.delta),
                    p.delta_over_reference_sigma
                        .map_or_else(|| "-".to_string(), |r| format!("{r:+.2}")),
                    scalar_test(&p.test)
                );
            }
        }
        let projection = match &camera.projection {
            ProjectionComparison::Compared {
                rotation_deg,
                fit_nodes,
                rms_fit_px,
                max_fit_px,
                max_valid_px,
                ..
            } => format!(
                "{rms_fit_px:.4} px RMS over {fit_nodes} covered nodes (max {max_fit_px:.4}; \
                 {max_valid_px:.3} over the image), rotation {rotation_deg:.4}°"
            ),
            ProjectionComparison::Unavailable { reason } => format!("not computed ({reason})"),
            _ => "-".to_string(),
        };
        let _ = writeln!(out, "    projection difference: {projection}");
    }
    for pose in d.rig.iter().chain(&d.handeye) {
        let _ = writeln!(out, "  {}", pose_line(pose));
    }
    for plane in &d.laser_planes {
        let _ = writeln!(
            out,
            "  laser plane {}: normal {:.4}°, distance {:+.6} m{}, {}",
            plane.camera,
            plane.normal_angle_deg,
            plane.distance_delta_m,
            plane
                .distance_over_reference_sigma
                .map_or_else(String::new, |r| format!(" ({r:+.2} σ_ref)")),
            scalar_test(&plane.distance_test)
        );
    }
}

fn value(v: f64, sigma: Option<f64>) -> String {
    match sigma {
        Some(s) => format!("{v:.6} ±{s:.2e}"),
        None => format!("{v:.6}"),
    }
}

fn scalar_test(test: &DifferenceTest) -> String {
    match test {
        DifferenceTest::Independent { z, p_value, .. } => {
            format!("independent z {z:+.2} (p {p_value:.3})")
        }
        DifferenceTest::Nested { z, p_value, .. } => format!("nested z {z:+.2} (p {p_value:.3})"),
        DifferenceTest::NotAvailable { reason } => format!("none ({})", tag(reason)),
        _ => "-".to_string(),
    }
}

fn pose_line(p: &PoseDifference) -> String {
    let materiality = p
        .delta_over_reference_sigma
        .map_or_else(String::new, |r| format!(", {r:.2} σ_ref"));
    let test = match &p.test {
        PoseTest::Independent { chi2, dof, p_value } => {
            format!("independent χ² {chi2:.2} ({dof} dof, p {p_value:.3})")
        }
        PoseTest::Nested { chi2, dof, p_value } => {
            format!("nested χ² {chi2:.2} ({dof} dof, p {p_value:.3})")
        }
        PoseTest::NotAvailable { reason } => format!("none ({})", tag(reason)),
        _ => "-".to_string(),
    };
    format!(
        "{}: rotation {:.4}°, translation {:.3} mm{materiality}, {test}",
        p.label,
        p.rotation_deg,
        p.translation_norm_m * 1e3
    )
}

/// A value's serde tag (`independent`, `shared_data`).
fn tag(value: &impl serde::Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::Object(map)) => map
            .get("kind")
            .and_then(|k| k.as_str())
            .map_or_else(|| name(value), str::to_string),
        _ => name(value),
    }
}
