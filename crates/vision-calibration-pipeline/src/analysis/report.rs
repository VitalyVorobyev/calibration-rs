//! A calibration result as one self-contained HTML page: [`html_report`].
//!
//! The page holds the quality verdict and every check, and per camera its
//! model, each parameter with its σ, the reprojection-error statistics and
//! figures drawn as inline SVG: the error histogram, the systematic error and
//! the local RMS error over the image (a [`residual_map`]), the mean error per
//! view and the projection-σ map. It loads
//! nothing (no script, font or stylesheet from elsewhere), so it opens
//! offline and can be attached or archived as it is, and the same result
//! always gives the same bytes.

use std::fmt::Write as _;

use serde::Serialize;
use vision_calibration_core::{CameraParams, ImageGrid, TargetFeatureResidual, Uncertainty};
use vision_calibration_optim::{ProjectionUncertainty, camera_parameters};

use super::{
    BandwidthSelection, CheckStatus, LevelStats, QualityReport, QualityThresholds,
    ResidualMapOptions, Verdict, assess, residual_map,
};
use crate::Error;
use crate::common::CalibratedCameras;

/// Options of [`html_report`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ReportOptions {
    /// The page's heading; empty for "Calibration report".
    pub title: String,
    /// Each camera's name, in camera order (a manifest's camera ids). A
    /// camera without one is shown by its index only.
    pub camera_names: Vec<String>,
    /// The limits of the quality checks, as [`assess`] takes them.
    pub thresholds: QualityThresholds,
}

impl ReportOptions {
    /// Construct default HTML report options.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A calibration result as one self-contained HTML page.
///
/// The page shows the [`assess`] verdict with every check and, per camera,
/// the model, each parameter with its σ, the reprojection-error statistics,
/// and the figures: the error histogram, the systematic error
/// (`observed − projected`) and the local RMS error over the image from one
/// [`residual_map`] (32 columns, default options), the mean error per view
/// and the projection-σ map ([`projection_uncertainty`](CalibratedCameras::projection_uncertainty)).
/// `image_sizes` is what [`assess`] takes: each camera's `[width, height]`
/// in pixels, or empty. Without a camera's size the residual maps span the
/// features' projections from the image origin, and the projection-σ map is
/// left out, as it is when the result has no covariance.
///
/// The page loads no script, font or stylesheet, and its bytes depend only on
/// the inputs and the crate version.
///
/// # Errors
///
/// What [`assess`] returns: [`Error::InvalidInput`] for an out-of-range
/// threshold or an `image_sizes` that is neither empty nor one size per
/// camera.
///
/// # Example
///
/// ```no_run
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let export: vision_calibration_pipeline::planar_intrinsics::PlanarIntrinsicsExport = todo!();
/// use vision_calibration_pipeline::analysis::{ReportOptions, html_report};
///
/// let mut options = ReportOptions::default();
/// options.title = "Left camera".to_string();
/// let page = html_report(&export, &[[1280, 960]], &options)?;
/// std::fs::write("report.html", page)?;
/// # Ok(())
/// # }
/// ```
pub fn html_report(
    export: &dyn CalibratedCameras,
    image_sizes: &[[u32; 2]],
    options: &ReportOptions,
) -> Result<String, Error> {
    let quality = assess(export, image_sizes, &options.thresholds)?;
    let cameras = export.camera_params();
    let residuals = &export.diagnostics().per_feature_residuals.target;
    let title = if options.title.is_empty() {
        "Calibration report"
    } else {
        options.title.as_str()
    };

    let mut page = String::with_capacity(64 * 1024);
    let _ = write!(
        page,
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>{}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n<main>\n",
        escape(title)
    );
    write_summary(&mut page, title, &quality, cameras.len(), residuals);
    write_checks(&mut page, &quality, &options.camera_names);
    for (index, camera) in cameras.iter().enumerate() {
        let camera_residuals: Vec<TargetFeatureResidual> = residuals
            .iter()
            .filter(|r| r.camera == index)
            .cloned()
            .collect();
        let section = CameraSection {
            index,
            name: options.camera_names.get(index).map(String::as_str),
            camera,
            prefix: (cameras.len() > 1).then(|| format!("cam{index}.")),
            uncertainty: export.parameter_uncertainty(),
            residuals: &camera_residuals,
            image_size: image_sizes.get(index).copied(),
            projection: image_sizes
                .get(index)
                .and_then(|&size| export.projection_uncertainty(index, size).ok()),
        };
        section.write(&mut page);
    }
    let _ = write!(
        page,
        "<footer>vision-calibration {}</footer>\n</main>\n</body>\n</html>\n",
        env!("CARGO_PKG_VERSION")
    );
    Ok(page)
}

fn write_summary(
    page: &mut String,
    title: &str,
    quality: &QualityReport,
    cameras: usize,
    residuals: &[TargetFeatureResidual],
) {
    let verdict = name(&quality.verdict);
    let stats = LevelStats::from_residuals(residuals);
    let _ = write!(
        page,
        "<header>\n<h1>{}</h1>\n<p class=\"verdict {verdict}\">{}</p>\n<p class=\"meta\">{cameras} \
         camera{}",
        escape(title),
        verdict_label(quality.verdict),
        if cameras == 1 { "" } else { "s" },
    );
    if stats.count > 0 {
        let _ = write!(
            page,
            " · {} features · reprojection error mean {} px, RMS {} px, 95 % below {} px, \
             max {} px",
            stats.count,
            sig(stats.mean, 3),
            sig(stats.rms, 3),
            sig(stats.p95, 3),
            sig(stats.max, 3)
        );
    }
    page.push_str("</p>\n</header>\n");
}

fn write_checks(page: &mut String, quality: &QualityReport, names: &[String]) {
    page.push_str(
        "<section>\n<h2>Quality checks</h2>\n<table>\n<thead><tr><th>Result</th><th>Check</th>\
         <th>Camera</th><th>Finding</th></tr></thead>\n<tbody>\n",
    );
    for check in &quality.checks {
        let status = name(&check.status);
        let camera = check
            .camera
            .map(|i| camera_label(i, names.get(i).map(String::as_str)))
            .unwrap_or_default();
        let _ = writeln!(
            page,
            "<tr><td><span class=\"status {status}\">{}</span></td><td>{}</td>\
             <td class=\"camera\">{}</td><td>{}</td></tr>",
            status_label(check.status),
            escape(&name(&check.kind).replace('_', " ")),
            escape(&camera),
            escape(&check.message)
        );
    }
    page.push_str("</tbody>\n</table>\n</section>\n");
}

struct CameraSection<'a> {
    index: usize,
    name: Option<&'a str>,
    camera: &'a CameraParams,
    /// The covariance labels' camera prefix (`cam1.`), for a result with
    /// several cameras.
    prefix: Option<String>,
    uncertainty: &'a Uncertainty,
    residuals: &'a [TargetFeatureResidual],
    image_size: Option<[u32; 2]>,
    projection: Option<ProjectionUncertainty>,
}

impl CameraSection<'_> {
    fn write(&self, page: &mut String) {
        let _ = write!(
            page,
            "<section class=\"camera\">\n<h2>{}</h2>\n<p class=\"meta\">{}",
            escape(&camera_label(self.index, self.name)),
            escape(&model_label(self.camera))
        );
        if let Some([width, height]) = self.image_size {
            let _ = write!(page, " · {width} × {height} px");
        }
        page.push_str("</p>\n<div class=\"panels\">\n");
        self.write_parameters(page);
        self.write_figures(page);
        page.push_str("</div>\n</section>\n");
    }

    fn write_parameters(&self, page: &mut String) {
        page.push_str("<div class=\"panel\">\n<h3>Parameters</h3>\n");
        let parameters = match camera_parameters(self.camera) {
            Ok(parameters) => parameters,
            Err(err) => {
                let _ = writeln!(page, "<p>{}</p>\n</div>", escape(&err.to_string()));
                return;
            }
        };
        page.push_str(
            "<table class=\"numbers\">\n<thead><tr><th>Parameter</th><th>Value</th><th>σ</th></tr>\
             </thead>\n<tbody>\n",
        );
        for (label, value) in parameters {
            let sigma = match self.uncertainty {
                Uncertainty::Available(covariance) => {
                    let key = format!("{}{label}", self.prefix.as_deref().unwrap_or(""));
                    covariance.std_dev(&key)
                }
                Uncertainty::Unavailable { .. } => None,
            };
            let (value, sigma) = match sigma {
                Some(s) => (with_sigma(value, s), sig(s, 2)),
                None if matches!(self.uncertainty, Uncertainty::Available(_)) => {
                    (sig(value, 6), "fixed".to_string())
                }
                None => (sig(value, 6), "—".to_string()),
            };
            let _ = writeln!(
                page,
                "<tr><td>{}</td><td>{value}</td><td>{sigma}</td></tr>",
                escape(label)
            );
        }
        page.push_str("</tbody>\n</table>\n");
        if let Uncertainty::Unavailable { reason } = self.uncertainty {
            let _ = writeln!(
                page,
                "<p class=\"note\">No parameter covariance: {}.</p>",
                escape(&name(reason).replace('_', " "))
            );
        }
        let stats = LevelStats::from_residuals(self.residuals);
        if stats.count > 0 {
            let _ = writeln!(
                page,
                "<h3>Reprojection error</h3>\n<table class=\"numbers\">\n<tbody>\n\
                 <tr><td>features</td><td>{}</td></tr>\n<tr><td>mean</td><td>{} px</td></tr>\n\
                 <tr><td>RMS</td><td>{} px</td></tr>\n<tr><td>median</td><td>{} px</td></tr>\n\
                 <tr><td>95th percentile</td><td>{} px</td></tr>\n\
                 <tr><td>max</td><td>{} px</td></tr>\n</tbody>\n</table>",
                stats.count,
                sig(stats.mean, 3),
                sig(stats.rms, 3),
                sig(stats.median, 3),
                sig(stats.p95, 3),
                sig(stats.max, 3)
            );
        }
        page.push_str("</div>\n");
    }

    fn write_figures(&self, page: &mut String) {
        let errors: Vec<f64> = self
            .residuals
            .iter()
            .filter_map(|r| r.error_px)
            .filter(|e| e.is_finite())
            .collect();
        if !errors.is_empty() {
            figure(
                page,
                &histogram(&errors),
                "Reprojection error per feature, pixels. The dashed line is the mean.",
            );
            for (svg, caption) in residual_figures(self.residuals, self.index, self.image_size) {
                figure(page, &svg, &caption);
            }
            figure(
                page,
                &per_view(self.residuals),
                "Mean reprojection error of each view, pixels. A view far above the dashed \
                 mean is worth a look.",
            );
        }
        if let Some(projection) = &self.projection {
            let corner = projection
                .corner
                .map_or_else(String::new, |c| format!(", corners {} px", sig(c, 3)));
            let caption = format!(
                "Projection σ: how far the calibration's uncertainty moves where a ray lands, \
                 pixels. Centre {} px{corner}, largest {} px.",
                sig(projection.center, 3),
                sig(projection.max, 3)
            );
            figure(page, &projection_map(projection), &caption);
        }
    }
}

fn figure(page: &mut String, svg: &str, caption: &str) {
    let _ = writeln!(
        page,
        "<figure class=\"panel\">\n{svg}\n<figcaption>{}</figcaption>\n</figure>",
        escape(caption)
    );
}

// ─── Figures ────────────────────────────────────────────────────────────────

/// Plot area of every figure, in SVG user units.
const W: f64 = 360.0;
const H: f64 = 200.0;
/// Margins: left (axis labels), bottom (axis labels), top and right.
const ML: f64 = 44.0;
const MB: f64 = 26.0;
const MT: f64 = 10.0;
const MR: f64 = 12.0;

fn svg_open(out: &mut String, height: f64, label: &str) {
    let _ = write!(
        out,
        "<svg viewBox=\"0 0 {W} {height}\" role=\"img\" aria-label=\"{}\">",
        escape(label)
    );
}

/// The histogram of `errors`: 24 bins from zero to the 99th percentile, the
/// last bin holding everything above it.
fn histogram(errors: &[f64]) -> String {
    const BINS: usize = 24;
    let mut sorted = errors.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p99 = sorted[((sorted.len() - 1) as f64 * 0.99).round() as usize];
    let upper = if p99 > 0.0 { p99 } else { 1.0 };
    let mut counts = [0usize; BINS];
    for &e in errors {
        let bin = ((e / upper) * BINS as f64) as usize;
        counts[bin.min(BINS - 1)] += 1;
    }
    let peak = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;

    let (pw, ph) = (W - ML - MR, H - MT - MB);
    let mut svg = String::new();
    svg_open(&mut svg, H, "Histogram of the reprojection errors");
    let bar = pw / BINS as f64;
    for (i, &count) in counts.iter().enumerate() {
        let h = ph * count as f64 / peak;
        let _ = write!(
            svg,
            "<rect class=\"bar\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{h:.2}\"/>",
            ML + i as f64 * bar + 0.5,
            MT + ph - h,
            bar - 1.0
        );
    }
    let mx = ML + pw * (mean / upper).min(1.0);
    let _ = write!(
        svg,
        "<line class=\"mean\" x1=\"{mx:.2}\" y1=\"{MT}\" x2=\"{mx:.2}\" y2=\"{:.2}\"/>",
        MT + ph
    );
    axes(&mut svg, pw, ph);
    x_ticks(&mut svg, pw, upper, "px");
    let _ = write!(
        svg,
        "<text class=\"tick\" x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\">{}</text>\
         <text class=\"tick\" x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\">0</text></svg>",
        ML - 4.0,
        MT + 8.0,
        peak as usize,
        ML - 4.0,
        MT + ph
    );
    svg
}

/// The columns of the report's residual maps.
const RESIDUAL_MAP_COLUMNS: u32 = 32;

/// `χ` above which a node of the systematic-error map is significant
/// (pointwise `p < 0.01`).
const SIGNIFICANT_CHI: f64 = 3.03;

/// The two figures of one [`residual_map`] of `camera` (default options,
/// [`RESIDUAL_MAP_COLUMNS`] columns): the length of the systematic error,
/// greyed where it is not significant, and the local RMS error. Without the
/// image's size the grid spans the projections from the image origin. None
/// when no record has a projection.
fn residual_figures(
    residuals: &[TargetFeatureResidual],
    camera: usize,
    image_size: Option<[u32; 2]>,
) -> Vec<(String, String)> {
    let size = match image_size {
        Some([w, h]) if w > 0 && h > 0 => [w, h],
        _ => {
            let far = |i: usize| {
                residuals
                    .iter()
                    .filter_map(|r| r.projected_px)
                    .map(|p| p[i])
                    .filter(|v| v.is_finite())
                    .fold(0.0f64, f64::max)
            };
            // Saturating casts: a projection far off the image only widens the grid.
            [(far(0) + 1.5) as u32, (far(1) + 1.5) as u32]
        }
    };
    let Ok(grid) = ImageGrid::with_columns(size, RESIDUAL_MAP_COLUMNS) else {
        return Vec::new();
    };
    let Ok(map) = residual_map(residuals, camera, &grid, &ResidualMapOptions::default()) else {
        return Vec::new();
    };
    if map.used == 0 {
        return Vec::new();
    }
    let aspect = f64::from(size[1]) / f64::from(size[0]);
    let mut largest = None::<f64>;
    let mean: Vec<Cell> = (0..grid.len())
        .map(|i| {
            let length = map.mean_dx[i].hypot(map.mean_dy[i]);
            if !length.is_finite() {
                Cell::Empty
            } else if map.significance[i] >= SIGNIFICANT_CHI {
                largest = Some(largest.map_or(length, |l: f64| l.max(length)));
                Cell::Value(length)
            } else {
                Cell::Grey(format!("{} px, not significant", sig(length, 2)))
            }
        })
        .collect();
    let rms: Vec<Cell> = map
        .rms
        .iter()
        .map(|&v| {
            if v.is_finite() {
                Cell::Value(v)
            } else {
                Cell::Empty
            }
        })
        .collect();
    let extent = if image_size.is_some() {
        ""
    } else {
        " (image size unknown: the map spans the features' projections)"
    };
    let selection = match map.bandwidth_selection {
        BandwidthSelection::CrossValidated { .. } => "cross-validated over views",
        BandwidthSelection::FewViews => "fewer than four views to cross-validate",
        _ => "given",
    };
    let finding = match largest {
        Some(l) => format!("The largest significant one is {} px.", sig(l, 2)),
        None => "No node is significant.".to_string(),
    };
    let mean_caption = format!(
        "Systematic reprojection error: the local mean of observed − projected over the {} \
         views, its length in pixels, kernel σ {} px ({selection}){extent}. Grey: not \
         significant (pointwise p ≥ 0.01, calibrated for the number of views); empty: too \
         few features. {finding} A pattern toward the \
         edges points at the camera model, but it averages this capture's depths and poses.",
        map.views,
        sig(map.bandwidth_px, 2),
    );
    let rms_caption = format!(
        "Local RMS reprojection error, pixels, same kernel{extent}. Empty: too few features."
    );
    vec![
        (
            heat_map(
                &mean,
                grid.columns(),
                grid.rows(),
                aspect,
                "Systematic reprojection error across the image",
            ),
            mean_caption,
        ),
        (
            heat_map(
                &rms,
                grid.columns(),
                grid.rows(),
                aspect,
                "Local RMS reprojection error across the image",
            ),
            rms_caption,
        ),
    ]
}

/// The projection-σ grid on the shared colour scale; a node no ray reaches
/// is greyed out.
fn projection_map(projection: &ProjectionUncertainty) -> String {
    let [w, h] = projection.image_size;
    let cells: Vec<Cell> = projection
        .sigma
        .iter()
        .map(|v| match v {
            Some(v) => Cell::Value(*v),
            None => Cell::Grey("no ray reaches here".to_string()),
        })
        .collect();
    heat_map(
        &cells,
        projection.columns,
        projection.rows,
        f64::from(h) / f64::from(w).max(1.0),
        "Projection uncertainty across the image",
    )
}

/// One cell of a [`heat_map`].
enum Cell {
    /// A value on the colour scale.
    Value(f64),
    /// Greyed out, with this tooltip.
    Grey(String),
    /// Left empty.
    Empty,
}

/// `cells` (row-major, `columns` × `rows`) in a plot of height `aspect` ×
/// width, with a colour bar from zero to the largest value.
fn heat_map(cells: &[Cell], columns: usize, rows: usize, aspect: f64, label: &str) -> String {
    // No axis labels on the left: the map takes the figure's width.
    let left = MR;
    let pw = W - left - MR;
    let ph = (pw * aspect).clamp(60.0, 260.0);
    let height = MT + ph + MB + 16.0;
    let top = cells
        .iter()
        .filter_map(|c| match c {
            Cell::Value(v) => Some(*v),
            _ => None,
        })
        .fold(0.0f64, f64::max);
    let scale = if top > 0.0 { top } else { 1.0 };
    let (cw, ch) = (pw / columns as f64, ph / rows as f64);
    let mut svg = String::new();
    svg_open(&mut svg, height, label);
    let _ = write!(
        svg,
        "<rect class=\"frame\" x=\"{left}\" y=\"{MT}\" width=\"{pw:.2}\" height=\"{ph:.2}\"/>"
    );
    for (i, cell) in cells.iter().enumerate() {
        let (c, r) = (i % columns, i / columns);
        let (x, y) = (left + c as f64 * cw, MT + r as f64 * ch);
        match cell {
            Cell::Value(v) => {
                let _ = write!(
                    svg,
                    "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{:.2}\" height=\"{:.2}\" \
                     fill=\"{}\"><title>{} px</title></rect>",
                    cw + 0.3,
                    ch + 0.3,
                    colour(v / scale),
                    sig(*v, 3)
                );
            }
            Cell::Grey(title) => {
                let _ = write!(
                    svg,
                    "<rect class=\"missing\" x=\"{x:.2}\" y=\"{y:.2}\" width=\"{:.2}\" \
                     height=\"{:.2}\"><title>{}</title></rect>",
                    cw + 0.3,
                    ch + 0.3,
                    escape(title)
                );
            }
            Cell::Empty => {}
        }
    }
    // The colour bar under the map.
    let bar_y = MT + ph + 8.0;
    for k in 0..48 {
        let t = (k as f64 + 0.5) / 48.0;
        let _ = write!(
            svg,
            "<rect x=\"{:.2}\" y=\"{bar_y:.2}\" width=\"{:.2}\" height=\"8\" fill=\"{}\"/>",
            left + pw * k as f64 / 48.0,
            pw / 48.0 + 0.3,
            colour(t)
        );
    }
    let _ = write!(
        svg,
        "<text class=\"tick\" x=\"{left}\" y=\"{:.2}\">0</text><text class=\"tick\" x=\"{:.2}\" \
         y=\"{:.2}\" text-anchor=\"end\">{} px</text></svg>",
        bar_y + 20.0,
        left + pw,
        bar_y + 20.0,
        sig(top, 3)
    );
    svg
}

/// The mean reprojection error of each view, as bars in view order.
fn per_view(residuals: &[TargetFeatureResidual]) -> String {
    let views = residuals
        .iter()
        .filter_map(|r| r.view.checked_add(1))
        .max()
        .unwrap_or(0);
    let mut sums = vec![(0.0f64, 0usize); views];
    for r in residuals {
        if let Some(e) = r.error_px.filter(|e| e.is_finite())
            && let Some(slot) = sums.get_mut(r.view)
        {
            slot.0 += e;
            slot.1 += 1;
        }
    }
    let means: Vec<Option<f64>> = sums
        .iter()
        .map(|&(sum, n)| (n > 0).then(|| sum / n as f64))
        .collect();
    let (total, count) = sums
        .iter()
        .fold((0.0, 0usize), |(s, n), &(sum, k)| (s + sum, n + k));
    let mean = if count > 0 { total / count as f64 } else { 0.0 };
    let top = means.iter().flatten().copied().fold(0.0f64, f64::max);
    let scale = if top.is_finite() && top > 0.0 {
        top
    } else {
        1.0
    };

    let (pw, ph) = (W - ML - MR, H - MT - MB);
    let mut svg = String::new();
    svg_open(&mut svg, H, "Mean reprojection error per view");
    let bar = pw / views.max(1) as f64;
    for (view, value) in means.iter().enumerate() {
        if let Some(v) = value {
            let h = ph * v / scale;
            let _ = write!(
                svg,
                "<rect class=\"bar\" x=\"{:.2}\" y=\"{:.2}\" width=\"{:.2}\" height=\"{h:.2}\">\
                 <title>view {view}: {} px</title></rect>",
                ML + view as f64 * bar + bar * 0.1,
                MT + ph - h,
                bar * 0.8,
                sig(*v, 3)
            );
        }
    }
    let my = MT + ph - ph * mean / scale;
    let _ = write!(
        svg,
        "<line class=\"mean\" x1=\"{ML}\" y1=\"{my:.2}\" x2=\"{:.2}\" y2=\"{my:.2}\"/>",
        ML + pw
    );
    axes(&mut svg, pw, ph);
    let _ = write!(
        svg,
        "<text class=\"tick\" x=\"{ML}\" y=\"{:.2}\">view 0</text><text class=\"tick\" \
         x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\">view {}</text><text class=\"tick\" \
         x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\">{} px</text></svg>",
        MT + ph + 16.0,
        ML + pw,
        MT + ph + 16.0,
        views.saturating_sub(1),
        ML - 4.0,
        MT + 8.0,
        sig(top, 2)
    );
    svg
}

fn axes(svg: &mut String, pw: f64, ph: f64) {
    let _ = write!(
        svg,
        "<path class=\"axis\" d=\"M{ML} {MT}V{:.2}H{:.2}\"/>",
        MT + ph,
        ML + pw
    );
}

/// Ticks at 0, half and all of `upper` under the x axis.
fn x_ticks(svg: &mut String, pw: f64, upper: f64, unit: &str) {
    for (t, anchor) in [(0.0, "start"), (0.5, "middle"), (1.0, "end")] {
        let _ = write!(
            svg,
            "<text class=\"tick\" x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"{anchor}\">{} {unit}</text>",
            ML + pw * t,
            H - MB + 16.0,
            sig(upper * t, 2)
        );
    }
}

/// A sequential colour for `t` in `[0, 1]`: dark blue through teal and
/// green to yellow (viridis), readable in both themes and for the common
/// colour-vision deficiencies.
fn colour(t: f64) -> String {
    const STOPS: [[f64; 3]; 9] = [
        [68.0, 1.0, 84.0],
        [71.0, 45.0, 123.0],
        [59.0, 82.0, 139.0],
        [44.0, 114.0, 142.0],
        [33.0, 145.0, 140.0],
        [39.0, 173.0, 129.0],
        [92.0, 200.0, 99.0],
        [170.0, 220.0, 50.0],
        [253.0, 231.0, 37.0],
    ];
    let x = t.clamp(0.0, 1.0) * (STOPS.len() - 1) as f64;
    let i = (x.floor() as usize).min(STOPS.len() - 2);
    let f = x - i as f64;
    let c = |k: usize| (STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * f).round() as u8;
    format!("#{:02x}{:02x}{:02x}", c(0), c(1), c(2))
}

// ─── Text ───────────────────────────────────────────────────────────────────

/// `value` to `digits` significant figures, without an exponent.
fn sig(value: f64, digits: i32) -> String {
    if !value.is_finite() {
        return "—".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).clamp(0, 12) as usize;
    format!("{value:.decimals$}")
}

/// `value` to one decimal past the second significant figure of its
/// standard deviation `sigma`: the digits the data supports, and one more.
fn with_sigma(value: f64, sigma: f64) -> String {
    if !value.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
        return sig(value, 6);
    }
    let decimals = (2 - sigma.log10().floor() as i32).clamp(0, 12) as usize;
    format!("{value:.decimals$}")
}

/// The name a value serializes to (`planar_intrinsics`, `warn`); for an
/// internally tagged enum, its `kind`.
fn name(value: &impl Serialize) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(name)) => name,
        Ok(serde_json::Value::Object(map)) => match map.get("kind") {
            Some(serde_json::Value::String(kind)) => kind.clone(),
            _ => serde_json::Value::Object(map).to_string(),
        },
        Ok(other) => other.to_string(),
        Err(_) => String::new(),
    }
}

fn verdict_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Pass => "Pass",
        Verdict::Warn => "Warn",
        Verdict::Fail => "Fail",
    }
}

fn status_label(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "pass",
        CheckStatus::Warn => "warn",
        CheckStatus::Fail => "fail",
        CheckStatus::NotApplicable => "n/a",
    }
}

fn camera_label(index: usize, name: Option<&str>) -> String {
    match name {
        Some(name) if !name.is_empty() => format!("Camera {index} ({name})"),
        _ => format!("Camera {index}"),
    }
}

/// "pinhole projection · brown_conrady5 distortion · identity sensor".
fn model_label(camera: &CameraParams) -> String {
    match camera.spec() {
        Some(spec) => format!(
            "{} projection · {} distortion · {} sensor",
            name(&spec.projection),
            name(&spec.distortion),
            name(&spec.sensor)
        ),
        None => String::new(),
    }
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

const STYLE: &str = "
:root{--bg:#ffffff;--fg:#1d2330;--muted:#5d6577;--line:#d9dde5;--panel:#f6f7f9;
--bar:#3b528b;--mean:#c2410c;--pass:#15803d;--warn:#b45309;--fail:#b91c1c;--na:#6b7280}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--fg:#e6e8ec;--muted:#9aa2b1;
--line:#2c323c;--panel:#1b1f26;--bar:#7aa2e3;--mean:#fb923c;--pass:#4ade80;--warn:#fbbf24;
--fail:#f87171;--na:#9ca3af}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.6rem;margin:0 0 8px}h2{font-size:1.25rem;margin:32px 0 4px}
h3{font-size:1rem;margin:0 0 8px}
.meta{color:var(--muted);margin:4px 0 12px}
.verdict{display:inline-block;margin:0;padding:2px 12px;border-radius:999px;font-weight:600;
color:var(--bg)}
.verdict.pass{background:var(--pass)}.verdict.warn{background:var(--warn)}
.verdict.fail{background:var(--fail)}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
table.numbers td:not(:first-child),table.numbers th:not(:first-child){
font-variant-numeric:tabular-nums;text-align:right}
td.camera{white-space:nowrap}
.status{font-weight:600}.status.pass{color:var(--pass)}.status.warn{color:var(--warn)}
.status.fail{color:var(--fail)}.status.not_applicable{color:var(--na)}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;
margin:0;min-width:0}
figure svg{width:100%;height:auto;display:block}
figcaption{color:var(--muted);font-size:.85rem;margin-top:6px}
.note{color:var(--muted)}
footer{color:var(--muted);font-size:.85rem;margin-top:40px}
svg .bar{fill:var(--bar)}svg .mean{stroke:var(--mean);stroke-width:1.5;stroke-dasharray:4 3}
svg .axis{fill:none;stroke:var(--muted);stroke-width:1}
svg .frame{fill:none;stroke:var(--line)}svg .missing{fill:var(--line)}
svg .tick{fill:var(--muted);font-size:10px}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significant_figures() {
        assert_eq!(sig(534.10791, 6), "534.108");
        assert_eq!(sig(0.000_934_2, 2), "0.00093");
        assert_eq!(sig(-0.291_93, 3), "-0.292");
        assert_eq!(sig(12_345.6, 2), "12346");
        assert_eq!(sig(0.0, 3), "0");
        assert_eq!(sig(f64::NAN, 3), "—");
    }

    #[test]
    fn a_value_keeps_the_digits_its_sigma_supports() {
        assert_eq!(with_sigma(560.913_42, 1.07), "560.91");
        assert_eq!(with_sigma(0.000_650_049, 0.0073), "0.00065");
        assert_eq!(with_sigma(-0.291_932, 0.0049), "-0.29193");
        assert_eq!(with_sigma(1234.5, 0.0), "1234.50");
    }

    #[test]
    fn escapes_markup() {
        assert_eq!(escape("a<b & \"c\"'"), "a&lt;b &amp; &quot;c&quot;&#39;");
    }

    #[test]
    fn colour_scale_ends() {
        assert_eq!(colour(0.0), "#440154");
        assert_eq!(colour(1.0), "#fde725");
        assert_eq!(colour(2.0), "#fde725");
    }

    /// Records whose observations sit 0.5 px right of their projections in
    /// the left half of the image, with a little scatter.
    fn shifted_records(views: usize) -> Vec<TargetFeatureResidual> {
        let mut out = Vec::new();
        for view in 0..views {
            for i in 0..600usize {
                let mut r = TargetFeatureResidual::default();
                r.view = view;
                let p = [
                    ((i * 37 + view * 11) % 1280) as f64,
                    ((i * 53 + view * 7) % 960) as f64,
                ];
                let wobble = 0.05 * (((i * 7919 + view * 104_729) % 200) as f64 / 100.0 - 1.0);
                let dx = if p[0] < 640.0 { 0.5 } else { 0.0 };
                r.projected_px = Some(p);
                r.observed_px = [p[0] + dx + wobble, p[1] - wobble];
                r.error_px = Some((dx + wobble).hypot(wobble));
                out.push(r);
            }
        }
        out
    }

    #[test]
    fn residual_figures_show_the_systematic_and_rms_maps() {
        let records = shifted_records(8);
        let figures = residual_figures(&records, 0, Some([1280, 960]));
        assert_eq!(figures.len(), 2);
        let (mean_svg, mean_caption) = &figures[0];
        assert!(mean_svg.starts_with("<svg") && mean_svg.contains("fill=\"#"));
        assert!(
            mean_caption.contains("observed − projected"),
            "{mean_caption}"
        );
        assert!(mean_caption.contains("8 views"), "{mean_caption}");
        assert!(mean_caption.contains("cross-validated"), "{mean_caption}");
        assert!(
            mean_caption.contains("largest significant one is 0.5"),
            "{mean_caption}"
        );
        assert!(figures[1].1.contains("RMS"));
        // Deterministic bytes.
        assert_eq!(residual_figures(&records, 0, Some([1280, 960])), figures);
        // Without the image size, the map spans the projections.
        let unsized_figures = residual_figures(&records, 0, None);
        assert_eq!(unsized_figures.len(), 2);
        assert!(unsized_figures[0].1.contains("image size unknown"));
        // Another camera's records, or none projected: no figure.
        assert!(residual_figures(&records, 1, Some([1280, 960])).is_empty());
        let mut unprojected = TargetFeatureResidual::default();
        unprojected.error_px = Some(0.1);
        assert!(residual_figures(&[unprojected], 0, Some([1280, 960])).is_empty());
    }
}
