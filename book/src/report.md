# The Calibration Report

`html_report` turns a calibration result into one HTML page to read, attach
to a ticket or archive next to the export. The page loads nothing from
elsewhere, no script, font or stylesheet, so it opens offline in any
browser, and the same result always gives the same page. It follows the
reader's light or dark theme.

## What the page holds

- **The verdict** of [Judging a Result](quality.md), pass, warn or fail, with
  the overall reprojection error: mean, RMS, 95th percentile and maximum.
- **Every quality check**, with its finding, so a warning says what to
  change.
- **Per camera**: the camera model, each parameter with its σ (`fixed` for
  one the solve held), the reprojection-error statistics, and the figures:
  - the **histogram** of the per-feature errors, with the mean marked;
  - the **systematic error over the image**: the length of the local mean
    of `observed − projected` across the views, greyed where it is not
    significant, from one [residual map](diagnostic_maps.md) (32 columns,
    cross-validated kernel). A pattern, such as a significant error toward
    the corners, points at the camera model; empty cells show where the
    capture never put the board;
  - the **local RMS error** over the image, from the same map;
  - the **mean error of each view**, where a view far above the others is
    the one to inspect or drop;
  - the **projection-σ map** ([Projection Uncertainty](uncertainty_projection.md)):
    how far the calibration's own uncertainty moves where a ray lands, over
    the image.

A parameter's value is rounded to one digit past what its σ supports: with
σ(k1) = 0.0073, k1 shows as `0.00065`, not `0.000650049`. A parameter
without a σ keeps six significant figures.

The residual-map and per-view figures read the export's per-feature
residuals, which every workflow records; without the camera's image size the
residual maps span the features' projections. The projection-σ map needs the
image size and the parameter covariance; without either it is left out.

## Making one

From the command line, `--report` adds `report.html` to what `vcal run`
writes ([The Command Line](cli.md)):

```bash
vcal run data/my_camera/dataset.toml --report
```

The desktop app writes `report.html` into every saved run's folder
([The Desktop App](app.md)).

In Rust, `html_report` takes any export and the image sizes that
[`assess`](quality.md) takes; `dataset_runner::calibrate_dataset` returns
them with the export:

```rust,no_run
use std::path::Path;
use vision_calibration::analysis::{ReportOptions, html_report};
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::control::RunControl;
use vision_calibration::dataset_runner::{Detection, calibrate_dataset};

fn main() -> anyhow::Result<()> {
    let dir = Path::new("data/my_camera");
    let spec = DatasetSpec::from_path(dir.join("dataset.toml"))?;
    let run = calibrate_dataset(&spec, dir, None, Detection::Uncached, None, &RunControl::default())?;

    let mut options = ReportOptions::default();
    options.title = "Bench camera".to_string();
    options.camera_names = spec.cameras.iter().map(|c| c.id.clone()).collect();
    let page = html_report(&run.export, &run.image_sizes, &options)?;
    std::fs::write(dir.join("report.html"), page)?;
    Ok(())
}
```

In Python, `vc.html_report` takes a result or an export loaded from JSON;
`calibrate_dataset` returns the sizes as `image_sizes`, and
`dataset_runner.camera_image_sizes` reads them from a manifest without running it:

```python
from pathlib import Path
import vision_calibration as vc

manifest = Path("data/my_camera/dataset.toml")
run = vc.calibrate_dataset(manifest)
page = vc.html_report(
    run.export,
    image_sizes=run.image_sizes,
    title="Bench camera",
)
(manifest.parent / "report.html").write_text(page, encoding="utf-8")
```

`ReportOptions::thresholds` (Python: `thresholds=`) sets the quality checks'
limits, as for `assess`.
