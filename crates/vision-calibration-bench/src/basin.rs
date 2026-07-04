//! Convergence-basin study for the seeded Scheimpflug initialization
//! (Q6-BASIN-STUDY; the quantitative evidence behind ADR 0022/0023).
//!
//! `calib-bench basin` answers: *how much spec error can the ADR
//! 0022/0023 seeded route absorb before it stops converging?* For every
//! registered `scheimpflug_intrinsics` entry it:
//!
//! 1. derives the device-spec seed and detects the camera's views once
//!    (`run::detect_scheimpflug_seeded_input` — detection is the
//!    expensive step, so it is never repeated per cell);
//! 2. perturbs the seed over a structured per-axis grid (focal
//!    multiplier / tilt offset / principal-point offset — never a full
//!    cross-product) and re-runs `run::solve_scheimpflug_seeded`
//!    for each cell, reusing the detected dataset;
//! 3. records whether the cell clears the entry's `accept` gate.
//!
//! Entries are grouped into dataset **families** by their shared
//! `data_root` (the 6 `rtv3d_ref_cam*` entries are one family, the 6
//! `rtv3d_ringgrid_cam*` entries are another); the report is one
//! pass-rate table per family plus the widest all-camera-pass interval
//! per perturbation axis. This subcommand is never wired into CI — it is
//! a standing diagnostic, run on demand against the private datasets.

#[cfg(feature = "tier-b")]
pub use tier_b::run_basin_study;

#[cfg(feature = "tier-b")]
mod tier_b {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use std::path::PathBuf;

    use anyhow::{Context, Result};
    use vision_calibration::scheimpflug_intrinsics::ScheimpflugManualInit;
    use vision_calibration_core::{PlanarDataset, ReprojectionStats};

    use crate::registry::{AcceptGate, BenchEntry, ProblemKind};
    use crate::run::{detect_scheimpflug_seeded_input, progress_label, solve_scheimpflug_seeded};

    /// Focal-length multiplier sweep, applied to both `fx` and `fy`.
    const FOCAL_MULTIPLIERS: [f64; 9] = [0.50, 0.70, 0.85, 0.95, 1.05, 1.15, 1.30, 1.50, 2.00];
    /// Scheimpflug tilt offset sweep in degrees, added to both `tilt_x`
    /// and `tilt_y` (converted to radians before perturbing the seed).
    const TILT_OFFSETS_DEG: [f64; 8] = [-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0];
    /// Principal-point offset sweep in pixels, applied to both `cx` and `cy`.
    const PP_OFFSETS_PX: [f64; 6] = [-100.0, -50.0, -20.0, 20.0, 50.0, 100.0];

    /// A single perturbation of the device-spec seed along one axis, or
    /// the unperturbed baseline.
    #[derive(Debug, Clone, Copy)]
    enum Cell {
        /// The spec seed exactly as derived — no perturbation.
        Baseline,
        /// Multiply `fx`/`fy` by this factor.
        Focal(f64),
        /// Add this many degrees to `tilt_x`/`tilt_y`.
        TiltDeg(f64),
        /// Add this many pixels to `cx`/`cy`.
        PrincipalPointPx(f64),
    }

    impl Cell {
        fn label(&self) -> String {
            match self {
                Cell::Baseline => "baseline (unperturbed spec seed)".to_string(),
                Cell::Focal(m) => format!("focal ×{m:.2}"),
                Cell::TiltDeg(d) => format!("tilt {d:+.2}°"),
                Cell::PrincipalPointPx(p) => format!("pp {p:+.0} px"),
            }
        }

        /// Apply this perturbation to a clone of `seed`. `scheimpflug_seed`
        /// always populates `intrinsics` and `sensor` (ADR 0022: both are
        /// load-bearing), so the `if let Some` guards are defensive, not
        /// load-bearing themselves.
        fn apply(&self, seed: &ScheimpflugManualInit) -> ScheimpflugManualInit {
            let mut out = seed.clone();
            match *self {
                Cell::Baseline => {}
                Cell::Focal(mult) => {
                    if let Some(k) = out.intrinsics.as_mut() {
                        k.fx *= mult;
                        k.fy *= mult;
                    }
                }
                Cell::TiltDeg(deg) => {
                    if let Some(s) = out.sensor.as_mut() {
                        s.tilt_x += deg.to_radians();
                        s.tilt_y += deg.to_radians();
                    }
                }
                Cell::PrincipalPointPx(px) => {
                    if let Some(k) = out.intrinsics.as_mut() {
                        k.cx += px;
                        k.cy += px;
                    }
                }
            }
            out
        }
    }

    /// The full per-axis sweep: baseline, then focal, then tilt, then
    /// principal point. ~25-35 solves per camera, never a cross-product.
    fn all_cells() -> Vec<Cell> {
        let mut cells = vec![Cell::Baseline];
        cells.extend(FOCAL_MULTIPLIERS.iter().map(|&m| Cell::Focal(m)));
        cells.extend(TILT_OFFSETS_DEG.iter().map(|&d| Cell::TiltDeg(d)));
        cells.extend(PP_OFFSETS_PX.iter().map(|&p| Cell::PrincipalPointPx(p)));
        cells
    }

    /// Per-cell pass/fail for one camera, aligned index-for-index with
    /// [`all_cells`].
    struct CameraBasinResult {
        camera_id: String,
        cells: Vec<bool>,
    }

    /// A cell "passes" iff the solve succeeds and its bench-recomputed mean
    /// reprojection error (the same statistic `calib-bench accept` gates
    /// on) is finite and at or under the entry's gate. A solve error
    /// (extreme perturbations can knock the non-linear solve over) is
    /// always a fail, never an operational abort of the whole study.
    fn cell_passes(
        dataset: &PlanarDataset,
        cell: Cell,
        seed: &ScheimpflugManualInit,
        gate: AcceptGate,
        label: &str,
    ) -> bool {
        let seeded = cell.apply(seed);
        match solve_scheimpflug_seeded(dataset.clone(), seeded, label) {
            Ok(solve) => {
                let errors: Vec<f64> = solve
                    .export
                    .per_feature_residuals
                    .target
                    .iter()
                    .filter_map(|r| r.error_px)
                    .collect();
                let mean = ReprojectionStats::from_errors(&errors).mean;
                mean.is_finite() && mean <= gate.max_per_cam_mean_px
            }
            Err(_) => false,
        }
    }

    /// Widest contiguous interval around `center` (walking outward on each
    /// side of the axis) for which every cell passed on **all** cameras.
    /// Stops at the first failing cell in each direction — the basin is
    /// assumed contiguous, so a pass past a failure is not counted. If the
    /// baseline itself does not pass on every camera, the interval
    /// collapses to `(center, center)`, which is itself the loud signal
    /// (the printed bounds degenerate to a point).
    fn axis_bounds(
        cells: &[Cell],
        all_pass: &[bool],
        center: f64,
        project: impl Fn(&Cell) -> Option<f64>,
    ) -> (f64, f64) {
        let baseline_idx = cells
            .iter()
            .position(|c| matches!(c, Cell::Baseline))
            .expect("all_cells always includes Baseline");
        if !all_pass[baseline_idx] {
            return (center, center);
        }

        let mut values: Vec<(f64, bool)> = cells
            .iter()
            .zip(all_pass)
            .filter_map(|(c, &pass)| project(c).map(|v| (v, pass)))
            .collect();
        values.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut lower = center;
        for &(v, pass) in values.iter().filter(|(v, _)| *v < center).rev() {
            if !pass {
                break;
            }
            lower = v;
        }
        let mut upper = center;
        for &(v, pass) in values.iter().filter(|(v, _)| *v > center) {
            if !pass {
                break;
            }
            upper = v;
        }
        (lower, upper)
    }

    /// Run the Q6 convergence-basin study over `entries` (already
    /// filesystem-resolved, i.e. `data_root` absolute) and render the
    /// markdown report. Returns `Err` only for operational failures
    /// (bad registry data, a family with zero available cameras across
    /// the whole run); a narrow basin is reported loudly in the text but
    /// never turned into an `Err`.
    pub fn run_basin_study(entries: &[BenchEntry]) -> Result<String> {
        anyhow::ensure!(
            !entries.is_empty(),
            "no scheimpflug_intrinsics entries provided to the basin study"
        );
        for entry in entries {
            anyhow::ensure!(
                entry.problem == ProblemKind::ScheimpflugIntrinsics,
                "basin study only supports scheimpflug_intrinsics entries, got `{}` ({:?})",
                entry.id,
                entry.problem
            );
        }

        let mut families: BTreeMap<PathBuf, Vec<&BenchEntry>> = BTreeMap::new();
        for entry in entries {
            families
                .entry(entry.data_root.clone())
                .or_default()
                .push(entry);
        }

        let cells = all_cells();
        let mut out = String::new();
        out.push_str("# Q6 Convergence-Basin Study — Seeded Scheimpflug Intrinsics\n\n");
        out.push_str(
            "Perturbs the ADR 0023 device-spec seed over a structured per-axis \
             grid and re-runs the ADR 0022 seeded route, gating each cell on \
             the entry's `accept.max_per_cam_mean_px`. See \
             `docs/notes/scheimpflug-intrinsics.md`.\n\n",
        );

        let mut envelope_violations: Vec<String> = Vec::new();

        for (data_root, mut family_entries) in families {
            family_entries.sort_by(|a, b| a.id.cmp(&b.id));
            let family_name = data_root
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| data_root.display().to_string());

            let mut camera_results: Vec<CameraBasinResult> = Vec::new();
            for entry in &family_entries {
                if !entry.data_root.is_dir() {
                    let _ = writeln!(
                        out,
                        "UNAVAILABLE  {} ({} not on disk — skipped, not passed)\n",
                        entry.id,
                        entry.data_root.display()
                    );
                    continue;
                }
                let Some(gate) = entry.accept else {
                    let _ = writeln!(
                        out,
                        "NO-GATE      {} (no `accept` gate — skipped in the basin study)\n",
                        entry.id
                    );
                    continue;
                };
                progress_label(&entry.id, "basin: detecting views (once)");
                let detected = detect_scheimpflug_seeded_input(entry)
                    .with_context(|| format!("entry `{}`: detection failed", entry.id))?;
                progress_label(&entry.id, format!("basin: sweeping {} cells", cells.len()));
                let cell_pass: Vec<bool> = cells
                    .iter()
                    .map(|&cell| {
                        let label = format!("{}/{}", entry.id, cell.label());
                        cell_passes(&detected.dataset, cell, &detected.seed, gate, &label)
                    })
                    .collect();
                camera_results.push(CameraBasinResult {
                    camera_id: detected.camera_id,
                    cells: cell_pass,
                });
            }

            if camera_results.is_empty() {
                let _ = writeln!(
                    out,
                    "## {family_name}\n\nNo cameras available for this family — skipped.\n"
                );
                continue;
            }

            render_family(
                &mut out,
                &family_name,
                &cells,
                &camera_results,
                &mut envelope_violations,
            );
        }

        render_decision_rule(&mut out, &envelope_violations);
        Ok(out)
    }

    fn render_family(
        out: &mut String,
        family_name: &str,
        cells: &[Cell],
        cameras: &[CameraBasinResult],
        envelope_violations: &mut Vec<String>,
    ) {
        let n_cams = cameras.len();
        let camera_ids = cameras
            .iter()
            .map(|c| c.camera_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "## {family_name} ({n_cams} camera{}: {camera_ids})\n",
            if n_cams == 1 { "" } else { "s" }
        );
        let _ = writeln!(out, "| cell | pass-rate |");
        let _ = writeln!(out, "|---|---:|");
        let mut all_pass: Vec<bool> = Vec::with_capacity(cells.len());
        for (idx, cell) in cells.iter().enumerate() {
            let pass_count = cameras.iter().filter(|c| c.cells[idx]).count();
            all_pass.push(pass_count == n_cams);
            let _ = writeln!(out, "| {} | {}/{} |", cell.label(), pass_count, n_cams);
        }
        out.push('\n');

        if !all_pass[0] {
            let fail_count = n_cams - cameras.iter().filter(|c| c.cells[0]).count();
            let _ = writeln!(
                out,
                "**baseline (unperturbed spec seed) FAILED on {fail_count}/{n_cams} camera(s)** \
                 — the axis bounds below collapse to the center and should not be trusted.\n"
            );
        }

        let focal_bounds = axis_bounds(cells, &all_pass, 1.0, |c| match c {
            Cell::Focal(m) => Some(*m),
            _ => None,
        });
        let tilt_bounds = axis_bounds(cells, &all_pass, 0.0, |c| match c {
            Cell::TiltDeg(d) => Some(*d),
            _ => None,
        });
        let pp_bounds = axis_bounds(cells, &all_pass, 0.0, |c| match c {
            Cell::PrincipalPointPx(p) => Some(*p),
            _ => None,
        });

        let _ = writeln!(
            out,
            "- focal basin: ×[{:.2}, {:.2}] all-pass",
            focal_bounds.0, focal_bounds.1
        );
        let _ = writeln!(
            out,
            "- tilt basin: [{:+.2}°, {:+.2}°] all-pass",
            tilt_bounds.0, tilt_bounds.1
        );
        let _ = writeln!(
            out,
            "- principal-point basin: [{:+.0}, {:+.0}] px all-pass\n",
            pp_bounds.0, pp_bounds.1
        );

        // Decision rule (ADR 0022/0023, Q6): the basin must comfortably
        // contain realistic spec error — focal ±5 %, tilt ±2°.
        if focal_bounds.0 > 0.95 || focal_bounds.1 < 1.05 {
            envelope_violations.push(format!(
                "{family_name}: focal basin ×[{:.2}, {:.2}] does not cover ±5% (needs ⊇ [0.95, 1.05])",
                focal_bounds.0, focal_bounds.1
            ));
        }
        if tilt_bounds.0 > -2.0 || tilt_bounds.1 < 2.0 {
            envelope_violations.push(format!(
                "{family_name}: tilt basin [{:+.2}°, {:+.2}°] does not cover ±2° (needs ⊇ [-2°, +2°])",
                tilt_bounds.0, tilt_bounds.1
            ));
        }
    }

    fn render_decision_rule(out: &mut String, violations: &[String]) {
        out.push_str("## Decision Rule\n\n");
        out.push_str(
            "The seeded route's basin must comfortably contain realistic spec \
             error: focal ±5 % and tilt ±2° (ADR 0022/0023). Basin narrowness \
             never fails this command's exit code — only operational errors do.\n\n",
        );
        if violations.is_empty() {
            out.push_str(
                "RESULT: PASS — every family's focal and tilt basin covers the \
                 ±5 % / ±2° envelope on all its cameras.\n",
            );
            return;
        }
        out.push_str("================================================================\n");
        out.push_str("WARNING: the seeded-init basin does NOT cover the realistic\n");
        out.push_str("spec-error envelope (focal ±5%, tilt ±2°) on every camera.\n");
        out.push_str("This finding REOPENS the Phase A sweep design (docs/backlog.md Q6).\n");
        out.push_str("================================================================\n\n");
        for v in violations {
            let _ = writeln!(out, "- {v}");
        }
    }
}
