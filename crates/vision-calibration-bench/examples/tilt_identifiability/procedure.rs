//! The determination procedure end to end, and the prior branch.
//!
//! - `prior` (C): a Gaussian prior on the tilt, right or wrong by 1° or 3°,
//!   with the tangential terms held or freed; on a determined, an
//!   undetermined and two decentred lenses. What the posterior σ says, what
//!   the estimate does, and whether a wrong prior shows in the data.
//! - `rule` (F): the procedure as the library would run it. Step 1 is the
//!   plain camera through the pipeline (untilted, Brown-Conrady, `p1, p2`
//!   free, cold linear init). Step 2 fits the tilted camera (M1) from step
//!   1, seeded by the decentring identity `τx ≈ atan(−p1/k1)`,
//!   `τy ≈ atan(p2/k1)` with `K` and the poses moved from the sensor-normal
//!   frame to the lens frame, unbounded. Step 3 decides on M1's stated σ and
//!   keeps M1 or step 1's camera.

use anyhow::Result;
use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration_core::{Iso3, PlanarDataset};
use vision_calibration_optim::SolverBackend;

use crate::determine::{Design, M0P, M1, M2, dof, fit_model, par_map, tilt_stat};
use crate::downstream::{Bench, Metrics, evaluate};
use crate::model::{Cam, FreeSet, Param, tilt_rotation};
use crate::scene::{CamSpec, ViewDesign};
use crate::solve::{Outcome, TiltPrior, backend_opts, route_plain, solve_direct, solve_param_with};
use crate::study::{Size, deg, header, num, row};
use crate::verdict::CHI2_2_99;

/// The pixel σ of a fit, `√(2·cost / dof)`.
fn pixel_sigma(fit: &Outcome, data: &PlanarDataset, model: usize) -> f64 {
    (2.0 * fit.cost / dof(data, model)).sqrt()
}

/// The largest ratio of `a`'s errors to `b`'s over the four metrics.
fn worst(a: &Metrics, b: &Metrics) -> f64 {
    a.values()
        .iter()
        .zip(b.values())
        .map(|(x, y)| x / y)
        .fold(0.0, f64::max)
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    (m, s)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// The cameras of the prior study.
fn prior_cells() -> Vec<(&'static str, CamSpec, ViewDesign)> {
    let spec = |hfov: f64, dist: f64, tilt: (f64, f64), p: (f64, f64)| CamSpec {
        hfov_deg: hfov,
        corner_distortion: dist,
        tilt_deg: tilt,
        p,
        ..CamSpec::baseline()
    };
    vec![
        (
            "determined: 50°, −0.05, tilt 5°",
            spec(50.0, -0.05, (5.0, 0.0), (0.0, 0.0)),
            ViewDesign::baseline(),
        ),
        (
            "undetermined: 70°, −0.005, tilt 10°",
            spec(70.0, -0.005, (10.0, 0.0), (0.0, 0.0)),
            ViewDesign::baseline(),
        ),
        (
            "undetermined, 8 central views: 50°, −0.02, tilt 10°",
            spec(50.0, -0.02, (10.0, 0.0), (0.0, 0.0)),
            Design::Poor.view_design(),
        ),
        (
            "decentred, determined: 50°, −0.05, tilt 5°, p1 = 5e-4",
            spec(50.0, -0.05, (5.0, 0.0), (5e-4, 0.0)),
            ViewDesign::baseline(),
        ),
        (
            "decentred, undetermined: 70°, −0.005, tilt 10°, p1 = 1e-4",
            spec(70.0, -0.005, (10.0, 0.0), (1e-4, 0.0)),
            ViewDesign::baseline(),
        ),
    ]
}

/// A prior variant: σ, offset of its mean from the truth (on `tilt_x`), and
/// whether `p1, p2` are freed.
#[derive(Clone, Copy)]
struct PriorVariant {
    sigma_deg: f64,
    offset_deg: f64,
    p_free: bool,
}

fn prior_variants() -> Vec<PriorVariant> {
    let mut v = Vec::new();
    for p_free in [false, true] {
        for sigma_deg in [0.25, 1.0] {
            for offset_deg in [0.0, 1.0, 3.0] {
                v.push(PriorVariant {
                    sigma_deg,
                    offset_deg,
                    p_free,
                });
            }
        }
    }
    v
}

/// One draw of a prior cell.
struct PriorDraw {
    /// M1 and M2 without a prior: tilt_x error (rad), stated σ(tilt_x).
    m1: (f64, f64),
    m2: (f64, f64),
    /// Per variant: tilt_x error, stated σ(tilt_x), p1 error, worst metric
    /// ratio against M1's (no prior) in the same draw, prior-data conflict
    /// `z` (the no-prior fit of the variant's free set against the prior).
    variants: Vec<(f64, f64, f64, f64, f64)>,
}

fn prior_draw(bench: &Bench, variants: &[PriorVariant], seed: u64) -> Result<PriorDraw> {
    let noise = 0.2;
    let data = bench.train.dataset(noise, seed);
    let data2 = bench.partner_data(noise, seed);
    let truth = bench.train.cam;
    let plain = fit_model(M0P, &data, &truth, &bench.train.poses)?;
    let sigma_px = pixel_sigma(&plain, &data, M0P);
    let m1 = fit_model(M1, &data, &truth, &bench.train.poses)?;
    let m2 = fit_model(M2, &data, &truth, &bench.train.poses)?;
    let m1_metrics = evaluate(bench, &m1, &data, &data2)?;
    let sx = |o: &Outcome| o.sigma_of("tilt_x").unwrap_or(f64::NAN);
    let mut out = Vec::new();
    for v in variants {
        let mean = [truth.tilt[0] + v.offset_deg.to_radians(), truth.tilt[1]];
        // Started at the prior mean, in the gauge-aligned coordinates: the
        // raw ones stop about one σ short of the posterior mean along the
        // tilt–decentring valley (table C0).
        let mut start = truth;
        start.tilt = mean;
        if !v.p_free {
            start.d[3] = 0.0;
            start.d[4] = 0.0;
        }
        let fit = solve_param_with(
            &data,
            &start,
            &bench.train.poses,
            Param::Gauge,
            if v.p_free {
                FreeSet::Tangential
            } else {
                FreeSet::Default
            },
            300,
            Some(TiltPrior {
                mean,
                sigma: v.sigma_deg.to_radians(),
                pixel_sigma: sigma_px,
            }),
        )?;
        let metrics = evaluate(bench, &fit, &data, &data2)?;
        let free_fit = if v.p_free { &m2 } else { &m1 };
        let conflict = (free_fit.cam.tilt[0] - mean[0])
            / (sx(free_fit).powi(2) + v.sigma_deg.to_radians().powi(2)).sqrt();
        out.push((
            fit.cam.tilt[0] - truth.tilt[0],
            sx(&fit),
            fit.cam.d[3] - truth.d[3],
            worst(&metrics, &m1_metrics),
            conflict,
        ));
    }
    Ok(PriorDraw {
        m1: (m1.cam.tilt[0] - truth.tilt[0], sx(&m1)),
        m2: (m2.cam.tilt[0] - truth.tilt[0], sx(&m2)),
        variants: out,
    })
}

/// C: the prior branch.
pub fn prior_study(size: &Size) -> Result<()> {
    let draws = size.label_draws;
    println!(
        "## A prior on the tilt ({draws} draws per cell, noise 0.2 px, started at the prior mean)\n"
    );
    println!(
        "The prior `tilt ~ N(τ₀, σₚ²)` per axis, weighed against the corners at the plain fit's \
         pixel σ (both stated, so the covariance is the posterior). Offset: `τ₀ − τ` on \
         `tilt_x`. Columns: `tilt_x` bias and scatter, the stated posterior σ, RMS error / \
         stated σ, `p1` bias, the median of the worst metric ratio against the no-prior M1 in \
         the same draw, and the share of draws whose prior-data conflict `z = (τ̂ − τ₀) / √(σ̂² \
         + σₚ²)` (τ̂, σ̂ from the same free set without the prior) exceeds 3.\n"
    );
    prior_solver_check()?;
    let variants = prior_variants();
    let cells = prior_cells();
    let results = par_map(&cells, |(_, spec, design)| -> Result<Vec<PriorDraw>> {
        let bench = Bench::new(*spec, design)?;
        (0..draws)
            .map(|d| prior_draw(&bench, &variants, 3000 + 97 * d as u64))
            .collect()
    });
    header(&[
        "camera",
        "fit",
        "tilt_x bias °",
        "scatter °",
        "stated σ °",
        "RMS / σ",
        "p1 bias",
        "metric ratio vs M1",
        "conflict |z| > 3",
    ]);
    for ((name, _, _), res) in cells.iter().zip(results) {
        let res = res?;
        for (label, pick) in [("M1, no prior", 0usize), ("M2, no prior", 1)] {
            let e: Vec<f64> = res
                .iter()
                .map(|d| if pick == 0 { d.m1.0 } else { d.m2.0 })
                .collect();
            let s: Vec<f64> = res
                .iter()
                .map(|d| if pick == 0 { d.m1.1 } else { d.m2.1 })
                .collect();
            let (b, sc) = mean_std(&e);
            let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
            let sm = median(s);
            row(&[
                (*name).into(),
                label.into(),
                deg(b),
                deg(sc),
                deg(sm),
                num(rms / sm, 2),
                "—".into(),
                "—".into(),
                "—".into(),
            ]);
        }
        for (vi, v) in variants.iter().enumerate() {
            let e: Vec<f64> = res.iter().map(|d| d.variants[vi].0).collect();
            let s: Vec<f64> = res.iter().map(|d| d.variants[vi].1).collect();
            let p: Vec<f64> = res.iter().map(|d| d.variants[vi].2).collect();
            let w: Vec<f64> = res.iter().map(|d| d.variants[vi].3).collect();
            let z = res.iter().filter(|d| d.variants[vi].4.abs() > 3.0).count();
            let (b, sc) = mean_std(&e);
            let rms = (e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt();
            let sm = median(s);
            row(&[
                (*name).into(),
                format!(
                    "prior σₚ {}°, offset {}°, p {}",
                    v.sigma_deg,
                    v.offset_deg,
                    if v.p_free { "free" } else { "held" }
                ),
                deg(b),
                deg(sc),
                deg(sm),
                num(rms / sm, 2),
                num(mean_std(&p).0, 2),
                num(median(w), 3),
                format!("{z}/{}", res.len()),
            ]);
        }
    }
    Ok(())
}

/// The tilted camera seeded from a plain one by the decentring identity:
/// `τx = atan(−p1/k1)`, `τy = atan(p2/k1)` (each clamped to ±0.5 rad), `K`
/// moved from the sensor-normal pinhole to the lens-axis one and the poses
/// from the sensor-normal frame to the lens frame; `p1, p2` zeroed.
pub fn decentring_seed(plain: &Outcome) -> (Cam, Vec<Iso3>) {
    let c = plain.cam;
    let k1 = c.d[0];
    let tilt = if k1.abs() > 1e-12 {
        [
            (-c.d[3] / k1).atan().clamp(-0.5, 0.5),
            (c.d[4] / k1).atan().clamp(-0.5, 0.5),
        ]
    } else {
        [0.0, 0.0]
    };
    let k = Param::Effective.raw_intrinsics(&c.k, &tilt);
    let mut d = c.d;
    d[3] = 0.0;
    d[4] = 0.0;
    let r = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(tilt_rotation(
        tilt[0], tilt[1],
    )));
    let to_lens = Iso3::from_parts(Translation3::identity(), r.inverse());
    let poses = plain.poses.iter().map(|p| to_lens * p).collect();
    (Cam { k, d, tilt }, poses)
}

/// What the procedure decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    /// The tilt is determined: estimated.
    Estimated,
    /// Not determined, and the data do not reject zero: fixed at zero, `p1,
    /// p2` free.
    FixedUndetermined,
    /// Not determined, but the data reject zero (likelihood ratio of the
    /// plain camera against the tilted one with `p` free): estimated, with
    /// a warning.
    EstimatedImprecise,
}

/// One draw of the procedure.
struct RuleDraw {
    branch: Branch,
    /// The pipeline's cold plain route failed (its linear init).
    plain_failed: bool,
    /// M1's stated σ (rad) and the likelihood ratio of step 1 against M2.
    sigma: f64,
    lr: f64,
    /// Whether the seeded M1 reached the reference minimum (within 0.5 of
    /// its σ in every camera coordinate it frees).
    reached: bool,
    /// Worst metric ratio of the procedure's camera against the reference
    /// M1's in the same draw, and the absolute held-out error.
    ratio: f64,
    heldout: f64,
    /// Frame of the reported camera, from the sensor-normal frame, degrees.
    frame_sensor: f64,
    /// Reported tilt_x error, degrees (the tilt reported is zero when fixed).
    tilt_err_deg: f64,
}

fn rule_draw(bench: &Bench, theta: f64, noise: f64, seed: u64) -> Result<RuleDraw> {
    let data = bench.train.dataset(noise, seed);
    let data2 = bench.partner_data(noise, seed);
    let truth = bench.train.cam;
    // Step 1: the plain camera through the pipeline; where its cold linear
    // init fails (strong distortion defeats Zhang's method), the plain fit
    // from the sensor-normal pinhole, as a seeded route would start it.
    let (plain, plain_failed) = match route_plain(&data, SolverBackend::TinySolver) {
        Ok(p) => (p, false),
        Err(_) => (fit_model(M0P, &data, &truth, &bench.train.poses)?, true),
    };
    // Step 2: the tilted camera seeded by the decentring, unbounded, run to
    // a tight stop (the default stopping rule halts along the weak-distortion
    // valley; backlog SOLVER-VALLEY-STOP).
    let (seed_cam, seed_poses) = decentring_seed(&plain);
    let m1 = solve_direct(
        &data,
        &seed_cam,
        &seed_poses,
        FreeSet::Default,
        None,
        backend_opts(SolverBackend::TinySolver, 500, true),
    )?;
    // The guard's alternative: tilt and `p` free, from M1.
    let m2 = solve_direct(
        &data,
        &m1.cam,
        &m1.poses,
        FreeSet::Tangential,
        None,
        backend_opts(SolverBackend::TinySolver, 500, true),
    )?;
    let reference = fit_model(M1, &data, &truth, &bench.train.poses)?;
    let reached = {
        let a = m1.cam.raw();
        let b = reference.cam.raw();
        FreeSet::Default.indices().iter().all(|&i| {
            let name = crate::model::RAW_NAMES[i];
            reference
                .sigma_of(name)
                .is_some_and(|s| (a[i] - b[i]).abs() <= 0.5 * s)
        })
    };
    let stat = tilt_stat(&m1);
    let sigma = stat.map_or(f64::INFINITY, |s| s.sigma);
    let lr = (plain.cost - m2.cost) * dof(&data, M2) / m2.cost;
    let branch = if sigma <= theta {
        Branch::Estimated
    } else if lr > CHI2_2_99 {
        Branch::EstimatedImprecise
    } else {
        Branch::FixedUndetermined
    };
    let chosen = match branch {
        Branch::Estimated | Branch::EstimatedImprecise => &m1,
        Branch::FixedUndetermined => &plain,
    };
    let metrics = evaluate(bench, chosen, &data, &data2)?;
    let ref_metrics = evaluate(bench, &reference, &data, &data2)?;
    Ok(RuleDraw {
        branch,
        plain_failed,
        sigma,
        lr,
        reached,
        ratio: worst(&metrics, &ref_metrics),
        heldout: metrics.heldout,
        frame_sensor: metrics.frame_sensor_deg,
        tilt_err_deg: (chosen.cam.tilt[0] - truth.tilt[0]).to_degrees(),
    })
}

/// F: the procedure end to end.
pub fn rule_study(size: &Size) -> Result<()> {
    let theta = 0.5_f64.to_radians();
    let draws = 2 * size.det_draws;
    println!(
        "## The procedure end to end (θ = 0.5°, guard at LR > {CHI2_2_99}; {draws} draws per \
         cell, noise 0.2 px)\n"
    );
    println!(
        "Step 1 the plain camera through the pipeline; step 2 the tilted camera seeded by the \
         decentring identity, unbounded; step 3 the decision. Columns: the branch taken (E \
         estimated, I estimated but imprecise, Z fixed at zero with `p` free), the seeded tilted \
         fit reaching the reference minimum, the median stated σ, the median likelihood ratio \
         of step 1 against M2, the worst metric ratio of the reported camera against the \
         reference M1 (median / max), its held-out reprojection error (median px), the reported \
         `tilt_x` error (median °) and the reported frame's rotation from the sensor normal \
         (median °).\n"
    );
    let mut cells: Vec<(String, CamSpec, ViewDesign)> = Vec::new();
    for design in [Design::Rich, Design::Poor] {
        for &hfov in &[35.0, 50.0, 90.0] {
            for &dist in &[-0.005, -0.02, -0.15] {
                for &tilt in &[0.0, 5.0, 20.0] {
                    cells.push((
                        format!("{}, {hfov}°, {dist}, tilt {tilt}°", design.name()),
                        CamSpec {
                            hfov_deg: hfov,
                            corner_distortion: dist,
                            tilt_deg: (tilt, 0.0),
                            ..CamSpec::baseline()
                        },
                        design.view_design(),
                    ));
                }
            }
        }
    }
    for (name, hfov, dist, p1) in [
        (
            "decentred, untilted: 50°, −0.05, p1 = 5e-4",
            50.0,
            -0.05,
            5e-4,
        ),
        (
            "decentred, untilted: 70°, −0.005, p1 = 1e-4",
            70.0,
            -0.005,
            1e-4,
        ),
    ] {
        cells.push((
            name.into(),
            CamSpec {
                hfov_deg: hfov,
                corner_distortion: dist,
                tilt_deg: (0.0, 0.0),
                p: (p1, 0.0),
                ..CamSpec::baseline()
            },
            ViewDesign::baseline(),
        ));
    }
    cells.push((
        "real-rig-like: 720×540, 35°, −0.07, tilt (−5°, 0.3°), 22 views".into(),
        CamSpec {
            size: (720.0, 540.0),
            hfov_deg: 35.0,
            corner_distortion: -0.07,
            tilt_deg: (-5.0, 0.3),
            p: (0.0, 0.0),
        },
        ViewDesign {
            views: 22,
            ..ViewDesign::baseline()
        },
    ));
    cells.push((
        "real-rig-like, 8 central views".into(),
        CamSpec {
            size: (720.0, 540.0),
            hfov_deg: 35.0,
            corner_distortion: -0.07,
            tilt_deg: (-5.0, 0.3),
            p: (0.0, 0.0),
        },
        Design::Poor.view_design(),
    ));
    let results = par_map(&cells, |(_, spec, design)| -> Result<Vec<RuleDraw>> {
        let bench = Bench::new(*spec, design)?;
        let n = if spec.size.0 < 1000.0 {
            4 * draws
        } else {
            draws
        };
        (0..n)
            .map(|d| rule_draw(&bench, theta, 0.2, 7000 + 131 * d as u64))
            .collect()
    });
    header(&[
        "cell",
        "E / I / Z",
        "step 1 init failed",
        "seed reached",
        "σ̂ °",
        "LR",
        "ratio median / max",
        "held-out px",
        "tilt_x error °",
        "frame from sensor normal °",
    ]);
    for ((name, _, _), res) in cells.iter().zip(results) {
        let res = res?;
        let count = |b: Branch| res.iter().filter(|d| d.branch == b).count();
        let ratios: Vec<f64> = res.iter().map(|d| d.ratio).collect();
        row(&[
            name.clone(),
            format!(
                "{} / {} / {}",
                count(Branch::Estimated),
                count(Branch::EstimatedImprecise),
                count(Branch::FixedUndetermined)
            ),
            res.iter().filter(|d| d.plain_failed).count().to_string(),
            format!("{}/{}", res.iter().filter(|d| d.reached).count(), res.len()),
            deg(median(res.iter().map(|d| d.sigma).collect())),
            num(median(res.iter().map(|d| d.lr).collect()), 2),
            format!(
                "{} / {}",
                num(median(ratios.clone()), 3),
                num(ratios.iter().copied().fold(0.0, f64::max), 3)
            ),
            num(median(res.iter().map(|d| d.heldout).collect()), 2),
            num(median(res.iter().map(|d| d.tilt_err_deg).collect()), 2),
            num(median(res.iter().map(|d| d.frame_sensor).collect()), 2),
        ]);
    }
    Ok(())
}

/// C0: does the prior solve reach its minimum? The undetermined lens
/// (70°, −0.005, tilt 10°), `p1, p2` free, prior σₚ 1° offset by 1°, from
/// the truth and from the prior mean, in the raw and the gauge-aligned
/// coordinates.
fn prior_solver_check() -> Result<()> {
    println!(
        "### C0 The prior solve's convergence (70°, −0.005, tilt 10°, `p` free, σₚ 1°, offset \
         1°)\n"
    );
    header(&[
        "draw",
        "coordinates, start",
        "tilt_x error °",
        "stated σ °",
        "cost",
        "iterations",
    ]);
    let spec = CamSpec {
        hfov_deg: 70.0,
        corner_distortion: -0.005,
        tilt_deg: (10.0, 0.0),
        ..CamSpec::baseline()
    };
    let bench = Bench::new(spec, &ViewDesign::baseline())?;
    let truth = bench.train.cam;
    for seed in 0..3u64 {
        let data = bench.train.dataset(0.2, 3000 + 97 * seed);
        let plain = fit_model(M0P, &data, &truth, &bench.train.poses)?;
        let sigma_px = pixel_sigma(&plain, &data, M0P);
        let mean = [truth.tilt[0] + 1f64.to_radians(), truth.tilt[1]];
        let prior = TiltPrior {
            mean,
            sigma: 1f64.to_radians(),
            pixel_sigma: sigma_px,
        };
        let mut at_mean = truth;
        at_mean.tilt = mean;
        for (name, param, start) in [
            ("raw, truth", Param::Raw, truth),
            ("raw, prior mean", Param::Raw, at_mean),
            ("gauge-aligned, truth", Param::Gauge, truth),
            ("gauge-aligned, prior mean", Param::Gauge, at_mean),
        ] {
            let fit = solve_param_with(
                &data,
                &start,
                &bench.train.poses,
                param,
                FreeSet::Tangential,
                300,
                Some(prior),
            )?;
            row(&[
                seed.to_string(),
                name.into(),
                num((fit.cam.tilt[0] - truth.tilt[0]).to_degrees(), 3),
                deg(fit.sigma_of("tilt_x").unwrap_or(f64::NAN)),
                format!("{:.3}", fit.cost),
                fit.iters.to_string(),
            ]);
        }
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::downstream::frame_rotation;
    use crate::scene::build_scene;

    fn plain_fit(spec: CamSpec) -> (crate::scene::Scene, Outcome) {
        let scene = build_scene(spec, &ViewDesign::baseline()).unwrap();
        let data = scene.dataset(0.0, 1);
        let fit = fit_model(M0P, &data, &scene.cam, &scene.poses).unwrap();
        (scene, fit)
    }

    /// The plain camera fitted to a tilted one carries the tilt in its
    /// decentring, and the seed reads it back to first order.
    #[test]
    fn decentring_seed_recovers_the_tilt() {
        let spec = CamSpec {
            corner_distortion: -0.02,
            tilt_deg: (5.0, -3.0),
            ..CamSpec::baseline()
        };
        let (scene, fit) = plain_fit(spec);
        let (seed, poses) = decentring_seed(&fit);
        for axis in 0..2 {
            let rel = (seed.tilt[axis] - scene.cam.tilt[axis]) / scene.cam.tilt[axis];
            assert!(rel.abs() < 0.15, "axis {axis}: {rel}");
        }
        // The seeded poses are back in the lens frame.
        let frame = frame_rotation(&poses, &scene.poses);
        let angle = UnitQuaternion::from_rotation_matrix(&frame)
            .angle()
            .to_degrees();
        assert!(angle < 1.0, "{angle}");
    }

    /// With the tilt fixed at zero and the decentring free, the camera frame
    /// is the sensor normal, not the lens axis.
    #[test]
    fn the_plain_camera_frame_is_the_sensor_normal() {
        let spec = CamSpec {
            corner_distortion: -0.005,
            tilt_deg: (10.0, 0.0),
            ..CamSpec::baseline()
        };
        let (scene, fit) = plain_fit(spec);
        let frame = frame_rotation(&fit.poses, &scene.poses);
        let r =
            Rotation3::from_matrix_unchecked(tilt_rotation(scene.cam.tilt[0], scene.cam.tilt[1]));
        let angle = |r: Rotation3<f64>| {
            UnitQuaternion::from_rotation_matrix(&r)
                .angle()
                .to_degrees()
        };
        assert!(
            angle(frame * r.inverse()) < 0.1,
            "{}",
            angle(frame * r.inverse())
        );
        assert!((angle(frame) - 10.0).abs() < 0.2, "{}", angle(frame));
    }
}
