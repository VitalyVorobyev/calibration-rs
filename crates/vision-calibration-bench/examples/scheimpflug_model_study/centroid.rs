//! `--parts centroid`: is the defocus-coupled part of the residual an
//! optical centroid shift proportional to *signed* defocus?
//!
//! Hypothesis: a pupil illuminated asymmetrically (vignetting) moves the
//! centroid of a defocused spot by the spot radius times the pupil
//! centroid's offset (as a fraction of the pupil radius), which changes
//! sign through focus. With `g = κ·(1/z − π·(x, y, 1))` the signed defocus
//! of the fitted plane of sharp focus (`focus.rs`, `κ = D·f/4`, so the spot
//! radius is `2|g|`), the displacement is `g·(a + B·n + …)`, `n` the
//! normalized image position, and the pupil-centroid offset is
//! `ε(n) = (a + B·n)/2` of the pupil radius. Radial vignetting about the
//! lens axis predicts `B` real and isotropic (the `z` term, imaginary and
//! `z̄` parts zero) and `a ≈ −B·n_axis`.
//!
//! Rows are linearized fits about the route, held out by view folds, the
//! term evaluated at the model's projection, its covariate (`g`, from the
//! full-data blur and the route's geometry, never the residual) fixed per
//! point: exactly the harness of the `field` part, against its unsigned
//! "geometric blur × deg 0–3" and "+ depth" rows. Null: the same on plain
//! synthetic scenes (`--parts synth,centroid synth_plain_cam1..6`).

use anyhow::Result;
use nalgebra::Point3;

use crate::field_study::{self, Local, Rig, eval_local, eval_shared, jack_ratio, with_term};
use crate::focus::{BlurObs, fit_pof, tilt_rotation};
use crate::linfield::{Basis, CMono, Concat, LinPoint, NCAM, Scaled};
use crate::{CamCtx, CameraData};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Fill `defocus` on every point of every camera from a PoF fitted to the
/// camera's full-data blur in the `K'` frame of its route.
fn add_defocus(rig: &mut Rig) {
    for c in &mut rig.cams {
        let r = tilt_rotation(c.base.tilt);
        let kp = |p: &LinPoint| r * (p.pose * Point3::new(p.board[0], p.board[1], 0.0)).coords;
        let obs: Vec<BlurObs> = c
            .full
            .iter()
            .filter_map(|p| p.blur.map(|b| BlurObs { x: kp(p), blur: b }))
            .collect();
        let Some(fit) = fit_pof(&obs, None) else {
            println!("  {}: no focus fit", c.id);
            continue;
        };
        println!(
            "  {}: PoF π ({:+.2}, {:+.2}, {:+.2}) /m, κ {:.3} px·m, σ0 {:.2} px from {} markers",
            c.id,
            fit.pi.x,
            fit.pi.y,
            fit.pi.z,
            fit.kappa,
            fit.s0,
            obs.len()
        );
        let set = |p: &mut LinPoint| p.defocus = Some(fit.g(&kp(p)));
        c.full.iter_mut().for_each(set);
        for (tr, te) in &mut c.folds {
            tr.iter_mut().for_each(set);
            te.iter_mut().for_each(set);
        }
    }
}

type Make = Box<dyn Fn() -> Box<dyn Basis>>;

pub fn run(cams: &[CameraData], ctxs: &[CamCtx]) -> Result<()> {
    let mut rigs = field_study::build(cams, ctxs)?;
    for rig in &mut rigs {
        println!("\n# Centroid shift ∝ signed defocus — rig {}", rig.name);
        add_defocus(rig);
        if !rig
            .cams
            .iter()
            .all(|c| c.full.iter().all(|p| p.defocus.is_some()))
        {
            println!("  (skipped: a camera has no focus fit)");
            continue;
        }
        study(rig);
    }
    Ok(())
}

fn study(rig: &Rig) {
    let size = rig.cams[0].size;
    let (w, h) = (f64::from(size[0]), f64::from(size[1]));
    let centre = [w / 2.0, h / 2.0];
    let deg = move |lo, hi| CMono::degrees(lo, hi, centre, w / 2.0);
    let pts = || rig.cams.iter().flat_map(|c| c.full.iter());
    let g_abs_med = median(pts().map(|p| p.defocus.unwrap_or(0.0).abs()).collect());
    let g_sq_med = median(pts().map(|p| p.defocus.unwrap_or(0.0).powi(2)).collect());
    let gblur_med = median(pts().filter_map(|p| p.gblur).collect());
    let zbar = pts().map(|p| p.depth).sum::<f64>() / pts().count() as f64;
    let n_neg = pts().filter(|p| p.defocus.unwrap_or(0.0) < 0.0).count();
    println!(
        "  signed defocus: median |g| {g_abs_med:.2} px, {:.0} % of markers beyond the PoF (g < 0)",
        100.0 * n_neg as f64 / pts().count() as f64
    );
    let signed = move |k: usize| -> Make {
        Box::new(move || {
            Box::new(Scaled {
                inner: deg(0, k),
                f: Box::new(|p: &LinPoint| p.defocus.unwrap_or(0.0)),
            })
        })
    };
    let unsigned = move |k: usize| -> Make {
        Box::new(move || {
            Box::new(Scaled {
                inner: deg(0, k),
                f: Box::new(move |p: &LinPoint| p.defocus.unwrap_or(0.0).abs() - g_abs_med),
            })
        })
    };
    let squared = move |k: usize| -> Make {
        Box::new(move || {
            Box::new(Scaled {
                inner: deg(0, k),
                f: Box::new(move |p: &LinPoint| {
                    (p.defocus.unwrap_or(0.0).powi(2) - g_sq_med) / g_abs_med
                }),
            })
        })
    };
    let geo = move || -> Box<dyn Basis> {
        Box::new(Scaled {
            inner: deg(0, 3),
            f: Box::new(move |p: &LinPoint| p.gblur.map_or(0.0, |b| b - gblur_med)),
        })
    };
    let depth = move || -> Box<dyn Basis> {
        Box::new(Scaled {
            inner: deg(1, 4),
            f: Box::new(move |p: &LinPoint| zbar / p.depth - 1.0),
        })
    };
    // Depth at the signed term's degrees: at a fixed image position `g` is
    // affine in `1/z`, so `g × field` is `depth × field` plus an image-fixed
    // field tied to the plane of sharp focus.
    let inv_depth = move |k: usize| -> Make {
        Box::new(move || {
            Box::new(Scaled {
                inner: deg(0, k),
                f: Box::new(move |p: &LinPoint| zbar / p.depth - 1.0),
            })
        })
    };
    let rows: Vec<(String, Make)> = vec![
        (
            "geometric blur (σ_geo − median) × deg 0-3".into(),
            Box::new(geo),
        ),
        (
            "geometric blur × deg 0-3 + depth × deg 1-4".into(),
            Box::new(move || Box::new(Concat(vec![geo(), depth()]))),
        ),
        ("signed g × deg 0-1 (a + B·n)".into(), signed(1)),
        ("signed g × deg 0-2".into(), signed(2)),
        ("signed g × deg 0-3".into(), signed(3)),
        ("depth (z̄/z − 1) × deg 0-1".into(), inv_depth(1)),
        ("depth (z̄/z − 1) × deg 0-3".into(), inv_depth(3)),
        ("|g| × deg 0-1".into(), unsigned(1)),
        ("|g| × deg 0-3".into(), unsigned(3)),
        ("g² × deg 0-1".into(), squared(1)),
        ("g² × deg 0-3".into(), squared(3)),
        (
            "signed g × deg 0-1 + |g| × deg 0-3".into(),
            Box::new(move || Box::new(Concat(vec![signed(1)(), unsigned(3)()]))),
        ),
        (
            "signed g × deg 0-3 + |g| × deg 0-3".into(),
            Box::new(move || Box::new(Concat(vec![signed(3)(), unsigned(3)()]))),
        ),
        (
            "depth × deg 0-1 + |g| × deg 0-3".into(),
            Box::new(move || Box::new(Concat(vec![inv_depth(1)(), unsigned(3)()]))),
        ),
        (
            "depth × deg 0-3 + |g| × deg 0-3".into(),
            Box::new(move || Box::new(Concat(vec![inv_depth(3)(), unsigned(3)()]))),
        ),
        (
            "signed g × deg 0-1 + geometric blur + depth".into(),
            Box::new(move || Box::new(Concat(vec![signed(1)(), geo(), depth()]))),
        ),
        (
            "signed g × deg 0-3 + geometric blur + depth".into(),
            Box::new(move || Box::new(Concat(vec![signed(3)(), geo(), depth()]))),
        ),
    ];
    let route = eval_local(rig, "route", &|_, _| Local::free(None));
    let r0 = route.pooled()[0];
    println!("\n## Per camera (camera free), held out by view folds");
    println!("| model | rms / mean / median | Δ var vs route | per-camera mean |");
    println!("{}", route.line(None));
    let mut signed1 = None;
    for (name, make) in &rows {
        let row = eval_local(rig, name, &|_, _| Local::free(Some(make())));
        println!("{}", row.line(Some(r0)));
        if name.starts_with("signed g × deg 0-1 (") {
            signed1 = Some(row);
        }
    }
    // Next to a free image field (φ ≤ 6, per camera): there `signed g × k`
    // and `depth × k` span nearly the same space, so what is left to tell
    // them apart is only what φ cannot hold.
    let phi_row = eval_local(rig, "φ ≤ 6 (camera free)", &|_, _| {
        Local::free(Some(Box::new(rig.phi.clone())))
    });
    let f0 = phi_row.pooled()[0];
    println!("\n## Next to the image field φ ≤ 6 (Δ var vs φ)");
    println!("{}", phi_row.line(Some(r0)));
    for (name, make) in [
        ("φ + signed g × deg 0-1", signed(1)),
        ("φ + depth × deg 0-1", inv_depth(1)),
        ("φ + |g| × deg 0-1", unsigned(1)),
        ("φ + signed g × deg 0-3", signed(3)),
        ("φ + depth × deg 0-3", inv_depth(3)),
        ("φ + |g| × deg 0-3", unsigned(3)),
    ] {
        println!("{}", with_term(rig, name, make.as_ref()).line(Some(f0)));
    }
    let both = |name: &str, a: Make, b: Make| {
        let row = with_term(rig, name, &|| Box::new(Concat(vec![a(), b()])));
        println!("{}", row.line(Some(f0)));
    };
    both(
        "φ + signed g × deg 0-3 + |g| × deg 0-3",
        signed(3),
        unsigned(3),
    );
    both(
        "φ + depth × deg 0-3 + |g| × deg 0-3",
        inv_depth(3),
        unsigned(3),
    );
    println!("\n## Shared across the rig's cameras (one coefficient set, cameras free)");
    for (name, make) in [
        ("signed g × deg 0-1, shared", signed(1)),
        ("signed g × deg 0-3, shared", signed(3)),
        ("|g| × deg 0-3, shared", unsigned(3)),
    ] {
        let (row, coefs) = eval_shared(rig, name, &|_| make());
        println!("{}", row.line(Some(r0)));
        if name.starts_with("signed g × deg 0-1") {
            let Some(full) = coefs.last().cloned() else {
                continue;
            };
            let nf = coefs.len() - 1;
            let jk: Vec<f64> = (0..full.len())
                .map(|k| {
                    let v: Vec<f64> = coefs[..nf].iter().map(|c| c[k]).collect();
                    let m = v.iter().sum::<f64>() / nf as f64;
                    ((nf as f64 - 1.0) / nf as f64 * v.iter().map(|x| (x - m).powi(2)).sum::<f64>())
                        .sqrt()
                })
                .collect();
            println!(
                "  shared coefficients (a_x, a_y, z re, z im, z̄ re, z̄ im): {} ; fold jackknife ± {}",
                full.iter()
                    .map(|v| format!("{v:+.4}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                jk.iter()
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
    }
    // Coefficients of the minimal signed term: per camera, fold stability,
    // and the pupil-centroid offset they imply.
    if let Some(row) = signed1 {
        println!(
            "\n## signed g × deg 0-1 coefficients per camera (a_x, a_y; z re (radial), z im (twist); z̄ re, z̄ im), ± stated σ"
        );
        let names = ["a_x", "a_y", "z re", "z im", "z̄ re", "z̄ im"];
        let mut per: Vec<Vec<f64>> = vec![Vec::new(); 6];
        for (c, s) in rig.cams.iter().zip(&row.full) {
            let txt: Vec<String> = (0..6)
                .map(|k| {
                    let sig = (s.sigma2 * s.cov[(NCAM + k, NCAM + k)]).sqrt();
                    per[k].push(s.local[k]);
                    format!("{:+.4} ± {:.4}", s.local[k], sig)
                })
                .collect();
            println!("  {}: {}", c.id, txt.join(", "));
        }
        for (k, n) in names.iter().enumerate() {
            let (m, sd) = crate::mean_std(&per[k]);
            println!(
                "  {n}: across cameras {m:+.4} ± {sd:.4}; fold jackknife / stated σ {:.2}",
                jack_ratio(&row, k, NCAM)
            );
        }
        let (a, b) = (
            crate::mean_std(&per[0]).0.hypot(crate::mean_std(&per[1]).0),
            crate::mean_std(&per[2]).0,
        );
        println!(
            "  implied pupil-centroid offset (fraction of the pupil radius): constant |a|/2 = {:.3}, radial gradient z re/2 = {:+.3} per unit n (n = 1 at the image's side edge)",
            a / 2.0,
            b / 2.0
        );
    }
    let _ = h;
}
