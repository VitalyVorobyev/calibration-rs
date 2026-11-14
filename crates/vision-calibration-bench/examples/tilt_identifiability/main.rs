//! Identifiability and convergence of the tilted (Scheimpflug) camera fit: a
//! synthetic study over camera, lens, tilt and view design.
//!
//! `--parts` picks what runs (all by default), each printing Markdown
//! tables:
//!
//! - `analytic`: the exact factorization `K·H(τ) = K'(τ)·R(τ)` and the
//!   decentering expansion, checked numerically through the library solver
//!   (an untilted camera with free tangential terms fitted to tilted data);
//! - `fisher`: the Fisher information at the truth over the sweep (σ,
//!   correlations, conditioning; free sets and parameterizations), from a
//!   central-difference Jacobian of the library's camera (`model.rs`);
//! - `mc`: Monte-Carlo through `optimize_planar_intrinsics` on both
//!   backends: bias, scatter against the stated σ, and the policy cells
//!   (true tangential terms held or freed);
//! - `basin`: convergence from perturbed starts: the pipeline's seeded and
//!   cold routes on both backends, the direct joint solve, and the four
//!   parameterizations through the library's solver (`solve.rs`), with the
//!   failure modes and whether the result flags them;
//! - `solver`: the stopping rule against a tight one, and the Jacobi
//!   scaling's drift from the start to the solution;
//! - `determine`: whether the tilt is determined (the decision statistic
//!   against its labels), what fixing it at zero costs downstream with the
//!   tangential terms held or freed, and the frame each fit ends in
//!   (`determine.rs`, `downstream.rs`, `verdict.rs`);
//! - `prior`: a Gaussian prior on the tilt, right or wrong, with the
//!   tangential terms held or freed (`procedure.rs`);
//! - `rule`: the determination procedure end to end, from the plain camera
//!   through the pipeline to the decision (`procedure.rs`).
//!
//! Findings: `docs/notes/tilt-identifiability.md` (the first five parts)
//! and `docs/notes/tilt-determination.md` (the last three).
//!
//! ```text
//! cargo run --release -p vision-calibration-bench --example tilt_identifiability -- \
//!     [--parts analytic,fisher,mc,basin,solver,determine,prior,rule] [--quick] [--dump draws.tsv]
//! ```

mod determine;
mod downstream;
mod model;
mod procedure;
mod scene;
mod solve;
mod study;
mod verdict;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Synthetic identifiability study of the tilted camera fit")]
struct Args {
    /// Comma-separated parts: analytic, fisher, mc, basin, solver,
    /// determine, prior, rule.
    #[arg(long, default_value = "analytic,fisher,mc,basin,solver")]
    parts: String,
    /// Fewer Monte-Carlo draws and starts (a smoke run).
    #[arg(long)]
    quick: bool,
    /// Write the determination sweep's draws to this tab-separated file.
    #[arg(long)]
    dump: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let parts: Vec<&str> = args.parts.split(',').map(str::trim).collect();
    let size = if args.quick {
        study::Size::quick()
    } else {
        study::Size::full()
    };
    for part in parts {
        let started = std::time::Instant::now();
        match part {
            "analytic" => study::analytic()?,
            "fisher" => study::fisher_tables()?,
            "mc" => study::monte_carlo(&size)?,
            "basin" => study::basin(&size)?,
            "solver" => study::solver(&size)?,
            "determine" => verdict::sweep(&size, args.dump.as_deref())?,
            "prior" => procedure::prior_study(&size)?,
            "rule" => procedure::rule_study(&size)?,
            other => anyhow::bail!("unknown part `{other}`"),
        }
        eprintln!("[{part}: {:.1} s]", started.elapsed().as_secs_f64());
    }
    Ok(())
}
