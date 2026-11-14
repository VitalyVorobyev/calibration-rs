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
//!   scaling's drift from the start to the solution.
//!
//! Findings: `docs/notes/tilt-identifiability.md`.
//!
//! ```text
//! cargo run --release -p vision-calibration-bench --example tilt_identifiability -- \
//!     [--parts analytic,fisher,mc,basin,solver] [--quick]
//! ```

mod model;
mod scene;
mod solve;
mod study;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Synthetic identifiability study of the tilted camera fit")]
struct Args {
    /// Comma-separated parts: analytic, fisher, mc, basin, solver.
    #[arg(long, default_value = "analytic,fisher,mc,basin,solver")]
    parts: String,
    /// Fewer Monte-Carlo draws and starts (a smoke run).
    #[arg(long)]
    quick: bool,
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
            other => anyhow::bail!("unknown part `{other}`"),
        }
        eprintln!("[{part}: {:.1} s]", started.elapsed().as_secs_f64());
    }
    Ok(())
}
