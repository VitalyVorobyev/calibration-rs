//! Workspace task runner.
//!
//! - `emit-schemas [--check]` writes JSON Schemas for every `*Config`,
//!   `*Input`, and `DatasetSpec` type into `app/src/schemas/`; the Tauri app
//!   reads them at build time to drive its schema-driven config forms.
//! - `check-docs` fails when user-facing docs cite internal material or pin
//!   a stale version (see `check_docs`).

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

mod check_docs;
mod emit_schemas;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().context(
        "usage: cargo xtask <command>\n\ncommands:\n  emit-schemas [--check]\n  check-docs\n",
    )?;

    match cmd.as_str() {
        "emit-schemas" => {
            let check = args.any(|a| a == "--check");
            emit_schemas::run(&workspace_root()?, check)
        }
        "check-docs" => check_docs::run(&workspace_root()?),
        other => bail!("unknown xtask `{other}`; available: emit-schemas [--check], check-docs"),
    }
}

fn workspace_root() -> Result<PathBuf> {
    // CARGO_MANIFEST_DIR points at xtask/, so the workspace root is its parent.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .context("CARGO_MANIFEST_DIR is unset; xtask must be run via `cargo xtask`")?;
    let root = Path::new(&manifest_dir)
        .parent()
        .context("xtask/ has no parent directory")?
        .to_path_buf();
    Ok(root)
}
