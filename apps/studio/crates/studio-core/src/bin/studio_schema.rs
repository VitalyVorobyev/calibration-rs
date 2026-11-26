//! `studio-schema [--check]`: write the back end's API schema,
//! `apps/studio/schema/studio-api.json`, from the command table. With
//! `--check` it only compares, and fails when the file is stale, so a Rust
//! type change that was not regenerated fails CI. `bun run generate:api`
//! runs it and then generates the TypeScript from the file.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn out_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schema/studio-api.json")
}

fn main() -> ExitCode {
    let check = std::env::args().skip(1).any(|a| a == "--check");
    let path = out_path();
    let mut text = match serde_json::to_string_pretty(&studio_core::api::Api::schema()) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("studio-schema: render the schema: {e}");
            return ExitCode::FAILURE;
        }
    };
    text.push('\n');
    if check {
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        if on_disk == text {
            println!("schema up to date: {}", path.display());
            return ExitCode::SUCCESS;
        }
        eprintln!("schema drift: {}", path.display());
        eprintln!("run `bun run generate:api` and commit schema/ and src/api/generated/");
        return ExitCode::FAILURE;
    }
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&path, &text));
    match written {
        Ok(()) => {
            println!("wrote {}", path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("studio-schema: write {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}
