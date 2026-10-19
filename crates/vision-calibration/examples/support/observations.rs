//! Loads the pre-detected observations committed under
//! `regression/observations/`.
//!
//! Each file holds the calibration input of one public dataset, with the
//! target corners already detected, so an example can run a calibration
//! without the images.

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

/// The path of observation set `id` (e.g. `stereo_rig`).
pub fn path(id: &str) -> PathBuf {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the facade crate lives two levels below the repository root");
    repo_root
        .join("regression/observations")
        .join(format!("{id}.json"))
}

/// The calibration input stored in observation set `id`.
pub fn load_input<T: DeserializeOwned>(id: &str) -> Result<T> {
    let path = path(id);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let mut file: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    serde_json::from_value(file["input"].take())
        .with_context(|| format!("{} does not hold the expected input", path.display()))
}
