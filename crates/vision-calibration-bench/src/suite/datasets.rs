//! The suite's image entries as benchmark datasets.
//!
//! The public datasets of `calib-bench run`, `diagnose` and `feature-cov`
//! are the suite's image entries: each one's dataset manifest and the
//! directory its paths resolve against. The suite is their single
//! definition; the bench registry holds only private datasets.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use vision_calibration_dataset::DatasetSpec;

use super::{Suite, SuiteEntry, fetch};
use crate::registry::{BenchEntry, CrossvalCfg, SpecRef, StabilityCfg, Tier, Visibility};

/// The dataset manifest of image entry `entry` and the directory its paths
/// resolve against: the manifest's own directory, or, with `images`, the
/// entry's directory in that `suite fetch` cache, after checking every file
/// of its fetch list.
///
/// # Errors
///
/// Fails when the entry has no dataset manifest, or when `images` is given
/// for an entry without a fetch source or its cached files are missing or
/// changed.
pub fn image_source(
    suite: &Suite,
    entry: &SuiteEntry,
    images: Option<&Path>,
) -> Result<(PathBuf, PathBuf)> {
    let id = &entry.id;
    let manifest = entry
        .manifest
        .as_ref()
        .with_context(|| format!("entry {id:?} has no dataset manifest"))?;
    let manifest = suite.root.join(manifest);
    let data_root = match images {
        None => manifest.parent().unwrap_or(Path::new(".")).to_path_buf(),
        Some(images) => fetched_dir(suite, entry, images)?,
    };
    Ok((manifest, data_root))
}

/// The directory of entry `entry` in the `suite fetch` cache `images`,
/// after checking every file of its fetch list.
///
/// # Errors
///
/// Fails when the entry has no fetch source, or its cached files are
/// missing or changed.
pub fn fetched_dir(suite: &Suite, entry: &SuiteEntry, images: &Path) -> Result<PathBuf> {
    let id = &entry.id;
    ensure!(
        entry.fetch.is_some(),
        "entry {id:?} cannot be fetched, so its images are not in {}; \
         read it without --images",
        images.display()
    );
    let dir = fetch::entry_dir(images, id);
    fetch::verify(&dir, &fetch::read_fetch_list(suite, id)?)
        .with_context(|| format!("run `suite fetch --only {id}` first"))?;
    Ok(dir)
}

/// The image entries of `suite` named in `only` (all of them when it is
/// empty) as public [`BenchEntry`]s, their images read as [`image_source`]
/// says. With `images`, only the entries that have a fetch source are read
/// from the cache; the others keep their manifest's directory. An id in
/// `only` that is not an image entry is skipped, so `only` may name
/// datasets defined elsewhere.
///
/// # Errors
///
/// Fails on an invalid dataset manifest, on cached files that are missing
/// or changed, and on an entry with `config` overrides: a benchmark run
/// does not apply them, so it would calibrate a different problem than the
/// suite's reference.
pub fn bench_entries(
    suite: &Suite,
    images: Option<&Path>,
    only: &[String],
) -> Result<Vec<BenchEntry>> {
    suite
        .manifest
        .entries
        .iter()
        .filter(|e| e.manifest.is_some() && (only.is_empty() || only.contains(&e.id)))
        .map(|entry| {
            ensure!(
                entry.config.is_none(),
                "suite entry {:?} has config overrides, which benchmark runs do not apply",
                entry.id
            );
            let images = images.filter(|_| entry.fetch.is_some());
            let (manifest, data_root) = image_source(suite, entry, images)?;
            let spec = DatasetSpec::from_path(&manifest)?;
            Ok(bench_entry(&entry.id, spec, data_root))
        })
        .collect()
}

fn bench_entry(id: &str, spec: DatasetSpec, data_root: PathBuf) -> BenchEntry {
    BenchEntry {
        id: id.to_owned(),
        visibility: Visibility::Public,
        tiers: vec![Tier::B],
        problem: spec.topology.into(),
        data_root,
        spec: Some(SpecRef::Inline(Box::new(spec))),
        detector: None,
        laser: None,
        board: None,
        cameras: Vec::new(),
        robot_poses: None,
        prior_export: None,
        fixture: None,
        seed: None,
        device_spec: None,
        accept: None,
        planar_intrinsics: None,
        single_cam_handeye: None,
        rig_handeye: None,
        solver_backend: None,
        stability: StabilityCfg::default(),
        crossval: CrossvalCfg::default(),
        notes: Some("a regression-suite image entry".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ProblemKind;

    fn workspace_suite() -> Suite {
        Suite::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../regression/suite.toml"))
            .unwrap()
    }

    #[test]
    fn every_image_entry_is_a_public_dataset() {
        let suite = workspace_suite();
        let entries = bench_entries(&suite, None, &[]).unwrap();
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids,
            ["kuka_1", "stereo_left", "stereo_right", "stereo_rig", "ds8"]
        );
        let ds8 = &entries[4];
        assert_eq!(ds8.problem, ProblemKind::RigHandeye);
        assert!(ds8.data_root.ends_with("data/DS8"));
        let spec = crate::manifest::dataset_spec(ds8).unwrap();
        assert_eq!(spec.cameras.len(), 3);
        assert!(entries.iter().all(|e| e.visibility == Visibility::Public));

        let only = ["ds8".to_string(), "not_in_the_suite".to_string()];
        let selected = bench_entries(&suite, None, &only).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "ds8");
    }
}
