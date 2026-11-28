//! The digest of a run's inputs ([`InputDigest`]) and the detection cache
//! that records the keys a run reads ([`RecordingCache`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::Serialize;
use vision_calibration_core::lexical_absolute;
use vision_calibration_dataset::DatasetSpec;
use vision_calibration_detect::{
    CacheError, CacheKey, CachedFeatures, DetectionCache, DetectionDetail,
};

use super::fs::{canonical_json, digest_of, read, sha256};
use super::{Exclusions, InputDigest, ProjectDigest, ProjectError};
use crate::analysis::QualityThresholds;
use crate::dataset_runner::CalibrationConfig;
use crate::device_seed::DeviceSeed;

/// A detection cache that passes every call through and records each key
/// a run looks up or stores: every image the run reads, with its detector
/// and the detector's configuration and version.
pub(crate) struct RecordingCache<'a> {
    inner: &'a dyn DetectionCache,
    keys: Mutex<BTreeSet<String>>,
    /// Each image file read (made absolute lexically) and its content hash.
    images: Mutex<BTreeMap<PathBuf, u64>>,
}

impl<'a> RecordingCache<'a> {
    pub(crate) fn new(inner: &'a dyn DetectionCache) -> Self {
        Self {
            inner,
            keys: Mutex::new(BTreeSet::new()),
            images: Mutex::new(BTreeMap::new()),
        }
    }

    /// The content hash of each image file read, by its lexically absolute
    /// path.
    pub(crate) fn images(&self) -> BTreeMap<PathBuf, u64> {
        match self.images.lock() {
            Ok(images) => images.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    fn record(&self, key: &CacheKey) {
        let mut keys = match self.keys.lock() {
            Ok(keys) => keys,
            Err(poisoned) => poisoned.into_inner(),
        };
        keys.insert(key.file_name());
    }

    /// The keys recorded, sorted.
    pub(crate) fn keys(&self) -> BTreeSet<String> {
        match self.keys.lock() {
            Ok(keys) => keys.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

impl DetectionCache for RecordingCache<'_> {
    fn get(&self, key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
        self.record(key);
        self.inner.get(key)
    }

    fn put(&self, key: &CacheKey, value: &CachedFeatures) -> Result<(), CacheError> {
        self.record(key);
        self.inner.put(key, value)
    }

    // The detail goes under a key `get` or `put` already recorded.
    fn get_detail(&self, key: &CacheKey) -> Result<Option<DetectionDetail>, CacheError> {
        self.inner.get_detail(key)
    }

    fn put_detail(&self, key: &CacheKey, detail: &DetectionDetail) -> Result<(), CacheError> {
        self.inner.put_detail(key, detail)
    }

    fn note_image(&self, image: &Path, key: &CacheKey) {
        let mut images = match self.images.lock() {
            Ok(images) => images,
            Err(poisoned) => poisoned.into_inner(),
        };
        images.insert(lexical_absolute(image), key.image_hash);
        drop(images);
        self.inner.note_image(image, key);
    }
}

/// What a run's inputs digest covers, resolved.
pub(crate) struct DigestInputs<'a> {
    pub(crate) dataset: &'a DatasetSpec,
    pub(crate) data_root: &'a Path,
    pub(crate) exclusions: &'a Exclusions,
    pub(crate) config: &'a CalibrationConfig,
    pub(crate) quality: &'a QualityThresholds,
    pub(crate) detections: &'a BTreeSet<String>,
    pub(crate) seed: Option<&'a DeviceSeed>,
}

impl DigestInputs<'_> {
    /// The digest. The data root's location enters none of it: only the
    /// content of the files the manifest names other than images, keyed by
    /// their paths as the manifest writes them.
    pub(crate) fn digest(&self) -> Result<InputDigest, ProjectError> {
        let ProjectDigest {
            dataset,
            exclusions,
            config,
            quality,
            seed,
        } = project_digest(
            self.dataset,
            self.data_root,
            self.exclusions,
            self.config,
            self.quality,
            self.seed,
        )?;
        let detections = digest_of(self.detections);
        // Without a seed, `inputs` covers the five parts alone, as it did
        // before seeds were recorded.
        let inputs = match &seed {
            Some(seed) => digest_of(&[&dataset, &exclusions, &config, &quality, &detections, seed]),
            None => digest_of(&[&dataset, &exclusions, &config, &quality, &detections]),
        };
        Ok(InputDigest {
            inputs,
            dataset,
            exclusions,
            config,
            quality,
            detections,
            seed,
        })
    }
}

/// The digests of the inputs known before any image is read, computed as a
/// run computes them: every part of an [`InputDigest`] but the detections
/// (and so `inputs`).
pub(crate) fn project_digest(
    dataset: &DatasetSpec,
    data_root: &Path,
    exclusions: &Exclusions,
    config: &CalibrationConfig,
    quality: &QualityThresholds,
    seed: Option<&DeviceSeed>,
) -> Result<ProjectDigest, ProjectError> {
    #[derive(Serialize)]
    struct Dataset<'a> {
        manifest: &'a DatasetSpec,
        /// Path as written → `sha256:<hex>` of the file's content.
        files: std::collections::BTreeMap<String, String>,
    }
    let mut files = std::collections::BTreeMap::new();
    let named = dataset
        .robot_poses
        .as_ref()
        .map(|p| p.path.as_path())
        .into_iter()
        .chain(dataset.upstream_calibration.as_deref());
    for path in named {
        let bytes = read(&data_root.join(path))?;
        files.insert(path.to_string_lossy().replace('\\', "/"), sha256(&bytes));
    }
    Ok(ProjectDigest {
        dataset: digest_of(&Dataset {
            manifest: dataset,
            files,
        }),
        exclusions: digest_of(&normalized(exclusions)),
        config: digest_of(config),
        quality: digest_of(quality),
        seed: seed.map(digest_of),
    })
}

/// The exclusions as a set: notes left out, each list's entries as their
/// canonical JSON, sorted and without repeats, a feature entry's points
/// sorted and without repeats.
fn normalized(exclusions: &Exclusions) -> serde_json::Value {
    fn canonical(value: &impl Serialize) -> String {
        // Exclusion entries are strings, paths and numbers.
        canonical_json(value).expect("an exclusion serializes to JSON")
    }
    let views: BTreeSet<String> = exclusions
        .views
        .iter()
        .map(|v| canonical(&v.view))
        .collect();
    let images: BTreeSet<String> = exclusions
        .images
        .iter()
        .map(|i| canonical(&(&i.camera, &i.image)))
        .collect();
    let features: BTreeSet<String> = exclusions
        .features
        .iter()
        .map(|f| {
            let mut points = f.points.clone();
            points.sort_by(|a, b| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            points.dedup();
            canonical(&(&f.camera, &f.image, points))
        })
        .collect();
    serde_json::json!({ "views": views, "images": images, "features": features })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{FeatureExclusion, ImageExclusion, ViewExclusion, ViewRef};

    #[test]
    fn the_exclusion_digest_ignores_notes_order_and_repeats() {
        let mut a = Exclusions::default();
        a.views.push(ViewExclusion::new(ViewRef::Token("1".into())));
        a.images.push(ImageExclusion::new("cam0", "a.png"));
        a.images.push(ImageExclusion::new("cam1", "b.png"));
        a.features.push(FeatureExclusion::new(
            "cam0",
            "a.png",
            vec![[0.0, 0.03, 0.0], [0.03, 0.0, 0.0]],
        ));
        let mut b = Exclusions::default();
        let mut image = ImageExclusion::new("cam1", "b.png");
        image.note = Some("blurred".into());
        b.images.push(image);
        b.images.push(ImageExclusion::new("cam0", "a.png"));
        b.images.push(ImageExclusion::new("cam0", "a.png"));
        b.views.push(ViewExclusion::new(ViewRef::Token("1".into())));
        b.features.push(FeatureExclusion::new(
            "cam0",
            "a.png",
            vec![[0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.03, 0.0, 0.0]],
        ));
        assert_eq!(digest_of(&normalized(&a)), digest_of(&normalized(&b)));
        b.images.push(ImageExclusion::new("cam1", "c.png"));
        assert_ne!(digest_of(&normalized(&a)), digest_of(&normalized(&b)));
    }
}
