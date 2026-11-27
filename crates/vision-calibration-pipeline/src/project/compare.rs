//! Runs compared through their records: [`Project::compare_runs`] and
//! [`compare_recorded_runs`], the data relation of each pair read from the
//! content of the images each run used.

use std::collections::BTreeSet;

use vision_calibration_core::FrameKind;

use super::{CalibratedRun, Exclusions, Project, ProjectError, RunInfo, RunSource, RunState};
use crate::Error;
use crate::analysis::{
    CompareRunsOptions, ComparedRun, DataOverlap, DataRelation, ExclusionChanges, RunComparison,
    RunSide, compare_runs,
};
use crate::common::{CalibratedCameras, calibrated_cameras};

impl Project {
    /// The project's runs `ids` side by side against the first
    /// ([`compare_recorded_runs`]).
    ///
    /// # Errors
    ///
    /// [`ProjectError::RunNotFound`] for an unknown id; the errors of
    /// [`compare_recorded_runs`].
    pub fn compare_runs(
        &self,
        ids: &[&str],
        options: &CompareRunsOptions,
    ) -> Result<RunComparison, ProjectError> {
        let runs = ids
            .iter()
            .map(|id| self.run(id))
            .collect::<Result<Vec<_>, _>>()?;
        compare_recorded_runs(&runs, options)
    }
}

/// Recorded runs side by side against the first, the reference
/// ([`compare_runs`](crate::analysis::compare_runs)), each named by its id;
/// the runs may belong to different projects.
///
/// Each run's camera ids and image sizes come from its record (the image
/// sizes from its images when the record has none), its configuration from
/// the record of a calibrated run. The data relation of each run to the
/// reference is read from the records, never given:
///
/// - **Independent** when the two used no image of the same content;
/// - otherwise by their features, each one target point (or laser pixel)
///   of one camera in one image, identified by the image's content and the
///   observed pixel: **Identical** for equal input digests; the same
///   features under one estimator (equal dataset and configuration
///   digests) is **Identical**, under another **SameData**; one run's
///   features a strict subset of the other's under one estimator,
///   **Nested**; anything else **Overlapping**;
/// - **Unknown** for a run that did not record the content of every image
///   it used (an imported or earlier run), which never yields a test.
///
/// Each difference also carries the [`DataOverlap`] behind the relation and,
/// between two calibrated runs, the [`ExclusionChanges`].
///
/// # Errors
///
/// [`ProjectError::Compare`] when `options` states relations (the records
/// decide them), a run's summary names no cameras, a camera's image size is
/// unknown, or the comparison's own checks fail;
/// [`ProjectError::Io`] and [`ProjectError::Parse`] when an export cannot
/// be read.
pub fn compare_recorded_runs(
    runs: &[RunInfo],
    options: &CompareRunsOptions,
) -> Result<RunComparison, ProjectError> {
    if !options.relations.is_empty() {
        return Err(ProjectError::Compare(Error::invalid_input(
            "the relations of recorded runs are read from their records; \
             leave CompareRunsOptions::relations empty",
        )));
    }
    let loaded = runs.iter().map(load).collect::<Result<Vec<_>, _>>()?;
    let reference = &loaded[..1];
    let mut relations = Vec::new();
    let mut overlaps = Vec::new();
    let mut exclusions = Vec::new();
    for other in &loaded[1..] {
        let (relation, overlap) = relation(&reference[0], other);
        relations.push(relation);
        overlaps.push(overlap);
        exclusions.push(exclusion_changes(&reference[0], other));
    }
    let compared: Vec<ComparedRun<'_>> = loaded
        .iter()
        .map(|run| {
            let compared = ComparedRun::new(
                run.id.clone(),
                run.export.as_ref(),
                run.camera_ids.clone(),
                run.image_sizes.clone(),
            );
            match run.calibrated.as_ref() {
                Some(c) => compared.with_config(c.inputs.config.clone()),
                None => compared,
            }
        })
        .collect();
    let mut comparison = compare_runs(&compared, &options.clone().with_relations(relations))
        .map_err(ProjectError::Compare)?;
    for ((difference, overlap), changes) in comparison
        .differences
        .iter_mut()
        .zip(overlaps)
        .zip(exclusions)
    {
        difference.data = overlap;
        difference.exclusion_changes = changes;
    }
    Ok(comparison)
}

/// A run as the comparison reads it.
struct Loaded {
    id: String,
    export: Box<dyn CalibratedCameras>,
    camera_ids: Vec<String>,
    image_sizes: Vec<[u32; 2]>,
    calibrated: Option<CalibratedRun>,
    /// The content of every image used and every feature, when the record
    /// has the content of each image the export's manifest names.
    observations: Option<Observations>,
}

/// What a run observed, by content.
struct Observations {
    images: BTreeSet<String>,
    features: BTreeSet<Feature>,
}

/// A feature: the image's content, the camera, target or laser, the target
/// point and the observed pixel (exact bits: equal detections of equal
/// images give equal values).
type Feature = (String, String, u8, [u64; 3], [u64; 2]);

fn load(run: &RunInfo) -> Result<Loaded, ProjectError> {
    let path = run.export_path();
    let text = std::fs::read_to_string(&path).map_err(|e| ProjectError::io(&path, e))?;
    let export = calibrated_cameras(&text).map_err(|e| ProjectError::parse(&path, e))?;
    let cameras = export.camera_params().len();
    let invalid = |message: String| ProjectError::Compare(Error::invalid_input(message));
    let summary = run
        .summary()
        .filter(|s| s.cameras.len() == cameras)
        .ok_or_else(|| invalid(format!("run {}: its record names no cameras", run.id)))?;
    let camera_ids: Vec<String> = summary.cameras.iter().map(|c| c.id.clone()).collect();
    let image_sizes = super::run::recorded_sizes(&summary.cameras, export.as_ref(), &run.dir)
        .ok_or_else(|| {
            invalid(format!(
                "run {}: the image size of its cameras is not recorded and its images cannot \
                 be read",
                run.id
            ))
        })?;
    let calibrated = match &run.state {
        RunState::Recorded(record) => match &record.source {
            RunSource::Calibrated(c) => Some((**c).clone()),
            _ => None,
        },
        _ => None,
    };
    let observations = calibrated
        .as_ref()
        .and_then(|c| observations(export.as_ref(), c, &camera_ids));
    Ok(Loaded {
        id: run.id.clone(),
        export,
        camera_ids,
        image_sizes,
        calibrated,
        observations,
    })
}

/// The run's images and features by content; `None` unless the record
/// has the content of every frame of the export's image manifest.
fn observations(
    export: &dyn CalibratedCameras,
    run: &CalibratedRun,
    ids: &[String],
) -> Option<Observations> {
    let manifest = export.diagnostics().image_manifest.as_ref()?;
    if run.images.len() != manifest.frames.len() || run.images.is_empty() {
        return None;
    }
    let content = |view: usize, camera: usize, kind: FrameKind| -> Option<String> {
        let i = manifest
            .frames
            .iter()
            .position(|f| f.view == view && f.camera == camera && f.kind == kind)?;
        Some(run.images[i].content.clone())
    };
    let camera = |c: usize| ids.get(c).cloned().unwrap_or_else(|| format!("cam{c}"));
    let residuals = &export.diagnostics().per_feature_residuals;
    let mut features = BTreeSet::new();
    for r in &residuals.target {
        features.insert((
            content(r.view, r.camera, FrameKind::Target)?,
            camera(r.camera),
            0,
            r.target_xyz_m.map(f64::to_bits),
            r.observed_px.map(f64::to_bits),
        ));
    }
    for r in &residuals.laser {
        features.insert((
            content(r.view, r.camera, FrameKind::Laser)?,
            camera(r.camera),
            1,
            [0; 3],
            r.observed_px.map(f64::to_bits),
        ));
    }
    Some(Observations {
        images: run.images.iter().map(|i| i.content.clone()).collect(),
        features,
    })
}

/// The relation of `other` to `reference` and the overlap behind it.
fn relation(reference: &Loaded, other: &Loaded) -> (DataRelation, Option<DataOverlap>) {
    let (Some(a), Some(b), Some(ca), Some(cb)) = (
        &reference.observations,
        &other.observations,
        &reference.calibrated,
        &other.calibrated,
    ) else {
        return (DataRelation::Unknown, None);
    };
    let shared_images = a.images.intersection(&b.images).count();
    let shared_features = a.features.intersection(&b.features).count();
    let overlap = DataOverlap {
        shared_images,
        reference_only_images: a.images.len() - shared_images,
        other_only_images: b.images.len() - shared_images,
        shared_features,
        reference_only_features: a.features.len() - shared_features,
        other_only_features: b.features.len() - shared_features,
    };
    let same_estimator =
        ca.digest.config == cb.digest.config && ca.digest.dataset == cb.digest.dataset;
    let relation = if shared_images == 0 {
        DataRelation::Independent
    } else if ca.digest.inputs == cb.digest.inputs {
        DataRelation::Identical
    } else if a.features == b.features {
        if same_estimator {
            DataRelation::Identical
        } else {
            DataRelation::SameData
        }
    } else if same_estimator && b.features.is_subset(&a.features) {
        DataRelation::Nested {
            smaller: RunSide::Other,
        }
    } else if same_estimator && a.features.is_subset(&b.features) {
        DataRelation::Nested {
            smaller: RunSide::Reference,
        }
    } else {
        DataRelation::Overlapping
    };
    (relation, Some(overlap))
}

/// The exclusions only one of two calibrated runs made, notes left out.
fn exclusion_changes(reference: &Loaded, other: &Loaded) -> Option<ExclusionChanges> {
    let a = stripped(&reference.calibrated.as_ref()?.inputs.exclusions);
    let b = stripped(&other.calibrated.as_ref()?.inputs.exclusions);
    let only = |x: &Exclusions, y: &Exclusions| Exclusions {
        views: x
            .views
            .iter()
            .filter(|v| !y.views.contains(v))
            .cloned()
            .collect(),
        images: x
            .images
            .iter()
            .filter(|v| !y.images.contains(v))
            .cloned()
            .collect(),
        features: x
            .features
            .iter()
            .filter(|v| !y.features.contains(v))
            .cloned()
            .collect(),
    };
    Some(ExclusionChanges {
        reference_only: only(&a, &b),
        other_only: only(&b, &a),
    })
}

/// The exclusions without their notes, which the calibration does not read.
fn stripped(exclusions: &Exclusions) -> Exclusions {
    let mut out = exclusions.clone();
    for v in &mut out.views {
        v.note = None;
    }
    for i in &mut out.images {
        i.note = None;
    }
    for f in &mut out.features {
        f.note = None;
    }
    out
}

#[cfg(test)]
mod tests;
