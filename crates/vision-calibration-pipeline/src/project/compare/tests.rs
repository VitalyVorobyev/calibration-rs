//! The data relation of recorded runs, read from their records, on the
//! synthetic planar project of the run tests.

use std::path::Path;

use vision_calibration_core::RunControl;
use vision_calibration_detect::{
    CacheKey, ChessboardConfig, DetectionCache, DetectorConfig, FsDetectionCache,
};

use super::*;
use crate::analysis::{DifferenceTest, NoTest, ParameterComparison};
use crate::dataset_runner::ConfigPatch;
use crate::project::run::tests::{calibrate, calibrated, options, planar_project, write_image};
use crate::project::{DETECTIONS_DIR, ImageExclusion, RunNotes};

fn compare(project: &Project, ids: &[&str]) -> RunComparison {
    project
        .compare_runs(ids, &CompareRunsOptions::new())
        .expect("comparison")
}

#[test]
fn a_run_records_the_content_of_each_image_it_used() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let run = calibrate(&project);
    let images = &calibrated(&run).images;
    // Seven views: image 3's detection is too short and left out.
    assert_eq!(images.len(), 7);
    assert!(
        images
            .iter()
            .all(|i| i.camera == "cam0" && i.content.len() == 16)
    );
    assert_eq!(images[0].image, Path::new("img/view00.png"));
    let bytes = std::fs::read(tmp.path().join("data/img/view00.png")).unwrap();
    let config = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
    let key = CacheKey::for_detection(&bytes, &config, None);
    assert_eq!(images[0].content, format!("{:016x}", key.image_hash));
    let summary = &record_of(&run).summary.cameras[0];
    assert_eq!(summary.image_size, Some([640, 480]));
}

fn record_of(run: &RunInfo) -> &crate::project::RunRecord {
    run.record().expect("record")
}

#[test]
fn equal_inputs_are_identical_with_no_difference() {
    let tmp = tempfile::tempdir().unwrap();
    let project = planar_project(tmp.path());
    let (a, b) = (calibrate(&project), calibrate(&project));
    let comparison = compare(&project, &[&a.id, &b.id]);
    let d = &comparison.differences[0];
    assert_eq!(d.relation, DataRelation::Identical);
    let overlap = d.data.expect("overlap");
    assert_eq!(overlap.shared_images, 7);
    assert_eq!(
        overlap.reference_only_features + overlap.other_only_features,
        0
    );
    let ParameterComparison::SameModel { parameters, .. } = &d.cameras[0].parameters else {
        panic!("same model");
    };
    assert!(parameters.iter().all(|p| p.delta == 0.0));
    assert!(parameters.iter().all(|p| p.test
        == DifferenceTest::NotAvailable {
            reason: NoTest::Identical
        }));
    let changes = d.exclusion_changes.as_ref().expect("both calibrated");
    assert!(changes.reference_only.is_empty() && changes.other_only.is_empty());
    assert_eq!(d.config_changes, Some(Vec::new()));
}

#[test]
fn an_exclusion_nests_and_another_configuration_shares_data() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let full = calibrate(&project);
    let mut other_config = options();
    other_config.config_patch = ConfigPatch::from_toml_str("solver.max_iters = 77").unwrap();
    let reconfigured = project
        .calibrate(&other_config, None, &RunControl::default())
        .unwrap();
    project
        .spec_mut()
        .exclude
        .images
        .push(ImageExclusion::new("cam0", "img/view01.png"));
    let fewer = calibrate(&project);
    let both = project
        .calibrate(&other_config, None, &RunControl::default())
        .unwrap();

    let comparison = compare(&project, &[&full.id, &fewer.id, &reconfigured.id, &both.id]);
    let [nested, same, overlapping] = &comparison.differences[..] else {
        panic!("three differences");
    };
    assert_eq!(
        nested.relation,
        DataRelation::Nested {
            smaller: RunSide::Other
        }
    );
    let overlap = nested.data.expect("overlap");
    assert_eq!(
        (overlap.shared_images, overlap.reference_only_images),
        (6, 1)
    );
    assert_eq!(overlap.other_only_features, 0);
    let changes = nested.exclusion_changes.as_ref().expect("changes");
    assert_eq!(changes.other_only.images.len(), 1);
    assert!(changes.reference_only.is_empty());

    assert_eq!(same.relation, DataRelation::SameData);
    let config = same.config_changes.as_ref().expect("both configured");
    assert_eq!(config.len(), 1);
    assert!(config[0].path.ends_with("solver.max_iters"), "{config:?}");

    // Fewer images under another configuration: shared, not nested.
    assert_eq!(overlapping.relation, DataRelation::Overlapping);
}

#[test]
fn disjoint_captures_are_independent_and_imports_unknown() {
    let tmp = tempfile::tempdir().unwrap();
    let first = planar_project(&tmp.path().join("first"));
    let second_dir = tmp.path().join("second");
    let second = planar_project(&second_dir);
    // Another capture: every image of the second project differs in content
    // and its detections by noise.
    let cache = FsDetectionCache::new(second_dir.join(DETECTIONS_DIR));
    let config = DetectorConfig::Chessboard(ChessboardConfig::new(6, 8, 0.03));
    let data = second_dir.join("data");
    for i in 0..8u32 {
        let rel = format!("img/view{i:02}.png");
        let old = CacheKey::for_detection(&std::fs::read(data.join(&rel)).unwrap(), &config, None);
        // Another capture's detections: the same corners a micropixel off
        // (bit-identical detections would be the same data).
        let mut features = cache.get(&old).unwrap().unwrap();
        for f in &mut features.features {
            f.image_xy[0] += 1e-6;
        }
        write_image(&data, &rel, 1000 + i);
        let new = CacheKey::for_detection(&std::fs::read(data.join(&rel)).unwrap(), &config, None);
        cache.put(&new, &features).unwrap();
    }
    let a = calibrate(&first);
    let b = calibrate(&second);
    let imported = first
        .import_export(b.export_path(), RunNotes::new())
        .unwrap();

    let comparison = compare_recorded_runs(&[a.clone(), b, imported], &CompareRunsOptions::new())
        .expect("comparison");
    let [independent, unknown] = &comparison.differences[..] else {
        panic!("two differences");
    };
    assert_eq!(independent.relation, DataRelation::Independent);
    assert_eq!(independent.data.expect("overlap").shared_images, 0);
    assert_eq!(unknown.relation, DataRelation::Unknown);
    assert_eq!(unknown.data, None);
    assert_eq!(unknown.exclusion_changes, None);
    assert_eq!(comparison.runs[2].cameras[0].image_size, [640, 480]);

    assert!(matches!(
        first.compare_runs(&[&a.id, "run-404"], &CompareRunsOptions::new()),
        Err(ProjectError::RunNotFound { .. })
    ));
}

/// A run seeded from another reaches the same optimum on the same data, but
/// a seed can bound or hold what the solve estimates: the two are the same
/// data under different estimators, the seed listed among the configuration
/// changes, and no test between them. A seeded run with an image fewer does
/// not nest in an unseeded one.
#[test]
fn a_seed_is_part_of_the_estimator() {
    let tmp = tempfile::tempdir().unwrap();
    let mut project = planar_project(tmp.path());
    let first = calibrate(&project);
    project.spec_mut().seed = Some(crate::project::SeedSource::Run("runs/run-001".into()));
    project.save().unwrap();
    let seeded = calibrate(&project);
    project
        .spec_mut()
        .exclude
        .images
        .push(ImageExclusion::new("cam0", "img/view01.png"));
    let fewer = calibrate(&project);

    let comparison = compare(&project, &[&first.id, &seeded.id, &fewer.id]);
    let [same, shared] = &comparison.differences[..] else {
        panic!("two differences");
    };
    assert_eq!(same.relation, DataRelation::SameData);
    let overlap = same.data.expect("overlap");
    assert_eq!(
        overlap.reference_only_features + overlap.other_only_features,
        0
    );
    let changes = same.config_changes.as_ref().expect("both configured");
    let paths: Vec<&str> = changes.iter().map(|c| c.path.as_str()).collect();
    assert_eq!(paths, ["seed.source", "seed.values"]);
    assert_eq!(changes[0].reference, None);
    assert_eq!(
        changes[0].other.as_deref(),
        Some(r#"{"run":"runs/run-001"}"#)
    );
    let ParameterComparison::SameModel { parameters, .. } = &same.cameras[0].parameters else {
        panic!("same model");
    };
    let mut largest = 0.0_f64;
    for p in parameters {
        assert_eq!(
            p.test,
            DifferenceTest::NotAvailable {
                reason: NoTest::SharedData
            }
        );
        // Fixed parameters have no sigma and no difference.
        match p.delta_over_reference_sigma {
            Some(materiality) => largest = largest.max(materiality.abs()),
            None => assert_eq!(p.delta, 0.0, "{}", p.label),
        }
    }
    // The same optimum (on this noise-free project, the same values to the
    // bit): the seed moved the start, not the result.
    assert!(largest < 1e-3, "{largest}");
    assert_eq!(shared.relation, DataRelation::Overlapping);
}
