//! What a project leaves out of its dataset: whole views, one camera's
//! image of a view, or single target points of a detection.
//!
//! Every exclusion names what it leaves out by what does not change when
//! the dataset grows: a view by its pairing token or one of its images, an
//! image by its camera and path, a feature by the target point the detector
//! labelled it with. None of them is an index.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vision_calibration_core::lexical_absolute;
use vision_calibration_dataset::{DatasetSpec, PosePairing};

use crate::dataset_runner::{ConvertOptions, RunError, pair_dataset_views};

/// What a project leaves out of its dataset.
///
/// Image paths are relative to the dataset's folder unless absolute, and
/// `/`-separated; they match the images the manifest lists after both are
/// made absolute and their `.` and `..` folded.
///
/// In TOML:
///
/// ```toml
/// [exclude]
/// views  = [{ view = { token = "0012" }, note = "robot pose glitch" }]
/// images = [{ camera = "cam1", image = "cam1/0003.png" }]
/// [[exclude.features]]
/// camera = "cam0"
/// image  = "cam0/0005.png"
/// points = [[0.03, 0.06, 0.0]]
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Exclusions {
    /// Whole views: every camera's image of the view, its robot pose and
    /// its laser frames.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<ViewExclusion>,
    /// One camera's image of a view; the view keeps its other cameras.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageExclusion>,
    /// Single features of one camera's detection in one image.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<FeatureExclusion>,
}

impl Exclusions {
    /// Whether nothing is left out.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty() && self.images.is_empty() && self.features.is_empty()
    }
}

/// A view left out, with every camera's image of it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ViewExclusion {
    /// Which view.
    pub view: ViewRef,
    /// Why, for the reader; the calibration does not read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ViewExclusion {
    /// Leave out `view`, without a note.
    pub fn new(view: ViewRef) -> Self {
        Self { view, note: None }
    }
}

/// How an exclusion names a view.
///
/// A view has no index that survives an image added to the dataset; its
/// pairing token and its images do. Serialized as `{ token = "0012" }` or
/// `{ image = "cam0/0012.png" }`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ViewRef {
    /// The view's pairing token, under `shared_filename_token` pairing.
    Token(String),
    /// Any one of the view's images.
    Image(PathBuf),
}

/// One camera's image of a view, left out.
///
/// The camera is named with the image because a tiled image holds several
/// cameras' views.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ImageExclusion {
    /// The camera's id in the manifest.
    pub camera: String,
    /// The image, one of the camera's.
    pub image: PathBuf,
    /// Why, for the reader; the calibration does not read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ImageExclusion {
    /// Leave out camera `camera`'s image `image`, without a note.
    pub fn new(camera: impl Into<String>, image: impl Into<PathBuf>) -> Self {
        Self {
            camera: camera.into(),
            image: image.into(),
            note: None,
        }
    }
}

/// Features of one camera's detection in one image, left out.
///
/// A feature is named by its target point as the detector labelled it,
/// before a rig turns a plain chessboard's labels to agree with its other
/// cameras: the detection a review shows. A point matches a feature within
/// a micrometre in each coordinate. A point that matches no feature of the
/// detection leaves nothing out and is not an error, since what the
/// detector finds is known only once it has run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct FeatureExclusion {
    /// The camera's id in the manifest.
    pub camera: String,
    /// The image, one of the camera's.
    pub image: PathBuf,
    /// The target points of the features to leave out, metres.
    pub points: Vec<[f64; 3]>,
    /// Why, for the reader; the calibration does not read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl FeatureExclusion {
    /// Leave out camera `camera`'s features at `points` in image `image`,
    /// without a note.
    pub fn new(
        camera: impl Into<String>,
        image: impl Into<PathBuf>,
        points: Vec<[f64; 3]>,
    ) -> Self {
        Self {
            camera: camera.into(),
            image: image.into(),
            points,
            note: None,
        }
    }
}

/// Why an exclusion names nothing in the dataset. A stale exclusion is an
/// error rather than one that silently leaves nothing out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ExclusionFault {
    /// The manifest has no camera of that id.
    #[error("the manifest has no camera of that id")]
    UnknownCamera,
    /// The image is not among the camera's images (or, for a view, any
    /// camera's).
    #[error("the image is not among the camera's images")]
    ImageNotFound,
    /// No view has that pairing token.
    #[error("no view has that pairing token")]
    TokenNotFound,
    /// The manifest does not pair its views by a filename token.
    #[error("the manifest does not pair its views by a filename token")]
    NoTokenPairing,
    /// The image belongs to more than one view.
    #[error("the image belongs to more than one view")]
    AmbiguousImage,
}

/// Why exclusions do not resolve against a dataset.
#[derive(Debug, Error)]
pub(crate) enum ResolveError {
    /// The dataset's images could not be listed or paired.
    #[error(transparent)]
    Run(#[from] RunError),
    /// An exclusion names something the dataset does not have.
    #[error("{entry}: {fault}")]
    Exclusion {
        /// The exclusion, as `exclude.<list>[<index>]`.
        entry: String,
        /// What is wrong with it.
        fault: ExclusionFault,
    },
}

impl Exclusions {
    /// The conversion options that leave out what these exclusions name in
    /// the dataset `spec` describes, whose relative paths resolve against
    /// `data_root`.
    ///
    /// Only the image lists are read, never an image: a stale exclusion
    /// fails before any detection. No exclusions resolve to the default
    /// options without listing anything.
    pub(crate) fn resolve(
        &self,
        spec: &DatasetSpec,
        data_root: &Path,
    ) -> Result<ConvertOptions, ResolveError> {
        let mut options = ConvertOptions::default();
        if self.is_empty() {
            return Ok(options);
        }
        let paired = pair_dataset_views(spec, data_root)?;
        // slots[view][camera]: each paired image, as the converters will
        // compare it.
        let slots: Vec<Vec<Option<PathBuf>>> = paired
            .paths
            .iter()
            .map(|cameras| {
                cameras
                    .iter()
                    .map(|path| path.as_deref().map(lexical_absolute))
                    .collect()
            })
            .collect();
        let token_pairing = matches!(
            spec.pose_pairing,
            Some(PosePairing::SharedFilenameToken { .. })
        );
        let resolve = |image: &Path| lexical_absolute(&data_root.join(image));
        let fault = |list: &str, index: usize, fault| ResolveError::Exclusion {
            entry: format!("exclude.{list}[{index}]"),
            fault,
        };
        // A camera's slot of `image`, checked to be one of its images.
        let slot = |list: &str, index: usize, camera: &str, image: &Path| {
            let camera = spec
                .cameras
                .iter()
                .position(|c| c.id == camera)
                .ok_or_else(|| fault(list, index, ExclusionFault::UnknownCamera))?;
            let image = resolve(image);
            if slots
                .iter()
                .any(|views| views[camera].as_ref() == Some(&image))
            {
                Ok((camera, image))
            } else {
                Err(fault(list, index, ExclusionFault::ImageNotFound))
            }
        };

        for (index, exclusion) in self.views.iter().enumerate() {
            let view = match &exclusion.view {
                ViewRef::Token(token) => {
                    if !token_pairing {
                        return Err(fault("views", index, ExclusionFault::NoTokenPairing));
                    }
                    paired
                        .tokens
                        .iter()
                        .position(|t| t == token)
                        .ok_or_else(|| fault("views", index, ExclusionFault::TokenNotFound))?
                }
                ViewRef::Image(image) => {
                    let image = resolve(image);
                    let views: BTreeSet<usize> = slots
                        .iter()
                        .enumerate()
                        .filter(|(_, cameras)| cameras.iter().flatten().any(|p| *p == image))
                        .map(|(view, _)| view)
                        .collect();
                    match (views.len(), views.first()) {
                        (1, Some(&view)) => view,
                        (0, _) => return Err(fault("views", index, ExclusionFault::ImageNotFound)),
                        _ => return Err(fault("views", index, ExclusionFault::AmbiguousImage)),
                    }
                }
            };
            for (camera, path) in slots[view].iter().enumerate() {
                if let Some(path) = path {
                    options.exclude_slot(camera, path);
                }
            }
        }
        for (index, exclusion) in self.images.iter().enumerate() {
            let (camera, image) = slot("images", index, &exclusion.camera, &exclusion.image)?;
            options.exclude_slot(camera, &image);
        }
        for (index, exclusion) in self.features.iter().enumerate() {
            let (camera, image) = slot("features", index, &exclusion.camera, &exclusion.image)?;
            options.mask_points(camera, &image, &exclusion.points);
        }
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A two-camera rig paired by filename token: views `0001` to `0003`,
    /// camera 1 missing view `0002`. The images are files that are not
    /// images: resolving reads none of them.
    fn token_rig(dir: &Path) -> DatasetSpec {
        for (cam, views) in [(0, vec![1, 2, 3]), (1, vec![1, 3])] {
            std::fs::create_dir_all(dir.join(format!("cam{cam}"))).unwrap();
            for view in views {
                std::fs::write(dir.join(format!("cam{cam}/img_{view:04}.png")), b"-").unwrap();
            }
        }
        serde_json::from_value(json!({
            "version": 1,
            "topology": "rig_extrinsics",
            "cameras": [
                { "id": "cam0", "images": { "kind": "glob", "pattern": "cam0/*.png" } },
                { "id": "cam1", "images": { "kind": "glob", "pattern": "cam1/*.png" } },
            ],
            "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.03 },
            "pose_pairing": {
                "kind": "shared_filename_token", "regex": "img_(?P<id>\\d+)", "group": "id"
            },
        }))
        .unwrap()
    }

    fn exclusions(value: serde_json::Value) -> Exclusions {
        serde_json::from_value(value).unwrap()
    }

    fn fault(result: Result<ConvertOptions, ResolveError>) -> (String, ExclusionFault) {
        match result {
            Err(ResolveError::Exclusion { entry, fault }) => (entry, fault),
            other => panic!("expected an exclusion fault, got {other:?}"),
        }
    }

    #[test]
    fn no_exclusions_resolve_to_the_default_options_without_listing_images() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = token_rig(tmp.path());
        spec.cameras[0].images = vision_calibration_dataset::ImagePattern::Glob {
            pattern: "[".into(),
        };
        let options = Exclusions::default()
            .resolve(&spec, Path::new("/nowhere"))
            .unwrap();
        assert_eq!(options, ConvertOptions::default());
    }

    #[test]
    fn a_view_is_named_by_its_token_or_any_of_its_images() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let spec = token_rig(dir);
        for view in [
            json!({ "token": "0003" }),
            json!({ "image": "cam1/img_0003.png" }),
        ] {
            let options = exclusions(json!({ "views": [{ "view": view }] }))
                .resolve(&spec, dir)
                .unwrap();
            assert!(options.excludes(0, &dir.join("cam0/img_0003.png")));
            assert!(options.excludes(1, &dir.join("cam1/img_0003.png")));
            assert!(!options.excludes(0, &dir.join("cam0/img_0001.png")));
        }
        // A view camera 1 has no image of: camera 0's is excluded.
        let options = exclusions(json!({ "views": [{ "view": { "token": "0002" } }] }))
            .resolve(&spec, dir)
            .unwrap();
        assert!(options.excludes(0, &dir.join("cam0/img_0002.png")));
    }

    #[test]
    fn an_image_or_its_features_name_one_camera_s_slot() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let spec = token_rig(dir);
        let absolute = dir.join("cam0/img_0001.png");
        let options = exclusions(json!({
            "images": [{ "camera": "cam1", "image": "cam1/./img_0001.png" }],
            "features": [{ "camera": "cam0", "image": absolute, "points": [[0.03, 0.0, 0.0]] }],
        }))
        .resolve(&spec, dir)
        .unwrap();
        assert!(options.excludes(1, &dir.join("cam1/img_0001.png")));
        assert!(!options.excludes(0, &absolute));
        let features = [
            vision_calibration_detect::Feature::new([0.0, 0.0], [0.0, 0.0, 0.0]),
            vision_calibration_detect::Feature::new([1.0, 0.0], [0.03, 0.0, 0.0]),
        ];
        assert_eq!(
            options.kept_features(0, &absolute, &features),
            Some(vec![true, false])
        );
    }

    /// A stale exclusion fails before any image is read, naming its entry.
    #[test]
    fn a_stale_exclusion_is_a_typed_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        let spec = token_rig(dir);
        let resolve = |value| exclusions(value).resolve(&spec, dir);
        assert_eq!(
            fault(resolve(json!({ "images": [
                { "camera": "cam0", "image": "cam0/img_0001.png" },
                { "camera": "cam9", "image": "cam0/img_0001.png" },
            ] }))),
            ("exclude.images[1]".into(), ExclusionFault::UnknownCamera)
        );
        // Camera 1 has no image of view 0002, and cam0's image is not its.
        for image in ["cam1/img_0002.png", "cam0/img_0001.png"] {
            assert_eq!(
                fault(resolve(json!({ "features": [
                    { "camera": "cam1", "image": image, "points": [] },
                ] }))),
                ("exclude.features[0]".into(), ExclusionFault::ImageNotFound)
            );
        }
        assert_eq!(
            fault(resolve(
                json!({ "views": [{ "view": { "token": "0009" } }] })
            )),
            ("exclude.views[0]".into(), ExclusionFault::TokenNotFound)
        );
        assert_eq!(
            fault(resolve(
                json!({ "views": [{ "view": { "image": "cam0/img_0009.png" } }] })
            )),
            ("exclude.views[0]".into(), ExclusionFault::ImageNotFound)
        );

        // Under index pairing there are no tokens, and an image listed
        // twice belongs to two views.
        let mut by_index = spec.clone();
        by_index.pose_pairing = Some(PosePairing::ByIndex);
        by_index.cameras.truncate(1);
        by_index.cameras[0].images = vision_calibration_dataset::ImagePattern::List {
            paths: [
                "cam0/img_0001.png",
                "cam0/img_0001.png",
                "cam0/img_0003.png",
            ]
            .iter()
            .map(PathBuf::from)
            .collect(),
        };
        let resolve = |value| exclusions(value).resolve(&by_index, dir);
        assert_eq!(
            fault(resolve(json!({ "views": [{ "view": { "token": "0" } }] }))),
            ("exclude.views[0]".into(), ExclusionFault::NoTokenPairing)
        );
        assert_eq!(
            fault(resolve(
                json!({ "views": [{ "view": { "image": "cam0/img_0001.png" } }] })
            )),
            ("exclude.views[0]".into(), ExclusionFault::AmbiguousImage)
        );
    }

    #[test]
    fn exclusions_round_trip_through_toml_and_json() {
        let text = r#"
            views  = [{ view = { token = "0012" }, note = "robot pose glitch" }]
            images = [{ camera = "cam1", image = "cam1/0003.png" }]
            [[features]]
            camera = "cam0"
            image  = "cam0/0005.png"
            points = [[0.03, 0.06, 0.0]]
        "#;
        let parsed: Exclusions = toml::from_str(text).unwrap();
        let mut expected = Exclusions::default();
        let mut view = ViewExclusion::new(ViewRef::Token("0012".into()));
        view.note = Some("robot pose glitch".into());
        expected.views.push(view);
        expected
            .images
            .push(ImageExclusion::new("cam1", "cam1/0003.png"));
        expected.features.push(FeatureExclusion::new(
            "cam0",
            "cam0/0005.png",
            vec![[0.03, 0.06, 0.0]],
        ));
        assert_eq!(parsed, expected);
        let toml_back: Exclusions = toml::from_str(&toml::to_string(&parsed).unwrap()).unwrap();
        assert_eq!(toml_back, parsed);
        let json = serde_json::to_value(&parsed).unwrap();
        assert_eq!(json["views"][0]["view"], json!({ "token": "0012" }));
        assert!(json["images"][0].get("note").is_none());
        let json_back: Exclusions = serde_json::from_value(json).unwrap();
        assert_eq!(json_back, parsed);
        assert_eq!(
            serde_json::to_value(Exclusions::default()).unwrap(),
            json!({})
        );
    }

    #[test]
    fn a_misspelt_field_is_an_error_not_a_silent_no_op() {
        for text in [
            r#"veiws = [{ view = { token = "1" } }]"#,
            r#"views = [{ view = { token = "1" }, notes = "x" }]"#,
            r#"images = [{ camera = "cam0", imgae = "a.png" }]"#,
        ] {
            assert!(toml::from_str::<Exclusions>(text).is_err(), "{text}");
        }
    }
}
