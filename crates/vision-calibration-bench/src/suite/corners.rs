//! Entries frozen from the corners their publisher released, instead of
//! from detection on the images.
//!
//! Some datasets ship their target corners next to the images — OpenCV's
//! own calibration test sets do, in `cv::FileStorage` files. Freezing those
//! corners makes calibration-rs and the publisher's own tests calibrate
//! exactly the same points, so the two results compare without a detector
//! in between. A [`PublishedCorners`] source names the files (fetched like
//! images, checksummed in `fetch/<id>.sha256`) and the cameras' image
//! sizes, which corner files do not record.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use vision_calibration_core::{
    CorrespondenceView, NoMeta, PlanarDataset, Pt2, Pt3, RigDataset, RigView, RigViewObs, View,
};

use super::manifest::safe_relative_path;
use super::{OBSERVATION_SET_VERSION, ObservationSet, ObservedCamera, ProblemInput};

/// The corner files of an entry and the problem they are calibrated as.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishedCorners {
    /// File format of every corner file.
    pub format: CornerFormat,
    /// The problem the views are calibrated as.
    pub problem: CornerProblem,
    /// Target points (metres), view by view, relative to the entry's
    /// fetched files.
    pub object: String,
    /// The cameras, in input order; view `i` of every camera shows the
    /// target points of view `i` of `object`.
    pub cameras: Vec<CornerCamera>,
}

/// File format of published corners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CornerFormat {
    /// OpenCV `cv::FileStorage` XML, as OpenCV's calibration tests read
    /// it: top-level nodes `image_0`, `image_1`, … in order, each a flat
    /// sequence of numbers (`x y` per image point, `x y z` per target
    /// point).
    OpencvStorage,
}

/// The problem published corners are calibrated as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CornerProblem {
    /// One camera's intrinsics.
    PlanarIntrinsics,
    /// A rig of two or more cameras that see the target in the same views.
    RigExtrinsics,
}

/// One camera of [`PublishedCorners`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CornerCamera {
    /// Camera id in the observations.
    pub id: String,
    /// Image points (pixels, OpenCV convention), view by view, relative to
    /// the entry's fetched files.
    pub points: String,
    /// Image `[width, height]` in pixels.
    pub image_size: [u32; 2],
}

impl PublishedCorners {
    /// Every file the source reads, relative to the entry's fetched files.
    pub fn files(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.object.as_str()).chain(self.cameras.iter().map(|c| c.points.as_str()))
    }

    /// Check the camera count fits the problem and every path is safe.
    ///
    /// # Errors
    ///
    /// Fails on a problem with the wrong number of cameras or an unsafe
    /// path (see [`safe_relative_path`]).
    pub fn validate(&self) -> Result<()> {
        match self.problem {
            CornerProblem::PlanarIntrinsics => ensure!(
                self.cameras.len() == 1,
                "planar_intrinsics corners name exactly one camera, not {}",
                self.cameras.len()
            ),
            CornerProblem::RigExtrinsics => ensure!(
                self.cameras.len() >= 2,
                "rig_extrinsics corners name two or more cameras, not {}",
                self.cameras.len()
            ),
        }
        for file in self.files() {
            safe_relative_path(file)?;
        }
        Ok(())
    }

    /// Build the observations from the files under `dir`.
    ///
    /// # Errors
    ///
    /// Fails on an unreadable or malformed file, on files with different
    /// view counts, and on a view whose target and image point counts
    /// differ.
    pub fn observations(&self, dir: &Path) -> Result<ObservationSet> {
        self.validate()?;
        let read = |rel: &str| -> Result<Vec<Vec<f64>>> {
            let path = dir.join(safe_relative_path(rel)?);
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            let nodes = match self.format {
                CornerFormat::OpencvStorage => parse_opencv_storage(&text),
            }
            .with_context(|| format!("parse {}", path.display()))?;
            numbered_views(nodes).with_context(|| format!("views of {}", path.display()))
        };
        let object: Vec<Vec<Pt3>> = read(&self.object)?
            .iter()
            .map(|v| {
                Ok(points::<3>(v)?
                    .into_iter()
                    .map(|[x, y, z]| Pt3::new(x, y, z))
                    .collect())
            })
            .collect::<Result<_>>()
            .with_context(|| format!("target points of {}", self.object))?;
        let per_camera: Vec<Vec<CorrespondenceView>> = self
            .cameras
            .iter()
            .map(|camera| {
                let views = read(&camera.points)?;
                ensure!(
                    views.len() == object.len(),
                    "{} has {} views, {} has {}",
                    camera.points,
                    views.len(),
                    self.object,
                    object.len()
                );
                views
                    .iter()
                    .zip(&object)
                    .enumerate()
                    .map(|(i, (pixels, targets))| {
                        let pixels: Vec<Pt2> = points::<2>(pixels)?
                            .into_iter()
                            .map(|[x, y]| Pt2::new(x, y))
                            .collect();
                        CorrespondenceView::new(targets.clone(), pixels)
                            .with_context(|| format!("view image_{i} of {}", camera.points))
                    })
                    .collect()
            })
            .collect::<Result<_>>()?;

        let input = match self.problem {
            CornerProblem::PlanarIntrinsics => {
                let views = per_camera
                    .into_iter()
                    .next()
                    .expect("validated: one camera")
                    .into_iter()
                    .map(View::without_meta)
                    .collect();
                ProblemInput::PlanarIntrinsics(PlanarDataset::new(views)?)
            }
            CornerProblem::RigExtrinsics => {
                let views = (0..object.len())
                    .map(|i| RigView {
                        meta: NoMeta,
                        obs: RigViewObs {
                            cameras: per_camera.iter().map(|cam| Some(cam[i].clone())).collect(),
                        },
                    })
                    .collect();
                ProblemInput::RigExtrinsics(RigDataset::new(views, self.cameras.len())?)
            }
        };
        Ok(ObservationSet {
            version: OBSERVATION_SET_VERSION,
            cameras: self
                .cameras
                .iter()
                .map(|c| ObservedCamera {
                    id: c.id.clone(),
                    image_size: c.image_size,
                })
                .collect(),
            input,
        })
    }
}

/// A flat number sequence as points of `N` coordinates.
fn points<const N: usize>(values: &[f64]) -> Result<Vec<[f64; N]>> {
    let (points, rest) = values.as_chunks::<N>();
    ensure!(
        rest.is_empty(),
        "{} numbers are not points of {N} coordinates",
        values.len()
    );
    Ok(points.to_vec())
}

/// The nodes `image_0 … image_<n-1>` of a file, in order; any other node
/// name, or a gap, is an error.
fn numbered_views(mut nodes: BTreeMap<String, Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let mut views = Vec::with_capacity(nodes.len());
    while let Some(values) = nodes.remove(&format!("image_{}", views.len())) {
        views.push(values);
    }
    if let Some(name) = nodes.keys().next() {
        bail!(
            "node {name:?} is not one of image_0 … image_{}",
            views.len().saturating_sub(1)
        );
    }
    ensure!(!views.is_empty(), "no image_0 node");
    Ok(views)
}

/// Parse an OpenCV `cv::FileStorage` XML file whose top-level nodes are
/// flat number sequences: `name → numbers`.
///
/// Only that subset is read: nodes with attributes (the `opencv-matrix`
/// type), nested nodes and comments are errors, not skipped.
///
/// # Errors
///
/// Fails on anything outside the subset, a number that does not parse and
/// a duplicate node.
pub fn parse_opencv_storage(text: &str) -> Result<BTreeMap<String, Vec<f64>>> {
    let mut rest = text.trim_start();
    if let Some(declaration) = rest.strip_prefix("<?xml") {
        rest = declaration
            .split_once("?>")
            .context("unterminated XML declaration")?
            .1;
    }
    let body = rest
        .trim()
        .strip_prefix("<opencv_storage>")
        .and_then(|b| b.strip_suffix("</opencv_storage>"))
        .context("expected one <opencv_storage> root element")?;
    let mut nodes = BTreeMap::new();
    let mut rest = body.trim_start();
    while !rest.is_empty() {
        let (name, after) = rest
            .strip_prefix('<')
            .and_then(|r| r.split_once('>'))
            .context("expected an element")?;
        ensure!(
            !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'),
            "element <{name}>: only plain named nodes of numbers are read"
        );
        let (content, after) = after
            .split_once(&format!("</{name}>"))
            .with_context(|| format!("element <{name}> is not closed"))?;
        ensure!(
            !content.contains('<'),
            "element <{name}>: only a flat sequence of numbers is read"
        );
        let values = content
            .split_whitespace()
            .map(|token| {
                token
                    .parse::<f64>()
                    .with_context(|| format!("element <{name}>: {token:?} is not a number"))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            nodes.insert(name.to_owned(), values).is_none(),
            "duplicate element <{name}>"
        );
        rest = after.trim_start();
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJECT: &str = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<image_0>\n  0. 0. 0. \
        2.4399999529123306e-02 0. 0. 0.\n  2.4399999529123306e-02 0. 1. 1. 0.</image_0>\n\
        <image_1>\n  0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 1. 0.</image_1>\n</opencv_storage>\n";
    const LEFT: &str = "<?xml version=\"1.0\"?>\n<opencv_storage>\n<image_0>\n  5.375e+02 \
        3.785e+02 5.84e+02 3.80e+02 5.3e+02 4.1e+02 6. 7.</image_0>\n<image_1>\n  1. 2. 3. \
        4. 5. 6. 7. 8.</image_1>\n</opencv_storage>\n";

    #[test]
    fn parses_flat_number_nodes() {
        let nodes = parse_opencv_storage(OBJECT).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes["image_0"].len(), 12);
        assert_eq!(nodes["image_0"][3], 2.4399999529123306e-02);
        assert_eq!(nodes["image_1"][3], 1.0);
    }

    #[test]
    fn rejects_what_it_does_not_read() {
        for (text, message) in [
            (
                "<opencv_storage><m type_id=\"opencv-matrix\">1</m></opencv_storage>",
                "plain named",
            ),
            (
                "<opencv_storage><a><b>1</b></a></opencv_storage>",
                "flat sequence",
            ),
            (
                "<opencv_storage><a>1 x</a></opencv_storage>",
                "not a number",
            ),
            (
                "<opencv_storage><a>1</a><a>2</a></opencv_storage>",
                "duplicate",
            ),
            ("<opencv_storage><a>1</opencv_storage>", "not closed"),
            ("<storage><a>1</a></storage>", "root element"),
        ] {
            let err = parse_opencv_storage(text).unwrap_err();
            assert!(format!("{err:#}").contains(message), "{text}: {err:#}");
        }
    }

    #[test]
    fn views_are_numbered_without_gaps() {
        let node = |n: &str| (n.to_owned(), vec![0.0]);
        let ok = BTreeMap::from([node("image_1"), node("image_0"), node("image_2")]);
        assert_eq!(numbered_views(ok).unwrap().len(), 3);
        let gap = BTreeMap::from([node("image_0"), node("image_2")]);
        assert!(numbered_views(gap).is_err());
        let other = BTreeMap::from([node("image_0"), node("camera_matrix")]);
        assert!(numbered_views(other).is_err());
    }

    fn corners(problem: CornerProblem, cameras: &[&str]) -> PublishedCorners {
        PublishedCorners {
            format: CornerFormat::OpencvStorage,
            problem,
            object: "object.xml".into(),
            cameras: cameras
                .iter()
                .map(|id| CornerCamera {
                    id: (*id).into(),
                    points: format!("{id}.xml"),
                    image_size: [1280, 800],
                })
                .collect(),
        }
    }

    fn files_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("calib-corners-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("object.xml"), OBJECT).unwrap();
        std::fs::write(dir.join("left.xml"), LEFT).unwrap();
        std::fs::write(dir.join("right.xml"), LEFT).unwrap();
        dir
    }

    #[test]
    fn builds_planar_and_rig_observations() {
        let dir = files_dir("build");
        let planar = corners(CornerProblem::PlanarIntrinsics, &["left"])
            .observations(&dir)
            .unwrap();
        assert_eq!(planar.input.problem_name(), "planar_intrinsics");
        assert_eq!(planar.cameras[0].image_size, [1280, 800]);
        let ProblemInput::PlanarIntrinsics(input) = &planar.input else {
            unreachable!()
        };
        assert_eq!(input.views.len(), 2);
        assert_eq!(input.views[0].obs.points_2d[1], Pt2::new(584.0, 380.0));
        assert_eq!(input.views[1].obs.points_3d[2], Pt3::new(0.0, 1.0, 0.0));

        let rig = corners(CornerProblem::RigExtrinsics, &["left", "right"])
            .observations(&dir)
            .unwrap();
        let ProblemInput::RigExtrinsics(input) = &rig.input else {
            unreachable!()
        };
        assert_eq!(input.num_cameras, 2);
        assert_eq!(input.views.len(), 2);
        assert!(input.views[1].obs.cameras.iter().all(Option::is_some));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mismatched_files_are_errors() {
        let dir = files_dir("mismatch");
        std::fs::write(
            dir.join("right.xml"),
            "<opencv_storage><image_0>1 2</image_0></opencv_storage>",
        )
        .unwrap();
        let err = corners(CornerProblem::RigExtrinsics, &["left", "right"])
            .observations(&dir)
            .unwrap_err();
        assert!(format!("{err:#}").contains("1 views"), "{err:#}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn camera_count_must_fit_the_problem() {
        assert!(
            corners(CornerProblem::PlanarIntrinsics, &["a", "b"])
                .validate()
                .is_err()
        );
        assert!(
            corners(CornerProblem::RigExtrinsics, &["a"])
                .validate()
                .is_err()
        );
        let mut unsafe_path = corners(CornerProblem::PlanarIntrinsics, &["a"]);
        unsafe_path.object = "../object.xml".into();
        assert!(unsafe_path.validate().is_err());
    }
}
