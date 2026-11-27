//! The views a project's dataset pairs into (`project_views`): each
//! camera's image, the robot pose and the laser frames each view pairs
//! with, and what a run would refuse, all without reading an image. A draft
//! spec is judged in place of the saved one, also in a folder that is no
//! project yet.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::dataset_runner::{PoseRow, UnpairedLaserFrame};
use vision_calibration::project::{DatasetSource, Project, ProjectSpec};

use super::project::check_spec_paths;
use crate::api::{Ctx, ErrorKind, StudioError};
use crate::runs::{open_project as open_checked, slash};
use crate::scope::UserPath;

/// The views the project's images pair into, without reading an image, and
/// what is wrong with the project as a run would find it before detection:
/// of the saved project, or of the draft `spec` when given (nothing is
/// written). Each problem is a message with the field it is about. With
/// robot poses or laser images in the manifest, each view also names the
/// pose and the laser frames it pairs with, and what pairs with no view is
/// listed.
///
/// With a `spec`, `dir` may also be a folder that is not a project yet (a
/// new project's data folder): the draft is then judged as a project in
/// that folder, its inline manifest's `root` read against it.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectViewsArgs {
    pub dir: UserPath,
    #[serde(default)]
    pub spec: Option<ProjectSpec>,
}

/// The dataset's views and the project's problems.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ProjectViews {
    /// The cameras' ids, in manifest order.
    pub cameras: Vec<String>,
    /// The views, in pairing order; empty when the images cannot be
    /// listed or paired (a message says why).
    pub views: Vec<ViewRow>,
    /// What a run would refuse, each with the field it is about when there
    /// is one; empty when the project would run.
    pub messages: Vec<StudioError>,
    /// How the views pair with the manifest's robot poses (each view's
    /// `pose`); absent when the manifest declares none, or when the pose
    /// file cannot be listed (a message says why).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poses: Option<RobotPosePairing>,
    /// How the views pair with the cameras' laser frames (each view's
    /// `laser_frames`); absent when no camera declares laser images, or
    /// when they cannot be listed (a message says why).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub laser_frames: Option<LaserFramePairing>,
}

/// One view of the dataset.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ViewRow {
    /// Its pairing token (the view's index under index pairing).
    pub token: String,
    /// Each camera's image, relative to the data folder (absolute outside
    /// it), `/`-separated; `null` where a camera has none.
    pub images: Vec<Option<String>>,
    /// The robot pose the view pairs with; absent when none does, or when
    /// the reply has no `poses`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose: Option<PoseRef>,
    /// Each camera's laser frame in the view, as `images` names images;
    /// `null` where a camera has none. Empty when the reply has no
    /// `laser_frames`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub laser_frames: Vec<Option<String>>,
}

/// One pose of the manifest's robot pose file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct PoseRef {
    /// Its zero-based position among the file's poses (a CSV's header
    /// excluded), as a pose parse error names it.
    pub index: usize,
    /// Its `pose_id`, when the manifest maps a `pose_id` column.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl From<&PoseRow> for PoseRef {
    fn from(row: &PoseRow) -> Self {
        Self {
            index: row.index,
            id: row.id.clone(),
        }
    }
}

/// What the robot pose pairing leaves over.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RobotPosePairing {
    /// The poses that pair with no view, in file order.
    pub unpaired: Vec<PoseRef>,
}

/// What the laser frame pairing leaves over.
#[derive(Debug, Serialize, JsonSchema)]
pub struct LaserFramePairing {
    /// The laser frames that pair with no view, by camera, each camera's in
    /// the order its images sort.
    pub unpaired: Vec<UnpairedLaserFrameRef>,
}

/// A laser frame that pairs with no view.
#[derive(Debug, Serialize, JsonSchema)]
pub struct UnpairedLaserFrameRef {
    /// Index of the camera in the manifest's `cameras`.
    pub camera: usize,
    /// The laser image, as `images` names images.
    pub image: String,
    /// The pairing token it carries: its index among the camera's laser
    /// images under index pairing, the token its file name yields under
    /// token pairing; absent when the file name yields none. A token of a
    /// view belongs to a view another frame of the camera pairs with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

pub fn project_views(ctx: &Ctx, args: ProjectViewsArgs) -> Result<ProjectViews, StudioError> {
    // The scratch project of a draft in a folder that is no project yet;
    // removed when this call ends.
    let mut _scratch = None;
    let project = match (open_checked(ctx, &args.dir), &args.spec) {
        (Err(e), Some(spec)) if e.kind == ErrorKind::NotAProject => {
            let (project, guard) = draft_project(ctx, &args.dir, spec)?;
            _scratch = Some(guard);
            project
        }
        (opened, _) => {
            let mut project = opened?;
            if let Some(spec) = args.spec {
                check_spec_paths(ctx, project.dir(), &spec)?;
                *project.spec_mut() = spec;
            }
            project
        }
    };
    let mut messages: Vec<StudioError> = Vec::new();
    let push = |error: StudioError, messages: &mut Vec<StudioError>| {
        if !messages.iter().any(|m| m.message == error.message) {
            messages.push(error);
        }
    };
    let dataset = match project.dataset() {
        Ok(dataset) => dataset,
        Err(e) => {
            return Ok(ProjectViews {
                cameras: Vec::new(),
                views: Vec::new(),
                messages: vec![e.into()],
                poses: None,
                laser_frames: None,
            });
        }
    };
    let data_root = project.data_root()?;
    ctx.scope.check_dataset(&dataset, &data_root)?;
    if let Err(e) = project.check() {
        push(e.into(), &mut messages);
    }
    let paired = match project.views() {
        Ok(views) => views,
        Err(e) => {
            push(e.into(), &mut messages);
            return Ok(ProjectViews {
                cameras: dataset.cameras.iter().map(|c| c.id.clone()).collect(),
                views: Vec::new(),
                messages,
                poses: None,
                laser_frames: None,
            });
        }
    };
    // Either pairing failing leaves the views, and the other pairing, as
    // they are: its error is one more message.
    let view_poses = project.view_poses(&paired).unwrap_or_else(|e| {
        push(e.into(), &mut messages);
        None
    });
    let view_lasers = project.view_laser_frames(&paired).unwrap_or_else(|e| {
        push(e.into(), &mut messages);
        None
    });
    let name = |path: &Path| relative(&data_root, path);
    let views = paired
        .tokens
        .iter()
        .zip(&paired.paths)
        .enumerate()
        .map(|(i, (token, paths))| ViewRow {
            token: token.clone(),
            images: paths.iter().map(|p| p.as_deref().map(name)).collect(),
            pose: view_poses
                .as_ref()
                .and_then(|poses| poses.views.get(i)?.as_ref().map(PoseRef::from)),
            laser_frames: view_lasers
                .as_ref()
                .and_then(|lasers| lasers.paths.get(i))
                .map(|paths| paths.iter().map(|p| p.as_deref().map(name)).collect())
                .unwrap_or_default(),
        })
        .collect();
    Ok(ProjectViews {
        cameras: dataset.cameras.iter().map(|c| c.id.clone()).collect(),
        views,
        messages,
        poses: view_poses.map(|poses| RobotPosePairing {
            unpaired: poses.unpaired.iter().map(PoseRef::from).collect(),
        }),
        laser_frames: view_lasers.map(|lasers| LaserFramePairing {
            unpaired: lasers
                .unpaired
                .iter()
                .map(|frame: &UnpairedLaserFrame| UnpairedLaserFrameRef {
                    camera: frame.camera,
                    image: name(&frame.path),
                    token: frame.token.clone(),
                })
                .collect(),
        }),
    })
}

/// A scratch folder for a draft project, removed on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover scratch folder holds one small file.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The draft `spec` as a project whose data folder is `folder` (a folder
/// that holds no project yet), so a new project's layout can be previewed
/// before anything is written to it: the project file goes to a scratch
/// folder under the cache, its manifest's `root` made the absolute data
/// folder. Only an inline manifest has a data folder of its own.
fn draft_project(
    ctx: &Ctx,
    folder: &UserPath,
    spec: &ProjectSpec,
) -> Result<(Project, Scratch), StudioError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let folder = ctx.scope.existing_dir(folder)?;
    let DatasetSource::Inline {
        root,
        spec: manifest,
        ..
    } = &spec.dataset
    else {
        return Err(StudioError::invalid(
            "a draft in a folder that holds no project needs its manifest written in the spec",
        ));
    };
    // What the draft names must lie inside the roots, as a saved project's
    // does (judged as if the project were the folder itself).
    check_spec_paths(ctx, &folder, spec)?;
    let data_root = folder.join(root);
    let mut draft = spec.clone();
    draft.dataset = DatasetSource::inline(data_root, manifest.as_ref().clone());
    let dir = ctx.dirs.cache.join("drafts").join(format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let guard = Scratch(dir.clone());
    let project = Project::create(&dir, draft)?;
    Ok((project, guard))
}

/// `path` relative to `root` when it lies under it, `/`-separated.
fn relative(root: &Path, path: &Path) -> String {
    slash(path.strip_prefix(root).unwrap_or(path))
}
