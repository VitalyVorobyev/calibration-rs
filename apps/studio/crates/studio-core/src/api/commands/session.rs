//! The session: what this back end is and serves (`server_info`), the
//! desktop's root folders (`add_root`, `remove_root`) and the recent
//! projects (`recent_projects`, `forget_recent`).

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::api::{Ctx, ErrorKind, StudioError};
use crate::scope::UserPath;

/// What this back end is and what it serves. The frontend branches on
/// these values, never on which transport carries them.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ServerInfoArgs {}

/// This back end.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ServerInfo {
    /// The app's version.
    pub version: String,
    /// The shell: `desktop` or `server`.
    pub shell: String,
    /// The root folders: the only ones it reads and writes, where the folder
    /// browser starts.
    pub roots: Vec<RootFolder>,
    /// Whether `add_root` and `remove_root` change them (the desktop); a
    /// server's are fixed when it starts.
    pub roots_editable: bool,
    /// The Quick Start samples `fetch_sample` makes into projects: the
    /// library's sample catalogue, empty when this back end was built
    /// without them.
    pub samples: Vec<SampleInfo>,
}

/// One Quick Start sample.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SampleInfo {
    /// The id `fetch_sample` takes.
    pub id: String,
    /// What it holds: the workflow, the views, the target.
    pub description: String,
    /// Its workflow.
    pub topology: vision_calibration::dataset::ProblemKind,
    /// Its licence (SPDX; `NOASSERTION` when the publisher states none).
    pub licence: String,
}

/// The samples this back end offers.
fn sample_list() -> Vec<SampleInfo> {
    if !cfg!(feature = "samples") {
        return Vec::new();
    }
    vision_calibration::dataset::samples::catalogue()
        .iter()
        .map(|s| SampleInfo {
            id: s.id.clone(),
            description: s.description.clone(),
            topology: s.manifest().topology,
            licence: s.licence.clone(),
        })
        .collect()
}

/// One root folder.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RootFolder {
    /// Its canonical path.
    pub path: String,
    /// The Quick Start samples' folder: always a root, never removed.
    pub samples: bool,
}

pub fn server_info(ctx: &Ctx, _: ServerInfoArgs) -> Result<ServerInfo, StudioError> {
    Ok(info(ctx))
}

fn info(ctx: &Ctx) -> ServerInfo {
    ServerInfo {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        shell: ctx.shell.to_owned(),
        roots: ctx
            .scope
            .roots()
            .iter()
            .map(|root| RootFolder {
                path: root.to_string_lossy().into_owned(),
                samples: dunce::canonicalize(&ctx.dirs.samples).is_ok_and(|s| s == *root),
            })
            .collect(),
        roots_editable: ctx.scope.is_editable(),
        samples: sample_list(),
    }
}

/// Add the folder `path` to the root folders (the desktop: the folder the
/// user picked in the native dialog, the one act that grants a folder).
/// `roots_fixed` on a server.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddRootArgs {
    pub path: UserPath,
}

pub fn add_root(ctx: &Ctx, args: AddRootArgs) -> Result<ServerInfo, StudioError> {
    if !ctx.scope.is_editable() {
        return Err(StudioError::new(
            ErrorKind::RootsFixed,
            "this server's folders are fixed when it starts (its --root flags)",
        ));
    }
    // The one path not judged by the scope: granting it is the point.
    let path = Path::new(args.path.as_str());
    if !path.is_absolute() {
        return Err(StudioError::new(
            ErrorKind::ForbiddenPath,
            format!("{} is not an absolute path", args.path.as_str()),
        )
        .at(args.path.as_str()));
    }
    let dir = dunce::canonicalize(path)
        .ok()
        .filter(|dir| dir.is_dir())
        .ok_or_else(|| {
            StudioError::new(
                ErrorKind::Io,
                format!("{} is not a folder", args.path.as_str()),
            )
            .at(args.path.as_str())
        })?;
    let stored = ctx.settings.add_root(&dir)?;
    ctx.scope.set_roots(ctx.root_folders(&stored))?;
    Ok(info(ctx))
}

/// Remove `path` from the root folders (the desktop). The samples' folder
/// is always a root.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemoveRootArgs {
    pub path: UserPath,
}

pub fn remove_root(ctx: &Ctx, args: RemoveRootArgs) -> Result<ServerInfo, StudioError> {
    let stored = ctx.settings.remove_root(args.path.as_str())?;
    ctx.scope.set_roots(ctx.root_folders(&stored))?;
    Ok(info(ctx))
}

/// The projects opened most recently on this back end, newest first.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecentProjectsArgs {}

pub fn recent_projects(ctx: &Ctx, _: RecentProjectsArgs) -> Result<Vec<String>, StudioError> {
    Ok(ctx.settings.recent())
}

/// Drop `dir` from the recent projects (one that no longer opens).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForgetRecentArgs {
    pub dir: UserPath,
}

pub fn forget_recent(ctx: &Ctx, args: ForgetRecentArgs) -> Result<Vec<String>, StudioError> {
    ctx.settings.forget_recent(args.dir.as_str())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::api::{Api, Dirs, ErrorKind, Setup};

    fn setup(tmp: &std::path::Path, fixed: Option<Vec<std::path::PathBuf>>) -> Setup {
        Setup {
            dirs: Dirs {
                cache: tmp.join("cache"),
                state: tmp.join("state"),
                samples: tmp.join("samples"),
            },
            fixed_roots: fixed,
            shell: "desktop",
        }
    }

    #[test]
    fn the_desktop_starts_with_no_roots_and_keeps_the_ones_added() {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let data = base.join("data");
        std::fs::create_dir(&data).unwrap();
        let samples = base.join("samples");
        let api = Api::new(setup(&base, None));
        let info = api.invoke("server_info", json!({})).unwrap();
        assert_eq!(info["roots_editable"], true);
        assert_eq!(
            info["roots"],
            json!([{ "path": samples.to_string_lossy(), "samples": true }])
        );
        let image = data.join("a.png");
        std::fs::write(&image, b"x").unwrap();
        assert_eq!(
            api.ctx().scope.check(&image).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );

        let info = api
            .invoke("add_root", json!({"path": data.to_string_lossy()}))
            .unwrap();
        assert_eq!(info["roots"].as_array().unwrap().len(), 2);
        assert!(api.ctx().scope.check(&image).is_ok());
        let err = api
            .invoke("add_root", json!({"path": "relative/dir"}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);

        // A restart reads them back.
        drop(api);
        let api = Api::new(setup(&base, None));
        assert!(api.ctx().scope.check(&image).is_ok());
        api.invoke("remove_root", json!({"path": data.to_string_lossy()}))
            .unwrap();
        assert!(api.ctx().scope.check(&image).is_err());
    }

    #[test]
    fn a_server_s_roots_are_fixed() {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let api = Api::new(setup(&base, Some(vec![base.clone()])));
        let info = api.invoke("server_info", json!({})).unwrap();
        assert_eq!(info["roots_editable"], false);
        let samples = info["samples"].as_array().unwrap();
        assert!(
            samples
                .iter()
                .any(|s| s["id"] == "kuka_1" && s["topology"] == "single_cam_handeye")
        );
        for command in ["add_root", "remove_root"] {
            let err = api
                .invoke(command, json!({"path": base.to_string_lossy()}))
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::RootsFixed, "{command}");
        }
        assert_eq!(api.invoke("recent_projects", json!({})).unwrap(), json!([]));
    }
}
