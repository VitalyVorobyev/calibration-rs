//! The folder browser: `list_dir`, over the roots only, on both transports.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::api::{Ctx, ErrorKind, StudioError};
use crate::scope::UserPath;
use vision_calibration::project::{PROJECT_FILE, PROJECT_FILE_JSON};

/// The most entries a listing answers.
pub const MAX_ENTRIES: usize = 5_000;

/// What a listing shows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Show {
    /// Folders and files.
    #[default]
    All,
    /// Folders only.
    Dirs,
}

/// The folder `path`, or the root folders without one. Entries whose name
/// starts with a dot are left out, and so is a symbolic link that leads out
/// of the roots; past 5 000 entries the listing is `truncated`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListDirArgs {
    #[serde(default)]
    pub path: Option<UserPath>,
    #[serde(default)]
    pub show: Show,
}

/// A folder's entries.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DirListing {
    /// The folder listed (canonical); `null` for the roots.
    pub path: Option<String>,
    /// Its parent while that is inside a root; `null` at a root (go back to
    /// the roots) and for the roots.
    pub parent: Option<String>,
    /// Folders first, then files, each by name.
    pub entries: Vec<DirEntry>,
    /// Whether entries were left out past the limit.
    pub truncated: bool,
}

/// One entry of a listing.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DirEntry {
    pub name: String,
    /// Its canonical path, for the next `list_dir` or a command.
    pub path: String,
    pub kind: EntryKind,
    /// A folder holding a calibration project.
    pub is_project: bool,
    /// A file's size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Dir,
    File,
}

pub fn list_dir(ctx: &Ctx, args: ListDirArgs) -> Result<DirListing, StudioError> {
    let Some(path) = &args.path else {
        let mut entries: Vec<DirEntry> = ctx
            .scope
            .roots()
            .iter()
            .filter_map(|root| entry(root, root.to_string_lossy().into_owned()))
            .collect();
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        return Ok(DirListing {
            path: None,
            parent: None,
            entries,
            truncated: false,
        });
    };
    let dir = ctx.scope.existing_dir(path)?;
    let parent = if ctx.scope.is_root(&dir) {
        None
    } else {
        dir.parent()
            .filter(|parent| ctx.scope.check(parent).is_ok())
            .map(|parent| parent.to_string_lossy().into_owned())
    };
    let read = std::fs::read_dir(&dir).map_err(|e| {
        StudioError::new(ErrorKind::Io, format!("{}: {e}", dir.display())).at(path.as_str())
    })?;
    let mut entries = Vec::new();
    let mut truncated = false;
    for item in read.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        // Resolved through the scope: a link out of the roots is left out.
        let Ok(target) = ctx.scope.check(&item.path()) else {
            continue;
        };
        let Some(entry) = entry(&target, name) else {
            continue;
        };
        if args.show == Show::Dirs && entry.kind != EntryKind::Dir {
            continue;
        }
        if entries.len() == MAX_ENTRIES {
            truncated = true;
            break;
        }
        entries.push(DirEntry {
            path: item.path().to_string_lossy().into_owned(),
            ..entry
        });
    }
    entries.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.name.cmp(&b.name)));
    Ok(DirListing {
        path: Some(dir.to_string_lossy().into_owned()),
        parent,
        entries,
        truncated,
    })
}

/// The entry for the canonical `target`, named `name`.
fn entry(target: &Path, name: String) -> Option<DirEntry> {
    let meta = std::fs::metadata(target).ok()?;
    let kind = if meta.is_dir() {
        EntryKind::Dir
    } else {
        EntryKind::File
    };
    Some(DirEntry {
        name,
        path: target.to_string_lossy().into_owned(),
        kind,
        is_project: kind == EntryKind::Dir
            && (target.join(PROJECT_FILE).is_file() || target.join(PROJECT_FILE_JSON).is_file()),
        size: (kind == EntryKind::File).then_some(meta.len()),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::api::{Api, ErrorKind};
    use crate::scope::RootScope;

    #[test]
    fn listings_stay_inside_the_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let root = base.join("root");
        std::fs::create_dir_all(root.join("proj")).unwrap();
        std::fs::write(root.join("proj/calibration-project.json"), "{}").unwrap();
        std::fs::create_dir_all(root.join("sub/deeper")).unwrap();
        std::fs::write(root.join("a.png"), b"12345").unwrap();
        std::fs::write(root.join(".hidden"), b"").unwrap();
        std::fs::write(base.join("secret.txt"), b"s").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&base, root.join("escape")).unwrap();
            std::os::unix::fs::symlink(root.join("sub"), root.join("inside")).unwrap();
        }
        let api = Api::for_tests(RootScope::fixed([root.clone()]));

        let roots = api.invoke("list_dir", json!({})).unwrap();
        assert_eq!(roots["path"], serde_json::Value::Null);
        assert_eq!(roots["entries"][0]["path"], root.to_string_lossy().as_ref());

        let listing = api
            .invoke("list_dir", json!({"path": root.to_string_lossy()}))
            .unwrap();
        // At a root there is no parent to go up to.
        assert_eq!(listing["parent"], serde_json::Value::Null);
        let names: Vec<&str> = listing["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        #[cfg(unix)]
        assert_eq!(names, ["inside", "proj", "sub", "a.png"]);
        let proj = &listing["entries"].as_array().unwrap()[1];
        assert_eq!(proj["is_project"], true);
        let file = listing["entries"].as_array().unwrap().last().unwrap();
        assert_eq!(
            (file["kind"].as_str(), file["size"].as_u64()),
            (Some("file"), Some(5))
        );

        let dirs = api
            .invoke(
                "list_dir",
                json!({"path": root.to_string_lossy(), "show": "dirs"}),
            )
            .unwrap();
        assert!(
            dirs["entries"]
                .as_array()
                .unwrap()
                .iter()
                .all(|e| e["kind"] == "dir")
        );
        let sub = api
            .invoke(
                "list_dir",
                json!({"path": root.join("sub").to_string_lossy()}),
            )
            .unwrap();
        assert_eq!(sub["parent"], root.to_string_lossy().as_ref());

        // Above the root, beside it, `..` out of it: refused.
        for outside in [base.clone(), root.join(".."), base.join("secret.txt")] {
            let err = api
                .invoke("list_dir", json!({"path": outside.to_string_lossy()}))
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::ForbiddenPath, "{}", outside.display());
        }
        #[cfg(unix)]
        {
            let err = api
                .invoke(
                    "list_dir",
                    json!({"path": root.join("escape").to_string_lossy()}),
                )
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        }
    }
}
