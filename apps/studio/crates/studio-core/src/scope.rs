//! The files the back end may read: those under a set of root folders,
//! judged after canonicalising, so `..` segments and symbolic links cannot
//! lead out of them. [`FileScope`] is a fixed set; [`RootScope`] is what
//! the commands go through, with the paths they receive as [`UserPath`]s.
//! [`check_manifest`] judges the paths a dataset manifest names before the
//! library reads them.

use std::path::{Component, Path, PathBuf};

use crate::api::{ErrorKind, StudioError};

/// Why a path was refused.
#[derive(Debug, PartialEq, Eq)]
pub enum ScopeError {
    /// The path names no file (or a folder that cannot be resolved).
    NotFound(String),
    /// The path is relative or resolves outside every root.
    Forbidden(String),
}

impl ScopeError {
    pub fn message(&self) -> &str {
        match self {
            Self::NotFound(m) | Self::Forbidden(m) => m,
        }
    }
}

/// Canonical root folders; a path is in scope when it resolves under one.
#[derive(Debug, Clone)]
pub struct FileScope {
    roots: Vec<PathBuf>,
}

impl FileScope {
    /// A scope over `roots`. A root that does not resolve is dropped: there
    /// is nothing under it to serve.
    pub fn new(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut canonical: Vec<PathBuf> = roots
            .into_iter()
            .filter_map(|root| dunce::canonicalize(root).ok())
            .collect();
        canonical.sort();
        canonical.dedup();
        Self { roots: canonical }
    }

    /// The canonical root folders.
    #[cfg(test)]
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// The canonical file `path` names, if it is an absolute path that
    /// resolves under a root.
    pub fn resolve(&self, path: &Path) -> Result<PathBuf, ScopeError> {
        if !path.is_absolute() {
            return Err(ScopeError::Forbidden(format!(
                "{} is not an absolute path",
                path.display()
            )));
        }
        let canonical = dunce::canonicalize(path)
            .map_err(|_| ScopeError::NotFound(format!("{} not found", path.display())))?;
        if self.roots.iter().any(|root| canonical.starts_with(root)) {
            Ok(canonical)
        } else {
            Err(ScopeError::Forbidden(format!(
                "{} is outside the folders this back end serves",
                path.display()
            )))
        }
    }
}

/// A path as a client wrote it. Not a `Path`: only a [`RootScope`] turns it
/// into one, so a command cannot read a path it was sent without the scope
/// judging it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(transparent)]
pub struct UserPath(String);

impl UserPath {
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// The path as written, for messages.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl schemars::JsonSchema for UserPath {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "UserPath".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "An absolute path on the back end's machine, inside one of its root folders. The back end refuses any other.",
            "type": "string"
        })
    }
}

/// The folders the back end may read and write under: the root folders.
/// Every path a command receives goes through it — [`RootScope::existing`],
/// [`RootScope::existing_file`], [`RootScope::existing_dir`] — and so does
/// every path the back end derives from a file ([`RootScope::check`],
/// [`RootScope::check_lexical`]). The roots never grow by opening
/// something: the server's are fixed, the desktop's change only through its
/// settings (`add_root`, the user's act of consent in a native dialog).
pub struct RootScope {
    editable: bool,
    roots: std::sync::RwLock<FileScope>,
}

impl std::fmt::Debug for RootScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootScope")
            .field("editable", &self.editable)
            .field("roots", &self.roots())
            .finish()
    }
}

impl RootScope {
    /// A scope over `roots` that never changes (the server's).
    pub fn fixed(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            editable: false,
            roots: std::sync::RwLock::new(FileScope::new(roots)),
        }
    }

    /// A scope over `roots` that the settings may replace (the desktop's).
    pub fn editable(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            editable: true,
            roots: std::sync::RwLock::new(FileScope::new(roots)),
        }
    }

    /// Whether the roots can change while the back end runs.
    pub fn is_editable(&self) -> bool {
        self.editable
    }

    /// The canonical root folders.
    pub fn roots(&self) -> Vec<PathBuf> {
        self.read().roots.clone()
    }

    /// Replace the roots (an editable scope only).
    pub fn set_roots(&self, roots: impl IntoIterator<Item = PathBuf>) -> Result<(), StudioError> {
        if !self.editable {
            return Err(StudioError::new(
                ErrorKind::RootsFixed,
                "this back end's folders are fixed when it starts",
            ));
        }
        *self.roots.write().unwrap_or_else(|e| e.into_inner()) = FileScope::new(roots);
        Ok(())
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, FileScope> {
        // A poisoned lock only means a reader panicked; the roots are a
        // plain list and stay valid.
        self.roots.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the canonical folder `dir` is one of the roots.
    pub fn is_root(&self, dir: &Path) -> bool {
        self.read().roots.iter().any(|root| root == dir)
    }

    /// The canonical path `path` names, if it exists under a root.
    pub fn existing(&self, path: &UserPath) -> Result<PathBuf, StudioError> {
        self.check(Path::new(path.as_str()))
    }

    /// The canonical file `path` names, if it is a file under a root.
    pub fn existing_file(&self, path: &UserPath) -> Result<PathBuf, StudioError> {
        let file = self.existing(path)?;
        if file.is_file() {
            Ok(file)
        } else {
            Err(
                StudioError::new(ErrorKind::Io, format!("{} is not a file", path.as_str()))
                    .at(path.as_str()),
            )
        }
    }

    /// The canonical folder `path` names, if it is a folder under a root.
    pub fn existing_dir(&self, path: &UserPath) -> Result<PathBuf, StudioError> {
        let dir = self.existing(path)?;
        if dir.is_dir() {
            Ok(dir)
        } else {
            Err(
                StudioError::new(ErrorKind::Io, format!("{} is not a folder", path.as_str()))
                    .at(path.as_str()),
            )
        }
    }

    /// A path the back end derived itself (a project's dataset folder, an
    /// export's image root), checked like a client's.
    pub fn check(&self, path: &Path) -> Result<PathBuf, StudioError> {
        self.read().resolve(path).map_err(|e| scope_error(path, e))
    }

    /// A path that need not exist yet (a manifest's glob base, a file a
    /// command will write): `..` folded lexically, the deepest existing
    /// ancestor canonicalised and judged against the roots, the rest plain
    /// names. Answers the path with that ancestor canonical.
    pub fn check_lexical(&self, path: &Path) -> Result<PathBuf, StudioError> {
        let forbidden = |why: String| scope_error(path, ScopeError::Forbidden(why));
        if !path.is_absolute() {
            return Err(forbidden(format!(
                "{} is not an absolute path",
                path.display()
            )));
        }
        let mut normal = PathBuf::new();
        for part in path.components() {
            match part {
                Component::ParentDir => {
                    if !normal.pop() {
                        return Err(forbidden(format!(
                            "{} leaves the file system",
                            path.display()
                        )));
                    }
                }
                Component::CurDir => {}
                other => normal.push(other),
            }
        }
        let mut existing = normal.as_path();
        let mut rest = Vec::new();
        while !existing.exists() {
            match (existing.parent(), existing.file_name()) {
                (Some(parent), Some(name)) => {
                    rest.push(name.to_owned());
                    existing = parent;
                }
                _ => {
                    return Err(forbidden(format!(
                        "{} has no existing folder",
                        path.display()
                    )));
                }
            }
        }
        let mut resolved = self.check(existing)?;
        for name in rest.into_iter().rev() {
            resolved.push(name);
        }
        Ok(resolved)
    }
}

/// Judge every path the dataset manifest `manifest` names — each camera's
/// image and laser-frame globs and lists, the robot-pose file, the upstream
/// calibration — resolved against `base` as the library resolves them,
/// before the library reads any of them. A glob is judged by its base (the
/// components before the first wildcard), lexically. A manifest can name
/// anything (`/etc`, `../..`); this keeps a run inside the roots.
pub fn check_manifest(
    scope: &RootScope,
    manifest: &serde_json::Value,
    base: &Path,
) -> Result<(), StudioError> {
    let mut paths: Vec<(String, bool)> = Vec::new();
    let mut images = |source: Option<&serde_json::Value>| {
        let Some(source) = source else { return };
        if let Some(pattern) = source.get("pattern").and_then(|p| p.as_str()) {
            paths.push((pattern.to_owned(), true));
        }
        for path in source
            .get("paths")
            .and_then(|p| p.as_array())
            .into_iter()
            .flatten()
        {
            if let Some(path) = path.as_str() {
                paths.push((path.to_owned(), false));
            }
        }
    };
    for camera in manifest
        .get("cameras")
        .and_then(|c| c.as_array())
        .into_iter()
        .flatten()
    {
        images(camera.get("images"));
        images(camera.get("laser_images"));
    }
    let mut file = |value: Option<&serde_json::Value>| {
        if let Some(path) = value.and_then(|v| v.as_str()) {
            paths.push((path.to_owned(), false));
        }
    };
    file(manifest.get("robot_poses").and_then(|r| r.get("path")));
    file(manifest.get("upstream_calibration"));
    for (path, glob) in paths {
        let written = Path::new(&path);
        // An absolute path replaces the base, as `Path::join` does.
        let mut target = if written.has_root() {
            PathBuf::new()
        } else {
            base.to_path_buf()
        };
        for part in written.components() {
            let wild = part
                .as_os_str()
                .to_str()
                .is_some_and(|p| p.contains(['*', '?', '[', '{']));
            if glob && wild {
                break;
            }
            target.push(part.as_os_str());
        }
        scope.check_lexical(&target).map_err(|e| {
            StudioError::new(
                ErrorKind::ForbiddenPath,
                format!(
                    "the manifest names {path}, which is outside the folders this back end serves"
                ),
            )
            .at(e.path.unwrap_or(path))
        })?;
    }
    Ok(())
}

fn scope_error(path: &Path, e: ScopeError) -> StudioError {
    let kind = match e {
        ScopeError::NotFound(_) => ErrorKind::Io,
        ScopeError::Forbidden(_) => ErrorKind::ForbiddenPath,
    };
    StudioError::new(kind, e.message()).at(path.to_string_lossy())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `<tmp>/run/{export.json, images/a.png}` beside `<tmp>/secret.txt`
    /// and `<tmp>/run-other/b.png`; the scope is `<tmp>/run`.
    fn layout() -> (tempfile::TempDir, FileScope, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let run = base.join("run");
        std::fs::create_dir_all(run.join("images")).unwrap();
        std::fs::create_dir_all(base.join("run-other")).unwrap();
        std::fs::write(run.join("export.json"), "{}").unwrap();
        std::fs::write(run.join("images/a.png"), "png").unwrap();
        std::fs::write(base.join("secret.txt"), "secret").unwrap();
        std::fs::write(base.join("run-other/b.png"), "png").unwrap();
        let scope = FileScope::new([run.clone()]);
        (tmp, scope, run)
    }

    #[test]
    fn files_under_a_root_resolve() {
        let (_tmp, scope, run) = layout();
        assert_eq!(
            scope.resolve(&run.join("images/a.png")).unwrap(),
            run.join("images/a.png")
        );
        // `..` that stays inside is fine.
        assert_eq!(
            scope.resolve(&run.join("images/../export.json")).unwrap(),
            run.join("export.json")
        );
    }

    #[test]
    fn dot_dot_out_of_the_root_is_forbidden() {
        let (_tmp, scope, run) = layout();
        let err = scope.resolve(&run.join("../secret.txt")).unwrap_err();
        assert!(matches!(err, ScopeError::Forbidden(_)), "{err:?}");
        let err = scope
            .resolve(&run.join("images/../../secret.txt"))
            .unwrap_err();
        assert!(matches!(err, ScopeError::Forbidden(_)), "{err:?}");
    }

    #[test]
    fn absolute_paths_elsewhere_and_sibling_prefixes_are_forbidden() {
        let (_tmp, scope, run) = layout();
        let outside = run.parent().unwrap().join("secret.txt");
        assert!(matches!(
            scope.resolve(&outside),
            Err(ScopeError::Forbidden(_))
        ));
        // `run-other` shares the root's name as a string prefix, not a path one.
        let sibling = run.parent().unwrap().join("run-other/b.png");
        assert!(matches!(
            scope.resolve(&sibling),
            Err(ScopeError::Forbidden(_))
        ));
    }

    #[test]
    fn relative_and_missing_paths_are_refused() {
        let (_tmp, scope, run) = layout();
        assert!(matches!(
            scope.resolve(Path::new("images/a.png")),
            Err(ScopeError::Forbidden(_))
        ));
        assert!(matches!(
            scope.resolve(&run.join("images/missing.png")),
            Err(ScopeError::NotFound(_))
        ));
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_root_is_forbidden() {
        let (_tmp, scope, run) = layout();
        let link = run.join("images/escape.txt");
        std::os::unix::fs::symlink(run.parent().unwrap().join("secret.txt"), &link).unwrap();
        assert!(matches!(
            scope.resolve(&link),
            Err(ScopeError::Forbidden(_))
        ));
        let dir_link = run.join("up");
        std::os::unix::fs::symlink(run.parent().unwrap(), &dir_link).unwrap();
        assert!(matches!(
            scope.resolve(&dir_link.join("secret.txt")),
            Err(ScopeError::Forbidden(_))
        ));
    }

    #[test]
    fn a_missing_root_is_dropped() {
        let (_tmp, _scope, run) = layout();
        let scope = FileScope::new([run.clone(), run.join("no-such-folder")]);
        assert_eq!(scope.roots(), [run]);
    }

    fn user(path: &Path) -> UserPath {
        UserPath::new(path.to_string_lossy())
    }

    #[test]
    fn a_fixed_scope_checks_paths_and_never_changes() {
        let (_tmp, _, run) = layout();
        let scope = RootScope::fixed([run.clone()]);
        assert_eq!(
            scope
                .existing_file(&user(&run.join("export.json")))
                .unwrap(),
            run.join("export.json")
        );
        let outside = run.parent().unwrap().join("run-other");
        let err = scope.existing_dir(&user(&outside)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = scope.existing_file(&user(&run.join("images"))).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Io);
        let err = scope.set_roots([outside]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::RootsFixed);
        assert_eq!(scope.roots(), std::slice::from_ref(&run));
        assert!(scope.is_root(&run));
    }

    #[test]
    fn an_editable_scope_changes_only_through_its_roots() {
        let (_tmp, _, run) = layout();
        let scope = RootScope::editable([]);
        let image = run.join("images/a.png");
        assert_eq!(
            scope.existing_file(&user(&image)).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );
        scope.set_roots([run.clone()]).unwrap();
        assert_eq!(scope.existing_file(&user(&image)).unwrap(), image);
        // Relative paths are never in scope.
        assert_eq!(
            scope
                .existing_dir(&UserPath::new("images"))
                .unwrap_err()
                .kind,
            ErrorKind::ForbiddenPath
        );
        // The sibling stays out.
        let sibling = run.parent().unwrap().join("run-other/b.png");
        assert_eq!(
            scope.existing_file(&user(&sibling)).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );
    }

    #[test]
    fn lexical_checks_fold_dot_dot_and_judge_the_existing_ancestor() {
        let (_tmp, _, run) = layout();
        let scope = RootScope::fixed([run.clone()]);
        // A path that does not exist yet, inside.
        assert_eq!(
            scope.check_lexical(&run.join("new/sub/x.png")).unwrap(),
            run.join("new/sub/x.png")
        );
        assert_eq!(
            scope.check_lexical(&run.join("images/../new")).unwrap(),
            run.join("new")
        );
        // `..` out of the root, before or after a missing folder.
        for escape in [
            run.join("../secret.txt"),
            run.join("missing/../../secret.txt"),
            run.join("images/../../run-other/x"),
        ] {
            assert_eq!(
                scope.check_lexical(&escape).unwrap_err().kind,
                ErrorKind::ForbiddenPath,
                "{}",
                escape.display()
            );
        }
        assert_eq!(
            scope.check_lexical(Path::new("images/x")).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );
    }

    #[test]
    fn a_manifest_may_name_only_paths_inside_the_roots() {
        let (_tmp, _, run) = layout();
        let scope = RootScope::fixed([run.clone()]);
        let manifest = |cameras: serde_json::Value, extra: serde_json::Value| {
            let mut m = serde_json::json!({ "version": 1, "cameras": cameras });
            m.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            m
        };
        let glob = |pattern: &str| serde_json::json!([{ "id": "c", "images": { "kind": "glob", "pattern": pattern } }]);
        let fine = manifest(
            glob("images/*.png"),
            serde_json::json!({ "robot_poses": { "path": "poses.json" } }),
        );
        check_manifest(&scope, &fine, &run).unwrap();
        check_manifest(
            &scope,
            &manifest(glob("**/*.png"), serde_json::json!({})),
            &run,
        )
        .unwrap();
        let outside = run.parent().unwrap();
        for bad in [
            manifest(glob("../*.png"), serde_json::json!({})),
            manifest(
                glob(&format!("{}/*.txt", outside.display())),
                serde_json::json!({}),
            ),
            manifest(
                serde_json::json!([{ "id": "c", "images": { "kind": "list", "paths": ["images/a.png", "/etc/passwd"] } }]),
                serde_json::json!({}),
            ),
            manifest(
                glob("images/*.png"),
                serde_json::json!({ "robot_poses": { "path": "../secret.txt" } }),
            ),
            manifest(
                glob("images/*.png"),
                serde_json::json!({ "upstream_calibration": "/etc/hosts" }),
            ),
            manifest(
                serde_json::json!([{ "id": "c", "images": { "kind": "glob", "pattern": "*.png" }, "laser_images": { "kind": "glob", "pattern": "../../*" } }]),
                serde_json::json!({}),
            ),
        ] {
            let err = check_manifest(&scope, &bad, &run).unwrap_err();
            assert_eq!(err.kind, ErrorKind::ForbiddenPath, "{bad}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_lexical_check_does_not_follow_a_symlink_out() {
        let (_tmp, _, run) = layout();
        let scope = RootScope::fixed([run.clone()]);
        std::os::unix::fs::symlink(run.parent().unwrap(), run.join("up")).unwrap();
        assert_eq!(
            scope
                .check_lexical(&run.join("up/new.png"))
                .unwrap_err()
                .kind,
            ErrorKind::ForbiddenPath
        );
    }
}
