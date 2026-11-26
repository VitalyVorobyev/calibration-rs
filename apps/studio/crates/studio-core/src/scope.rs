//! The files the back end may read: those under a set of root folders,
//! judged after canonicalising, so `..` segments and symbolic links cannot
//! lead out of them. [`FileScope`] is a fixed set; [`RootScope`] is what
//! the commands go through, with the paths they receive as [`UserPath`]s.

use std::path::{Path, PathBuf};

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
                "{} is outside the folders this viewer serves",
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
            "description": "An absolute path on the back end's machine. The back end refuses one outside the folders the user opened.",
            "type": "string"
        })
    }
}

/// How a [`RootScope`] gets its roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootMode {
    /// Fixed when the scope is made (the browser viewer: the run folder and
    /// its image root). Opening a path outside them is refused.
    Fixed,
    /// Seeded by what the user opens (the desktop app: a folder picked in a
    /// native dialog, a project, an export): opening a path grants it.
    Growable,
}

/// Called with each folder a growable scope grants, so a shell can mirror
/// it (the desktop app's asset protocol scope).
pub type GrantHook = Box<dyn Fn(&Path) + Send + Sync>;

/// The folders the back end may read and write under. Every path a command
/// receives goes through it: [`RootScope::existing_file`] and
/// [`RootScope::existing_dir`] check a path against the roots,
/// [`RootScope::open_dir`] and [`RootScope::open_file`] are the acts of
/// opening that grant a root in [`RootMode::Growable`] and check it in
/// [`RootMode::Fixed`].
pub struct RootScope {
    mode: RootMode,
    roots: std::sync::RwLock<FileScope>,
    on_grant: Option<GrantHook>,
}

impl std::fmt::Debug for RootScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RootScope")
            .field("mode", &self.mode)
            .field("roots", &self.roots())
            .finish()
    }
}

impl RootScope {
    /// A scope over `roots` that never grows.
    pub fn fixed(roots: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            mode: RootMode::Fixed,
            roots: std::sync::RwLock::new(FileScope::new(roots)),
            on_grant: None,
        }
    }

    /// An empty scope that grows with what is opened; `on_grant` hears of
    /// each folder granted.
    pub fn growable(on_grant: Option<GrantHook>) -> Self {
        Self {
            mode: RootMode::Growable,
            roots: std::sync::RwLock::new(FileScope::new([])),
            on_grant,
        }
    }

    pub fn mode(&self) -> RootMode {
        self.mode
    }

    /// The canonical root folders.
    pub fn roots(&self) -> Vec<PathBuf> {
        self.read().roots.clone()
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, FileScope> {
        // A poisoned lock only means a reader panicked; the roots are a
        // plain list and stay valid.
        self.roots.read().unwrap_or_else(|e| e.into_inner())
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

    /// The user opens the folder `path`: granted in a growable scope,
    /// checked in a fixed one.
    pub fn open_dir(&self, path: &UserPath) -> Result<PathBuf, StudioError> {
        self.open_path(Path::new(path.as_str()))?;
        self.existing_dir(path)
    }

    /// The user opens the file `path`: its folder is granted in a growable
    /// scope; the file is checked in a fixed one.
    pub fn open_file(&self, path: &UserPath) -> Result<PathBuf, StudioError> {
        if self.mode == RootMode::Growable {
            let file = Path::new(path.as_str());
            let parent = file.parent().unwrap_or(file);
            self.open_path(parent)?;
        }
        self.existing_file(path)
    }

    /// Grant the folder `path` (growable), or check it (fixed). A grant needs
    /// an absolute path to an existing folder.
    pub fn open_path(&self, path: &Path) -> Result<PathBuf, StudioError> {
        if self.mode == RootMode::Fixed {
            return self.check(path);
        }
        if !path.is_absolute() {
            return Err(scope_error(
                path,
                ScopeError::Forbidden(format!("{} is not an absolute path", path.display())),
            ));
        }
        let dir = dunce::canonicalize(path).map_err(|e| {
            StudioError::new(ErrorKind::Io, format!("{}: {e}", path.display()))
                .at(path.to_string_lossy())
        })?;
        if !dir.is_dir() {
            return Err(StudioError::new(
                ErrorKind::Io,
                format!("{} is not a folder", path.display()),
            )
            .at(path.to_string_lossy()));
        }
        let added = {
            let mut roots = self.roots.write().unwrap_or_else(|e| e.into_inner());
            if roots.roots.iter().any(|root| dir.starts_with(root)) {
                false
            } else {
                let mut all = roots.roots.clone();
                all.push(dir.clone());
                *roots = FileScope::new(all);
                true
            }
        };
        if added && let Some(hook) = &self.on_grant {
            hook(&dir);
        }
        Ok(dir)
    }
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
    fn a_fixed_scope_checks_what_is_opened_and_never_grows() {
        let (_tmp, _, run) = layout();
        let scope = RootScope::fixed([run.clone()]);
        assert_eq!(
            scope.open_file(&user(&run.join("export.json"))).unwrap(),
            run.join("export.json")
        );
        let outside = run.parent().unwrap().join("run-other");
        let err = scope.open_dir(&user(&outside)).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        assert_eq!(scope.roots(), std::slice::from_ref(&run));
        let err = scope.existing_file(&user(&run.join("images"))).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Io);
    }

    #[test]
    fn a_growable_scope_grants_what_is_opened_and_tells_the_hook() {
        let (_tmp, _, run) = layout();
        let granted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = std::sync::Arc::clone(&granted);
        let scope = RootScope::growable(Some(Box::new(move |dir: &Path| {
            seen.lock().unwrap().push(dir.to_path_buf());
        })));
        let image = run.join("images/a.png");
        assert_eq!(
            scope.existing_file(&user(&image)).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );
        // Opening a file grants its folder; a folder inside a root adds none.
        scope.open_file(&user(&run.join("export.json"))).unwrap();
        scope.open_dir(&user(&run.join("images"))).unwrap();
        assert_eq!(scope.existing_file(&user(&image)).unwrap(), image);
        assert_eq!(*granted.lock().unwrap(), std::slice::from_ref(&run));
        // Relative paths are never granted.
        assert_eq!(
            scope.open_dir(&UserPath::new("images")).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );
        // The sibling stays out until it is opened itself.
        let sibling = run.parent().unwrap().join("run-other/b.png");
        assert_eq!(
            scope.existing_file(&user(&sibling)).unwrap_err().kind,
            ErrorKind::ForbiddenPath
        );
    }
}
