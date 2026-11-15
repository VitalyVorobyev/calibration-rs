//! The files a viewer may serve: those under a fixed set of root folders,
//! judged after canonicalising, so `..` segments and symbolic links cannot
//! lead out of them.

use std::path::{Path, PathBuf};

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
}
