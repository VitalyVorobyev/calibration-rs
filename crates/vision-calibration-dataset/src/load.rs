//! Reading a [`DatasetSpec`] from a manifest file.

use std::path::{Path, PathBuf};

use crate::DatasetSpec;

/// Why a manifest file could not be loaded.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ManifestError {
    /// The file could not be read.
    #[error("cannot read manifest {}: {source}", path.display())]
    Io {
        /// The manifest path.
        path: PathBuf,
        /// The read failure.
        #[source]
        source: std::io::Error,
    },
    /// The file is not a valid manifest in its format.
    #[error("invalid manifest {}: {message}", path.display())]
    Parse {
        /// The manifest path.
        path: PathBuf,
        /// The TOML or JSON parser's message.
        message: String,
    },
}

impl DatasetSpec {
    /// Read a manifest file: TOML for a `.toml` extension, JSON otherwise.
    ///
    /// This parses the manifest only; [`validate`](crate::validate) checks it
    /// against the dataset folder, and the dataset runner does so before it
    /// reads any image.
    ///
    /// # Errors
    ///
    /// [`ManifestError::Io`] when the file cannot be read,
    /// [`ManifestError::Parse`] when it is not a manifest.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ManifestError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let is_toml = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"));
        let parsed = if is_toml {
            toml::from_str(&text).map_err(|err| err.to_string())
        } else {
            serde_json::from_str(&text).map_err(|err| err.to_string())
        };
        parsed.map_err(|message| ManifestError::Parse {
            path: path.to_path_buf(),
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST_TOML: &str = r#"
version = 1
topology = "planar_intrinsics"

[[cameras]]
id = "cam0"

[cameras.images]
kind = "glob"
pattern = "*.png"

[target]
kind = "chessboard"
rows = 7
cols = 11
square_size_m = 0.03
"#;

    #[test]
    fn reads_toml_and_json_by_extension() {
        let dir = std::env::temp_dir().join(format!("vc-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let toml_path = dir.join("dataset.toml");
        std::fs::write(&toml_path, MANIFEST_TOML).unwrap();
        let spec = DatasetSpec::from_path(&toml_path).unwrap();
        let json_path = dir.join("dataset.json");
        std::fs::write(&json_path, serde_json::to_string(&spec).unwrap()).unwrap();
        let again = DatasetSpec::from_path(&json_path).unwrap();
        assert_eq!(again.cameras.len(), 1);
        assert_eq!(again.topology, spec.topology);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn names_the_file_on_failure() {
        let missing = DatasetSpec::from_path("no/such/dataset.toml").unwrap_err();
        assert!(matches!(missing, ManifestError::Io { .. }));
        assert!(missing.to_string().contains("no/such/dataset.toml"));

        let dir = std::env::temp_dir().join(format!("vc-manifest-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bad = dir.join("dataset.json");
        std::fs::write(&bad, "{ not json").unwrap();
        let err = DatasetSpec::from_path(&bad).unwrap_err();
        assert!(matches!(err, ManifestError::Parse { .. }));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
