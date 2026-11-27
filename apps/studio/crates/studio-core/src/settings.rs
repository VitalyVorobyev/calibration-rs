//! What the back end keeps between runs: the desktop's root folders
//! (`settings.json`) and the recent projects (`state.json`), both versioned
//! JSON in the state folder the shell names. The server's roots are its
//! command line's and are not stored; its recent projects are the device's,
//! shared by its clients.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::api::{ErrorKind, StudioError};

/// How many recent projects are kept.
pub const RECENT_LIMIT: usize = 12;
/// The settings file's name in the state folder.
pub const SETTINGS_FILE: &str = "settings.json";
/// The state file's name in the state folder.
pub const STATE_FILE: &str = "state.json";

/// The desktop's settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Settings {
    /// The file's format version (1).
    #[serde(default = "version_1")]
    pub version: u32,
    /// The root folders the user added: the only folders the app reads and
    /// writes, and where its folder browser starts.
    #[serde(default)]
    pub roots: Vec<String>,
}

/// What the back end remembers for the user.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct State {
    /// The file's format version (1).
    #[serde(default = "version_1")]
    pub version: u32,
    /// The projects opened most recently, newest first.
    #[serde(default)]
    pub recent_projects: Vec<String>,
}

fn version_1() -> u32 {
    1
}

/// The settings and state files, with their contents in memory.
pub struct SettingsStore {
    dir: PathBuf,
    /// The desktop's settings; `None` on the server (fixed roots).
    settings: Option<Mutex<Settings>>,
    state: Mutex<State>,
}

impl SettingsStore {
    /// The store in `dir`, read now. `editable_roots` keeps `settings.json`
    /// (the desktop); without it the roots are not stored. A file that is
    /// missing or does not parse reads as empty: it is a convenience, never a
    /// reason not to start.
    pub fn open(dir: &Path, editable_roots: bool) -> Self {
        let settings = editable_roots.then(|| Mutex::new(read_json(&dir.join(SETTINGS_FILE))));
        let state = Mutex::new(read_json(&dir.join(STATE_FILE)));
        Self {
            dir: dir.to_path_buf(),
            settings,
            state,
        }
    }

    /// The stored roots (the desktop's), as written.
    pub fn roots(&self) -> Vec<String> {
        self.settings
            .as_ref()
            .map(|s| lock(s).roots.clone())
            .unwrap_or_default()
    }

    /// Add `dir` (canonical) to the stored roots and save them; the roots
    /// as they are now.
    pub fn add_root(&self, dir: &Path) -> Result<Vec<String>, StudioError> {
        let settings = self.editable()?;
        let mut settings = lock(settings);
        let dir = dir.to_string_lossy().into_owned();
        if !settings.roots.contains(&dir) {
            settings.roots.push(dir);
            settings.roots.sort();
        }
        write_json(&self.dir.join(SETTINGS_FILE), &*settings)?;
        Ok(settings.roots.clone())
    }

    /// Remove `dir` from the stored roots and save them.
    pub fn remove_root(&self, dir: &str) -> Result<Vec<String>, StudioError> {
        let settings = self.editable()?;
        let mut settings = lock(settings);
        settings.roots.retain(|root| root != dir);
        write_json(&self.dir.join(SETTINGS_FILE), &*settings)?;
        Ok(settings.roots.clone())
    }

    fn editable(&self) -> Result<&Mutex<Settings>, StudioError> {
        self.settings.as_ref().ok_or_else(|| {
            StudioError::new(
                ErrorKind::RootsFixed,
                "this server's folders are fixed when it starts (its --root flags)",
            )
        })
    }

    /// The recent projects, newest first.
    pub fn recent(&self) -> Vec<String> {
        lock(&self.state).recent_projects.clone()
    }

    /// Put `dir` first among the recent projects. Failing to save is not
    /// the command's failure.
    pub fn touch_recent(&self, dir: &Path) {
        let dir = dir.to_string_lossy().into_owned();
        let mut state = lock(&self.state);
        state.recent_projects.retain(|d| *d != dir);
        state.recent_projects.insert(0, dir);
        state.recent_projects.truncate(RECENT_LIMIT);
        let _ = write_json(&self.dir.join(STATE_FILE), &*state);
    }

    /// Drop `dir` from the recent projects; the list as it is now.
    pub fn forget_recent(&self, dir: &str) -> Result<Vec<String>, StudioError> {
        let mut state = lock(&self.state);
        state.recent_projects.retain(|d| d != dir);
        write_json(&self.dir.join(STATE_FILE), &*state)?;
        Ok(state.recent_projects.clone())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // The values are plain data; a panic elsewhere leaves them whole.
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn read_json<T: serde::de::DeserializeOwned + Default>(path: &Path) -> T {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// Write `value` to `path` through a temporary file beside it.
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), StudioError> {
    let fail = |e: String| StudioError::new(ErrorKind::Io, format!("{}: {e}", path.display()));
    let text = serde_json::to_string_pretty(value).map_err(|e| fail(e.to_string()))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| fail(e.to_string()))?;
    }
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text + "\n").map_err(|e| fail(e.to_string()))?;
    std::fs::rename(&tmp, path).map_err(|e| fail(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_and_recents_survive_a_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let store = SettingsStore::open(tmp.path(), true);
        assert!(store.roots().is_empty());
        store.add_root(Path::new("/data/b")).unwrap();
        store.add_root(Path::new("/data/a")).unwrap();
        store.add_root(Path::new("/data/a")).unwrap();
        for i in 0..(RECENT_LIMIT + 2) {
            store.touch_recent(Path::new(&format!("/p/{i}")));
        }
        store.touch_recent(Path::new("/p/3"));

        let again = SettingsStore::open(tmp.path(), true);
        assert_eq!(again.roots(), ["/data/a", "/data/b"]);
        let recent = again.recent();
        assert_eq!(recent.len(), RECENT_LIMIT);
        assert_eq!(recent[0], "/p/3");
        assert_eq!(recent.iter().filter(|d| *d == "/p/3").count(), 1);
        assert_eq!(again.remove_root("/data/a").unwrap(), ["/data/b"]);
        assert!(
            !again
                .forget_recent("/p/3")
                .unwrap()
                .contains(&"/p/3".into())
        );
    }

    #[test]
    fn a_server_store_keeps_no_roots_and_a_broken_file_reads_empty() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(STATE_FILE), "{ not json").unwrap();
        let store = SettingsStore::open(tmp.path(), false);
        assert!(store.recent().is_empty());
        let err = store.add_root(Path::new("/x")).unwrap_err();
        assert_eq!(err.kind, ErrorKind::RootsFixed);
        assert!(!tmp.path().join(SETTINGS_FILE).exists());
    }
}
