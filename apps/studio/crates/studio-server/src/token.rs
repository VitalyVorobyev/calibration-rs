//! The session token: 32 random bytes, base64url, from a file, the
//! environment, or generated once into `<state-dir>/token` (mode 0600).
//! There is no `--token <value>` flag: it would show in `ps`.

use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

/// The stored token's file name in the state folder.
pub const TOKEN_FILE: &str = "token";

/// A new random token.
pub fn generate() -> Result<String, String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| format!("no randomness for a token: {e}"))?;
    Ok(URL_SAFE_NO_PAD.encode(bytes))
}

/// The token the server runs with, the first of: the file `token_file`, the
/// value of `$CALIBRATION_STUDIO_TOKEN` (`env`), and — unless the bind is a
/// loopback address — the token stored in `state_dir` (generated when there
/// is none). `None` only on a loopback bind with nothing configured.
pub fn resolve(
    token_file: Option<&Path>,
    env: Option<&str>,
    state_dir: &Path,
    loopback: bool,
) -> Result<Option<String>, String> {
    if let Some(file) = token_file {
        return read(file).map(Some);
    }
    if let Some(value) = env.map(str::trim).filter(|v| !v.is_empty()) {
        return checked(value.to_owned(), "$CALIBRATION_STUDIO_TOKEN").map(Some);
    }
    if loopback {
        return Ok(None);
    }
    stored(state_dir, false).map(Some)
}

/// The token in `state_dir`, generated (or replaced, with `rotate`) and
/// written with owner-only permissions.
pub fn stored(state_dir: &Path, rotate: bool) -> Result<String, String> {
    let file = state_dir.join(TOKEN_FILE);
    if !rotate && file.is_file() {
        return read(&file);
    }
    let token = generate()?;
    std::fs::create_dir_all(state_dir).map_err(|e| format!("{}: {e}", state_dir.display()))?;
    write_private(&file, &token).map_err(|e| format!("{}: {e}", file.display()))?;
    Ok(token)
}

fn read(file: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    checked(text.trim().to_owned(), &file.display().to_string())
}

/// A token long enough to be one: at least 16 characters.
fn checked(token: String, source: &str) -> Result<String, String> {
    if token.len() < 16 {
        return Err(format!("{source}: a token has at least 16 characters"));
    }
    Ok(token)
}

#[cfg(unix)]
fn write_private(file: &Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(file)?;
    out.write_all(format!("{token}\n").as_bytes())
}

#[cfg(not(unix))]
fn write_private(file: &Path, token: &str) -> std::io::Result<()> {
    std::fs::write(file, format!("{token}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_source_wins_and_loopback_needs_none() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("t");
        std::fs::write(&file, "from-the-file-0123456789\n").unwrap();
        let state = tmp.path().join("state");
        assert_eq!(
            resolve(Some(&file), Some("from-the-environment-01"), &state, false).unwrap(),
            Some("from-the-file-0123456789".into())
        );
        assert_eq!(
            resolve(None, Some("from-the-environment-01"), &state, true).unwrap(),
            Some("from-the-environment-01".into())
        );
        assert_eq!(resolve(None, None, &state, true).unwrap(), None);
        assert!(!state.exists());
        let generated = resolve(None, None, &state, false).unwrap().unwrap();
        assert_eq!(generated.len(), 43);
        assert_eq!(
            resolve(None, None, &state, false).unwrap().unwrap(),
            generated
        );
        assert_ne!(stored(&state, true).unwrap(), generated);
        assert!(resolve(None, Some("short"), &state, false).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn the_stored_token_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        stored(tmp.path(), false).unwrap();
        let mode = std::fs::metadata(tmp.path().join(TOKEN_FILE))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}
