//! The command line and the config file, merged into one [`Config`].

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use serde::Deserialize;

/// The default address: this machine only.
pub const DEFAULT_BIND: &str = "127.0.0.1:8757";

/// Calibration Studio in a browser: serves the app and the folders under its
/// roots to the browsers that reach it.
///
/// Bound to a loopback address it needs no token unless one is configured;
/// on any other address a token is always required: it is read from
/// --token-file, $CALIBRATION_STUDIO_TOKEN or the config file's
/// `token_file`, else generated once into <state-dir>/token. The start-up
/// line prints the address with the token in its fragment.
#[derive(Debug, Parser)]
#[command(
    name = "calibration-studio-server",
    version,
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
    #[command(flatten)]
    pub serve: ServeArgs,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Print the stored token's address fragment (`#token=…`), generating
    /// the token when there is none.
    Token {
        /// Replace the stored token: every session signed with the old one
        /// ends.
        #[arg(long)]
        rotate: bool,
        #[arg(long, value_name = "DIR")]
        state_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Args, Default)]
pub struct ServeArgs {
    /// A project folder to serve alone (with its dataset folder) and open.
    #[arg(value_name = "PROJECT")]
    pub project: Option<PathBuf>,
    /// A root folder: the server reads and writes only under its roots.
    /// Repeatable.
    #[arg(long = "root", value_name = "DIR")]
    pub roots: Vec<PathBuf>,
    /// The address to listen on [default: 127.0.0.1:8757].
    #[arg(long, value_name = "ADDR:PORT")]
    pub bind: Option<SocketAddr>,
    /// A TOML file with `roots`, `bind`, `token_file`, `allowed_hosts`,
    /// `public_origin`, `state_dir`, `cache_dir`; flags win.
    #[arg(long, value_name = "FILE")]
    pub config: Option<PathBuf>,
    /// The file holding the token.
    #[arg(long, value_name = "FILE")]
    pub token_file: Option<PathBuf>,
    /// Where the token, the recent projects and the samples go
    /// [default: $XDG_STATE_HOME/calibration-studio].
    #[arg(long, value_name = "DIR")]
    pub state_dir: Option<PathBuf>,
    /// Where derived images go [default: $XDG_CACHE_HOME/calibration-studio].
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,
    /// Serve the frontend from this folder instead of the embedded one.
    #[arg(long, value_name = "DIR")]
    pub dist: Option<PathBuf>,
    /// Another Host name the server answers (a reverse proxy's). Repeatable.
    #[arg(long = "allowed-host", value_name = "HOST")]
    pub allowed_hosts: Vec<String>,
    /// The origin browsers see behind a reverse proxy
    /// (`https://calib.example.lan`): accepted as Origin, and https makes
    /// the session cookie Secure.
    #[arg(long, value_name = "URL")]
    pub public_origin: Option<String>,
    /// Open the address in the default browser.
    #[arg(long)]
    pub open: bool,
}

/// The config file.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    #[serde(default)]
    pub roots: Vec<PathBuf>,
    pub bind: Option<SocketAddr>,
    pub token_file: Option<PathBuf>,
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    pub public_origin: Option<String>,
    pub state_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

impl ConfigFile {
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Where the frontend comes from.
#[derive(Debug, Clone)]
pub enum Dist {
    /// Inside the binary.
    Embedded,
    /// A folder (`--dist`), for development.
    Folder(PathBuf),
}

/// Everything the server runs with.
#[derive(Debug, Clone)]
pub struct Config {
    pub roots: Vec<PathBuf>,
    pub bind: SocketAddr,
    /// The token every request needs; `None` only on a loopback bind with
    /// no token configured.
    pub token: Option<String>,
    pub state_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub dist: Dist,
    pub allowed_hosts: Vec<String>,
    pub public_origin: Option<String>,
    /// The project to open in the page (`PROJECT`).
    pub project: Option<PathBuf>,
    pub open: bool,
}

/// `$XDG_STATE_HOME/calibration-studio` (Linux), else the local data
/// folder's.
pub fn default_state_dir() -> PathBuf {
    dirs::state_dir()
        .or_else(dirs::data_local_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("calibration-studio")
}

/// `$XDG_CACHE_HOME/calibration-studio`.
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("calibration-studio")
}

impl Config {
    /// The flags and the config file, merged: flags win, lists add up.
    pub fn from_args(args: ServeArgs) -> Result<Self, String> {
        let file = match &args.config {
            Some(path) => ConfigFile::read(path)?,
            None => ConfigFile::default(),
        };
        let bind = args
            .bind
            .or(file.bind)
            .unwrap_or_else(|| DEFAULT_BIND.parse().expect("the default address parses"));
        let state_dir = args
            .state_dir
            .or(file.state_dir)
            .unwrap_or_else(default_state_dir);
        let cache_dir = args
            .cache_dir
            .or(file.cache_dir)
            .unwrap_or_else(default_cache_dir);
        let mut roots = args.roots;
        roots.extend(file.roots);
        if let Some(project) = &args.project {
            let info = studio_core::project::open(project).map_err(|e| {
                format!(
                    "{}: {} (pass a project folder, or serve folders with --root)",
                    project.display(),
                    e.message
                )
            })?;
            if roots.is_empty() {
                roots.push(PathBuf::from(&info.dir));
                roots.push(PathBuf::from(&info.dataset_dir));
            }
        }
        if roots.is_empty() {
            return Err("give at least one --root folder (or a project folder)".into());
        }
        let roots = roots
            .into_iter()
            .map(|root| {
                dunce::canonicalize(&root)
                    .ok()
                    .filter(|dir| dir.is_dir())
                    .ok_or_else(|| format!("--root {}: not a folder", root.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token = crate::token::resolve(
            args.token_file.as_deref().or(file.token_file.as_deref()),
            std::env::var("CALIBRATION_STUDIO_TOKEN").ok().as_deref(),
            &state_dir,
            bind.ip().is_loopback(),
        )?;
        let mut allowed_hosts = args.allowed_hosts;
        allowed_hosts.extend(file.allowed_hosts);
        let public_origin = args
            .public_origin
            .or(file.public_origin)
            .map(|o| o.trim_end_matches('/').to_owned());
        Ok(Self {
            roots,
            bind,
            token,
            state_dir,
            cache_dir,
            dist: args.dist.map_or(Dist::Embedded, Dist::Folder),
            allowed_hosts,
            public_origin,
            project: args.project.and_then(|p| dunce::canonicalize(p).ok()),
            open: args.open,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_help_parses_and_flags_win_over_the_file() {
        use clap::CommandFactory;
        Cli::command().debug_assert();

        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let file = tmp.path().join("server.toml");
        std::fs::write(
            &file,
            format!(
                "roots = [{:?}]\nbind = \"127.0.0.1:9000\"\nallowed_hosts = [\"calib.lan\"]\n",
                b.display().to_string()
            ),
        )
        .unwrap();
        let cli = Cli::parse_from([
            "calibration-studio-server",
            "--root",
            a.to_str().unwrap(),
            "--config",
            file.to_str().unwrap(),
            "--bind",
            "127.0.0.1:9001",
            "--state-dir",
            tmp.path().join("state").to_str().unwrap(),
        ]);
        let config = Config::from_args(cli.serve).unwrap();
        assert_eq!(config.bind.port(), 9001);
        assert_eq!(config.roots.len(), 2);
        assert_eq!(config.allowed_hosts, ["calib.lan"]);
        // Loopback without a configured token: none.
        assert!(config.token.is_none());

        let missing = Cli::parse_from(["calibration-studio-server", "--root", "/no/such/dir"]);
        assert!(Config::from_args(missing.serve).is_err());
        let none = Cli::parse_from(["calibration-studio-server"]);
        assert!(
            Config::from_args(none.serve)
                .unwrap_err()
                .contains("--root")
        );
    }

    #[test]
    fn a_non_loopback_bind_always_has_a_token() {
        let tmp = tempfile::tempdir().unwrap();
        let cli = Cli::parse_from([
            "calibration-studio-server",
            "--root",
            tmp.path().to_str().unwrap(),
            "--bind",
            "0.0.0.0:0",
            "--state-dir",
            tmp.path().join("state").to_str().unwrap(),
        ]);
        let config = Config::from_args(cli.serve).unwrap();
        let token = config.token.expect("a token is generated");
        assert!(token.len() >= 43);
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("state/token"))
                .unwrap()
                .trim(),
            token
        );
    }
}
