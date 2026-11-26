//! `calib-viewer`: the app's views over one calibration run, in a browser.
//!
//! `calib-viewer <run-dir | export.json> [--port N] [--no-open] [--dist DIR]`
//! serves the built frontend (`apps/studio/dist`) and the run on `127.0.0.1` only,
//! prints the address and opens it in the default browser. The page is the
//! desktop app's own UI with its HTTP back end: Diagnose, the 3D viewer,
//! Epipolar and Depth work; what needs the desktop (projects, running a
//! calibration) is hidden.
//!
//! A run is a `vcal run -o` folder, a run the app saved in a project, or one
//! export file ([`source`]). The server reads only files under the run
//! folder and under the export's image root ([`scope`]); the command bodies
//! are the desktop app's ([`crate::ops`]).

mod http;
mod scope;
mod source;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use crate::ops;

const USAGE: &str = "\
usage: calib-viewer <run-dir | export.json> [--port N] [--no-open] [--dist DIR]

Serves a calibration run to a browser on 127.0.0.1.

  <run-dir>     a `vcal run -o` folder (calibration.json), a run the app
                saved (runs/<name>/export.json), or an export file
  --port N      the port to listen on (default: a free one)
  --no-open     do not open the browser
  --dist DIR    the built frontend (default: the app's dist/, from
                `bun run build`)";

/// The command line, parsed.
#[derive(Debug, PartialEq)]
struct Args {
    run: PathBuf,
    port: u16,
    open: bool,
    dist: PathBuf,
}

/// The frontend `bun run build` writes, next to this crate.
fn default_dist() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../dist")
}

fn parse_args(args: Vec<OsString>) -> Result<Option<Args>, String> {
    let mut run = None;
    let mut port = 0;
    let mut open = true;
    let mut dist = default_dist();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-h" | "--help") => return Ok(None),
            Some("--no-open") => open = false,
            Some("--port") => {
                let value = args.next().ok_or("--port needs a value")?;
                port = value.to_str().and_then(|v| v.parse().ok()).ok_or_else(|| {
                    format!("--port {}: not a port number", value.to_string_lossy())
                })?;
            }
            Some("--dist") => dist = args.next().ok_or("--dist needs a folder")?.into(),
            Some(flag) if flag.starts_with('-') => return Err(format!("unknown option {flag}")),
            _ if run.is_none() => run = Some(PathBuf::from(arg)),
            _ => return Err("give one run folder or export file".into()),
        }
    }
    let run = run.ok_or("give a run folder or an export file")?;
    Ok(Some(Args {
        run,
        port,
        open,
        dist,
    }))
}

/// The built frontend in `dist`, or an error that says how to build it. The
/// repository carries a placeholder `index.html` without its `assets/`.
fn check_dist(dist: &Path) -> Result<(), String> {
    if dist.join("index.html").is_file() && dist.join("assets").is_dir() {
        Ok(())
    } else {
        Err(format!(
            "the frontend is not built ({} has no index.html and assets/): \
             run `bun run build` in apps/studio/, or start the viewer with \
             `bun run viewer -- <run>`, which builds it first",
            dist.display()
        ))
    }
}

/// The `calib-viewer` program: `args` without the program name.
pub fn main(args: Vec<OsString>) -> ExitCode {
    let args = match parse_args(args) {
        Ok(Some(args)) => args,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("calib-viewer: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match serve(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("calib-viewer: {e}");
            ExitCode::FAILURE
        }
    }
}

fn serve(args: &Args) -> Result<(), String> {
    check_dist(&args.dist)?;
    let run = source::find_run(&args.run)?;
    // Read the export now: a file the viewer cannot show fails here, before
    // a browser opens on it.
    let loaded = ops::read_export(&run.export)?;
    let images = loaded.image_root;
    let viewer = http::Viewer::start(&run, images.clone(), &args.dist, args.port)?;
    let url = viewer.url();
    println!("calib-viewer: {} at {url}", run.export.display());
    match images.as_deref().map(dunce::canonicalize) {
        Some(Ok(root)) => println!("  images from {}", root.display()),
        Some(Err(e)) => println!("  the image root does not resolve ({e}): no images"),
        None => println!("  this export has no image list: residuals and statistics only"),
    }
    println!("  Ctrl-C stops the server.");
    if args.open && open_browser(&url).is_err() {
        println!("  could not open a browser; open {url} yourself");
    }
    viewer.wait();
    Ok(())
}

/// Open `url` in the default browser.
fn open_browser(url: &str) -> std::io::Result<()> {
    use std::process::{Command, Stdio};
    #[cfg(target_os = "macos")]
    let mut command = Command::new("open");
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut command = Command::new("xdg-open");
    command
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(drop)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        parse_args(args.iter().map(OsString::from).collect())
    }

    #[test]
    fn arguments_parse_with_defaults() {
        let args = parse(&["runs/run-001"]).unwrap().unwrap();
        assert_eq!(args.run, PathBuf::from("runs/run-001"));
        assert_eq!((args.port, args.open), (0, true));
        assert_eq!(args.dist, default_dist());

        let args = parse(&["--port", "8765", "out", "--no-open", "--dist", "d"])
            .unwrap()
            .unwrap();
        assert_eq!((args.port, args.open), (8765, false));
        assert_eq!(args.dist, PathBuf::from("d"));

        assert_eq!(parse(&["--help"]).unwrap(), None);
        assert!(parse(&[]).is_err());
        assert!(parse(&["a", "b"]).is_err());
        assert!(parse(&["a", "--port", "http"]).is_err());
        assert!(parse(&["a", "--verbose"]).is_err());
    }

    #[test]
    fn an_unbuilt_frontend_says_how_to_build_it() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("index.html"), "placeholder").unwrap();
        assert!(
            check_dist(tmp.path())
                .unwrap_err()
                .contains("bun run build")
        );
        std::fs::create_dir(tmp.path().join("assets")).unwrap();
        assert!(check_dist(tmp.path()).is_ok());
    }
}
