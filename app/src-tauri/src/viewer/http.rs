//! The viewer's localhost HTTP server.
//!
//! Routes:
//!
//! - `GET /api/session` — the run being served (`Session`).
//! - `POST /api/<command>` — a viewer command, by the name and with the
//!   camelCase argument object of its Tauri command, answered with the same
//!   JSON (or PNG bytes for `load_undistorted_image`). A failure is the
//!   command's error string as JSON, with a 4xx status.
//! - `GET /api/file?path=<absolute path>` — a file under the run folder or
//!   the export's image root, the browser's stand-in for the asset protocol.
//! - `GET /<path>` — the built frontend; `index.html` is marked with
//!   `<meta name="calib-backend" content="http">` so the page picks the
//!   HTTP back end.
//!
//! The server binds 127.0.0.1 only, accepts only its own `Host` (so a page
//! from another origin cannot reach it through DNS rebinding) and refuses a
//! `POST` that names a foreign `Origin`. It sends no CORS headers.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server};
use vision_calibration::core::PixelRect;

use super::scope::{FileScope, ScopeError};
use super::source::{RunLayout, RunSource};
use crate::disparity::DisparityParams;
use crate::export_cache::ExportCache;
use crate::ops;

/// Requests are served by this many threads, so an image keeps loading
/// while an undistortion or a disparity map is computed.
const WORKERS: usize = 4;
/// The largest request body read (an export sent to `set_active_export`).
const MAX_BODY: u64 = 256 * 1024 * 1024;
/// What marks the page as served by this back end.
const BACKEND_META: &str = r#"<meta name="calib-backend" content="http" />"#;

/// What `GET /api/session` answers: the run the viewer serves.
#[derive(Debug, Clone, Serialize)]
pub struct Session {
    /// The run folder's name, for the page's header.
    pub name: String,
    pub layout: RunLayout,
    /// The run folder (canonical, `/`-separated).
    pub run_dir: String,
    /// The export file to load (canonical, `/`-separated).
    pub export_path: String,
}

impl Session {
    pub fn of(run: &RunSource) -> Self {
        let name = run
            .dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| slash(&run.dir));
        Self {
            name,
            layout: run.layout,
            run_dir: slash(&run.dir),
            export_path: slash(&run.export),
        }
    }
}

fn slash(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Everything a request handler reads.
struct State {
    session: Session,
    /// The run folder and the export's image root.
    files: FileScope,
    /// The built frontend.
    dist: PathBuf,
    dist_scope: FileScope,
    cache: ExportCache,
    /// The `Host` values the server answers to.
    hosts: [String; 2],
    /// The `Origin` values a `POST` may carry.
    origins: [String; 2],
}

/// A running viewer server.
pub struct Viewer {
    /// Kept to stop the server in tests; the workers hold it otherwise.
    #[cfg(test)]
    server: Arc<Server>,
    workers: Vec<JoinHandle<()>>,
    port: u16,
}

impl Viewer {
    /// Serve `run` (whose export's image root is `images`, if it has one)
    /// and the frontend in `dist` on `127.0.0.1:port` (`0`: a free port).
    pub fn start(
        run: &RunSource,
        images: Option<PathBuf>,
        dist: &Path,
        port: u16,
    ) -> Result<Self, String> {
        let server = Server::http(("127.0.0.1", port))
            .map_err(|e| format!("cannot listen on 127.0.0.1:{port}: {e}"))?;
        let port = server
            .server_addr()
            .to_ip()
            .map(|addr| addr.port())
            .ok_or("the server is not listening on an IP address")?;
        let dist = dunce::canonicalize(dist).map_err(|e| format!("{}: {e}", dist.display()))?;
        let mut roots = vec![run.dir.clone()];
        roots.extend(images);
        let state = Arc::new(State {
            session: Session::of(run),
            files: FileScope::new(roots),
            dist_scope: FileScope::new([dist.clone()]),
            dist,
            cache: ExportCache::new(),
            hosts: [format!("127.0.0.1:{port}"), format!("localhost:{port}")],
            origins: [
                format!("http://127.0.0.1:{port}"),
                format!("http://localhost:{port}"),
            ],
        });
        let server = Arc::new(server);
        let workers = (0..WORKERS)
            .map(|_| {
                let server = Arc::clone(&server);
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    while let Ok(request) = server.recv() {
                        handle(&state, request);
                    }
                })
            })
            .collect();
        Ok(Self {
            #[cfg(test)]
            server,
            workers,
            port,
        })
    }

    #[cfg(test)]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// The page's address.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    /// Serve until the process ends.
    pub fn wait(self) {
        for worker in self.workers {
            let _ = worker.join();
        }
    }

    /// Stop serving and wait for the workers to finish.
    #[cfg(test)]
    pub fn stop(self) {
        for _ in &self.workers {
            self.server.unblock();
        }
        for worker in self.workers {
            let _ = worker.join();
        }
    }
}

/// A response: its status, type and body, plus what to log on failure.
struct Reply {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
    no_store: bool,
}

impl Reply {
    fn json<T: Serialize>(value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(body) => Self {
                status: 200,
                content_type: "application/json",
                body,
                no_store: true,
            },
            Err(e) => Self::error(500, format!("serialize the response: {e}")),
        }
    }

    /// A failure: the message as a JSON string, as a rejected Tauri
    /// command's error reaches the frontend.
    fn error(status: u16, message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(&message).unwrap_or_default(),
            no_store: true,
        }
    }

    fn bytes(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type,
            body,
            no_store: false,
        }
    }

    fn into_response(self) -> Response<std::io::Cursor<Vec<u8>>> {
        let mut response = Response::from_data(self.body).with_status_code(self.status);
        let mut headers = vec![
            ("Content-Type", self.content_type),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
        ];
        if self.no_store {
            headers.push(("Cache-Control", "no-store"));
        }
        for (name, value) in headers {
            if let Ok(header) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
                response.add_header(header);
            }
        }
        response
    }
}

fn handle(state: &State, mut request: Request) {
    let reply = route(state, &mut request).unwrap_or_else(|e| e);
    if reply.status >= 400 {
        let path = request.url().split('?').next().unwrap_or_default();
        let message = serde_json::from_slice::<String>(&reply.body).unwrap_or_default();
        eprintln!("{} {path}: {} {message}", request.method(), reply.status);
    }
    let _ = request.respond(reply.into_response());
}

fn header<'r>(request: &'r Request, name: &'static str) -> Option<&'r str> {
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str())
}

fn route(state: &State, request: &mut Request) -> Result<Reply, Reply> {
    if !header(request, "Host").is_some_and(|host| state.hosts.iter().any(|h| h == host)) {
        return Err(Reply::error(403, "unexpected Host header"));
    }
    let url = request.url().to_owned();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    match (request.method(), path) {
        (Method::Get, "/api/session") => Ok(Reply::json(&state.session)),
        (Method::Get, "/api/file") => {
            let requested = query_param(query, "path")
                .ok_or_else(|| Reply::error(400, "`path` is missing or not UTF-8"))?;
            let file = scoped(&state.files, &requested)?;
            let bytes = std::fs::read(&file)
                .map_err(|e| Reply::error(404, format!("read {}: {e}", file.display())))?;
            Ok(Reply::bytes(content_type(&file), bytes))
        }
        (Method::Post, _) if path.starts_with("/api/") => {
            if let Some(origin) = header(request, "Origin")
                && !state.origins.iter().any(|o| o == origin)
            {
                return Err(Reply::error(403, format!("foreign origin {origin}")));
            }
            let mut body = Vec::new();
            request
                .as_reader()
                .take(MAX_BODY)
                .read_to_end(&mut body)
                .map_err(|e| Reply::error(400, format!("read the request body: {e}")))?;
            command(state, &path["/api/".len()..], &body)
        }
        (_, _) if path.starts_with("/api/") => Err(Reply::error(405, "method not allowed")),
        (Method::Get, _) => serve_static(state, path),
        _ => Err(Reply::error(405, "method not allowed")),
    }
}

/// `path` resolved inside `scope`: a 403 outside it, a 404 for no file.
fn scoped(scope: &FileScope, path: &str) -> Result<PathBuf, Reply> {
    scope.resolve(Path::new(path)).map_err(|e| match e {
        ScopeError::NotFound(_) => Reply::error(404, e.message()),
        ScopeError::Forbidden(_) => Reply::error(403, e.message()),
    })
}

/// A file of the built frontend; `/` is `index.html`, marked as served by
/// this back end.
fn serve_static(state: &State, path: &str) -> Result<Reply, Reply> {
    let relative = percent_decode(path.trim_start_matches('/'))
        .ok_or_else(|| Reply::error(400, "the path is not UTF-8"))?;
    let relative = if relative.is_empty() {
        "index.html".to_owned()
    } else {
        relative
    };
    // Plain relative segments only; the scope check below also catches
    // symbolic links out of `dist`.
    if relative
        .split('/')
        .any(|segment| segment.is_empty() || segment == ".." || segment.contains(['\\', ':']))
    {
        return Err(Reply::error(404, "not found"));
    }
    let file = scoped(&state.dist_scope, &slash(&state.dist.join(&relative)))?;
    if file.is_dir() {
        return Err(Reply::error(404, "not found"));
    }
    let bytes =
        std::fs::read(&file).map_err(|e| Reply::error(404, format!("{}: {e}", file.display())))?;
    if relative == "index.html" {
        let page = String::from_utf8_lossy(&bytes).replacen(
            "<head>",
            &format!("<head>\n    {BACKEND_META}"),
            1,
        );
        let mut reply = Reply::bytes("text/html; charset=utf-8", page.into_bytes());
        reply.no_store = true;
        return Ok(reply);
    }
    Ok(Reply::bytes(content_type(&file), bytes))
}

fn content_type(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match ext.as_str() {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "tif" | "tiff" => "image/tiff",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "txt" | "toml" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// The value of `name` in a query string, percent-decoded.
fn query_param(query: &str, name: &str) -> Option<String> {
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(key, _)| *key == name)
        .and_then(|(_, value)| percent_decode(value))
}

/// `%XX` escapes decoded (`+` is a literal plus, as `encodeURIComponent`
/// writes a space as `%20`). `None` for a broken escape or non-UTF-8.
fn percent_decode(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = text.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

// The argument objects of the commands, as the frontend sends them (the
// Tauri command parameters, camelCase).

#[derive(Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
struct ExportArgs {
    export: serde_json::Value,
}

#[derive(Deserialize)]
struct UndistortedImageArgs {
    path: String,
    camera: usize,
    roi: Option<PixelRect>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UndistortPointsArgs {
    camera: usize,
    points_px: Vec<[f64; 2]>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OverlayArgs {
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UndistortedOverlayArgs {
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
    image_width_b: f64,
    image_height_b: f64,
}

#[derive(Deserialize)]
struct LabelsArgs {
    labels: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProjectionArgs {
    camera: usize,
    image_width: u32,
    image_height: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DisparityArgs {
    path_a: String,
    path_b: String,
    cam_a: usize,
    cam_b: usize,
    pose: usize,
    params: DisparityParams,
}

fn args<T: DeserializeOwned>(body: &[u8]) -> Result<T, Reply> {
    serde_json::from_slice(body).map_err(|e| Reply::error(400, format!("invalid arguments: {e}")))
}

/// A command's result as a reply: its JSON, or its error with 422.
fn answer<T: Serialize>(result: Result<T, String>) -> Result<Reply, Reply> {
    result
        .map(|value| Reply::json(&value))
        .map_err(|e| Reply::error(422, e))
}

/// Run the viewer command `name`. The commands the browser does not offer
/// (projects, calibration runs, dialogs) are 404s.
fn command(state: &State, name: &str, body: &[u8]) -> Result<Reply, Reply> {
    let cache = &state.cache;
    match name {
        "load_export" => {
            let a: PathArgs = args(body)?;
            let path = scoped(&state.files, &a.path)?;
            answer(ops::read_export(&path).map(|loaded| loaded.result))
        }
        "set_active_export" => {
            let a: ExportArgs = args(body)?;
            cache.set(a.export);
            answer(Ok(()))
        }
        "load_text_file" => {
            let a: PathArgs = args(body)?;
            answer(ops::read_text_file(&scoped(&state.files, &a.path)?))
        }
        "load_undistorted_image" => {
            let a: UndistortedImageArgs = args(body)?;
            let path = scoped(&state.files, &a.path)?;
            ops::load_undistorted_image(cache, &path, a.camera, a.roi)
                .map(|png| Reply::bytes("image/png", png))
                .map_err(|e| Reply::error(422, e))
        }
        "undistort_points" => {
            let a: UndistortPointsArgs = args(body)?;
            answer(ops::undistort_points(cache, a.camera, a.points_px))
        }
        "compute_epipolar_overlay" => {
            let a: OverlayArgs = args(body)?;
            answer(ops::compute_epipolar_overlay(
                cache, a.cam_a, a.cam_b, a.point_px,
            ))
        }
        "compute_epipolar_overlay_undistorted" => {
            let a: UndistortedOverlayArgs = args(body)?;
            answer(ops::compute_epipolar_overlay_undistorted(
                cache,
                a.cam_a,
                a.cam_b,
                a.point_px,
                [a.image_width_b, a.image_height_b],
            ))
        }
        "camera_parameter_values" => {
            let a: LabelsArgs = args(body)?;
            answer(ops::camera_parameter_values(cache, &a.labels))
        }
        "projection_uncertainty" => {
            let a: ProjectionArgs = args(body)?;
            answer(ops::projection_uncertainty(
                cache,
                a.camera,
                [a.image_width, a.image_height],
            ))
        }
        "compute_disparity" => {
            let a: DisparityArgs = args(body)?;
            let path_a = scoped(&state.files, &a.path_a)?;
            let path_b = scoped(&state.files, &a.path_b)?;
            answer(ops::compute_disparity(
                cache,
                [&path_a, &path_b],
                a.cam_a,
                a.cam_b,
                a.pose,
                &a.params,
            ))
        }
        other => Err(Reply::error(
            404,
            format!("`{other}` is not available in the browser viewer"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viewer::source::find_run;
    use std::io::Write;
    use std::net::TcpStream;

    struct Reply {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    }

    impl Reply {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.body).into_owned()
        }
        fn json(&self) -> serde_json::Value {
            serde_json::from_slice(&self.body).unwrap()
        }
    }

    fn send(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> Reply {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut head = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
        if !headers.iter().any(|(name, _)| *name == "Host") {
            head.push_str(&format!("Host: 127.0.0.1:{port}\r\n"));
        }
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(body.as_bytes()).unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).unwrap();
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status = head.split(' ').nth(1).unwrap().parse().unwrap();
        let content_type = head
            .lines()
            .find_map(|l| {
                l.split_once(':')
                    .filter(|(k, _)| k.eq_ignore_ascii_case("content-type"))
            })
            .map(|(_, v)| v.trim().to_owned())
            .unwrap_or_default();
        Reply {
            status,
            content_type,
            body: raw[split + 4..].to_vec(),
        }
    }

    fn get(port: u16, path: &str) -> Reply {
        send(port, "GET", path, &[], "")
    }

    fn post(port: u16, command: &str, body: serde_json::Value) -> Reply {
        send(
            port,
            "POST",
            &format!("/api/{command}"),
            &[("Content-Type", "application/json")],
            &body.to_string(),
        )
    }

    fn encode(path: &str) -> String {
        path.bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                    (b as char).to_string()
                }
                _ => format!("%{b:02X}"),
            })
            .collect()
    }

    /// The committed fixture run (a two-camera rig with images), served
    /// with a stand-in frontend.
    fn start() -> (Viewer, RunSource, tempfile::TempDir) {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../e2e/fixtures/viewer-run");
        let run = find_run(&fixture).unwrap();
        let loaded = ops::read_export(&run.export).unwrap();
        let dist = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dist.path().join("assets")).unwrap();
        std::fs::write(
            dist.path().join("index.html"),
            "<!doctype html><html><head><title>t</title></head><body></body></html>",
        )
        .unwrap();
        std::fs::write(dist.path().join("assets/app.js"), "export {};").unwrap();
        let viewer = Viewer::start(&run, loaded.image_root, dist.path(), 0).unwrap();
        (viewer, run, dist)
    }

    #[test]
    fn the_page_is_marked_and_the_session_names_the_run() {
        let (viewer, run, _dist) = start();
        let port = viewer.port();

        let page = get(port, "/");
        assert_eq!(page.status, 200);
        assert!(page.text().contains(BACKEND_META), "{}", page.text());
        let script = get(port, "/assets/app.js");
        assert_eq!(script.status, 200);
        assert!(script.content_type.starts_with("text/javascript"));
        assert_eq!(get(port, "/assets/missing.js").status, 404);
        assert_eq!(get(port, "/../Cargo.toml").status, 404);
        assert_eq!(get(port, "/assets/%2e%2e/%2e%2e/Cargo.toml").status, 404);

        let session = get(port, "/api/session").json();
        assert_eq!(session["name"], "viewer-run");
        assert_eq!(session["layout"], "project_run");
        assert_eq!(session["export_path"], slash(&run.export));
        viewer.stop();
    }

    #[test]
    fn commands_answer_as_the_tauri_commands_do() {
        let (viewer, run, _dist) = start();
        let port = viewer.port();

        // Before an export is active, a cached command fails with the
        // command's own error string.
        let early = post(
            port,
            "undistort_points",
            serde_json::json!({"camera": 0, "pointsPx": [[1.0, 2.0]]}),
        );
        assert_eq!(early.status, 422);
        assert_eq!(early.json(), "no export loaded yet");

        let loaded = post(
            port,
            "load_export",
            serde_json::json!({"path": slash(&run.export)}),
        );
        assert_eq!(loaded.status, 200, "{}", loaded.text());
        let loaded = loaded.json();
        assert_eq!(loaded["export_dir"], slash(&run.dir));
        let export = loaded["export"].clone();
        assert_eq!(export["kind"], "rig_extrinsics");

        let set = post(
            port,
            "set_active_export",
            serde_json::json!({"export": export}),
        );
        assert_eq!((set.status, set.json()), (200, serde_json::Value::Null));

        // A distortion-free camera maps pixels to themselves.
        let points = post(
            port,
            "undistort_points",
            serde_json::json!({"camera": 1, "pointsPx": [[10.0, 20.0]]}),
        );
        assert_eq!(points.status, 200, "{}", points.text());
        let px = points.json();
        assert!((px[0][0].as_f64().unwrap() - 10.0).abs() < 1e-6, "{px}");

        let values = post(
            port,
            "camera_parameter_values",
            serde_json::json!({"labels": ["cam1.fx", "cam0.cy"]}),
        );
        assert_eq!(values.json(), serde_json::json!([200.0, 59.5]));

        let map = post(
            port,
            "projection_uncertainty",
            serde_json::json!({"camera": 0, "imageWidth": 160, "imageHeight": 120}),
        );
        assert_eq!(map.status, 200, "{}", map.text());

        let overlay = post(
            port,
            "compute_epipolar_overlay_undistorted",
            serde_json::json!({
                "camA": 0, "camB": 1, "pointPx": [80.0, 60.0], "imageWidthB": 160.0, "imageHeightB": 120.0
            }),
        );
        assert_eq!(overlay.status, 200, "{}", overlay.text());

        let image = run.dir.join("images/cam0_view0.png");
        let png = post(
            port,
            "load_undistorted_image",
            serde_json::json!({"path": slash(&image), "camera": 0, "roi": null}),
        );
        assert_eq!(png.status, 200, "{}", png.text());
        assert_eq!(png.content_type, "image/png");
        assert!(png.body.starts_with(b"\x89PNG"));

        let bad = post(
            port,
            "undistort_points",
            serde_json::json!({"camera": "zero"}),
        );
        assert_eq!(bad.status, 400);
        let desktop = post(port, "run_calibration_cmd", serde_json::json!({}));
        assert_eq!(desktop.status, 404);
        viewer.stop();
    }

    #[test]
    fn files_are_served_only_from_the_run_and_its_images() {
        let (viewer, run, _dist) = start();
        let port = viewer.port();
        let file = |path: &Path| get(port, &format!("/api/file?path={}", encode(&slash(path))));

        let image = file(&run.dir.join("images/cam1_view0.png"));
        assert_eq!(image.status, 200);
        assert_eq!(image.content_type, "image/png");
        assert_eq!(file(&run.dir.join("export.json")).status, 200);

        // `..` out of the run folder, an absolute path elsewhere, a
        // relative path: all refused.
        assert_eq!(
            file(&run.dir.join("../viewer-run-vcal/calibration.json")).status,
            403
        );
        assert_eq!(
            file(&run.dir.join("images/../../../../src-tauri/Cargo.toml")).status,
            403
        );
        assert_eq!(
            file(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("Cargo.toml")
                    .as_path()
            )
            .status,
            403
        );
        assert_eq!(
            get(port, "/api/file?path=images%2Fcam0_view0.png").status,
            403
        );
        assert_eq!(file(&run.dir.join("images/missing.png")).status, 404);
        // A command's path is scoped the same way.
        let outside = post(
            port,
            "load_text_file",
            serde_json::json!({
                "path": slash(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            }),
        );
        assert_eq!(outside.status, 403);
        viewer.stop();
    }

    #[test]
    fn foreign_hosts_and_origins_are_refused() {
        let (viewer, _run, _dist) = start();
        let port = viewer.port();
        let rebinding = send(
            port,
            "GET",
            "/api/session",
            &[("Host", "evil.example:80")],
            "",
        );
        assert_eq!(rebinding.status, 403);
        let localhost = format!("localhost:{port}");
        let ok = send(port, "GET", "/api/session", &[("Host", &localhost)], "");
        assert_eq!(ok.status, 200);
        let foreign = send(
            port,
            "POST",
            "/api/set_active_export",
            &[("Origin", "http://evil.example")],
            r#"{"export": {}}"#,
        );
        assert_eq!(foreign.status, 403);
        let own = format!("http://127.0.0.1:{port}");
        let same = send(
            port,
            "POST",
            "/api/set_active_export",
            &[("Origin", &own)],
            r#"{"export": {}}"#,
        );
        assert_eq!(same.status, 200);
        viewer.stop();
    }

    #[test]
    fn percent_decoding_reads_escapes_and_refuses_broken_ones() {
        assert_eq!(percent_decode("%2Fa%20b+c").as_deref(), Some("/a b+c"));
        assert_eq!(percent_decode("%2"), None);
        assert_eq!(percent_decode("%ZZ"), None);
        assert_eq!(query_param("x=1&path=%2Fp", "path").as_deref(), Some("/p"));
        assert_eq!(query_param("x=1", "path"), None);
    }
}
