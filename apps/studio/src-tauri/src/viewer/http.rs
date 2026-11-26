//! The viewer's localhost HTTP server.
//!
//! Routes:
//!
//! - `GET /api/session` — the run being served (`ViewerSession`).
//! - `POST /api/cmd/<name>` — a command of studio-core's table, one of the
//!   read-only ones this viewer offers ([`COMMANDS`]), with its JSON
//!   argument object; answered with its JSON reply, or with the
//!   `StudioError` envelope and a status from its kind.
//! - `GET /api/r/<name>?q=<json>` — a resource of the table ([`RESOURCES`]),
//!   `q` its query object, percent-encoded: a source image, an undistorted
//!   view, a blob; with `ETag`/`If-None-Match` and single `Range`s.
//! - `GET /api/events` — the event stream (server-sent events), resumed
//!   after `Last-Event-ID` or `?after=`; the disparity job reports on it.
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
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::Serialize;
use studio_core::api::{Api, Conditions, Dirs, ErrorKind, Kind, ServerEvent, StudioError};
use studio_core::run_source::{RunSource, ViewerSession, slash};
use studio_core::scope::{FileScope, RootScope, ScopeError};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

/// Requests are served by this many threads, so an image keeps loading
/// while an undistortion or a disparity map is computed.
const WORKERS: usize = 4;
/// The largest request body read (an export sent to `set_active_export`).
const MAX_BODY: u64 = 256 * 1024 * 1024;
/// What marks the page as served by this back end.
const BACKEND_META: &str = r#"<meta name="calib-backend" content="http" />"#;

/// The commands of the table the browser viewer answers: the viewers'
/// own, which read the run's files and the active export, the disparity job
/// and the job registry. Projects, the runner, folder sniffing and samples
/// are the desktop app's. (`set_active_export` writes only the server's
/// memory.)
const COMMANDS: &[&str] = &[
    "load_export",
    "set_active_export",
    "read_manifest",
    "undistort_points",
    "epipolar_overlay",
    "epipolar_overlay_undistorted",
    "camera_parameter_values",
    "projection_uncertainty",
    "disparity",
    "list_jobs",
    "job_status",
    "cancel_job",
];

/// The resources of the table the browser viewer answers.
const RESOURCES: &[&str] = &["image", "undistorted_image", "blob"];

/// Everything a request handler reads.
struct State {
    session: ViewerSession,
    /// The built frontend.
    dist: PathBuf,
    dist_scope: FileScope,
    /// The command table over the run folder and the export's image root.
    api: Api,
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
            session: ViewerSession::of(run),
            dist_scope: FileScope::new([dist.clone()]),
            dist,
            api: Api::new(RootScope::fixed(roots), Dirs::platform()),
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
    /// More headers (a resource's `ETag`, `Cache-Control`, `Content-Range`).
    headers: Vec<(&'static str, String)>,
}

impl Reply {
    fn json<T: Serialize>(value: &T) -> Self {
        match serde_json::to_vec(value) {
            Ok(body) => Self {
                status: 200,
                content_type: "application/json",
                body,
                no_store: true,
                headers: Vec::new(),
            },
            Err(e) => Self::error(500, format!("serialize the response: {e}")),
        }
    }

    /// A command's failure: the error envelope, with its kind's status.
    fn envelope(error: &StudioError) -> Self {
        let mut reply = Self::json(error);
        reply.status = error.http_status();
        reply
    }

    /// A failure of the file and page routes: the message as a JSON string.
    fn error(status: u16, message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            status,
            content_type: "application/json",
            body: serde_json::to_vec(&message).unwrap_or_default(),
            no_store: true,
            headers: Vec::new(),
        }
    }

    fn bytes(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type,
            body,
            no_store: false,
            headers: Vec::new(),
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
        for (name, value) in &self.headers {
            if let Ok(header) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
                response.add_header(header);
            }
        }
        response
    }
}

fn handle(state: &State, mut request: Request) {
    let url = request.url().to_owned();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    if request.method() == &Method::Get
        && path == "/api/events"
        && header(&request, "Host").is_some_and(|host| state.hosts.iter().any(|h| h == host))
    {
        let after = header(&request, "Last-Event-ID")
            .map(str::to_owned)
            .or_else(|| query_param(query, "after"))
            .and_then(|seq| seq.parse().ok());
        events(state, request, after);
        return;
    }
    let reply = route(state, &mut request).unwrap_or_else(|e| e);
    if reply.status >= 400 {
        let path = request.url().split('?').next().unwrap_or_default();
        let message = String::from_utf8_lossy(&reply.body);
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
        (Method::Post, _) if path.starts_with("/api/cmd/") => {
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
            command(state, &path["/api/cmd/".len()..], &body)
        }
        (Method::Get, _) if path.starts_with("/api/r/") => {
            let query = query_param(query, "q").unwrap_or_else(|| "{}".to_owned());
            let conditions = Conditions {
                if_none_match: header(request, "If-None-Match").map(str::to_owned),
                range: header(request, "Range").map(str::to_owned),
            };
            resource(state, &path["/api/r/".len()..], &query, &conditions)
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

fn unknown(name: &str) -> StudioError {
    StudioError::new(
        ErrorKind::UnknownCommand,
        format!("`{name}` is not available in the browser viewer"),
    )
}

/// Run the command `name` on the JSON argument object `body` (empty: no
/// arguments). The commands the browser does not offer are 404s.
fn command(state: &State, name: &str, body: &[u8]) -> Result<Reply, Reply> {
    let offered = matches!(state.api.kind(name), Some(Kind::Plain | Kind::Job { .. }));
    if !COMMANDS.contains(&name) || !offered {
        return Err(Reply::envelope(&unknown(name)));
    }
    let args = if body.iter().all(u8::is_ascii_whitespace) {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(body).map_err(|e| {
            Reply::envelope(&StudioError::new(
                ErrorKind::InvalidArgs,
                format!("{name}: the body is not JSON: {e}"),
            ))
        })?
    };
    state
        .api
        .invoke(name, args)
        .map(|reply| Reply::json(&reply))
        .map_err(|e| Reply::envelope(&e))
}

/// Resource `name` for the JSON query object `query`, under the request's
/// `conditions`.
fn resource(
    state: &State,
    name: &str,
    query: &str,
    conditions: &Conditions,
) -> Result<Reply, Reply> {
    if !RESOURCES.contains(&name) {
        return Err(Reply::envelope(&unknown(name)));
    }
    let query = serde_json::from_str(query).map_err(|e| {
        Reply::envelope(&StudioError::new(
            ErrorKind::InvalidArgs,
            format!("{name}: `q` is not JSON: {e}"),
        ))
    })?;
    let resource = state
        .api
        .resource(name, query, conditions)
        .map_err(|e| Reply::envelope(&e))?;
    let body = resource
        .read_body()
        .map_err(|e| Reply::error(500, format!("read the resource: {e}")))?;
    let mut reply = Reply::bytes(resource.media_type, body);
    reply.status = resource.status;
    reply.headers = vec![
        ("ETag", resource.etag.clone()),
        ("Cache-Control", resource.cache_control.to_owned()),
        ("Accept-Ranges", "bytes".to_owned()),
    ];
    if let Some(range) = &resource.content_range {
        reply.headers.push(("Content-Range", range.clone()));
    }
    Ok(reply)
}

/// The event stream as server-sent events, from after `after`: one
/// `id:`/`data:` record per event, a comment every 15 s so a closed
/// connection is noticed. Holds this worker until the client goes.
fn events(state: &State, request: Request, after: Option<u64>) {
    let subscription = state.api.subscribe(after);
    let reader = EventReader {
        events: subscription.events,
        pending: Vec::new(),
        at: 0,
    };
    let headers = [
        ("Content-Type", "text/event-stream"),
        ("Cache-Control", "no-store"),
        ("X-Content-Type-Options", "nosniff"),
    ]
    .into_iter()
    .filter_map(|(name, value)| Header::from_bytes(name.as_bytes(), value.as_bytes()).ok())
    .collect();
    let response = Response::new(StatusCode(200), headers, reader, None, None);
    let _ = request.respond(response);
    state.api.unsubscribe(subscription.id);
}

/// The events of a subscription, as the bytes of an SSE stream.
struct EventReader {
    events: Receiver<ServerEvent>,
    pending: Vec<u8>,
    at: usize,
}

impl Read for EventReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.at == self.pending.len() {
            self.pending = match self.events.recv_timeout(Duration::from_secs(15)) {
                Ok(event) => {
                    let data = serde_json::to_string(&event).unwrap_or_default();
                    format!("id: {}\ndata: {data}\n\n", event.seq).into_bytes()
                }
                Err(RecvTimeoutError::Timeout) => b": keep-alive\n\n".to_vec(),
                Err(RecvTimeoutError::Disconnected) => return Ok(0),
            };
            self.at = 0;
        }
        let n = out.len().min(self.pending.len() - self.at);
        out[..n].copy_from_slice(&self.pending[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpStream;
    use studio_core::ops;
    use studio_core::run_source::find_run;

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
            &format!("/api/cmd/{command}"),
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
    fn commands_answer_through_the_table() {
        let (viewer, run, _dist) = start();
        let port = viewer.port();

        // Before an export is active, a command on it fails with the
        // error envelope.
        let early = post(
            port,
            "undistort_points",
            serde_json::json!({"camera": 0, "points_px": [[1.0, 2.0]]}),
        );
        assert_eq!(early.status, 422);
        assert_eq!(
            early.json(),
            serde_json::json!({"kind": "invalid", "message": "no export loaded yet"})
        );

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
            serde_json::json!({"camera": 1, "points_px": [[10.0, 20.0]]}),
        );
        assert_eq!(points.status, 200, "{}", points.text());
        let px = points.json();
        assert!((px[0][0].as_f64().unwrap() - 10.0).abs() < 1e-6, "{px}");

        let overlay = post(
            port,
            "epipolar_overlay_undistorted",
            serde_json::json!({
                "cam_a": 0, "cam_b": 1, "point_px": [80.0, 60.0],
                "image_width_b": 160.0, "image_height_b": 120.0
            }),
        );
        assert_eq!(overlay.status, 200, "{}", overlay.text());

        // A resource: its query as JSON in `q`.
        let query = serde_json::json!({
            "path": slash(&run.dir.join("images/cam0_view0.png")), "camera": 0
        });
        let png = get(
            port,
            &format!("/api/r/undistorted_image?q={}", encode(&query.to_string())),
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
        assert_eq!(bad.json()["kind"], "invalid_args");
        let not_json = send(port, "POST", "/api/cmd/undistort_points", &[], "{");
        assert_eq!(not_json.status, 400);
        // The desktop's commands are not offered here.
        for name in ["run_calibration", "open_project", "fetch_sample", "nope"] {
            let desktop = post(port, name, serde_json::json!({}));
            assert_eq!(desktop.status, 404, "{name}");
            assert_eq!(desktop.json()["kind"], "unknown_command");
        }
        assert_eq!(get(port, "/api/r/nope?q=%7B%7D").status, 404);
        viewer.stop();
    }

    #[test]
    fn files_are_served_only_from_the_run_and_its_images() {
        let (viewer, run, _dist) = start();
        let port = viewer.port();
        let file = |path: &Path| {
            let query = serde_json::json!({ "path": slash(path) }).to_string();
            get(port, &format!("/api/r/image?q={}", encode(&query)))
        };

        let image = file(&run.dir.join("images/cam1_view0.png"));
        assert_eq!(image.status, 200);
        assert_eq!(image.content_type, "image/png");
        assert!(image.body.starts_with(b"\x89PNG"));

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
        assert_eq!(file(Path::new("images/cam0_view0.png")).status, 403);
        // A missing file is an I/O failure of the request, not a 200.
        assert_eq!(file(&run.dir.join("images/missing.png")).status, 422);
        // A command's path is scoped the same way.
        let outside = post(
            port,
            "read_manifest",
            serde_json::json!({
                "path": slash(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            }),
        );
        assert_eq!(outside.status, 403);
        assert_eq!(outside.json()["kind"], "forbidden_path");
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
            "/api/cmd/set_active_export",
            &[("Origin", "http://evil.example")],
            r#"{"export": {}}"#,
        );
        assert_eq!(foreign.status, 403);
        let own = format!("http://127.0.0.1:{port}");
        let same = send(
            port,
            "POST",
            "/api/cmd/set_active_export",
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
