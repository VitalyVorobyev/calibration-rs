//! The server's routes end to end, in process (no socket): authentication,
//! the Host and Origin checks, the limits, the path scope and the event
//! stream. Each test serves a temporary root holding one image, beside a
//! file outside it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use http_body_util::BodyExt;
use studio_server::config::{Config, Dist};
use studio_server::{AppState, router};
use tower::ServiceExt;

const TOKEN: &str = "a-test-token-0123456789abcdef";
const HOST: &str = "127.0.0.1:8757";
const ORIGIN: &str = "http://127.0.0.1:8757";

struct Fixture {
    _tmp: tempfile::TempDir,
    app: Router,
    root: PathBuf,
    outside: PathBuf,
}

/// `<tmp>/root/{a.png, sub/}`, `<tmp>/secret.txt`, a link `root/escape`
/// to `<tmp>`, a built frontend in `<tmp>/dist`.
fn fixture(token: Option<&str>) -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let base = dunce::canonicalize(tmp.path()).unwrap();
    let root = base.join("root");
    std::fs::create_dir_all(root.join("sub")).unwrap();
    let png = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../e2e/fixtures/library/rig-demo/images/cam0_view0.png");
    std::fs::copy(png, root.join("a.png")).unwrap();
    std::fs::write(base.join("secret.txt"), "secret").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(&base, root.join("escape")).unwrap();
    let dist = base.join("dist");
    std::fs::create_dir_all(dist.join("assets")).unwrap();
    std::fs::write(
        dist.join("index.html"),
        "<!doctype html><title>page</title>",
    )
    .unwrap();
    std::fs::write(dist.join("assets/app.js"), "export {};").unwrap();
    let config = Config {
        roots: vec![root.clone()],
        bind: HOST.parse().unwrap(),
        token: token.map(str::to_owned),
        state_dir: base.join("state"),
        cache_dir: base.join("cache"),
        dist: Dist::Folder(dist),
        allowed_hosts: vec!["calib.lan".into()],
        public_origin: None,
        project: None,
        open: false,
    };
    let app = router(Arc::new(AppState::new(&config)));
    Fixture {
        _tmp: tmp,
        app,
        outside: base.join("secret.txt"),
        root,
    }
}

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Vec<u8>,
}

impl Reply {
    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or(serde_json::Value::Null)
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

async fn send(app: &Router, request: Request<Body>) -> Reply {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    Reply {
        status,
        headers,
        body,
    }
}

fn get(path: &str) -> axum::http::request::Builder {
    Request::get(path).header(header::HOST, HOST)
}

fn post(path: &str) -> axum::http::request::Builder {
    Request::post(path)
        .header(header::HOST, HOST)
        .header(header::ORIGIN, ORIGIN)
        .header(header::CONTENT_TYPE, "application/json")
}

fn bearer(builder: axum::http::request::Builder) -> axum::http::request::Builder {
    builder.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
}

fn json(value: serde_json::Value) -> Body {
    Body::from(value.to_string())
}

fn query(value: serde_json::Value) -> String {
    value
        .to_string()
        .bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[tokio::test]
async fn nothing_but_the_page_health_and_login_without_the_token() {
    let f = fixture(Some(TOKEN));
    for request in [
        post("/api/cmd/server_info")
            .body(json(serde_json::json!({})))
            .unwrap(),
        post("/api/cmd/list_dir")
            .body(json(serde_json::json!({})))
            .unwrap(),
        get("/api/events").body(Body::empty()).unwrap(),
        get(&format!(
            "/api/r/image?q={}",
            query(serde_json::json!({"path": f.root.join("a.png")}))
        ))
        .body(Body::empty())
        .unwrap(),
        get("/api/nothing").body(Body::empty()).unwrap(),
        post("/api/logout").body(Body::empty()).unwrap(),
    ] {
        let path = request.uri().path().to_owned();
        let reply = send(&f.app, request).await;
        assert_eq!(reply.status, StatusCode::UNAUTHORIZED, "{path}");
        let body = reply.json();
        assert_eq!(body["kind"], "unauthorized", "{path}");
        // Nothing about the server: no version, roots, jobs.
        let text = reply.text();
        assert!(!text.contains(&*f.root.to_string_lossy()) && !text.contains("version"));
    }
    let health = send(&f.app, get("/api/health").body(Body::empty()).unwrap()).await;
    assert_eq!(
        (health.status, health.json()),
        (StatusCode::OK, serde_json::json!({"ok": true}))
    );
    let page = send(&f.app, get("/").body(Body::empty()).unwrap()).await;
    assert_eq!(page.status, StatusCode::OK);
    assert!(page.text().contains("<title>page</title>"));
}

#[tokio::test]
async fn login_sets_a_strict_http_only_cookie_and_the_bearer_token_works() {
    let f = fixture(Some(TOKEN));
    let started = Instant::now();
    let wrong = send(
        &f.app,
        post("/api/login")
            .body(json(serde_json::json!({"token": "nope-nope-nope-nope"})))
            .unwrap(),
    )
    .await;
    assert_eq!(wrong.status, StatusCode::UNAUTHORIZED);
    assert!(
        started.elapsed() >= Duration::from_millis(250),
        "a failed login waits"
    );
    assert!(wrong.headers.get(header::SET_COOKIE).is_none());

    let login = send(
        &f.app,
        post("/api/login")
            .body(json(serde_json::json!({"token": TOKEN})))
            .unwrap(),
    )
    .await;
    assert_eq!(login.status, StatusCode::NO_CONTENT);
    let cookie = login.headers[header::SET_COOKIE]
        .to_str()
        .unwrap()
        .to_owned();
    for flag in ["HttpOnly", "SameSite=Strict", "Path=/"] {
        assert!(cookie.contains(flag), "{cookie}");
    }
    assert!(!cookie.contains(TOKEN), "the cookie is not the token");
    let session = cookie.split(';').next().unwrap().to_owned();

    let with_cookie = send(
        &f.app,
        post("/api/cmd/server_info")
            .header(header::COOKIE, &session)
            .body(json(serde_json::json!({})))
            .unwrap(),
    )
    .await;
    assert_eq!(with_cookie.status, StatusCode::OK);
    assert_eq!(with_cookie.json()["roots_editable"], false);
    assert_eq!(with_cookie.headers[header::CACHE_CONTROL], "no-store");
    let with_bearer = send(
        &f.app,
        bearer(post("/api/cmd/list_dir"))
            .body(json(serde_json::json!({})))
            .unwrap(),
    )
    .await;
    assert_eq!(with_bearer.status, StatusCode::OK);

    let logout = send(
        &f.app,
        post("/api/logout")
            .header(header::COOKIE, &session)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(logout.status, StatusCode::NO_CONTENT);
    assert!(
        logout.headers[header::SET_COOKIE]
            .to_str()
            .unwrap()
            .contains("Max-Age=0")
    );
}

#[tokio::test]
async fn a_loopback_server_without_a_token_needs_none() {
    let f = fixture(None);
    let reply = send(
        &f.app,
        post("/api/cmd/server_info")
            .body(json(serde_json::json!({})))
            .unwrap(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::OK);
}

#[tokio::test]
async fn foreign_hosts_origins_and_methods_are_refused_without_cors() {
    let f = fixture(Some(TOKEN));
    let mut replies = Vec::new();
    for (host, expected) in [
        ("evil.example:8757", StatusCode::FORBIDDEN),
        ("localhost.evil.example", StatusCode::FORBIDDEN),
        ("calib.lan", StatusCode::OK),
        ("10.1.2.3:8757", StatusCode::OK),
        ("localhost:8757", StatusCode::OK),
    ] {
        let reply = send(
            &f.app,
            Request::get("/api/health")
                .header(header::HOST, host)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(reply.status, expected, "{host}");
        replies.push(reply);
    }
    let no_host = send(&f.app, Request::get("/").body(Body::empty()).unwrap()).await;
    assert_eq!(no_host.status, StatusCode::FORBIDDEN);

    let body = || json(serde_json::json!({}));
    let no_origin = Request::post("/api/cmd/server_info")
        .header(header::HOST, HOST)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .body(body())
        .unwrap();
    let foreign = bearer(post("/api/cmd/server_info"))
        .header(header::ORIGIN, "http://evil.example")
        .body(body())
        .unwrap();
    // A form post (`text/plain`) cannot reach a command either.
    let form = Request::post("/api/cmd/server_info")
        .header(header::HOST, HOST)
        .header(header::ORIGIN, ORIGIN)
        .header(header::CONTENT_TYPE, "text/plain")
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .body(body())
        .unwrap();
    let login_elsewhere = Request::post("/api/login")
        .header(header::HOST, HOST)
        .header(header::ORIGIN, "http://evil.example")
        .header(header::CONTENT_TYPE, "application/json")
        .body(json(serde_json::json!({"token": TOKEN})))
        .unwrap();
    for (request, expected) in [
        (no_origin, StatusCode::FORBIDDEN),
        (foreign, StatusCode::FORBIDDEN),
        (form, StatusCode::UNSUPPORTED_MEDIA_TYPE),
        (login_elsewhere, StatusCode::FORBIDDEN),
    ] {
        let reply = send(&f.app, request).await;
        assert_eq!(reply.status, expected);
        replies.push(reply);
    }
    let preflight = send(
        &f.app,
        Request::options("/api/cmd/server_info")
            .header(header::HOST, HOST)
            .header(header::ORIGIN, "http://evil.example")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(preflight.status, StatusCode::METHOD_NOT_ALLOWED);
    replies.push(preflight);
    for reply in &replies {
        assert!(
            reply
                .headers
                .keys()
                .all(|name| !name.as_str().starts_with("access-control-")),
            "no CORS header: {:?}",
            reply.headers
        );
        assert_eq!(reply.headers["x-content-type-options"], "nosniff");
        assert_eq!(reply.headers["referrer-policy"], "no-referrer");
        assert!(reply.headers.contains_key(header::CONTENT_SECURITY_POLICY));
    }
}

#[tokio::test]
async fn bodies_and_queries_are_limited() {
    let f = fixture(Some(TOKEN));
    let big = format!("{{\"path\": \"{}\"}}", "x".repeat(5 << 20));
    let reply = send(
        &f.app,
        bearer(post("/api/cmd/list_dir"))
            .body(Body::from(big))
            .unwrap(),
    )
    .await;
    assert_eq!(reply.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(reply.json()["kind"], "too_large");
    let long = format!("/api/r/image?q={}", "a".repeat(9 << 10));
    let reply = send(&f.app, bearer(get(&long)).body(Body::empty()).unwrap()).await;
    assert_eq!(reply.status, StatusCode::URI_TOO_LONG);
}

#[tokio::test]
async fn paths_outside_the_root_are_refused() {
    let f = fixture(Some(TOKEN));
    let image = |path: String| {
        bearer(get(&format!(
            "/api/r/image?q={}",
            query(serde_json::json!({"path": path}))
        )))
        .body(Body::empty())
        .unwrap()
    };
    let inside = send(&f.app, image(f.root.join("a.png").to_string_lossy().into())).await;
    assert_eq!(inside.status, StatusCode::OK);
    assert_eq!(inside.headers[header::CONTENT_TYPE], "image/png");
    let mut escapes = vec![
        f.outside.to_string_lossy().into_owned(),
        f.root.join("../secret.txt").to_string_lossy().into_owned(),
        f.root
            .join("sub/../../secret.txt")
            .to_string_lossy()
            .into_owned(),
        "secret.txt".to_owned(),
        "/etc/passwd".to_owned(),
    ];
    if cfg!(unix) {
        escapes.push(
            f.root
                .join("escape/secret.txt")
                .to_string_lossy()
                .into_owned(),
        );
    }
    for path in escapes {
        let reply = send(&f.app, image(path.clone())).await;
        assert_eq!(reply.status, StatusCode::FORBIDDEN, "{path}");
        assert_eq!(reply.json()["kind"], "forbidden_path", "{path}");
        assert!(!reply.text().contains("secret\""), "{path}");
    }
    // No listing outside the roots; the roots themselves without a path.
    let list =
        |args: serde_json::Value| bearer(post("/api/cmd/list_dir")).body(json(args)).unwrap();
    let roots = send(&f.app, list(serde_json::json!({}))).await.json();
    let listed: Vec<&str> = roots["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["path"].as_str().unwrap())
        .collect();
    assert!(listed.contains(&&*f.root.to_string_lossy()));
    for path in [f.root.join(".."), f.root.join("escape")] {
        let reply = send(&f.app, list(serde_json::json!({"path": path}))).await;
        assert_eq!(reply.status, StatusCode::FORBIDDEN, "{}", path.display());
    }
    let inside = send(&f.app, list(serde_json::json!({"path": f.root})))
        .await
        .json();
    let names: Vec<&str> = inside["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(
        !names.contains(&"escape"),
        "a link out of the root is not listed: {names:?}"
    );
    // The frontend's files never reach past the frontend.
    for path in [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/..%2f..%2fsecret.txt",
    ] {
        let reply = send(&f.app, get(path).body(Body::empty()).unwrap()).await;
        assert!(!reply.text().contains("secret"), "{path}: {}", reply.text());
    }
}

#[tokio::test]
async fn images_answer_ranges_and_revalidation() {
    let f = fixture(Some(TOKEN));
    let url = format!(
        "/api/r/image?q={}",
        query(serde_json::json!({"path": f.root.join("a.png")}))
    );
    let whole = send(&f.app, bearer(get(&url)).body(Body::empty()).unwrap()).await;
    assert_eq!(whole.status, StatusCode::OK);
    let etag = whole.headers[header::ETAG].to_str().unwrap().to_owned();
    assert_eq!(whole.headers[header::CACHE_CONTROL], "private, no-cache");
    let part = send(
        &f.app,
        bearer(get(&url))
            .header(header::RANGE, "bytes=0-7")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(part.status, StatusCode::PARTIAL_CONTENT);
    assert_eq!(part.body, b"\x89PNG\r\n\x1a\n");
    assert!(
        part.headers[header::CONTENT_RANGE]
            .to_str()
            .unwrap()
            .starts_with("bytes 0-7/")
    );
    let same = send(
        &f.app,
        bearer(get(&url))
            .header(header::IF_NONE_MATCH, &etag)
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(same.status, StatusCode::NOT_MODIFIED);
}

/// The `id:` and `data:` of the next SSE record of `body`.
async fn next_record(body: &mut Body) -> (u64, serde_json::Value) {
    let mut text = String::new();
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(10), body.frame())
            .await
            .expect("an event within 10 s")
            .expect("the stream stays open")
            .unwrap();
        if let Ok(data) = frame.into_data() {
            text.push_str(&String::from_utf8_lossy(&data));
        }
        if let Some((record, _)) = text.split_once("\n\n") {
            let mut id = None;
            let mut data = None;
            for line in record.lines() {
                if let Some(v) = line.strip_prefix("id:") {
                    id = v.trim().parse().ok();
                }
                if let Some(v) = line.strip_prefix("data:") {
                    data = serde_json::from_str(v.trim()).ok();
                }
            }
            if let (Some(id), Some(data)) = (id, data) {
                return (id, data);
            }
            text = text
                .split_once("\n\n")
                .map(|(_, rest)| rest.to_owned())
                .unwrap_or_default();
        }
    }
}

#[tokio::test]
async fn the_event_stream_reports_a_job_and_resumes_after_the_last_id() {
    let f = fixture(Some(TOKEN));
    let stream = f
        .app
        .clone()
        .oneshot(bearer(get("/api/events")).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(stream.status(), StatusCode::OK);
    assert_eq!(stream.headers()[header::CONTENT_TYPE], "text/event-stream");
    assert_eq!(stream.headers()["x-accel-buffering"], "no");
    let mut body = stream.into_body();
    // A job that fails at once: an unknown sample.
    let started = send(
        &f.app,
        bearer(post("/api/cmd/fetch_sample"))
            .body(json(serde_json::json!({"id": "no-such-sample"})))
            .unwrap(),
    )
    .await;
    assert_eq!(started.status, StatusCode::OK);
    let id = started.json()["id"].as_str().unwrap().to_owned();
    let mut seen = Vec::new();
    loop {
        let (seq, event) = next_record(&mut body).await;
        assert_eq!(event["body"]["id"], id.as_str());
        seen.push(seq);
        if event["body"]["state"]["kind"] == "failed" {
            break;
        }
    }
    assert!(seen.len() >= 2, "{seen:?}");
    drop(body);

    // Reconnecting with the first id gets the rest again, in order.
    let resumed = f
        .app
        .clone()
        .oneshot(
            bearer(get("/api/events"))
                .header("last-event-id", seen[0].to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let mut body = resumed.into_body();
    for expected in &seen[1..] {
        assert_eq!(next_record(&mut body).await.0, *expected);
    }
}
