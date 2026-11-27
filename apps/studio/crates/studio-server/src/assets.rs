//! The frontend: embedded in the binary (`embed-dist`) or a folder
//! (`--dist`). Public: it is the app's code, nothing about the device. A
//! path that is not a file of the frontend answers `index.html` (the page
//! routes itself); hashed `assets/*` are immutable, `index.html` is
//! revalidated. A folder is served by plain relative names only, judged
//! after canonicalising, so `..` and symbolic links cannot reach past it.

use std::borrow::Cow;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};

use crate::AppState;
use crate::config::{Config, Dist};

#[cfg(feature = "embed-dist")]
#[derive(rust_embed::Embed)]
#[folder = "../../dist"]
struct Embedded;

/// The file `name` of the frontend.
fn file(dist: &Dist, name: &str) -> Option<Cow<'static, [u8]>> {
    match dist {
        #[cfg(feature = "embed-dist")]
        Dist::Embedded => Embedded::get(name).map(|f| f.data),
        #[cfg(not(feature = "embed-dist"))]
        Dist::Embedded => None,
        Dist::Folder(dir) => folder_file(dir, name).map(Cow::Owned),
    }
}

/// `name` under `dir`, if it is a plain relative path to a file there.
fn folder_file(dir: &Path, name: &str) -> Option<Vec<u8>> {
    if name
        .split('/')
        .any(|part| part.is_empty() || part == "." || part == ".." || part.contains(['\\', ':']))
    {
        return None;
    }
    let root = dunce::canonicalize(dir).ok()?;
    let path = dunce::canonicalize(root.join(name)).ok()?;
    if !path.starts_with(&root) || !path.is_file() {
        return None;
    }
    std::fs::read(path).ok()
}

fn media_type(name: &str) -> &'static str {
    let ext = name.rsplit_once('.').map_or("", |(_, e)| e);
    match ext.to_ascii_lowercase().as_str() {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// The fallback route: a file of the frontend, else `index.html`.
pub async fn serve(State(state): State<Arc<AppState>>, request: Request) -> Response {
    if request.method() != Method::GET && request.method() != Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let path = request.uri().path().trim_start_matches('/');
    let (name, bytes) = match (!path.is_empty())
        .then(|| file(&state.dist, path))
        .flatten()
    {
        Some(bytes) => (path.to_owned(), bytes),
        None => match file(&state.dist, "index.html") {
            Some(bytes) => ("index.html".to_owned(), bytes),
            None => return StatusCode::NOT_FOUND.into_response(),
        },
    };
    let cache = if name.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    let mut response = Response::new(Body::from(bytes.into_owned()));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(media_type(&name)),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    response
}

/// The `sha256-<base64>` source of each inline `<script>` of the frontend's
/// `index.html`, for the Content Security Policy.
pub fn inline_script_hashes(dist: &Dist) -> Vec<String> {
    use base64::Engine;
    use sha2::{Digest, Sha256};
    let Some(page) = file(dist, "index.html") else {
        return Vec::new();
    };
    let page = String::from_utf8_lossy(&page);
    let mut hashes = Vec::new();
    let mut rest: &str = &page;
    while let Some(start) = rest.find("<script>") {
        let body = &rest[start + "<script>".len()..];
        let Some(end) = body.find("</script>") else {
            break;
        };
        let digest = Sha256::digest(&body.as_bytes()[..end]);
        hashes.push(format!(
            "sha256-{}",
            base64::engine::general_purpose::STANDARD.encode(digest)
        ));
        rest = &body[end..];
    }
    hashes
}

/// Whether the frontend has its built assets (not the placeholder page).
fn built(dist: &Dist) -> bool {
    match dist {
        #[cfg(feature = "embed-dist")]
        Dist::Embedded => Embedded::iter().any(|name| name.starts_with("assets/")),
        #[cfg(not(feature = "embed-dist"))]
        Dist::Embedded => false,
        Dist::Folder(dir) => dir.join("assets").is_dir(),
    }
}

/// Print the start-up lines on stderr; the address to open (with the token
/// in its fragment, never sent to a server, and the project to open).
pub fn announce(config: &Config, bound: SocketAddr, state: &AppState) -> String {
    let host = if bound.ip().is_unspecified() {
        gethostname::gethostname()
            .into_string()
            .unwrap_or_else(|_| "localhost".into())
    } else {
        bound.ip().to_string()
    };
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host
    };
    let mut fragment = Vec::new();
    if let Some(token) = &config.token {
        fragment.push(format!("token={token}"));
    }
    if let Some(project) = &config.project {
        fragment.push(format!("project={}", encode(&project.to_string_lossy())));
    }
    let base = config
        .public_origin
        .clone()
        .unwrap_or_else(|| format!("http://{host}:{}", bound.port()));
    let url = if fragment.is_empty() {
        format!("{base}/")
    } else {
        format!("{base}/#{}", fragment.join("&"))
    };
    eprintln!("calibration-studio-server {}", env!("CARGO_PKG_VERSION"));
    for root in state.api.ctx().scope.roots() {
        eprintln!("  root {}", root.display());
    }
    if !built(&config.dist) {
        eprintln!("  the frontend is not built: run `bun run build` in apps/studio/");
    }
    eprintln!("  open {url}");
    eprintln!("  Ctrl-C stops the server.");
    url
}

/// `text` percent-encoded for a URL fragment value.
fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_serves_plain_names_inside_it_only() {
        let tmp = tempfile::tempdir().unwrap();
        let dist = tmp.path().join("dist");
        std::fs::create_dir_all(dist.join("assets")).unwrap();
        std::fs::write(dist.join("index.html"), "page").unwrap();
        std::fs::write(dist.join("assets/app.js"), "js").unwrap();
        std::fs::write(tmp.path().join("secret.txt"), "secret").unwrap();
        assert_eq!(folder_file(&dist, "assets/app.js").unwrap(), b"js");
        for bad in [
            "../secret.txt",
            "assets/../../secret.txt",
            "/etc/passwd",
            "",
            "assets//app.js",
        ] {
            assert!(folder_file(&dist, bad).is_none(), "{bad}");
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(tmp.path().join("secret.txt"), dist.join("link.txt"))
                .unwrap();
            assert!(folder_file(&dist, "link.txt").is_none());
        }
        assert_eq!(media_type("assets/x.JS"), "text/javascript; charset=utf-8");
        std::fs::write(
            dist.join("index.html"),
            "<head><script>paint()</script><script type=\"module\" src=\"/a.js\"></script></head>",
        )
        .unwrap();
        let hashes = inline_script_hashes(&Dist::Folder(dist.clone()));
        // echo -n 'paint()' | openssl dgst -sha256 -binary | base64
        assert_eq!(
            hashes,
            ["sha256-6Z05mR7jGVRSmyHx7iStJaiOybZYZUjxiGbPhYYXh1o="]
        );
        assert_eq!(encode("/a b/ü"), "/a%20b/%C3%BC");
    }
}
