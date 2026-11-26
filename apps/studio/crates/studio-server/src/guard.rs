//! The checks every request passes before its route, and the headers every
//! response carries.
//!
//! - **Host** must name this server: its bind address, `localhost`,
//!   `127.0.0.1`, any IP literal (DNS rebinding needs a name), the machine's
//!   name, or an `--allowed-host`. Else 403.
//! - A **non-GET** must carry an `Origin` equal to the request's own
//!   `http://<Host>` or `--public-origin` (403 otherwise), and a JSON
//!   `Content-Type` (415 otherwise). GET routes have no side effects.
//! - **No CORS**: no `Access-Control-*` header is ever sent; `OPTIONS` is
//!   405.
//! - The **query** is at most 8 KiB (414).
//! - Every response: `X-Content-Type-Options: nosniff`, `Referrer-Policy:
//!   no-referrer`, `Cross-Origin-Resource-Policy: same-origin` and the app's
//!   Content Security Policy.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;
use studio_core::api::{ErrorKind, StudioError};

use crate::config::Config;
use crate::{AppState, MAX_QUERY, envelope};

/// The page's Content Security Policy; `script_hashes` are the `'sha256-…'`
/// sources of `index.html`'s inline scripts (the theme painted before the
/// first frame), the only inline code allowed.
pub fn csp(script_hashes: &[String]) -> String {
    let scripts: String = script_hashes.iter().map(|h| format!(" '{h}'")).collect();
    format!(
        "default-src 'self'; script-src 'self'{scripts}; img-src 'self' blob: data:; \
         style-src 'self' 'unsafe-inline'; font-src 'self' data:; connect-src 'self'; \
         frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
}

/// The Host names and origins the server answers.
pub struct Guard {
    /// Lower-case names a `Host` may carry (without its port).
    names: Vec<String>,
    public_origin: Option<String>,
}

impl Guard {
    pub fn new(config: &Config) -> Self {
        let mut names = vec!["localhost".to_owned(), "127.0.0.1".to_owned()];
        if let Ok(name) = gethostname::gethostname().into_string() {
            let name = name.to_ascii_lowercase();
            names.push(format!("{}.local", name.trim_end_matches(".local")));
            names.push(name);
        }
        names.extend(config.allowed_hosts.iter().map(|h| h.to_ascii_lowercase()));
        if let Some(host) = config
            .public_origin
            .as_deref()
            .and_then(|o| o.split_once("://"))
            .map(|(_, rest)| host_name(rest).to_ascii_lowercase())
        {
            names.push(host);
        }
        Self {
            names,
            public_origin: config.public_origin.clone(),
        }
    }

    /// Whether `host` (a `Host` header) names this server.
    pub fn host_allowed(&self, host: &str) -> bool {
        let name = host_name(host).to_ascii_lowercase();
        let bare = name.trim_start_matches('[').trim_end_matches(']');
        bare.parse::<IpAddr>().is_ok() || self.names.contains(&name)
    }

    /// Whether `origin` may send a non-GET to `host`.
    pub fn origin_allowed(&self, origin: &str, host: &str) -> bool {
        origin == format!("http://{host}") || self.public_origin.as_deref() == Some(origin)
    }
}

/// `host` without its port (`[::1]:80` → `[::1]`).
fn host_name(host: &str) -> &str {
    if host.starts_with('[') {
        return host
            .split_once(']')
            .map_or(host, |(name, _)| &host[..=name.len()]);
    }
    host.rsplit_once(':').map_or(host, |(name, _)| name)
}

fn refuse(status: StatusCode, kind: ErrorKind, message: &str) -> Response {
    let mut response = envelope(&StudioError::new(kind, message));
    *response.status_mut() = status;
    response
}

/// The checks of every request (see the module doc).
pub async fn guard(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    match refusal(&state.guard, &request) {
        Some(response) => response,
        None => next.run(request).await,
    }
}

/// Why `request` is refused, if it is.
fn refusal(guard: &Guard, request: &Request) -> Option<Response> {
    // A header counts only when it is there once: two Origins (or Hosts)
    // are not a choice the request gets to make.
    let header = |name: header::HeaderName| {
        let mut values = request.headers().get_all(name).iter();
        match (values.next(), values.next()) {
            (Some(value), None) => value.to_str().ok().map(str::to_owned),
            _ => None,
        }
    };
    let Some(host) = header(header::HOST).filter(|h| guard.host_allowed(h)) else {
        return Some(refuse(
            StatusCode::FORBIDDEN,
            ErrorKind::ForbiddenPath,
            "this server does not answer that Host",
        ));
    };
    if request
        .uri()
        .query()
        .is_some_and(|query| query.len() > MAX_QUERY)
    {
        return Some(refuse(
            StatusCode::URI_TOO_LONG,
            ErrorKind::TooLarge,
            "the query is longer than 8 KiB",
        ));
    }
    let method = request.method().clone();
    if method == Method::OPTIONS {
        return Some(refuse(
            StatusCode::METHOD_NOT_ALLOWED,
            ErrorKind::UnknownCommand,
            "no cross-origin requests",
        ));
    }
    if method != Method::GET && method != Method::HEAD {
        let origin_ok = header(header::ORIGIN).is_some_and(|o| guard.origin_allowed(&o, &host));
        if !origin_ok {
            return Some(refuse(
                StatusCode::FORBIDDEN,
                ErrorKind::ForbiddenPath,
                "a request that changes something must come from this server's own page",
            ));
        }
        let json = header(header::CONTENT_TYPE).is_some_and(|t| {
            t.split(';')
                .next()
                .is_some_and(|m| m.trim().eq_ignore_ascii_case("application/json"))
        });
        if !json {
            return Some(refuse(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                ErrorKind::InvalidArgs,
                "the body must be JSON (Content-Type: application/json)",
            ));
        }
    }
    None
}

/// The headers of every response.
pub async fn security_headers(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    for (name, value) in [
        (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
        (header::REFERRER_POLICY, "no-referrer"),
        (header::X_FRAME_OPTIONS, "DENY"),
    ] {
        headers
            .entry(name)
            .or_insert(HeaderValue::from_static(value));
    }
    if let Ok(csp) = HeaderValue::from_str(&state.csp) {
        headers
            .entry(header::CONTENT_SECURITY_POLICY)
            .or_insert(csp);
    }
    headers.insert(
        "cross-origin-resource-policy",
        HeaderValue::from_static("same-origin"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard(allowed: &[&str], public: Option<&str>) -> Guard {
        let config = Config {
            roots: Vec::new(),
            bind: "127.0.0.1:8757".parse().unwrap(),
            token: None,
            state_dir: "/s".into(),
            cache_dir: "/c".into(),
            dist: crate::config::Dist::Embedded,
            allowed_hosts: allowed.iter().map(|h| (*h).to_owned()).collect(),
            public_origin: public.map(str::to_owned),
            project: None,
            open: false,
        };
        Guard::new(&config)
    }

    #[test]
    fn hosts_are_this_machine_or_named() {
        let g = guard(&["calib.lan"], Some("https://proxy.example"));
        for ok in [
            "localhost:8757",
            "127.0.0.1:8757",
            "192.168.1.20:8757",
            "[::1]:8757",
            "[fe80::1]",
            "CALIB.lan",
            "proxy.example",
        ] {
            assert!(g.host_allowed(ok), "{ok}");
        }
        for bad in [
            "evil.example:8757",
            "localhost.evil.example",
            "",
            "calib.lan.evil",
        ] {
            assert!(!g.host_allowed(bad), "{bad}");
        }
        assert_eq!(host_name("[::1]:80"), "[::1]");
        assert_eq!(host_name("a.b:80"), "a.b");
    }

    #[test]
    fn origins_are_the_page_s_own() {
        let g = guard(&[], Some("https://proxy.example"));
        assert!(g.origin_allowed("http://10.0.0.2:8757", "10.0.0.2:8757"));
        assert!(g.origin_allowed("https://proxy.example", "127.0.0.1:8757"));
        assert!(!g.origin_allowed("http://evil.example", "10.0.0.2:8757"));
        assert!(!g.origin_allowed("null", "10.0.0.2:8757"));
    }
}
