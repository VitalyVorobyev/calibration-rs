//! Who may use the API: whoever holds the token. The page posts it once to
//! `/api/login` and gets a session cookie — `studio_session=<hex SHA-256 of
//! "studio-session-v1" ‖ token>; HttpOnly; SameSite=Strict; Path=/` (and
//! `Secure` behind an https public origin) — so `<img>` and `EventSource`
//! authenticate without script access to anything. No session table: the
//! cookie survives a restart and dies when the token rotates.
//! `Authorization: Bearer <token>` works too (curl, scripts). Comparisons
//! are constant-time; a failed login waits 250 ms.

use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use studio_core::api::{ErrorKind, StudioError};
use subtle::ConstantTimeEq;

use crate::{AppState, envelope};

/// The cookie's name.
pub const COOKIE: &str = "studio_session";
/// How long a failed login waits.
pub const FAILED_LOGIN_DELAY: Duration = Duration::from_millis(250);

/// The token check.
pub struct Auth {
    /// SHA-256 of the token; `None`: no token (a loopback bind).
    token_digest: Option<[u8; 32]>,
    /// The cookie value a session carries.
    session: Option<String>,
    secure: bool,
}

fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    hash.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

impl Auth {
    pub fn new(token: Option<String>, public_origin: Option<&str>) -> Self {
        Self {
            token_digest: token.as_deref().map(|t| digest(&[t.as_bytes()])),
            session: token
                .as_deref()
                .map(|t| hex(&digest(&[b"studio-session-v1", t.as_bytes()]))),
            secure: public_origin.is_some_and(|o| o.starts_with("https://")),
        }
    }

    /// Whether a token is required.
    pub fn required(&self) -> bool {
        self.token_digest.is_some()
    }

    /// Whether `token` is the token.
    pub fn token_matches(&self, token: &str) -> bool {
        self.token_digest
            .is_some_and(|expected| bool::from(digest(&[token.as_bytes()]).ct_eq(&expected)))
    }

    /// Whether the request carries the session cookie or the bearer token
    /// (always, without a token).
    pub fn authorized(&self, headers: &HeaderMap) -> bool {
        let Some(session) = &self.session else {
            return true;
        };
        let cookie = headers
            .get_all(header::COOKIE)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .flat_map(|v| v.split(';'))
            .filter_map(|pair| pair.trim().split_once('='))
            .any(|(name, value)| {
                name == COOKIE && bool::from(value.as_bytes().ct_eq(session.as_bytes()))
            });
        let bearer = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .is_some_and(|token| self.token_matches(token.trim()));
        cookie || bearer
    }

    /// The `Set-Cookie` of a new session.
    fn cookie(&self) -> Option<String> {
        let session = self.session.as_ref()?;
        let secure = if self.secure { "; Secure" } else { "" };
        Some(format!(
            "{COOKIE}={session}; HttpOnly; SameSite=Strict; Path=/{secure}"
        ))
    }

    /// The `Set-Cookie` that ends a session.
    fn expired(&self) -> String {
        let secure = if self.secure { "; Secure" } else { "" };
        format!("{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{secure}")
    }
}

/// `POST /api/login`'s body.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Login {
    pub token: String,
}

fn unauthorized() -> StudioError {
    StudioError::new(
        ErrorKind::Unauthorized,
        "this server needs its token: open the address it printed, or log in",
    )
}

/// `POST /api/login`: the token in, the session cookie out (204).
pub async fn login(State(state): State<Arc<AppState>>, body: Json<Login>) -> Response {
    let auth = &state.auth;
    if !auth.required() {
        return StatusCode::NO_CONTENT.into_response();
    }
    if !auth.token_matches(body.token.trim()) {
        tokio::time::sleep(FAILED_LOGIN_DELAY).await;
        return envelope(&unauthorized());
    }
    let mut response = StatusCode::NO_CONTENT.into_response();
    if let Some(cookie) = auth.cookie().and_then(|c| HeaderValue::from_str(&c).ok()) {
        response.headers_mut().insert(header::SET_COOKIE, cookie);
    }
    response
}

/// `POST /api/logout`: the cookie expired.
pub async fn logout(State(state): State<Arc<AppState>>) -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    if let Ok(cookie) = HeaderValue::from_str(&state.auth.expired()) {
        response.headers_mut().insert(header::SET_COOKIE, cookie);
    }
    response
}

/// The `/api` routes behind the token: 401 with the envelope otherwise, and
/// nothing about the server in it.
pub async fn require(State(state): State<Arc<AppState>>, request: Request, next: Next) -> Response {
    if state.auth.authorized(request.headers()) {
        next.run(request).await
    } else {
        envelope(&unauthorized())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cookie_and_the_bearer_token_authorize() {
        let auth = Auth::new(Some("a-token-of-sixteen+".into()), None);
        let mut headers = HeaderMap::new();
        assert!(!auth.authorized(&headers));
        let cookie = auth.cookie().unwrap();
        assert!(cookie.contains("HttpOnly") && cookie.contains("SameSite=Strict"));
        assert!(!cookie.contains("Secure"));
        let value = cookie.split(';').next().unwrap().to_owned();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("a=b; {value}")).unwrap(),
        );
        assert!(auth.authorized(&headers));

        let mut bearer = HeaderMap::new();
        bearer.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer a-token-of-sixteen+"),
        );
        assert!(auth.authorized(&bearer));
        bearer.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer nope"),
        );
        assert!(!auth.authorized(&bearer));
        // The cookie is not the token: presenting the token as the cookie
        // fails.
        let mut raw = HeaderMap::new();
        raw.insert(
            header::COOKIE,
            HeaderValue::from_static("studio_session=a-token-of-sixteen+"),
        );
        assert!(!auth.authorized(&raw));

        let secure = Auth::new(Some("a-token-of-sixteen+".into()), Some("https://c.lan"));
        assert!(secure.cookie().unwrap().ends_with("; Secure"));
        assert!(Auth::new(None, None).authorized(&HeaderMap::new()));
    }
}
