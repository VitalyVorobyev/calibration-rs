//! The `studio:` URI scheme: the desktop app's resources by URL, as the
//! browser gets them from `GET /api/r/<name>?q=<query>`. The webview loads
//! `studio://localhost/r/<name>?q=<query JSON>` (`http://studio.localhost/…`
//! on Windows) in an `<img>` or a `fetch`; the handler calls the same
//! `Api::resource` on a blocking thread and copies its reply out: status,
//! media type, `ETag`, `Cache-Control`, `Content-Range`.
//!
//! The scheme is another origin than the page, so the reply names the page's
//! origin in `Access-Control-Allow-Origin`: a canvas that reads an image's
//! pixels stays untainted. Only the app's own origins are named.

use std::sync::Arc;

use studio_core::api::{Api, Conditions, ResourceReply, StudioError};
use tauri::http::{Request, Response, StatusCode, header};

/// The page origins the app is served from: the bundled app on macOS and
/// Linux, on Windows, and the Vite dev server of `tauri dev`.
const APP_ORIGINS: &[&str] = &[
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:1420",
];

/// The URI scheme's name.
pub const SCHEME: &str = "studio";

/// `%XX` escapes decoded. `None` for a broken escape or non-UTF-8.
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

/// The resource name and query of a request target. The frontend builds the
/// URL with Tauri's `convertFileSrc("r/<name>?q=<json>", "studio")`, which
/// percent-encodes the whole `r/…` part into the path; a plain
/// `/r/<name>?q=<encoded json>` is read too.
pub fn parse_target(path: &str, query: Option<&str>) -> Option<(String, String)> {
    let decoded = percent_decode(path.trim_start_matches('/'))?;
    let (route, inline_query) = match decoded.split_once('?') {
        Some((route, q)) => (route.to_owned(), Some(q.to_owned())),
        None => (decoded, None),
    };
    let name = route.strip_prefix("r/")?;
    if name.is_empty() || name.contains('/') {
        return None;
    }
    let json = match (inline_query, query) {
        (Some(q), _) => q.strip_prefix("q=")?.to_owned(),
        (None, Some(q)) => q
            .split('&')
            .find_map(|pair| pair.strip_prefix("q="))
            .and_then(percent_decode)?,
        (None, None) => "{}".to_owned(),
    };
    Some((name.to_owned(), json))
}

/// Answer one `studio:` request.
pub fn respond(api: &Arc<Api>, request: &Request<Vec<u8>>) -> Response<Vec<u8>> {
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|o| o.to_str().ok())
        .filter(|o| APP_ORIGINS.contains(o))
        .map(str::to_owned);
    let mut response = match answer(api, request) {
        Ok(reply) => reply,
        Err(e) => envelope(&e),
    };
    let headers = response.headers_mut();
    if let Some(origin) = origin
        && let Ok(value) = origin.parse()
    {
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
        headers.insert(header::VARY, header::HeaderValue::from_static("Origin"));
    }
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        header::HeaderValue::from_static("nosniff"),
    );
    response
}

fn answer(api: &Arc<Api>, request: &Request<Vec<u8>>) -> Result<Response<Vec<u8>>, StudioError> {
    if request.method() != tauri::http::Method::GET {
        return Err(StudioError::new(
            studio_core::api::ErrorKind::UnknownCommand,
            "the studio: scheme answers GET only",
        ));
    }
    let uri = request.uri();
    let (name, json) = parse_target(uri.path(), uri.query()).ok_or_else(|| {
        StudioError::new(
            studio_core::api::ErrorKind::UnknownCommand,
            format!("no resource at {uri}"),
        )
    })?;
    let query: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
        StudioError::new(
            studio_core::api::ErrorKind::InvalidArgs,
            format!("{name}: the query is not JSON: {e}"),
        )
    })?;
    let header_value = |name: header::HeaderName| {
        request
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    let conditions = Conditions {
        if_none_match: header_value(header::IF_NONE_MATCH),
        range: header_value(header::RANGE),
    };
    let reply = api.resource(&name, query, &conditions)?;
    http_reply(&reply)
}

/// `reply` as a webview response, its body read into memory.
fn http_reply(reply: &ResourceReply) -> Result<Response<Vec<u8>>, StudioError> {
    let body = reply
        .read_body()
        .map_err(|e| StudioError::internal(format!("read the resource: {e}")))?;
    let mut builder = Response::builder()
        .status(reply.status)
        .header(header::CONTENT_TYPE, reply.media_type)
        .header(header::ETAG, &reply.etag)
        .header(header::CACHE_CONTROL, reply.cache_control)
        .header(header::ACCEPT_RANGES, "bytes");
    if let Some(range) = &reply.content_range {
        builder = builder.header(header::CONTENT_RANGE, range);
    }
    builder
        .body(body)
        .map_err(|e| StudioError::internal(format!("build the response: {e}")))
}

/// A failure: the error envelope with its kind's status.
fn envelope(error: &StudioError) -> Response<Vec<u8>> {
    let body = serde_json::to_vec(error).unwrap_or_default();
    let status =
        StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = Response::new(body);
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_parse_as_convert_file_src_writes_them() {
        // `convertFileSrc("r/image?q={...}", "studio")` encodes it all.
        let encoded = "/r%2Fimage%3Fq%3D%7B%22path%22%3A%22%2Fa%20b.png%22%7D";
        assert_eq!(
            parse_target(encoded, None),
            Some(("image".into(), r#"{"path":"/a b.png"}"#.into()))
        );
        // The plain form.
        assert_eq!(
            parse_target("/r/blob", Some("q=%7B%22id%22%3A%22x%22%7D")),
            Some(("blob".into(), r#"{"id":"x"}"#.into()))
        );
        assert_eq!(
            parse_target("/r/blob", None),
            Some(("blob".into(), "{}".into()))
        );
        assert_eq!(parse_target("/x/image", None), None);
        assert_eq!(parse_target("/r/a/b", None), None);
        assert_eq!(parse_target("/r/%zz", None), None);
    }
}
