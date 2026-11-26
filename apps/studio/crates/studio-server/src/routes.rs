//! The routes: login and health in the open, the command table behind the
//! token, the frontend as the fallback.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::Stream;
use serde::Deserialize;
use studio_core::api::{Conditions, ErrorKind, ServerEvent, StudioError};

use crate::{AppState, COMMAND_TIMEOUT, MAX_BODY, MAX_SUBSCRIBERS, assets, auth, envelope};

/// The routes over `state` (without the guard, see [`crate::router`]).
pub fn router(state: Arc<AppState>) -> Router {
    let api = Router::new()
        .route("/api/logout", post(auth::logout))
        .route("/api/cmd/{name}", post(command))
        .route("/api/events", get(events))
        .route("/api/r/{name}", get(resource))
        // Any other `/api` path is no route (and not the page).
        .route("/api", get(no_api).post(no_api))
        .route("/api/{*rest}", get(no_api).post(no_api))
        .route_layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            auth::require,
        ));
    Router::new()
        .route("/api/login", post(auth::login))
        .route("/api/health", get(health))
        .merge(api)
        .fallback(assets::serve)
        .with_state(state)
}

async fn health() -> Response {
    axum::Json(serde_json::json!({ "ok": true })).into_response()
}

async fn no_api() -> Response {
    envelope(&StudioError::new(
        ErrorKind::UnknownCommand,
        "no such route",
    ))
}

/// `POST /api/cmd/{name}`: a plain command's reply, or a job's `JobInfo`.
async fn command(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    request: Request,
) -> Response {
    let bytes = match to_bytes(request.into_body(), MAX_BODY).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return envelope(&StudioError::new(
                ErrorKind::TooLarge,
                "the request body is larger than 4 MiB",
            ));
        }
    };
    let args = if bytes.iter().all(u8::is_ascii_whitespace) {
        serde_json::Value::Null
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(args) => args,
            Err(e) => {
                return envelope(&StudioError::new(
                    ErrorKind::InvalidArgs,
                    format!("{name}: the body is not JSON: {e}"),
                ));
            }
        }
    };
    let Ok(_permit) = state.commands.acquire().await else {
        return envelope(&StudioError::internal("the server is shutting down"));
    };
    let api = Arc::clone(&state.api);
    let task = tokio::task::spawn_blocking(move || api.invoke(&name, args));
    let reply = match tokio::time::timeout(COMMAND_TIMEOUT, task).await {
        Err(_) => Err(StudioError::internal(
            "the command took longer than 120 s; it keeps running, its reply is lost",
        )),
        Ok(Err(e)) => Err(StudioError::internal(format!("the command failed: {e}"))),
        Ok(Ok(reply)) => reply,
    };
    match reply {
        Ok(value) => {
            let mut response = axum::Json(value).into_response();
            response
                .headers_mut()
                .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
            response
        }
        Err(e) => envelope(&e),
    }
}

#[derive(Debug, Deserialize)]
struct ResourceQuery {
    #[serde(default)]
    q: Option<String>,
}

/// `GET /api/r/{name}?q=<query JSON>`: a resource, with its ETag, range and
/// caching.
async fn resource(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<ResourceQuery>,
    headers: HeaderMap,
) -> Response {
    let query: serde_json::Value = match serde_json::from_str(query.q.as_deref().unwrap_or("{}")) {
        Ok(query) => query,
        Err(e) => {
            return envelope(&StudioError::new(
                ErrorKind::InvalidArgs,
                format!("{name}: `q` is not JSON: {e}"),
            ));
        }
    };
    let header_value = |name: header::HeaderName| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    let conditions = Conditions {
        if_none_match: header_value(header::IF_NONE_MATCH),
        range: header_value(header::RANGE),
    };
    let api = Arc::clone(&state.api);
    let task = tokio::task::spawn_blocking(move || {
        let reply = api.resource(&name, query, &conditions)?;
        let body = reply
            .read_body()
            .map_err(|e| StudioError::internal(format!("read the resource: {e}")))?;
        Ok::<_, StudioError>((reply, body))
    });
    let (reply, body) = match task.await {
        Ok(Ok(answer)) => answer,
        Ok(Err(e)) => return envelope(&e),
        Err(e) => return envelope(&StudioError::internal(format!("the resource failed: {e}"))),
    };
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = StatusCode::from_u16(reply.status).unwrap_or(StatusCode::OK);
    let headers = response.headers_mut();
    let mut set = |name: header::HeaderName, value: &str| {
        if let Ok(value) = HeaderValue::from_str(value) {
            headers.insert(name, value);
        }
    };
    set(header::CONTENT_TYPE, reply.media_type);
    set(header::ETAG, &reply.etag);
    set(header::CACHE_CONTROL, reply.cache_control);
    set(header::ACCEPT_RANGES, "bytes");
    if let Some(range) = &reply.content_range {
        set(header::CONTENT_RANGE, range);
    }
    response
}

#[derive(Debug, Deserialize)]
struct EventsQuery {
    #[serde(default)]
    after: Option<u64>,
}

/// One open event stream; its slot is given back when it ends.
struct Slot(Arc<AppState>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.subscribers.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `GET /api/events`: the event stream as server-sent events, from after
/// `Last-Event-ID` (a reconnecting `EventSource`) or `?after=`.
async fn events(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EventsQuery>,
    headers: HeaderMap,
) -> Response {
    let open = state.subscribers.fetch_add(1, Ordering::SeqCst);
    let slot = Slot(Arc::clone(&state));
    if open >= MAX_SUBSCRIBERS {
        return envelope(&StudioError::new(
            ErrorKind::Busy,
            "too many event streams are open",
        ));
    }
    let after = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse().ok())
        .or(query.after);
    let subscription = state.api.subscribe(after);
    let (sender, receiver) = tokio::sync::mpsc::channel::<ServerEvent>(64);
    // Core's stream is a std channel: a thread carries it over, and notices a
    // closed connection within a second even when no event comes.
    let api = Arc::clone(&state.api);
    std::thread::spawn(move || {
        loop {
            match subscription.events.recv_timeout(Duration::from_secs(1)) {
                Ok(event) => {
                    if sender.blocking_send(event).is_err() {
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if sender.is_closed() {
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        api.unsubscribe(subscription.id);
    });
    let stream = event_stream(receiver, slot);
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response();
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

fn event_stream(
    receiver: tokio::sync::mpsc::Receiver<ServerEvent>,
    slot: Slot,
) -> impl Stream<Item = Result<Event, Infallible>> {
    futures_util::stream::unfold((receiver, slot), |(mut receiver, slot)| async move {
        let event = receiver.recv().await?;
        let data = serde_json::to_string(&event).unwrap_or_default();
        Some((
            Ok(Event::default().id(event.seq.to_string()).data(data)),
            (receiver, slot),
        ))
    })
}
