//! `calibration-studio-server`: Calibration Studio in a browser. One axum
//! server carries studio-core's command table over HTTP, as the desktop
//! app's Tauri shell carries it over IPC.
//!
//! Routes:
//!
//! ```text
//! POST /api/login       {token} → Set-Cookie                 (public)
//! POST /api/logout
//! GET  /api/health      {"ok":true}                          (public)
//! POST /api/cmd/{name}  JSON args → JSON reply | StudioError
//! GET  /api/events      SSE; resumes from Last-Event-ID or ?after=
//! GET  /api/r/{name}?q= resource bytes; q = the query as JSON
//! GET  /*               the frontend; unknown paths → index.html (public)
//! ```
//!
//! Every request passes [`guard`] (the Host, the Origin and content type of
//! a non-GET, the query's size, no CORS, security headers) and every `/api`
//! route but login and health passes [`auth`]. Commands run on blocking
//! threads behind a semaphore with a time limit; jobs run on core's own
//! workers.

pub mod assets;
pub mod auth;
pub mod config;
pub mod guard;
pub mod routes;
pub mod token;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use axum::Router;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use studio_core::api::{Api, Dirs, Setup, StudioError};
use tokio::sync::Semaphore;

use config::{Config, Dist};

/// The largest JSON body read (a command's arguments).
pub const MAX_BODY: usize = 4 << 20;
/// The longest query string (a resource's query).
pub const MAX_QUERY: usize = 8 << 10;
/// The most event streams open at once.
pub const MAX_SUBSCRIBERS: usize = 32;
/// How long a plain command may run.
pub const COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// What every handler reads.
pub struct AppState {
    pub api: Arc<Api>,
    pub auth: auth::Auth,
    pub guard: guard::Guard,
    pub dist: Dist,
    /// Plain commands running at once.
    pub commands: Semaphore,
    /// Event streams open.
    pub subscribers: AtomicUsize,
    /// The Content Security Policy of every response.
    pub csp: String,
}

impl AppState {
    /// The state `config` describes, with the API's job workers running.
    pub fn new(config: &Config) -> Self {
        let api = Api::new(Setup {
            dirs: Dirs {
                cache: config.cache_dir.clone(),
                state: config.state_dir.clone(),
                samples: config.state_dir.join("samples"),
            },
            fixed_roots: Some(config.roots.clone()),
            shell: "server",
        });
        let parallel = std::thread::available_parallelism().map_or(2, |n| n.get());
        Self {
            api: Arc::new(api),
            auth: auth::Auth::new(config.token.clone(), config.public_origin.as_deref()),
            guard: guard::Guard::new(config),
            csp: guard::csp(&assets::inline_script_hashes(&config.dist)),
            dist: config.dist.clone(),
            commands: Semaphore::new(parallel),
            subscribers: AtomicUsize::new(0),
        }
    }
}

/// The server's routes over `state`.
pub fn router(state: Arc<AppState>) -> Router {
    routes::router(Arc::clone(&state))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            guard::guard,
        ))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            guard::security_headers,
        ))
}

/// `error` as the response: the envelope, with its kind's status.
pub fn envelope(error: &StudioError) -> Response {
    let status =
        StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = (status, axum::Json(error)).into_response();
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

/// Serve `config` until Ctrl-C.
pub async fn serve(config: Config) -> Result<(), String> {
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .map_err(|e| format!("cannot listen on {}: {e}", config.bind))?;
    let bound = listener
        .local_addr()
        .map_err(|e| format!("the listener has no address: {e}"))?;
    let state = Arc::new(AppState::new(&Config {
        bind: bound,
        ..config.clone()
    }));
    let url = assets::announce(&config, bound, &state);
    if config.open && open_browser(&url).is_err() {
        eprintln!("  could not open a browser; open the address above yourself");
    }
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .map_err(|e| format!("the server stopped: {e}"))
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
