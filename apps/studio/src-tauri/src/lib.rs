//! Tauri 2 shell of Calibration Studio, the calibration-rs desktop app.
//!
//! What the app computes lives in `studio_core`, behind its command table
//! (`studio_core::api`). This crate carries the table to the React frontend:
//!
//! - [`commands`] — the Tauri transport: `studio_invoke` and the event
//!   subscription, over one `Api`.
//! - [`scheme`] — the `studio:` URI scheme: the table's resources (source
//!   images, undistorted views, blobs) by URL, as the browser gets them from
//!   `/api/r/…` of `calibration-studio-server`.
//!
//! The desktop's root folders start empty and are kept in its settings: the
//! only native dialog left, Settings → "Add folder…", is how the user grants
//! one (`add_root`).

mod commands;
mod scheme;

use std::sync::Arc;

use studio_core::api::{Api, Dirs, Setup};
use tauri::Manager;

/// The API of the desktop app: its platform folders, its roots from its
/// settings.
pub fn desktop_api() -> Api {
    Api::new(Setup {
        dirs: Dirs::desktop(),
        fixed_roots: None,
        shell: "desktop",
    })
}

/// `base` with the shell's commands and the `studio:` scheme over `api`
/// (the app's builder, and the mock runtime's in the tests).
pub fn builder<R: tauri::Runtime>(base: tauri::Builder<R>, api: Arc<Api>) -> tauri::Builder<R> {
    base.manage::<commands::SharedApi>(api)
        .register_asynchronous_uri_scheme_protocol(scheme::SCHEME, |ctx, request, responder| {
            let api = Arc::clone(ctx.app_handle().state::<commands::SharedApi>().inner());
            tauri::async_runtime::spawn_blocking(move || {
                responder.respond(scheme::respond(&api, &request));
            });
        })
        .invoke_handler(tauri::generate_handler![
            commands::studio_invoke,
            commands::studio_subscribe,
            commands::studio_unsubscribe,
        ])
}

/// Entry point invoked from `main.rs`.
pub fn run() {
    builder(tauri::Builder::default(), Arc::new(desktop_api()))
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("failed to launch Calibration Studio");
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use serde_json::{Value, json};
    use studio_core::api::{Api, Dirs, Setup};
    use tauri::ipc::{CallbackFn, InvokeBody};
    use tauri::test::{
        INVOKE_KEY, MockRuntime, get_ipc_response, mock_builder, mock_context, noop_assets,
    };
    use tauri::webview::InvokeRequest;

    /// The shell on the mock runtime over the e2e fixture library.
    fn app(state: &Path) -> (tauri::App<MockRuntime>, std::path::PathBuf) {
        let library = dunce::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../e2e/fixtures/library"),
        )
        .unwrap();
        let api = Api::new(Setup {
            dirs: Dirs {
                cache: state.join("cache"),
                state: state.join("state"),
                samples: state.join("samples"),
            },
            fixed_roots: Some(vec![library.clone()]),
            shell: "desktop",
        });
        let app = super::builder(mock_builder(), Arc::new(api))
            .build(mock_context(noop_assets()))
            .unwrap();
        (app, library)
    }

    fn invoke(
        webview: &tauri::WebviewWindow<MockRuntime>,
        name: &str,
        args: Value,
    ) -> Result<Value, Value> {
        get_ipc_response(
            webview,
            InvokeRequest {
                cmd: "studio_invoke".into(),
                callback: CallbackFn(0),
                error: CallbackFn(1),
                url: "tauri://localhost".parse().unwrap(),
                body: InvokeBody::Json(json!({ "name": name, "args": args })),
                headers: Default::default(),
                invoke_key: INVOKE_KEY.to_string(),
            },
        )
        .map(|body| body.deserialize::<Value>().unwrap())
    }

    #[test]
    fn commands_and_jobs_answer_over_ipc() {
        let tmp = tempfile::tempdir().unwrap();
        let (app, library) = app(tmp.path());
        let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .unwrap();

        let listing = invoke(
            &webview,
            "list_dir",
            json!({ "path": library.join("rig-demo") }),
        )
        .unwrap();
        assert!(
            listing["entries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["name"] == "runs")
        );
        // The envelope comes back as the rejection.
        let refused =
            invoke(&webview, "list_dir", json!({ "path": library.join("..") })).unwrap_err();
        assert_eq!(refused["kind"], "forbidden_path");

        // A job: started over IPC, followed by its status to its end.
        let job = invoke(&webview, "fetch_sample", json!({ "id": "no-such-sample" })).unwrap();
        let id = job["id"].clone();
        let mut state = Value::Null;
        for _ in 0..200 {
            state = invoke(&webview, "job_status", json!({ "id": id })).unwrap()["state"].clone();
            if state["kind"] == "failed" {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(state["kind"], "failed", "{state}");
    }

    #[test]
    fn the_studio_scheme_serves_an_image_inside_the_roots_only() {
        let tmp = tempfile::tempdir().unwrap();
        let (app, library) = app(tmp.path());
        let api: tauri::State<'_, super::commands::SharedApi> = tauri::Manager::state(&app);
        let target = |path: &Path| {
            let query = json!({ "path": path }).to_string();
            // As `convertFileSrc("r/image?q=…", "studio")` writes it.
            let encoded: String = format!("r/image?q={query}")
                .bytes()
                .map(|b| match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        (b as char).to_string()
                    }
                    _ => format!("%{b:02X}"),
                })
                .collect();
            format!("studio://localhost/{encoded}")
        };
        let request = |uri: String, origin: &str| {
            tauri::http::Request::get(uri)
                .header("origin", origin)
                .body(Vec::new())
                .unwrap()
        };
        let image = library.join("rig-demo/images/cam0_view0.png");
        let ok = super::scheme::respond(&api, &request(target(&image), "tauri://localhost"));
        assert_eq!(ok.status(), 200);
        assert_eq!(ok.headers()["content-type"], "image/png");
        assert_eq!(
            ok.headers()["access-control-allow-origin"],
            "tauri://localhost"
        );
        assert!(ok.body().starts_with(b"\x89PNG"));
        let outside = super::scheme::respond(
            &api,
            &request(
                target(&library.join("../../../package.json")),
                "tauri://localhost",
            ),
        );
        assert_eq!(outside.status(), 403);
        // A page from elsewhere is not named as an allowed origin.
        let foreign = super::scheme::respond(&api, &request(target(&image), "http://evil.example"));
        assert!(
            foreign
                .headers()
                .get("access-control-allow-origin")
                .is_none()
        );
    }
}
