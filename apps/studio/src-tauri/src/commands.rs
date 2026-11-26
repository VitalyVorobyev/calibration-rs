//! The Tauri transport of studio-core's command table: four Tauri commands
//! carry every command, job, event and resource the frontend uses.
//!
//! - `studio_invoke { name, args }` — [`Api::invoke`]: a plain command's
//!   reply or a job's `JobInfo`; rejects with the `StudioError` envelope.
//! - `studio_subscribe { after, onEvent }` — the event stream (job states
//!   and progress) over one channel; answers the subscription id.
//! - `studio_unsubscribe { id }` — ends it.
//! - `studio_resource { name, query }` — a resource's bytes as a binary IPC
//!   response.
//!
//! Every body runs on a blocking thread: commands are synchronous and some
//! (opening a project, reading an export) touch the disk.

use std::sync::Arc;

use serde_json::Value;
use studio_core::api::{Api, ServerEvent, StudioError};
use tauri::State;
use tauri::ipc::{Channel, Response};

/// The API the app's commands run against.
pub type SharedApi = Arc<Api>;

fn panicked(e: impl std::fmt::Display) -> StudioError {
    StudioError::internal(format!("the command's task failed: {e}"))
}

/// Run command `name` with its argument object `args`.
#[tauri::command]
pub async fn studio_invoke(
    name: String,
    args: Value,
    api: State<'_, SharedApi>,
) -> Result<Value, StudioError> {
    let api = Arc::clone(&api);
    tauri::async_runtime::spawn_blocking(move || api.invoke(&name, args))
        .await
        .map_err(panicked)?
}

/// Subscribe to the event stream from after event `after` (from the latest
/// state of each live job without it); events arrive on `on_event` until
/// [`studio_unsubscribe`] or until the webview drops the channel.
#[tauri::command]
pub fn studio_subscribe(
    after: Option<u64>,
    on_event: Channel<ServerEvent>,
    api: State<'_, SharedApi>,
) -> u64 {
    let subscription = api.subscribe(after);
    let id = subscription.id;
    let api = Arc::clone(&api);
    std::thread::spawn(move || {
        for event in subscription.events {
            if on_event.send(event).is_err() {
                break;
            }
        }
        api.unsubscribe(id);
    });
    id
}

/// End subscription `id`.
#[tauri::command]
pub fn studio_unsubscribe(id: u64, api: State<'_, SharedApi>) {
    api.unsubscribe(id);
}

/// The bytes of resource `name` for `query` (a PNG for
/// `undistorted_image`), as a binary IPC response rather than text.
#[tauri::command]
pub async fn studio_resource(
    name: String,
    query: Value,
    api: State<'_, SharedApi>,
) -> Result<Response, StudioError> {
    let api = Arc::clone(&api);
    let resource = tauri::async_runtime::spawn_blocking(move || api.resource(&name, query))
        .await
        .map_err(panicked)??;
    Ok(Response::new(resource.body))
}
