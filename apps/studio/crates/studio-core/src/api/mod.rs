//! The back end's one API, which both shells (the Tauri app and the
//! `calib-viewer` HTTP server) call: [`Api::invoke`] a command by name with
//! its JSON arguments, [`Api::resource`] for bytes (a file or a blob, with
//! its HTTP semantics), [`Api::subscribe`] to the event stream. The commands are the [`table`]; their schema, from which
//! the TypeScript client is generated, is [`Api::schema`].

mod commands;
mod error;
mod events;
mod jobs;
pub mod resource;
mod schema;
mod table;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

pub use error::{ErrorKind, StudioError};
pub use events::{EventBody, EventBus, JobEvent, ServerEvent, Subscription};
pub use jobs::{JobCx, JobInfo, JobLimits, JobManager, JobState, Lane};
pub use resource::{Body, Conditions, Resource, ResourceReply};
pub use table::{CommandSpec, Kind, job, plain, resource, table};

use crate::blobs::BlobStore;
use crate::export_cache::ExportCache;
use crate::images::ImageCache;
use crate::scope::RootScope;
use table::Handler;

/// The folders the back end keeps its own files in, chosen by the shell.
#[derive(Debug, Clone)]
pub struct Dirs {
    /// Derived images (transcodes, thumbnails, undistorted views).
    pub cache: PathBuf,
}

impl Dirs {
    /// The platform's folders for this app: the cache under the user's
    /// cache folder (the system's temporary folder without one).
    pub fn platform() -> Self {
        let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
        Self {
            cache: base.join("calibration-studio"),
        }
    }
}

/// What every command runs against.
pub struct Ctx {
    /// The folders commands may read and write under.
    pub scope: RootScope,
    /// The export the viewers show.
    pub exports: ExportCache,
    /// Derived images, on disk.
    pub images: ImageCache,
    /// Derived blobs, in memory.
    pub blobs: BlobStore,
    pub jobs: JobManager,
    pub events: EventBus,
}

impl Ctx {
    pub(crate) fn with_jobs(
        scope: RootScope,
        jobs: JobManager,
        events: EventBus,
        dirs: &Dirs,
    ) -> Self {
        Self {
            scope,
            exports: ExportCache::new(),
            images: ImageCache::new(&dirs.cache),
            blobs: BlobStore::default(),
            jobs,
            events,
        }
    }
}

/// The API over one [`Ctx`]: the command table and the job workers.
pub struct Api {
    ctx: Arc<Ctx>,
    specs: Vec<CommandSpec>,
    index: BTreeMap<&'static str, usize>,
}

impl Api {
    /// The API over `scope`, keeping its files in `dirs`, with its job
    /// workers running.
    pub fn new(scope: RootScope, dirs: Dirs) -> Self {
        let events = EventBus::new();
        let ctx = Arc::new(Ctx::with_jobs(
            scope,
            JobManager::new(events.clone()),
            events,
            &dirs,
        ));
        ctx.jobs.start_workers(Arc::downgrade(&ctx));
        let specs = table();
        let index = specs
            .iter()
            .enumerate()
            .map(|(i, spec)| (spec.name, i))
            .collect();
        Self { ctx, specs, index }
    }

    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    /// The kind of command `name`, if there is one.
    pub fn kind(&self, name: &str) -> Option<Kind> {
        self.spec(name).ok().map(|spec| spec.kind)
    }

    fn spec(&self, name: &str) -> Result<&CommandSpec, StudioError> {
        self.index
            .get(name)
            .map(|&i| &self.specs[i])
            .ok_or_else(|| {
                StudioError::new(ErrorKind::UnknownCommand, format!("no command {name}"))
            })
    }

    /// Run command `name` on `args`: a plain command's reply, or a job's
    /// `JobInfo` once it is queued.
    pub fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, StudioError> {
        let spec = self.spec(name)?;
        match (&spec.handler, spec.kind) {
            (Handler::Plain(handler), _) => handler(&self.ctx, args),
            (Handler::Job(handler), Kind::Job { lane }) => {
                let task = handler(args)?;
                let info = self.ctx.jobs.enqueue(name, lane, task)?;
                serde_json::to_value(info)
                    .map_err(|e| StudioError::internal(format!("serialize the job: {e}")))
            }
            _ => Err(StudioError::new(
                ErrorKind::UnknownCommand,
                format!("{name} is a resource, not a command"),
            )),
        }
    }

    /// Resource `name` for `query`, answered under the request's
    /// `conditions` (a 304 for a matching `If-None-Match`, a 206 for a
    /// `Range` on a file).
    pub fn resource(
        &self,
        name: &str,
        query: serde_json::Value,
        conditions: &Conditions,
    ) -> Result<ResourceReply, StudioError> {
        match &self.spec(name)?.handler {
            Handler::Resource(handler) => resource::reply(handler(&self.ctx, query)?, conditions),
            _ => Err(StudioError::new(
                ErrorKind::UnknownCommand,
                format!("{name} is a command, not a resource"),
            )),
        }
    }

    /// The event stream, from after `after` (see [`EventBus::subscribe`]).
    pub fn subscribe(&self, after: Option<u64>) -> Subscription {
        self.ctx
            .events
            .subscribe(after, || self.ctx.jobs.live_events())
    }

    pub fn unsubscribe(&self, id: u64) {
        self.ctx.events.unsubscribe(id);
    }

    /// An API over `scope` for a test that makes no derived image: its cache
    /// folder is never written.
    #[cfg(test)]
    pub(crate) fn for_tests(scope: RootScope) -> Self {
        Self::new(
            scope,
            Dirs {
                cache: std::env::temp_dir().join("studio-core-tests-unused-cache"),
            },
        )
    }

    /// `studio-api.json`: every command with its types, the event and error
    /// types, and every wire type the frontend reads.
    pub fn schema() -> serde_json::Value {
        schema::build(&table())
    }
}
