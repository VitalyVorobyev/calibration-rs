//! The back end's one API, which both shells (the Tauri app and the
//! `calib-viewer` HTTP server) call: [`Api::invoke`] a command by name with
//! its JSON arguments, [`Api::resource`] for bytes, [`Api::subscribe`] to the
//! event stream. The commands are the [`table`]; their schema, from which
//! the TypeScript client is generated, is [`Api::schema`].

mod commands;
mod error;
mod events;
mod jobs;
mod schema;
mod table;

use std::collections::BTreeMap;
use std::sync::Arc;

pub use error::{ErrorKind, StudioError};
pub use events::{EventBody, EventBus, JobEvent, ServerEvent, Subscription};
pub use jobs::{JobCx, JobInfo, JobLimits, JobManager, JobState, Lane};
pub use table::{CommandSpec, Kind, Resource, job, plain, resource, table};

use crate::export_cache::ExportCache;
use crate::scope::RootScope;
use table::Handler;

/// What every command runs against.
pub struct Ctx {
    /// The folders commands may read and write under.
    pub scope: RootScope,
    /// The export the viewers show.
    pub exports: ExportCache,
    pub jobs: JobManager,
    pub events: EventBus,
}

impl Ctx {
    pub(crate) fn with_jobs(scope: RootScope, jobs: JobManager, events: EventBus) -> Self {
        Self {
            scope,
            exports: ExportCache::new(),
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
    /// The API over `scope`, with its job workers running.
    pub fn new(scope: RootScope) -> Self {
        let events = EventBus::new();
        let ctx = Arc::new(Ctx::with_jobs(
            scope,
            JobManager::new(events.clone()),
            events,
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

    /// The bytes of resource `name` for `query`.
    pub fn resource(&self, name: &str, query: serde_json::Value) -> Result<Resource, StudioError> {
        match &self.spec(name)?.handler {
            Handler::Resource(handler) => handler(&self.ctx, query),
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

    /// `studio-api.json`: every command with its types, the event and error
    /// types, and every wire type the frontend reads.
    pub fn schema() -> serde_json::Value {
        schema::build(&table())
    }
}
