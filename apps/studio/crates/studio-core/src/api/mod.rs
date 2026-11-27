//! The back end's one API, which both shells (the Tauri app and
//! `calibration-studio-server`) call: [`Api::invoke`] a command by name with
//! its JSON arguments, [`Api::resource`] for bytes (a file or a blob, with
//! its HTTP semantics), [`Api::subscribe`] to the event stream. The
//! commands are the [`table`]; their schema, from which the TypeScript
//! client is generated, is [`Api::schema`]. A shell sets the API up with a
//! [`Setup`]: where its files go and which folders it serves.

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
use crate::settings::SettingsStore;
use table::Handler;

#[cfg(test)]
impl Setup {
    /// A server setup over shared temporary folders.
    pub(crate) fn for_tests(cache: Option<PathBuf>) -> Self {
        let base = std::env::temp_dir().join("studio-core-tests");
        Self {
            dirs: Dirs {
                cache: cache.unwrap_or_else(|| base.join("cache")),
                state: base.join("state"),
                samples: base.join("samples"),
            },
            fixed_roots: Some(Vec::new()),
            shell: "server",
        }
    }
}

/// The folders the back end keeps its own files in, chosen by the shell.
#[derive(Debug, Clone)]
pub struct Dirs {
    /// Derived images (transcodes, thumbnails, undistorted views).
    pub cache: PathBuf,
    /// `settings.json` (the desktop's roots) and `state.json` (recent
    /// projects).
    pub state: PathBuf,
    /// The Quick Start sample datasets, downloaded on first use: always a
    /// root.
    pub samples: PathBuf,
}

impl Dirs {
    /// The desktop app's folders: the cache, config and data folders of the
    /// platform (`~/Library/…` on macOS, the XDG folders on Linux,
    /// `AppData` on Windows), each with a `calibration-studio` folder;
    /// `VISION_CALIBRATION_DATA` overrides the samples' folder.
    pub fn desktop() -> Self {
        let app = |base: Option<PathBuf>| {
            base.unwrap_or_else(std::env::temp_dir)
                .join("calibration-studio")
        };
        let samples = std::env::var_os("VISION_CALIBRATION_DATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| app(dirs::data_dir()).join("samples"));
        Self {
            cache: app(dirs::cache_dir()),
            state: app(dirs::config_dir()),
            samples,
        }
    }
}

/// How a shell sets the back end up.
#[derive(Debug, Clone)]
pub struct Setup {
    pub dirs: Dirs,
    /// The server's root folders, fixed while it runs. `None`: the
    /// desktop's, kept in `settings.json` and changed by `add_root` and
    /// `remove_root` (empty on the first run).
    pub fixed_roots: Option<Vec<PathBuf>>,
    /// The shell, for `server_info` (`"desktop"`, `"server"`).
    pub shell: &'static str,
}

/// What every command runs against.
pub struct Ctx {
    /// The folders commands may read and write under.
    pub scope: RootScope,
    /// The desktop's roots and everyone's recent projects.
    pub settings: SettingsStore,
    pub dirs: Dirs,
    /// The shell, for `server_info`.
    pub shell: &'static str,
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
        setup: Setup,
    ) -> Self {
        let settings = SettingsStore::open(&setup.dirs.state, setup.fixed_roots.is_none());
        Self {
            scope,
            settings,
            images: ImageCache::new(&setup.dirs.cache),
            dirs: setup.dirs,
            shell: setup.shell,
            exports: ExportCache::new(),
            blobs: BlobStore::default(),
            jobs,
            events,
        }
    }

    /// The roots the scope should hold: the fixed or stored ones, and the
    /// samples' folder.
    pub(crate) fn root_folders(&self, stored: &[String]) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = stored.iter().map(PathBuf::from).collect();
        roots.push(self.dirs.samples.clone());
        roots
    }
}

/// The API over one [`Ctx`]: the command table and the job workers.
pub struct Api {
    ctx: Arc<Ctx>,
    specs: Vec<CommandSpec>,
    index: BTreeMap<&'static str, usize>,
}

impl Api {
    /// The API `setup` describes, with its job workers running. The
    /// samples' folder is created (it is always a root).
    pub fn new(setup: Setup) -> Self {
        let _ = std::fs::create_dir_all(&setup.dirs.samples);
        let scope = match &setup.fixed_roots {
            Some(roots) => {
                let mut roots = roots.clone();
                roots.push(setup.dirs.samples.clone());
                RootScope::fixed(roots)
            }
            None => {
                let stored = SettingsStore::open(&setup.dirs.state, true).roots();
                let mut roots: Vec<PathBuf> = stored.iter().map(PathBuf::from).collect();
                roots.push(setup.dirs.samples.clone());
                RootScope::editable(roots)
            }
        };
        Self::with_scope(scope, setup)
    }

    /// The API over `scope` (tests choose theirs).
    pub(crate) fn with_scope(scope: RootScope, setup: Setup) -> Self {
        let events = EventBus::new();
        let ctx = Arc::new(Ctx::with_jobs(
            scope,
            JobManager::new(events.clone()),
            events,
            setup,
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

    /// An API over `scope` for a test that makes no derived image: its
    /// folders are a shared temporary one.
    #[cfg(test)]
    pub(crate) fn for_tests(scope: RootScope) -> Self {
        Self::with_scope(scope, Setup::for_tests(None))
    }

    /// `studio-api.json`: every command with its types, the event and error
    /// types, and every wire type the frontend reads.
    pub fn schema() -> serde_json::Value {
        schema::build(&table())
    }
}
