//! The command table: every command the back end offers, declared once as
//! a typed function and listed once in [`table`]. Its argument, reply, event
//! and output types come from the function's signature; the transports
//! (Tauri IPC, the server's HTTP) dispatch through the table, and
//! the TypeScript client is generated from the same list.

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::Ctx;
use super::commands::{files, forms, jobs, project, resources, session, setup, viewer};
use super::error::{ErrorKind, StudioError};
use super::jobs::{JobCx, JobTask, Lane};
use super::resource::Resource;

/// What kind of command an entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Arguments in, reply out.
    Plain,
    /// Queued on `lane`; answers its `JobInfo`, then reports events and ends
    /// with an output (or an error) on the event stream.
    Job { lane: Lane },
    /// A typed query in, a file or bytes out (behind a URL).
    Resource,
}

type PlainHandler =
    Box<dyn Fn(&Ctx, serde_json::Value) -> Result<serde_json::Value, StudioError> + Send + Sync>;
type JobHandler = Box<dyn Fn(serde_json::Value) -> Result<JobTask, StudioError> + Send + Sync>;
type ResourceHandler =
    Box<dyn Fn(&Ctx, serde_json::Value) -> Result<Resource, StudioError> + Send + Sync>;

pub(crate) enum Handler {
    Plain(PlainHandler),
    Job(JobHandler),
    Resource(ResourceHandler),
}

/// The schemas of one command's types, as `$ref`s into one generator.
pub(crate) struct CommandSchema {
    pub args: Schema,
    pub reply: Option<Schema>,
    pub event: Option<Schema>,
    pub output: Option<Schema>,
}

type SchemaFn = Box<dyn Fn(&mut SchemaGenerator) -> CommandSchema + Send + Sync>;

/// One line of the table.
pub struct CommandSpec {
    pub name: &'static str,
    pub kind: Kind,
    pub(crate) handler: Handler,
    pub(crate) schema: SchemaFn,
}

/// `args` as the argument object `A` (no arguments at all reads as `{}`).
fn parse<A: DeserializeOwned>(name: &str, args: serde_json::Value) -> Result<A, StudioError> {
    let args = if args.is_null() {
        serde_json::json!({})
    } else {
        args
    };
    serde_json::from_value(args)
        .map_err(|e| StudioError::new(ErrorKind::InvalidArgs, format!("{name}: {e}")))
}

fn to_value<R: Serialize>(name: &str, reply: &R) -> Result<serde_json::Value, StudioError> {
    serde_json::to_value(reply)
        .map_err(|e| StudioError::internal(format!("{name}: serialize the reply: {e}")))
}

/// A plain command: `f` answers its arguments.
pub fn plain<A, R>(name: &'static str, f: fn(&Ctx, A) -> Result<R, StudioError>) -> CommandSpec
where
    A: DeserializeOwned + JsonSchema + 'static,
    R: Serialize + JsonSchema + 'static,
{
    CommandSpec {
        name,
        kind: Kind::Plain,
        handler: Handler::Plain(Box::new(move |ctx, args| {
            let reply = f(ctx, parse(name, args)?)?;
            to_value(name, &reply)
        })),
        schema: Box::new(|generator| CommandSchema {
            args: generator.subschema_for::<A>(),
            reply: Some(generator.subschema_for::<R>()),
            event: None,
            output: None,
        }),
    }
}

/// A job: `f` runs on `lane`'s worker, reports `E`s and ends with an `R`.
/// Its arguments are checked when it is started, before it is queued.
pub fn job<A, E, R>(
    name: &'static str,
    lane: Lane,
    f: fn(&Ctx, A, &JobCx<E>) -> Result<R, StudioError>,
) -> CommandSpec
where
    A: DeserializeOwned + JsonSchema + Send + 'static,
    E: Serialize + JsonSchema + 'static,
    R: Serialize + JsonSchema + 'static,
{
    CommandSpec {
        name,
        kind: Kind::Job { lane },
        handler: Handler::Job(Box::new(move |args| {
            let args: A = parse(name, args)?;
            let task: JobTask = Box::new(move |ctx, sink| {
                let output = f(ctx, args, &JobCx::new(sink))?;
                to_value(name, &output)
            });
            Ok(task)
        })),
        schema: Box::new(|generator| CommandSchema {
            args: generator.subschema_for::<A>(),
            reply: None,
            event: Some(generator.subschema_for::<E>()),
            output: Some(generator.subschema_for::<R>()),
        }),
    }
}

/// A resource: `f` answers its query with a file or bytes.
pub fn resource<Q>(
    name: &'static str,
    f: fn(&Ctx, Q) -> Result<Resource, StudioError>,
) -> CommandSpec
where
    Q: DeserializeOwned + JsonSchema + 'static,
{
    CommandSpec {
        name,
        kind: Kind::Resource,
        handler: Handler::Resource(Box::new(move |ctx, query| f(ctx, parse(name, query)?))),
        schema: Box::new(|generator| CommandSchema {
            args: generator.subschema_for::<Q>(),
            reply: None,
            event: None,
            output: None,
        }),
    }
}

/// The one list. Its order is the order of the generated client.
pub fn table() -> Vec<CommandSpec> {
    vec![
        // The session: this back end, its roots, the recent projects.
        plain("server_info", session::server_info),
        plain("add_root", session::add_root),
        plain("remove_root", session::remove_root),
        plain("recent_projects", session::recent_projects),
        plain("forget_recent", session::forget_recent),
        // The folder browser.
        plain("list_dir", files::list_dir),
        // Projects.
        plain("create_project", project::create_project),
        plain("open_project", project::open_project),
        plain("save_project_manifest", project::save_project_manifest),
        plain("save_project_run", project::save_project_run),
        // The forms, from the library's schemas and defaults.
        plain("config_form", forms::config_form),
        plain("dataset_form", forms::dataset_form),
        plain("resolve_config", forms::resolve_config),
        // The Run workspace's setup and the runner.
        plain("read_manifest", setup::read_manifest),
        plain("sniff_folder", setup::sniff_folder),
        job("fetch_sample", Lane::Light, setup::fetch_sample),
        job("run_calibration", Lane::Heavy, setup::run_calibration),
        // The viewers, over the active export.
        plain("load_export", viewer::load_export),
        plain("set_active_export", viewer::set_active_export),
        plain("undistort_points", viewer::undistort_points),
        plain("epipolar_overlay", viewer::epipolar_overlay),
        plain(
            "epipolar_overlay_undistorted",
            viewer::epipolar_overlay_undistorted,
        ),
        plain("camera_parameter_values", viewer::camera_parameter_values),
        plain("projection_uncertainty", viewer::projection_uncertainty),
        job("disparity", Lane::Heavy, viewer::disparity),
        // Resources: the bytes behind the frontend's URLs.
        resource("image", resources::image),
        resource("undistorted_image", resources::undistorted_image),
        resource("blob", resources::blob),
        // Jobs.
        plain("list_jobs", jobs::list_jobs),
        plain("job_status", jobs::job_status),
        plain("cancel_job", jobs::cancel_job),
    ]
}
