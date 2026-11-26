//! The job registry's commands: what is running or finished, one job's
//! state, and cancelling one.

use schemars::JsonSchema;
use serde::Deserialize;

use crate::api::{Ctx, JobInfo, StudioError};

/// Every job the back end still knows of, live or recently finished, in
/// start order.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListJobsArgs {}

pub fn list_jobs(ctx: &Ctx, _: ListJobsArgs) -> Result<Vec<JobInfo>, StudioError> {
    Ok(ctx.jobs.list())
}

/// The job `id`: its state, its last progress report, its output once
/// done.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobStatusArgs {
    pub id: String,
}

pub fn job_status(ctx: &Ctx, args: JobStatusArgs) -> Result<JobInfo, StudioError> {
    ctx.jobs.status(&args.id)
}

/// Cancel the job `id`: a queued job at once, a running one at its next
/// image or solver iteration. A finished job is answered as it is.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CancelJobArgs {
    pub id: String,
}

pub fn cancel_job(ctx: &Ctx, args: CancelJobArgs) -> Result<JobInfo, StudioError> {
    ctx.jobs.cancel(&args.id)
}
