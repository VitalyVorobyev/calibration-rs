//! Jobs: long commands that run on the back end's own threads, report
//! progress as events, and can be cancelled.
//!
//! Starting a job only queues it and answers its [`JobInfo`]; the state
//! changes (`queued` with its position, `running`, then `done`, `failed` or
//! `cancelled`) and each progress report go out on the [`EventBus`]. Each
//! [`Lane`] has one worker and a FIFO queue, so one heavy job (a
//! calibration) runs at a time. The registry keeps every live job and the
//! latest terminal ones, with their last report and output, so a client that
//! reloads finds them again.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::Serialize;
use vision_calibration::control::CancelToken;

use super::Ctx;
use super::error::{ErrorKind, StudioError};
use super::events::{EventBody, EventBus, JobEvent};

/// Which worker runs a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// Calibrations: CPU- and memory-heavy, one at a time.
    Heavy,
    /// Downloads and other waits that should not queue behind a
    /// calibration.
    Light,
}

const LANES: [Lane; 2] = [Lane::Heavy, Lane::Light];

/// Where a job is in its life.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobState {
    /// Waiting for its lane's worker; `position` 1 runs next.
    Queued {
        position: usize,
    },
    Running,
    /// Finished; `output` is the job command's output type.
    Done {
        output: serde_json::Value,
    },
    Failed {
        error: StudioError,
    },
    Cancelled,
}

impl JobState {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Done { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }
}

/// A job as the registry knows it.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct JobInfo {
    /// `<counter>-<8 random hex digits>`: unique within the process and not
    /// reused by the next one.
    pub id: String,
    /// The job command that started it.
    pub command: String,
    pub state: JobState,
    /// Its latest progress report, if it sent one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event: Option<serde_json::Value>,
}

/// How much the manager queues and keeps.
#[derive(Debug, Clone, Copy)]
pub struct JobLimits {
    /// Queued (not running) jobs per lane; one more is `busy`.
    pub queue: usize,
    /// Terminal jobs kept, newest first.
    pub keep_terminal: usize,
    /// How long a terminal job is kept at most.
    pub keep_for: Duration,
}

impl Default for JobLimits {
    fn default() -> Self {
        Self {
            queue: 8,
            keep_terminal: 32,
            keep_for: Duration::from_secs(3600),
        }
    }
}

/// The work of one job, run once on its lane's worker.
pub(crate) type JobTask =
    Box<dyn FnOnce(&Ctx, JobSink) -> Result<serde_json::Value, StudioError> + Send>;

struct Entry {
    info: JobInfo,
    lane: Lane,
    token: CancelToken,
    task: Option<JobTask>,
    /// When it reached a terminal state, and in which order.
    finished: Option<(Instant, u64)>,
}

#[derive(Default)]
struct Registry {
    /// In start order: the queues are this list filtered by lane and state.
    jobs: Vec<Entry>,
    finished: u64,
    shutdown: bool,
}

struct Shared {
    registry: Mutex<Registry>,
    wake: Condvar,
    bus: EventBus,
    limits: JobLimits,
    counter: AtomicU64,
    random: RandomState,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Registry> {
        // A job body never runs under this lock, so a poisoned lock means a
        // panic in the bookkeeping itself; the list is still whole.
        self.registry.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn publish(&self, info: &JobInfo, event: Option<serde_json::Value>) {
        self.bus.publish(EventBody::Job(JobEvent {
            id: info.id.clone(),
            command: info.command.clone(),
            state: info.state.clone(),
            event,
        }));
    }

    /// Renumber `lane`'s queue; the jobs whose position changed.
    fn renumber(registry: &mut Registry, lane: Lane) -> Vec<JobInfo> {
        let mut changed = Vec::new();
        let mut position = 0;
        for entry in registry.jobs.iter_mut().filter(|e| e.lane == lane) {
            if let JobState::Queued { position: old } = &mut entry.info.state {
                position += 1;
                if *old != position {
                    *old = position;
                    changed.push(entry.info.clone());
                }
            }
        }
        changed
    }

    /// Mark job `id` terminal, forget the oldest terminal jobs past the
    /// limits, and announce it.
    fn finish(&self, id: &str, state: JobState) {
        let info = {
            let mut registry = self.lock();
            registry.finished += 1;
            let order = registry.finished;
            let Some(entry) = registry.jobs.iter_mut().find(|e| e.info.id == id) else {
                return;
            };
            entry.info.state = state;
            entry.task = None;
            entry.finished = Some((Instant::now(), order));
            let info = entry.info.clone();
            self.prune(&mut registry);
            info
        };
        self.publish(&info, None);
    }

    fn prune(&self, registry: &mut Registry) {
        let now = Instant::now();
        let mut terminal: Vec<u64> = registry
            .jobs
            .iter()
            .filter_map(|e| e.finished.map(|(_, order)| order))
            .collect();
        terminal.sort_unstable();
        let cut = terminal.len().saturating_sub(self.limits.keep_terminal);
        let oldest_kept = terminal.get(cut).copied().unwrap_or(u64::MAX);
        registry.jobs.retain(|e| match e.finished {
            None => true,
            Some((at, order)) => order >= oldest_kept && now - at <= self.limits.keep_for,
        });
    }
}

/// The job registry and its workers.
pub struct JobManager {
    shared: Arc<Shared>,
}

impl JobManager {
    pub fn new(bus: EventBus) -> Self {
        Self::with_limits(bus, JobLimits::default())
    }

    pub fn with_limits(bus: EventBus, limits: JobLimits) -> Self {
        Self {
            shared: Arc::new(Shared {
                registry: Mutex::new(Registry::default()),
                wake: Condvar::new(),
                bus,
                limits,
                counter: AtomicU64::new(0),
                random: RandomState::new(),
            }),
        }
    }

    /// Start one worker per lane. They run jobs against `ctx` while it
    /// lives and stop when this manager is dropped.
    pub(crate) fn start_workers(&self, ctx: Weak<Ctx>) {
        for lane in LANES {
            let shared = Arc::clone(&self.shared);
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .name(format!("studio-jobs-{lane:?}").to_lowercase())
                .spawn(move || work(&shared, lane, &ctx))
                .map(drop)
                .unwrap_or_else(|e| panic!("cannot start a job worker thread: {e}"));
        }
    }

    /// Queue `task` as a job of `command` on `lane`.
    pub(crate) fn enqueue(
        &self,
        command: &str,
        lane: Lane,
        task: JobTask,
    ) -> Result<JobInfo, StudioError> {
        let shared = &self.shared;
        let info = {
            let mut registry = shared.lock();
            let queued = registry
                .jobs
                .iter()
                .filter(|e| e.lane == lane && matches!(e.info.state, JobState::Queued { .. }))
                .count();
            if queued >= shared.limits.queue {
                return Err(StudioError::new(
                    ErrorKind::Busy,
                    format!("{queued} jobs are already waiting; try again when one has run"),
                ));
            }
            let n = shared.counter.fetch_add(1, Ordering::Relaxed) + 1;
            let salt = shared.random.hash_one(n) as u32;
            let info = JobInfo {
                id: format!("{n}-{salt:08x}"),
                command: command.to_owned(),
                state: JobState::Queued {
                    position: queued + 1,
                },
                last_event: None,
            };
            registry.jobs.push(Entry {
                info: info.clone(),
                lane,
                token: CancelToken::new(),
                task: Some(task),
                finished: None,
            });
            info
        };
        shared.publish(&info, None);
        shared.wake.notify_all();
        Ok(info)
    }

    /// The job `id`.
    pub fn status(&self, id: &str) -> Result<JobInfo, StudioError> {
        self.shared
            .lock()
            .jobs
            .iter()
            .find(|e| e.info.id == id)
            .map(|e| e.info.clone())
            .ok_or_else(|| not_found(id))
    }

    /// Every job retained, in start order.
    pub fn list(&self) -> Vec<JobInfo> {
        self.shared
            .lock()
            .jobs
            .iter()
            .map(|e| e.info.clone())
            .collect()
    }

    /// Cancel job `id`: a queued job is cancelled at once, a running one at
    /// its next image or solver iteration; a finished one is left as it is.
    pub fn cancel(&self, id: &str) -> Result<JobInfo, StudioError> {
        let shared = &self.shared;
        let (info, renumbered, dequeued) = {
            let mut registry = shared.lock();
            let entry = registry
                .jobs
                .iter_mut()
                .find(|e| e.info.id == id)
                .ok_or_else(|| not_found(id))?;
            match entry.info.state {
                JobState::Queued { .. } => {
                    entry.task = None;
                    entry.info.state = JobState::Cancelled;
                    let lane = entry.lane;
                    let info = entry.info.clone();
                    (info, Shared::renumber(&mut registry, lane), true)
                }
                JobState::Running => {
                    entry.token.cancel();
                    (entry.info.clone(), Vec::new(), false)
                }
                _ => (entry.info.clone(), Vec::new(), false),
            }
        };
        if dequeued {
            shared.finish(id, JobState::Cancelled);
            for queued in &renumbered {
                shared.publish(queued, None);
            }
        }
        Ok(info)
    }

    /// Each live job's latest state, as the events a new subscriber starts
    /// from.
    pub(crate) fn live_events(&self) -> Vec<EventBody> {
        self.shared
            .lock()
            .jobs
            .iter()
            .filter(|e| !e.info.state.is_terminal())
            .map(|e| {
                EventBody::Job(JobEvent {
                    id: e.info.id.clone(),
                    command: e.info.command.clone(),
                    state: e.info.state.clone(),
                    event: e.info.last_event.clone(),
                })
            })
            .collect()
    }
}

impl Drop for JobManager {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_all();
    }
}

fn not_found(id: &str) -> StudioError {
    StudioError::new(ErrorKind::JobNotFound, format!("no job {id}"))
}

/// A lane's worker: take the oldest queued job, run it, record how it
/// ended; until the manager shuts down or the context is gone.
fn work(shared: &Arc<Shared>, lane: Lane, ctx: &Weak<Ctx>) {
    loop {
        let (info, task, token, renumbered) = {
            let mut registry = shared.lock();
            loop {
                if registry.shutdown {
                    return;
                }
                let next = registry
                    .jobs
                    .iter_mut()
                    .find(|e| e.lane == lane && matches!(e.info.state, JobState::Queued { .. }));
                if let Some(entry) = next {
                    entry.info.state = JobState::Running;
                    let picked = (entry.info.clone(), entry.task.take(), entry.token.clone());
                    let renumbered = Shared::renumber(&mut registry, lane);
                    break (picked.0, picked.1, picked.2, renumbered);
                }
                registry = shared
                    .wake
                    .wait(registry)
                    .unwrap_or_else(|e| e.into_inner());
            }
        };
        shared.publish(&info, None);
        for queued in &renumbered {
            shared.publish(queued, None);
        }
        let Some(ctx) = ctx.upgrade() else {
            return;
        };
        let Some(task) = task else {
            continue;
        };
        let sink = JobSink {
            id: info.id.clone(),
            command: info.command.clone(),
            shared: Arc::clone(shared),
            token,
        };
        let ended = catch_unwind(AssertUnwindSafe(|| task(&ctx, sink)));
        drop(ctx);
        let state = match ended {
            Ok(Ok(output)) => JobState::Done { output },
            Ok(Err(error)) if error.kind == ErrorKind::Cancelled => JobState::Cancelled,
            Ok(Err(error)) => JobState::Failed { error },
            Err(_) => JobState::Failed {
                error: StudioError::internal(format!("job {} panicked", info.command)),
            },
        };
        shared.finish(&info.id, state);
    }
}

/// A running job's channel back to the registry: its progress reports and
/// its cancellation.
#[derive(Clone)]
pub struct JobSink {
    id: String,
    command: String,
    shared: Arc<Shared>,
    token: CancelToken,
}

impl JobSink {
    fn emit(&self, event: serde_json::Value) {
        let info = {
            let mut registry = self.shared.lock();
            let Some(entry) = registry.jobs.iter_mut().find(|e| e.info.id == self.id) else {
                return;
            };
            entry.info.last_event = Some(event.clone());
            entry.info.clone()
        };
        debug_assert_eq!(info.command, self.command);
        self.shared.publish(&info, Some(event));
    }
}

/// What a job's function gets besides its arguments: where its progress
/// reports (`E`, its command's event type) go, and its cancel token.
pub struct JobCx<E> {
    sink: JobSink,
    _event: PhantomData<fn(E)>,
}

impl<E> Clone for JobCx<E> {
    fn clone(&self) -> Self {
        Self {
            sink: self.sink.clone(),
            _event: PhantomData,
        }
    }
}

impl<E: Serialize> JobCx<E> {
    pub(crate) fn new(sink: JobSink) -> Self {
        Self {
            sink,
            _event: PhantomData,
        }
    }

    /// Report progress.
    pub fn emit(&self, event: &E) {
        // A progress type that does not serialize is a bug of its own; the
        // job goes on without the report.
        if let Ok(value) = serde_json::to_value(event) {
            self.sink.emit(value);
        }
    }

    /// The token cancelling this job.
    pub fn token(&self) -> CancelToken {
        self.sink.token.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    use crate::scope::RootScope;

    /// A context whose job manager has `limits`, with its workers started.
    fn ctx(limits: JobLimits) -> Arc<Ctx> {
        let events = EventBus::new();
        let ctx = Arc::new(Ctx::with_jobs(
            RootScope::fixed([]),
            JobManager::with_limits(events.clone(), limits),
            events,
            &crate::api::Dirs {
                cache: std::env::temp_dir().join("studio-core-tests-unused-cache"),
            },
        ));
        ctx.jobs.start_workers(Arc::downgrade(&ctx));
        ctx
    }

    /// A job that waits for `gate`, reports one event and answers `value`.
    fn gated(gate: mpsc::Receiver<()>, value: i32) -> JobTask {
        Box::new(move |_, sink| {
            let cx = JobCx::<i32>::new(sink);
            let _ = gate.recv();
            cx.emit(&value);
            Ok(serde_json::json!(value))
        })
    }

    fn wait_for(ctx: &Ctx, id: &str, done: impl Fn(&JobState) -> bool) -> JobInfo {
        for _ in 0..500 {
            let info = ctx.jobs.status(id).unwrap();
            if done(&info.state) {
                return info;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("job {id} did not get there: {:?}", ctx.jobs.status(id));
    }

    #[test]
    fn a_lane_runs_one_job_at_a_time_in_order_and_cancels_queued_ones() {
        let ctx = ctx(JobLimits {
            queue: 2,
            ..JobLimits::default()
        });
        let (open_a, gate_a) = mpsc::channel();
        let (open_b, gate_b) = mpsc::channel();
        let (_open_c, gate_c) = mpsc::channel();
        let a = ctx
            .jobs
            .enqueue("t", Lane::Heavy, gated(gate_a, 1))
            .unwrap();
        wait_for(&ctx, &a.id, |s| *s == JobState::Running);
        let b = ctx
            .jobs
            .enqueue("t", Lane::Heavy, gated(gate_b, 2))
            .unwrap();
        let c = ctx
            .jobs
            .enqueue("t", Lane::Heavy, gated(gate_c, 3))
            .unwrap();
        assert_eq!(b.state, JobState::Queued { position: 1 });
        assert_eq!(c.state, JobState::Queued { position: 2 });
        // The queue is full.
        let (_o, g) = mpsc::channel();
        let busy = ctx.jobs.enqueue("t", Lane::Heavy, gated(g, 4)).unwrap_err();
        assert_eq!(busy.kind, ErrorKind::Busy);
        // Another lane is not behind them.
        let (open_l, gate_l) = mpsc::channel();
        let light = ctx
            .jobs
            .enqueue("t", Lane::Light, gated(gate_l, 9))
            .unwrap();
        open_l.send(()).unwrap();
        wait_for(&ctx, &light.id, JobState::is_terminal);

        // Cancelling a queued job takes it out without running it.
        assert_eq!(ctx.jobs.cancel(&c.id).unwrap().state, JobState::Cancelled);
        open_a.send(()).unwrap();
        let a = wait_for(&ctx, &a.id, JobState::is_terminal);
        assert_eq!(
            a.state,
            JobState::Done {
                output: serde_json::json!(1)
            }
        );
        assert_eq!(a.last_event, Some(serde_json::json!(1)));
        wait_for(&ctx, &b.id, |s| *s == JobState::Running);
        open_b.send(()).unwrap();
        wait_for(&ctx, &b.id, JobState::is_terminal);
        assert_eq!(ctx.jobs.status(&c.id).unwrap().state, JobState::Cancelled);
        assert_eq!(
            ctx.jobs.status("nope").unwrap_err().kind,
            ErrorKind::JobNotFound
        );
    }

    #[test]
    fn a_running_job_stops_at_its_token_and_failures_keep_their_error() {
        let ctx = ctx(JobLimits::default());
        let (started, running) = mpsc::channel();
        let job = ctx
            .jobs
            .enqueue(
                "t",
                Lane::Heavy,
                Box::new(move |_, sink| {
                    let cx = JobCx::<()>::new(sink);
                    let _ = started.send(());
                    while !cx.token().is_cancelled() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(StudioError::cancelled())
                }),
            )
            .unwrap();
        running.recv().unwrap();
        assert_eq!(ctx.jobs.cancel(&job.id).unwrap().state, JobState::Running);
        let job = wait_for(&ctx, &job.id, JobState::is_terminal);
        assert_eq!(job.state, JobState::Cancelled);

        let failing = ctx
            .jobs
            .enqueue(
                "t",
                Lane::Heavy,
                Box::new(|_, _| Err(StudioError::invalid("no"))),
            )
            .unwrap();
        let failing = wait_for(&ctx, &failing.id, JobState::is_terminal);
        assert_eq!(
            failing.state,
            JobState::Failed {
                error: StudioError::invalid("no")
            }
        );
        let panicking = ctx
            .jobs
            .enqueue("t", Lane::Heavy, Box::new(|_, _| panic!("boom")))
            .unwrap();
        let panicking = wait_for(&ctx, &panicking.id, JobState::is_terminal);
        assert!(matches!(
            panicking.state,
            JobState::Failed { error } if error.kind == ErrorKind::Internal
        ));
    }

    #[test]
    fn events_follow_a_job_and_a_resuming_subscriber_catches_up() {
        let ctx = ctx(JobLimits::default());
        let (open, gate) = mpsc::channel();
        let job = ctx.jobs.enqueue("t", Lane::Heavy, gated(gate, 5)).unwrap();
        wait_for(&ctx, &job.id, |s| *s == JobState::Running);
        // A fresh subscriber hears of the running job at once.
        let fresh = ctx.events.subscribe(None, || ctx.jobs.live_events());
        let first = fresh.events.recv().unwrap();
        let EventBody::Job(event) = &first.body;
        assert_eq!(
            (event.id.as_str(), &event.state),
            (job.id.as_str(), &JobState::Running)
        );
        open.send(()).unwrap();
        wait_for(&ctx, &job.id, JobState::is_terminal);
        let rest: Vec<_> = fresh.events.try_iter().collect();
        let states: Vec<_> = rest
            .iter()
            .map(|e| {
                let EventBody::Job(j) = &e.body;
                (j.state.clone(), j.event.clone())
            })
            .collect();
        assert_eq!(
            states,
            [
                (JobState::Running, Some(serde_json::json!(5))),
                (
                    JobState::Done {
                        output: serde_json::json!(5)
                    },
                    None
                ),
            ]
        );
        // Resuming after the first live event replays the same two.
        let resumed = ctx
            .events
            .subscribe(Some(rest[0].seq - 1), || ctx.jobs.live_events());
        assert_eq!(resumed.events.try_iter().collect::<Vec<_>>(), rest);
    }

    #[test]
    fn the_registry_keeps_only_the_newest_terminal_jobs() {
        let ctx = ctx(JobLimits {
            keep_terminal: 2,
            ..JobLimits::default()
        });
        let mut ids = Vec::new();
        for i in 0..4 {
            let job = ctx
                .jobs
                .enqueue(
                    "t",
                    Lane::Light,
                    Box::new(move |_, _| Ok(serde_json::json!(i))),
                )
                .unwrap();
            wait_for(&ctx, &job.id, JobState::is_terminal);
            ids.push(job.id);
        }
        let kept: Vec<_> = ctx.jobs.list().into_iter().map(|j| j.id).collect();
        assert_eq!(kept, ids[2..]);
        assert_eq!(
            ctx.jobs.status(&ids[0]).unwrap_err().kind,
            ErrorKind::JobNotFound
        );

        // A terminal job older than `keep_for` goes too.
        let ctx = super::tests::ctx(JobLimits {
            keep_for: Duration::ZERO,
            ..JobLimits::default()
        });
        let job = ctx
            .jobs
            .enqueue(
                "t",
                Lane::Light,
                Box::new(|_, _| Ok(serde_json::Value::Null)),
            )
            .unwrap();
        for _ in 0..500 {
            if ctx.jobs.status(&job.id).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(ctx.jobs.list().is_empty());
    }
}
