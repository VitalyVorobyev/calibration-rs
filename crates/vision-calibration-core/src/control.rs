//! Progress reports and cooperative cancellation of a long-running call.
//!
//! A [`RunControl`] travels with a calibration: through a solve's options,
//! a session, or the dataset runner. It carries an optional
//! [`CancelToken`], which the call polls at every Levenberg–Marquardt
//! iteration, between pipeline steps and per image read, and an
//! optional observer, which it calls with a [`Progress`] at the same points.
//!
//! Progress is a side channel: what the observer sees never enters a
//! result, and a run with an observer returns exactly what it returns
//! without one. A cancelled call returns its crate's `Cancelled` error,
//! never a partial result.
//!
//! ```
//! use std::sync::{Arc, Mutex};
//! use vision_calibration_core::{CancelToken, Progress, RunControl};
//!
//! let token = CancelToken::new();
//! let seen = Arc::new(Mutex::new(Vec::new()));
//! let log = Arc::clone(&seen);
//! let control = RunControl::new()
//!     .with_cancel(token.clone())
//!     .with_observer(move |progress| log.lock().unwrap().push(progress));
//!
//! control.report(Progress::step("init"));
//! assert!(control.check().is_ok());
//! token.cancel(); // from any thread: a Ctrl-C handler, a UI button
//! assert!(control.check().is_err());
//! assert_eq!(seen.lock().unwrap().len(), 1);
//! ```

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

/// A shared cancellation flag.
///
/// Clones share one flag: cancel any clone, from any thread, and every call
/// polling the token stops at its next check. Cancellation cannot be
/// undone; a new run takes a new token.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

/// A call stopped because its [`CancelToken`] was cancelled.
///
/// Each crate's error type has a `Cancelled` variant this converts into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

/// What a long-running call reports to its observer.
///
/// In JSON the variant is the `kind` field:
/// `{"kind": "iteration", "iter": 3, "max_iters": 100, "cost": 12.5}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Progress {
    /// A target image was detected, or read from the detection cache.
    #[non_exhaustive]
    Detection {
        /// Target images done so far, `1..=total`.
        done: usize,
        /// Target images to detect, over every camera.
        total: usize,
    },
    /// A laser image was extracted, read from the cache, or skipped because
    /// its view was left out.
    #[non_exhaustive]
    LaserExtraction {
        /// Laser images done so far, `1..=total`.
        done: usize,
        /// Laser images to extract, over every camera.
        total: usize,
    },
    /// A step of a workflow's `run_calibration` begins: the step function's
    /// name without its `step_` prefix (`"intrinsics_init_all"`).
    #[non_exhaustive]
    Step {
        /// The step's name.
        name: &'static str,
    },
    /// A Levenberg–Marquardt iteration finished. `iter` counts from 1 within
    /// one solve and starts again at the next.
    #[non_exhaustive]
    Iteration {
        /// The iteration, `1..=max_iters`.
        iter: usize,
        /// The solve's iteration budget.
        max_iters: usize,
        /// The objective after the iteration, `½ Σᵢ ρᵢ(‖rᵢ‖²)` (the
        /// convention of a solve report's `final_cost`).
        cost: f64,
    },
}

impl Progress {
    /// A [`Progress::Detection`].
    pub fn detection(done: usize, total: usize) -> Self {
        Self::Detection { done, total }
    }

    /// A [`Progress::LaserExtraction`].
    pub fn laser_extraction(done: usize, total: usize) -> Self {
        Self::LaserExtraction { done, total }
    }

    /// A [`Progress::Step`].
    pub fn step(name: &'static str) -> Self {
        Self::Step { name }
    }

    /// A [`Progress::Iteration`].
    pub fn iteration(iter: usize, max_iters: usize, cost: f64) -> Self {
        Self::Iteration {
            iter,
            max_iters,
            cost,
        }
    }
}

type Observer = dyn Fn(Progress) + Send + Sync;

/// How a long-running call reports progress and learns it should stop.
///
/// The default has neither a token nor an observer: the call runs to the
/// end and reports nothing, at the cost of one branch per check. Clones
/// share the token and the observers.
///
/// The observer runs on the calling thread, synchronously, at every
/// report: keep it cheap (forward the report, update a counter). It cannot
/// change the result; to stop the call, cancel the token.
#[derive(Clone, Default)]
pub struct RunControl {
    cancel: Option<CancelToken>,
    observer: Option<Arc<Observer>>,
}

impl RunControl {
    /// No token, no observer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Poll `token`, replacing any token set before.
    #[must_use]
    pub fn with_cancel(mut self, token: CancelToken) -> Self {
        self.cancel = Some(token);
        self
    }

    /// Add an observer. Observers set before still see every report, ahead
    /// of this one.
    #[must_use]
    pub fn with_observer(mut self, observer: impl Fn(Progress) + Send + Sync + 'static) -> Self {
        self.observer = Some(match self.observer.take() {
            None => Arc::new(observer),
            Some(earlier) => Arc::new(move |progress| {
                earlier(progress);
                observer(progress);
            }),
        });
        self
    }

    /// The token this control polls, if any.
    pub fn cancel_token(&self) -> Option<&CancelToken> {
        self.cancel.as_ref()
    }

    /// Whether the token was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
    }

    /// `Err(Cancelled)` once the token is cancelled.
    ///
    /// # Errors
    ///
    /// [`Cancelled`] when the token was cancelled.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }

    /// Hand `progress` to the observers.
    pub fn report(&self, progress: Progress) {
        if let Some(observer) = &self.observer {
            observer(progress);
        }
    }
}

impl fmt::Debug for RunControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunControl")
            .field("cancel", &self.cancel)
            .field("observer", &self.observer.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn default_control_never_cancels_and_reports_nowhere() {
        let control = RunControl::default();
        control.report(Progress::step("init"));
        assert!(control.check().is_ok());
        assert!(control.cancel_token().is_none());
    }

    #[test]
    fn clones_share_the_token() {
        let token = CancelToken::new();
        let control = RunControl::new().with_cancel(token.clone());
        let copy = control.clone();
        assert!(!copy.is_cancelled());
        token.clone().cancel();
        assert_eq!(control.check(), Err(Cancelled));
        assert!(copy.is_cancelled());
    }

    #[test]
    fn observers_compose_in_the_order_added() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (a, b) = (Arc::clone(&seen), Arc::clone(&seen));
        let control = RunControl::new()
            .with_observer(move |_| a.lock().unwrap().push("first"))
            .with_observer(move |_| b.lock().unwrap().push("second"));
        control.report(Progress::iteration(1, 10, 0.5));
        assert_eq!(*seen.lock().unwrap(), ["first", "second"]);
    }

    #[test]
    fn progress_serializes_with_a_kind_tag() {
        let json = serde_json::to_value(Progress::detection(3, 9)).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"kind": "detection", "done": 3, "total": 9})
        );
        let json = serde_json::to_value(Progress::step("rig_init")).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"kind": "step", "name": "rig_init"})
        );
    }

    #[test]
    fn run_control_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RunControl>();
        assert_send_sync::<CancelToken>();
    }
}
