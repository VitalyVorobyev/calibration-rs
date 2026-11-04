//! Progress callbacks and Ctrl-C for the long calls.
//!
//! A long call runs with the GIL released, so CPython's own Ctrl-C handling
//! (a flag its signal handler sets, turned into `KeyboardInterrupt` the next
//! time the interpreter checks) never runs until the call returns. A
//! [`PyRunControl`] closes that gap: its observer, which the Rust side calls
//! at every report (each image read, each step, each solver iteration),
//! briefly takes the GIL, hands the report to the caller's `progress`
//! callable if there is one, and checks for signals. A pending
//! `KeyboardInterrupt`, or an exception the callable raised, is kept and the
//! run's token cancelled; when the call returns, that exception is raised in
//! place of its result. A callable stops a run by raising.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use vision_calibration::control::{CancelToken, Progress, RunControl};

use crate::session::to_py;

/// The run control of one Python call.
pub(crate) struct PyRunControl {
    control: RunControl,
    raised: Arc<Mutex<Option<PyErr>>>,
}

impl PyRunControl {
    /// A control reporting to `progress` (when given) and polling Ctrl-C.
    pub(crate) fn new(progress: Option<Py<PyAny>>) -> Self {
        let token = CancelToken::new();
        let raised: Arc<Mutex<Option<PyErr>>> = Arc::new(Mutex::new(None));
        let observer_token = token.clone();
        let observer_raised = Arc::clone(&raised);
        let control =
            RunControl::new()
                .with_cancel(token)
                .with_observer(move |report: Progress| {
                    if observer_token.is_cancelled() {
                        return;
                    }
                    let outcome = Python::attach(|py| -> PyResult<()> {
                        if let Some(callback) = &progress {
                            callback.call1(py, (to_py(py, &report)?,))?;
                        }
                        py.check_signals()
                    });
                    if let Err(err) = outcome {
                        if let Ok(mut slot) = observer_raised.lock() {
                            slot.get_or_insert(err);
                        }
                        observer_token.cancel();
                    }
                });
        Self { control, raised }
    }

    /// The control to hand to the Rust call.
    pub(crate) fn control(&self) -> &RunControl {
        &self.control
    }

    /// The call's outcome as Python sees it: the exception the progress
    /// callable raised, or the `KeyboardInterrupt`, in place of whatever the
    /// call returned; otherwise the result, its error mapped by `err`.
    pub(crate) fn finish<T, E>(
        &self,
        result: Result<T, E>,
        err: impl FnOnce(E) -> PyErr,
    ) -> PyResult<T> {
        let raised = self.raised.lock().ok().and_then(|mut slot| slot.take());
        match raised {
            Some(exception) => Err(exception),
            None => result.map_err(err),
        }
    }
}
