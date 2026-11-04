# ADR 0029: Progress and Cancellation

## Context

A calibration runs for seconds to minutes: detection reads every image, and
a rig or hand-eye workflow solves several Levenberg–Marquardt problems in a
row. The app could report three coarse stages and not stop a running
solve; the CLI could not be interrupted cleanly; a Python call, which runs
with the GIL released, ignored Ctrl-C until it returned. Progress and
cancellation have to reach the innermost loop (the LM iteration, the image
loop) through every layer above it, without changing a result (AGENTS.md
§5: deterministic, no wall-clock values in results).

## Decision

### One control value, in core

`vision_calibration_core::control` (on the facade as
`vision_calibration::control`) holds:

- `CancelToken`: a shared flag (`Arc<AtomicBool>`); clones share it,
  `cancel()` is sticky and callable from any thread.
- `Progress`: what a call reports, a `#[non_exhaustive]` enum with
  `#[non_exhaustive]` variants — `Detection { done, total }`,
  `LaserExtraction { done, total }`, `Step { name }`,
  `Iteration { iter, max_iters, cost }`. It is `Copy`, serializes with a
  `kind` tag, and has a published schema (`schemas/progress.json`), so the
  app and Python receive it as data.
- `RunControl`: an optional token plus observers
  (`Arc<dyn Fn(Progress) + Send + Sync>`, composed in the order added). The
  default has neither, so every check is one branch.

Core is the lowest crate both optim and the pipeline see, and the dataset
runner and the solver report through the same type.

### Explicit, never ambient

The control travels as a value: `BackendSolveOptions::control` (not
serialized) into the shared LM loop; `SolveOptions::control` for
`optim::problem::solve`; `CalibrationSession::set_control` for a session,
which every step's solver options take through
`SolverConfig::backend_options(control)`, so a step cannot forget it; a
last `control: &RunControl` argument of `calibrate_dataset` and of every
`build_*_input` converter. Nested sessions (the joint laser workflow's
warm starts) get the parent's control.

### Where it is checked, what it reports

- The LM loop checks the token before every outer iteration and once more
  before returning a solution, and reports each finished iteration (`iter`
  from 1 within a solve, `cost` = ½Σρ as in `SolveReport`). The check is one
  atomic load; nothing per residual.
- `run_calibration` reports each step it begins and checks the token before
  and after the report.
- The converters check before each image and report after it; a laser image
  of a dropped view counts as done, so `done` reaches `total`.

### Cancellation is an error, never a result

A cancelled call returns its crate's `Cancelled` variant (`optim::Error`,
`pipeline::Error`, `RunError`, `CalibrateDatasetError`); the pipeline lifts
an optim cancellation to its own variant, so one match answers "was this
cancelled?". A cancelled step stores nothing. Code that swallows a failed
solve (multistarts, staged fallbacks) checks the token before returning
what it kept, so a partial multistart is never returned as its result.

### Progress is a side channel

The observer cannot change the solve: an uncancelled run with observers is
bit-identical to one without (tested on both backends and end to end).
Reports carry no time; a consumer that throttles (the app, the CLI line)
reads the clock itself, for display only.

### Front ends

- **CLI**: the binary's Ctrl-C handler (`ctrlc`) cancels the token; a second
  Ctrl-C exits at once. `vcal run` exits with 130 and writes nothing; it
  draws one status line on standard error when it is a terminal.
- **Python**: the binding installs an observer that takes the GIL, calls the
  user's `progress` callable and `Python::check_signals()`. A pending
  `KeyboardInterrupt`, or an exception the callable raised, is kept and the
  token cancelled; the call then raises it. Ctrl-C works with or without a
  callable; Python has no token type — raising is how Python code stops a
  run.
- **App**: the run registry holds a token per run; reports go to the
  frontend over the existing channel (iterations throttled to 50 ms).

## Consequences

- A cancel takes effect within one LM iteration (zero further iterations
  once the token is seen) or one image. Work between checks — a linear
  initializer, the covariance after a solve — runs to its end.
- `calibrate_dataset`, the converters and `vcal`'s `run` gained an argument,
  and `SolveOptions` lost `Copy`/`Eq`: a breaking release.
- A Python progress callable runs on the calibrating thread under the GIL
  at every report; a slow callable slows the run.

## Alternatives rejected

- **A thread-local "current control"** installed around a call. It would
  have touched no signatures, but it is hidden coupling: a library call's
  behaviour would depend on ambient state, it would not cross a thread, and
  forgetting to install it is silent.
- **A control field on every step's options.** Options are serialized
  configuration; a cancellation handle is not, and the session already
  carries what a run shares.
- **An observer trait with one method per event.** The enum is one
  extension point (`#[non_exhaustive]`), serializes for the app and Python
  as is, and a closure is enough for every front end.
- **Python signal handling by a Python-side handler.** A Python signal
  handler runs only when the main thread executes bytecode, which it does
  not while the GIL is released; polling `check_signals` from the observer
  does.
- **Returning the last iterate on cancel.** A partial solve has no valid
  covariance and would be indistinguishable from a converged one downstream.
