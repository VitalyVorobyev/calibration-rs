//! The `vcal` binary; the commands are in the library.

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use vision_calibration::control::{CancelToken, RunControl};

fn main() -> ExitCode {
    let token = CancelToken::new();
    let handler_token = token.clone();
    let interrupted = AtomicBool::new(false);
    // The first Ctrl-C stops the run at its next check; a second one, for a
    // run stuck between checks, exits at once. Without a handler (another
    // one is installed) Ctrl-C keeps its default: the process ends.
    let _ = ctrlc::set_handler(move || {
        if interrupted.swap(true, Ordering::SeqCst) {
            std::process::exit(i32::from(vision_calibration_cli::INTERRUPTED));
        }
        handler_token.cancel();
    });
    let control = RunControl::new().with_cancel(token);
    ExitCode::from(vision_calibration_cli::run(std::env::args_os(), &control))
}
