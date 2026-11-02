//! The `vcal` binary; the commands are in the library.

use std::process::ExitCode;

fn main() -> ExitCode {
    ExitCode::from(vision_calibration_cli::run(std::env::args_os()))
}
