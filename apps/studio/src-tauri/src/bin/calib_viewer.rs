//! `calib-viewer <run-dir | export.json> [--port N] [--no-open] [--dist DIR]`:
//! the calibration report viewer in a browser. See
//! `calibration_studio_lib::viewer`.

fn main() -> std::process::ExitCode {
    calibration_studio_lib::viewer::main(std::env::args_os().skip(1).collect())
}
