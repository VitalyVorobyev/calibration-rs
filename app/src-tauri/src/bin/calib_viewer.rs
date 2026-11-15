//! `calib-viewer <run-dir | export.json> [--port N] [--no-open] [--dist DIR]`:
//! the calibration report viewer in a browser. See
//! `calibration_diagnose_lib::viewer`.

fn main() -> std::process::ExitCode {
    calibration_diagnose_lib::viewer::main(std::env::args_os().skip(1).collect())
}
