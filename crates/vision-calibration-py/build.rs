//! Build script for `vision-calibration-py`: configures PyO3 extension-module link args.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
