//! `calibration-studio-server`: see the crate documentation and `--help`.

use std::process::ExitCode;

use clap::Parser;
use studio_server::config::{Cli, Command, Config, default_state_dir};

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(Command::Token { rotate, state_dir }) = cli.command {
        let state_dir = state_dir.unwrap_or_else(default_state_dir);
        return match studio_server::token::stored(&state_dir, rotate) {
            Ok(token) => {
                println!("#token={token}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("calibration-studio-server: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let config = match Config::from_args(cli.serve) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("calibration-studio-server: {e}");
            return ExitCode::from(2);
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("calibration-studio-server: cannot start the runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(studio_server::serve(config)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("calibration-studio-server: {e}");
            ExitCode::FAILURE
        }
    }
}
