use std::{error::Error, fs, path::Path};

use calib_pipeline::{run_planar_intrinsics, PlanarIntrinsicsConfig, PlanarIntrinsicsInput};
use clap::Parser;

/// Calibration CLI for planar camera intrinsics.
#[derive(Debug, Parser)]
#[command(author, version, about = "Planar intrinsics calibration pipeline")]
struct Args {
    /// Path to JSON file containing PlanarIntrinsicsInput.
    #[arg(long)]
    input: String,

    /// Optional path to JSON PlanarIntrinsicsConfig. Defaults are used if omitted.
    #[arg(long)]
    config: Option<String>,
}

fn load_json_file<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn Error>> {
    let data = fs::read_to_string(path)?;
    let value = serde_json::from_str(&data)?;
    Ok(value)
}

fn write_report_json(
    report: &calib_pipeline::PlanarIntrinsicsReport,
) -> Result<String, Box<dyn Error>> {
    Ok(serde_json::to_string_pretty(report)?)
}

fn run_planar_intrinsics_from_files(
    input_path: &str,
    config_path: Option<&str>,
) -> Result<String, Box<dyn Error>> {
    let input: PlanarIntrinsicsInput = load_json_file(Path::new(input_path))?;

    let config = if let Some(cfg_path) = config_path {
        load_json_file::<PlanarIntrinsicsConfig>(Path::new(cfg_path))?
    } else {
        PlanarIntrinsicsConfig::default()
    };

    let report = run_planar_intrinsics(&input, &config)?;
    write_report_json(&report)
}

fn main() {
    if let Err(err) = try_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let json = run_planar_intrinsics_from_files(&args.input, args.config.as_deref())?;
    println!("{}", json);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{
        synthetic::planar, BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole,
    };
    use calib_pipeline::{PlanarIntrinsicsConfig, PlanarIntrinsicsInput};
    use std::{fs, path::Path};
    use tempfile::NamedTempFile;

    fn write_json<T: serde::Serialize>(value: &T, path: &Path) {
        serde_json::to_writer_pretty(fs::File::create(path).unwrap(), value).unwrap();
    }

    fn synthetic_input() -> (PlanarIntrinsicsInput, PlanarIntrinsicsConfig) {
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);

        let board_points = planar::grid_points(5, 4, 0.05);
        let poses = planar::poses_yaw_y_z(3, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).expect("projection");

        let input = PlanarIntrinsicsInput { views };
        let mut config = PlanarIntrinsicsConfig::default();
        config.backend_opts.max_iters = 200;
        (input, config)
    }

    #[test]
    fn helper_smoke_test() {
        let (input, config) = synthetic_input();
        let input_file = NamedTempFile::new().unwrap();
        let config_file = NamedTempFile::new().unwrap();

        write_json(&input, input_file.path());
        write_json(&config, config_file.path());

        let json = run_planar_intrinsics_from_files(
            input_file.path().to_str().unwrap(),
            Some(config_file.path().to_str().unwrap()),
        )
        .expect("cli helper should succeed");

        let report: calib_pipeline::PlanarIntrinsicsReport = serde_json::from_str(&json).unwrap();
        assert!(
            report.final_cost < 1e-6,
            "final cost too high: {}",
            report.final_cost
        );
    }
}
