use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "calib-cli")]
#[command(about = "Rust calibration CLI (WIP)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Minimal homography demo like calib_example_homography
    Homography {},
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Homography {} => run_homography_demo(),
    }
}

fn run_homography_demo() {
    use calib_core::Pt2;
    use calib_linear::dlt_homography;

    let world = vec![
        Pt2::new(0.0, 0.0),
        Pt2::new(100.0, 0.0),
        Pt2::new(100.0, 50.0),
        Pt2::new(0.0, 50.0),
    ];
    let image = vec![
        Pt2::new(10.0, 20.0),
        Pt2::new(110.0, 18.0),
        Pt2::new(120.0, 70.0),
        Pt2::new(8.0, 72.0),
    ];

    let h = dlt_homography(&world, &image).expect("homography");
    println!("Estimated H:\n{}", h);
}
