//! Runs the pipeline on a video file and writes one JSON line per frame.

use std::io::Write;
use std::path::PathBuf;

use auto_aim::config::{GeneralConfig, RobotConfig, load};
use auto_aim::io::video::VideoSource;
use auto_aim::io::{CameraSource, RobotSource};
use auto_aim::pipeline::{self, RunOptions};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    /// Input video.
    #[arg(long)]
    video: PathBuf,
    /// Per-frame JSON lines output.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "configs/general.yaml")]
    general: PathBuf,
    #[arg(long, default_value = "configs/robots/realsense_test.yaml")]
    robot: PathBuf,
    /// Aimer tick rate, Hz.
    #[arg(long, default_value_t = 200.0)]
    control_rate: f64,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let args = Args::parse();
    let general: GeneralConfig = load(&args.general)?;
    let robot: RobotConfig = load(&args.robot)?;
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    let source = CameraSource::Video(VideoSource::open(&args.video)?);
    let options = RunOptions { control_rate: args.control_rate };
    pipeline::run(&general, &robot, source, &RobotSource::Defaults(robot.default_state), options, |frame, record| {
        let line = serde_json::json!({
            "frame_id": frame.frame.frame_id,
            "t": frame.frame.t,
            "detections": frame.detections,
            "record": record,
        });
        writeln!(out, "{line}")?;
        Ok(())
    })?;
    out.flush()?;
    Ok(())
}
