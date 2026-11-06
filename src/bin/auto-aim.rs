//! Runs the pipeline on a video file or a recorded MCAP log.
//!
//! Outputs, each optional: per-frame JSON lines (for analysis), an MCAP log (for replay and
//! Foxglove), a live Foxglove websocket, and an overlay video of the estimate drawn on the footage.

use std::io::Write;
use std::path::PathBuf;

use auto_aim::config::{GeneralConfig, LoggingConfig, RobotConfig, load};
use auto_aim::geometry::Camera;
use auto_aim::io::video::VideoSource;
use auto_aim::io::{CameraSource, RobotSource, mcap_log};
use auto_aim::pipeline::{self, RunOptions};
use auto_aim::viz::foxglove::Publisher;
use auto_aim::viz::overlay::{OverlayWriter, render};
use clap::Parser;
use color_eyre::eyre::bail;

#[derive(Debug, Parser)]
struct Args {
    /// Input video (no robot state; config defaults are used).
    #[arg(long, conflicts_with = "replay")]
    video: Option<PathBuf>,
    /// Input MCAP log recorded by this program (frames + robot state).
    #[arg(long)]
    replay: Option<PathBuf>,
    /// Per-frame JSON lines output.
    #[arg(long)]
    jsonl: Option<PathBuf>,
    /// MCAP log output.
    #[arg(long)]
    mcap: Option<PathBuf>,
    /// Serve a live Foxglove websocket on this port.
    #[arg(long)]
    ws: Option<u16>,
    /// Overlay video output (.mp4).
    #[arg(long)]
    overlay: Option<PathBuf>,
    #[arg(long, default_value = "configs/general.yaml")]
    general: PathBuf,
    #[arg(long, default_value = "configs/robots/realsense_test.yaml")]
    robot: PathBuf,
    #[arg(long, default_value = "configs/logging.yaml")]
    logging: PathBuf,
    /// Aimer tick rate, Hz.
    #[arg(long, default_value_t = 200.0)]
    control_rate: f64,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let args = Args::parse();
    let general: GeneralConfig = load(&args.general)?;
    let robot: RobotConfig = load(&args.robot)?;
    let logging: LoggingConfig = load(&args.logging)?;
    let camera = Camera::from(&robot.camera);

    let (source, robot_source, fps) = match (&args.video, &args.replay) {
        (Some(video), None) => {
            let v = VideoSource::open(video)?;
            let fps = v.fps();
            (CameraSource::Video(v), RobotSource::Defaults(robot.default_state), fps)
        }
        (None, Some(log)) => {
            let (frames, states) = mcap_log::open(log)?;
            let robot_source = if states.is_empty() { RobotSource::Defaults(robot.default_state) } else { RobotSource::Log(states) };
            (CameraSource::Log(Box::new(frames)), robot_source, 60.0)
        }
        _ => bail!("pass exactly one of --video or --replay"),
    };

    for path in [&args.jsonl, &args.mcap, &args.overlay].into_iter().flatten() {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let mut jsonl = args.jsonl.as_ref().map(std::fs::File::create).transpose()?.map(std::io::BufWriter::new);
    let mut publisher = if args.mcap.is_some() || args.ws.is_some() {
        Some(Publisher::new(args.mcap.as_deref(), args.ws, logging.frame_rate)?)
    } else {
        None
    };
    let mut overlay: Option<OverlayWriter> = None;

    pipeline::run(&general, &robot, source, &robot_source, RunOptions { control_rate: args.control_rate }, |frame, record| {
        if let Some(out) = jsonl.as_mut() {
            let line = serde_json::json!({ "frame_id": frame.frame.frame_id, "t": frame.frame.t, "detections": frame.detections, "record": record });
            writeln!(out, "{line}")?;
        }
        if let Some(p) = publisher.as_mut() {
            p.publish(frame, record, &camera)?;
        }
        if let Some(path) = &args.overlay {
            if overlay.is_none() {
                overlay = Some(OverlayWriter::create(path, frame.frame.image.width(), frame.frame.image.height(), fps)?);
            }
            if let Some(w) = overlay.as_mut() {
                w.write(&render(frame, record, &camera))?;
            }
        }
        Ok(())
    })?;

    if let Some(mut out) = jsonl {
        out.flush()?;
    }
    if let Some(p) = publisher {
        p.finish()?;
    }
    if let Some(w) = overlay {
        w.finish()?;
    }
    Ok(())
}
