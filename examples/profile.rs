//! Per-stage latency of the pipeline on one frame.
//!
//! Runs the detector and both tracker variants over the frames before `--frame` so the track is in
//! its steady state, then repeats every stage on that frame and reports the spread.
//!
//! `cargo run --release --example profile -- --video assets/videos/spinning_3m.mp4 --frame 400`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use auto_aim::aimer::Aimer;
use auto_aim::config::{GeneralConfig, RobotConfig, load};
use auto_aim::detector::{ArmorColor, Detector};
use auto_aim::geometry::Camera;
use auto_aim::io::video::VideoSource;
use auto_aim::tracker::measurement::solve_armor;
use auto_aim::tracker::{MeasurementSpace, Tracker};
use clap::Parser;
use color_eyre::eyre::eyre;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "assets/videos/spinning_3m.mp4")]
    video: PathBuf,
    /// Frame to profile; earlier frames warm up the tracker.
    #[arg(long, default_value_t = 400)]
    frame: u64,
    /// Repetitions of each stage.
    #[arg(long, default_value_t = 200)]
    reps: usize,
    #[arg(long, default_value = "configs/general.yaml")]
    general: PathBuf,
    #[arg(long, default_value = "configs/robots/realsense_test.yaml")]
    robot: PathBuf,
}

#[derive(Debug, Default)]
struct Samples(Vec<Duration>);

impl Samples {
    fn time<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let out = f();
        self.0.push(start.elapsed());
        out
    }

    fn ms(&self, percent: usize) -> f64 {
        let mut v: Vec<f64> = self.0.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        v.sort_by(f64::total_cmp);
        v.get(v.len().saturating_sub(1) * percent / 100).copied().unwrap_or(f64::NAN)
    }
}

#[expect(clippy::print_stdout, reason = "the report is the output")]
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let general: GeneralConfig = load(&args.general)?;
    let robot: RobotConfig = load(&args.robot)?;
    let camera = Camera::from(&robot.camera);
    let q = robot.default_state.q;
    let enemy = if robot.default_state.robot_id > 100 { ArmorColor::Red } else { ArmorColor::Blue };

    let mut detector = Detector::new(&general.detector)?;
    let mut pose = Tracker::new(general.tracker.clone(), camera.clone());
    let mut pixel_cfg = general.tracker.clone();
    pixel_cfg.measurement = MeasurementSpace::Pixel;
    let mut pixel = Tracker::new(pixel_cfg, camera.clone());
    let mut aimer = Aimer::new(general.aimer, robot.gimbal.clone());

    let mut video = VideoSource::open(&args.video)?;
    let mut decode = Samples::default();
    let mut estimate = None;
    let frame = loop {
        let frame = decode.time(|| video.next_frame())?.ok_or_else(|| eyre!("video ends before frame {}", args.frame))?;
        if frame.frame_id == args.frame {
            break frame;
        }
        let dets = detector.detect(&frame.image, enemy)?;
        pixel.update(frame.t, &dets, &q);
        aimer.update(frame.t, estimate.as_ref(), robot.default_state.bullet_speed, robot.downlink_delay);
        estimate = pose.update(frame.t, &dets, &q);
    };
    let est = estimate.as_ref().ok_or_else(|| eyre!("no track at frame {}", args.frame))?;
    println!("{} frame {}: {:?} track, {:?} mode, w = {:.1} rad/s, backend {:?}", args.video.display(), args.frame, est.state, est.mode, est.x[7], detector.active_backend());

    let [mut pre, mut inf, mut postprocess, mut refine, mut total]: [Samples; 5] = Default::default();
    let mut dets = Vec::new();
    for _ in 0..args.reps {
        dets = Samples::time(&mut total, || detector.detect(&frame.image, enemy))?;
        let stages = detector.last_times();
        for (s, d) in [(&mut pre, stages.preprocess), (&mut inf, stages.inference), (&mut postprocess, stages.postprocess), (&mut refine, stages.refine)] {
            s.0.push(d);
        }
    }
    let [mut pnp, mut track_pose, mut track_pixel, mut aim]: [Samples; 4] = Default::default();
    for _ in 0..args.reps * 10 {
        Samples::time(&mut pnp, || dets.iter().filter_map(|d| solve_armor(d, &camera, &q)).count());
        Samples::time(&mut track_pose, || pose.clone().update(frame.t, &dets, &q));
        Samples::time(&mut track_pixel, || pixel.clone().update(frame.t, &dets, &q));
        Samples::time(&mut aim, || aimer.clone().update(frame.t, estimate.as_ref(), robot.default_state.bullet_speed, robot.downlink_delay));
    }
    let clone_cost = {
        let mut s = Samples::default();
        for _ in 0..args.reps * 10 {
            s.time(|| (pose.clone(), aimer.clone()));
        }
        s.ms(50)
    };

    println!("{} detection(s), {} repetitions ({}x for the tracker and aimer)\n", dets.len(), args.reps, 10);
    println!("| Stage | median ms | p90 ms |");
    println!("|---|---|---|");
    let row = |name: &str, s: &Samples| println!("| {name} | {:.3} | {:.3} |", s.ms(50), s.ms(90));
    row("Video decode (ffmpeg pipe, per frame)", &decode);
    row("Detector: letterbox + tensor", &pre);
    row("Detector: ONNX Runtime inference", &inf);
    row("Detector: decode + NMS", &postprocess);
    row("Detector: corner refinement", &refine);
    row("Detector total", &total);
    row("Plate pose solve (all detections)", &pnp);
    row("Tracker update, pose measurement (includes pose solve)", &track_pose);
    row("Tracker update, pixel measurement (includes pose solve)", &track_pixel);
    row("Aimer, one 200 Hz tick", &aim);
    println!("\nTracker and aimer rows include cloning their state, {clone_cost:.4} ms.");
    Ok(())
}
