//! Talks to the MCU over a real port: prints every frame received, decoding the ones we know, and
//! optionally sends a hold-still gimbal command at a fixed rate.
//!
//! `cargo run --release --example serial_monitor -- --port /dev/ttyACM0 --baud 921600 --send-hz 100`
//! (`COM3` etc. on Windows). Stop with Ctrl-C.

use std::time::{Duration, Instant};

use auto_aim::io::serial::{GimbalCommand, GimbalState, SerialLink};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    port: String,
    #[arg(long, default_value_t = 921_600)]
    baud: u32,
    /// Send a zero `GimbalCommand` (no fire) at this rate; 0 to only listen.
    #[arg(long, default_value_t = 0.0)]
    send_hz: f64,
}

#[expect(clippy::print_stdout, reason = "the monitor's output is the point")]
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let mut link = SerialLink::open(&args.port, args.baud)?;
    println!("listening on {} at {} baud", args.port, args.baud);
    let start = Instant::now();
    let period = (args.send_hz > 0.0).then(|| Duration::from_secs_f64(1.0 / args.send_hz));
    let mut next_send = Instant::now();
    let mut last_report = Instant::now();
    let mut count = 0u64;
    loop {
        if let Some(period) = period
            && Instant::now() >= next_send
        {
            link.send(&GimbalCommand { yaw: 0.0, pitch: 0.0, yaw_vel: 0.0, pitch_vel: 0.0, yaw_acc: 0.0, pitch_acc: 0.0, fire: false })?;
            next_send += period;
        }
        if let Ok(r) = link.incoming().recv_timeout(Duration::from_millis(1)) {
            count += 1;
            let t = r.at.duration_since(start).as_secs_f64();
            match r.frame.parse::<GimbalState>() {
                Some(s) => println!("{t:9.4} seq {:3} GimbalState q {:?} bullet {:.2} m/s id {}", r.frame.seq, s.q, s.bullet_speed, s.robot_id),
                None => println!("{t:9.4} seq {:3} cmd 0x{:04X} {} bytes {:02X?}", r.frame.seq, r.frame.cmd_id, r.frame.payload.len(), r.frame.payload),
            }
        }
        if last_report.elapsed() >= Duration::from_secs(1) {
            println!("-- {count} frames/s, decoder {:?}", link.stats());
            count = 0;
            last_report = Instant::now();
        }
    }
}
