//! The MCAP log round-trips through replay, and the live websocket accepts connections.

use std::sync::Arc;

use auto_aim::detector::DetectedFrame;
use auto_aim::geometry::{Camera, CameraConfig};
use auto_aim::io::{CameraFrame, RobotState, mcap_log};
use auto_aim::pipeline::FrameRecord;
use auto_aim::viz::foxglove::Publisher;
use image::{Rgb, RgbImage};
use nalgebra::UnitQuaternion;

fn camera() -> Camera {
    Camera::from(&CameraConfig { k: [700.0, 700.0, 48.0, 27.0], dist: [0.0; 4], rpy: [0.0; 3], translation: [0.0; 3] })
}

fn publish(p: &mut Publisher, i: u32, robot: RobotState) {
    let t = f64::from(i) / 60.0;
    let image = RgbImage::from_pixel(96, 54, Rgb([200, 40, u8::try_from(i * 40).expect("small")]));
    let frame = DetectedFrame { frame: CameraFrame { frame_id: u64::from(i), t, image: Arc::new(image) }, detections: vec![] };
    let record = FrameRecord { robot: RobotState { t, ..robot }, estimate: None, commands: vec![], target: None };
    p.publish(&frame, &record, &camera()).expect("publish");
}

#[test]
fn log_replays_frames_and_robot_states() {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("roundtrip.mcap");
    let robot = RobotState { t: 0.0, q: UnitQuaternion::from_euler_angles(0.0, 0.1, 0.4), bullet_speed: 23.5, robot_id: 103 };
    let mut p = Publisher::new(Some(&path), None, 0.0).expect("open log");
    for i in 0..5 {
        publish(&mut p, i, robot);
    }
    p.finish().expect("close log");

    let (mut frames, states) = mcap_log::open(&path).expect("reopen log");
    for i in 0..5u32 {
        let f = frames.next_frame().expect("decodes").expect("five frames");
        assert!((f.t - f64::from(i) / 60.0).abs() < 1e-6, "frame {i} keeps its data time");
        assert_eq!(f.image.dimensions(), (96, 54), "frame {i} keeps its size");
        let blue = i32::from(f.image.get_pixel(48, 27)[2]);
        assert!((blue - i32::try_from(i * 40).expect("small")).abs() < 8, "frame {i} pixels survive JPEG");
    }
    assert!(frames.next_frame().expect("decodes").is_none(), "exactly five frames");
    assert_eq!(states.len(), 5, "one robot state per frame");
    assert!(states[0].q.angle_to(&robot.q) < 1e-9 && (states[0].bullet_speed - 23.5).abs() < 1e-12, "robot state round-trips");
}

#[test]
fn live_websocket_accepts_a_client() {
    let p = Publisher::new(None, Some(18765), 0.0).expect("server starts");
    assert!(std::net::TcpStream::connect("127.0.0.1:18765").is_ok(), "port is listening");
    p.finish().expect("server stops");
}
