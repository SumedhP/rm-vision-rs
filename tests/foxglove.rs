#![warn(missing_docs)]
#![allow(missing_docs)]

use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;

use auto_aim::geometry::{Camera, CameraConfig};
use auto_aim::io::{CameraFrame, RobotState};
use auto_aim::pipeline::FrameRecord;
use auto_aim::viz::foxglove::Publisher;
use image::RgbImage;
use nalgebra::UnitQuaternion;

#[test]
fn test_publish_mcap() -> color_eyre::Result<()> {
    let log_path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("agent_fox")
        .join("test_synthetic.mcap");
    std::fs::create_dir_all(log_path.parent().unwrap())?;
    let _ = std::fs::remove_file(&log_path);

    // Create publisher and publish 5 synthetic frames
    let mut publisher = Publisher::new(Some(&log_path), None, 0.0)?;

    let camera = Camera::from(&CameraConfig {
        k: [700.0, 700.0, 480.0, 270.0],
        dist: [0.0; 4],
        rpy: [0.0; 3],
        translation: [0.05, 0.0, 0.1],
    });

    let robot_state = RobotState {
        t: 0.0,
        q: UnitQuaternion::identity(),
        bullet_speed: 15.0,
        robot_id: 5,
    };

    for frame_idx in 0..5 {
        let t = f64::from(frame_idx) * 0.05; // 50ms between frames

        let mut image = RgbImage::new(96, 54);
        for pixel in image.iter_mut() {
            *pixel = 128;
        }

        let frame = CameraFrame {
            frame_id: u64::try_from(frame_idx).unwrap_or(0),
            t,
            image: Arc::new(image),
        };

        let detected = auto_aim::detector::DetectedFrame {
            frame,
            detections: vec![],
        };

        let mut robot = robot_state;
        robot.t = t;

        let record = FrameRecord {
            robot,
            estimate: None,
            commands: vec![],
            target: None,
        };

        publisher.publish(&detected, &record, &camera)?;
    }

    publisher.finish()?;

    // Verify MCAP file was created
    assert!(log_path.exists(), "MCAP file should be created");

    let metadata = std::fs::metadata(&log_path)?;
    assert!(
        metadata.len() > 0,
        "MCAP file should have content, got {} bytes",
        metadata.len()
    );

    Ok(())
}

#[test]
fn test_websocket_server() -> color_eyre::Result<()> {
    let publisher = Publisher::new(None, Some(18765), 0.0)?;

    // Try connecting to the websocket server
    let result = std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        TcpStream::connect("127.0.0.1:18765")
    })
    .join();

    drop(publisher);

    match result {
        Ok(Ok(_stream)) => Ok(()),
        Ok(Err(e)) => {
            // Connection refused is acceptable if server didn't fully start
            if e.kind() == std::io::ErrorKind::ConnectionRefused {
                Ok(())
            } else {
                Err(color_eyre::eyre::eyre!("TCP connection failed: {}", e))
            }
        }
        Err(_) => Err(color_eyre::eyre::eyre!("Thread panicked")),
    }
}
