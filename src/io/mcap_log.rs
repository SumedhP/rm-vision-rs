//! Camera source that replays frames from an MCAP log written by this pipeline.

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use image::RgbImage;
use mcap::io::MessageReader;
use nalgebra::UnitQuaternion;

use super::CameraFrame;
use crate::io::RobotState;

/// Replays logged camera frames in timestamp order.
pub struct McapCameraSource {
    reader: MessageReader<BufReader<File>>,
    next_frame_id: u64,
}

impl fmt::Debug for McapCameraSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McapCameraSource")
            .field("next_frame_id", &self.next_frame_id)
            .finish_non_exhaustive()
    }
}

impl McapCameraSource {
    /// Next frame, or `None` at end of log.
    ///
    /// # Errors
    /// Corrupt log or undecodable image.
    pub fn next_frame(&mut self) -> color_eyre::Result<Option<CameraFrame>> {
        for message in &mut self.reader {
            let msg = message?;
            if msg.channel.topic == "/camera/image" {
                // TODO: decode CompressedImage protobuf to extract JPEG data
                // For now, return stub frames
                let t = (msg.log_time as f64) * 1e-9;
                let frame_id = self.next_frame_id;
                self.next_frame_id += 1;

                let img = RgbImage::new(96, 54);
                return Ok(Some(CameraFrame {
                    frame_id,
                    t,
                    image: std::sync::Arc::new(img),
                }));
            }
        }
        Ok(None)
    }
}

/// Opens an MCAP log file, reads it, and returns frames and robot states.
///
/// # Errors
/// File could not be opened, is corrupt, or contains invalid data.
pub fn open(path: &Path) -> color_eyre::Result<(McapCameraSource, Vec<RobotState>)> {
    let file = File::open(path)?;
    let reader = MessageReader::new(BufReader::new(file));
    let mut robot_states = Vec::new();

    // First pass: collect robot states
    let file = File::open(path)?;
    let reader_pass1 = MessageReader::new(BufReader::new(file));
    for message in reader_pass1 {
        let msg = message?;
        if msg.channel.topic == "/robot_state" {
            let state_json: serde_json::Value = serde_json::from_slice(&msg.data)?;
            let state = parse_robot_state(&state_json)?;
            robot_states.push(state);
        }
    }

    let source = McapCameraSource {
        reader,
        next_frame_id: 0,
    };

    Ok((source, robot_states))
}

fn parse_robot_state(json: &serde_json::Value) -> color_eyre::Result<RobotState> {
    let t = json["t"]
        .as_f64()
        .ok_or_else(|| color_eyre::eyre::eyre!("missing or invalid t field"))?;

    let q_arr = json["q"]
        .as_array()
        .ok_or_else(|| color_eyre::eyre::eyre!("missing or invalid q field"))?;

    if q_arr.len() != 4 {
        return Err(color_eyre::eyre::eyre!("q array must have 4 elements"));
    }

    let [w, x, y, z] = [
        q_arr[0].as_f64().ok_or_else(|| color_eyre::eyre::eyre!("invalid q[0]"))?,
        q_arr[1].as_f64().ok_or_else(|| color_eyre::eyre::eyre!("invalid q[1]"))?,
        q_arr[2].as_f64().ok_or_else(|| color_eyre::eyre::eyre!("invalid q[2]"))?,
        q_arr[3].as_f64().ok_or_else(|| color_eyre::eyre::eyre!("invalid q[3]"))?,
    ];

    let bullet_speed = json["bullet_speed"]
        .as_f64()
        .ok_or_else(|| color_eyre::eyre::eyre!("missing or invalid bullet_speed field"))?;

    let robot_id = json["robot_id"]
        .as_u64()
        .ok_or_else(|| color_eyre::eyre::eyre!("missing or invalid robot_id field"))? as u8;

    Ok(RobotState {
        t,
        q: UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z)),
        bullet_speed,
        robot_id,
    })
}
