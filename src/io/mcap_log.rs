//! Replays an MCAP log written by [`crate::viz::foxglove::Publisher`]: camera frames from
//! `/camera/image` and robot states from `/robot_state`, both stamped with data time.

use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use color_eyre::eyre::eyre;
use foxglove::Decode;
use foxglove::messages::CompressedImage;
use mcap::io::MessageReader;
use nalgebra::{Quaternion, UnitQuaternion};
use serde::Deserialize;

use super::{CameraFrame, RobotState};

/// Logged camera frames in log order.
pub struct McapCameraSource {
    reader: MessageReader<BufReader<File>>,
    next_frame_id: u64,
}

impl fmt::Debug for McapCameraSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("McapCameraSource").field("next_frame_id", &self.next_frame_id).finish_non_exhaustive()
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
            if msg.channel.topic != "/camera/image" {
                continue;
            }
            let compressed = CompressedImage::decode(&msg.data[..]).map_err(|e| eyre!("decoding /camera/image: {e}"))?;
            let image = image::load_from_memory(&compressed.data)?.to_rgb8();
            let frame = CameraFrame { frame_id: self.next_frame_id, t: seconds(msg.log_time), image: std::sync::Arc::new(image) };
            self.next_frame_id += 1;
            return Ok(Some(frame));
        }
        Ok(None)
    }
}

#[derive(Deserialize)]
struct LoggedState {
    t: f64,
    q: [f64; 4],
    bullet_speed: f64,
    robot_id: u8,
}

/// Opens a log: the camera source plus every logged robot state.
///
/// # Errors
/// File missing or corrupt.
pub fn open(path: &Path) -> color_eyre::Result<(McapCameraSource, Vec<RobotState>)> {
    let mut states = Vec::new();
    for message in MessageReader::new(BufReader::new(File::open(path)?)) {
        let msg = message?;
        if msg.channel.topic == "/robot_state" {
            let state: LoggedState = serde_json::from_slice(&msg.data)?;
            let [w, x, y, z] = state.q;
            states.push(RobotState { t: state.t, q: UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)), bullet_speed: state.bullet_speed, robot_id: state.robot_id });
        }
    }
    let reader = MessageReader::new(BufReader::new(File::open(path)?));
    Ok((McapCameraSource { reader, next_frame_id: 0 }, states))
}

#[expect(clippy::cast_precision_loss, clippy::as_conversions, reason = "nanosecond log times stay well inside f64's exact range for a match")]
fn seconds(ns: u64) -> f64 {
    ns as f64 * 1e-9
}
