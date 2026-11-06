//! Pipeline inputs: camera frames and robot (turret + referee) state.
//!
//! Each input has a closed set of sources. `Live` hardware sources (camera SDK, serial) are
//! added as new variants when the hardware is available.

use std::sync::Arc;

use image::RgbImage;
use nalgebra::UnitQuaternion;
use serde::Deserialize;

pub mod mcap_log;
pub mod video;

/// One camera image. `t` is seconds on the pipeline clock, already corrected by `camera_delay`.
#[derive(Debug, Clone)]
pub struct CameraFrame {
    /// Monotonic frame counter from the source.
    pub frame_id: u64,
    /// Exposure time, seconds.
    pub t: f64,
    /// RGB pixels.
    pub image: Arc<RgbImage>,
}

/// Turret and referee state from the MCU.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct RobotState {
    /// Sample time, seconds on the pipeline clock.
    #[serde(default)]
    pub t: f64,
    /// Gimbal orientation (world from gimbal), `[w, x, y, z]` in config.
    #[serde(deserialize_with = "quat_wxyz")]
    pub q: UnitQuaternion<f64>,
    /// Last measured bullet speed, m/s.
    pub bullet_speed: f64,
    /// Own robot id. Ids above 100 are blue, matching the referee convention.
    pub robot_id: u8,
}

fn quat_wxyz<'de, D: serde::Deserializer<'de>>(d: D) -> Result<UnitQuaternion<f64>, D::Error> {
    let [w, x, y, z] = <[f64; 4]>::deserialize(d)?;
    Ok(UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w, x, y, z)))
}

/// Where camera frames come from.
#[derive(Debug)]
#[non_exhaustive]
pub enum CameraSource {
    /// A video file decoded by ffmpeg. Timestamps are `frame_id / fps`.
    Video(video::VideoSource),
    /// Frames recorded in an MCAP log by this pipeline.
    Log(mcap_log::McapCameraSource),
}

impl CameraSource {
    /// Next frame, or `None` at end of stream.
    ///
    /// # Errors
    /// Decoding or I/O failure in the underlying source.
    pub fn next_frame(&mut self) -> color_eyre::Result<Option<CameraFrame>> {
        match self {
            CameraSource::Video(s) => s.next_frame(),
            CameraSource::Log(s) => s.next_frame(),
        }
    }
}

/// Where robot state comes from.
#[derive(Debug)]
#[non_exhaustive]
pub enum RobotSource {
    /// No robot data (e.g. a plain video). Every query returns this state from config.
    Defaults(RobotState),
    /// Robot states recorded in an MCAP log, in timestamp order.
    Log(Vec<RobotState>),
}
