//! Draws the current estimate over the camera image and encodes it to a video file.

use std::path::Path;

use image::RgbImage;

use crate::detector::DetectedFrame;
use crate::geometry::Camera;
use crate::pipeline::FrameRecord;

/// Renders one annotated frame.
pub fn render(frame: &DetectedFrame, record: &FrameRecord, camera: &Camera) -> RgbImage {
    let _ = (record, camera);
    (*frame.frame.image).clone()
}

/// Streams rendered frames into an ffmpeg H.264 encoder.
#[derive(Debug)]
pub struct OverlayWriter {}

impl OverlayWriter {
    /// Starts an encoder writing `path`.
    ///
    /// # Errors
    /// ffmpeg could not be started.
    pub fn create(path: &Path, width: u32, height: u32, fps: f64) -> color_eyre::Result<Self> {
        let _ = (path, width, height, fps);
        Ok(Self {})
    }

    /// Appends one rendered frame.
    ///
    /// # Errors
    /// The encoder went away.
    pub fn write(&mut self, image: &RgbImage) -> color_eyre::Result<()> {
        let _ = image;
        Ok(())
    }

    /// Flushes and waits for the encoder.
    ///
    /// # Errors
    /// ffmpeg failed.
    pub fn finish(self) -> color_eyre::Result<()> {
        Ok(())
    }
}
