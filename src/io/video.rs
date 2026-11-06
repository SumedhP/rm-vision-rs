//! Video file camera source, decoded by an `ffmpeg` child process.

use super::CameraFrame;

/// Decodes a video file frame by frame.
#[derive(Debug)]
pub struct VideoSource {}

impl VideoSource {
    /// Next frame, or `None` at end of file.
    ///
    /// # Errors
    /// ffmpeg failed or produced a short read.
    pub fn next_frame(&mut self) -> color_eyre::Result<Option<CameraFrame>> {
        Ok(None)
    }
}
