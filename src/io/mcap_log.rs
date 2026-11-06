//! Camera source that replays frames from an MCAP log written by this pipeline.

use super::CameraFrame;

/// Replays logged camera frames in timestamp order.
#[derive(Debug)]
pub struct McapCameraSource {}

impl McapCameraSource {
    /// Next frame, or `None` at end of log.
    ///
    /// # Errors
    /// Corrupt log or undecodable image.
    pub fn next_frame(&mut self) -> color_eyre::Result<Option<CameraFrame>> {
        Ok(None)
    }
}
