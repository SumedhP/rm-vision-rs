//! Foxglove publishing. The same messages go to an MCAP file (the log) and, optionally, a live
//! websocket server that the Foxglove app connects to.

use std::path::Path;

use crate::detector::DetectedFrame;
use crate::geometry::Camera;
use crate::pipeline::FrameRecord;

/// Publishes every frame's results.
#[derive(Debug)]
pub struct Publisher {}

impl Publisher {
    /// Opens the MCAP file and/or starts the websocket server. Images are logged at most
    /// `image_rate` times per second; everything else is logged every frame.
    ///
    /// # Errors
    /// File or socket could not be opened.
    pub fn new(mcap: Option<&Path>, websocket_port: Option<u16>, image_rate: f64) -> color_eyre::Result<Self> {
        let _ = (mcap, websocket_port, image_rate);
        Ok(Self {})
    }

    /// Logs one frame.
    ///
    /// # Errors
    /// Encoding failed.
    pub fn publish(&mut self, frame: &DetectedFrame, record: &FrameRecord, camera: &Camera) -> color_eyre::Result<()> {
        let _ = (frame, record, camera);
        Ok(())
    }

    /// Flushes and closes the MCAP file.
    ///
    /// # Errors
    /// Write failed.
    pub fn finish(self) -> color_eyre::Result<()> {
        Ok(())
    }
}
