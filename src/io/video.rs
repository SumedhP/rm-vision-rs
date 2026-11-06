//! Video file camera source, decoded by an `ffmpeg` child process.

use std::fmt;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};

use image::RgbImage;

use super::CameraFrame;

/// Decodes a video file frame by frame via ffmpeg.
pub struct VideoSource {
    width: u32,
    height: u32,
    fps: f64,
    frame_id: u64,
    reader: Box<dyn Read + Send>,
    buf: Vec<u8>,
}

impl fmt::Debug for VideoSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoSource")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fps", &self.fps)
            .field("frame_id", &self.frame_id)
            .finish_non_exhaustive()
    }
}

impl VideoSource {
    /// Open a video file.
    ///
    /// # Errors
    /// ffprobe failed, ffmpeg failed, or file not found.
    pub fn open(path: &Path) -> color_eyre::Result<Self> {
        let path_str = path.to_string_lossy();
        let probe = Command::new("ffprobe")
            .args(&[
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "csv=p=0",
            ])
            .arg(path_str.as_ref())
            .output()?;

        if !probe.status.success() {
            return Err(color_eyre::eyre::eyre!("ffprobe failed for {:?}", path));
        }

        let output = String::from_utf8(probe.stdout)?;
        let parts: Vec<&str> = output.trim().split(',').collect();
        if parts.len() < 3 {
            return Err(color_eyre::eyre::eyre!("ffprobe: unexpected output"));
        }

        let width: u32 = parts[0].parse()?;
        let height: u32 = parts[1].parse()?;

        let fps_str = parts[2];
        let fps_parts: Vec<&str> = fps_str.split('/').collect();
        let fps = if fps_parts.len() == 2 {
            let num: f64 = fps_parts[0].parse()?;
            let denom: f64 = fps_parts[1].parse()?;
            num / denom
        } else {
            fps_str.parse()?
        };

        let ffmpeg = Command::new("ffmpeg")
            .args(&[
                "-v", "error",
                "-i",
            ])
            .arg(path_str.as_ref())
            .args(&[
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let reader = ffmpeg.stdout.ok_or_else(|| color_eyre::eyre::eyre!("ffmpeg stdout failed"))?;

        Ok(Self {
            width,
            height,
            fps,
            frame_id: 0,
            reader: Box::new(reader),
            buf: vec![0u8; (width * height * 3) as usize],
        })
    }

    /// Next frame, or `None` at end of file.
    ///
    /// # Errors
    /// ffmpeg failed or produced a short read.
    pub fn next_frame(&mut self) -> color_eyre::Result<Option<CameraFrame>> {
        match self.reader.read_exact(&mut self.buf) {
            Ok(()) => {
                let image = RgbImage::from_raw(self.width, self.height, self.buf.clone())
                    .ok_or_else(|| color_eyre::eyre::eyre!("invalid RGB buffer size"))?;
                let t = self.frame_id as f64 / self.fps;
                let frame = CameraFrame {
                    frame_id: self.frame_id,
                    t,
                    image: std::sync::Arc::new(image),
                };
                self.frame_id += 1;
                Ok(Some(frame))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
