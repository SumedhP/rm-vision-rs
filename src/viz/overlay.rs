//! Draws the detections and the robot estimate over the camera image and encodes it to a video.
//!
//! Everything is drawn as thin outlines so the footage stays visible. Green = detections (icon
//! and confidence above each), cyan = the four estimated plates (orange = the one aimed at).

use std::io::Write;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::OnceLock;

use ab_glyph::{FontArc, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_line_segment_mut, draw_text_mut};
use nalgebra::{Point2, Vector3};

use crate::detector::DetectedFrame;
use crate::geometry::{Camera, armor_corners_world};
use crate::pipeline::FrameRecord;

const GREEN: Rgb<u8> = Rgb([60, 255, 60]);
const CYAN: Rgb<u8> = Rgb([0, 230, 255]);
const ORANGE: Rgb<u8> = Rgb([255, 150, 0]);
const WHITE: Rgb<u8> = Rgb([240, 240, 240]);

/// Renders one annotated frame.
pub fn render(frame: &DetectedFrame, record: &FrameRecord, camera: &Camera) -> RgbImage {
    let mut img = (*frame.frame.image).clone();
    let font = font();

    for det in &frame.detections {
        polyline(&mut img, &det.corners, GREEN);
        if let Some(font) = font {
            let top = det.corners.iter().fold(Point2::new(f64::MAX, f64::MAX), |m, p| Point2::new(m.x.min(p.x), m.y.min(p.y)));
            let label = format!("{:?} {:.2}", det.icon, det.confidence);
            draw_text_mut(&mut img, GREEN, px(top.x), px(top.y) - 16, PxScale::from(14.0), font, &label);
        }
    }

    if let Some(est) = &record.estimate {
        let aimed = record.target.map(|tg| tg.plate);
        for plate in est.plates_at(frame.frame.t) {
            let corners: Option<Vec<Point2<f64>>> = armor_corners_world(est.icon.armor_size(), &plate.position, plate.yaw)
                .iter()
                .map(|p: &Vector3<f64>| camera.project(&record.robot.q, p))
                .collect();
            if let Some(c) = corners {
                polyline(&mut img, &c, if aimed == Some(plate.id) { ORANGE } else { CYAN });
            }
        }
        if let Some(font) = font {
            let text = format!("target {:?} ({:.0}% icon agreement)  seen {} frames", est.icon, est.icon_agreement * 100.0, est.frames_seen);
            draw_text_mut(&mut img, WHITE, 8, 8, PxScale::from(18.0), font, &text);
        }
    }
    img
}
fn font() -> Option<&'static FontArc> {
    static FONT: OnceLock<Option<FontArc>> = OnceLock::new();
    FONT.get_or_init(|| {
        ["C:\\Windows\\Fonts\\consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
            .iter()
            .find_map(|p| std::fs::read(p).ok().and_then(|d| FontArc::try_from_vec(d).ok()))
    })
    .as_ref()
}

fn polyline(img: &mut RgbImage, pts: &[Point2<f64>], color: Rgb<u8>) {
    let n = pts.len();
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        draw_line_segment_mut(img, (to_f32(a.x), to_f32(a.y)), (to_f32(b.x), to_f32(b.y)), color);
    }
}

#[expect(clippy::cast_possible_truncation, clippy::as_conversions, reason = "pixel coordinates; imageproc clips off-image values")]
fn to_f32(v: f64) -> f32 {
    v as f32
}

#[expect(clippy::cast_possible_truncation, clippy::as_conversions, reason = "pixel coordinates; imageproc clips off-image values")]
fn px(v: f64) -> i32 {
    v.round() as i32
}

/// Streams rendered frames into an ffmpeg H.264 encoder.
#[derive(Debug)]
pub struct OverlayWriter {
    process: Child,
    stdin: ChildStdin,
}

impl OverlayWriter {
    /// Starts an encoder writing `path`.
    ///
    /// # Errors
    /// ffmpeg could not be started.
    pub fn create(path: &Path, width: u32, height: u32, fps: f64) -> color_eyre::Result<Self> {
        let mut process = Command::new("ffmpeg")
            .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{width}x{height}"), "-r", &fps.to_string()])
            .args(["-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()?;
        let stdin = process.stdin.take().ok_or_else(|| color_eyre::eyre::eyre!("could not pipe to ffmpeg"))?;
        Ok(Self { process, stdin })
    }

    /// Appends one rendered frame.
    ///
    /// # Errors
    /// The encoder went away.
    pub fn write(&mut self, image: &RgbImage) -> color_eyre::Result<()> {
        self.stdin.write_all(image.as_raw())?;
        Ok(())
    }

    /// Flushes and waits for the encoder.
    ///
    /// # Errors
    /// ffmpeg failed.
    pub fn finish(self) -> color_eyre::Result<()> {
        let Self { mut process, stdin } = self;
        drop(stdin);
        let status = process.wait()?;
        if status.success() { Ok(()) } else { Err(color_eyre::eyre::eyre!("ffmpeg exited with {status}")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CameraConfig;
    use crate::io::{CameraFrame, RobotState};
    use crate::tracker::model::ix;
    use crate::tracker::{RobotEstimate, State, TrackMode, TrackState};
    use nalgebra::UnitQuaternion;

    #[test]
    fn draws_the_estimate_where_the_robot_projects() {
        let camera = Camera::from(&CameraConfig { k: [691.0, 691.0, 486.0, 272.0], dist: [0.0; 4], rpy: [0.0; 3], translation: [0.0; 3] });
        let frame = DetectedFrame { frame: CameraFrame { frame_id: 0, t: 0.0, image: std::sync::Arc::new(RgbImage::new(960, 540)) }, detections: vec![] };
        let mut x = State::zeros();
        x[ix::CX] = 3.0;
        x[ix::R] = 0.25;
        let estimate = RobotEstimate {
            t: 0.0,
            icon: crate::detector::Icon::Infantry3,
            state: TrackState::Confirmed,
            mode: TrackMode::Robot,
            plate_count: 4,
            x,
            p: nalgebra::SMatrix::zeros(),
            last_plate: None,
            updates: vec![],
            frames_seen: 10,
            icon_agreement: 1.0,
        };
        let robot = RobotState { t: 0.0, q: UnitQuaternion::identity(), bullet_speed: 23.0, robot_id: 103 };
        let record = FrameRecord { robot, estimate: Some(estimate), commands: vec![], target: None };
        let img = render(&frame, &record, &camera);
        let lit = |x0: u32, x1: u32, y0: u32, y1: u32| (x0..x1).flat_map(|x| (y0..y1).map(move |y| (x, y))).any(|(x, y)| img.get_pixel(x, y)[1] > 100);
        assert!(lit(440, 530, 255, 290), "front plate outline drawn around the principal point");
        assert!(!lit(700, 960, 400, 540), "nothing drawn far from the robot");
    }
}
