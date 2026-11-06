//! Foxglove publishing. The same messages go to an MCAP file (the log) and, optionally, a live
//! websocket server that the Foxglove app connects to.

use std::path::Path;
use std::sync::Arc;

use foxglove::messages::{
    Color, CompressedImage, CubePrimitive, ImageAnnotations, Point2, PointsAnnotation,
    Pose, Quaternion, SceneEntity, SceneUpdate, Vector3 as FxVector3,
};
use foxglove::{Channel, Context, McapWriter, RawChannel, WebSocketServer, WebSocketServerHandle};
use image::RgbImage;
use serde_json::json;

use crate::detector::DetectedFrame;
use crate::geometry::{armor_corners_world, Camera};
use crate::pipeline::FrameRecord;

/// Publishes every frame's results.
#[derive(Debug)]
pub struct Publisher {
    #[expect(dead_code, reason = "Arc<Context> kept alive for side effects")]
    context: Arc<Context>,
    image_channel: Channel<CompressedImage>,
    annotations_channel: Channel<ImageAnnotations>,
    robot_state_channel: Arc<RawChannel>,
    frame_channel: Arc<RawChannel>,
    scene_channel: Channel<SceneUpdate>,
    mcap: Option<foxglove::McapWriterHandle<std::io::BufWriter<std::fs::File>>>,
    websocket: Option<WebSocketServerHandle>,
    image_rate: f64,
    last_image_time: f64,
}

impl Publisher {
    /// Opens the MCAP file and/or starts the websocket server. Images are logged at most
    /// `image_rate` times per second; everything else is logged every frame.
    ///
    /// # Errors
    /// File or socket could not be opened.
    pub fn new(mcap: Option<&Path>, websocket_port: Option<u16>, image_rate: f64) -> color_eyre::Result<Self> {
        let context = Context::new();

        let mcap_writer = if let Some(path) = mcap {
            let writer = McapWriter::new()
                .context(&context)
                .create(std::io::BufWriter::new(std::fs::File::create(path)?))?;
            Some(writer)
        } else {
            None
        };

        let websocket_handle = if let Some(port) = websocket_port {
            let server = WebSocketServer::new()
                .bind("127.0.0.1", port)
                .context(&context)
                .start_blocking()?;
            Some(server)
        } else {
            None
        };

        let robot_state_channel = context
            .channel_builder("/robot_state")
            .message_encoding("json")
            .build_raw()?;
        let frame_channel = context
            .channel_builder("/frame")
            .message_encoding("json")
            .build_raw()?;

        Ok(Self {
            context: Arc::clone(&context),
            image_channel: context.channel_builder("/camera/image").build(),
            annotations_channel: context.channel_builder("/camera/annotations").build(),
            robot_state_channel,
            frame_channel,
            scene_channel: context.channel_builder("/scene").build(),
            mcap: mcap_writer,
            websocket: websocket_handle,
            image_rate,
            last_image_time: -1e9,
        })
    }

    /// Logs one frame.
    ///
    /// # Errors
    /// Encoding failed.
    pub fn publish(&mut self, frame: &DetectedFrame, record: &FrameRecord, camera: &Camera) -> color_eyre::Result<()> {
        let t = frame.frame.t;
        let log_time_ns = time_to_ns(t);

        // Publish image at rate-limited frequency
        if self.image_rate <= 0.0 || t - self.last_image_time >= 1.0 / self.image_rate {
            self.last_image_time = t;

            let jpeg_data = encode_jpeg(&frame.frame.image, 80)?;
            let img_msg = CompressedImage {
                timestamp: Some(ns_to_timestamp(log_time_ns)),
                frame_id: "camera".into(),
                data: jpeg_data.into(),
                format: "jpeg".into(),
            };
            self.image_channel.log_with_meta(&img_msg, at(log_time_ns));
        }

        // Publish image annotations
        let annotations = build_annotations(frame, record, camera);
        self.annotations_channel.log_with_meta(&annotations, at(log_time_ns));

        // Publish robot state as JSON
        let robot_json = json!({
            "t": record.robot.t,
            "q": [
                record.robot.q.w,
                record.robot.q.i,
                record.robot.q.j,
                record.robot.q.k,
            ],
            "bullet_speed": record.robot.bullet_speed,
            "robot_id": record.robot.robot_id,
        });
        let robot_bytes = serde_json::to_vec(&robot_json)?;
        self.robot_state_channel.log_with_meta(&robot_bytes, at(log_time_ns));

        // Publish full frame record
        let mut frame_json = serde_json::to_value(record)?;
        if let serde_json::Value::Object(ref mut obj) = frame_json {
            obj.insert("frame_id".into(), json!(frame.frame.frame_id));
            obj.insert("detections".into(), serde_json::to_value(&frame.detections)?);
        }
        let frame_bytes = serde_json::to_vec(&frame_json)?;
        self.frame_channel.log_with_meta(&frame_bytes, at(log_time_ns));

        // Publish scene (predicted plates as cubes)
        if let Some(est) = &record.estimate {
            let plates = est.plates_at(t);
            let mut entities = Vec::new();
            for plate in &plates {
                let center = &plate.position;
                let yaw = plate.yaw;

                let corners = armor_corners_world(est.icon.armor_size(), center, yaw);
                let valid = corners.iter().all(|c| camera.project(&record.robot.q, c).is_some());
                if !valid {
                    continue;
                }

                let pose = Pose {
                    position: Some(FxVector3 { x: center.x, y: center.y, z: center.z }),
                    orientation: Some(quaternion_from_yaw(yaw)),
                };

                let size_m = est.icon.armor_size();
                let cube = CubePrimitive {
                    pose: Some(pose),
                    size: Some(FxVector3 {
                        x: size_m.width(),
                        y: 0.02,
                        z: 0.056,
                    }),
                    color: Some(Color { r: 0.0, g: 1.0, b: 1.0, a: 0.5 }),
                };

                let center_sphere = foxglove::messages::SpherePrimitive {
                    pose: Some(Pose {
                        position: Some(FxVector3 { x: center.x, y: center.y, z: center.z }),
                        orientation: Some(Quaternion { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }),
                    }),
                    size: Some(FxVector3 { x: 0.05, y: 0.05, z: 0.05 }),
                    color: Some(Color { r: 1.0, g: 1.0, b: 0.0, a: 0.8 }),
                };

                let entity = SceneEntity {
                    timestamp: Some(ns_to_timestamp(log_time_ns)),
                    frame_id: "world".into(),
                    id: format!("plate_{}", plate.id),
                    lifetime: None,
                    frame_locked: false,
                    metadata: vec![],
                    arrows: vec![],
                    cubes: vec![cube],
                    spheres: vec![center_sphere],
                    cylinders: vec![],
                    lines: vec![],
                    triangles: vec![],
                    texts: vec![],
                    models: vec![],
                };
                entities.push(entity);
            }

            let scene = SceneUpdate { deletions: vec![], entities };
            self.scene_channel.log_with_meta(&scene, at(log_time_ns));
        }

        Ok(())
    }

    /// Flushes and closes the MCAP file.
    ///
    /// # Errors
    /// Write failed.
    pub fn finish(self) -> color_eyre::Result<()> {
        if let Some(websocket) = self.websocket {
            let _ = websocket.stop();
        }
        if let Some(writer) = self.mcap {
            writer.close()?;
        }
        Ok(())
    }
}

/// Messages are stamped with the frame's data time, not the wall clock, so a replayed log has
/// the same timeline as the run that recorded it.
fn at(log_time_ns: u64) -> foxglove::PartialMetadata {
    foxglove::PartialMetadata { log_time: Some(log_time_ns) }
}

#[expect(clippy::as_conversions, reason = "protobuf enum to i32 conversion")]
fn enum_to_i32(e: foxglove::messages::points_annotation::Type) -> i32 {
    e as i32
}

fn time_to_ns(t: f64) -> u64 {
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::as_conversions, reason = "nanosecond conversion is intentional")]
    let ns = (t * 1e9) as u64;
    ns
}

fn ns_to_timestamp(ns: u64) -> foxglove::messages::Timestamp {
    #[expect(clippy::cast_possible_truncation, clippy::as_conversions, reason = "time fits in u32 seconds")]
    {
        foxglove::messages::Timestamp::new(
            (ns / 1_000_000_000) as u32,
            (ns % 1_000_000_000) as u32,
        )
    }
}

fn quaternion_from_yaw(yaw: f64) -> Quaternion {
    let half_yaw = yaw / 2.0;
    Quaternion {
        x: 0.0,
        y: 0.0,
        z: half_yaw.sin(),
        w: half_yaw.cos(),
    }
}

fn encode_jpeg(img: &RgbImage, quality: u8) -> color_eyre::Result<Vec<u8>> {
    let buf = Vec::new();
    let mut cursor = std::io::Cursor::new(buf);
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cursor, quality);
    encoder.encode(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgb8)?;
    Ok(cursor.into_inner())
}

fn build_annotations(
    frame: &DetectedFrame,
    record: &FrameRecord,
    camera: &Camera,
) -> ImageAnnotations {
    let t = frame.frame.t;
    let q = record.robot.q;
    let log_time_ns = time_to_ns(t);

    let mut points_annot = Vec::new();

    // Green detection polygons
    if !frame.detections.is_empty() {
        let det_points: Vec<Point2> = frame
            .detections
            .iter()
            .flat_map(|d| d.corners.iter())
            .map(|p| Point2 { x: p.x, y: p.y })
            .collect();

        points_annot.push(PointsAnnotation {
            timestamp: None,
            r#type: enum_to_i32(foxglove::messages::points_annotation::Type::LineLoop),
            points: det_points,
            outline_color: Some(Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }),
            outline_colors: vec![],
            fill_color: None,
            thickness: 2.0,
            metadata: vec![],
        });
    }

    // Cyan predicted plate polygons
    if let Some(est) = &record.estimate {
        let plates = est.plates_at(t);
        for plate in &plates {
            let center = camera.project(&q, &plate.position);
            if center.is_none() {
                continue;
            }

            let plate_to_viewer = plate.position.y.atan2(plate.position.x);
            let facing = crate::tracker::model::wrap_angle(plate.yaw - plate_to_viewer).abs();
            if facing > std::f64::consts::FRAC_PI_2 {
                continue;
            }

            let corners = armor_corners_world(est.icon.armor_size(), &plate.position, plate.yaw);
            let projected: Vec<Point2> = corners
                .iter()
                .filter_map(|c| camera.project(&q, c))
                .map(|p| Point2 { x: p.x, y: p.y })
                .collect();

            if projected.len() == 4 {
                points_annot.push(PointsAnnotation {
                    timestamp: None,
                    r#type: enum_to_i32(foxglove::messages::points_annotation::Type::LineLoop),
                    points: projected,
                    outline_color: Some(Color { r: 0.0, g: 1.0, b: 1.0, a: 1.0 }),
                    outline_colors: vec![],
                    fill_color: None,
                    thickness: 1.0,
                    metadata: vec![],
                });
            }
        }
    }

    // Red aim point
    if let Some(target) = &record.target && let Some(impact_px) = camera.project(&q, &target.position) {
        points_annot.push(PointsAnnotation {
            timestamp: None,
            r#type: enum_to_i32(foxglove::messages::points_annotation::Type::Points),
            points: vec![Point2 { x: impact_px.x, y: impact_px.y }],
            outline_color: Some(Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }),
            outline_colors: vec![],
            fill_color: Some(Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }),
            thickness: 8.0,
            metadata: vec![],
        });
    }

    ImageAnnotations {
        timestamp: Some(ns_to_timestamp(log_time_ns)),
        circles: vec![],
        points: points_annot,
        texts: vec![],
        metadata: vec![],
    }
}
