//! Armor detector: any YOLO-pose armor model described by a [`ModelSpec`] (configured for SZURP's
//! `Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`).

use nalgebra::Point2;
use serde::Deserialize;

use crate::geometry::ArmorSize;
use crate::io::CameraFrame;

pub mod onnx;
pub mod pool;
pub mod refine;
pub mod spec;

pub use onnx::{Backend, Detector, StageTimes};
pub use onnx::DetectorConfig;
pub use spec::ModelSpec;

/// Lightbar color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArmorColor {
    /// Blue team.
    Blue,
    /// Red team.
    Red,
    /// Unlit (destroyed or not powered).
    Gray,
    /// Purple (invincible). Never a valid target.
    Purple,
}

/// Plate sticker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Icon {
    /// Sentry.
    Sentry,
    /// Hero (1).
    Hero,
    /// Engineer (2).
    Engineer,
    /// Infantry 3.
    Infantry3,
    /// Infantry 4.
    Infantry4,
    /// Infantry 5.
    Infantry5,
    /// Outpost.
    Outpost,
    /// Base, lower plate.
    BaseBottom,
    /// Base, upper plate.
    BaseTop,
}

impl Icon {
    /// Physical plate size for this icon.
    pub fn armor_size(self) -> ArmorSize {
        match self {
            Icon::Hero | Icon::BaseBottom => ArmorSize::Large,
            _ => ArmorSize::Small,
        }
    }
}

/// One detected armor plate.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Detection {
    /// Lightbar color.
    pub color: ArmorColor,
    /// Sticker class.
    pub icon: Icon,
    /// Best class score, 0..1.
    pub confidence: f32,
    /// Lightbar end points in original image pixels: TL, BL, BR, TR.
    pub corners: [Point2<f64>; 4],
}

/// A frame plus everything the detector found in it.
#[derive(Debug, Clone)]
pub struct DetectedFrame {
    /// The input frame.
    pub frame: CameraFrame,
    /// Detections after color filtering, NMS and corner refinement.
    pub detections: Vec<Detection>,
}
