//! What a YOLO-pose armor model expects and outputs, so one detector serves any net of that
//! family. A new model of the same shape is a new `spec` block in the config, not new code.

use serde::Deserialize;

use super::{ArmorColor, Icon};

/// Input and output layout of one model (`detector.spec` in `configs/general.yaml`).
///
/// The output is one tensor with a row per value and a column per candidate: color scores, then
/// class scores, then `(x, y)` for each of the 4 keypoints, in model input pixels.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    /// Input tensor name.
    pub input_name: String,
    /// Output tensor name.
    pub output_name: String,
    /// Input `[width, height]`, pixels. The image is letterboxed into it, RGB scaled to `[0, 1]`.
    pub input_size: [u32; 2],
    /// Letterbox fill value, 0-255.
    pub pad: u8,
    /// Color score rows, in output order.
    pub colors: Vec<ArmorColor>,
    /// Class score rows, in output order.
    pub icons: Vec<Icon>,
    /// Scores are logits and need a sigmoid.
    pub logits: bool,
    /// Output is `[candidates, rows]` instead of `[rows, candidates]`.
    pub transposed: bool,
    /// Model keypoint index of each corner, in our order: top-left, bottom-left, bottom-right,
    /// top-right.
    pub corners: [usize; 4],
}

impl ModelSpec {
    /// Values per candidate.
    pub fn rows(&self) -> usize {
        self.colors.len() + self.icons.len() + 8
    }
}
