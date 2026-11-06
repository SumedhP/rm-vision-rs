//! ONNX Runtime inference for the SZURP 21-row YOLOv8-pose armor model.

use std::path::PathBuf;

use color_eyre::eyre::{WrapErr, bail};
use image::{RgbImage, imageops};
use nalgebra::Point2;
use ndarray::Array4;
use ort::environment::Environment;
use ort::ep::{CPU, ExecutionProviderDispatch};
#[cfg(feature = "gpu")]
use ort::ep::{CUDA, ExecutionProvider, TensorRT};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Deserialize;

use crate::detector::{ArmorColor, Detection, Icon};

const IN_W: u32 = 640;
const IN_H: u32 = 480;
const PAD: u8 = 124;

/// Execution provider, tried in the configured order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Backend {
    /// NVIDIA TensorRT.
    #[serde(rename = "tensorrt")]
    TensorRt,
    /// NVIDIA CUDA.
    Cuda,
    /// CPU, always available.
    Cpu,
}

impl Backend {
    fn is_available(self) -> bool {
        match self {
            #[cfg(feature = "gpu")]
            Backend::TensorRt => TensorRT::is_available().unwrap_or(false),
            #[cfg(feature = "gpu")]
            Backend::Cuda => CUDA::is_available().unwrap_or(false),
            #[cfg(not(feature = "gpu"))]
            Backend::TensorRt | Backend::Cuda => false,
            Backend::Cpu => true,
        }
    }
}

/// Detector settings (`detector:` section of `configs/general.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConfig {
    /// ONNX model path.
    pub model: PathBuf,
    /// Execution providers in priority order; CPU is the implicit last resort.
    pub backends: Vec<Backend>,
    /// Fail at startup unless the first backend is available.
    pub require_backend: bool,
    /// FP16 for TensorRT.
    pub fp16: bool,
    /// TensorRT engine cache directory.
    pub engine_cache: PathBuf,
    /// Minimum class score.
    pub score_threshold: f32,
    /// IoU above which the weaker of two overlapping detections is dropped.
    pub nms_iou: f64,
    /// Snap keypoints to lightbar edges.
    pub refine_corners: bool,
    /// Detector worker threads.
    pub workers: usize,
}

/// One ONNX Runtime session plus pre/post-processing.
#[derive(Debug)]
pub struct Detector {
    session: Session,
    backend: Backend,
    cfg: DetectorConfig,
}

impl Detector {
    /// Loads the model on the first available backend.
    ///
    /// # Errors
    /// Model missing or invalid, or `require_backend` set and the first backend unavailable.
    pub fn new(cfg: &DetectorConfig) -> color_eyre::Result<Self> {
        let backend = cfg.backends.iter().copied().find(|b| b.is_available()).unwrap_or(Backend::Cpu);
        let first = cfg.backends.first().copied().unwrap_or(Backend::Cpu);
        if backend != first {
            if cfg.require_backend {
                bail!("required backend {first:?} is unavailable");
            }
            tracing::warn!(requested = ?first, active = ?backend, "detector backend fell back");
        } else {
            tracing::info!(active = ?backend, "detector backend");
        }
        let providers: Vec<ExecutionProviderDispatch> = cfg.backends.iter().map(|b| provider(*b, cfg)).collect();
        let session = Session::builder(environment()?)
            .map_err(ort_err)?
            .with_execution_providers(providers)
            .map_err(ort_err)?
            .with_intra_threads(1)
            .map_err(ort_err)?
            .commit_from_file(&cfg.model)
            .map_err(ort_err)
            .wrap_err_with(|| format!("loading {}", cfg.model.display()))?;
        Ok(Self { session, backend, cfg: cfg.clone() })
    }

    /// Backend the session actually runs on.
    pub fn active_backend(&self) -> Backend {
        self.backend
    }

    /// Detects `enemy` armor plates in an RGB image.
    ///
    /// # Errors
    /// Inference failed.
    pub fn detect(&mut self, image: &RgbImage, enemy: ArmorColor) -> color_eyre::Result<Vec<Detection>> {
        let lb = Letterbox::new(image.width(), image.height());
        let input = lb.tensor(image);
        let outputs = self
            .session
            .run(ort::inputs!["images" => TensorRef::from_array_view(&input).map_err(ort_err)?])
            .map_err(ort_err)?;
        let (_, raw) = outputs["output0"].try_extract_tensor::<f32>().map_err(ort_err)?;
        let mut dets = decode(raw, &lb, enemy, self.cfg.score_threshold);
        dets = nms(dets, self.cfg.nms_iou);
        if self.cfg.refine_corners {
            dets = dets.into_iter().map(|d| super::refine::refine(image, &d)).collect();
        }
        Ok(dets)
    }
}

fn environment() -> color_eyre::Result<&'static Environment> {
    static ENV: std::sync::OnceLock<Environment> = std::sync::OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::environment::init().with_name("auto_aim").build().map_err(ort_err)?;
    Ok(ENV.get_or_init(|| env))
}

fn provider(b: Backend, cfg: &DetectorConfig) -> ExecutionProviderDispatch {
    let dispatch = match b {
        #[cfg(feature = "gpu")]
        Backend::TensorRt => TensorRT::default()
            .with_fp16(cfg.fp16)
            .with_engine_cache(true)
            .with_engine_cache_path(cfg.engine_cache.display())
            .build(),
        #[cfg(feature = "gpu")]
        Backend::Cuda => CUDA::default().build(),
        _ => CPU::default().build(),
    };
    if cfg.require_backend && cfg.backends.first() == Some(&b) { dispatch.error_on_failure() } else { dispatch }
}

fn ort_err<R>(e: ort::Error<R>) -> color_eyre::Report {
    color_eyre::eyre::eyre!("onnxruntime: {e}")
}

/// Aspect-preserving resize into the model input, centered with gray padding.
#[derive(Debug, Clone, Copy)]
struct Letterbox {
    scale: f64,
    pad_x: u32,
    pad_y: u32,
    w: u32,
    h: u32,
    src_w: u32,
    src_h: u32,
}

impl Letterbox {
    fn new(src_w: u32, src_h: u32) -> Self {
        let scale = (f64::from(IN_H) / f64::from(src_h)).min(f64::from(IN_W) / f64::from(src_w));
        let w = round_px(f64::from(src_w) * scale).min(IN_W);
        let h = round_px(f64::from(src_h) * scale).min(IN_H);
        Self { scale, pad_x: (IN_W - w) / 2, pad_y: (IN_H - h) / 2, w, h, src_w, src_h }
    }

    fn tensor(&self, image: &RgbImage) -> Array4<f32> {
        let resized = imageops::resize(image, self.w, self.h, imageops::FilterType::Triangle);
        let mut t = Array4::from_elem((1, 3, usize_of(IN_H), usize_of(IN_W)), f32::from(PAD) / 255.0);
        for (x, y, px) in resized.enumerate_pixels() {
            let (row, col) = (usize_of(y + self.pad_y), usize_of(x + self.pad_x));
            for c in 0..3 {
                t[[0, c, row, col]] = f32::from(px[c]) / 255.0;
            }
        }
        t
    }

    fn to_source(&self, x: f32, y: f32) -> Option<Point2<f64>> {
        let p = Point2::new(
            (f64::from(x) - f64::from(self.pad_x)) / self.scale,
            (f64::from(y) - f64::from(self.pad_y)) / self.scale,
        );
        let inside = (0.0..=f64::from(self.src_w)).contains(&p.x) && (0.0..=f64::from(self.src_h)).contains(&p.y);
        inside.then_some(p)
    }
}

/// Output layout per candidate column: 4 color scores, 9 class scores, 4 keypoints (x, y).
fn decode(raw: &[f32], lb: &Letterbox, enemy: ArmorColor, threshold: f32) -> Vec<Detection> {
    let n = raw.len() / 21;
    let at = |row: usize, i: usize| raw[row * n + i];
    let argmax = |rows: std::ops::Range<usize>, i: usize| {
        rows.clone().max_by(|&a, &b| at(a, i).total_cmp(&at(b, i))).map_or(0, |r| r - rows.start)
    };
    (0..n)
        .filter_map(|i| {
            let class = argmax(4..13, i);
            let confidence = at(4 + class, i);
            if confidence < threshold {
                return None;
            }
            let color = [ArmorColor::Blue, ArmorColor::Red, ArmorColor::Gray, ArmorColor::Purple][argmax(0..4, i)];
            if color != enemy {
                return None;
            }
            let mut corners = [Point2::origin(); 4];
            for (k, c) in corners.iter_mut().enumerate() {
                *c = lb.to_source(at(13 + 2 * k, i), at(14 + 2 * k, i))?;
            }
            Some(Detection { color, icon: Icon::from_class(class)?, confidence, corners })
        })
        .collect()
}

fn nms(mut dets: Vec<Detection>, iou_max: f64) -> Vec<Detection> {
    dets.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    let mut kept: Vec<Detection> = Vec::new();
    for d in dets {
        if kept.iter().all(|k| iou(&bbox(&k.corners), &bbox(&d.corners)) <= iou_max) {
            kept.push(d);
        }
    }
    kept
}

fn bbox(c: &[Point2<f64>; 4]) -> [f64; 4] {
    c.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |[x0, y0, x1, y1], p| {
        [x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y)]
    })
}

fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let area = |r: &[f64; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let inter = area(&[a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])]);
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::as_conversions, reason = "pixel sizes are small and non-negative")]
fn round_px(v: f64) -> u32 {
    v.round() as u32
}

fn usize_of(v: u32) -> usize {
    usize::try_from(v).expect("u32 fits in usize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_matches_szurp_for_960x540() {
        let lb = Letterbox::new(960, 540);
        assert_eq!((lb.w, lb.h, lb.pad_x, lb.pad_y), (640, 360, 0, 60), "scale 2/3, 60 px bars top and bottom");
        let p = lb.to_source(320.0, 240.0).expect("center maps inside");
        assert!((p.x - 480.0).abs() < 1e-9 && (p.y - 270.0).abs() < 1e-9, "model center is image center");
    }

    #[test]
    fn nms_keeps_strongest_of_overlapping() {
        let d = |conf: f32, x: f64| Detection {
            color: ArmorColor::Red,
            icon: Icon::Infantry3,
            confidence: conf,
            corners: [Point2::new(x, 0.0), Point2::new(x, 10.0), Point2::new(x + 30.0, 10.0), Point2::new(x + 30.0, 0.0)],
        };
        let kept = nms(vec![d(0.6, 1.0), d(0.9, 0.0), d(0.8, 100.0)], 0.4);
        assert_eq!(kept.len(), 2, "overlapping pair collapses");
        assert!((kept[0].confidence - 0.9).abs() < 1e-6, "strongest first");
    }
}
