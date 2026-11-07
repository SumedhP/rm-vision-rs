//! ONNX Runtime inference for YOLO-pose armor models, laid out as described by a [`ModelSpec`].

use std::path::PathBuf;
use std::time::{Duration, Instant};

use color_eyre::eyre::{WrapErr, bail, eyre};
use image::RgbImage;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::preprocess::{Preprocessor, ResizeMode};
use kornia_tensor::Tensor;
use nalgebra::Point2;
use ort::environment::Environment;
use ort::ep::{CPU, ExecutionProviderDispatch};
#[cfg(feature = "gpu")]
use ort::ep::{CUDA, ExecutionProvider, TensorRT};
use ort::session::Session;
use ort::value::TensorRef;
use serde::Deserialize;

use crate::detector::{ArmorColor, Detection, ModelSpec};

/// Execution provider, tried in the configured order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Backend {
    /// NVIDIA `TensorRT`.
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
    /// The model's input and output layout.
    pub spec: ModelSpec,
    /// Execution providers in priority order; CPU is the implicit last resort.
    pub backends: Vec<Backend>,
    /// Fail at startup unless the first backend is available.
    pub require_backend: bool,
    /// FP16 for `TensorRT`.
    pub fp16: bool,
    /// `TensorRT` engine cache directory.
    pub engine_cache: PathBuf,
    /// Minimum class score.
    pub score_threshold: f32,
    /// `IoU` above which the weaker of two overlapping detections is dropped.
    pub nms_iou: f64,
    /// Snap keypoints to lightbar edges.
    pub refine_corners: bool,
    /// Detector worker threads.
    pub workers: usize,
}

/// Wall time of each stage of the last [`Detector::detect`] call.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimes {
    /// Resize, letterbox and convert to the input tensor.
    pub preprocess: Duration,
    /// ONNX Runtime session run.
    pub inference: Duration,
    /// Decode candidates and non-maximum suppression.
    pub postprocess: Duration,
    /// Lightbar corner refinement.
    pub refine: Duration,
}

/// One ONNX Runtime session plus pre/post-processing.
#[derive(Debug)]
pub struct Detector {
    session: Session,
    backend: Backend,
    cfg: DetectorConfig,
    times: StageTimes,
}

impl Detector {
    /// Loads the model on the first available backend.
    ///
    /// # Errors
    /// Model missing or invalid, or `require_backend` set and the first backend unavailable.
    pub fn new(cfg: &DetectorConfig) -> color_eyre::Result<Self> {
        let backend = cfg.backends.iter().copied().find(|b| b.is_available()).unwrap_or(Backend::Cpu);
        let first = cfg.backends.first().copied().unwrap_or(Backend::Cpu);
        if backend == first {
            tracing::info!(active = ?backend, "detector backend");
        } else {
            if cfg.require_backend {
                bail!("required backend {first:?} is unavailable");
            }
            tracing::warn!(requested = ?first, active = ?backend, "detector backend fell back");
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
        Ok(Self { session, backend, cfg: cfg.clone(), times: StageTimes::default() })
    }

    /// Backend the session actually runs on.
    pub fn active_backend(&self) -> Backend {
        self.backend
    }

    /// Stage timings of the last [`Detector::detect`] call.
    pub fn last_times(&self) -> StageTimes {
        self.times
    }

    /// Detects `enemy` armor plates in an RGB image.
    ///
    /// # Errors
    /// Inference failed.
    pub fn detect(&mut self, image: &RgbImage, enemy: ArmorColor) -> color_eyre::Result<Vec<Detection>> {
        let start = Instant::now();
        let spec = &self.cfg.spec;
        let [in_w, in_h] = spec.input_size;
        let lb = Letterbox::new(image.width(), image.height(), in_w, in_h);
        let input = preprocess(image, spec)?;
        let shape = [1, 3, usize_of(in_h), usize_of(in_w)];
        let preprocessed = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![spec.input_name.as_str() => TensorRef::from_array_view((shape, input.as_slice())).map_err(ort_err)?])
            .map_err(ort_err)?;
        let inferred = Instant::now();
        let output = outputs.get(spec.output_name.as_str()).ok_or_else(|| eyre!("model has no output {:?}", spec.output_name))?;
        let (dims, raw) = output.try_extract_tensor::<f32>().map_err(ort_err)?;
        let rows_dim = if spec.transposed { dims.last() } else { dims.get(dims.len().saturating_sub(2)) };
        if rows_dim.and_then(|&d| usize::try_from(d).ok()) != Some(spec.rows()) {
            bail!("model output {dims:?} does not match the spec's {} values per candidate (transposed: {})", spec.rows(), spec.transposed);
        }
        let mut dets = decode(raw, spec, &lb, enemy, self.cfg.score_threshold);
        dets = nms(dets, self.cfg.nms_iou);
        let decoded = Instant::now();
        if self.cfg.refine_corners {
            dets = dets.into_iter().map(|d| super::refine::refine(image, &d)).collect();
        }
        self.times = StageTimes {
            preprocess: preprocessed - start,
            inference: inferred - preprocessed,
            postprocess: decoded - inferred,
            refine: decoded.elapsed(),
        };
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

#[expect(clippy::needless_pass_by_value, reason = "used with map_err which requires ownership")]
fn ort_err<R>(e: ort::Error<R>) -> color_eyre::Report {
    color_eyre::eyre::eyre!("onnxruntime: {e}")
}

/// Letterbox, normalize to `[0, 1]` and convert to the CHW model input, in one fused pass (kornia).
fn preprocess(image: &RgbImage, spec: &ModelSpec) -> color_eyre::Result<Tensor<f32, 4>> {
    let size = ImageSize { width: usize_of(image.width()), height: usize_of(image.height()) };
    let src = Image::<u8, 3>::from_size_slice(size, image.as_raw())?;
    let [w, h] = spec.input_size;
    let mut input = Tensor::from_shape_val([1, 3, usize_of(h), usize_of(w)], 0.0f32);
    Preprocessor::builder().mode(ResizeMode::Letterbox).pad_value(spec.pad).build()?.run(&src, &mut input)?;
    Ok(input)
}

/// Where the letterbox puts the image inside the model input; maps keypoints back to the source.
#[derive(Debug, Clone, Copy)]
struct Letterbox {
    scale: f64,
    pad_x: u32,
    pad_y: u32,
    src_w: u32,
    src_h: u32,
}

impl Letterbox {
    fn new(src_w: u32, src_h: u32, in_w: u32, in_h: u32) -> Self {
        let scale = (f64::from(in_h) / f64::from(src_h)).min(f64::from(in_w) / f64::from(src_w));
        let w = round_px(f64::from(src_w) * scale).min(in_w);
        let h = round_px(f64::from(src_h) * scale).min(in_h);
        Self { scale, pad_x: (in_w - w) / 2, pad_y: (in_h - h) / 2, src_w, src_h }
    }

    fn to_source(self, x: f32, y: f32) -> Option<Point2<f64>> {
        let p = Point2::new(
            (f64::from(x) - f64::from(self.pad_x)) / self.scale,
            (f64::from(y) - f64::from(self.pad_y)) / self.scale,
        );
        let inside = (0.0..=f64::from(self.src_w)).contains(&p.x) && (0.0..=f64::from(self.src_h)).contains(&p.y);
        inside.then_some(p)
    }
}

/// Candidates above `threshold` in the `enemy` color, keypoints mapped back to source pixels.
fn decode(raw: &[f32], spec: &ModelSpec, lb: &Letterbox, enemy: ArmorColor, threshold: f32) -> Vec<Detection> {
    let rows = spec.rows();
    let n = raw.len() / rows;
    let at = |row: usize, i: usize| if spec.transposed { raw[i * rows + row] } else { raw[row * n + i] };
    let argmax = |rows: std::ops::Range<usize>, i: usize| {
        rows.clone().max_by(|&a, &b| at(a, i).total_cmp(&at(b, i))).map_or(0, |r| r - rows.start)
    };
    let (n_colors, n_icons) = (spec.colors.len(), spec.icons.len());
    let keypoints = n_colors + n_icons;
    (0..n)
        .filter_map(|i| {
            let class = argmax(n_colors..keypoints, i);
            let score = at(n_colors + class, i);
            let confidence = if spec.logits { 1.0 / (1.0 + (-score).exp()) } else { score };
            if confidence < threshold {
                return None;
            }
            let color = spec.colors[argmax(0..n_colors, i)];
            if color != enemy {
                return None;
            }
            let mut corners = [Point2::origin(); 4];
            for (corner, &k) in corners.iter_mut().zip(&spec.corners) {
                *corner = lb.to_source(at(keypoints + 2 * k, i), at(keypoints + 2 * k + 1, i))?;
            }
            Some(Detection { color, icon: spec.icons[class], confidence, corners })
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
    use crate::detector::Icon;

    fn spec(logits: bool, transposed: bool, corners: [usize; 4]) -> ModelSpec {
        ModelSpec {
            input_name: "images".into(),
            output_name: "output0".into(),
            input_size: [640, 480],
            pad: 124,
            colors: vec![ArmorColor::Blue, ArmorColor::Red],
            icons: vec![Icon::Hero, Icon::Infantry3],
            logits,
            transposed,
            corners,
        }
    }

    /// Two candidates: a red Infantry3 and a blue Hero. Keypoint `k` is stored at `(10k, 100 + k)`.
    fn candidates(score: f32) -> [[f32; 12]; 2] {
        let keypoints = |base: f32| [base, 100.0, base + 10.0, 101.0, base + 20.0, 102.0, base + 30.0, 103.0];
        let mut red = [0.1, 0.9, 0.2, score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        red[4..].copy_from_slice(&keypoints(0.0));
        let mut blue = [0.9, 0.1, score, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        blue[4..].copy_from_slice(&keypoints(200.0));
        [red, blue]
    }

    /// The same candidates in every layout the spec describes decode to the same detection.
    #[test]
    fn decodes_any_described_layout() {
        let lb = Letterbox::new(640, 480, 640, 480);
        let decode_with = |s: &ModelSpec, score: f32| {
            let c = candidates(score);
            let raw: Vec<f32> = if s.transposed { c.iter().flatten().copied().collect() } else { (0..12).flat_map(|row| c.iter().map(move |cand| cand[row])).collect() };
            decode(&raw, s, &lb, ArmorColor::Red, 0.5)
        };
        let plain = decode_with(&spec(false, false, [0, 1, 2, 3]), 0.8);
        assert_eq!(plain.len(), 1, "only the red candidate");
        assert_eq!((plain[0].icon, plain[0].color), (Icon::Infantry3, ArmorColor::Red), "class and color from the spec's tables");
        assert_eq!(plain[0].corners[1], Point2::new(10.0, 101.0), "keypoint 1 is bottom-left");
        assert_eq!(decode_with(&spec(false, true, [0, 1, 2, 3]), 0.8), plain, "transposed output");
        let logit = decode_with(&spec(true, false, [0, 1, 2, 3]), 1.386_294_4);
        assert!((logit[0].confidence - 0.8).abs() < 1e-6 && logit[0].corners == plain[0].corners, "sigmoid applied to logits");
        let swapped = decode_with(&spec(false, false, [1, 0, 3, 2]), 0.8);
        assert_eq!(swapped[0].corners[0], plain[0].corners[1], "corner order follows the spec");
    }

    #[test]
    fn letterbox_matches_szurp_for_960x540() {
        let lb = Letterbox::new(960, 540, 640, 480);
        assert_eq!((lb.pad_x, lb.pad_y), (0, 60),"scale 2/3, 60 px bars top and bottom");
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
