//! Detector on the real RealSense recordings with the SZURP model.

use std::path::Path;

use auto_aim::config::{GeneralConfig, load};
use auto_aim::detector::pool::DetectorPool;
use auto_aim::detector::{ArmorColor, Backend, Detector, Icon};
use auto_aim::io::video::VideoSource;

fn config() -> GeneralConfig {
    load(Path::new("configs/general.yaml")).expect("general config parses")
}

fn hit_rate(video: &str, frames: u64) -> f64 {
    let mut src = VideoSource::open(Path::new(video)).expect("video opens");
    let mut pool = DetectorPool::new(&config().detector, ArmorColor::Red).expect("pool starts");
    let mut hits = 0u32;
    for _ in 0..frames {
        let frame = src.next_frame().expect("decodes").expect("video long enough");
        pool.submit(frame).expect("submit");
        if pool.in_flight() > 4 {
            let f = pool.recv().expect("frame back");
            hits += u32::from(f.detections.iter().any(|d| d.icon == Icon::Infantry3 && d.confidence > 0.7));
        }
    }
    while let Some(f) = pool.recv() {
        hits += u32::from(f.detections.iter().any(|d| d.icon == Icon::Infantry3 && d.confidence > 0.7));
    }
    f64::from(hits) / f64::from(u32::try_from(frames).expect("small"))
}

#[test]
fn falls_back_to_cpu_without_gpu() {
    let d = Detector::new(&config().detector).expect("loads on cpu");
    assert_eq!(d.active_backend(), Backend::Cpu, "laptop has no CUDA/TensorRT");
}

#[test]
fn required_gpu_backend_fails_loudly() {
    let mut cfg = config().detector;
    cfg.require_backend = true;
    assert!(Detector::new(&cfg).is_err(), "require_backend refuses to fall back");
}

#[test]
fn finds_the_red_infantry_at_3m_and_6m() {
    let near = hit_rate("assets/videos/static_3m.mp4", 120);
    let far = hit_rate("assets/videos/static_6m.mp4", 120);
    assert!(near >= 0.95, "3 m hit rate {near}");
    assert!(far >= 0.90, "6 m hit rate {far}");
}

#[test]
fn pool_returns_frames_in_order() {
    let mut src = VideoSource::open(Path::new("assets/videos/spinning_3m.mp4")).expect("video opens");
    let mut pool = DetectorPool::new(&config().detector, ArmorColor::Red).expect("pool starts");
    let mut ids = Vec::new();
    for _ in 0..40 {
        pool.submit(src.next_frame().expect("decodes").expect("frame")).expect("submit");
        if pool.in_flight() > 5 {
            ids.push(pool.recv().expect("frame").frame.frame_id);
        }
    }
    while let Some(f) = pool.recv() {
        ids.push(f.frame.frame_id);
    }
    assert_eq!(ids, (0..40).collect::<Vec<_>>(), "strict frame order");
}
