//! Detector worker pool. Frames go out to N workers and come back strictly in `frame_id` order,
//! because the tracker's filter breaks on out-of-order frames.

use std::collections::{BTreeMap, VecDeque};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};

use super::{ArmorColor, DetectedFrame, Detector, DetectorConfig};
use crate::io::CameraFrame;

/// N detector sessions behind an ordered queue.
#[derive(Debug)]
pub struct DetectorPool {
    tx: Option<Sender<CameraFrame>>,
    rx: Receiver<DetectedFrame>,
    workers: Vec<JoinHandle<()>>,
    pending: BTreeMap<u64, DetectedFrame>,
    submitted: VecDeque<u64>,
}

impl DetectorPool {
    /// Starts `cfg.workers` threads, each with its own ONNX Runtime session.
    ///
    /// # Errors
    /// A detector failed to load.
    pub fn new(cfg: &DetectorConfig, enemy: ArmorColor) -> color_eyre::Result<Self> {
        let n = cfg.workers.max(1);
        let (tx, work) = bounded::<CameraFrame>(n * 2);
        let (done, rx) = unbounded::<DetectedFrame>();
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            let mut detector = Detector::new(cfg)?;
            let (work, done) = (work.clone(), done.clone());
            workers.push(std::thread::spawn(move || {
                for frame in work {
                    let detections = detector.detect(&frame.image, enemy).unwrap_or_else(|e| {
                        tracing::warn!(frame_id = frame.frame_id, "detection failed: {e}");
                        Vec::new()
                    });
                    if done.send(DetectedFrame { frame, detections }).is_err() {
                        return;
                    }
                }
            }));
        }
        Ok(Self { tx: Some(tx), rx, workers, pending: BTreeMap::new(), submitted: VecDeque::new() })
    }

    /// Number of frames submitted but not yet returned by [`DetectorPool::recv`].
    pub fn in_flight(&self) -> usize {
        self.submitted.len()
    }

    /// Queues a frame. Blocks when all workers are busy and the queue is full.
    ///
    /// # Errors
    /// The pool has shut down.
    pub fn submit(&mut self, frame: CameraFrame) -> color_eyre::Result<()> {
        let id = frame.frame_id;
        self.tx.as_ref().ok_or_else(|| color_eyre::eyre::eyre!("pool closed"))?.send(frame)?;
        self.submitted.push_back(id);
        Ok(())
    }

    /// Oldest submitted frame with its detections. Blocks until it is ready; `None` when nothing is
    /// in flight. Frames must be submitted in increasing `frame_id` order.
    pub fn recv(&mut self) -> Option<DetectedFrame> {
        let want = self.submitted.pop_front()?;
        while !self.pending.contains_key(&want) {
            let done = self.rx.recv().ok()?;
            self.pending.insert(done.frame.frame_id, done);
        }
        self.pending.remove(&want)
    }
}

impl Drop for DetectorPool {
    fn drop(&mut self) {
        self.tx = None;
        self.rx = crossbeam_channel::never();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
