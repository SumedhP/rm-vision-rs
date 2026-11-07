//! The serial link: one thread owns the port, writes queued frames and decodes incoming ones.
//!
//! Generic over any `Read + Write` transport, so the same code runs on a real port
//! ([`SerialLink::open`]) and on an in-memory pipe in tests. The port is read with a short timeout;
//! queued frames go out between reads, so a send waits at most that timeout.

use std::io::{ErrorKind, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, unbounded};

use super::frame::{self, DecodeStats, Decoder, Frame};
use super::messages::Message;

/// How long a read blocks before the thread checks for frames to send.
pub const POLL: Duration = Duration::from_millis(1);

/// A frame and when its last byte was read.
#[derive(Debug, Clone)]
pub struct Received {
    /// Arrival time (host monotonic clock).
    pub at: Instant,
    /// The frame.
    pub frame: Frame,
}

impl Frame {
    /// The typed message, if this frame carries `M`.
    pub fn parse<M: Message>(&self) -> Option<M> {
        (self.cmd_id == M::CMD_ID).then(|| M::decode(&self.payload)).flatten()
    }
}

/// Handle to the link thread. Dropping it stops the thread.
#[derive(Debug)]
pub struct SerialLink {
    outgoing: Sender<Vec<u8>>,
    incoming: Receiver<Received>,
    seq: u8,
    stats: Arc<Mutex<DecodeStats>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<std::io::Result<()>>>,
}

impl SerialLink {
    /// Opens a serial port (8N1, no flow control).
    ///
    /// # Errors
    /// The port can't be opened.
    pub fn open(path: &str, baud: u32) -> color_eyre::Result<Self> {
        let port = serialport::new(path, baud).timeout(POLL).open()?;
        Ok(Self::spawn(port))
    }

    /// Runs the link over any byte transport. Reads must time out (`ErrorKind::TimedOut` or
    /// `WouldBlock`) rather than block forever, so the thread can send and notice shutdown.
    pub fn spawn<T: Read + Write + Send + 'static>(mut port: T) -> Self {
        let (outgoing, to_send) = unbounded::<Vec<u8>>();
        let (deliver, incoming) = unbounded();
        let stats = Arc::new(Mutex::new(DecodeStats::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (thread_stats, thread_stop) = (Arc::clone(&stats), Arc::clone(&stop));
        let thread = std::thread::spawn(move || -> std::io::Result<()> {
            let mut decoder = Decoder::new();
            let mut buf = [0u8; 512];
            while !thread_stop.load(Ordering::Relaxed) {
                for bytes in to_send.try_iter() {
                    port.write_all(&bytes)?;
                }
                port.flush()?;
                match port.read(&mut buf) {
                    Ok(n) => {
                        let at = Instant::now();
                        for frame in decoder.push(&buf[..n]) {
                            if deliver.send(Received { at, frame }).is_err() {
                                return Ok(());
                            }
                        }
                        *thread_stats.lock().unwrap_or_else(PoisonError::into_inner) = decoder.stats();
                    }
                    Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::Interrupted) => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        });
        Self { outgoing, incoming, seq: 0, stats, stop, thread: Some(thread) }
    }

    /// Queues `msg` to be written.
    ///
    /// # Errors
    /// The link thread has stopped (port error or disconnect).
    pub fn send<M: Message>(&mut self, msg: &M) -> color_eyre::Result<()> {
        let mut body = Vec::new();
        msg.encode(&mut body);
        self.send_raw(M::CMD_ID, &body)
    }

    /// Queues an already encoded body under `cmd_id`, for messages without a type yet.
    ///
    /// # Errors
    /// The link thread has stopped (port error or disconnect).
    pub fn send_raw(&mut self, cmd_id: u16, body: &[u8]) -> color_eyre::Result<()> {
        let bytes = frame::encode(cmd_id, self.seq, body);
        self.seq = self.seq.wrapping_add(1);
        self.outgoing.send(bytes).map_err(|_| self.stopped())
    }

    /// Frames received so far, in arrival order.
    pub fn incoming(&self) -> &Receiver<Received> {
        &self.incoming
    }

    /// Decoder counters (frames, skipped bytes, rejected headers and frames).
    pub fn stats(&self) -> DecodeStats {
        *self.stats.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn stopped(&mut self) -> color_eyre::Report {
        if let Some(Ok(Err(e))) = self.thread.take().map(JoinHandle::join) {
            color_eyre::eyre::eyre!("serial link stopped: {e}")
        } else {
            color_eyre::eyre::eyre!("serial link stopped")
        }
    }
}

impl Drop for SerialLink {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
