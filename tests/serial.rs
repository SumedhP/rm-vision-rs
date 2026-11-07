//! Serial link end to end over an in-memory null-modem cable: two links talking to each other,
//! and a link fed a noisy hand-built byte stream.

use std::io::{ErrorKind, Read, Write};
use std::time::Duration;

use auto_aim::io::serial::frame::{self, SOF};
use auto_aim::io::serial::{GimbalCommand, GimbalState, Message, Received, SerialLink};
use crossbeam_channel::{Receiver, Sender, unbounded};

/// One end of a byte pipe whose reads time out like a serial port.
struct End {
    rx: Receiver<Vec<u8>>,
    tx: Sender<Vec<u8>>,
    pending: Vec<u8>,
}

impl Read for End {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pending.is_empty() {
            match self.rx.recv_timeout(Duration::from_millis(1)) {
                Ok(bytes) => self.pending = bytes,
                Err(_) => return Err(ErrorKind::TimedOut.into()),
            }
        }
        let n = buf.len().min(self.pending.len());
        buf[..n].copy_from_slice(&self.pending[..n]);
        self.pending.drain(..n);
        Ok(n)
    }
}

impl Write for End {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Deliver in small pieces, as a UART would, so frames arrive split across reads.
        for chunk in buf.chunks(7) {
            self.tx.send(chunk.to_vec()).map_err(|_| std::io::Error::from(ErrorKind::BrokenPipe))?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn cable() -> (End, End) {
    let (a_tx, b_rx) = unbounded();
    let (b_tx, a_rx) = unbounded();
    (End { rx: a_rx, tx: a_tx, pending: Vec::new() }, End { rx: b_rx, tx: b_tx, pending: Vec::new() })
}

fn collect(link: &SerialLink, n: usize) -> Vec<Received> {
    (0..n).map_while(|_| link.incoming().recv_timeout(Duration::from_secs(2)).ok()).collect()
}

#[test]
fn host_and_mcu_exchange_typed_messages_both_ways() {
    let (host_end, mcu_end) = cable();
    let mut host = SerialLink::spawn(host_end);
    let mut mcu = SerialLink::spawn(mcu_end);

    let commands: Vec<GimbalCommand> = (0..300u16)
        .map(|i| {
            let x = f32::from(i) * 0.01;
            GimbalCommand { yaw: x, pitch: -x, yaw_vel: 1.0, pitch_vel: 0.5, yaw_acc: 10.0, pitch_acc: -3.0, fire: i % 2 == 0 }
        })
        .collect();
    for c in &commands {
        host.send(c).expect("host link running");
    }
    let state = GimbalState { q: [std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2], bullet_speed: 23.4, robot_id: 103 };
    mcu.send(&state).expect("mcu link running");

    let got = collect(&mcu, commands.len());
    assert_eq!(got.len(), commands.len(), "every command arrives");
    for (i, (r, sent)) in got.iter().zip(&commands).enumerate() {
        assert_eq!(r.frame.seq, u8::try_from(i % 256).expect("fits"), "sequence numbers count up and wrap");
        assert_eq!(r.frame.parse::<GimbalCommand>().as_ref(), Some(sent), "command {i} intact and in order");
    }

    let back = collect(&host, 1);
    let received = back[0].frame.parse::<GimbalState>().expect("a gimbal state");
    assert_eq!(received, state, "state intact");
    let robot = received.to_robot_state(1.5);
    assert!((robot.q.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-3 && robot.robot_id == 103, "converts to the pipeline's robot state");
    assert_eq!(mcu.stats().bad_frames + mcu.stats().bad_headers, 0, "clean wire, no rejects");
}

#[test]
fn noise_and_corruption_cost_only_the_frames_they_hit() {
    let (mut wire, link_end) = cable();
    let link = SerialLink::spawn(link_end);

    let sent = GimbalState { q: [1.0, 0.0, 0.0, 0.0], bullet_speed: 22.0, robot_id: 3 };
    let mut body = Vec::new();
    sent.encode(&mut body);
    let mut corrupted = frame::encode(GimbalState::CMD_ID, 1, &body);
    corrupted[9] ^= 0x40;
    let referee_game_status = frame::encode(0x0001, 2, &[0x14, 0x2c, 0x01, 0, 0, 0, 0, 0, 0, 0, 0]);

    let mut stream = vec![0xFF, SOF, 0x00, SOF];
    stream.extend_from_slice(&corrupted);
    stream.extend_from_slice(&frame::encode(GimbalState::CMD_ID, 3, &body));
    stream.extend_from_slice(&[0x55; 9]);
    stream.extend_from_slice(&referee_game_status);
    wire.write_all(&stream).expect("pipe open");

    let got = collect(&link, 2);
    assert_eq!(got.len(), 2, "the intact state and the referee frame");
    assert_eq!(got[0].frame.parse::<GimbalState>(), Some(sent), "state after the corrupted copy survives");
    assert_eq!(got[0].frame.seq, 3, "and it is the intact copy");
    assert_eq!((got[1].frame.cmd_id, got[1].frame.parse::<GimbalState>()), (0x0001, None), "unknown ids pass through untyped");
    assert!(link.incoming().recv_timeout(Duration::from_millis(50)).is_err(), "nothing else");
    let stats = link.stats();
    assert!(stats.bad_frames + stats.bad_headers >= 1 && stats.skipped_bytes > 0, "rejects are counted: {stats:?}");
}
