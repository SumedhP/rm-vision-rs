//! DJI `RoboMaster` serial frame (the referee system protocol), used for every message on the link.
//!
//! ```text
//! | SOF 0xA5 | data_length u16 | seq u8 | CRC8 | cmd_id u16 | data[data_length] | CRC16 |
//! ```
//! Little-endian. CRC8 covers the first 4 bytes (init `0xFF`, reflected polynomial `0x31`). CRC16
//! covers everything before it (init `0xFFFF`, reflected polynomial `0x1021`, i.e. CRC-16/MCRF4XX).

/// Start of frame.
pub const SOF: u8 = 0xA5;
const HEADER: usize = 5;
const OVERHEAD: usize = HEADER + 2 + 2;
/// Largest payload accepted. Referee frames stay well under this, so a longer `data_length` means a
/// corrupt header that happened to pass CRC8.
pub const MAX_PAYLOAD: usize = 512;

const CRC8_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        #[expect(clippy::cast_possible_truncation, clippy::as_conversions, reason = "i < 256")]
        let mut crc = i as u8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0x8C } else { crc >> 1 };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

const CRC16_TABLE: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        #[expect(clippy::cast_possible_truncation, clippy::as_conversions, reason = "i < 256")]
        let mut crc = i as u16;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0x8408 } else { crc >> 1 };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Header checksum.
pub fn crc8(data: &[u8]) -> u8 {
    data.iter().fold(0xFF, |crc, &b| CRC8_TABLE[usize::from(crc ^ b)])
}

/// Frame checksum.
pub fn crc16(data: &[u8]) -> u16 {
    data.iter().fold(0xFFFF, |crc, &b| (crc >> 8) ^ CRC16_TABLE[usize::from(crc.to_le_bytes()[0] ^ b)])
}

/// One decoded frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Message id.
    pub cmd_id: u16,
    /// Sender's sequence number.
    pub seq: u8,
    /// Message body.
    pub payload: Vec<u8>,
}

/// Serializes one frame.
///
/// # Panics
/// Panics if `payload` is longer than [`MAX_PAYLOAD`].
pub fn encode(cmd_id: u16, seq: u8, payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() <= MAX_PAYLOAD, "payload of {} bytes exceeds {MAX_PAYLOAD}", payload.len());
    let len = u16::try_from(payload.len()).expect("checked above");
    let mut out = Vec::with_capacity(OVERHEAD + payload.len());
    out.push(SOF);
    out.extend_from_slice(&len.to_le_bytes());
    out.push(seq);
    out.push(crc8(&out));
    out.extend_from_slice(&cmd_id.to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&crc16(&out).to_le_bytes());
    out
}

/// Counts of what the decoder threw away.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecodeStats {
    /// Frames delivered.
    pub frames: u64,
    /// Bytes skipped while looking for a start of frame.
    pub skipped_bytes: u64,
    /// Headers rejected (CRC8 mismatch or impossible length).
    pub bad_headers: u64,
    /// Whole frames rejected by CRC16.
    pub bad_frames: u64,
}

/// Streaming decoder: feed it bytes as they arrive, in any chunking, and get whole frames back.
/// On a corrupt header or frame it skips one byte and searches for the next start of frame, so a
/// bad byte costs at most the frame it lands in.
#[derive(Debug, Default)]
pub struct Decoder {
    buf: Vec<u8>,
    stats: DecodeStats,
}

impl Decoder {
    /// Empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// What has been delivered and discarded so far.
    pub fn stats(&self) -> DecodeStats {
        self.stats
    }

    /// Appends `bytes` and returns every frame now complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Frame> {
        self.buf.extend_from_slice(bytes);
        let mut frames = Vec::new();
        let mut start = 0;
        loop {
            let Some(offset) = self.buf[start..].iter().position(|&b| b == SOF) else {
                self.stats.skipped_bytes += u64::try_from(self.buf.len() - start).unwrap_or(0);
                start = self.buf.len();
                break;
            };
            self.stats.skipped_bytes += u64::try_from(offset).unwrap_or(0);
            start += offset;
            let rest = &self.buf[start..];
            if rest.len() < HEADER {
                break;
            }
            let len = usize::from(u16::from_le_bytes([rest[1], rest[2]]));
            if crc8(&rest[..4]) != rest[4] || len > MAX_PAYLOAD {
                self.stats.bad_headers += 1;
                start += 1;
                continue;
            }
            let total = OVERHEAD + len;
            if rest.len() < total {
                break;
            }
            if crc16(&rest[..total - 2]) != u16::from_le_bytes([rest[total - 2], rest[total - 1]]) {
                self.stats.bad_frames += 1;
                start += 1;
                continue;
            }
            frames.push(Frame { cmd_id: u16::from_le_bytes([rest[5], rest[6]]), seq: rest[3], payload: rest[7..total - 2].to_vec() });
            self.stats.frames += 1;
            start += total;
        }
        self.buf.drain(..start);
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entries from the DJI tables as copied in awakening's `radar_io/crc.cpp`.
    #[test]
    fn tables_match_the_dji_reference() {
        assert_eq!([CRC8_TABLE[1], CRC8_TABLE[2], CRC8_TABLE[128], CRC8_TABLE[255]], [0x5e, 0xbc, 0x8c, 0x35], "CRC8 table");
        assert_eq!([CRC16_TABLE[1], CRC16_TABLE[2], CRC16_TABLE[128], CRC16_TABLE[255]], [0x1189, 0x2312, 0x8408, 0x0f78], "CRC16 table");
    }

    /// CRC-16/MCRF4XX catalogue check value.
    #[test]
    fn crc16_check_value() {
        assert_eq!(crc16(b"123456789"), 0x6F91, "MCRF4XX check");
    }

    #[test]
    fn round_trips_through_noise_and_any_chunking() {
        let a = encode(0x0A01, 7, &[1, 2, 3]);
        let b = encode(0x0301, 8, &[]);
        let mut stream = vec![0x00, SOF, 0x13, 0xA5];
        stream.extend_from_slice(&a);
        stream.extend_from_slice(&[0xA5, 0xA5]);
        stream.extend_from_slice(&b);
        for chunk in [1, 2, 3, 5, 64] {
            let mut d = Decoder::new();
            let frames: Vec<Frame> = stream.chunks(chunk).flat_map(|c| d.push(c)).collect();
            assert_eq!(frames, vec![Frame { cmd_id: 0x0A01, seq: 7, payload: vec![1, 2, 3] }, Frame { cmd_id: 0x0301, seq: 8, payload: vec![] }], "chunk size {chunk}");
        }
    }

    #[test]
    fn a_corrupted_frame_is_dropped_and_the_next_one_survives() {
        let mut bad = encode(0x0A01, 1, &[9; 20]);
        bad[10] ^= 0xFF;
        let good = encode(0x0A01, 2, &[4; 20]);
        let mut d = Decoder::new();
        let frames = d.push(&[bad, good].concat());
        assert_eq!(frames.len(), 1, "only the intact frame");
        assert_eq!(frames[0].seq, 2, "it is the second frame");
        assert_eq!(d.stats().bad_frames, 1, "the corrupt one is counted");
    }
}
