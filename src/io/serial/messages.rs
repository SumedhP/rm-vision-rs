//! Typed messages carried in serial frames.
//!
//! A message is any type implementing [`Message`]: a command id plus encode/decode of its body.
//! To add one, write the struct and its impl here; nothing else needs to know about it. Receivers
//! call [`super::Frame::parse`] for the ids they understand and can ignore or log the rest, so the
//! MCU may also forward referee frames (ids `0x0001..=0x0305`) unchanged.
//!
//! Our ids start at `0x0A00` to stay clear of the referee range. Bodies are little-endian, floats
//! are `f32`.

use nalgebra::{Quaternion, UnitQuaternion};

use crate::aimer::AimCommand;
use crate::io::RobotState;

/// A typed frame body.
pub trait Message: Sized {
    /// Frame `cmd_id`.
    const CMD_ID: u16;
    /// Appends the body.
    fn encode(&self, out: &mut Vec<u8>);
    /// Parses a body, `None` if malformed.
    fn decode(body: &[u8]) -> Option<Self>;
}

/// MCU -> host: gimbal orientation and shooter state, sent at the IMU rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GimbalState {
    /// World-from-gimbal orientation `[w, x, y, z]`.
    pub q: [f32; 4],
    /// Last measured bullet speed, m/s.
    pub bullet_speed: f32,
    /// Own robot id (referee numbering; above 100 is blue).
    pub robot_id: u8,
}

impl GimbalState {
    /// Pipeline robot state, given the sample time on the pipeline clock (arrival minus `imu_delay`).
    pub fn to_robot_state(&self, t: f64) -> RobotState {
        let q = self.q.map(f64::from);
        RobotState { t, q: UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3])), bullet_speed: f64::from(self.bullet_speed), robot_id: self.robot_id }
    }
}

impl Message for GimbalState {
    const CMD_ID: u16 = 0x0A01;

    fn encode(&self, out: &mut Vec<u8>) {
        for v in self.q.iter().chain([&self.bullet_speed]) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(self.robot_id);
    }

    fn decode(body: &[u8]) -> Option<Self> {
        let mut r = Reader(body);
        let msg = Self { q: [r.f32()?, r.f32()?, r.f32()?, r.f32()?], bullet_speed: r.f32()?, robot_id: r.u8()? };
        r.end().then_some(msg)
    }
}

/// Host -> MCU: gimbal setpoint with feedforward, and whether to fire.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GimbalCommand {
    /// Yaw, radians, world frame.
    pub yaw: f32,
    /// Pitch, radians, positive up.
    pub pitch: f32,
    /// Yaw rate, rad/s.
    pub yaw_vel: f32,
    /// Pitch rate, rad/s.
    pub pitch_vel: f32,
    /// Yaw acceleration, rad/s^2.
    pub yaw_acc: f32,
    /// Pitch acceleration, rad/s^2.
    pub pitch_acc: f32,
    /// Fire now.
    pub fire: bool,
}

impl From<&AimCommand> for GimbalCommand {
    fn from(c: &AimCommand) -> Self {
        Self {
            yaw: narrow(c.yaw),
            pitch: narrow(c.pitch),
            yaw_vel: narrow(c.yaw_vel),
            pitch_vel: narrow(c.pitch_vel),
            yaw_acc: narrow(c.yaw_acc),
            pitch_acc: narrow(c.pitch_acc),
            fire: c.fire,
        }
    }
}

impl Message for GimbalCommand {
    const CMD_ID: u16 = 0x0A81;

    fn encode(&self, out: &mut Vec<u8>) {
        for v in [self.yaw, self.pitch, self.yaw_vel, self.pitch_vel, self.yaw_acc, self.pitch_acc] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(u8::from(self.fire));
    }

    fn decode(body: &[u8]) -> Option<Self> {
        let mut r = Reader(body);
        let msg = Self { yaw: r.f32()?, pitch: r.f32()?, yaw_vel: r.f32()?, pitch_vel: r.f32()?, yaw_acc: r.f32()?, pitch_acc: r.f32()?, fire: r.u8()? != 0 };
        r.end().then_some(msg)
    }
}

#[expect(clippy::cast_possible_truncation, clippy::as_conversions, reason = "the wire format is f32")]
fn narrow(v: f64) -> f32 {
    v as f32
}

/// Little-endian cursor over a body.
struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }

    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|[b]| b)
    }

    fn f32(&mut self) -> Option<f32> {
        self.take().map(f32::from_le_bytes)
    }

    fn end(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gimbal_state_round_trips_and_rejects_wrong_lengths() {
        let msg = GimbalState { q: [1.0, 0.0, 0.0, 0.0], bullet_speed: 23.5, robot_id: 103 };
        let mut body = Vec::new();
        msg.encode(&mut body);
        assert_eq!(body.len(), 21, "4 + 1 floats and a byte");
        assert_eq!(GimbalState::decode(&body), Some(msg), "round trip");
        assert_eq!(GimbalState::decode(&body[..20]), None, "short body");
        assert_eq!(GimbalState::decode(&[body.as_slice(), &[0]].concat()), None, "long body");
    }

    #[test]
    fn command_carries_the_aimer_output() {
        let aim = AimCommand { t: 1.0, yaw: 0.25, pitch: -0.05, yaw_vel: 1.5, pitch_vel: 0.0, yaw_acc: -30.0, pitch_acc: 2.0, fire: true };
        let msg = GimbalCommand::from(&aim);
        let mut body = Vec::new();
        msg.encode(&mut body);
        assert_eq!(GimbalCommand::decode(&body), Some(msg), "round trip");
        assert!(msg.fire && (msg.yaw_acc + 30.0).abs() < 1e-6, "fields copied");
    }
}
