//! Aimer: robot estimate + robot state -> gimbal command.

use serde::Deserialize;

/// What the host sends to the MCU each control tick.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AimCommand {
    /// Time the command is for (`now + downlink_delay`), seconds.
    pub t: f64,
    /// Yaw setpoint, radians, world frame.
    pub yaw: f64,
    /// Pitch setpoint, radians, positive up.
    pub pitch: f64,
    /// Yaw rate feedforward, rad/s.
    pub yaw_vel: f64,
    /// Pitch rate feedforward, rad/s.
    pub pitch_vel: f64,
    /// Yaw acceleration feedforward, rad/s^2.
    pub yaw_acc: f64,
    /// Pitch acceleration feedforward, rad/s^2.
    pub pitch_acc: f64,
    /// Whether to fire now.
    pub fire: bool,
}

/// Aimer settings (`aimer:` section of `configs/general.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AimerConfig {}

/// Gimbal capability, per robot.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GimbalLimits {
    /// Max yaw acceleration, rad/s^2.
    pub max_yaw_acc: f64,
    /// Max pitch acceleration, rad/s^2.
    pub max_pitch_acc: f64,
}
