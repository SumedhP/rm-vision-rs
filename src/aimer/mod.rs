//! Aimer: robot estimate + robot state -> gimbal command.

pub mod aimer;
pub mod ballistics;
pub mod trajectory;

pub use aimer::{AimTarget, Aimer};
pub use ballistics::Shot;

use serde::Deserialize;

/// What the host sends to the MCU each control tick.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize)]
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
pub struct AimerConfig {
    /// Quadratic drag `k` in `a = -k |v| v`, 1/m. 0.019 for a 17 mm, 3.2 g ball (Cd 0.47, air 1.169 kg/m^3).
    pub drag: f64,
    /// Command-to-muzzle delay (trigger, feed), seconds.
    pub shoot_delay: f64,
    /// Keep the current plate until it turns further than this from the line of sight, radians.
    pub switch_angle: f64,
    /// Never fire at a plate turned further than this, radians.
    pub max_fire_angle: f64,
    /// Bullet speed used when the reported one is outside 10..30 m/s.
    pub default_bullet_speed: f64,
    /// Setpoint smoother bandwidth, rad/s.
    pub bandwidth: f64,
}

/// Gimbal capability, per robot.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GimbalLimits {
    /// Max yaw acceleration, rad/s^2.
    pub max_yaw_acc: f64,
    /// Max pitch acceleration, rad/s^2.
    pub max_pitch_acc: f64,
}
