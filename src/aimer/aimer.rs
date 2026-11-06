//! Turns the robot estimate into a smoothed gimbal command and a fire decision.

use nalgebra::Vector3;

use super::ballistics::{self, Shot};
use super::trajectory::Axis;
use super::{AimCommand, AimerConfig, GimbalLimits};
use crate::geometry::LIGHTBAR_LENGTH;
use crate::tracker::model::wrap_angle;
use crate::tracker::{RobotEstimate, TrackMode, TrackState};

/// What the aimer is pointing at this tick (for logging and overlays).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct AimTarget {
    /// Plate id being aimed at.
    pub plate: usize,
    /// Predicted plate center at impact, world frame.
    pub position: Vector3<f64>,
    /// Angle between the plate normal and the line of sight at impact, radians.
    pub facing: f64,
    /// Unsmoothed ballistic solution.
    pub shot: Shot,
    /// Predicted impact time.
    pub t_hit: f64,
    /// Angle between the shot and the plate center at impact, radians. Zero when aiming at the
    /// plate; nonzero when aiming at the robot's front and waiting for the plate to come around.
    pub miss: f64,
}

/// Stateful aimer: remembers the selected plate and the smoothed gimbal setpoint.
#[derive(Debug)]
pub struct Aimer {
    cfg: AimerConfig,
    limits: GimbalLimits,
    yaw: Option<Axis>,
    pitch: Axis,
    last_t: Option<f64>,
    plate: Option<usize>,
    target: Option<AimTarget>,
}

impl Aimer {
    /// New aimer at rest.
    pub fn new(cfg: AimerConfig, limits: GimbalLimits) -> Self {
        Self { cfg, limits, yaw: None, pitch: Axis::at(0.0), last_t: None, plate: None, target: None }
    }

    /// Target chosen on the last call to [`Aimer::update`].
    pub fn target(&self) -> Option<AimTarget> {
        self.target
    }

    /// One control tick at host time `now`.
    pub fn update(&mut self, now: f64, est: Option<&RobotEstimate>, bullet_speed: f64, downlink_delay: f64) -> AimCommand {
        let dt = self.last_t.map_or(0.005, |t| (now - t).clamp(1e-4, 0.05));
        self.last_t = Some(now);
        let speed = if (10.0..=30.0).contains(&bullet_speed) { bullet_speed } else { self.cfg.default_bullet_speed };
        let t_cmd = now + downlink_delay;

        self.target = est.and_then(|e| self.solve(e, t_cmd, speed));
        let Some(target) = self.target else {
            let (yaw, pitch) = (self.yaw.unwrap_or(Axis::at(0.0)), self.pitch);
            return AimCommand { t: t_cmd, yaw: yaw.pos, pitch: pitch.pos, ..AimCommand::default() };
        };

        let est = est.expect("target implies an estimate");
        let ahead = self.solve_for(est, t_cmd + 0.01, speed, target.plate);
        let rate = |a: f64, b: f64| if ahead.is_some() { wrap_angle(b - a) / 0.01 } else { 0.0 };
        let want_yaw = Axis { pos: target.shot.yaw, vel: rate(target.shot.yaw, ahead.map_or(0.0, |s| s.shot.yaw)), acc: 0.0 };
        let want_pitch = Axis { pos: target.shot.pitch, vel: rate(target.shot.pitch, ahead.map_or(0.0, |s| s.shot.pitch)), acc: 0.0 };

        let yaw = self.yaw.get_or_insert(Axis::at(want_yaw.pos));
        let unwrapped = Axis { pos: yaw.pos + wrap_angle(want_yaw.pos - yaw.pos), ..want_yaw };
        yaw.step(&unwrapped, dt, self.limits.max_yaw_acc, self.cfg.bandwidth);
        self.pitch.step(&want_pitch, dt, self.limits.max_pitch_acc, self.cfg.bandwidth);
        let yaw = *yaw;

        let distance = target.position.norm();
        let yaw_tol = 0.5 * est.icon.armor_size().width() * target.facing.cos() / distance;
        let pitch_tol = 0.5 * LIGHTBAR_LENGTH / distance;
        let fire = est.state == TrackState::Confirmed
            && target.facing < self.cfg.max_fire_angle
            && target.miss < yaw_tol
            && wrap_angle(yaw.pos - target.shot.yaw).abs() < yaw_tol
            && (self.pitch.pos - target.shot.pitch).abs() < pitch_tol;

        AimCommand {
            t: t_cmd,
            yaw: wrap_angle(yaw.pos),
            pitch: self.pitch.pos,
            yaw_vel: yaw.vel,
            pitch_vel: self.pitch.vel,
            yaw_acc: yaw.acc,
            pitch_acc: self.pitch.acc,
            fire,
        }
    }

    /// Picks the plate (with switching hysteresis) and iterates impact time with flight time.
    fn solve(&mut self, est: &RobotEstimate, t_cmd: f64, speed: f64) -> Option<AimTarget> {
        let first = self.solve_plate(est, t_cmd, speed, self.plate.unwrap_or(0))?;
        let plate = match est.mode {
            TrackMode::Plate => est.last_plate.map_or(0, |(_, p)| p.id),
            _ => {
                let keep = self.plate.filter(|_| first.facing < self.cfg.switch_angle);
                keep.unwrap_or_else(|| {
                    (0..est.plate_count)
                        .filter_map(|id| self.solve_plate(est, t_cmd, speed, id))
                        .min_by(|a, b| a.facing.total_cmp(&b.facing))
                        .map_or(0, |t| t.plate)
                })
            }
        };
        self.plate = Some(plate);
        self.solve_for(est, t_cmd, speed, plate)
    }

    /// Spinning too fast to follow each plate (the gimbal can't slew and settle before the next one
    /// comes around), so hold on the robot's front and fire as a plate crosses it.
    fn center_aim(&self, est: &RobotEstimate) -> bool {
        est.mode == TrackMode::Robot && est.x[crate::tracker::model::ix::W].abs() > self.cfg.center_aim_spin
    }

    fn solve_for(&self, est: &RobotEstimate, t_cmd: f64, speed: f64, plate: usize) -> Option<AimTarget> {
        if self.center_aim(est) { self.solve_center(est, t_cmd, speed) } else { self.solve_plate(est, t_cmd, speed, plate) }
    }

    fn solve_center(&self, est: &RobotEstimate, t_cmd: f64, speed: f64) -> Option<AimTarget> {
        use crate::tracker::model::ix;
        let mut flight = 0.0;
        let mut out = None;
        for _ in 0..3 {
            let t_hit = t_cmd + self.cfg.shoot_delay + flight;
            let x = est.state_at(t_hit);
            let center = Vector3::new(x[ix::CX], x[ix::CY], x[ix::CZ]);
            let toward_us = -center.xy().normalize();
            let facing_of = |p: &crate::tracker::PlatePose| wrap_angle(p.yaw - p.position.y.atan2(p.position.x)).abs();
            let plate = est.plates_at(t_hit).into_iter().min_by(|a, b| facing_of(a).total_cmp(&facing_of(b)))?;
            let front = center.xy() + toward_us * (x[ix::R] + if plate.id % 2 == 1 { x[ix::L] } else { 0.0 });
            let position = Vector3::new(front.x, front.y, plate.position.z);
            let shot = ballistics::solve(&position, speed, self.cfg.drag)?;
            flight = shot.flight_time;
            let miss = wrap_angle(plate.position.y.atan2(plate.position.x) - shot.yaw).abs();
            out = Some(AimTarget { plate: plate.id, position, facing: facing_of(&plate), shot, t_hit, miss });
        }
        out
    }

    fn solve_plate(&self, est: &RobotEstimate, t_cmd: f64, speed: f64, plate: usize) -> Option<AimTarget> {
        let mut flight = 0.0;
        let mut out = None;
        for _ in 0..3 {
            let t_hit = t_cmd + self.cfg.shoot_delay + flight;
            let (position, yaw) = match (est.mode, est.last_plate) {
                (TrackMode::Plate, Some((t_seen, p))) => {
                    let x = est.state_at(t_hit);
                    let v = Vector3::new(x[1], x[3], x[5]);
                    (p.position + v * (t_hit - t_seen), p.yaw)
                }
                _ => {
                    let p = est.plates_at(t_hit).get(plate).copied()?;
                    (p.position, p.yaw)
                }
            };
            let shot = ballistics::solve(&position, speed, self.cfg.drag)?;
            flight = shot.flight_time;
            let facing = wrap_angle(yaw - position.y.atan2(position.x)).abs();
            out = Some(AimTarget { plate, position, facing, shot, t_hit, miss: 0.0 });
        }
        out
    }
}
