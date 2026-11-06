//! Turns the robot estimate into a smoothed gimbal command and a fire decision.
//!
//! Plate choice follows rmcs's attack window. A plate is engaged only while its angle around the
//! robot center is inside a window. That window narrows when the gimbal's speed and acceleration
//! can't follow the plate across it and come back for the next one. When no plate is inside, the
//! gimbal waits on the window edge where the next plate will enter. Above `degraded_spin` it aims
//! at the robot center instead. In both of those cases the shot is timed to a plate crossing the
//! line of fire at impact.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use nalgebra::Vector3;

use super::ballistics::{self, Shot};
use super::trajectory::Axis;
use super::{AimCommand, AimerConfig, GimbalLimits};
use crate::geometry::LIGHTBAR_LENGTH;
use crate::tracker::model::{self, State, ix, wrap_angle};
use crate::tracker::{PlatePose, RobotEstimate, TrackMode, TrackState};

/// How the aimer is engaging the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum AimKind {
    /// Following a plate inside the attack window.
    Plate,
    /// No plate in the window: waiting on the edge the next plate enters from.
    PreAim,
    /// Spinning faster than `degraded_spin`: aiming at the robot center.
    Center,
}

/// What the aimer is pointing at this tick (for logging and overlays).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct AimTarget {
    /// How the target is engaged.
    pub kind: AimKind,
    /// Plate being aimed at ([`AimKind::Plate`]), or the plate closest to the line of fire.
    pub plate: usize,
    /// Aim point at impact, world frame.
    pub position: Vector3<f64>,
    /// Angle between `plate`'s normal and the line of sight at impact, radians.
    pub facing: f64,
    /// Unsmoothed ballistic solution.
    pub shot: Shot,
    /// Predicted impact time.
    pub t_hit: f64,
    /// Angle between the shot and `plate`'s center at impact, radians. Zero for
    /// [`AimKind::Plate`].
    pub miss: f64,
}

#[derive(Debug, Clone, Copy)]
enum Selection {
    Plate(usize),
    /// Window edge as a signed angle around the robot center.
    PreAim(f64),
    Center,
}

/// Stateful aimer: remembers the selected plate and the smoothed gimbal setpoint.
#[derive(Debug, Clone)]
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

        let solved = est.and_then(|e| {
            let (selection, flight) = self.select(e, t_cmd, speed)?;
            Some((e, selection, self.aim_at(e, t_cmd, speed, selection, flight, 2)?))
        });
        self.target = solved.map(|(_, _, t)| t);
        let Some((est, selection, target)) = solved else {
            let (yaw, pitch) = (self.yaw.unwrap_or(Axis::at(0.0)), self.pitch);
            return AimCommand { t: t_cmd, yaw: yaw.pos, pitch: pitch.pos, ..AimCommand::default() };
        };

        let ahead = self.aim_at(est, t_cmd + 0.01, speed, selection, target.shot.flight_time, 1);
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

    /// Chooses what to aim at, judged at the predicted impact time. Also returns the flight time
    /// to the robot center, which seeds the impact-time iteration.
    fn select(&mut self, est: &RobotEstimate, t_cmd: f64, speed: f64) -> Option<(Selection, f64)> {
        let t_shot = t_cmd + self.cfg.shoot_delay;
        let flight = ballistics::solve(&center(&est.state_at(t_shot)), speed, self.cfg.drag)?.flight_time;
        if est.mode == TrackMode::Plate {
            self.plate = None;
            return Some((Selection::Plate(est.last_plate.map_or(0, |(_, p)| p.id)), flight));
        }
        let x = est.state_at(t_shot + flight);
        let n = est.plate_count;
        let Some(half) = self.half_window(&x, n) else {
            self.plate = None;
            return Some((Selection::Center, flight));
        };
        let angle = |id: usize| body_angle(&x, &model::plate(&x, id, n)).abs();
        let h = self.cfg.window_hysteresis;
        if let Some(id) = self.plate.filter(|&id| id < n && angle(id) <= half * (1.0 + h)) {
            return Some((Selection::Plate(id), flight));
        }
        self.plate = (0..n).filter(|&id| angle(id) <= half * (1.0 - h)).min_by(|&a, &b| angle(a).total_cmp(&angle(b)));
        let selection = match self.plate {
            Some(id) => Selection::Plate(id),
            None => Selection::PreAim(if x[ix::W] < 0.0 { half } else { -half }),
        };
        Some((selection, flight))
    }

    /// Half width of the attack window, or `None` when spinning too fast to engage plates.
    fn half_window(&self, x: &State, plate_count: usize) -> Option<f64> {
        let w = x[ix::W].abs();
        if w > self.cfg.degraded_spin {
            return None;
        }
        let r = self.cfg.window_redundancy;
        let dynamic = tracking_window(w, x[ix::R], center(x).xy().norm(), plate_count, self.limits.max_yaw_vel * r, self.limits.max_yaw_acc * r);
        Some((self.cfg.attack_window / 2.0).min(dynamic))
    }

    /// Iterates impact time with flight time, starting from the flight-time guess `flight`.
    fn aim_at(&self, est: &RobotEstimate, t_cmd: f64, speed: f64, selection: Selection, mut flight: f64, iterations: usize) -> Option<AimTarget> {
        let mut out = None;
        for _ in 0..iterations {
            let t_hit = t_cmd + self.cfg.shoot_delay + flight;
            let x = est.state_at(t_hit);
            let plates = est.plates_at(t_hit);
            let target = match selection {
                Selection::Plate(id) => {
                    let (position, yaw) = if let (TrackMode::Plate, Some((t_seen, p))) = (est.mode, est.last_plate) {
                        (p.position + Vector3::new(x[ix::VX], x[ix::VY], x[ix::VZ]) * (t_hit - t_seen), p.yaw)
                    } else {
                        let p = plates.get(id)?;
                        (p.position, p.yaw)
                    };
                    let shot = ballistics::solve(&position, speed, self.cfg.drag)?;
                    let facing = wrap_angle(yaw - position.y.atan2(position.x)).abs();
                    AimTarget { kind: AimKind::Plate, plate: id, position, facing, shot, t_hit, miss: 0.0 }
                }
                Selection::PreAim(edge) => {
                    let incoming = plates
                        .iter()
                        .map(|p| (p, body_angle(&x, p)))
                        .filter(|(_, a)| (*a > 0.0) == (edge > 0.0))
                        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                        .map_or(plates.first()?, |(p, _)| p);
                    let c = center(&x);
                    let radius = (incoming.position - c).xy().norm();
                    let a = (-c.y).atan2(-c.x) + edge;
                    self.timed(AimKind::PreAim, Vector3::new(c.x + radius * a.cos(), c.y + radius * a.sin(), incoming.position.z), &x, &plates, speed, t_hit)?
                }
                Selection::Center => {
                    let c = center(&x);
                    let front = &plates.iter().min_by(|a, b| body_angle(&x, a).abs().total_cmp(&body_angle(&x, b).abs()))?;
                    let radius = (front.position - c).xy().norm();
                    let toward_us = -c.xy().normalize() * radius;
                    self.timed(AimKind::Center, Vector3::new(c.x + toward_us.x, c.y + toward_us.y, front.position.z), &x, &plates, speed, t_hit)?
                }
            };
            flight = target.shot.flight_time;
            out = Some(target);
        }
        out
    }

    /// Shot at a fixed point, with the plate that crosses its line of fire closest at impact.
    fn timed(&self, kind: AimKind, position: Vector3<f64>, x: &State, plates: &[PlatePose], speed: f64, t_hit: f64) -> Option<AimTarget> {
        let shot = ballistics::solve(&position, speed, self.cfg.drag)?;
        let miss = |p: &PlatePose| wrap_angle(p.position.y.atan2(p.position.x) - shot.yaw).abs();
        let plate = plates.iter().filter(|p| body_angle(x, p).abs() < FRAC_PI_2).min_by(|a, b| miss(a).total_cmp(&miss(b)))?;
        let facing = wrap_angle(plate.yaw - plate.position.y.atan2(plate.position.x)).abs();
        Some(AimTarget { kind, plate: plate.id, position, facing, shot, t_hit, miss: miss(plate) })
    }
}

fn center(x: &State) -> Vector3<f64> {
    Vector3::new(x[ix::CX], x[ix::CY], x[ix::CZ])
}

/// Signed angle of a plate around the robot center, zero when it faces the gimbal; grows with
/// the spin direction.
fn body_angle(x: &State, plate: &PlatePose) -> f64 {
    wrap_angle(plate.yaw - x[ix::CY].atan2(x[ix::CX]))
}

/// rmcs's sawtooth window: the half width (angle around the robot center) inside which a gimbal
/// limited to `v_max`, `a_max` can follow a plate at a steady rate, given that it must also slew back
/// for the next plate. `INFINITY` when it never limits.
///
/// Seen from the gimbal, each plate sweeps a fixed angle (`stroke`) once per `TAU / (n w)`, so the
/// yaw target is a sawtooth. Tracking it at `v_track` and returning with a bang-bang (or
/// trapezoidal, when speed-limited) slew leaves the remaining stroke as the steady window.
pub fn tracking_window(w: f64, radius: f64, distance: f64, plate_count: usize, v_max: f64, a_max: f64) -> f64 {
    let n = f64::from(u32::try_from(plate_count).unwrap_or(4));
    if w < 0.1 || radius <= 0.0 || distance <= radius || v_max <= 0.0 || a_max <= 0.0 {
        return f64::INFINITY;
    }
    let half = PI / n;
    let sight = (distance * distance + radius * radius - 2.0 * distance * radius * half.cos()).sqrt();
    let stroke = 2.0 * (radius * half.sin() / sight).asin();
    let v_track = stroke * n * w / TAU;
    let stable = if (stroke * a_max).sqrt() <= v_max + v_track {
        stroke - 2.0 * v_track * (stroke / a_max).sqrt()
    } else {
        v_max * stroke / (v_max + v_track) - v_track * (v_max + v_track) / a_max
    };
    let phi = stable.max(0.0) / 2.0;
    let s = distance / radius * phi.sin();
    if s >= 1.0 { f64::INFINITY } else { s.asin() - phi }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_shrinks_with_spin_and_opens_with_distance() {
        let window = |w, d| tracking_window(w, 0.25, d, 4, 8.0, 48.0);
        assert!(window(0.05, 3.0).is_infinite(), "no spin: unlimited");
        assert!(window(4.0, 3.0) > window(8.0, 3.0), "faster spin, narrower window");
        assert!(window(8.0, 6.0) > window(8.0, 3.0), "further away, the gimbal sweeps less");
        assert!(window(30.0, 3.0) < 0.05, "a gimbal can't follow 30 rad/s plates");
    }

    /// At 3 m and 11.7 rad/s (the `translate_spinning` video), the steady window is about 9.5 degrees
    /// either side of straight-on, worked by hand from the formula.
    #[test]
    fn window_matches_hand_calculation() {
        let half = tracking_window(11.7, 0.25, 3.0, 4, 8.0, 48.0).to_degrees();
        assert!((8.0..11.0).contains(&half), "half window {half} deg");
    }
}
