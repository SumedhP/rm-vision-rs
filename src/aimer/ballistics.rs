//! Projectile with gravity and quadratic drag, and the pitch that hits a target.

use nalgebra::Vector3;

/// Standard gravity, m/s^2.
pub const G: f64 = 9.81;
const STEP: f64 = 1e-3;

/// Gun angles and time of flight for one target.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Shot {
    /// Yaw, radians.
    pub yaw: f64,
    /// Pitch, radians, positive up.
    pub pitch: f64,
    /// Time from muzzle to target, seconds.
    pub flight_time: f64,
}

/// Height and time when the bullet reaches horizontal distance `range`, fired at `pitch`.
/// `drag` is `k` in `a = -k |v| v` (m^-1). `None` if it never gets there.
pub fn height_at(range: f64, speed: f64, pitch: f64, drag: f64) -> Option<(f64, f64)> {
    let accel = |v: [f64; 2]| {
        let s = v[0].hypot(v[1]);
        [-drag * s * v[0], -drag * s * v[1] - G]
    };
    let (mut pos, mut vel, mut t) = ([0.0f64; 2], [speed * pitch.cos(), speed * pitch.sin()], 0.0);
    while t < 3.0 {
        let k1v = accel(vel);
        let v2 = [vel[0] + 0.5 * STEP * k1v[0], vel[1] + 0.5 * STEP * k1v[1]];
        let k2v = accel(v2);
        let v3 = [vel[0] + 0.5 * STEP * k2v[0], vel[1] + 0.5 * STEP * k2v[1]];
        let k3v = accel(v3);
        let v4 = [vel[0] + STEP * k3v[0], vel[1] + STEP * k3v[1]];
        let k4v = accel(v4);
        let next_pos = [0, 1].map(|i| pos[i] + STEP / 6.0 * (vel[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]));
        let next_vel = [0, 1].map(|i| vel[i] + STEP / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]));
        if next_pos[0] >= range {
            let f = (range - pos[0]) / (next_pos[0] - pos[0]);
            return Some((pos[1] + f * (next_pos[1] - pos[1]), t + f * STEP));
        }
        if next_vel[0] <= 0.0 {
            return None;
        }
        (pos, vel, t) = (next_pos, next_vel, t + STEP);
    }
    None
}

/// Lowest-arc shot from the muzzle (origin) through `target` (world frame, meters).
pub fn solve(target: &Vector3<f64>, speed: f64, drag: f64) -> Option<Shot> {
    let range = target.xy().norm();
    let miss = |pitch: f64| height_at(range, speed, pitch, drag).map(|(z, _)| z - target.z);
    let (mut lo, mut hi) = (-0.6f64, 0.7f64);
    if miss(lo)? > 0.0 || miss(hi)? < 0.0 {
        return None;
    }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if miss(mid)? < 0.0 { lo = mid } else { hi = mid }
    }
    let pitch = 0.5 * (lo + hi);
    Some(Shot { yaw: target.y.atan2(target.x), pitch, flight_time: height_at(range, speed, pitch, drag)?.1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuum_matches_closed_form() {
        let (v, s, z) = (23.0f64, 5.0f64, 0.3f64);
        let shot = solve(&Vector3::new(s, 0.0, z), v, 0.0).expect("reachable");
        let closed = ((v * v - (v.powi(4) - G * (G * s * s + 2.0 * z * v * v)).sqrt()) / (G * s)).atan();
        assert!((shot.pitch - closed).abs() < 1e-6, "pitch {} vs {closed}", shot.pitch);
        assert!((shot.flight_time - s / (v * closed.cos())).abs() < 1e-4, "vacuum flight time");
    }

    #[test]
    fn drag_raises_pitch_and_still_hits() {
        let target = Vector3::new(6.0, 1.0, -0.2);
        let vac = solve(&target, 23.0, 0.0).expect("reachable");
        let shot = solve(&target, 23.0, 0.019).expect("reachable");
        assert!(shot.pitch > vac.pitch && shot.flight_time > vac.flight_time, "drag needs a higher, slower shot");
        let (z, _) = height_at(target.xy().norm(), 23.0, shot.pitch, 0.019).expect("arrives");
        assert!((z - target.z).abs() < 1e-3, "re-simulated hit within 1 mm");
    }

    #[test]
    fn out_of_range_is_none() {
        assert!(solve(&Vector3::new(80.0, 0.0, 0.0), 15.0, 0.019).is_none(), "too far for 15 m/s");
    }
}
