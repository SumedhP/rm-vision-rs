//! Projectile with gravity and quadratic drag, and the pitch that hits a target.

use nalgebra::Vector3;

/// Standard gravity, m/s^2.
pub const G: f64 = 9.81;
/// RK4 step. Over the 0.1 to 0.4 s flights here the integration error is far below a millimeter.
const STEP: f64 = 5e-3;
/// Stop iterating once the simulated shot passes this close to the target height, meters.
const HEIGHT_TOLERANCE: f64 = 1e-6;

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
        let remaining = range - pos[0];
        if remaining < 1e-7 {
            return Some((pos[1], t));
        }
        // Shorten the last steps to land on `range` instead of interpolating across a full step.
        // Drag only slows the bullet, so this undershoots and converges in two or three steps.
        let h = (remaining / vel[0]).min(STEP);
        let k1v = accel(vel);
        let v2 = [vel[0] + 0.5 * h * k1v[0], vel[1] + 0.5 * h * k1v[1]];
        let k2v = accel(v2);
        let v3 = [vel[0] + 0.5 * h * k2v[0], vel[1] + 0.5 * h * k2v[1]];
        let k3v = accel(v3);
        let v4 = [vel[0] + h * k3v[0], vel[1] + h * k3v[1]];
        let k4v = accel(v4);
        pos = [0, 1].map(|i| pos[i] + h / 6.0 * (vel[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]));
        vel = [0, 1].map(|i| vel[i] + h / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]));
        t += h;
        if vel[0] <= 0.0 {
            return None;
        }
    }
    None
}

/// Lowest-arc shot from the muzzle (origin) through `target` (world frame, meters).
///
/// Secant iteration on the simulated miss height, started from the drag-free closed-form pitch,
/// which is already within a few millimeters of a hit at these ranges. Usually 3 to 4 simulations.
pub fn solve(target: &Vector3<f64>, speed: f64, drag: f64) -> Option<Shot> {
    let range = target.xy().norm();
    let miss = |pitch: f64| height_at(range, speed, pitch, drag).map(|(z, t)| (z - target.z, t));
    let mut a = vacuum_pitch(range, target.z, speed)?;
    let (mut miss_a, mut flight) = miss(a)?;
    let mut b = a + 0.01;
    for _ in 0..10 {
        if miss_a.abs() < HEIGHT_TOLERANCE {
            return Some(Shot { yaw: target.y.atan2(target.x), pitch: a, flight_time: flight });
        }
        let (miss_b, flight_b) = miss(b)?;
        let next = b - miss_b * (b - a) / (miss_b - miss_a);
        (a, miss_a, flight) = (b, miss_b, flight_b);
        b = next;
        if !b.is_finite() {
            return None;
        }
    }
    None
}

/// Lowest-arc pitch without drag. `None` when out of reach even in a vacuum, which drag only shortens.
fn vacuum_pitch(range: f64, height: f64, speed: f64) -> Option<f64> {
    let a = G * range * range / (2.0 * speed * speed);
    let disc = range * range - 4.0 * a * (a + height);
    (disc >= 0.0 && a > 0.0).then(|| ((range - disc.sqrt()) / (2.0 * a)).atan())
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

    /// Reference pitches from an independent 0.1 ms RK4 + bisection (`k = 0.019`, 23 m/s, level target).
    #[test]
    fn matches_fine_step_reference() {
        for (range, reference_deg) in [(2.0, 1.091), (4.0, 2.240), (6.0, 3.454), (8.0, 4.738)] {
            let shot = solve(&Vector3::new(range, 0.0, 0.0), 23.0, 0.019).expect("reachable");
            let height_err = (shot.pitch.to_degrees() - reference_deg).to_radians() * range;
            assert!(height_err.abs() < 1e-3, "{range} m: pitch {} deg, {height_err} m off the reference", shot.pitch.to_degrees());
        }
    }

    #[test]
    fn out_of_range_is_none() {
        assert!(solve(&Vector3::new(80.0, 0.0, 0.0), 15.0, 0.019).is_none(), "too far for 15 m/s");
    }
}
