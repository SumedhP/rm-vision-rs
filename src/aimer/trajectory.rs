//! Per-axis setpoint smoother. Tracks a moving target (position + velocity + acceleration) with a
//! critically damped feedback law whose output acceleration is clamped to the gimbal limit, so a
//! plate switch becomes a fast but feasible slew instead of a step.

/// One gimbal axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Axis {
    /// Commanded angle, radians.
    pub pos: f64,
    /// Commanded rate, rad/s.
    pub vel: f64,
    /// Commanded acceleration, rad/s^2.
    pub acc: f64,
}

impl Axis {
    /// At rest at `pos`.
    pub fn at(pos: f64) -> Self {
        Self { pos, vel: 0.0, acc: 0.0 }
    }

    /// Advances `dt` toward `target`. `bandwidth` is the closed-loop natural frequency (rad/s).
    pub fn step(&mut self, target: &Axis, dt: f64, max_acc: f64, bandwidth: f64) {
        let error = target.pos - self.pos;
        let acc = target.acc + bandwidth * bandwidth * error + 2.0 * bandwidth * (target.vel - self.vel);
        self.acc = acc.clamp(-max_acc, max_acc);
        self.pos += self.vel * dt + 0.5 * self.acc * dt * dt;
        self.vel += self.acc * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_respects_limit_and_settles() {
        let mut a = Axis::at(0.0);
        let target = Axis::at(0.5);
        let mut peak = 0.0f64;
        for _ in 0..400 {
            a.step(&target, 0.005, 60.0, 30.0);
            peak = peak.max(a.acc.abs());
        }
        assert!(peak <= 60.0, "acceleration {peak} within limit");
        assert!((a.pos - 0.5).abs() < 1e-3 && a.vel.abs() < 1e-2, "settles on the target");
    }

    #[test]
    fn tracks_a_sinusoid_with_feedforward() {
        let mut a = Axis::at(0.0);
        let mut worst = 0.0f64;
        for i in 0..1000 {
            let t = f64::from(i) * 0.005;
            let target = Axis { pos: 0.2 * (3.0 * t).sin(), vel: 0.6 * (3.0 * t).cos(), acc: -1.8 * (3.0 * t).sin() };
            a.step(&target, 0.005, 60.0, 30.0);
            if i > 200 {
                worst = worst.max((a.pos - 0.2 * (3.0 * (t + 0.005)).sin()).abs());
            }
        }
        assert!(worst < 2e-3, "tracking error {worst} rad");
    }
}
