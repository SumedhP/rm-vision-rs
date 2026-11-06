//! Buffered gimbal orientations, interpolated to a frame's exposure time.

use std::collections::VecDeque;

use nalgebra::UnitQuaternion;

/// Time-ordered ring buffer of IMU orientations.
#[derive(Debug, Clone)]
pub struct PoseBuffer {
    capacity: usize,
    samples: VecDeque<(f64, UnitQuaternion<f64>)>,
}

impl PoseBuffer {
    /// Buffer holding at most `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self { capacity: capacity.max(2), samples: VecDeque::new() }
    }

    /// Appends a sample. Samples must arrive in time order.
    pub fn push(&mut self, t: f64, q: UnitQuaternion<f64>) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back((t, q));
    }

    /// Orientation at `t`, slerped between the bracketing samples. Outside the buffered range the
    /// nearest sample is used, because frames can land a few ms past the newest IMU packet.
    pub fn at(&self, t: f64) -> Option<UnitQuaternion<f64>> {
        let i = self.samples.partition_point(|&(ts, _)| ts <= t);
        match (i.checked_sub(1).and_then(|j| self.samples.get(j)), self.samples.get(i)) {
            (Some(&(t0, q0)), Some(&(t1, q1))) => Some(q0.slerp(&q1, (t - t0) / (t1 - t0))),
            (Some(&(_, q)), None) | (None, Some(&(_, q))) => Some(q),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn slerps_between_samples_and_clamps_outside() {
        let mut buf = PoseBuffer::new(4);
        let q1 = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 1.0);
        buf.push(0.0, UnitQuaternion::identity());
        buf.push(1.0, q1);
        let mid = buf.at(0.25).expect("buffer has samples");
        assert!((mid.angle() - 0.25).abs() < 1e-9, "quarter of the way");
        assert!((buf.at(5.0).expect("clamps").angle() - 1.0).abs() < 1e-12, "uses newest past the end");
        assert!(PoseBuffer::new(4).at(0.0).is_none(), "empty buffer has no pose");
    }
}
