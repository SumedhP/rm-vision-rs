//! Whole-robot motion model shared by the EKF, the aimer and the visualizers.
//!
//! State `x` (world frame, meters / radians / seconds):
//! `[cx, vx, cy, vy, cz, vz, yaw, w, r, l, h]`
//! - `(cx, cy, cz)`: rotation center, `cz` is the height of the even plates.
//! - `yaw`: yaw of plate 0, `w`: spin rate.
//! - `r`: radius of even plates, `r + l`: radius of odd plates, `cz + h`: height of odd plates.

use nalgebra::{SVector, Vector3};

/// State dimension.
pub const N: usize = 11;
/// Robot state vector.
pub type State = SVector<f64, N>;

/// Index of each state component.
#[expect(missing_docs, reason = "names mirror the state layout documented above")]
pub mod ix {
    pub const CX: usize = 0;
    pub const VX: usize = 1;
    pub const CY: usize = 2;
    pub const VY: usize = 3;
    pub const CZ: usize = 4;
    pub const VZ: usize = 5;
    pub const YAW: usize = 6;
    pub const W: usize = 7;
    pub const R: usize = 8;
    pub const L: usize = 9;
    pub const H: usize = 10;
}

/// One armor plate predicted from the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlatePose {
    /// Plate index, 0..plate_count.
    pub id: usize,
    /// Plate center, world frame.
    pub position: Vector3<f64>,
    /// Plate yaw: direction from the plate to the rotation center.
    pub yaw: f64,
}

/// Constant-velocity center, constant spin rate, fixed geometry.
pub fn predict(x: &State, dt: f64) -> State {
    let mut y = *x;
    y[ix::CX] += x[ix::VX] * dt;
    y[ix::CY] += x[ix::VY] * dt;
    y[ix::CZ] += x[ix::VZ] * dt;
    y[ix::YAW] = wrap_angle(x[ix::YAW] + x[ix::W] * dt);
    y
}

/// Pose of plate `id` out of `plate_count` evenly spaced plates.
pub fn plate(x: &State, id: usize, plate_count: usize) -> PlatePose {
    let odd = id % 2 == 1 && plate_count == 4;
    let step = std::f64::consts::TAU / f64::from(u32::try_from(plate_count).expect("plate count is 3 or 4"));
    let yaw = wrap_angle(x[ix::YAW] + step * f64::from(u32::try_from(id).expect("plate id is small")));
    let r = x[ix::R] + if odd { x[ix::L] } else { 0.0 };
    let z = x[ix::CZ] + if odd { x[ix::H] } else { 0.0 };
    PlatePose {
        id,
        position: Vector3::new(x[ix::CX] - r * yaw.cos(), x[ix::CY] - r * yaw.sin(), z),
        yaw,
    }
}

/// All plates.
pub fn plates(x: &State, plate_count: usize) -> Vec<PlatePose> {
    (0..plate_count).map(|id| plate(x, id, plate_count)).collect()
}

/// Wraps an angle to `(-pi, pi]`.
pub fn wrap_angle(a: f64) -> f64 {
    let w = a.rem_euclid(std::f64::consts::TAU);
    if w > std::f64::consts::PI { w - std::f64::consts::TAU } else { w }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plate_zero_faces_viewer_when_yaw_zero() {
        let mut x = State::zeros();
        x[ix::CX] = 3.25;
        x[ix::R] = 0.25;
        let p = plate(&x, 0, 4);
        assert!((p.position - Vector3::new(3.0, 0.0, 0.0)).norm() < 1e-12, "plate 0 is between viewer and center");
    }

    #[test]
    fn odd_plates_use_second_radius_and_height() {
        let mut x = State::zeros();
        x[ix::R] = 0.25;
        x[ix::L] = 0.05;
        x[ix::H] = 0.08;
        let p = plate(&x, 1, 4);
        assert!((p.position.xy().norm() - 0.30).abs() < 1e-12, "odd radius is r + l");
        assert!((p.position.z - 0.08).abs() < 1e-12, "odd height is cz + h");
    }

    #[test]
    fn predict_spins_and_translates() {
        let mut x = State::zeros();
        x[ix::VX] = 1.0;
        x[ix::W] = 2.0;
        let y = predict(&x, 0.5);
        assert!((y[ix::CX] - 0.5).abs() < 1e-12 && (y[ix::YAW] - 1.0).abs() < 1e-12, "cv + constant spin");
    }
}
