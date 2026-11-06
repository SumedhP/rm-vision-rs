//! Iterated extended Kalman filter over the whole-robot state, with a chi-square innovation gate.
//!
//! The measurement model is passed in per update, so the same filter takes either a solved plate
//! pose (4 values) or the plate's four corner pixels (8 values).

use nalgebra::{SMatrix, SVector, UnitQuaternion};

use super::model::{self, N, State, ix, wrap_angle};
use crate::geometry::{ArmorSize, Camera, armor_corners_world};

/// State covariance.
pub type Covariance = SMatrix<f64, N, N>;

/// Process noise densities.
#[derive(Debug, Clone, Copy)]
pub struct ProcessNoise {
    /// Horizontal center acceleration, m/s^2.
    pub acc: f64,
    /// Vertical center acceleration, m/s^2 (robots don't change height much).
    pub acc_z: f64,
    /// Spin acceleration, rad/s^2.
    pub alpha: f64,
    /// Geometry (r, l, h) drift, m/sqrt(s).
    pub geometry: f64,
}

/// One measurement: its value, noise, and which components are angles (wrapped in residuals).
#[derive(Debug, Clone, Copy)]
pub struct Observation<const M: usize> {
    /// Measured value.
    pub z: SVector<f64, M>,
    /// Measurement noise covariance.
    pub r: SMatrix<f64, M, M>,
    /// Indices of angle components.
    pub angles: &'static [usize],
}

/// Mean and covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct Ekf {
    /// Mean.
    pub x: State,
    /// Covariance.
    pub p: Covariance,
}

impl Ekf {
    /// Propagates by `dt` seconds. Noise is per second, so it scales correctly with frame time.
    pub fn predict(&mut self, dt: f64, noise: &ProcessNoise) {
        let mut f = Covariance::identity();
        for (pos, vel) in [(ix::CX, ix::VX), (ix::CY, ix::VY), (ix::CZ, ix::VZ), (ix::YAW, ix::W)] {
            f[(pos, vel)] = dt;
        }
        let mut q = Covariance::zeros();
        let (dt2, dt3, dt4) = (dt * dt, dt * dt * dt, dt * dt * dt * dt);
        for (pos, vel, sigma) in [
            (ix::CX, ix::VX, noise.acc),
            (ix::CY, ix::VY, noise.acc),
            (ix::CZ, ix::VZ, noise.acc_z),
            (ix::YAW, ix::W, noise.alpha),
        ] {
            let s2 = sigma * sigma;
            q[(pos, pos)] = s2 * dt4 / 4.0;
            q[(pos, vel)] = s2 * dt3 / 2.0;
            q[(vel, pos)] = s2 * dt3 / 2.0;
            q[(vel, vel)] = s2 * dt2;
        }
        for i in [ix::R, ix::L, ix::H] {
            q[(i, i)] = noise.geometry * noise.geometry * dt;
        }
        self.x = model::predict(&self.x, dt);
        self.p = f * self.p * f.transpose() + q;
    }

    /// Normalized innovation squared of `obs` against the prediction `h(x)`.
    pub fn nis<const M: usize>(&self, obs: &Observation<M>, h: impl Fn(&State) -> Option<SVector<f64, M>>) -> Option<f64> {
        let (y, jac) = linearize(obs, &h, &self.x)?;
        let s_inv = (jac * self.p * jac.transpose() + obs.r).try_inverse()?;
        Some((y.transpose() * s_inv * y)[(0, 0)])
    }

    /// Updates with `obs` unless its NIS against the prior exceeds `gate`, and returns that NIS.
    ///
    /// `iterations > 1` relinearizes `h` around each new estimate (Gauss-Newton on the MAP cost,
    /// Bell & Cathey 1993). Each step restarts from the prior mean and keeps the prior term
    /// `H_i (x_prior - x_i)`, which is the term the iterated filters in the reference repos drop.
    pub fn update<const M: usize>(
        &mut self,
        obs: &Observation<M>,
        h: impl Fn(&State) -> Option<SVector<f64, M>>,
        gate: f64,
        iterations: usize,
    ) -> Option<f64> {
        let prior = self.x;
        let nis = self.nis(obs, &h)?;
        if nis > gate {
            return Some(nis);
        }
        let mut xi = prior;
        let mut gain = None;
        for _ in 0..iterations.max(1) {
            let (y, jac) = linearize(obs, &h, &xi)?;
            let k = self.p * jac.transpose() * (jac * self.p * jac.transpose() + obs.r).try_inverse()?;
            let mut next = prior + k * (y - jac * state_diff(&prior, &xi));
            next[ix::YAW] = wrap_angle(next[ix::YAW]);
            let step = state_diff(&next, &xi).norm();
            xi = next;
            gain = Some((k, jac));
            if step < 1e-9 {
                break;
            }
        }
        let (k, jac) = gain?;
        self.x = xi;
        let i_kh = Covariance::identity() - k * jac;
        self.p = i_kh * self.p * i_kh.transpose() + k * obs.r * k.transpose();
        Some(nis)
    }
}

/// Residual `z - h(x)` and the Jacobian of `h` at `x` (central differences).
fn linearize<const M: usize>(
    obs: &Observation<M>,
    h: impl Fn(&State) -> Option<SVector<f64, M>>,
    x: &State,
) -> Option<(SVector<f64, M>, SMatrix<f64, M, N>)> {
    let wrap = |mut d: SVector<f64, M>| {
        for &i in obs.angles {
            d[i] = wrap_angle(d[i]);
        }
        d
    };
    let mut jac = SMatrix::<f64, M, N>::zeros();
    for i in 0..N {
        let eps = 1e-6;
        let (mut a, mut b) = (*x, *x);
        a[i] += eps;
        b[i] -= eps;
        jac.set_column(i, &wrap(h(&a)? - h(&b)?).scale(0.5 / eps));
    }
    Some((wrap(obs.z - h(x)?), jac))
}

fn state_diff(a: &State, b: &State) -> State {
    let mut d = a - b;
    d[ix::YAW] = wrap_angle(d[ix::YAW]);
    d
}

/// `[bearing yaw, bearing pitch, distance, plate yaw]` of plate `id` as seen from the world origin.
pub fn measure_pose(x: &State, id: usize, plate_count: usize) -> SVector<f64, 4> {
    let plate = model::plate(x, id, plate_count);
    let p = plate.position;
    SVector::<f64, 4>::new(p.y.atan2(p.x), p.z.atan2(p.xy().norm()), p.norm(), plate.yaw)
}

/// Pixel coordinates `[u, v] x (TL, BL, BR, TR)` of plate `id`'s corners. `None` if a corner is
/// behind the camera.
pub fn measure_corners(
    x: &State,
    id: usize,
    plate_count: usize,
    size: ArmorSize,
    camera: &Camera,
    q_imu: &UnitQuaternion<f64>,
) -> Option<SVector<f64, 8>> {
    let plate = model::plate(x, id, plate_count);
    let mut z = SVector::<f64, 8>::zeros();
    for (i, c) in armor_corners_world(size, &plate.position, plate.yaw).iter().enumerate() {
        let px = camera.project(q_imu, c)?;
        z[2 * i] = px.x;
        z[2 * i + 1] = px.y;
    }
    Some(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::CameraConfig;

    const POSE_ANGLES: &[usize] = &[0, 3];

    fn truth() -> State {
        let mut x = State::zeros();
        x[ix::CX] = 3.25;
        x[ix::YAW] = 0.4;
        x[ix::R] = 0.25;
        x
    }

    #[test]
    fn linear_part_converges_and_gate_rejects_outliers() {
        let truth = truth();
        let mut ekf = Ekf { x: truth, p: Covariance::identity() * 0.01 };
        ekf.x[ix::CX] += 0.1;
        let r = SMatrix::<f64, 4, 4>::from_diagonal(&SVector::<f64, 4>::new(1e-6, 1e-6, 1e-4, 1e-3));
        let noise = ProcessNoise { acc: 1.0, acc_z: 1.0, alpha: 1.0, geometry: 1e-3 };
        let h = |x: &State| Some(measure_pose(x, 0, 4));
        for _ in 0..30 {
            ekf.predict(0.01, &noise);
            ekf.update(&Observation { z: measure_pose(&truth, 0, 4), r, angles: POSE_ANGLES }, h, 1e9, 1);
        }
        let plate_err = (model::plate(&ekf.x, 0, 4).position - model::plate(&truth, 0, 4).position).norm();
        assert!(plate_err < 0.01, "plate converges, error {plate_err}");

        let mut wild = measure_pose(&truth, 0, 4);
        wild[2] += 2.0;
        let before = ekf.x;
        let nis = ekf.update(&Observation { z: wild, r, angles: POSE_ANGLES }, h, 13.28, 1).expect("invertible");
        assert!(nis > 13.28 && ekf.x == before, "a 2 m distance jump is gated out");
    }

    /// With a precise, strongly nonlinear measurement (corner pixels) and a poor prior, one EKF step
    /// lands off the measurement; iterating converges onto it.
    #[test]
    fn iterating_fits_a_nonlinear_measurement_better() {
        let camera = Camera::from(&CameraConfig { k: [691.0, 691.0, 486.0, 272.0], dist: [0.0; 4], rpy: [0.0; 3], translation: [0.0; 3] });
        let q = UnitQuaternion::identity();
        let truth = truth();
        let h = |x: &State| measure_corners(x, 0, 4, ArmorSize::Small, &camera, &q);
        let z = h(&truth).expect("visible");
        let obs = Observation { z, r: SMatrix::<f64, 8, 8>::identity() * 0.25, angles: &[] };
        let mut prior = Ekf { x: truth, p: Covariance::from_diagonal(&State::from_element(0.04)) };
        prior.x[ix::YAW] += 0.5;
        prior.x[ix::CX] += 0.3;
        let residual = |iterations| {
            let mut ekf = prior.clone();
            ekf.update(&obs, h, f64::INFINITY, iterations).expect("updates");
            (z - h(&ekf.x).expect("visible")).norm()
        };
        let (once, iterated) = (residual(1), residual(10));
        assert!(once > 1.0 && iterated < 0.5 * once,"iterated residual {iterated} px vs single step {once} px");
    }
}
