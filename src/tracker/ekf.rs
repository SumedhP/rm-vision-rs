//! Extended Kalman filter over the whole-robot state, with a chi-square innovation gate.

use nalgebra::{SMatrix, SVector};

use super::model::{self, N, State, ix, wrap_angle};

/// Measurement of one plate: `[bearing yaw, bearing pitch, distance, plate yaw]`.
pub type Measurement = SVector<f64, 4>;
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

    /// Normalized innovation squared of plate `id`'s measurement against the prediction.
    pub fn nis(&self, z: &Measurement, r: &SMatrix<f64, 4, 4>, id: usize, plate_count: usize) -> Option<f64> {
        self.innovation(z, r, id, plate_count).map(|i| i.nis)
    }

    /// Updates with plate `id`'s measurement unless its NIS exceeds `gate`. Returns the NIS,
    /// computed from the prior innovation covariance, either way.
    pub fn update(&mut self, z: &Measurement, r: &SMatrix<f64, 4, 4>, id: usize, plate_count: usize, gate: f64) -> Option<f64> {
        let Innovation { y, s_inv, jac, nis } = self.innovation(z, r, id, plate_count)?;
        if nis > gate {
            return Some(nis);
        }
        let k = self.p * jac.transpose() * s_inv;
        self.x += k * y;
        self.x[ix::YAW] = wrap_angle(self.x[ix::YAW]);
        let i_kh = Covariance::identity() - k * jac;
        self.p = i_kh * self.p * i_kh.transpose() + k * r * k.transpose();
        Some(nis)
    }

    #[expect(clippy::many_single_char_names, reason = "z, r: Kalman filter measurement/covariance; h: observation fn; a, b: finite difference states")]
    fn innovation(&self, z: &Measurement, r: &SMatrix<f64, 4, 4>, id: usize, plate_count: usize) -> Option<Innovation> {
        let h = |x: &State| measure(x, id, plate_count);
        let mut jac = SMatrix::<f64, 4, N>::zeros();
        for i in 0..N {
            let eps = 1e-6;
            let (mut a, mut b) = (self.x, self.x);
            a[i] += eps;
            b[i] -= eps;
            jac.set_column(i, &diff(&h(&a), &h(&b)).scale(0.5 / eps));
        }
        let y = diff(z, &h(&self.x));
        let s_inv = (jac * self.p * jac.transpose() + r).try_inverse()?;
        let nis = (y.transpose() * s_inv * y)[(0, 0)];
        Some(Innovation { y, s_inv, jac, nis })
    }
}

struct Innovation {
    y: Measurement,
    s_inv: SMatrix<f64, 4, 4>,
    jac: SMatrix<f64, 4, N>,
    nis: f64,
}
/// `[bearing yaw, bearing pitch, distance, plate yaw]` of plate `id` as seen from the world origin.
pub fn measure(x: &State, id: usize, plate_count: usize) -> Measurement {
    let plate = model::plate(x, id, plate_count);
    let p = plate.position;
    Measurement::new(p.y.atan2(p.x), p.z.atan2(p.xy().norm()), p.norm(), plate.yaw)
}

fn diff(a: &Measurement, b: &Measurement) -> Measurement {
    let mut d = a - b;
    d[0] = wrap_angle(d[0]);
    d[3] = wrap_angle(d[3]);
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_part_converges_and_gate_rejects_outliers() {
        let mut truth = State::zeros();
        truth[ix::CX] = 3.25;
        truth[ix::R] = 0.25;
        let mut ekf = Ekf { x: truth, p: Covariance::identity() * 0.01 };
        ekf.x[ix::CX] += 0.1;
        let r = SMatrix::<f64, 4, 4>::from_diagonal(&Measurement::new(1e-6, 1e-6, 1e-4, 1e-3));
        let noise = ProcessNoise { acc: 1.0, acc_z: 1.0, alpha: 1.0, geometry: 1e-3 };
        for _ in 0..30 {
            ekf.predict(0.01, &noise);
            ekf.update(&measure(&truth, 0, 4), &r, 0, 4, 1e9);
        }
        let plate_err = (model::plate(&ekf.x, 0, 4).position - model::plate(&truth, 0, 4).position).norm();
        assert!(plate_err < 0.01, "plate converges, error {plate_err}");

        let mut wild = measure(&truth, 0, 4);
        wild[2] += 2.0;
        let before = ekf.x;
        let nis = ekf.update(&wild, &r, 0, 4, 13.28).expect("invertible");
        assert!(nis > 13.28 && ekf.x == before, "a 2 m distance jump is gated out");
    }
}
