//! Armor plate pose from its four keypoints.
//!
//! The plate's tilt is fixed by the rules, so the unknowns are position and yaw. A yaw grid search
//! picks the right side of the planar two-fold ambiguity; Levenberg-Marquardt on the eight pixel
//! residuals then refines all four parameters.

use nalgebra::{SMatrix, SVector, UnitQuaternion, Vector3};

use crate::detector::Detection;
use crate::geometry::{ARMOR_TILT, ArmorSize, Camera, LIGHTBAR_LENGTH, armor_corners_world};

/// One plate's pose in the world frame.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ArmorObservation {
    /// Plate center, meters.
    pub position: Vector3<f64>,
    /// Plate yaw (direction from the plate to the robot center), radians.
    pub yaw: f64,
    /// RMS reprojection error, pixels.
    pub rms_px: f64,
}

type Params = SVector<f64, 4>;
type Residual = SVector<f64, 8>;

/// Solves the plate pose. `None` if the corners are degenerate or the fit does not converge
/// to a plausible pose.
pub fn solve_armor(det: &Detection, cam: &Camera, q_imu: &UnitQuaternion<f64>) -> Option<ArmorObservation> {
    let size = det.icon.armor_size();
    let [tl, bl, br, tr] = det.corners;
    let bar_px = 0.5 * ((bl - tl).norm() + (br - tr).norm());
    if bar_px < 2.0 {
        return None;
    }
    let center_px = nalgebra::Point2::from((tl.coords + bl.coords + br.coords + tr.coords) / 4.0);
    let depth = LIGHTBAR_LENGTH * ARMOR_TILT.cos() * cam.intrinsics.fy / bar_px;
    let position = cam.origin(q_imu) + cam.ray(q_imu, &center_px) * depth;

    let view_yaw = position.y.atan2(position.x);
    let cost = |p: &Params| residual(det, cam, q_imu, size, p).map_or(f64::INFINITY, |r| r.norm_squared());
    let yaw = (-80..=80)
        .map(|deg| view_yaw + f64::from(deg).to_radians())
        .min_by(|a, b| cost(&with_yaw(position, *a)).total_cmp(&cost(&with_yaw(position, *b))))?;

    let mut p = with_yaw(position, yaw);
    let mut lambda = 1e-3;
    let mut current = cost(&p);
    for _ in 0..20 {
        let r = residual(det, cam, q_imu, size, &p)?;
        let mut j = SMatrix::<f64, 8, 4>::zeros();
        for i in 0..4 {
            let h = if i == 3 { 1e-5 } else { 1e-6 };
            let mut dp = p;
            dp[i] += h;
            j.set_column(i, &((residual(det, cam, q_imu, size, &dp)? - r) / h));
        }
        let jtj = j.transpose() * j;
        let step = (jtj + SMatrix::<f64, 4, 4>::from_diagonal(&jtj.diagonal()) * lambda).try_inverse()? * j.transpose() * r;
        let next = p - step;
        let next_cost = cost(&next);
        if next_cost < current {
            p = next;
            current = next_cost;
            lambda *= 0.3;
            if step.norm() < 1e-9 {
                break;
            }
        } else {
            lambda *= 10.0;
        }
    }
    let rms_px = (current / 4.0).sqrt();
    rms_px.is_finite().then(|| ArmorObservation { position: p.xyz(), yaw: crate::tracker::model::wrap_angle(p[3]), rms_px })
}

fn with_yaw(position: Vector3<f64>, yaw: f64) -> Params {
    Params::new(position.x, position.y, position.z, yaw)
}

fn residual(det: &Detection, cam: &Camera, q_imu: &UnitQuaternion<f64>, size: ArmorSize, p: &Params) -> Option<Residual> {
    let corners = armor_corners_world(size, &p.xyz(), p[3]);
    let mut r = Residual::zeros();
    for (i, c) in corners.iter().enumerate() {
        let d = cam.project(q_imu, c)? - det.corners[i];
        r[2 * i] = d.x;
        r[2 * i + 1] = d.y;
    }
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{ArmorColor, Icon};
    use crate::geometry::CameraConfig;

    #[test]
    fn recovers_pose_across_yaws_with_pixel_noise() {
        let cam = Camera::from(&CameraConfig { k: [691.0, 691.0, 486.0, 272.0], dist: [0.0; 4], rpy: [0.0; 3], translation: [0.0; 3] });
        let q = UnitQuaternion::identity();
        let mut seed = 7u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (f64::from(u32::try_from(seed >> 40).expect("24 bits")) / f64::from(1u32 << 24) - 0.5) * 0.6
        };
        for deg in (-50i32..=50).step_by(10) {
            let truth = Vector3::<f64>::new(3.0, 0.4, -0.1);
            let yaw = truth.y.atan2(truth.x) + f64::from(deg).to_radians();
            let corners = armor_corners_world(ArmorSize::Small, &truth, yaw)
                .map(|c| cam.project(&q, &c).expect("visible") + nalgebra::Vector2::new(noise(), noise()));
            let det = Detection { color: ArmorColor::Red, icon: Icon::Infantry3, confidence: 1.0, corners };
            let obs = solve_armor(&det, &cam, &q).expect("solves");
            assert!((obs.position - truth).norm() < 0.08, "yaw {deg}: position error {}", (obs.position - truth).norm());
            let yaw_err = crate::tracker::model::wrap_angle(obs.yaw - yaw).abs().to_degrees();
            let limit = if deg.abs() <= 10 { 15.0 } else { 6.0 };
            assert!(yaw_err < limit, "yaw {deg}: yaw error {yaw_err} deg");
        }
    }
}
