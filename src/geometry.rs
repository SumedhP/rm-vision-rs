//! Camera model, coordinate frames and armor plate geometry.
//!
//! Frames:
//! - **camera**: optical frame, x right, y down, z forward.
//! - **gimbal**: x forward, y left, z up, rotates with the turret.
//! - **world**: gimbal frame with the IMU rotation removed (gravity-aligned, co-moving with the robot).
//!   `p_world = q_imu * p_gimbal`.

use kornia_3d::camera::PinholeCamera;
use kornia_algebra::Vec2F64;
use nalgebra::{Isometry3, Point2, Point3, Rotation3, Translation3, UnitQuaternion, Vector3};
use serde::Deserialize;

/// Intrinsics (kornia's pinhole + Brown-Conrady model) and mounting of one camera.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Pixel projection model.
    pub intrinsics: PinholeCamera,
    /// Camera optical frame to gimbal frame.
    pub to_gimbal: Isometry3<f64>,
}

/// Config form of [`Camera`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    /// `[fx, fy, cx, cy]` in pixels.
    pub k: [f64; 4],
    /// Brown-Conrady `[k1, k2, p1, p2]`.
    pub dist: [f64; 4],
    /// Extra camera rotation relative to the gimbal, `[roll, pitch, yaw]` radians.
    pub rpy: [f64; 3],
    /// Camera optical center in the gimbal frame, meters.
    pub translation: [f64; 3],
}

impl From<&CameraConfig> for Camera {
    fn from(c: &CameraConfig) -> Self {
        let [fx, fy, cx, cy] = c.k;
        let [k1, k2, p1, p2] = c.dist;
        let optical_to_gimbal = Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
            0.0, 0.0, 1.0, //
            -1.0, 0.0, 0.0, //
            0.0, -1.0, 0.0,
        ));
        let r = Rotation3::from_euler_angles(c.rpy[0], c.rpy[1], c.rpy[2]) * optical_to_gimbal;
        Camera {
            intrinsics: PinholeCamera { fx, fy, cx, cy, k1, k2, p1, p2 },
            to_gimbal: Isometry3::from_parts(Translation3::from(Vector3::from(c.translation)), UnitQuaternion::from_rotation_matrix(&r)),
        }
    }
}

impl Camera {
    /// World point to pixel. `None` when behind the camera.
    pub fn project(&self, q_imu: &UnitQuaternion<f64>, p_world: &Vector3<f64>) -> Option<Point2<f64>> {
        let p = self.to_gimbal.inverse_transform_point(&Point3::from(q_imu.inverse() * p_world));
        if p.z <= 1e-6 {
            return None;
        }
        let k = &self.intrinsics;
        let (x, y) = (p.x / p.z, p.y / p.z);
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k.k1 + r2 * k.k2);
        let xd = x * radial + 2.0 * k.p1 * x * y + k.p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + k.p1 * (r2 + 2.0 * y * y) + 2.0 * k.p2 * x * y;
        Some(Point2::new(k.fx * xd + k.cx, k.fy * yd + k.cy))
    }

    /// Unit ray in the world frame through a pixel.
    pub fn ray(&self, q_imu: &UnitQuaternion<f64>, px: &Point2<f64>) -> Vector3<f64> {
        let n = self.intrinsics.normalize(Vec2F64::new(px.x, px.y));
        q_imu * (self.to_gimbal.rotation * Vector3::new(n.x, n.y, 1.0).normalize())
    }

    /// Camera optical center in the world frame.
    pub fn origin(&self, q_imu: &UnitQuaternion<f64>) -> Vector3<f64> {
        q_imu * self.to_gimbal.translation.vector
    }
}

/// Physical armor plate size. Keypoints sit on the lightbar ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArmorSize {
    /// 135 mm between lightbars.
    Small,
    /// 230 mm between lightbars.
    Large,
}

impl ArmorSize {
    /// Distance between the two lightbars, meters.
    pub fn width(self) -> f64 {
        match self {
            ArmorSize::Small => 0.135,
            ArmorSize::Large => 0.230,
        }
    }
}

/// Lightbar length, meters (keypoint top-to-bottom distance).
pub const LIGHTBAR_LENGTH: f64 = 0.056;

/// Armor plates lean back by this angle (top further from the viewer).
pub const ARMOR_TILT: f64 = 15.0 * std::f64::consts::PI / 180.0;

/// Corners in the plate frame, in keypoint order TL, BL, BR, TR as seen by a viewer facing the plate.
///
/// Plate frame: x points into the robot (away from the viewer), y left, z up.
pub fn armor_corners_local(size: ArmorSize) -> [Vector3<f64>; 4] {
    let (w, h) = (size.width() / 2.0, LIGHTBAR_LENGTH / 2.0);
    [Vector3::new(0.0, w, h), Vector3::new(0.0, w, -h), Vector3::new(0.0, -w, -h), Vector3::new(0.0, -w, h)]
}

/// Plate orientation in world: yaw about z, then the fixed backwards tilt.
pub fn armor_rotation(yaw: f64) -> Rotation3<f64> {
    Rotation3::from_axis_angle(&Vector3::z_axis(), yaw) * Rotation3::from_axis_angle(&Vector3::y_axis(), ARMOR_TILT)
}

/// World positions of a plate's corners (TL, BL, BR, TR).
pub fn armor_corners_world(size: ArmorSize, center: &Vector3<f64>, yaw: f64) -> [Vector3<f64>; 4] {
    let r = armor_rotation(yaw);
    armor_corners_local(size).map(|c| center + r * c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(dist: [f64; 4]) -> Camera {
        Camera::from(&CameraConfig { k: [700.0, 700.0, 480.0, 270.0], dist, rpy: [0.0; 3], translation: [0.05, 0.0, 0.1] })
    }

    #[test]
    fn project_and_ray_agree_with_distortion() {
        let cam = camera([-0.08, 0.1, 0.001, -0.001]);
        let q = UnitQuaternion::from_euler_angles(0.0, 0.1, 0.3);
        let p = Vector3::new(3.0, 0.4, -0.2);
        let px = cam.project(&q, &p).expect("in front");
        let ray = cam.ray(&q, &px);
        let to_p = (p - cam.origin(&q)).normalize();
        assert!((ray - to_p).norm() < 1e-6, "ray through the projection points back at the point");
    }

    #[test]
    fn forward_point_projects_to_principal_point() {
        let cam = camera([0.0; 4]);
        let px = cam.project(&UnitQuaternion::identity(), &Vector3::new(4.0, 0.0, 0.1)).expect("in front");
        assert!((px - Point2::new(480.0, 270.0)).norm() < 1e-9, "level camera, point straight ahead");
    }

    #[test]
    fn facing_plate_projects_with_tl_top_left() {
        let cam = camera([0.0; 4]);
        let q = UnitQuaternion::identity();
        let c = armor_corners_world(ArmorSize::Small, &Vector3::new(3.0, 0.0, 0.0), 0.0).map(|p| cam.project(&q, &p).expect("visible"));
        assert!(c[0].x < c[3].x && c[0].y < c[1].y, "TL is left of TR and above BL");
        assert!(c[1].x < c[2].x, "BL is left of BR");
    }
}
