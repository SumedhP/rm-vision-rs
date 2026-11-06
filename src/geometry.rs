//! Camera model, coordinate frames and armor plate geometry.
//!
//! Frames:
//! - **camera**: optical frame, x right, y down, z forward.
//! - **gimbal**: x forward, y left, z up, rotates with the turret.
//! - **world**: gimbal frame with the IMU rotation removed (gravity-aligned, co-moving with the robot).
//!   `p_world = q_imu * p_gimbal`.

use nalgebra::{Matrix3, Point2, Rotation3, UnitQuaternion, Vector3};
use serde::Deserialize;

/// Pinhole camera with Brown-Conrady distortion `[k1, k2, p1, p2, k3]`.
#[derive(Debug, Clone, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length x in pixels.
    pub fx: f64,
    /// Focal length y in pixels.
    pub fy: f64,
    /// Principal point x in pixels.
    pub cx: f64,
    /// Principal point y in pixels.
    pub cy: f64,
    /// Distortion `[k1, k2, p1, p2, k3]`.
    pub dist: [f64; 5],
}

impl CameraIntrinsics {
    /// Projects a point in the camera frame to pixels. `None` when behind the camera.
    pub fn project(&self, p: &Vector3<f64>) -> Option<Point2<f64>> {
        if p.z <= 1e-6 {
            return None;
        }
        let (x, y) = self.distort(p.x / p.z, p.y / p.z);
        Some(Point2::new(self.fx * x + self.cx, self.fy * y + self.cy))
    }

    /// Unit bearing in the camera frame for a pixel.
    pub fn bearing(&self, px: &Point2<f64>) -> Vector3<f64> {
        let (xd, yd) = ((px.x - self.cx) / self.fx, (px.y - self.cy) / self.fy);
        let (mut x, mut y) = (xd, yd);
        for _ in 0..10 {
            let (dx, dy) = self.distort(x, y);
            x -= dx - xd;
            y -= dy - yd;
        }
        Vector3::new(x, y, 1.0).normalize()
    }

    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let [k1, k2, p1, p2, k3] = self.dist;
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        (
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    }
}

/// Camera mounting on the gimbal.
#[derive(Debug, Clone, Deserialize)]
pub struct CameraExtrinsics {
    /// Extra rotation of the camera relative to the gimbal, `[roll, pitch, yaw]` in radians.
    pub rpy: [f64; 3],
    /// Camera optical center in the gimbal frame, meters.
    pub translation: [f64; 3],
}

impl CameraExtrinsics {
    /// Rotation taking camera-frame vectors into the gimbal frame.
    pub fn rotation(&self) -> Rotation3<f64> {
        let axes = Rotation3::from_matrix_unchecked(Matrix3::new(
            0.0, 0.0, 1.0, //
            -1.0, 0.0, 0.0, //
            0.0, -1.0, 0.0,
        ));
        Rotation3::from_euler_angles(self.rpy[0], self.rpy[1], self.rpy[2]) * axes
    }

    /// Camera point to world point, given the gimbal orientation.
    pub fn camera_to_world(&self, q_imu: &UnitQuaternion<f64>, p_cam: &Vector3<f64>) -> Vector3<f64> {
        q_imu * (self.rotation() * p_cam + Vector3::from(self.translation))
    }

    /// World point to camera point, given the gimbal orientation.
    pub fn world_to_camera(&self, q_imu: &UnitQuaternion<f64>, p_world: &Vector3<f64>) -> Vector3<f64> {
        self.rotation().inverse() * (q_imu.inverse() * p_world - Vector3::from(self.translation))
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
    [
        Vector3::new(0.0, w, h),
        Vector3::new(0.0, w, -h),
        Vector3::new(0.0, -w, -h),
        Vector3::new(0.0, -w, h),
    ]
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

    #[test]
    fn project_bearing_roundtrip_with_distortion() {
        let k = CameraIntrinsics { fx: 700.0, fy: 700.0, cx: 480.0, cy: 270.0, dist: [-0.08, 0.1, 0.001, -0.001, 0.0] };
        let p = Vector3::new(0.4, -0.2, 3.0);
        let px = k.project(&p).expect("in front of camera");
        let b = k.bearing(&px);
        assert!((b - p.normalize()).norm() < 1e-9, "bearing should invert projection");
    }

    #[test]
    fn identity_extrinsics_map_optical_axis_to_forward() {
        let e = CameraExtrinsics { rpy: [0.0; 3], translation: [0.0; 3] };
        let q = UnitQuaternion::identity();
        let fwd = e.camera_to_world(&q, &Vector3::new(0.0, 0.0, 2.0));
        assert!((fwd - Vector3::new(2.0, 0.0, 0.0)).norm() < 1e-12, "camera z is world x");
        let back = e.world_to_camera(&q, &fwd);
        assert!((back - Vector3::new(0.0, 0.0, 2.0)).norm() < 1e-12, "roundtrip");
    }

    #[test]
    fn facing_plate_projects_with_tl_top_left() {
        let k = CameraIntrinsics { fx: 700.0, fy: 700.0, cx: 480.0, cy: 270.0, dist: [0.0; 5] };
        let e = CameraExtrinsics { rpy: [0.0; 3], translation: [0.0; 3] };
        let q = UnitQuaternion::identity();
        let c = armor_corners_world(ArmorSize::Small, &Vector3::new(3.0, 0.0, 0.0), 0.0)
            .map(|p| k.project(&e.world_to_camera(&q, &p)).expect("visible"));
        assert!(c[0].x < c[3].x && c[0].y < c[1].y, "TL is left of TR and above BL");
        assert!(c[1].x < c[2].x, "BL is left of BR");
    }
}
