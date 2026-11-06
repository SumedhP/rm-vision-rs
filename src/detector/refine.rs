//! Corner refinement: fit a line through each lightbar's bright pixels and move the network's
//! keypoints onto it. Only the lateral position and direction change; the keypoints keep their
//! spacing along the bar, since glow makes the bar's visible length unreliable.

use image::RgbImage;
use nalgebra::{Point2, Vector2};

use super::{ArmorColor, Detection};

/// Returns `det` with each lightbar's keypoints snapped to the fitted lightbar axis. A bar whose fit
/// disagrees with the network (too few pixels, or tilted more than 20 degrees) is left unchanged.
pub fn refine(image: &RgbImage, det: &Detection) -> Detection {
    let [tl, bl, br, tr] = det.corners;
    let mut out = *det;
    if let Some([a, b]) = refine_bar(image, tl, bl, det.color) {
        out.corners[0] = a;
        out.corners[1] = b;
    }
    if let Some([a, b]) = refine_bar(image, tr, br, det.color) {
        out.corners[3] = a;
        out.corners[2] = b;
    }
    out
}

fn refine_bar(image: &RgbImage, top: Point2<f64>, bottom: Point2<f64>, color: ArmorColor) -> Option<[Point2<f64>; 2]> {
    let len = (bottom - top).norm();
    if len < 4.0 {
        return None;
    }
    let (mx, my) = (0.3 * len + 2.0, 0.15 * len + 2.0);
    let x0 = clamp_px(top.x.min(bottom.x) - mx, image.width());
    let x1 = clamp_px(top.x.max(bottom.x) + mx, image.width());
    let y0 = clamp_px(top.y.min(bottom.y) - my, image.height());
    let y1 = clamp_px(top.y.max(bottom.y) + my, image.height());

    let channel = |x: u32, y: u32| {
        let p = image.get_pixel(x, y);
        f64::from(match color {
            ArmorColor::Red => p[0],
            ArmorColor::Blue => p[2],
            _ => p[0].max(p[1]).max(p[2]),
        })
    };
    let roi: Vec<(f64, f64, f64)> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .map(|(x, y)| (f64::from(x), f64::from(y), channel(x, y)))
        .collect();
    let max = roi.iter().map(|p| p.2).fold(0.0, f64::max);
    let threshold = 0.6 * max;
    let bright: Vec<Vector2<f64>> = roi.iter().filter(|p| p.2 >= threshold && max > 60.0).map(|p| Vector2::new(p.0, p.1)).collect();
    if bright.len() < 6 {
        return None;
    }

    let n = f64::from(u32::try_from(bright.len()).ok()?);
    let c = bright.iter().sum::<Vector2<f64>>() / n;
    let (sxx, syy, sxy) = bright.iter().fold((0.0, 0.0, 0.0), |(a, b, d), p| {
        let q = p - c;
        (a + q.x * q.x, b + q.y * q.y, d + q.x * q.y)
    });
    let lambda = 0.5 * (sxx + syy) + (0.25 * (sxx - syy).powi(2) + sxy * sxy).sqrt();
    let mut axis = if sxy.abs() > 1e-9 { Vector2::new(sxy, lambda - sxx) } else if sxx > syy { Vector2::x() } else { Vector2::y() };
    axis.normalize_mut();
    let net_axis = (bottom - top) / len;
    if axis.dot(&net_axis) < 0.0 {
        axis = -axis;
    }
    if axis.dot(&net_axis) < 20f64.to_radians().cos() {
        return None;
    }
    let onto = |p: Point2<f64>| Point2::from(c + axis * axis.dot(&(p.coords - c)));
    Some([onto(top), onto(bottom)])
}

#[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::as_conversions, reason = "clamped to the image size first")]
fn clamp_px(v: f64, size: u32) -> u32 {
    v.clamp(0.0, f64::from(size)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::Icon;
    use image::Rgb;

    #[test]
    fn snaps_offset_keypoints_onto_vertical_bars() {
        let mut img = RgbImage::new(100, 60);
        for y in 10..50 {
            for x in [20u32, 21, 70, 71] {
                img.put_pixel(x, y, Rgb([250, 40, 40]));
            }
        }
        let det = Detection {
            color: ArmorColor::Red,
            icon: Icon::Infantry3,
            confidence: 0.9,
            corners: [Point2::new(23.0, 12.0), Point2::new(22.0, 48.0), Point2::new(68.0, 48.0), Point2::new(69.0, 12.0)],
        };
        let r = refine(&img, &det);
        for (i, want_x) in [(0, 20.5), (1, 20.5), (2, 70.5), (3, 70.5)] {
            assert!((r.corners[i].x - want_x).abs() < 0.2, "corner {i} at x={}, want {want_x}", r.corners[i].x);
        }
        assert!((r.corners[0].y - 12.0).abs() < 1.0, "spacing along the bar kept");
    }
}
