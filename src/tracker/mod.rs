//! Tracker: detections + gimbal pose -> whole-robot estimate.

use std::collections::HashMap;

use nalgebra::{SMatrix, UnitQuaternion};
use serde::Deserialize;

use crate::detector::{Detection, Icon};
use crate::geometry::Camera;

pub mod ekf;
pub mod measurement;
pub mod model;

pub use measurement::ArmorObservation;
pub use model::{PlatePose, State};

use ekf::{Covariance, Ekf, Measurement, ProcessNoise};
use model::{ix, wrap_angle};

/// Tracker settings (`tracker:` section of `configs/general.yaml`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackerConfig {
    /// Horizontal center acceleration noise, m/s^2.
    pub sigma_acc: f64,
    /// Vertical center acceleration noise, m/s^2.
    pub sigma_acc_z: f64,
    /// Spin acceleration noise, rad/s^2.
    pub sigma_alpha: f64,
    /// Geometry drift noise, m/sqrt(s).
    pub sigma_geometry: f64,
    /// Bearing noise, radians.
    pub sigma_bearing: f64,
    /// Distance noise per squared meter of distance (`PnP` depth error grows with d^2).
    pub sigma_distance_per_m2: f64,
    /// Plate yaw noise when the plate is seen edge-on, radians.
    pub sigma_plate_yaw_oblique: f64,
    /// Plate yaw noise when the plate faces the camera (yaw is least observable there), radians.
    pub sigma_plate_yaw_facing: f64,
    /// Plates with a worse `PnP` fit than this are ignored, pixels.
    pub max_reprojection_px: f64,
    /// NIS gate (99% chi-square, 4 dof = 13.28).
    pub chi2_gate: f64,
    /// After this many consecutive frames with every plate gated out, the filter is the one that is
    /// wrong (e.g. a violent spin-up), so the next frame is applied ungated.
    pub gate_bypass_after: usize,
    /// Enter robot mode above this |spin rate|, rad/s.
    pub robot_mode_enter: f64,
    /// Leave robot mode below this |spin rate|, rad/s.
    pub robot_mode_exit: f64,
    /// Updates before a track is confirmed.
    pub confirm_updates: usize,
    /// Frames without an update before the track is dropped.
    pub lost_frames: usize,
    /// Updates a track needs before its geometry is kept as a prior.
    pub prior_min_updates: usize,
    /// Share of frames that must agree on the icon for the prior to be kept.
    pub prior_min_agreement: f64,
    /// Radius used when no prior exists, meters.
    pub default_radius: f64,
}

/// Which output the aimer should trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum TrackMode {
    /// Low spin: radius and center are unobservable, aim at the observed plate directly.
    Plate,
    /// Clearly spinning: aim from the full robot model.
    Robot,
}

/// Track lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[non_exhaustive]
pub enum TrackState {
    /// Seen, not yet trusted.
    Tentative,
    /// Trusted and updated this frame.
    Confirmed,
    /// Trusted, but no matching detection this frame; prediction only.
    Coasting,
}

/// Current belief about the tracked robot.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RobotEstimate {
    /// Time the estimate is valid at, seconds.
    pub t: f64,
    /// Robot identity.
    pub icon: Icon,
    /// Lifecycle state.
    pub state: TrackState,
    /// Aiming mode.
    pub mode: TrackMode,
    /// Number of plates (4, outpost 3).
    pub plate_count: usize,
    /// EKF mean.
    pub x: State,
    /// EKF covariance.
    pub p: Covariance,
    /// Last observed plate (world frame) and the time it was seen. Used in plate mode.
    pub last_plate: Option<(f64, PlatePose)>,
    /// Plates used in this frame's update, with their NIS. Empty when coasting.
    pub updates: Vec<PlateUpdate>,
    /// Frames this target has been tracked for.
    pub frames_seen: usize,
    /// Share of those frames whose detections agreed with `icon`.
    pub icon_agreement: f64,
}

impl RobotEstimate {
    /// Mean state propagated to time `t`.
    pub fn state_at(&self, t: f64) -> State {
        model::predict(&self.x, t - self.t)
    }

    /// All plates at time `t`.
    pub fn plates_at(&self, t: f64) -> Vec<PlatePose> {
        model::plates(&self.state_at(t), self.plate_count)
    }
}

/// One plate observation and what the filter did with it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct PlateUpdate {
    /// Plate id it was associated with.
    pub id: usize,
    /// Solved plate pose.
    pub observation: ArmorObservation,
    /// Normalized innovation squared against the prediction.
    pub nis: f64,
    /// Whether the gate accepted it.
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    r: f64,
    l: f64,
    h: f64,
}

#[derive(Debug)]
struct Track {
    icon: Icon,
    plate_count: usize,
    ekf: Ekf,
    t: f64,
    state: TrackState,
    mode: TrackMode,
    updates: usize,
    misses: usize,
    frames_seen: usize,
    frames_with_icon: usize,
    last_plate: Option<(f64, PlatePose)>,
}

/// Selects one robot, tracks it with the EKF, and remembers each robot's geometry between tracks.
#[derive(Debug)]
pub struct Tracker {
    cfg: TrackerConfig,
    camera: Camera,
    track: Option<Track>,
    priors: HashMap<Icon, Geometry>,
}

impl Tracker {
    /// New tracker with no target.
    pub fn new(cfg: TrackerConfig, camera: Camera) -> Self {
        Self { cfg, camera, track: None, priors: HashMap::new() }
    }

    /// Processes one frame's detections taken at time `t` with gimbal orientation `q_imu`.
    pub fn update(&mut self, t: f64, detections: &[Detection], q_imu: &UnitQuaternion<f64>) -> Option<RobotEstimate> {
        let observations: Vec<(&Detection, ArmorObservation)> = detections
            .iter()
            .filter_map(|d| measurement::solve_armor(d, &self.camera, q_imu).map(|o| (d, o)))
            .filter(|(_, o)| o.rms_px <= self.cfg.max_reprojection_px)
            .collect();

        let Some(track) = self.track.as_mut() else {
            let (cx, cy) = (self.camera.intrinsics.cx, self.camera.intrinsics.cy);
            let off_center = |d: &Detection| {
                let c = d.corners.iter().fold(nalgebra::Vector2::zeros(), |a, p| a + p.coords) / 4.0;
                (c.x - cx).hypot(c.y - cy)
            };
            let (det, obs) = observations.iter().min_by(|a, b| off_center(a.0).total_cmp(&off_center(b.0)))?;
            self.track = Some(self.start(det.icon, obs, t));
            return self.estimate(Vec::new());
        };

        track.ekf.predict(t - track.t, &ProcessNoise {
            acc: self.cfg.sigma_acc,
            acc_z: self.cfg.sigma_acc_z,
            alpha: self.cfg.sigma_alpha,
            geometry: self.cfg.sigma_geometry,
        });
        track.t = t;
        track.frames_seen += 1;
        let mine: Vec<ArmorObservation> = observations.iter().filter(|(d, _)| d.icon == track.icon).map(|(_, o)| *o).collect();
        track.frames_with_icon += usize::from(!mine.is_empty() || observations.is_empty());

        let gate = if track.misses >= self.cfg.gate_bypass_after { f64::INFINITY } else { self.cfg.chi2_gate };
        let mut applied = Vec::new();
        for obs in mine {
            let used: Vec<usize> = applied.iter().map(|u: &PlateUpdate| u.id).collect();
            let z = Measurement::new(
                obs.position.y.atan2(obs.position.x),
                obs.position.z.atan2(obs.position.xy().norm()),
                obs.position.norm(),
                obs.yaw,
            );
            let r = measurement_noise(&self.cfg, &obs);
            let Some(id) = associate(&track.ekf, track.plate_count, &z, &r, &used) else { continue };
            let Some(nis) = track.ekf.update(&z, &r, id, track.plate_count, gate)
            else {
                continue;
            };
            let accepted = nis <= gate;
            if accepted {
                track.last_plate = Some((t, PlatePose { id, position: obs.position, yaw: obs.yaw }));
            }
            applied.push(PlateUpdate { id, observation: obs, nis, accepted });
        }

        if applied.iter().any(|u| u.accepted) {
            track.updates += 1;
            track.misses = 0;
            if track.updates >= self.cfg.confirm_updates {
                track.state = TrackState::Confirmed;
            }
        } else {
            track.misses += 1;
            if track.state == TrackState::Confirmed {
                track.state = TrackState::Coasting;
            }
            if track.misses > self.cfg.lost_frames {
                self.drop_track();
                return None;
            }
        }
        let w = track.ekf.x[ix::W].abs();
        if w > self.cfg.robot_mode_enter {
            track.mode = TrackMode::Robot;
        } else if w < self.cfg.robot_mode_exit {
            track.mode = TrackMode::Plate;
        }
        self.estimate(applied)
    }

    fn start(&self, icon: Icon, obs: &ArmorObservation, t: f64) -> Track {
        let prior = self.priors.get(&icon);
        let geo = prior.copied().unwrap_or(Geometry { r: self.cfg.default_radius, l: 0.0, h: 0.0 });
        let mut x = State::zeros();
        x[ix::CX] = obs.position.x + geo.r * obs.yaw.cos();
        x[ix::CY] = obs.position.y + geo.r * obs.yaw.sin();
        x[ix::CZ] = obs.position.z;
        x[ix::YAW] = obs.yaw;
        x[ix::R] = geo.r;
        x[ix::L] = geo.l;
        x[ix::H] = geo.h;
        let geo_sigma = if prior.is_some() { 0.02 } else { 0.08 };
        let sigmas = [0.1, 1.0, 0.1, 1.0, 0.05, 0.3, 0.3, 8.0, geo_sigma, geo_sigma, geo_sigma];
        Track {
            icon,
            plate_count: if icon == Icon::Outpost { 3 } else { 4 },
            ekf: Ekf { x, p: Covariance::from_diagonal(&State::from_iterator(sigmas.iter().map(|s| s * s))) },
            t,
            state: TrackState::Tentative,
            mode: TrackMode::Plate,
            updates: 1,
            misses: 0,
            frames_seen: 1,
            frames_with_icon: 1,
            last_plate: Some((t, PlatePose { id: 0, position: obs.position, yaw: obs.yaw })),
        }
    }

    fn drop_track(&mut self) {
        let Some(track) = self.track.take() else { return };
        let agreement = ratio(track.frames_with_icon, track.frames_seen);
        if track.updates >= self.cfg.prior_min_updates && agreement >= self.cfg.prior_min_agreement && track.mode == TrackMode::Robot {
            let x = track.ekf.x;
            self.priors.insert(track.icon, Geometry { r: x[ix::R], l: x[ix::L], h: x[ix::H] });
        }
    }

    fn estimate(&self, updates: Vec<PlateUpdate>) -> Option<RobotEstimate> {
        let track = self.track.as_ref()?;
        Some(RobotEstimate {
            t: track.t,
            icon: track.icon,
            state: track.state,
            mode: track.mode,
            plate_count: track.plate_count,
            x: track.ekf.x,
            p: track.ekf.p,
            last_plate: track.last_plate,
            updates,
            frames_seen: track.frames_seen,
            icon_agreement: ratio(track.frames_with_icon, track.frames_seen),
        })
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    f64::from(u32::try_from(a).unwrap_or(u32::MAX)) / f64::from(u32::try_from(b.max(1)).unwrap_or(u32::MAX))
}

/// The plate id that explains the observation best (lowest NIS), skipping plates already matched
/// this frame. Using the full innovation, not just yaw, lets the plate height and radius tell the
/// even and odd plates apart, so a spin-up transient can't swap them.
fn associate(ekf: &Ekf, plate_count: usize, z: &Measurement, r: &SMatrix<f64, 4, 4>, used: &[usize]) -> Option<usize> {
    (0..plate_count)
        .filter(|id| !used.contains(id))
        .filter_map(|id| ekf.nis(z, r, id, plate_count).map(|nis| (id, nis)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
}
fn measurement_noise(cfg: &TrackerConfig, obs: &ArmorObservation) -> SMatrix<f64, 4, 4> {
    let d = obs.position.norm();
    let facing = wrap_angle(obs.yaw - obs.position.y.atan2(obs.position.x)).cos();
    let yaw_sigma = cfg.sigma_plate_yaw_oblique + (cfg.sigma_plate_yaw_facing - cfg.sigma_plate_yaw_oblique) * facing * facing;
    let dist_sigma = cfg.sigma_distance_per_m2 * d * d;
    SMatrix::from_diagonal(&Measurement::new(cfg.sigma_bearing.powi(2), cfg.sigma_bearing.powi(2), dist_sigma.powi(2), yaw_sigma.powi(2)))
}
