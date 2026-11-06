//! Tracker: detections + gimbal pose -> whole-robot estimate.

use nalgebra::SMatrix;

use crate::detector::Icon;

pub mod model;

pub use model::{PlatePose, State};

/// Which output the aimer should trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrackMode {
    /// Low spin: radius and center are unobservable, aim at the observed plate directly.
    Plate,
    /// Clearly spinning: aim from the full robot model.
    Robot,
}

/// Track lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrackState {
    /// Seen, not yet trusted.
    Tentative,
    /// Trusted and updated this frame.
    Confirmed,
    /// Trusted, but no matching detection recently; prediction only.
    Coasting,
}

/// Current belief about the tracked robot.
#[derive(Debug, Clone, PartialEq)]
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
    pub p: SMatrix<f64, { model::N }, { model::N }>,
    /// Last observed plate (world frame) and the time it was seen. Used in plate mode.
    pub last_plate: Option<(f64, PlatePose)>,
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
