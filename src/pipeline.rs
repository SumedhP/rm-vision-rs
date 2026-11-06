//! The pipeline loop: source -> detector pool -> tracker -> aimer.
//!
//! Time comes from the data, never the wall clock: the tracker runs at each frame's exposure time
//! and the aimer ticks at a fixed rate between frames. Replaying the same input therefore gives the
//! same output, and the same loop drives live hardware once live sources exist.

use serde::Serialize;

use crate::aimer::{AimCommand, AimTarget, Aimer};
use crate::config::{GeneralConfig, RobotConfig};
use crate::detector::pool::DetectorPool;
use crate::detector::{ArmorColor, DetectedFrame};
use crate::geometry::Camera;
use crate::io::{CameraSource, RobotSource, RobotState};
use crate::sync::PoseBuffer;
use crate::tracker::{RobotEstimate, Tracker};

/// Everything the pipeline decided for one frame.
#[derive(Debug, Clone, Serialize)]
pub struct FrameRecord {
    /// Robot state interpolated to the frame time.
    pub robot: RobotState,
    /// Tracker output after this frame.
    pub estimate: Option<RobotEstimate>,
    /// Aimer ticks since the previous frame, computed from the previous frame's estimate.
    pub commands: Vec<AimCommand>,
    /// Aimer target at the last tick.
    pub target: Option<AimTarget>,
}

/// Pipeline settings that are not owned by a stage.
#[derive(Debug, Clone, Copy)]
pub struct RunOptions {
    /// Aimer tick rate, Hz.
    pub control_rate: f64,
}

/// Runs the source to the end, calling `sink` with every frame in order.
///
/// # Errors
/// Source or detector failure, or `sink` returning an error.
pub fn run(
    general: &GeneralConfig,
    robot_cfg: &RobotConfig,
    mut camera_source: CameraSource,
    robot_source: &RobotSource,
    options: RunOptions,
    mut sink: impl FnMut(&DetectedFrame, &FrameRecord) -> color_eyre::Result<()>,
) -> color_eyre::Result<()> {
    let mut poses = PoseBuffer::new(4096);
    let default_state = match robot_source {
        RobotSource::Defaults(s) => *s,
        RobotSource::Log(states) => {
            for s in states {
                poses.push(s.t - robot_cfg.imu_delay, s.q);
            }
            states.first().copied().unwrap_or(robot_cfg.default_state)
        }
    };
    let enemy = if default_state.robot_id > 100 { ArmorColor::Red } else { ArmorColor::Blue };
    let mut pool = DetectorPool::new(&general.detector, enemy)?;
    let mut tracker = Tracker::new(general.tracker.clone(), Camera::from(&robot_cfg.camera));
    let mut aimer = Aimer::new(general.aimer.clone(), robot_cfg.gimbal.clone());
    let period = 1.0 / options.control_rate;
    let mut next_tick: Option<f64> = None;
    let mut previous: Option<RobotEstimate> = None;
    let depth = 2 * general.detector.workers.max(1);

    let mut process = |done: DetectedFrame| -> color_eyre::Result<()> {
        let t = done.frame.t;
        let robot = RobotState { t, q: poses.at(t).unwrap_or(default_state.q), ..default_state };
        let mut commands = Vec::new();
        let mut tick = next_tick.unwrap_or(t);
        while tick <= t {
            commands.push(aimer.update(tick, previous.as_ref(), robot.bullet_speed, robot_cfg.downlink_delay));
            tick += period;
        }
        next_tick = Some(tick);
        let estimate = tracker.update(t, &done.detections, &robot.q);
        previous.clone_from(&estimate);
        sink(&done, &FrameRecord { robot, estimate, commands, target: aimer.target() })
    };

    while let Some(mut frame) = camera_source.next_frame()? {
        frame.t -= robot_cfg.camera_delay;
        pool.submit(frame)?;
        if pool.in_flight() >= depth {
            if let Some(done) = pool.recv() {
                process(done)?;
            }
        }
    }
    while let Some(done) = pool.recv() {
        process(done)?;
    }
    Ok(())
}
